"""Source-bound Cosmos refresh; raw metadata stays in the ignored input cache.

Packet construction intentionally matches prepare_live_input. The separate batch
ledger distinguishes reused vectors, resumed batches, logical API calls and HTTP
attempts (including the existing Azure client's retries).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np

from sampling_comparison.matryoshka_experiment import canonical, load_input, normalized_prefix, sha256_file, write_json
from sampling_comparison.matryoshka_inputs import AzureBatchEmbedder, BatchEmbedder
from sampling_comparison.v7_dataset import load_cosmos_otel_expected_dataset
from trace_sampling.session_embedding import TiktokenTokenizer
from trace_sampling.token_representation import CanonicalizationOptions, TokenSessionEvidencePacketBuilder

ENDPOINT = "https://bugboss-foundry.services.ai.azure.com"
MODEL = "text-embedding-3-small"
LABEL_SOURCE = (
    "Existing task-design expected_outcome: good=1, bad=0, partial=0. "
    "Not fresh judge results or verified actual completion."
)
CONTAINERS = frozenset((
    "catalogs", "cost_snapshots", "eval_labels_e2e", "eval_labels_smoke",
    "labels", "labels_redteam_smoke", "labels_vp_e2e", "summaries", "world_store", "spans",
))


class RefreshError(ValueError):
    """Only fixed, aggregate-safe messages belong in this exception."""


def require_target(endpoint: str, deployment: str) -> None:
    if endpoint.rstrip("/") != ENDPOINT or deployment != MODEL:
        raise RefreshError("Embedding endpoint or model is not the authorized target.")


class MeteredAzureEmbedder(AzureBatchEmbedder):
    """Use the existing Azure implementation; count retries without logging URLs."""

    def __init__(self, **kwargs: Any):
        require_target(kwargs["endpoint"], kwargs["deployment"])
        super().__init__(**kwargs)
        from openai import DefaultHttpxClient

        self.http_attempts = 0
        self.http_successes = 0
        self.client = self.client.with_options(http_client=DefaultHttpxClient(
            follow_redirects=False,
            event_hooks={"request": [self._request], "response": [self._response]},
        ))

    def _request(self, request: Any) -> None:
        # Refuse redirects or client misconfiguration before disclosing input.
        if str(request.url) != ENDPOINT + "/openai/v1/embeddings":
            raise RefreshError("Refusing an unauthorized embedding request target.")
        self.http_attempts += 1

    def _response(self, response: Any) -> None:
        self.http_successes += int(200 <= response.status_code < 300)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_state(path: Path, *, count_lines: bool = False) -> dict[str, Any]:
    digest, size, lines, last = hashlib.sha256(), 0, 0, b""
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            if count_lines:
                lines += chunk.count(b"\n")
                last = chunk[-1:]
    result = {"sha256": digest.hexdigest(), "bytes": size}
    if count_lines:
        result["documents"] = lines + int(bool(last) and last != b"\n")
    return result


def verify_snapshot(root: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    rows = preflight["containers"]
    if len(rows) != 10 or {r["container"] for r in rows} != CONTAINERS:
        raise RefreshError("Preflight must bind all ten top-level containers.")
    if preflight["required_model"] != MODEL or preflight["required_dimensions"] != 1536:
        raise RefreshError("Preflight embedding model or dimensions mismatch.")
    files = {}
    for row in rows:
        name = f"genesis_observability__{row['container']}.jsonl"
        state = _file_state(root / name, count_lines=True)
        if state != {key: row[key] for key in ("sha256", "bytes", "documents")}:
            raise RefreshError("Snapshot checksum, size or document count mismatch; stop refresh.")
        files[name] = state
    if sum(r["bytes"] for r in rows) != preflight["snapshot_bytes"] or sum(
        r["documents"] for r in rows
    ) != preflight["snapshot_documents"]:
        raise RefreshError("Preflight aggregate snapshot counts mismatch.")
    if set(preflight["metadata_sha256"]) != {"manifest.json", "refresh_report.json"}:
        raise RefreshError("Preflight snapshot metadata binding is incomplete.")
    for name, expected in preflight["metadata_sha256"].items():
        state = _file_state(root / name)
        if state["sha256"] != expected:
            raise RefreshError("Snapshot metadata checksum mismatch; stop refresh.")
        files[name] = state
    return files


def _reuse(manifest_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    before = {name: sha256_file(manifest_path.parent / name)
              for name in (manifest_path.name, "units.json", "vectors.npz")}
    manifest = _read(manifest_path)
    if manifest["files"] != {"units": "units.json", "vectors": "vectors.npz"}:
        raise RefreshError("Reuse cache must contain local standard input artifacts.")
    data = load_input(manifest_path)
    if data.dataset_id != "cosmos_otel" or data.vectors.dtype != np.float32:
        raise RefreshError("Reuse cache requires Cosmos native float32 vectors.")
    vectors = {}
    for row, vector in zip(_read(manifest_path.parent / "units.json"), data.vectors, strict=True):
        packet_hash = row.get("packet_hash", "")
        if not re.fullmatch("[0-9a-f]{64}", packet_hash):
            raise RefreshError("Reuse cache lacks a valid canonical packet hash.")
        if packet_hash in vectors and vectors[packet_hash].tobytes() != vector.tobytes():
            raise RefreshError("Reuse cache has conflicting vectors for one canonical packet.")
        vectors[packet_hash] = vector
    after = {name: sha256_file(manifest_path.parent / name) for name in before}
    if before != after:
        raise RefreshError("Reuse cache changed while loading.")
    return {"manifest_path": str(manifest_path.resolve()), "sha256": before}, vectors


def _verify_reuse(binding: dict[str, Any]) -> None:
    parent = Path(binding["manifest_path"]).parent
    if any(sha256_file(parent / name) != expected for name, expected in binding["sha256"].items()):
        raise RefreshError("Reuse cache checksum changed; stop refresh.")


def _batch_paths(output: Path, start: int) -> tuple[Path, Path]:
    path = output / "batches" / f"{start:06}.npz"
    return path, path.with_suffix(".json")


def _load_batches(output: Path, hashes: list[str], reuse: dict[str, np.ndarray],
                  batch_size: int) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], set[int]]:
    vectors, ledgers, done = {}, [], set()
    expected_paths = set()
    for start in range(0, len(hashes), batch_size):
        batch_hashes = hashes[start:start + batch_size]
        path, ledger_path = _batch_paths(output, start)
        expected_paths.update((path, ledger_path))
        if not path.exists() and not ledger_path.exists():
            continue
        if not path.exists() or not ledger_path.exists():
            raise RefreshError("Incomplete or uncertain batch requires explicit recovery; no automatic resend.")
        ledger = _read(ledger_path)
        missing = sum(h not in reuse for h in batch_hashes)
        if ledger.get("status") != "complete" or sha256_file(path) != ledger.get("sha256"):
            raise RefreshError("Cached batch checksum or completion mismatch.")
        if (ledger["requested_vectors"] != missing or
                ledger["reused_vectors"] != len(batch_hashes) - missing or
                ledger["logical_requests"] != int(missing > 0)):
            raise RefreshError("Cached batch accounting mismatch.")
        for field in ("api_input_tokens", "http_attempts", "http_successes"):
            if type(ledger[field]) is not int or ledger[field] < 0:
                raise RefreshError("Cached batch usage is incomplete.")
        if ledger["http_attempts"] < ledger["logical_requests"] or ledger["http_successes"] != ledger["logical_requests"]:
            raise RefreshError("Cached batch HTTP accounting mismatch.")
        with np.load(path, allow_pickle=False) as arrays:
            if arrays["hashes"].tolist() != batch_hashes:
                raise RefreshError("Cached batch canonical row order mismatch.")
            matrix = arrays["vectors"].copy()
        if matrix.shape != (len(batch_hashes), 1536) or matrix.dtype != np.float32:
            raise RefreshError("Cached batch dimensions or dtype mismatch.")
        normalized_prefix(matrix, 1536)
        for h, vector in zip(batch_hashes, matrix, strict=True):
            if h in reuse and vector.tobytes() != reuse[h].tobytes():
                raise RefreshError("Reused vector is not bit-for-bit identical.")
            vectors[h] = vector
        ledgers.append(ledger)
        done.add(start)
    if (output / "batches").exists() and set((output / "batches").iterdir()) - expected_paths:
        raise RefreshError("Unexpected batch artifacts require explicit recovery.")
    return vectors, ledgers, done


USAGE_FIELDS = ("requested_vectors", "reused_vectors", "logical_requests",
                "http_attempts", "http_successes", "api_input_tokens")


def _usage(ledgers: list[dict[str, Any]]) -> dict[str, int]:
    return {key: sum(row[key] for row in ledgers) for key in USAGE_FIELDS}


def safe_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Allowlist aggregates, never serialize raw records or arbitrary metadata."""
    fields = (
        "status", "sessions", "agents", "positive_count", "unique_packets",
        "truncated_sessions", "representation_sources", "outcome_counts",
        "embedding_model_id", "embedding_dimensions", "reused_vectors",
        "requested_vectors", "logical_requests", "http_attempts", "http_successes",
        "api_input_tokens", "stored_embedding_batches", "live_judge_calls",
        "fresh_embedding_batches_this_invocation", "resumed_embedding_batches_this_invocation",
        "fresh_requested_vectors_this_invocation", "resumed_requested_vectors_this_invocation",
        "fresh_logical_requests_this_invocation", "fresh_api_input_tokens_this_invocation",
        "snapshot_cutoff_utc", "snapshot_export_completed_at",
    )
    summary = {key: result[key] for key in fields if key in result}
    # Even nested dictionaries are filtered, not trusted wholesale.
    for key, allowed in (
        ("representation_sources", ("linked_raw_spans", "synthetic_label_document")),
        ("outcome_counts", ("good", "bad", "partial")),
    ):
        if key in summary:
            summary[key] = {name: int(result[key].get(name, 0)) for name in allowed}
    return summary


def prepare_refresh_input(
    cosmos_root: Path, output: Path, *, preflight_path: Path, reuse_manifest: Path,
    embedder: BatchEmbedder | None = None, endpoint: str = ENDPOINT, deployment: str = MODEL,
    batch_size: int = 16, progress: Callable[[str], None] | None = None,
    loader: Callable[..., Any] = load_cosmos_otel_expected_dataset,
) -> dict[str, Any]:
    """No cloud by default. Resume pins both source generations BEFORE any API."""
    require_target(endpoint, deployment)
    if not 1 <= batch_size <= 32:
        raise RefreshError("Batch size must be between 1 and 32.")
    cosmos_root, output, reuse_manifest = cosmos_root.resolve(), output.resolve(), reuse_manifest.resolve()
    if (output == cosmos_root or output.is_relative_to(cosmos_root) or
            cosmos_root.is_relative_to(output) or output == reuse_manifest.parent or
            output.is_relative_to(reuse_manifest.parent) or reuse_manifest.parent.is_relative_to(output)):
        raise RefreshError("Output must not overlap the source or reuse cache.")
    log = progress or (lambda message: None)
    preflight_hash = sha256_file(preflight_path)
    preflight = _read(preflight_path)
    snapshot = verify_snapshot(cosmos_root, preflight)
    reuse_binding, reused = _reuse(reuse_manifest)
    watched = [cosmos_root / name for name in snapshot]
    watched += [reuse_manifest.parent / name for name in reuse_binding["sha256"]]
    watched.append(preflight_path)
    source_stats = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in watched}

    def check_source_stats() -> None:
        if any((path.stat().st_size, path.stat().st_mtime_ns) != state
               for path, state in source_stats.items()):
            raise RefreshError("Source or reuse cache changed during preparation; stop refresh.")

    binding = {
        "version": "matryoshka-refresh-preparation-v1", "source_root": str(cosmos_root),
        "snapshot_files": snapshot, "preflight_sha256": preflight_hash,
        "reuse_source": reuse_binding, "endpoint": ENDPOINT, "deployment": MODEL,
        "embedding_model_id": MODEL, "embedding_dimensions": 1536,
        "batch_size": batch_size, "label_source": LABEL_SOURCE,
        "packet_policy": "prepare_live_input: cl100k_base, max8191, original order",
    }

    def verify_sources() -> None:
        if sha256_file(preflight_path) != preflight_hash or verify_snapshot(cosmos_root, preflight) != snapshot:
            raise RefreshError("Snapshot or preflight fingerprint changed; stop refresh.")
        _verify_reuse(reuse_binding)

    config_path, manifest_path = output / "preparation.json", output / "manifest.json"
    prior = _read(config_path) if config_path.exists() else None
    if prior is not None and prior["binding"] != binding:
        raise RefreshError("Preparation fingerprint mismatch; use a fresh output.")
    if prior is None and output.exists() and any(output.iterdir()):
        raise RefreshError("Nonempty output lacks matching preparation provenance.")
    if manifest_path.exists():
        data = load_input(manifest_path)
        records = _read(output / "units.json")
        if hashlib.sha256(canonical(records).encode()).hexdigest() != prior["units_sha256"]:
            raise RefreshError("Completed input preparation fingerprint mismatch.")
        hashes = sorted({r["packet_hash"] for r in records})
        vectors, ledgers, done = _load_batches(output, hashes, reused, batch_size)
        if len(vectors) != len(hashes):
            raise RefreshError("Completed input is missing retained batches.")
        expected = np.asarray([vectors[r["packet_hash"]] for r in records])
        if data.vectors.tobytes() != expected.tobytes():
            raise RefreshError("Completed vectors differ from retained batch evidence.")
        result = _read(manifest_path)
        if any(result[key] != value for key, value in _usage(ledgers).items()):
            raise RefreshError("Completed input usage differs from batch ledgers.")
        if result["refresh_binding"] != binding:
            raise RefreshError("Completed input source binding mismatch.")
        verify_sources()
        return {**result, "fresh_embedding_batches_this_invocation": 0,
                "resumed_embedding_batches_this_invocation": len(done),
                "fresh_requested_vectors_this_invocation": 0,
                "resumed_requested_vectors_this_invocation": result["requested_vectors"],
                "fresh_logical_requests_this_invocation": 0, "fresh_api_input_tokens_this_invocation": 0}

    log("Snapshot and reuse checksums verified; loading expected labels and spans.")
    data = loader(cosmos_root, partial_label=0)
    tokenizer = TiktokenTokenizer(model_name=MODEL, encoding_name="cl100k_base")
    builder = TokenSessionEvidencePacketBuilder(
        options=CanonicalizationOptions(tokenizer=tokenizer, max_tokens=8191),
        max_size=max(4096, len(data.ordered_unit_ids)),
    )
    records, texts = [], {}
    for ordinal, uid in enumerate(data.ordered_unit_ids, 1):
        trace = data.traces_by_unit_id[uid]
        packet = builder.build(trace)
        h = hashlib.sha256(packet.canonical_json.encode()).hexdigest()
        texts[h] = packet.canonical_json
        label = data.labels_by_unit[uid]
        if label not in (0, 1):
            raise RefreshError("Missing or nonbinary expected reference label.")
        records.append({
            "unit_id": uid, "agent_id": data.agent_id_by_unit[uid], "signature": list(trace.signature),
            "concept_key": data.concept_key_by_unit[uid], "label": label, "packet_hash": h,
            "original_tokens": packet.original_tokens, "emitted_tokens": packet.emitted_tokens,
            "truncated": packet.audit.truncated, "representation_source": data.representation_source_by_unit[uid],
        })
        if ordinal % 50 == 0 or ordinal == len(data.ordered_unit_ids):
            log(f"Canonical packets: {ordinal}/{len(data.ordered_unit_ids)}.")
    if data.dataset_id != "cosmos_otel" or len(records) < 2 or len({r["unit_id"] for r in records}) != len(records):
        raise RefreshError("Refresh requires at least two unique Cosmos units.")
    hashes = sorted(texts)
    reused_count = sum(h in reused for h in hashes)
    outcomes = dict(Counter(data.metadata_by_unit[uid]["expected_outcome"] for uid in data.ordered_unit_ids))
    plan = {
        "dataset_id": data.dataset_id, "status": "prepared_without_embeddings",
        "embedding_model_id": MODEL, "embedding_dimensions": 1536,
        "endpoint": ENDPOINT, "deployment": MODEL, "label_source": LABEL_SOURCE,
        "source_paths": data.source_paths,
        "source_hashes": {key: sha256_file(Path(path)) for key, path in data.source_paths.items()},
        "sessions": len(records), "agents": len(set(data.agent_id_by_unit.values())),
        "positive_count": sum(r["label"] for r in records), "outcome_counts": outcomes,
        "unique_packets": len(hashes), "reused_vectors": reused_count,
        "requested_vectors": len(hashes) - reused_count,
        "truncated_sessions": sum(r["truncated"] for r in records),
        "representation_sources": dict(Counter(r["representation_source"] for r in records)),
        "input_tokens_before_api": sum(tokenizer.count(texts[h]) for h in hashes),
        "missing_input_tokens_before_api": sum(tokenizer.count(texts[h]) for h in hashes if h not in reused),
        "live_judge_calls": 0, "refresh_binding": binding,
        "snapshot_cutoff_utc": preflight["snapshot_cutoff_utc"],
        "snapshot_export_completed_at": preflight["snapshot_export_completed_at"],
        "source_read_only": True, "point_in_time_snapshot": preflight["point_in_time_snapshot"],
    }
    for actual, expected in (
        ("sessions", "eligible_units"), ("agents", "agents"), ("unique_packets", "unique_canonical_packets"),
        ("reused_vectors", "cached_packet_matches"), ("requested_vectors", "missing_unique_native_embeddings"),
        ("outcome_counts", "outcome_counts"), ("truncated_sessions", "truncated_packets"),
        ("representation_sources", "representation_sources"),
    ):
        if plan[actual] != preflight[expected]:
            raise RefreshError("Canonical population differs from verified preflight; no API permitted.")
    config = {"binding": binding, "units_sha256": hashlib.sha256(canonical(records).encode()).hexdigest()}
    if prior is not None and prior != config:
        raise RefreshError("Canonical preparation fingerprint mismatch.")
    # Validate every existing batch before sending even the first missing batch.
    vectors, ledgers, done = _load_batches(output, hashes, reused, batch_size)
    resumed_usage, resumed_batches = _usage(ledgers), len(done)
    verify_sources()
    write_json(config_path, config)
    write_json(output / "units.json", records)
    write_json(output / "readiness.json", plan)
    if embedder is None:
        return plan
    for start in range(0, len(hashes), batch_size):
        if start in done:
            continue
        check_source_stats()
        batch_hashes = hashes[start:start + batch_size]
        missing = [h for h in batch_hashes if h not in reused]
        path, ledger_path = _batch_paths(output, start)
        ledger = {key: 0 for key in USAGE_FIELDS}
        ledger.update(requested_vectors=len(missing), reused_vectors=len(batch_hashes) - len(missing))
        new = {}
        if missing:
            # A durable reservation prevents an interrupted/ambiguous paid call
            # from being silently sent again on resume.
            write_json(ledger_path, {**ledger, "status": "requesting"})
            attempts_before = getattr(embedder, "http_attempts", 0)
            successes_before = getattr(embedder, "http_successes", 0)
            try:
                matrix, tokens = embedder.embed([texts[h] for h in missing])
                if matrix.shape != (len(missing), 1536) or matrix.dtype != np.float32:
                    raise RefreshError("Embedding response must be native float32 [N,1536].")
                normalized_prefix(matrix, 1536)
                if type(tokens) is not int or tokens < 0:
                    raise RefreshError("Embedding API omitted billable token usage.")
            except Exception:
                write_json(ledger_path, {
                    **ledger, "status": "failed_or_uncertain", "logical_requests": 1,
                    "http_attempts": getattr(embedder, "http_attempts", attempts_before + 1) - attempts_before,
                    "http_successes": getattr(embedder, "http_successes", successes_before) - successes_before,
                    "api_input_tokens": None,
                })
                raise RefreshError("Embedding request failed or response invalid; no success manifest; explicit recovery required.") from None
            ledger.update(
                logical_requests=1, api_input_tokens=tokens,
                http_attempts=getattr(embedder, "http_attempts", attempts_before + 1) - attempts_before,
                http_successes=getattr(embedder, "http_successes", successes_before + 1) - successes_before,
            )
            new = dict(zip(missing, matrix, strict=True))
        matrix = np.asarray([reused[h] if h in reused else new[h] for h in batch_hashes], dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, hashes=np.asarray(batch_hashes), vectors=matrix)
        tmp.replace(path)
        ledger.update(status="complete", sha256=sha256_file(path))
        write_json(ledger_path, ledger)
        ledgers.append(ledger)
        vectors.update(zip(batch_hashes, matrix, strict=True))
        log(f"Vectors available: {min(start + batch_size, len(hashes))}/{len(hashes)}; "
            f"new API vectors this invocation: {_usage(ledgers)['requested_vectors'] - resumed_usage['requested_vectors']}.")
    verify_sources()
    # Revalidate durable evidence before publishing a success manifest.
    vectors, ledgers, done = _load_batches(output, hashes, reused, batch_size)
    usage = _usage(ledgers)
    tmp = output / "vectors.tmp.npz"
    np.savez_compressed(tmp, unit_ids=np.asarray([r["unit_id"] for r in records]),
                        vectors=np.asarray([vectors[r["packet_hash"]] for r in records]))
    tmp.replace(output / "vectors.npz")
    result = {
        **plan, **usage, "version": "matryoshka-input-v1", "status": "complete",
        "representation_policy": "V3 weighted canonical full-session packet, cl100k_base, max 8191 tokens; prefixes taken only after full-vector API generation.",
        "concept_definition": "Dataset loader's cohort-relative concept key; not a validated business-use-case classifier.",
        "excluded_sessions": 0, "files": {"units": "units.json", "vectors": "vectors.npz"},
        "hashes": {key: sha256_file(output / name) for key, name in (("units", "units.json"), ("vectors", "vectors.npz"))},
        "live_embedding_calls": usage["logical_requests"], "stored_embedding_batches": len(done),
        "fresh_embedding_batches_this_invocation": len(done) - resumed_batches,
        "resumed_embedding_batches_this_invocation": resumed_batches,
        "fresh_requested_vectors_this_invocation": usage["requested_vectors"] - resumed_usage["requested_vectors"],
        "resumed_requested_vectors_this_invocation": resumed_usage["requested_vectors"],
        "fresh_logical_requests_this_invocation": usage["logical_requests"] - resumed_usage["logical_requests"],
        "fresh_api_input_tokens_this_invocation": usage["api_input_tokens"] - resumed_usage["api_input_tokens"],
        "api_usage_basis": "API-reported total_tokens summed over successful logical requests; HTTP attempts include SDK retries.",
        "preservation_verified": True,
    }
    # Validate through the consumer contract without ever publishing premature
    # success. This pending artifact is also confined to the ignored cache.
    pending = output / "manifest.pending.json"
    write_json(pending, result)
    load_input(pending)
    verify_sources()
    write_json(output / "readiness.json", safe_summary(result))
    pending.replace(manifest_path)
    return result
