from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

from trace_sampling.vector_store import VectorStore

from .v2_experiment import CombinedDataset
from .v3_experiment import (
    V3Runtime,
    run_v3_outcome_comparison,
    run_v3_quadrant_experiment,
    run_v3_throughput_grid_experiment,
    select_v3_membership,
)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_canonical_json(dict(payload)) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + "\n")
    os.replace(tmp, path)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _concept_key(meta: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(meta.get("corpus_id") or "unknown"),
            str(meta.get("domain") or "unknown"),
            str(meta.get("task") or "unknown"),
            str(meta.get("difficulty") or "unknown"),
        )
    )


def _corpus_audit_v3(data: CombinedDataset) -> dict[str, Any]:
    units_by_id = {str(unit.unit_id or ""): unit for unit in data.units}
    source_files: dict[str, Any] = {}
    for corpus_id, source_path in data.source_paths.items():
        unit_ids = [uid for uid in data.unit_ids if data.corpus_id_by_unit[uid] == corpus_id]
        labels = [bool(data.labels_by_unit[uid]) for uid in unit_ids]
        agents = {
            f"{units_by_id[uid].tenant_id}|{units_by_id[uid].agent_id}"
            for uid in unit_ids
            if uid in units_by_id
        }
        concepts = {_concept_key(data.metadata_by_unit[uid]) for uid in unit_ids}
        path = Path(source_path)
        source_files[corpus_id] = {
            "path": source_path,
            "sha256": (_sha256_file(path) if path.exists() else None),
            "counts": {
                "units": len(unit_ids),
                "labels": len(unit_ids),
                "agents": len(agents),
                "concepts": len(concepts),
            },
            "label_pass_rate": (sum(1 for x in labels if x) / len(labels)) if labels else 0.0,
        }

    all_labels = [bool(data.labels_by_unit[uid]) for uid in data.unit_ids]
    all_concepts = {_concept_key(data.metadata_by_unit[uid]) for uid in data.unit_ids}
    return {
        "version": "sampling-v3-corpus-audit-v1",
        "source_files": source_files,
        "combined": {
            "counts": {
                "units": len(data.unit_ids),
                "labels": len(data.unit_ids),
                "agents": len(data.scoped_identities),
                "concepts": len(all_concepts),
            },
            "label_pass_rate": (sum(1 for x in all_labels if x) / len(all_labels)) if all_labels else 0.0,
        },
    }


def _build_token_inventory(runtime: V3Runtime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for uid, rec in sorted(runtime.packet_records_by_unit_id.items()):
        rows.append(
            {
                "unit_id": uid,
                "content_sha256": rec.content_sha256,
                "original_tokens": int(rec.original_tokens),
                "emitted_tokens": int(rec.emitted_tokens),
                "truncated": bool(rec.truncated),
            }
        )
    return rows


def _embedding_ledger(runtime: V3Runtime) -> dict[str, Any]:
    return {
        "packet_builds": int(runtime.ledger.packet_builds),
        "packet_cache_hits": int(runtime.ledger.packet_cache_hits),
        "embedding_calls": int(runtime.ledger.embedding_calls),
        "embedding_inputs": int(runtime.ledger.embedding_inputs),
        "embedding_input_tokens": int(runtime.ledger.embedding_input_tokens),
        "embedding_latency_seconds": float(runtime.ledger.embedding_latency_seconds),
        "embedding_content_hash_count": len(runtime.ledger.embedding_content_hashes),
        "embedding_model_id": runtime.ledger.embedding_model_id,
        "embedding_deployment_id": runtime.ledger.embedding_deployment_id,
        "embedding_embedder_class": runtime.ledger.embedding_embedder_class,
    }


def _code_hashes() -> dict[str, str | None]:
    repo_root = Path(__file__).resolve().parents[1]
    rel_paths = (
        "sampling_comparison/v3_experiment.py",
        "sampling_comparison/v3_outputs.py",
        "sampling_comparison/v3_report.py",
        "sampling_comparison/v2_experiment.py",
        "scripts/run_sampling_v3.py",
        "scripts/build_sampling_v3_report.py",
        "trace_sampling/token_representation.py",
        "trace_sampling/samplers.py",
        "trace_sampling/stats.py",
        "trace_sampling/reservoir.py",
        "trace_sampling/backpressure.py",
        "trace_sampling/variety.py",
        "trace_sampling/cluster_index.py",
        "trace_sampling/vector_store.py",
        "trace_sampling/model.py",
        "trace_sampling/session_embedding.py",
        "trace_sampling/embedding.py",
        "trace_sampling/azure_config.py",
        "minhash_sampling/config.py",
        "minhash_sampling/index.py",
        "minhash_sampling/signature.py",
        "random_sampling/datasets.py",
        "random_sampling/agent365_otel.py",
        "random_sampling/models.py",
    )
    out: dict[str, str | None] = {}
    for rel in rel_paths:
        path = repo_root / rel
        out[rel] = _sha256_file(path) if path.exists() else None
    return out


def _methodology_delta(runtime: V3Runtime) -> str:
    packet_records = list(runtime.packet_records_by_unit_id.values())
    total_packets = len(packet_records)
    truncated_packets = sum(1 for rec in packet_records if bool(rec.truncated))
    max_emitted_tokens = max((int(rec.emitted_tokens) for rec in packet_records), default=0)
    cap_binding = "binding" if truncated_packets > 0 else "non-binding"
    return "\n".join(
        [
            "# V3 Methodology Delta",
            "",
            "- Live embeddings are 1536-dimensional Foundry/Azure OpenAI vectors with explicit deployment provenance.",
            "- Embedding-cell novelty uses live scoped Azure AI Search HNSW filtered by tenant, run scope, and semantic scope.",
            "- A 4096-entry exact recent-leader buffer resolves many decisions before HNSW lookup and shields index lag effects.",
            "- Token representation is token-v3 packetized evidence with max packet tokens of 8191.",
            "- Budgets are exact token-mass floors derived from eligible population token mass.",
            "- V3 does not use Cochran sample sizing or finite-population correction; every arm packs as many whole sessions as its exact token budget permits.",
            "- Sessions are indivisible token units; membership is maximal feasible under exact budget slack constraints.",
            "- Adaptive policy is native proposal followed by deterministic maximal fill under remaining budget.",
            "- Random arm is descriptive only; no probability-based confidence interval is reported.",
            "- Expected labels are joined after label-blind membership selection for diagnostics/scoring only.",
            "- Legacy percent tiers are stored only as provenance for conversion to exact token budgets.",
            "- Live resource and threshold limits are explicit: embedding/vector dim 1536, packet max 8191, tau 0.55.",
            (
                "- Packet cap binding check from runtime inventory: "
                f"{truncated_packets}/{total_packets} packets truncated; max emitted tokens {max_emitted_tokens}; "
                f"cap is {cap_binding}."
            ),
            "",
            "## Runtime Provenance",
            "",
            f"- token_profile_id: {runtime.token_profile_id}",
            f"- minhash_profile_id: {runtime.minhash_profile_id}",
            f"- embedding_profile_id: {runtime.embedding_profile_id}",
            f"- embedding_semantic_scope: {runtime.embedding_semantic_scope}",
        ]
    ) + "\n"


def _artifact_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest payload must be an object")
    if str(payload.get("version")) != "sampling-v3-manifest-v1":
        raise ValueError("manifest version must be sampling-v3-manifest-v1")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts must be an object")
    return payload


def register_manifest_artifact(*, manifest_path: Path, key: str, artifact_path: Path) -> dict[str, Any]:
    payload = _load_manifest(manifest_path)
    artifacts = dict(payload.get("artifacts") or {})
    artifacts[str(key)] = _artifact_meta(artifact_path)
    payload["artifacts"] = artifacts
    _write_json_atomic(manifest_path, payload)
    return dict(artifacts[str(key)])


def write_run_source_manifest(
    *,
    output_dir: str | Path,
    pre_run_source_hashes: Mapping[str, str | None],
    branch: str,
    captured_at: str,
    note: str,
    manifest_path: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    path = out / "run_source_manifest.json"
    payload = {
        "version": "sampling-v3-run-source-manifest-v1",
        "captured_at": str(captured_at),
        "branch": str(branch),
        "note": str(note),
        "source_hashes": dict(pre_run_source_hashes),
    }
    _write_json_atomic(path, payload)
    entry = register_manifest_artifact(
        manifest_path=Path(manifest_path),
        key="run_source_manifest",
        artifact_path=path,
    )
    return {
        "path": str(path),
        "manifest_entry": entry,
    }


def write_search_cleanup_audit(
    *,
    output_dir: str | Path,
    tenant_id: str,
    checked_at: str,
    remaining_count: int,
    scopes: Mapping[str, Any],
    manifest_path: str | Path,
    allow_nonzero: bool = False,
) -> dict[str, Any]:
    remaining = int(remaining_count)
    if remaining != 0 and not bool(allow_nonzero):
        raise ValueError(
            "search cleanup audit requires remaining_count == 0 unless allow_nonzero=True; "
            f"got {remaining}"
        )

    out = Path(output_dir)
    path = out / "search_cleanup_audit.json"
    payload = {
        "version": "sampling-v3-search-cleanup-audit-v1",
        "tenant_id": str(tenant_id),
        "checked_at": str(checked_at),
        "remaining_count": remaining,
        "scopes": dict(scopes),
        "allow_nonzero": bool(allow_nonzero),
    }
    _write_json_atomic(path, payload)
    entry = register_manifest_artifact(
        manifest_path=Path(manifest_path),
        key="search_cleanup_audit",
        artifact_path=path,
    )
    return {
        "path": str(path),
        "manifest_entry": entry,
    }


def run_v3_experiment_bundle(
    *,
    runtime: V3Runtime,
    data: CombinedDataset,
    output_dir: str | Path,
    vector_store_factory: Callable[[str, str], VectorStore] | None,
    outcome_repetitions: int = 3,
    quadrant_replays: int = 3,
    throughput_replays: int = 2,
    legacy_outcome_tiers_pct: Sequence[int] = (5, 10, 20, 30, 50),
    legacy_quadrant_tiers_pct: Sequence[int] = (15, 30),
    throughput_arrival_rates_sessions_per_second: Sequence[float] = (0.25, 1.0, 4.0, 16.0),
    throughput_capacity_rates_sessions_per_second: Sequence[float] = (0.25, 1.0, 4.0, 16.0),
    seed: int = 13,
    tenant_id: str = "sampling-v3-experiment",
    cleanup_max_attempts: int = 10,
    cleanup_settle_seconds: float = 0.0,
    skip_quadrant: bool = False,
    skip_throughput: bool = False,
    aggregate_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    outcome = run_v3_outcome_comparison(
        data,
        runtime=runtime,
        methods=(
            "random_sampling_token_priority",
            "adaptive_minhash_32x4_token",
            "adaptive_embedding_fullsession_token",
        ),
        legacy_outcome_tiers_pct=legacy_outcome_tiers_pct,
        repetitions=outcome_repetitions,
        seed=seed,
        vector_store_factory=vector_store_factory,
        tenant_id=tenant_id,
        cleanup_max_attempts=cleanup_max_attempts,
        cleanup_settle_seconds=cleanup_settle_seconds,
    )

    quadrant = None
    if not skip_quadrant:
        quadrant = run_v3_quadrant_experiment(
            data,
            runtime=runtime,
            methods=(
                "random_sampling_token_priority",
                "adaptive_minhash_32x4_token",
                "adaptive_embedding_fullsession_token",
            ),
            legacy_quadrant_tiers_pct=legacy_quadrant_tiers_pct,
            replay_count=quadrant_replays,
            seed=seed,
            tenant_id=tenant_id,
            vector_store_factory=vector_store_factory,
            cleanup_max_attempts=cleanup_max_attempts,
            cleanup_settle_seconds=cleanup_settle_seconds,
        )

    throughput = None
    if not skip_throughput:
        throughput = run_v3_throughput_grid_experiment(
            data,
            runtime=runtime,
            methods=(
                "random_sampling_token_priority",
                "adaptive_minhash_32x4_token",
                "adaptive_embedding_fullsession_token",
            ),
            legacy_budget_tiers_pct=legacy_quadrant_tiers_pct,
            arrival_rates_sessions_per_second=throughput_arrival_rates_sessions_per_second,
            eval_capacity_rates_sessions_per_second=throughput_capacity_rates_sessions_per_second,
            replay_count=throughput_replays,
            seed=seed,
            tenant_id=tenant_id,
            vector_store_factory=vector_store_factory,
            cleanup_max_attempts=cleanup_max_attempts,
            cleanup_settle_seconds=cleanup_settle_seconds,
        )

    corpus_audit = _corpus_audit_v3(data)
    token_inventory = _build_token_inventory(runtime)

    mid_budget = int((20 / 100.0) * sum(runtime.token_cost_by_unit_id[uid] for uid in data.unit_ids))
    selected_membership = select_v3_membership(
        data,
        runtime=runtime,
        method="adaptive_embedding_fullsession_token",
        eligible_unit_ids=data.unit_ids,
        budget_tokens=mid_budget,
        seed=seed,
        tenant_id=tenant_id,
        run_scope=f"v3-membership-20-{seed}",
        vector_store_factory=vector_store_factory,
        cleanup_max_attempts=cleanup_max_attempts,
        cleanup_settle_seconds=cleanup_settle_seconds,
    )

    budget_manifest = {
        "outcome": {
            "eligible_token_mass": int(outcome["eligible_token_mass"]),
            "legacy_outcome_tiers_pct": list(legacy_outcome_tiers_pct),
        },
        "quadrant": {
            "legacy_quadrant_tiers_pct": list(legacy_quadrant_tiers_pct),
        },
        "throughput": {
            "arrival_rates_sessions_per_second": [float(x) for x in throughput_arrival_rates_sessions_per_second],
            "capacity_rates_sessions_per_second": [float(x) for x in throughput_capacity_rates_sessions_per_second],
        },
    }

    runtime_seconds = perf_counter() - started
    generated_at = _iso_utc_now()
    cfg = dict(aggregate_config or {})
    cfg["selection_budget_policy"] = {
        "unit": "tokens",
        "cochran_sample_sizing": False,
        "finite_population_correction": False,
        "sessions_are_indivisible": True,
        "selection_rule": "maximal_feasible_greedy_pack",
    }
    cfg.setdefault("runtime", {})
    cfg["runtime"]["actual_wall_runtime_seconds"] = float(runtime_seconds)
    cfg["runtime"]["embedding_ledger"] = _embedding_ledger(runtime)

    aggregate = {
        "version": "sampling-v3-bundle-v1",
        "generated_at": generated_at,
        "population_count": len(data.unit_ids),
        "runtime_seconds": float(runtime_seconds),
        "config": cfg,
        "outcome": {
            "version": outcome["version"],
            "aggregate": outcome["aggregate"],
            "eligible_token_mass": outcome["eligible_token_mass"],
        },
        "quadrant": None if quadrant is None else {
            "version": quadrant["version"],
            "aggregate_groups": quadrant["aggregate_groups"],
            "quadrant_summary": quadrant["quadrants"].get("quadrant_summary", {}),
        },
        "throughput": None if throughput is None else {
            "version": throughput["version"],
            "aggregate_grid": throughput["aggregate_grid"],
            "eval_tokens_per_second_map": throughput["config"]["eval_tokens_per_second_map"],
        },
        "provenance": {
            "code_hashes": _code_hashes(),
            "source_hashes": corpus_audit.get("source_files", {}),
        },
    }

    aggregate_path = out / "aggregate.json"
    runs_path = out / "runs.jsonl"
    quadrant_path = out / "quadrant.json"
    throughput_path = out / "throughput.json"
    corpus_audit_path = out / "corpus_audit.json"
    token_inventory_path = out / "token_inventory.jsonl"
    budget_manifest_path = out / "budget_manifest.json"
    embedding_ledger_path = out / "embedding_ledger.json"
    selected_membership_path = out / "selected_membership.json"
    methodology_path = out / "methodology_delta.md"

    _write_json_atomic(aggregate_path, aggregate)
    _write_jsonl_atomic(runs_path, outcome["runs"])
    _write_json_atomic(quadrant_path, quadrant or {"version": "sampling-v3-quadrant-v1", "skipped": True})
    _write_json_atomic(throughput_path, throughput or {"version": "sampling-v3-throughput-v1", "skipped": True})
    _write_json_atomic(corpus_audit_path, corpus_audit)
    _write_jsonl_atomic(token_inventory_path, token_inventory)
    _write_json_atomic(budget_manifest_path, budget_manifest)
    _write_json_atomic(embedding_ledger_path, _embedding_ledger(runtime))
    _write_json_atomic(selected_membership_path, {
        "version": "sampling-v3-selected-membership-v1",
        "legacy_tier_pct_provenance": 20,
        "budget_tokens": int(mid_budget),
        "membership": selected_membership,
    })
    methodology_path.write_text(_methodology_delta(runtime), encoding="utf-8")

    manifest = {
        "version": "sampling-v3-manifest-v1",
        "generated_at": generated_at,
        "artifacts": {
            "aggregate": _artifact_meta(aggregate_path),
            "runs_jsonl": _artifact_meta(runs_path),
            "quadrant": _artifact_meta(quadrant_path),
            "throughput": _artifact_meta(throughput_path),
            "corpus_audit": _artifact_meta(corpus_audit_path),
            "token_inventory": _artifact_meta(token_inventory_path),
            "budget_manifest": _artifact_meta(budget_manifest_path),
            "embedding_ledger": _artifact_meta(embedding_ledger_path),
            "selected_membership": _artifact_meta(selected_membership_path),
            "methodology_delta": _artifact_meta(methodology_path),
        },
        "notes": [
            "No raw packet text or embedding vectors are persisted.",
            "V3 bundle intentionally omits V2 ExternalEvalSnapshot artifacts.",
            "Legacy percent tiers are provenance only; exact token budgets are primary axes.",
        ],
    }
    manifest_path = out / "manifest.json"
    _write_json_atomic(manifest_path, manifest)

    return {
        "aggregate": aggregate,
        "outcome": outcome,
        "quadrant": quadrant,
        "throughput": throughput,
        "corpus_audit": corpus_audit,
        "token_inventory": token_inventory,
        "budget_manifest": budget_manifest,
        "embedding_ledger": _embedding_ledger(runtime),
        "selected_membership": selected_membership,
        "output_paths": {
            "aggregate": str(aggregate_path),
            "runs_jsonl": str(runs_path),
            "quadrant": str(quadrant_path),
            "throughput": str(throughput_path),
            "corpus_audit": str(corpus_audit_path),
            "token_inventory": str(token_inventory_path),
            "budget_manifest": str(budget_manifest_path),
            "embedding_ledger": str(embedding_ledger_path),
            "selected_membership": str(selected_membership_path),
            "methodology_delta": str(methodology_path),
            "manifest": str(manifest_path),
        },
    }
