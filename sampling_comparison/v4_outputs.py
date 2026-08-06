from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from trace_sampling.vector_store import VectorStore

from .v2_experiment import CombinedDataset
from .v3_experiment import V3Runtime
from .v3_outputs import run_v3_experiment_bundle
from .v4_experiment import IDWConfig, augment_v3_outcome_with_idw


V4_BUNDLE_VERSION = "sampling-v4-bundle-v1"
V4_MANIFEST_VERSION = "sampling-v4-manifest-v1"
SOURCE_V3_MANIFEST_VERSION = "sampling-v3-manifest-v1"
SOURCE_SUBDIR = "source_v3"


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


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_canonical_json(dict(payload)) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + "\n")
    os.replace(tmp, path)


def _rel_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _artifact_meta(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _rel_path(path, root),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _default_token_budget_policy() -> dict[str, Any]:
    return {
        "unit": "tokens",
        "cochran_sample_sizing": False,
        "finite_population_correction": False,
        "sessions_are_indivisible": True,
        "selection_rule": "maximal_feasible_greedy_pack",
    }


def _code_hashes_for_v4() -> dict[str, str | None]:
    repo_root = Path(__file__).resolve().parents[1]
    rel_paths = (
        "sampling_comparison/v2_experiment.py",
        "sampling_comparison/v3_experiment.py",
        "sampling_comparison/v3_outputs.py",
        "sampling_comparison/v4_experiment.py",
        "sampling_comparison/v4_idw.py",
        "sampling_comparison/v4_outputs.py",
        "trace_sampling/token_representation.py",
        "trace_sampling/samplers.py",
        "trace_sampling/vector_store.py",
        "minhash_sampling/config.py",
        "minhash_sampling/index.py",
        "minhash_sampling/signature.py",
        "random_sampling/datasets.py",
        "random_sampling/models.py",
    )
    out: dict[str, str | None] = {}
    for rel in rel_paths:
        path = repo_root / rel
        out[rel] = _sha256_file(path) if path.exists() else None
    return out


def _load_source_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"source V3 manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source V3 manifest payload must be an object")
    version = str(payload.get("version") or "")
    if version != SOURCE_V3_MANIFEST_VERSION:
        raise ValueError(
            "source V3 manifest version must be "
            f"{SOURCE_V3_MANIFEST_VERSION}; got {version or '<empty>'}"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("source V3 manifest artifacts must be an object")
    return payload


def _sanitize_for_persistence(value: Any) -> Any:
    banned_exact = {
        "labels",
        "label_by_unit",
        "labels_by_unit",
        "vectors",
        "packet_text",
        "packet_texts",
        "donor_ids",
        "donor_distances",
        "donor_weights",
        "per_unit_estimators",
        "per_unit_estimator_rows",
        "per_unit_rows",
        "expected_labels",
    }

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            key_l = key.lower()
            if key_l in banned_exact:
                continue
            if key_l.startswith("donor_") and (
                key_l.endswith("ids") or key_l.endswith("distances") or key_l.endswith("weights")
            ):
                continue
            out[key] = _sanitize_for_persistence(v)
        return out

    if isinstance(value, list):
        return [_sanitize_for_persistence(v) for v in value]

    if isinstance(value, tuple):
        return [_sanitize_for_persistence(v) for v in value]

    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    return str(value)


def _build_methodology_delta() -> str:
    return "\n".join(
        [
            "# V4 Methodology Delta",
            "",
            "- Outcome cells preserve exact token mass budgets from live V3 execution with no budget reinterpretation.",
            "- V4 continues to use no Cochran sample sizing and no finite-population correction (FPC).",
            "- Selection remains whole-session maximal packing under exact token budgets.",
            "- Random and MinHash arms remain selected-only estimators in V4.",
            "- MinHash bucket miss is treated as novelty/no-candidate and does not trigger exhaustive scan fallback.",
            "- Embedding arm reports selected-only metrics plus judged+IDW model-assisted estimates.",
            "- IDW uses same-agent k=8 angular neighbors with the configured fallback chain from V4 IDW logic.",
            "- Deterministic expected labels are treated as pseudo-judge outputs after membership freeze.",
            "- V4 makes no design-unbiasedness claim; IDW outputs are model-assisted diagnostics.",
            "- V3 source outcomes did not already perform IDW augmentation.",
        ]
    ) + "\n"


def run_v4_experiment_bundle(
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
    idw_config: IDWConfig = IDWConfig(),
    aggregate_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    source_dir = root / SOURCE_SUBDIR
    source_bundle = run_v3_experiment_bundle(
        runtime=runtime,
        data=data,
        output_dir=source_dir,
        vector_store_factory=vector_store_factory,
        outcome_repetitions=outcome_repetitions,
        quadrant_replays=quadrant_replays,
        throughput_replays=throughput_replays,
        legacy_outcome_tiers_pct=legacy_outcome_tiers_pct,
        legacy_quadrant_tiers_pct=legacy_quadrant_tiers_pct,
        throughput_arrival_rates_sessions_per_second=throughput_arrival_rates_sessions_per_second,
        throughput_capacity_rates_sessions_per_second=throughput_capacity_rates_sessions_per_second,
        seed=seed,
        tenant_id=tenant_id,
        cleanup_max_attempts=cleanup_max_attempts,
        cleanup_settle_seconds=cleanup_settle_seconds,
        skip_quadrant=skip_quadrant,
        skip_throughput=skip_throughput,
        aggregate_config=aggregate_config,
    )

    source_manifest_path = source_dir / "manifest.json"
    source_manifest = _load_source_manifest(source_manifest_path)

    v3_outcome = source_bundle.get("outcome")
    if not isinstance(v3_outcome, Mapping):
        raise ValueError("source V3 bundle must include an outcome mapping")

    augmented_outcome = augment_v3_outcome_with_idw(
        data=data,
        runtime=runtime,
        v3_outcome=v3_outcome,
        idw_config=idw_config,
    )

    idw_payload = {
        "version": "sampling-v4-idw-config-v1",
        "idw_config": asdict(idw_config),
    }

    source_aggregate = source_bundle.get("aggregate") if isinstance(source_bundle, Mapping) else None
    source_cfg = dict(source_aggregate.get("config") or {}) if isinstance(source_aggregate, Mapping) else {}
    source_runtime_cfg = dict(source_cfg.get("runtime") or {}) if isinstance(source_cfg.get("runtime"), Mapping) else {}
    token_budget_policy = dict(source_cfg.get("selection_budget_policy") or _default_token_budget_policy())
    embedding_ledger = dict(source_bundle.get("embedding_ledger") or source_runtime_cfg.get("embedding_ledger") or {})

    cfg = dict(aggregate_config or {})
    cfg["selection_budget_policy"] = token_budget_policy
    cfg["idw_config"] = asdict(idw_config)

    source_manifest_meta = _artifact_meta(source_manifest_path, root)
    source_manifest_meta["version"] = str(source_manifest.get("version") or "")

    source_lineage = {
        "version": "sampling-v4-source-lineage-v1",
        "source_bundle_subdir": SOURCE_SUBDIR,
        "source_bundle_version": str((source_aggregate or {}).get("version") or ""),
        "source_outcome_version": str((v3_outcome or {}).get("version") or ""),
        "source_manifest": source_manifest_meta,
        "selection_rerun": False,
        "augmentation": "augment_v3_outcome_with_idw",
    }

    generated_at = _iso_utc_now()
    aggregate = {
        "version": V4_BUNDLE_VERSION,
        "generated_at": generated_at,
        "population_count": int(augmented_outcome.get("population_count") or len(data.unit_ids)),
        "runtime_seconds": float((source_aggregate or {}).get("runtime_seconds") or 0.0),
        "runtime": {
            "token_profile_id": getattr(runtime, "token_profile_id", None),
            "minhash_profile_id": getattr(runtime, "minhash_profile_id", None),
            "embedding_profile_id": getattr(runtime, "embedding_profile_id", None),
            "embedding_semantic_scope": getattr(runtime, "embedding_semantic_scope", None),
            "embedding_ledger": embedding_ledger,
        },
        "config": cfg,
        "outcome": {
            "version": str(augmented_outcome.get("version") or ""),
            "aggregate": augmented_outcome.get("aggregate"),
            "eligible_token_mass": int(augmented_outcome.get("eligible_token_mass") or 0),
        },
        "source_v3": {
            "subdir": SOURCE_SUBDIR,
            "bundle_version": str((source_aggregate or {}).get("version") or ""),
            "manifest": source_manifest_meta,
            "manifest_relative_path": _rel_path(source_manifest_path, root),
        },
        "provenance": {
            "code_hashes": _code_hashes_for_v4(),
            "token_budget_policy": token_budget_policy,
        },
        "notes": [
            "V4 executes V3 once and augments V3 outcome rows with IDW without rerunning selection.",
            "MinHash no-candidate buckets are preserved as novelty misses with no exhaustive-scan fallback.",
            "IDW results are model-assisted and not design-unbiased estimates.",
            "V3 outcomes do not already include IDW.",
        ],
    }

    runs_rows: list[dict[str, Any]] = []
    for raw_row in list(augmented_outcome.get("runs") or []):
        if not isinstance(raw_row, Mapping):
            continue
        runs_rows.append(dict(_sanitize_for_persistence(raw_row)))

    aggregate_path = root / "aggregate.json"
    runs_path = root / "runs.jsonl"
    idw_config_path = root / "idw_config.json"
    methodology_path = root / "methodology_delta.md"
    source_lineage_path = root / "source_lineage.json"
    manifest_path = root / "manifest.json"

    _write_json_atomic(aggregate_path, aggregate)
    _write_jsonl_atomic(runs_path, runs_rows)
    _write_json_atomic(idw_config_path, idw_payload)
    _write_text_atomic(methodology_path, _build_methodology_delta())
    _write_json_atomic(source_lineage_path, source_lineage)

    manifest = {
        "version": V4_MANIFEST_VERSION,
        "generated_at": generated_at,
        "artifacts": {
            "aggregate": _artifact_meta(aggregate_path, root),
            "runs_jsonl": _artifact_meta(runs_path, root),
            "idw_config": _artifact_meta(idw_config_path, root),
            "methodology_delta": _artifact_meta(methodology_path, root),
            "source_lineage": _artifact_meta(source_lineage_path, root),
            "source_v3_manifest": source_manifest_meta,
        },
        "source": {
            "source_subdir": SOURCE_SUBDIR,
            "source_manifest_version": str(source_manifest.get("version") or ""),
            "source_manifest_sha256": source_manifest_meta["sha256"],
            "source_manifest_relative_path": source_manifest_meta["path"],
        },
        "integrity": {
            "deterministic_json": True,
            "atomic_writes": True,
            "manifest_hash_basis": _sha256_text(_canonical_json({
                "aggregate": _artifact_meta(aggregate_path, root),
                "runs_jsonl": _artifact_meta(runs_path, root),
                "idw_config": _artifact_meta(idw_config_path, root),
                "methodology_delta": _artifact_meta(methodology_path, root),
                "source_lineage": _artifact_meta(source_lineage_path, root),
                "source_v3_manifest": source_manifest_meta,
            })),
        },
    }

    _write_json_atomic(manifest_path, manifest)

    return {
        "aggregate": aggregate,
        "outcome": augmented_outcome,
        "source_v3": source_bundle,
        "output_paths": {
            "aggregate": str(aggregate_path),
            "runs_jsonl": str(runs_path),
            "idw_config": str(idw_config_path),
            "methodology_delta": str(methodology_path),
            "source_lineage": str(source_lineage_path),
            "manifest": str(manifest_path),
            "source_v3_manifest": str(source_manifest_path),
        },
    }
