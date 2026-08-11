from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sampling_comparison.v2_experiment import load_combined_dataset
from sampling_comparison.v3_experiment import (
    Deterministic1536Embedder,
    V3_DEFAULT_EMBEDDING_BATCH_SIZE,
    V3_EMBEDDING_DIMENSIONS,
    V3_EMBEDDING_MODEL,
    build_v3_runtime,
)
from sampling_comparison.v3_report import default_inputs, load_v3_artifacts, validate_v3_artifacts
from sampling_comparison.v4_idw import (
    IDWConfig,
    estimate_embedding_population,
    freeze_membership,
    leave_one_out_donor_diagnostics,
    validate_embedding_population,
)
from trace_sampling.azure_config import AzureConfig
from trace_sampling.embedding import AzureOpenAIEmbedder


SCRIPT_VERSION = "tune-sampling-v4-idw-v1"
EXPECTED_METHOD = "adaptive_embedding_fullsession_token"
EXPECTED_EMBEDDING_CELLS = 15

DEFAULT_SOURCE_DIR = Path("outputs_sampling_v4") / "runs" / "full-20260805" / "source_v3"
DEFAULT_OUTPUT_PATH = Path("outputs_sampling_v4") / "runs" / "full-20260805" / "idw-parameter-sweep.json"

DEFAULT_K_GRID = "1,2,3,5,8,16,32"
DEFAULT_POWER_GRID = "1,1.5,2,3,4"
DEFAULT_EPS_GRID = "1e-6,1e-3,1e-2"
DEFAULT_EXACT_COSINE_EPS_GRID = "1e-8"
DEFAULT_PRIOR_GRID = "0.5"


@dataclass(frozen=True)
class EvalCell:
    cell_id: str
    budget_tokens: int
    legacy_tier_pct: int
    repetition: int
    order_hash: str
    selected_count: int
    selected_only_abs_error: float
    membership: Any


@dataclass(frozen=True)
class CandidateMetrics:
    config: IDWConfig
    weighted_tuning_loo_brier: float
    weighted_tuning_loo_mae: float
    weighted_holdout_loo_brier: float
    weighted_holdout_loo_mae: float


class _EmbedConfigOverride:
    def __init__(self, base: AzureConfig, deployment: str) -> None:
        self.openai_endpoint = base.openai_endpoint
        self.openai_api_version = base.openai_api_version
        self.embedding_deployment = deployment


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_csv_ints(raw: str, *, field_name: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"{field_name} contains non-integer token: {item}") from exc
        out.append(value)
    if not out:
        raise ValueError(f"{field_name} must not be empty")
    return out


def _parse_csv_floats(raw: str, *, field_name: str) -> list[float]:
    out: list[float] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"{field_name} contains non-float token: {item}") from exc
        out.append(value)
    if not out:
        raise ValueError(f"{field_name} must not be empty")
    return out


def _config_to_dict(cfg: IDWConfig) -> dict[str, Any]:
    return {
        "k": int(cfg.k),
        "power": float(cfg.power),
        "eps": float(cfg.eps),
        "exact_cosine_eps": float(cfg.exact_cosine_eps),
        "prior": float(cfg.prior),
    }


def _candidate_sort_key(metrics: CandidateMetrics) -> tuple[Any, ...]:
    cfg = metrics.config
    return (
        float(metrics.weighted_tuning_loo_brier),
        float(metrics.weighted_tuning_loo_mae),
        int(cfg.k),
        float(cfg.power),
        float(cfg.eps),
        float(cfg.exact_cosine_eps),
        abs(float(cfg.prior) - 0.5),
        float(cfg.prior),
    )


def _load_embedding_cells(
    *,
    source_dir: Path,
    eligible_ids: Sequence[str],
) -> tuple[list[EvalCell], dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifacts = load_v3_artifacts(default_inputs(source_dir))
    validate_v3_artifacts(artifacts)

    aggregate = artifacts.aggregate
    manifest = artifacts.manifest
    selected_membership = artifacts.selected_membership

    rows = [row for row in artifacts.runs_jsonl if str(row.get("method")) == EXPECTED_METHOD]
    if len(rows) != EXPECTED_EMBEDDING_CELLS:
        raise ValueError(
            "expected exactly 15 persisted adaptive_embedding_fullsession_token cells in runs.jsonl; "
            f"found {len(rows)}"
        )

    eval_cells: list[EvalCell] = []
    unique_ids: set[str] = set()
    for row in rows:
        try:
            repetition = int(row["repetition"])
            budget_tokens = int(row["budget_tokens"])
            legacy_tier_pct = int(row["legacy_tier_pct"])
            order_hash = str(row["order_hash"])
            selected_ids_raw = row["selected_ids"]
        except KeyError as exc:
            raise ValueError(f"missing required field in runs.jsonl row: {exc}") from exc

        if not isinstance(selected_ids_raw, list):
            raise ValueError("runs.jsonl selected_ids must be a list")
        selected_ids = [str(x) for x in selected_ids_raw]
        cell_id = f"budget={budget_tokens}|rep={repetition}|order={order_hash}"
        if cell_id in unique_ids:
            raise ValueError(f"duplicate embedding cell encountered: {cell_id}")
        unique_ids.add(cell_id)

        membership = freeze_membership(
            cell_id=cell_id,
            eligible_ids=eligible_ids,
            selected_ids=selected_ids,
        )
        if "absolute_error" not in row:
            raise ValueError("runs.jsonl embedding row is missing required absolute_error")
        selected_only_abs_error = float(row["absolute_error"])
        eval_cells.append(
            EvalCell(
                cell_id=cell_id,
                budget_tokens=budget_tokens,
                legacy_tier_pct=legacy_tier_pct,
                repetition=repetition,
                order_hash=order_hash,
                selected_count=len(selected_ids),
                selected_only_abs_error=selected_only_abs_error,
                membership=membership,
            )
        )

    repetitions = sorted({cell.repetition for cell in eval_cells})
    if repetitions != [0, 1, 2]:
        raise ValueError(
            "expected embedding cells to cover repetitions [0,1,2]; "
            f"found {repetitions}"
        )

    return eval_cells, aggregate, manifest, selected_membership


def _dataset_identity(data: Any) -> dict[str, Any]:
    unit_ids = [str(uid) for uid in data.unit_ids]
    return {
        "unit_count": len(unit_ids),
        "unit_ids_sha256": _sha256_text(_canonical_json(unit_ids)),
        "source_paths": dict(data.source_paths),
    }


def _build_vectors_with_runtime(
    *,
    data: Any,
    mode: str,
    embedding_model_id: str,
    embedding_deployment: str | None,
    embedding_batch_size: int,
    deterministic_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if mode == "azure":
        azure = AzureConfig.from_env()
        deployment = embedding_deployment or azure.embedding_deployment
        embedder = AzureOpenAIEmbedder(_EmbedConfigOverride(azure, deployment))
        runtime = build_v3_runtime(
            data,
            embedder=embedder,
            embedding_model_id=embedding_model_id,
            embedding_deployment_id=deployment,
            embedding_batch_size=embedding_batch_size,
            embedding_dimensions=V3_EMBEDDING_DIMENSIONS,
        )
        mode_label = "azure-authoritative"
        deployment_id = deployment
    elif mode == "deterministic":
        embedder = Deterministic1536Embedder(seed=deterministic_seed, dimensions=V3_EMBEDDING_DIMENSIONS)
        deployment_id = embedding_deployment or "deterministic-1536"
        runtime = build_v3_runtime(
            data,
            embedder=embedder,
            embedding_model_id=embedding_model_id,
            embedding_deployment_id=deployment_id,
            embedding_batch_size=embedding_batch_size,
            embedding_dimensions=V3_EMBEDDING_DIMENSIONS,
        )
        mode_label = "deterministic-non-authoritative"
    else:
        raise ValueError(f"unsupported mode: {mode}")

    vector_by_unit: dict[str, np.ndarray] = {}
    for uid in data.unit_ids:
        trace_id = int(data.trace_by_unit_id[uid].trace_id)
        vector = runtime.embedding_vector_by_trace_id.get(trace_id)
        if vector is None:
            raise ValueError(f"missing embedding vector for unit_id={uid} trace_id={trace_id}")
        vector_by_unit[str(uid)] = np.asarray(vector, dtype=np.float32)

    runtime_meta = {
        "mode_label": mode_label,
        "runtime_version": runtime.version,
        "embedding_model_id": embedding_model_id,
        "embedding_deployment_id": deployment_id,
        "embedding_profile_id": runtime.embedding_profile_id,
        "embedding_semantic_scope": runtime.embedding_semantic_scope,
        "ledger": {
            "embedding_calls": int(runtime.ledger.embedding_calls),
            "embedding_inputs": int(runtime.ledger.embedding_inputs),
            "embedding_input_tokens": int(runtime.ledger.embedding_input_tokens),
            "embedding_latency_seconds": float(runtime.ledger.embedding_latency_seconds),
            "embedding_embedder_class": str(runtime.ledger.embedding_embedder_class),
        },
        "packet_hashes_trace_count": int(len(runtime.packet_hashes_by_trace_id)),
    }
    return vector_by_unit, runtime_meta


def _load_vector_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload = np.load(path, allow_pickle=False)
    unit_ids = payload["unit_ids"]
    vectors = payload["vectors"]
    metadata_json = str(payload["metadata_json"][0])
    metadata = json.loads(metadata_json)

    if vectors.ndim != 2:
        raise ValueError("embedding cache vectors must be a 2D matrix")
    if len(unit_ids) != vectors.shape[0]:
        raise ValueError("embedding cache unit_ids length must match vectors rows")

    out: dict[str, np.ndarray] = {}
    for idx, uid in enumerate(unit_ids.tolist()):
        out[str(uid)] = np.asarray(vectors[idx], dtype=np.float32)
    return out, metadata


def _write_vector_cache(path: Path, *, vector_by_unit: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    ordered_ids = sorted(vector_by_unit)
    vectors = np.stack([vector_by_unit[uid] for uid in ordered_ids], axis=0)
    metadata_json = json.dumps(metadata, sort_keys=True, ensure_ascii=True)
    max_unit_id_length = max((len(uid) for uid in ordered_ids), default=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        unit_ids=np.asarray(ordered_ids, dtype=f"<U{max_unit_id_length}"),
        vectors=vectors.astype(np.float32),
        metadata_json=np.asarray([metadata_json], dtype=f"<U{max(1, len(metadata_json))}"),
    )


def _get_vectors(
    *,
    data: Any,
    mode: str,
    source_manifest: dict[str, Any],
    embedding_model_id: str,
    embedding_deployment: str | None,
    embedding_batch_size: int,
    deterministic_seed: int,
    embedding_cache: Path | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    dataset_identity = _dataset_identity(data)
    if mode == "azure":
        azure_config = AzureConfig.from_env()
        effective_deployment = embedding_deployment or azure_config.embedding_deployment
    else:
        effective_deployment = embedding_deployment or "deterministic-1536"
    expected_cache_meta = {
        "cache_version": 1,
        "dataset_unit_ids_sha256": dataset_identity["unit_ids_sha256"],
        "source_runs_sha256": str(source_manifest.get("artifacts", {}).get("runs_jsonl", {}).get("sha256", "")),
        "source_manifest_sha256": _sha256_text(_canonical_json(source_manifest)),
        "mode": mode,
        "embedding_model_id": embedding_model_id,
        "embedding_deployment_id": effective_deployment,
        "deterministic_seed": int(deterministic_seed) if mode == "deterministic" else None,
        "dimensions": V3_EMBEDDING_DIMENSIONS,
    }

    if embedding_cache is not None and embedding_cache.is_file():
        cached_vectors, cached_meta = _load_vector_cache(embedding_cache)
        cache_ok = (
            int(cached_meta.get("cache_version", -1)) == int(expected_cache_meta["cache_version"])
            and str(cached_meta.get("dataset_unit_ids_sha256", "")) == expected_cache_meta["dataset_unit_ids_sha256"]
            and str(cached_meta.get("source_runs_sha256", "")) == expected_cache_meta["source_runs_sha256"]
            and str(cached_meta.get("source_manifest_sha256", "")) == expected_cache_meta["source_manifest_sha256"]
            and str(cached_meta.get("mode", "")) == expected_cache_meta["mode"]
            and str(cached_meta.get("embedding_model_id", "")) == expected_cache_meta["embedding_model_id"]
            and str(cached_meta.get("embedding_deployment_id", "")) == expected_cache_meta["embedding_deployment_id"]
            and cached_meta.get("deterministic_seed") == expected_cache_meta["deterministic_seed"]
            and int(cached_meta.get("dimensions", -1)) == V3_EMBEDDING_DIMENSIONS
            and len(cached_vectors) == int(dataset_identity["unit_count"])
        )
        if cache_ok:
            missing = [uid for uid in data.unit_ids if uid not in cached_vectors]
            if not missing:
                return (
                    cached_vectors,
                    {
                        "cache": {
                            "used": True,
                            "path": str(embedding_cache),
                            "status": "hit",
                            "metadata": cached_meta,
                        },
                    },
                    dataset_identity,
                )

    vectors, runtime_meta = _build_vectors_with_runtime(
        data=data,
        mode=mode,
        embedding_model_id=embedding_model_id,
        embedding_deployment=embedding_deployment,
        embedding_batch_size=embedding_batch_size,
        deterministic_seed=deterministic_seed,
    )

    if embedding_cache is not None:
        cache_meta = {
            **expected_cache_meta,
            "created_at": _utc_now_iso(),
        }
        _write_vector_cache(embedding_cache, vector_by_unit=vectors, metadata=cache_meta)

    return (
        vectors,
        {
            "cache": {
                "used": False,
                "path": str(embedding_cache) if embedding_cache is not None else None,
                "status": "miss_or_disabled",
            },
            "runtime": runtime_meta,
        },
        dataset_identity,
    )


def _make_expected_label_maps(data: Any) -> tuple[dict[str, float], dict[str, str]]:
    expected_labels: dict[str, float] = {}
    agent_id_by_unit: dict[str, str] = {}
    for uid in data.unit_ids:
        trace = data.trace_by_unit_id[uid]
        expected_labels[str(uid)] = 1.0 if bool(data.labels_by_unit[uid]) else 0.0
        agent_id_by_unit[str(uid)] = str(trace.agent_id)
    return expected_labels, agent_id_by_unit


def _materialize_selected_expected_labels(
    membership: Any,
    expected_labels: dict[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for uid in membership.selected_ids:
        if uid not in expected_labels:
            raise ValueError(f"expected label missing for selected unit_id={uid}")
        out[uid] = float(expected_labels[uid])
    return out


def _weighted_loo(
    *,
    cells: Sequence[EvalCell],
    agent_id_by_unit: dict[str, str],
    vector_by_unit: dict[str, np.ndarray],
    expected_labels: dict[str, float],
    config: IDWConfig,
) -> tuple[float, float]:
    total_weight = 0
    weighted_brier = 0.0
    weighted_mae = 0.0

    for cell in cells:
        judged_values = _materialize_selected_expected_labels(cell.membership, expected_labels)
        loo = leave_one_out_donor_diagnostics(
            membership=cell.membership,
            agent_id_by_unit=agent_id_by_unit,
            vector_by_unit=vector_by_unit,
            judged_values_by_unit=judged_values,
            config=config,
        )
        weight = int(len(judged_values))
        total_weight += weight
        weighted_brier += float(loo.brier_score) * weight
        weighted_mae += float(loo.mae) * weight

    if total_weight <= 0:
        return 0.0, 0.0
    return weighted_brier / total_weight, weighted_mae / total_weight


def _summarize_cell_oracle(
    *,
    cell: EvalCell,
    expected_labels: dict[str, float],
    agent_id_by_unit: dict[str, str],
    vector_by_unit: dict[str, np.ndarray],
    config: IDWConfig,
) -> dict[str, Any]:
    judged_values = _materialize_selected_expected_labels(cell.membership, expected_labels)
    estimates = estimate_embedding_population(
        membership=cell.membership,
        agent_id_by_unit=agent_id_by_unit,
        vector_by_unit=vector_by_unit,
        judged_values_by_unit=judged_values,
        config=config,
    )
    validation = validate_embedding_population(estimates, expected_labels)

    idw_abs = float(validation.absolute_aggregate_rate_error)
    selected_abs = float(cell.selected_only_abs_error)
    delta = idw_abs - selected_abs
    if abs(delta) < 1e-12:
        selected_cmp = "tied"
    elif delta < 0.0:
        selected_cmp = "improved"
    else:
        selected_cmp = "regressed"

    return {
        "cell_id": cell.cell_id,
        "budget_tokens": int(cell.budget_tokens),
        "legacy_tier_pct": int(cell.legacy_tier_pct),
        "repetition": int(cell.repetition),
        "selected_count": int(cell.selected_count),
        "selected_only_abs_error": selected_abs,
        "idw_abs_error": idw_abs,
        "idw_minus_selected_only_abs_error": delta,
        "selected_only_comparison": selected_cmp,
        "unjudged_mae": validation.unjudged_only_mae,
        "unjudged_brier": validation.unjudged_only_brier,
        "ece": float(validation.expected_calibration_error),
        "zero_donor_agent_count": int(estimates.aggregate.zero_donor_agent_count),
        "prior_count": int(estimates.aggregate.prior_count),
        "provenance_counts": {k: int(v) for k, v in estimates.aggregate.provenance_counts.items()},
    }


def _mean(values: Iterable[float | None]) -> float | None:
    collected = [float(v) for v in values if v is not None]
    if not collected:
        return None
    return float(np.mean(np.asarray(collected, dtype=np.float64)))


def _aggregate_oracle_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    improved = sum(1 for row in rows if row["selected_only_comparison"] == "improved")
    regressed = sum(1 for row in rows if row["selected_only_comparison"] == "regressed")
    tied = sum(1 for row in rows if row["selected_only_comparison"] == "tied")
    return {
        "cell_count": len(rows),
        "idw_abs_error_mean": _mean(row["idw_abs_error"] for row in rows),
        "selected_only_abs_error_mean": _mean(row["selected_only_abs_error"] for row in rows),
        "idw_minus_selected_only_abs_error_mean": _mean(
            row["idw_minus_selected_only_abs_error"] for row in rows
        ),
        "selected_only_comparison": {
            "improved": int(improved),
            "regressed": int(regressed),
            "tied": int(tied),
        },
        "unjudged_mae_mean": _mean(row["unjudged_mae"] for row in rows),
        "unjudged_brier_mean": _mean(row["unjudged_brier"] for row in rows),
        "ece_mean": _mean(row["ece"] for row in rows),
        "zero_donor_agent_count_total": int(sum(int(row["zero_donor_agent_count"]) for row in rows)),
        "prior_count_total": int(sum(int(row["prior_count"]) for row in rows)),
    }


def _build_candidates(
    *,
    k_values: Sequence[int],
    power_values: Sequence[float],
    eps_values: Sequence[float],
    exact_eps_values: Sequence[float],
    prior_values: Sequence[float],
) -> list[IDWConfig]:
    configs: list[IDWConfig] = []
    for k in k_values:
        for power in power_values:
            for eps in eps_values:
                for exact_eps in exact_eps_values:
                    for prior in prior_values:
                        configs.append(
                            IDWConfig(
                                k=int(k),
                                power=float(power),
                                eps=float(eps),
                                exact_cosine_eps=float(exact_eps),
                                prior=float(prior),
                            )
                        )
    return configs


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune V4 IDW parameters from persisted source_v3 embedding memberships using expected labels."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--mode", choices=("azure", "deterministic"), default="azure")

    parser.add_argument("--k-grid", type=str, default=DEFAULT_K_GRID)
    parser.add_argument("--power-grid", type=str, default=DEFAULT_POWER_GRID)
    parser.add_argument("--eps-grid", type=str, default=DEFAULT_EPS_GRID)
    parser.add_argument("--exact-cosine-eps-grid", type=str, default=DEFAULT_EXACT_COSINE_EPS_GRID)
    parser.add_argument("--prior-grid", type=str, default=DEFAULT_PRIOR_GRID)

    parser.add_argument("--quick", action="store_true", help="Use a tiny grid for fast dry runs.")
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=V3_DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--embedding-model-id", type=str, default=V3_EMBEDDING_MODEL)
    parser.add_argument("--embedding-deployment", type=str, default=None)
    parser.add_argument("--deterministic-seed", type=int, default=13)
    parser.add_argument("--top-oracle", type=int, default=1)

    return parser


def main() -> None:
    args = _arg_parser().parse_args()
    t0 = perf_counter()

    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be > 0")
    if args.top_oracle <= 0:
        raise ValueError("--top-oracle must be > 0")

    k_grid = _parse_csv_ints(args.k_grid, field_name="--k-grid")
    power_grid = _parse_csv_floats(args.power_grid, field_name="--power-grid")
    eps_grid = _parse_csv_floats(args.eps_grid, field_name="--eps-grid")
    exact_eps_grid = _parse_csv_floats(args.exact_cosine_eps_grid, field_name="--exact-cosine-eps-grid")
    prior_grid = _parse_csv_floats(args.prior_grid, field_name="--prior-grid")

    if args.quick:
        k_grid = [1, 8]
        power_grid = [1.0, 2.0]
        eps_grid = [1e-6]
        exact_eps_grid = [1e-8]
        prior_grid = [0.5]

    data = load_combined_dataset()
    expected_labels, agent_id_by_unit = _make_expected_label_maps(data)

    eval_cells, aggregate, manifest, selected_membership = _load_embedding_cells(
        source_dir=args.source_dir,
        eligible_ids=data.unit_ids,
    )

    vector_by_unit, vector_provenance, dataset_identity = _get_vectors(
        data=data,
        mode=args.mode,
        source_manifest=manifest,
        embedding_model_id=args.embedding_model_id,
        embedding_deployment=args.embedding_deployment,
        embedding_batch_size=args.embedding_batch_size,
        deterministic_seed=args.deterministic_seed,
        embedding_cache=args.embedding_cache,
    )

    tuning_cells = [cell for cell in eval_cells if cell.repetition in (0, 1)]
    holdout_cells = [cell for cell in eval_cells if cell.repetition == 2]
    if len(tuning_cells) != 10 or len(holdout_cells) != 5:
        raise ValueError(
            "expected 10 tuning cells (repetitions 0/1) and 5 hold-out cells (repetition 2); "
            f"found tuning={len(tuning_cells)} holdout={len(holdout_cells)}"
        )

    candidates = _build_candidates(
        k_values=k_grid,
        power_values=power_grid,
        eps_values=eps_grid,
        exact_eps_values=exact_eps_grid,
        prior_values=prior_grid,
    )

    baseline = IDWConfig()
    all_metrics: list[CandidateMetrics] = []

    for cfg in candidates:
        tune_brier, tune_mae = _weighted_loo(
            cells=tuning_cells,
            agent_id_by_unit=agent_id_by_unit,
            vector_by_unit=vector_by_unit,
            expected_labels=expected_labels,
            config=cfg,
        )
        hold_brier, hold_mae = _weighted_loo(
            cells=holdout_cells,
            agent_id_by_unit=agent_id_by_unit,
            vector_by_unit=vector_by_unit,
            expected_labels=expected_labels,
            config=cfg,
        )
        all_metrics.append(
            CandidateMetrics(
                config=cfg,
                weighted_tuning_loo_brier=tune_brier,
                weighted_tuning_loo_mae=tune_mae,
                weighted_holdout_loo_brier=hold_brier,
                weighted_holdout_loo_mae=hold_mae,
            )
        )

    ranked = sorted(all_metrics, key=_candidate_sort_key)
    winner = ranked[0]

    baseline_metrics: CandidateMetrics | None = None
    for item in ranked:
        if _config_to_dict(item.config) == _config_to_dict(baseline):
            baseline_metrics = item
            break
    if baseline_metrics is None:
        base_tune_brier, base_tune_mae = _weighted_loo(
            cells=tuning_cells,
            agent_id_by_unit=agent_id_by_unit,
            vector_by_unit=vector_by_unit,
            expected_labels=expected_labels,
            config=baseline,
        )
        base_hold_brier, base_hold_mae = _weighted_loo(
            cells=holdout_cells,
            agent_id_by_unit=agent_id_by_unit,
            vector_by_unit=vector_by_unit,
            expected_labels=expected_labels,
            config=baseline,
        )
        baseline_metrics = CandidateMetrics(
            config=baseline,
            weighted_tuning_loo_brier=base_tune_brier,
            weighted_tuning_loo_mae=base_tune_mae,
            weighted_holdout_loo_brier=base_hold_brier,
            weighted_holdout_loo_mae=base_hold_mae,
        )

    top_oracle_candidates = [item.config for item in ranked[: max(1, args.top_oracle)]]
    oracle_configs: list[IDWConfig] = []
    seen = set()
    for cfg in [baseline_metrics.config, winner.config, *top_oracle_candidates]:
        key = _canonical_json(_config_to_dict(cfg))
        if key not in seen:
            seen.add(key)
            oracle_configs.append(cfg)

    full_oracle: dict[str, Any] = {}
    for cfg in oracle_configs:
        per_cell_rows = [
            _summarize_cell_oracle(
                cell=cell,
                expected_labels=expected_labels,
                agent_id_by_unit=agent_id_by_unit,
                vector_by_unit=vector_by_unit,
                config=cfg,
            )
            for cell in eval_cells
        ]

        by_budget: dict[int, list[dict[str, Any]]] = {}
        for row in per_cell_rows:
            by_budget.setdefault(int(row["budget_tokens"]), []).append(row)

        budget_summary = {
            str(budget): _aggregate_oracle_rows(rows)
            for budget, rows in sorted(by_budget.items(), key=lambda item: item[0])
        }

        cfg_key = _canonical_json(_config_to_dict(cfg))
        full_oracle[cfg_key] = {
            "config": _config_to_dict(cfg),
            "aggregate": _aggregate_oracle_rows(per_cell_rows),
            "by_budget": budget_summary,
            "per_cell": per_cell_rows,
        }

    ranked_rows = [
        {
            "rank": idx + 1,
            "config": _config_to_dict(item.config),
            "weighted_tuning_loo_brier": float(item.weighted_tuning_loo_brier),
            "weighted_tuning_loo_mae": float(item.weighted_tuning_loo_mae),
            "weighted_holdout_loo_brier": float(item.weighted_holdout_loo_brier),
            "weighted_holdout_loo_mae": float(item.weighted_holdout_loo_mae),
        }
        for idx, item in enumerate(ranked)
    ]

    source_pre_run = (
        aggregate.get("config", {}).get("source_v3_pre_run", {})
        if isinstance(aggregate, dict)
        else {}
    )

    caveats = [
        "Expected labels act as a synthetic pseudo-judge; tuning on these labels can overfit this retained corpus.",
    ]
    if args.mode == "deterministic":
        caveats.append(
            "Deterministic mode uses Deterministic1536Embedder and does not reproduce authoritative Azure V4 embeddings/results."
        )

    report = {
        "version": SCRIPT_VERSION,
        "generated_at": _utc_now_iso(),
        "mode": {
            "requested": args.mode,
            "label": "azure-authoritative" if args.mode == "azure" else "deterministic-non-authoritative",
            "authoritative": args.mode == "azure",
            "deterministic_seed": int(args.deterministic_seed) if args.mode == "deterministic" else None,
        },
        "source": {
            "source_dir": str(args.source_dir),
            "manifest_sha256": _sha256_text(_canonical_json(manifest)),
            "manifest_artifacts": manifest.get("artifacts", {}),
            "selected_membership_sha256": _sha256_text(_canonical_json(selected_membership)),
            "source_v3_pre_run": source_pre_run,
            "dataset_identity": dataset_identity,
        },
        "embedding_vector_provenance": vector_provenance,
        "grid": {
            "k": k_grid,
            "power": power_grid,
            "eps": eps_grid,
            "exact_cosine_eps": exact_eps_grid,
            "prior": prior_grid,
            "candidate_count": len(candidates),
            "quick": bool(args.quick),
        },
        "selection_protocol": {
            "frozen_membership_before_expected_labels": True,
            "embedding_cells_expected": EXPECTED_EMBEDDING_CELLS,
            "embedding_cells_used": len(eval_cells),
            "tuning_repetitions": [0, 1],
            "holdout_repetition": 2,
            "criterion": "judged-count-weighted leave-one-out Brier on tuning repetitions",
            "tie_break": [
                "judged-count-weighted leave-one-out MAE",
                "smaller k",
                "lower power",
                "lower eps",
                "lower exact_cosine_eps",
                "prior closest to 0.5",
            ],
            "top_oracle": int(args.top_oracle),
            "all_labels_restricted_to_selected_for_judged_values": True,
            "full_labels_used_only_for_post_estimation_validation": True,
        },
        "winner": {
            "config": _config_to_dict(winner.config),
            "weighted_tuning_loo_brier": float(winner.weighted_tuning_loo_brier),
            "weighted_tuning_loo_mae": float(winner.weighted_tuning_loo_mae),
            "weighted_holdout_loo_brier": float(winner.weighted_holdout_loo_brier),
            "weighted_holdout_loo_mae": float(winner.weighted_holdout_loo_mae),
        },
        "baseline": {
            "config": _config_to_dict(baseline_metrics.config),
            "weighted_tuning_loo_brier": float(baseline_metrics.weighted_tuning_loo_brier),
            "weighted_tuning_loo_mae": float(baseline_metrics.weighted_tuning_loo_mae),
            "weighted_holdout_loo_brier": float(baseline_metrics.weighted_holdout_loo_brier),
            "weighted_holdout_loo_mae": float(baseline_metrics.weighted_holdout_loo_mae),
        },
        "heldout_metrics": {
            "winner": {
                "weighted_loo_brier": float(winner.weighted_holdout_loo_brier),
                "weighted_loo_mae": float(winner.weighted_holdout_loo_mae),
            },
            "baseline": {
                "weighted_loo_brier": float(baseline_metrics.weighted_holdout_loo_brier),
                "weighted_loo_mae": float(baseline_metrics.weighted_holdout_loo_mae),
            },
        },
        "ranked_candidates": ranked_rows,
        "full_oracle_metrics": full_oracle,
        "interpretation": {
            "summary": "Winner selected by weighted LOO Brier on repetitions 0/1; repetition 2 is held out for selection evaluation.",
            "caveats": caveats,
        },
        "runtime_seconds": perf_counter() - t0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(
        "[idw-sweep] "
        f"wrote={args.output} "
        f"mode={report['mode']['label']} "
        f"candidates={len(candidates)} "
        f"winner={_canonical_json(_config_to_dict(winner.config))} "
        f"winner_tune_brier={winner.weighted_tuning_loo_brier:.6f} "
        f"winner_holdout_brier={winner.weighted_holdout_loo_brier:.6f} "
        f"baseline_holdout_brier={baseline_metrics.weighted_holdout_loo_brier:.6f} "
        f"runtime_s={report['runtime_seconds']:.2f}"
    )


if __name__ == "__main__":
    main()
