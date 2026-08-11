from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import floor
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sampling_comparison.v2_experiment import load_combined_dataset
from sampling_comparison.v3_experiment import (
    V3_EMBEDDING_DIMENSIONS,
    V3ReadonlyEmbeddingCache,
    _TelemetryVectorStore,
    _pack_maximal,
    _paired_permutation,
    _prepare_replay,
    _stable_rank,
)
from sampling_comparison.v3_report import default_inputs, load_v3_artifacts, validate_v3_artifacts
from trace_sampling import AdaptiveSampler, SamplerConfig
from trace_sampling.cluster_index import AzureClusterIndex
from trace_sampling.vector_store import InMemoryVectorStore

SCRIPT_VERSION = "tune-sampling-v4-coverage-v1"
EXPECTED_METHOD = "adaptive_embedding_fullsession_token"
EXPECTED_BASELINE_CELL_COUNT = 15
EXPECTED_BUDGET_COUNT = 5
EXPECTED_REPETITIONS = (0, 1, 2)

DEFAULT_SOURCE_DIR = Path("outputs_sampling_v4") / "runs" / "full-20260805" / "source_v3"
DEFAULT_VECTOR_CACHE = Path("outputs_sampling_v4") / "runs" / "full-20260805" / "idw-vectors.deterministic-seed13.npz"
DEFAULT_OUTPUT_PATH = (
    Path("outputs_sampling_v4")
    / "runs"
    / "full-20260805"
    / "coverage-parameter-sweep.deterministic.json"
)

DEFAULT_TAU_GRID = "0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.85"
DEFAULT_K_GRID = "4,8,16,32"
DEFAULT_IAT_ALPHA_GRID = "0.1,0.3,0.5"

BASELINE_TAU = 0.55
BASELINE_K = 16.0
BASELINE_IAT_ALPHA = 0.3


@dataclass(frozen=True)
class CandidateConfig:
    tau: float
    k: float
    iat_alpha: float
    recent_buffer_size: int

    def key(self) -> str:
        return (
            f"tau={self.tau:.12g}|k={self.k:.12g}|"
            f"iat_alpha={self.iat_alpha:.12g}|recent_buffer_size={self.recent_buffer_size}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau": float(self.tau),
            "k": float(self.k),
            "iat_alpha": float(self.iat_alpha),
            "recent_buffer_size": int(self.recent_buffer_size),
            "ttl_seconds": 90.0,
            "purge_every": 200,
            "embed_budget_per_tick": 10_000_000,
            "agent_floor": 0.0,
            "enforce_keep_one_floor": False,
            "native_then_fill": True,
        }


@dataclass(frozen=True)
class CellResult:
    candidate_key: str
    stage: str
    repetition: int
    budget_tokens: int
    legacy_tier_pct: int
    order_hash: str
    selection_runtime_ms: float
    decision_runtime_ms_p95: float
    decision_runtime_ms_p50: float
    selected_count: int
    native_count: int
    fill_count: int
    selected_tokens: int
    native_tokens: int
    fill_tokens: int
    slack_tokens: int
    min_unselected_token_cost: int | None
    budget_utilization_tokens: float
    fraction_saved: float
    concept_coverage: float
    zero_selection_agent_rate: float
    selected_ids_sha256: str
    maximality_unavoidable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "stage": self.stage,
            "repetition": int(self.repetition),
            "budget_tokens": int(self.budget_tokens),
            "legacy_tier_pct": int(self.legacy_tier_pct),
            "order_hash": self.order_hash,
            "selection_runtime_ms": float(self.selection_runtime_ms),
            "decision_runtime_ms_p95": float(self.decision_runtime_ms_p95),
            "decision_runtime_ms_p50": float(self.decision_runtime_ms_p50),
            "selected_count": int(self.selected_count),
            "native_count": int(self.native_count),
            "fill_count": int(self.fill_count),
            "selected_tokens": int(self.selected_tokens),
            "native_tokens": int(self.native_tokens),
            "fill_tokens": int(self.fill_tokens),
            "slack_tokens": int(self.slack_tokens),
            "min_unselected_token_cost": (
                None if self.min_unselected_token_cost is None else int(self.min_unselected_token_cost)
            ),
            "budget_utilization_tokens": float(self.budget_utilization_tokens),
            "fraction_saved": float(self.fraction_saved),
            "concept_coverage": float(self.concept_coverage),
            "zero_selection_agent_rate": float(self.zero_selection_agent_rate),
            "selected_ids_sha256": self.selected_ids_sha256,
            "maximality_unavoidable": bool(self.maximality_unavoidable),
        }


@dataclass(frozen=True)
class CandidateSummary:
    candidate_key: str
    config: CandidateConfig
    tuning_mean_concept_coverage: float
    tuning_mean_zero_selection_agent_rate: float
    tuning_mean_decision_runtime_ms_p95: float
    tuning_mean_budget_utilization_tokens: float
    holdout_mean_concept_coverage: float
    holdout_mean_zero_selection_agent_rate: float
    holdout_mean_decision_runtime_ms_p95: float
    holdout_mean_budget_utilization_tokens: float
    stage_membership: tuple[str, ...]

    def rank_key(self) -> tuple[Any, ...]:
        return (
            -float(self.tuning_mean_concept_coverage),
            float(self.tuning_mean_zero_selection_agent_rate),
            float(self.tuning_mean_decision_runtime_ms_p95),
            abs(float(self.config.tau) - BASELINE_TAU),
            float(self.config.tau),
            float(self.config.k),
            float(self.config.iat_alpha),
            int(self.config.recent_buffer_size),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "config": self.config.to_dict(),
            "stage_membership": list(self.stage_membership),
            "tuning": {
                "mean_concept_coverage": float(self.tuning_mean_concept_coverage),
                "mean_zero_selection_agent_rate": float(self.tuning_mean_zero_selection_agent_rate),
                "mean_decision_runtime_ms_p95": float(self.tuning_mean_decision_runtime_ms_p95),
                "mean_budget_utilization_tokens": float(self.tuning_mean_budget_utilization_tokens),
            },
            "holdout": {
                "mean_concept_coverage": float(self.holdout_mean_concept_coverage),
                "mean_zero_selection_agent_rate": float(self.holdout_mean_zero_selection_agent_rate),
                "mean_decision_runtime_ms_p95": float(self.holdout_mean_decision_runtime_ms_p95),
                "mean_budget_utilization_tokens": float(self.holdout_mean_budget_utilization_tokens),
            },
        }


class ScopeFactory:
    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self._counter = 0

    def next_scope(self, *, candidate_key: str, repetition: int, budget_tokens: int, stage: str) -> str:
        self._counter += 1
        key = f"{candidate_key}|rep={repetition}|budget={budget_tokens}|stage={stage}|n={self._counter}|seed={self._seed}"
        return f"coverage-{_sha256_text(key)[:20]}"


class LightweightRuntime:
    def __init__(self, *, embedding_profile_id: str, embedding_vector_by_trace_id: Mapping[int, np.ndarray], token_cost_by_unit_id: Mapping[str, int]) -> None:
        self.embedding_profile_id = embedding_profile_id
        self.embedding_vector_by_trace_id = dict(embedding_vector_by_trace_id)
        self.token_cost_by_unit_id = dict(token_cost_by_unit_id)


class _VectorCacheValidationError(ValueError):
    pass


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_csv_floats(raw: str, *, field_name: str) -> list[float]:
    vals: list[float] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"{field_name} contains non-float token: {item}") from exc
        vals.append(value)
    if not vals:
        raise ValueError(f"{field_name} must not be empty")
    return vals


def _parse_csv_ints(raw: str, *, field_name: str) -> list[int]:
    vals: list[int] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"{field_name} contains non-integer token: {item}") from exc
        vals.append(value)
    if not vals:
        raise ValueError(f"{field_name} must not be empty")
    return vals


def _concept_key(meta: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(meta.get("corpus_id") or "unknown"),
            str(meta.get("domain") or "unknown"),
            str(meta.get("task") or "unknown"),
            str(meta.get("difficulty") or "unknown"),
        )
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _dataset_identity(data: Any) -> dict[str, Any]:
    unit_ids = [str(uid) for uid in data.unit_ids]
    return {
        "unit_count": int(len(unit_ids)),
        "unit_ids_sha256": _sha256_text(_canonical_json(unit_ids)),
        "source_paths": dict(data.source_paths),
    }


def _token_cost_by_unit_from_inventory(*, artifacts: Any, data: Any) -> dict[str, int]:
    costs: dict[str, int] = {}
    for row in artifacts.token_inventory:
        uid = str(row.get("unit_id") or "")
        if not uid:
            raise ValueError("token_inventory row missing unit_id")
        emitted = int(row.get("emitted_tokens") or 0)
        if emitted <= 0:
            raise ValueError(f"token_inventory emitted_tokens must be > 0 for unit_id={uid}")
        costs[uid] = emitted

    missing = [str(uid) for uid in data.unit_ids if str(uid) not in costs]
    if missing:
        raise ValueError(f"token_inventory missing emitted token costs for {len(missing)} dataset unit ids")

    return {str(uid): int(costs[str(uid)]) for uid in data.unit_ids}


def _load_vector_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload = np.load(path, allow_pickle=False)
    required_keys = {"unit_ids", "vectors", "metadata_json"}
    if set(payload.files) != required_keys:
        raise _VectorCacheValidationError(
            f"vector cache keys must be {sorted(required_keys)}; got {sorted(payload.files)}"
        )

    unit_ids = payload["unit_ids"]
    vectors = payload["vectors"]
    metadata_json = str(payload["metadata_json"][0])
    metadata = json.loads(metadata_json)

    if vectors.ndim != 2:
        raise _VectorCacheValidationError("vectors matrix must be 2D")
    if len(unit_ids) != vectors.shape[0]:
        raise _VectorCacheValidationError("unit_ids length must match vector rows")
    if int(vectors.shape[1]) != int(V3_EMBEDDING_DIMENSIONS):
        raise _VectorCacheValidationError(
            f"vector dimension must be {V3_EMBEDDING_DIMENSIONS}, got {vectors.shape[1]}"
        )

    vector_by_unit: dict[str, np.ndarray] = {}
    for idx, uid in enumerate(unit_ids.tolist()):
        vector_by_unit[str(uid)] = np.asarray(vectors[idx], dtype=np.float32)
    return vector_by_unit, metadata


def _validate_vector_cache_identity(
    *,
    cache_meta: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_runs_sha256: str,
    deterministic_seed: int,
) -> None:
    expected_manifest_sha = _sha256_text(_canonical_json(source_manifest))

    expected = {
        "cache_version": 1,
        "dataset_unit_ids_sha256": str(dataset_identity["unit_ids_sha256"]),
        "source_runs_sha256": str(source_runs_sha256),
        "source_manifest_sha256": expected_manifest_sha,
        "mode": "deterministic",
        "deterministic_seed": int(deterministic_seed),
        "dimensions": int(V3_EMBEDDING_DIMENSIONS),
    }

    for field, value in expected.items():
        actual = cache_meta.get(field)
        if actual != value:
            raise _VectorCacheValidationError(
                f"cache metadata mismatch for {field}: expected={value!r} actual={actual!r}"
            )


def _expected_budget_rows(artifacts: Any) -> list[dict[str, int]]:
    outcome = (artifacts.budget_manifest.get("outcome") or {})
    eligible_mass = int(outcome.get("eligible_token_mass") or 0)
    legacy_tiers = list(outcome.get("legacy_outcome_tiers_pct") or [])
    if eligible_mass <= 0:
        raise ValueError("budget_manifest outcome.eligible_token_mass must be > 0")
    if len(legacy_tiers) != EXPECTED_BUDGET_COUNT:
        raise ValueError(
            "expected 5 persisted outcome budget tiers in budget_manifest; "
            f"found {len(legacy_tiers)}"
        )

    rows: list[dict[str, int]] = []
    for tier in legacy_tiers:
        pct = int(tier)
        rows.append({
            "legacy_tier_pct": pct,
            "budget_tokens": int(floor((pct / 100.0) * eligible_mass)),
        })
    return rows


def _collect_persisted_embedding_rows(artifacts: Any) -> list[dict[str, Any]]:
    rows = [row for row in artifacts.runs_jsonl if str(row.get("method")) == EXPECTED_METHOD]
    if len(rows) != EXPECTED_BASELINE_CELL_COUNT:
        raise ValueError(
            "expected exactly 15 persisted adaptive embedding rows in runs.jsonl; "
            f"found {len(rows)}"
        )
    reps = sorted({int(row["repetition"]) for row in rows})
    if tuple(reps) != EXPECTED_REPETITIONS:
        raise ValueError(
            f"expected repetitions {list(EXPECTED_REPETITIONS)} in persisted embedding rows; got {reps}"
        )
    budgets = sorted({int(row["budget_tokens"]) for row in rows})
    if len(budgets) != EXPECTED_BUDGET_COUNT:
        raise ValueError(f"expected {EXPECTED_BUDGET_COUNT} unique persisted budgets; got {budgets}")

    return rows


def _build_persisted_order_hash_by_rep(embedding_rows: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for row in embedding_rows:
        rep = int(row["repetition"])
        order_hash = str(row.get("order_hash") or "")
        if not order_hash:
            raise ValueError("persisted embedding row missing order_hash")
        existing = out.get(rep)
        if existing is None:
            out[rep] = order_hash
        elif existing != order_hash:
            raise ValueError(
                f"persisted embedding rows contain multiple order hashes for repetition={rep}: "
                f"{existing} vs {order_hash}"
            )
    return out


def _verify_paired_orders(
    *,
    data: Any,
    seed: int,
    expected_order_hash_by_rep: Mapping[int, str],
) -> dict[int, tuple[str, ...]]:
    order_by_rep: dict[int, tuple[str, ...]] = {}
    for rep in EXPECTED_REPETITIONS:
        expected_hash = str(expected_order_hash_by_rep.get(rep) or "")
        if not expected_hash:
            raise ValueError(f"missing persisted order hash for repetition={rep}")

        order = _paired_permutation(data.unit_ids, token=f"v3|outcome|seed={seed}|rep={rep}")
        actual_hash = _sha256_text("|".join(order))
        if actual_hash != expected_hash:
            raise ValueError(
                "paired replay order hash mismatch for repetition="
                f"{rep}: expected={expected_hash} actual={actual_hash}"
            )
        order_by_rep[rep] = tuple(str(uid) for uid in order)
    return order_by_rep


def _build_budget_map(expected_budget_rows: Sequence[Mapping[str, int]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for row in expected_budget_rows:
        budget = int(row["budget_tokens"])
        legacy = int(row["legacy_tier_pct"])
        mapping[budget] = legacy
    return mapping


def _selected_ids_sha256(selected_ids: Sequence[str]) -> str:
    payload = [str(uid) for uid in selected_ids]
    return _sha256_text(_canonical_json(payload))


def _zero_selection_agent_rate(data: Any, selected_ids: Sequence[str], eligible_ids: Sequence[str]) -> float:
    selected = set(selected_ids)
    by_agent: dict[str, list[str]] = {}
    for uid in eligible_ids:
        trace = data.trace_by_unit_id[uid]
        agent = str(trace.agent_id)
        by_agent.setdefault(agent, []).append(uid)

    if not by_agent:
        return 0.0

    zero_agents = 0
    for uids in by_agent.values():
        if not any(uid in selected for uid in uids):
            zero_agents += 1
    return float(zero_agents / len(by_agent))


def _concept_coverage(data: Any, selected_ids: Sequence[str], eligible_ids: Sequence[str]) -> float:
    universe = {_concept_key(data.metadata_by_unit[uid]) for uid in eligible_ids}
    covered = {_concept_key(data.metadata_by_unit[uid]) for uid in selected_ids}
    if not universe:
        return 0.0
    return float(len(covered) / len(universe))


def _select_configurable_embedding_membership(
    *,
    data: Any,
    runtime: LightweightRuntime,
    eligible_unit_ids: Sequence[str],
    budget_tokens: int,
    seed: int,
    candidate: CandidateConfig,
    tenant_id: str,
    run_scope: str,
    semantic_scope: str,
) -> CellResult:
    cfg = SamplerConfig(
        llm_throughput=1000.0,
        agent_floor=0.0,
        enforce_keep_one_floor=False,
    )

    base_store = InMemoryVectorStore()
    store = _TelemetryVectorStore(base_store)
    cache = V3ReadonlyEmbeddingCache(runtime.embedding_profile_id, runtime.embedding_vector_by_trace_id)

    index = AzureClusterIndex(
        cache,
        store,
        tau=float(candidate.tau),
        ttl=90.0,
        purge_every=200,
        embed_budget_per_tick=10_000_000,
        recent_buffer_size=int(candidate.recent_buffer_size),
        k=float(candidate.k),
        iat_alpha=float(candidate.iat_alpha),
        semantic_scope=semantic_scope,
        tenant_id=tenant_id,
        run_scope=run_scope,
    )

    store.delete_scope_settled(
        tenant_id,
        run_scope,
        semantic_scope=semantic_scope,
        max_attempts=3,
        settle_seconds=0.0,
    )

    sampler = AdaptiveSampler(cfg, seed=seed, variety_index=index, use_novelty=True)
    trace_rows = _prepare_replay(data, eligible_unit_ids)

    proposed_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    decision_latencies_ms: list[float] = []

    t_start = perf_counter()
    for uid, trace in trace_rows:
        t0 = perf_counter()
        _ = sampler.decide(trace, admit_keep=True)
        decision_latencies_ms.append((perf_counter() - t0) * 1000.0)
        obs = sampler.last_observation
        proposed = bool(sampler.last_proposed_keep)

        row = {
            "unit_id": uid,
            "novelty": float(getattr(obs, "novelty", 0.0) or 0.0),
            "rarity": float(getattr(obs, "rarity", 0.0) or 0.0),
            "rank": _stable_rank(seed, f"native|{uid}"),
        }
        if proposed:
            proposed_rows.append(row)
        else:
            rejected_rows.append(row)

    native_order = [row["unit_id"] for row in proposed_rows]
    selected_native, native_tokens = _pack_maximal(native_order, runtime.token_cost_by_unit_id, budget_tokens)
    selected_set = set(selected_native)

    rejected_rows.sort(key=lambda row: (-row["novelty"], -row["rarity"], row["rank"]))
    fill_order = [row["unit_id"] for row in rejected_rows if row["unit_id"] not in selected_set]

    remaining_budget = int(budget_tokens - native_tokens)
    selected_fill, fill_tokens = _pack_maximal(fill_order, runtime.token_cost_by_unit_id, remaining_budget)

    selected_ids = list(selected_native) + list(selected_fill)
    selected_tokens = int(native_tokens + fill_tokens)
    slack_tokens = int(budget_tokens - selected_tokens)

    unselected = [uid for uid in eligible_unit_ids if uid not in set(selected_ids)]
    min_unselected = min((runtime.token_cost_by_unit_id[uid] for uid in unselected), default=None)

    if min_unselected is not None and not (slack_tokens < int(min_unselected)):
        raise RuntimeError("selection is not maximal under token budget")

    maximality_unavoidable = bool(min_unselected is None or slack_tokens < int(min_unselected))
    budget_util = (float(selected_tokens) / float(budget_tokens)) if budget_tokens > 0 else 0.0
    if budget_util < 0.98 and not maximality_unavoidable:
        raise RuntimeError(
            "budget utilization hard fail: cell below 0.98 without maximal-packing justification "
            f"(util={budget_util:.6f})"
        )

    pop_count = len(eligible_unit_ids)
    selection_runtime_ms = (perf_counter() - t_start) * 1000.0

    store.delete_scope_settled(
        tenant_id,
        run_scope,
        semantic_scope=semantic_scope,
        max_attempts=3,
        settle_seconds=0.0,
    )

    return CellResult(
        candidate_key="",
        stage="",
        repetition=-1,
        budget_tokens=int(budget_tokens),
        legacy_tier_pct=-1,
        order_hash="",
        selection_runtime_ms=float(selection_runtime_ms),
        decision_runtime_ms_p95=_percentile(decision_latencies_ms, 95.0),
        decision_runtime_ms_p50=_percentile(decision_latencies_ms, 50.0),
        selected_count=int(len(selected_ids)),
        native_count=int(len(selected_native)),
        fill_count=int(len(selected_fill)),
        selected_tokens=int(selected_tokens),
        native_tokens=int(native_tokens),
        fill_tokens=int(fill_tokens),
        slack_tokens=int(slack_tokens),
        min_unselected_token_cost=(None if min_unselected is None else int(min_unselected)),
        budget_utilization_tokens=float(budget_util),
        fraction_saved=(1.0 - (float(len(selected_ids)) / float(pop_count))) if pop_count > 0 else 0.0,
        concept_coverage=_concept_coverage(data, selected_ids, eligible_unit_ids),
        zero_selection_agent_rate=_zero_selection_agent_rate(data, selected_ids, eligible_unit_ids),
        selected_ids_sha256=_selected_ids_sha256(selected_ids),
        maximality_unavoidable=bool(maximality_unavoidable),
    )


def _materialize_cell_result(
    *,
    base: CellResult,
    candidate_key: str,
    stage: str,
    repetition: int,
    budget_tokens: int,
    legacy_tier_pct: int,
    order_hash: str,
) -> CellResult:
    return CellResult(
        candidate_key=candidate_key,
        stage=stage,
        repetition=int(repetition),
        budget_tokens=int(budget_tokens),
        legacy_tier_pct=int(legacy_tier_pct),
        order_hash=str(order_hash),
        selection_runtime_ms=float(base.selection_runtime_ms),
        decision_runtime_ms_p95=float(base.decision_runtime_ms_p95),
        decision_runtime_ms_p50=float(base.decision_runtime_ms_p50),
        selected_count=int(base.selected_count),
        native_count=int(base.native_count),
        fill_count=int(base.fill_count),
        selected_tokens=int(base.selected_tokens),
        native_tokens=int(base.native_tokens),
        fill_tokens=int(base.fill_tokens),
        slack_tokens=int(base.slack_tokens),
        min_unselected_token_cost=base.min_unselected_token_cost,
        budget_utilization_tokens=float(base.budget_utilization_tokens),
        fraction_saved=float(base.fraction_saved),
        concept_coverage=float(base.concept_coverage),
        zero_selection_agent_rate=float(base.zero_selection_agent_rate),
        selected_ids_sha256=str(base.selected_ids_sha256),
        maximality_unavoidable=bool(base.maximality_unavoidable),
    )


def _evaluate_candidate(
    *,
    stage: str,
    candidate: CandidateConfig,
    data: Any,
    runtime: LightweightRuntime,
    reps: Sequence[int],
    budgets: Sequence[int],
    budget_to_tier: Mapping[int, int],
    order_by_rep: Mapping[int, Sequence[str]],
    order_hash_by_rep: Mapping[int, str],
    scope_factory: ScopeFactory,
    seed: int,
    semantic_scope_prefix: str,
) -> list[CellResult]:
    out: list[CellResult] = []
    candidate_key = candidate.key()

    for rep in reps:
        order = tuple(order_by_rep[rep])
        order_hash = str(order_hash_by_rep[rep])
        for budget in budgets:
            tier = int(budget_to_tier[budget])
            scope = scope_factory.next_scope(
                candidate_key=candidate_key,
                repetition=rep,
                budget_tokens=budget,
                stage=stage,
            )
            semantic_scope = f"{semantic_scope_prefix}|{candidate_key}|{stage}|rep={rep}|budget={budget}"

            base = _select_configurable_embedding_membership(
                data=data,
                runtime=runtime,
                eligible_unit_ids=order,
                budget_tokens=int(budget),
                seed=int(seed + rep),
                candidate=candidate,
                tenant_id="sampling-v4-coverage",
                run_scope=scope,
                semantic_scope=semantic_scope,
            )
            row = _materialize_cell_result(
                base=base,
                candidate_key=candidate_key,
                stage=stage,
                repetition=rep,
                budget_tokens=budget,
                legacy_tier_pct=tier,
                order_hash=order_hash,
            )
            out.append(row)
    return out


def _candidate_summary(
    *,
    candidate: CandidateConfig,
    candidate_key: str,
    tuning_cells: Sequence[CellResult],
    holdout_cells: Sequence[CellResult],
    stage_membership: Sequence[str],
) -> CandidateSummary:
    return CandidateSummary(
        candidate_key=candidate_key,
        config=candidate,
        tuning_mean_concept_coverage=_mean([row.concept_coverage for row in tuning_cells]),
        tuning_mean_zero_selection_agent_rate=_mean([row.zero_selection_agent_rate for row in tuning_cells]),
        tuning_mean_decision_runtime_ms_p95=_mean([row.decision_runtime_ms_p95 for row in tuning_cells]),
        tuning_mean_budget_utilization_tokens=_mean([row.budget_utilization_tokens for row in tuning_cells]),
        holdout_mean_concept_coverage=_mean([row.concept_coverage for row in holdout_cells]),
        holdout_mean_zero_selection_agent_rate=_mean([row.zero_selection_agent_rate for row in holdout_cells]),
        holdout_mean_decision_runtime_ms_p95=_mean([row.decision_runtime_ms_p95 for row in holdout_cells]),
        holdout_mean_budget_utilization_tokens=_mean([row.budget_utilization_tokens for row in holdout_cells]),
        stage_membership=tuple(sorted(set(stage_membership))),
    )


def _budget_means(
    *,
    cells: Sequence[CellResult],
    budgets: Sequence[int],
    budget_to_tier: Mapping[int, int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for budget in budgets:
        bucket = [row for row in cells if int(row.budget_tokens) == int(budget)]
        out.append(
            {
                "budget_tokens": int(budget),
                "legacy_tier_pct": int(budget_to_tier[budget]),
                "mean_concept_coverage": _mean([row.concept_coverage for row in bucket]),
                "mean_zero_selection_agent_rate": _mean([row.zero_selection_agent_rate for row in bucket]),
                "mean_budget_utilization_tokens": _mean([row.budget_utilization_tokens for row in bucket]),
                "mean_selected_count": _mean([float(row.selected_count) for row in bucket]),
                "mean_native_count": _mean([float(row.native_count) for row in bucket]),
                "mean_fill_count": _mean([float(row.fill_count) for row in bucket]),
                "mean_fraction_saved": _mean([row.fraction_saved for row in bucket]),
                "mean_decision_runtime_ms_p95": _mean([row.decision_runtime_ms_p95 for row in bucket]),
            }
        )
    return out


def _index_by_budget(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[int(row["budget_tokens"])] = dict(row)
    return out


def _source_concept_coverage_by_method_budget(artifacts: Any) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    outcome = artifacts.aggregate.get("outcome") or {}
    aggregate_rows = outcome.get("aggregate") or []
    for row in aggregate_rows:
        method = str(row.get("method") or "")
        budget = int(row.get("budget_tokens") or 0)
        concept_obj = row.get("concept_coverage") or {}
        coverage = float(concept_obj.get("mean") or 0.0)
        out.setdefault(method, {})[budget] = coverage
    return out


def _validate_no_labels_used_statement() -> dict[str, Any]:
    return {
        "selection_uses_labels": False,
        "ranking_uses_labels": False,
        "metrics_use_labels": False,
        "selection_inputs": [
            "paired deterministic replay order",
            "token budgets",
            "session token costs",
            "trace agent ids",
            "cached embedding vectors",
            "adaptive novelty/rarity signals",
        ],
        "scoring_only_metadata": "concept coverage uses synthetic metadata key corpus_id|domain|task|difficulty",
        "disallowed_fields": [
            "labels_by_unit",
            "expected labels",
            "pass rates",
            "MAE",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune V4/V3 deterministic embedding selection for concept coverage "
            "using persisted budgets, paired replay orders, and cached vectors."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--vector-cache", type=Path, default=DEFAULT_VECTOR_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--tau-grid", type=str, default=DEFAULT_TAU_GRID)
    parser.add_argument("--k-grid", type=str, default=DEFAULT_K_GRID)
    parser.add_argument("--iat-alpha-grid", type=str, default=DEFAULT_IAT_ALPHA_GRID)
    parser.add_argument("--top-tau-count", type=int, default=2)
    parser.add_argument("--recent-buffer-size", type=int, default=4096)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = perf_counter()

    tau_grid = _parse_csv_floats(args.tau_grid, field_name="--tau-grid")
    k_grid = _parse_csv_ints(args.k_grid, field_name="--k-grid")
    iat_grid = _parse_csv_floats(args.iat_alpha_grid, field_name="--iat-alpha-grid")

    if args.top_tau_count <= 0:
        raise ValueError("--top-tau-count must be > 0")
    if args.recent_buffer_size <= 0:
        raise ValueError("--recent-buffer-size must be > 0")

    source_dir = Path(args.source_dir)
    vector_cache_path = Path(args.vector_cache)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"source dir does not exist: {source_dir}")
    if not vector_cache_path.is_file():
        raise FileNotFoundError(f"vector cache file does not exist: {vector_cache_path}")

    artifacts = load_v3_artifacts(default_inputs(source_dir))
    validate_v3_artifacts(artifacts)

    data = load_combined_dataset(enforce_integrity_counts=True)
    dataset_identity = _dataset_identity(data)

    embedding_rows = _collect_persisted_embedding_rows(artifacts)
    expected_budget_rows = _expected_budget_rows(artifacts)

    budget_tokens_from_manifest = sorted(int(row["budget_tokens"]) for row in expected_budget_rows)
    budget_tokens_from_rows = sorted({int(row["budget_tokens"]) for row in embedding_rows})
    if budget_tokens_from_manifest != budget_tokens_from_rows:
        raise ValueError(
            "persisted embedding row budgets do not match budget_manifest computed exact budgets: "
            f"manifest={budget_tokens_from_manifest} rows={budget_tokens_from_rows}"
        )

    budget_to_tier = _build_budget_map(expected_budget_rows)

    persisted_order_hash_by_rep = _build_persisted_order_hash_by_rep(embedding_rows)
    order_by_rep = _verify_paired_orders(
        data=data,
        seed=int(args.seed),
        expected_order_hash_by_rep=persisted_order_hash_by_rep,
    )

    vector_by_unit, cache_meta = _load_vector_cache(vector_cache_path)

    source_runs_sha = str((artifacts.manifest.get("artifacts") or {}).get("runs_jsonl", {}).get("sha256") or "")
    if len(source_runs_sha) != 64:
        raise ValueError("source manifest is missing a valid runs_jsonl sha256")

    _validate_vector_cache_identity(
        cache_meta=cache_meta,
        dataset_identity=dataset_identity,
        source_manifest=artifacts.manifest,
        source_runs_sha256=source_runs_sha,
        deterministic_seed=int(args.seed),
    )

    missing_uids = [uid for uid in data.unit_ids if uid not in vector_by_unit]
    if missing_uids:
        raise ValueError(f"vector cache missing {len(missing_uids)} dataset unit ids")

    vectors_by_trace_id: dict[int, np.ndarray] = {}
    for uid in data.unit_ids:
        trace_id = int(data.trace_by_unit_id[uid].trace_id)
        vectors_by_trace_id[trace_id] = np.asarray(vector_by_unit[uid], dtype=np.float32)

    token_cost_by_unit_id = _token_cost_by_unit_from_inventory(artifacts=artifacts, data=data)

    runtime = LightweightRuntime(
        embedding_profile_id=f"coverage-replay-profile|{dataset_identity['unit_ids_sha256']}",
        embedding_vector_by_trace_id=vectors_by_trace_id,
        token_cost_by_unit_id=token_cost_by_unit_id,
    )

    baseline = CandidateConfig(
        tau=float(BASELINE_TAU),
        k=float(BASELINE_K),
        iat_alpha=float(BASELINE_IAT_ALPHA),
        recent_buffer_size=int(args.recent_buffer_size),
    )

    if args.quick:
        stage1_tau_values = sorted(set([float(v) for v in tau_grid[: min(3, len(tau_grid))]] + [BASELINE_TAU]))
        tuning_reps = (0,)
        holdout_reps = (2,)
        tuning_budgets = tuple(budget_tokens_from_manifest[: min(3, len(budget_tokens_from_manifest))])
        holdout_budgets = tuple(budget_tokens_from_manifest[: min(2, len(budget_tokens_from_manifest))])
        stage2_k_values = [16]
        stage2_iat_values = [0.3]
    else:
        stage1_tau_values = sorted(set(float(v) for v in tau_grid))
        tuning_reps = (0, 1)
        holdout_reps = (2,)
        tuning_budgets = tuple(budget_tokens_from_manifest)
        holdout_budgets = tuple(budget_tokens_from_manifest)
        stage2_k_values = [int(v) for v in k_grid]
        stage2_iat_values = [float(v) for v in iat_grid]

    scope_factory = ScopeFactory(seed=int(args.seed))
    semantic_scope_prefix = (
        "sampling-v4-coverage"
        f"|dataset={dataset_identity['unit_ids_sha256']}"
        f"|cache={cache_meta.get('source_manifest_sha256', '')}"
    )

    stage1_candidates: list[CandidateConfig] = [
        CandidateConfig(
            tau=float(tau),
            k=float(BASELINE_K),
            iat_alpha=float(BASELINE_IAT_ALPHA),
            recent_buffer_size=int(args.recent_buffer_size),
        )
        for tau in stage1_tau_values
    ]
    if baseline.key() not in {c.key() for c in stage1_candidates}:
        stage1_candidates.append(baseline)

    all_cells_by_candidate: dict[str, list[CellResult]] = {}
    stage_by_candidate: dict[str, set[str]] = {}
    config_by_key: dict[str, CandidateConfig] = {}

    for candidate in stage1_candidates:
        key = candidate.key()
        config_by_key[key] = candidate
        stage_by_candidate.setdefault(key, set()).add("stage1")
        cells = _evaluate_candidate(
            stage="stage1",
            candidate=candidate,
            data=data,
            runtime=runtime,
            reps=tuning_reps,
            budgets=tuning_budgets,
            budget_to_tier=budget_to_tier,
            order_by_rep=order_by_rep,
            order_hash_by_rep=persisted_order_hash_by_rep,
            scope_factory=scope_factory,
            seed=int(args.seed),
            semantic_scope_prefix=semantic_scope_prefix,
        )
        all_cells_by_candidate.setdefault(key, []).extend(cells)

    stage1_summaries: list[CandidateSummary] = []
    for candidate in stage1_candidates:
        key = candidate.key()
        tuning_cells = [row for row in all_cells_by_candidate.get(key, []) if row.repetition in tuning_reps]
        summary = _candidate_summary(
            candidate=candidate,
            candidate_key=key,
            tuning_cells=tuning_cells,
            holdout_cells=[],
            stage_membership=sorted(stage_by_candidate.get(key, {"stage1"})),
        )
        stage1_summaries.append(summary)

    stage1_summaries_sorted = sorted(stage1_summaries, key=lambda row: row.rank_key())

    top_taus: list[float] = []
    for row in stage1_summaries_sorted:
        tau = float(row.config.tau)
        if tau not in top_taus:
            top_taus.append(tau)
        if len(top_taus) >= int(args.top_tau_count):
            break

    stage2_candidates_by_key: dict[str, CandidateConfig] = {}
    for tau in top_taus:
        for k in stage2_k_values:
            for iat in stage2_iat_values:
                cfg = CandidateConfig(
                    tau=float(tau),
                    k=float(k),
                    iat_alpha=float(iat),
                    recent_buffer_size=int(args.recent_buffer_size),
                )
                stage2_candidates_by_key[cfg.key()] = cfg

    if baseline.key() not in stage2_candidates_by_key:
        stage2_candidates_by_key[baseline.key()] = baseline

    for key, candidate in stage2_candidates_by_key.items():
        config_by_key[key] = candidate
        stage_by_candidate.setdefault(key, set()).add("stage2")
        if key in {c.key() for c in stage1_candidates}:
            continue
        cells = _evaluate_candidate(
            stage="stage2",
            candidate=candidate,
            data=data,
            runtime=runtime,
            reps=tuning_reps,
            budgets=tuning_budgets,
            budget_to_tier=budget_to_tier,
            order_by_rep=order_by_rep,
            order_hash_by_rep=persisted_order_hash_by_rep,
            scope_factory=scope_factory,
            seed=int(args.seed),
            semantic_scope_prefix=semantic_scope_prefix,
        )
        all_cells_by_candidate.setdefault(key, []).extend(cells)

    union_keys = sorted(all_cells_by_candidate.keys())
    for key in union_keys:
        candidate = config_by_key[key]
        holdout_cells = _evaluate_candidate(
            stage="holdout",
            candidate=candidate,
            data=data,
            runtime=runtime,
            reps=holdout_reps,
            budgets=holdout_budgets,
            budget_to_tier=budget_to_tier,
            order_by_rep=order_by_rep,
            order_hash_by_rep=persisted_order_hash_by_rep,
            scope_factory=scope_factory,
            seed=int(args.seed),
            semantic_scope_prefix=semantic_scope_prefix,
        )
        all_cells_by_candidate.setdefault(key, []).extend(holdout_cells)

    all_summaries: list[CandidateSummary] = []
    for key in union_keys:
        candidate = config_by_key[key]
        cells = all_cells_by_candidate[key]
        tuning_cells = [row for row in cells if row.repetition in tuning_reps]
        holdout_cells = [row for row in cells if row.repetition in holdout_reps]
        summary = _candidate_summary(
            candidate=candidate,
            candidate_key=key,
            tuning_cells=tuning_cells,
            holdout_cells=holdout_cells,
            stage_membership=sorted(stage_by_candidate.get(key, set())),
        )
        all_summaries.append(summary)

    ranked = sorted(all_summaries, key=lambda row: row.rank_key())
    winner = ranked[0]

    baseline_key = baseline.key()
    baseline_summary = next((row for row in ranked if row.candidate_key == baseline_key), None)
    if baseline_summary is None:
        raise RuntimeError("baseline candidate summary missing from ranked candidates")

    baseline_cells_all = all_cells_by_candidate[baseline_key]
    winner_cells_all = all_cells_by_candidate[winner.candidate_key]

    baseline_tuning_cells = [row for row in baseline_cells_all if row.repetition in tuning_reps]
    baseline_holdout_cells = [row for row in baseline_cells_all if row.repetition in holdout_reps]
    winner_tuning_cells = [row for row in winner_cells_all if row.repetition in tuning_reps]
    winner_holdout_cells = [row for row in winner_cells_all if row.repetition in holdout_reps]

    expected_baseline_cell_count = EXPECTED_BASELINE_CELL_COUNT
    actual_baseline_cell_count = len(
        [
            row
            for row in baseline_cells_all
            if row.repetition in EXPECTED_REPETITIONS and int(row.budget_tokens) in set(budget_tokens_from_manifest)
        ]
    )
    if not args.quick and actual_baseline_cell_count != expected_baseline_cell_count:
        raise RuntimeError(
            "baseline run does not contain expected 15 cells across 5 budgets x 3 reps: "
            f"expected={expected_baseline_cell_count} actual={actual_baseline_cell_count}"
        )

    baseline_tuning_budget = _budget_means(
        cells=baseline_tuning_cells,
        budgets=tuning_budgets,
        budget_to_tier=budget_to_tier,
    )
    winner_tuning_budget = _budget_means(
        cells=winner_tuning_cells,
        budgets=tuning_budgets,
        budget_to_tier=budget_to_tier,
    )
    baseline_holdout_budget = _budget_means(
        cells=baseline_holdout_cells,
        budgets=holdout_budgets,
        budget_to_tier=budget_to_tier,
    )
    winner_holdout_budget = _budget_means(
        cells=winner_holdout_cells,
        budgets=holdout_budgets,
        budget_to_tier=budget_to_tier,
    )

    baseline_tuning_idx = _index_by_budget(baseline_tuning_budget)
    winner_tuning_idx = _index_by_budget(winner_tuning_budget)
    baseline_holdout_idx = _index_by_budget(baseline_holdout_budget)
    winner_holdout_idx = _index_by_budget(winner_holdout_budget)

    deltas_tuning: list[dict[str, Any]] = []
    for budget in tuning_budgets:
        b = baseline_tuning_idx[budget]
        w = winner_tuning_idx[budget]
        deltas_tuning.append(
            {
                "budget_tokens": int(budget),
                "legacy_tier_pct": int(budget_to_tier[budget]),
                "winner_minus_baseline_concept_coverage": float(
                    w["mean_concept_coverage"] - b["mean_concept_coverage"]
                ),
                "winner_minus_baseline_zero_selection_agent_rate": float(
                    w["mean_zero_selection_agent_rate"] - b["mean_zero_selection_agent_rate"]
                ),
                "winner_minus_baseline_budget_utilization_tokens": float(
                    w["mean_budget_utilization_tokens"] - b["mean_budget_utilization_tokens"]
                ),
            }
        )

    deltas_holdout: list[dict[str, Any]] = []
    for budget in holdout_budgets:
        b = baseline_holdout_idx[budget]
        w = winner_holdout_idx[budget]
        deltas_holdout.append(
            {
                "budget_tokens": int(budget),
                "legacy_tier_pct": int(budget_to_tier[budget]),
                "winner_minus_baseline_concept_coverage": float(
                    w["mean_concept_coverage"] - b["mean_concept_coverage"]
                ),
                "winner_minus_baseline_zero_selection_agent_rate": float(
                    w["mean_zero_selection_agent_rate"] - b["mean_zero_selection_agent_rate"]
                ),
                "winner_minus_baseline_budget_utilization_tokens": float(
                    w["mean_budget_utilization_tokens"] - b["mean_budget_utilization_tokens"]
                ),
            }
        )

    source_cov = _source_concept_coverage_by_method_budget(artifacts)

    method_random = "random_sampling_token_priority"
    method_minhash = "adaptive_minhash_32x4_token"
    method_embedding = "adaptive_embedding_fullsession_token"

    source_comparison_rows: list[dict[str, Any]] = []
    for budget in budget_tokens_from_manifest:
        deterministic_baseline_tuning = baseline_tuning_idx.get(budget, {}).get("mean_concept_coverage")
        deterministic_winner_tuning = winner_tuning_idx.get(budget, {}).get("mean_concept_coverage")
        deterministic_baseline_holdout = baseline_holdout_idx.get(budget, {}).get("mean_concept_coverage")
        deterministic_winner_holdout = winner_holdout_idx.get(budget, {}).get("mean_concept_coverage")
        authoritative_embedding = float(source_cov.get(method_embedding, {}).get(budget, 0.0))
        source_comparison_rows.append(
            {
                "budget_tokens": int(budget),
                "legacy_tier_pct": int(budget_to_tier[budget]),
                "authoritative_azure_random_mean_concept_coverage": float(source_cov.get(method_random, {}).get(budget, 0.0)),
                "authoritative_azure_minhash_mean_concept_coverage": float(source_cov.get(method_minhash, {}).get(budget, 0.0)),
                "authoritative_azure_embedding_mean_concept_coverage": authoritative_embedding,
                "deterministic_baseline_tuning_mean_concept_coverage": deterministic_baseline_tuning,
                "deterministic_winner_tuning_mean_concept_coverage": deterministic_winner_tuning,
                "deterministic_baseline_holdout_mean_concept_coverage": deterministic_baseline_holdout,
                "deterministic_winner_holdout_mean_concept_coverage": deterministic_winner_holdout,
                "deterministic_baseline_minus_authoritative_embedding": (
                    None
                    if deterministic_baseline_tuning is None
                    else float(deterministic_baseline_tuning - authoritative_embedding)
                ),
            }
        )

    runtime_seconds = float(perf_counter() - started)

    stage1_ranked_payload = []
    for idx, row in enumerate(stage1_summaries_sorted):
        payload = row.to_dict()
        payload["holdout"] = None
        stage1_ranked_payload.append({"rank": idx + 1, **payload})

    ranked_payload = [{"rank": idx + 1, **row.to_dict()} for idx, row in enumerate(ranked)]

    output = {
        "version": SCRIPT_VERSION,
        "generated_at": _utc_now_iso(),
        "mode": {
            "selection_mode": "deterministic-non-authoritative",
            "deterministic_non_authoritative": True,
            "note": (
                "Vector cache uses deterministic embeddings and does not reproduce authoritative "
                "Azure V4 vectors."
            ),
        },
        "source": {
            "source_dir": str(source_dir),
            "source_manifest_sha256": _sha256_text(_canonical_json(artifacts.manifest)),
            "source_runs_sha256": str(source_runs_sha),
            "validated_with": [
                "sampling_comparison.v3_report.default_inputs",
                "sampling_comparison.v3_report.load_v3_artifacts",
                "sampling_comparison.v3_report.validate_v3_artifacts",
            ],
        },
        "cache": {
            "path": str(vector_cache_path),
            "metadata": cache_meta,
            "identity_validated": True,
        },
        "dataset": dataset_identity,
        "no_label_guarantee": _validate_no_labels_used_statement(),
        "protocol": {
            "goal": "Tune embedding selection for concept coverage only",
            "selection_semantics": {
                "whole_session_maximal_packing": True,
                "native_then_fill": True,
                "agent_floor": 0.0,
                "enforce_keep_one_floor": False,
                "sampler_other_defaults": "SamplerConfig defaults retained",
            },
            "deterministic_replay": {
                "seed": int(args.seed),
                "repetitions": list(EXPECTED_REPETITIONS),
                "order_hash_by_repetition": {
                    str(rep): str(persisted_order_hash_by_rep[rep]) for rep in EXPECTED_REPETITIONS
                },
                "order_hash_validation": "recomputed via _paired_permutation token v3|outcome|seed=13|rep={rep}",
            },
            "budgets": {
                "count": len(budget_tokens_from_manifest),
                "rows": [
                    {
                        "legacy_tier_pct": int(row["legacy_tier_pct"]),
                        "budget_tokens": int(row["budget_tokens"]),
                    }
                    for row in expected_budget_rows
                ],
                "baseline_expected_cells": EXPECTED_BASELINE_CELL_COUNT,
                "baseline_actual_cells": int(actual_baseline_cell_count),
            },
            "stages": {
                "stage1": {
                    "repetitions": list(tuning_reps),
                    "budgets": [int(v) for v in tuning_budgets],
                    "fixed_k": BASELINE_K,
                    "fixed_iat_alpha": BASELINE_IAT_ALPHA,
                    "tau_grid": [float(v) for v in stage1_tau_values],
                    "ranking": [
                        "higher mean concept coverage",
                        "lower mean zero-selection-agent rate",
                        "lower mean p95 decision latency",
                        "closer tau to baseline 0.55",
                    ],
                },
                "stage2": {
                    "repetitions": list(tuning_reps),
                    "budgets": [int(v) for v in tuning_budgets],
                    "top_tau_count": int(args.top_tau_count),
                    "top_taus_from_stage1": [float(v) for v in top_taus],
                    "k_grid": [int(v) for v in stage2_k_values],
                    "iat_alpha_grid": [float(v) for v in stage2_iat_values],
                    "ranking": [
                        "higher mean concept coverage",
                        "lower mean zero-selection-agent rate",
                        "lower mean p95 decision latency",
                        "closer tau to baseline 0.55",
                    ],
                },
                "holdout": {
                    "repetitions": list(holdout_reps),
                    "budgets": [int(v) for v in holdout_budgets],
                    "selection_only": True,
                    "winner_selection_uses_holdout": False,
                },
            },
            "quick_mode": bool(args.quick),
        },
        "grids": {
            "tau_grid_input": [float(v) for v in tau_grid],
            "k_grid_input": [int(v) for v in k_grid],
            "iat_alpha_grid_input": [float(v) for v in iat_grid],
            "recent_buffer_size": int(args.recent_buffer_size),
            "stage1_candidate_count": len(stage1_candidates),
            "stage2_candidate_count": len(stage2_candidates_by_key),
            "union_candidate_count": len(ranked),
        },
        "baseline": {
            "candidate_key": baseline_summary.candidate_key,
            "config": baseline_summary.config.to_dict(),
            "tuning": baseline_summary.to_dict()["tuning"],
            "holdout": baseline_summary.to_dict()["holdout"],
        },
        "winner": {
            "candidate_key": winner.candidate_key,
            "config": winner.config.to_dict(),
            "tuning": winner.to_dict()["tuning"],
            "holdout": winner.to_dict()["holdout"],
        },
        "stage1_ranked": stage1_ranked_payload,
        "all_candidates_ranked": ranked_payload,
        "per_budget": {
            "tuning": {
                "baseline_means": baseline_tuning_budget,
                "winner_means": winner_tuning_budget,
                "winner_minus_baseline": deltas_tuning,
            },
            "holdout": {
                "baseline_means": baseline_holdout_budget,
                "winner_means": winner_holdout_budget,
                "winner_minus_baseline": deltas_holdout,
            },
        },
        "comparisons_to_source_at_matched_budgets": {
            "comparison_warning": (
                "Authoritative source methods use Azure text-embedding-3-small vectors. "
                "Deterministic baseline/winner values use reconstructed deterministic vectors and are not "
                "directly comparable; the baseline-minus-source field quantifies this vector-regime gap."
            ),
            "rows": source_comparison_rows,
            "source_methods": {
                method_random: "random",
                method_minhash: "minhash",
                method_embedding: "embedding",
            },
        },
        "runtime": {
            "seconds": runtime_seconds,
            "candidate_evaluations": int(
                sum(len(rows) for rows in all_cells_by_candidate.values())
            ),
        },
        "interpretation": {
            "summary": (
                "Winner chosen by tuning-only concept coverage ranking with tie-breakers. "
                "Holdout repetition is reported for replay-order robustness and does not influence winner."
            ),
            "selection_objective_caveat": (
                "Concept metadata is synthetic scoring-only ground truth for coverage. "
                "Parameter tuning can overfit this metadata structure."
            ),
            "holdout_caveat": (
                "Holdout repetition tests deterministic paired replay-order robustness only; "
                "it does not evaluate generalization to new populations."
            ),
            "holdout_result": (
                f"Winner-minus-baseline holdout coverage was "
                f"{winner.holdout_mean_concept_coverage - baseline_summary.holdout_mean_concept_coverage:+.6f}; "
                "the tuning gain did not replicate consistently across budgets and should be treated as replay-order noise."
            ),
            "recommendation": (
                "Keep the baseline selector configuration for authoritative use until the candidate is validated "
                "with the original Azure embedding profile across additional repetitions or a new population."
            ),
            "cross_regime_warning": (
                "Do not compare deterministic tuned coverage directly with authoritative Random, MinHash, or embedding coverage. "
                "The deterministic baseline itself differs materially from the authoritative embedding baseline."
            ),
        },
        "cells": {
            "baseline": [row.to_dict() for row in sorted(baseline_cells_all, key=lambda r: (r.repetition, r.budget_tokens))],
            "winner": [row.to_dict() for row in sorted(winner_cells_all, key=lambda r: (r.repetition, r.budget_tokens))],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    baseline_tuning_cov = float(baseline_summary.tuning_mean_concept_coverage)
    winner_tuning_cov = float(winner.tuning_mean_concept_coverage)
    baseline_holdout_cov = float(baseline_summary.holdout_mean_concept_coverage)
    winner_holdout_cov = float(winner.holdout_mean_concept_coverage)

    concise = {
        "output": str(output_path),
        "winner": winner.config.to_dict(),
        "baseline_tuning_mean_concept_coverage": baseline_tuning_cov,
        "winner_tuning_mean_concept_coverage": winner_tuning_cov,
        "tuning_delta": float(winner_tuning_cov - baseline_tuning_cov),
        "baseline_holdout_mean_concept_coverage": baseline_holdout_cov,
        "winner_holdout_mean_concept_coverage": winner_holdout_cov,
        "holdout_delta": float(winner_holdout_cov - baseline_holdout_cov),
        "runtime_seconds": runtime_seconds,
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
