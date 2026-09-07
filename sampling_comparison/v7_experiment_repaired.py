from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from datetime import datetime, timezone
import math
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Mapping, Sequence

from sampling_comparison.v7_report_rich import build_v7_report_html as _canonical_build_v7_report_html

import numpy as np
from scipy.stats import t as student_t
from sklearn.decomposition import PCA

from minhash_sampling import BandedMinHashLSHIndex, MinHashConfig
from minhash_sampling.signature import MinHashRecord
from sampling_comparison.v3_experiment import V3ReadonlyMinHashProvider
from sampling_comparison.v6_experiment import (
    METHOD_IDS as V6_METHOD_IDS,
    SessionDescriptor,
    arm4_inclusion_probabilities,
    compute_trial_metrics,
    select_arm1,
    select_arm4,
    select_arm5,
)
from sampling_comparison.v7_search import AzureV7SearchAdapter
from trace_sampling.model import SessionEvent, Trace

V7_VERSION = "sampling-v7"
V7_METHOD_IDS: dict[str, str] = {
    "random": "random_sampling",
    "minhash": "minhash_lsh",
    "pca8_cosine": "pca8_idw_binary_cosine",
    "pca8_euclidean": "pca8_idw_binary_euclidean",
    "arm5": "arm5_hajek_weighted",
}
V7_METHOD_ID_ORDER: tuple[str, ...] = (
    V7_METHOD_IDS["random"],
    V7_METHOD_IDS["minhash"],
    V7_METHOD_IDS["pca8_cosine"],
    V7_METHOD_IDS["pca8_euclidean"],
    V7_METHOD_IDS["arm5"],
)
V7_BUDGET_LEVELS: tuple[int, ...] = (1, 3, 5, 10, 20)
V7_PCA_DIMENSIONS = 8
V7_DEFAULT_REPETITIONS = 10
V7_DEFAULT_BASE_SEED = 13


@dataclass(frozen=True)
class ReplayPlan:
    dataset_id: str
    repetition_index: int
    replay_seed: int
    replay_id: str
    occurrence_ids: tuple[str, ...]
    source_unit_ids: tuple[str, ...]
    source_id_by_occurrence: dict[str, str]
    frequency_by_source: dict[str, int]
    order_sha256: str
    frequency_sha256: str
    unique_source_count: int
    unique_source_fraction: float
    min_frequency: int
    max_frequency: int
    mean_frequency: float
    std_frequency: float
    duplicate_event_count: int

    def validate(self, *, source_population: Sequence[str]) -> None:
        population = tuple(str(uid) for uid in source_population)
        pop_set = set(population)
        n_events = len(self.source_unit_ids)
        if n_events != len(population):
            raise ValueError("replay must draw exactly N events from N sources")
        if len(self.occurrence_ids) != n_events:
            raise ValueError("occurrence count must equal replay event count")
        if len(set(self.occurrence_ids)) != len(self.occurrence_ids):
            raise ValueError("occurrence_ids must be unique")
        if any(uid not in pop_set for uid in self.source_unit_ids):
            raise ValueError("replay includes source_unit_ids not present in source population")
        if set(self.frequency_by_source.keys()) != pop_set:
            raise ValueError("frequency_by_source must include the full source population")
        freq_sum = int(sum(int(v) for v in self.frequency_by_source.values()))
        if freq_sum != n_events:
            raise ValueError("frequency_by_source must sum to N replay events")
        for occurrence_id, source_id in self.source_id_by_occurrence.items():
            if occurrence_id not in set(self.occurrence_ids):
                raise ValueError("source_id_by_occurrence keys must match occurrence_ids")
            if source_id not in pop_set:
                raise ValueError("source_id_by_occurrence must map to valid source IDs")


def _replay_seed(base_seed: int, repetition_index: int) -> int:
    if repetition_index <= 0:
        raise ValueError("repetition_index must be 1-based")
    return int(base_seed) + int(repetition_index) - 1


def build_replay_plan(*, dataset_id: str, ordered_source_unit_ids: Sequence[str], repetition_index: int, base_seed: int = V7_DEFAULT_BASE_SEED, replay_seed_override: int | None = None) -> ReplayPlan:
    source_ids = tuple(str(uid) for uid in ordered_source_unit_ids)
    if not source_ids:
        raise ValueError("ordered_source_unit_ids must not be empty")
    n = len(source_ids)
    replay_seed = int(replay_seed_override) if replay_seed_override is not None else _replay_seed(int(base_seed), int(repetition_index))
    rng = np.random.default_rng(replay_seed)
    draw_idx = rng.integers(0, n, size=n, dtype=np.int64)
    drawn_sources = tuple(source_ids[int(i)] for i in draw_idx.tolist())
    replay_id = f"{dataset_id}|r{int(repetition_index):03d}|s{replay_seed}"
    occurrence_ids = tuple(f"{replay_id}|e{idx:06d}" for idx in range(1, n + 1))
    source_id_by_occurrence = {occurrence_ids[i]: drawn_sources[i] for i in range(n)}
    frequency_counter = Counter(drawn_sources)
    frequency_by_source = {uid: int(frequency_counter.get(uid, 0)) for uid in source_ids}
    ordered_payload = [source_id_by_occurrence[occ] for occ in occurrence_ids]
    order_sha256 = hashlib.sha256(_compact_json(ordered_payload).encode("utf-8")).hexdigest()
    frequency_sha256 = hashlib.sha256(_compact_json(frequency_by_source).encode("utf-8")).hexdigest()
    freqs = np.asarray([frequency_by_source[uid] for uid in source_ids], dtype=np.float64)
    unique_source_count = int(np.sum(freqs > 0.0))
    plan = ReplayPlan(
        dataset_id=str(dataset_id),
        repetition_index=int(repetition_index),
        replay_seed=int(replay_seed),
        replay_id=replay_id,
        occurrence_ids=occurrence_ids,
        source_unit_ids=drawn_sources,
        source_id_by_occurrence=source_id_by_occurrence,
        frequency_by_source=frequency_by_source,
        order_sha256=order_sha256,
        frequency_sha256=frequency_sha256,
        unique_source_count=unique_source_count,
        unique_source_fraction=float(unique_source_count / n),
        min_frequency=int(np.min(freqs)),
        max_frequency=int(np.max(freqs)),
        mean_frequency=float(np.mean(freqs)),
        std_frequency=float(np.std(freqs, ddof=0)),
        duplicate_event_count=int(n - unique_source_count),
    )
    plan.validate(source_population=source_ids)
    return plan


@dataclass(frozen=True)
class BinaryPopulationResult:
    estimate_binary_pass_rate: float
    estimate_probability_pass_rate: float
    observed_count: int
    imputed_count: int
    rows: tuple[dict[str, Any], ...]
    latency_seconds: float


def _stable_hash(seed: int, *parts: Any) -> str:
    payload = "|".join([str(seed)] + [str(part) for part in parts])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_int(value: str) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_compact_json(dict(payload)) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_compact_json(dict(row)) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def derive_budget_caps(population_size: int, budget_levels: Sequence[int] = V7_BUDGET_LEVELS) -> dict[int, int]:
    size = max(0, int(population_size))
    out: dict[int, int] = {}
    for level in budget_levels:
        pct = int(level)
        if pct < 0:
            raise ValueError("budget levels must be non-negative")
        cap = int(round(size * (pct / 100.0)))
        if size > 0 and pct > 0:
            cap = max(1, cap)
        out[pct] = min(size, max(0, cap))
    return out


def fit_pca8_embeddings(*, ordered_unit_ids: Sequence[str], full_vectors_by_unit: Mapping[str, Any], n_components: int = V7_PCA_DIMENSIONS) -> dict[str, Any]:
    if int(n_components) <= 0:
        raise ValueError("n_components must be positive")
    unit_ids = [str(uid) for uid in ordered_unit_ids]
    if not unit_ids:
        raise ValueError("ordered_unit_ids must not be empty")

    rows: list[np.ndarray] = []
    for uid in unit_ids:
        if uid not in full_vectors_by_unit:
            raise ValueError(f"missing full-session embedding for unit_id={uid}")
        vec = np.asarray(full_vectors_by_unit[uid], dtype=np.float64)
        if vec.ndim != 1 or vec.size == 0:
            raise ValueError(f"embedding for unit_id={uid} must be a non-empty 1D vector")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"embedding for unit_id={uid} has non-finite values")
        rows.append(vec)

    matrix = np.vstack(rows)
    if matrix.shape[0] < int(n_components):
        raise ValueError("population is smaller than requested PCA components")

    t0 = perf_counter()
    pca = PCA(n_components=int(n_components), svd_solver="full", random_state=0)
    reduced = pca.fit_transform(matrix)
    latency_seconds = perf_counter() - t0

    reduced = np.asarray(reduced, dtype=np.float64)
    reduced_by_unit = {uid: reduced[idx].astype(np.float32) for idx, uid in enumerate(unit_ids)}
    return {
        "reduced_vectors_by_unit": reduced_by_unit,
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_.tolist()],
        "explained_variance_total": float(np.sum(pca.explained_variance_ratio_)),
        "latency_seconds": float(latency_seconds),
        "n_components": int(n_components),
        "population_count": int(matrix.shape[0]),
        "input_dimensions": int(matrix.shape[1]),
    }


def build_reduced_vector_export_documents(*, ordered_unit_ids: Sequence[str], reduced_vectors_by_unit: Mapping[str, Any], agent_id_by_unit: Mapping[str, str], run_scope: str, semantic_scope: str, vector_dimensions: int = V7_PCA_DIMENSIONS) -> list[dict[str, Any]]:
    expected_dim = int(vector_dimensions)
    if expected_dim <= 0:
        raise ValueError("vector_dimensions must be positive")
    out: list[dict[str, Any]] = []
    for uid in ordered_unit_ids:
        vec = np.asarray(reduced_vectors_by_unit[uid], dtype=np.float32)
        if vec.ndim != 1 or int(vec.size) != expected_dim:
            raise ValueError(f"reduced vector for unit_id={uid} has dim={int(vec.size)}; expected dim={expected_dim}")
        out.append({"cluster_id": str(uid), "original_unit_id": str(uid), "agent_id": str(agent_id_by_unit[uid]), "run_scope": str(run_scope), "semantic_scope": str(semantic_scope), "last_seen": 0.0, "vector": vec.tolist()})
    return out


def _distance(a: np.ndarray, b: np.ndarray, *, metric: str) -> float:
    if metric == "cosine":
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 1.0
        cosine = float(np.dot(a, b) / (na * nb))
        cosine = max(-1.0, min(1.0, cosine))
        return 1.0 - cosine
    if metric == "euclidean":
        return float(np.linalg.norm(a - b))
    raise ValueError(f"unsupported metric: {metric}")


def select_diverse_exact_cap(*, ordered_unit_ids: Sequence[str], vectors_by_unit: Mapping[str, Any], cap: int, metric: str, seed: int = 13, anchor_unit_id: str | None = None) -> dict[str, Any]:
    unit_ids = [str(uid) for uid in ordered_unit_ids]
    if cap < 0:
        raise ValueError("cap must be non-negative")
    target = min(int(cap), len(unit_ids))
    if target == 0:
        return {"selected_ids": [], "latency_seconds": 0.0, "metric": metric, "diagnostics": {"mean_min_distance": 0.0}}

    vectors = {uid: np.asarray(vectors_by_unit[uid], dtype=np.float64) for uid in unit_ids}
    ranks = {uid: _stable_hash(seed, "v7-diversity", metric, uid) for uid in unit_ids}
    selected: list[str] = []
    remaining = set(unit_ids)
    t0 = perf_counter()

    if anchor_unit_id is not None:
        start = str(anchor_unit_id)
        if start not in remaining:
            raise ValueError("anchor_unit_id must belong to ordered_unit_ids")
    else:
        start = min(remaining, key=lambda uid: (ranks[uid], uid))
    selected.append(start)
    remaining.remove(start)

    min_dists = {uid: _distance(vectors[uid], vectors[start], metric=metric) for uid in remaining}
    while remaining and len(selected) < target:
        best_uid = max(remaining, key=lambda uid: (min_dists[uid], ranks[uid], uid))
        selected.append(best_uid)
        remaining.remove(best_uid)
        for uid in remaining:
            d = _distance(vectors[uid], vectors[best_uid], metric=metric)
            if d < min_dists[uid]:
                min_dists[uid] = d

    return {"selected_ids": selected, "latency_seconds": float(perf_counter() - t0), "metric": metric, "diagnostics": {"candidate_count": len(unit_ids), "selected_count": len(selected)}}


def idw_binary_population(*, ordered_unit_ids: Sequence[str], selected_ids: Sequence[str], labels_by_unit: Mapping[str, int | bool], vectors_by_unit: Mapping[str, Any], metric: str, agent_id_by_unit: Mapping[str, str] | None = None, k: int = 8, power: float = 2.0, eps: float = 1e-6, prior: float = 0.5) -> BinaryPopulationResult:
    unit_ids = [str(uid) for uid in ordered_unit_ids]
    selected = tuple(sorted({str(uid) for uid in selected_ids}))
    selected_set = set(selected)

    labels = {uid: int(1 if bool(labels_by_unit[uid]) else 0) for uid in unit_ids}
    vectors = {uid: np.asarray(vectors_by_unit[uid], dtype=np.float64) for uid in unit_ids}
    all_donors = [uid for uid in unit_ids if uid in selected_set]
    agents = {uid: str((agent_id_by_unit or {}).get(uid, "unknown-agent")) for uid in unit_ids}

    t0 = perf_counter()
    rows: list[dict[str, Any]] = []
    for uid in unit_ids:
        observed = uid in selected_set
        if observed:
            prob = float(labels[uid])
            provenance = "observed"
            donor_scope = "observed"
            donor_ids: list[str] = []
        else:
            within_agent_donors = [donor for donor in all_donors if agents[donor] == agents[uid]]
            donor_pool = within_agent_donors if within_agent_donors else all_donors
            donor_scope = "within_agent" if within_agent_donors else ("global_fallback" if donor_pool else "none")
            distances = sorted(((donor_id, _distance(vectors[uid], vectors[donor_id], metric=metric)) for donor_id in donor_pool), key=lambda row: (row[1], row[0]))
            if not distances:
                prob = float(prior)
                provenance = "prior_no_donors"
                donor_ids = []
            else:
                nearest = distances[: max(1, min(int(k), len(distances)))]
                exact = [donor_id for donor_id, d in nearest if d <= float(eps)]
                if exact:
                    prob = float(np.mean([labels[donor_id] for donor_id in exact]))
                    provenance = "exact_match_within_agent" if donor_scope == "within_agent" else "exact_match_global_fallback"
                    donor_ids = exact
                else:
                    weights = np.asarray([1.0 / ((d + eps) ** power) for _, d in nearest], dtype=np.float64)
                    values = np.asarray([labels[donor_id] for donor_id, _ in nearest], dtype=np.float64)
                    denom = float(np.sum(weights))
                    prob = float(np.sum(weights * values) / denom) if denom > 0.0 else float(prior)
                    provenance = "idw_within_agent" if donor_scope == "within_agent" else "idw_global_fallback"
                    donor_ids = [donor_id for donor_id, _ in nearest]

        rows.append({"unit_id": uid, "observed": observed, "label": labels[uid], "probability": float(max(0.0, min(1.0, prob))), "binary": int(1 if prob >= 0.5 else 0), "provenance": provenance, "donor_scope": donor_scope, "donor_ids": donor_ids})

    return BinaryPopulationResult(
        estimate_binary_pass_rate=float(np.mean([row["binary"] for row in rows])) if rows else 0.0,
        estimate_probability_pass_rate=float(np.mean([row["probability"] for row in rows])) if rows else 0.0,
        observed_count=len(selected),
        imputed_count=len(unit_ids) - len(selected),
        rows=tuple(rows),
        latency_seconds=float(perf_counter() - t0),
    )


def _binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    return {"accuracy": float((tp + tn) / n), "precision": precision, "recall": recall, "f1": f1}


def embedding_binary_metrics(*, ordered_unit_ids: Sequence[str], labels_by_unit: Mapping[str, int | bool], idw_result: BinaryPopulationResult) -> dict[str, Any]:
    unit_ids = [str(uid) for uid in ordered_unit_ids]
    rows_by_id = {row["unit_id"]: row for row in idw_result.rows}
    y_true_all = [int(1 if bool(labels_by_unit[uid]) else 0) for uid in unit_ids]
    y_pred_all = [int(rows_by_id[uid]["binary"]) for uid in unit_ids]
    imputed_ids = [uid for uid in unit_ids if not bool(rows_by_id[uid]["observed"])]
    y_true_imputed = [int(1 if bool(labels_by_unit[uid]) else 0) for uid in imputed_ids]
    y_pred_imputed = [int(rows_by_id[uid]["binary"]) for uid in imputed_ids]
    return {
        "judged_plus_imputed": _binary_metrics(y_true_all, y_pred_all),
        "judged_plus_imputed_observed_inflated": True,
        "imputed_only": _binary_metrics(y_true_imputed, y_pred_imputed),
        "counts": {"population": len(unit_ids), "observed": idw_result.observed_count, "imputed": idw_result.imputed_count},
    }


def select_minhash_exact_cap(*, ordered_unit_ids: Sequence[str], traces_by_unit_id: Mapping[str, Trace], cap: int, seed: int = 13, minhash_records_by_unit_id: Mapping[str, MinHashRecord] | None = None) -> dict[str, Any]:
    cfg = MinHashConfig(ngram_size=3, permutations=128, lsh_bands=32, lsh_rows=4, seed=seed, similarity_threshold=0.55, ttl_s=90.0, max_clusters_per_agent=256, max_clusters_total=4096)
    provider = None
    if minhash_records_by_unit_id is not None:
        records_by_trace_id: dict[int, MinHashRecord] = {}
        for uid in ordered_unit_ids:
            records_by_trace_id[int(traces_by_unit_id[uid].trace_id)] = minhash_records_by_unit_id[uid]
        provider = V3ReadonlyMinHashProvider(records_by_trace_id, cfg)
    index = BandedMinHashLSHIndex(cfg, signature_provider=provider)
    scored: list[tuple[float, str]] = []
    t0 = perf_counter()
    for uid in (str(uid) for uid in ordered_unit_ids):
        obs = index.observe(traces_by_unit_id[uid])
        scored.append((float(obs.novelty) + float(obs.rarity), uid))
    scored.sort(key=lambda row: (-row[0], _stable_hash(seed, "v7-minhash-score", row[1]), row[1]))
    return {"selected_ids": [uid for _, uid in scored[: min(max(0, int(cap)), len(scored))]], "latency_seconds": float(perf_counter() - t0), "telemetry": dict(index.telemetry())}


def compute_overlap_diagnostics(selections_by_method: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    methods = [str(method) for method in selections_by_method]
    out: dict[str, Any] = {}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            left = methods[i]
            right = methods[j]
            left_set = set(str(uid) for uid in selections_by_method[left])
            right_set = set(str(uid) for uid in selections_by_method[right])
            inter = len(left_set & right_set)
            union = len(left_set | right_set)
            out[f"{left}__{right}"] = {"overlap_count": inter, "jaccard": (float(inter) / float(union)) if union > 0 else 1.0}
    return out


def _census_rate(ordered_unit_ids: Sequence[str], labels_by_unit: Mapping[str, int | bool]) -> float:
    return float(np.mean([1.0 if bool(labels_by_unit[str(uid)]) else 0.0 for uid in ordered_unit_ids])) if ordered_unit_ids else 0.0


def _to_v6_descriptors(*, ordered_unit_ids: Sequence[str], labels_by_unit: Mapping[str, int | bool], agent_id_by_unit: Mapping[str, str], concept_key_by_unit: Mapping[str, str], use_case_id_by_unit: Mapping[str, str], business_use_case_guid_by_unit: Mapping[str, str]) -> tuple[SessionDescriptor, ...]:
    return tuple(SessionDescriptor(unit_id=str(uid), agent_id=str(agent_id_by_unit[uid]), use_case_id=str(use_case_id_by_unit[uid]), business_use_case_guid=str(business_use_case_guid_by_unit[uid]), concept_key=str(concept_key_by_unit[uid]), label=bool(labels_by_unit[uid])) for uid in ordered_unit_ids)


def materialize_replay_dataset_view(
    *,
    replay_plan: ReplayPlan,
    source_ordered_unit_ids: Sequence[str],
    source_labels_by_unit: Mapping[str, int | bool],
    source_full_vectors_by_unit: Mapping[str, Any],
    source_reduced_vectors_by_unit: Mapping[str, Any],
    source_traces_by_unit_id: Mapping[str, Trace],
    source_agent_id_by_unit: Mapping[str, str],
    source_concept_key_by_unit: Mapping[str, str],
    source_use_case_id_by_unit: Mapping[str, str],
    source_business_use_case_guid_by_unit: Mapping[str, str],
    source_token_count_by_unit: Mapping[str, int],
    source_metadata_by_unit: Mapping[str, Mapping[str, Any]],
    source_minhash_records_by_unit_id: Mapping[str, MinHashRecord] | None = None,
) -> dict[str, Any]:
    replay_plan.validate(source_population=source_ordered_unit_ids)
    source_ids = tuple(str(uid) for uid in source_ordered_unit_ids)
    replay_ids = tuple(replay_plan.occurrence_ids)

    labels_by_unit: dict[str, int] = {}
    full_vectors_by_unit: dict[str, np.ndarray] = {}
    reduced_vectors_by_unit: dict[str, np.ndarray] = {}
    traces_by_unit_id: dict[str, Trace] = {}
    agent_id_by_unit: dict[str, str] = {}
    concept_key_by_unit: dict[str, str] = {}
    use_case_id_by_unit: dict[str, str] = {}
    business_use_case_guid_by_unit: dict[str, str] = {}
    token_count_by_unit: dict[str, int] = {}
    metadata_by_unit: dict[str, dict[str, Any]] = {}
    minhash_records_by_unit_id: dict[str, MinHashRecord] = {}
    source_unit_id_by_unit: dict[str, str] = {}

    source_set = set(source_ids)
    for event_idx, occurrence_id in enumerate(replay_ids):
        source_id = str(replay_plan.source_id_by_occurrence[occurrence_id])
        if source_id not in source_set:
            raise ValueError(f"replay source unit is unknown: {source_id}")
        source_unit_id_by_unit[occurrence_id] = source_id
        labels_by_unit[occurrence_id] = int(1 if bool(source_labels_by_unit[source_id]) else 0)
        full_vectors_by_unit[occurrence_id] = np.asarray(source_full_vectors_by_unit[source_id], dtype=np.float64)
        reduced_vectors_by_unit[occurrence_id] = np.asarray(source_reduced_vectors_by_unit[source_id], dtype=np.float32)
        source_trace = source_traces_by_unit_id[source_id]
        trace_seed = _stable_int(f"{replay_plan.replay_id}|{occurrence_id}|trace")
        traces_by_unit_id[occurrence_id] = Trace(
            trace_id=trace_seed,
            agent_id=source_trace.agent_id,
            timestamp=float(event_idx),
            signature=tuple(source_trace.signature),
            span_count=int(source_trace.span_count),
            duration_ms=float(source_trace.duration_ms),
            status=str(source_trace.status),
            concept_id=int(source_trace.concept_id),
            events=tuple(source_trace.events),
        )
        agent_id_by_unit[occurrence_id] = str(source_agent_id_by_unit[source_id])
        concept_key_by_unit[occurrence_id] = str(source_concept_key_by_unit[source_id])
        use_case_id_by_unit[occurrence_id] = str(source_use_case_id_by_unit[source_id])
        business_use_case_guid_by_unit[occurrence_id] = str(source_business_use_case_guid_by_unit[source_id])
        token_count_by_unit[occurrence_id] = int(source_token_count_by_unit[source_id])
        md = dict(source_metadata_by_unit.get(source_id) or {})
        md["source_unit_id"] = source_id
        md["replay_id"] = replay_plan.replay_id
        md["repetition_index"] = int(replay_plan.repetition_index)
        md["replay_event_index"] = int(event_idx)
        metadata_by_unit[occurrence_id] = md
        if source_minhash_records_by_unit_id is not None:
            minhash_records_by_unit_id[occurrence_id] = source_minhash_records_by_unit_id[source_id]

    return {
        "ordered_unit_ids": replay_ids,
        "labels_by_unit": labels_by_unit,
        "full_vectors_by_unit": full_vectors_by_unit,
        "reduced_vectors_by_unit": reduced_vectors_by_unit,
        "traces_by_unit_id": traces_by_unit_id,
        "agent_id_by_unit": agent_id_by_unit,
        "concept_key_by_unit": concept_key_by_unit,
        "use_case_id_by_unit": use_case_id_by_unit,
        "business_use_case_guid_by_unit": business_use_case_guid_by_unit,
        "token_count_by_unit": token_count_by_unit,
        "metadata_by_unit": metadata_by_unit,
        "minhash_records_by_unit_id": minhash_records_by_unit_id if source_minhash_records_by_unit_id is not None else None,
        "source_unit_id_by_unit": source_unit_id_by_unit,
    }


def run_v7_core(*, ordered_unit_ids: Sequence[str], labels_by_unit: Mapping[str, int | bool], full_vectors_by_unit: Mapping[str, Any], traces_by_unit_id: Mapping[str, Trace], agent_id_by_unit: Mapping[str, str], concept_key_by_unit: Mapping[str, str], use_case_id_by_unit: Mapping[str, str], business_use_case_guid_by_unit: Mapping[str, str], token_count_by_unit: Mapping[str, int], metadata_by_unit: Mapping[str, Mapping[str, Any]] | None = None, budget_levels: Sequence[int] = V7_BUDGET_LEVELS, seed: int = 13, window_id: str = "window-1", minhash_records_by_unit_id: Mapping[str, MinHashRecord] | None = None, reduced_vectors_by_unit: Mapping[str, Any] | None = None, pca_shared_latency_seconds: float = 0.0, replay_plan: ReplayPlan | None = None, source_corpus_census_pass_rate: float | None = None, source_unit_id_by_unit: Mapping[str, str] | None = None) -> dict[str, Any]:
    ordered = tuple(str(uid) for uid in ordered_unit_ids)
    for uid in ordered:
        if uid not in token_count_by_unit:
            raise ValueError(f"missing actual token count for unit_id={uid}")
    caps = derive_budget_caps(len(ordered), budget_levels)
    labels_bool = {uid: bool(labels_by_unit[uid]) for uid in ordered}
    descriptors = _to_v6_descriptors(ordered_unit_ids=ordered, labels_by_unit=labels_bool, agent_id_by_unit=agent_id_by_unit, concept_key_by_unit=concept_key_by_unit, use_case_id_by_unit=use_case_id_by_unit, business_use_case_guid_by_unit=business_use_case_guid_by_unit)
    reduced_vectors = {str(uid): np.asarray(vec, dtype=np.float32) for uid, vec in (reduced_vectors_by_unit or fit_pca8_embeddings(ordered_unit_ids=ordered, full_vectors_by_unit=full_vectors_by_unit)["reduced_vectors_by_unit"]).items()}
    replay_census = _census_rate(ordered, labels_bool)
    source_census = float(source_corpus_census_pass_rate) if source_corpus_census_pass_rate is not None else replay_census
    replay_meta = replay_plan
    runs: list[dict[str, Any]] = []
    for budget_pct in budget_levels:
        cap = caps[int(budget_pct)]
        selections: dict[str, list[str]] = {}
        selections[V7_METHOD_IDS["random"]] = list(select_arm1(descriptors=descriptors, cap=cap, trial_seed=seed, window_id=window_id).selected_ids)
        minhash_result = select_minhash_exact_cap(ordered_unit_ids=ordered, traces_by_unit_id=traces_by_unit_id, cap=cap, seed=seed, minhash_records_by_unit_id=minhash_records_by_unit_id)
        selections[V7_METHOD_IDS["minhash"]] = list(minhash_result["selected_ids"])
        cos_sel = select_diverse_exact_cap(ordered_unit_ids=ordered, vectors_by_unit=reduced_vectors, cap=cap, metric="cosine", seed=seed)
        cos_idw = idw_binary_population(ordered_unit_ids=ordered, selected_ids=cos_sel["selected_ids"], labels_by_unit=labels_bool, vectors_by_unit=reduced_vectors, metric="cosine", agent_id_by_unit=agent_id_by_unit)
        selections[V7_METHOD_IDS["pca8_cosine"]] = list(cos_sel["selected_ids"])
        euc_sel = select_diverse_exact_cap(ordered_unit_ids=ordered, vectors_by_unit=reduced_vectors, cap=cap, metric="euclidean", seed=seed)
        euc_idw = idw_binary_population(ordered_unit_ids=ordered, selected_ids=euc_sel["selected_ids"], labels_by_unit=labels_bool, vectors_by_unit=reduced_vectors, metric="euclidean", agent_id_by_unit=agent_id_by_unit)
        selections[V7_METHOD_IDS["pca8_euclidean"]] = list(euc_sel["selected_ids"])
        arm4 = select_arm4(descriptors=descriptors, cap=cap, trial_seed=seed, window_id=window_id)
        arm5 = select_arm5(descriptors=descriptors, arm4_outcome=arm4, labels_by_unit=labels_bool, trial_seed=seed, window_id=window_id)
        selections[V7_METHOD_IDS["arm5"]] = list(arm5.selected_ids)
        overlap = compute_overlap_diagnostics(selections)
        for method_id in V7_METHOD_ID_ORDER:
            selected = selections[method_id]
            selected_rate = float(np.mean([1.0 if labels_bool[uid] else 0.0 for uid in selected])) if selected else 0.0
            estimate = selected_rate
            estimator_type = "selected_rate"
            diagnostics: dict[str, Any] = {}
            embedding_metrics: dict[str, Any] | None = None
            imputation_rows: list[dict[str, Any]] = []
            latency_sel = 0.0
            latency_idw = 0.0
            if method_id == V7_METHOD_IDS["minhash"]:
                latency_sel = float(minhash_result["latency_seconds"])
                diagnostics["minhash_telemetry"] = dict(minhash_result["telemetry"])
            elif method_id == V7_METHOD_IDS["pca8_cosine"]:
                estimate = float(cos_idw.estimate_binary_pass_rate)
                estimator_type = "pca_binary_imputed_population"
                latency_sel = float(cos_sel["latency_seconds"])
                latency_idw = float(cos_idw.latency_seconds)
                embedding_metrics = embedding_binary_metrics(ordered_unit_ids=ordered, labels_by_unit=labels_bool, idw_result=cos_idw)
                imputation_rows = [dict(row) for row in cos_idw.rows]
            elif method_id == V7_METHOD_IDS["pca8_euclidean"]:
                estimate = float(euc_idw.estimate_binary_pass_rate)
                estimator_type = "pca_binary_imputed_population"
                latency_sel = float(euc_sel["latency_seconds"])
                latency_idw = float(euc_idw.latency_seconds)
                embedding_metrics = embedding_binary_metrics(ordered_unit_ids=ordered, labels_by_unit=labels_bool, idw_result=euc_idw)
                imputation_rows = [dict(row) for row in euc_idw.rows]
            elif method_id == V7_METHOD_IDS["arm5"]:
                metrics = compute_trial_metrics(descriptors=descriptors, selected_ids=arm5.selected_ids, method_id=V6_METHOD_IDS["arm5"], trial_seed=seed, window_id=window_id, nominal_budget=cap, labels_by_unit=labels_bool, arm4_outcome=arm4)
                estimate = float(metrics.estimate)
                estimator_type = "hajek_weighted"
                diagnostics["arm4_inclusion"] = arm4_inclusion_probabilities(descriptors=descriptors, cap=cap, trial_seed=seed, window_id=window_id)
            runs.append({
                "version": V7_VERSION,
                "method_id": method_id,
                "budget_pct": int(budget_pct),
                "cap": int(cap),
                "sample_size": len(selected),
                "selected_ids": list(selected),
                "selected_rate": selected_rate,
                "estimate": estimate,
                "estimator_type": estimator_type,
                "census_pass_rate": replay_census,
                "replay_census_pass_rate": replay_census,
                "source_corpus_census_pass_rate": source_census,
                "aggregate_pass_rate_mae": abs(estimate - source_census),
                "selected_only_pass_rate_mae": abs(selected_rate - source_census),
                "replay_aggregate_pass_rate_mae": abs(estimate - replay_census),
                "replay_selected_only_pass_rate_mae": abs(selected_rate - replay_census),
                "actual_token_count": int(sum(int(token_count_by_unit[uid]) for uid in selected)),
                "latency_seconds": {"selection": float(latency_sel), "idw": float(latency_idw), "search_evidence": 0.0, "per_method_total": float(latency_sel + latency_idw), "shared_preprocessing_reference": "dataset_profile.shared_preprocessing"},
                "embedding_metrics": embedding_metrics,
                "imputation_rows": imputation_rows,
                "coverage": _coverage_snapshot(selected_ids=selected, ordered_unit_ids=ordered, agent_id_by_unit=agent_id_by_unit, concept_key_by_unit=concept_key_by_unit, metadata_by_unit=metadata_by_unit or {}),
                "diagnostics": diagnostics,
                "overlap": overlap,
                "repetition_index": int(replay_meta.repetition_index) if replay_meta is not None else 1,
                "replay_seed": int(replay_meta.replay_seed) if replay_meta is not None else int(seed),
                "replay_id": str(replay_meta.replay_id) if replay_meta is not None else f"single|{seed}",
                "replay_order_sha256": str(replay_meta.order_sha256) if replay_meta is not None else _stable_hash(seed, "order", window_id),
                "replay_frequency_sha256": str(replay_meta.frequency_sha256) if replay_meta is not None else _stable_hash(seed, "freq", window_id),
                "replay_frequency_summary": {
                    "unique_source_count": int(replay_meta.unique_source_count) if replay_meta is not None else len(set(ordered)),
                    "unique_source_fraction": float(replay_meta.unique_source_fraction) if replay_meta is not None else 1.0,
                    "min_frequency": int(replay_meta.min_frequency) if replay_meta is not None else 1,
                    "max_frequency": int(replay_meta.max_frequency) if replay_meta is not None else 1,
                    "mean_frequency": float(replay_meta.mean_frequency) if replay_meta is not None else 1.0,
                    "std_frequency": float(replay_meta.std_frequency) if replay_meta is not None else 0.0,
                    "duplicate_event_count": int(replay_meta.duplicate_event_count) if replay_meta is not None else 0,
                },
                "source_unit_id_by_unit": dict(source_unit_id_by_unit or {}),
            })
    return {"version": V7_VERSION, "method_id_order": list(V7_METHOD_ID_ORDER), "budget_caps": caps, "runs": runs}


def _coverage_snapshot(*, selected_ids: Sequence[str], ordered_unit_ids: Sequence[str], agent_id_by_unit: Mapping[str, str], concept_key_by_unit: Mapping[str, str], metadata_by_unit: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    selected = set(str(uid) for uid in selected_ids)
    ordered = [str(uid) for uid in ordered_unit_ids]
    all_agents = {str(agent_id_by_unit.get(uid, "unknown")) for uid in ordered}
    sel_agents = {str(agent_id_by_unit.get(uid, "unknown")) for uid in ordered if uid in selected}
    all_domains = {str((metadata_by_unit.get(uid) or {}).get("domain") or "unknown") for uid in ordered}
    sel_domains = {str((metadata_by_unit.get(uid) or {}).get("domain") or "unknown") for uid in ordered if uid in selected}
    all_tasks = {str((metadata_by_unit.get(uid) or {}).get("task_id") or (metadata_by_unit.get(uid) or {}).get("task") or "unknown") for uid in ordered}
    sel_tasks = {str((metadata_by_unit.get(uid) or {}).get("task_id") or (metadata_by_unit.get(uid) or {}).get("task") or "unknown") for uid in ordered if uid in selected}
    all_concepts = {str(concept_key_by_unit.get(uid, "unknown")) for uid in ordered}
    sel_concepts = {str(concept_key_by_unit.get(uid, "unknown")) for uid in ordered if uid in selected}
    return {
        "agent": float(len(sel_agents) / len(all_agents)) if all_agents else 1.0,
        "domain": float(len(sel_domains) / len(all_domains)) if all_domains else 1.0,
        "task": float(len(sel_tasks) / len(all_tasks)) if all_tasks else 1.0,
        "concept": {
            "selected_distinct": len(sel_concepts),
            "population_distinct": len(all_concepts),
            "coverage_ratio": float(len(sel_concepts) / len(all_concepts)) if all_concepts else 1.0,
        },
        "selected_count": len(selected),
        "imputed_count": max(0, len(ordered) - len(selected)),
    }


def make_synthetic_trace_for_text(*, unit_id: str, agent_id: str, text: str) -> Trace:
    return Trace(trace_id=int(int(hashlib.sha256(unit_id.encode("utf-8")).hexdigest()[:16], 16)), agent_id=str(agent_id), timestamp=0.0, signature=("synthetic",), span_count=1, duration_ms=0.0, status="ok", concept_id=0, events=(SessionEvent(role="assistant", text=str(text)),))


def build_v7_method_matrix(*, repetitions: int = V7_DEFAULT_REPETITIONS, base_seed: int = V7_DEFAULT_BASE_SEED, budget_levels: Sequence[int] = V7_BUDGET_LEVELS, seeds: Sequence[int] | None = None) -> list[tuple[int, int]]:
    replay_seeds = tuple(int(seed) for seed in (seeds or ()))
    if not replay_seeds:
        replay_seeds = tuple(_replay_seed(int(base_seed), idx) for idx in range(1, max(1, int(repetitions)) + 1))
    return [(int(seed), int(budget)) for seed in replay_seeds for budget in budget_levels]


def build_v7_report_html(payload: Mapping[str, Any]) -> str:
    return _canonical_build_v7_report_html(payload)


def _value_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _metric_from_row(row: Mapping[str, Any], metric_id: str) -> float | None:
    if metric_id == "aggregate_pass_rate_mae":
        return _value_or_none(row.get("aggregate_pass_rate_mae"))
    if metric_id == "selected_only_pass_rate_mae":
        return _value_or_none(row.get("selected_only_pass_rate_mae"))
    if metric_id == "replay_aggregate_pass_rate_mae":
        return _value_or_none(row.get("replay_aggregate_pass_rate_mae"))
    if metric_id == "replay_selected_only_pass_rate_mae":
        return _value_or_none(row.get("replay_selected_only_pass_rate_mae"))
    if metric_id == "coverage_concept_ratio":
        return _value_or_none((((row.get("coverage") or {}).get("concept") or {}).get("coverage_ratio")))
    if metric_id == "coverage_task":
        return _value_or_none((row.get("coverage") or {}).get("task"))
    if metric_id == "coverage_domain":
        return _value_or_none((row.get("coverage") or {}).get("domain"))
    if metric_id == "coverage_agent":
        return _value_or_none((row.get("coverage") or {}).get("agent"))
    if metric_id == "actual_token_count":
        return _value_or_none(row.get("actual_token_count"))
    if metric_id == "latency_per_method_total_seconds":
        return _value_or_none(((row.get("latency_seconds") or {}).get("per_method_total")))
    if metric_id.startswith("imputed_only_"):
        key = metric_id.replace("imputed_only_", "")
        return _value_or_none((((row.get("embedding_metrics") or {}).get("imputed_only") or {}).get(key)))
    if metric_id.startswith("judged_plus_imputed_"):
        key = metric_id.replace("judged_plus_imputed_", "")
        return _value_or_none((((row.get("embedding_metrics") or {}).get("judged_plus_imputed") or {}).get(key)))
    return None


def _metric_summary(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "sample_std": 0.0,
            "standard_error": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "mean_ci95_lower": 0.0,
            "mean_ci95_upper": 0.0,
        }
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    sample_std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    standard_error = float(sample_std / math.sqrt(n)) if n > 0 else 0.0
    critical_value = float(student_t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    ci_delta = float(critical_value * standard_error)
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "mean_ci95_lower": float(mean - ci_delta),
        "mean_ci95_upper": float(mean + ci_delta),
    }


def _build_aggregate_metrics(run_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_specs: tuple[tuple[str, str], ...] = (
        ("aggregate_pass_rate_mae", "lower_better"),
        ("selected_only_pass_rate_mae", "lower_better"),
        ("replay_aggregate_pass_rate_mae", "lower_better"),
        ("replay_selected_only_pass_rate_mae", "lower_better"),
        ("coverage_concept_ratio", "higher_better"),
        ("coverage_task", "higher_better"),
        ("coverage_domain", "higher_better"),
        ("coverage_agent", "higher_better"),
        ("actual_token_count", "lower_better"),
        ("latency_per_method_total_seconds", "lower_better"),
        ("imputed_only_accuracy", "higher_better"),
        ("imputed_only_precision", "higher_better"),
        ("imputed_only_recall", "higher_better"),
        ("imputed_only_f1", "higher_better"),
        ("judged_plus_imputed_accuracy", "higher_better"),
        ("judged_plus_imputed_precision", "higher_better"),
        ("judged_plus_imputed_recall", "higher_better"),
        ("judged_plus_imputed_f1", "higher_better"),
    )
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in run_rows:
        key = (str(row.get("dataset_id")), int(row.get("budget_pct") or 0), str(row.get("method_id")))
        grouped.setdefault(key, []).append(row)

    out_rows: list[dict[str, Any]] = []
    for (dataset_id, budget_pct, method_id), rows in sorted(grouped.items()):
        for metric_id, direction in metric_specs:
            values: list[float] = []
            for row in rows:
                value = _metric_from_row(row, metric_id)
                if value is not None:
                    values.append(value)
            stats = _metric_summary(values)
            out_rows.append(
                {
                    "dataset_id": dataset_id,
                    "budget_pct": int(budget_pct),
                    "method_id": method_id,
                    "metric_id": metric_id,
                    "direction": direction,
                    **stats,
                    "ci95_note": "Empirical replay uncertainty summary (Student-t interval); not a population-generalization guarantee.",
                }
            )
    return {"rows": out_rows}


def _build_paired_differences(run_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    method_cos = V7_METHOD_IDS["pca8_cosine"]
    method_euc = V7_METHOD_IDS["pca8_euclidean"]
    metric_specs: tuple[tuple[str, str], ...] = (
        ("aggregate_pass_rate_mae", "lower_better"),
        ("selected_only_pass_rate_mae", "lower_better"),
        ("replay_aggregate_pass_rate_mae", "lower_better"),
        ("replay_selected_only_pass_rate_mae", "lower_better"),
        ("coverage_concept_ratio", "higher_better"),
        ("imputed_only_accuracy", "higher_better"),
        ("imputed_only_precision", "higher_better"),
        ("imputed_only_recall", "higher_better"),
        ("imputed_only_f1", "higher_better"),
        ("latency_per_method_total_seconds", "lower_better"),
        ("actual_token_count", "lower_better"),
    )

    cell_index: dict[tuple[str, int, int, str], Mapping[str, Any]] = {}
    for row in run_rows:
        cell_key = (
            str(row.get("dataset_id")),
            int(row.get("budget_pct") or 0),
            int(row.get("repetition_index") or 0),
            str(row.get("method_id")),
        )
        cell_index[cell_key] = row

    paired_rows: list[dict[str, Any]] = []
    grouped_deltas: dict[tuple[str, int, str], list[float]] = {}
    grouped_cells: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    keys = sorted({(d, b, r) for (d, b, r, _) in cell_index.keys()})
    for dataset_id, budget_pct, repetition_index in keys:
        left = cell_index.get((dataset_id, budget_pct, repetition_index, method_cos))
        right = cell_index.get((dataset_id, budget_pct, repetition_index, method_euc))
        if left is None or right is None:
            continue
        for metric_id, direction in metric_specs:
            lv = _metric_from_row(left, metric_id)
            rv = _metric_from_row(right, metric_id)
            if lv is None or rv is None:
                continue
            delta = float(lv - rv)
            gk = (dataset_id, int(budget_pct), metric_id)
            grouped_deltas.setdefault(gk, []).append(delta)
            grouped_cells.setdefault(gk, []).append(
                {
                    "dataset_id": dataset_id,
                    "budget_pct": int(budget_pct),
                    "repetition_index": int(repetition_index),
                    "metric_id": metric_id,
                    "direction": direction,
                    "delta_cosine_minus_euclidean": delta,
                }
            )

    for (dataset_id, budget_pct, metric_id), deltas in sorted(grouped_deltas.items()):
        direction = next(spec[1] for spec in metric_specs if spec[0] == metric_id)
        tol = 1e-12
        wins = 0
        ties = 0
        losses = 0
        for delta in deltas:
            if abs(delta) <= tol:
                ties += 1
                continue
            if direction == "higher_better":
                if delta > 0:
                    wins += 1
                else:
                    losses += 1
            else:
                if delta < 0:
                    wins += 1
                else:
                    losses += 1
        stats = _metric_summary(deltas)
        paired_rows.append(
            {
                "dataset_id": dataset_id,
                "budget_pct": int(budget_pct),
                "pair_id": f"{method_cos}_minus_{method_euc}",
                "left_method_id": method_cos,
                "right_method_id": method_euc,
                "metric_id": metric_id,
                "direction": direction,
                "n": int(stats["n"]),
                "mean_delta": float(stats["mean"]),
                "median_delta": float(stats["median"]),
                "p05_delta": float(stats["p05"]),
                "p95_delta": float(stats["p95"]),
                "wins": int(wins),
                "ties": int(ties),
                "losses": int(losses),
                "win_rate": float((wins / stats["n"]) if stats["n"] > 0 else 0.0),
            }
        )
    return {"rows": paired_rows, "cells": [cell for rows in grouped_cells.values() for cell in rows]}


def run_v7_experiment(*, datasets: Mapping[str, Mapping[str, Any]], output_dir: str | Path | None = None, seed_values: Sequence[int] | None = None, repetitions: int = V7_DEFAULT_REPETITIONS, base_seed: int = V7_DEFAULT_BASE_SEED, budget_levels: Sequence[int] = V7_BUDGET_LEVELS, method_ids: Sequence[str] | None = None, partial_label: int = 0, azure_config: Any | None = None, search_adapter: Any | None = None, skip_search_sync: bool = False, cosine_index_name: str = "trace-clusters-sampling-v7-cosine", euclidean_index_name: str = "trace-clusters-sampling-v7-euclidean", search_probe_attempts: int = 3, search_probe_delay_seconds: float = 1.0) -> dict[str, Any]:
    _ = partial_label
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("datasets must be a non-empty mapping of dataset_id -> dataset payload")
    method_order = tuple(method_ids or V7_METHOD_ID_ORDER)
    explicit_seeds = tuple(int(seed) for seed in (seed_values or ()) if str(seed).strip())
    if explicit_seeds:
        repetition_seeds = explicit_seeds
        repetition_count = len(repetition_seeds)
    else:
        repetition_count = max(1, int(repetitions))
        repetition_seeds = tuple(_replay_seed(int(base_seed), rep_idx) for rep_idx in range(1, repetition_count + 1))
    output_path = Path(output_dir) if output_dir is not None else Path("outputs_sampling_v7") / "runs" / f"v7-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    output_path.mkdir(parents=True, exist_ok=True)

    active_search_adapter = search_adapter
    if active_search_adapter is None and azure_config is not None:
        active_search_adapter = AzureV7SearchAdapter(azure_config=azure_config, cosine_index_name=cosine_index_name, euclidean_index_name=euclidean_index_name)
    if active_search_adapter is None and not skip_search_sync:
        raise ValueError("v7 requires live Search sync unless --skip-search-sync is explicitly enabled")

    run_rows: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
    imputed_predictions: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    dataset_profiles: list[dict[str, Any]] = []
    embedding_ledger: list[dict[str, Any]] = []
    search_ledger: list[dict[str, Any]] = []
    shared_preprocessing: list[dict[str, Any]] = []

    for dataset_id, dataset in datasets.items():
        ordered = [str(uid) for uid in dataset["ordered_unit_ids"]]
        labels_by_unit = {str(k): int(v) for k, v in (dataset.get("labels_by_unit") or {}).items()}
        full_vectors_by_unit = {str(k): np.asarray(v, dtype=np.float64) for k, v in (dataset.get("full_vectors_by_unit") or {}).items()}
        traces_by_unit_id = dataset.get("traces_by_unit_id") or {}
        agent_id_by_unit = {str(k): str(v) for k, v in (dataset.get("agent_id_by_unit") or {}).items()}
        concept_key_by_unit = {str(k): str(v) for k, v in (dataset.get("concept_key_by_unit") or {}).items()}
        use_case_id_by_unit = {str(k): str(v) for k, v in (dataset.get("use_case_id_by_unit") or {}).items()}
        business_use_case_guid_by_unit = {str(k): str(v) for k, v in (dataset.get("business_use_case_guid_by_unit") or {}).items()}
        metadata_by_unit = {str(k): dict(v) for k, v in (dataset.get("metadata_by_unit") or {}).items()}
        token_count_by_unit = {str(k): int(v) for k, v in (dataset.get("token_count_by_unit") or {}).items()}
        minhash_records_by_unit_id = dataset.get("minhash_records_by_unit_id")
        runtime_ledger = dict(dataset.get("runtime_ledger") or {})
        representation_source_by_unit = {
            str(k): str(v)
            for k, v in (dataset.get("representation_source_by_unit") or {}).items()
        }
        if not representation_source_by_unit:
            representation_source_by_unit = {
                uid: str((metadata_by_unit.get(uid) or {}).get("representation_source") or "unknown")
                for uid in ordered
            }

        source_census_pass_rate = _census_rate(ordered, labels_by_unit)
        pca = fit_pca8_embeddings(ordered_unit_ids=ordered, full_vectors_by_unit=full_vectors_by_unit)
        reduced = pca["reduced_vectors_by_unit"]
        label_positive_count = int(sum(1 for uid in ordered if bool(labels_by_unit.get(uid, 0))))
        linkage_counts = Counter(
            str((metadata_by_unit.get(uid) or {}).get("linkage_path") or "none")
            for uid in ordered
        )
        rep_counts = Counter(representation_source_by_unit.get(uid, "unknown") for uid in ordered)
        dataset_profile = {
            "dataset_id": dataset_id,
            "population": len(ordered),
            "label_positive_count": label_positive_count,
            "label_rate": (float(label_positive_count) / float(len(ordered))) if ordered else 0.0,
            "agent_count": len({agent_id_by_unit.get(uid, "unknown") for uid in ordered}),
            "domain_count": len({str((metadata_by_unit.get(uid) or {}).get("domain") or "unknown") for uid in ordered}),
            "task_count": len({str((metadata_by_unit.get(uid) or {}).get("task_id") or (metadata_by_unit.get(uid) or {}).get("task") or "unknown") for uid in ordered}),
            "concept_count": len({concept_key_by_unit.get(uid, "unknown") for uid in ordered}),
            "representation_source_counts": dict(rep_counts),
            "linkage_path_counts": dict(linkage_counts),
            "pca": {
                "n_components": pca["n_components"],
                "explained_variance_ratio": list(pca["explained_variance_ratio"]),
                "explained_variance_total": pca["explained_variance_total"],
                "latency_seconds": pca["latency_seconds"],
            },
            "runtime_ledger": runtime_ledger,
            "source_paths": dataset.get("source_paths") or {},
            "shared_preprocessing": {
                "embedding_model_id": str(runtime_ledger.get("embedding_model_id") or dataset.get("embedding_model_id") or "text-embedding-3-small"),
                "embedding_deployment_id": str(runtime_ledger.get("embedding_deployment_id") or dataset.get("embedding_deployment_id") or ""),
                "embedding_calls": int(runtime_ledger.get("embedding_calls") or 0),
                "embedding_inputs": int(runtime_ledger.get("embedding_inputs") or 0),
                "embedding_input_tokens": int(runtime_ledger.get("embedding_input_tokens") or 0),
                "embedding_latency_seconds": float(runtime_ledger.get("embedding_latency_seconds") or 0.0),
                "pca_fit_latency_seconds": float(pca["latency_seconds"]),
                "pca_explained_variance_total": float(pca["explained_variance_total"]),
            },
        }
        dataset_profiles.append(dataset_profile)
        shared_preprocessing.append({"dataset_id": dataset_id, **dict(dataset_profile["shared_preprocessing"])})
        embedding_ledger.append({"dataset_id": dataset_id, **dict(dataset_profile["shared_preprocessing"]), "cache_hit": bool(runtime_ledger.get("cache_hit") or False), "cache_rows": int(runtime_ledger.get("cache_rows") or 0)})

        index_names = {"cosine": cosine_index_name, "euclidean": euclidean_index_name}
        docs_by_metric = {
            "cosine": build_reduced_vector_export_documents(ordered_unit_ids=ordered, reduced_vectors_by_unit=reduced, agent_id_by_unit=agent_id_by_unit, run_scope=f"{dataset_id}-{V7_VERSION}", semantic_scope="v7-pca8-cosine"),
            "euclidean": build_reduced_vector_export_documents(ordered_unit_ids=ordered, reduced_vectors_by_unit=reduced, agent_id_by_unit=agent_id_by_unit, run_scope=f"{dataset_id}-{V7_VERSION}", semantic_scope="v7-pca8-euclidean"),
        }
        if not skip_search_sync and active_search_adapter is not None:
            for metric_name in ("cosine", "euclidean"):
                t0 = perf_counter()
                write_count = int(active_search_adapter.sync_documents(index_name=index_names[metric_name], documents=docs_by_metric[metric_name], dimensions=V7_PCA_DIMENSIONS, metric=metric_name))
                expected = len(docs_by_metric[metric_name])
                if write_count != expected:
                    raise ValueError(f"v7 search sync wrote {write_count} documents; expected {expected} for dataset={dataset_id} metric={metric_name}")
                search_ledger.append({"dataset_id": dataset_id, "metric": metric_name, "index_name": index_names[metric_name], "write_count": write_count, "expected_write_count": expected, "sync_latency_seconds": float(perf_counter() - t0)})

        shared_search_probe_by_metric: dict[str, dict[str, Any]] = {
            "cosine": {"query_count": 0, "neighbor_hits": 0, "latency_seconds": 0.0, "index_name": index_names["cosine"], "metric": "cosine", "shared_probe": False},
            "euclidean": {"query_count": 0, "neighbor_hits": 0, "latency_seconds": 0.0, "index_name": index_names["euclidean"], "metric": "euclidean", "shared_probe": False},
        }
        if not skip_search_sync and active_search_adapter is not None:
            for metric_name in ("cosine", "euclidean"):
                scope_name = f"v7-pca8-{metric_name}"
                index_name = index_names[metric_name]
                probe_ids = list(ordered[: min(3, len(ordered))])
                t0 = perf_counter()
                max_attempts = max(1, int(search_probe_attempts))
                delay_s = max(0.0, float(search_probe_delay_seconds))
                if active_search_adapter.__class__.__name__.lower().startswith("fake"):
                    delay_s = 0.0
                total_queries = 0
                hits_by_attempt: list[int] = []
                for attempt_idx in range(max_attempts):
                    hits = 0
                    for source_uid in probe_ids:
                        neighbors = active_search_adapter.query_neighbors(
                            index_name=index_name,
                            vector=np.asarray(reduced[source_uid], dtype=np.float32),
                            k=1,
                            metric=metric_name,
                            run_scope=f"{dataset_id}-{V7_VERSION}",
                            semantic_scope=scope_name,
                        )
                        total_queries += 1
                        if neighbors:
                            hits += 1
                    hits_by_attempt.append(hits)
                    if hits > 0 or attempt_idx + 1 >= max_attempts:
                        break
                    if delay_s > 0.0:
                        sleep(delay_s)
                shared_search_probe_by_metric[metric_name] = {
                    "query_count": total_queries,
                    "neighbor_hits": max(hits_by_attempt) if hits_by_attempt else 0,
                    "attempt_count": len(hits_by_attempt),
                    "hits_by_attempt": hits_by_attempt,
                    "latency_seconds": float(perf_counter() - t0),
                    "index_name": index_name,
                    "metric": metric_name,
                    "shared_probe": True,
                }

        for repetition_index, replay_seed in enumerate(repetition_seeds, start=1):
            plan = build_replay_plan(
                dataset_id=dataset_id,
                ordered_source_unit_ids=ordered,
                repetition_index=int(repetition_index),
                base_seed=int(base_seed),
                replay_seed_override=int(replay_seed) if explicit_seeds else None,
            )
            replay_rows.append(
                {
                    "dataset_id": dataset_id,
                    "repetition_index": int(repetition_index),
                    "replay_seed": int(plan.replay_seed),
                    "replay_id": plan.replay_id,
                    "occurrence_count": len(plan.occurrence_ids),
                    "occurrence_ids": list(plan.occurrence_ids),
                    "source_unit_ids_in_order": list(plan.source_unit_ids),
                    "source_id_by_occurrence": dict(plan.source_id_by_occurrence),
                    "frequency_by_source": dict(plan.frequency_by_source),
                    "replay_order_sha256": plan.order_sha256,
                    "replay_frequency_sha256": plan.frequency_sha256,
                    "unique_source_count": int(plan.unique_source_count),
                    "unique_source_fraction": float(plan.unique_source_fraction),
                    "min_frequency": int(plan.min_frequency),
                    "max_frequency": int(plan.max_frequency),
                    "mean_frequency": float(plan.mean_frequency),
                    "std_frequency": float(plan.std_frequency),
                    "duplicate_event_count": int(plan.duplicate_event_count),
                }
            )
            replay_view = materialize_replay_dataset_view(
                replay_plan=plan,
                source_ordered_unit_ids=ordered,
                source_labels_by_unit=labels_by_unit,
                source_full_vectors_by_unit=full_vectors_by_unit,
                source_reduced_vectors_by_unit=reduced,
                source_traces_by_unit_id=traces_by_unit_id,
                source_agent_id_by_unit=agent_id_by_unit,
                source_concept_key_by_unit=concept_key_by_unit,
                source_use_case_id_by_unit=use_case_id_by_unit,
                source_business_use_case_guid_by_unit=business_use_case_guid_by_unit,
                source_token_count_by_unit=token_count_by_unit,
                source_metadata_by_unit=metadata_by_unit,
                source_minhash_records_by_unit_id=minhash_records_by_unit_id,
            )
            seed_result = run_v7_core(
                ordered_unit_ids=replay_view["ordered_unit_ids"],
                labels_by_unit=replay_view["labels_by_unit"],
                full_vectors_by_unit=replay_view["full_vectors_by_unit"],
                traces_by_unit_id=replay_view["traces_by_unit_id"],
                agent_id_by_unit=replay_view["agent_id_by_unit"],
                concept_key_by_unit=replay_view["concept_key_by_unit"],
                use_case_id_by_unit=replay_view["use_case_id_by_unit"],
                business_use_case_guid_by_unit=replay_view["business_use_case_guid_by_unit"],
                token_count_by_unit=replay_view["token_count_by_unit"],
                metadata_by_unit=replay_view["metadata_by_unit"],
                budget_levels=budget_levels,
                seed=int(replay_seed),
                window_id=f"{dataset_id}|rep-{repetition_index}",
                minhash_records_by_unit_id=replay_view["minhash_records_by_unit_id"],
                reduced_vectors_by_unit=replay_view["reduced_vectors_by_unit"],
                pca_shared_latency_seconds=float(pca["latency_seconds"]),
                replay_plan=plan,
                source_corpus_census_pass_rate=float(source_census_pass_rate),
                source_unit_id_by_unit=replay_view["source_unit_id_by_unit"],
            )
            for row in seed_result["runs"]:
                    method_id = str(row["method_id"])
                    if method_id not in method_order:
                        continue
                    search_evidence = {"query_count": 0, "neighbor_hits": 0, "latency_seconds": 0.0, "index_name": None, "metric": None}
                    if not skip_search_sync and active_search_adapter is not None and method_id in (V7_METHOD_IDS["pca8_cosine"], V7_METHOD_IDS["pca8_euclidean"]):
                        metric_name = "cosine" if method_id == V7_METHOD_IDS["pca8_cosine"] else "euclidean"
                        source_map = row.get("source_unit_id_by_unit") or {}
                        search_evidence = {
                            **dict(shared_search_probe_by_metric.get(metric_name) or {}),
                            "selected_occurrence_probe_ids": list(row.get("selected_ids") or [])[:3],
                            "selected_source_probe_ids": [str(source_map.get(uid, uid)) for uid in list(row.get("selected_ids") or [])[:3]],
                            "source_index_scope": f"{dataset_id}-{V7_VERSION}",
                        }

                    lat = dict(row.get("latency_seconds") or {})
                    lat["search_evidence"] = float(search_evidence["latency_seconds"])
                    lat["per_method_total"] = float(lat.get("per_method_total") or 0.0) + float(search_evidence["latency_seconds"])
                    base = {
                        "version": V7_VERSION,
                        "transductive": True,
                        "dataset_id": dataset_id,
                        "seed": int(replay_seed),
                        "budget_pct": int(row.get("budget_pct") or 0),
                        "cap": int(row.get("cap") or 0),
                        "method_id": method_id,
                        "estimator_type": str(row.get("estimator_type") or "unknown"),
                        "estimate": float(row.get("estimate") or 0.0),
                        "selected_rate": float(row.get("selected_rate") or 0.0),
                        "census_pass_rate": float(row.get("census_pass_rate") or 0.0),
                        "replay_census_pass_rate": float(row.get("replay_census_pass_rate") or 0.0),
                        "source_corpus_census_pass_rate": float(row.get("source_corpus_census_pass_rate") or 0.0),
                        "aggregate_pass_rate_mae": float(row.get("aggregate_pass_rate_mae") or 0.0),
                        "selected_only_pass_rate_mae": float(row.get("selected_only_pass_rate_mae") or 0.0),
                        "replay_aggregate_pass_rate_mae": float(row.get("replay_aggregate_pass_rate_mae") or 0.0),
                        "replay_selected_only_pass_rate_mae": float(row.get("replay_selected_only_pass_rate_mae") or 0.0),
                        "actual_token_count": int(row.get("actual_token_count") or 0),
                        "selected_ids": list(row.get("selected_ids") or []),
                        "latency_seconds": lat,
                        "coverage": dict(row.get("coverage") or {}),
                        "diagnostics": dict(row.get("diagnostics") or {}),
                        "embedding_metrics": row.get("embedding_metrics"),
                        "search_evidence": search_evidence,
                        "repetition_index": int(row.get("repetition_index") or repetition_index),
                        "replay_seed": int(row.get("replay_seed") or replay_seed),
                        "replay_id": str(row.get("replay_id") or plan.replay_id),
                        "replay_order_sha256": str(row.get("replay_order_sha256") or plan.order_sha256),
                        "replay_frequency_sha256": str(row.get("replay_frequency_sha256") or plan.frequency_sha256),
                        "replay_frequency_summary": dict(row.get("replay_frequency_summary") or {}),
                        "paired_replay": True,
                        "randomized_order": True,
                        "randomized_frequency": True,
                    }
                    run_rows.append(base)
                    source_map = row.get("source_unit_id_by_unit") or {}
                    memberships.append({
                        "version": V7_VERSION,
                        "dataset_id": dataset_id,
                        "seed": int(replay_seed),
                        "repetition_index": int(row.get("repetition_index") or repetition_index),
                        "replay_id": str(row.get("replay_id") or plan.replay_id),
                        "replay_order_sha256": str(row.get("replay_order_sha256") or plan.order_sha256),
                        "replay_frequency_sha256": str(row.get("replay_frequency_sha256") or plan.frequency_sha256),
                        "budget_pct": int(row.get("budget_pct") or 0),
                        "method_id": method_id,
                        "cap": int(row.get("cap") or 0),
                        "sample_size": int(len(row.get("selected_ids") or [])),
                        "selected_occurrence_ids": list(row.get("selected_ids") or []),
                        "selected_source_unit_ids": [str(source_map.get(uid, uid)) for uid in list(row.get("selected_ids") or [])],
                        "transductive": True,
                    })
                    for item in list(row.get("imputation_rows") or []):
                        if bool(item.get("observed")):
                            continue
                        uid = str(item["unit_id"])
                        source_map = row.get("source_unit_id_by_unit") or {}
                        imputed_predictions.append({
                            "version": V7_VERSION,
                            "transductive": True,
                            "dataset_id": dataset_id,
                            "seed": int(replay_seed),
                            "repetition_index": int(row.get("repetition_index") or repetition_index),
                            "replay_id": str(row.get("replay_id") or plan.replay_id),
                            "budget_pct": int(row.get("budget_pct") or 0),
                            "method_id": method_id,
                            "unit_id": uid,
                            "source_unit_id": str(source_map.get(uid, uid)),
                            "expected_label": int((replay_view["labels_by_unit"] or {}).get(uid, 0)),
                            "predicted_binary": int(item["binary"]),
                            "probability": float(item["probability"]),
                            "observed": False,
                            "provenance": str(item.get("provenance") or "idw"),
                            "donor_scope": str(item.get("donor_scope") or "unknown"),
                        })

    aggregate_metrics_payload = _build_aggregate_metrics(run_rows)
    paired_differences_payload = _build_paired_differences(run_rows)

    summary = {
        "version": V7_VERSION,
        "transductive": True,
        "dataset_count": len(datasets),
        "seed_count": len(repetition_seeds),
        "repetition_count": int(repetition_count),
        "base_seed": int(base_seed),
        "paired_replay": True,
        "randomized_order": True,
        "randomized_frequency": True,
        "budget_count": len(budget_levels),
        "method_count": len(method_order),
        "run_count": len(run_rows),
        "output_dir": str(output_path),
        "generated_at": _utc_now_iso(),
        "is_live": bool(azure_config is not None),
        "skip_search_sync": bool(skip_search_sync),
        "method_id_order": list(method_order),
        "expected_default_shape": len(datasets) * V7_DEFAULT_REPETITIONS * len(V7_BUDGET_LEVELS) * len(V7_METHOD_ID_ORDER),
        "configured_shape": len(datasets) * len(repetition_seeds) * len(budget_levels) * len(method_order),
    }
    aggregate = {
        "version": V7_VERSION,
        "transductive": True,
        "summary": summary,
        "datasets": dataset_profiles,
        "shared_preprocessing": shared_preprocessing,
        "runs": run_rows,
        "replays": replay_rows,
        "aggregate_metrics": aggregate_metrics_payload,
        "paired_differences": paired_differences_payload,
        "source": "sampling_comparison.v7_experiment",
        "cloud_metadata": {
            "azure_config_provided": azure_config is not None,
            "search_sync_skipped": bool(skip_search_sync),
            "live_search_required": not bool(skip_search_sync),
        },
    }
    _write_json(output_path / "aggregate.json", aggregate)
    _write_jsonl(output_path / "runs.jsonl", run_rows)
    _write_jsonl(output_path / "memberships.jsonl", memberships)
    _write_jsonl(output_path / "imputed_predictions.jsonl", imputed_predictions)
    _write_jsonl(output_path / "replays.jsonl", replay_rows)
    _write_json(output_path / "aggregate_metrics.json", aggregate_metrics_payload)
    _write_json(output_path / "paired_differences.json", paired_differences_payload)
    _write_json(output_path / "dataset_profile.json", {"datasets": dataset_profiles, "shared_preprocessing": shared_preprocessing, "generated_at": _utc_now_iso()})
    (output_path / "methodology.md").write_text(
        "\n".join(
            [
                "# Sampling V7 Methodology",
                "",
                "## Arms",
                "1. random_sampling: deterministic exact-cap baseline.",
                "2. minhash_lsh: stream-order MinHash novelty+rarity scoring; exact-cap top selection.",
                "3. pca8_idw_binary_cosine: corpus-level transductive PCA-8 fit once on unique source corpus, cosine diversity selection on replay occurrences, then binary IDW with threshold 0.5.",
                "4. pca8_idw_binary_euclidean: same shared source-corpus PCA-8 projection with euclidean distance for both selection and IDW on replay occurrences.",
                "5. arm5_hajek_weighted: v6 Arm5 Hajek-weighted estimator.",
                "",
                "## Replay Method",
                "- For each dataset and repetition, N replay occurrences are sampled with replacement from N source sessions.",
                "- Replay order and source frequency are randomized per repetition and paired across all methods/budgets.",
                "- Repeated occurrences are a sensitivity analysis of frequency/order perturbations, not independent new labels.",
                "",
                "## Imputation and Labeling",
                "- Binary threshold is fixed at 0.5 for embedding IDW probabilities.",
                "- IDW donor selection uses within-agent neighbors first, then global fallback if no within-agent judged donors exist.",
                "- Expected label mapping is good=1, bad=0, partial defaults to 0 unless overridden.",
                "- expected_outcome and correlated fields are excluded from representation/embedding packets.",
                "",
                "## Search Evidence",
                "- Search sync writes reduced vectors and validates successful indexing responses.",
                "- Search evidence checks non-empty neighbor retrieval on selected probes and retries with bounded attempts for eventual consistency.",
                "- Search evidence is validation-only; it does not alter deterministic selection outputs.",
                "",
                "## Estimator Comparability",
                "- Headline aggregate and selected-only MAE use the fixed source-corpus census target.",
                "- Replay-relative MAE is retained separately to expose sensitivity to each bootstrap frequency draw.",
                "- Aggregate MAE spans heterogeneous estimators and is reported with estimator_type per row.",
                "- Selected-only MAE is reported for apples-to-apples selection quality comparison.",
                "",
                "## Shared Cloud Cost",
                "- Embedding and PCA preprocessing are shared corpus costs and recorded once per dataset in shared_preprocessing/embedding_ledger.",
                "- Per-row latency excludes additive duplication of shared preprocessing overhead.",
                "- Search writes are synced per unique source corpus and metric, then shared across replay repetitions.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(output_path / "search_ledger.json", {"rows": search_ledger, "generated_at": _utc_now_iso()})
    _write_json(output_path / "embedding_ledger.json", {"rows": embedding_ledger, "generated_at": _utc_now_iso()})
    report_html = build_v7_report_html(
        {
            "runs": run_rows,
            "summary": summary,
            "datasets": dataset_profiles,
            "aggregate_metrics": aggregate_metrics_payload,
            "paired_differences": paired_differences_payload,
        }
    )
    (output_path / "interactive_report.html").write_text(report_html, encoding="utf-8")
    artifact_files = {
        "aggregate": output_path / "aggregate.json",
        "runs": output_path / "runs.jsonl",
        "replays": output_path / "replays.jsonl",
        "memberships": output_path / "memberships.jsonl",
        "imputed_predictions": output_path / "imputed_predictions.jsonl",
        "aggregate_metrics": output_path / "aggregate_metrics.json",
        "paired_differences": output_path / "paired_differences.json",
        "dataset_profile": output_path / "dataset_profile.json",
        "methodology": output_path / "methodology.md",
        "interactive_report": output_path / "interactive_report.html",
        "search_ledger": output_path / "search_ledger.json",
        "embedding_ledger": output_path / "embedding_ledger.json",
    }
    manifest = {
        "version": V7_VERSION,
        "transductive": True,
        "summary": summary,
        "files": {k: str(v) for k, v in artifact_files.items()},
        "hashes": {k: _sha256_file(v) for k, v in artifact_files.items()},
        "search_sync": {
            "skip": bool(skip_search_sync),
            "required_for_pca8": not bool(skip_search_sync),
            "cosine_index": cosine_index_name,
            "euclidean_index": euclidean_index_name,
        },
        "shared_preprocessing": shared_preprocessing,
        "embedding_ledger": embedding_ledger,
        "search_ledger": search_ledger,
    }
    _write_json(output_path / "manifest.json", manifest)
    return {"version": V7_VERSION, "transductive": True, "summary": summary, "runs": run_rows, "replays": replay_rows, "memberships": memberships, "imputed_predictions": imputed_predictions, "dataset_profile": dataset_profiles, "output_dir": str(output_path), "report_path": str(output_path / "interactive_report.html"), "aggregate": aggregate}


__all__ = [
    "BinaryPopulationResult",
    "ReplayPlan",
    "V7_BUDGET_LEVELS",
    "V7_DEFAULT_BASE_SEED",
    "V7_DEFAULT_REPETITIONS",
    "V7_METHOD_IDS",
    "V7_METHOD_ID_ORDER",
    "V7_PCA_DIMENSIONS",
    "V7_VERSION",
    "build_reduced_vector_export_documents",
    "build_v7_method_matrix",
    "build_v7_report_html",
    "compute_overlap_diagnostics",
    "derive_budget_caps",
    "embedding_binary_metrics",
    "fit_pca8_embeddings",
    "idw_binary_population",
    "make_synthetic_trace_for_text",
    "materialize_replay_dataset_view",
    "build_replay_plan",
    "run_v7_core",
    "run_v7_experiment",
    "select_diverse_exact_cap",
    "select_minhash_exact_cap",
]
