"""Offline IMDb occurrence-bootstrap replay; no embedding or judge API calls.

Input rows are source reviews, not replay occurrences. Production callers supply
the actual native text-embedding-3-small cache (50,000 x 1,536) and its provenance
in ``profile``. Small native-width fixtures are supported, but never represented
as measured IMDb results. Each seed draws len(labels) occurrences with replacement.

ARM2 membership is label-blind but ranked over the FULL schedule. Only inference
and calibration are time-causal. Exact distances cost O(N * budget * dimension)
time, not O(N**2) memory. Selection deliberately reuses the original ARM2 code,
including its TTL, novelty, rarity and proposed-keep semantics.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import scipy
from threadpoolctl import threadpool_info

from sampling_comparison.idw_threshold_experiment import exact_roc, threshold_metrics
from sampling_comparison.matryoshka_experiment import (
    build_schedule, canonical, normalized_prefix, rank_membership, sha256_file, write_json,
)
from sampling_comparison.v4_idw import IDWConfig
from trace_sampling.lipschitz import (
    LipschitzEstimate, LipschitzEstimatorConfig, calculate_conditional_geodesic_bounds,
)


VERSION = "imdb-sampling-v1"
DIMENSIONS = (1536, 32, 24, 16, 12, 8)
RATES = (.01, .02, .05, .10, .20)
SCHEDULES = ("uniformly_random", "bursty")
PROVENANCE_CODES = {"prior": 0, "global_mean": 1, "idw": 2, "exact_match": 3}
LIPSCHITZ_CONFIG = LipschitzEstimatorConfig(
    quantile=.90, theta_floor=.01, conservative_fallback=1.0,
)
METRICS = ("mae", "accuracy", "precision", "recall", "f1", "auc", "brier")
POPULATIONS = (
    "all_unselected", "eligible_point", "eligible_lower", "novel_source",
    "repeated_source", "novel_eligible_point", "novel_eligible_lower",
    "repeated_eligible_point", "repeated_eligible_lower", "combined_secondary",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Bind shape, dtype and C-order bytes, without constructing a bytes copy."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(canonical({"shape": array.shape, "dtype": array.dtype.str}).encode())
    raw = memoryview(array).cast("B")
    for start in range(0, len(raw), 8 * 1024 * 1024):
        digest.update(raw[start:start + 8 * 1024 * 1024])
    return digest.hexdigest()


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def bootstrap_replay(source_count: int, seed: int, schedule: str) -> dict[str, np.ndarray]:
    """Draws are frozen across schedules; schedules vary order AND timestamps.

    An occurrence's stable integer ID is its draw index, scoped by seed. It is
    never the source ID: repeated reviews are separate budget/arrival units.
    """
    source_count = _positive_integer(source_count, "source_count")
    if schedule not in SCHEDULES or seed < 0:
        raise ValueError("invalid replay seed or schedule")
    sources = np.random.default_rng(seed).integers(0, source_count, source_count, dtype=np.int64)
    ids = tuple(f"seed-{seed}:occ-{i:012d}" for i in range(source_count))
    order, times = build_schedule(ids, ("imdb",) * source_count, schedule, seed)
    return {
        "occurrence_id": np.arange(source_count, dtype=np.int64),
        "source_id": sources, "order": order.astype(np.int64), "timestamps": times,
    }


def _angular_block(targets: np.ndarray, donors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inputs are float64 unit vectors; largest temporary is target x donor."""
    cosine = np.clip(targets @ donors.T, -1.0, 1.0)
    return cosine, np.arccos(cosine) / math.pi


def _nearest_k(distances: np.ndarray, ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact partial selection with deterministic (distance, occurrence ID) ties."""
    width = min(k, distances.shape[1])
    indices = np.argpartition(distances, width - 1, axis=1)[:, :width]
    cutoff = np.take_along_axis(distances, indices, axis=1).max(axis=1)
    # argpartition is deliberately not relied on for kth-boundary ties.
    ties = distances == cutoff[:, None]
    below = distances < cutoff[:, None]
    repair = ties.sum(axis=1) > width - below.sum(axis=1)
    for row in np.flatnonzero(repair):
        first = np.flatnonzero(below[row])
        equal = np.flatnonzero(ties[row])
        equal = equal[np.argsort(ids[row, equal], kind="stable")]
        indices[row] = np.r_[first, equal[:width - len(first)]]
    d = np.take_along_axis(distances, indices, axis=1)
    chosen = np.take_along_axis(ids, indices, axis=1)
    order = np.lexsort((chosen, d), axis=1)
    return np.take_along_axis(d, order, axis=1), np.take_along_axis(chosen, order, axis=1)


def _fallback_scores(counts: np.ndarray, sums: np.ndarray, prior: float) -> tuple[np.ndarray, np.ndarray]:
    scores = np.divide(sums, counts, out=np.full(len(counts), prior, dtype=float), where=counts > 0)
    provenance = np.where(counts > 0, PROVENANCE_CODES["global_mean"], PROVENANCE_CODES["prior"])
    return scores, provenance.astype(np.uint8)


class _CausalCalibration:
    """Bottom-hash reservoir of earlier selected occurrences, NOT latest labels.

    Admission depends only on (seed, occurrence ID), even for repeated sources.
    A changed slot updates O(reservoir_size * dimension) geometry. Quantiles of
    the bounded slope matrix are cached until another admission/replacement.
    This uses unsmoothed binary-label slopes in normalized-angular units, NOT
    the smoothed cluster-rate estimator in the live value prototype.
    """

    def __init__(self, vectors: np.ndarray, labels: np.ndarray, sources: np.ndarray,
                 seed: int, capacity: int, config: LipschitzEstimatorConfig):
        self.vectors, self.labels, self.sources = vectors, labels, sources
        self.seed, self.capacity, self.config = seed, capacity, config
        self.ids: list[int] = []
        self.heap: list[tuple[int, int, int]] = []
        self.slopes = np.zeros((capacity, capacity), dtype=np.float64)
        self.contradictions = np.zeros((capacity, capacity), dtype=bool)
        self.events: list[tuple[int, int, int, int]] = []
        self.estimate = self._estimate()
        self.max_position = -1
        self.contradiction_count = 0

    def _estimate(self) -> LipschitzEstimate:
        size = len(self.ids)
        pairs = self.slopes[np.triu_indices(size, 1)]
        fallback = size < 2
        return LipschitzEstimate(
            value=self.config.conservative_fallback if fallback else float(np.quantile(pairs, self.config.quantile)),
            provenance="configured_fallback_sparse_reservoir" if fallback else "empirical_reservoir_label_slopes",
            usable_clusters=size, pair_count=len(pairs), quantile=self.config.quantile,
            median_slope=0.0 if fallback else float(np.median(pairs)), mean_rate_variance=0.0,
        )

    def observe(self, occurrence: int, position: int) -> bool:
        priority = int(hashlib.sha256(f"{self.seed}|calibration|{occurrence}".encode()).hexdigest(), 16)
        if len(self.ids) < self.capacity:
            slot, evicted = len(self.ids), -1
            self.ids.append(occurrence)
        else:
            worst_priority, worst_id, slot = self.heap[0]
            if (priority, occurrence) >= (-worst_priority, -worst_id):
                return False
            heapq.heappop(self.heap)
            evicted = self.ids[slot]
            self.ids[slot] = occurrence
        heapq.heappush(self.heap, (-priority, -occurrence, slot))
        source = self.sources[occurrence]
        members = self.sources[self.ids]
        cosine, distances = _angular_block(self.vectors[source:source + 1], self.vectors[members])
        distances[0, members == source] = 0.0
        delta = np.abs(self.labels[source] - self.labels[members])
        slopes = delta / np.maximum(distances[0], self.config.theta_floor)
        self.slopes[slot, :len(members)] = slopes
        self.slopes[:len(members), slot] = slopes
        contradict = (delta > 0) & ((1.0 - cosine[0]) <= IDWConfig().exact_cosine_eps)
        self.contradictions[slot, :len(members)] = contradict
        self.contradictions[:len(members), slot] = contradict
        self.estimate = self._estimate()
        self.contradiction_count = int(np.count_nonzero(self.contradictions) // 2)
        self.max_position = position  # this admission is later than all retained members
        self.events.append((occurrence, position, slot, evicted))
        return True


def evaluate_replay(
    vectors: np.ndarray, labels: np.ndarray, source_ids: np.ndarray,
    order: np.ndarray, selected_ids: np.ndarray, *, seed: int = 13,
    target_block_size: int = 256, donor_block_size: int = 2048,
    calibration_reservoir_size: int = 128, config: IDWConfig = IDWConfig(),
) -> dict[str, np.ndarray]:
    """Return lossless per-unselected-occurrence evidence in arrival order.

    Only earlier selected occurrences supply labels. All earlier exact matches
    are averaged, even when there are more than k. Non-exact IDW uses the exact
    nearest k=8 in normalized angular geometry, inverse square (d+1e-6).
    Mean/prior fallback states are distinct. With valid single-agent vectors a
    donor always supports IDW, so global_mean is normally absent.
    """
    target_block_size = _positive_integer(target_block_size, "target_block_size")
    donor_block_size = _positive_integer(donor_block_size, "donor_block_size")
    capacity = _positive_integer(calibration_reservoir_size, "calibration_reservoir_size")
    if capacity > 128:
        raise ValueError("calibration reservoir must not exceed 128")
    if config != IDWConfig():
        raise ValueError("this protocol fixes IDWConfig to k=8, power=2, epsilon=1e-6")
    vectors = np.array(vectors, dtype=np.float64, copy=True)
    labels = np.asarray(labels)
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("vectors must be a finite nonempty matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("vectors must have finite nonzero norms")
    vectors /= norms
    if labels.shape != (len(vectors),) or not np.isin(labels, (0, 1)).all():
        raise ValueError("source labels must be aligned binary outcomes")
    # Cast after validation so negative signed deltas cannot wrap uint8.
    labels = labels.astype(np.float64)
    for values in (source_ids, order, selected_ids):
        if np.asarray(values).ndim != 1 or np.asarray(values).dtype.kind not in "iu":
            raise ValueError("occurrence/source indices must be integer vectors")
    source_ids, order, selected_ids = (np.asarray(x, dtype=np.int64) for x in (source_ids, order, selected_ids))
    n = len(source_ids)
    if n == 0 or np.any((source_ids < 0) | (source_ids >= len(vectors))):
        raise ValueError("source indices out of range")
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("order must be an occurrence permutation")
    if len(np.unique(selected_ids)) != len(selected_ids) or np.any((selected_ids < 0) | (selected_ids >= n)):
        raise ValueError("selected occurrence IDs must be unique and in range")
    selected = np.zeros(n, dtype=bool)
    selected[selected_ids] = True
    positions = np.empty(n, dtype=np.int64)
    positions[order] = np.arange(n)
    target_ids = order[~selected[order]]
    target_positions = positions[target_ids]
    donors = np.sort(selected_ids)
    donor_positions = positions[donors]
    # Distance blocks traverse chronological donors so wholly future blocks can
    # be skipped. Top-k ties still use occurrence ID, not traversal order.
    chronological_donors = order[selected[order]]
    ordered_selected = selected[order]
    earlier_count = np.cumsum(ordered_selected) - ordered_selected
    ordered_labels = np.where(ordered_selected, labels[source_ids[order]], 0.0)
    earlier_sum = np.cumsum(ordered_labels) - ordered_labels
    scores, provenance = _fallback_scores(earlier_count[target_positions], earlier_sum[target_positions], config.prior)
    m = len(target_ids)
    evidence = {
        "occurrence_id": target_ids, "source_id": source_ids[target_ids], "position": target_positions,
        "label": labels[source_ids[target_ids]].astype(np.uint8),
        "score": scores, "provenance": provenance,
        "earlier_selected_count": earlier_count[target_positions],
        "exact_count": np.zeros(m, dtype=np.int64),
        "used_donor_count": np.zeros(m, dtype=np.int64),
        "max_donor_position": np.full(m, -1, dtype=np.int64),
        "neighbor_occurrence_id": np.full((m, config.k), -1, dtype=np.int64),
        "neighbor_distance": np.full((m, config.k), np.nan),
        "weighted_distance": np.full(m, np.nan),
    }
    first_judged = np.full(len(vectors), n, dtype=np.int64)
    np.minimum.at(first_judged, source_ids[donors], donor_positions)
    # Source identity is the input ROW INDEX, not an embedding/text identity.
    # Distinct source reviews with duplicate content can be novel_source AND
    # exact_match. Only an earlier judgment of this same row makes it repeated.
    evidence["novel_source"] = first_judged[source_ids[target_ids]] >= target_positions
    for start in range(0, m, target_block_size):
        stop = min(start + target_block_size, m)
        target = target_ids[start:stop]
        pos = positions[target]
        size = len(target)
        nearest_d = np.full((size, config.k), np.inf)
        nearest_id = np.full((size, config.k), n, dtype=np.int64)
        exact_count = np.zeros(size, dtype=np.int64)
        exact_sum = np.zeros(size)
        exact_distance = np.zeros(size)
        exact_max_position = np.full(size, -1, dtype=np.int64)
        target_vectors = vectors[source_ids[target]]
        for begin in range(0, len(chronological_donors), donor_block_size):
            ids = chronological_donors[begin:begin + donor_block_size]
            dpos = positions[ids]
            if dpos.min() >= pos.max():
                break
            cosine, distance = _angular_block(target_vectors, vectors[source_ids[ids]])
            same_source = source_ids[target, None] == source_ids[ids][None, :]
            distance[same_source] = 0.0
            valid = dpos[None, :] < pos[:, None]
            exact = valid & (((1.0 - cosine) <= config.exact_cosine_eps) | same_source)
            exact_count += exact.sum(axis=1)
            exact_sum += exact @ labels[source_ids[ids]]
            exact_distance += np.where(exact, distance, 0.0).sum(axis=1)
            exact_max_position = np.maximum(exact_max_position, np.where(exact, dpos[None, :], -1).max(axis=1))
            distance[~valid] = np.inf
            candidate_d = np.concatenate((nearest_d, distance), axis=1)
            candidate_id = np.concatenate((nearest_id, np.broadcast_to(ids, distance.shape)), axis=1)
            nearest_d, nearest_id = _nearest_k(candidate_d, candidate_id, config.k)
        valid = np.isfinite(nearest_d)
        safe_ids = np.where(valid, nearest_id, 0)
        weights = np.where(valid, 1.0 / (nearest_d + config.eps) ** config.power, 0.0)
        totals = weights.sum(axis=1)
        supported = totals > 0
        estimates = np.divide(
            (weights * labels[source_ids[safe_ids]]).sum(axis=1), totals,
            out=scores[start:stop].copy(), where=supported,
        )
        weighted_distance = np.divide(
            (weights * np.where(valid, nearest_d, 0.0)).sum(axis=1), totals,
            out=np.full(size, np.nan), where=supported,
        )
        codes = provenance[start:stop].copy()
        codes[supported] = PROVENANCE_CODES["idw"]
        exact_supported = exact_count > 0
        estimates[exact_supported] = exact_sum[exact_supported] / exact_count[exact_supported]
        weighted_distance[exact_supported] = exact_distance[exact_supported] / exact_count[exact_supported]
        codes[exact_supported] = PROVENANCE_CODES["exact_match"]
        scores[start:stop] = np.clip(estimates, 0.0, 1.0)
        provenance[start:stop] = codes
        evidence["exact_count"][start:stop] = exact_count
        evidence["used_donor_count"][start:stop] = np.where(exact_supported, exact_count, valid.sum(axis=1))
        max_position = np.where(valid, positions[safe_ids], -1).max(axis=1)
        evidence["max_donor_position"][start:stop] = np.where(exact_supported, exact_max_position, max_position)
        # Neighbors describe ordinary IDW only. Exact donors are not truncated;
        # recover them from membership + geometry; count/mean distance are kept.
        ordinary = valid & ~exact_supported[:, None]
        evidence["neighbor_occurrence_id"][start:stop] = np.where(ordinary, nearest_id, -1)
        evidence["neighbor_distance"][start:stop] = np.where(ordinary, nearest_d, np.nan)
        evidence["weighted_distance"][start:stop] = weighted_distance

    for name in ("lower", "upper", "raw_lower", "raw_upper", "lipschitz"):
        evidence[name] = np.full(m, np.nan)
    for name in ("calibration_donor_count", "calibration_pair_count", "calibration_version",
                 "calibration_exact_contradiction_count"):
        evidence[name] = np.zeros(m, dtype=np.int64)
    evidence["calibration_fallback"] = np.zeros(m, dtype=bool)
    evidence["max_calibration_position"] = np.full(m, -1, dtype=np.int64)
    calibration = _CausalCalibration(vectors, labels, source_ids, seed, capacity, LIPSCHITZ_CONFIG)
    offset = 0
    for position, occurrence in enumerate(order):
        if selected[occurrence]:
            calibration.observe(int(occurrence), position)
            continue
        estimate = calibration.estimate
        evidence["calibration_donor_count"][offset] = len(calibration.ids)
        evidence["calibration_pair_count"][offset] = estimate.pair_count
        evidence["calibration_version"][offset] = len(calibration.events)
        evidence["max_calibration_position"][offset] = calibration.max_position
        evidence["calibration_fallback"][offset] = estimate.provenance.startswith("configured_fallback")
        evidence["calibration_exact_contradiction_count"][offset] = calibration.contradiction_count
        if np.isfinite(evidence["weighted_distance"][offset]):
            bounds = calculate_conditional_geodesic_bounds(
                float(scores[offset]), float(evidence["weighted_distance"][offset]), estimate,
            ).probability
            for name in ("lower", "upper", "raw_lower", "raw_upper"):
                evidence[name][offset] = getattr(bounds, name)
            evidence["lipschitz"][offset] = estimate.value
        offset += 1
    evidence["calibration_events"] = np.asarray(calibration.events, dtype=np.int64).reshape(-1, 4)
    evidence["selected_occurrence_id"] = donors
    eligible = np.isfinite(evidence["lower"])
    if np.any(evidence["lower"][eligible] > scores[eligible] + 1e-12):
        raise AssertionError("lower envelope exceeds point score")
    if np.any(evidence["max_donor_position"] >= target_positions):
        raise AssertionError("target/future donor used")
    if np.any(evidence["max_calibration_position"] >= target_positions):
        raise AssertionError("target/future calibration used")
    if not np.isfinite(scores).all():
        raise AssertionError("nonfinite prediction")
    return evidence


def _quality(labels: np.ndarray, scores: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    roc = exact_roc(labels, scores)
    metrics = {
        **threshold_metrics(labels, scores, .5),
        "mae": float(np.mean(np.abs(labels - scores))) if len(labels) else None,
        "brier": float(np.mean((labels - scores) ** 2)) if len(labels) else None,
        "auc": roc["auc"], "auc_reason": roc["reason"],
    }
    # A report-size display grid, NEVER the input to AUC. Lossless float64
    # retained scores permit full >=-threshold ROC/threshold-curve reconstruction.
    grid = np.linspace(0, 1, 101)
    curve = {
        "fpr": grid.tolist(),
        "tpr": np.interp(grid, roc["fpr"], roc["tpr"]).tolist() if roc["auc"] is not None else [None] * len(grid),
        "auc": roc["auc"], "reason": roc["reason"],
        "exact_point_count": len(roc["thresholds"]),
        "representation": "display_grid_interpolated_from_exact_roc",
    }
    return metrics, curve


def cell_metrics(evidence: dict[str, np.ndarray], selected_labels: np.ndarray) -> dict[str, Any]:
    y, point, lower = evidence["label"], evidence["score"], evidence["lower"]
    eligible = np.isfinite(lower)
    novel = evidence["novel_source"]
    populations = {
        "all_unselected": (np.ones(len(y), dtype=bool), point),
        "eligible_point": (eligible, point), "eligible_lower": (eligible, lower),
        "novel_source": (novel, point), "repeated_source": (~novel, point),
        "novel_eligible_point": (novel & eligible, point),
        "novel_eligible_lower": (novel & eligible, lower),
        "repeated_eligible_point": (~novel & eligible, point),
        "repeated_eligible_lower": (~novel & eligible, lower),
    }
    result: dict[str, Any] = {"roc": {}}
    for name, (mask, values) in populations.items():
        result[name], curve = _quality(y[mask], values[mask])
        if name in ("all_unselected", "eligible_point", "eligible_lower"):
            key = {"eligible_point": "point", "eligible_lower": "lower"}.get(name, name)
            result["roc"][key] = curve
    combined_y = np.r_[y, selected_labels]
    result["combined_secondary"], _ = _quality(combined_y, np.r_[point, selected_labels])
    provenance = Counter(evidence["provenance"].tolist())
    eligible_count = int(eligible.sum())
    result["counts"] = {
        "occurrences": len(combined_y), "selected": len(selected_labels), "unselected": len(y),
        "envelope_eligible": eligible_count,
        "envelope_ineligible": int((~eligible).sum()),
        "envelope_coverage": eligible_count / len(y) if len(y) else None,
        "novel_source": int(novel.sum()), "repeated_source": int((~novel).sum()),
        "repeated_source_exact_reuse": int(np.sum(~novel & (evidence["provenance"] == 3))),
        "calibration_fallback_unselected": int(evidence["calibration_fallback"].sum()),
        "calibration_fallback_eligible": int(np.sum(evidence["calibration_fallback"] & eligible)),
        "calibration_empirical_eligible": int(np.sum(~evidence["calibration_fallback"] & eligible)),
        "calibration_updates": len(evidence["calibration_events"]),
        "provenance": {name: provenance[code] for name, code in PROVENANCE_CODES.items()},
    }
    result["envelope_label_coverage"] = (
        float(np.mean((y[eligible] >= lower[eligible]) & (y[eligible] <= evidence["upper"][eligible])))
        if eligible_count else None
    )
    return result


def _method_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    # Conservative local dependency closure. Unrelated report/CLI changes do
    # not invalidate a run; all trace helper changes do.
    paths = list((root / "trace_sampling").rglob("*.py"))
    paths += [Path(__file__).resolve()] + [
        root / "sampling_comparison" / name
        for name in ("matryoshka_experiment.py", "idw_threshold_experiment.py", "v4_idw.py")
    ]
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(set(paths))}


def _runtime() -> dict[str, Any]:
    keys = ("user_api", "internal_api", "version", "num_threads", "threading_layer", "architecture", "prefix")
    return {
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "threadpools": [{k: pool.get(k) for k in keys} for pool in threadpool_info()],
        "thread_environment": {k: os.environ.get(k) for k in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        )},
    }


def _write_record(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, {**payload, "record_sha256": _digest(payload)})


def _read_record(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        digest = payload.pop("record_sha256")
        if digest != _digest(payload):
            raise ValueError("content hash mismatch")
        return payload
    except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"checkpoint hash/format mismatch: {path}") from exc


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _load_checkpoint(path: Path, binding: str, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    record = _read_record(path)
    if record["binding_sha256"] != binding or record["identity"] != identity:
        raise ValueError(f"incompatible checkpoint binding: {path}")
    evidence = path.with_suffix(".npz")
    if not evidence.exists() or sha256_file(evidence) != record["evidence_sha256"]:
        raise ValueError(f"evidence hash mismatch: {evidence}")
    return record


def _checkpoint(path: Path, binding: str, identity: dict[str, Any],
                arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> dict[str, Any]:
    evidence = path.with_suffix(".npz")
    _save_npz(evidence, arrays)
    record = {
        "binding_sha256": binding, "identity": identity,
        "evidence_sha256": sha256_file(evidence), **metadata,
    }
    _write_record(path, record)
    return record


def _quantiles(values: Sequence[float | None]) -> dict[str, Any]:
    values = [v for v in values if v is not None]
    return {
        "n": len(values), "mean": float(np.mean(values)) if values else None,
        "q025": float(np.quantile(values, .025)) if values else None,
        "q975": float(np.quantile(values, .975)) if values else None,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Equal-replay summaries; paired differences use the same seed/schedule/rate."""
    native = {(r["seed"], r["schedule"], r["rate"]): r for r in rows if r["dimension"] == 1536}
    groups: dict[tuple[int, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        reference = native[(row["seed"], row["schedule"], row["rate"])]
        row["paired_delta_native"] = {
            population: {
                metric: row[population][metric] - reference[population][metric]
                if row[population][metric] is not None and reference[population][metric] is not None else None
                for metric in METRICS
            } for population in POPULATIONS
        }
        groups.setdefault((row["dimension"], row["schedule"], row["rate"]), []).append(row)
    summaries = []
    for (dimension, schedule, rate), cells in sorted(groups.items()):
        item = {
            "dimension": dimension, "schedule": schedule, "rate": rate,
            "budget": cells[0]["budget"], "replay_count": len(cells),
            "interval_kind": "2.5/97.5 percentiles of replay sensitivity; not production CI",
            "paired_delta_native": {},
            "envelope_coverage": _quantiles([c["counts"]["envelope_coverage"] for c in cells]),
            "envelope_label_coverage": _quantiles([c["envelope_label_coverage"] for c in cells]),
            "counts": {
                name: _quantiles([c["counts"][name] for c in cells])
                for name in cells[0]["counts"] if name != "provenance"
            },
        }
        item["counts"]["provenance"] = {
            name: _quantiles([c["counts"]["provenance"][name] for c in cells])
            for name in PROVENANCE_CODES
        }
        for population in POPULATIONS:
            item[population] = {
                metric: _quantiles([c[population][metric] for c in cells])
                for metric in ("n",) + METRICS
            }
            item["paired_delta_native"][population] = {
                metric: _quantiles([c["paired_delta_native"][population][metric] for c in cells])
                for metric in METRICS
            }
        summaries.append(item)
    return summaries


def run_experiment(
    vectors: np.ndarray, labels: np.ndarray, output: Path, *, profile: dict,
    dimensions: Sequence[int] = DIMENSIONS, repetitions: int = 40, base_seed: int = 13,
    rates: Sequence[float] = RATES, schedules: Sequence[str] = SCHEDULES,
    resume: bool = False, target_block_size: int = 256, donor_block_size: int = 2048,
    calibration_reservoir_size: int = 128,
) -> dict[str, Any]:
    """Run/resume a pre-registered paired bootstrap, using only supplied inputs.

    Production defaults: 40 x 2 schedules x 6 dimensions x 5 rates = 2,400
    cells, N=50,000 if supplied all 50k source reviews. Repetitions/dimensions
    can be reduced for offline fixtures. Nonempty output requires resume=True;
    input, method, runtime, configuration and checkpoint hashes must all match.
    A single writer per output directory is required.
    """
    vectors, labels = np.asarray(vectors), np.asarray(labels)
    if vectors.ndim != 2 or vectors.shape[1] != 1536 or len(vectors) < 2 or vectors.dtype.kind != "f":
        raise ValueError("provide native floating-point vectors with shape [N,1536], N >= 2")
    if not np.isfinite(vectors).all():
        raise ValueError("nonfinite native vectors")
    if labels.shape != (len(vectors),) or not np.isin(labels, (0, 1)).all():
        raise ValueError("provide aligned binary source labels")
    if not isinstance(profile, dict):
        raise ValueError("profile must be a JSON dataset provenance dictionary")
    canonical(profile)
    if any(profile[key] != "text-embedding-3-small" for key in ("embedding_model_id", "model") if key in profile):
        raise ValueError("production input must use text-embedding-3-small")
    repetitions = _positive_integer(repetitions, "repetitions")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    dimensions, rates, schedules = tuple(dimensions), tuple(rates), tuple(schedules)
    if (not dimensions or dimensions[0] != 1536 or len(set(dimensions)) != len(dimensions)
            or any(isinstance(d, bool) or not isinstance(d, int) or not 1 <= d <= 1536 for d in dimensions)):
        raise ValueError("dimensions must be unique integers starting with native 1536")
    if not rates or len(set(rates)) != len(rates) or any(isinstance(r, bool) or not 0 < r < 1 for r in rates):
        raise ValueError("rates must be unique fractions in (0,1)")
    if not schedules or len(set(schedules)) != len(schedules) or set(schedules) - set(SCHEDULES):
        raise ValueError("schedules must be unique known schedules")
    target_block_size = _positive_integer(target_block_size, "target_block_size")
    donor_block_size = _positive_integer(donor_block_size, "donor_block_size")
    calibration_reservoir_size = _positive_integer(calibration_reservoir_size, "calibration_reservoir_size")
    if calibration_reservoir_size > 128:
        raise ValueError("calibration reservoir must not exceed 128")
    # Validate every prefix BEFORE registering or spending time on any cells.
    for dimension in dimensions:
        normalized_prefix(vectors, dimension)
    seeds = list(range(base_seed, base_seed + repetitions))
    protocol = {
        "dimensions": list(dimensions), "repetitions": repetitions, "seeds": seeds,
        "rates": list(rates), "schedules": list(schedules), "mode": "end_to_end",
        "source_count": len(labels), "occurrences_per_replay": len(labels), "agents": 1,
        "planned_cells": len(dimensions) * repetitions * len(rates) * len(schedules),
        "embedding": "caller-supplied text-embedding-3-small native 1536; first d coordinates L2-normalized; no PCA",
        "embedding_provenance": "profile supplied by preparation; engine makes no embedding or judge calls",
        "bootstrap": "N draws with replacement from ALL source rows; PCG64(seed); same draws across schedules",
        "occurrence_identity": "draw index scoped by seed; vectors/labels indexed by source_id",
        "pairing": "identical draw/order/timestamps across dimensions and budgets within seed/schedule",
        "budget_rule": "max(1,floor(N*rate)) occurrences; no warm start or free reused-source judgments",
        "membership": "unmodified ARM2 rank_membership, tau=.55, TTL=90; constant imdb-review signature, single imdb agent",
        "online_boundary": "observations sequential; FULL-SCHEDULE membership ranking is NOT strictly online",
        "idw": asdict(IDWConfig()), "distance": "float64 renormalized prefix; acos(clipped cosine)/pi; same source distance zero",
        "donor_rule": "only strictly earlier selected OCCURRENCES by arrival position, including timestamp ties in frozen order",
        "neighbor_ties": "distance then integer occurrence ID",
        "exact_match": "all earlier selected donors with 1-cos<=1e-8, not only k; equal weights",
        "fallbacks": "earlier selected global mean vs prior=.5; mean unreachable with valid single-agent donor geometry",
        "distance_blocks": {"targets": target_block_size, "donors": donor_block_size},
        "scaling": "exact O(N*budget*dimension) distance work in bounded blocks; no N x N matrix; original TTL ARM2 selector unchanged",
        "lipschitz": {
            **asdict(LIPSCHITZ_CONFIG), "reservoir_size": calibration_reservoir_size,
            "admission": "lowest SHA256(seed|calibration|occurrence_id), label-blind, earlier selected only",
            "estimator": "q90 abs(label_i-label_j)/max(normalized angular distance,.01) for all reservoir pairs",
            "deviation_from_prior": "bounded donor reservoir instead of all earlier selected pairs; no smoothing",
            "sparse_fallback": "L=1 for fewer than two reservoir donors; fallback envelopes explicitly counted",
            "cache": "incremental distance row; recompute quantile only when reservoir changes",
            "interpretation": "conditional sensitivity envelope, not confidence interval or calibrated guarantee",
        },
        "primary": "all_unselected point scores including separately counted mean/prior fallbacks; never direct labels",
        "matched_comparison": "eligible_point and eligible_lower share exactly the same targets within each cell",
        "source_strata": "novel_source: no earlier SELECTED occurrence of that input source row index; repeated_source otherwise; distinct rows with duplicate text/vectors remain distinct sources even for exact_match",
        "secondary": "combined_secondary mixes direct selected labels with unselected predictions",
        "metrics": "MAE, Brier, accuracy/precision/recall/F1 at score>=.5; exact tied >=-threshold ROC AUC",
        "roc_storage": "lossless float64 scores in cell NPZ; JSON curves are 101-point interpolated display grids, NOT AUC quadrature",
        "uncertainty": "equal-replay mean and 2.5/97.5 replay quantiles, NOT production CI",
        "paired_deltas": "dimension minus native per seed/schedule/rate; membership and eligible sets may differ across dimensions",
        "local_retention": "replay mappings, label-free rankings and per-target evidence stay local; not publication artifacts",
        "resume": "single writer; orphan NPZ without a committed JSON checkpoint may be recomputed; corrupted committed artifacts fail closed",
    }
    binding_payload = {
        "version": VERSION, "dataset": profile, "protocol": protocol,
        "input_hashes": {"vectors": array_sha256(vectors), "labels": array_sha256(labels)},
        "method_code_hashes": _method_hashes(), "runtime": _runtime(),
    }
    binding = _digest(binding_payload)
    output = Path(output)
    preregistration = output / "preregistration.json"
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError("nonempty output requires explicit resume=True")
        if not preregistration.exists():
            raise ValueError("cannot resume without preregistration")
        previous = _read_record(preregistration)
        if previous["binding_sha256"] != binding or previous["binding"] != binding_payload:
            raise ValueError("incompatible resume: inputs, method code, runtime, profile or configuration changed")
    else:
        output.mkdir(parents=True, exist_ok=True)
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        _write_record(preregistration, {
            "binding_sha256": binding, "binding": binding_payload, "source_revision": revision,
            "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    # A completed bundle is accepted only after verifying EVERY retained hash.
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = _read_record(manifest_path)
        if manifest["binding_sha256"] != binding or manifest["status"] != "completed":
            raise ValueError("incompatible final manifest")
        for name, digest in manifest["files"].items():
            path = output / name
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"final artifact hash mismatch: {name}")
        return json.loads((output / "aggregate.json").read_text(encoding="utf-8"))

    retained = {preregistration}
    rows = []
    for seed in seeds:
        unit_ids = tuple(f"seed-{seed}:occ-{i:012d}" for i in range(len(labels)))
        for schedule in schedules:
            replay_path = output / "replays" / f"s{seed}-{schedule}.json"
            identity = {"kind": "replay", "seed": seed, "schedule": schedule}
            replay_record = _load_checkpoint(replay_path, binding, identity)
            if replay_record is None:
                replay = bootstrap_replay(len(labels), seed, schedule)
                replay_record = _checkpoint(replay_path, binding, identity, replay, {
                    "hashes": {name: array_sha256(value) for name, value in replay.items()},
                })
            with np.load(replay_path.with_suffix(".npz"), allow_pickle=False) as arrays:
                replay = {name: arrays[name] for name in arrays.files}
            retained.update((replay_path, replay_path.with_suffix(".npz")))
            sources, order, times = replay["source_id"], replay["order"], replay["timestamps"]
            for dimension in dimensions:
                rank_path = output / "memberships" / f"s{seed}-{schedule}-d{dimension}.json"
                rank_identity = {**identity, "kind": "membership", "dimension": dimension}
                rank_record = _load_checkpoint(rank_path, binding, rank_identity)
                prefix = None  # lazy on resume; no recomputation for completed cells
                if rank_record is None:
                    prefix = normalized_prefix(vectors, dimension)
                    ranking, selector_counts = rank_membership(
                        unit_ids, ("imdb",) * len(sources), (("imdb-review",),) * len(sources),
                        prefix[sources], order, times, seed,
                    )
                    rank_record = _checkpoint(rank_path, binding, rank_identity, {"ranking": ranking}, {
                        "ranking_sha256": array_sha256(ranking), "selector_counts": selector_counts,
                        "replay_hashes": replay_record["hashes"], "label_blind": True,
                    })
                with np.load(rank_path.with_suffix(".npz"), allow_pickle=False) as arrays:
                    ranking = arrays["ranking"]
                retained.update((rank_path, rank_path.with_suffix(".npz")))
                for rate_index, rate in enumerate(rates):
                    budget = max(1, math.floor(len(sources) * rate))
                    selected = ranking[:budget]
                    cell_path = output / "cells" / f"s{seed}-{schedule}-d{dimension}-r{rate_index}.json"
                    cell_identity = {**rank_identity, "kind": "cell", "rate": rate, "budget": budget}
                    cell = _load_checkpoint(cell_path, binding, cell_identity)
                    if cell is None:
                        started = time.perf_counter()
                        if prefix is None:
                            prefix = normalized_prefix(vectors, dimension)
                        evidence = evaluate_replay(
                            prefix, labels, sources, order, selected, seed=seed,
                            target_block_size=target_block_size, donor_block_size=donor_block_size,
                            calibration_reservoir_size=calibration_reservoir_size,
                        )
                        row = {
                            "dimension": dimension, "seed": seed, "schedule": schedule, "rate": rate,
                            "budget": budget, **cell_metrics(evidence, labels[sources[selected]]),
                            "replay_hashes": replay_record["hashes"],
                            "ranking_sha256": rank_record["ranking_sha256"],
                            "membership_sha256": array_sha256(np.sort(selected)),
                            "selector_counts": rank_record["selector_counts"],
                            "evidence": cell_path.with_suffix(".npz").relative_to(output).as_posix(),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                        cell = _checkpoint(cell_path, binding, cell_identity, evidence, {"row": row})
                    row = dict(cell["row"])
                    row["evidence_sha256"] = cell["evidence_sha256"]
                    rows.append(row)
                    retained.update((cell_path, cell_path.with_suffix(".npz")))
    if len(rows) != protocol["planned_cells"]:
        raise AssertionError("incomplete replay grid cannot be finalized")
    summaries = summarize(rows)
    aggregate = {
        "version": VERSION, "status": "completed", "dataset": profile,
        "protocol": protocol, "binding_sha256": binding, "provenance": binding_payload,
        "rows": rows, "summaries": summaries,
    }
    aggregate_path = output / "aggregate.json"
    write_json(aggregate_path, aggregate)
    retained.add(aggregate_path)
    _write_record(manifest_path, {
        "version": VERSION, "status": "completed", "binding_sha256": binding,
        "completed_cells": len(rows),
        "files": {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(retained)},
    })
    return aggregate
