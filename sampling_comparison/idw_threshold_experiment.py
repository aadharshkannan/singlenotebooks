from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t
from threadpoolctl import threadpool_info

from sampling_comparison.matryoshka_experiment import (
    InputDataset,
    build_schedule,
    canonical,
    distance_blocks,
    evaluate,
    load_input,
    normalized_prefix,
    sha256_file,
    write_json,
)
from sampling_comparison.v4_idw import IDWConfig
from trace_sampling.lipschitz import (
    LipschitzEstimate,
    LipschitzEstimatorConfig,
    calculate_conditional_geodesic_bounds,
)


VERSION = "idw-threshold-envelope-v1"
DATASETS = ("historical_300", "dense_2500", "cosmos_otel")
DIMENSIONS = (1536, 8)
SEEDS = tuple(range(13, 43))
SCHEDULES = ("evenly_spaced", "uniformly_random", "bursty", "front_loaded", "agent_blocked")
RATES = (0.01, 0.02, 0.05, 0.10, 0.20)
PROVENANCE_CODES = {"prior": 0, "global_mean": 1, "idw": 2, "exact_match": 3}
INPUTS = {
    "historical_300": Path("outputs_matryoshka/cache/live-20260914/historical_300/manifest.json"),
    "dense_2500": Path("outputs_matryoshka/cache/dense_2500/manifest.json"),
    "cosmos_otel": Path("outputs_matryoshka/cache/live-20260914/cosmos_otel/manifest.json"),
}
BASELINE = Path("outputs_matryoshka/runs/mrl-three-datasets-30-seed-20260914")
LIPSCHITZ_CONFIG = LipschitzEstimatorConfig(
    quantile=0.90, theta_floor=0.01, conservative_fallback=1.0, seed=0
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_snapshot(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(set(paths), key=lambda value: str(value))
    }


def input_artifacts(manifests: Mapping[str, Path]) -> list[Path]:
    paths: list[Path] = []
    for path in manifests.values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        paths.append(path)
        paths.extend(path.parent / payload["files"][name] for name in ("units", "vectors"))
    return paths


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must align")
    if labels.size == 0:
        return {
            "n": 0, "positive_count": 0, "predicted_positive_count": 0,
            "accuracy": None, "precision": None, "recall": None, "f1": None,
            "reason": "empty_population",
        }
    predicted = scores >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    actual_positive = tp + fn
    predicted_positive = tp + fp
    precision = tp / predicted_positive if predicted_positive else 0.0
    recall = tp / actual_positive if actual_positive else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall > 0
        else (0.0 if recall is not None else None)
    )
    return {
        "n": int(labels.size), "positive_count": actual_positive,
        "predicted_positive_count": predicted_positive,
        "accuracy": (tp + tn) / labels.size, "precision": precision,
        "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "reason": None,
    }


def exact_roc(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Exact >=-threshold ROC with stable tie aggregation and explicit endpoints."""
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must align")
    if labels.size == 0:
        return {
            "thresholds": np.asarray([], dtype=np.float64),
            "fpr": np.asarray([], dtype=np.float64),
            "tpr": np.asarray([], dtype=np.float64),
            "auc": None, "reason": "empty_population",
        }
    if not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("ROC inputs must be finite scores and binary labels")
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0), labels.size - 1]
    tp = np.r_[0, np.cumsum(ordered_labels)[ends]].astype(np.float64)
    fp = np.r_[0, (ends + 1) - np.cumsum(ordered_labels)[ends]].astype(np.float64)
    thresholds = np.r_[np.inf, ordered_scores[ends]]
    tpr = tp / positives if positives else np.full(tp.shape, np.nan)
    fpr = fp / negatives if negatives else np.full(fp.shape, np.nan)
    if not positives:
        auc, reason = None, "no_positive_labels"
    elif not negatives:
        auc, reason = None, "no_negative_labels"
    else:
        auc, reason = float(np.trapezoid(tpr, fpr)), None
    return {"thresholds": thresholds, "fpr": fpr, "tpr": tpr, "auc": auc, "reason": reason}


def _display_roc(roc: Mapping[str, Any], max_points: int = 500) -> dict[str, Any]:
    n = len(roc["thresholds"])
    if n <= max_points:
        indices = np.arange(n)
    else:
        indices = np.unique(np.rint(np.linspace(0, n - 1, max_points)).astype(int))
    return {
        "fpr": [None if not np.isfinite(x) else float(x) for x in np.asarray(roc["fpr"])[indices]],
        "tpr": [None if not np.isfinite(x) else float(x) for x in np.asarray(roc["tpr"])[indices]],
        "exact_point_count": n, "display_point_count": len(indices),
    }


def _membership_hash(selected: Sequence[int]) -> str:
    return hashlib.sha256(canonical(sorted(int(x) for x in selected)).encode()).hexdigest()


def load_verified_baseline(
    baseline_dir: Path, datasets: Mapping[str, InputDataset]
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]], dict[str, Any]]:
    manifest_path = baseline_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aggregate_path = baseline_dir / "aggregate.json"
    memberships_path = baseline_dir / "memberships.jsonl.gz"
    for name, path in (("aggregate", aggregate_path), ("memberships", memberships_path)):
        expected = manifest["files"][name]["sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"baseline {name} SHA256 mismatch")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    verify_baseline_inputs(aggregate, datasets)
    protocol = aggregate["protocol"]
    expected_protocol = {
        "seeds": list(SEEDS), "schedules": list(SCHEDULES), "rates": list(RATES),
    }
    for name, expected in expected_protocol.items():
        if protocol[name] != expected:
            raise ValueError(f"baseline {name} differs from pre-registration")
    if protocol["idw"] != asdict(IDWConfig()):
        raise ValueError("baseline IDW configuration differs from pre-registration")
    rows = {
        (row["dataset_id"], row["dimension"], row["seed"], row["schedule"], row["rate"]): row
        for row in aggregate["rows"]
        if row["dataset_id"] in datasets and row["dimension"] in DIMENSIONS and row["mode"] == "end_to_end"
    }
    memberships: dict[tuple[Any, ...], dict[str, Any]] = {}
    with gzip.open(memberships_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if (
                record["dataset_id"] not in datasets
                or record["dimension"] not in DIMENSIONS
                or record["mode"] != "end_to_end"
            ):
                continue
            key = (
                record["dataset_id"], record["dimension"], record["seed"],
                record["schedule"], record["rate"],
            )
            if key in memberships:
                raise ValueError(f"duplicate baseline membership: {key}")
            selected = record["selected_indices"]
            if _membership_hash(selected) != record["membership_sha256"]:
                raise ValueError(f"membership hash mismatch: {key}")
            if key not in rows or rows[key]["membership_sha256"] != record["membership_sha256"]:
                raise ValueError("baseline membership disagrees with measured result")
            data = datasets[record["dataset_id"]]
            order, _ = build_schedule(data.unit_ids, data.agents, record["schedule"], record["seed"])
            order_hash = hashlib.sha256(canonical(order.tolist()).encode()).hexdigest()
            if order_hash != record["order_sha256"]:
                raise ValueError(f"schedule order hash mismatch: {key}")
            expected_count = max(1, math.floor(len(data.unit_ids) * record["rate"]))
            if len(selected) != expected_count or len(set(selected)) != expected_count:
                raise ValueError(f"membership count/uniqueness mismatch: {key}")
            if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(data.unit_ids) for i in selected):
                raise ValueError("baseline membership index outside input population")
            memberships[key] = record
    expected_count = len(datasets) * len(DIMENSIONS) * len(SEEDS) * len(SCHEDULES) * len(RATES)
    if len(rows) != expected_count or len(memberships) != expected_count:
        raise ValueError("baseline does not contain the complete 4,500-cell replay grid")
    return rows, memberships, {
        "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "aggregate": str(aggregate_path), "aggregate_sha256": sha256_file(aggregate_path),
        "memberships": str(memberships_path), "memberships_sha256": sha256_file(memberships_path),
        "source_revision": manifest["source_revision"],
    }


def verify_baseline_inputs(aggregate: Mapping[str, Any], datasets: Mapping[str, InputDataset]) -> None:
    profiles = {row["dataset_id"]: row for row in aggregate["datasets"] if row["status"] == "completed"}
    if set(profiles) != set(datasets):
        raise ValueError("baseline and supplied input cohorts differ")
    for name, data in datasets.items():
        profile = profiles[name]
        if data.dataset_id != name or profile["n"] != len(data.unit_ids):
            raise ValueError("baseline input population mismatch")
        if profile["provenance"]["input_manifest_sha256"] != data.profile["input_manifest_sha256"]:
            raise ValueError("baseline input manifest hash mismatch")


def _calibration_estimate(donors: Sequence[int], slopes: Sequence[float]) -> LipschitzEstimate:
    if len(donors) < 2:
        return LipschitzEstimate(
            value=LIPSCHITZ_CONFIG.conservative_fallback,
            provenance="configured_fallback_insufficient_observed_donors",
            usable_clusters=len(donors), pair_count=0, quantile=LIPSCHITZ_CONFIG.quantile,
            median_slope=0.0, mean_rate_variance=0.0,
        )
    values = np.asarray(slopes, dtype=np.float64)
    if values.size == 0:
        return LipschitzEstimate(
            value=LIPSCHITZ_CONFIG.conservative_fallback,
            provenance="configured_fallback_no_usable_pairs",
            usable_clusters=len(donors), pair_count=0, quantile=LIPSCHITZ_CONFIG.quantile,
            median_slope=0.0, mean_rate_variance=0.0,
        )
    return LipschitzEstimate(
        value=float(np.quantile(values, LIPSCHITZ_CONFIG.quantile)),
        provenance="empirical_observed_label_pair_slopes_normalized_angular",
        usable_clusters=len(donors), pair_count=len(values),
        quantile=LIPSCHITZ_CONFIG.quantile, median_slope=float(np.median(values)),
        mean_rate_variance=0.0,
    )


def envelope_evidence(
    data: InputDataset,
    blocks: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    order: np.ndarray,
    selected_ids: np.ndarray,
    scores: np.ndarray,
    provenance: np.ndarray,
    config: IDWConfig = IDWConfig(),
) -> dict[str, np.ndarray]:
    """Build target-time envelope evidence without recomputing the point estimate."""
    n = len(data.unit_ids)
    selected = np.zeros(n, dtype=bool)
    selected[selected_ids] = True
    positions = np.empty(n, dtype=np.int64)
    positions[order] = np.arange(n)
    local_maps: dict[str, dict[int, int]] = {}
    for agent, (ids, _, _) in blocks.items():
        local_maps[agent] = {int(value): i for i, value in enumerate(ids)}
    target_ids = np.flatnonzero(~selected)
    lower = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    raw_lower = np.full(n, np.nan, dtype=np.float64)
    raw_upper = np.full(n, np.nan, dtype=np.float64)
    weighted_distance = np.full(n, np.nan, dtype=np.float64)
    lipschitz = np.full(n, np.nan, dtype=np.float64)
    donor_count = np.zeros(n, dtype=np.uint16)
    pair_count = np.zeros(n, dtype=np.uint32)
    contradiction_count = np.zeros(n, dtype=np.uint16)
    max_donor_position = np.full(n, -1, dtype=np.int32)
    calibration_fallback = np.zeros(n, dtype=np.uint8)
    state: dict[str, dict[str, Any]] = {
        agent: {"donors": [], "slopes": [], "contradictions": 0, "estimate": None}
        for agent in blocks
    }
    for position, target in enumerate(order):
        target = int(target)
        agent = data.agents[target]
        ids, cosine, angular = blocks[agent]
        local = local_maps[agent]
        current = state[agent]
        donors: list[int] = current["donors"]
        if not selected[target] and donors:
            target_local = local[target]
            donor_locals = np.asarray([local[value] for value in donors], dtype=np.int64)
            exact_mask = (1.0 - cosine[target_local, donor_locals]) <= config.exact_cosine_eps
            if np.any(exact_mask):
                used = [donors[i] for i in np.flatnonzero(exact_mask)]
                distances = angular[target_local, donor_locals[exact_mask]].astype(np.float64)
                distance = float(np.mean(distances))
                if provenance[target] != "exact_match":
                    raise AssertionError("point estimator exact-match provenance disagreement")
            else:
                ranked = sorted(
                    range(len(donors)),
                    key=lambda i: (float(angular[target_local, donor_locals[i]]), data.unit_ids[donors[i]]),
                )[:config.k]
                used = [donors[i] for i in ranked]
                distances = np.asarray(
                    [float(angular[target_local, donor_locals[i]]) for i in ranked], dtype=np.float64
                )
                weights = 1.0 / np.square(distances + config.eps)
                distance = float(np.dot(weights / weights.sum(), distances))
                if provenance[target] != "idw":
                    raise AssertionError("point estimator IDW provenance disagreement")
            estimate = current["estimate"] or _calibration_estimate(donors, current["slopes"])
            bounds = calculate_conditional_geodesic_bounds(float(scores[target]), distance, estimate)
            lower[target] = bounds.probability.lower
            upper[target] = bounds.probability.upper
            raw_lower[target] = bounds.probability.raw_lower
            raw_upper[target] = bounds.probability.raw_upper
            weighted_distance[target] = distance
            lipschitz[target] = estimate.value
            donor_count[target] = len(donors)
            pair_count[target] = estimate.pair_count
            contradiction_count[target] = current["contradictions"]
            max_donor_position[target] = max(int(positions[value]) for value in used)
            calibration_fallback[target] = int(estimate.provenance.startswith("configured_fallback"))
            if max_donor_position[target] >= position:
                raise AssertionError("target or future donor crossed temporal boundary")
        elif not selected[target] and provenance[target] not in ("global_mean", "prior"):
            raise AssertionError("no same-agent donors must use ordinary fallback")

        if selected[target]:
            target_local = local[target]
            for earlier in donors:
                earlier_local = local[earlier]
                distance = float(angular[target_local, earlier_local])
                delta = abs(float(data.labels[target]) - float(data.labels[earlier]))
                current["slopes"].append(delta / max(distance, LIPSCHITZ_CONFIG.theta_floor))
                if (
                    delta > 0
                    and (1.0 - float(cosine[target_local, earlier_local])) <= config.exact_cosine_eps
                ):
                    current["contradictions"] += 1
            donors.append(target)
            current["estimate"] = _calibration_estimate(donors, current["slopes"])
    eligible = np.isfinite(lower[target_ids])
    if np.any(lower[target_ids][eligible] > scores[target_ids][eligible] + 1e-12):
        raise AssertionError("lower envelope exceeds point score")
    return {
        "target_id": target_ids.astype(np.uint16),
        "position": positions[target_ids].astype(np.uint16),
        "label": data.labels[target_ids].astype(np.uint8),
        "score": scores[target_ids].astype(np.float32),
        "lower": lower[target_ids].astype(np.float32),
        "upper": upper[target_ids].astype(np.float32),
        "raw_lower": raw_lower[target_ids].astype(np.float32),
        "raw_upper": raw_upper[target_ids].astype(np.float32),
        "weighted_distance": weighted_distance[target_ids].astype(np.float32),
        "lipschitz": lipschitz[target_ids].astype(np.float32),
        "provenance": np.asarray([PROVENANCE_CODES[value] for value in provenance[target_ids]], dtype=np.uint8),
        "calibration_donor_count": donor_count[target_ids],
        "calibration_pair_count": pair_count[target_ids],
        "exact_contradiction_count": contradiction_count[target_ids],
        "max_donor_position": max_donor_position[target_ids].astype(np.int16),
        "calibration_fallback": calibration_fallback[target_ids],
    }


def _assert_threshold_dominance(labels: np.ndarray, point: np.ndarray, lower: np.ndarray) -> None:
    thresholds = np.unique(np.r_[np.linspace(0.0, 1.0, 101), point, lower, np.inf])
    positive = labels == 1
    negative = ~positive
    for threshold in thresholds:
        p = point >= threshold
        b = lower >= threshold
        if np.any(b & ~p):
            raise AssertionError("lower-bound positives are not a subset of point positives")
        if positive.any() and np.mean(b[positive]) > np.mean(p[positive]) + 1e-12:
            raise AssertionError("lower-bound recall exceeds point recall")
        if negative.any() and np.mean(b[negative]) > np.mean(p[negative]) + 1e-12:
            raise AssertionError("lower-bound FPR exceeds point FPR")


def _t_interval(values: Sequence[float]) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        return None
    half = float(student_t.ppf(0.975, len(array) - 1) * np.std(array, ddof=1) / math.sqrt(len(array)))
    return [float(np.mean(array) - half), float(np.mean(array) + half)]


def _grid_metrics(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    thresholds = np.r_[np.linspace(0.0, 1.0, 101), np.nextafter(1.0, np.inf)]
    return [{"threshold": float(value), **threshold_metrics(labels, scores, float(value))} for value in thresholds]


def _confusion_add(acc: dict[str, int], labels: np.ndarray, scores: np.ndarray) -> None:
    metrics = threshold_metrics(labels, scores, 0.5)
    if metrics["n"] == 0:
        return
    for name in ("n", "positive_count", "predicted_positive_count", "tp", "fp", "fn", "tn"):
        acc[name] += int(metrics[name])


def _confusion_finish(acc: Mapping[str, int]) -> dict[str, Any]:
    if not acc["n"]:
        return threshold_metrics(np.asarray([], dtype=np.uint8), np.asarray([], dtype=float), 0.5)
    tp, fp, fn, tn = (acc[name] for name in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    return {
        **dict(acc), "accuracy": (tp + tn) / acc["n"], "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall else (0.0 if recall is not None else None),
        "reason": None,
    }


def run_experiment(
    output: Path,
    *,
    baseline_dir: Path = BASELINE,
    input_manifests: Mapping[str, Path] = INPUTS,
    source_revision: str,
    require_canonical_runtime: bool = False,
) -> dict[str, Any]:
    output = Path(output)
    if set(input_manifests) != set(DATASETS):
        raise ValueError("provide the three configured input cohorts")
    permitted_existing = {"preregistration.json", "failed_runs.json", "runtime_variants", "runtime_correction.json"}
    if output.exists() and {path.name for path in output.iterdir()} - permitted_existing:
        raise FileExistsError("use a fresh empty run directory")
    output.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "version": VERSION, "registered_at": _utc_now(), "analysis_status": "confirmatory_and_exploratory",
        "hypotheses": [
            "baseline replay accuracy is exact and MAE differs by at most 1e-12",
            "on matched eligible cohorts lower-bound positives/FPR/recall cannot exceed point score at the same threshold",
            "all source hashes and temporal boundaries remain valid",
        ],
        "primary_metrics": ["exact ROC/AUROC on paired eligible targets", "accuracy", "precision", "recall", "F1"],
        "secondary_metric": "all-unjudged point-score metrics including global-mean/prior fallbacks",
        "stopping_rule": "complete all 4500 cells or fail closed on hash, replay, temporal, or pairing violation",
        "acceptance": {
            "mae_absolute_tolerance": 1e-12, "threshold_dominance_tolerance": 1e-12,
            "canonical_runtime_mae_absolute_tolerance": 0.0,
            "canonical_runtime_accuracy_absolute_tolerance": 0.0,
        },
        "budget": {"cpu_hours_max": 2.0, "artifact_mib_max": 500, "network_calls_max": 0, "external_cost_usd_max": 0},
        "exploratory": "dimension/dataset/budget/schedule/agent differences; no deployable threshold selection",
    }
    preregistration_path = output / "preregistration.json"
    if preregistration_path.exists():
        preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    else:
        write_json(preregistration_path, preregistration)
    started_at = _utc_now()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    pools = threadpool_info()
    if require_canonical_runtime:
        expected_threads = {
            name: "12"
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        }
        if sys.version_info[:2] != (3, 13) or np.__version__ != "2.5.3":
            raise RuntimeError("canonical run requires Python 3.13 and NumPy 2.5.3")
        if any(os.environ.get(name) != value for name, value in expected_threads.items()):
            raise RuntimeError("canonical run requires OPENBLAS/OMP/MKL/NUMEXPR thread counts set to 12")
        if not any(
            pool.get("internal_api") == "openblas" and pool.get("num_threads") == 12
            for pool in pools
        ):
            raise RuntimeError("threadpoolctl did not observe a 12-thread OpenBLAS pool")
    before = _file_snapshot(input_artifacts(input_manifests))
    datasets = {name: load_input(path) for name, path in input_manifests.items()}
    baseline_rows, memberships, baseline_provenance = load_verified_baseline(baseline_dir, datasets)
    keys = sorted(
        memberships,
        key=lambda key: (
            DATASETS.index(key[0]), DIMENSIONS.index(key[1]), key[2],
            SCHEDULES.index(key[3]), RATES.index(key[4]),
        ),
    )
    expected_cells = len(DATASETS) * len(DIMENSIONS) * len(SEEDS) * len(SCHEDULES) * len(RATES)
    if len(keys) != expected_cells:
        raise AssertionError("unexpected cell count")
    total_targets = sum(len(datasets[key[0]].unit_ids) - len(memberships[key]["selected_indices"]) for key in keys)
    columns: dict[str, np.ndarray] = {
        "target_id": np.empty(total_targets, dtype=np.uint16),
        "position": np.empty(total_targets, dtype=np.uint16),
        "label": np.empty(total_targets, dtype=np.uint8),
        "score": np.empty(total_targets, dtype=np.float32),
        "lower": np.empty(total_targets, dtype=np.float32),
        "upper": np.empty(total_targets, dtype=np.float32),
        "raw_lower": np.empty(total_targets, dtype=np.float32),
        "raw_upper": np.empty(total_targets, dtype=np.float32),
        "weighted_distance": np.empty(total_targets, dtype=np.float32),
        "lipschitz": np.empty(total_targets, dtype=np.float32),
        "provenance": np.empty(total_targets, dtype=np.uint8),
        "calibration_donor_count": np.empty(total_targets, dtype=np.uint16),
        "calibration_pair_count": np.empty(total_targets, dtype=np.uint32),
        "exact_contradiction_count": np.empty(total_targets, dtype=np.uint16),
        "max_donor_position": np.empty(total_targets, dtype=np.int16),
        "calibration_fallback": np.empty(total_targets, dtype=np.uint8),
    }
    offsets = np.empty(expected_cells + 1, dtype=np.uint64)
    rows: list[dict[str, Any]] = []
    agent_confusions: dict[tuple[str, int, str, str], dict[str, int]] = defaultdict(
        lambda: {name: 0 for name in ("n", "positive_count", "predicted_positive_count", "tp", "fp", "fn", "tn")}
    )
    cursor = 0
    block_cache: dict[tuple[str, int], Any] = {}
    baseline_max_mae_error = 0.0
    baseline_accuracy_mismatches = 0
    baseline_max_accuracy_error = 0.0
    for cell_id, key in enumerate(keys):
        dataset_id, dimension, seed, schedule, rate = key
        data = datasets[dataset_id]
        blocks = block_cache.get((dataset_id, dimension))
        if blocks is None:
            vectors = normalized_prefix(data.vectors, dimension)
            blocks = distance_blocks(data, vectors)
            block_cache[(dataset_id, dimension)] = blocks
        order, _ = build_schedule(data.unit_ids, data.agents, schedule, seed)
        selected_ids = np.asarray(memberships[key]["selected_indices"], dtype=np.int64)
        metrics, scores, provenance = evaluate(data, blocks, order, selected_ids)
        baseline = baseline_rows[key]
        mae_error = abs(float(metrics["mae"]) - float(baseline["mae"]))
        baseline_max_mae_error = max(baseline_max_mae_error, mae_error)
        execution_tolerance = (
            preregistration["acceptance"]["canonical_runtime_mae_absolute_tolerance"]
            if require_canonical_runtime
            else preregistration["acceptance"].get(
                "amended_execution_mae_absolute_tolerance",
                preregistration["acceptance"]["mae_absolute_tolerance"],
            )
        )
        if mae_error > execution_tolerance:
            raise AssertionError(f"baseline MAE replay failed for {key}: {mae_error}")
        accuracy_error = abs(float(metrics["accuracy"]) - float(baseline["accuracy"]))
        baseline_max_accuracy_error = max(baseline_max_accuracy_error, accuracy_error)
        if accuracy_error > 0:
            baseline_accuracy_mismatches += 1
        accuracy_tolerance = (
            preregistration["acceptance"]["canonical_runtime_accuracy_absolute_tolerance"]
            if require_canonical_runtime
            else preregistration["acceptance"].get("amended_execution_accuracy_absolute_tolerance", 0.0)
        )
        if accuracy_error > accuracy_tolerance:
            raise AssertionError(f"baseline accuracy replay failed for {key}: {accuracy_error}")
        evidence = envelope_evidence(data, blocks, order, selected_ids, scores, provenance)
        size = len(evidence["target_id"])
        offsets[cell_id] = cursor
        for name, values in evidence.items():
            columns[name][cursor:cursor + size] = values
        start, end = cursor, cursor + size
        cursor = end
        eligible = np.isfinite(evidence["lower"])
        paired_labels = evidence["label"][eligible]
        paired_point = evidence["score"][eligible]
        paired_lower = evidence["lower"][eligible]
        _assert_threshold_dominance(paired_labels, paired_point, paired_lower)
        point_roc = exact_roc(paired_labels, paired_point)
        lower_roc = exact_roc(paired_labels, paired_lower)
        point_at_half = threshold_metrics(paired_labels, paired_point, 0.5)
        lower_at_half = threshold_metrics(paired_labels, paired_lower, 0.5)
        all_at_half = threshold_metrics(evidence["label"], evidence["score"], 0.5)
        row = {
            "cell_id": cell_id, "dataset_id": dataset_id, "dimension": dimension,
            "seed": seed, "schedule": schedule, "rate": rate,
            "selected_count": len(selected_ids), "unjudged_count": size,
            "eligible_count": int(eligible.sum()), "envelope_missing_count": int((~eligible).sum()),
            "provenance_counts": {
                name: int(np.sum(evidence["provenance"] == code))
                for name, code in PROVENANCE_CODES.items()
            },
            "calibration_fallback_count": int(evidence["calibration_fallback"][eligible].sum()),
            "exact_contradiction_target_count": int(np.sum(evidence["exact_contradiction_count"][eligible] > 0)),
            "point_auc": point_roc["auc"], "point_auc_reason": point_roc["reason"],
            "lower_auc": lower_roc["auc"], "lower_auc_reason": lower_roc["reason"],
            "paired_point_at_0_5": point_at_half, "paired_lower_at_0_5": lower_at_half,
            "all_unjudged_point_at_0_5": all_at_half,
            "baseline_replay": {
                "mae_absolute_error": mae_error, "accuracy_absolute_error": accuracy_error,
                "accuracy_exact": accuracy_error == 0,
            },
            "evidence_offset": [start, end],
        }
        rows.append(row)
        target_agents = np.asarray(data.agents, dtype=object)[evidence["target_id"]]
        for agent in sorted(set(target_agents.tolist())):
            agent_mask = target_agents == agent
            _confusion_add(agent_confusions[(dataset_id, dimension, agent, "all_point")],
                           evidence["label"][agent_mask], evidence["score"][agent_mask])
            paired_agent = agent_mask & eligible
            _confusion_add(agent_confusions[(dataset_id, dimension, agent, "paired_point")],
                           evidence["label"][paired_agent], evidence["score"][paired_agent])
            _confusion_add(agent_confusions[(dataset_id, dimension, agent, "paired_lower")],
                           evidence["label"][paired_agent], evidence["lower"][paired_agent])
        if (cell_id + 1) % 50 == 0:
            print(f"completed {cell_id + 1}/{expected_cells} cells", flush=True)
    offsets[-1] = cursor
    if cursor != total_targets:
        raise AssertionError("target allocation mismatch")

    selector_summaries: list[dict[str, Any]] = []
    curve_arrays: dict[str, np.ndarray] = {}
    for dataset_id in DATASETS:
        for dimension in DIMENSIONS:
            for rate in RATES:
                selected_rows = [
                    row for row in rows
                    if row["dataset_id"] == dataset_id and row["dimension"] == dimension and row["rate"] == rate
                ]
                spans = [np.arange(*row["evidence_offset"], dtype=np.int64) for row in selected_rows]
                indexes = np.concatenate(spans)
                eligible = np.isfinite(columns["lower"][indexes])
                paired_indexes = indexes[eligible]
                labels = columns["label"][paired_indexes]
                point = columns["score"][paired_indexes]
                lower = columns["lower"][paired_indexes]
                all_labels = columns["label"][indexes]
                all_scores = columns["score"][indexes]
                point_roc = exact_roc(labels, point)
                lower_roc = exact_roc(labels, lower)
                selector_id = f"{dataset_id}|{dimension}|{rate:g}"
                for method, roc in (("point", point_roc), ("lower", lower_roc)):
                    prefix = f"{selector_id}|{method}"
                    curve_arrays[prefix + "|thresholds"] = roc["thresholds"]
                    curve_arrays[prefix + "|fpr"] = roc["fpr"]
                    curve_arrays[prefix + "|tpr"] = roc["tpr"]
                selector_summaries.append({
                    "selector_id": selector_id, "dataset_id": dataset_id, "dimension": dimension,
                    "rate": rate, "cell_count": len(selected_rows), "pooled_unjudged_n": len(indexes),
                    "pooled_eligible_n": len(paired_indexes),
                    "pooled_occurrences_note": "Repeated replay target occurrences; not independent sessions.",
                    "point_auc": point_roc["auc"], "point_auc_reason": point_roc["reason"],
                    "lower_auc": lower_roc["auc"], "lower_auc_reason": lower_roc["reason"],
                    "point_roc_display": _display_roc(point_roc),
                    "lower_roc_display": _display_roc(lower_roc),
                    "point_grid": _grid_metrics(labels, point),
                    "lower_grid": _grid_metrics(labels, lower),
                    "all_unjudged_point_at_0_5": threshold_metrics(all_labels, all_scores, 0.5),
                })

    equal_cell: list[dict[str, Any]] = []
    for dataset_id in DATASETS:
        for dimension in DIMENSIONS:
            for rate in RATES:
                subset = [
                    row for row in rows
                    if row["dataset_id"] == dataset_id and row["dimension"] == dimension and row["rate"] == rate
                ]
                item: dict[str, Any] = {
                    "dataset_id": dataset_id, "dimension": dimension, "rate": rate, "cell_count": len(subset)
                }
                for method, field in (("point", "point_auc"), ("lower", "lower_auc")):
                    available = [float(row[field]) for row in subset if row[field] is not None]
                    seed_means = [
                        float(np.mean([row[field] for row in subset if row["seed"] == seed and row[field] is not None]))
                        for seed in SEEDS
                        if any(row["seed"] == seed and row[field] is not None for row in subset)
                    ]
                    item[method + "_auc_mean"] = float(np.mean(available)) if available else None
                    item[method + "_auc_seed_t95"] = _t_interval(seed_means)
                    item[method + "_auc_available_cells"] = len(available)
                item["eligible_share"] = sum(row["eligible_count"] for row in subset) / sum(
                    row["unjudged_count"] for row in subset
                )
                equal_cell.append(item)

    agent_summary = []
    for (dataset_id, dimension, agent, method), counts in sorted(agent_confusions.items()):
        agent_summary.append({
            "dataset_id": dataset_id, "dimension": dimension, "agent_id": agent,
            "method": method, **_confusion_finish(counts),
        })

    np.savez_compressed(output / "target_evidence.npz", offsets=offsets, **columns)
    np.savez_compressed(output / "exact_roc_curves.npz", **curve_arrays)
    write_json(output / "target_evidence_schema.json", {
        "version": VERSION, "row_alignment": "arrays are contiguous by aggregate.rows[cell_id]; offsets bound each cell",
        "provenance_codes": PROVENANCE_CODES,
        "fields": {
            "target_id": "source-row index", "position": "target replay position",
            "score": "ordinary evaluate() point score", "lower/upper": "clamped envelope",
            "raw_lower/raw_upper": "unclamped envelope", "weighted_distance": "normalized angular IDW donor distance",
            "lipschitz": "target-time empirical/fallback L", "max_donor_position": "latest used donor replay position; always < target position",
            "calibration_pair_count": "earlier same-agent observed-label pairs",
            "exact_contradiction_count": "earlier exact-geometry donor pairs with conflicting labels",
        },
        "missing": "NaN envelope fields mean global_mean/prior point fallback; no Lipschitz band is claimed",
        "score_precision": "Point scores and envelope fields are retained as float32. Threshold metrics and exact tied-score ROC use those retained values; baseline replay parity is checked before this serialization.",
    })
    after = _file_snapshot(input_artifacts(input_manifests))
    if before != after:
        raise AssertionError("input cache artifacts changed during offline run")
    wall_seconds, cpu_seconds = time.perf_counter() - wall_start, time.process_time() - cpu_start
    profiles = [
        {
            "dataset_id": name, "sessions": len(data.unit_ids), "agents": len(set(data.agents)),
            "positive_count": int(data.labels.sum()), "pass_rate": float(data.labels.mean()),
            "manifest": str(input_manifests[name]), "manifest_sha256": sha256_file(input_manifests[name]),
            "label_source": data.profile["label_source"],
            "representation_policy": data.profile["representation_policy"],
            "representation_sources": data.profile.get("representation_sources", {}),
            "embedding_preparation": {
                key: data.profile[key] for key in (
                    "live_embedding_calls", "api_input_tokens", "reused_vectors",
                    "requested_vectors", "logical_requests", "http_attempts", "http_successes",
                ) if key in data.profile
            },
            "snapshot_cutoff_utc": data.profile.get("snapshot_cutoff_utc"),
            "point_in_time_snapshot": data.profile.get("point_in_time_snapshot"),
            "truncated_sessions": data.profile.get("truncated_sessions", 0),
            "excluded_sessions": data.profile.get("excluded_sessions", 0),
            "source_hashes": data.profile.get("source_hashes", {}),
        }
        for name, data in datasets.items()
    ]
    code_paths = [
        Path(__file__), Path("scripts/run_idw_threshold_experiment.py"),
        Path("sampling_comparison/matryoshka_experiment.py"), Path("trace_sampling/lipschitz.py"),
    ]
    result = {
        "version": VERSION, "run_id": output.name, "status": "complete",
        "started_at": started_at, "completed_at": _utc_now(), "source_revision": source_revision,
        "preregistration": preregistration,
        "protocol": {
            "dimensions": list(DIMENSIONS), "seeds": list(SEEDS), "schedules": list(SCHEDULES),
            "rates": list(RATES), "modes": ["end_to_end"], "idw": asdict(IDWConfig()),
            "lipschitz": asdict(LIPSCHITZ_CONFIG), "angular_units": "arccos(cosine)/pi",
            "budget_rule": "max(1,floor(N*rate))", "selection": "reused verified label-blind full-schedule ranking",
            "target_rule": "only earlier same-agent selected donors; selected observations excluded",
            "threshold_rule": "score >= threshold", "grid": "0..1 step .01 plus all-negative endpoint",
            "precision_zero_division": 0, "f1_zero_predicted_positive_when_positive_class_exists": 0,
            "network_calls": 0, "embedding_calls": 0, "judge_calls": 0,
            "canonical_runtime_required": require_canonical_runtime,
        },
        "baseline": baseline_provenance, "datasets": profiles, "rows": rows,
        "selectors": selector_summaries, "equal_cell_summary": equal_cell,
        "agent_summary": agent_summary,
        "validation": {
            "expected_cells": expected_cells, "actual_cells": len(rows),
            "target_occurrences": total_targets,
            "eligible_target_occurrences": int(np.isfinite(columns["lower"]).sum()),
            "fallback_target_occurrences": int(np.isnan(columns["lower"]).sum()),
            "baseline_max_mae_absolute_error": baseline_max_mae_error,
            "baseline_original_preregistered_mae_tolerance": preregistration["acceptance"]["mae_absolute_tolerance"],
            "baseline_amended_execution_mae_tolerance": preregistration["acceptance"].get(
                "amended_execution_mae_absolute_tolerance",
                preregistration["acceptance"]["mae_absolute_tolerance"],
            ),
            "baseline_original_preregistered_tolerance_passed": (
                baseline_max_mae_error <= preregistration["acceptance"]["mae_absolute_tolerance"]
            ),
            "baseline_accuracy_mismatches": baseline_accuracy_mismatches,
            "baseline_max_accuracy_absolute_error": baseline_max_accuracy_error,
            "baseline_original_exact_accuracy_passed": baseline_accuracy_mismatches == 0,
            "threshold_dominance": "passed_all_cells",
            "cache_source_preservation": "passed", "cache_snapshot_before": before, "cache_snapshot_after": after,
        },
        "environment": {
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
            "threadpool_info": pools,
        },
        "costs": {
            "wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds,
            "network_calls": 0, "embedding_calls": 0, "judge_calls": 0, "external_cost_usd": 0,
        },
        "code_hashes": {str(path): sha256_file(path) for path in code_paths},
        "files": {
            "aggregate": str(output / "aggregate.json"),
            "preregistration": str(output / "preregistration.json"),
            "target_evidence": str(output / "target_evidence.npz"),
            "target_evidence_schema": str(output / "target_evidence_schema.json"),
            "exact_roc_curves": str(output / "exact_roc_curves.npz"),
        },
    }
    write_json(output / "aggregate.json", result)
    _write_manifest(output, result)
    artifact_bytes = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    if artifact_bytes > preregistration["budget"]["artifact_mib_max"] * 1024 * 1024:
        raise RuntimeError("artifact storage budget exceeded")
    if wall_seconds > preregistration["budget"]["cpu_hours_max"] * 3600:
        raise RuntimeError("recorded execution-time budget exceeded")
    return result


def _write_manifest(output: Path, aggregate: Mapping[str, Any]) -> None:
    files = {}
    for name, value in aggregate["files"].items():
        path = Path(value)
        if path.exists():
            files[name] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "version": VERSION, "status": aggregate["status"], "source_revision": aggregate["source_revision"],
        "code_hashes": aggregate["code_hashes"], "baseline": aggregate["baseline"], "files": files,
        "generated_at": _utc_now(),
    }
    manifest["fingerprint"] = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    write_json(output / "manifest.json", manifest)


def refresh_manifest(aggregate_path: Path) -> None:
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    _write_manifest(aggregate_path.parent, aggregate)
