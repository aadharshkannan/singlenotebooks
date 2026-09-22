"""Independently check retained real-study scores, cohorts, causality and hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sampling_comparison.idw_threshold_experiment import exact_roc, threshold_metrics
from sampling_comparison.imdb_inputs import load_input
from sampling_comparison.matryoshka_experiment import canonical, sha256_file, write_json


DIMENSION_GRIDS = (
    (1536, 32, 24, 16, 12, 8),
    (1536, 256, 128, 64, 32, 24, 16, 12, 8),
)


def validate_extension(aggregate: dict, manifest: dict, run: Path) -> dict | None:
    if "extension" not in aggregate:
        return None
    extension = aggregate["extension"]
    matching_sources = [
        run / name for name, digest in manifest["files"].items()
        if digest == extension["baseline_aggregate_sha256"]
    ]
    if len(matching_sources) != 1:
        raise ValueError("extension must retain exactly one hash-bound original aggregate")
    baseline_path = matching_sources[0]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if sha256_file(baseline_path) != extension["baseline_aggregate_sha256"]:
        raise ValueError("original aggregate changed")
    baseline_manifest = baseline_path.parent / "manifest.json"
    if sha256_file(baseline_manifest) != extension["baseline_manifest_sha256"]:
        raise ValueError("original manifest changed")
    key = lambda row: (row["seed"], row["schedule"], row["dimension"], row["rate"])
    original = {key(row): row for row in baseline["rows"]}
    combined = {key(row): row for row in aggregate["rows"]}
    if len(original) != extension["reused_cells"] or not original.keys() <= combined.keys():
        raise ValueError("extension dropped or duplicated original cells")
    for identity, row in original.items():
        current = combined[identity]
        expected_values = {k: v for k, v in row.items() if k != "evidence"}
        current_values = {k: v for k, v in current.items() if k != "evidence"}
        if canonical(expected_values) != canonical(current_values):
            raise ValueError("an original measured row changed")
        if (baseline_path.parent / row["evidence"]).resolve() != (run / current["evidence"]).resolve():
            raise ValueError("original evidence was replaced rather than referenced")
    native_replays = {
        (row["seed"], row["schedule"]): row["replay_hashes"]
        for row in baseline["rows"] if row["dimension"] == 1536
    }
    added = [row for identity, row in combined.items() if identity not in original]
    if len(added) != extension["added_cells"]:
        raise ValueError("additional cell count mismatch")
    for row in added:
        if row["replay_hashes"] != native_replays[(row["seed"], row["schedule"])]:
            raise ValueError("new dimension is not paired to the original replay")
    if (extension["baseline_rows_unchanged"] is not True
            or extension["replay_pairing_exact"] is not True or extension["embedding_calls"] != 0):
        raise ValueError("extension provenance claim is invalid")
    return {
        "baseline_rows_unchanged": True, "replay_pairing_exact": True,
        "reused_cells": len(original), "added_cells": len(added),
        "embedding_calls": 0,
        "baseline_aggregate_sha256": extension["baseline_aggregate_sha256"],
        "baseline_manifest_sha256": extension["baseline_manifest_sha256"],
    }


def validate_scores(row: dict, evidence: dict[str, np.ndarray], labels: np.ndarray) -> dict:
    y, score = evidence["label"], evidence["score"]
    lower, upper = evidence["lower"], evidence["upper"]
    eligible = np.isfinite(lower) & np.isfinite(upper)
    novel = evidence["novel_source"]
    if row["budget"] != max(1, int(len(labels) * row["rate"])):
        raise ValueError("selected budget does not match requested label rate")
    if not np.array_equal(y, labels[evidence["source_id"]]):
        raise ValueError("retained labels differ from original source labels")
    if len(score) != row["counts"]["unselected"] or len(score) != len(labels) - row["budget"]:
        raise ValueError("unselected denominator mismatch")
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("invalid point score")
    if not np.array_equal(np.isfinite(lower), np.isfinite(upper)):
        raise ValueError("lower/upper eligibility mismatch")
    if np.any(lower[eligible] > score[eligible]) or np.any(score[eligible] > upper[eligible]):
        raise ValueError("point score is outside conditional envelope")
    if np.any(lower[eligible] < 0) or np.any(upper[eligible] > 1):
        raise ValueError("display envelope outside [0,1]")
    if np.any(evidence["max_donor_position"] >= evidence["position"]):
        raise ValueError("noncausal donor position")
    if np.any(evidence["max_calibration_position"] >= evidence["position"]):
        raise ValueError("noncausal calibration position")
    if np.intersect1d(evidence["occurrence_id"], evidence["selected_occurrence_id"]).size:
        raise ValueError("direct selected labels leaked into unselected metrics")
    if len(np.unique(evidence["selected_occurrence_id"])) != row["budget"]:
        raise ValueError("selected membership budget mismatch")
    cohorts = {
        "all_unselected": (np.ones(len(y), dtype=bool), score),
        "eligible_point": (eligible, score), "eligible_lower": (eligible, lower),
        "novel_source": (novel, score), "repeated_source": (~novel, score),
    }
    maximum_error = 0.0
    for name, (mask, values) in cohorts.items():
        measured = threshold_metrics(y[mask], values[mask], .5)
        measured["mae"] = float(np.mean(np.abs(y[mask] - values[mask]))) if mask.any() else None
        measured["auc"] = exact_roc(y[mask], values[mask])["auc"]
        for metric in ("n", "mae", "accuracy", "precision", "recall", "f1", "auc"):
            actual, retained = measured[metric], row[name][metric]
            if actual is None or retained is None:
                if actual is not retained:
                    raise ValueError(f"{name}/{metric} undefined-state mismatch")
            else:
                error = abs(actual - retained)
                maximum_error = max(maximum_error, error)
                if error > 1e-12:
                    raise ValueError(f"{name}/{metric} differs from retained scores by {error}")
    coverage = float(np.mean((y[eligible] >= lower[eligible]) & (y[eligible] <= upper[eligible]))) if eligible.any() else None
    if coverage != row["envelope_label_coverage"]:
        raise ValueError("observed envelope label coverage mismatch")
    return {
        "max_metric_error": maximum_error,
        "lower_zero_fraction": float(np.mean(lower[eligible] == 0)) if eligible.any() else None,
        "mean_envelope_width": float(np.mean(upper[eligible] - lower[eligible])) if eligible.any() else None,
    }


def validate_experiment(run: Path, input_manifest: Path) -> dict:
    _, labels, profile = load_input(input_manifest)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected_hash in manifest["files"].items():
        if sha256_file(run / relative) != expected_hash:
            raise ValueError(f"retained run artifact hash mismatch: {relative}")
    aggregate = json.loads((run / "aggregate.json").read_text(encoding="utf-8"))
    protocol = aggregate["protocol"]
    planned_cells = len(protocol["dimensions"]) * 40 * 2 * 5
    if (aggregate["status"] != "completed" or manifest["completed_cells"] != planned_cells
            or len(aggregate["rows"]) != planned_cells or len(labels) != 50_000
            or tuple(protocol["dimensions"]) not in DIMENSION_GRIDS
            or protocol["planned_cells"] != planned_cells
            or protocol["repetitions"] != 40
            or protocol["rates"] != [.01, .02, .05, .1, .2]
            or protocol["schedules"] != ["uniformly_random", "bursty"]):
        raise ValueError("requested full real-data study grid is incomplete")
    if aggregate["dataset"]["input_manifest_sha256"] != profile["input_manifest_sha256"]:
        raise ValueError("study is not bound to this real input manifest")
    extension_validation = validate_extension(aggregate, manifest, run)
    if tuple(protocol["dimensions"]) == DIMENSION_GRIDS[1] and extension_validation is None:
        raise ValueError("extended grid requires preserved-baseline provenance")
    fields = (
        "label", "score", "lower", "upper", "source_id", "novel_source", "position",
        "max_donor_position", "max_calibration_position", "occurrence_id", "selected_occurrence_id",
    )
    maximum_error, inspected = 0.0, 0
    native_five = []
    identities = set()
    for row in aggregate["rows"]:
        identities.add((row["seed"], row["schedule"], row["dimension"], row["rate"]))
        with np.load(run / row["evidence"], allow_pickle=False) as arrays:
            evidence = {key: arrays[key] for key in fields}
        result = validate_scores(row, evidence, labels)
        maximum_error = max(maximum_error, result["max_metric_error"])
        inspected += len(evidence["label"])
        if row["dimension"] == 1536 and row["rate"] == .05:
            native_five.append(result)
    seeds = {row["seed"] for row in aggregate["rows"]}
    expected_identities = {
        (seed, schedule, dimension, rate)
        for seed in seeds for schedule in protocol["schedules"]
        for dimension in protocol["dimensions"] for rate in protocol["rates"]
    }
    if len(seeds) != 40 or identities != expected_identities or len(native_five) != 80:
        raise ValueError("cell identities or native 5% scope mismatch")
    result = {
        "ok": True, "cells_checked": planned_cells, "retained_artifacts_hash_checked": len(manifest["files"]),
        "dimensions": protocol["dimensions"],
        "unselected_occurrences_checked": inspected,
        "aggregate_sha256": sha256_file(run / "aggregate.json"),
        "input_manifest_sha256": profile["input_manifest_sha256"],
        "native_vectors_sha256": profile["hashes"]["vectors"],
        "model": profile["model"], "live_judge_calls": profile["live_judge_calls"],
        "maximum_absolute_metric_difference": maximum_error,
        "validator_sha256": sha256_file(Path(__file__)),
        "checks": [
            f"Complete requested {planned_cells}-cell grid and original-label alignment",
            "All retained artifact hashes and immutable real native embedding cache",
            "Recomputed MAE, accuracy, precision, recall, F1 and exact AUROC in five cohorts",
            "Strict earlier-only donor and calibration positions",
            "Disjoint directly selected and unselected occurrence sets",
            "Nested point/lower score ordering and observed full-envelope label coverage",
        ],
        "native_5pct_equal_cell_diagnostics": {
            "lower_zero_fraction": float(np.mean([r["lower_zero_fraction"] for r in native_five])),
            "mean_envelope_width": float(np.mean([r["mean_envelope_width"] for r in native_five])),
        },
    }
    if extension_validation is not None:
        result["extension"] = extension_validation
        result["checks"].append("Unchanged original rows/evidence and exact new-dimension replay pairing")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs_imdb/private_runs/imdb-40-replay"))
    parser.add_argument("--input", type=Path, default=Path("outputs_imdb/cache/imdb-50000/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs_imdb/reports/imdb-40-replay/numerical_validation.json"))
    args = parser.parse_args()
    result = validate_experiment(args.run, args.input)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
