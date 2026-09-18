from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.idw_threshold_experiment import (  # noqa: E402
    BASELINE,
    DATASETS,
    INPUTS,
    PROVENANCE_CODES,
    envelope_evidence,
    exact_roc,
    load_verified_baseline,
    refresh_manifest,
    threshold_metrics,
)
from sampling_comparison.matryoshka_experiment import (  # noqa: E402
    build_schedule,
    distance_blocks,
    evaluate,
    load_input,
    normalized_prefix,
    sha256_file,
    write_json,
)


def _same_metric(actual: dict, expected: dict) -> bool:
    for name in ("n", "positive_count", "predicted_positive_count", "tp", "fp", "fn", "tn"):
        if actual.get(name) != expected.get(name):
            return False
    for name in ("accuracy", "precision", "recall", "f1"):
        left, right = actual.get(name), expected.get(name)
        if left is None or right is None:
            if left is not right:
                return False
        elif abs(float(left) - float(right)) > 1e-15:
            return False
    return True


def validate(aggregate_path: Path) -> dict:
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    run = aggregate_path.parent
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"].values():
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise AssertionError("retained threshold artifact hash mismatch")
    datasets = load_recorded_inputs(aggregate)
    baseline_dir = Path(aggregate["baseline"]["manifest"]).parent
    for key in ("manifest", "aggregate", "memberships"):
        if sha256_file(Path(aggregate["baseline"][key])) != aggregate["baseline"][key + "_sha256"]:
            raise AssertionError("recorded baseline changed")
    baseline_rows, memberships, _ = load_verified_baseline(baseline_dir, datasets)
    if len(aggregate["rows"]) != 4500 or len(memberships) != 4500:
        raise AssertionError("canonical grid or membership count is incomplete")
    if aggregate["validation"]["baseline_max_mae_absolute_error"] != 0:
        raise AssertionError("canonical baseline MAE replay was not exact")
    if aggregate["validation"]["baseline_accuracy_mismatches"] != 0:
        raise AssertionError("canonical baseline accuracy replay was not exact")

    with np.load(run / "target_evidence.npz", allow_pickle=False) as evidence:
        retained_arrays = {name: evidence[name] for name in evidence.files}
        offsets = retained_arrays["offsets"]
        if len(offsets) != len(aggregate["rows"]) + 1:
            raise AssertionError("offset count does not match result cells")
        if int(offsets[-1]) != aggregate["validation"]["target_occurrences"]:
            raise AssertionError("target evidence length mismatch")
        metric_cells = 0
        for row in aggregate["rows"]:
            start, end = map(int, row["evidence_offset"])
            if (start, end) != (int(offsets[row["cell_id"]]), int(offsets[row["cell_id"] + 1])):
                raise AssertionError("aggregate and NPZ offsets disagree")
            labels = retained_arrays["label"][start:end]
            point = retained_arrays["score"][start:end]
            lower = retained_arrays["lower"][start:end]
            eligible = np.isfinite(lower)
            if np.any(lower[eligible] > point[eligible]):
                raise AssertionError("lower envelope exceeds retained point score")
            if np.any(retained_arrays["max_donor_position"][start:end][eligible] >= retained_arrays["position"][start:end][eligible]):
                raise AssertionError("donor crosses target-time boundary")
            provenance = retained_arrays["provenance"][start:end]
            if not np.array_equal(eligible, np.isin(provenance, (2, 3))):
                raise AssertionError("claimed envelope cohort differs from IDW-supported targets")
            for field in ("lower", "upper"):
                raw = retained_arrays["raw_" + field][start:end][eligible]
                if not np.array_equal(np.clip(raw, 0, 1), retained_arrays[field][start:end][eligible]):
                    raise AssertionError("raw/clamped envelope disagreement")
            comparisons = (
                (threshold_metrics(labels, point, 0.5), row["all_unjudged_point_at_0_5"]),
                (threshold_metrics(labels[eligible], point[eligible], 0.5), row["paired_point_at_0_5"]),
                (threshold_metrics(labels[eligible], lower[eligible], 0.5), row["paired_lower_at_0_5"]),
            )
            if not all(_same_metric(actual, expected) for actual, expected in comparisons):
                raise AssertionError(f"threshold metric mismatch in cell {row['cell_id']}")
            key = (row["dataset_id"], row["dimension"], row["seed"], row["schedule"], row["rate"])
            if comparisons[0][0]["accuracy"] != baseline_rows[key]["accuracy"]:
                raise AssertionError("retained float32 classifications differ from baseline")
            for field, expected_auc in (("score", row["point_auc"]), ("lower", row["lower_auc"])):
                actual_auc = exact_roc(labels[eligible], retained_arrays[field][start:end][eligible])["auc"]
                if actual_auc is None or expected_auc is None:
                    if actual_auc is not expected_auc:
                        raise AssertionError(f"undefined AUROC mismatch in cell {row['cell_id']}")
                elif abs(actual_auc - expected_auc) > 1e-15:
                    raise AssertionError(f"AUROC mismatch in cell {row['cell_id']}")
            metric_cells += 1

        # Recompute one seed/schedule cell for every dataset/dimension/budget
        # selector. This independently checks source labels, membership exclusion,
        # target ordering, point scores, envelope evidence, and NPZ serialization.
        replay_cells = 0
        block_cache = {}
        for row in aggregate["rows"]:
            if row["seed"] != 13 or row["schedule"] != "evenly_spaced":
                continue
            key = (row["dataset_id"], row["dimension"], row["seed"], row["schedule"], row["rate"])
            data = datasets[row["dataset_id"]]
            blocks = block_cache.get((row["dataset_id"], row["dimension"]))
            if blocks is None:
                blocks = distance_blocks(data, normalized_prefix(data.vectors, row["dimension"]))
                block_cache[(row["dataset_id"], row["dimension"])] = blocks
            order, _ = build_schedule(data.unit_ids, data.agents, row["schedule"], row["seed"])
            selected = np.asarray(memberships[key]["selected_indices"], dtype=np.int64)
            _, scores, provenance = evaluate(data, blocks, order, selected)
            rebuilt = envelope_evidence(data, blocks, order, selected, scores, provenance)
            start, end = map(int, row["evidence_offset"])
            for name, values in rebuilt.items():
                retained = retained_arrays[name][start:end]
                if np.issubdtype(values.dtype, np.floating):
                    equal = np.array_equal(values, retained, equal_nan=True)
                else:
                    equal = np.array_equal(values, retained)
                if not equal:
                    raise AssertionError(f"source-to-score mismatch for {key}, field={name}")
            replay_cells += 1
        if replay_cells != 30:
            raise AssertionError(f"expected 30 stratified source-score replay cells, got {replay_cells}")

        exact_arrays_checked = 0
        with np.load(run / "exact_roc_curves.npz", allow_pickle=False) as curves:
            for selector in aggregate["selectors"]:
                rows = [
                    row for row in aggregate["rows"]
                    if row["dataset_id"] == selector["dataset_id"]
                    and row["dimension"] == selector["dimension"]
                    and row["rate"] == selector["rate"]
                ]
                indexes = np.concatenate([
                    np.arange(*map(int, row["evidence_offset"]), dtype=np.int64) for row in rows
                ])
                eligible = np.isfinite(retained_arrays["lower"][indexes])
                labels = retained_arrays["label"][indexes][eligible]
                for method, field in (("point", "score"), ("lower", "lower")):
                    roc = exact_roc(labels, retained_arrays[field][indexes][eligible])
                    prefix = selector["selector_id"] + "|" + method
                    for name in ("thresholds", "fpr", "tpr"):
                        if not np.array_equal(roc[name], curves[prefix + "|" + name], equal_nan=True):
                            raise AssertionError(f"exact ROC mismatch: {prefix}|{name}")
                    exact_arrays_checked += 1

    current_cache = aggregate["validation"]["cache_snapshot_after"]
    if not all(sha256_file(Path(path)) == item["sha256"] for path, item in current_cache.items()):
        raise AssertionError("input cache changed after the canonical run")
    return {
        "version": aggregate["version"], "checked_at": datetime.now(timezone.utc).isoformat(),
        "ok": True, "canonical_cells_checked": metric_cells,
        "stratified_source_to_score_replay_cells": replay_cells,
        "exact_roc_method_selector_arrays_checked": exact_arrays_checked,
        "valid_point_lower_pairs": aggregate["validation"]["eligible_target_occurrences"],
        "missing_envelope_fallback_targets": aggregate["validation"]["fallback_target_occurrences"],
        "empirical_l_targets": (
            aggregate["validation"]["eligible_target_occurrences"]
            - sum(row["calibration_fallback_count"] for row in aggregate["rows"])
        ),
        "configured_l_fallback_targets": sum(row["calibration_fallback_count"] for row in aggregate["rows"]),
        "targets_with_exact_match_contradictions": sum(
            row["exact_contradiction_target_count"] for row in aggregate["rows"]
        ),
        "baseline_exact_mae": True, "baseline_exact_accuracy": True,
        "membership_records_verified": len(memberships), "input_cache_hashes_preserved": True,
    }


def load_recorded_inputs(aggregate: dict) -> dict:
    profiles = aggregate["datasets"]
    if len(profiles) != len(DATASETS) or {p["dataset_id"] for p in profiles} != set(DATASETS):
        raise ValueError("recorded input cohorts are incomplete or duplicated")
    datasets = {}
    for profile in profiles:
        path = Path(profile["manifest"])
        if sha256_file(path) != profile["manifest_sha256"]:
            raise ValueError("recorded input manifest hash mismatch")
        data = load_input(path)
        if data.dataset_id != profile["dataset_id"] or len(data.unit_ids) != profile["sessions"]:
            raise ValueError("recorded input population mismatch")
        datasets[data.dataset_id] = data
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate IDW threshold scores and retained ROC evidence.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    aggregate_path, output_path = Path(args.input), Path(args.output)
    result = validate(aggregate_path)
    write_json(output_path, result)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["score_consistency_validation"] = result
    aggregate["files"]["score_consistency_validation"] = str(output_path)
    aggregate["code_hashes"]["scripts/validate_idw_threshold_experiment.py"] = sha256_file(Path(__file__))
    write_json(aggregate_path, aggregate)
    refresh_manifest(aggregate_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
