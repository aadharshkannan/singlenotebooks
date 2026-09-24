"""Independently check retained real-study scores, cohorts, causality and hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from threadpoolctl import threadpool_limits

from sampling_comparison.idw_threshold_experiment import exact_roc, threshold_metrics
from sampling_comparison.imdb_experiment import _digest, _read_record, array_sha256
from sampling_comparison.imdb_inputs import load_input
from sampling_comparison.matryoshka_experiment import canonical, normalized_prefix, sha256_file, write_json


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


def validate_execution(aggregate: dict, manifest: dict, run: Path) -> dict | None:
    if "execution" not in aggregate:
        return None
    execution = aggregate["execution"]
    transition_path = run / "parallel_transition.json"
    transition_hash = sha256_file(transition_path)
    transition = _read_record(transition_path)
    if (execution != manifest.get("execution")
            or execution["transition_sha256"] != transition_hash
            or manifest["files"].get("parallel_transition.json") != transition_hash
            or transition["execution"]["scientific_binding_sha256"] != aggregate["binding_sha256"]
            or not execution["completed_checkpoints_unchanged"]
            or not execution["scientific_binding_unchanged"]
            or execution["workers"] != transition["execution"]["workers"]
            or execution["blas_threads_per_worker"] != 4
            or execution["checkpoint_cells_preserved_at_switch"] != transition["preserved_cells"]):
        raise ValueError("parallel execution provenance mismatch")
    preserved_cells = 0
    for name, expected_hash in transition["preserved_files"].items():
        if sha256_file(run / name) != expected_hash:
            raise ValueError("checkpoint changed after parallel transition")
        if name.startswith("cells/") and name.endswith(".json"):
            preserved_cells += 1
    if preserved_cells != transition["preserved_cells"]:
        raise ValueError("parallel preserved checkpoint count mismatch")
    return {
        "workers": execution["workers"], "blas_threads_per_worker": 4,
        "completed_checkpoints_unchanged": True, "scientific_binding_unchanged": True,
        "checkpoint_cells_preserved_at_switch": preserved_cells,
        "transition_sha256": transition_hash,
    }


def validate_pca_study(aggregate: dict, manifest: dict, run: Path) -> dict | None:
    if "pca_study" not in aggregate:
        return None
    study = aggregate["pca_study"]
    source_paths = [run / name for name, digest in manifest["files"].items()
                    if digest == study["baseline_aggregate_sha256"]]
    pca_paths = [run / name for name, digest in manifest["files"].items()
                 if digest == study["pca_manifest_sha256"]]
    if len(source_paths) != 1 or len(pca_paths) != 1:
        raise ValueError("PCA requires uniquely bound prefix source and preparation manifests")
    source_path, pca_path = source_paths[0], pca_paths[0]
    if (sha256_file(source_path) != study["baseline_aggregate_sha256"]
            or sha256_file(source_path.parent / "manifest.json") != study["baseline_manifest_sha256"]
            or sha256_file(pca_path) != study["pca_manifest_sha256"]):
        raise ValueError("PCA source or preparation manifest changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    key = lambda row: (row["seed"], row["schedule"], row["dimension"], row["rate"])
    old = {key(row): row for row in source["rows"]}
    references = {key(row): row for row in aggregate["rows"] if row.get("representation") != "pca"}
    pca_rows = [row for row in aggregate["rows"] if row.get("representation") == "pca"]
    if (old.keys() != references.keys() or len(old) != study["reused_cells"]
            or len(pca_rows) != study["added_cells"]):
        raise ValueError("PCA combined study changes the preserved source cell grid")
    for identity, original in old.items():
        current = references[identity]
        if canonical({k: v for k, v in original.items() if k != "evidence"}) != canonical({
            k: v for k, v in current.items() if k != "evidence"
        }):
            raise ValueError("PCA comparison changed a preserved reference measurement")
        if (source_path.parent / original["evidence"]).resolve() != (run / current["evidence"]).resolve():
            raise ValueError("PCA comparison replaced original reference evidence")
    for row in pca_rows:
        if row.get("representation_id") != f"pca_{row['dimension']}":
            raise ValueError("PCA representation identity is ambiguous")
        reference = old.get(key(row))
        if reference is None or reference["replay_hashes"] != row["replay_hashes"]:
            raise ValueError("PCA row is not paired to its original reference stream")
        native = old[row["seed"], row["schedule"], 1536, row["rate"]]
        for field, baseline in (("paired_delta_native", native), ("paired_delta_prefix", reference)):
            if field not in row:
                continue
            for population, metrics in row[field].items():
                for metric, recorded in metrics.items():
                    value, target = row[population][metric], baseline[population][metric]
                    expected = None if value is None or target is None else value - target
                    if recorded != expected:
                        raise ValueError("PCA paired difference uses the wrong reference")
    if (study["baseline_rows_unchanged"] is not True or study["replay_pairing_exact"] is not True
            or study["embedding_calls"] != 0 or study["judge_calls"] != 0
            or study["fit_scope"] != "full_unlabeled_source_population"
            or study["fit_source_count"] != aggregate["dataset"]["sessions"]
            or study["fit_once"] is not True or study["whiten"] is not False
            or study["solver"] != "full"):
        raise ValueError("PCA fit or preservation provenance is incompatible")
    return {
        "baseline_rows_unchanged": True, "replay_pairing_exact": True,
        "reused_cells": len(old), "added_cells": len(pca_rows), "embedding_calls": 0, "judge_calls": 0,
        "fit_scope": study["fit_scope"], "fit_source_count": study["fit_source_count"],
        "solver": study["solver"], "whiten": False, "fit_once": True,
        "pca_manifest_sha256": study["pca_manifest_sha256"],
        "baseline_aggregate_sha256": study["baseline_aggregate_sha256"],
    }


def validate_pca_fit(native: np.ndarray, manifest_path: Path) -> dict:
    """Verify every retained projected row against the frozen numerical model."""
    manifest = _read_record(manifest_path)
    binding, fit = manifest["binding"], manifest["fit"]
    if (manifest["status"] != "completed" or manifest["binding_sha256"] != _digest(binding)
            or binding["solver"] != "full" or binding["whiten"] is not False
            or binding["fit_once"] is not True or binding["labels_used_for_fit"] is not False
            or binding["deduplicated"] is not False or binding["transductive"] is not True
            or binding["fit_scope"] != "full_unlabeled_source_population"
            or binding["input_shape"] != list(native.shape)
            or binding["fit_source_count"] != len(native) or fit["fit_count"] != 1
            or binding["input_hashes"]["vectors"] != array_sha256(native)):
        raise ValueError("PCA preparation binding or fit scope is incompatible")
    for name, expected in manifest["files"].items():
        if sha256_file(manifest_path.parent / name) != expected:
            raise ValueError("PCA preparation artifact hash mismatch")
    keys = ("mean", "components", "singular_values", "explained_variance",
            "explained_variance_ratio", "projected")
    arrays = {name: np.load(manifest_path.parent / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in keys}
    n, width = native.shape
    components_count = binding["n_components"]
    shapes = {"mean": (width,), "components": (components_count, width),
              "singular_values": (components_count,), "explained_variance": (components_count,),
              "explained_variance_ratio": (components_count,), "projected": (n, components_count)}
    for name, value in arrays.items():
        if (value.shape != shapes[name] or value.dtype != np.float32 or not np.isfinite(value).all()
                or array_sha256(value) != manifest["array_hashes"][name]):
            raise ValueError(f"PCA {name} shape/precision/array hash mismatch")
    x = normalized_prefix(native, width)
    if array_sha256(x) != fit["normalized_input_sha256"]:
        raise ValueError("PCA fit did not use the recorded normalized native input")
    mean = np.asarray(arrays["mean"], dtype=np.float64)
    components = np.asarray(arrays["components"], dtype=np.float64)
    expected_mean = np.mean(x, axis=0)
    if not np.allclose(mean, expected_mean, atol=2e-6, rtol=2e-5):
        raise ValueError("PCA mean does not match full source population")
    max_projection_error = max_inverse_error = 0.
    with threadpool_limits(limits=4):
        gram = components @ components.T
        if not np.allclose(gram, np.eye(components_count), atol=1e-5, rtol=1e-5):
            raise ValueError("PCA components are not orthonormal")
        for start in range(0, n, 512):
            centered = np.asarray(x[start:start + 512], dtype=np.float64) - mean
            projected = np.asarray(arrays["projected"][start:start + 512], dtype=np.float64)
            expected = centered @ components.T
            max_projection_error = max(max_projection_error, float(np.max(np.abs(projected - expected))))
            if not np.allclose(projected, expected, atol=1e-5, rtol=5e-5):
                raise ValueError("PCA projected rows do not match the unwhitened transform")
            if components_count == width:
                reconstructed = projected @ components
                max_inverse_error = max(max_inverse_error, float(np.max(np.abs(reconstructed - centered))))
                if not np.allclose(reconstructed, centered, atol=1e-5, rtol=5e-5):
                    raise ValueError("Full-rank PCA does not reconstruct centered native input")
    variance = np.var(arrays["projected"], axis=0, ddof=1, dtype=np.float64)
    explained = np.asarray(arrays["explained_variance"], dtype=np.float64)
    singular = np.asarray(arrays["singular_values"], dtype=np.float64)
    total_variance = np.var(x, axis=0, ddof=1, dtype=np.float64).sum()
    if (not np.allclose(variance, explained, atol=2e-7, rtol=1e-4)
            or not np.allclose(singular ** 2 / (n - 1), explained, atol=2e-7, rtol=1e-4)
            or not np.allclose(explained / total_variance, arrays["explained_variance_ratio"], atol=2e-7, rtol=1e-4)):
        raise ValueError("PCA variances or singular values are inconsistent")
    for dimension in binding["dimensions"]:
        projected = normalized_prefix(arrays["projected"], dimension)
        if not np.allclose(np.linalg.norm(projected, axis=1), 1., atol=2e-6, rtol=2e-6):
            raise ValueError("PCA dimension normalization failed")
    return {
        "ok": True, "source_rows_checked": n, "components": components_count,
        "full_rank_centered_control": components_count == width,
        "maximum_projection_absolute_error": max_projection_error,
        "maximum_full_rank_reconstruction_absolute_error": max_inverse_error if components_count == width else None,
        "projection_tolerances": {"atol": 1e-5, "rtol": 5e-5},
        "solver": "full", "whiten": False, "fit_count": 1,
        "labels_used_for_fit": False, "transductive": True,
    }


def validate_pca_scaling(aggregate: dict, run: Path) -> dict | None:
    execution = aggregate.get("pca_execution", {})
    if "scaling" not in execution:
        return None
    scaling = execution["scaling"]
    path = run / "scaling_transition.json"
    transition = _read_record(path)
    if (sha256_file(path) != scaling["transition_sha256"]
            or transition["execution"]["scientific_binding_sha256"] != aggregate["binding_sha256"]
            or transition["execution"]["workers"] != execution["workers"]
            or not 1 <= execution["workers"] <= 6 or execution["blas_threads_per_worker"] != 4
            or transition["preserved_cells"] != scaling["preserved_cells"]
            or transition["preserved_files"] != scaling["preserved_files"]
            or scaling["scientific_binding_unchanged"] is not True
            or scaling["completed_checkpoints_unchanged"] is not True):
        raise ValueError("PCA worker scaling provenance mismatch")
    cells = 0
    for name, digest in scaling["preserved_files"].items():
        artifact = (run / name).resolve()
        if not artifact.is_relative_to(run.resolve()) or sha256_file(artifact) != digest:
            raise ValueError("PCA checkpoint changed at worker scaling")
        cells += name.startswith("cells/") and name.endswith(".json")
    if cells != scaling["preserved_cells"]:
        raise ValueError("PCA scaling preserved-cell count mismatch")
    return {
        "workers": execution["workers"], "blas_threads_per_worker": 4,
        "preserved_cells": cells, "completed_checkpoints_unchanged": True,
        "scientific_binding_unchanged": True, "transition_sha256": scaling["transition_sha256"],
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
    native_vectors, labels, profile = load_input(input_manifest)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected_hash in manifest["files"].items():
        if sha256_file(run / relative) != expected_hash:
            raise ValueError(f"retained run artifact hash mismatch: {relative}")
    aggregate = json.loads((run / "aggregate.json").read_text(encoding="utf-8"))
    protocol = aggregate["protocol"]
    has_pca = "pca_study" in aggregate
    families = ("reference", "pca") if has_pca else ("reference",)
    planned_cells = len(protocol["dimensions"]) * len(families) * 40 * 2 * 5
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
    execution_validation = validate_execution(aggregate, manifest, run)
    pca_validation = validate_pca_study(aggregate, manifest, run)
    pca_scaling = validate_pca_scaling(aggregate, run)
    if tuple(protocol["dimensions"]) == DIMENSION_GRIDS[1] and extension_validation is None and not has_pca:
        raise ValueError("extended grid requires preserved-baseline provenance")
    if has_pca:
        expected_representations = {
            ("native_1536" if d == 1536 else f"prefix_{d}", "native" if d == 1536 else "prefix", d)
            for d in protocol["dimensions"]
        } | {(f"pca_{d}", "pca", d) for d in protocol["dimensions"]}
        recorded_representations = {(r["id"], r["family"], r["dimension"])
                                    for r in protocol["representations"]}
        if (recorded_representations != expected_representations
                or len(recorded_representations) != len(protocol["representations"])
                or aggregate["pca_study"]["dimensions"] != protocol["dimensions"]):
            raise ValueError("PCA family/dimension protocol grid is incomplete")
        pca_manifest = next(run / name for name, digest in manifest["files"].items()
                            if digest == aggregate["pca_study"]["pca_manifest_sha256"])
        pca_fit_validation = validate_pca_fit(native_vectors, pca_manifest)
    fields = (
        "label", "score", "lower", "upper", "source_id", "novel_source", "position",
        "max_donor_position", "max_calibration_position", "occurrence_id", "selected_occurrence_id",
    )
    maximum_error, inspected = 0.0, 0
    native_five = []
    identities = set()
    for row in aggregate["rows"]:
        family = "pca" if row.get("representation") == "pca" else "reference"
        identities.add((family, row["seed"], row["schedule"], row["dimension"], row["rate"]))
        with np.load(run / row["evidence"], allow_pickle=False) as arrays:
            evidence = {key: arrays[key] for key in fields}
        result = validate_scores(row, evidence, labels)
        maximum_error = max(maximum_error, result["max_metric_error"])
        inspected += len(evidence["label"])
        if family == "reference" and row["dimension"] == 1536 and row["rate"] == .05:
            native_five.append(result)
    seeds = {row["seed"] for row in aggregate["rows"]}
    expected_identities = {
        (family, seed, schedule, dimension, rate)
        for family in families for seed in seeds for schedule in protocol["schedules"]
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
    if execution_validation is not None:
        result["execution"] = execution_validation
        result["checks"].append("Byte-identical pre-parallel checkpoints and unchanged scientific binding")
    if pca_validation is not None:
        result["pca_study"] = pca_validation
        result["pca_fit"] = pca_fit_validation
        result["checks"].append("Distinct centered PCA identity, preserved prefix evidence, exact stream pairing and paired deltas")
        result["checks"].append("All PCA projected rows reconstructed from frozen full-population mean/components; no whitening")
    if pca_scaling is not None:
        result["pca_execution"] = pca_scaling
        result["checks"].append("Byte-identical PCA checkpoints preserved at the six-worker scheduling transition")
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
