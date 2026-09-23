import copy

import numpy as np
import pytest

from sampling_comparison.imdb_experiment import _digest, _write_record, array_sha256
from sampling_comparison.matryoshka_experiment import normalized_prefix, sha256_file, write_json
from scripts.validate_imdb_experiment import validate_pca_fit, validate_pca_study


def pca_fixture(tmp_path):
    source, fit, output = tmp_path / "reference", tmp_path / "fit", tmp_path / "pca"
    for path in (source / "cells", fit, output):
        path.mkdir(parents=True)
    original = []
    for dimension, mae in ((1536, .2), (8, .4)):
        (source / "cells" / f"{dimension}.npz").write_bytes(b"fixture-only evidence")
        original.append({
            "seed": 13, "schedule": "bursty", "rate": .05, "dimension": dimension,
            "all_unselected": {"mae": mae}, "replay_hashes": {"order": "same", "draw": "same"},
            "evidence": f"cells/{dimension}.npz",
        })
    write_json(source / "aggregate.json", {"rows": original})
    write_json(source / "manifest.json", {"fixture": "source"})
    write_json(fit / "manifest.json", {"fixture": "fit"})
    rows = [{**copy.deepcopy(r), "evidence": "../reference/" + r["evidence"]} for r in original]
    for original_row, mae in zip(original, (.21, .25), strict=True):
        dimension = original_row["dimension"]
        rows.append({
            **copy.deepcopy(original_row), "representation": "pca", "representation_id": f"pca_{dimension}",
            "all_unselected": {"mae": mae}, "evidence": f"cells/pca-{dimension}.npz",
            "paired_delta_native": {"all_unselected": {"mae": mae - .2}},
            "paired_delta_prefix": {"all_unselected": {"mae": mae - original_row["all_unselected"]["mae"]}},
        })
    study = {
        "baseline_aggregate_sha256": sha256_file(source / "aggregate.json"),
        "baseline_manifest_sha256": sha256_file(source / "manifest.json"),
        "pca_manifest_sha256": sha256_file(fit / "manifest.json"),
        "reused_cells": 2, "added_cells": 2, "baseline_rows_unchanged": True,
        "replay_pairing_exact": True, "embedding_calls": 0, "judge_calls": 0,
        "fit_scope": "full_unlabeled_source_population", "fit_source_count": 2000,
        "fit_once": True, "whiten": False, "solver": "full",
    }
    manifest = {"files": {
        "../reference/aggregate.json": study["baseline_aggregate_sha256"],
        "../fit/manifest.json": study["pca_manifest_sha256"],
    }}
    return {"dataset": {"sessions": 2000}, "rows": rows, "pca_study": study}, manifest, output


def test_pca_and_native_full_dimensions_remain_distinct(tmp_path):
    aggregate, manifest, output = pca_fixture(tmp_path)
    result = validate_pca_study(aggregate, manifest, output)
    assert result["reused_cells"] == result["added_cells"] == 2
    assert result["fit_once"] and result["baseline_rows_unchanged"]
    assert result["embedding_calls"] == result["judge_calls"] == 0


@pytest.mark.parametrize(("mutation", "message"), [
    ("reference", "preserved reference measurement"), ("path", "replaced original"),
    ("replay", "not paired"), ("identity", "identity is ambiguous"),
    ("delta", "wrong reference"), ("whiten", "provenance is incompatible"),
    ("fit", "preparation manifest changed"), ("source_manifest", "preparation manifest changed"),
])
def test_pca_validation_rejects_source_or_reference_drift(tmp_path, mutation, message):
    aggregate, manifest, output = pca_fixture(tmp_path)
    if mutation == "reference":
        aggregate["rows"][0]["all_unselected"]["mae"] = .9
    elif mutation == "path":
        aggregate["rows"][0]["evidence"] = "different.npz"
    elif mutation == "replay":
        aggregate["rows"][2]["replay_hashes"]["order"] = "different"
    elif mutation == "identity":
        aggregate["rows"][2]["representation_id"] = "native_1536"
    elif mutation == "delta":
        aggregate["rows"][2]["paired_delta_native"]["all_unselected"]["mae"] = 0
    elif mutation == "whiten":
        aggregate["pca_study"]["whiten"] = True
    elif mutation == "fit":
        write_json(tmp_path / "fit" / "manifest.json", {"changed": True})
    else:
        write_json(tmp_path / "reference" / "manifest.json", {"changed": True})
    with pytest.raises(ValueError, match=message):
        validate_pca_study(aggregate, manifest, output)


def fit_fixture(tmp_path, *, whiten_scores=False):
    native = np.random.default_rng(77).normal(size=(32, 4)).astype(np.float32)
    x = normalized_prefix(native, 4)
    mean = x.mean(axis=0)
    _, singular, components = np.linalg.svd(x - mean, full_matrices=False)
    projected = (x - mean) @ components.T
    explained = singular ** 2 / (len(x) - 1)
    if whiten_scores:
        projected /= np.sqrt(explained)
    arrays = {
        "mean": mean, "components": components, "singular_values": singular,
        "explained_variance": explained,
        "explained_variance_ratio": explained / np.var(x, axis=0, ddof=1).sum(),
        "projected": projected,
    }
    for name, value in arrays.items():
        np.save(tmp_path / f"{name}.npy", value.astype(np.float32))
    binding = {
        "solver": "full", "whiten": False, "fit_once": True, "labels_used_for_fit": False,
        "deduplicated": False, "transductive": True, "fit_scope": "full_unlabeled_source_population",
        "input_shape": list(native.shape), "fit_source_count": len(x), "n_components": 4,
        "dimensions": [4, 2], "input_hashes": {"vectors": array_sha256(native)},
    }
    path = tmp_path / "manifest.json"
    _write_record(path, {
        "status": "completed", "binding": binding, "binding_sha256": _digest(binding),
        "fit": {"fit_count": 1, "normalized_input_sha256": array_sha256(x)},
        "array_hashes": {name: array_sha256(value.astype(np.float32)) for name, value in arrays.items()},
        "files": {f"{name}.npy": sha256_file(tmp_path / f"{name}.npy") for name in arrays},
    })
    return native, path


def test_independent_fit_validation_checks_every_projection_and_full_rank_control(tmp_path):
    native, manifest = fit_fixture(tmp_path)
    result = validate_pca_fit(native, manifest)
    assert result["ok"] and result["full_rank_centered_control"]
    assert result["source_rows_checked"] == 32
    assert result["maximum_projection_absolute_error"] < 1e-5
    assert result["maximum_full_rank_reconstruction_absolute_error"] < 1e-5


def test_independent_fit_validation_rejects_whitening_even_with_consistent_hashes(tmp_path):
    native, manifest = fit_fixture(tmp_path, whiten_scores=True)
    with pytest.raises(ValueError, match="unwhitened transform"):
        validate_pca_fit(native, manifest)
