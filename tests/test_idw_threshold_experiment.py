from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from sampling_comparison.idw_threshold_experiment import (
    _assert_threshold_dominance,
    envelope_evidence,
    exact_roc,
    threshold_metrics,
)
from sampling_comparison.matryoshka_experiment import (
    InputDataset,
    distance_blocks,
    evaluate,
    normalized_prefix,
    sha256_file,
)


def fixture_data(labels=None, vectors=None, agents=None) -> InputDataset:
    n = 8
    rng = np.random.default_rng(77)
    labels = np.asarray(labels if labels is not None else [0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
    vectors = np.asarray(
        vectors if vectors is not None else rng.normal(size=(n, 1536)), dtype=np.float32
    )
    agents = tuple(agents if agents is not None else ["a"] * n)
    return InputDataset(
        "historical_300", tuple(f"u{i}" for i in range(n)), agents,
        tuple(("tool",) for _ in range(n)), tuple(f"c{i}" for i in range(n)),
        labels, vectors, {},
    )


def calculate(data, selected=(0, 2, 5)):
    vectors = normalized_prefix(data.vectors, 8)
    blocks = distance_blocks(data, vectors)
    order = np.arange(len(data.unit_ids))
    chosen = np.asarray(selected, dtype=int)
    _, scores, provenance = evaluate(data, blocks, order, chosen)
    evidence = envelope_evidence(data, blocks, order, chosen, scores, provenance)
    return scores, evidence


def test_exact_roc_known_answer_ties_and_endpoints():
    labels = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    scores = np.asarray([0.1, 0.4, 0.35, 0.8])
    result = exact_roc(labels, scores)
    assert result["auc"] == pytest.approx(0.75)
    assert np.isinf(result["thresholds"][0])
    assert (result["fpr"][0], result["tpr"][0]) == (0, 0)
    assert (result["fpr"][-1], result["tpr"][-1]) == (1, 1)

    tied = exact_roc(np.asarray([0, 1]), np.asarray([0.5, 0.5]))
    assert tied["auc"] == pytest.approx(0.5)
    assert tied["thresholds"].tolist()[1:] == [0.5]
    assert tied["fpr"].tolist() == [0.0, 1.0]
    assert tied["tpr"].tolist() == [0.0, 1.0]


def test_exact_roc_empty_and_one_class_are_null_with_reasons():
    empty = exact_roc(np.asarray([], dtype=np.uint8), np.asarray([], dtype=float))
    assert empty["auc"] is None and empty["reason"] == "empty_population"
    positive = exact_roc(np.ones(3, dtype=np.uint8), np.asarray([0.1, 0.2, 0.3]))
    assert positive["auc"] is None and positive["reason"] == "no_negative_labels"
    assert np.isnan(positive["fpr"]).all()
    negative = exact_roc(np.zeros(3, dtype=np.uint8), np.asarray([0.1, 0.2, 0.3]))
    assert negative["auc"] is None and negative["reason"] == "no_positive_labels"


def test_threshold_metrics_zero_division_and_empty_conventions():
    metric = threshold_metrics(np.asarray([0, 1]), np.asarray([0.1, 0.2]), np.inf)
    assert metric["precision"] == 0
    assert metric["recall"] == 0
    assert metric["f1"] == 0
    empty = threshold_metrics(np.asarray([], dtype=np.uint8), np.asarray([]), 0.5)
    assert empty["accuracy"] is None and empty["reason"] == "empty_population"


def test_no_target_or_future_label_leakage():
    data = fixture_data()
    before_scores, before = calculate(data)
    # Mutating an unselected target's own label cannot alter its score or bound.
    target = 4
    changed_labels = data.labels.copy()
    changed_labels[target] = 1 - changed_labels[target]
    after_scores, after = calculate(replace(data, labels=changed_labels))
    target_offset = np.flatnonzero(before["target_id"] == target)[0]
    assert before_scores[target] == after_scores[target]
    assert before["lower"][target_offset] == after["lower"][target_offset]
    # A later selected label cannot alter earlier target evidence.
    changed_labels = data.labels.copy()
    changed_labels[5] = 1 - changed_labels[5]
    _, future = calculate(replace(data, labels=changed_labels))
    early = before["position"] < 5
    np.testing.assert_array_equal(before["score"][early], future["score"][early])
    np.testing.assert_array_equal(before["lower"][early], future["lower"][early])


def test_exact_match_contradiction_and_sparse_calibration_are_audited():
    vectors = np.zeros((8, 1536), dtype=np.float32)
    vectors[:, 0] = 1
    data = fixture_data(labels=[0, 1, 0, 1, 1, 0, 1, 0], vectors=vectors)
    _, evidence = calculate(data, selected=(0, 1, 4))
    target_two = np.flatnonzero(evidence["target_id"] == 2)[0]
    assert evidence["provenance"][target_two] == 3  # exact_match
    assert evidence["exact_contradiction_count"][target_two] == 1
    assert evidence["lipschitz"][target_two] >= 99
    target_three = np.flatnonzero(evidence["target_id"] == 3)[0]
    assert evidence["max_donor_position"][target_three] < evidence["position"][target_three]

    _, sparse = calculate(data, selected=(0,))
    first_target = np.flatnonzero(sparse["target_id"] == 1)[0]
    assert sparse["calibration_fallback"][first_target] == 1
    assert sparse["calibration_donor_count"][first_target] == 1


def test_no_same_agent_donor_has_no_claimed_envelope():
    data = fixture_data(agents=("a", "b", "b", "b", "b", "b", "b", "b"))
    _, evidence = calculate(data, selected=(0,))
    assert np.isnan(evidence["lower"]).all()
    assert set(evidence["provenance"]) <= {0, 1}


def test_matched_cohort_dominance_and_pairing():
    labels = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    point = np.asarray([0.8, 0.9, 0.4, 0.7])
    lower = np.asarray([0.2, 0.8, 0.1, 0.5])
    _assert_threshold_dominance(labels, point, lower)
    with pytest.raises(AssertionError, match="subset"):
        _assert_threshold_dominance(labels, point, np.asarray([0.9, 0.8, 0.1, 0.5]))


def test_hash_tampering_detected_and_source_hash_utility_preserves_file(tmp_path: Path):
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    before = sha256_file(path)
    assert sha256_file(path) == before
    path.write_text(json.dumps({"a": 2}), encoding="utf-8")
    assert sha256_file(path) != before


def test_fresh_canonical_run_registers_exact_acceptance_without_manual_amendments(tmp_path, monkeypatch):
    import sampling_comparison.idw_threshold_experiment as experiment

    class ReachedInputs(Exception):
        pass

    def stop_at_inputs(_):
        raise ReachedInputs

    monkeypatch.setattr(experiment.sys, "version_info", (3, 13, 13))
    monkeypatch.setattr(experiment.np, "__version__", "2.5.3")
    monkeypatch.setattr(experiment, "threadpool_info", lambda: [
        {"internal_api": "openblas", "num_threads": 12},
    ])
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "12")
    monkeypatch.setattr(experiment, "input_artifacts", stop_at_inputs)
    with pytest.raises(ReachedInputs):
        experiment.run_experiment(tmp_path / "fresh", source_revision="fixture", require_canonical_runtime=True)
    acceptance = json.loads((tmp_path / "fresh" / "preregistration.json").read_text())["acceptance"]
    assert acceptance["canonical_runtime_mae_absolute_tolerance"] == 0.0
    assert acceptance["canonical_runtime_accuracy_absolute_tolerance"] == 0.0


def test_refreshed_input_must_match_its_baseline_not_just_dataset_name():
    from sampling_comparison.idw_threshold_experiment import verify_baseline_inputs

    data = replace(fixture_data(), profile={"input_manifest_sha256": "fresh"})
    baseline = {"datasets": [{
        "dataset_id": "historical_300", "status": "completed", "n": 8,
        "provenance": {"input_manifest_sha256": "fresh"},
    }]}
    verify_baseline_inputs(baseline, {"historical_300": data})
    baseline["datasets"][0]["provenance"]["input_manifest_sha256"] = "old"
    with pytest.raises(ValueError, match="hash"):
        verify_baseline_inputs(baseline, {"historical_300": data})


def test_explicit_input_override_preserves_other_corpora():
    from scripts.run_idw_threshold_experiment import resolve_inputs
    from sampling_comparison.idw_threshold_experiment import INPUTS

    inputs = resolve_inputs([r"cosmos_otel=private\fresh\manifest.json"])
    assert inputs["cosmos_otel"] == Path(r"private\fresh\manifest.json")
    assert inputs["dense_2500"] == INPUTS["dense_2500"]
    for invalid in (["unknown=x"], ["cosmos_otel="], ["cosmos_otel=x", "cosmos_otel=y"]):
        with pytest.raises(ValueError):
            resolve_inputs(invalid)


def test_validator_uses_recorded_input_paths_and_checks_hashes(monkeypatch):
    import scripts.validate_idw_threshold_experiment as validator

    visited = []
    monkeypatch.setattr(validator, "sha256_file", lambda path: "digest")
    def load(path):
        visited.append(str(path))
        return replace(fixture_data(), dataset_id=path.parent.name)
    monkeypatch.setattr(validator, "load_input", load)
    aggregate = {"datasets": [
        {"dataset_id": name, "manifest": str(Path("private") / name / "manifest.json"),
         "manifest_sha256": "digest", "sessions": 8}
        for name in validator.DATASETS
    ]}
    assert set(validator.load_recorded_inputs(aggregate)) == set(validator.DATASETS)
    assert all(path.startswith("private") for path in visited)
    aggregate["datasets"][0]["manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        validator.load_recorded_inputs(aggregate)
