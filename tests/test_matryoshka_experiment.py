from dataclasses import replace
import json
from pathlib import Path
import socket

import numpy as np
import pytest

from sampling_comparison.matryoshka_experiment import (
    DATASETS, SCHEDULES, InputDataset, binary_metrics, build_schedule, distance_blocks,
    evaluate, load_input, normalized_prefix, rank_membership, run_experiment, sha256_file,
    summarize, write_json,
)
from sampling_comparison.v4_idw import IDWConfig


def fixture_data(n=30, dataset_id="dense_2500"):
    rng = np.random.default_rng(123)
    return InputDataset(
        dataset_id, tuple(f"unit-{i:03}" for i in range(n)), tuple(f"tenant|agent-{i % 3}" for i in range(n)),
        tuple(("tool",) for _ in range(n)), tuple(f"concept-{i % 4}" for i in range(n)),
        rng.integers(0, 2, n).astype(float), rng.normal(size=(n, 1536)).astype(np.float32), {},
    )


def save_fixture(path, data):
    path.mkdir()
    units = [{"unit_id": uid, "agent_id": data.agents[i], "signature": list(data.signatures[i]),
              "concept_key": data.concepts[i], "label": float(data.labels[i])} for i, uid in enumerate(data.unit_ids)]
    write_json(path / "units.json", units)
    np.savez(path / "vectors.npz", unit_ids=np.asarray(data.unit_ids), vectors=data.vectors)
    manifest = {
        "version": "matryoshka-input-v1", "dataset_id": data.dataset_id,
        "embedding_model_id": "text-embedding-3-small", "embedding_dimensions": 1536,
        "label_source": "TEST fixture only", "representation_policy": "TEST random vectors, never a measured run",
        "files": {"units": "units.json", "vectors": "vectors.npz"},
        "hashes": {"units": sha256_file(path / "units.json"), "vectors": sha256_file(path / "vectors.npz")},
    }
    write_json(path / "manifest.json", manifest)
    return path / "manifest.json"


def test_prefix_is_first_coordinates_normalized_without_mutation():
    original = np.asarray([[3, 4, 9], [-3, 4, 8]], dtype=np.float32)
    before = original.copy()
    result = normalized_prefix(original, 2)
    np.testing.assert_allclose(result, [[0.6, 0.8], [-0.6, 0.8]])
    np.testing.assert_array_equal(original, before)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1)


@pytest.mark.parametrize("dimension", [0, 4, -1, 1.5, True])
def test_invalid_dimension_rejected(dimension):
    with pytest.raises(ValueError):
        normalized_prefix(np.ones((2, 3)), dimension)


def test_zero_prefix_and_nonfinite_are_errors_not_synthetic_fallbacks():
    with pytest.raises(ValueError, match="zero-norm"):
        normalized_prefix(np.asarray([[0, 0, 1]]), 2)
    with pytest.raises(ValueError, match="non-finite"):
        normalized_prefix(np.asarray([[1, np.nan]]), 1)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_paired_schedule_is_deterministic_and_complete(schedule):
    data = fixture_data()
    first = build_schedule(data.unit_ids, data.agents, schedule, 13)
    second = build_schedule(data.unit_ids, data.agents, schedule, 13)
    np.testing.assert_array_equal(first, second)
    assert set(first[0]) == set(range(len(data.unit_ids)))
    assert np.all(np.diff(first[1]) >= 0)


def reference_predictions(data, blocks, order, selected):
    cfg = IDWConfig()
    result = np.full(len(order), cfg.prior)
    provenance = []
    earlier = []
    for i in order:
        if i in selected:
            result[i] = data.labels[i]
            provenance.append((i, "observed"))
            earlier.append(i)
            continue
        ids, cos, ang = blocks[data.agents[i]]
        donors = [j for j in earlier if data.agents[j] == data.agents[i]]
        if not donors:
            result[i] = np.mean(data.labels[earlier]) if earlier else cfg.prior
            provenance.append((i, "global_mean" if earlier else "prior"))
            continue
        local = {int(index): pos for pos, index in enumerate(ids)}
        donors.sort(key=lambda j: (ang[local[i], local[j]], data.unit_ids[j]))
        exact = [j for j in donors if 1 - cos[local[i], local[j]] <= cfg.exact_cosine_eps]
        if exact:
            result[i] = np.mean(data.labels[exact])
            provenance.append((i, "exact_match"))
        else:
            nearest = donors[:cfg.k]
            weights = np.asarray([1 / (float(ang[local[i], local[j]]) + cfg.eps) ** cfg.power for j in nearest])
            result[i] = np.dot(weights / weights.sum(), data.labels[nearest])
            provenance.append((i, "idw"))
    return result, np.asarray([value for _, value in sorted(provenance)])


@pytest.mark.parametrize("seed", [13, 42])
@pytest.mark.parametrize("dimension", [8, 512, 1536])
def test_vectorized_idw_matches_scalar_causal_oracle(seed, dimension):
    data = fixture_data(60)
    vectors = normalized_prefix(data.vectors, dimension)
    blocks = distance_blocks(data, vectors)
    order, _ = build_schedule(data.unit_ids, data.agents, "uniformly_random", seed)
    selected = order[::5]
    metrics, values, provenance = evaluate(data, blocks, order, selected)
    expected, expected_provenance = reference_predictions(data, blocks, order, selected)
    np.testing.assert_allclose(values, expected, atol=1e-12)
    np.testing.assert_array_equal(provenance, expected_provenance)
    assert metrics["selected_count"] + metrics["unjudged_count"] == 60
    assert sum(metrics["provenance_counts"].values()) == metrics["unjudged_count"]
    assert metrics["accuracy"] == binary_metrics(data.labels[~np.isin(np.arange(60), selected)],
                                                 values[~np.isin(np.arange(60), selected)])["accuracy"]


def test_future_and_unselected_labels_cannot_change_earlier_predictions():
    data = fixture_data(12)
    order = np.arange(12)
    selected = np.asarray([2, 5, 10])
    blocks = distance_blocks(data, normalized_prefix(data.vectors, 8))
    _, before, _ = evaluate(data, blocks, order, selected)
    labels = data.labels.copy()
    labels[10:] = 1 - labels[10:]
    _, after, _ = evaluate(replace(data, labels=labels), blocks, order, selected)
    np.testing.assert_array_equal(before[:10], after[:10])
    labels = data.labels.copy()
    labels[~np.isin(order, selected)] = 1 - labels[~np.isin(order, selected)]
    _, changed, _ = evaluate(replace(data, labels=labels), blocks, order, selected)
    np.testing.assert_array_equal(before, changed)


def test_exact_match_all_donors_and_prior_fallback():
    data = fixture_data(12)
    vectors = np.zeros_like(data.vectors)
    vectors[:, 0] = 1
    labels = np.asarray([0, 1] * 6, dtype=float)
    data = replace(data, vectors=vectors, labels=labels, agents=("a",) * 12)
    blocks = distance_blocks(data, vectors)
    _, predictions, provenance = evaluate(data, blocks, np.arange(12), np.arange(10))
    assert predictions[10] == 0.5
    assert provenance[10] == "exact_match"
    _, predictions, provenance = evaluate(data, blocks, np.arange(12), np.asarray([], dtype=int))
    np.testing.assert_array_equal(predictions, np.full(12, 0.5))
    assert set(provenance) == {"prior"}


def test_selector_label_blind_and_representation_sensitive():
    data = fixture_data()
    order, times = build_schedule(data.unit_ids, data.agents, "bursty", 13)
    vectors = normalized_prefix(data.vectors, 8)
    first, telemetry = rank_membership(data.unit_ids, data.agents, data.signatures, vectors, order, times, 13)
    second, _ = rank_membership(data.unit_ids, data.agents, data.signatures, vectors, order, times, 13)
    np.testing.assert_array_equal(first, second)
    assert telemetry["representation_fallbacks"] == 0
    assert set(first) == set(range(30))


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_all_four_dataset_ids_load_without_network(tmp_path, dataset_id):
    path = save_fixture(tmp_path / "input", fixture_data(dataset_id=dataset_id))
    assert load_input(path).dataset_id == dataset_id
    with (path.parent / "units.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_input(path)


def test_grid_pairing_partial_status_resume_and_offline(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("experiment attempted network access")
    monkeypatch.setattr(socket, "create_connection", no_network)
    data = fixture_data(12)
    path = save_fixture(tmp_path / "input", data)
    kwargs = dict(source_revision="test", dimensions=(1536, 8), seeds=(13, 14), rates=(0.1, 0.2), schedules=("bursty",))
    result = run_experiment({"dense_2500": path}, tmp_path / "run", **kwargs)
    assert result["status"] == "partial"
    assert len(result["rows"]) == 16
    assert sum(d["status"] == "blocked" for d in result["datasets"]) == 3
    for row in result["rows"]:
        if row["dimension"] == 1536:
            assert row["accuracy_delta_native"] == 0
        if row["mode"] == "fixed_membership":
            assert row["membership_jaccard_native"] == 1
    repeat = run_experiment({"dense_2500": path}, tmp_path / "run", resume=True, **kwargs)
    assert result["rows"] == repeat["rows"]
    with pytest.raises(FileExistsError):
        run_experiment({"dense_2500": path}, tmp_path / "run", **kwargs)
    with pytest.raises(ValueError, match="fingerprint"):
        run_experiment({"dense_2500": path}, tmp_path / "run", resume=True, **{**kwargs, "seeds": (13,)})


def test_decision_requires_all_larger_prefixes_and_keeps_negative_result():
    rows = []
    for dimension, delta in [(1536, 0), (512, -0.03), (8, 0.02)]:
        for seed in [13, 14]:
            row = {key: 0.5 for key in ("accuracy", "mae", "f1", "brier", "macro_agent_accuracy",
                                        "combined_accuracy", "aggregate_rate_error", "concept_coverage", "agent_coverage")}
            rows.append({**row, "dataset_id": "dense_2500", "mode": "end_to_end",
                         "dimension": dimension, "seed": seed, "accuracy_delta_native": delta})
    _, decisions = summarize(rows, 0.01)
    assert decisions[0]["smallest_non_degrading_dimension"] is None
    assert decisions[0]["one_pp_candidate_dimension"] is None
