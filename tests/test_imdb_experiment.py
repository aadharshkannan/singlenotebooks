"""Small offline fixtures: these tests are not IMDb performance measurements."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import sampling_comparison.imdb_experiment as engine
from sampling_comparison.matryoshka_experiment import normalized_prefix, rank_membership


def fixture_inputs(n=24):
    vectors = np.random.default_rng(7).normal(size=(n, 1536)).astype(np.float32)
    return vectors, (np.arange(n) % 2).astype(np.uint8)


def run_small(path, **overrides):
    vectors, labels = fixture_inputs()
    arguments = dict(
        profile={"dataset_id": "offline-fixture", "embedding_model_id": "text-embedding-3-small",
                 "representation_policy": "random fixture vectors, NOT actual IMDb embeddings"},
        dimensions=(1536, 8), repetitions=2, rates=(.2, .4),
        schedules=engine.SCHEDULES, target_block_size=5, donor_block_size=3,
        calibration_reservoir_size=4,
    )
    arguments.update(overrides)
    return engine.run_experiment(vectors, labels, path, **arguments)


def reference(vectors, labels, sources, order, selected):
    """Intentionally simple, exhaustive small-fixture oracle, not production."""
    vectors = np.asarray(vectors, dtype=np.float64)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    chosen = set(selected)
    earlier = []
    answer = {}
    for position, occurrence in enumerate(order):
        if occurrence in chosen:
            earlier.append(int(occurrence))
            continue
        if not earlier:
            answer[occurrence] = (.5, 0, np.nan, [], 0)
            continue
        ids = np.asarray(sorted(earlier))
        cosine = np.clip(vectors[sources[ids]] @ vectors[sources[occurrence]], -1, 1)
        distance = np.arccos(cosine) / np.pi
        same = sources[ids] == sources[occurrence]
        distance[same] = 0
        exact = (1 - cosine <= 1e-8) | same
        if exact.any():
            used = ids[exact]
            score = np.mean(labels[sources[used]])
            weighted_distance = np.mean(distance[exact])
            provenance = 3
        else:
            nearest = sorted(range(len(ids)), key=lambda j: (distance[j], ids[j]))[:8]
            used = ids[nearest]
            weights = 1 / (distance[nearest] + 1e-6) ** 2
            score = np.dot(weights, labels[sources[used]]) / weights.sum()
            weighted_distance = np.dot(weights, distance[nearest]) / weights.sum()
            provenance = 2
        answer[occurrence] = (score, provenance, weighted_distance, list(used), int(exact.sum()))
    return answer


def test_bootstrap_occurrences_unique_pairing_and_different_schedules():
    random = engine.bootstrap_replay(80, 13, "uniformly_random")
    bursty = engine.bootstrap_replay(80, 13, "bursty")
    again = engine.bootstrap_replay(80, 13, "uniformly_random")
    assert len(np.unique(random["occurrence_id"])) == 80
    assert len(np.unique(random["source_id"])) < 80
    for name in ("occurrence_id", "source_id"):
        np.testing.assert_array_equal(random[name], bursty[name])
    for name in random:
        np.testing.assert_array_equal(random[name], again[name])
    assert not np.array_equal(random["order"], bursty["order"])
    assert not np.array_equal(random["timestamps"], bursty["timestamps"])
    assert not np.array_equal(random["source_id"], engine.bootstrap_replay(80, 14, "bursty")["source_id"])
    assert np.all(np.diff(random["timestamps"]) >= 0)
    assert np.all(np.diff(bursty["timestamps"]) >= 0)


def test_prefix_is_first_coordinates_normalized_not_pca():
    vectors, _ = fixture_inputs(4)
    for dimension in engine.DIMENSIONS:
        actual = normalized_prefix(vectors, dimension)
        expected = vectors[:, :dimension] / np.linalg.norm(vectors[:, :dimension], axis=1, keepdims=True)
        np.testing.assert_array_equal(actual, expected)
    vectors[:, 8:] *= 100
    np.testing.assert_array_equal(normalized_prefix(vectors, 8), expected)


@pytest.mark.parametrize("target_block,donor_block", [(1, 1), (3, 4), (7, 3), (200, 200)])
def test_bounded_blocks_match_exhaustive_reference(target_block, donor_block, monkeypatch):
    rng = np.random.default_rng(27)
    vectors = rng.normal(size=(19, 12))
    labels = (np.arange(19) % 2).astype(np.uint8)
    sources = rng.integers(0, 19, 41)
    order = rng.permutation(len(sources))
    selected = order[[0, 2, 4, 5, 7, 9, 10, 12, 17, 20, 24, 28, 35]]
    calls = []
    original = engine._angular_block

    def bounded(targets, donors):
        calls.append((len(targets), len(donors)))
        # Calibration has a separately bounded 1 x reservoir block.
        assert len(targets) <= max(target_block, 1)
        assert len(donors) <= max(donor_block, 4)
        return original(targets, donors)

    monkeypatch.setattr(engine, "_angular_block", bounded)
    actual = engine.evaluate_replay(
        vectors, labels, sources, order, selected, target_block_size=target_block,
        donor_block_size=donor_block, calibration_reservoir_size=4,
    )
    expected = reference(vectors, labels, sources, order, selected)
    assert calls
    for offset, occurrence in enumerate(actual["occurrence_id"]):
        point, code, distance, donors, exact_count = expected[occurrence]
        assert actual["score"][offset] == pytest.approx(point, abs=1e-14)
        assert actual["provenance"][offset] == code
        assert actual["exact_count"][offset] == exact_count
        assert actual["weighted_distance"][offset] == pytest.approx(distance, abs=1e-8)
        if code == 2:
            used = actual["neighbor_occurrence_id"][offset]
            assert used[used >= 0].tolist() == donors
    eligible = np.isfinite(actual["lower"])
    assert np.all(actual["lower"][eligible] <= actual["score"][eligible])
    assert np.all(actual["max_donor_position"] < actual["position"])
    assert np.all(actual["max_calibration_position"] < actual["position"])
    assert actual["calibration_donor_count"].max() <= 4
    assert actual["calibration_pair_count"].max() <= 6


def test_nearest_ties_across_blocks_use_occurrence_id():
    vectors = np.zeros((15, 2))
    vectors[:12, 0] = 1
    vectors[12:, 1] = 1
    labels = np.r_[np.zeros(8), np.ones(7)]
    sources = np.arange(15)
    order = np.r_[np.arange(12)[::-1], np.arange(12, 15)]
    expected = np.arange(8)
    for block in (1, 3, 15):
        evidence = engine.evaluate_replay(
            vectors, labels, sources, order, np.arange(12), donor_block_size=block,
        )
        np.testing.assert_array_equal(evidence["neighbor_occurrence_id"][0], expected)
        assert evidence["score"][0] == 0


def test_exact_match_averages_all_earlier_exact_donors_not_only_eight():
    vectors = np.ones((14, 2))
    labels = np.r_[np.zeros(8), np.ones(6)]
    evidence = engine.evaluate_replay(
        vectors, labels, np.arange(14), np.arange(14), np.arange(12), donor_block_size=3,
    )
    np.testing.assert_allclose(evidence["score"], 4 / 12)
    assert evidence["provenance"].tolist() == [3, 3]
    assert evidence["used_donor_count"].tolist() == [12, 12]
    assert np.all(evidence["neighbor_occurrence_id"] == -1)
    assert np.all(evidence["calibration_exact_contradiction_count"] > 0)


def test_empty_donors_prior_and_mean_fallback_are_distinct():
    vectors, labels = fixture_inputs(5)
    evidence = engine.evaluate_replay(
        vectors, labels, np.arange(5), np.arange(5), np.asarray([], dtype=int),
    )
    assert evidence["score"].tolist() == [.5] * 5
    assert evidence["provenance"].tolist() == [0] * 5
    assert np.isnan(evidence["lower"]).all()
    scores, codes = engine._fallback_scores(np.array([0, 2, 3]), np.array([0, 2, 1]), .5)
    np.testing.assert_allclose(scores, [.5, 1, 1 / 3])
    assert codes.tolist() == [0, 1, 1]
    metrics = engine.cell_metrics(evidence, np.array([]))
    assert metrics["eligible_point"]["n"] == metrics["eligible_lower"]["n"] == 0
    assert metrics["eligible_point"]["auc"] is None
    assert metrics["counts"]["provenance"]["prior"] == 5


def test_duplicate_source_no_warm_start_or_future_source_leakage():
    vectors = np.array([[1., 0], [0, 1], [-1, 0]])
    labels = np.array([1, 0, 0])
    # Sources 0 and 1 occur repeatedly BEFORE their first selected occurrence.
    sources = np.array([0, 0, 1, 0, 1, 0, 1, 2])
    evidence = engine.evaluate_replay(vectors, labels, sources, np.arange(8), np.array([3, 6]))
    assert evidence["occurrence_id"].tolist() == [0, 1, 2, 4, 5, 7]
    assert evidence["novel_source"].tolist() == [True, True, True, True, False, True]
    assert evidence["provenance"][:3].tolist() == [0, 0, 0]
    assert evidence["provenance"][4] == 3
    assert evidence["score"][4] == 1
    metrics = engine.cell_metrics(evidence, labels[sources[[3, 6]]])
    assert metrics["all_unselected"]["n"] == 6
    assert metrics["novel_source"]["n"] == 5
    assert metrics["repeated_source"]["n"] == 1
    assert metrics["counts"]["repeated_source_exact_reuse"] == 1
    assert metrics["combined_secondary"]["n"] == 8


def test_duplicate_content_exact_match_is_not_same_source_reuse():
    # Two independently labeled source rows share an embedding (duplicate
    # normalized review text). The embedding cache must not collapse their IDs.
    vectors = np.array([[1., 0], [1., 0]])
    labels = np.array([1, 0])
    sources = np.array([0, 1, 0, 1, 1])
    selected = np.array([0, 3])
    evidence = engine.evaluate_replay(vectors, labels, sources, np.arange(5), selected)
    assert evidence["occurrence_id"].tolist() == [1, 2, 4]
    assert evidence["source_id"].tolist() == [1, 0, 1]
    assert evidence["provenance"].tolist() == [3, 3, 3]
    assert evidence["novel_source"].tolist() == [True, False, False]
    assert evidence["score"].tolist() == [1., 1., .5]
    metrics = engine.cell_metrics(evidence, labels[sources[selected]])
    assert metrics["novel_source"]["n"] == 1
    assert metrics["novel_source"]["mae"] == 1
    assert metrics["repeated_source"]["n"] == 2
    assert metrics["counts"]["repeated_source_exact_reuse"] == 2
    assert metrics["counts"]["provenance"]["exact_match"] == 3


def test_future_labels_cannot_change_points_bounds_or_calibration_and_target_not_used():
    vectors, labels = fixture_inputs(18)
    sources, order = np.arange(18), np.arange(18)
    selected = np.array([1, 3, 6, 9, 12, 16])
    before = engine.evaluate_replay(vectors, labels, sources, order, selected, calibration_reservoir_size=3)
    changed = labels.copy()
    changed[9:] = 1 - changed[9:]
    future = engine.evaluate_replay(vectors, changed, sources, order, selected, calibration_reservoir_size=3)
    early = before["position"] < 9
    for name in ("score", "lower", "upper", "lipschitz", "calibration_version", "calibration_pair_count"):
        np.testing.assert_array_equal(before[name][early], future[name][early])
    changed = labels.copy()
    changed[4] = 1 - changed[4]
    target = engine.evaluate_replay(vectors, changed, sources, order, selected, calibration_reservoir_size=3)
    for name in ("score", "lower", "upper", "lipschitz", "calibration_events"):
        np.testing.assert_array_equal(before[name], target[name])
    # Reservoir admission/eviction is independent of ANY labels.
    flipped = engine.evaluate_replay(vectors, 1 - labels, sources, order, selected, calibration_reservoir_size=3)
    np.testing.assert_array_equal(before["calibration_events"], flipped["calibration_events"])


def test_reservoir_bottom_hash_predictable_cached_only_on_changes():
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(60, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    calibration = engine._CausalCalibration(
        vectors, np.arange(60) % 2, np.arange(60), 13, 4, engine.LIPSCHITZ_CONFIG,
    )
    unchanged = 0
    for occurrence in range(60):
        previous = calibration.estimate
        changed = calibration.observe(occurrence, occurrence)
        if not changed:
            unchanged += 1
            assert calibration.estimate is previous
        expected = sorted(range(occurrence + 1), key=lambda i: (
            int(engine.hashlib.sha256(f"13|calibration|{i}".encode()).hexdigest(), 16), i,
        ))[:4]
        assert set(calibration.ids) == set(expected)
    assert unchanged > 0
    assert len(calibration.events) < 60
    # Check q90 against the actual reservoir pairs, not a latest-donor window.
    ids = np.array(calibration.ids)
    cosine = np.clip(vectors[ids] @ vectors[ids].T, -1, 1)
    delta = np.abs((ids % 2)[:, None] - (ids % 2)[None, :])
    slopes = delta / np.maximum(np.arccos(cosine) / np.pi, .01)
    assert calibration.estimate.value == pytest.approx(np.quantile(slopes[np.triu_indices(4, 1)], .9))


def test_sparse_calibration_fallback_and_shared_bound_helper(monkeypatch):
    calls = []
    original = engine.calculate_conditional_geodesic_bounds

    def record(point, distance, estimate):
        calls.append(estimate)
        return original(point, distance, estimate)

    monkeypatch.setattr(engine, "calculate_conditional_geodesic_bounds", record)
    vectors = np.array([[1., 0], [0, 1], [-1, 0]])
    evidence = engine.evaluate_replay(vectors, np.array([1, 0, 1]), np.arange(3), np.arange(3), np.array([0]))
    assert len(calls) == 2
    assert all(estimate.value == 1 for estimate in calls)
    assert evidence["calibration_fallback"].all()
    assert evidence["calibration_pair_count"].tolist() == [0, 0]
    np.testing.assert_allclose(evidence["lower"], [0.5, 0.])
    metrics = engine.cell_metrics(evidence, np.array([1]))
    assert metrics["counts"]["calibration_fallback_eligible"] == 2
    assert metrics["counts"]["envelope_coverage"] == 1


def test_small_full_reservoir_matches_all_earlier_pair_calibration():
    vectors, labels = fixture_inputs(14)
    vectors = normalized_prefix(vectors, 8)
    sources = np.array([0, 1, 2, 0, 4, 5, 1, 7, 8, 9, 10, 11, 12, 13])
    selected = np.array([0, 2, 3, 5, 7, 10])
    evidence = engine.evaluate_replay(vectors, labels, sources, np.arange(14), selected)
    unit = vectors.astype(np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    for offset, position in enumerate(evidence["position"]):
        earlier = selected[selected < position]
        if len(earlier) < 2:
            expected = 1.
        else:
            ids = sources[earlier]
            distance = np.arccos(np.clip(unit[ids] @ unit[ids].T, -1, 1)) / np.pi
            distance[ids[:, None] == ids[None, :]] = 0
            outcomes = labels[ids].astype(float)
            slope = np.abs(outcomes[:, None] - outcomes[None, :]) / np.maximum(distance, .01)
            expected = np.quantile(slope[np.triu_indices(len(ids), 1)], .9)
        assert evidence["lipschitz"][offset] == pytest.approx(expected)
        assert evidence["lower"][offset] == pytest.approx(
            max(0, evidence["score"][offset] - expected * evidence["weighted_distance"][offset]),
        )
        assert evidence["calibration_pair_count"][offset] == len(earlier) * (len(earlier) - 1) // 2
    # Lower-threshold positives are a subset on the same eligible population.
    for threshold in np.r_[np.inf, evidence["score"], evidence["lower"], -np.inf]:
        assert np.all(~(evidence["lower"] >= threshold) | (evidence["score"] >= threshold))


def test_arrival_positions_do_not_overflow_int16_and_donors_are_not_warm_started():
    n = 33_000
    evidence = engine.evaluate_replay(
        np.array([[1., 0]]), np.array([1]), np.zeros(n, dtype=int),
        np.arange(n), np.array([n - 2]),
    )
    assert evidence["position"][-1] == n - 1
    assert evidence["max_donor_position"][-1] == n - 2
    assert evidence["max_calibration_position"][-1] == n - 2
    assert evidence["occurrence_id"][-1] == n - 1
    assert evidence["novel_source"].sum() == n - 2
    assert np.all(evidence["score"][:-1] == .5)
    assert evidence["score"][-1] == 1


def test_metrics_use_exact_ge_threshold_auc_and_never_observed_primary():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([.1, .4, .35, .8])
    metrics, curve = engine._quality(labels, scores)
    assert metrics["auc"] == pytest.approx(.75)
    assert curve["auc"] == .75
    assert curve["exact_point_count"] == 5
    tied, _ = engine._quality(np.array([0, 1]), np.array([.5, .5]))
    assert tied["accuracy"] == .5 and tied["recall"] == 1 and tied["precision"] == .5
    assert tied["auc"] == .5
    empty, curve = engine._quality(np.array([]), np.array([]))
    assert empty["mae"] is None and empty["auc"] is None
    assert all(x is None for x in curve["tpr"])
    one_class, _ = engine._quality(np.array([0, 0]), np.array([.1, .5]))
    assert one_class["auc"] is None and one_class["recall"] is None


def test_direct_labels_only_affect_combined_secondary_metrics():
    vectors, labels = fixture_inputs(8)
    evidence = engine.evaluate_replay(vectors, labels, np.arange(8), np.arange(8), np.array([1, 3, 6]))
    first = engine.cell_metrics(evidence, np.zeros(3))
    second = engine.cell_metrics(evidence, np.ones(3))
    for population in engine.POPULATIONS:
        if population != "combined_secondary":
            assert first[population] == second[population]
    assert first["combined_secondary"] != second["combined_secondary"]


def test_end_to_end_artifacts_pairing_summary_roc_and_resume(tmp_path, monkeypatch):
    result = run_small(tmp_path)
    assert result["version"] == engine.VERSION and result["status"] == "completed"
    assert len(result["rows"]) == 16
    assert len(result["summaries"]) == 8
    assert "NOT strictly online" in result["protocol"]["online_boundary"]
    assert "bounded donor reservoir" in result["protocol"]["lipschitz"]["deviation_from_prior"]
    prereg = json.loads((tmp_path / "preregistration.json").read_text())
    assert prereg["binding"]["protocol"] == result["protocol"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["completed_cells"] == 16
    memberships = {}
    for row in result["rows"]:
        path = tmp_path / row["evidence"]
        assert engine.sha256_file(path) == row["evidence_sha256"]
        with np.load(path, allow_pickle=False) as evidence:
            assert evidence["score"].dtype == np.float64
            assert row["all_unselected"]["n"] == 24 - row["budget"]
            eligible = np.isfinite(evidence["lower"])
            for name, field in (("point", "score"), ("lower", "lower")):
                roc = engine.exact_roc(evidence["label"][eligible], evidence[field][eligible])
                assert row["roc"][name]["auc"] == roc["auc"]
            assert row["eligible_point"]["n"] == row["eligible_lower"]["n"] == int(eligible.sum())
            key = row["seed"], row["schedule"], row["dimension"]
            memberships.setdefault(key, []).append(set(evidence["selected_occurrence_id"]))
        paired = [r for r in result["rows"] if (r["seed"], r["schedule"]) == (row["seed"], row["schedule"])]
        assert all(r["replay_hashes"] == row["replay_hashes"] for r in paired)
        if row["dimension"] == 1536:
            assert row["paired_delta_native"]["all_unselected"]["mae"] == 0
    for smaller, larger in memberships.values():
        assert smaller < larger
    for summary in result["summaries"]:
        cells = [r for r in result["rows"] if all(r[k] == summary[k] for k in ("dimension", "schedule", "rate"))]
        values = [r["all_unselected"]["mae"] for r in cells]
        assert summary["all_unselected"]["mae"]["mean"] == pytest.approx(np.mean(values))
        assert summary["all_unselected"]["mae"]["q025"] == pytest.approx(np.quantile(values, .025))
        assert summary["all_unselected"]["mae"]["q975"] == pytest.approx(np.quantile(values, .975))
    monkeypatch.setattr(engine, "evaluate_replay", lambda *a, **k: pytest.fail("completed cells recomputed"))
    monkeypatch.setattr(engine, "rank_membership", lambda *a, **k: pytest.fail("membership recomputed"))
    assert run_small(tmp_path, resume=True) == result
    with pytest.raises(FileExistsError):
        run_small(tmp_path)


def test_readonly_native_mmap_and_parent_profile_contract(tmp_path):
    vectors, labels = fixture_inputs(10)
    path = tmp_path / "float32.npy"
    np.save(path, vectors, allow_pickle=False)
    digest = engine.sha256_file(path)
    mapped = np.load(path, allow_pickle=False, mmap_mode="r")
    profile = {
        "dataset_id": "offline-fixture", "model": "text-embedding-3-small",
        "sessions": 10, "agents": 1, "source_sha256": "fixture-source",
        "input_manifest_sha256": "fixture-manifest", "label_source": "fixture labels",
        "representation_policy": "random fixture vectors; not measured IMDb embeddings",
    }
    result = engine.run_experiment(
        mapped, labels, tmp_path / "run", profile=profile, dimensions=(1536, 8),
        repetitions=1, rates=(.2,), schedules=("uniformly_random",),
    )
    assert result["dataset"] == profile
    assert result["protocol"]["source_count"] == 10
    assert result["protocol"]["occurrences_per_replay"] == 10
    assert engine.sha256_file(path) == digest
    assert not mapped.flags.writeable
    for row in result["rows"]:
        for kind, population in (("point", "eligible_point"), ("lower", "eligible_lower")):
            curve = row["roc"][kind]
            assert {"fpr", "tpr", "auc"} <= curve.keys()
            assert len(curve["fpr"]) == len(curve["tpr"]) == 101
            assert curve["auc"] == row[population]["auc"]


def test_membership_reuses_arm2_and_is_label_blind(tmp_path):
    vectors, labels = fixture_inputs()
    one = run_small(tmp_path / "one", repetitions=1, rates=(.2,), schedules=("bursty",))
    arguments = dict(
        profile=one["dataset"], dimensions=(1536, 8), repetitions=1, rates=(.2,),
        schedules=("bursty",), target_block_size=5, donor_block_size=3, calibration_reservoir_size=4,
    )
    two = engine.run_experiment(vectors, 1 - labels, tmp_path / "two", **arguments)
    for first, second in zip(one["rows"], two["rows"], strict=True):
        assert first["membership_sha256"] == second["membership_sha256"]
        assert first["ranking_sha256"] == second["ranking_sha256"]
    replay = engine.bootstrap_replay(24, 13, "bursty")
    ids = tuple(f"seed-13:occ-{i:012d}" for i in range(24))
    expected, _ = rank_membership(
        ids, ("imdb",) * 24, (("imdb-review",),) * 24,
        normalized_prefix(vectors, 8)[replay["source_id"]], replay["order"], replay["timestamps"], 13,
    )
    with np.load(tmp_path / "one" / "memberships" / "s13-bursty-d8.npz") as arrays:
        np.testing.assert_array_equal(expected, arrays["ranking"])
        assert arrays.files == ["ranking"]  # no outcomes in membership provenance


def test_preregistration_precedes_cells_and_partial_resume(tmp_path, monkeypatch):
    original = engine.evaluate_replay
    calls = []

    def interrupt(*args, **kwargs):
        assert (tmp_path / "preregistration.json").exists()
        assert not (tmp_path / "aggregate.json").exists()
        assert not (tmp_path / "manifest.json").exists()
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "evaluate_replay", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        run_small(tmp_path, repetitions=1)
    committed = list((tmp_path / "cells").glob("*.json"))
    assert len(committed) == 1
    before = committed[0].read_bytes()
    monkeypatch.setattr(engine, "evaluate_replay", original)
    completed = run_small(tmp_path, repetitions=1, resume=True)
    assert len(completed["rows"]) == 8
    assert committed[0].read_bytes() == before


@pytest.mark.parametrize("artifact", ["cell_evidence", "cell_metric", "rank_evidence", "replay_evidence"])
def test_partial_resume_rejects_committed_checkpoint_tampering(tmp_path, monkeypatch, artifact):
    original = engine.evaluate_replay
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "evaluate_replay", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_small(tmp_path, repetitions=1)
    assert not (tmp_path / "manifest.json").exists()
    if artifact == "cell_metric":
        path = next((tmp_path / "cells").glob("*.json"))
        record = json.loads(path.read_text())
        record["row"]["all_unselected"]["mae"] = -1
        path.write_text(json.dumps(record), encoding="utf-8")
    else:
        directory = {"cell_evidence": "cells", "rank_evidence": "memberships", "replay_evidence": "replays"}[artifact]
        path = next((tmp_path / directory).glob("*.npz"))
        path.write_bytes(path.read_bytes() + b"tamper")
    monkeypatch.setattr(engine, "evaluate_replay", original)
    with pytest.raises(ValueError, match="hash"):
        run_small(tmp_path, repetitions=1, resume=True)
    assert not (tmp_path / "aggregate.json").exists()


@pytest.mark.parametrize("artifact", ["cells_npz", "cells_json", "replay", "membership", "aggregate", "preregistration", "manifest"])
def test_completed_resume_rejects_tampering(tmp_path, artifact):
    run_small(tmp_path, repetitions=1, dimensions=(1536,), rates=(.2,), schedules=("bursty",))
    paths = {
        "cells_npz": next((tmp_path / "cells").glob("*.npz")),
        "cells_json": next((tmp_path / "cells").glob("*.json")),
        "replay": next((tmp_path / "replays").glob("*.npz")),
        "membership": next((tmp_path / "memberships").glob("*.npz")),
        "aggregate": tmp_path / "aggregate.json",
        "preregistration": tmp_path / "preregistration.json",
        "manifest": tmp_path / "manifest.json",
    }
    path = paths[artifact]
    path.write_bytes(path.read_bytes() + b"tampering")
    with pytest.raises(ValueError, match="hash|format|resume"):
        run_small(tmp_path, repetitions=1, dimensions=(1536,), rates=(.2,), schedules=("bursty",), resume=True)


@pytest.mark.parametrize("change", ["vectors", "labels", "profile", "dimensions", "code", "runtime"])
def test_resume_binds_inputs_profile_config_code_and_runtime(tmp_path, monkeypatch, change):
    run_small(tmp_path, repetitions=1)
    if change == "code":
        monkeypatch.setattr(engine, "_method_hashes", lambda: {"different.py": "changed"})
    elif change == "runtime":
        monkeypatch.setattr(engine, "_runtime", lambda: {"python": "incompatible"})
    vectors, labels = fixture_inputs()
    if change == "vectors":
        vectors[0, 0] += 1
    if change == "labels":
        labels[0] = 1 - labels[0]
    if change in ("vectors", "labels"):
        monkeypatch.setattr(sys.modules[__name__], "fixture_inputs", lambda: (vectors, labels))
    overrides = {"resume": True, "repetitions": 1}
    if change == "profile":
        overrides["profile"] = {"dataset_id": "changed"}
    if change == "dimensions":
        overrides["dimensions"] = (1536,)
    with pytest.raises(ValueError, match="incompatible resume"):
        run_small(tmp_path, **overrides)


@pytest.mark.parametrize("overrides", [
    {"dimensions": (8,)}, {"repetitions": 0}, {"rates": (0,)}, {"rates": (.2, .2)},
    {"schedules": ("future",)}, {"base_seed": -1}, {"target_block_size": 0},
    {"donor_block_size": False}, {"calibration_reservoir_size": 129},
    {"profile": {"embedding_model_id": "surrogate"}},
    {"profile": {"model": "surrogate"}},
])
def test_invalid_configuration_writes_no_protocol(tmp_path, overrides):
    with pytest.raises(ValueError):
        run_small(tmp_path, **overrides)
    assert not (tmp_path / "preregistration.json").exists()


@pytest.mark.parametrize("bad_input", ["width", "nan", "zero_prefix", "label_shape", "nonbinary"])
def test_invalid_inputs_rejected_before_registration(tmp_path, bad_input):
    vectors, labels = fixture_inputs()
    if bad_input == "width":
        vectors = vectors[:, :32]
    elif bad_input == "nan":
        vectors[0, 0] = np.nan
    elif bad_input == "zero_prefix":
        vectors[0, :8] = 0
    elif bad_input == "label_shape":
        labels = labels[:-1]
    elif bad_input == "nonbinary":
        labels[0] = 2
    with pytest.raises(ValueError):
        engine.run_experiment(vectors, labels, tmp_path, profile={"dataset_id": "fixture"})
    assert not (tmp_path / "preregistration.json").exists()
