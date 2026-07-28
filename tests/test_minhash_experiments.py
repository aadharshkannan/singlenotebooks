from __future__ import annotations

from minhash_sampling import run_minhash_experiment, sweep_minhash_experiments


def test_experiment_is_deterministic_and_json_friendly() -> None:
    a = run_minhash_experiment(ngram_size=3, permutations=128, threshold=0.5)
    b = run_minhash_experiment(ngram_size=3, permutations=128, threshold=0.5)

    left = a.to_dict()
    right = b.to_dict()
    for arm in ("exact", "minhash"):
        left[arm].pop("decision_latency_ms_p50")
        left[arm].pop("decision_latency_ms_p95")
        right[arm].pop("decision_latency_ms_p50")
        right[arm].pop("decision_latency_ms_p95")
    assert left == right
    assert isinstance(a.to_dict()["telemetry"]["minhash_index"]["live_clusters"], int)


def test_minhash_exposes_purity_separation_and_fragmentation_tradeoff() -> None:
    result = run_minhash_experiment(ngram_size=3, permutations=128, threshold=0.5)

    # Deterministic stream is constructed so exact signatures collide across latent concepts,
    # while lexical MinHash should separate near-duplicate task families at similar keep volume.
    assert abs(result.minhash.keep_count - result.exact.keep_count) <= 6
    assert result.minhash.concept_coverage == result.exact.concept_coverage == 1.0
    assert result.separation_gain > 0.0
    assert result.purity_gain > 0.0
    assert result.minhash.clusters_per_concept_mean > result.exact.clusters_per_concept_mean
    assert result.minhash.ari < result.exact.ari
    assert result.telemetry["minhash_index"]["purges"] > 0
    assert result.telemetry["minhash_index"]["fallbacks"] > 0


def test_signature_calibration_mae_is_reasonable() -> None:
    result = run_minhash_experiment(ngram_size=3, permutations=256, threshold=0.5)
    assert result.signature_jaccard_mae <= 0.10


def test_sweep_runs_expected_grid_size() -> None:
    results = sweep_minhash_experiments()
    assert len(results) == 18
