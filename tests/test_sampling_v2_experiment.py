from __future__ import annotations

from pathlib import Path

from sampling_comparison.v2_experiment import (
    assign_population_quadrants,
    load_combined_dataset,
    run_paired_repeated_comparison,
    run_throughput_grid_experiment,
    run_v2_experiment_bundle,
    score_selection,
    select_ids_for_method,
    slice_dataset,
    with_permuted_labels,
)


def test_load_combined_dataset_integrity_counts_and_provenance():
    data = load_combined_dataset()

    assert len(data.units) == 2800
    assert len(data.unit_ids) == 2800
    assert len(set(data.unit_ids)) == 2800
    assert len(data.labels_by_unit) == 2800
    assert len(data.scoped_identities) == 105
    assert set(data.source_paths) == {"historical_300", "dense_2500"}

    for unit_id in data.unit_ids[:20]:
        meta = data.metadata_by_unit[unit_id]
        assert meta["corpus_id"] in {"historical_300", "dense_2500"}
        assert unit_id.startswith(meta["corpus_id"] + ":")
        assert meta["original_unit_id"] == data.original_unit_id_by_unit[unit_id]


def test_label_permutation_invariance_for_all_selectors_on_small_slice():
    full = load_combined_dataset()
    data = slice_dataset(full, limit=240)
    permuted = with_permuted_labels(data, seed=101)

    methods = (
        ("random_sampling_stratified", 20),
        ("adaptive_minhash_32x4", 20),
        ("adaptive_embedding_fullsession", 20),
    )
    for method, budget in methods:
        left = select_ids_for_method(
            data,
            method=method,
            budget_pct=budget,
            repetition_seed=17,
        )
        right = select_ids_for_method(
            permuted,
            method=method,
            budget_pct=budget,
            repetition_seed=17,
        )
        assert left == right


def test_scoring_includes_per_corpus_and_per_agent_error_metrics():
    full = load_combined_dataset()
    data = slice_dataset(full, limit=240)

    selected_ids = select_ids_for_method(
        data,
        method="random_sampling_stratified",
        budget_pct=20,
        repetition_seed=13,
    )
    scored = score_selection(
        data,
        method="random_sampling_stratified",
        budget_pct=20,
        repetition=0,
        selected_ids=selected_ids,
    )

    assert "selected_pass_rate" in scored
    assert "census_pass_rate" in scored
    assert "absolute_error" in scored
    assert "fraction_saved" in scored
    assert "concept_coverage" in scored

    assert scored["per_corpus"]
    for corpus in scored["per_corpus"].values():
        assert set(corpus).issuperset(
            {
                "population_count",
                "sampled_count",
                "keep_rate",
                "selected_pass_rate",
                "census_pass_rate",
                "absolute_error",
                "fraction_saved",
                "concept_coverage",
            }
        )

    assert scored["per_agent"]
    first_agent = next(iter(scored["per_agent"].values()))
    assert set(first_agent).issuperset(
        {
            "population_count",
            "sampled_count",
            "keep_rate",
            "selected_pass_rate",
            "census_pass_rate",
            "absolute_error",
        }
    )


def test_run_paired_repeated_comparison_shape_and_pairing_behavior():
    full = load_combined_dataset()
    data = slice_dataset(full, limit=300)

    out = run_paired_repeated_comparison(
        data,
        budget_pcts=(5, 10, 20, 30, 50),
        repetitions=2,
        seed=13,
    )

    assert out["version"] == "sampling-v2-outcome-v1"
    assert out["population_count"] == 300
    assert out["census_baseline"]["method"] == "census"
    assert out["census_baseline"]["selected_count"] == 300

    runs = out["runs"]
    assert len(runs) == 2 * 5 * 3
    by_key = {(row["method"], row["budget_pct"], row["repetition"]): row for row in runs}
    assert len(by_key) == len(runs)

    pair_manifest = out["pairing"]["paired_order_manifest"]
    assert len(pair_manifest) == 2
    assert pair_manifest[0]["order_hash"] != pair_manifest[1]["order_hash"]

    for repetition in (0, 1):
        manifest_row = pair_manifest[repetition]
        for budget in (5, 10, 20, 30, 50):
            random_row = by_key[("random_sampling_stratified", budget, repetition)]
            minhash_row = by_key[("adaptive_minhash_32x4", budget, repetition)]
            embed_row = by_key[("adaptive_embedding_fullsession", budget, repetition)]

            expected_target = int(round(300 * (budget / 100.0)))
            assert random_row["selected_count"] == expected_target
            assert minhash_row["selected_count"] <= expected_target
            assert embed_row["selected_count"] <= expected_target

            assert minhash_row["order_hash"] == manifest_row["order_hash"]
            assert embed_row["order_hash"] == manifest_row["order_hash"]
            assert random_row["order_hash"] == "temporal"


def test_quadrant_assignment_total_equals_2800():
    data = load_combined_dataset()
    quadrants = assign_population_quadrants(data)

    assert quadrants["counts"]["total_units"] == 2800
    total = sum(row["unit_count"] for row in quadrants["quadrant_summary"].values())
    assert total == 2800


def test_throughput_matrix_shape_on_small_slice():
    full = load_combined_dataset()
    data = slice_dataset(full, limit=360)

    out = run_throughput_grid_experiment(
        data,
        budgets=(15, 30),
        arrival_rates=(0.25, 1.0),
        eval_throughputs=(0.25, 1.0),
        replay_count=2,
        seed=31,
    )

    expected = 2 * 2 * 2 * 3 * 2
    assert len(out["runs"]) == expected
    assert len(out["aggregate_grid"]) == 2 * 2 * 2 * 3


def test_bundle_writes_expected_artifacts(tmp_path: Path):
    full = load_combined_dataset()
    data = slice_dataset(full, limit=320)
    out_dir = tmp_path / "v2"

    result = run_v2_experiment_bundle(
        data=data,
        enforce_integrity_counts=False,
        budget_pcts=(20,),
        outcome_repetitions=1,
        quadrant_replays=1,
        throughput_replays=1,
        seed=7,
        output_dir=out_dir,
    )

    paths = result["output_paths"]
    assert paths is not None
    expected = {
        "aggregate",
        "runs_jsonl",
        "quadrant",
        "throughput",
        "corpus_audit",
        "selected_membership_20pct",
        "external_eval_snapshots_manifest",
        "production_storage_manifest",
        "manifest",
    }
    assert set(paths) == expected
    for path in paths.values():
        assert Path(path).exists()
