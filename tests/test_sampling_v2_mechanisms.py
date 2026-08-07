from __future__ import annotations

from sampling_comparison.v2_experiment import (
    LAST_SELECTION_MECHANISMS,
    load_combined_dataset,
    select_ids_for_method,
    slice_dataset,
    with_permuted_labels,
)


def _slice() -> object:
    full = load_combined_dataset()
    return slice_dataset(full, limit=260)


def test_v2_arms_invoke_owning_production_packages() -> None:
    data = _slice()

    select_ids_for_method(data, method="random_sampling_stratified", budget_pct=20, repetition_seed=13)
    select_ids_for_method(data, method="adaptive_minhash_32x4", budget_pct=20, repetition_seed=13)
    select_ids_for_method(data, method="adaptive_embedding_fullsession", budget_pct=20, repetition_seed=13)

    assert LAST_SELECTION_MECHANISMS["random_sampling_stratified"]["owner"] == "random_sampling"
    assert LAST_SELECTION_MECHANISMS["adaptive_minhash_32x4"]["owner"] == "minhash_sampling"
    assert LAST_SELECTION_MECHANISMS["adaptive_embedding_fullsession"]["owner"] == "trace_sampling"


def test_minhash_and_embedding_selections_differ_on_meaningful_slice() -> None:
    data = _slice()
    minhash = select_ids_for_method(data, method="adaptive_minhash_32x4", budget_pct=20, repetition_seed=17)
    embed = select_ids_for_method(data, method="adaptive_embedding_fullsession", budget_pct=20, repetition_seed=17)
    assert minhash != embed


def test_label_permutation_does_not_change_selection() -> None:
    data = _slice()
    permuted = with_permuted_labels(data, seed=77)
    for method in ("random_sampling_stratified", "adaptive_minhash_32x4", "adaptive_embedding_fullsession"):
        left = select_ids_for_method(data, method=method, budget_pct=20, repetition_seed=19)
        right = select_ids_for_method(permuted, method=method, budget_pct=20, repetition_seed=19)
        assert left == right


def test_fresh_state_isolation_repeated_run_same_seed() -> None:
    data = _slice()
    first = select_ids_for_method(data, method="adaptive_minhash_32x4", budget_pct=20, repetition_seed=23)
    second = select_ids_for_method(data, method="adaptive_minhash_32x4", budget_pct=20, repetition_seed=23)
    assert first == second


def test_adaptive_methods_reuse_precompute_without_posthoc_fill() -> None:
    data = _slice()
    for method in ("adaptive_minhash_32x4", "adaptive_embedding_fullsession"):
        selected = select_ids_for_method(data, method=method, budget_pct=20, repetition_seed=31)
        mechanism = LAST_SELECTION_MECHANISMS[method]
        assert len(selected) <= round(len(data.unit_ids) * 0.20)
        assert mechanism["trim_fill"] == "none"
        assert mechanism["precomputed_records"] == len(data.unit_ids)


def test_no_label_fields_enter_serialized_representations() -> None:
    data = _slice()
    select_ids_for_method(data, method="adaptive_embedding_fullsession", budget_pct=20, repetition_seed=29)
    mechanism = LAST_SELECTION_MECHANISMS["adaptive_embedding_fullsession"]
    # The mechanism payload intentionally excludes expected-label values.
    assert "label" not in "|".join(sorted(mechanism.keys())).lower()