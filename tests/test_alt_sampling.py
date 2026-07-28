import pytest

from trace_sampling_alt import (
    SamplePolicy,
    allocate_strata,
    cochran_sample_size,
    plan_sample,
)


def test_reference_characterization_plans_and_allocates_exactly():
    plan = plan_sample(population=300, margin=0.10, confidence=0.95)

    assert cochran_sample_size(0.10, 0.95) == 97
    assert plan.recommended == 74
    assert plan.selected == 74
    assert plan.probability_selected == 74
    assert plan.diversity_selected == 0
    assert [(row.key, row.selected) for row in allocate_strata(
        {"large": 200, "small": 100}, plan.selected
    )] == [("large", 49), ("small", 25)]


def test_capacity_shortfall_is_explicit_and_total_budget_preserved():
    plan = plan_sample(population=300, capacity=40)

    assert plan.recommended == 74
    assert plan.selected == 40
    assert plan.probability_selected == 40
    assert plan.diversity_selected == 0
    assert plan.precision_status == "capacity_limited_precision_shortfall"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"margin": 0}, "margin"),
        ({"confidence": 1}, "confidence"),
        ({"diversity_fraction": -0.1}, "diversity_fraction"),
    ],
)
def test_sample_policy_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SamplePolicy(**kwargs)