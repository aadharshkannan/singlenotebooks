from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from random_sampling import AgentKey, EvaluationUnit, EvaluationWindow, SamplePolicy, SamplingEngine, Turn


def _unit(index: int, *, tenant: str = "tenant-a", agent: str = "agent-a", small: bool = False) -> EvaluationUnit:
    turns = (Turn(f"u{index}", f"a{index}"),) if small else (
        Turn(f"u{index}", f"a{index}"),
        Turn(f"u{index}-2", f"a{index}-2"),
    )
    return EvaluationUnit(
        tenant_id=tenant,
        agent_id=agent,
        conversation_id=f"conv-{index:04d}",
        session_id=f"session-{index:04d}",
        channel="copilot" if small else "teams",
        source_trace_ids=(f"trace-{index:04d}",),
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        had_error=False,
        turns=turns,
        tool_calls=(),
    )


def _population() -> list[EvaluationUnit]:
    return [_unit(i) for i in range(200)] + [_unit(i, small=True) for i in range(200, 300)]


def test_random_stratified_selection_is_exact_and_weighted():
    sample = SamplingEngine().sample(_population(), SamplePolicy(seed=13)).agents[0]

    assert sample.plan.selected == 74
    assert len(sample.units) == 74
    assert [(row.key, row.selected) for row in sample.strata] == [
        ("1|copilot", 25),
        ("2-3|teams", 49),
    ]
    assert all(row.estimand_eligible for row in sample.units)
    assert all(row.inclusion_probability is not None for row in sample.units)
    assert all(row.sampling_weight is not None for row in sample.units)


def test_reversed_input_replays_identical_manifest():
    population = _population()
    engine = SamplingEngine()
    policy = SamplePolicy(seed=17)

    left = engine.sample(population, policy)
    right = engine.sample(reversed(population), policy)

    assert left.run_id == right.run_id
    assert [row.unit.unit_id for row in left.agents[0].units] == [
        row.unit.unit_id for row in right.agents[0].units
    ]


def test_low_volume_agent_is_a_census():
    sample = SamplingEngine().sample([_unit(i) for i in range(10)], SamplePolicy()).agents[0]

    assert sample.plan.census is True
    assert len(sample.units) == 10
    assert all(row.inclusion_probability == 1.0 for row in sample.units)


def test_agent_capacity_caps_total_random_sample():
    agent = AgentKey("tenant-a", "agent-a")
    sample = SamplingEngine().sample(
        _population(),
        SamplePolicy(seed=23),
        capacities={agent: 40},
        agents=[agent],
    ).agents[0]

    assert sample.plan.recommended == 74
    assert sample.plan.selected == 40
    assert len(sample.units) == 40


def test_tenant_capacity_is_conserved_and_surplus_unused():
    first = _population()
    second = [_unit(1000 + i, agent="agent-b") for i in range(100)]
    engine = SamplingEngine()

    limited = engine.sample(first + second, SamplePolicy(seed=29), tenant_capacities={"tenant-a": 60})
    assert sum(len(sample.units) for sample in limited.agents) == 60
    assert limited.tenant_capacities[0].selected == 60

    surplus = engine.sample(first, SamplePolicy(), tenant_capacities={"tenant-a": 100})
    assert surplus.tenant_capacities[0].selected == 74
    assert surplus.tenant_capacities[0].unused == 26


def test_capacity_modes_are_mutually_exclusive():
    agent = AgentKey("tenant-a", "agent-a")
    with pytest.raises(ValueError, match="mutually exclusive"):
        SamplingEngine().sample(
            _population(),
            SamplePolicy(),
            capacities={agent: 20},
            tenant_capacities={"tenant-a": 20},
        )


def test_window_changes_run_id():
    end = datetime(2025, 1, 2, tzinfo=timezone.utc)
    first = EvaluationWindow(end - timedelta(hours=24), end)
    second = EvaluationWindow(end - timedelta(hours=12), end)
    engine = SamplingEngine()
    population = _population()

    assert engine.sample(population, SamplePolicy(), evaluation_window=first).run_id != engine.sample(
        population, SamplePolicy(), evaluation_window=second
    ).run_id
