from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trace_sampling_alt import (
    AgentKey,
    EvaluationUnit,
    EvaluationWindow,
    SamplePolicy,
    SamplingEngine,
    ToolCall,
    Turn,
    build_minhash_signature,
)
from trace_sampling_alt.sampling import _farthest_first


def _make_unit(
    idx: int,
    *,
    tenant: str = "tenant-a",
    agent: str = "agent-a",
    stratum: str = "large",
    had_error: bool = False,
) -> EvaluationUnit:
    if stratum == "large":
        turns = (
            Turn(user_text=f"u{idx}", assistant_text=f"a{idx}"),
            Turn(user_text=f"u{idx}-2", assistant_text=f"a{idx}-2"),
        )
        channel = "teams"
    else:
        turns = (
            Turn(user_text=f"u{idx}", assistant_text=f"a{idx}"),
            Turn(user_text=f"u{idx}-2", assistant_text=f"a{idx}-2"),
            Turn(user_text=f"u{idx}-3", assistant_text=f"a{idx}-3"),
            Turn(user_text=f"u{idx}-4", assistant_text=f"a{idx}-4"),
        )
        channel = "copilot"

    return EvaluationUnit(
        tenant_id=tenant,
        agent_id=agent,
        conversation_id=f"conv-{idx:04d}",
        session_id=f"sess-{idx:04d}",
        channel=channel,
        source_trace_ids=(f"trace-{idx:04d}",),
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        had_error=had_error,
        turns=turns,
        tool_calls=(),
    )


def _make_signature_unit(
    *,
    user_text: str = "where is order",
    assistant_text: str = "order is pending",
    tool_name: str = "lookup_order",
    tool_input: str = "id=42",
    tool_output: str = "status=pending",
) -> EvaluationUnit:
    return EvaluationUnit(
        tenant_id="tenant-a",
        agent_id="agent-a",
        conversation_id="conv-signature",
        session_id="sess-signature",
        channel="teams",
        source_trace_ids=("trace-signature",),
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        had_error=False,
        turns=(Turn(user_text=user_text, assistant_text=assistant_text),),
        tool_calls=(
            ToolCall(
                name=tool_name,
                input_text=tool_input,
                output_text=tool_output,
            ),
        ),
    )


def _make_population_200_100() -> list[EvaluationUnit]:
    units: list[EvaluationUnit] = []
    for i in range(200):
        units.append(_make_unit(i, stratum="large", had_error=(i % 7 == 0)))
    for i in range(200, 300):
        units.append(_make_unit(i, stratum="small", had_error=(i % 5 == 0)))
    return units


def test_reference_allocation_200_100_to_74_is_49_25():
    units = _make_population_200_100()
    engine = SamplingEngine()
    policy = SamplePolicy(margin=0.10, confidence=0.95, diversity_enabled=False, seed=13)

    batch = engine.sample(units, policy)
    assert len(batch.agents) == 1
    sample = batch.agents[0]

    assert sample.plan.selected == 74
    assert [(row.key, row.selected) for row in sample.strata] == [
        ("2-3|teams", 49),
        ("4-7|copilot", 25),
    ]


def test_reversed_input_is_identical_manifest_and_run_id():
    units = _make_population_200_100()
    engine = SamplingEngine()
    policy = SamplePolicy(seed=17, diversity_enabled=True, diversity_fraction=0.20)

    batch_a = engine.sample(units, policy)
    batch_b = engine.sample(list(reversed(units)), policy)

    assert batch_a.run_id == batch_b.run_id
    a_sample = batch_a.agents[0]
    b_sample = batch_b.agents[0]
    assert [u.unit.unit_id for u in a_sample.core] == [
        u.unit.unit_id for u in b_sample.core
    ]
    assert [u.unit.unit_id for u in a_sample.diversity] == [
        u.unit.unit_id for u in b_sample.diversity
    ]


def test_core_and_diversity_disjoint_with_expected_diversity_metadata():
    units = _make_population_200_100()
    engine = SamplingEngine()
    policy = SamplePolicy(seed=19, diversity_enabled=True, diversity_fraction=0.20)

    sample = engine.sample(units, policy).agents[0]
    core_ids = {u.unit.unit_id for u in sample.core}
    diversity_ids = {u.unit.unit_id for u in sample.diversity}

    assert core_ids.isdisjoint(diversity_ids)
    for row in sample.diversity:
        assert row.sample_kind == "diversity"
        assert row.estimand_eligible is False
        assert row.inclusion_probability is None
        assert row.sampling_weight is None
        assert row.selection_reason == "diversity_minhash_farthest_first"


def test_census_selects_all_core_and_diversity_empty_even_when_enabled():
    units = [_make_unit(i, stratum="large") for i in range(10)]
    engine = SamplingEngine()
    policy = SamplePolicy(diversity_enabled=True, diversity_fraction=0.50)

    sample = engine.sample(units, policy).agents[0]
    assert sample.plan.census is True
    assert sample.plan.probability_selected == len(units)
    assert sample.plan.diversity_selected == 0
    assert len(sample.core) == len(units)
    assert sample.diversity == ()


def test_enabled_and_disabled_policy_splits_are_exact_for_n300():
    units = _make_population_200_100()
    engine = SamplingEngine()

    enabled = SamplePolicy(seed=31, diversity_enabled=True, diversity_fraction=0.20)
    disabled = SamplePolicy(seed=31, diversity_enabled=False, diversity_fraction=0.20)

    enabled_sample = engine.sample(units, enabled).agents[0]
    assert enabled_sample.plan.selected == 74
    assert enabled_sample.plan.probability_selected == 59
    assert enabled_sample.plan.diversity_selected == 15
    assert len(enabled_sample.core) == 59
    assert len(enabled_sample.diversity) == 15
    assert len(enabled_sample.core) + len(enabled_sample.diversity) == 74

    disabled_sample = engine.sample(units, disabled).agents[0]
    assert disabled_sample.plan.selected == 74
    assert disabled_sample.plan.probability_selected == 74
    assert disabled_sample.plan.diversity_selected == 0
    assert len(disabled_sample.core) == 74
    assert len(disabled_sample.diversity) == 0


def test_capacity_shortfall_and_agent_filtering():
    a_units = _make_population_200_100()
    b_units = [_make_unit(i + 1000, tenant="tenant-b", agent="agent-b", stratum="large") for i in range(40)]

    units = a_units + b_units
    engine = SamplingEngine()
    policy = SamplePolicy(diversity_enabled=True, diversity_fraction=0.20, seed=23)

    cap_agent_a = AgentKey(tenant_id="tenant-a", agent_id="agent-a")
    batch = engine.sample(
        units,
        policy,
        capacities={cap_agent_a: 40},
        agents=[cap_agent_a],
    )

    assert len(batch.agents) == 1
    sample = batch.agents[0]
    assert sample.agent == cap_agent_a
    assert sample.plan.recommended == 74
    assert sample.plan.selected == 40
    assert sample.plan.probability_selected == 32
    assert sample.plan.diversity_selected == 8
    assert len(sample.core) == 32
    assert len(sample.diversity) == 8
    assert sample.plan.precision_status == (
        "capacity_limited_precision_shortfall+diversity_reserved_precision_shortfall"
    )


def test_capacity_50_splits_40_core_10_diversity():
    units = _make_population_200_100()
    engine = SamplingEngine()
    policy = SamplePolicy(diversity_enabled=True, diversity_fraction=0.20, seed=41)

    cap_agent_a = AgentKey(tenant_id="tenant-a", agent_id="agent-a")
    sample = engine.sample(
        units,
        policy,
        capacities={cap_agent_a: 50},
        agents=[cap_agent_a],
    ).agents[0]

    assert sample.plan.selected == 50
    assert sample.plan.probability_selected == 40
    assert sample.plan.diversity_selected == 10
    assert len(sample.core) == 40
    assert len(sample.diversity) == 10


def test_tenant_capacity_is_conserved_across_agent_recommendations():
    agent_a = _make_population_200_100()
    agent_b = [
        _make_unit(
            i + 1000,
            tenant="tenant-a",
            agent="agent-b",
            stratum="large",
        )
        for i in range(100)
    ]
    engine = SamplingEngine()
    policy = SamplePolicy(diversity_enabled=True, diversity_fraction=0.20, seed=29)

    batch = engine.sample(
        agent_a + agent_b,
        policy,
        tenant_capacities={"tenant-a": 60},
    )

    assert sum(sample.plan.selected for sample in batch.agents) == 60
    assert sum(len(sample.core) + len(sample.diversity) for sample in batch.agents) == 60
    assert [sample.plan.selected for sample in batch.agents] == [36, 24]
    assert len(batch.tenant_capacities) == 1
    capacity = batch.tenant_capacities[0]
    assert capacity.granted == 60
    assert capacity.statistical_recommended == 124
    assert capacity.selected == 60
    assert capacity.unused == 0
    assert capacity.precision_status == "capacity_limited_precision_shortfall"

    replay = engine.sample(
        list(reversed(agent_a + agent_b)),
        policy,
        tenant_capacities={"tenant-a": 60},
    )
    assert replay.run_id == batch.run_id


def test_tenant_capacity_surplus_is_left_unused_and_modes_do_not_mix():
    units = _make_population_200_100()
    engine = SamplingEngine()
    policy = SamplePolicy(diversity_enabled=False)

    batch = engine.sample(
        units,
        policy,
        tenant_capacities={"tenant-a": 100},
    )

    capacity = batch.tenant_capacities[0]
    assert capacity.selected == 74
    assert capacity.unused == 26

    try:
        engine.sample(
            units,
            policy,
            capacities={AgentKey("tenant-a", "agent-a"): 20},
            tenant_capacities={"tenant-a": 20},
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("mixed capacity modes must be rejected")


def test_minhash_signature_changes_for_each_tagged_field_family():
    policy = SamplePolicy(
        diversity_enabled=True,
        minhash_ngram_size=3,
        minhash_permutations=128,
        seed=13,
    )
    baseline = _make_signature_unit()
    baseline_sig = build_minhash_signature(baseline, policy)

    assert build_minhash_signature(
        _make_signature_unit(user_text="different user words"), policy
    ) != baseline_sig
    assert build_minhash_signature(
        _make_signature_unit(assistant_text="different assistant words"), policy
    ) != baseline_sig
    assert build_minhash_signature(
        _make_signature_unit(tool_name="different_tool"), policy
    ) != baseline_sig
    assert build_minhash_signature(
        _make_signature_unit(tool_input="id=999"), policy
    ) != baseline_sig
    assert build_minhash_signature(
        _make_signature_unit(tool_output="status=cancelled"), policy
    ) != baseline_sig


def test_farthest_first_updates_distance_after_first_diversity_pick():
    policy = SamplePolicy(
        diversity_enabled=True,
        minhash_ngram_size=2,
        minhash_permutations=128,
        seed=53,
    )
    core = _make_signature_unit(
        user_text="alpha core request",
        assistant_text="alpha core response",
    )
    duplicate_a = _make_unit(700)
    duplicate_b = _make_unit(701)
    distinct = _make_unit(702)
    duplicate_content = (
        Turn("rare identical request", "rare identical response"),
    )
    object.__setattr__(duplicate_a, "turns", duplicate_content)
    object.__setattr__(duplicate_b, "turns", duplicate_content)
    object.__setattr__(distinct, "turns", (Turn("different gamma task", "different gamma outcome"),))

    chosen = _farthest_first(
        candidates=(duplicate_a, duplicate_b, distinct),
        budget=2,
        seed=53,
        policy=policy,
        core_signatures=(build_minhash_signature(core, policy),),
    )

    chosen_ids = {unit.unit_id for unit in chosen}
    duplicate_ids = {duplicate_a.unit_id, duplicate_b.unit_id}
    assert len(chosen_ids & duplicate_ids) == 1
    assert distinct.unit_id in chosen_ids


def test_window_changes_run_id():
    units = _make_population_200_100()
    engine = SamplingEngine()
    policy = SamplePolicy(seed=43, diversity_enabled=True, diversity_fraction=0.20)

    end = datetime(2025, 1, 2, tzinfo=timezone.utc)
    window_a = EvaluationWindow(start_at=end - timedelta(hours=24), end_at=end)
    window_b = EvaluationWindow(start_at=end - timedelta(hours=12), end_at=end)

    batch_a = engine.sample(units, policy, evaluation_window=window_a)
    batch_b = engine.sample(units, policy, evaluation_window=window_b)

    assert batch_a.run_id != batch_b.run_id
    assert batch_a.evaluation_window == window_a
    assert batch_b.evaluation_window == window_b
