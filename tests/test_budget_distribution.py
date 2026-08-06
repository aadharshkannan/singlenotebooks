from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from random_sampling.budget_distribution import (
    AllocationConfig,
    BatchCheckpoint,
    BudgetDeductions,
    CheckpointCASConflictError,
    CheckpointTransitionError,
    FairnessState,
    FenceRejectedError,
    JsonCheckpointStore,
    JsonReferenceStore,
    LeaseConflictError,
    RollingTokenPacer,
    SessionDemand,
    SimulationScenario,
    UnpaceableReservationError,
    allocate_hierarchical_tokens,
    build_batch_plan,
    build_batch_telemetry,
    build_eligible_frame,
    calculate_batch_budget,
    resolve_batch_window,
    reserve_with_claim,
    select_within_agent_grants,
    simulate_policies,
)
from random_sampling.budget_distribution.selection import AgentSelectionGrant


def _session(
    sid: str,
    *,
    tenant: str = "t1",
    agent: str = "a1",
    complete_minute: int = 0,
    ingest_minute: int = 0,
    cost: int = 100,
    version: str = "v1",
) -> SessionDemand:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    return SessionDemand(
        tenant_id=tenant,
        agent_id=agent,
        session_id=sid,
        completed_at=base + timedelta(minutes=complete_minute),
        ingested_at=base + timedelta(minutes=ingest_minute),
        estimated_input_tokens=cost - 20,
        expected_output_tokens=20,
        session_version=version,
    )


def test_budget_examples_and_non_negative_effective_budget() -> None:
    five = calculate_batch_budget(5.0)
    hour = calculate_batch_budget(60.0)
    assert five.nominal_tokens == 100_000
    assert hour.nominal_tokens == 1_200_000

    bounded = calculate_batch_budget(
        5.0,
        deductions=BudgetDeductions(safety_tokens=5_000, output_tokens=10_000, retry_tokens=5_000),
    )
    assert bounded.effective_tokens == 80_000

    zeroed = calculate_batch_budget(
        1.0,
        deductions=BudgetDeductions(safety_tokens=99_999, output_tokens=99_999, retry_tokens=99_999),
    )
    assert zeroed.effective_tokens == 0


def test_window_bootstrap_fractional_and_catchup() -> None:
    cutoff = datetime(2026, 1, 1, 13, 30, tzinfo=timezone.utc)
    bootstrap = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    window = resolve_batch_window(
        previous_successful_watermark=None,
        cutoff=cutoff,
        bootstrap_watermark=bootstrap,
        lookback=timedelta(minutes=15),
        max_catchup_minutes=60.0,
    )
    assert window.bootstrap_used is True
    assert window.elapsed_minutes == 90.0
    assert window.clamped_minutes == 60.0
    assert window.source_scan_start == bootstrap - timedelta(minutes=15)

    fractional_cutoff = datetime(2026, 1, 1, 12, 2, 30, tzinfo=timezone.utc)
    fractional = resolve_batch_window(
        previous_successful_watermark=bootstrap,
        cutoff=fractional_cutoff,
        max_catchup_minutes=60.0,
    )
    assert fractional.elapsed_minutes == 2.5

    capped = resolve_batch_window(
        previous_successful_watermark=bootstrap,
        cutoff=bootstrap + timedelta(minutes=90),
        max_catchup_minutes=60.0,
    )
    budget = calculate_batch_budget(window=capped)
    assert budget.nominal_tokens == 1_200_000


def test_window_requires_bootstrap_on_first_run() -> None:
    with pytest.raises(ValueError, match="bootstrap_watermark"):
        resolve_batch_window(
            previous_successful_watermark=None,
            cutoff=datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc),
        )


def test_eligible_frame_canonical_membership_late_and_dedup() -> None:
    previous = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    cutoff = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    window = resolve_batch_window(
        previous_successful_watermark=previous,
        cutoff=cutoff,
        lookback=timedelta(minutes=20),
    )

    s1 = _session("s1", complete_minute=1, ingest_minute=2)
    s1_dup = _session("s1", complete_minute=1, ingest_minute=3)
    late_unseen = _session("s2", complete_minute=-5, ingest_minute=10)
    late_processed = _session("s3", complete_minute=-5, ingest_minute=11)
    outside = _session("s4", complete_minute=90, ingest_minute=91)

    frame = build_eligible_frame(
        source_sessions=[s1, s1_dup, late_unseen, late_processed, outside],
        window=window,
        processed_session_keys={late_processed.dedup_key},
    )

    assert [s.session_id for s in frame.sessions] == ["s2", "s1"]
    assert frame.duplicate_count == 1
    assert frame.lookback_admitted_count == 1
    assert frame.processed_count == 1


def test_allocation_and_selection_deterministic_with_oversize_and_redistribution() -> None:
    sessions = (
        _session("a", tenant="t1", agent="a1", cost=300),
        _session("b", tenant="t1", agent="a1", cost=300),
        _session("c", tenant="t1", agent="a2", cost=200),
        _session("d", tenant="t2", agent="a3", cost=950),
        _session("e", tenant="t2", agent="a3", cost=1500),
    )
    allocation, next_state = allocate_hierarchical_tokens(
        sessions=sessions,
        total_budget_tokens=1_000,
        fairness_state=FairnessState(),
        config=AllocationConfig(tenant_floor_tokens=100, agent_floor_tokens=100),
    )
    assert allocation.total_budget_tokens == 1_000
    assert allocation.agent_nodes

    grants = tuple(
        AgentSelectionGrant(key=key, grant_tokens=node.grant_tokens)
        for key, node in allocation.agent_nodes.items()
    )
    selection = select_within_agent_grants(
        sessions=sessions,
        grants=grants,
        seed="seed",
        frame_hash="frame",
        effective_budget_tokens=1_000,
    )
    assert selection.unserviceable_count == 1
    assert selection.selected_total_tokens <= 1_000
    assert selection.selected_ids == tuple(sorted(selection.selected_ids))
    assert next_state.tenant_deficit_tokens


def test_selection_audit_integrity_and_composite_selected_ids() -> None:
    sessions = (
        _session("same", tenant="t1", agent="a1", cost=300, version="v1"),
        _session("same", tenant="t1", agent="a1", cost=300, version="v2"),
        _session("same", tenant="t2", agent="a2", cost=100, version="v1"),
    )
    grants = (
        AgentSelectionGrant(key=sessions[0].key, grant_tokens=300, deficit_priority=1),
        AgentSelectionGrant(key=sessions[2].key, grant_tokens=400, deficit_priority=0),
    )
    selection = select_within_agent_grants(
        sessions=sessions,
        grants=grants,
        seed="seed",
        frame_hash="frame",
        effective_budget_tokens=700,
    )
    selected_keys = {record.demand.dedup_key for record in selection.selected}
    unselected_keys = {record.demand.dedup_key for record in selection.unselected}
    assert selected_keys.isdisjoint(unselected_keys)
    assert selected_keys | unselected_keys == {session.dedup_key for session in sessions}
    assert len(selected_keys) == len(selection.selected)
    assert len(unselected_keys) == len(selection.unselected)
    assert set(selection.selected_ids) == selected_keys
    assert any("/same/v2" in sid for sid in selection.selected_ids)


def test_dominant_tenant_and_many_agents_zero_grants_possible() -> None:
    sessions = tuple(
        _session(f"s{i}", tenant="big", agent=f"a{i}", cost=300)
        for i in range(6)
    ) + (
        _session("small-1", tenant="small", agent="x", cost=100),
    )
    allocation, _ = allocate_hierarchical_tokens(
        sessions=sessions,
        total_budget_tokens=3,
        fairness_state=FairnessState(),
        config=AllocationConfig(tenant_floor_tokens=0, agent_floor_tokens=0),
    )
    assert allocation.zero_grant_agents >= 1


def test_label_blind_model_has_no_outcome_fields() -> None:
    demand = _session("blind")
    assert not hasattr(demand, "label")
    assert not hasattr(demand, "outcome")


def test_batch_plan_contains_required_outputs() -> None:
    previous = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    cutoff = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    window = resolve_batch_window(previous_successful_watermark=previous, cutoff=cutoff)
    budget = calculate_batch_budget(60.0)
    frame = build_eligible_frame(
        source_sessions=[_session("p1", cost=100), _session("p2", cost=120)],
        window=window,
        processed_session_keys=set(),
    )
    plan = build_batch_plan(
        pipeline_id="pipe",
        batch_id="batch-1",
        seed="seed",
        window=window,
        budget=budget,
        frame=frame,
        fairness_state=FairnessState(),
        allocation_config=AllocationConfig(tenant_floor_tokens=20, agent_floor_tokens=10),
    )
    assert plan.batch_id == "batch-1"
    assert plan.seed == "seed"
    assert plan.frame_hash
    assert plan.membership_hash
    assert plan.config_hash
    assert plan.selection.selected_ids
    assert plan.planned_usage_tokens >= 0


def test_batch_plan_marks_over_tpm_requests_unserviceable() -> None:
    previous = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    cutoff = datetime(2026, 1, 1, 12, 5, tzinfo=timezone.utc)
    window = resolve_batch_window(previous_successful_watermark=previous, cutoff=cutoff)
    budget = calculate_batch_budget(clamped_minutes=5.0, tpm_limit=20_000)
    frame = build_eligible_frame(
        source_sessions=[_session("huge", cost=25_000)],
        window=window,
        processed_session_keys=set(),
    )
    plan = build_batch_plan(
        pipeline_id="pipe",
        batch_id="batch-oversize",
        seed="seed",
        window=window,
        budget=budget,
        frame=frame,
        fairness_state=FairnessState(),
    )
    assert len(plan.schedule) == 0
    assert plan.selection.unserviceable_count == 1


def test_batch_plan_fairness_state_uses_served_tokens() -> None:
    previous = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    cutoff = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    window = resolve_batch_window(previous_successful_watermark=previous, cutoff=cutoff)
    budget = calculate_batch_budget(clamped_minutes=0.05, tpm_limit=20_000)
    frame = build_eligible_frame(
        source_sessions=[
            _session("f1", tenant="t1", agent="a1", cost=700),
            _session("f2", tenant="t1", agent="a1", cost=700),
        ],
        window=window,
        processed_session_keys=set(),
    )
    plan = build_batch_plan(
        pipeline_id="pipe",
        batch_id="batch-fair",
        seed="seed",
        window=window,
        budget=budget,
        frame=frame,
        fairness_state=FairnessState(),
        allocation_config=AllocationConfig(tenant_floor_tokens=0, agent_floor_tokens=0),
    )
    assert plan.planned_usage_tokens == 700
    assert plan.fairness_state_out.agent_deficit_tokens["t1/a1"] == 700


def test_checkpoint_lifecycle_retry_and_success_commit(tmp_path) -> None:
    store = JsonCheckpointStore(tmp_path / "checkpoint")
    created = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    checkpoint = BatchCheckpoint(
        pipeline_id="pipe",
        batch_id="batch-a",
        status="PREPARED",
        previous_successful_watermark=created - timedelta(minutes=60),
        cutoff=created,
        elapsed_minutes=60.0,
        nominal_budget_tokens=1_200_000,
        effective_budget_tokens=1_150_000,
        seed="seed",
        frame_hash="fh",
        config_hash="ch",
        membership_hash="mh",
        selected_ids=("t1/a1/s1/v1",),
        planned_usage_tokens=100,
        actual_usage_tokens=0,
        retry_count=0,
        fairness_state={"tenant": {}},
        created_at=created,
    )
    first = store.prepare(checkpoint, frame_hash="fh", config_hash="ch", seed="seed")
    second = store.prepare(checkpoint, frame_hash="fh", config_hash="ch", seed="seed")
    assert first.batch_id == second.batch_id

    store.mark_running("batch-a")
    store.settle("batch-a", actual_usage_tokens=90)
    store.commit("batch-a", success=False)
    assert store.get("batch-a").status == "FAILED"

    checkpoint_b = BatchCheckpoint(
        pipeline_id="pipe",
        batch_id="batch-b",
        status="PREPARED",
        previous_successful_watermark=created - timedelta(minutes=60),
        cutoff=created + timedelta(minutes=1),
        elapsed_minutes=61.0,
        nominal_budget_tokens=1_220_000,
        effective_budget_tokens=1_150_000,
        seed="seed-2",
        frame_hash="fh-2",
        config_hash="ch",
        membership_hash="mh",
        selected_ids=("t1/a1/s2/v1",),
        planned_usage_tokens=80,
        actual_usage_tokens=0,
        retry_count=0,
        fairness_state={"tenant": {}},
        created_at=created,
    )
    store.prepare(checkpoint_b, frame_hash="fh-2", config_hash="ch", seed="seed-2")
    store.mark_running("batch-b")
    store.settle("batch-b", actual_usage_tokens=80)
    store.commit("batch-b", success=True, new_watermark=created, expected_previous_watermark=None)
    assert store.get("batch-b").status == "COMMITTED"
    assert store.latest_successful_watermark() == created


def test_checkpoint_commit_requires_legal_transition_and_cas(tmp_path) -> None:
    store = JsonCheckpointStore(tmp_path / "checkpoint")
    created = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    checkpoint = BatchCheckpoint(
        pipeline_id="pipe",
        batch_id="batch-c",
        status="PREPARED",
        previous_successful_watermark=created - timedelta(minutes=60),
        cutoff=created,
        elapsed_minutes=60.0,
        nominal_budget_tokens=1_200_000,
        effective_budget_tokens=1_150_000,
        seed="seed",
        frame_hash="fh",
        config_hash="ch",
        membership_hash="mh",
        selected_ids=("t1/a1/s1/v1",),
        planned_usage_tokens=100,
        actual_usage_tokens=0,
        retry_count=0,
        fairness_state={"tenant": {}},
        created_at=created,
    )
    store.prepare(checkpoint, frame_hash="fh", config_hash="ch", seed="seed")
    with pytest.raises(CheckpointTransitionError):
        store.commit("batch-c", success=True, new_watermark=created, expected_previous_watermark=None)

    store.mark_running("batch-c")
    store.settle("batch-c", actual_usage_tokens=90)
    with pytest.raises(CheckpointCASConflictError):
        store.commit(
            "batch-c",
            success=True,
            new_watermark=created,
            expected_previous_watermark=created - timedelta(minutes=1),
        )


def test_reference_store_lease_and_fencing_and_claims(tmp_path) -> None:
    ref = JsonReferenceStore(tmp_path / "reference")
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    lease1 = ref.acquire_lease(
        pipeline_id="pipe",
        holder="worker-a",
        ttl=timedelta(minutes=5),
        now=now,
    )
    with pytest.raises(LeaseConflictError):
        ref.acquire_lease(
            pipeline_id="pipe",
            holder="worker-b",
            ttl=timedelta(minutes=5),
            now=now + timedelta(minutes=1),
        )

    lease2 = ref.acquire_lease(
        pipeline_id="pipe",
        holder="worker-a",
        ttl=timedelta(minutes=5),
        now=now + timedelta(minutes=6),
    )
    assert lease2.generation > lease1.generation
    with pytest.raises(FenceRejectedError):
        ref.assert_fence(lease1)
    ref.assert_fence(lease2)

    assert ref.claim_request(request_id="r1", batch_id="b1") is True
    assert ref.claim_request(request_id="r1", batch_id="b1") is True
    assert ref.claim_request(request_id="r1", batch_id="b2") is False


def test_pacing_reserve_reconcile_and_overlap_claim_prevention(tmp_path) -> None:
    ref = JsonReferenceStore(tmp_path / "reference")
    pacer = RollingTokenPacer(tpm_limit=20_000)

    first = reserve_with_claim(
        pacer=pacer,
        reference_store=ref,
        batch_id="b1",
        request_id="req-1",
        estimated_tokens=12_000,
    )
    second = reserve_with_claim(
        pacer=pacer,
        reference_store=ref,
        batch_id="b1",
        request_id="req-2",
        estimated_tokens=12_000,
    )
    assert second.scheduled_at_seconds >= first.scheduled_at_seconds
    assert pacer.is_tpm_compliant() is True

    delta = pacer.reconcile("req-2", actual_tokens=10_000)
    assert delta == -2_000
    assert pacer.is_tpm_compliant() is True

    with pytest.raises(RuntimeError):
        reserve_with_claim(
            pacer=pacer,
            reference_store=ref,
            batch_id="b2",
            request_id="req-1",
            estimated_tokens=100,
        )


def test_pacing_rejects_unpaceable_requests_and_rolls_back_claim(tmp_path) -> None:
    ref = JsonReferenceStore(tmp_path / "reference")
    pacer = RollingTokenPacer(tpm_limit=20_000)
    with pytest.raises(UnpaceableReservationError):
        reserve_with_claim(
            pacer=pacer,
            reference_store=ref,
            batch_id="b1",
            request_id="req-big",
            estimated_tokens=50_000,
        )
    assert ref.claim_request(request_id="req-big", batch_id="b2") is True


def test_pacing_dense_window_and_reconcile_behavior() -> None:
    pacer = RollingTokenPacer(tpm_limit=20_000)
    r1 = pacer.reserve("r1", 10_000)
    r2 = pacer.reserve("r2", 10_000)
    r3 = pacer.reserve("r3", 5_000)
    assert r1.scheduled_at_seconds == 0.0
    assert r2.scheduled_at_seconds == 0.0
    assert r3.scheduled_at_seconds >= 60.0
    assert pacer.is_tpm_compliant() is True

    delta = pacer.reconcile("r2", actual_tokens=15_000)
    assert delta == 5_000
    assert pacer.is_tpm_compliant() is False


def test_telemetry_and_serialization_shape(tmp_path) -> None:
    sessions = (_session("m1", cost=100), _session("m2", cost=200))
    allocation, _ = allocate_hierarchical_tokens(
        sessions=sessions,
        total_budget_tokens=500,
        fairness_state=FairnessState(),
    )
    selection = select_within_agent_grants(
        sessions=sessions,
        grants=(AgentSelectionGrant(key=sessions[0].key, grant_tokens=500),),
        seed="seed",
        frame_hash="frame",
        effective_budget_tokens=500,
    )
    telemetry = build_batch_telemetry(
        allocation=allocation,
        selection=selection,
        total_eligible_sessions=len(sessions),
        tpm_compliance=True,
    )
    assert 0.0 <= telemetry.utilization <= 1.0
    assert 0.0 <= telemetry.coverage <= 1.0


def test_simulation_basic_and_replay_metrics() -> None:
    scenario = SimulationScenario(
        name="mini",
        sessions_by_batch=(
            (_session("s1", cost=200), _session("s2", tenant="t2", agent="a2", cost=150)),
            (_session("s3", cost=200), _session("s4", tenant="t2", agent="a2", cost=150)),
        ),
        budget_tokens_per_batch=300,
    )
    result = simulate_policies((scenario,), seed="deterministic")
    assert result.metrics
    assert {metric.policy for metric in result.metrics} == {
        "fcfs",
        "global_random",
        "equal_tenant",
        "equal_agent",
        "proportional",
        "hierarchical",
    }
    assert all(metric.replay_match for metric in result.metrics)
    assert all(0.0 <= metric.utilization <= 1.0 for metric in result.metrics)


def test_simulation_scarcity_yields_distinct_policy_outcomes() -> None:
    scenario = SimulationScenario(
        name="scarce-differentiation",
        sessions_by_batch=(
            (
                _session("s1", tenant="t1", agent="a1", complete_minute=0, cost=900),
                _session("s2", tenant="t1", agent="a1", complete_minute=1, cost=900),
                _session("s3", tenant="t1", agent="a2", complete_minute=2, cost=300),
                _session("s4", tenant="t2", agent="b1", complete_minute=3, cost=300),
            ),
        ),
        budget_tokens_per_batch=1_200,
    )
    result = simulate_policies((scenario,), seed="scarcity-seed")
    rows = [metric for metric in result.metrics if metric.scenario == scenario.name]
    distinct_outcomes = {(row.selected, round(row.utilization, 6), round(row.coverage, 6)) for row in rows}
    assert len(distinct_outcomes) > 1

    fcfs = next(metric for metric in rows if metric.policy == "fcfs")
    assert fcfs.selected == 2
    assert fcfs.coverage == pytest.approx(0.5)
    assert fcfs.utilization == pytest.approx(1.0)
    assert fcfs.utilization != pytest.approx(fcfs.coverage)


def test_simulation_fairness_uses_cumulative_actual_served_distribution() -> None:
    scenario = SimulationScenario(
        name="served-fairness-denominator",
        sessions_by_batch=(
            (
                _session("big", tenant="t1", agent="a1", complete_minute=0, cost=1_000),
                _session("small", tenant="t2", agent="a2", complete_minute=1, cost=100),
            ),
        ),
        budget_tokens_per_batch=100,
    )
    result = simulate_policies((scenario,), seed="served-jain-seed")
    fcfs = next(metric for metric in result.metrics if metric.policy == "fcfs" and metric.scenario == scenario.name)
    assert fcfs.selected == 1
    assert fcfs.fairness_jain == pytest.approx(0.5)


def test_simulation_equal_baselines_do_not_artificially_zero_low_demand_batches() -> None:
    scenario = SimulationScenario(
        name="equal-baseline-low-demand-regression",
        sessions_by_batch=(
            (
                _session("s1", tenant="t1", agent="a1", cost=80),
                _session("s2", tenant="t1", agent="a2", cost=90),
                _session("s3", tenant="t2", agent="b1", cost=70),
            ),
        ),
        budget_tokens_per_batch=500,
    )
    result = simulate_policies((scenario,), seed="equal-baseline-regression")

    equal_agent = next(metric for metric in result.metrics if metric.policy == "equal_agent" and metric.scenario == scenario.name)
    equal_tenant = next(metric for metric in result.metrics if metric.policy == "equal_tenant" and metric.scenario == scenario.name)

    assert equal_agent.selected == 3
    assert equal_tenant.selected == 3
    assert equal_agent.coverage == pytest.approx(1.0)
    assert equal_tenant.coverage == pytest.approx(1.0)
    assert equal_agent.utilization > 0.0
    assert equal_tenant.utilization > 0.0


def test_batch_plan_config_hash_changes_when_reserve_breakdown_changes() -> None:
    previous = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    cutoff = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    window = resolve_batch_window(previous_successful_watermark=previous, cutoff=cutoff)
    frame = build_eligible_frame(
        source_sessions=[_session("h1", cost=200), _session("h2", cost=220)],
        window=window,
        processed_session_keys=set(),
    )

    budget_a = calculate_batch_budget(
        window=window,
        deductions=BudgetDeductions(safety_tokens=6_000, retry_tokens=5_000, output_tokens=4_000),
    )
    budget_b = calculate_batch_budget(
        window=window,
        deductions=BudgetDeductions(safety_tokens=8_000, retry_tokens=2_000, output_tokens=5_000),
    )
    assert budget_a.effective_tokens == budget_b.effective_tokens

    plan_a = build_batch_plan(
        pipeline_id="pipe",
        batch_id="batch-hash-a",
        seed="seed",
        window=window,
        budget=budget_a,
        frame=frame,
        fairness_state=FairnessState(),
    )
    plan_b = build_batch_plan(
        pipeline_id="pipe",
        batch_id="batch-hash-b",
        seed="seed",
        window=window,
        budget=budget_b,
        frame=frame,
        fairness_state=FairnessState(),
    )
    assert plan_a.config_hash != plan_b.config_hash
