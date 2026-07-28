from __future__ import annotations

from datetime import datetime, timezone

from trace_sampling_alt import (
    AgentKey,
    AgentSample,
    BinaryValue,
    CategoricalValue,
    EvaluationFailure,
    EvaluationRun,
    EvaluationUnit,
    JudgeDescriptor,
    LikertValue,
    MetricObservation,
    MetricSpec,
    SampleBatch,
    SamplePlan,
    SamplePolicy,
    SampledUnit,
    ScalarValue,
    StratumPlan,
    TASK_COMPLETION_V1,
    Turn,
    build_report,
)


def _sampled(conversation_id: str, kind: str = "core", eligible: bool = True) -> SampledUnit:
    unit = EvaluationUnit(
        tenant_id="tenant-a",
        agent_id="agent-a",
        conversation_id=conversation_id,
        session_id="sess",
        channel="teams",
        source_trace_ids=("trace",),
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        had_error=False,
        turns=(Turn(user_text="u", assistant_text="a"),),
        tool_calls=(),
    )
    return SampledUnit(
        unit=unit,
        sample_kind=kind,
        estimand_eligible=eligible,
        stratum_key="2-3|teams",
        inclusion_probability=1.0 if eligible else None,
        sampling_weight=1.0 if eligible else None,
        selection_reason="test",
    )


def _batch() -> SampleBatch:
    core = (_sampled("conv-1"), _sampled("conv-2"), _sampled("conv-3"))
    diversity = (_sampled("conv-4", kind="diversity", eligible=False),)
    return SampleBatch(
        policy=SamplePolicy(confidence=0.95, diversity_enabled=True, diversity_fraction=0.3),
        version="v1",
        run_id="run-id",
        agents=(
            AgentSample(
                agent=AgentKey("tenant-a", "agent-a"),
                plan=SamplePlan(
                    population=20,
                    recommended=3,
                    selected=3,
                    capacity=None,
                    census=False,
                    precision_status="diversity_reserved_precision_shortfall",
                    effective_rate=0.15,
                ),
                strata=(StratumPlan("2-3|teams", 20, 3),),
                core=core,
                diversity=diversity,
            ),
        ),
    )


def _obs(metric: MetricSpec, conv: str, value, kind: str = "core", eligible: bool = True) -> MetricObservation:
    unit_id = f"session:sess-{conv}"
    return MetricObservation(
        request_id=f"req-{metric.id}-{conv}-{kind}",
        agent=AgentKey("tenant-a", "agent-a"),
        tenant_id="tenant-a",
        agent_id="agent-a",
        unit_id=unit_id,
        session_id=f"sess-{conv}",
        conversation_ids=(conv,),
        metric=metric,
        value=value,
        sample_kind=kind,
        estimand_eligible=eligible,
        judge=JudgeDescriptor("stub", "deterministic", "v1"),
        evidence_sha256="abc",
    )


def test_binary_core_wilson_fpc_and_counts():
    batch = _batch()
    run = EvaluationRun(
        status="completed",
        observations=(
            _obs(TASK_COMPLETION_V1, "conv-1", BinaryValue(True)),
            _obs(TASK_COMPLETION_V1, "conv-2", BinaryValue(False)),
            _obs(TASK_COMPLETION_V1, "conv-3", BinaryValue(True)),
        ),
        failures=(),
        selected_count=4,
        request_count=3,
        judge_descriptor=JudgeDescriptor("stub", "deterministic", "v1"),
        metrics=(TASK_COMPLETION_V1,),
    )
    report = build_report(batch, run)
    core = next(row for row in report.summaries if row.metric_id == "task_completion" and row.sample_kind == "core")

    assert core.binary is not None
    assert core.binary.passes == 2
    assert core.binary.failures == 1
    assert core.binary.pass_rate == 2 / 3
    assert core.binary.wilson_interval is not None
    assert core.binary.wilson_interval.lower <= core.binary.pass_rate <= core.binary.wilson_interval.upper
    assert any("reserved diversity capacity" in note for note in report.notes)
    assert report.status == "completed"


def test_all_failing_diversity_does_not_change_core_headline_and_judge_failure_is_missing_not_fail():
    batch = _batch()
    diag_metric = MetricSpec(
        id="diag_binary",
        name="DIAG_BINARY_V1",
        version="v1",
        kind="binary",
        display_name="Diagnostic Binary",
    )
    run = EvaluationRun(
        status="partial",
        observations=(
            _obs(TASK_COMPLETION_V1, "conv-1", BinaryValue(True)),
            _obs(TASK_COMPLETION_V1, "conv-2", BinaryValue(True)),
            _obs(diag_metric, "conv-4", BinaryValue(False), kind="diversity", eligible=False),
        ),
        failures=(
            EvaluationFailure(
                request_id="req-missing",
                tenant_id="tenant-a",
                agent_id="agent-a",
                unit_id="session:sess-conv-3",
                session_id="sess-conv-3",
                metric_id="task_completion",
                metric_version="v1",
                sample_kind="core",
                code="judge_timeout",
                attempts=3,
                retryable=True,
                message="timeout",
            ),
        ),
        selected_count=4,
        request_count=4,
        judge_descriptor=JudgeDescriptor("stub", "deterministic", "v1"),
    )
    report = build_report(batch, run)

    core = next(row for row in report.summaries if row.metric_id == "task_completion" and row.sample_kind == "core")
    assert core.binary is not None
    assert core.binary.passes == 2
    assert core.binary.failures == 0
    assert core.binary.succeeded_count == 2
    assert core.binary.failed_count == 1
    assert core.binary.wilson_interval is not None
    assert core.binary.wilson_interval.lower <= 1.0 <= core.binary.wilson_interval.upper

    diversity = next(row for row in report.summaries if row.metric_id == "diag_binary" and row.sample_kind == "diversity")
    assert diversity.binary is not None
    assert diversity.binary.pass_rate == 0.0
    assert diversity.binary.wilson_interval is None
    assert diversity.binary.population_eligible_count is None
    assert diversity.binary.estimand_eligible is False


def test_all_failed_core_keeps_structured_summary():
    batch = _batch()
    failures = tuple(
        EvaluationFailure(
            request_id=f"req-{index}",
            tenant_id="tenant-a",
            agent_id="agent-a",
            unit_id=f"session:sess-conv-{index}",
            session_id=f"sess-conv-{index}",
            metric_id="task_completion",
            metric_version="v1",
            sample_kind="core",
            code="judge_timeout",
            attempts=3,
            retryable=True,
            message="timeout",
        )
        for index in range(1, 4)
    )
    run = EvaluationRun(
        status="failed",
        observations=(),
        failures=failures,
        selected_count=3,
        request_count=3,
        judge_descriptor=JudgeDescriptor("stub", "deterministic", "v1"),
        metrics=(TASK_COMPLETION_V1,),
    )

    report = build_report(batch, run)
    core = next(row for row in report.summaries if row.sample_kind == "core")

    assert core.binary is not None
    assert core.binary.selected_count == 3
    assert core.binary.submitted_count == 3
    assert core.binary.succeeded_count == 0
    assert core.binary.failed_count == 3
    assert core.binary.pass_rate is None
    assert core.binary.wilson_interval is None


def test_likert_scalar_categorical_summaries():
    batch = _batch()
    likert = MetricSpec(
        id="helpfulness",
        name="HELPFULNESS_V1",
        version="v1",
        kind="likert",
        display_name="Helpfulness",
        likert_min=1,
        likert_max=5,
    )
    scalar = MetricSpec(
        id="latency",
        name="LATENCY_V1",
        version="v1",
        kind="scalar",
        display_name="Latency",
    )
    categorical = MetricSpec(
        id="route",
        name="ROUTE_V1",
        version="v1",
        kind="categorical",
        display_name="Route",
        categories=("search", "tool", "other"),
    )

    run = EvaluationRun(
        status="completed",
        observations=(
            _obs(likert, "conv-1", LikertValue(score=2, min_score=1, max_score=5)),
            _obs(likert, "conv-2", LikertValue(score=4, min_score=1, max_score=5)),
            _obs(scalar, "conv-1", ScalarValue(10.0)),
            _obs(scalar, "conv-2", ScalarValue(20.0)),
            _obs(categorical, "conv-1", CategoricalValue("search")),
            _obs(categorical, "conv-2", CategoricalValue("tool")),
            _obs(categorical, "conv-3", CategoricalValue("tool")),
        ),
        failures=(),
        selected_count=4,
        request_count=7,
        judge_descriptor=JudgeDescriptor("stub", "deterministic", "v1"),
    )

    report = build_report(batch, run)
    likert_summary = next(row for row in report.summaries if row.metric_id == "helpfulness")
    assert likert_summary.likert is not None
    assert likert_summary.likert.distribution == ((2, 1), (4, 1))
    assert likert_summary.likert.mean == 3.0
    assert likert_summary.selected_count == 3
    assert likert_summary.succeeded_count == 2
    assert likert_summary.response_rate == 2 / 3

    scalar_summary = next(row for row in report.summaries if row.metric_id == "latency")
    assert scalar_summary.scalar is not None
    assert scalar_summary.scalar.mean == 15.0
    assert scalar_summary.scalar.min_value == 10.0
    assert scalar_summary.scalar.max_value == 20.0

    cat_summary = next(row for row in report.summaries if row.metric_id == "route")
    assert cat_summary.categorical is not None
    assert cat_summary.categorical.counts == (("search", 1), ("tool", 2))


def test_all_failed_likert_keeps_missingness_counts_without_values():
    batch = _batch()
    likert = MetricSpec(
        id="helpfulness",
        name="HELPFULNESS_V1",
        version="v1",
        kind="likert",
        display_name="Helpfulness",
        likert_min=1,
        likert_max=5,
    )
    failure = EvaluationFailure(
        request_id="req-likert-failed",
        tenant_id="tenant-a",
        agent_id="agent-a",
        unit_id="session:sess-conv-1",
        session_id="sess-conv-1",
        metric_id=likert.id,
        metric_version=likert.version,
        sample_kind="core",
        code="judge_timeout",
        attempts=3,
        retryable=True,
        message="timeout",
    )
    run = EvaluationRun(
        status="partial",
        observations=(),
        failures=(failure,),
        selected_count=3,
        request_count=1,
        judge_descriptor=JudgeDescriptor("stub", "deterministic", "v1"),
        metrics=(likert,),
    )

    report = build_report(batch, run)
    summary = next(
        row
        for row in report.summaries
        if row.metric_id == likert.id and row.sample_kind == "core"
    )

    assert summary.metric_kind == "likert"
    assert summary.selected_count == 3
    assert summary.submitted_count == 1
    assert summary.succeeded_count == 0
    assert summary.failed_count == 1
    assert summary.response_rate == 0.0
    assert summary.likert is None
