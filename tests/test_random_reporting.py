from __future__ import annotations

from datetime import datetime, timezone

from random_sampling import (
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


def _sampled(conversation_id: str) -> SampledUnit:
    unit = EvaluationUnit(
        tenant_id="tenant-a",
        agent_id="agent-a",
        conversation_id=conversation_id,
        session_id=f"session-{conversation_id}",
        channel="teams",
        source_trace_ids=(f"trace-{conversation_id}",),
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        had_error=False,
        turns=(Turn("u", "a"),),
        tool_calls=(),
    )
    return SampledUnit(
        unit=unit,
        estimand_eligible=True,
        stratum_key="1|teams",
        inclusion_probability=0.15,
        sampling_weight=1 / 0.15,
        selection_reason="test",
    )


def _batch() -> SampleBatch:
    units = (_sampled("conv-1"), _sampled("conv-2"), _sampled("conv-3"))
    return SampleBatch(
        policy=SamplePolicy(confidence=0.95),
        version="random-v1",
        run_id="run-id",
        agents=(
            AgentSample(
                agent=AgentKey("tenant-a", "agent-a"),
                plan=SamplePlan(20, 3, 3, None, False, "meets_statistical_recommendation", 0.15),
                strata=(StratumPlan("1|teams", 20, 3),),
                units=units,
            ),
        ),
    )


def _observation(metric: MetricSpec, conversation_id: str, value) -> MetricObservation:
    return MetricObservation(
        request_id=f"request-{metric.id}-{conversation_id}",
        agent=AgentKey("tenant-a", "agent-a"),
        tenant_id="tenant-a",
        agent_id="agent-a",
        unit_id=f"unit-{conversation_id}",
        session_id=f"session-{conversation_id}",
        conversation_ids=(conversation_id,),
        metric=metric,
        value=value,
        estimand_eligible=True,
        judge=JudgeDescriptor("stub", "deterministic", "v1"),
        evidence_sha256="abc",
    )


def test_binary_random_sample_wilson_fpc_and_counts():
    run = EvaluationRun(
        status="completed",
        observations=(
            _observation(TASK_COMPLETION_V1, "conv-1", BinaryValue(True)),
            _observation(TASK_COMPLETION_V1, "conv-2", BinaryValue(False)),
            _observation(TASK_COMPLETION_V1, "conv-3", BinaryValue(True)),
        ),
        failures=(),
        selected_count=3,
        request_count=3,
        judge_descriptor=JudgeDescriptor("stub", "deterministic", "v1"),
        metrics=(TASK_COMPLETION_V1,),
    )

    report = build_report(_batch(), run)
    summary = report.summaries[0]

    assert report.version == "random-report-v1"
    assert report.status == "completed"
    assert summary.binary is not None
    assert summary.binary.passes == 2
    assert summary.binary.failures == 1
    assert summary.binary.pass_rate == 2 / 3
    assert summary.binary.wilson_interval is not None
    assert summary.binary.wilson_interval.lower <= 2 / 3 <= summary.binary.wilson_interval.upper
    assert summary.binary.estimand_eligible is True


def test_judge_failure_is_missing_not_outcome_failure():
    failure = EvaluationFailure(
        request_id="failed",
        tenant_id="tenant-a",
        agent_id="agent-a",
        unit_id="unit-conv-3",
        session_id="session-conv-3",
        metric_id="task_completion",
        metric_version="v1",
        code="judge_timeout",
        attempts=3,
        retryable=True,
        message="timeout",
    )
    run = EvaluationRun(
        status="partial",
        observations=(
            _observation(TASK_COMPLETION_V1, "conv-1", BinaryValue(True)),
            _observation(TASK_COMPLETION_V1, "conv-2", BinaryValue(True)),
        ),
        failures=(failure,),
        selected_count=3,
        request_count=3,
        judge_descriptor=JudgeDescriptor("stub", "deterministic", "v1"),
        metrics=(TASK_COMPLETION_V1,),
    )

    report = build_report(_batch(), run)
    binary = report.summaries[0].binary

    assert report.status == "partial"
    assert binary is not None
    assert binary.succeeded_count == 2
    assert binary.failed_count == 1
    assert binary.passes == 2
    assert binary.failures == 0
    assert binary.response_rate == 2 / 3


def test_all_failed_sample_keeps_structured_summary():
    failures = tuple(
        EvaluationFailure(
            request_id=f"failed-{index}",
            tenant_id="tenant-a",
            agent_id="agent-a",
            unit_id=f"unit-conv-{index}",
            session_id=f"session-conv-{index}",
            metric_id="task_completion",
            metric_version="v1",
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

    binary = build_report(_batch(), run).summaries[0].binary
    assert binary is not None
    assert binary.succeeded_count == 0
    assert binary.failed_count == 3
    assert binary.pass_rate is None
    assert binary.wilson_interval is None


def test_likert_scalar_and_categorical_summaries():
    likert = MetricSpec("helpfulness", "HELPFULNESS_V1", "v1", "likert", "Helpfulness", likert_min=1, likert_max=5)
    scalar = MetricSpec("latency", "LATENCY_V1", "v1", "scalar", "Latency")
    categorical = MetricSpec("route", "ROUTE_V1", "v1", "categorical", "Route", categories=("search", "tool"))
    run = EvaluationRun(
        status="completed",
        observations=(
            _observation(likert, "conv-1", LikertValue(2, 1, 5)),
            _observation(likert, "conv-2", LikertValue(4, 1, 5)),
            _observation(scalar, "conv-1", ScalarValue(10.0)),
            _observation(scalar, "conv-2", ScalarValue(20.0)),
            _observation(categorical, "conv-1", CategoricalValue("search")),
            _observation(categorical, "conv-2", CategoricalValue("tool")),
            _observation(categorical, "conv-3", CategoricalValue("tool")),
        ),
        failures=(),
        selected_count=3,
        request_count=7,
        judge_descriptor=JudgeDescriptor("stub", "deterministic", "v1"),
        metrics=(likert, scalar, categorical),
    )

    summaries = {row.metric_id: row for row in build_report(_batch(), run).summaries}
    assert summaries["helpfulness"].likert.mean == 3.0
    assert summaries["helpfulness"].likert.distribution == ((2, 1), (4, 1))
    assert summaries["latency"].scalar.mean == 15.0
    assert summaries["route"].categorical.counts == (("search", 1), ("tool", 2))
