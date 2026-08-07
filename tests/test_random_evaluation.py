from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from random_sampling import (
    AgentKey,
    BinaryValue,
    DeterministicJudgeStub,
    EvaluationRunner,
    EvaluationUnit,
    InMemoryOutcomeSink,
    JudgeDescriptor,
    JudgeRequest,
    JudgeResponse,
    JudgeResponseError,
    JudgeTransientError,
    MetricSpec,
    RunnerConfig,
    SampleBatch,
    SamplePlan,
    SamplePolicy,
    SampledUnit,
    StratumPlan,
    SyncJudgeAdapter,
    Turn,
)
from random_sampling.azure_openai_judge import (
    JudgeAuthenticationError,
    JudgeContentFilteredError,
    JudgeEmptyResponseError,
    JudgeLengthExhaustedError,
    JudgeMalformedJsonError,
    JudgeProviderError,
    JudgeSkippedError,
    JudgeTerminalHttpError,
)


def _unit(i: int) -> EvaluationUnit:
    return EvaluationUnit(
        tenant_id="tenant-a",
        agent_id="agent-a",
        conversation_id=f"conv-{i}",
        session_id=f"sess-{i}",
        channel="teams",
        source_trace_ids=(f"trace-{i}",),
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        had_error=False,
        turns=(Turn(user_text="u", assistant_text="a"),),
        tool_calls=(),
    )


def _batch(count: int = 3) -> SampleBatch:
    sampled = tuple(
        SampledUnit(
            unit=_unit(i),
            estimand_eligible=True,
            stratum_key="2-3|teams",
            inclusion_probability=1.0,
            sampling_weight=1.0,
            selection_reason="test",
        )
        for i in range(count)
    )
    agent = AgentKey("tenant-a", "agent-a")
    return SampleBatch(
        policy=SamplePolicy(),
        version="vtest",
        run_id="run-test",
        agents=(
            __import__("random_sampling").AgentSample(
                agent=agent,
                plan=SamplePlan(
                    population=count,
                    recommended=count,
                    selected=count,
                    capacity=None,
                    census=True,
                    precision_status="meets_statistical_recommendation",
                    effective_rate=1.0,
                ),
                strata=(StratumPlan(key="2-3|teams", population=count, selected=count),),
                units=sampled,
            ),
        ),
    )


@dataclass(frozen=True)
class _SyncBinaryJudge:
    descriptor: JudgeDescriptor = JudgeDescriptor(provider="stub", name="sync", version="v1")

    def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        return JudgeResponse(
            request_id=request.request_id,
            metric=request.metric,
            value=BinaryValue(passed=True),
            reasoning="ok",
        )


class _FlakyJudge:
    descriptor = JudgeDescriptor(provider="stub", name="flaky", version="v1")

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        c = self.calls.get(request.request_id, 0) + 1
        self.calls[request.request_id] = c
        if c < 2:
            raise JudgeTransientError("retry me")
        return JudgeResponse(request_id=request.request_id, metric=request.metric, value=BinaryValue(True))


class _BadJudge:
    descriptor = JudgeDescriptor(provider="stub", name="bad", version="v1")

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        likert = MetricSpec(
            id="likert-x",
            name="LIKERT_X",
            version="v1",
            kind="likert",
            display_name="Likert X",
            likert_min=1,
            likert_max=5,
        )
        return JudgeResponse(
            request_id=request.request_id,
            metric=likert,
            value=__import__("random_sampling").LikertValue(score=3, min_score=1, max_score=5),
        )


class _UnexpectedJudge:
    descriptor = JudgeDescriptor(provider="stub", name="unexpected", version="v1")

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        if request.unit_id == _unit(0).unit_id:
            raise RuntimeError("provider SDK failed")
        return JudgeResponse(
            request_id=request.request_id,
            metric=request.metric,
            value=BinaryValue(True),
        )


class _SafeResponseErrorJudge:
    descriptor = JudgeDescriptor(provider="stub", name="safe-errors", version="v1")

    def __init__(self, error: JudgeResponseError) -> None:
        self.error = error

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        raise self.error


def test_sync_adapter_parity_and_sink_idempotence():
    async def _case() -> None:
        batch = _batch(2)
        metric = MetricSpec(
            id="task_completion",
            name="TASK_COMPLETION_V1",
            version="v1",
            kind="binary",
            display_name="Task Completion",
        )

        sink = InMemoryOutcomeSink()
        runner = EvaluationRunner(
            judge=SyncJudgeAdapter(_SyncBinaryJudge()),
            config=RunnerConfig(max_concurrency=1, max_attempts=2, base_backoff_seconds=0.0),
            outcome_sink=sink,
        )
        run1 = await runner.run(batch, (metric,))
        run2 = await runner.run(batch, (metric,))

        assert run1.status == "completed"
        assert run2.status == "completed"
        assert len(run1.observations) == 2
        assert len(sink.values()) == 2

    asyncio.run(_case())


def test_stub_stable_and_runner_deterministic_ordering():
    async def _case() -> None:
        batch = _batch(3)
        metric = MetricSpec(
            id="task_completion",
            name="TASK_COMPLETION_V1",
            version="v1",
            kind="binary",
            display_name="Task Completion",
        )
        runner = EvaluationRunner(
            judge=DeterministicJudgeStub(),
            config=RunnerConfig(max_concurrency=3, max_attempts=1, base_backoff_seconds=0.0),
        )
        run_a = await runner.run(batch, (metric,))
        run_b = await runner.run(batch, (metric,))

        assert [o.request_id for o in run_a.observations] == [o.request_id for o in run_b.observations]
        assert [o.value for o in run_a.observations] == [o.value for o in run_b.observations]

    asyncio.run(_case())


def test_transient_retry_then_success():
    async def _case() -> None:
        batch = _batch(1)
        metric = MetricSpec(
            id="task_completion",
            name="TASK_COMPLETION_V1",
            version="v1",
            kind="binary",
            display_name="Task Completion",
        )
        runner = EvaluationRunner(
            judge=_FlakyJudge(),
            config=RunnerConfig(max_attempts=3, base_backoff_seconds=0.0),
        )
        run = await runner.run(batch, (metric,))

        assert run.status == "completed"
        assert len(run.observations) == 1
        assert not run.failures

    asyncio.run(_case())


def test_terminal_malformed_response_and_no_replacement():
    async def _case() -> None:
        batch = _batch(2)
        metric = MetricSpec(
            id="task_completion",
            name="TASK_COMPLETION_V1",
            version="v1",
            kind="binary",
            display_name="Task Completion",
        )
        runner = EvaluationRunner(
            judge=_BadJudge(),
            config=RunnerConfig(max_attempts=2, base_backoff_seconds=0.0),
        )
        run = await runner.run(batch, (metric,))

        assert run.status == "failed"
        assert len(run.observations) == 0
        assert len(run.failures) == 2
        assert all(f.code == "judge_malformed_response" for f in run.failures)
        assert run.selected_count == 2

    asyncio.run(_case())


def test_unexpected_judge_error_is_terminal_for_one_request_only():
    async def _case() -> None:
        batch = _batch(2)
        metric = MetricSpec(
            id="task_completion",
            name="TASK_COMPLETION_V1",
            version="v1",
            kind="binary",
            display_name="Task Completion",
        )
        run = await EvaluationRunner(
            judge=_UnexpectedJudge(),
            config=RunnerConfig(max_attempts=2, base_backoff_seconds=0.0),
        ).run(batch, (metric,))

        assert run.status == "partial"
        assert len(run.observations) == 1
        assert len(run.failures) == 1
        assert run.failures[0].code == "judge_unexpected_error"
        assert run.failures[0].attempts == 1
        assert run.metrics == (metric,)

    asyncio.run(_case())


def test_unit_id_filter_unknown_raises_value_error():
    async def _case() -> None:
        batch = _batch(2)
        metric = MetricSpec(
            id="task_completion",
            name="TASK_COMPLETION_V1",
            version="v1",
            kind="binary",
            display_name="Task Completion",
        )
        runner = EvaluationRunner(
            judge=DeterministicJudgeStub(),
            config=RunnerConfig(max_attempts=1, base_backoff_seconds=0.0),
        )
        try:
            await runner.run(batch, (metric,), unit_ids={"unit-does-not-exist"})
            raise AssertionError("Expected ValueError for unknown unit_ids")
        except ValueError as exc:
            assert "Unknown unit_ids" in str(exc)

    asyncio.run(_case())


def test_safe_response_error_codes_map_distinctly():
    async def _case() -> None:
        metric = MetricSpec(
            id="task_completion",
            name="TASK_COMPLETION_V1",
            version="v1",
            kind="binary",
            display_name="Task Completion",
        )
        expected_codes = {
            JudgeLengthExhaustedError("x"): "response_length_exhausted",
            JudgeEmptyResponseError("x"): "response_empty",
            JudgeContentFilteredError("x"): "response_content_filtered",
            JudgeMalformedJsonError("x"): "response_malformed_json",
            JudgeSkippedError("x"): "response_skipped",
            JudgeAuthenticationError("x"): "provider_authentication",
            JudgeTerminalHttpError(400): "provider_http_400",
            JudgeProviderError("x"): "provider_terminal",
        }

        for error, code in expected_codes.items():
            run = await EvaluationRunner(
                judge=_SafeResponseErrorJudge(error),
                config=RunnerConfig(max_attempts=1, base_backoff_seconds=0.0),
            ).run(_batch(1), (metric,))
            assert len(run.failures) == 1
            assert run.failures[0].code == code

    asyncio.run(_case())
