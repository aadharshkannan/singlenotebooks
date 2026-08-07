from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json

from random_sampling import DeterministicJudgeStub, EvaluationWindow, RandomSamplingPipeline, SamplePolicy


BASE = datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc)


def _flat_session(
    *,
    tenant: str,
    agent: str,
    conversation: str,
    session: str,
    ended_at: datetime,
) -> dict[str, object]:
    started_at = ended_at - timedelta(minutes=2)
    return {
        "TenantId": tenant,
        "AgentId": agent,
        "ConversationId": conversation,
        "SessionIdentity": session,
        "name": "invoke_agent",
        "TimeGenerated": started_at.isoformat().replace("+00:00", "Z"),
        "CompletionTime": ended_at.isoformat().replace("+00:00", "Z"),
        "RequestMessages": json.dumps({"messages": [{"role": "user", "content": "q"}]}),
        "ResponseMessages": json.dumps({"messages": [{"role": "assistant", "content": "a"}]}),
    }


def test_pipeline_end_to_end_otlp_and_flat_with_issue_propagation():
    async def _case() -> None:
        otlp = {
            "traceRequest": {
                "resourceSpans": [
                    {
                        "resource": {
                            "attributes": [
                                {"key": "microsoft.tenant.id", "value": {"stringValue": "tenant-1"}},
                            ]
                        },
                        "scopeSpans": [
                            {
                                "spans": [
                                    {
                                        "traceId": "t-1",
                                        "spanId": "s-1",
                                        "name": "invoke_agent",
                                        "startTimeUnixNano": str(int((BASE - timedelta(minutes=5)).timestamp() * 1_000_000_000)),
                                        "endTimeUnixNano": str(int((BASE - timedelta(minutes=4)).timestamp() * 1_000_000_000)),
                                        "attributes": {
                                            "gen_ai.agent.id": {"stringValue": "agent-a"},
                                            "gen_ai.conversation.id": {"stringValue": "conv-1"},
                                            "microsoft.session.id": {"stringValue": "sess-1"},
                                            "gen_ai.input.messages": {
                                                "stringValue": json.dumps(
                                                    {"messages": [{"role": "user", "content": "hello"}]}
                                                )
                                            },
                                            "gen_ai.output.messages": {
                                                "stringValue": json.dumps(
                                                    {"messages": [{"role": "assistant", "content": "world"}]}
                                                )
                                            },
                                        },
                                    }
                                ]
                            }
                        ],
                    }
                ]
            }
        }

        flat = {
            "TenantId": "tenant-1",
            "AgentId": "agent-a",
            "ConversationId": "conv-2",
            "SessionIdentity": "sess-2",
            "name": "invoke_agent",
            "TimeGenerated": (BASE - timedelta(minutes=3)).isoformat().replace("+00:00", "Z"),
            "CompletionTime": (BASE - timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
            "RequestMessages": json.dumps({"messages": [{"role": "user", "content": "q"}]}),
            "ResponseMessages": json.dumps({"messages": [{"role": "assistant", "content": "a"}]}),
        }

        broken = {
            "ConversationId": "conv-x",
            "name": "invoke_agent",
            "RequestMessages": "[]",
            "ResponseMessages": "[]",
        }

        pipeline = RandomSamplingPipeline(judge=DeterministicJudgeStub())
        result = await pipeline.run([otlp, flat, broken], window_end=BASE)

        assert len(result.normalization.units) == 2
        assert len(result.sample_batch.agents) == 1
        assert result.evaluation.request_count >= 2
        assert result.report.version == "random-report-v1"
        assert result.report.ingest_issue_count >= 1

    asyncio.run(_case())


def test_pipeline_default_24h_with_explicit_window_end_and_half_open_boundary():
    async def _case() -> None:
        rows = [
            _flat_session(
                tenant="tenant-1",
                agent="agent-a",
                conversation="conv-in",
                session="sess-in",
                ended_at=BASE - timedelta(minutes=1),
            ),
            _flat_session(
                tenant="tenant-1",
                agent="agent-a",
                conversation="conv-edge",
                session="sess-edge",
                ended_at=BASE,
            ),
            _flat_session(
                tenant="tenant-1",
                agent="agent-a",
                conversation="conv-out",
                session="sess-out",
                ended_at=BASE - timedelta(hours=24, seconds=1),
            ),
        ]

        pipeline = RandomSamplingPipeline(judge=DeterministicJudgeStub())
        result = await pipeline.run(rows, window_end=BASE)

        assert result.sample_batch.evaluation_window is not None
        window = result.sample_batch.evaluation_window
        assert window is not None
        assert window.start_at == BASE - timedelta(hours=24)
        assert window.end_at == BASE
        kept = {unit.session_id for unit in result.normalization.units}
        assert kept == {"sess-in"}

    asyncio.run(_case())


def test_pipeline_configurable_one_hour_window_filters_sessions_outside_window():
    async def _case() -> None:
        rows = [
            _flat_session(
                tenant="tenant-1",
                agent="agent-a",
                conversation="conv-hour-in",
                session="sess-hour-in",
                ended_at=BASE - timedelta(minutes=1),
            ),
            _flat_session(
                tenant="tenant-1",
                agent="agent-a",
                conversation="conv-hour-out",
                session="sess-hour-out",
                ended_at=BASE - timedelta(hours=1, seconds=1),
            ),
        ]

        pipeline = RandomSamplingPipeline(
            judge=DeterministicJudgeStub(),
            window_duration=timedelta(hours=1),
        )
        result = await pipeline.run(rows, window_end=BASE)

        assert result.sample_batch.evaluation_window is not None
        assert result.sample_batch.evaluation_window.duration == timedelta(hours=1)
        kept = {unit.session_id for unit in result.normalization.units}
        assert kept == {"sess-hour-in"}

    asyncio.run(_case())


def test_default_schedule_windows_align_to_completed_utc_boundaries():
    now = datetime(2025, 1, 2, 15, 37, 12, tzinfo=timezone.utc)

    daily = RandomSamplingPipeline(judge=DeterministicJudgeStub())
    hourly = RandomSamplingPipeline(
        judge=DeterministicJudgeStub(),
        window_duration=timedelta(hours=1),
    )

    daily_window = daily.default_evaluation_window(now)
    hourly_window = hourly.default_evaluation_window(now)

    assert daily_window.start_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert daily_window.end_at == datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert hourly_window.start_at == datetime(2025, 1, 2, 14, tzinfo=timezone.utc)
    assert hourly_window.end_at == datetime(2025, 1, 2, 15, tzinfo=timezone.utc)


def test_pipeline_uses_explicit_evaluation_window_for_deterministic_replay():
    async def _case() -> None:
        rows = [
            _flat_session(
                tenant="tenant-1",
                agent="agent-a",
                conversation="conv-a",
                session="sess-a",
                ended_at=BASE - timedelta(hours=2),
            ),
            _flat_session(
                tenant="tenant-1",
                agent="agent-a",
                conversation="conv-b",
                session="sess-b",
                ended_at=BASE - timedelta(hours=3),
            ),
        ]

        explicit_window = EvaluationWindow(
            start_at=BASE - timedelta(hours=3),
            end_at=BASE - timedelta(hours=1),
        )
        pipeline = RandomSamplingPipeline(judge=DeterministicJudgeStub())
        result = await pipeline.run(rows, evaluation_window=explicit_window)

        assert result.sample_batch.evaluation_window == explicit_window
        assert {unit.session_id for unit in result.normalization.units} == {"sess-a", "sess-b"}

    asyncio.run(_case())


def test_pipeline_cochran_population_is_filtered_per_agent_session_count():
    async def _case() -> None:
        rows = [
            _flat_session(
                tenant="tenant-1",
                agent="agent-a",
                conversation="conv-a1",
                session="sess-a1",
                ended_at=BASE - timedelta(minutes=10),
            ),
            _flat_session(
                tenant="tenant-1",
                agent="agent-a",
                conversation="conv-a2",
                session="sess-a2",
                ended_at=BASE - timedelta(minutes=20),
            ),
            _flat_session(
                tenant="tenant-1",
                agent="agent-a",
                conversation="conv-a-old",
                session="sess-a-old",
                ended_at=BASE - timedelta(days=2),
            ),
            _flat_session(
                tenant="tenant-1",
                agent="agent-b",
                conversation="conv-b1",
                session="sess-b1",
                ended_at=BASE - timedelta(minutes=15),
            ),
            _flat_session(
                tenant="tenant-1",
                agent="agent-b",
                conversation="conv-b-old",
                session="sess-b-old",
                ended_at=BASE - timedelta(days=3),
            ),
        ]

        pipeline = RandomSamplingPipeline(
            judge=DeterministicJudgeStub(),
            sample_policy=SamplePolicy(),
        )
        result = await pipeline.run(rows, window_end=BASE)

        plans = {
            (agent_sample.agent.tenant_id, agent_sample.agent.agent_id): agent_sample.plan
            for agent_sample in result.sample_batch.agents
        }
        assert plans[("tenant-1", "agent-a")].population == 2
        assert plans[("tenant-1", "agent-a")].selected == 2
        assert plans[("tenant-1", "agent-b")].population == 1
        assert plans[("tenant-1", "agent-b")].selected == 1

    asyncio.run(_case())
