from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from random_sampling import (
    EvaluationWindow,
    SessionizationPolicy,
    filter_sessions_to_window,
    normalize_agent365_records,
)


BASE = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _span(
    *,
    tenant: str = "tenant-1",
    agent: str = "agent-1",
    conversation: str | None = "conv-1",
    session: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    trace: str = "trace-1",
    span: str = "span-1",
    channel: str | None = "teams",
    user: str | None = "hello",
    assistant: str | None = "hi",
) -> dict[str, object]:
    row: dict[str, object] = {
        "TenantId": tenant,
        "AgentId": agent,
        "traceId": trace,
        "spanId": span,
        "name": "invoke_agent",
        "ChannelName": channel,
    }
    if conversation is not None:
        row["ConversationId"] = conversation
    if session is not None:
        row["SessionIdentity"] = session
    if start is not None:
        row["TimeGenerated"] = start.isoformat().replace("+00:00", "Z")
    if end is not None:
        row["CompletionTime"] = end.isoformat().replace("+00:00", "Z")
    if user is not None:
        row["RequestMessages"] = json.dumps([{"role": "user", "content": user}])
    if assistant is not None:
        row["ResponseMessages"] = json.dumps([{"role": "assistant", "content": assistant}])
    return row


def test_same_session_id_across_multiple_conversations_merges_to_one_unit():
    rows = [
        _span(
            conversation="conv-a",
            session="sess-1",
            start=BASE,
            end=BASE + timedelta(minutes=1),
            trace="trace-a",
            span="span-a",
            channel="teams",
        ),
        _span(
            conversation="conv-b",
            session="sess-1",
            start=BASE + timedelta(minutes=2),
            end=BASE + timedelta(minutes=3),
            trace="trace-b",
            span="span-b",
            channel="copilot",
        ),
    ]

    result = normalize_agent365_records(rows)

    assert not result.issues
    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.session_id == "sess-1"
    assert unit.conversation_ids == ("conv-a", "conv-b")
    assert unit.conversation_id == "conv-a"
    assert unit.channel == "multi"
    assert unit.sessionization_kind == "session_id"
    assert unit.unit_id is not None
    assert unit.unit_id.startswith("session:")


def test_two_session_ids_in_same_conversation_split():
    rows = [
        _span(
            conversation="conv-1",
            session="sess-1",
            start=BASE,
            end=BASE + timedelta(minutes=1),
            trace="trace-1",
            span="span-1",
        ),
        _span(
            conversation="conv-1",
            session="sess-2",
            start=BASE + timedelta(minutes=2),
            end=BASE + timedelta(minutes=3),
            trace="trace-2",
            span="span-2",
        ),
    ]

    result = normalize_agent365_records(rows)

    assert not result.issues
    assert len(result.units) == 2
    assert {unit.session_id for unit in result.units} == {"sess-1", "sess-2"}


def test_inactivity_fallback_uses_strict_greater_than_timeout_boundary():
    rows = [
        _span(
            conversation="conv-fallback",
            session=None,
            start=BASE,
            end=BASE + timedelta(minutes=1),
            trace="trace-a",
            span="span-a",
        ),
        _span(
            conversation="conv-fallback",
            session=None,
            start=BASE + timedelta(minutes=30),
            end=BASE + timedelta(minutes=32),
            trace="trace-b",
            span="span-b",
        ),
        _span(
            conversation="conv-fallback",
            session=None,
            start=BASE + timedelta(minutes=63),
            end=BASE + timedelta(minutes=64),
            trace="trace-c",
            span="span-c",
        ),
    ]

    result = normalize_agent365_records(rows, SessionizationPolicy(inactivity_timeout=timedelta(minutes=30)))

    assert not result.issues
    assert len(result.units) == 2
    first, second = sorted(result.units, key=lambda unit: unit.started_at or BASE)
    assert first.started_at == BASE
    assert first.ended_at == BASE + timedelta(minutes=32)
    assert second.started_at == BASE + timedelta(minutes=63)


def test_session_id_scope_is_per_tenant_and_agent():
    rows = [
        _span(
            tenant="tenant-1",
            agent="agent-1",
            conversation="conv-a",
            session="sess-shared",
            start=BASE,
            end=BASE + timedelta(minutes=1),
            trace="trace-a",
            span="span-a",
        ),
        _span(
            tenant="tenant-1",
            agent="agent-2",
            conversation="conv-b",
            session="sess-shared",
            start=BASE,
            end=BASE + timedelta(minutes=1),
            trace="trace-b",
            span="span-b",
        ),
    ]

    result = normalize_agent365_records(rows)

    assert not result.issues
    assert len(result.units) == 2
    assert {(unit.tenant_id, unit.agent_id, unit.session_id) for unit in result.units} == {
        ("tenant-1", "agent-1", "sess-shared"),
        ("tenant-1", "agent-2", "sess-shared"),
    }
    assert len({unit.unit_id for unit in result.units}) == 2


def test_missing_conversation_is_allowed_when_session_id_exists():
    rows = [
        _span(
            conversation=None,
            session="sess-1",
            start=BASE,
            end=BASE + timedelta(minutes=1),
            trace="trace-1",
            span="span-1",
        )
    ]

    result = normalize_agent365_records(rows)

    assert not result.issues
    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.session_id == "sess-1"
    assert unit.conversation_id == ""
    assert unit.conversation_ids == ()


def test_missing_timestamps_form_deterministic_segment_with_uncertainty_issue():
    rows = [
        _span(
            conversation="conv-1",
            session=None,
            start=None,
            end=None,
            trace="trace-2",
            span="span-b",
        ),
        _span(
            conversation="conv-1",
            session=None,
            start=None,
            end=None,
            trace="trace-1",
            span="span-a",
        ),
    ]

    result = normalize_agent365_records(rows)

    assert len(result.units) == 1
    issue_codes = [issue.code for issue in result.issues]
    assert "fallback_sessionization_uncertainty" in issue_codes


def test_filter_sessions_to_window_uses_inferred_completion_and_applies_half_open_window():
    rows = [
        _span(
            conversation="conv-in",
            session="sess-in",
            start=BASE,
            end=BASE + timedelta(hours=23, minutes=59),
            trace="trace-in",
            span="span-in",
        ),
        _span(
            conversation="conv-out",
            session="sess-out",
            start=BASE,
            end=BASE + timedelta(hours=24),
            trace="trace-out",
            span="span-out",
        ),
        _span(
            conversation="conv-missing-end",
            session="sess-missing",
            start=BASE + timedelta(hours=1),
            end=None,
            trace="trace-missing",
            span="span-missing",
        ),
    ]

    normalized = normalize_agent365_records(rows)
    window = EvaluationWindow.ending_at(BASE + timedelta(hours=24))

    filtered = filter_sessions_to_window(normalized, window)

    assert {unit.session_id for unit in filtered.units} == {"sess-in", "sess-missing"}
    assert any(issue.code == "session_completion_inferred" for issue in filtered.issues)


def test_evaluation_window_default_24h_and_explicit_1h_half_open():
    rows = [
        _span(
            conversation="conv-default",
            session="sess-default",
            start=BASE,
            end=BASE + timedelta(hours=23, minutes=59),
            trace="trace-default",
            span="span-default",
        ),
        _span(
            conversation="conv-hour-in",
            session="sess-hour-in",
            start=BASE,
            end=BASE + timedelta(minutes=59),
            trace="trace-hour-in",
            span="span-hour-in",
        ),
        _span(
            conversation="conv-hour-edge",
            session="sess-hour-edge",
            start=BASE,
            end=BASE + timedelta(hours=1),
            trace="trace-hour-edge",
            span="span-hour-edge",
        ),
    ]

    normalized = normalize_agent365_records(rows)

    day_window = EvaluationWindow.ending_at(BASE + timedelta(hours=24))
    one_hour_window = EvaluationWindow.ending_at(BASE + timedelta(hours=1), timedelta(hours=1))

    day_filtered = filter_sessions_to_window(normalized, day_window)
    hour_filtered = filter_sessions_to_window(normalized, one_hour_window)

    assert {unit.session_id for unit in day_filtered.units} == {
        "sess-default",
        "sess-hour-in",
        "sess-hour-edge",
    }
    assert {unit.session_id for unit in hour_filtered.units} == {"sess-hour-in"}
