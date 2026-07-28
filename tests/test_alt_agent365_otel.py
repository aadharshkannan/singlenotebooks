from __future__ import annotations

import json
from datetime import timezone

from trace_sampling_alt import normalize_agent365_records


def test_normalize_typed_otlp_prefers_invoke_agent_and_dedupes_spans():
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
                                    "traceId": "trace-1",
                                    "spanId": "span-invoke",
                                    "name": "invoke_agent",
                                    "startTimeUnixNano": "1735689600000000000",
                                    "endTimeUnixNano": "1735689601000000000",
                                    "attributes": {
                                        "gen_ai.agent.id": {"stringValue": "agent-a"},
                                        "gen_ai.conversation.id": {"stringValue": "conv-1"},
                                        "microsoft.session.id": {"stringValue": "sess-1"},
                                        "microsoft.channel.name": {"stringValue": "teams"},
                                        "gen_ai.input.messages": {
                                            "stringValue": json.dumps(
                                                {
                                                    "messages": [
                                                        {"role": "user", "content": "Where is ticket 42?"}
                                                    ]
                                                }
                                            )
                                        },
                                        "gen_ai.output.messages": {
                                            "stringValue": json.dumps(
                                                {
                                                    "messages": [
                                                        {"role": "assistant", "content": "Ticket 42 is in backlog."}
                                                    ]
                                                }
                                            )
                                        },
                                    },
                                    "status": {"code": 0},
                                },
                                {
                                    "traceId": "trace-1",
                                    "spanId": "span-infer",
                                    "name": "inference",
                                    "startTimeUnixNano": "1735689602000000000",
                                    "endTimeUnixNano": "1735689603000000000",
                                    "attributes": [
                                        {"key": "gen_ai.agent.id", "value": {"stringValue": "agent-a"}},
                                        {"key": "gen_ai.conversation.id", "value": {"stringValue": "conv-1"}},
                                        {
                                            "key": "input",
                                            "value": {
                                                "stringValue": json.dumps(
                                                    [
                                                        {"role": "user", "content": "Where is ticket 42?"},
                                                    ]
                                                )
                                            },
                                        },
                                        {
                                            "key": "output",
                                            "value": {
                                                "stringValue": json.dumps(
                                                    [
                                                        {"role": "assistant", "content": "Ticket 42 is in backlog."},
                                                    ]
                                                )
                                            },
                                        },
                                    ],
                                },
                                {
                                    "traceId": "trace-1",
                                    "spanId": "span-tool",
                                    "name": "execute_tool",
                                    "startTimeUnixNano": "1735689604000000000",
                                    "endTimeUnixNano": "1735689605000000000",
                                    "attributes": {
                                        "gen_ai.agent.id": {"stringValue": "agent-a"},
                                        "gen_ai.conversation.id": {"stringValue": "conv-1"},
                                        "tool.name": {"stringValue": "search"},
                                        "tool.input": {"stringValue": "ticket 42"},
                                        "tool.output": {"stringValue": "found in backlog"},
                                    },
                                },
                                {
                                    "traceId": "trace-1",
                                    "spanId": "span-tool",
                                    "name": "execute_tool",
                                    "startTimeUnixNano": "1735689604000000000",
                                    "endTimeUnixNano": "1735689605000000000",
                                    "attributes": {
                                        "gen_ai.agent.id": {"stringValue": "agent-a"},
                                        "gen_ai.conversation.id": {"stringValue": "conv-1"},
                                        "tool.name": {"stringValue": "search"},
                                    },
                                },
                            ]
                        }
                    ],
                }
            ]
        }
    }

    result = normalize_agent365_records([otlp])

    assert not result.issues
    assert len(result.units) == 1
    unit = result.units[0]

    assert unit.tenant_id == "tenant-1"
    assert unit.agent_id == "agent-a"
    assert unit.conversation_id == "conv-1"
    assert unit.session_id == "sess-1"
    assert unit.channel == "teams"
    assert unit.source_trace_ids == ("trace-1",)
    assert len(unit.turns) == 1
    assert unit.turns[0].user_text == "Where is ticket 42?"
    assert unit.turns[0].assistant_text == "Ticket 42 is in backlog."
    assert len(unit.tool_calls) == 1
    assert unit.tool_calls[0].name == "search"
    assert unit.started_at is not None
    assert unit.started_at.tzinfo == timezone.utc
    assert unit.ended_at is not None


def test_normalize_esp_and_kusto_variants_with_incomplete_issue():
    esp = {
        "documents": [
            {
                "jsonContent": json.dumps(
                    [
                        {
                            "microsoft.tenant.id": "tenant-2",
                            "gen_ai.agent.id": "agent-b",
                            "gen_ai.conversation.id": "conv-2",
                            "microsoft.session.id": "sess-2",
                            "gen_ai.execution.source.name": "copilot",
                            "name": "invoke_agent",
                            "traceId": "trace-esp-1",
                            "spanId": "span-esp-1",
                            "startTime": "2025-01-01T00:00:00Z",
                            "endTime": "2025-01-01T00:00:02Z",
                            "RequestMessages": [
                                {"role": "user", "parts": [{"type": "text", "text": "hello"}]}
                            ],
                            "ResponseMessages": [
                                {
                                    "role": "assistant",
                                    "parts": [
                                        {"type": "tool_call", "text": "calling weather tool"},
                                        {"type": "text", "text": "hi there"},
                                    ],
                                }
                            ],
                        }
                    ]
                )
            }
        ]
    }

    kusto_row = {
        "TenantId": "tenant-3",
        "AgentId": "agent-c",
        "ConversationId": "conv-3",
        "SessionIdentity": "sess-3",
        "ChannelName": "teams",
        "TimeGenerated": "2025-01-02T00:00:00Z",
        "CompletionTime": "2025-01-02T00:00:03Z",
        "RequestMessages": json.dumps({"messages": [{"role": "user", "content": "what time"}]}),
        "ResponseMessages": json.dumps({"messages": [{"role": "assistant", "content": "12:00 UTC"}]}),
        "traceId": "trace-kusto-1",
        "spanId": "span-kusto-1",
    }

    incomplete = {
        "TenantId": "tenant-3",
        "AgentId": "agent-c",
        "ConversationId": "conv-missing-assistant",
        "SessionIdentity": "sess-4",
        "TimeGenerated": "2025-01-02T01:00:00Z",
        "name": "invoke_agent",
        "RequestMessages": json.dumps([{"role": "user", "content": "only user"}]),
        "traceId": "trace-kusto-2",
        "spanId": "span-kusto-2",
    }

    missing_identity = {
        "ConversationId": "conv-no-identities",
        "name": "invoke_agent",
        "RequestMessages": "[]",
        "ResponseMessages": "[]",
        "traceId": "trace-kusto-3",
        "spanId": "span-kusto-3",
    }

    result = normalize_agent365_records([esp, kusto_row, incomplete, missing_identity])

    assert len(result.units) == 2
    by_conversation = {unit.conversation_id: unit for unit in result.units}

    assert "conv-2" in by_conversation
    assert by_conversation["conv-2"].channel == "copilot"
    assert by_conversation["conv-2"].turns[-1].assistant_text == "calling weather tool\nhi there"

    assert "conv-3" in by_conversation
    assert by_conversation["conv-3"].session_id == "sess-3"
    assert by_conversation["conv-3"].turns[-1].assistant_text == "12:00 UTC"

    issue_codes = {issue.code for issue in result.issues}
    assert "incomplete_session" in issue_codes
    assert "missing_identity" in issue_codes
    assert any(
        issue.code == "incomplete_session" and issue.source_kind == "session"
        for issue in result.issues
    )


def test_numeric_epoch_units_are_normalized_to_utc():
    rows = []
    for suffix, timestamp in (
        ("ms", 1735689600000),
        ("us", 1735689600000000),
        ("ns", 1735689600000000000),
    ):
        rows.append(
            {
                "TenantId": "tenant-time",
                "AgentId": "agent-time",
                "ConversationId": f"conv-{suffix}",
                "RequestMessages": '[{"role":"user","content":"hi"}]',
                "ResponseMessages": '[{"role":"assistant","content":"hello"}]',
                "TimeGenerated": timestamp,
            }
        )

    result = normalize_agent365_records(rows)

    assert {issue.code for issue in result.issues} == {"session_completion_inferred"}
    assert {unit.started_at.year for unit in result.units if unit.started_at} == {2025}


def test_strict_otlp_root_supports_synthetic_agent365_aliases_and_tools():
    document = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "microsoft.agent365.tenant.id",
                            "value": {"stringValue": "tenant-synthetic"},
                        },
                        {
                            "key": "gen_ai.agent.id",
                            "value": {"stringValue": "agent-synthetic"},
                        },
                        {
                            "key": "microsoft.agent365.channel",
                            "value": {"stringValue": "msteams:COPILOT"},
                        },
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "a" * 32,
                                "spanId": "b" * 16,
                                "name": "invoke_agent agent-synthetic",
                                "startTimeUnixNano": "1735689600000000000",
                                "endTimeUnixNano": "1735689601000000000",
                                "attributes": [
                                    {
                                        "key": "gen_ai.conversation.id",
                                        "value": {"stringValue": "conv-synthetic"},
                                    },
                                    {
                                        "key": "gen_ai.input.messages",
                                        "value": {
                                            "stringValue": '[{"role":"user","content":"find order"}]'
                                        },
                                    },
                                    {
                                        "key": "gen_ai.output.messages",
                                        "value": {
                                            "stringValue": '[{"role":"assistant","content":"order found"}]'
                                        },
                                    },
                                ],
                            },
                            {
                                "traceId": "a" * 32,
                                "spanId": "c" * 16,
                                "parentSpanId": "b" * 16,
                                "name": "execute_tool LookupOrder",
                                "startTimeUnixNano": "1735689600200000000",
                                "endTimeUnixNano": "1735689600300000000",
                                "attributes": [
                                    {
                                        "key": "gen_ai.conversation.id",
                                        "value": {"stringValue": "conv-synthetic"},
                                    },
                                    {
                                        "key": "gen_ai.tool.name",
                                        "value": {"stringValue": "LookupOrder"},
                                    },
                                    {
                                        "key": "gen_ai.tool.call.arguments",
                                        "value": {"stringValue": "{\"id\":42}"},
                                    },
                                    {
                                        "key": "gen_ai.tool.call.result",
                                        "value": {"stringValue": "{\"status\":\"found\"}"},
                                    },
                                ],
                            },
                        ]
                    }
                ],
            }
        ]
    }

    result = normalize_agent365_records([document])

    assert not result.issues
    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.tenant_id == "tenant-synthetic"
    assert unit.agent_id == "agent-synthetic"
    assert unit.channel == "msteams:COPILOT"
    assert unit.tool_calls[0].name == "LookupOrder"
    assert unit.tool_calls[0].input_text == '{"id":42}'
    assert unit.tool_calls[0].output_text == '{"status":"found"}'


def test_chat_operation_is_used_when_invoke_span_has_no_messages():
    shared = [
        {
            "key": "gen_ai.conversation.id",
            "value": {"stringValue": "conv-chat"},
        }
    ]
    document = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "microsoft.agent365.tenant.id",
                            "value": {"stringValue": "tenant-chat"},
                        },
                        {
                            "key": "gen_ai.agent.id",
                            "value": {"stringValue": "agent-chat"},
                        },
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "d" * 32,
                                "spanId": "e" * 16,
                                "name": "invoke_agent agent-chat",
                                "startTimeUnixNano": "1735689600000000000",
                                "endTimeUnixNano": "1735689601000000000",
                                "attributes": shared
                                + [
                                    {
                                        "key": "gen_ai.operation.name",
                                        "value": {"stringValue": "invoke_agent"},
                                    }
                                ],
                            },
                            {
                                "traceId": "d" * 32,
                                "spanId": "f" * 16,
                                "parentSpanId": "e" * 16,
                                "name": "chat agent-chat",
                                "startTimeUnixNano": "1735689600200000000",
                                "endTimeUnixNano": "1735689600900000000",
                                "attributes": shared
                                + [
                                    {
                                        "key": "gen_ai.operation.name",
                                        "value": {"stringValue": "chat"},
                                    },
                                    {
                                        "key": "gen_ai.input.messages",
                                        "value": {
                                            "stringValue": '[{"role":"user","content":"hello"}]'
                                        },
                                    },
                                    {
                                        "key": "gen_ai.output.messages",
                                        "value": {
                                            "stringValue": '[{"role":"assistant","content":"hi"}]'
                                        },
                                    },
                                ],
                            },
                        ]
                    }
                ],
            }
        ]
    }

    result = normalize_agent365_records([document])

    assert not result.issues
    assert len(result.units) == 1
    assert result.units[0].turns[0].user_text == "hello"
    assert result.units[0].turns[0].assistant_text == "hi"
