from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from trace_sampling_alt import EvaluationUnit, ToolCall, Turn, build_evidence_packet


def _unit_with_unicode() -> EvaluationUnit:
    return EvaluationUnit(
        tenant_id="tenant-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        session_id="sess-1",
        channel="teams",
        source_trace_ids=("trace-1",),
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        had_error=False,
        turns=(
            Turn(user_text="Hello \u03bb", assistant_text="Hi \U0001f31f"),
            Turn(user_text="Long " + "x" * 400, assistant_text="Reply " + "y" * 400),
        ),
        tool_calls=(
            ToolCall(name="search", input_text="i" * 300, output_text="o" * 300, status="ok"),
        ),
    )


def _parse_packet(packet_json: str) -> dict[str, object]:
    parsed = json.loads(packet_json)
    assert isinstance(parsed, dict)
    return parsed


def test_evidence_packet_deterministic_and_utf8_bounded():
    unit = _unit_with_unicode()
    packet_a = build_evidence_packet(unit, max_bytes=1024)
    packet_b = build_evidence_packet(unit, max_bytes=1024)
    payload = _parse_packet(packet_a.canonical_json)
    audit = payload["audit"]

    assert packet_a.version == "evidence-packet-v2"
    assert packet_a.canonical_json == packet_b.canonical_json
    assert packet_a.sha256 == packet_b.sha256
    assert packet_a.encoded_bytes <= 1024
    assert packet_a.original_bytes >= packet_a.encoded_bytes
    assert packet_a.truncated is True
    assert isinstance(packet_a.canonical_json.encode("utf-8"), bytes)
    assert payload["version"] == "evidence-packet-v2"
    assert payload["audit"]["version"] == "v2"
    assert packet_a.omitted_turn_count == audit["omitted_turn_count"]
    assert packet_a.omitted_tool_call_count == audit["omitted_tool_call_count"]


def test_evidence_packet_structural_long_session_budget_retains_required_slots_and_exact_omissions():
    turns = tuple(
        Turn(
            user_text=(
                "FIRST_USER_TASK_" + "\u03bb" * 40 + "_" + "u" * 120
                if i == 0
                else f"user-{i}-" + ("\u4f60\u597d" * 20)
            ),
            assistant_text=(
                f"assistant-{i}-" + "\U0001f31f" * 24
                if i < 239
                else "FINAL_ASSISTANT_OUTCOME_" + "\U0001f31f" * 50 + "_" + "z" * 200
            ),
        )
        for i in range(240)
    )
    tools = tuple(
        ToolCall(
            name=f"tool-{i}",
            input_text=f"input-{i}-" + ("\u03c0" * 50),
            output_text=(
                f"output-{i}-" + ("\u03a9" * 50)
                if i < 209
                else "LATEST_TOOL_OUTPUT_" + "\u03a9" * 120 + "_" + "out" * 90
            ),
            status="ok",
        )
        for i in range(210)
    )

    unit = EvaluationUnit(
        tenant_id="tenant-1",
        agent_id="agent-1",
        conversation_id="conv-main",
        conversation_ids=("conv-main", "conv-side-a", "conv-side-b"),
        session_id="sess-struct",
        channel="teams",
        source_trace_ids=tuple(f"trace-{i}" for i in range(6)),
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        had_error=False,
        turns=turns,
        tool_calls=tools,
    )

    packet = build_evidence_packet(unit, max_bytes=3072)
    payload = _parse_packet(packet.canonical_json)
    audit = payload["audit"]

    assert packet.encoded_bytes <= 3072
    assert payload["version"] == "evidence-packet-v2"
    assert payload["scope"]["unit_id"] == unit.unit_id
    assert payload["scope"]["session_id"] == "sess-struct"
    assert payload["scope"]["conversation_ids"] == sorted(["conv-main", "conv-side-a", "conv-side-b"])
    assert payload["scope"]["sessionization_kind"] == "session_id"
    assert payload["timing"]["had_error"] is False
    assert payload["trace_ids"] == list(unit.source_trace_ids)

    emitted_turns = payload["turns"]
    emitted_tools = payload["tool_calls"]
    assert emitted_turns[0]["original_index"] == 0
    assert emitted_turns[-1]["original_index"] == 239
    assert emitted_tools[-1]["original_index"] == 209

    first_user = emitted_turns[0]["user"]
    final_assistant = emitted_turns[-1]["assistant"]
    latest_tool = emitted_tools[-1]
    latest_tool_name = latest_tool["name"]
    latest_tool_output = latest_tool["output"]

    assert first_user
    assert final_assistant
    assert latest_tool_name
    assert latest_tool_output
    assert "FIRST_USER_TASK_" in first_user
    assert "FINAL_ASSISTANT_OUTCOME_" in final_assistant
    assert latest_tool_name.startswith("tool-209")
    assert "LATEST_TOOL_OUTPUT_" in latest_tool_output

    assert audit["original_turn_count"] == 240
    assert audit["emitted_turn_count"] == len(emitted_turns)
    assert audit["omitted_turn_count"] == 240 - len(emitted_turns)
    assert audit["original_tool_call_count"] == 210
    assert audit["emitted_tool_call_count"] == len(emitted_tools)
    assert audit["omitted_tool_call_count"] == 210 - len(emitted_tools)
    assert packet.omitted_turn_count == audit["omitted_turn_count"]
    assert packet.omitted_tool_call_count == audit["omitted_tool_call_count"]


def test_evidence_packet_full_fit_emits_all_and_zero_omissions():
    unit = _unit_with_unicode()
    packet = build_evidence_packet(unit, max_bytes=32768)
    payload = _parse_packet(packet.canonical_json)
    audit = payload["audit"]

    assert packet.truncated is False
    assert payload["version"] == "evidence-packet-v2"
    assert [turn["original_index"] for turn in payload["turns"]] == [0, 1]
    assert [call["original_index"] for call in payload["tool_calls"]] == [0]
    assert payload["turns"][0]["user"] == "Hello \u03bb"
    assert payload["turns"][1]["assistant"].startswith("Reply ")
    assert payload["tool_calls"][0]["name"] == "search"
    assert payload["tool_calls"][0]["status"] == "ok"
    assert audit["omitted_turn_count"] == 0
    assert audit["omitted_tool_call_count"] == 0
    assert packet.omitted_turn_count == 0
    assert packet.omitted_tool_call_count == 0


def test_evidence_packet_too_small_structural_floor_errors():
    unit = _unit_with_unicode()
    with pytest.raises(ValueError, match="structural floor"):
        build_evidence_packet(unit, max_bytes=40)


def test_evidence_packet_too_small_mandatory_floor_errors():
    unit = EvaluationUnit(
        tenant_id="tenant-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        session_id="sess-1",
        channel="teams",
        source_trace_ids=("trace-1",),
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        had_error=False,
        turns=(Turn(user_text="\u03bb" * 64, assistant_text="\U0001f31f" * 64),),
        tool_calls=(ToolCall(name="tool-x", input_text="", output_text="\u03a9" * 80, status="ok"),),
    )

    found_budget: int | None = None
    for budget in range(1, 2048):
        try:
            build_evidence_packet(unit, max_bytes=budget)
        except ValueError as exc:
            if "mandatory evidence floor" in str(exc):
                found_budget = budget
                break

    assert found_budget is not None
    with pytest.raises(ValueError, match="mandatory evidence floor"):
        build_evidence_packet(unit, max_bytes=found_budget)


def test_evidence_packet_utf8_marker_and_prefix_behavior():
    unit = EvaluationUnit(
        tenant_id="tenant-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        session_id="sess-utf8",
        channel="teams",
        source_trace_ids=("trace-1",),
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        had_error=False,
        turns=(
            Turn(user_text="FIRST_" + "\u4f60\u597d" * 100, assistant_text="A"),
            Turn(user_text="B", assistant_text="FINAL_" + "\U0001f31f" * 120),
        ),
        tool_calls=(
            ToolCall(name="tool-end", input_text="\u03c0" * 120, output_text="OUTPUT_" + "\u03a9" * 120, status="ok"),
        ),
    )

    packet = build_evidence_packet(unit, max_bytes=900)
    payload = _parse_packet(packet.canonical_json)
    turns = payload["turns"]
    tools = payload["tool_calls"]

    final_assistant = next(turn for turn in turns if turn["original_index"] == 1)["assistant"]
    first_user = next(turn for turn in turns if turn["original_index"] == 0)["user"]
    latest_output = tools[-1]["output"]

    for value in (final_assistant, first_user, latest_output):
        assert isinstance(value.encode("utf-8"), bytes)

    assert packet.encoded_bytes <= 900
    assert packet.truncated is True
    assert final_assistant.startswith("FINAL_")
    assert latest_output.startswith("OUTPUT_")
    assert (" [truncated]" in final_assistant) or (" [truncated]" in latest_output) or (" [truncated]" in first_user)


def test_evidence_packet_minimal_structural_floor_emits_valid_json():
    unit = EvaluationUnit(
        tenant_id="tenant-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        session_id="sess-1",
        channel="teams",
        source_trace_ids=("trace-1",),
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        had_error=False,
        turns=(Turn(user_text="Q" * 10, assistant_text="R" * 10),),
        tool_calls=(),
    )

    packet = build_evidence_packet(unit, max_bytes=700)
    payload = _parse_packet(packet.canonical_json)
    assert payload["version"] == "evidence-packet-v2"
    assert payload["turns"][0]["original_index"] == 0
    assert packet.encoded_bytes <= 700
