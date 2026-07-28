import json

import pytest

from trace_sampling.model import SessionEvent, Trace
from trace_sampling.representation import (
    CANONICAL_POLICY,
    CANONICAL_VERSION,
    SessionEvidencePacketBuilder,
    normalize_trace,
)


def _trace(events, trace_id=1):
    return Trace(
        trace_id=trace_id,
        agent_id="agent-one",
        timestamp=1_774_761_600.0,
        signature=("search", "write"),
        span_count=2,
        duration_ms=125.0,
        status="ok",
        events=tuple(events),
    )


def test_semantically_identical_traces_are_byte_identical():
    first = _trace(
        [
            SessionEvent("user", "hello\r\nworld"),
            SessionEvent("tool", tool_name="search", arguments={"b": 2, "a": 1}),
        ]
    )
    same = _trace(
        [
            SessionEvent("user", "hello\nworld"),
            SessionEvent("tool", tool_name="search", arguments={"a": 1, "b": 2}),
        ]
    )

    first_representation = normalize_trace(first, max_utf8_bytes=10_000)
    same_representation = normalize_trace(same, max_utf8_bytes=10_000)

    assert first_representation.canonical_json == same_representation.canonical_json
    assert first_representation.policy == CANONICAL_POLICY
    assert first_representation.version == CANONICAL_VERSION
    assert first_representation.truncated is False


def test_complete_packet_is_unchanged_when_it_fits():
    trace = _trace([SessionEvent("user", "hello-世界")])

    roomy = normalize_trace(trace, max_utf8_bytes=10_000)
    exact = normalize_trace(trace, max_utf8_bytes=roomy.emitted_utf8_bytes)

    assert exact.canonical_json == roomy.canonical_json
    assert exact.truncated is False
    assert exact.original_utf8_bytes == exact.emitted_utf8_bytes


def test_truncation_is_deterministic_bounded_and_utf8_safe():
    trace = _trace(
        [
            SessionEvent("user", "世界" * 2_000),
            SessionEvent("assistant", "完了" * 2_000),
        ]
    )

    first = normalize_trace(trace, max_utf8_bytes=800)
    second = normalize_trace(trace, max_utf8_bytes=800)
    events = json.loads(first.canonical_json)["session"]["events"]

    assert first.truncated is True
    assert first.emitted_utf8_bytes <= 800
    assert first.canonical_json == second.canonical_json
    assert len(first.canonical_json.encode("utf-8")) == first.emitted_utf8_bytes
    assert events[0]["text"] and events[1]["text"]
    events[0]["text"].encode("utf-8", errors="strict")
    events[1]["text"].encode("utf-8", errors="strict")


def test_large_arguments_are_truncated_instead_of_rejecting_trace():
    trace = _trace(
        [
            SessionEvent("user", "find it"),
            SessionEvent("tool", tool_name="search", arguments={"query": "世界" * 2_000}),
        ]
    )

    result = normalize_trace(trace, max_utf8_bytes=700)
    arguments_json = json.loads(result.canonical_json)["session"]["events"][1][
        "arguments_json"
    ]

    assert result.truncated is True
    assert result.emitted_utf8_bytes <= 700
    assert arguments_json
    assert len(arguments_json) < len(json.dumps({"query": "世界" * 2_000}))


def test_weighted_allocation_preserves_late_outcome_and_tool_result():
    trace = _trace(
        [
            SessionEvent("user", "A" * 15_000),
            SessionEvent(
                "tool",
                tool_name="resolver",
                arguments={"query": "latest status"},
                output='{"status":"complete","evidence":"late-tool-result"}',
            ),
            SessionEvent(
                "assistant",
                "Final outcome: task completed successfully with confirmation.",
            ),
        ]
    )

    result = normalize_trace(trace, max_utf8_bytes=900)
    events = json.loads(result.canonical_json)["session"]["events"]

    assert events[0]["text"] != trace.events[0].text
    assert events[1]["output"].startswith('{"status":"complete"')
    assert events[2]["text"].startswith("Final outcome")


def test_all_mandatory_evidence_classes_receive_an_extract():
    trace = _trace(
        [
            SessionEvent("system", "Policy context " + ("P" * 2_000)),
            SessionEvent("user", "Initial user goal " + ("G" * 8_000)),
            SessionEvent("user", "Later refinement " + ("R" * 4_000)),
            SessionEvent("tool", tool_name="commit", output="saved " + ("T" * 2_000)),
            SessionEvent("assistant", "Final outcome confirmed " + ("O" * 2_000)),
        ]
    )

    result = normalize_trace(trace, max_utf8_bytes=1_400)
    events = json.loads(result.canonical_json)["session"]["events"]

    assert result.truncated is True
    assert all(events[index]["text"] for index in (0, 1, 2, 4))
    assert events[3]["output"]


def test_late_tool_results_receive_budget_before_early_results():
    trace = _trace(
        [
            SessionEvent("user", "Complete the task"),
            SessionEvent("tool", tool_name="early", output="E" * 3_000),
            SessionEvent("tool", tool_name="late", output="L" * 3_000),
            SessionEvent("assistant", "Done"),
        ]
    )

    result = normalize_trace(trace, max_utf8_bytes=950)
    events = json.loads(result.canonical_json)["session"]["events"]

    assert len(events[2]["output"]) >= len(events[1]["output"])


def test_structural_and_mandatory_floor_failures_are_explicit():
    trace = _trace(
        [
            SessionEvent("user", "G" * 2_000),
            SessionEvent("user", "R" * 2_000),
            SessionEvent("assistant", "O" * 2_000),
        ]
    )

    with pytest.raises(ValueError, match="non-content canonical structure"):
        normalize_trace(trace, max_utf8_bytes=5)

    structural_size = normalize_trace(
        _trace([SessionEvent("user"), SessionEvent("user"), SessionEvent("assistant")]),
        max_utf8_bytes=10_000,
    ).emitted_utf8_bytes
    with pytest.raises(ValueError, match="mandatory task-completion evidence"):
        normalize_trace(trace, max_utf8_bytes=structural_size + 20)


def test_legacy_representation_identity_is_rejected():
    with pytest.raises(ValueError, match="unsupported canonical representation"):
        normalize_trace(
            _trace([SessionEvent("user", "hello")]),
            max_utf8_bytes=10_000,
            policy="complete_session_content_truncate",
            version="1.0",
        )


def test_packet_builder_reuses_same_representation_object():
    builder = SessionEvidencePacketBuilder()
    trace = _trace([SessionEvent("user", "hello")])

    first = builder.build(trace)
    second = builder.build(trace)

    assert first is second
    assert builder.n_builds == 1
    assert builder.n_hits == 1