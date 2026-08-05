import json

import pytest

from trace_sampling.model import SessionEvent, Trace
from trace_sampling.token_representation import (
    CANONICAL_POLICY,
    CANONICAL_VERSION,
    CanonicalizationOptions,
    TokenSessionEvidencePacketBuilder,
    normalize_trace,
)


class CharTokenizer:
    name = "char"
    version = "1"

    def count(self, text: str) -> int:
        return len(text)


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


def test_complete_packet_is_unchanged_when_it_fits():
    trace = _trace([SessionEvent("user", "hello-世界")])
    tokenizer = CharTokenizer()

    roomy = normalize_trace(trace, tokenizer=tokenizer, max_tokens=10_000)
    exact = normalize_trace(trace, tokenizer=tokenizer, max_tokens=roomy.emitted_tokens)

    assert exact.canonical_json == roomy.canonical_json
    assert exact.truncated is False
    assert exact.original_tokens == exact.emitted_tokens
    assert exact.policy == CANONICAL_POLICY
    assert exact.version == CANONICAL_VERSION


def test_truncation_is_deterministic_under_hard_token_bound():
    trace = _trace(
        [
            SessionEvent("user", "世界" * 1_000),
            SessionEvent("assistant", "完了" * 1_000),
        ]
    )
    tokenizer = CharTokenizer()

    first = normalize_trace(trace, tokenizer=tokenizer, max_tokens=700)
    second = normalize_trace(trace, tokenizer=tokenizer, max_tokens=700)

    assert first.truncated is True
    assert first.emitted_tokens <= 700
    assert first.canonical_json == second.canonical_json
    json.loads(first.canonical_json)


def test_mandatory_evidence_is_retained_when_truncated():
    trace = _trace(
        [
            SessionEvent("system", "Policy context " + ("P" * 2_000)),
            SessionEvent("user", "Initial user goal " + ("G" * 6_000)),
            SessionEvent("user", "Later refinement " + ("R" * 3_000)),
            SessionEvent("tool", tool_name="commit", output="saved " + ("T" * 2_000)),
            SessionEvent("assistant", "Final outcome confirmed " + ("O" * 2_000)),
        ]
    )

    result = normalize_trace(trace, tokenizer=CharTokenizer(), max_tokens=1_300)
    events = json.loads(result.canonical_json)["session"]["events"]

    assert result.truncated is True
    assert all(events[index]["text"] for index in (0, 1, 2, 4))
    assert events[3]["output"]


def test_late_tool_result_gets_priority_on_ties():
    trace = _trace(
        [
            SessionEvent("user", "Complete the task"),
            SessionEvent("tool", tool_name="early", output="E" * 3_000),
            SessionEvent("tool", tool_name="late", output="L" * 3_000),
            SessionEvent("assistant", "Done"),
        ]
    )

    result = normalize_trace(trace, tokenizer=CharTokenizer(), max_tokens=900)
    events = json.loads(result.canonical_json)["session"]["events"]

    assert len(events[2]["output"]) >= len(events[1]["output"])


def test_structural_and_mandatory_floor_failures_are_explicit():
    tokenizer = CharTokenizer()
    trace = _trace(
        [
            SessionEvent("user", "G" * 2_000),
            SessionEvent("user", "R" * 2_000),
            SessionEvent("assistant", "O" * 2_000),
        ]
    )

    with pytest.raises(ValueError, match="non-content canonical structure"):
        normalize_trace(trace, tokenizer=tokenizer, max_tokens=5)

    structural_size = normalize_trace(
        _trace([SessionEvent("user"), SessionEvent("user"), SessionEvent("assistant")]),
        tokenizer=tokenizer,
        max_tokens=10_000,
    ).emitted_tokens

    with pytest.raises(ValueError, match="mandatory task-completion evidence"):
        normalize_trace(trace, tokenizer=tokenizer, max_tokens=structural_size + 20)


def test_cache_reuse_and_options_identity():
    tokenizer = CharTokenizer()
    options = CanonicalizationOptions(tokenizer=tokenizer, max_tokens=512)
    builder = TokenSessionEvidencePacketBuilder(options=options)
    trace = _trace([SessionEvent("user", "hello")])

    first = builder.build(trace)
    second = builder.build(trace)

    assert first is second
    assert builder.n_builds == 1
    assert builder.n_hits == 1


def test_semantically_identical_traces_are_byte_identical():
    tokenizer = CharTokenizer()
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

    first_representation = normalize_trace(first, tokenizer=tokenizer, max_tokens=10_000)
    same_representation = normalize_trace(same, tokenizer=tokenizer, max_tokens=10_000)

    assert first_representation.canonical_json == same_representation.canonical_json