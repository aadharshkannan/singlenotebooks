from __future__ import annotations

import math

import pytest

from minhash_sampling import MinHashBuildError, MinHashConfig, MinHashSignatureProvider
from trace_sampling.model import SessionEvent, Trace


def _trace(
    trace_id: int,
    *,
    text: str,
    tool_name: str = "search",
    tool_output: str = "ok",
    arguments: dict | None = None,
) -> Trace:
    return Trace(
        trace_id=trace_id,
        agent_id="agent-a",
        timestamp=float(trace_id),
        signature=(tool_name,),
        span_count=1,
        duration_ms=1.0,
        status="ok",
        concept_id=0,
        events=(
            SessionEvent(role="system", text="policy"),
            SessionEvent(role="user", text=text),
            SessionEvent(role="assistant", text="working"),
            SessionEvent(role="tool", tool_name=tool_name, arguments=arguments or {"q": text}, output=tool_output),
            SessionEvent(role="assistant", text=tool_output),
        ),
    )


def test_signatures_are_deterministic_and_exclude_identity_fields() -> None:
    provider = MinHashSignatureProvider(MinHashConfig(permutations=128, seed=13))
    t1 = _trace(1, text="reset password")
    t2 = _trace(2, text="reset password")

    r1 = provider.build(t1)
    r2 = provider.build(t2)

    assert r1.signature == r2.signature
    assert r1.content_sha256 == r2.content_sha256
    assert r1.profile_id == r2.profile_id
    assert r1.shingle_count == r2.shingle_count
    assert provider.n_builds == 1
    assert provider.n_hits == 1
    assert provider._debug_shingles == {}


def test_content_hash_excludes_agent_identity() -> None:
    provider = MinHashSignatureProvider(MinHashConfig(permutations=64, seed=13))
    first = _trace(1, text="reset password")
    second = Trace(
        trace_id=2,
        agent_id="agent-b",
        timestamp=99.0,
        signature=first.signature,
        span_count=first.span_count,
        duration_ms=first.duration_ms,
        status=first.status,
        concept_id=first.concept_id,
        events=first.events,
    )
    assert provider.build(first).content_sha256 == provider.build(second).content_sha256


def test_calibration_shingle_retention_is_opt_in_and_bounded() -> None:
    provider = MinHashSignatureProvider(
        MinHashConfig(permutations=64, cache_size=1, retain_debug_shingles=True)
    )
    first = _trace(1, text="first content")
    second = _trace(2, text="second content")
    provider.build(first)
    provider.build(second)
    assert len(provider._debug_shingles) == 1
    assert 0.0 <= provider.shingle_jaccard(first, second) <= 1.0


def test_identical_tool_signature_but_disjoint_content_differs() -> None:
    provider = MinHashSignatureProvider(MinHashConfig(permutations=128, seed=13))
    a = provider.build(_trace(1, text="reset password", tool_name="search", tool_output="done"))
    b = provider.build(_trace(2, text="issue refund", tool_name="search", tool_output="paid"))

    assert a.signature != b.signature


def test_field_families_affect_signature() -> None:
    provider = MinHashSignatureProvider(MinHashConfig(permutations=128, seed=13))
    base = provider.build(_trace(1, text="hello", tool_name="search", tool_output="result", arguments={"q": "hello"}))
    changed_user = provider.build(_trace(2, text="different", tool_name="search", tool_output="result", arguments={"q": "hello"}))
    changed_tool = provider.build(_trace(3, text="hello", tool_name="lookup", tool_output="result", arguments={"q": "hello"}))
    changed_args = provider.build(_trace(4, text="hello", tool_name="search", tool_output="result", arguments={"q": "xyz"}))
    changed_out = provider.build(_trace(5, text="hello", tool_name="search", tool_output="alt-result", arguments={"q": "hello"}))

    assert base.signature != changed_user.signature
    assert base.signature != changed_tool.signature
    assert base.signature != changed_args.signature
    assert base.signature != changed_out.signature


def test_jaccard_estimate_tracks_exact_within_tolerance() -> None:
    provider = MinHashSignatureProvider(MinHashConfig(permutations=256, seed=13))
    t1 = _trace(1, text="reset account password quickly")
    t2 = _trace(2, text="recover account password quickly")

    r1 = provider.build(t1)
    r2 = provider.build(t2)
    est = sum(1 for a, b in zip(r1.signature, r2.signature) if a == b) / len(r1.signature)
    exact = provider.shingle_jaccard(t1, t2)

    assert math.fabs(est - exact) <= 0.10


def test_cache_hits_and_profile_separation() -> None:
    t = _trace(1, text="hello")
    p1 = MinHashSignatureProvider(MinHashConfig(permutations=64, seed=13))
    p2 = MinHashSignatureProvider(MinHashConfig(permutations=128, seed=13))

    r1a = p1.build(t)
    r1b = p1.build(t)
    r2 = p2.build(t)

    assert p1.n_hits == 1
    assert r1a.signature == r1b.signature
    assert r1a.profile_id != r2.profile_id


def test_utf8_and_representation_truncation_flag() -> None:
    cfg = MinHashConfig(representation_max_utf8_bytes=1200, permutations=64)
    provider = MinHashSignatureProvider(cfg)
    long_text = "你好" * 400
    t = _trace(1, text=long_text, tool_output=long_text)
    r = provider.build(t)

    assert r.representation_truncated is True
    assert provider.n_truncations >= 1


def test_empty_events_and_signature_fallback_source_still_builds() -> None:
    provider = MinHashSignatureProvider(MinHashConfig())
    # Events empty but tool signature present -> provider fallback event synthesis should work.
    trace = Trace(1, "agent-a", 0.0, ("search",), 1, 1.0, "ok", events=())
    record = provider.build(trace)
    assert len(record.signature) == provider.cfg.permutations

    # Completely empty evidence (no events, no signature) should fail.
    empty = Trace(2, "agent-a", 0.0, (), 0, 1.0, "ok", events=())
    with pytest.raises(MinHashBuildError):
        provider.build(empty)
