import numpy as np
from trace_sampling.model import AgentConfig
from trace_sampling.generator import generate_stream


def _configs():
    return [
        AgentConfig("fast_lowvar", velocity=10.0, vocab_size=2, zipf_s=2.0),
        AgentConfig("slow_highvar", velocity=1.0, vocab_size=20, zipf_s=0.6),
        AgentConfig("late", velocity=5.0, vocab_size=5, zipf_s=1.0, start_time=5.0),
    ]


def test_stream_is_time_sorted_and_deterministic():
    s1 = generate_stream(_configs(), duration=10.0, seed=7)
    s2 = generate_stream(_configs(), duration=10.0, seed=7)
    assert [t.trace_id for t in s1] == [t.trace_id for t in s2]
    assert [t.timestamp for t in s1] == [t.timestamp for t in s2]
    assert [t.signature for t in s1] == [t.signature for t in s2]
    assert [t.agent_id for t in s1] == [t.agent_id for t in s2]
    ts = [t.timestamp for t in s1]
    assert ts == sorted(ts)
    assert all(0.0 <= t.timestamp <= 10.0 for t in s1)


def test_late_agent_only_appears_after_start_time():
    s = generate_stream(_configs(), duration=10.0, seed=7)
    late = [t for t in s if t.agent_id == "late"]
    assert late, "late agent should emit some traces"
    assert min(t.timestamp for t in late) >= 5.0


def test_velocity_controls_relative_volume():
    s = generate_stream(_configs(), duration=20.0, seed=3)
    counts = {}
    for t in s:
        counts[t.agent_id] = counts.get(t.agent_id, 0) + 1
    assert counts["fast_lowvar"] > counts["slow_highvar"]


def test_variety_controls_distinct_signatures():
    s = generate_stream(_configs(), duration=30.0, seed=3)
    distinct = {}
    for t in s:
        distinct.setdefault(t.agent_id, set()).add(t.signature)
    assert len(distinct["slow_highvar"]) > len(distinct["fast_lowvar"])
