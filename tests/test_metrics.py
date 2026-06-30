import math
from trace_sampling.model import Trace
from trace_sampling.metrics import (
    signature_coverage, min_active_keep_rate, representativeness,
    kept_rate_timeseries,
)


def _t(tid, agent, ts, sig):
    return Trace(tid, agent, ts, sig, len(sig), 1.0, "ok")


def _stream():
    out = []
    tid = 0
    for ts in range(20):
        out.append(_t(tid, "a", float(ts), ("x",))); tid += 1
        out.append(_t(tid, "a", float(ts) + 0.5, ("y",))); tid += 1
        out.append(_t(tid, "b", float(ts), ("z",))); tid += 1
    return out


def test_coverage_full_when_all_kept():
    s = _stream()
    cov = signature_coverage(s, s)
    assert cov["a"] == 1.0 and cov["b"] == 1.0


def test_coverage_partial_when_one_signature_dropped():
    s = _stream()
    kept = [t for t in s if t.signature != ("y",)]
    cov = signature_coverage(s, kept)
    assert math.isclose(cov["a"], 0.5)


def test_min_active_keep_rate_zero_when_agent_starved():
    s = _stream()
    kept = [t for t in s if t.agent_id == "a"]   # b fully starved
    assert min_active_keep_rate(s, kept, window=5.0) == 0.0


def test_min_active_keep_rate_positive_when_all_served():
    s = _stream()
    kept = s[::2]
    assert min_active_keep_rate(s, kept, window=50.0) > 0.0


def test_representativeness_zero_divergence_when_proportional():
    s = _stream()
    rep = representativeness(s, s)
    assert rep["a"]["tv"] < 1e-9
    assert rep["a"]["kl"] < 1e-9


def test_kept_rate_timeseries_shapes():
    s = _stream()
    bucket = 5.0
    times, rates = kept_rate_timeseries(s, bucket=bucket)
    assert len(times) == len(rates)
    # rates are per-second; multiplying back by bucket recovers total count.
    assert abs(sum(rates) * bucket - len(s)) < 1e-9
