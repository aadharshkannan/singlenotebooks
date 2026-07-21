import numpy as np
import pandas as pd
import pytest

from trace_sampling.model import Trace
from trace_sampling.value_sim import (make_smooth_value_field,
                                       run_value_simulation)


class _Cache:
    """dict-backed stand-in for EmbeddingCache: supports `sig in cache` and get()."""

    def __init__(self, mapping):
        self._m = dict(mapping)

    def __contains__(self, sig):
        return sig in self._m

    def get(self, sig):
        return self._m[sig]


class _RunResult:
    """Minimal stand-in for eval_harness.RunResult: exposes .log and .index._cache."""

    def __init__(self, log, cache):
        self.log = log
        self.index = type("Idx", (), {"_cache": cache})()


def _two_cluster_run():
    """A stream with two well-separated clusters for one agent. The first trace of
    each cluster is KEPT (judged); the rest are DROPPED. Cluster c1 sits at high
    value, c2 at low value, so a global agent-mean fill is badly wrong while
    cluster-local IDW is accurate."""
    v1 = np.array([1.0, 0.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0, 0.0])
    cache = _Cache({("s1",): v1, ("s2",): v2})
    # 6x c1 then 4x c2; keep the first occurrence of each cluster.
    plan = [("s1", "c1")] * 6 + [("s2", "c2")] * 4
    rows, stream = [], []
    seen = set()
    for tid, (sig_tok, cid) in enumerate(plan):
        sig = (sig_tok,)
        kept = cid not in seen
        seen.add(cid)
        stream.append(Trace(tid, "a", float(tid), sig, 1, 1.0, "ok"))
        rows.append(dict(kept=kept, key_kind="cluster", variety_key=cid))
    return stream, _RunResult(pd.DataFrame(rows), cache)


def test_field_is_deterministic_and_in_unit_range():
    f1 = make_smooth_value_field(dim=16, seed=123)
    f2 = make_smooth_value_field(dim=16, seed=123)
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.normal(size=16)
        a, b = f1(v), f2(v)
        assert a == b
        assert 0.0 <= a <= 1.0


def test_idw_beats_agent_mean_fill_when_clusters_differ():
    stream, run = _two_cluster_run()
    field = lambda vec: 0.9 if vec[0] > 0.5 else 0.1
    res = run_value_simulation(stream, run, field=field, k=8, sigma=0.0, seed=1)
    # 10 traces, 2 kept -> 8 dropped embedded traces scored.
    assert res.n_dropped == 8
    assert res.idw_coverage == 1.0
    # Cluster-local IDW recovers the near-true value; the global agent-mean fill
    # (~0.5) is far off for both clusters.
    assert res.idw_mse < res.agent_fill_mse
    assert res.idw_mse < 1e-6
    # The causal agent-mean fill is wrong for the low-value c2 drops (its running
    # mean is pulled toward the high-value c1 evals judged first).
    assert res.agent_fill_mse > 0.05


def test_per_trace_and_per_agent_shapes():
    stream, run = _two_cluster_run()
    field = lambda vec: 0.9 if vec[0] > 0.5 else 0.1
    res = run_value_simulation(stream, run, field=field, sigma=0.0, seed=1)
    assert set(res.per_trace.columns) == {
        "agent_id", "true", "idw_pred", "agent_fill_pred", "provenance"}
    assert len(res.per_trace) == res.n_dropped
    assert list(res.per_agent.index) == ["a"]
    assert int(res.per_agent.loc["a", "n_dropped"]) == res.n_dropped


def test_stream_length_mismatch_raises():
    stream, run = _two_cluster_run()
    with pytest.raises(ValueError):
        run_value_simulation(stream[:-1], run, field=lambda v: 0.5)
