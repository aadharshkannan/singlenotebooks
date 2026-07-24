import numpy as np
import pytest
from trace_sampling.model import Trace
from trace_sampling.variety import VarietyKey, VarietyObservation
from trace_sampling.value_reservoir import ClusterValueReservoir
from trace_sampling.value_pipeline import ValuePipeline, TraceValue


class _FakeSampler:
    """Minimal sampler stub: decide() returns a queued verdict and sets
    last_observation to a preset VarietyObservation."""

    def __init__(self):
        self._verdicts = []
        self._obs = []
        self.last_observation = None

    def queue(self, keep: bool, key: VarietyKey):
        self._verdicts.append(keep)
        self._obs.append(VarietyObservation(key, rarity=0.0, novelty=0.0))

    def decide(self, trace):
        self.last_observation = self._obs.pop(0)
        return self._verdicts.pop(0)


class _Cache:
    """dict-backed stand-in for EmbeddingCache: supports `sig in cache` and get()."""

    def __init__(self, mapping=None):
        self._m = dict(mapping or {})
        self.get_calls = []

    def __contains__(self, sig):
        return sig in self._m

    def get(self, sig):
        self.get_calls.append(sig)
        return self._m[sig]


def _trace(tid=1, agent="a", sig=("search",)):
    return Trace(tid, agent, 0.0, sig, len(sig), 1.0, "ok")


def test_drop_path_returns_immediate_imputed_value():
    sig = ("search",)
    vec = np.array([1.0, 0.0])
    sampler = _FakeSampler()
    sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    res.record_eval("c1", "a", vec, 0.7)          # one judged donor
    cache = _Cache({sig: vec})
    emitted = []
    pipe = ValuePipeline(sampler, cache, res, submit_judge=None, on_value=emitted.append)

    tv = pipe.process(_trace(sig=sig))
    assert tv.kept is False
    assert tv.provenance == "idw"
    assert tv.value == pytest.approx(0.7, abs=1e-3)
    assert tv.conditional_geodesic_bounds is not None
    assert tv.conditional_geodesic_bounds.is_confidence_interval is False
    assert emitted == [tv]


def test_keep_path_pending_then_judged_and_is_causal():
    sig = ("edit",)
    vec = np.array([0.0, 1.0])
    sampler = _FakeSampler()
    # first a keep, then a later drop in the same cluster
    sampler.queue(keep=True, key=VarietyKey("cluster", "c1"))
    sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    cache = _Cache({sig: vec})

    pending_judge = {}

    def submit_judge(trace, on_done):
        pending_judge[trace.trace_id] = on_done   # defer -> async

    emitted = []
    pipe = ValuePipeline(sampler, cache, res, submit_judge, on_value=emitted.append)

    tv_keep = pipe.process(_trace(tid=1, sig=sig))
    assert tv_keep.kept is True and tv_keep.value is None and tv_keep.provenance == "pending"
    assert tv_keep.conditional_geodesic_bounds is None

    # BEFORE the judge returns, a drop cannot use the pending eval -> not idw
    tv_drop_early = pipe.process(_trace(tid=2, sig=sig))
    assert tv_drop_early.provenance != "idw"
    assert tv_drop_early.conditional_geodesic_bounds is None

    # judge returns -> recorded, "judged" emitted
    pending_judge[1](0.9)
    assert emitted[-1] == TraceValue(1, True, 0.9, "judged")

    # a NEW drop now sees the donor -> idw
    sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    tv_drop_late = pipe.process(_trace(tid=3, sig=sig))
    assert tv_drop_late.provenance == "idw"
    assert tv_drop_late.value == pytest.approx(0.9, abs=1e-3)
    assert tv_drop_late.conditional_geodesic_bounds is not None


def test_synchronous_judge_emits_pending_before_judged():
    sig = ("fetch",)
    vec = np.array([1.0, 0.0])
    sampler = _FakeSampler()
    sampler.queue(keep=True, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    cache = _Cache({sig: vec})
    emitted = []

    def sync_judge(trace, on_done):
        on_done(0.6)                              # fire inline (synchronous judge)

    pipe = ValuePipeline(sampler, cache, res, sync_judge, on_value=emitted.append)
    pipe.process(_trace(tid=1, sig=sig))
    # natural order preserved even for a synchronous callback
    assert [tv.provenance for tv in emitted] == ["pending", "judged"]
    assert emitted[0].conditional_geodesic_bounds is None
    assert emitted[1].conditional_geodesic_bounds is None


def test_idw_drop_band_uses_only_completed_judged_observations_causally():
    sig = ("causal",)
    vec = np.array([1.0, 0.0])
    sampler = _FakeSampler()
    sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    sampler.queue(keep=True, key=VarietyKey("cluster", "c1"))
    sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    cache = _Cache({sig: vec})
    res = ClusterValueReservoir()
    pending = {}

    def submit_judge(trace, on_done):
        pending[trace.trace_id] = on_done

    pipe = ValuePipeline(sampler, cache, res, submit_judge=submit_judge)

    first_drop = pipe.process(_trace(tid=1, sig=sig))
    assert first_drop.provenance != "idw"
    assert first_drop.conditional_geodesic_bounds is None

    kept_pending = pipe.process(_trace(tid=2, sig=sig))
    assert kept_pending.provenance == "pending"
    assert kept_pending.conditional_geodesic_bounds is None

    second_drop_before_judge = pipe.process(_trace(tid=3, sig=sig))
    assert second_drop_before_judge.provenance != "idw"
    assert second_drop_before_judge.conditional_geodesic_bounds is None

    pending[2](0.8)
    sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    third_drop_after_judge = pipe.process(_trace(tid=4, sig=sig))
    assert third_drop_after_judge.provenance == "idw"
    assert third_drop_after_judge.conditional_geodesic_bounds is not None


def test_out_of_range_idw_value_does_not_raise_or_claim_probability_bounds():
    sig = ("unbounded",)
    vec = np.array([1.0, 0.0])
    sampler = _FakeSampler()
    sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    res.record_eval("c1", "a", vec, 2.0)
    pipe = ValuePipeline(sampler, _Cache({sig: vec}), res, submit_judge=None)

    tv = pipe.process(_trace(sig=sig))

    assert tv.value == pytest.approx(2.0)
    assert tv.conditional_geodesic_bounds is None


def test_lipschitz_failure_does_not_break_drop_path(monkeypatch):
    sig = ("best-effort-band",)
    vec = np.array([1.0, 0.0])
    sampler = _FakeSampler()
    sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    res.record_eval("c1", "a", vec, 0.7)
    monkeypatch.setattr(res, "estimate_lipschitz", lambda **kwargs: (_ for _ in ()).throw(ValueError()))
    pipe = ValuePipeline(sampler, _Cache({sig: vec}), res, submit_judge=None)

    tv = pipe.process(_trace(sig=sig))

    assert tv.value == pytest.approx(0.7)
    assert tv.conditional_geodesic_bounds is None
    assert pipe.n_lipschitz_failures == 1


def test_judge_returning_non_finite_does_not_emit_judged():
    sig = ("run",)
    vec = np.array([1.0, 0.0])
    sampler = _FakeSampler()
    sampler.queue(keep=True, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    cache = _Cache({sig: vec})
    captured = {}
    pipe = ValuePipeline(sampler, cache, res,
                         submit_judge=lambda t, cb: captured.setdefault("cb", cb))
    emitted = []
    pipe.on_value = emitted.append
    pipe.process(_trace(tid=1, sig=sig))
    captured["cb"](float("nan"))                  # bad eval
    assert all(tv.provenance != "judged" for tv in emitted)
    assert res._global.count == 0                 # nothing recorded


def test_fallback_signature_yields_no_embed_and_mean_fallback():
    sig = ("plan",)
    sampler = _FakeSampler()
    sampler.queue(keep=False, key=VarietyKey("fallback-signature", sig))
    res = ClusterValueReservoir()
    res.record_eval(None, "a", None, 0.4)         # seed agent mean
    cache = _Cache({})                            # sig NOT cached
    pipe = ValuePipeline(sampler, cache, res, submit_judge=None)

    tv = pipe.process(_trace(sig=sig))
    assert cache.get_calls == []                  # NEVER embedded on the hot path
    assert tv.provenance == "agent_mean"
    assert tv.value == pytest.approx(0.4)


def test_uncached_cluster_trace_does_not_embed():
    sig = ("test",)
    sampler = _FakeSampler()
    sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    cache = _Cache({})                            # cluster kind but sig not cached
    pipe = ValuePipeline(sampler, cache, res, submit_judge=None)
    pipe.process(_trace(sig=sig))
    assert cache.get_calls == []                  # guard prevents embed


def test_submit_judge_failure_still_returns_pending_no_record():
    sig = ("write",)
    vec = np.array([1.0, 0.0])
    sampler = _FakeSampler()
    sampler.queue(keep=True, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    cache = _Cache({sig: vec})

    def boom(trace, on_done):
        raise RuntimeError("judge queue full")

    pipe = ValuePipeline(sampler, cache, res, submit_judge=boom)
    tv = pipe.process(_trace(sig=sig))
    assert tv.kept is True and tv.provenance == "pending"
    assert res._global.count == 0 and res._members == {}


def test_submit_failure_increments_counter():
    sig = ("write",)
    vec = np.array([1.0, 0.0])
    sampler = _FakeSampler()
    sampler.queue(keep=True, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    cache = _Cache({sig: vec})

    def boom(trace, on_done):
        raise RuntimeError("judge queue full")

    pipe = ValuePipeline(sampler, cache, res, submit_judge=boom)
    tv = pipe.process(_trace(sig=sig))
    assert tv.provenance == "pending"
    assert pipe.n_submit_failures == 1


def test_non_finite_eval_increments_rejected_counter():
    sig = ("run",)
    vec = np.array([1.0, 0.0])
    sampler = _FakeSampler()
    sampler.queue(keep=True, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    cache = _Cache({sig: vec})
    captured = {}
    pipe = ValuePipeline(sampler, cache, res,
                         submit_judge=lambda t, cb: captured.setdefault("cb", cb))
    pipe.process(_trace(tid=1, sig=sig))
    captured["cb"](float("nan"))
    assert pipe.n_rejected_evals == 1
    assert res._global.count == 0


def test_process_triggers_purge_on_cadence():
    sig = ("search",)
    vec = np.array([1.0, 0.0])
    res = ClusterValueReservoir()
    purge_calls = []
    _original_purge = res.purge_stale
    res.purge_stale = lambda **kw: purge_calls.append(kw) or _original_purge(**kw)

    sampler = _FakeSampler()
    for _ in range(3):
        sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    cache = _Cache({sig: vec})
    pipe = ValuePipeline(sampler, cache, res, submit_judge=None, purge_every=3)

    pipe.process(_trace(tid=1, sig=sig))
    assert len(purge_calls) == 0          # not yet
    pipe.process(_trace(tid=2, sig=sig))
    assert len(purge_calls) == 0          # not yet
    pipe.process(_trace(tid=3, sig=sig))
    assert len(purge_calls) == 1          # exactly on the 3rd
