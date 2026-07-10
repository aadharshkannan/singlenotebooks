import math
import time
import numpy as np
import pytest
from trace_sampling.value_reservoir import ClusterValueReservoir, Imputation, _Running


def test_running_mean_updates_incrementally():
    r = _Running()
    assert r.count == 0
    r.update(1.0); r.update(3.0)
    assert r.count == 2
    assert r.mean == pytest.approx(2.0)


def test_ctor_defaults_and_validation():
    res = ClusterValueReservoir()
    assert res.k == 64 and res.power == 2.0 and res.eps == 1e-6
    assert res.prior == 0.5 and res.ttl == 60.0
    for bad in dict(k=0), dict(power=0.0), dict(eps=0.0), dict(ttl=0.0):
        with pytest.raises(ValueError):
            ClusterValueReservoir(**bad)


def _vec(*xs):
    return np.array(xs, dtype=np.float64)


def test_record_eval_updates_means_and_returns_accept_flag():
    res = ClusterValueReservoir()
    assert res.record_eval("c1", "a", _vec(1.0, 0.0), 0.8) is True
    # means updated
    assert res._global.mean == pytest.approx(0.8)
    assert res._agent_mean["a"].mean == pytest.approx(0.8)
    # member stored under c1
    assert len(res._members["c1"]) == 1


def test_record_eval_no_member_when_cluster_or_vec_missing():
    res = ClusterValueReservoir()
    assert res.record_eval(None, "a", _vec(1.0), 0.4) is True      # no cluster
    assert res.record_eval("c1", "a", None, 0.6) is True           # no vec
    assert res._members == {}                                       # nothing stored
    assert res._agent_mean["a"].count == 2                          # both means updated


def test_record_eval_rejects_non_finite():
    res = ClusterValueReservoir()
    assert res.record_eval("c1", "a", _vec(1.0), float("nan")) is False
    assert res.record_eval("c1", "a", _vec(1.0), float("inf")) is False
    assert res._global.count == 0 and res._members == {}


def test_record_eval_copies_donor_vector():
    res = ClusterValueReservoir()
    original = np.array([1.0, 0.0], dtype=np.float64)
    res.record_eval("c1", "a", original, 0.7)
    original[0] = 999.0   # mutate after recording
    stored_vec, stored_val = res._members["c1"][0]
    assert stored_vec[0] == pytest.approx(1.0)   # copy, not aliased
    assert res.impute("c1", "a", _vec(1.0, 0.0)).value == pytest.approx(0.7, abs=1e-3)


def test_impute_idw_weighted_average():
    res = ClusterValueReservoir(power=2.0, eps=1e-6)
    # two orthogonal-ish donors; query closer to the first
    res.record_eval("c1", "a", _vec(1.0, 0.0), 0.0)
    res.record_eval("c1", "a", _vec(0.0, 1.0), 1.0)
    imp = res.impute("c1", "a", _vec(1.0, 0.2))   # nearer donor-1 (value 0.0)
    assert imp.provenance == "idw"
    assert imp.n_donors == 2
    assert 0.0 <= imp.value < 0.5                  # pulled toward the near donor
    assert imp.nearest_dist == pytest.approx(0.0, abs=0.05)


def test_impute_higher_power_favors_nearest_donor():
    donors = [(_vec(1.0, 0.0), 0.0), (_vec(0.0, 1.0), 1.0)]
    q = _vec(1.0, 0.3)
    lo = ClusterValueReservoir(power=1.0)
    hi = ClusterValueReservoir(power=6.0)
    for r in (lo, hi):
        for v, val in donors:
            r.record_eval("c1", "a", v, val)
    assert hi.impute("c1", "a", q).value < lo.impute("c1", "a", q).value


def test_impute_eps_floor_near_duplicate_donor():
    res = ClusterValueReservoir(power=2.0, eps=1e-6)
    res.record_eval("c1", "a", _vec(1.0, 0.0), 0.3)   # value 0.3
    res.record_eval("c1", "a", _vec(0.0, 1.0), 0.9)
    imp = res.impute("c1", "a", _vec(1.0, 0.0))       # exact duplicate of donor-1
    assert imp.value == pytest.approx(0.3, abs=1e-3)  # near-dup dominates, no div0


def test_impute_fallback_chain():
    res = ClusterValueReservoir(prior=0.5)
    # cold: nothing anywhere -> prior
    assert res.impute("c1", "a", _vec(1.0, 0.0)).provenance == "prior"
    # global-only (recorded with no cluster/vec): agent "b" query with no members -> agent_mean
    res.record_eval(None, "b", None, 0.2)
    assert res.impute("cX", "b", None).provenance == "agent_mean"
    # unknown agent falls to global_mean
    assert res.impute("cX", "zzz", None).provenance == "global_mean"


def test_impute_no_vec_skips_idw():
    res = ClusterValueReservoir()
    res.record_eval("c1", "a", _vec(1.0, 0.0), 0.7)
    imp = res.impute("c1", "a", None)                 # no query vec -> cannot IDW
    assert imp.provenance == "agent_mean"


def test_ring_buffer_caps_at_k():
    res = ClusterValueReservoir(k=3)
    for i in range(5):
        res.record_eval("c1", "a", _vec(float(i), 0.0), float(i))
    assert len(res._members["c1"]) == 3          # only last k retained
    vals = [v for _, v in res._members["c1"]]
    assert vals == [2.0, 3.0, 4.0]


def test_purge_stale_drops_and_returns_ids():
    res = ClusterValueReservoir(ttl=10.0)
    res.record_eval("c1", "a", _vec(1.0), 0.5, now=0.0)
    res.record_eval("c2", "a", _vec(1.0), 0.5, now=100.0)
    dropped = res.purge_stale(now=105.0)          # c1 is stale (>10s), c2 fresh
    assert dropped == ["c1"]
    assert "c1" not in res._members and "c1" not in res._last_seen
    assert "c2" in res._members


def test_evict_removes_named_clusters():
    res = ClusterValueReservoir()
    res.record_eval("c1", "a", _vec(1.0), 0.5, now=0.0)
    res.record_eval("c2", "a", _vec(1.0), 0.5, now=0.0)
    res.evict(["c1"])
    assert "c1" not in res._members and "c1" not in res._last_seen
    assert "c2" in res._members


def test_record_eval_without_now_is_fresh_not_immediately_purgeable():
    res = ClusterValueReservoir(ttl=3600.0)
    res.record_eval("c1", "a", _vec(1.0), 0.5)          # no explicit now -> monotonic()
    dropped = res.purge_stale(now=time.monotonic())      # same clock, just recorded
    assert "c1" not in dropped
    assert "c1" in res._members


import threading


def test_concurrent_record_and_impute_stays_consistent():
    res = ClusterValueReservoir(k=128)
    errors = []

    def writer():
        try:
            for i in range(500):
                res.record_eval("c1", "a", _vec(float(i % 7), 1.0), (i % 10) / 10.0)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def reader():
        try:
            for _ in range(500):
                imp = res.impute("c1", "a", _vec(1.0, 1.0))
                assert math.isfinite(imp.value)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=writer) for _ in range(2)] + \
         [threading.Thread(target=reader) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    assert res._global.count == 1000
