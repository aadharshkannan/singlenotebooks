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
