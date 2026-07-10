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
