import numpy as np
from trace_sampling.reservoir import WeightedReservoir


def test_capacity_is_bounded():
    r = WeightedReservoir(capacity=5, seed=1)
    for i in range(100):
        r.offer(item=i, weight=1.0)
    assert len(r.items()) == 5


def test_high_weight_items_favored():
    rng_seed = 2
    r = WeightedReservoir(capacity=10, seed=rng_seed)
    # 1000 low-weight items, 10 high-weight items
    for i in range(1000):
        r.offer(item=("low", i), weight=1.0)
    for i in range(10):
        r.offer(item=("high", i), weight=100.0)
    kept = r.items()
    high_kept = sum(1 for it in kept if it[0] == "high")
    assert high_kept >= 5  # high-weight items dominate the reservoir


def test_offer_returns_admission_flag():
    r = WeightedReservoir(capacity=1, seed=0)
    assert r.offer(item="first", weight=1.0) is True  # fills empty slot
    # A much higher weight item should be admitted (replace).
    admitted_any = any(r.offer(item=f"x{i}", weight=1000.0) for i in range(20))
    assert admitted_any is True
