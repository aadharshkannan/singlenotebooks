from trace_sampling.stats import AgentStats


def test_coldstart_until_min_samples():
    s = AgentStats(coldstart_min_samples=3, max_signatures=10, ewma_alpha=0.5)
    assert s.is_coldstart()
    for i in range(3):
        s.observe(timestamp=float(i), signature=("a",))
    assert not s.is_coldstart()


def test_velocity_ewma_tracks_rate():
    s = AgentStats(coldstart_min_samples=1, max_signatures=10, ewma_alpha=0.5)
    # One trace per second -> rate ~1.0
    for i in range(1, 11):
        s.observe(timestamp=float(i), signature=("a",))
    assert 0.5 < s.velocity() < 1.5


def test_lru_eviction_bounds_memory():
    s = AgentStats(coldstart_min_samples=1, max_signatures=3, ewma_alpha=0.5)
    for i in range(10):
        s.observe(timestamp=float(i), signature=(f"sig{i}",))
    assert s.distinct_estimate() == 3  # capped


def test_rarity_higher_for_rare_signature():
    s = AgentStats(coldstart_min_samples=1, max_signatures=10, ewma_alpha=0.5)
    for i in range(100):
        s.observe(timestamp=float(i), signature=("common",))
    s.observe(timestamp=100.0, signature=("rare",))
    assert s.rarity(("rare",)) > s.rarity(("common",))


def test_entropy_nonnegative():
    s = AgentStats(coldstart_min_samples=1, max_signatures=10, ewma_alpha=0.5)
    for i in range(5):
        s.observe(timestamp=float(i), signature=(f"s{i % 2}",))
    assert s.entropy() >= 0.0
