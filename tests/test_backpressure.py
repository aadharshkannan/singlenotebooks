from trace_sampling.backpressure import BackpressureController


def _ctrl():
    return BackpressureController(
        throughput=10.0, queue_high=20.0, queue_low=5.0,
        aimd_increase=0.05, aimd_decrease=0.5, min_multiplier=0.01)


def test_multiplier_drops_under_backpressure():
    c = _ctrl()
    now = 0.0
    for _ in range(100):       # flood without draining time
        c.on_kept()
    c.tick(now)                # queue huge -> decrease
    assert c.multiplier < 1.0


def test_multiplier_recovers_with_slack():
    c = _ctrl()
    c.multiplier = 0.2
    # Advance time with empty queue -> additive increase each tick.
    for i in range(1, 50):
        c.tick(float(i))
    assert c.multiplier > 0.2


def test_multiplier_clamped():
    c = _ctrl()
    for _ in range(10000):
        c.on_kept()
    for i in range(1, 50):
        c.tick(0.0)
    assert c.multiplier >= 0.01
    assert c.multiplier <= 1.0


def test_queue_drains_over_time():
    c = _ctrl()
    for _ in range(50):
        c.on_kept()
    c.tick(10.0)               # 10s * 10/s = 100 drained -> empty
    assert c.queue_len == 0.0
