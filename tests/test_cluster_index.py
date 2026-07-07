from trace_sampling.cluster_index import CircuitBreaker


def test_breaker_opens_after_threshold_and_recovers():
    b = CircuitBreaker(fail_threshold=2, cooldown_s=10.0)
    assert b.allow(now=0.0)
    b.on_failure(now=0.0); b.on_failure(now=0.0)     # 2 failures -> open
    assert not b.allow(now=1.0)                       # within cooldown
    assert b.allow(now=11.0)                          # half-open after cooldown
    b.on_success()                                    # closes
    assert b.allow(now=12.0)
