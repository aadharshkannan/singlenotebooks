"""VarietyIndex building blocks: CircuitBreaker for Azure resilience."""


class CircuitBreaker:
    def __init__(self, fail_threshold: int = 5, cooldown_s: float = 30.0):
        self.fail_threshold = fail_threshold
        self.cooldown_s = cooldown_s
        self._fails = 0
        self._opened_at = None

    def allow(self, now: float) -> bool:
        if self._opened_at is None:
            return True
        if now - self._opened_at >= self.cooldown_s:
            return True  # half-open: allow a trial
        return False

    def on_success(self) -> None:
        self._fails = 0
        self._opened_at = None

    def on_failure(self, now: float) -> None:
        self._fails += 1
        if self._fails >= self.fail_threshold:
            self._opened_at = now
