class BackpressureController:
    """AIMD admission multiplier driven by a drained consumer queue."""

    def __init__(self, throughput: float, queue_high: float, queue_low: float,
                 aimd_increase: float = 0.05, aimd_decrease: float = 0.5,
                 min_multiplier: float = 0.01):
        self.throughput = throughput
        self.queue_high = queue_high
        self.queue_low = queue_low
        self.aimd_increase = aimd_increase
        self.aimd_decrease = aimd_decrease
        self.min_multiplier = min_multiplier
        self.multiplier = 1.0
        self.queue_len = 0.0
        self._last_drain = 0.0

    def on_kept(self) -> None:
        self.queue_len += 1.0

    def tick(self, now: float) -> None:
        drained = self.throughput * max(now - self._last_drain, 0.0)
        self.queue_len = max(0.0, self.queue_len - drained)
        self._last_drain = now
        if self.queue_len > self.queue_high:
            self.multiplier *= self.aimd_decrease
        elif self.queue_len < self.queue_low:
            self.multiplier += self.aimd_increase
        self.multiplier = min(1.0, max(self.min_multiplier, self.multiplier))
