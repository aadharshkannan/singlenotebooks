import math
from collections import OrderedDict
from typing import Tuple


class AgentStats:
    """Bounded-memory live statistics for a single agent."""

    def __init__(self, coldstart_min_samples: int = 20,
                 max_signatures: int = 256, ewma_alpha: float = 0.1):
        self.coldstart_min_samples = coldstart_min_samples
        self.max_signatures = max_signatures
        self.ewma_alpha = ewma_alpha
        self._counts: "OrderedDict[Tuple[str, ...], int]" = OrderedDict()
        self._total = 0
        self._velocity = 0.0
        self._last_ts = None

    def observe(self, timestamp: float, signature: Tuple[str, ...]) -> None:
        if self._last_ts is not None:
            dt = max(timestamp - self._last_ts, 1e-9)
            inst_rate = 1.0 / dt
            if self._velocity == 0.0:
                self._velocity = inst_rate
            else:
                a = self.ewma_alpha
                self._velocity = a * inst_rate + (1 - a) * self._velocity
        self._last_ts = timestamp
        self._total += 1
        if signature in self._counts:
            self._counts[signature] += 1
            self._counts.move_to_end(signature)
        else:
            self._counts[signature] = 1
            self._counts.move_to_end(signature)
            if len(self._counts) > self.max_signatures:
                self._counts.popitem(last=False)  # evict least-recently-seen

    def is_coldstart(self) -> bool:
        return self._total < self.coldstart_min_samples

    def velocity(self) -> float:
        return self._velocity

    def distinct_estimate(self) -> int:
        return len(self._counts)

    def total(self) -> int:
        return self._total

    def rarity(self, signature: Tuple[str, ...]) -> float:
        """In [0, 1]; ~1 for unseen/rare signatures, ->0 for very frequent ones."""
        if self._total == 0:
            return 1.0
        count = self._counts.get(signature, 0)
        return 1.0 / (1.0 + count)

    def entropy(self) -> float:
        if self._total == 0:
            return 0.0
        total = sum(self._counts.values())
        h = 0.0
        for c in self._counts.values():
            p = c / total
            h -= p * math.log(p + 1e-12)
        return h
