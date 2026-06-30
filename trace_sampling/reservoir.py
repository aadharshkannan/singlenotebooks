import heapq
from typing import Any, List
import numpy as np


class WeightedReservoir:
    """A-Res weighted reservoir: retains the top-`capacity` items by key
    u**(1/weight). Higher weight -> higher retention probability."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        # min-heap of (key, tiebreak, item); smallest key at top for eviction
        self._heap: List = []
        self._tiebreak = 0

    def offer(self, item: Any, weight: float) -> bool:
        weight = max(weight, 1e-9)
        u = self._rng.random()
        key = u ** (1.0 / weight)
        self._tiebreak += 1
        entry = (key, self._tiebreak, item)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, entry)
            return True
        if key > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)
            return True
        return False

    def items(self) -> List[Any]:
        return [e[2] for e in self._heap]

    def __len__(self) -> int:
        return len(self._heap)
