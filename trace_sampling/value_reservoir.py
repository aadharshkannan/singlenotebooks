"""Snap IDW-imputed eval values for dropped traces.

ClusterValueReservoir stores the judged eval of kept traces per agent-scoped
cluster (bounded ring buffer of (embedding, value)) plus per-agent and global
running means. When a trace is dropped, impute() returns an inverse-distance-
weighted average of nearby judged members of the same cluster, degrading through
agent-mean -> global-mean -> prior. It never influences keep/drop or variety
scoring; see docs/superpowers/specs/2026-07-09-snap-imputed-evals-cluster-value-reservoir-design.md
"""
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .vector_store import _cosine


@dataclass(frozen=True)
class Imputation:
    value: float
    provenance: str      # "idw" | "agent_mean" | "global_mean" | "prior"
    n_donors: int        # reservoir members used (0 for non-idw)
    nearest_dist: float  # min (1 - cosine) to a donor; NaN if none


class _Running:
    """Incremental (count, mean) accumulator. O(1) update, O(1) memory."""

    __slots__ = ("count", "mean")

    def __init__(self):
        self.count = 0
        self.mean = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.mean += (value - self.mean) / self.count


class ClusterValueReservoir:
    def __init__(self, k: int = 64, power: float = 2.0,
                 eps: float = 1e-6, prior: float = 0.5, ttl: float = 60.0):
        if not k >= 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not power > 0:
            raise ValueError(f"power must be > 0, got {power}")
        if not eps > 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if not ttl > 0:
            raise ValueError(f"ttl must be > 0, got {ttl}")
        self.k = k
        self.power = power
        self.eps = eps
        self.prior = prior
        self.ttl = ttl
        self._members: Dict[str, Deque[Tuple[np.ndarray, float]]] = {}
        self._last_seen: Dict[str, float] = {}
        self._agent_mean: Dict[str, _Running] = {}
        self._global = _Running()
        self._lock = threading.Lock()
