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

    def record_eval(self, cluster_id: Optional[str], agent_id: str,
                    vec: Optional[np.ndarray], value: float,
                    now: Optional[float] = None) -> bool:
        """Record a judged eval. Returns True iff the value was accepted (finite).

        Always updates the agent + global running means for a finite value.
        Appends an IDW donor (vec, value) to the cluster ring buffer only when
        BOTH cluster_id and vec are present; otherwise no _members entry is made.
        `now` defaults to a monotonic-clock reading (time.monotonic) so a donor
        recorded without an explicit timestamp is fresh, not immediately purgeable."""
        if not math.isfinite(value):
            return False
        ts = time.monotonic() if now is None else now
        with self._lock:
            self._global.update(value)
            self._agent_mean.setdefault(agent_id, _Running()).update(value)
            if cluster_id is not None and vec is not None:
                buf = self._members.get(cluster_id)
                if buf is None:
                    buf = self._members[cluster_id] = deque(maxlen=self.k)
                buf.append((np.asarray(vec, dtype=np.float64), float(value)))
                self._last_seen[cluster_id] = ts
        return True

    def impute(self, cluster_id: Optional[str], agent_id: str,
               vec: Optional[np.ndarray]) -> Imputation:
        """Snap value for a dropped trace: IDW over the cluster's judged members,
        falling back to agent-mean -> global-mean -> prior (first applicable)."""
        with self._lock:
            members = self._members.get(cluster_id) if cluster_id is not None else None
            if vec is not None and members:
                q = np.asarray(vec, dtype=np.float64)
                num = 0.0
                den = 0.0
                nearest = math.inf
                for mvec, mval in members:
                    d = max(0.0, 1.0 - _cosine(q, mvec))
                    nearest = min(nearest, d)
                    w = 1.0 / (d + self.eps) ** self.power
                    num += w * mval
                    den += w
                return Imputation(num / den, "idw", len(members), nearest)
            am = self._agent_mean.get(agent_id)
            if am is not None and am.count > 0:
                return Imputation(am.mean, "agent_mean", 0, math.nan)
            if self._global.count > 0:
                return Imputation(self._global.mean, "global_mean", 0, math.nan)
            return Imputation(self.prior, "prior", 0, math.nan)

    def purge_stale(self, now: float, ttl: Optional[float] = None) -> List[str]:
        """Drop cluster reservoirs untouched for longer than ttl. Returns dropped ids."""
        ttl = self.ttl if ttl is None else ttl
        with self._lock:
            stale = [cid for cid, ts in self._last_seen.items() if now - ts > ttl]
            for cid in stale:
                self._members.pop(cid, None)
                self._last_seen.pop(cid, None)
        return stale

    def evict(self, cluster_ids: Iterable[str]) -> None:
        """Forget the named cluster reservoirs (e.g. mirroring the index's purge)."""
        with self._lock:
            for cid in cluster_ids:
                self._members.pop(cid, None)
                self._last_seen.pop(cid, None)
