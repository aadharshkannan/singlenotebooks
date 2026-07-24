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

from .lipschitz import (
    BoundedClusterSummary,
    LipschitzEstimate,
    LipschitzEstimatorConfig,
    estimate_bounded_lipschitz,
)
from .vector_store import _cosine


@dataclass(frozen=True)
class Imputation:
    value: float
    provenance: str      # "idw" | "agent_mean" | "global_mean" | "prior"
    n_donors: int        # reservoir members used (0 for non-idw)
    nearest_dist: float  # min (1 - cosine) to a donor; NaN if none
    weighted_geodesic_angle: float = math.nan  # weighted arccos(cosine); NaN if non-idw


class _Running:
    """Incremental (count, mean) accumulator. O(1) update, O(1) memory."""

    __slots__ = ("count", "mean")

    def __init__(self):
        self.count = 0
        self.mean = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.mean += (value - self.mean) / self.count


@dataclass
class _ClusterCalibration:
    agent_id: str
    count: int
    value_sum: float
    centroid_sum: np.ndarray

    def update(self, unit_vec: np.ndarray, value: float) -> None:
        self.count += 1
        self.value_sum += value
        self.centroid_sum += unit_vec


class ClusterValueReservoir:
    def __init__(self, k: int = 64, power: float = 2.0,
                 eps: float = 1e-6, prior: float = 0.5, ttl: float = 60.0):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if ttl <= 0:
            raise ValueError(f"ttl must be > 0, got {ttl}")
        self.k = k
        self.power = power
        self.eps = eps
        self.prior = prior
        self.ttl = ttl
        self._members: Dict[str, Deque[Tuple[np.ndarray, float]]] = {}
        self._cluster_agents: Dict[str, str] = {}
        # IDW donors stay ring-bounded; these sufficient statistics retain all
        # completed normalized evals so calibration support is not capped by k.
        self._calibration: Dict[str, _ClusterCalibration] = {}
        self._lipschitz_cache: Dict[tuple, Tuple[int, LipschitzEstimate]] = {}
        self._lipschitz_versions: Dict[str, int] = {}
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
                stored_vec = np.array(vec, dtype=np.float64)
                buf.append((stored_vec, float(value)))
                self._cluster_agents[cluster_id] = agent_id
                self._last_seen[cluster_id] = ts
                if 0.0 <= value <= 1.0:
                    norm = float(np.linalg.norm(stored_vec))
                    if stored_vec.ndim == 1 and stored_vec.size > 0 and math.isfinite(norm) and norm > 0.0:
                        # Unit vectors make the accumulated centroid angular;
                        # embedding magnitude must not bias geodesic distances.
                        unit_vec = stored_vec / norm
                        summary = self._calibration.get(cluster_id)
                        if summary is None:
                            self._calibration[cluster_id] = _ClusterCalibration(
                                agent_id=agent_id,
                                count=1,
                                value_sum=float(value),
                                centroid_sum=unit_vec,
                            )
                        elif summary.agent_id == agent_id and summary.centroid_sum.shape == unit_vec.shape:
                            summary.update(unit_vec, float(value))
                        self._invalidate_lipschitz(agent_id)
        return True

    def _invalidate_lipschitz(self, agent_id: str) -> None:
        self._lipschitz_versions[agent_id] = self._lipschitz_versions.get(agent_id, 0) + 1
        stale = [key for key in self._lipschitz_cache if key[0] == agent_id]
        for key in stale:
            self._lipschitz_cache.pop(key, None)

    def impute(self, cluster_id: Optional[str], agent_id: str,
               vec: Optional[np.ndarray]) -> Imputation:
        """Snap value for a dropped trace: IDW over the cluster's judged members,
        falling back to agent-mean -> global-mean -> prior (first applicable)."""
        with self._lock:
            members = self._members.get(cluster_id) if cluster_id is not None else None
            # Invariant: _members never holds an empty deque — deques are created on
            # first append and removed wholesale by purge_stale/evict, so `members`
            # being truthy (non-None, non-empty) is sufficient to proceed with IDW.
            if vec is not None and members:
                q = np.asarray(vec, dtype=np.float64)
                num = 0.0
                den = 0.0
                angle_num = 0.0
                nearest = math.inf
                for mvec, mval in members:
                    cos = max(-1.0, min(1.0, _cosine(q, mvec)))
                    d = max(0.0, 1.0 - cos)
                    nearest = min(nearest, d)
                    w = 1.0 / (d + self.eps) ** self.power
                    num += w * mval
                    den += w
                    angle_num += w * math.acos(cos)
                return Imputation(num / den, "idw", len(members), nearest, angle_num / den)
            am = self._agent_mean.get(agent_id)
            if am is not None and am.count > 0:
                return Imputation(am.mean, "agent_mean", 0, math.nan, math.nan)
            if self._global.count > 0:
                return Imputation(self._global.mean, "global_mean", 0, math.nan, math.nan)
            return Imputation(self.prior, "prior", 0, math.nan, math.nan)

    def purge_stale(self, now: float, ttl: Optional[float] = None) -> List[str]:
        """Drop cluster reservoirs untouched for longer than ttl. Returns dropped ids.

        `now` must come from the same clock used by `record_eval` — i.e.
        `time.monotonic()` when records were inserted with the default `now=None`.
        Mixing clock epochs (e.g. passing `time.time()` against monotonic timestamps)
        will produce incorrect `now - ts` comparisons."""
        ttl = self.ttl if ttl is None else ttl
        with self._lock:
            stale = [cid for cid, ts in self._last_seen.items() if now - ts > ttl]
            for cid in stale:
                agent_id = self._cluster_agents.get(cid)
                self._members.pop(cid, None)
                self._cluster_agents.pop(cid, None)
                self._calibration.pop(cid, None)
                self._last_seen.pop(cid, None)
                if agent_id is not None:
                    self._invalidate_lipschitz(agent_id)
        return stale

    def evict(self, cluster_ids: Iterable[str]) -> None:
        """Forget the named cluster reservoirs (e.g. mirroring the index's purge)."""
        with self._lock:
            for cid in cluster_ids:
                agent_id = self._cluster_agents.get(cid)
                self._members.pop(cid, None)
                self._cluster_agents.pop(cid, None)
                self._calibration.pop(cid, None)
                self._last_seen.pop(cid, None)
                if agent_id is not None:
                    self._invalidate_lipschitz(agent_id)

    def estimate_lipschitz(
        self,
        *,
        agent_id: str,
        embedding_dimension: int,
        config: LipschitzEstimatorConfig,
    ) -> LipschitzEstimate:
        """Estimate a bounded-value Lipschitz constant from judged cluster donors.

        Calibration is agent-scoped and uses only retained values in [0, 1].
        Their sums are kept continuous rather than rounded to binary pseudo-counts.
        """
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be >= 1")
        cache_key = (agent_id, embedding_dimension, config)
        with self._lock:
            version = self._lipschitz_versions.get(agent_id, 0)
            cached = self._lipschitz_cache.get(cache_key)
            if cached is not None and cached[0] == version:
                return cached[1]
            # Copy sufficient statistics under the lock, then release it before
            # the pairwise O(cluster^2) estimator work.
            snapshots = tuple(
                (
                    cid,
                    summary.count,
                    summary.value_sum,
                    np.array(summary.centroid_sum, copy=True),
                )
                for cid, summary in self._calibration.items()
                if summary.agent_id == agent_id and summary.centroid_sum.size == embedding_dimension
            )

        clusters: list[BoundedClusterSummary] = []
        for cid, count, value_sum, centroid_sum in snapshots:
            norm = float(np.linalg.norm(centroid_sum))
            if not math.isfinite(norm) or norm <= 0.0:
                continue
            centroid = centroid_sum / norm
            clusters.append(
                BoundedClusterSummary(
                    cluster_id=cid,
                    centroid=tuple(float(x) for x in centroid),
                    value_sum=min(float(count), max(0.0, value_sum)),
                    count=count,
                )
            )

        estimate = estimate_bounded_lipschitz(tuple(clusters), config)
        with self._lock:
            # A concurrent judge callback may have advanced the agent version;
            # return this snapshot estimate, but never cache it as current.
            if self._lipschitz_versions.get(agent_id, 0) == version:
                self._lipschitz_cache[cache_key] = (version, estimate)
        return estimate
