"""VarietyIndex building blocks: CircuitBreaker for Azure resilience."""
import itertools
import numpy as np

_EPS = 1e-9

from .model import Trace
from .variety import VarietyObservation, VarietyKey, ExactSignatureIndex
from .embedding import EmbeddingCache
from .vector_store import VectorStore, VectorDoc


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


def _cos(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


class AzureClusterIndex:
    """Treatment VarietyIndex: TTL leader-clustering over a vector store.

    Per trace: embed (cached) -> nearest centroid (recent in-process buffer first,
    then the vector store) -> if cosine >= tau join that cluster else create a new
    one. Stale centroids (now - last_seen > ttl) are purged so returning behavior
    is re-flagged as fresh and memory stays bounded. Degrades to exact-signature
    scoring when the per-tick embed budget is exhausted or Azure calls fail
    (circuit breaker)."""

    def __init__(self, cache: EmbeddingCache, store: VectorStore, tau: float = 0.85,
                 ttl: float = 60.0, purge_every: int = 200,
                 embed_budget_per_tick: int = 8, recent_buffer_size: int = 64,
                 breaker=None, k: float = 8.0, iat_alpha: float = 0.3):
        if not k > 0:
            raise ValueError(f"k must be > 0, got {k}")
        if not (0.0 < iat_alpha <= 1.0):
            raise ValueError(f"iat_alpha must be in (0, 1], got {iat_alpha}")
        self._cache = cache
        self._store = store
        self.tau = tau
        self.ttl = ttl
        self.purge_every = purge_every
        self.embed_budget_per_tick = embed_budget_per_tick
        self._recent = []
        self._recent_max = recent_buffer_size
        self._breaker = breaker
        self._ids = itertools.count()
        self.k = k
        self.iat_alpha = iat_alpha
        self._last_seen = {}        # cluster_id -> last observe timestamp
        self._iat = {}              # cluster_id -> EWMA of inter-observe gap (seeded on first join)
        self._since_purge = 0
        self._embeds_this_tick = 0
        self._last_tick = None
        self._fallback_index = ExactSignatureIndex()
        self.n_fallbacks = 0

    def _new_id(self) -> str:
        return f"c{next(self._ids)}"

    def _tick(self, ts: float):
        if ts != self._last_tick:
            self._embeds_this_tick = 0
            self._last_tick = ts

    def _fallback(self, trace: Trace) -> VarietyObservation:
        self.n_fallbacks += 1
        obs = self._fallback_index.observe(trace)
        return VarietyObservation(
            key=VarietyKey("fallback-signature", trace.signature),
            rarity=obs.rarity, novelty=obs.novelty)

    def _evict_recent(self, now: float):
        self._recent = [e for e in self._recent if now - e[3] <= self.ttl]

    def _touch_recent(self, cluster_id: str, now: float):
        for i, (cid, aid, v, _) in enumerate(self._recent):
            if cid == cluster_id:
                self._recent.pop(i)
                self._recent.append((cid, aid, v, now))
                return

    def _recent_nearest(self, agent_id, vec):
        best = None
        for cid, aid, v, _ in self._recent:
            if aid != agent_id:
                continue
            s = _cos(vec, v)
            if best is None or s > best[1]:
                best = (cid, s)
        return best

    def _bump(self, cluster_id: str, ts: float) -> float:
        prev_ts = self._last_decay_ts.get(cluster_id, ts)
        decay = 0.5 ** ((ts - prev_ts) / self._decay_half_life) if self._decay_half_life else 1.0
        c = self._counts.get(cluster_id, 0.0) * decay + 1.0
        self._counts[cluster_id] = c
        self._last_decay_ts[cluster_id] = ts
        return 1.0 / (1.0 + c)

    def observe(self, trace: Trace) -> VarietyObservation:
        self._tick(trace.timestamp)
        if self._breaker and not self._breaker.allow(trace.timestamp):
            return self._fallback(trace)

        if trace.signature not in self._cache:
            if self._embeds_this_tick >= self.embed_budget_per_tick:
                return self._fallback(trace)
            self._embeds_this_tick += 1

        try:
            self._evict_recent(trace.timestamp)
            self._since_purge += 1
            if self._since_purge >= self.purge_every:
                for cid in self._store.purge_stale(now=trace.timestamp, ttl=self.ttl):
                    self._counts.pop(cid, None)
                    self._last_decay_ts.pop(cid, None)
                self._since_purge = 0
            vec = self._cache.get(trace.signature)
            near = self._recent_nearest(trace.agent_id, vec)
            if near is None or near[1] < self.tau:
                store_near = self._store.nearest(vec, agent_id=trace.agent_id)
                if store_near is not None and (near is None or store_near[1] > near[1]):
                    near = store_near
            if near is not None and near[1] >= self.tau:
                cluster_id, score = near
                novelty = 0.0
                self._store.touch(cluster_id, trace.timestamp)
                self._touch_recent(cluster_id, trace.timestamp)
            else:
                cluster_id = self._new_id()
                novelty = 1.0
                self._store.upsert(VectorDoc(cluster_id, vec, trace.agent_id, trace.timestamp))
                self._recent.append((cluster_id, trace.agent_id, vec, trace.timestamp))
                if len(self._recent) > self._recent_max:
                    self._recent.pop(0)
            if self._breaker:
                self._breaker.on_success()
        except Exception:
            if self._breaker:
                self._breaker.on_failure(trace.timestamp)
            return self._fallback(trace)

        rarity = self._bump(cluster_id, trace.timestamp)
        return VarietyObservation(VarietyKey("cluster", cluster_id), rarity, novelty)
