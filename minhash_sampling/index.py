from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Iterable
from typing import Optional

from trace_sampling.model import Trace
from trace_sampling.representation import RepresentationError
from trace_sampling.variety import ExactSignatureIndex, VarietyKey, VarietyObservation

from .config import MinHashConfig
from .signature import MinHashBuildError, MinHashRecord, MinHashSignatureProvider, minhash_similarity


_EPS = 1e-9


class MinHashClusterIndex:
    """Per-agent immutable-leader MinHash clustering with bounded memory and fallback."""

    def __init__(
        self,
        config: Optional[MinHashConfig] = None,
        signature_provider: Optional[MinHashSignatureProvider] = None,
    ) -> None:
        self.cfg = config or MinHashConfig()
        self._provider = signature_provider or MinHashSignatureProvider(self.cfg)
        self._fallback_index = ExactSignatureIndex(max_signatures_per_agent=self.cfg.max_clusters_per_agent)

        self._agent_clusters: dict[str, "OrderedDict[str, dict[str, object]]"] = {}
        self._global_lru: "OrderedDict[str, tuple[str, str]]" = OrderedDict()
        self._agent_watermark: dict[str, float] = {}
        self._since_purge = 0

        # Telemetry counters.
        self.n_builds = 0
        self.n_cache_hits = 0
        self.n_comparisons = 0
        self.n_clusters = 0
        self.n_purges = 0
        self.n_evictions = 0
        self.n_fallbacks = 0
        self.n_fallback_build_errors = 0
        self.n_fallback_runtime_errors = 0
        self.n_truncations = 0

    def _fallback(self, trace: Trace, *, cause: str = "runtime") -> VarietyObservation:
        self.n_fallbacks += 1
        if cause == "build":
            self.n_fallback_build_errors += 1
        else:
            self.n_fallback_runtime_errors += 1
        obs = self._fallback_index.observe(trace)
        return VarietyObservation(
            key=VarietyKey("fallback-signature", trace.signature),
            rarity=obs.rarity,
            novelty=obs.novelty,
        )

    def _watermark_ts(self, trace: Trace) -> float:
        prior = self._agent_watermark.get(trace.agent_id)
        if prior is None:
            self._agent_watermark[trace.agent_id] = trace.timestamp
            return trace.timestamp
        ts = trace.timestamp if trace.timestamp >= prior else prior
        self._agent_watermark[trace.agent_id] = ts
        return ts

    def _agent_map(self, agent_id: str) -> "OrderedDict[str, dict[str, object]]":
        agent_map = self._agent_clusters.get(agent_id)
        if agent_map is None:
            agent_map = OrderedDict()
            self._agent_clusters[agent_id] = agent_map
        return agent_map

    def _global_key(self, agent_id: str, cluster_id: str) -> str:
        return f"{agent_id}|{cluster_id}"

    def _touch_cluster(self, agent_id: str, cluster_id: str) -> None:
        agent_map = self._agent_map(agent_id)
        if cluster_id in agent_map:
            agent_map.move_to_end(cluster_id)
        gkey = self._global_key(agent_id, cluster_id)
        if gkey in self._global_lru:
            self._global_lru.move_to_end(gkey)

    def _remove_cluster(self, agent_id: str, cluster_id: str) -> None:
        agent_map = self._agent_clusters.get(agent_id)
        if agent_map is not None:
            agent_map.pop(cluster_id, None)
            if not agent_map:
                self._agent_clusters.pop(agent_id, None)
        self._global_lru.pop(self._global_key(agent_id, cluster_id), None)

    def _evict_agent_overflow(self, agent_id: str) -> None:
        agent_map = self._agent_map(agent_id)
        while len(agent_map) > self.cfg.max_clusters_per_agent:
            cid, _ = agent_map.popitem(last=False)
            self._global_lru.pop(self._global_key(agent_id, cid), None)
            self.n_evictions += 1

    def _evict_global_overflow(self) -> None:
        while len(self._global_lru) > self.cfg.max_clusters_total:
            _, (agent_id, cluster_id) = self._global_lru.popitem(last=False)
            self._remove_cluster(agent_id, cluster_id)
            self.n_evictions += 1

    def _purge_stale(self, agent_id: str, now: float) -> None:
        removed = 0
        amap = self._agent_clusters.get(agent_id, {})
        stale = [
            cid
            for cid, cluster in amap.items()
            if now - float(cluster["last_seen"]) > self.cfg.ttl_s
        ]
        for cid in stale:
            self._remove_cluster(agent_id, cid)
            removed += 1
        if removed:
            self.n_purges += removed

    def _new_cluster_id(self, trace: Trace, record: MinHashRecord, now: float) -> str:
        material = (
            f"agent={trace.agent_id}|profile={record.profile_id}|"
            f"sig={','.join(str(v) for v in record.signature)}|"
            f"content={record.content_sha256}|generation={now:.9f}"
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
        return f"mh-{digest}"

    def _cluster_staleness(self, cluster: dict[str, object], now: float) -> float:
        dt = max(0.0, now - float(cluster["last_seen"]))
        iat = cluster.get("iat")
        if iat is None:
            iat_ref = max(dt, _EPS)
            cluster["iat"] = iat_ref
        else:
            iat_ref = max(float(iat), _EPS)
            cluster["iat"] = self.cfg.iat_alpha * dt + (1.0 - self.cfg.iat_alpha) * float(iat)
        half_life = max(self.cfg.staleness_k * iat_ref, _EPS)
        cluster["last_seen"] = now
        return 1.0 - 0.5 ** (dt / half_life)

    def _best_match(self, trace: Trace, record: MinHashRecord) -> tuple[Optional[str], float]:
        amap = self._agent_clusters.get(trace.agent_id)
        if not amap:
            return None, 0.0
        best_id = None
        best_sim = -1.0
        for cid, cluster in amap.items():
            self.n_comparisons += 1
            sim = minhash_similarity(record.signature, cluster["leader_signature"])
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        return best_id, best_sim

    def telemetry(self) -> dict[str, int]:
        return {
            "builds": self.n_builds,
            "cache_hits": self.n_cache_hits,
            "comparisons": self.n_comparisons,
            "clusters": self.n_clusters,
            "purges": self.n_purges,
            "evictions": self.n_evictions,
            "fallbacks": self.n_fallbacks,
            "fallback_build_errors": self.n_fallback_build_errors,
            "fallback_runtime_errors": self.n_fallback_runtime_errors,
            "truncations": self.n_truncations,
            "live_clusters": sum(len(v) for v in self._agent_clusters.values()),
        }

    def observe(self, trace: Trace) -> VarietyObservation:
        now = self._watermark_ts(trace)
        self._since_purge += 1
        if self._since_purge >= self.cfg.purge_every:
            self._purge_stale(trace.agent_id, now)
            self._since_purge = 0

        try:
            before_hits = self._provider.n_hits
            before_builds = self._provider.n_builds
            record = self._provider.build(trace)
            self.n_cache_hits += self._provider.n_hits - before_hits
            self.n_builds += self._provider.n_builds - before_builds
            if record.representation_truncated:
                self.n_truncations += 1
        except RepresentationError:
            raise
        except MinHashBuildError:
            return self._fallback(trace, cause="build")
        except Exception:
            return self._fallback(trace, cause="runtime")

        match_id, match_sim = self._best_match(trace, record)
        if match_id is not None and match_sim >= self.cfg.similarity_threshold:
            amap = self._agent_map(trace.agent_id)
            cluster = amap[match_id]
            rarity = self._cluster_staleness(cluster, now)
            novelty = 0.0
            self._touch_cluster(trace.agent_id, match_id)
            return VarietyObservation(
                key=VarietyKey("cluster", match_id),
                rarity=rarity,
                novelty=novelty,
            )

        cluster_id = self._new_cluster_id(trace, record, now)
        cluster = {
            "leader_signature": record.signature,
            "profile_id": record.profile_id,
            "created_at": now,
            "last_seen": now,
            "iat": None,
        }
        amap = self._agent_map(trace.agent_id)
        amap[cluster_id] = cluster
        amap.move_to_end(cluster_id)
        self._global_lru[self._global_key(trace.agent_id, cluster_id)] = (trace.agent_id, cluster_id)
        self._global_lru.move_to_end(self._global_key(trace.agent_id, cluster_id))
        self.n_clusters += 1

        self._evict_agent_overflow(trace.agent_id)
        self._evict_global_overflow()

        return VarietyObservation(
            key=VarietyKey("cluster", cluster_id),
            rarity=0.5,
            novelty=1.0,
        )


class BandedMinHashLSHIndex(MinHashClusterIndex):
    """MinHash clustering accelerated by per-agent LSH band buckets.

    This implementation keeps immutable leader signatures, bounded state (TTL + LRU),
    and falls back to exact-signature behavior on build/runtime errors. Candidate
    lookup is sublinear in the number of live leaders when band buckets are selective.

    If an agent already has live leaders but a trace produces no valid LSH bucket
    candidates, the trace is treated as novel and creates a new cluster. This
    intentionally allows LSH false negatives to split clusters instead of forcing an
    exhaustive leader scan.
    """

    def __init__(
        self,
        config: Optional[MinHashConfig] = None,
        signature_provider: Optional[MinHashSignatureProvider] = None,
    ) -> None:
        super().__init__(config=config, signature_provider=signature_provider)
        self._band_buckets: dict[str, "OrderedDict[str, set[str]]"] = {}
        self.n_candidate_lookups = 0
        self.n_candidate_unions = 0
        self.n_no_candidate_novel = 0
        self.n_last_candidates = 0

    def _band_map(self, agent_id: str) -> "OrderedDict[str, set[str]]":
        bands = self._band_buckets.get(agent_id)
        if bands is None:
            bands = OrderedDict()
            self._band_buckets[agent_id] = bands
        return bands

    def _band_keys_for_signature(self, signature: tuple[int, ...]) -> tuple[str, ...]:
        rows = self.cfg.lsh_rows
        out: list[str] = []
        for band in range(self.cfg.lsh_bands):
            start = band * rows
            end = start + rows
            band_tuple = signature[start:end]
            payload = ",".join(str(value) for value in band_tuple)
            digest = hashlib.sha256(f"{band}|{payload}".encode("utf-8")).hexdigest()[:24]
            out.append(f"b{band}:{digest}")
        return tuple(out)

    def _add_to_buckets(self, agent_id: str, cluster_id: str, signature: tuple[int, ...]) -> None:
        bands = self._band_map(agent_id)
        for bucket_key in self._band_keys_for_signature(signature):
            members = bands.get(bucket_key)
            if members is None:
                members = set()
                bands[bucket_key] = members
            members.add(cluster_id)

    def _remove_from_buckets(self, agent_id: str, cluster_id: str) -> None:
        bands = self._band_buckets.get(agent_id)
        if not bands:
            return
        stale_keys: list[str] = []
        for bucket_key, members in bands.items():
            if cluster_id in members:
                members.discard(cluster_id)
            if not members:
                stale_keys.append(bucket_key)
        for bucket_key in stale_keys:
            bands.pop(bucket_key, None)
        if not bands:
            self._band_buckets.pop(agent_id, None)

    def _remove_cluster(self, agent_id: str, cluster_id: str) -> None:
        self._remove_from_buckets(agent_id, cluster_id)
        super()._remove_cluster(agent_id, cluster_id)

    def _evict_agent_overflow(self, agent_id: str) -> None:
        agent_map = self._agent_map(agent_id)
        while len(agent_map) > self.cfg.max_clusters_per_agent:
            cluster_id = next(iter(agent_map))
            self._remove_cluster(agent_id, cluster_id)
            self.n_evictions += 1

    def _candidate_cluster_ids(self, trace: Trace, record: MinHashRecord) -> set[str]:
        self.n_candidate_lookups += 1
        bands = self._band_buckets.get(trace.agent_id)
        if not bands:
            return set()
        out: set[str] = set()
        for bucket_key in self._band_keys_for_signature(record.signature):
            members = bands.get(bucket_key)
            if members:
                out.update(members)
        self.n_candidate_unions += 1
        return out

    def _iter_candidate_clusters(
        self,
        trace: Trace,
        record: MinHashRecord,
    ) -> Iterable[tuple[str, dict[str, object]]]:
        amap = self._agent_clusters.get(trace.agent_id)
        if not amap:
            self.n_last_candidates = 0
            return ()
        candidate_ids = self._candidate_cluster_ids(trace, record)
        valid = [cid for cid in sorted(candidate_ids) if cid in amap]
        self.n_last_candidates = len(valid)
        if valid:
            return tuple((cid, amap[cid]) for cid in valid)
        self.n_no_candidate_novel += 1
        return ()

    def _best_match(self, trace: Trace, record: MinHashRecord) -> tuple[Optional[str], float]:
        candidates = self._iter_candidate_clusters(trace, record)
        if not candidates:
            return None, 0.0
        best_id = None
        best_sim = -1.0
        for cid, cluster in candidates:
            self.n_comparisons += 1
            sim = minhash_similarity(record.signature, cluster["leader_signature"])
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        return best_id, best_sim

    def telemetry(self) -> dict[str, int]:
        base = super().telemetry()
        base.update(
            {
                "candidate_lookups": self.n_candidate_lookups,
                "candidate_unions": self.n_candidate_unions,
                "full_scan_fallbacks": 0,
                "no_candidate_novel": self.n_no_candidate_novel,
                "last_candidates": self.n_last_candidates,
            }
        )
        return base

    def observe(self, trace: Trace) -> VarietyObservation:
        now = self._watermark_ts(trace)
        self._since_purge += 1
        if self._since_purge >= self.cfg.purge_every:
            self._purge_stale(trace.agent_id, now)
            self._since_purge = 0

        try:
            before_hits = self._provider.n_hits
            before_builds = self._provider.n_builds
            record = self._provider.build(trace)
            self.n_cache_hits += self._provider.n_hits - before_hits
            self.n_builds += self._provider.n_builds - before_builds
            if record.representation_truncated:
                self.n_truncations += 1
        except RepresentationError:
            raise
        except MinHashBuildError:
            return self._fallback(trace, cause="build")
        except Exception:
            return self._fallback(trace, cause="runtime")

        match_id, match_sim = self._best_match(trace, record)
        if match_id is not None and match_sim >= self.cfg.similarity_threshold:
            amap = self._agent_map(trace.agent_id)
            cluster = amap[match_id]
            rarity = self._cluster_staleness(cluster, now)
            novelty = 0.0
            self._touch_cluster(trace.agent_id, match_id)
            return VarietyObservation(
                key=VarietyKey("cluster", match_id),
                rarity=rarity,
                novelty=novelty,
            )

        cluster_id = self._new_cluster_id(trace, record, now)
        cluster = {
            "leader_signature": record.signature,
            "profile_id": record.profile_id,
            "created_at": now,
            "last_seen": now,
            "iat": None,
        }
        amap = self._agent_map(trace.agent_id)
        amap[cluster_id] = cluster
        amap.move_to_end(cluster_id)
        self._global_lru[self._global_key(trace.agent_id, cluster_id)] = (trace.agent_id, cluster_id)
        self._global_lru.move_to_end(self._global_key(trace.agent_id, cluster_id))
        self._add_to_buckets(trace.agent_id, cluster_id, record.signature)
        self.n_clusters += 1

        self._evict_agent_overflow(trace.agent_id)
        self._evict_global_overflow()

        return VarietyObservation(
            key=VarietyKey("cluster", cluster_id),
            rarity=0.5,
            novelty=1.0,
        )
