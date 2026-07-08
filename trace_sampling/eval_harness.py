from dataclasses import dataclass
from typing import Optional
import time
import numpy as np
import pandas as pd

from .samplers import AdaptiveSampler, SamplerConfig
from .variety import ExactSignatureIndex


@dataclass
class RunResult:
    log: pd.DataFrame     # per-trace assignment log
    ledger: dict          # cost/latency counters (embed_calls, cache_hits, fallbacks, kept)
    index: object         # the VarietyIndex used (exposes _cache for treatment arms)


def _make_index(arm: str, cfg: SamplerConfig, synonym_map=None):
    if arm == "baseline":
        return ExactSignatureIndex(max_signatures_per_agent=cfg.max_signatures_per_agent), False
    if arm == "treatment_offline":
        from .embedding import FakeEmbedder, EmbeddingCache
        from .vector_store import InMemoryVectorStore
        from .cluster_index import AzureClusterIndex
        fe = FakeEmbedder(dim=64, synonym_map=synonym_map, noise=0.01, seed=0)
        return AzureClusterIndex(EmbeddingCache(fe), InMemoryVectorStore(), tau=0.9), True
    if arm == "treatment_azure":
        from .azure_config import AzureConfig
        from .embedding import AzureOpenAIEmbedder, EmbeddingCache
        from .vector_store import AzureSearchVectorStore
        from .cluster_index import AzureClusterIndex, CircuitBreaker
        c = AzureConfig.from_env()
        return AzureClusterIndex(
            EmbeddingCache(AzureOpenAIEmbedder(c)),
            AzureSearchVectorStore(c, dim=1536, ensure_index=True),
            tau=0.50, breaker=CircuitBreaker()), True
    raise ValueError(arm)


def _ledger(index, kept_count: int, decide_latencies_ms=None) -> dict:
    cache = getattr(index, "_cache", None)
    store = getattr(index, "_store", None)
    lat = list(getattr(cache, "embed_latencies_ms", []) or [])
    dlat = list(decide_latencies_ms or [])
    calls = getattr(cache, "n_calls", 0)
    hits = getattr(cache, "n_hits", 0)
    total = calls + hits
    cost_per_call = 10.0 / 1_000_000.0 * 0.02
    return dict(
        embed_calls=calls,
        cache_hits=hits,
        cache_hit_rate=(hits / total) if total else 0.0,
        search_queries=getattr(store, "n_queries", 0),
        embed_latency_p50_ms=float(np.percentile(lat, 50)) if lat else 0.0,
        embed_latency_p95_ms=float(np.percentile(lat, 95)) if lat else 0.0,
        added_latency_p50_ms=float(np.percentile(dlat, 50)) if dlat else 0.0,
        added_latency_p95_ms=float(np.percentile(dlat, 95)) if dlat else 0.0,
        est_cost_usd=calls * cost_per_call,
        fallbacks=getattr(index, "n_fallbacks", 0),
        kept=kept_count,
    )


def run_arm(stream, cfg: SamplerConfig, arm: str, seed: int = 0,
            synonym_map=None) -> RunResult:
    index, use_novelty = _make_index(arm, cfg, synonym_map)
    sampler = AdaptiveSampler(cfg, seed=seed, variety_index=index, use_novelty=use_novelty)
    rows = []
    kept_count = 0
    decide_latencies_ms = []
    for t in stream:
        t0 = time.perf_counter()
        kept = sampler.decide(t)
        decide_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        kept_count += int(kept)
        obs = sampler.last_observation
        rows.append(dict(
            timestamp=t.timestamp, agent_id=t.agent_id, concept_id=t.concept_id,
            signature=t.signature, variety_key=str(obs.key.value),
            key_kind=obs.key.kind, kept=kept,
        ))
    return RunResult(pd.DataFrame(rows), _ledger(index, kept_count, decide_latencies_ms), index)


def kept_embeddings(result: RunResult) -> Optional[np.ndarray]:
    """Re-embed the KEPT traces' signatures via the run's cache (treatment arms
    only; returns None for baseline). Row order matches result.log[kept] order,
    so it lines up with cross_agent_unification's kept_log argument."""
    cache = getattr(result.index, "_cache", None)
    if cache is None:
        return None
    kept = result.log[result.log["kept"]]
    return np.array([cache.get(tuple(sig)) for sig in kept["signature"]], dtype=float)
