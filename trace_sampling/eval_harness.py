from dataclasses import dataclass
from typing import Optional
import time
import numpy as np
import pandas as pd

from .samplers import AdaptiveSampler, BaselineSampler, SamplerConfig
from .variety import ExactSignatureIndex
from .embedding import cache_peek_trace


@dataclass
class RunResult:
    log: pd.DataFrame     # per-trace assignment log
    ledger: dict          # cost/latency counters (embed_calls, cache_hits, fallbacks, kept)
    index: object         # the VarietyIndex used (exposes _cache for treatment arms)


def _make_embedding_cache(embedder, default_model_id: str):
    from .embedding import EmbeddingCache
    from .embedding_config import EmbeddingConfig

    config = EmbeddingConfig.from_env(default_model_id)
    if not config.full_session_enabled:
        return EmbeddingCache(embedder)

    from .session_embedding import EmbeddingProfile, SessionEmbeddingCache, TiktokenTokenizer

    tokenizer = TiktokenTokenizer(
        model_name=config.tokenizer_id,
        encoding_name=config.tokenizer_encoding,
    )
    profile = EmbeddingProfile(
        model_id=config.model_id,
        model_version=config.model_version,
        tokenizer_id=tokenizer.name,
        tokenizer_version=tokenizer.version,
        max_input_tokens=config.max_input_tokens,
        max_representation_utf8_bytes=config.max_representation_utf8_bytes,
    )
    return SessionEmbeddingCache(embedder, tokenizer, profile)


def _make_index(arm: str, cfg: SamplerConfig, synonym_map=None):
    if arm == "adaptive_exact":
        return ExactSignatureIndex(max_signatures_per_agent=cfg.max_signatures_per_agent), False
    if arm == "adaptive_cluster_offline":
        from .embedding import FakeEmbedder
        from .vector_store import InMemoryVectorStore
        from .cluster_index import AzureClusterIndex
        fe = FakeEmbedder(dim=64, synonym_map=synonym_map, noise=0.01, seed=0)
        cache = _make_embedding_cache(fe, default_model_id="text-embedding-3-small")
        return AzureClusterIndex(cache, InMemoryVectorStore(), tau=0.9), True
    if arm == "adaptive_cluster_azure":
        from .azure_config import AzureConfig
        from .embedding import AzureOpenAIEmbedder
        from .vector_store import AzureSearchVectorStore
        from .cluster_index import AzureClusterIndex, CircuitBreaker
        c = AzureConfig.from_env()
        store = AzureSearchVectorStore(c, dim=1536, ensure_index=True)
        store.clear()  # start each eval run from a clean index for reproducibility
        cache = _make_embedding_cache(
            AzureOpenAIEmbedder(c), default_model_id=c.embedding_deployment
        )
        return AzureClusterIndex(
            cache,
            store,
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
    embedded_tokens = getattr(cache, "n_tokens", 0)
    return dict(
        embed_calls=calls,
        cache_hits=hits,
        cache_hit_rate=(hits / total) if total else 0.0,
        search_queries=getattr(store, "n_queries", 0),
        embed_latency_p50_ms=float(np.percentile(lat, 50)) if lat else 0.0,
        embed_latency_p95_ms=float(np.percentile(lat, 95)) if lat else 0.0,
        added_latency_p50_ms=float(np.percentile(dlat, 50)) if dlat else 0.0,
        added_latency_p95_ms=float(np.percentile(dlat, 95)) if dlat else 0.0,
        est_cost_usd=(embedded_tokens / 1_000_000.0 * 0.02)
        if embedded_tokens else calls * cost_per_call,
        embed_chunks=getattr(cache, "n_chunks", calls),
        embed_tokens=getattr(cache, "n_tokens", 0),
        embed_failures=getattr(cache, "n_failures", 0),
        embed_failed_chunks=getattr(cache, "n_failed_chunks", 0),
        embed_failed_tokens=getattr(cache, "n_failed_tokens", 0),
        fallbacks=getattr(index, "n_fallbacks", 0),
        kept=kept_count,
    )


def run_arm(stream, cfg: SamplerConfig, arm: str, seed: int = 0,
            synonym_map=None, keep_prob: Optional[float] = None) -> RunResult:
    """Run one sampling arm over the stream.

    Arms:
      * ``baseline`` -- ``BaselineSampler`` (random uniform, no variety index).
        Pass ``keep_prob`` to match its keep volume to an adaptive arm for a fair
        comparison; defaults to 0.5 if omitted.
      * ``adaptive_exact`` -- ``AdaptiveSampler`` + ``ExactSignatureIndex`` (the
        current-production, exact tool-call-signature variety comparison).
      * ``adaptive_cluster_offline`` / ``adaptive_cluster_azure`` --
        ``AdaptiveSampler`` + embedding-based ``AzureClusterIndex``.
    """
    if arm == "baseline":
        sampler = BaselineSampler(keep_prob=keep_prob if keep_prob is not None else 0.5,
                                  seed=seed)
        index = None
    else:
        index, use_novelty = _make_index(arm, cfg, synonym_map)
        sampler = AdaptiveSampler(cfg, seed=seed, variety_index=index,
                                  use_novelty=use_novelty)
    rows = []
    kept_count = 0
    decide_latencies_ms = []
    for t in stream:
        t0 = time.perf_counter()
        kept = sampler.decide(t)
        decide_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        kept_count += int(kept)
        if index is None:
            # Random baseline has no variety index: log the raw signature so
            # keep-based metrics (coverage/redundancy/latency) still apply, but
            # clustering metrics (ARI/V-measure) are intentionally not computed.
            variety_key = str(t.signature)
            key_kind = "signature"
        else:
            obs = sampler.last_observation
            variety_key = str(obs.key.value)
            key_kind = obs.key.kind
        cache = getattr(index, "_cache", None) if index is not None else None
        vector = cache_peek_trace(cache, t) if cache is not None else None
        rows.append(dict(
            timestamp=t.timestamp, agent_id=t.agent_id, concept_id=t.concept_id,
            signature=t.signature, variety_key=variety_key,
            key_kind=key_kind, kept=kept, embedding=vector,
        ))
    return RunResult(pd.DataFrame(rows), _ledger(index, kept_count, decide_latencies_ms), index)


def kept_embeddings(result: RunResult) -> Optional[np.ndarray]:
    """Read KEPT traces' existing vectors without triggering new embeddings."""
    cache = getattr(result.index, "_cache", None)
    if cache is None:
        return None
    kept = result.log[result.log["kept"]]
    vectors = kept["embedding"].tolist()
    available = [np.asarray(vector, dtype=float) for vector in vectors if vector is not None]
    if not available:
        return None
    dimensions = available[0].shape
    return np.array([
        np.asarray(vector, dtype=float)
        if vector is not None
        else np.full(dimensions, np.nan, dtype=float)
        for vector in vectors
    ])
