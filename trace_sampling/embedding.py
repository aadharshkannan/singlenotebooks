from collections import OrderedDict
from typing import List, Optional, Protocol, Tuple
import numpy as np


def _is_modern_foundry_endpoint(endpoint: str) -> bool:
    if not endpoint:
        return False
    host = endpoint.lower().split("//", 1)[-1].split("/", 1)[0]
    return host.endswith(".services.ai.azure.com") or host == "services.ai.azure.com"


def _openai_base_url(endpoint: str) -> str:
    return endpoint.rstrip("/") + "/openai/v1/"


def build_openai_embedding_client(config, *, openai_cls=None, azure_openai_cls=None):
    """Construct the correct OpenAI client for modern Foundry vs classic Azure endpoints."""
    if openai_cls is None:
        from openai import OpenAI as openai_cls
    if azure_openai_cls is None:
        from openai import AzureOpenAI as azure_openai_cls

    endpoint = (config.openai_endpoint or "").rstrip("/")
    if _is_modern_foundry_endpoint(config.openai_endpoint):
        api_key = config.openai_api_key
        if api_key:
            return openai_cls(base_url=_openai_base_url(endpoint), api_key=api_key)
        from .azure_config import get_openai_token

        token = get_openai_token()
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError(
                "No usable OpenAI credential for Foundry endpoint. Set AZURE_OPENAI_API_KEY or ensure Entra token acquisition succeeds."
            )
        return openai_cls(base_url=_openai_base_url(endpoint), api_key=token)

    if config.openai_api_key:
        return azure_openai_cls(
            azure_endpoint=endpoint,
            api_version=config.openai_api_version,
            api_key=config.openai_api_key,
        )

    from .azure_config import openai_token_provider
    return azure_openai_cls(
        azure_endpoint=endpoint,
        api_version=config.openai_api_version,
        azure_ad_token_provider=openai_token_provider(),
    )


def sequence_to_text(signature: Tuple[str, ...]) -> str:
    return " -> ".join(signature)


class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray: ...


class FakeEmbedder:
    """Deterministic, offline embedder for unit tests.

    Builds each text's vector as the SUM of per-token basis vectors, where every
    token is first mapped to its canonical synonym (via an optional ``SynonymMap``).
    Because synonyms collapse to the same canonical token, two surface sequences
    that express the same concept with different vocabulary embed to (near-)identical
    vectors — mimicking a real semantic embedder — while unrelated tokens are far
    apart. This is what lets the offline unit tests exercise concept unification
    deterministically, with no network. noise: gaussian scale added per-text
    (deterministic per text+seed), so the same text always embeds identically."""

    def __init__(self, dim: int = 64, synonym_map=None, noise: float = 0.01, seed: int = 0):
        self.dim = dim
        self.synonym_map = synonym_map
        self.noise = noise
        self.seed = seed

    def _token_vec(self, token: str) -> np.ndarray:
        import hashlib
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % (2**31)
        v = np.random.default_rng([self.seed, h]).normal(size=self.dim)
        n = np.linalg.norm(v)
        return v / (n or 1.0)

    def _canonical(self, token: str) -> str:
        return self.synonym_map.canonical(token) if self.synonym_map is not None else token

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, txt in enumerate(texts):
            tokens = [t.strip() for t in txt.split("->") if t.strip()]
            vec = np.zeros(self.dim, dtype=np.float64)
            for tok in tokens:
                vec += self._token_vec(self._canonical(tok))
            if self.noise:
                import hashlib
                h = int(hashlib.md5(txt.encode("utf-8")).hexdigest(), 16) % (2**31)
                vec += np.random.default_rng([self.seed, h]).normal(scale=self.noise, size=self.dim)
            out[i] = vec.astype(np.float32)
        return out


class AzureOpenAIEmbedder:
    """Live Azure OpenAI embeddings via API key or Entra tokens."""

    def __init__(self, config):
        self._deployment = config.embedding_deployment
        self._client = build_openai_embedding_client(config)

    def embed(self, texts: List[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._deployment, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


class EmbeddingCache:
    """Bounded LRU cache keyed by the signature tuple -> vector. Tracks call/hit
    counters and per-miss embed latency for the eval cost/latency ledger."""

    def __init__(self, embedder: Embedder, max_size: int = 4096):
        self._embedder = embedder
        self._max = max_size
        self._cache: "OrderedDict[Tuple[str, ...], np.ndarray]" = OrderedDict()
        self.n_calls = 0                 # underlying embed() invocations (cache misses)
        self.n_hits = 0                  # cache hits
        self.embed_latencies_ms = []     # wall-clock ms per miss (for p50/p95)

    def get(self, signature: Tuple[str, ...]) -> np.ndarray:
        if signature in self._cache:
            self._cache.move_to_end(signature)
            self.n_hits += 1
            return self._cache[signature]
        import time
        t0 = time.perf_counter()
        vec = self._embedder.embed([sequence_to_text(signature)])[0]
        self.embed_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        self.n_calls += 1
        self._cache[signature] = vec
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return vec

    def contains_trace(self, trace) -> bool:
        return trace.signature in self

    def peek_trace(self, trace):
        return self._cache.get(trace.signature)

    def get_trace(self, trace) -> np.ndarray:
        return self.get(trace.signature)

    def __contains__(self, signature) -> bool:
        return signature in self._cache


def cache_contains_trace(cache, trace) -> bool:
    method = getattr(cache, "contains_trace", None)
    return method(trace) if method is not None else trace.signature in cache


def cache_get_trace(cache, trace) -> np.ndarray:
    method = getattr(cache, "get_trace", None)
    return method(trace) if method is not None else cache.get(trace.signature)


def cache_peek_trace(cache, trace):
    method = getattr(cache, "peek_trace", None)
    if method is not None:
        record = method(trace)
        return None if record is None else getattr(record, "vector", record)
    return cache.get(trace.signature) if trace.signature in cache else None
