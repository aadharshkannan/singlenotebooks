import numpy as np
import pytest
from trace_sampling.embedding import FakeEmbedder, EmbeddingCache, sequence_to_text
from trace_sampling.model import Trace


def test_sequence_to_text():
    assert sequence_to_text(("search", "read", "edit")) == "search -> read -> edit"


def test_fake_embedder_groups_synonyms_by_canonical():
    from trace_sampling.concepts import SynonymMap
    sm = SynonymMap([["search", "query", "find"], ["run", "exec"]])
    fe = FakeEmbedder(dim=64, synonym_map=sm, noise=0.0, seed=0)

    def cos(x, y): return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    v_search = fe.embed(["search"])[0]
    v_query = fe.embed(["query"])[0]     # synonym of "search"
    v_run = fe.embed(["run"])[0]         # different concept
    assert cos(v_search, v_query) > 0.99
    assert cos(v_search, v_run) < 0.5


def test_fake_embedder_is_deterministic():
    fe = FakeEmbedder(dim=32, noise=0.0, seed=0)
    a = fe.embed(["search -> read"])[0]
    b = fe.embed(["search -> read"])[0]
    assert np.allclose(a, b)


def test_embedding_cache_hits_by_signature():
    calls = {"n": 0}
    class Counting:
        def embed(self, texts):
            calls["n"] += len(texts)
            return np.ones((len(texts), 4), dtype=np.float32)
    cache = EmbeddingCache(Counting(), max_size=8)
    cache.get(("search", "read"))
    cache.get(("search", "read"))   # cache hit -> no new call
    assert calls["n"] == 1
    assert cache.n_calls == 1 and cache.n_hits == 1


def test_build_openai_embedding_client_modern_endpoint_no_key_uses_string_token(monkeypatch):
    from trace_sampling.azure_config import AzureConfig
    from trace_sampling.embedding import build_openai_embedding_client

    seen = {}

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key):
            seen["base_url"] = base_url
            seen["api_key"] = api_key

    class FakeAzureOpenAI:
        def __init__(self, **kwargs):
            raise AssertionError("legacy client should not be used")

    monkeypatch.setattr("trace_sampling.azure_config.get_openai_token", lambda: "entra-token-value")

    cfg = AzureConfig(
        openai_endpoint="https://example.services.ai.azure.com",
        openai_api_version="2024-02-01",
        embedding_deployment="text-embedding-3-small",
        search_endpoint="https://search.example.com",
        search_index="trace-clusters",
        openai_api_key=None,
        search_api_key="search-secret",
    )

    client = build_openai_embedding_client(cfg, openai_cls=FakeOpenAI, azure_openai_cls=FakeAzureOpenAI)
    assert client is not None
    assert seen["base_url"] == "https://example.services.ai.azure.com/openai/v1/"
    assert isinstance(seen["api_key"], str)
    assert seen["api_key"] == "entra-token-value"
    assert not callable(seen["api_key"])


def test_build_openai_embedding_client_modern_endpoint_no_key_raises_when_no_token(monkeypatch):
    from trace_sampling.azure_config import AzureConfig
    from trace_sampling.embedding import build_openai_embedding_client

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key):
            self.base_url = base_url
            self.api_key = api_key

    monkeypatch.setattr("trace_sampling.azure_config.get_openai_token", lambda: "")

    cfg = AzureConfig(
        openai_endpoint="https://example.services.ai.azure.com",
        openai_api_version="2024-02-01",
        embedding_deployment="text-embedding-3-small",
        search_endpoint="https://search.example.com",
        search_index="trace-clusters",
        openai_api_key=None,
        search_api_key="search-secret",
    )

    with pytest.raises(RuntimeError, match="No usable OpenAI credential"):
        build_openai_embedding_client(cfg, openai_cls=FakeOpenAI)
