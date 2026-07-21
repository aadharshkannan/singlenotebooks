import numpy as np
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
