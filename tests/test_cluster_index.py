import os
import numpy as np
import pytest
from trace_sampling.model import SessionEvent, Trace
from trace_sampling.embedding import FakeEmbedder, EmbeddingCache
from trace_sampling.session_embedding import EmbeddingProfile, SessionEmbeddingCache
from trace_sampling.vector_store import InMemoryVectorStore
from trace_sampling.cluster_index import CircuitBreaker, AzureClusterIndex
from trace_sampling.variety import VarietyKey


def test_breaker_opens_after_threshold_and_recovers():
    b = CircuitBreaker(fail_threshold=2, cooldown_s=10.0)
    assert b.allow(now=0.0)
    b.on_failure(now=0.0); b.on_failure(now=0.0)     # 2 failures -> open
    assert not b.allow(now=1.0)                       # within cooldown
    assert b.allow(now=11.0)                          # half-open after cooldown
    b.on_success()                                    # closes
    assert b.allow(now=12.0)


def _index(**kw):
    from trace_sampling.concepts import SynonymMap
    sm = SynonymMap([["search", "query", "find"], ["edit", "modify"], ["deploy", "release"]])
    fe = FakeEmbedder(dim=64, synonym_map=sm, noise=0.005, seed=0)
    return AzureClusterIndex(EmbeddingCache(fe), InMemoryVectorStore(), tau=0.9, **kw)


def _t(sig, agent="a", ts=0.0, cid=0):
    return Trace(0, agent, ts, sig, len(sig), 1.0, "ok", concept_id=cid)


def test_first_trace_is_new_cluster_and_novel():
    idx = _index()
    obs = idx.observe(_t(("search",), cid=0))
    assert obs.key.kind == "cluster"
    assert obs.novelty == 1.0

def test_same_concept_joins_same_cluster():
    idx = _index()
    a = idx.observe(_t(("search",), agent="a", cid=0))
    b = idx.observe(_t(("query",), agent="a", cid=0))
    assert a.key == b.key
    assert b.novelty < 0.5

def test_different_concept_makes_new_cluster():
    idx = _index()
    idx.observe(_t(("search",), agent="a", cid=0))
    obs = idx.observe(_t(("deploy",), agent="a", cid=2))
    assert obs.novelty > 0.5

def test_ttl_purge_reflags_returning_behavior_as_new():
    idx = _index(ttl=10.0, purge_every=1)
    first = idx.observe(_t(("search",), agent="a", ts=0.0, cid=0))
    later = idx.observe(_t(("search",), agent="a", ts=100.0, cid=0))
    assert later.key != first.key
    assert later.novelty == 1.0

def test_embed_budget_exhaustion_falls_back_to_signature():
    idx = _index(embed_budget_per_tick=0)
    obs = idx.observe(_t(("search",), cid=0))
    assert obs.key.kind == "fallback-signature"


def test_session_cache_clusters_same_signature_by_full_session_content():
    class CharacterTokenizer:
        name = "characters"
        version = "1"

        def count(self, text):
            return len(text)

    class ContentEmbedder:
        def __init__(self):
            self.inputs = []

        def embed(self, texts):
            self.inputs.extend(texts)
            return np.array([
                [1.0, 0.0] if "alpha-result" in text else [0.0, 1.0]
                for text in texts
            ], dtype=np.float32)

    embedder = ContentEmbedder()
    profile = EmbeddingProfile(
        model_id="test",
        model_version="1",
        tokenizer_id="characters",
        tokenizer_version="1",
        max_input_tokens=1000,
    )
    cache = SessionEmbeddingCache(embedder, CharacterTokenizer(), profile)
    index = AzureClusterIndex(cache, InMemoryVectorStore(), tau=0.9)
    alpha = Trace(
        1, "a", 0.0, ("search",), 1, 1.0, "ok",
        events=(SessionEvent("tool", tool_name="search", output="alpha-result"),),
    )
    beta = Trace(
        2, "a", 1.0, ("search",), 1, 1.0, "ok",
        events=(SessionEvent("tool", tool_name="search", output="beta-result"),),
    )

    first = index.observe(alpha)
    second = index.observe(beta)

    assert first.key != second.key
    assert cache.n_calls == 2
    assert any("alpha-result" in text for text in embedder.inputs)
    assert any("beta-result" in text for text in embedder.inputs)

def test_fallback_preserves_exact_signature_rarity():
    idx = _index(embed_budget_per_tick=0)
    a = idx.observe(_t(("search",), agent="a", cid=0))
    b = idx.observe(_t(("search",), agent="a", cid=0))
    assert a.key.kind == "fallback-signature" and b.key.kind == "fallback-signature"
    assert a.rarity == 0.5
    assert b.rarity == 1.0 / 3.0

def test_recent_low_sim_does_not_suppress_store_match():
    idx = _index(recent_buffer_size=64)
    # create the "deploy" cluster; it goes into the recent buffer
    idx.observe(_t(("deploy",), agent="a", cid=2))
    # create a "search" cluster, then evict it from the recent buffer only,
    # leaving it live in the store, by pushing enough NEW clusters past the buffer cap.
    first_search = idx.observe(_t(("search",), agent="a", cid=0))
    # shrink the recent buffer so the search cluster is dropped but stays in the store
    idx._recent = [e for e in idx._recent if e[0] != first_search.key.value]
    # a synonym of search should JOIN the store's search cluster, not spawn a new one
    again = idx.observe(_t(("query",), agent="a", cid=0))
    assert again.key == first_search.key
    assert again.novelty < 0.5


def test_novelty_is_binary():
    idx = _index()
    new = idx.observe(_t(("search",), agent="a", cid=0))
    assert new.novelty == 1.0                      # new cluster
    join = idx.observe(_t(("query",), agent="a", cid=0))
    assert join.novelty == 0.0                     # joins existing cluster -> exactly 0


def test_purge_drops_per_cluster_state():
    idx = _index(ttl=10.0, purge_every=1)
    first = idx.observe(_t(("search",), agent="a", ts=0.0, cid=0))
    cid = first.key.value
    idx.observe(_t(("query",), agent="a", ts=1.0, cid=0))     # join -> seeds _iat[cid]
    assert cid in idx._iat and cid in idx._last_seen
    # far-future trace triggers purge of the now-stale cluster
    idx.observe(_t(("search",), agent="a", ts=100.0, cid=0))
    assert cid not in idx._iat
    assert cid not in idx._last_seen


def test_creation_has_zero_staleness_rarity():
    idx = _index()
    new = idx.observe(_t(("search",), agent="a", ts=0.0, cid=0))
    assert new.rarity == 0.0                        # nothing stale on first sight

def test_zero_gap_join_has_zero_staleness():
    # Two joins at the identical timestamp -> dt=0 -> staleness exactly 0, no div-by-zero.
    idx = _index()
    idx.observe(_t(("search",), agent="a", ts=5.0, cid=0))   # create
    idx.observe(_t(("query",),  agent="a", ts=6.0, cid=0))   # first join seeds iat
    same_ts = idx.observe(_t(("find",), agent="a", ts=6.0, cid=0))  # dt = 0
    assert same_ts.rarity == 0.0

def test_staleness_grows_with_gap():
    # Two indices, identical except the returning gap; larger gap -> larger rarity.
    def run(gap):
        idx = _index(k=8.0)
        idx.observe(_t(("search",), agent="a", ts=0.0, cid=0))   # create
        idx.observe(_t(("query",),  agent="a", ts=1.0, cid=0))   # first join, seeds iat=1
        return idx.observe(_t(("find",), agent="a", ts=1.0 + gap, cid=0)).rarity
    assert run(50.0) > run(1.0) > 0.0

def test_cadence_normalization_converges_across_speeds():
    # A regularly-hit cluster reaches steady-state staleness ~ 1 - 2^(-1/k),
    # independent of absolute cadence. Compare a fast and a slow cluster.
    k = 8.0
    expected = 1 - 2 ** (-1.0 / k)
    def steady(step):
        idx = _index(k=k)
        idx.observe(_t(("search",), agent="a", ts=0.0, cid=0))    # create
        r = 0.0
        for i in range(1, 6):
            r = idx.observe(_t(("query",), agent="a", ts=i * step, cid=0)).rarity
        return r
    fast, slow = steady(0.5), steady(50.0)
    assert abs(fast - expected) < 0.05
    assert abs(slow - expected) < 0.05
    assert abs(fast - slow) < 0.05


def test_ctor_rejects_bad_k():
    with pytest.raises(ValueError):
        _index(k=0.0)

def test_ctor_rejects_bad_iat_alpha():
    with pytest.raises(ValueError):
        _index(iat_alpha=0.0)
    with pytest.raises(ValueError):
        _index(iat_alpha=1.5)

def test_ctor_defaults_present():
    idx = _index()
    assert idx.k == 16.0
    assert idx.iat_alpha == 0.3
    assert not hasattr(idx, "_decay_half_life")


@pytest.mark.azure
def test_azure_cluster_index_end_to_end():
    from trace_sampling.azure_config import AzureConfig
    from trace_sampling.embedding import AzureOpenAIEmbedder, EmbeddingCache
    from trace_sampling.vector_store import AzureSearchVectorStore
    from trace_sampling.cluster_index import AzureClusterIndex
    cfg = AzureConfig.from_env()
    idx = AzureClusterIndex(
        EmbeddingCache(AzureOpenAIEmbedder(cfg)),
        AzureSearchVectorStore(cfg, dim=1536, ensure_index=True),
        tau=0.55)
    a = idx.observe(_t(("search", "read"), agent="live", cid=0))
    b = idx.observe(_t(("query", "read"), agent="live", cid=0))
    assert a.key.kind == "cluster" and b.key.kind == "cluster"
    assert a.key == b.key
    c = idx.observe(_t(("deploy", "release"), agent="live", cid=1))
    assert c.key != a.key
