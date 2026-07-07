import numpy as np
from trace_sampling.model import Trace
from trace_sampling.embedding import FakeEmbedder, EmbeddingCache
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
