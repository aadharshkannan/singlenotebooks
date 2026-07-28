import numpy as np
import pytest
from trace_sampling.vector_store import InMemoryVectorStore, VectorDoc


def test_inmemory_nearest_and_upsert():
    vs = InMemoryVectorStore()
    assert vs.nearest(np.array([1.0, 0.0]), agent_id="a") is None
    vs.upsert(VectorDoc("c0", np.array([1.0, 0.0]), "a", last_seen=0.0))
    cid, score = vs.nearest(np.array([0.9, 0.1]), agent_id="a")
    assert cid == "c0" and score > 0.9


def test_inmemory_agent_scoping():
    vs = InMemoryVectorStore()
    vs.upsert(VectorDoc("c0", np.array([1.0, 0.0]), "a", last_seen=0.0))
    assert vs.nearest(np.array([1.0, 0.0]), agent_id="b") is None


def test_inmemory_semantic_scope_isolates_representation_lineage():
    vs = InMemoryVectorStore()
    vector = np.array([1.0, 0.0])
    vs.upsert(
        VectorDoc(
            "c0",
            vector,
            "a",
            last_seen=0.0,
            semantic_scope="representation-v1",
        )
    )

    assert vs.nearest(vector, agent_id="a", semantic_scope="representation-v2") is None
    assert vs.nearest(vector, agent_id="a", semantic_scope="representation-v1") == (
        "c0",
        1.0,
    )


def test_inmemory_purge_stale():
    vs = InMemoryVectorStore()
    vs.upsert(VectorDoc("c0", np.array([1.0, 0.0]), "a", last_seen=0.0))
    vs.upsert(VectorDoc("c1", np.array([0.0, 1.0]), "a", last_seen=100.0))
    removed = vs.purge_stale(now=100.0, ttl=50.0)
    assert removed == ["c0"]
    res = vs.nearest(np.array([1.0, 0.0]), agent_id="a")
    assert res is not None and res[0] == "c1"


def test_inmemory_unscoped_purge_removes_stale_docs_across_lineages():
    vs = InMemoryVectorStore()
    vector = np.array([1.0, 0.0])
    vs.upsert(VectorDoc("old-v1", vector, "a", 0.0, semantic_scope="v1"))
    vs.upsert(VectorDoc("old-v2", vector, "a", 0.0, semantic_scope="v2"))

    assert set(vs.purge_stale(now=100.0, ttl=50.0)) == {"old-v1", "old-v2"}


def test_inmemory_touch_updates_last_seen_only():
    vs = InMemoryVectorStore()
    vs.upsert(VectorDoc("c0", np.array([1.0, 0.0]), "a", last_seen=10.0))
    vs.touch("c0", 20.0)
    d = vs._docs["c0"]
    assert d.last_seen == 20.0
    assert np.allclose(d.vector, np.array([1.0, 0.0]))  # centroid unchanged


def test_inmemory_delete_removes_doc():
    vs = InMemoryVectorStore()
    vs.upsert(VectorDoc("c0", np.array([1.0, 0.0]), "a", last_seen=0.0))
    vs.delete("c0")
    assert vs.nearest(np.array([1.0, 0.0]), agent_id="a") is None


@pytest.mark.azure
def test_azure_search_roundtrip():
    from trace_sampling.azure_config import AzureConfig
    from trace_sampling.vector_store import AzureSearchVectorStore
    vs = AzureSearchVectorStore(AzureConfig.from_env(), dim=1536, ensure_index=True)
    v = np.random.default_rng(0).normal(size=1536).astype("float32")
    vs.upsert(VectorDoc("smoke-c0", v, "smoke-agent", last_seen=0.0))
    import time; time.sleep(2)
    res = vs.nearest(v, agent_id="smoke-agent")
    assert res is not None and res[0] == "smoke-c0"
    vs.delete("smoke-c0")
