import numpy as np
import pytest

from trace_sampling.cosmos_vector_store import CosmosVectorStore
from trace_sampling.vector_store import VectorDoc


class FakeContainer:
    def __init__(self):
        self.documents = []
        self.nearest_call = None
        self.patch_call = None

    def upsert_item(self, document):
        self.documents.append(document)

    def nearest(self, **kwargs):
        self.nearest_call = kwargs
        return {"cluster_id": "c0", "distance": 0.25}

    def patch_cluster(self, **kwargs):
        self.patch_call = kwargs

    def delete_stale(self, **kwargs):
        assert kwargs == {
            "vector_space_id": "session-v1",
            "older_than": 40.0,
        }
        return ["c-old"]


def _store(container):
    return CosmosVectorStore(
        container,
        vector_space_id="session-v1",
        dimensions=2,
        distance_to_cosine=lambda distance: 1.0 - distance,
    )


def test_cosmos_store_scopes_documents_and_queries_to_vector_space_and_agent():
    container = FakeContainer()
    store = _store(container)
    vector = np.array([1.0, 0.0], dtype=np.float32)

    store.upsert(
        VectorDoc(
            "c0",
            vector,
            "agent-a",
            last_seen=10.0,
            semantic_scope="representation-v2",
        )
    )
    nearest = store.nearest(
        vector,
        agent_id="agent-a",
        semantic_scope="representation-v2",
    )

    document = container.documents[0]
    assert document["id"] == "session-v1|c0"
    assert document["partition_key"] == "session-v1|representation-v2|agent-a"
    assert document["schema_version"] == "cluster-vector-v1"
    assert document["semantic_scope"] == "representation-v2"
    assert document["vector"] == [1.0, 0.0]
    assert container.nearest_call["partition_key"] == (
        "session-v1|representation-v2|agent-a"
    )
    assert nearest == ("c0", 0.75)


def test_cosmos_touch_is_metadata_only_and_purge_returns_cluster_ids():
    container = FakeContainer()
    store = _store(container)

    store.touch("c0", 20.0)

    assert container.patch_call["changes"] == {"last_seen": 20.0}
    assert store.purge_stale(now=100.0, ttl=60.0) == ["c-old"]


def test_cosmos_store_rejects_cross_partition_and_non_unit_vectors():
    store = _store(FakeContainer())

    with pytest.raises(ValueError, match="agent_id"):
        store.nearest(np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="unit-normalized"):
        store.upsert(VectorDoc("c0", np.array([2.0, 0.0]), "a", 0.0))
