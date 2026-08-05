import numpy as np
import pytest
from trace_sampling.vector_store import (
    AzureSearchVectorStore,
    InMemoryVectorStore,
    VectorDoc,
)


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


def test_vectordoc_backcompat_defaults_for_tenant_and_run_scope():
    doc = VectorDoc("c0", np.array([1.0, 0.0]), "a", last_seen=0.0)
    assert doc.tenant_id == "legacy"
    assert doc.run_scope == "legacy"


def test_inmemory_nearest_enforces_tenant_and_run_scope_filters():
    vs = InMemoryVectorStore()
    vector = np.array([1.0, 0.0])
    vs.upsert(
        VectorDoc(
            "c0",
            vector,
            "a",
            last_seen=0.0,
            semantic_scope="s1",
            tenant_id="tenant-a",
            run_scope="run-1",
        )
    )
    assert (
        vs.nearest(
            vector,
            agent_id="a",
            semantic_scope="s1",
            tenant_id="tenant-a",
            run_scope="run-1",
        )
        == ("c0", 1.0)
    )
    assert (
        vs.nearest(
            vector,
            agent_id="a",
            semantic_scope="s1",
            tenant_id="tenant-x",
            run_scope="run-1",
        )
        is None
    )
    assert (
        vs.nearest(
            vector,
            agent_id="a",
            semantic_scope="s1",
            tenant_id="tenant-a",
            run_scope="run-x",
        )
        is None
    )


def test_inmemory_purge_stale_respects_tenant_and_run_scope_isolation():
    vs = InMemoryVectorStore()
    vector = np.array([1.0, 0.0])
    vs.upsert(VectorDoc("a", vector, "agent", 0.0, tenant_id="t1", run_scope="r1"))
    vs.upsert(VectorDoc("b", vector, "agent", 0.0, tenant_id="t1", run_scope="r2"))
    vs.upsert(VectorDoc("c", vector, "agent", 0.0, tenant_id="t2", run_scope="r1"))
    removed = vs.purge_stale(now=100.0, ttl=50.0, tenant_id="t1", run_scope="r1")
    assert removed == ["a"]
    assert set(vs._docs.keys()) == {"b", "c"}


def test_inmemory_delete_scope_does_not_delete_other_tenant_or_run():
    vs = InMemoryVectorStore()
    vector = np.array([1.0, 0.0])
    vs.upsert(VectorDoc("a", vector, "agent", 0.0, tenant_id="t1", run_scope="r1"))
    vs.upsert(VectorDoc("b", vector, "agent", 0.0, tenant_id="t1", run_scope="r2"))
    vs.upsert(VectorDoc("c", vector, "agent", 0.0, tenant_id="t2", run_scope="r1"))
    ids, count = vs.delete_scope("t1", "r1")
    assert ids == ["a"]
    assert count == 1
    assert set(vs._docs.keys()) == {"b", "c"}


def test_inmemory_delete_scope_settled_completes_immediately_without_sleep():
    vs = InMemoryVectorStore()
    vector = np.array([1.0, 0.0])
    vs.upsert(VectorDoc("a", vector, "agent", 0.0, tenant_id="t1", run_scope="r1", semantic_scope="s1"))
    vs.upsert(VectorDoc("b", vector, "agent", 0.0, tenant_id="t1", run_scope="r2", semantic_scope="s1"))
    ids, count = vs.delete_scope_settled("t1", "r1", semantic_scope="s1", max_attempts=5, settle_seconds=0.0)
    assert ids == ["a"]
    assert count == 1
    assert set(vs._docs.keys()) == {"b"}


class _FakeSearchClient:
    def __init__(self, pages=None):
        self.pages = pages if pages is not None else [[{"cluster_id": "doc-1", "@search.score": 0.9}]]
        self.search_calls = []
        self.upload_calls = []
        self.delete_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        vector_queries = kwargs.get("vector_queries")
        if vector_queries is not None:
            return iter(self.pages[0])
        skip = kwargs.get("skip", 0)
        top = kwargs.get("top", 1000)
        page_idx = int(skip / top)
        if page_idx >= len(self.pages):
            return iter([])
        return iter(self.pages[page_idx])

    def merge_or_upload_documents(self, docs):
        self.upload_calls.append(docs)

    def delete_documents(self, docs):
        self.delete_calls.append(docs)


class _SettlingFakeSearchClient(_FakeSearchClient):
    def __init__(self):
        super().__init__(pages=[])
        self._search_seq = 0

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        vector_queries = kwargs.get("vector_queries")
        if vector_queries is not None:
            return iter([])
        if int(kwargs.get("skip", 0)) > 0:
            return iter([])

        self._search_seq += 1
        if self._search_seq == 1:
            return iter([{"cluster_id": "late-doc"}])
        if self._search_seq == 2:
            return iter([{"cluster_id": "late-doc"}])
        return iter([])


def _build_azure_store(fake_client: _FakeSearchClient) -> AzureSearchVectorStore:
    store = AzureSearchVectorStore.__new__(AzureSearchVectorStore)
    store._client = fake_client
    store._dim = 1536
    store._nearest_k = 2
    store._nearest_top = 3
    store.n_queries = 0
    return store


def test_azure_nearest_sets_prefilter_and_escapes_odata_literals():
    fake = _FakeSearchClient(pages=[[{"cluster_id": "doc-1", "@search.score": 0.75}]])
    store = _build_azure_store(fake)
    vec = np.array([1.0, 0.0], dtype=np.float32)
    cid, cosine = store.nearest(
        vec,
        agent_id="agent'o",
        semantic_scope="scope'o",
        tenant_id="tenant'o",
        run_scope="run'o",
    )

    assert cid == "doc-1"
    assert cosine == pytest.approx(0.5)
    call = fake.search_calls[-1]
    assert call["vector_filter_mode"] == "preFilter"
    assert call["top"] == 3
    assert "agent_id eq 'agent''o'" in call["filter"]
    assert "semantic_scope eq 'scope''o'" in call["filter"]
    assert "tenant_id eq 'tenant''o'" in call["filter"]
    assert "run_scope eq 'run''o'" in call["filter"]


def test_azure_upsert_includes_tenant_and_run_scope_fields():
    fake = _FakeSearchClient()
    store = _build_azure_store(fake)
    doc = VectorDoc(
        "c1",
        np.array([1.0, 0.0], dtype=np.float32),
        "agent-1",
        12.0,
        semantic_scope="scope-1",
        tenant_id="tenant-1",
        run_scope="run-1",
    )
    store.upsert(doc)
    payload = fake.upload_calls[-1][0]
    assert payload["tenant_id"] == "tenant-1"
    assert payload["run_scope"] == "run-1"


def test_azure_purge_stale_scoped_and_paged_delete():
    fake = _FakeSearchClient(
        pages=[
            [{"cluster_id": "a"}],
            [{"cluster_id": "b"}],
            [],
        ]
    )
    store = _build_azure_store(fake)
    ids = store.purge_stale(
        now=100.0,
        ttl=50.0,
        semantic_scope="scope-1",
        tenant_id="tenant-1",
        run_scope="run-1",
    )
    assert ids == ["a", "b"]
    assert len(fake.delete_calls) == 1
    assert fake.delete_calls[0] == [{"cluster_id": "a"}, {"cluster_id": "b"}]
    first_search = fake.search_calls[0]
    assert "last_seen lt 50.0" in first_search["filter"]
    assert "semantic_scope eq 'scope-1'" in first_search["filter"]
    assert "tenant_id eq 'tenant-1'" in first_search["filter"]
    assert "run_scope eq 'run-1'" in first_search["filter"]


def test_azure_delete_scope_only_target_tenant_run_and_optional_semantic_scope():
    fake = _FakeSearchClient(
        pages=[
            [{"cluster_id": "t1r1s1-a"}, {"cluster_id": "t1r1s1-b"}],
            [],
        ]
    )
    store = _build_azure_store(fake)
    ids, count = store.delete_scope("tenant-1", "run-1", semantic_scope="scope-1")
    assert ids == ["t1r1s1-a", "t1r1s1-b"]
    assert count == 2
    flt = fake.search_calls[0]["filter"]
    assert "tenant_id eq 'tenant-1'" in flt
    assert "run_scope eq 'run-1'" in flt
    assert "semantic_scope eq 'scope-1'" in flt
    assert len(fake.delete_calls) == 1
    assert fake.delete_calls[0] == [
        {"cluster_id": "t1r1s1-a"},
        {"cluster_id": "t1r1s1-b"},
    ]


def test_azure_delete_scope_settled_eventual_consistency_deletes_late_arrival():
    fake = _SettlingFakeSearchClient()
    store = _build_azure_store(fake)

    ids, count = store.delete_scope_settled(
        "tenant-1",
        "run-1",
        semantic_scope="scope-1",
        max_attempts=4,
        settle_seconds=0.0,
    )

    assert ids == ["late-doc"]
    assert count == 1
    assert len(fake.delete_calls) == 2
    assert fake.delete_calls[0] == [{"cluster_id": "late-doc"}]
    assert fake.delete_calls[1] == [{"cluster_id": "late-doc"}]
    assert len(fake.search_calls) >= 3
    flt = fake.search_calls[0]["filter"]
    assert "tenant_id eq 'tenant-1'" in flt
    assert "run_scope eq 'run-1'" in flt
    assert "semantic_scope eq 'scope-1'" in flt


def test_azure_delete_scope_settled_waits_for_late_document_after_initial_empty(monkeypatch):
    class _InitiallyEmptyClient(_FakeSearchClient):
        def __init__(self):
            super().__init__(pages=[])
            self.scope_scans = 0

        def search(self, **kwargs):
            self.search_calls.append(kwargs)
            if kwargs.get("vector_queries") is not None:
                return iter([])
            self.scope_scans += 1
            if self.scope_scans == 1:
                return iter([])
            if self.scope_scans == 2:
                return iter([{"cluster_id": "late-after-empty"}])
            return iter([])

    sleeps = []
    monkeypatch.setattr("trace_sampling.vector_store.time.sleep", sleeps.append)
    fake = _InitiallyEmptyClient()
    store = _build_azure_store(fake)

    ids, count = store.delete_scope_settled(
        "tenant-1",
        "run-1",
        semantic_scope="scope-1",
        max_attempts=3,
        settle_seconds=1.0,
    )

    assert ids == ["late-after-empty"]
    assert count == 1
    assert fake.delete_calls == [[{"cluster_id": "late-after-empty"}]]
    assert sleeps == [1.0, 1.0]
    assert fake.scope_scans >= 4


def test_azure_delete_scope_settled_raises_when_not_settled_after_attempts():
    class _NeverSettlesClient(_FakeSearchClient):
        def search(self, **kwargs):
            self.search_calls.append(kwargs)
            if kwargs.get("vector_queries") is not None:
                return iter([])
            return iter([{"cluster_id": "stuck-doc"}])

    fake = _NeverSettlesClient()
    store = _build_azure_store(fake)

    with pytest.raises(RuntimeError, match="did not settle"):
        store.delete_scope_settled(
            "tenant-1",
            "run-1",
            semantic_scope="scope-1",
            max_attempts=2,
            settle_seconds=0.0,
        )


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
