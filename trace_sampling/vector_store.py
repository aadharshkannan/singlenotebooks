from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Protocol, Tuple
import numpy as np


@dataclass
class VectorDoc:
    cluster_id: str
    vector: np.ndarray
    agent_id: str
    last_seen: float
    hits: int = 1
    semantic_scope: str = "legacy"
    tenant_id: str = "legacy"
    run_scope: str = "legacy"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


class VectorStore(Protocol):
    def nearest(
        self,
        vec: np.ndarray,
        agent_id: Optional[str] = None,
        semantic_scope: Optional[str] = None,
        tenant_id: Optional[str] = None,
        run_scope: Optional[str] = None,
    ) -> Optional[Tuple[str, float]]: ...
    def upsert(self, doc: VectorDoc) -> None: ...
    def touch(self, cluster_id: str, now: float) -> None: ...
    def purge_stale(
        self,
        now: float,
        ttl: float,
        semantic_scope: Optional[str] = None,
        tenant_id: Optional[str] = None,
        run_scope: Optional[str] = None,
    ) -> "List[str]": ...
    def delete_scope(
        self,
        tenant_id: str,
        run_scope: str,
        semantic_scope: Optional[str] = None,
    ) -> Tuple[List[str], int]: ...
    def delete_scope_settled(
        self,
        tenant_id: str,
        run_scope: str,
        semantic_scope: Optional[str] = None,
        *,
        max_attempts: int = 3,
        settle_seconds: float = 0.0,
    ) -> Tuple[List[str], int]: ...


class InMemoryVectorStore:
    """Deterministic exact-NN store for unit tests / offline metric passes."""

    def __init__(self):
        self._docs: Dict[str, VectorDoc] = {}
        self.n_queries = 0   # nearest() calls (ledger telemetry)

    def nearest(
        self,
        vec,
        agent_id=None,
        semantic_scope=None,
        tenant_id=None,
        run_scope=None,
    ):
        self.n_queries += 1
        best = None
        for doc in self._docs.values():
            if agent_id is not None and doc.agent_id != agent_id:
                continue
            if semantic_scope is not None and doc.semantic_scope != semantic_scope:
                continue
            if tenant_id is not None and doc.tenant_id != tenant_id:
                continue
            if run_scope is not None and doc.run_scope != run_scope:
                continue
            s = _cosine(vec, doc.vector)
            if best is None or s > best[1]:
                best = (doc.cluster_id, s)
        return best

    def upsert(self, doc: VectorDoc) -> None:
        self._docs[doc.cluster_id] = doc

    def touch(self, cluster_id: str, now: float) -> None:
        # refresh TTL only; the centroid vector is intentionally left unchanged
        d = self._docs.get(cluster_id)
        if d:
            d.last_seen = now

    def delete(self, cluster_id: str) -> None:
        self._docs.pop(cluster_id, None)

    def clear(self) -> int:
        n = len(self._docs)
        self._docs.clear()
        return n

    def purge_stale(
        self,
        now: float,
        ttl: float,
        semantic_scope: Optional[str] = None,
        tenant_id: Optional[str] = None,
        run_scope: Optional[str] = None,
    ) -> List[str]:
        stale = [
            cid
            for cid, doc in self._docs.items()
            if now - doc.last_seen > ttl
            and (semantic_scope is None or doc.semantic_scope == semantic_scope)
            and (tenant_id is None or doc.tenant_id == tenant_id)
            and (run_scope is None or doc.run_scope == run_scope)
        ]
        for cid in stale:
            del self._docs[cid]
        return stale

    def delete_scope(
        self,
        tenant_id: str,
        run_scope: str,
        semantic_scope: Optional[str] = None,
    ) -> Tuple[List[str], int]:
        ids = [
            cid
            for cid, doc in self._docs.items()
            if doc.tenant_id == tenant_id
            and doc.run_scope == run_scope
            and (semantic_scope is None or doc.semantic_scope == semantic_scope)
        ]
        for cid in ids:
            del self._docs[cid]
        return ids, len(ids)

    def delete_scope_settled(
        self,
        tenant_id: str,
        run_scope: str,
        semantic_scope: Optional[str] = None,
        *,
        max_attempts: int = 3,
        settle_seconds: float = 0.0,
    ) -> Tuple[List[str], int]:
        # In-memory deletes are immediately consistent; one scoped delete is sufficient.
        return self.delete_scope(tenant_id, run_scope, semantic_scope=semantic_scope)


class AzureSearchVectorStore:
    """Live Azure AI Search vector store. Uses API key when supplied, otherwise Entra."""

    def __init__(self, config, dim: int = 1536, ensure_index: bool = False):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from azure.identity import DefaultAzureCredential
        self._index = config.search_index
        self._endpoint = config.search_endpoint
        self._cred = AzureKeyCredential(config.search_api_key) if config.search_api_key else DefaultAzureCredential()
        self._dim = dim
        self._nearest_k = 1
        self._nearest_top = 1
        self.n_queries = 0   # nearest() calls (ledger telemetry)
        if ensure_index:
            self._ensure_index(config)
        self._client = SearchClient(self._endpoint, self._index, self._cred)

    @staticmethod
    def _odata_literal(value: str) -> str:
        return "'{}'".format(value.replace("'", "''"))

    def _build_filter(
        self,
        *,
        agent_id: Optional[str] = None,
        semantic_scope: Optional[str] = None,
        tenant_id: Optional[str] = None,
        run_scope: Optional[str] = None,
        older_than: Optional[float] = None,
    ) -> Optional[str]:
        clauses: List[str] = []
        if agent_id is not None:
            clauses.append(f"agent_id eq {self._odata_literal(agent_id)}")
        if semantic_scope is not None:
            clauses.append(f"semantic_scope eq {self._odata_literal(semantic_scope)}")
        if tenant_id is not None:
            clauses.append(f"tenant_id eq {self._odata_literal(tenant_id)}")
        if run_scope is not None:
            clauses.append(f"run_scope eq {self._odata_literal(run_scope)}")
        if older_than is not None:
            clauses.append(f"last_seen lt {older_than}")
        return " and ".join(clauses) or None

    def _search_ids(
        self,
        *,
        filter_expr: Optional[str],
        page_size: int = 1000,
        max_scan: int = 10000,
    ) -> List[str]:
        ids: List[str] = []
        skip = 0
        while skip < max_scan:
            page = list(
                self._client.search(
                    search_text=None,
                    filter=filter_expr,
                    select=["cluster_id"],
                    top=page_size,
                    skip=skip,
                )
            )
            if not page:
                break
            ids.extend(r["cluster_id"] for r in page)
            skip += page_size
        return ids

    def _delete_ids(self, ids: List[str], batch_size: int = 1000) -> None:
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            self._client.delete_documents([{"cluster_id": cid} for cid in batch])

    def _ensure_index(self, config):
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            SearchIndex, SimpleField, SearchField, SearchFieldDataType,
            VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
        )
        ic = SearchIndexClient(self._endpoint, self._cred)
        fields = [
            SimpleField(name="cluster_id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="agent_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="tenant_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="run_scope", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="semantic_scope", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="last_seen", type=SearchFieldDataType.Double,
                        filterable=True, sortable=True),
            SimpleField(name="hits", type=SearchFieldDataType.Int64),
            SearchField(name="vector",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=self._dim,
                        vector_search_profile_name="hnsw-cosine"),
        ]
        vs = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="hnsw-cosine", algorithm_configuration_name="hnsw")],
        )
        ic.create_or_update_index(SearchIndex(name=self._index, fields=fields, vector_search=vs))

    def nearest(
        self,
        vec,
        agent_id=None,
        semantic_scope=None,
        tenant_id=None,
        run_scope=None,
    ):
        from azure.search.documents.models import VectorizedQuery
        self.n_queries += 1
        vq = VectorizedQuery(
            vector=vec.tolist(),
            k_nearest_neighbors=self._nearest_k,
            fields="vector",
        )
        flt = self._build_filter(
            agent_id=agent_id,
            semantic_scope=semantic_scope,
            tenant_id=tenant_id,
            run_scope=run_scope,
        )
        results = self._client.search(
            search_text=None,
            vector_queries=[vq],
            filter=flt,
            top=self._nearest_top,
            vector_filter_mode="preFilter",
        )
        for r in results:
            # Azure AI Search maps cosine to @search.score = (1 + cosine) / 2 in [0, 1];
            # invert it back to raw cosine in [-1, 1] so callers (recent-buffer NN uses
            # raw cosine) and tau are all evaluated in one consistent scale.
            cosine = 2.0 * float(r["@search.score"]) - 1.0
            return (r["cluster_id"], cosine)
        return None

    def upsert(self, doc: VectorDoc) -> None:
        self._client.merge_or_upload_documents([{
            "cluster_id": doc.cluster_id, "agent_id": doc.agent_id,
            "tenant_id": doc.tenant_id, "run_scope": doc.run_scope,
            "semantic_scope": doc.semantic_scope,
            "last_seen": doc.last_seen, "hits": doc.hits,
            "vector": doc.vector.astype("float32").tolist(),
        }])

    def touch(self, cluster_id: str, now: float) -> None:
        # merge updates only the provided fields, leaving the centroid vector intact
        self._client.merge_or_upload_documents([{
            "cluster_id": cluster_id, "last_seen": now,
        }])

    def delete(self, cluster_id: str) -> None:
        self._client.delete_documents([{"cluster_id": cluster_id}])

    def clear(self) -> int:
        """Delete every document in the index so an eval run starts from a clean
        slate (the index otherwise persists clusters across runs, which pollutes
        clustering metrics). Returns the number of documents removed."""
        import time as _time
        total = 0
        for _ in range(20):
            ids = [r["cluster_id"] for r in
                   self._client.search(search_text="*", select=["cluster_id"], top=1000)]
            if not ids:
                break
            self._client.delete_documents([{"cluster_id": c} for c in ids])
            total += len(ids)
            _time.sleep(1.0)  # deletes are eventually consistent; let them commit
        return total

    def purge_stale(
        self,
        now: float,
        ttl: float,
        semantic_scope: Optional[str] = None,
        tenant_id: Optional[str] = None,
        run_scope: Optional[str] = None,
    ) -> List[str]:
        flt = self._build_filter(
            semantic_scope=semantic_scope,
            tenant_id=tenant_id,
            run_scope=run_scope,
            older_than=now - ttl,
        )
        ids = self._search_ids(filter_expr=flt)
        if ids:
            self._delete_ids(ids)
        return ids

    def delete_scope(
        self,
        tenant_id: str,
        run_scope: str,
        semantic_scope: Optional[str] = None,
    ) -> Tuple[List[str], int]:
        flt = self._build_filter(
            tenant_id=tenant_id,
            run_scope=run_scope,
            semantic_scope=semantic_scope,
        )
        ids = self._search_ids(filter_expr=flt)
        if ids:
            self._delete_ids(ids)
        return ids, len(ids)

    def delete_scope_settled(
        self,
        tenant_id: str,
        run_scope: str,
        semantic_scope: Optional[str] = None,
        *,
        max_attempts: int = 3,
        settle_seconds: float = 0.0,
    ) -> Tuple[List[str], int]:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if settle_seconds < 0.0:
            raise ValueError("settle_seconds must be >= 0")

        flt = self._build_filter(
            tenant_id=tenant_id,
            run_scope=run_scope,
            semantic_scope=semantic_scope,
        )
        deleted_all: List[str] = []
        deleted_seen: set[str] = set()
        remaining: List[str] = []
        consecutive_empty = 0

        for attempt in range(max_attempts):
            remaining = self._search_ids(filter_expr=flt)
            if not remaining:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    return deleted_all, len(deleted_all)
            else:
                consecutive_empty = 0
                self._delete_ids(remaining)
                for cid in remaining:
                    if cid not in deleted_seen:
                        deleted_all.append(cid)
                        deleted_seen.add(cid)
            if settle_seconds > 0.0 and attempt < max_attempts - 1:
                time.sleep(settle_seconds)

        remaining = self._search_ids(filter_expr=flt)
        if remaining:
            raise RuntimeError(
                "delete_scope_settled did not settle after "
                f"{max_attempts} attempts for tenant_id={tenant_id}, run_scope={run_scope}, "
                f"semantic_scope={semantic_scope!r}; remaining={len(remaining)}"
            )
        consecutive_empty += 1
        while consecutive_empty < 2:
            if settle_seconds > 0.0:
                time.sleep(settle_seconds)
            remaining = self._search_ids(filter_expr=flt)
            if remaining:
                raise RuntimeError(
                    "delete_scope_settled observed documents during the final quiet verification "
                    f"for tenant_id={tenant_id}, run_scope={run_scope}, "
                    f"semantic_scope={semantic_scope!r}; remaining={len(remaining)}"
                )
            consecutive_empty += 1
        return deleted_all, len(deleted_all)
