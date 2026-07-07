from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple
import numpy as np


@dataclass
class VectorDoc:
    cluster_id: str
    vector: np.ndarray
    agent_id: str
    last_seen: float
    hits: int = 1


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


class VectorStore(Protocol):
    def nearest(self, vec: np.ndarray, agent_id: Optional[str] = None) -> Optional[Tuple[str, float]]: ...
    def upsert(self, doc: VectorDoc) -> None: ...
    def touch(self, cluster_id: str, now: float) -> None: ...
    def purge_stale(self, now: float, ttl: float) -> "List[str]": ...


class InMemoryVectorStore:
    """Deterministic exact-NN store for unit tests / offline metric passes."""

    def __init__(self):
        self._docs: Dict[str, VectorDoc] = {}
        self.n_queries = 0   # nearest() calls (ledger telemetry)

    def nearest(self, vec, agent_id=None):
        self.n_queries += 1
        best = None
        for doc in self._docs.values():
            if agent_id is not None and doc.agent_id != agent_id:
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

    def purge_stale(self, now: float, ttl: float) -> List[str]:
        stale = [cid for cid, d in self._docs.items() if now - d.last_seen > ttl]
        for cid in stale:
            del self._docs[cid]
        return stale


class AzureSearchVectorStore:
    """Live Azure AI Search vector store (Entra-only). HNSW + cosine."""

    def __init__(self, config, dim: int = 1536, ensure_index: bool = False):
        from azure.search.documents import SearchClient
        from .azure_config import get_credential
        self._index = config.search_index
        self._endpoint = config.search_endpoint
        self._cred = get_credential()
        self._dim = dim
        self.n_queries = 0   # nearest() calls (ledger telemetry)
        if ensure_index:
            self._ensure_index(config)
        self._client = SearchClient(self._endpoint, self._index, self._cred)

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

    def nearest(self, vec, agent_id=None):
        from azure.search.documents.models import VectorizedQuery
        self.n_queries += 1
        vq = VectorizedQuery(vector=vec.tolist(), k_nearest_neighbors=1, fields="vector")
        if agent_id is not None:
            flt = "agent_id eq '{}'".format(agent_id.replace("'", "''"))
        else:
            flt = None
        results = self._client.search(search_text=None, vector_queries=[vq], filter=flt, top=1)
        for r in results:
            return (r["cluster_id"], float(r["@search.score"]))
        return None

    def upsert(self, doc: VectorDoc) -> None:
        self._client.merge_or_upload_documents([{
            "cluster_id": doc.cluster_id, "agent_id": doc.agent_id,
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

    def purge_stale(self, now: float, ttl: float) -> List[str]:
        flt = f"last_seen lt {now - ttl}"
        stale = list(self._client.search(search_text=None, filter=flt, select=["cluster_id"], top=1000))
        ids = [r["cluster_id"] for r in stale]
        if ids:
            self._client.delete_documents([{"cluster_id": cid} for cid in ids])
        return ids
