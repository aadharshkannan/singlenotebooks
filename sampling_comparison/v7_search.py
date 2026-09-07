from __future__ import annotations

import base64
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class V7SearchEvidence:
    metric: str
    index_name: str
    query_count: int
    write_count: int
    latency_seconds: float
    validated_dimensions: int
    validated_neighbor_hits: int


def encode_search_cluster_id(unit_id: str) -> str:
    """Encode arbitrary unit IDs into Azure Search key-safe deterministic IDs."""
    raw = str(unit_id).encode("utf-8")
    token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"b64_{token}"


def decode_search_cluster_id(cluster_id: str) -> str:
    """Best-effort reverse of ``encode_search_cluster_id`` for test validation."""
    value = str(cluster_id)
    if not value.startswith("b64_"):
        return value
    token = value[4:]
    pad = "=" * ((4 - (len(token) % 4)) % 4)
    decoded = base64.urlsafe_b64decode((token + pad).encode("ascii"))
    return decoded.decode("utf-8")


class FakeV7SearchAdapter:
    """Deterministic in-memory stand-in for v7 Search integration tests."""

    def __init__(self) -> None:
        self._indexes: dict[str, dict[str, dict[str, Any]]] = {}
        self.telemetry: dict[str, dict[str, Any]] = {}

    def ensure_index(self, *, index_name: str, dimensions: int, metric: str) -> None:
        _ = (dimensions, metric)
        self._indexes.setdefault(index_name, {})
        self.telemetry.setdefault(index_name, {"writes": 0, "queries": 0, "latency_seconds": 0.0})

    def sync_documents(self, *, index_name: str, documents: Sequence[Mapping[str, Any]], dimensions: int, metric: str) -> int:
        self.ensure_index(index_name=index_name, dimensions=dimensions, metric=metric)
        store = self._indexes[index_name]
        writes = 0
        for row in documents:
            original_unit_id = str(row.get("original_unit_id") or row["cluster_id"])
            cluster_id = encode_search_cluster_id(original_unit_id)
            vector = np.asarray(row["vector"], dtype=np.float32)
            if vector.ndim != 1 or int(vector.size) != int(dimensions):
                raise ValueError("v7 fake search document vector dimensions mismatch")
            payload = dict(row)
            payload["cluster_id"] = cluster_id
            payload["original_unit_id"] = original_unit_id
            store[cluster_id] = payload
            writes += 1
        self.telemetry[index_name]["writes"] += writes
        return writes

    def query_neighbors(
        self,
        *,
        index_name: str,
        vector: np.ndarray,
        k: int,
        metric: str,
        run_scope: str,
        semantic_scope: str,
    ) -> list[dict[str, Any]]:
        t0 = perf_counter()
        docs = self._indexes.get(index_name, {})
        q = np.asarray(vector, dtype=np.float32)
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in docs.values():
            if str(row.get("run_scope") or "") != str(run_scope):
                continue
            if str(row.get("semantic_scope") or "") != str(semantic_scope):
                continue
            v = np.asarray(row["vector"], dtype=np.float32)
            if metric == "cosine":
                qn = float(np.linalg.norm(q))
                vn = float(np.linalg.norm(v))
                score = 0.0 if qn == 0.0 or vn == 0.0 else float(np.dot(q, v) / (qn * vn))
            elif metric == "euclidean":
                score = -float(np.linalg.norm(q - v))
            else:
                raise ValueError(f"unsupported metric: {metric}")
            scored.append((score, dict(row)))
        scored.sort(key=lambda row: row[0], reverse=True)
        out = [row for _, row in scored[: max(0, int(k))]]
        self.telemetry.setdefault(index_name, {"writes": 0, "queries": 0, "latency_seconds": 0.0})
        self.telemetry[index_name]["queries"] += 1
        self.telemetry[index_name]["latency_seconds"] += float(perf_counter() - t0)
        return out


class AzureV7SearchAdapter:
    """Azure AI Search adapter for v7 PCA-8 evidence indexes."""

    def __init__(self, *, azure_config: Any, cosine_index_name: str, euclidean_index_name: str) -> None:
        self._cfg = azure_config
        self._index_names = {
            "cosine": str(cosine_index_name),
            "euclidean": str(euclidean_index_name),
        }

    @property
    def index_names(self) -> dict[str, str]:
        return dict(self._index_names)

    @staticmethod
    def _credential(cfg: Any) -> Any:
        if getattr(cfg, "search_api_key", None):
            from azure.core.credentials import AzureKeyCredential

            return AzureKeyCredential(cfg.search_api_key)
        from azure.identity import DefaultAzureCredential

        return DefaultAzureCredential()

    def ensure_index(self, *, index_name: str, dimensions: int, metric: str) -> None:
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            HnswParameters,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchAlgorithmMetric,
            VectorSearchProfile,
        )

        if int(dimensions) <= 0:
            raise ValueError("dimensions must be positive")
        metric_name = str(metric).strip().lower()
        if metric_name not in {"cosine", "euclidean"}:
            raise ValueError(f"unsupported v7 metric: {metric}")

        algorithm_name = f"hnsw-{metric_name}-algo"
        profile_name = f"hnsw-{metric_name}"
        metric_value = (
            VectorSearchAlgorithmMetric.COSINE
            if metric_name == "cosine"
            else VectorSearchAlgorithmMetric.EUCLIDEAN
        )

        fields = [
            SimpleField(name="cluster_id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="original_unit_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="agent_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="run_scope", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="semantic_scope", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="last_seen", type=SearchFieldDataType.Double, filterable=True, sortable=True),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=int(dimensions),
                vector_search_profile_name=profile_name,
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name=algorithm_name,
                    parameters=HnswParameters(metric=metric_value),
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name=profile_name,
                    algorithm_configuration_name=algorithm_name,
                )
            ],
        )
        client = SearchIndexClient(self._cfg.search_endpoint, self._credential(self._cfg))
        client.create_or_update_index(SearchIndex(name=index_name, fields=fields, vector_search=vector_search))

    def sync_documents(self, *, index_name: str, documents: Sequence[Mapping[str, Any]], dimensions: int, metric: str) -> int:
        self.ensure_index(index_name=index_name, dimensions=dimensions, metric=metric)
        from azure.search.documents import SearchClient

        client = SearchClient(self._cfg.search_endpoint, index_name, self._credential(self._cfg))
        payload: list[dict[str, Any]] = []
        for row in documents:
            vector = np.asarray(row["vector"], dtype=np.float32)
            if vector.ndim != 1 or int(vector.size) != int(dimensions):
                raise ValueError("v7 search sync vector dimensions mismatch")
            original_unit_id = str(row.get("original_unit_id") or row["cluster_id"])
            payload.append(
                {
                    "cluster_id": encode_search_cluster_id(original_unit_id),
                    "original_unit_id": original_unit_id,
                    "agent_id": str(row["agent_id"]),
                    "run_scope": str(row["run_scope"]),
                    "semantic_scope": str(row["semantic_scope"]),
                    "last_seen": float(row.get("last_seen") or 0.0),
                    "vector": vector.tolist(),
                }
            )
        writes = 0
        batch_size = 500
        for start in range(0, len(payload), batch_size):
            chunk = payload[start : start + batch_size]
            if not chunk:
                continue
            result = client.merge_or_upload_documents(chunk)
            rows = list(result or [])
            if len(rows) != len(chunk):
                raise ValueError(
                    f"v7 search sync returned {len(rows)} indexing results for {len(chunk)} documents"
                )
            for idx, row_result in enumerate(rows):
                succeeded = bool(getattr(row_result, "succeeded", False))
                if not succeeded:
                    key = getattr(row_result, "key", None) or chunk[idx].get("cluster_id")
                    status_code = getattr(row_result, "status_code", None)
                    error_message = getattr(row_result, "error_message", None)
                    raise ValueError(
                        "v7 search sync failed for key="
                        f"{key} status_code={status_code} error={error_message}"
                    )
            writes += len(chunk)
        return writes

    def query_neighbors(
        self,
        *,
        index_name: str,
        vector: np.ndarray,
        k: int,
        metric: str,
        run_scope: str,
        semantic_scope: str,
    ) -> list[dict[str, Any]]:
        _ = metric
        from azure.search.documents import SearchClient
        from azure.search.documents.models import VectorizedQuery

        client = SearchClient(self._cfg.search_endpoint, index_name, self._credential(self._cfg))
        query = VectorizedQuery(
            vector=np.asarray(vector, dtype=np.float32).tolist(),
            k_nearest_neighbors=max(1, int(k)),
            fields="vector",
        )
        safe_run_scope = str(run_scope).replace("'", "''")
        safe_semantic_scope = str(semantic_scope).replace("'", "''")
        scope_filter = (
            f"run_scope eq '{safe_run_scope}' and "
            f"semantic_scope eq '{safe_semantic_scope}'"
        )
        results = client.search(
            search_text=None,
            vector_queries=[query],
            filter=scope_filter,
            top=max(1, int(k)),
            vector_filter_mode="preFilter",
        )
        out: list[dict[str, Any]] = []
        for row in results:
            out.append(dict(row))
        return out


__all__ = [
    "AzureV7SearchAdapter",
    "FakeV7SearchAdapter",
    "V7SearchEvidence",
    "decode_search_cluster_id",
    "encode_search_cluster_id",
]
