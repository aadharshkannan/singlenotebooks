from typing import Callable, Optional

import numpy as np

from .vector_store import VectorDoc


class CosmosVectorStore:
    """ESP/Cosmos vector repository around an injected container-like client.

    The concrete ESP SDK/authentication and distance convention are intentionally
    injected until those private contracts are available.
    """

    schema_version = "cluster-vector-v1"

    def __init__(
        self,
        container,
        vector_space_id: str,
        dimensions: int,
        distance_to_cosine: Callable[[float], float],
    ):
        if not vector_space_id:
            raise ValueError("vector_space_id is required")
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        self._container = container
        self.vector_space_id = vector_space_id
        self.dimensions = dimensions
        self._distance_to_cosine = distance_to_cosine
        self.n_queries = 0

    def _partition_key(self, agent_id: str, semantic_scope: str) -> str:
        return f"{self.vector_space_id}|{semantic_scope}|{agent_id}"

    def _document_id(self, cluster_id: str) -> str:
        return f"{self.vector_space_id}|{cluster_id}"

    def nearest(
        self,
        vec,
        agent_id: Optional[str] = None,
        semantic_scope: Optional[str] = None,
    ):
        if agent_id is None:
            raise ValueError("Cosmos vector queries require agent_id partition scope")
        vector = self._validate_vector(vec)
        self.n_queries += 1
        result = self._container.nearest(
            partition_key=self._partition_key(agent_id, semantic_scope or "legacy"),
            vector=vector.tolist(),
            vector_path="/vector",
            top=1,
        )
        if result is None:
            return None
        cosine = float(self._distance_to_cosine(float(result["distance"])))
        if not -1.0 <= cosine <= 1.0:
            raise ValueError("ESP/Cosmos distance mapping returned an invalid cosine")
        return result["cluster_id"], cosine

    def upsert(self, doc: VectorDoc) -> None:
        vector = self._validate_vector(doc.vector)
        self._container.upsert_item(
            {
                "id": self._document_id(doc.cluster_id),
                "schema_version": self.schema_version,
                "vector_space_id": self.vector_space_id,
                "semantic_scope": doc.semantic_scope,
                "partition_key": self._partition_key(doc.agent_id, doc.semantic_scope),
                "cluster_id": doc.cluster_id,
                "agent_id": doc.agent_id,
                "last_seen": doc.last_seen,
                "hits": doc.hits,
                "vector": vector.tolist(),
            }
        )

    def touch(self, cluster_id: str, now: float) -> None:
        self._container.patch_cluster(
            document_id=self._document_id(cluster_id),
            vector_space_id=self.vector_space_id,
            changes={"last_seen": now},
        )

    def purge_stale(
        self,
        now: float,
        ttl: float,
        semantic_scope: Optional[str] = None,
    ):
        request = {
            "vector_space_id": self.vector_space_id,
            "older_than": now - ttl,
        }
        if semantic_scope is not None:
            request["semantic_scope"] = semantic_scope
        return list(
            self._container.delete_stale(**request)
        )

    def delete(self, cluster_id: str) -> None:
        self._container.delete_cluster(
            document_id=self._document_id(cluster_id),
            vector_space_id=self.vector_space_id,
        )

    def _validate_vector(self, vec) -> np.ndarray:
        vector = np.asarray(vec, dtype=np.float32)
        if vector.shape != (self.dimensions,):
            raise ValueError(
                f"vector shape {vector.shape} does not match configured dimensions "
                f"{self.dimensions}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("vector must contain only finite values")
        norm = float(np.linalg.norm(vector))
        if not np.isclose(norm, 1.0, atol=1e-5):
            raise ValueError("Cosmos vectors must be unit-normalized for cosine search")
        return vector
