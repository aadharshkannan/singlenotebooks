from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .cluster_index import AzureClusterIndex
from .model import Trace
from .representation import NormalizedRepresentation, SessionEvidencePacketBuilder
from .session_embedding import SessionEmbeddingCache, SessionEmbeddingRecord
from .vector_store import InMemoryVectorStore, VectorStore


@dataclass(frozen=True)
class PreparedSession:
    """Prepared session packet for retrieval and judge evidence.

    `canonical_json` is the bounded evidence payload intended for judge submission.
    `vector` is retrieval state and should not be sent to an LLM by default.
    """

    trace: Trace
    canonical_json: str
    representation: NormalizedRepresentation
    embedding_record: SessionEmbeddingRecord
    vector: np.ndarray


class FullSessionEmbeddingPrototype:
    """Production-oriented facade for one-pass packet/vector preparation."""

    def __init__(
        self,
        cache: SessionEmbeddingCache,
        *,
        packet_builder: Optional[SessionEvidencePacketBuilder] = None,
        store: Optional[VectorStore] = None,
        similarity_threshold: float = 0.75,
        tenant_id: Optional[str] = "legacy",
        run_scope: Optional[str] = "legacy",
    ) -> None:
        self.cache = cache
        self.packet_builder = packet_builder or cache.packet_builder
        if self.packet_builder.options != cache.packet_builder.options:
            raise ValueError("prototype and cache packet builders must use identical options")
        self.store = store or InMemoryVectorStore()
        self.index = AzureClusterIndex(
            cache,
            self.store,
            tau=similarity_threshold,
            semantic_scope=cache.profile.cache_version,
            tenant_id=tenant_id,
            run_scope=run_scope,
        )

    def prepare(self, trace: Trace) -> PreparedSession:
        representation = self.packet_builder.build(trace)
        vector = self.cache.get_trace(trace)
        record = self.cache.peek_trace(trace)
        if record is None:
            raise RuntimeError("expected embedding record after get_trace")
        return PreparedSession(
            trace=trace,
            canonical_json=representation.canonical_json,
            representation=representation,
            embedding_record=record,
            vector=vector,
        )

    def observe(self, trace: Trace):
        """Cluster/session variety observation scoped by embedding profile."""
        return self.index.observe(trace)

    def build_judge_payload(
        self,
        prepared: PreparedSession,
        *,
        include_vector: bool = False,
    ) -> dict[str, Any]:
        """Serialize local judge input without performing a network request.

        Compact canonical evidence is always included. The embedding vector is
        opt-in because most LLM judges should receive auditable evidence rather
        than retrieval coordinates.
        """
        payload: dict[str, Any] = {
            "trace_id": prepared.trace.trace_id,
            "agent_id": prepared.trace.agent_id,
            "evidence": prepared.canonical_json,
            "evidence_sha256": prepared.embedding_record.key.content_sha256,
            "embedding_profile": self.cache.profile.cache_version,
        }
        if include_vector:
            payload["embedding_vector"] = prepared.vector.astype("float32").tolist()
        return payload