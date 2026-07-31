from __future__ import annotations

import numpy as np
import pytest

from trace_sampling.full_session_prototype import FullSessionEmbeddingPrototype
from trace_sampling.model import SessionEvent, Trace
from trace_sampling.session_embedding import EmbeddingProfile, SessionEmbeddingCache


class _Tokenizer:
    name = "chars"
    version = "1"

    def count(self, text: str) -> int:
        return len(text)


class _Embedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        rows = []
        for text in texts:
            rows.append([float(len(text) % 17 + 1), float(text.count("task") + 1)])
        return np.asarray(rows, dtype=np.float32)


def _profile(max_bytes: int = 32768) -> EmbeddingProfile:
    return EmbeddingProfile(
        model_id="deterministic",
        model_version="v1",
        tokenizer_id="chars",
        tokenizer_version="1",
        max_input_tokens=4096,
        max_representation_utf8_bytes=max_bytes,
    )


def _trace(trace_id: int, text: str, agent: str = "tenant|agent") -> Trace:
    return Trace(
        trace_id=trace_id,
        agent_id=agent,
        timestamp=float(trace_id),
        signature=("search",),
        span_count=1,
        duration_ms=10.0,
        status="ok",
        concept_id=1,
        events=(
            SessionEvent(role="user", text=text),
            SessionEvent(role="assistant", text="task complete"),
        ),
    )


def test_prepare_uses_single_canonical_packet_for_vector_and_judge_evidence() -> None:
    embedder = _Embedder()
    cache = SessionEmbeddingCache(embedder, _Tokenizer(), _profile())
    proto = FullSessionEmbeddingPrototype(cache)
    trace = _trace(1, "reset account password")

    prepared = proto.prepare(trace)

    assert prepared.canonical_json == prepared.representation.canonical_json
    assert prepared.embedding_record.representation_audit == prepared.representation.audit
    assert np.array_equal(prepared.vector, prepared.embedding_record.vector)
    assert embedder.calls == 1


def test_prepare_fails_early_on_mandatory_evidence_floor() -> None:
    embedder = _Embedder()
    cache = SessionEmbeddingCache(embedder, _Tokenizer(), _profile(max_bytes=120))
    proto = FullSessionEmbeddingPrototype(cache)
    trace = _trace(2, "G" * 2000)

    with pytest.raises(ValueError, match="mandatory task-completion evidence|non-content canonical structure"):
        proto.prepare(trace)

    assert embedder.calls == 0


def test_in_memory_vector_clustering_is_scoped_by_agent_and_profile() -> None:
    embedder = _Embedder()
    cache = SessionEmbeddingCache(embedder, _Tokenizer(), _profile())
    proto = FullSessionEmbeddingPrototype(cache, similarity_threshold=0.7)

    a1 = proto.observe(_trace(1, "shared words", agent="tenant-a|agent-1"))
    a2 = proto.observe(_trace(2, "shared words", agent="tenant-a|agent-1"))
    b1 = proto.observe(_trace(3, "shared words", agent="tenant-b|agent-1"))

    assert a1.key.kind == "cluster"
    assert a2.key == a1.key
    assert b1.key != a1.key


def test_judge_payload_defaults_to_compact_evidence_and_can_include_vector() -> None:
    cache = SessionEmbeddingCache(_Embedder(), _Tokenizer(), _profile())
    proto = FullSessionEmbeddingPrototype(cache)
    prepared = proto.prepare(_trace(4, "verify policy-compliant answer"))

    compact = proto.build_judge_payload(prepared)
    assert compact["evidence"] == prepared.canonical_json
    assert "embedding_vector" not in compact

    with_vector = proto.build_judge_payload(prepared, include_vector=True)
    assert with_vector["evidence"] == prepared.canonical_json
    assert len(with_vector["embedding_vector"]) == prepared.vector.shape[0]