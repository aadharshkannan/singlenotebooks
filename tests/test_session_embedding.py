import json

import numpy as np
import pytest

from trace_sampling.model import SessionEvent, Trace
from trace_sampling.session_embedding import (
    EmbeddingProfile,
    SessionChunker,
    SessionEmbeddingCache,
    SessionEmbeddingError,
    SessionSerializer,
    pool_chunk_embeddings,
)


class CharacterTokenizer:
    name = "characters"
    version = "1"

    def count(self, text):
        return len(text)


class CountingEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.array(
            [[len(text), text.count("result") + 1] for text in texts],
            dtype=np.float32,
        )


def _trace(events):
    return Trace(1, "agent", 0.0, ("search",), 1, 1.0, "ok", events=tuple(events))


def _profile(max_input_tokens=400, model_version="2026-07-01"):
    return EmbeddingProfile(
        model_id="capi-embedding",
        model_version=model_version,
        tokenizer_id="characters",
        tokenizer_version="1",
        max_input_tokens=max_input_tokens,
    )


def test_serializer_is_deterministic_but_preserves_event_order():
    serializer = SessionSerializer()
    first = (
        SessionEvent("user", "hello\r\nworld"),
        SessionEvent("assistant", tool_name="search", arguments={"b": 2, "a": 1}),
    )
    same = (
        SessionEvent("user", "hello\nworld"),
        SessionEvent("assistant", tool_name="search", arguments={"a": 1, "b": 2}),
    )

    assert serializer.serialize(serializer.snapshot(first)) == serializer.serialize(
        serializer.snapshot(same)
    )
    assert serializer.serialize(serializer.snapshot(tuple(reversed(first)))) != serializer.serialize(
        serializer.snapshot(first)
    )


def test_chunker_keeps_events_whole_when_they_fit():
    serializer = SessionSerializer()
    snapshot = serializer.snapshot(
        (SessionEvent("user", "first"), SessionEvent("assistant", "second"))
    )
    one_event_tokens = len(serializer.serialize_events([{**snapshot[0], "event_index": 0}]))
    two_event_tokens = len(
        serializer.serialize_events(
            [{**snapshot[0], "event_index": 0}, {**snapshot[1], "event_index": 1}]
        )
    )
    chunker = SessionChunker(
        serializer, CharacterTokenizer(), max_input_tokens=two_event_tokens - 1
    )

    chunks = chunker.chunk(snapshot)

    assert len(chunks) == 2
    assert all(chunk.token_count <= two_event_tokens - 1 for chunk in chunks)
    assert one_event_tokens <= chunks[0].token_count


def test_chunker_splits_huge_tool_output_without_loss():
    serializer = SessionSerializer()
    output = "0123456789" * 80
    snapshot = serializer.snapshot(
        (
            SessionEvent(
                "tool",
                tool_name="search",
                arguments={"query": "incident"},
                output=output,
            ),
        )
    )
    empty_frame = {
        **snapshot[0],
        "event_index": 0,
        "output": "x",
        "output_part_index": 0,
    }
    limit = len(serializer.serialize_events([empty_frame])) + 90
    chunks = SessionChunker(serializer, CharacterTokenizer(), limit).chunk(snapshot)

    reconstructed = "".join(json.loads(chunk.text)["events"][0]["output"] for chunk in chunks)
    assert len(chunks) > 1
    assert reconstructed == output
    assert all(chunk.token_count <= limit for chunk in chunks)


def test_pooling_normalizes_chunks_then_token_weights_the_result():
    vectors = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)

    pooled = pool_chunk_embeddings(vectors, [1, 3])

    expected = np.array([0.25, 0.75])
    expected /= np.linalg.norm(expected)
    assert pooled.dtype == np.float32
    assert np.allclose(pooled, expected)
    assert np.isclose(np.linalg.norm(pooled), 1.0)


def test_cache_keys_by_content_and_profile_and_records_usage():
    embedder = CountingEmbedder()
    trace = _trace([SessionEvent("tool", tool_name="search", output="result")])
    cache = SessionEmbeddingCache(embedder, CharacterTokenizer(), _profile())

    first = cache.get_trace(trace)
    second = cache.get_trace(trace)

    assert np.array_equal(first, second)
    assert len(embedder.calls) == 1
    assert cache.n_calls == 1
    assert cache.n_hits == 1
    assert cache.last_record.chunk_count == 1
    assert cache.last_record.token_count > 0
    assert cache.contains_trace(trace)
    assert cache.peek_trace(trace).key.content_sha256 == cache.last_record.key.content_sha256

    changed_content = _trace([SessionEvent("tool", tool_name="search", output="other")])
    cache.get_trace(changed_content)
    assert len(embedder.calls) == 2

    changed_version = SessionEmbeddingCache(
        embedder, CharacterTokenizer(), _profile(model_version="2026-08-01")
    )
    changed_version.get_trace(trace)
    assert len(embedder.calls) == 3


def test_empty_event_sessions_fall_back_to_ordered_signature_content():
    embedder = CountingEmbedder()
    cache = SessionEmbeddingCache(embedder, CharacterTokenizer(), _profile())
    first = Trace(1, "agent", 0.0, ("search",), 1, 1.0, "ok")
    second = Trace(2, "agent", 1.0, ("read",), 1, 1.0, "ok")

    cache.get_trace(first)
    cache.get_trace(second)

    assert len(embedder.calls) == 2
    assert "search" in embedder.calls[0][0]
    assert "read" in embedder.calls[1][0]


def test_oversized_non_output_is_recorded_as_an_explicit_failure():
    embedder = CountingEmbedder()
    trace = _trace([SessionEvent("user", text="x" * 300)])
    cache = SessionEmbeddingCache(
        embedder, CharacterTokenizer(), _profile(max_input_tokens=120)
    )

    with pytest.raises(SessionEmbeddingError, match="only be split"):
        cache.get_trace(trace)

    assert not embedder.calls
    assert cache.n_failures == 1
    assert cache.failures[0].stage == "chunking"
    assert cache.failures[0].reason == "oversized_non_output"
    assert cache.failures[0].content_sha256 is not None


def test_provider_failure_records_planned_chunks_and_tokens():
    class FailingEmbedder:
        def embed(self, texts):
            raise RuntimeError("CAPI unavailable")

    trace = _trace([SessionEvent("user", "hello")])
    cache = SessionEmbeddingCache(
        FailingEmbedder(), CharacterTokenizer(), _profile(max_input_tokens=400)
    )

    with pytest.raises(SessionEmbeddingError, match="RuntimeError"):
        cache.get_trace(trace)

    assert cache.n_failures == 1
    assert cache.failures[0].stage == "provider"
    assert cache.failures[0].chunk_count == 1
    assert cache.failures[0].token_count > 0
    assert "CAPI unavailable" not in cache.failures[0].message
    assert cache.n_calls == 1
    assert cache.n_failed_chunks == 1
    assert cache.n_failed_tokens == cache.failures[0].token_count