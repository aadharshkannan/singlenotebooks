from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple
import unicodedata

import numpy as np

from .embedding import Embedder
from .model import SessionEvent, Trace
from .representation import (
    CANONICAL_POLICY,
    CANONICAL_VERSION,
    CanonicalizationOptions,
    DEFAULT_MAX_UTF8_BYTES,
    NormalizedRepresentation,
    RepresentationAudit,
    RepresentationError,
    SessionEvidencePacketBuilder,
)


class Tokenizer(Protocol):
    name: str
    version: str

    def count(self, text: str) -> int: ...


class TiktokenTokenizer:
    """Tokenizer backed by the encoding for the deployed embedding model."""

    def __init__(self, model_name: str, encoding_name: Optional[str] = None):
        import tiktoken

        self.name = encoding_name or model_name
        self.version = getattr(tiktoken, "__version__", "unknown")
        self._encoding = (
            tiktoken.get_encoding(encoding_name)
            if encoding_name
            else tiktoken.encoding_for_model(model_name)
        )

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))


@dataclass(frozen=True)
class EmbeddingProfile:
    model_id: str
    model_version: str
    tokenizer_id: str
    tokenizer_version: str
    max_input_tokens: int
    pooling_version: str = "token-weighted-unit-v1"
    representation_policy: str = CANONICAL_POLICY
    representation_version: str = CANONICAL_VERSION
    max_representation_utf8_bytes: Optional[int] = None

    def __post_init__(self):
        if self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be >= 1")
        if not self.model_id or not self.model_version:
            raise ValueError("model_id and model_version are required")
        CanonicalizationOptions(
            max_utf8_bytes=self.representation_max_utf8_bytes,
            policy=self.representation_policy,
            version=self.representation_version,
        )

    @property
    def representation_max_utf8_bytes(self) -> int:
        return (
            self.max_representation_utf8_bytes
            if self.max_representation_utf8_bytes is not None
            else DEFAULT_MAX_UTF8_BYTES
        )

    @property
    def cache_version(self) -> str:
        payload = json.dumps(
            {
                "max_input_tokens": self.max_input_tokens,
                "model_id": self.model_id,
                "model_version": self.model_version,
                "pooling_version": self.pooling_version,
                "representation_max_utf8_bytes": self.representation_max_utf8_bytes,
                "representation_policy": self.representation_policy,
                "representation_version": self.representation_version,
                "tokenizer_id": self.tokenizer_id,
                "tokenizer_version": self.tokenizer_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SessionCacheKey:
    content_sha256: str
    serializer_version: str
    model_version: str


@dataclass(frozen=True)
class SessionChunk:
    text: str
    token_count: int


@dataclass(frozen=True)
class SessionEmbeddingRecord:
    key: SessionCacheKey
    vector: np.ndarray
    chunk_count: int
    token_count: int
    representation_audit: RepresentationAudit


@dataclass(frozen=True)
class EmbeddingFailure:
    stage: str
    reason: str
    content_sha256: Optional[str]
    chunk_count: int
    token_count: int
    message: str


class SessionEmbeddingError(RuntimeError):
    def __init__(self, stage: str, reason: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.reason = reason


class SessionSerializer:
    """Canonical JSON serializer for ordered session events."""

    version = f"session-events-json-v1-unicode-{unicodedata.unidata_version}"

    def snapshot(self, events: Sequence[SessionEvent]) -> Tuple[Dict[str, Any], ...]:
        return tuple(
            {
                "arguments": self._canonicalize(event.arguments),
                "output": self._canonicalize(event.output),
                "role": self._canonicalize(event.role),
                "text": self._canonicalize(event.text),
                "tool_name": self._canonicalize(event.tool_name),
            }
            for event in events
        )

    def serialize(self, snapshot: Sequence[Mapping[str, Any]]) -> str:
        return self.serialize_events(snapshot)

    def serialize_events(self, events: Sequence[Mapping[str, Any]]) -> str:
        return json.dumps(
            {
                "events": list(events),
                "schema": "session-events",
                "version": self.version,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _canonicalize(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise SessionEmbeddingError(
                    "serialization", "non_finite_number", "NaN and infinity are not supported"
                )
            return value
        if isinstance(value, str):
            return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
        if isinstance(value, Mapping):
            canonical = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise SessionEmbeddingError(
                        "serialization", "non_string_key", "argument object keys must be strings"
                    )
                canonical[self._canonicalize(key)] = self._canonicalize(item)
            return canonical
        if isinstance(value, (list, tuple)):
            return [self._canonicalize(item) for item in value]
        raise SessionEmbeddingError(
            "serialization",
            "unsupported_type",
            f"unsupported session value type: {type(value).__name__}",
        )


class SessionChunker:
    def __init__(self, serializer: SessionSerializer, tokenizer: Tokenizer, max_input_tokens: int):
        self.serializer = serializer
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.split_output_events = 0

    def chunk(self, snapshot: Sequence[Mapping[str, Any]]) -> List[SessionChunk]:
        chunks: List[SessionChunk] = []
        pending: List[Mapping[str, Any]] = []

        for event_index, event in enumerate(snapshot):
            indexed = dict(event)
            indexed["event_index"] = event_index
            event_tokens = self._count_events([indexed])
            if event_tokens > self.max_input_tokens:
                if pending:
                    chunks.append(self._make_chunk(pending))
                    pending = []
                chunks.extend(self._split_output(indexed))
                self.split_output_events += 1
                continue

            candidate = [*pending, indexed]
            if pending and self._count_events(candidate) > self.max_input_tokens:
                chunks.append(self._make_chunk(pending))
                pending = [indexed]
            else:
                pending = candidate

        if pending:
            chunks.append(self._make_chunk(pending))
        if not chunks:
            chunks.append(self._make_chunk([]))
        return chunks

    def _split_output(self, event: Mapping[str, Any]) -> List[SessionChunk]:
        output = event.get("output")
        if not isinstance(output, str) or not output:
            raise SessionEmbeddingError(
                "chunking",
                "oversized_non_output",
                "an oversized event can only be split when it has non-empty string output",
            )

        chunks: List[SessionChunk] = []
        offset = 0
        part_index = 0
        while offset < len(output):
            low = offset + 1
            high = len(output)
            best = offset
            while low <= high:
                middle = (low + high) // 2
                fragment = self._output_fragment(event, output[offset:middle], part_index)
                if self._count_events([fragment]) <= self.max_input_tokens:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == offset:
                raise SessionEmbeddingError(
                    "chunking",
                    "oversized_non_output",
                    "event metadata leaves no room for any tool output",
                )
            fragment = self._output_fragment(event, output[offset:best], part_index)
            chunks.append(self._make_chunk([fragment]))
            offset = best
            part_index += 1
        return chunks

    @staticmethod
    def _output_fragment(
        event: Mapping[str, Any], output: str, part_index: int
    ) -> Mapping[str, Any]:
        fragment = dict(event)
        fragment["output"] = output
        fragment["output_part_index"] = part_index
        return fragment

    def _count_events(self, events: Sequence[Mapping[str, Any]]) -> int:
        return self.tokenizer.count(self.serializer.serialize_events(events))

    def _make_chunk(self, events: Sequence[Mapping[str, Any]]) -> SessionChunk:
        text = self.serializer.serialize_events(events)
        token_count = self.tokenizer.count(text)
        if token_count > self.max_input_tokens:
            raise SessionEmbeddingError(
                "chunking", "token_limit", "constructed chunk exceeds max_input_tokens"
            )
        return SessionChunk(text=text, token_count=token_count)


def pool_chunk_embeddings(vectors: np.ndarray, token_counts: Sequence[int]) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != len(token_counts) or array.shape[0] == 0:
        raise SessionEmbeddingError(
            "pooling", "invalid_shape", "provider returned an unexpected embedding shape"
        )
    if not np.all(np.isfinite(array)):
        raise SessionEmbeddingError(
            "pooling", "non_finite_vector", "provider returned a non-finite embedding"
        )
    weights = np.asarray(token_counts, dtype=np.float64)
    if np.any(weights <= 0):
        raise SessionEmbeddingError(
            "pooling", "invalid_token_count", "chunk token counts must be positive"
        )
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0):
        raise SessionEmbeddingError(
            "pooling", "zero_norm_vector", "provider returned a zero-norm embedding"
        )
    pooled = np.average(array / norms[:, None], axis=0, weights=weights)
    pooled_norm = np.linalg.norm(pooled)
    if not math.isfinite(float(pooled_norm)) or pooled_norm == 0:
        raise SessionEmbeddingError(
            "pooling", "zero_norm_pool", "weighted chunk embeddings cancel to zero"
        )
    return (pooled / pooled_norm).astype(np.float32)


class SessionEmbeddingCache:
    """Content-addressed LRU cache that resolves each session to one unit vector."""

    def __init__(
        self,
        embedder: Embedder,
        tokenizer: Tokenizer,
        profile: EmbeddingProfile,
        max_size: int = 4096,
        serializer: Optional[SessionSerializer] = None,
        packet_builder: Optional[SessionEvidencePacketBuilder] = None,
    ):
        self._embedder = embedder
        self._tokenizer = tokenizer
        self.profile = profile
        self.serializer = serializer or SessionSerializer()
        self.packet_builder = packet_builder or SessionEvidencePacketBuilder(
            CanonicalizationOptions(
                max_utf8_bytes=profile.representation_max_utf8_bytes,
                policy=profile.representation_policy,
                version=profile.representation_version,
            ),
            max_size=max_size,
        )
        self._max = max_size
        self._cache: "OrderedDict[SessionCacheKey, SessionEmbeddingRecord]" = OrderedDict()
        self.n_calls = 0
        self.n_hits = 0
        self.n_failures = 0
        self.n_chunks = 0
        self.n_tokens = 0
        self.n_failed_chunks = 0
        self.n_failed_tokens = 0
        self.embed_latencies_ms: List[float] = []
        self.failure_counts: Counter = Counter()
        self.failures: List[EmbeddingFailure] = []
        self.last_record: Optional[SessionEmbeddingRecord] = None

    @property
    def split_output_events(self) -> int:
        return 0

    def _representation_and_key(
        self, trace: Trace
    ) -> Tuple[NormalizedRepresentation, SessionCacheKey]:
        representation = self.packet_builder.build(trace)
        digest = hashlib.sha256(representation.canonical_json.encode("utf-8")).hexdigest()
        return representation, SessionCacheKey(
            content_sha256=digest,
            serializer_version=f"{representation.policy}:{representation.version}",
            model_version=self.profile.cache_version,
        )

    def contains_trace(self, trace: Trace) -> bool:
        try:
            _, key = self._representation_and_key(trace)
        except (SessionEmbeddingError, ValueError):
            return False
        return key in self._cache

    def peek_trace(self, trace: Trace) -> Optional[SessionEmbeddingRecord]:
        try:
            _, key = self._representation_and_key(trace)
        except (SessionEmbeddingError, ValueError):
            return None
        return self._cache.get(key)

    def get_trace(self, trace: Trace) -> np.ndarray:
        content_sha256 = None
        chunks: List[SessionChunk] = []
        try:
            representation, key = self._representation_and_key(trace)
            content_sha256 = key.content_sha256
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self.n_hits += 1
                self.last_record = cached
                return cached.vector

            token_count = self._tokenizer.count(representation.canonical_json)
            if token_count > self.profile.max_input_tokens:
                raise SessionEmbeddingError(
                    "representation",
                    "token_limit",
                    "bounded canonical representation exceeds max_input_tokens",
                )
            chunks = [SessionChunk(representation.canonical_json, token_count)]
            texts = [representation.canonical_json]
            token_counts = [token_count]
            started = time.perf_counter()
            self.n_calls += 1
            try:
                vectors = self._embedder.embed(texts)
            except Exception as exc:
                raise SessionEmbeddingError(
                    "provider",
                    "request_failed",
                    f"embedding provider request failed: {type(exc).__name__}",
                ) from exc
            self.embed_latencies_ms.append((time.perf_counter() - started) * 1000.0)
            self.n_chunks += len(chunks)
            self.n_tokens += sum(token_counts)
            vector = pool_chunk_embeddings(vectors, token_counts)
            record = SessionEmbeddingRecord(
                key=key,
                vector=vector,
                chunk_count=len(chunks),
                token_count=sum(token_counts),
                representation_audit=representation.audit,
            )
            self._cache[key] = record
            self.last_record = record
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)
            return vector
        except SessionEmbeddingError as exc:
            self.n_failures += 1
            if exc.stage == "provider":
                self.n_failed_chunks += len(chunks)
                self.n_failed_tokens += sum(chunk.token_count for chunk in chunks)
            self.failure_counts[(exc.stage, exc.reason)] += 1
            self.failures.append(
                EmbeddingFailure(
                    stage=exc.stage,
                    reason=exc.reason,
                    content_sha256=content_sha256,
                    chunk_count=len(chunks),
                    token_count=sum(chunk.token_count for chunk in chunks),
                    message=str(exc),
                )
            )
            raise
        except RepresentationError as exc:
            self.n_failures += 1
            reason = (
                "mandatory_evidence_floor"
                if "mandatory task-completion evidence" in str(exc)
                else "structural_floor"
                if "non-content canonical structure" in str(exc)
                else "invalid_representation"
            )
            self.failure_counts[("representation", reason)] += 1
            self.failures.append(
                EmbeddingFailure(
                    stage="representation",
                    reason=reason,
                    content_sha256=content_sha256,
                    chunk_count=0,
                    token_count=0,
                    message=str(exc),
                )
            )
            raise
