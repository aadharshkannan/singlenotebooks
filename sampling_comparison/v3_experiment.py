from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from math import floor
from statistics import median
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from minhash_sampling import BandedMinHashLSHIndex, MinHashConfig
from minhash_sampling.signature import MinHashRecord
from trace_sampling import AdaptiveSampler, SamplerConfig
from trace_sampling.cluster_index import AzureClusterIndex
from trace_sampling.model import Trace
from trace_sampling.session_embedding import TiktokenTokenizer
from trace_sampling.token_representation import CanonicalizationOptions, TokenSessionEvidencePacketBuilder
from trace_sampling.vector_store import InMemoryVectorStore, VectorStore

from .v2_experiment import CombinedDataset, assign_population_quadrants

V3_VERSION = "sampling-v3"
V3_OUTCOME_VERSION = "sampling-v3-outcome-v1"
V3_MAX_SESSION_PACKET_TOKENS = 8191
V3_EMBEDDING_DIMENSIONS = 1536
V3_EMBEDDING_MODEL = "text-embedding-3-small"
V3_EMBEDDING_ENCODING = "cl100k_base"
V3_DEFAULT_EMBEDDING_BATCH_SIZE = 32
V3_DEFAULT_EMBEDDING_TAU = 0.55

V3_OUTCOME_METHODS: tuple[str, ...] = (
    "random_sampling_token_priority",
    "adaptive_minhash_32x4_token",
    "adaptive_embedding_fullsession_token",
)

V3_QUADRANT_METHODS: tuple[str, ...] = V3_OUTCOME_METHODS
V3_THROUGHPUT_METHODS: tuple[str, ...] = V3_OUTCOME_METHODS


@dataclass(frozen=True)
class V3TokenPacketRecord:
    unit_id: str
    trace_id: int
    content_sha256: str
    canonical_json: str
    original_tokens: int
    emitted_tokens: int
    truncated: bool


@dataclass(frozen=True)
class V3EmbeddingRecord:
    content_sha256: str
    vector: np.ndarray
    dimensions: int
    input_tokens: int


@dataclass(frozen=True)
class V3BuildLedger:
    packet_builds: int
    packet_cache_hits: int
    embedding_calls: int
    embedding_inputs: int
    embedding_input_tokens: int
    embedding_latency_seconds: float
    embedding_content_hashes: tuple[str, ...]
    embedding_model_id: str
    embedding_deployment_id: str
    embedding_embedder_class: str


@dataclass(frozen=True)
class V3Runtime:
    version: str
    token_profile_id: str
    minhash_profile_id: str
    embedding_profile_id: str
    embedding_semantic_scope: str
    packet_records_by_unit_id: dict[str, V3TokenPacketRecord]
    minhash_records_by_unit_id: dict[str, MinHashRecord]
    embedding_records_by_content_sha256: dict[str, V3EmbeddingRecord]
    embedding_vector_by_trace_id: dict[int, np.ndarray]
    packet_hashes_by_trace_id: dict[int, str]
    token_cost_by_unit_id: dict[str, int]
    unit_id_by_trace_id: dict[int, str]
    ledger: V3BuildLedger


class _UnitNorm1536Embedder:
    def __init__(self, embedder: Any, dimensions: int) -> None:
        self._embedder = embedder
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        raw = self._embedder.embed(texts)
        matrix = np.asarray(raw, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("embedder must return a 2D matrix")
        if matrix.shape[0] != len(texts):
            raise ValueError("embedder output row count does not match inputs")
        if matrix.shape[1] != self._dimensions:
            raise ValueError(f"embedder output must have {self._dimensions} dimensions")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("embedder produced non-finite values")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0):
            raise ValueError("embedder produced a zero-norm vector")
        matrix = matrix / norms[:, None]
        return matrix.astype(np.float32)


class V3Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _paired_permutation(unit_ids: Sequence[str], token: str) -> tuple[str, ...]:
    scored = [(_sha256_text(f"{token}|{uid}"), uid) for uid in unit_ids]
    scored.sort(key=lambda row: row[0])
    return tuple(uid for _, uid in scored)


@lru_cache(maxsize=16384)
def _token_vec(token: str, seed: int, dim: int) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}|{token}".encode("utf-8")).digest()
    values = [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) for i in range(dim)]
    arr = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    arr.setflags(write=False)
    return arr


class Deterministic1536Embedder:
    """Deterministic offline embedder for tests; mirrors batch API."""

    def __init__(self, seed: int = 13, dimensions: int = V3_EMBEDDING_DIMENSIONS) -> None:
        self.seed = seed
        self.dimensions = dimensions
        self.calls = 0
        self.inputs = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.inputs += len(texts)
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [tok for tok in text.split() if tok]
            if not tokens:
                tokens = ["empty"]
            vec = np.zeros(self.dimensions, dtype=np.float32)
            for token in tokens:
                vec += _token_vec(token, self.seed, self.dimensions)
            norm = float(np.linalg.norm(vec))
            out[row] = vec if norm == 0 else (vec / norm)
        return out


class _TokenProfile:
    def __init__(
        self,
        tokenizer: Any,
        max_tokens: int,
        *,
        embedding_model_id: str,
        embedding_deployment_id: str,
    ) -> None:
        self.model_name = getattr(tokenizer, "model_name", V3_EMBEDDING_MODEL)
        self.encoding_name = getattr(
            tokenizer,
            "encoding_name",
            getattr(tokenizer, "name", V3_EMBEDDING_ENCODING),
        )
        self.encoding_id = getattr(tokenizer, "encoding_id", self.encoding_name)
        self.version = getattr(tokenizer, "version", "unknown")
        self.max_tokens = max_tokens
        self.embedding_model_id = embedding_model_id
        self.embedding_deployment_id = embedding_deployment_id

    @property
    def profile_id(self) -> str:
        return (
            f"token-profile-v3|model={self.model_name}|encoding={self.encoding_name}|"
            f"encoding_id={self.encoding_id}|version={self.version}|max_tokens={self.max_tokens}|"
            f"embedding_model_id={self.embedding_model_id}|embedding_deployment_id={self.embedding_deployment_id}"
        )


def _token_minhash_profile_id(token_profile_id: str, config: MinHashConfig) -> str:
    return (
        "v3-token-minhash-v1"
        f"|token_profile={token_profile_id}"
        f"|seed={config.seed}|n={config.ngram_size}|perms={config.permutations}"
        f"|bands={config.lsh_bands}|rows={config.lsh_rows}|max_shingles={config.max_shingles}"
    )


_M61 = (1 << 61) - 1


def _stable_hash_m61(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % _M61


def _normalize_text(text: str) -> str:
    lowered = text.casefold().replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for ch in lowered:
        out.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return " ".join("".join(out).split())


def _packet_to_shingles(packet_json: str, ngram_size: int, max_shingles: int) -> set[str]:
    payload = json.loads(packet_json)
    events = payload.get("session", {}).get("events", [])
    fields: list[str] = []
    for row in events:
        role = _normalize_text(str(row.get("role") or ""))
        text = str(row.get("text") or "")
        tool_name = row.get("tool_name")
        arguments_json = row.get("arguments_json")
        output = row.get("output")
        if text:
            fields.append(f"role:{role}|text:{_normalize_text(text)}")
        if tool_name:
            fields.append(f"role:{role}|tool_name:{_normalize_text(str(tool_name))}")
        if arguments_json:
            fields.append(f"role:{role}|arguments:{_normalize_text(str(arguments_json))}")
        if output:
            fields.append(f"role:{role}|output:{_normalize_text(str(output))}")
    shingles: set[str] = set()
    for field in fields:
        toks = [tok for tok in field.split(" ") if tok]
        if not toks:
            continue
        if len(toks) < ngram_size:
            shingles.add(" ".join(toks))
            continue
        for i in range(len(toks) - ngram_size + 1):
            shingles.add(" ".join(toks[i : i + ngram_size]))
    if not shingles:
        shingles.add("empty-evidence")
    if len(shingles) > max_shingles:
        ranked = sorted((_stable_int(s), s) for s in shingles)
        shingles = {s for _, s in ranked[:max_shingles]}
    return shingles


def _build_token_minhash_record(
    packet: V3TokenPacketRecord,
    *,
    config: MinHashConfig,
    token_profile_id: str,
) -> MinHashRecord:
    seed_text = f"seed={config.seed}|perms={config.permutations}|n={config.ngram_size}"
    perm_a = [(((_stable_int(seed_text + f"|a|{i}") % (_M61 - 1)) + 1)) for i in range(config.permutations)]
    perm_b = [(_stable_int(seed_text + f"|b|{i}") % _M61) for i in range(config.permutations)]
    shingles = _packet_to_shingles(packet.canonical_json, config.ngram_size, config.max_shingles)
    hashed = tuple(_stable_hash_m61(shingle) for shingle in shingles)
    signature: list[int] = []
    for a, b in zip(perm_a, perm_b):
        signature.append(min((((a * x + b) % _M61) for x in hashed), default=0))
    return MinHashRecord(
        content_sha256=packet.content_sha256,
        profile_id=_token_minhash_profile_id(token_profile_id, config),
        signature=tuple(signature),
        shingle_count=len(shingles),
        representation_truncated=packet.truncated,
    )


class V3ReadonlyMinHashProvider:
    def __init__(self, records_by_trace_id: Mapping[int, MinHashRecord], config: MinHashConfig) -> None:
        self.cfg = config
        self._records = dict(records_by_trace_id)
        self.n_hits = 0
        self.n_builds = 0

    def build(self, trace: Trace) -> MinHashRecord:
        rec = self._records.get(int(trace.trace_id))
        if rec is None:
            raise KeyError(f"Missing V3 MinHash record for trace_id={trace.trace_id}")
        self.n_hits += 1
        return rec


class _V3EmbeddingProfileView:
    def __init__(self, cache_version: str) -> None:
        self.cache_version = cache_version


class V3ReadonlyEmbeddingCache:
    """Readonly embedding cache compatible with AzureClusterIndex."""

    def __init__(self, profile_cache_version: str, vectors_by_trace_id: Mapping[int, np.ndarray]) -> None:
        self.profile = _V3EmbeddingProfileView(profile_cache_version)
        self._vectors = dict(vectors_by_trace_id)

    def contains_trace(self, trace: Trace) -> bool:
        return int(trace.trace_id) in self._vectors

    def peek_trace(self, trace: Trace) -> np.ndarray | None:
        return self._vectors.get(int(trace.trace_id))

    def get_trace(self, trace: Trace) -> np.ndarray:
        vec = self.peek_trace(trace)
        if vec is None:
            raise KeyError(f"Missing V3 embedding for trace_id={trace.trace_id}")
        return vec


class _TelemetryVectorStore(VectorStore):
    def __init__(self, inner: VectorStore) -> None:
        self._inner = inner
        self.search_queries = 0
        self.writes = 0
        self.cleanup_calls = 0
        self.cleanup_deleted = 0

    def nearest(self, vec, agent_id=None, semantic_scope=None, tenant_id=None, run_scope=None):
        self.search_queries += 1
        return self._inner.nearest(
            vec,
            agent_id=agent_id,
            semantic_scope=semantic_scope,
            tenant_id=tenant_id,
            run_scope=run_scope,
        )

    def upsert(self, doc):
        self.writes += 1
        self._inner.upsert(doc)

    def touch(self, cluster_id: str, now: float) -> None:
        self._inner.touch(cluster_id, now)

    def purge_stale(self, now: float, ttl: float, semantic_scope=None, tenant_id=None, run_scope=None):
        return self._inner.purge_stale(
            now=now,
            ttl=ttl,
            semantic_scope=semantic_scope,
            tenant_id=tenant_id,
            run_scope=run_scope,
        )

    def delete_scope(self, tenant_id: str, run_scope: str, semantic_scope=None):
        self.cleanup_calls += 1
        ids, count = self._inner.delete_scope(tenant_id, run_scope, semantic_scope=semantic_scope)
        self.cleanup_deleted += int(count)
        return ids, count

    def delete_scope_settled(
        self,
        tenant_id: str,
        run_scope: str,
        semantic_scope=None,
        *,
        max_attempts: int = 3,
        settle_seconds: float = 0.0,
    ):
        self.cleanup_calls += 1
        ids, count = self._inner.delete_scope_settled(
            tenant_id,
            run_scope,
            semantic_scope=semantic_scope,
            max_attempts=max_attempts,
            settle_seconds=settle_seconds,
        )
        self.cleanup_deleted += int(count)
        return ids, count


def build_v3_runtime(
    data: CombinedDataset,
    *,
    tokenizer: Any | None = None,
    embedder: V3Embedder,
    embedding_model_id: str = V3_EMBEDDING_MODEL,
    embedding_deployment_id: str | None = None,
    max_session_packet_tokens: int = V3_MAX_SESSION_PACKET_TOKENS,
    embedding_batch_size: int = V3_DEFAULT_EMBEDDING_BATCH_SIZE,
    embedding_dimensions: int = V3_EMBEDDING_DIMENSIONS,
    minhash_config: MinHashConfig | None = None,
) -> V3Runtime:
    if embedder is None:
        raise ValueError("embedder is required; pass an explicit embedder instance")
    if not embedding_model_id:
        raise ValueError("embedding_model_id is required")

    tok = tokenizer or TiktokenTokenizer(model_name=V3_EMBEDDING_MODEL, encoding_name=V3_EMBEDDING_ENCODING)
    deployment_id = embedding_deployment_id or embedding_model_id
    token_profile = _TokenProfile(
        tok,
        max_session_packet_tokens,
        embedding_model_id=embedding_model_id,
        embedding_deployment_id=deployment_id,
    )
    minhash_cfg = minhash_config or MinHashConfig(
        ngram_size=3,
        permutations=128,
        lsh_bands=32,
        lsh_rows=4,
        seed=13,
        similarity_threshold=0.55,
        max_shingles=4096,
        ttl_s=90.0,
        max_clusters_per_agent=256,
        max_clusters_total=4096,
    )
    packet_builder = TokenSessionEvidencePacketBuilder(
        options=CanonicalizationOptions(tokenizer=tok, max_tokens=max_session_packet_tokens),
        max_size=max(4096, len(data.unit_ids)),
    )
    unit_norm_embedder = _UnitNorm1536Embedder(embedder, embedding_dimensions)

    packet_records_by_unit_id: dict[str, V3TokenPacketRecord] = {}
    minhash_records_by_unit_id: dict[str, MinHashRecord] = {}
    token_cost_by_unit_id: dict[str, int] = {}
    unit_id_by_trace_id: dict[int, str] = {}
    packet_hashes_by_trace_id: dict[int, str] = {}
    unique_packets: dict[str, tuple[str, int]] = {}

    for uid in data.unit_ids:
        trace = data.trace_by_unit_id[uid]
        packed = packet_builder.build(trace)
        content_sha = _sha256_text(packed.canonical_json)
        packet = V3TokenPacketRecord(
            unit_id=uid,
            trace_id=int(trace.trace_id),
            content_sha256=content_sha,
            canonical_json=packed.canonical_json,
            original_tokens=int(packed.original_tokens),
            emitted_tokens=int(packed.emitted_tokens),
            truncated=bool(packed.truncated),
        )
        packet_records_by_unit_id[uid] = packet
        token_cost = int(packet.emitted_tokens)
        if token_cost <= 0:
            raise ValueError(f"non-positive emitted token cost for unit_id={uid}: {token_cost}")
        token_cost_by_unit_id[uid] = token_cost
        unit_id_by_trace_id[int(trace.trace_id)] = uid
        packet_hashes_by_trace_id[int(trace.trace_id)] = content_sha
        unique_packets.setdefault(content_sha, (packet.canonical_json, packet.emitted_tokens))
        minhash_records_by_unit_id[uid] = _build_token_minhash_record(
            packet,
            config=minhash_cfg,
            token_profile_id=token_profile.profile_id,
        )

    unique_hashes = sorted(unique_packets)
    embedding_records_by_content_sha256: dict[str, V3EmbeddingRecord] = {}
    embedding_calls = 0
    embedding_inputs = 0
    embedding_input_tokens = 0
    embedding_latency_seconds = 0.0

    for start in range(0, len(unique_hashes), max(1, embedding_batch_size)):
        batch_hashes = unique_hashes[start : start + max(1, embedding_batch_size)]
        texts = [unique_packets[h][0] for h in batch_hashes]
        embedding_inputs += len(texts)
        embedding_input_tokens += sum(unique_packets[h][1] for h in batch_hashes)
        t0 = perf_counter()
        vectors = unit_norm_embedder.embed(texts)
        embedding_latency_seconds += perf_counter() - t0
        embedding_calls += 1
        for h, vec in zip(batch_hashes, vectors):
            embedding_records_by_content_sha256[h] = V3EmbeddingRecord(
                content_sha256=h,
                vector=np.asarray(vec, dtype=np.float32),
                dimensions=embedding_dimensions,
                input_tokens=unique_packets[h][1],
            )

    embedding_vector_by_trace_id: dict[int, np.ndarray] = {}
    for uid, packet in packet_records_by_unit_id.items():
        trace = data.trace_by_unit_id[uid]
        rec = embedding_records_by_content_sha256[packet.content_sha256]
        embedding_vector_by_trace_id[int(trace.trace_id)] = rec.vector

    embedding_profile_payload = {
        "model_id": embedding_model_id,
        "deployment_id": deployment_id,
        "dimensions": embedding_dimensions,
        "token_profile": token_profile.profile_id,
    }
    embedding_profile_id = _sha256_text(_canonical_json(embedding_profile_payload))
    embedding_scope = _sha256_text(_canonical_json({**embedding_profile_payload, "tau": V3_DEFAULT_EMBEDDING_TAU}))

    return V3Runtime(
        version=V3_VERSION,
        token_profile_id=token_profile.profile_id,
        minhash_profile_id=_token_minhash_profile_id(token_profile.profile_id, minhash_cfg),
        embedding_profile_id=embedding_profile_id,
        embedding_semantic_scope=embedding_scope,
        packet_records_by_unit_id=packet_records_by_unit_id,
        minhash_records_by_unit_id=minhash_records_by_unit_id,
        embedding_records_by_content_sha256=embedding_records_by_content_sha256,
        embedding_vector_by_trace_id=embedding_vector_by_trace_id,
        packet_hashes_by_trace_id=packet_hashes_by_trace_id,
        token_cost_by_unit_id=token_cost_by_unit_id,
        unit_id_by_trace_id=unit_id_by_trace_id,
        ledger=V3BuildLedger(
            packet_builds=packet_builder.n_builds,
            packet_cache_hits=packet_builder.n_hits,
            embedding_calls=embedding_calls,
            embedding_inputs=embedding_inputs,
            embedding_input_tokens=embedding_input_tokens,
            embedding_latency_seconds=embedding_latency_seconds,
            embedding_content_hashes=tuple(unique_hashes),
            embedding_model_id=embedding_model_id,
            embedding_deployment_id=deployment_id,
            embedding_embedder_class=type(embedder).__name__,
        ),
    )


def build_v3_offline_runtime(
    data: CombinedDataset,
    *,
    tokenizer: Any | None = None,
    seed: int = 13,
    max_session_packet_tokens: int = V3_MAX_SESSION_PACKET_TOKENS,
    embedding_batch_size: int = V3_DEFAULT_EMBEDDING_BATCH_SIZE,
    embedding_dimensions: int = V3_EMBEDDING_DIMENSIONS,
    minhash_config: MinHashConfig | None = None,
) -> V3Runtime:
    return build_v3_runtime(
        data,
        tokenizer=tokenizer,
        embedder=Deterministic1536Embedder(seed=seed, dimensions=embedding_dimensions),
        embedding_model_id=V3_EMBEDDING_MODEL,
        embedding_deployment_id=V3_EMBEDDING_MODEL,
        max_session_packet_tokens=max_session_packet_tokens,
        embedding_batch_size=embedding_batch_size,
        embedding_dimensions=embedding_dimensions,
        minhash_config=minhash_config,
    )


def build_exact_token_budget_manifest(
    runtime: V3Runtime,
    *,
    eligible_unit_ids: Sequence[str] | None = None,
    outcome_tiers_pct: Sequence[int] = (5, 10, 20, 30, 50),
    quadrant_tiers_pct: Sequence[int] = (15, 30),
    extra_tiers_pct: Sequence[int] = (),
) -> dict[str, Any]:
    eligible = tuple(eligible_unit_ids or tuple(runtime.token_cost_by_unit_id.keys()))
    eligible_mass = int(sum(runtime.token_cost_by_unit_id[uid] for uid in eligible))

    def _as_rows(tiers: Sequence[int]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for share in tiers:
            rows.append(
                {
                    "legacy_tier_pct": int(share),
                    "budget_tokens": int(floor((share / 100.0) * eligible_mass)),
                }
            )
        return rows

    return {
        "eligible_unit_count": len(eligible),
        "eligible_token_mass": eligible_mass,
        "outcome": _as_rows(outcome_tiers_pct),
        "quadrant": _as_rows(quadrant_tiers_pct),
        "extra": _as_rows(extra_tiers_pct),
    }


def _pack_maximal(order: Sequence[str], token_cost_by_unit_id: Mapping[str, int], budget_tokens: int) -> tuple[list[str], int]:
    selected: list[str] = []
    total = 0
    for uid in order:
        cost = int(token_cost_by_unit_id[uid])
        if cost <= 0:
            continue
        if total + cost <= budget_tokens:
            selected.append(uid)
            total += cost
    return selected, total


def _stable_rank(seed: int, key: str) -> str:
    return _sha256_text(f"{seed}|{key}")


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    return float(np.percentile(arr, p))


def _concept_key(meta: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(meta.get("corpus_id") or "unknown"),
            str(meta.get("domain") or "unknown"),
            str(meta.get("task") or "unknown"),
            str(meta.get("difficulty") or "unknown"),
        )
    )


def _representation_ratio(data: CombinedDataset, selected_ids: Sequence[str], eligible_ids: Sequence[str]) -> float:
    all_concepts = {_concept_key(data.metadata_by_unit[uid]) for uid in eligible_ids}
    selected_concepts = {_concept_key(data.metadata_by_unit[uid]) for uid in selected_ids}
    return (len(selected_concepts) / len(all_concepts)) if all_concepts else 0.0


def _zero_selection_agent_rate(data: CombinedDataset, selected_ids: Sequence[str], eligible_ids: Sequence[str]) -> float:
    units_by_id = {str(unit.unit_id or ""): unit for unit in data.units}
    selected = set(selected_ids)
    by_agent: dict[str, list[str]] = {}
    for uid in eligible_ids:
        unit = units_by_id[uid]
        agent = f"{unit.tenant_id}|{unit.agent_id}"
        by_agent.setdefault(agent, []).append(uid)
    if not by_agent:
        return 0.0
    zero = 0
    for _, uids in by_agent.items():
        if not any(uid in selected for uid in uids):
            zero += 1
    return float(zero / len(by_agent))


def _prepare_replay(
    data: CombinedDataset,
    ordered_unit_ids: Sequence[str],
) -> list[tuple[str, Trace]]:
    rows: list[tuple[str, Trace]] = []
    for idx, uid in enumerate(ordered_unit_ids):
        base = data.trace_by_unit_id[uid]
        rows.append(
            (
                uid,
                Trace(
                    trace_id=base.trace_id,
                    agent_id=base.agent_id,
                    timestamp=float(idx),
                    signature=base.signature,
                    span_count=base.span_count,
                    duration_ms=base.duration_ms,
                    status=base.status,
                    concept_id=base.concept_id,
                    events=base.events,
                ),
            )
        )
    return rows


def _select_random_token_priority(
    *,
    eligible_unit_ids: Sequence[str],
    token_cost_by_unit_id: Mapping[str, int],
    budget_tokens: int,
    seed: int,
) -> dict[str, Any]:
    t0 = perf_counter()
    ranked = sorted(eligible_unit_ids, key=lambda uid: _stable_rank(seed, uid))
    selected, selected_tokens = _pack_maximal(ranked, token_cost_by_unit_id, budget_tokens)
    unselected = [uid for uid in ranked if uid not in set(selected)]
    min_unselected = min((token_cost_by_unit_id[uid] for uid in unselected), default=None)
    slack = int(budget_tokens - selected_tokens)
    elapsed_ms = (perf_counter() - t0) * 1000.0
    return {
        "selected_ids": selected,
        "selected_tokens": int(selected_tokens),
        "slack_tokens": slack,
        "native_ids": selected,
        "fill_ids": [],
        "native_proposed_ids": ranked,
        "native_candidate_ids": ranked,
        "fill_candidate_ids": [],
        "native_proposed_count": len(ranked),
        "native_proposed_tokens": int(sum(token_cost_by_unit_id[uid] for uid in ranked)),
        "native_pack_order": ranked,
        "fill_pack_order": [],
        "min_unselected_token_cost": None if min_unselected is None else int(min_unselected),
        "telemetry": {
            "selector": "random_token_priority",
            "proposal_mode": "random_ranked_token_pack",
            "fallbacks": 0,
            "search_queries": 0,
            "writes": 0,
            "cleanup_deleted": 0,
            "decision_count": 1,
            "decision_runtime_ms_p50": float(elapsed_ms),
            "decision_runtime_ms_p95": float(elapsed_ms),
            "decision_runtime_ms_total": float(elapsed_ms),
        },
    }


def _build_minhash_variety_index(runtime: V3Runtime, seed: int) -> BandedMinHashLSHIndex:
    cfg = MinHashConfig(
        ngram_size=3,
        permutations=128,
        lsh_bands=32,
        lsh_rows=4,
        seed=seed,
        similarity_threshold=0.55,
        ttl_s=90.0,
        max_clusters_per_agent=256,
        max_clusters_total=4096,
    )
    provider = V3ReadonlyMinHashProvider(
        {
            trace_id: runtime.minhash_records_by_unit_id[uid]
            for trace_id, uid in runtime.unit_id_by_trace_id.items()
            if uid in runtime.minhash_records_by_unit_id
        },
        cfg,
    )
    return BandedMinHashLSHIndex(cfg, signature_provider=provider)


class V3EmbeddingSelector:
    def __init__(
        self,
        runtime: V3Runtime,
        *,
        vector_store_factory: Callable[[str, str], VectorStore] | None = None,
        tau: float = V3_DEFAULT_EMBEDDING_TAU,
        recent_buffer_size: int = 4096,
        cleanup_max_attempts: int = 3,
        cleanup_settle_seconds: float = 0.0,
    ) -> None:
        self.runtime = runtime
        self.vector_store_factory = vector_store_factory or (lambda _tenant, _scope: InMemoryVectorStore())
        self.tau = tau
        self.recent_buffer_size = recent_buffer_size
        self.cleanup_max_attempts = cleanup_max_attempts
        self.cleanup_settle_seconds = cleanup_settle_seconds

    def build_index(self, *, tenant_id: str, run_scope: str) -> tuple[AzureClusterIndex, _TelemetryVectorStore]:
        base_store = self.vector_store_factory(tenant_id, run_scope)
        store = _TelemetryVectorStore(base_store)
        cache = V3ReadonlyEmbeddingCache(self.runtime.embedding_profile_id, self.runtime.embedding_vector_by_trace_id)
        # We intentionally freeze tau at 0.55 for live text-embedding-3-small.
        index = AzureClusterIndex(
            cache,
            store,
            tau=self.tau,
            ttl=90.0,
            purge_every=200,
            embed_budget_per_tick=10_000_000,
            recent_buffer_size=self.recent_buffer_size,
            semantic_scope=self.runtime.embedding_semantic_scope,
            tenant_id=tenant_id,
            run_scope=run_scope,
        )
        return index, store


def _select_adaptive_with_fill(
    *,
    data: CombinedDataset,
    ordered_unit_ids: Sequence[str],
    runtime: V3Runtime,
    budget_tokens: int,
    seed: int,
    method: str,
    llm_throughput: float,
    vector_store_factory: Callable[[str, str], VectorStore] | None = None,
    tenant_id: str,
    run_scope: str,
    cleanup_max_attempts: int = 10,
    cleanup_settle_seconds: float = 0.0,
) -> dict[str, Any]:
    cfg = SamplerConfig(
        llm_throughput=max(float(llm_throughput), 1e-6),
        agent_floor=0.0,
        enforce_keep_one_floor=False,
    )

    store_telemetry: dict[str, int] = {
        "search_queries": 0,
        "writes": 0,
        "cleanup_deleted": 0,
    }

    if method == "adaptive_minhash_32x4_token":
        index = _build_minhash_variety_index(runtime, seed=seed)
        sampler = AdaptiveSampler(cfg, seed=seed, variety_index=index, use_novelty=True)
    elif method == "adaptive_embedding_fullsession_token":
        selector = V3EmbeddingSelector(
            runtime,
            vector_store_factory=vector_store_factory,
            cleanup_max_attempts=cleanup_max_attempts,
            cleanup_settle_seconds=cleanup_settle_seconds,
        )
        index, store = selector.build_index(tenant_id=tenant_id, run_scope=run_scope)
        _, deleted_before = store.delete_scope_settled(
            tenant_id,
            run_scope,
            semantic_scope=runtime.embedding_semantic_scope,
            max_attempts=selector.cleanup_max_attempts,
            settle_seconds=selector.cleanup_settle_seconds,
        )
        sampler = AdaptiveSampler(cfg, seed=seed, variety_index=index, use_novelty=True)
        store_telemetry["cleanup_deleted"] += int(deleted_before)
    else:
        raise ValueError(f"unknown adaptive method: {method}")

    trace_rows = _prepare_replay(data, ordered_unit_ids)
    proposed_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    decision_latencies_ms: list[float] = []
    t_total_start = perf_counter()

    for uid, trace in trace_rows:
        t0 = perf_counter()
        _ = sampler.decide(trace, admit_keep=True)
        decision_latencies_ms.append((perf_counter() - t0) * 1000.0)
        obs = sampler.last_observation
        proposed = bool(sampler.last_proposed_keep)
        row = {
            "unit_id": uid,
            "novelty": float(getattr(obs, "novelty", 0.0) or 0.0),
            "rarity": float(getattr(obs, "rarity", 0.0) or 0.0),
            "rank": _stable_rank(seed, f"native|{uid}"),
        }
        if proposed:
            proposed_rows.append(row)
        else:
            rejected_rows.append(row)

    native_order = [row["unit_id"] for row in proposed_rows]
    selected_native, native_tokens = _pack_maximal(native_order, runtime.token_cost_by_unit_id, budget_tokens)
    selected_set = set(selected_native)

    rejected_rows.sort(
        key=lambda row: (
            -row["novelty"],
            -row["rarity"],
            row["rank"],
        )
    )
    fill_order = [row["unit_id"] for row in rejected_rows]
    remaining_budget = int(budget_tokens - native_tokens)
    selected_fill, fill_tokens = _pack_maximal(
        [uid for uid in fill_order if uid not in selected_set],
        runtime.token_cost_by_unit_id,
        remaining_budget,
    )

    selected_ids = selected_native + selected_fill
    selected_tokens = native_tokens + fill_tokens
    unselected = [uid for uid in ordered_unit_ids if uid not in set(selected_ids)]
    min_unselected = min((runtime.token_cost_by_unit_id[uid] for uid in unselected), default=None)

    fallbacks = getattr(getattr(sampler, "_variety", None), "n_fallbacks", 0)
    if method == "adaptive_embedding_fullsession_token":
        _, deleted_after = store.delete_scope_settled(
            tenant_id,
            run_scope,
            semantic_scope=runtime.embedding_semantic_scope,
            max_attempts=selector.cleanup_max_attempts,
            settle_seconds=selector.cleanup_settle_seconds,
        )
        store_telemetry["cleanup_deleted"] += int(deleted_after)
        store_telemetry["search_queries"] = store.search_queries
        store_telemetry["writes"] = store.writes

    total_runtime_ms = (perf_counter() - t_total_start) * 1000.0

    return {
        "selected_ids": selected_ids,
        "selected_tokens": int(selected_tokens),
        "slack_tokens": int(budget_tokens - selected_tokens),
        "native_ids": selected_native,
        "fill_ids": selected_fill,
        "native_proposed_ids": native_order,
        "native_candidate_ids": native_order,
        "fill_candidate_ids": fill_order,
        "native_proposed_count": len(native_order),
        "native_proposed_tokens": int(sum(runtime.token_cost_by_unit_id[uid] for uid in native_order)),
        "native_pack_order": native_order,
        "fill_pack_order": fill_order,
        "min_unselected_token_cost": None if min_unselected is None else int(min_unselected),
        "telemetry": {
            "selector": method,
            "proposal_mode": "adaptive_native_then_fill",
            "fallbacks": int(fallbacks),
            "decision_count": len(decision_latencies_ms),
            "decision_runtime_ms_p50": _percentile(decision_latencies_ms, 50.0),
            "decision_runtime_ms_p95": _percentile(decision_latencies_ms, 95.0),
            "decision_runtime_ms_total": float(total_runtime_ms),
            **store_telemetry,
        },
    }


def select_v3_membership(
    data: CombinedDataset,
    *,
    runtime: V3Runtime,
    method: str,
    eligible_unit_ids: Sequence[str],
    budget_tokens: int,
    seed: int,
    llm_throughput: float = 1_000.0,
    vector_store_factory: Callable[[str, str], VectorStore] | None = None,
    tenant_id: str = "sampling-v3",
    run_scope: str = "default",
    cleanup_max_attempts: int = 10,
    cleanup_settle_seconds: float = 0.0,
) -> dict[str, Any]:
    t0 = perf_counter()
    if method == "random_sampling_token_priority":
        result = _select_random_token_priority(
            eligible_unit_ids=eligible_unit_ids,
            token_cost_by_unit_id=runtime.token_cost_by_unit_id,
            budget_tokens=budget_tokens,
            seed=seed,
        )
    elif method in {"adaptive_minhash_32x4_token", "adaptive_embedding_fullsession_token"}:
        result = _select_adaptive_with_fill(
            data=data,
            ordered_unit_ids=eligible_unit_ids,
            runtime=runtime,
            budget_tokens=budget_tokens,
            seed=seed,
            method=method,
            llm_throughput=llm_throughput,
            vector_store_factory=vector_store_factory,
            tenant_id=tenant_id,
            run_scope=run_scope,
            cleanup_max_attempts=cleanup_max_attempts,
            cleanup_settle_seconds=cleanup_settle_seconds,
        )
    else:
        raise ValueError(f"unknown v3 method: {method}")

    selected_ids = list(result["selected_ids"])
    selected_set = set(selected_ids)
    selected_tokens = int(sum(runtime.token_cost_by_unit_id[uid] for uid in selected_ids))
    if selected_tokens != int(result["selected_tokens"]):
        raise RuntimeError("selected token accounting mismatch")

    slack_tokens = int(budget_tokens - selected_tokens)
    min_unselected = result["min_unselected_token_cost"]
    if min_unselected is not None and not (slack_tokens < int(min_unselected)):
        raise RuntimeError("selection is not maximal under token budget")

    native_ids = list(result["native_ids"])
    fill_ids = list(result["fill_ids"])
    native_tokens = int(sum(runtime.token_cost_by_unit_id[uid] for uid in native_ids))
    fill_tokens = int(sum(runtime.token_cost_by_unit_id[uid] for uid in fill_ids))

    selected_pass = sum(1 for uid in selected_ids if data.labels_by_unit[uid])
    selected_count = len(selected_ids)
    population_count = len(eligible_unit_ids)
    census_pass = sum(1 for uid in eligible_unit_ids if data.labels_by_unit[uid])

    all_concepts = {
        "|".join(
            (
                str(data.metadata_by_unit[uid].get("corpus_id") or "unknown"),
                str(data.metadata_by_unit[uid].get("domain") or "unknown"),
                str(data.metadata_by_unit[uid].get("task") or "unknown"),
                str(data.metadata_by_unit[uid].get("difficulty") or "unknown"),
            )
        )
        for uid in eligible_unit_ids
    }
    selected_concepts = {
        "|".join(
            (
                str(data.metadata_by_unit[uid].get("corpus_id") or "unknown"),
                str(data.metadata_by_unit[uid].get("domain") or "unknown"),
                str(data.metadata_by_unit[uid].get("task") or "unknown"),
                str(data.metadata_by_unit[uid].get("difficulty") or "unknown"),
            )
        )
        for uid in selected_ids
    }

    selected_original_tokens = int(sum(runtime.packet_records_by_unit_id[uid].original_tokens for uid in selected_ids))
    selected_truncated_count = int(sum(1 for uid in selected_ids if runtime.packet_records_by_unit_id[uid].truncated))
    zero_rate = _zero_selection_agent_rate(data, selected_ids, eligible_unit_ids)
    total_runtime_ms = (perf_counter() - t0) * 1000.0

    return {
        "method": method,
        "budget_tokens": int(budget_tokens),
        "selected_tokens": int(selected_tokens),
        "slack_tokens": int(slack_tokens),
        "budget_utilization_tokens": (float(selected_tokens) / float(budget_tokens)) if budget_tokens > 0 else 0.0,
        "selected_count": selected_count,
        "native_count": len(native_ids),
        "native_tokens": native_tokens,
        "fill_count": len(fill_ids),
        "fill_tokens": fill_tokens,
        "token_original_selected": selected_original_tokens,
        "token_emitted_selected": int(selected_tokens),
        "token_truncated_selected_count": selected_truncated_count,
        "selected_ids": selected_ids,
        "selected_pass_rate": (selected_pass / selected_count) if selected_count else 0.0,
        "census_pass_rate": (census_pass / population_count) if population_count else 0.0,
        "absolute_error": abs(((selected_pass / selected_count) if selected_count else 0.0) - ((census_pass / population_count) if population_count else 0.0)),
        "fraction_saved": 1.0 - ((selected_count / population_count) if population_count else 0.0),
        "concept_coverage": (len(selected_concepts) / len(all_concepts)) if all_concepts else 0.0,
        "representation": _representation_ratio(data, selected_ids, eligible_unit_ids),
        "zero_selection_agent_rate": float(zero_rate),
        "telemetry": result["telemetry"],
        "native_candidate_count": len(result["native_candidate_ids"]),
        "native_proposed_tokens": int(result["native_proposed_tokens"]),
        "native_proposed_ids": list(result["native_proposed_ids"]),
        "min_unselected_token_cost": result["min_unselected_token_cost"],
        "decision_runtime_ms_p50": float(result["telemetry"].get("decision_runtime_ms_p50", 0.0)),
        "decision_runtime_ms_p95": float(result["telemetry"].get("decision_runtime_ms_p95", 0.0)),
        "selection_runtime_ms": float(total_runtime_ms),
    }


def _aggregate_outcome_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in runs:
        key = (str(row["method"]), int(row["budget_tokens"]))
        grouped.setdefault(key, []).append(row)

    def _stat(rows: Sequence[dict[str, Any]], field: str) -> dict[str, float]:
        vals = [float(r[field]) for r in rows]
        return {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "empirical_low": float(min(vals)) if vals else 0.0,
            "empirical_high": float(max(vals)) if vals else 0.0,
        }

    out: list[dict[str, Any]] = []
    for (method, budget_tokens), bucket in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        legacy_tiers = sorted({int(r["legacy_tier_pct"]) for r in bucket})
        out.append(
            {
                "method": method,
                "budget_tokens": int(budget_tokens),
                "legacy_tier_pct_provenance": legacy_tiers,
                "replays": len(bucket),
                "mae": _stat(bucket, "absolute_error"),
                "concept_coverage": _stat(bucket, "concept_coverage"),
                "fraction_saved": _stat(bucket, "fraction_saved"),
                "selected_count": _stat(bucket, "selected_count"),
                "selected_tokens": _stat(bucket, "selected_tokens"),
                "token_utilization": _stat(bucket, "budget_utilization_tokens"),
                "token_slack": _stat(bucket, "slack_tokens"),
                "native_count": _stat(bucket, "native_count"),
                "native_tokens": _stat(bucket, "native_tokens"),
                "fill_count": _stat(bucket, "fill_count"),
                "fill_tokens": _stat(bucket, "fill_tokens"),
                "measured_telemetry": {
                    "decision_runtime_ms_p50": _stat(bucket, "decision_runtime_ms_p50"),
                    "decision_runtime_ms_p95": _stat(bucket, "decision_runtime_ms_p95"),
                    "selection_runtime_ms": _stat(bucket, "selection_runtime_ms"),
                    "search_queries": {
                        "mean": float(np.mean([float(r["telemetry"].get("search_queries", 0.0)) for r in bucket])) if bucket else 0.0,
                        "empirical_low": float(min([float(r["telemetry"].get("search_queries", 0.0)) for r in bucket])) if bucket else 0.0,
                        "empirical_high": float(max([float(r["telemetry"].get("search_queries", 0.0)) for r in bucket])) if bucket else 0.0,
                    },
                    "writes": {
                        "mean": float(np.mean([float(r["telemetry"].get("writes", 0.0)) for r in bucket])) if bucket else 0.0,
                        "empirical_low": float(min([float(r["telemetry"].get("writes", 0.0)) for r in bucket])) if bucket else 0.0,
                        "empirical_high": float(max([float(r["telemetry"].get("writes", 0.0)) for r in bucket])) if bucket else 0.0,
                    },
                    "fallbacks": {
                        "mean": float(np.mean([float(r["telemetry"].get("fallbacks", 0.0)) for r in bucket])) if bucket else 0.0,
                        "empirical_low": float(min([float(r["telemetry"].get("fallbacks", 0.0)) for r in bucket])) if bucket else 0.0,
                        "empirical_high": float(max([float(r["telemetry"].get("fallbacks", 0.0)) for r in bucket])) if bucket else 0.0,
                    },
                },
            }
        )
    return out


def run_v3_outcome_comparison(
    data: CombinedDataset,
    *,
    runtime: V3Runtime | None = None,
    methods: Sequence[str] = V3_OUTCOME_METHODS,
    legacy_outcome_tiers_pct: Sequence[int] = (5, 10, 20, 30, 50),
    repetitions: int = 3,
    seed: int = 13,
    llm_throughput: float = 1_000.0,
    vector_store_factory: Callable[[str, str], VectorStore] | None = None,
    cleanup_max_attempts: int = 10,
    cleanup_settle_seconds: float = 0.0,
) -> dict[str, Any]:
    if runtime is None:
        raise ValueError(
            "runtime is required; build it with build_v3_runtime(..., embedder=...) "
            "or build_v3_offline_runtime(...)"
        )
    rt = runtime
    budget_manifest = build_exact_token_budget_manifest(
        rt,
        eligible_unit_ids=data.unit_ids,
        outcome_tiers_pct=legacy_outcome_tiers_pct,
    )

    runs: list[dict[str, Any]] = []
    pair_manifest: list[dict[str, Any]] = []

    for repetition in range(repetitions):
        order = _paired_permutation(data.unit_ids, token=f"v3|outcome|seed={seed}|rep={repetition}")
        order_hash = _sha256_text("|".join(order))
        pair_manifest.append(
            {
                "repetition": repetition,
                "order_hash": order_hash,
                "unit_count": len(order),
            }
        )

        for tier in budget_manifest["outcome"]:
            legacy_tier = int(tier["legacy_tier_pct"])
            budget_tokens = int(tier["budget_tokens"])
            for method in methods:
                run_scope = f"v3-r{repetition}-t{legacy_tier}-{method}-{order_hash[:10]}"
                row = select_v3_membership(
                    data,
                    runtime=rt,
                    method=method,
                    eligible_unit_ids=order,
                    budget_tokens=budget_tokens,
                    seed=seed + repetition,
                    llm_throughput=llm_throughput,
                    vector_store_factory=vector_store_factory,
                    run_scope=run_scope,
                    cleanup_max_attempts=cleanup_max_attempts,
                    cleanup_settle_seconds=cleanup_settle_seconds,
                )
                row["repetition"] = repetition
                row["legacy_tier_pct"] = legacy_tier
                row["order_hash"] = order_hash
                runs.append(row)

    return {
        "version": V3_OUTCOME_VERSION,
        "runtime_version": rt.version,
        "population_count": len(data.unit_ids),
        "eligible_token_mass": budget_manifest["eligible_token_mass"],
        "pairing": {
            "paired_order_manifest": pair_manifest,
            "repetitions": repetitions,
        },
        "aggregate": _aggregate_outcome_runs(runs),
        "runs": runs,
    }


def simulate_token_throughput(
    *,
    native_rows: Sequence[dict[str, Any]],
    token_cost_by_unit_id: Mapping[str, int],
    eval_tokens_per_second: float,
    max_queue_tokens: float,
) -> dict[str, Any]:
    queue_tokens = 0.0
    max_queue_observed = 0.0
    proposed_tokens = 0
    admitted_tokens = 0
    admitted_count = 0
    proposed_count = 0
    last_ts: float | None = None
    series: list[dict[str, Any]] = []

    for row in native_rows:
        uid = str(row["unit_id"])
        ts = float(row["timestamp"])
        if last_ts is not None:
            dt = max(0.0, ts - last_ts)
            queue_tokens = max(0.0, queue_tokens - (float(eval_tokens_per_second) * dt))
        last_ts = ts

        cost = int(token_cost_by_unit_id[uid])
        proposed_count += 1
        proposed_tokens += cost

        admitted = queue_tokens + cost <= max_queue_tokens
        if admitted:
            queue_tokens += cost
            admitted_count += 1
            admitted_tokens += cost
            if queue_tokens > max_queue_observed:
                max_queue_observed = queue_tokens

        series.append(
            {
                "unit_id": uid,
                "timestamp": ts,
                "token_cost": cost,
                "queue_tokens": queue_tokens,
                "admitted": admitted,
            }
        )

    return {
        "proposed_count": proposed_count,
        "proposed_tokens": proposed_tokens,
        "admitted_count": admitted_count,
        "admitted_tokens": admitted_tokens,
        "max_queue_tokens": float(max_queue_observed),
        "final_queue_tokens": queue_tokens,
        "series": series,
    }


def run_v3_quadrant_experiment(
    data: CombinedDataset,
    *,
    runtime: V3Runtime,
    methods: Sequence[str] = V3_QUADRANT_METHODS,
    legacy_quadrant_tiers_pct: Sequence[int] = (15, 30),
    replay_count: int = 3,
    seed: int = 13,
    llm_throughput: float = 1_000.0,
    tenant_id: str = "sampling-v3-experiment",
    vector_store_factory: Callable[[str, str], VectorStore] | None = None,
    cleanup_max_attempts: int = 10,
    cleanup_settle_seconds: float = 0.0,
) -> dict[str, Any]:
    quadrants = assign_population_quadrants(data)
    by_quadrant: dict[str, list[str]] = {}
    for row in quadrants["assignments"]:
        by_quadrant.setdefault(str(row["quadrant"]), []).append(str(row["unit_id"]))

    runs: list[dict[str, Any]] = []
    budget_manifest_by_quadrant: dict[str, Any] = {}
    for quadrant_name, eligible in sorted(by_quadrant.items()):
        q_manifest = build_exact_token_budget_manifest(
            runtime,
            eligible_unit_ids=eligible,
            outcome_tiers_pct=(),
            quadrant_tiers_pct=legacy_quadrant_tiers_pct,
        )
        budget_manifest_by_quadrant[quadrant_name] = q_manifest
        for replay in range(replay_count):
            order = _paired_permutation(eligible, f"v3|quadrant|{quadrant_name}|seed={seed}|rep={replay}")
            order_hash = _sha256_text("|".join(order))
            for tier in q_manifest["quadrant"]:
                legacy_tier = int(tier["legacy_tier_pct"])
                budget_tokens = int(tier["budget_tokens"])
                for method in methods:
                    run_scope = (
                        f"v3-q-{quadrant_name}-r{replay}-t{legacy_tier}-{method}-{order_hash[:10]}"
                    )
                    row = select_v3_membership(
                        data,
                        runtime=runtime,
                        method=method,
                        eligible_unit_ids=order,
                        budget_tokens=budget_tokens,
                        seed=seed + replay,
                        llm_throughput=llm_throughput,
                        vector_store_factory=vector_store_factory,
                        tenant_id=tenant_id,
                        run_scope=run_scope,
                        cleanup_max_attempts=cleanup_max_attempts,
                        cleanup_settle_seconds=cleanup_settle_seconds,
                    )
                    row.update(
                        {
                            "quadrant": quadrant_name,
                            "replay": replay,
                            "legacy_tier_pct": legacy_tier,
                            "eligible_count": len(order),
                            "eligible_token_mass": q_manifest["eligible_token_mass"],
                            "order_hash": order_hash,
                        }
                    )
                    runs.append(row)

    aggregate_groups: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in runs:
        key = (str(row["method"]), str(row["quadrant"]), int(row["budget_tokens"]))
        grouped.setdefault(key, []).append(row)
    for (method, quadrant_name, budget_tokens), bucket in sorted(grouped.items()):
        aggregate_groups.append(
            {
                "method": method,
                "quadrant": quadrant_name,
                "budget_tokens": int(budget_tokens),
                "legacy_tier_pct_provenance": sorted({int(r["legacy_tier_pct"]) for r in bucket}),
                "replays": len(bucket),
                "representation_mean": float(np.mean([r["representation"] for r in bucket])) if bucket else 0.0,
                "concept_coverage_mean": float(np.mean([r["concept_coverage"] for r in bucket])) if bucket else 0.0,
                "budget_utilization_tokens_mean": float(np.mean([r["budget_utilization_tokens"] for r in bucket])) if bucket else 0.0,
                "zero_selection_agent_rate_mean": float(np.mean([r["zero_selection_agent_rate"] for r in bucket])) if bucket else 0.0,
                "mae_mean": float(np.mean([r["absolute_error"] for r in bucket])) if bucket else 0.0,
                "decision_runtime_ms_p50_mean": float(np.mean([r["decision_runtime_ms_p50"] for r in bucket])) if bucket else 0.0,
                "decision_runtime_ms_p95_mean": float(np.mean([r["decision_runtime_ms_p95"] for r in bucket])) if bucket else 0.0,
                "native_count_mean": float(np.mean([r["native_count"] for r in bucket])) if bucket else 0.0,
                "fill_count_mean": float(np.mean([r["fill_count"] for r in bucket])) if bucket else 0.0,
                "native_tokens_mean": float(np.mean([r["native_tokens"] for r in bucket])) if bucket else 0.0,
                "fill_tokens_mean": float(np.mean([r["fill_tokens"] for r in bucket])) if bucket else 0.0,
            }
        )

    return {
        "version": "sampling-v3-quadrant-v1",
        "quadrants": quadrants,
        "budget_manifest_by_quadrant": budget_manifest_by_quadrant,
        "config": {
            "legacy_quadrant_tiers_pct": list(legacy_quadrant_tiers_pct),
            "replay_count": int(replay_count),
            "methods": list(methods),
        },
        "runs": runs,
        "aggregate_groups": aggregate_groups,
    }


def run_v3_throughput_grid_experiment(
    data: CombinedDataset,
    *,
    runtime: V3Runtime,
    methods: Sequence[str] = V3_THROUGHPUT_METHODS,
    legacy_budget_tiers_pct: Sequence[int] = (15, 30),
    arrival_rates_sessions_per_second: Sequence[float] = (0.25, 1.0, 4.0, 16.0),
    eval_capacity_rates_sessions_per_second: Sequence[float] = (0.25, 1.0, 4.0, 16.0),
    replay_count: int = 2,
    seed: int = 13,
    tenant_id: str = "sampling-v3-experiment",
    vector_store_factory: Callable[[str, str], VectorStore] | None = None,
    cleanup_max_attempts: int = 10,
    cleanup_settle_seconds: float = 0.0,
) -> dict[str, Any]:
    quadrants = assign_population_quadrants(data)
    high_variety = [str(row["unit_id"]) for row in quadrants["assignments"] if str(row["variety_band"]) == "high"]
    eligible = tuple(high_variety)
    if not eligible:
        raise ValueError("throughput experiment requires at least one high-variety eligible unit")

    budget_manifest = build_exact_token_budget_manifest(
        runtime,
        eligible_unit_ids=eligible,
        outcome_tiers_pct=(),
        quadrant_tiers_pct=legacy_budget_tiers_pct,
    )
    median_packet_tokens = float(median([runtime.token_cost_by_unit_id[uid] for uid in eligible]))
    eval_tokens_per_second_map = {
        float(rate): float(median_packet_tokens * float(rate)) for rate in eval_capacity_rates_sessions_per_second
    }

    runs: list[dict[str, Any]] = []
    for method in methods:
        for arrival_rate in arrival_rates_sessions_per_second:
            step = 1.0 / max(float(arrival_rate), 1e-9)
            for capacity_rate in eval_capacity_rates_sessions_per_second:
                eval_tokens_per_second = float(eval_tokens_per_second_map[float(capacity_rate)])
                for tier in budget_manifest["quadrant"]:
                    legacy_tier = int(tier["legacy_tier_pct"])
                    budget_tokens = int(tier["budget_tokens"])
                    for replay in range(replay_count):
                        order = _paired_permutation(
                            eligible,
                            (
                                "v3|throughput|"
                                f"{method}|a={arrival_rate}|c={capacity_rate}|"
                                f"t={legacy_tier}|seed={seed}|rep={replay}"
                            ),
                        )
                        order_hash = _sha256_text("|".join(order))
                        run_scope = (
                            f"v3-tp-{method}-a{arrival_rate}-c{capacity_rate}-"
                            f"t{legacy_tier}-r{replay}-{order_hash[:10]}"
                        )
                        membership = select_v3_membership(
                            data,
                            runtime=runtime,
                            method=method,
                            eligible_unit_ids=order,
                            budget_tokens=budget_tokens,
                            seed=seed + replay,
                            llm_throughput=eval_tokens_per_second,
                            vector_store_factory=vector_store_factory,
                            tenant_id=tenant_id,
                            run_scope=run_scope,
                            cleanup_max_attempts=cleanup_max_attempts,
                            cleanup_settle_seconds=cleanup_settle_seconds,
                        )
                        ts_by_uid = {uid: float(idx) * step for idx, uid in enumerate(order)}
                        native_rows = [
                            {
                                "unit_id": uid,
                                "timestamp": float(ts_by_uid.get(uid, float(idx) * step)),
                            }
                            for idx, uid in enumerate(membership["native_proposed_ids"])
                        ]
                        queue = simulate_token_throughput(
                            native_rows=native_rows,
                            token_cost_by_unit_id=runtime.token_cost_by_unit_id,
                            eval_tokens_per_second=eval_tokens_per_second,
                            max_queue_tokens=float(budget_tokens),
                        )
                        token_pressure_ratio = float(
                            queue["proposed_tokens"] / max(1.0, float(queue["admitted_tokens"]))
                        )
                        runs.append(
                            {
                                "method": method,
                                "proposal_mode": membership["telemetry"].get("proposal_mode"),
                                "arrival_rate_sessions_per_second": float(arrival_rate),
                                "eval_capacity_sessions_per_second": float(capacity_rate),
                                "eval_tokens_per_second": float(eval_tokens_per_second),
                                "legacy_tier_pct": legacy_tier,
                                "budget_tokens": int(budget_tokens),
                                "replay": replay,
                                "eligible_count": len(order),
                                "eligible_token_mass": int(budget_manifest["eligible_token_mass"]),
                                "order_hash": order_hash,
                                "queue_proposed_count": int(queue["proposed_count"]),
                                "queue_proposed_tokens": int(queue["proposed_tokens"]),
                                "queue_admitted_count": int(queue["admitted_count"]),
                                "queue_admitted_tokens": int(queue["admitted_tokens"]),
                                "queue_max_tokens": float(queue["max_queue_tokens"]),
                                "queue_final_tokens": float(queue["final_queue_tokens"]),
                                "token_pressure_ratio": float(token_pressure_ratio),
                                "selected_ids": list(membership["selected_ids"]),
                                "selected_count": int(membership["selected_count"]),
                                "selected_tokens": int(membership["selected_tokens"]),
                                "slack_tokens": int(membership["slack_tokens"]),
                                "budget_utilization_tokens": float(membership["budget_utilization_tokens"]),
                                "representation": float(membership["representation"]),
                                "concept_coverage": float(membership["concept_coverage"]),
                                "zero_selection_agent_rate": float(membership["zero_selection_agent_rate"]),
                                "decision_runtime_ms_p50": float(membership["decision_runtime_ms_p50"]),
                                "decision_runtime_ms_p95": float(membership["decision_runtime_ms_p95"]),
                                "selection_runtime_ms": float(membership["selection_runtime_ms"]),
                            }
                        )

    aggregate_grid: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float, float, int], list[dict[str, Any]]] = {}
    for row in runs:
        key = (
            str(row["method"]),
            float(row["arrival_rate_sessions_per_second"]),
            float(row["eval_capacity_sessions_per_second"]),
            int(row["budget_tokens"]),
        )
        grouped.setdefault(key, []).append(row)
    for (method, arrival_rate, capacity_rate, budget_tokens), bucket in sorted(grouped.items()):
        aggregate_grid.append(
            {
                "method": method,
                "arrival_rate_sessions_per_second": float(arrival_rate),
                "eval_capacity_sessions_per_second": float(capacity_rate),
                "budget_tokens": int(budget_tokens),
                "replays": len(bucket),
                "queue_admitted_tokens_mean": float(np.mean([r["queue_admitted_tokens"] for r in bucket])) if bucket else 0.0,
                "queue_max_tokens_mean": float(np.mean([r["queue_max_tokens"] for r in bucket])) if bucket else 0.0,
                "token_pressure_ratio_mean": float(np.mean([r["token_pressure_ratio"] for r in bucket])) if bucket else 0.0,
                "budget_utilization_tokens_mean": float(np.mean([r["budget_utilization_tokens"] for r in bucket])) if bucket else 0.0,
                "decision_runtime_ms_p95_mean": float(np.mean([r["decision_runtime_ms_p95"] for r in bucket])) if bucket else 0.0,
            }
        )

    return {
        "version": "sampling-v3-throughput-v1",
        "high_variety_population": {
            "eligible_count": len(eligible),
            "eligible_token_mass": int(budget_manifest["eligible_token_mass"]),
            "median_packet_tokens": float(median_packet_tokens),
        },
        "config": {
            "legacy_budget_tiers_pct": list(legacy_budget_tiers_pct),
            "arrival_rates_sessions_per_second": [float(x) for x in arrival_rates_sessions_per_second],
            "eval_capacity_rates_sessions_per_second": [float(x) for x in eval_capacity_rates_sessions_per_second],
            "eval_tokens_per_second_map": {
                str(rate): float(tokens) for rate, tokens in sorted(eval_tokens_per_second_map.items())
            },
            "queue_capacity_policy": "max_queue_tokens_equals_budget_tokens",
            "replay_count": int(replay_count),
            "methods": list(methods),
        },
        "budget_manifest": budget_manifest,
        "runs": runs,
        "aggregate_grid": aggregate_grid,
    }
