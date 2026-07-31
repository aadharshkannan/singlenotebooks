"""Dataset adapters and offline embedding index wiring for sampling comparison."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from minhash_sampling import MinHashClusterIndex, MinHashConfig
from random_sampling import EvaluationUnit, load_synthetic_a365_otel
from trace_sampling.cluster_index import AzureClusterIndex
from trace_sampling.model import SessionEvent, Trace
from trace_sampling.representation import CanonicalizationOptions, SessionEvidencePacketBuilder
from trace_sampling.session_embedding import EmbeddingProfile, SessionEmbeddingCache
from trace_sampling.vector_store import InMemoryVectorStore, VectorStore


def _stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _duration_ms(started_at: datetime | None, ended_at: datetime | None) -> float:
    if started_at is None or ended_at is None:
        return 0.0
    return max(0.0, (ended_at - started_at).total_seconds() * 1000.0)


def _signature_from_unit(unit: EvaluationUnit) -> tuple[str, ...]:
    names = tuple((call.name or "").strip() for call in unit.tool_calls if (call.name or "").strip())
    return names if names else ("no-tool",)


def _concept_id(meta: Mapping[str, Any]) -> int:
    domain = str(meta.get("domain") or "unknown")
    task = str(meta.get("task") or "unknown")
    difficulty = str(meta.get("difficulty") or "unknown")
    key = f"{domain}|{task}|{difficulty}"
    return _stable_int(key) % 1000003


def _events_from_unit(unit: EvaluationUnit) -> tuple[SessionEvent, ...]:
    events: list[SessionEvent] = []
    for turn in unit.turns:
        events.append(SessionEvent(role="user", text=turn.user_text))
        events.append(SessionEvent(role="assistant", text=turn.assistant_text))
    for call in unit.tool_calls:
        events.append(
            SessionEvent(
                role="tool",
                tool_name=call.name,
                arguments={"input": call.input_text} if call.input_text else None,
                output=call.output_text,
            )
        )
    return tuple(events)


@dataclass(frozen=True)
class AdaptedDataset:
    units: tuple[EvaluationUnit, ...]
    unit_ids: tuple[str, ...]
    traces: tuple[Trace, ...]
    trace_by_unit_id: dict[str, Trace]
    labels_by_unit: dict[str, bool]
    metadata_by_unit: dict[str, dict[str, Any]]


def load_adapted_dataset(data_path: str) -> AdaptedDataset:
    dataset = load_synthetic_a365_otel(data_path)
    ordered_units = sorted(
        dataset.normalization.units,
        key=lambda row: (
            row.ended_at.isoformat() if row.ended_at is not None else "",
            row.unit_id or "",
        ),
    )
    traces: list[Trace] = []
    trace_by_unit_id: dict[str, Trace] = {}
    for ordinal, unit in enumerate(ordered_units, start=1):
        unit_id = unit.unit_id or ""
        meta = dataset.metadata_by_unit.get(unit_id, {})
        trace = Trace(
            trace_id=_stable_int(unit_id),
            agent_id=f"{unit.tenant_id}|{unit.agent_id}",
            timestamp=float(ordinal),
            signature=_signature_from_unit(unit),
            span_count=len(unit.tool_calls),
            duration_ms=_duration_ms(unit.started_at, unit.ended_at),
            status="error" if unit.had_error else "ok",
            concept_id=_concept_id(meta),
            events=_events_from_unit(unit),
        )
        traces.append(trace)
        trace_by_unit_id[unit_id] = trace

    unit_ids = tuple(unit.unit_id or "" for unit in ordered_units)
    return AdaptedDataset(
        units=tuple(ordered_units),
        unit_ids=unit_ids,
        traces=tuple(traces),
        trace_by_unit_id=trace_by_unit_id,
        labels_by_unit=dict(dataset.labels_by_unit),
        metadata_by_unit=dict(dataset.metadata_by_unit),
    )


class DeterministicTokenizer:
    name = "deterministic-whitespace"
    version = "v1"

    def count(self, text: str) -> int:
        if not text:
            return 1
        return max(1, len(text.split()))


class DeterministicSessionEmbedder:
    """Offline deterministic embedder over canonical session payload text."""

    def __init__(self, dim: int = 64, seed: int = 13):
        self.dim = dim
        self.seed = seed

    def _vec_for_token(self, token: str) -> np.ndarray:
        digest = hashlib.sha256(f"{self.seed}|{token}".encode("utf-8")).digest()
        values = []
        for index in range(self.dim):
            byte = digest[index % len(digest)]
            values.append((byte / 255.0) * 2.0 - 1.0)
        arr = np.asarray(values, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr
        return arr / norm

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [tok for tok in text.split() if tok]
            if not tokens:
                tokens = ["empty"]
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in tokens:
                vec += self._vec_for_token(token)
            norm = np.linalg.norm(vec)
            matrix[row] = vec if norm == 0 else (vec / norm)
        return matrix


def build_offline_embedding_index(
    *,
    embedder: Any | None = None,
    store: VectorStore | None = None,
    seed: int = 13,
) -> tuple[AzureClusterIndex, str]:
    chosen_embedder = embedder or DeterministicSessionEmbedder(seed=seed)
    chosen_store = store or InMemoryVectorStore()
    tokenizer = DeterministicTokenizer()
    profile = EmbeddingProfile(
        model_id="offline-deterministic",
        model_version="offline-deterministic-v1",
        tokenizer_id=tokenizer.name,
        tokenizer_version=tokenizer.version,
        max_input_tokens=1024,
    )
    packet_builder = SessionEvidencePacketBuilder(
        CanonicalizationOptions(),
        max_size=4096,
    )
    cache = SessionEmbeddingCache(
        chosen_embedder,
        tokenizer,
        profile,
        max_size=4096,
        packet_builder=packet_builder,
    )
    index = AzureClusterIndex(cache, chosen_store, tau=0.75, ttl=90.0, embed_budget_per_tick=1000)
    return index, "offline"


def build_minhash_index(seed: int = 13) -> MinHashClusterIndex:
    return MinHashClusterIndex(
        MinHashConfig(
            seed=seed,
            ngram_size=3,
            permutations=128,
            similarity_threshold=0.55,
            ttl_s=90.0,
            max_clusters_per_agent=256,
        )
    )


def build_velocity_variety_stress_stream(count: int = 180, seed: int = 13) -> tuple[Trace, ...]:
    """Deterministic stress stream with bursty velocity, paraphrases, and repeated concepts."""
    rng = np.random.default_rng(seed)
    agents = (
        ("tenantA|agent-fast", 0.70),
        ("tenantA|agent-slow", 0.20),
        ("tenantB|agent-bursty", 0.10),
    )
    paraphrases = [
        ("search", "read", "edit"),
        ("query", "inspect", "modify"),
        ("lookup", "scan", "update"),
    ]
    traces: list[Trace] = []
    timestamp = 0.0
    for index in range(count):
        timestamp += 1.0
        if 60 <= index <= 95:
            agent = "tenantB|agent-bursty"
        else:
            draw = rng.random()
            cumulative = 0.0
            agent = agents[0][0]
            for candidate, weight in agents:
                cumulative += weight
                if draw <= cumulative:
                    agent = candidate
                    break
        phrase = paraphrases[index % len(paraphrases)]
        concept = index % 9
        traces.append(
            Trace(
                trace_id=_stable_int(f"stress-{index}"),
                agent_id=agent,
                timestamp=timestamp,
                signature=phrase,
                span_count=len(phrase),
                duration_ms=300.0 + float((index % 11) * 50),
                status="ok",
                concept_id=concept,
                events=(
                    SessionEvent(role="user", text=f"goal {concept}"),
                    SessionEvent(role="assistant", text=f"response {concept}"),
                    SessionEvent(role="tool", tool_name=phrase[0], output=f"result {concept}"),
                ),
            )
        )
    return tuple(traces)
