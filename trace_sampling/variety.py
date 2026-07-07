"""Swappable variety abstraction for adaptive trace sampling.

VarietyIndex scores behavioral variety (rarity + novelty) for a trace. The baseline
ExactSignatureIndex uses exact tool-signature matching and preserves the current
sampler's post-increment rarity semantics; a future AzureClusterIndex will implement
the same protocol using embedding clusters.
"""
from dataclasses import dataclass
from typing import Any, Protocol, Tuple

from .model import Trace
from .stats import AgentStats


@dataclass(frozen=True)
class VarietyKey:
    kind: str   # "signature" | "cluster" | "fallback-signature"
    value: Any  # tuple[str,...] for signatures, str for cluster ids


@dataclass(frozen=True)
class VarietyObservation:
    key: VarietyKey
    rarity: float    # in [0,1]
    novelty: float   # in [0,1]


class VarietyIndex(Protocol):
    def observe(self, trace: Trace) -> VarietyObservation: ...


class ExactSignatureIndex:
    """Baseline: exact tool-signature stratification. Preserves current sampler
    scoring exactly — rarity is post-increment (first-seen == 0.5)."""

    def __init__(self, max_signatures_per_agent: int = 256):
        self.max_signatures_per_agent = max_signatures_per_agent
        self._stats = {}

    def _stats_for(self, agent_id: str) -> AgentStats:
        if agent_id not in self._stats:
            self._stats[agent_id] = AgentStats(max_signatures=self.max_signatures_per_agent)
        return self._stats[agent_id]

    def observe(self, trace: Trace) -> VarietyObservation:
        stats = self._stats_for(trace.agent_id)
        seen_before = stats.has_seen(trace.signature)  # pre-observe state
        stats.observe(trace.timestamp, trace.signature)  # increments first (as today)
        rarity = stats.rarity(trace.signature)           # post-increment: first-seen=0.5
        novelty = 0.0 if seen_before else 1.0
        return VarietyObservation(
            key=VarietyKey("signature", trace.signature),
            rarity=rarity, novelty=novelty,
        )
