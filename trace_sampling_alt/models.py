"""Core immutable models for the alternative Agent 365 sampling pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from typing import Any, Literal, Mapping


@dataclass(frozen=True, order=True)
class AgentKey:
    """Tenant-scoped Agent 365 identity."""

    tenant_id: str
    agent_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must not be blank")
        if not self.agent_id.strip():
            raise ValueError("agent_id must not be blank")


@dataclass(frozen=True)
class SamplePolicy:
    """Versioned policy for the probability and diversity samples."""

    margin: float = 0.10
    confidence: float = 0.95
    diversity_enabled: bool = False
    diversity_fraction: float = 0.20
    minhash_ngram_size: int = 3
    minhash_permutations: int = 128
    seed: int = 13
    version: str = "alt-session-stratified-v2"

    def __post_init__(self) -> None:
        if not 0 < self.margin <= 1:
            raise ValueError(f"margin must be in (0, 1], got {self.margin}")
        if not 0 < self.confidence < 1:
            raise ValueError(
                f"confidence must be in (0, 1), got {self.confidence}"
            )
        if not 0 <= self.diversity_fraction < 1:
            raise ValueError(
                "diversity_fraction must be in [0, 1), "
                f"got {self.diversity_fraction}"
            )
        if self.minhash_ngram_size <= 0:
            raise ValueError("minhash_ngram_size must be positive")
        if self.minhash_permutations <= 0:
            raise ValueError("minhash_permutations must be positive")
        if not self.version.strip():
            raise ValueError("version must not be blank")


@dataclass(frozen=True)
class SessionizationPolicy:
    """How spans without a session ID are split into time-bound sessions."""

    inactivity_timeout: timedelta = timedelta(minutes=30)
    version: str = "agent365-sessionization-v1"

    def __post_init__(self) -> None:
        if self.inactivity_timeout <= timedelta(0):
            raise ValueError("inactivity_timeout must be positive")
        if not self.version.strip():
            raise ValueError("version must not be blank")


@dataclass(frozen=True)
class EvaluationWindow:
    """Half-open completed-session frame: ``[start_at, end_at)``."""

    start_at: datetime
    end_at: datetime

    def __post_init__(self) -> None:
        for name, value in (("start_at", self.start_at), ("end_at", self.end_at)):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.start_at >= self.end_at:
            raise ValueError("start_at must be before end_at")
        object.__setattr__(self, "start_at", self.start_at.astimezone(timezone.utc))
        object.__setattr__(self, "end_at", self.end_at.astimezone(timezone.utc))

    @classmethod
    def ending_at(
        cls,
        end_at: datetime,
        duration: timedelta = timedelta(hours=24),
    ) -> "EvaluationWindow":
        if duration <= timedelta(0):
            raise ValueError("duration must be positive")
        return cls(start_at=end_at - duration, end_at=end_at)

    @property
    def duration(self) -> timedelta:
        return self.end_at - self.start_at

    def contains_completion(self, completed_at: datetime) -> bool:
        if completed_at.tzinfo is None or completed_at.utcoffset() is None:
            raise ValueError("completed_at must be timezone-aware")
        completed_utc = completed_at.astimezone(timezone.utc)
        return self.start_at <= completed_utc < self.end_at


@dataclass(frozen=True)
class SamplePlan:
    population: int
    recommended: int
    selected: int
    capacity: int | None
    census: bool
    precision_status: str
    effective_rate: float
    probability_selected: int = 0
    diversity_selected: int = 0


@dataclass(frozen=True)
class StratumPlan:
    key: str
    population: int
    selected: int

    @property
    def inclusion_probability(self) -> float:
        if self.population == 0:
            return 0.0
        return self.selected / self.population

    @property
    def sampling_weight(self) -> float | None:
        probability = self.inclusion_probability
        return 1.0 / probability if probability > 0 else None


@dataclass(frozen=True)
class TenantCapacityPlan:
    tenant_id: str
    granted: int
    statistical_recommended: int
    selected: int
    unused: int
    precision_status: str


class FrozenMapping(Mapping[str, Any]):
    """Small hashable mapping for deeply immutable metadata payloads."""

    def __init__(self, values: Mapping[str, Any]):
        self._values = {key: _freeze(value) for key, value in values.items()}

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._values.items())))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class Turn:
    user_text: str
    assistant_text: str


@dataclass(frozen=True)
class ToolCall:
    name: str | None
    input_text: str | None = None
    output_text: str | None = None
    status: str | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.details is not None:
            object.__setattr__(self, "details", FrozenMapping(self.details))


@dataclass(frozen=True)
class EvaluationUnit:
    """One completed Agent 365 session, the sampling and judging unit."""

    tenant_id: str
    agent_id: str
    conversation_id: str
    session_id: str | None
    channel: str | None
    source_trace_ids: tuple[str, ...]
    started_at: datetime | None
    ended_at: datetime | None
    had_error: bool
    turns: tuple[Turn, ...]
    tool_calls: tuple[ToolCall, ...]
    conversation_ids: tuple[str, ...] = ()
    sessionization_kind: Literal["session_id", "inactivity"] | None = None
    unit_id: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip() or not self.agent_id.strip():
            raise ValueError("tenant_id and agent_id must not be blank")
        conversation_ids = tuple(
            sorted(
                {
                    value.strip()
                    for value in self.conversation_ids + ((self.conversation_id,) if self.conversation_id else ())
                    if value and value.strip()
                }
            )
        )
        object.__setattr__(self, "conversation_ids", conversation_ids)

        kind = self.sessionization_kind or (
            "session_id" if self.session_id else "inactivity"
        )
        object.__setattr__(self, "sessionization_kind", kind)

        if self.unit_id is None:
            if self.session_id:
                material = "||".join(
                    (self.tenant_id, self.agent_id, self.session_id)
                )
                digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
                unit_id = f"session:{digest}"
            else:
                material = "||".join(
                    (
                        self.tenant_id,
                        self.agent_id,
                        ",".join(conversation_ids),
                        self.started_at.isoformat() if self.started_at else "",
                        self.ended_at.isoformat() if self.ended_at else "",
                    )
                )
                digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
                unit_id = f"inactivity:{digest}"
            object.__setattr__(self, "unit_id", unit_id)
        elif not self.unit_id.strip():
            raise ValueError("unit_id must not be blank")

    @property
    def is_judgeable(self) -> bool:
        has_user = any(turn.user_text.strip() for turn in self.turns)
        final_assistant = self.turns[-1].assistant_text.strip() if self.turns else ""
        return has_user and bool(final_assistant)


@dataclass(frozen=True)
class SampledUnit:
    unit: EvaluationUnit
    sample_kind: Literal["core", "diversity"]
    estimand_eligible: bool
    stratum_key: str
    inclusion_probability: float | None
    sampling_weight: float | None
    selection_reason: str


@dataclass(frozen=True)
class AgentSample:
    agent: AgentKey
    plan: SamplePlan
    strata: tuple[StratumPlan, ...]
    core: tuple[SampledUnit, ...]
    diversity: tuple[SampledUnit, ...]

    def all_units(self) -> tuple[SampledUnit, ...]:
        return self.core + self.diversity

    @property
    def diagnostic(self) -> tuple[SampledUnit, ...]:
        """Back-compat alias; ownership moved to diversity."""
        return self.diversity


@dataclass(frozen=True)
class SampleBatch:
    policy: SamplePolicy
    version: str
    run_id: str
    agents: tuple[AgentSample, ...]
    tenant_capacities: tuple[TenantCapacityPlan, ...] = ()
    evaluation_window: EvaluationWindow | None = None

    def all_units(self) -> tuple[SampledUnit, ...]:
        out: list[SampledUnit] = []
        for agent_sample in self.agents:
            out.extend(agent_sample.all_units())
        return tuple(out)


@dataclass(frozen=True)
class IngestIssue:
    code: str
    message: str
    source_kind: str
    source_ref: str | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("code must not be blank")
        if not self.message.strip():
            raise ValueError("message must not be blank")
        if not self.source_kind.strip():
            raise ValueError("source_kind must not be blank")
        if self.details is not None:
            object.__setattr__(self, "details", FrozenMapping(self.details))