"""Metric contracts and immutable value/observation models for alt evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Literal

from .models import AgentKey

if TYPE_CHECKING:
    from .judge import JudgeDescriptor


MetricKind = Literal["binary", "likert", "scalar", "categorical"]


def _clean_text(value: str) -> str:
    return value.strip()


def sanitize_reasoning(reasoning: str | None, max_chars: int = 512) -> str | None:
    if reasoning is None:
        return None
    text = reasoning.strip()
    if not text:
        return None
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
    text = re.sub(r"(?i)(user[_ ]?id\s*[:=]\s*)([^\s,;]+)", r"\1[redacted]", text)
    if len(text) > max_chars:
        return text[: max_chars - 14].rstrip() + " [truncated]"
    return text


@dataclass(frozen=True)
class MetricSpec:
    """Versioned metric definition and optional display metadata."""

    id: str
    name: str
    version: str
    kind: MetricKind
    display_name: str
    likert_min: int | None = None
    likert_max: int | None = None
    categories: tuple[str, ...] | None = None
    pass_threshold: float | None = None

    def __post_init__(self) -> None:
        if not _clean_text(self.id):
            raise ValueError("MetricSpec.id must not be blank")
        if not _clean_text(self.name):
            raise ValueError("MetricSpec.name must not be blank")
        if not _clean_text(self.version):
            raise ValueError("MetricSpec.version must not be blank")
        if self.kind not in {"binary", "likert", "scalar", "categorical"}:
            raise ValueError(f"Unsupported metric kind: {self.kind}")
        if not _clean_text(self.display_name):
            raise ValueError("MetricSpec.display_name must not be blank")

        if self.kind == "likert":
            if self.likert_min is None or self.likert_max is None:
                raise ValueError("Likert metrics require likert_min and likert_max")
            if self.likert_min >= self.likert_max:
                raise ValueError("Likert bounds must satisfy likert_min < likert_max")
        else:
            if self.likert_min is not None or self.likert_max is not None:
                raise ValueError("Likert bounds may only be set on likert metrics")

        if self.kind == "categorical":
            if not self.categories:
                raise ValueError("Categorical metrics require non-empty categories")
            cleaned = tuple(cat.strip() for cat in self.categories if cat and cat.strip())
            if not cleaned or len(set(cleaned)) != len(cleaned):
                raise ValueError("Categorical categories must be unique non-empty strings")
            object.__setattr__(self, "categories", cleaned)
        elif self.categories is not None:
            raise ValueError("categories may only be set on categorical metrics")

        if self.pass_threshold is not None and self.kind not in {"binary", "likert", "scalar"}:
            raise ValueError("pass_threshold is only valid for binary/likert/scalar metrics")


@dataclass(frozen=True)
class BinaryValue:
    passed: bool
    kind: Literal["binary"] = "binary"


@dataclass(frozen=True)
class LikertValue:
    score: int
    min_score: int
    max_score: int
    kind: Literal["likert"] = "likert"

    def __post_init__(self) -> None:
        if self.min_score >= self.max_score:
            raise ValueError("LikertValue requires min_score < max_score")
        if self.score < self.min_score or self.score > self.max_score:
            raise ValueError("LikertValue.score must be inside [min_score, max_score]")


@dataclass(frozen=True)
class ScalarValue:
    value: float
    kind: Literal["scalar"] = "scalar"


@dataclass(frozen=True)
class CategoricalValue:
    category: str
    kind: Literal["categorical"] = "categorical"

    def __post_init__(self) -> None:
        if not self.category.strip():
            raise ValueError("CategoricalValue.category must not be blank")


MetricValue = BinaryValue | LikertValue | ScalarValue | CategoricalValue


@dataclass(frozen=True)
class MetricObservation:
    """Single metric value for one sampled session-level unit."""

    request_id: str
    agent: AgentKey
    tenant_id: str
    agent_id: str
    unit_id: str
    session_id: str | None
    conversation_ids: tuple[str, ...]
    metric: MetricSpec
    value: MetricValue
    sample_kind: Literal["core", "diversity"]
    estimand_eligible: bool
    judge: "JudgeDescriptor"
    evidence_sha256: str
    reasoning: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be blank")
        if not self.tenant_id.strip() or not self.agent_id.strip() or not self.unit_id.strip():
            raise ValueError("tenant_id, agent_id, and unit_id must not be blank")
        if self.sample_kind not in {"core", "diversity"}:
            raise ValueError(f"Unsupported sample_kind: {self.sample_kind}")
        if self.value.kind != self.metric.kind:
            raise ValueError(
                f"Value kind {self.value.kind} does not match metric kind {self.metric.kind}"
            )
        cleaned = tuple(sorted({value.strip() for value in self.conversation_ids if value and value.strip()}))
        object.__setattr__(self, "conversation_ids", cleaned)
        object.__setattr__(self, "reasoning", sanitize_reasoning(self.reasoning))


TASK_COMPLETION_V1 = MetricSpec(
    id="task_completion",
    name="TASK_COMPLETION_V1",
    version="v1",
    kind="binary",
    display_name="Task Completion",
    pass_threshold=0.5,
)
