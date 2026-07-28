"""Judge boundary contracts and deterministic local stub implementation."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Protocol

from .evidence import EvidencePacket
from .metrics import (
    BinaryValue,
    CategoricalValue,
    LikertValue,
    MetricSpec,
    MetricValue,
    ScalarValue,
)


class JudgeTransientError(RuntimeError):
    """Retryable transport or service-side transient issue."""


class JudgeResponseError(ValueError):
    """Terminal malformed/invalid judge response."""


@dataclass(frozen=True)
class JudgeDescriptor:
    provider: str
    name: str
    version: str

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.name.strip() or not self.version.strip():
            raise ValueError("JudgeDescriptor fields must not be blank")


@dataclass(frozen=True)
class JudgeRequest:
    request_id: str
    idempotency_key: str
    tenant_id: str
    agent_id: str
    unit_id: str
    session_id: str | None
    conversation_ids: tuple[str, ...]
    metric: MetricSpec
    evidence: EvidencePacket

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.idempotency_key.strip():
            raise ValueError("request_id and idempotency_key must not be blank")
        if not self.tenant_id.strip() or not self.agent_id.strip() or not self.unit_id.strip():
            raise ValueError("scoped identity must not be blank")
        cleaned = tuple(sorted({value.strip() for value in self.conversation_ids if value and value.strip()}))
        object.__setattr__(self, "conversation_ids", cleaned)


@dataclass(frozen=True)
class JudgeResponse:
    request_id: str
    metric: MetricSpec
    value: MetricValue
    reasoning: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be blank")
        if self.metric.kind != self.value.kind:
            raise JudgeResponseError(
                f"Judge response value kind {self.value.kind} does not match metric kind {self.metric.kind}"
            )


class AsyncJudge(Protocol):
    descriptor: JudgeDescriptor

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        ...


class SyncJudge(Protocol):
    descriptor: JudgeDescriptor

    def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        ...


@dataclass(frozen=True)
class SyncJudgeAdapter:
    """Bridge sync judges into async runner calls with to_thread."""

    sync_judge: SyncJudge

    @property
    def descriptor(self) -> JudgeDescriptor:
        return self.sync_judge.descriptor

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        return await asyncio.to_thread(self.sync_judge.evaluate, request)


ResultOverride = MetricValue | Callable[[JudgeRequest], MetricValue]


@dataclass(frozen=True)
class DeterministicJudgeStub:
    """Non-production deterministic judge for local testing and smoke runs.

    Overrides are keyed by ``unit_id``.
    """

    overrides: dict[str, ResultOverride] | None = None
    descriptor: JudgeDescriptor = JudgeDescriptor(
        provider="stub",
        name="deterministic-local",
        version="v1",
    )

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        if self.overrides and request.unit_id in self.overrides:
            override = self.overrides[request.unit_id]
            value = override(request) if callable(override) else override
        else:
            value = self._deterministic_value(request)

        response = JudgeResponse(
            request_id=request.request_id,
            metric=request.metric,
            value=value,
            reasoning="stub_result",
        )
        return response

    def _deterministic_value(self, request: JudgeRequest) -> MetricValue:
        digest = hashlib.sha256(
            f"{request.unit_id}|{request.metric.id}|{request.metric.version}|{request.evidence.sha256}".encode("utf-8")
        ).digest()

        if request.metric.kind == "binary":
            return BinaryValue(passed=(digest[0] % 2 == 0))
        if request.metric.kind == "likert":
            low = request.metric.likert_min if request.metric.likert_min is not None else 1
            high = request.metric.likert_max if request.metric.likert_max is not None else 5
            score = low + (digest[0] % (high - low + 1))
            return LikertValue(score=score, min_score=low, max_score=high)
        if request.metric.kind == "scalar":
            raw = int.from_bytes(digest[:8], "big", signed=False)
            return ScalarValue(value=(raw % 10001) / 100.0)
        if request.metric.kind == "categorical":
            categories = request.metric.categories or ("unknown",)
            return CategoricalValue(category=categories[digest[0] % len(categories)])
        raise JudgeResponseError(f"Unsupported metric kind: {request.metric.kind}")
