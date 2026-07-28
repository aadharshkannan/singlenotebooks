"""Async evaluation runner for sampled units and pluggable judges."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
from typing import Protocol

from .evidence import build_evidence_packet
from .azure_openai_judge import (
    JudgeAuthenticationError,
    JudgeContentFilteredError,
    JudgeEmptyResponseError,
    JudgeLengthExhaustedError,
    JudgeMalformedJsonError,
    JudgeProviderError,
    JudgeSkippedError,
    JudgeTerminalHttpError,
)
from .judge import (
    AsyncJudge,
    JudgeDescriptor,
    JudgeRequest,
    JudgeResponse,
    JudgeResponseError,
    JudgeTransientError,
)
from .metrics import MetricObservation, MetricSpec
from .models import AgentKey, SampleBatch, SampledUnit


@dataclass(frozen=True)
class RunnerConfig:
    max_concurrency: int = 8
    timeout_seconds: float = 30.0
    max_attempts: int = 3
    base_backoff_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")
        if self.base_backoff_seconds < 0:
            raise ValueError("base_backoff_seconds must be >= 0")


@dataclass(frozen=True)
class EvaluationFailure:
    request_id: str
    tenant_id: str
    agent_id: str
    unit_id: str
    session_id: str | None
    metric_id: str
    metric_version: str
    sample_kind: str
    code: str
    attempts: int
    retryable: bool
    message: str


@dataclass(frozen=True)
class EvaluationRun:
    status: str
    observations: tuple[MetricObservation, ...]
    failures: tuple[EvaluationFailure, ...]
    selected_count: int
    request_count: int
    judge_descriptor: JudgeDescriptor
    judge_prompt_schema_fingerprint: str | None = None
    metrics: tuple[MetricSpec, ...] = ()


class OutcomeSink(Protocol):
    async def upsert(self, observation: MetricObservation) -> None:
        ...


@dataclass
class InMemoryOutcomeSink:
    """Idempotent request_id keyed in-memory sink for runner outputs."""

    _rows: dict[str, MetricObservation] | None = None

    def __post_init__(self) -> None:
        if self._rows is None:
            self._rows = {}

    async def upsert(self, observation: MetricObservation) -> None:
        assert self._rows is not None
        self._rows[observation.request_id] = observation

    def values(self) -> tuple[MetricObservation, ...]:
        assert self._rows is not None
        return tuple(self._rows[key] for key in sorted(self._rows))


@dataclass(frozen=True)
class _EvalTask:
    index: int
    sampled: SampledUnit
    metric: MetricSpec


def _idempotency_key(
    batch: SampleBatch,
    sampled: SampledUnit,
    metric: MetricSpec,
    evidence_sha256: str,
    judge_descriptor: JudgeDescriptor,
) -> str:
    descriptor = "/".join(
        (judge_descriptor.provider, judge_descriptor.name, judge_descriptor.version)
    )
    material = "||".join(
        [
            batch.run_id,
            batch.policy.version,
            sampled.unit.tenant_id,
            sampled.unit.agent_id,
            sampled.unit.unit_id or "",
            sampled.sample_kind,
            metric.id,
            metric.version,
            evidence_sha256,
            descriptor,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _request_id(idempotency_key: str) -> str:
    return idempotency_key[:24]


def _sort_key_observation(observation: MetricObservation) -> tuple[str, str, str, str, str]:
    return (
        observation.tenant_id,
        observation.agent_id,
        observation.unit_id,
        observation.metric.id,
        observation.sample_kind,
    )


def _sort_key_failure(failure: EvaluationFailure) -> tuple[str, str, str, str, str]:
    return (
        failure.tenant_id,
        failure.agent_id,
        failure.unit_id,
        failure.metric_id,
        failure.sample_kind,
    )


class EvaluationRunner:
    def __init__(
        self,
        judge: AsyncJudge,
        config: RunnerConfig | None = None,
        outcome_sink: OutcomeSink | None = None,
        max_evidence_bytes: int = 32768,
    ) -> None:
        self._judge = judge
        self._config = config or RunnerConfig()
        self._sink = outcome_sink
        self._max_evidence_bytes = max_evidence_bytes

    async def run(
        self,
        batch: SampleBatch,
        metrics: tuple[MetricSpec, ...],
        unit_ids: set[str] | frozenset[str] | None = None,
    ) -> EvaluationRun:
        semaphore = asyncio.Semaphore(self._config.max_concurrency)
        selected_filter: frozenset[str] | None = None
        if unit_ids is not None:
            selected_filter = frozenset(unit_ids)
            if not selected_filter:
                selected_filter = frozenset()

            known_ids = {
                sampled.unit.unit_id
                for sampled in batch.all_units()
                if sampled.unit.unit_id is not None
            }
            unknown = sorted(selected_filter - known_ids)
            if unknown:
                unknown_csv = ", ".join(unknown)
                raise ValueError(f"Unknown unit_ids for batch {batch.run_id}: {unknown_csv}")

        tasks: list[_EvalTask] = []
        selected_units_attempted: set[str] = set()
        for sampled in batch.all_units():
            unit = sampled.unit
            if unit.unit_id is None:
                continue
            if selected_filter is not None and unit.unit_id not in selected_filter:
                continue
            selected_units_attempted.add(unit.unit_id)
            for metric in metrics:
                tasks.append(_EvalTask(index=len(tasks), sampled=sampled, metric=metric))

        observations: list[MetricObservation] = []
        failures: list[EvaluationFailure] = []

        async def worker(task: _EvalTask) -> None:
            sampled = task.sampled
            unit = sampled.unit
            assert unit.unit_id is not None
            try:
                evidence = build_evidence_packet(unit, max_bytes=self._max_evidence_bytes)
            except ValueError as exc:
                failures.append(
                    EvaluationFailure(
                        request_id=f"evidence-{task.index}",
                        tenant_id=unit.tenant_id,
                        agent_id=unit.agent_id,
                        unit_id=unit.unit_id,
                        session_id=unit.session_id,
                        metric_id=task.metric.id,
                        metric_version=task.metric.version,
                        sample_kind=sampled.sample_kind,
                        code="evidence_too_large",
                        attempts=0,
                        retryable=False,
                        message=str(exc),
                    )
                )
                return

            idempotency_key = _idempotency_key(
                batch,
                sampled,
                task.metric,
                evidence.sha256,
                self._judge.descriptor,
            )
            request_id = _request_id(idempotency_key)
            request = JudgeRequest(
                request_id=request_id,
                idempotency_key=idempotency_key,
                tenant_id=unit.tenant_id,
                agent_id=unit.agent_id,
                unit_id=unit.unit_id,
                session_id=unit.session_id,
                conversation_ids=unit.conversation_ids,
                metric=task.metric,
                evidence=evidence,
            )

            attempts = 0
            while attempts < self._config.max_attempts:
                attempts += 1
                try:
                    async with semaphore:
                        response = await asyncio.wait_for(
                            self._judge.evaluate(request),
                            timeout=self._config.timeout_seconds,
                        )
                    observation = self._observation_from_response(sampled, request, response)
                    if self._sink is not None:
                        await self._sink.upsert(observation)
                    observations.append(observation)
                    return
                except asyncio.TimeoutError:
                    if attempts >= self._config.max_attempts:
                        failures.append(
                            EvaluationFailure(
                                request_id=request_id,
                                tenant_id=unit.tenant_id,
                                agent_id=unit.agent_id,
                                unit_id=unit.unit_id,
                                session_id=unit.session_id,
                                metric_id=task.metric.id,
                                metric_version=task.metric.version,
                                sample_kind=sampled.sample_kind,
                                code="judge_timeout",
                                attempts=attempts,
                                retryable=True,
                                message="Judge request timed out",
                            )
                        )
                        return
                except JudgeTransientError as exc:
                    if attempts >= self._config.max_attempts:
                        failures.append(
                            EvaluationFailure(
                                request_id=request_id,
                                tenant_id=unit.tenant_id,
                                agent_id=unit.agent_id,
                                unit_id=unit.unit_id,
                                session_id=unit.session_id,
                                metric_id=task.metric.id,
                                metric_version=task.metric.version,
                                sample_kind=sampled.sample_kind,
                                code="judge_transient_error",
                                attempts=attempts,
                                retryable=True,
                                message=str(exc),
                            )
                        )
                        return
                except JudgeResponseError as exc:
                    code = "judge_malformed_response"
                    if isinstance(exc, JudgeLengthExhaustedError):
                        code = "response_length_exhausted"
                    elif isinstance(exc, JudgeEmptyResponseError):
                        code = "response_empty"
                    elif isinstance(exc, JudgeContentFilteredError):
                        code = "response_content_filtered"
                    elif isinstance(exc, JudgeMalformedJsonError):
                        code = "response_malformed_json"
                    elif isinstance(exc, JudgeSkippedError):
                        code = "response_skipped"
                    elif isinstance(exc, JudgeAuthenticationError):
                        code = "provider_authentication"
                    elif isinstance(exc, JudgeTerminalHttpError):
                        code = f"provider_http_{exc.status_code}"
                    elif isinstance(exc, JudgeProviderError):
                        code = "provider_terminal"
                    failures.append(
                        EvaluationFailure(
                            request_id=request_id,
                            tenant_id=unit.tenant_id,
                            agent_id=unit.agent_id,
                            unit_id=unit.unit_id,
                            session_id=unit.session_id,
                            metric_id=task.metric.id,
                            metric_version=task.metric.version,
                            sample_kind=sampled.sample_kind,
                            code=code,
                            attempts=attempts,
                            retryable=False,
                            message=str(exc),
                        )
                    )
                    return
                except Exception as exc:  # Provider boundary: preserve the rest of the batch.
                    failures.append(
                        EvaluationFailure(
                            request_id=request_id,
                            tenant_id=unit.tenant_id,
                            agent_id=unit.agent_id,
                            unit_id=unit.unit_id,
                            session_id=unit.session_id,
                            metric_id=task.metric.id,
                            metric_version=task.metric.version,
                            sample_kind=sampled.sample_kind,
                            code="judge_unexpected_error",
                            attempts=attempts,
                            retryable=False,
                            message=f"Unexpected judge error: {type(exc).__name__}",
                        )
                    )
                    return

                if attempts < self._config.max_attempts:
                    delay = min(
                        self._config.base_backoff_seconds * (2 ** (attempts - 1)),
                        self._config.timeout_seconds,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)

        await asyncio.gather(*(worker(task) for task in tasks))

        sorted_observations = tuple(sorted(observations, key=_sort_key_observation))
        sorted_failures = tuple(sorted(failures, key=_sort_key_failure))
        status = "completed"
        if not sorted_observations and sorted_failures:
            status = "failed"
        elif sorted_failures:
            status = "partial"

        return EvaluationRun(
            status=status,
            observations=sorted_observations,
            failures=sorted_failures,
            selected_count=len(selected_units_attempted),
            request_count=len(tasks),
            judge_descriptor=self._judge.descriptor,
            judge_prompt_schema_fingerprint=getattr(self._judge, "prompt_schema_fingerprint", None),
            metrics=metrics,
        )

    def _observation_from_response(
        self,
        sampled: SampledUnit,
        request: JudgeRequest,
        response: JudgeResponse,
    ) -> MetricObservation:
        if response.request_id != request.request_id:
            raise JudgeResponseError("Judge response request_id mismatch")
        if response.metric.id != request.metric.id or response.metric.version != request.metric.version:
            raise JudgeResponseError("Judge response metric mismatch")
        if response.value.kind != request.metric.kind:
            raise JudgeResponseError("Judge response value kind mismatch")

        unit = sampled.unit
        return MetricObservation(
            request_id=request.request_id,
            agent=AgentKey(tenant_id=unit.tenant_id, agent_id=unit.agent_id),
            tenant_id=unit.tenant_id,
            agent_id=unit.agent_id,
            unit_id=unit.unit_id,
            session_id=unit.session_id,
            conversation_ids=unit.conversation_ids,
            metric=request.metric,
            value=response.value,
            sample_kind=sampled.sample_kind,
            estimand_eligible=sampled.estimand_eligible,
            judge=self._judge.descriptor,
            evidence_sha256=request.evidence.sha256,
            reasoning=response.reasoning,
        )
