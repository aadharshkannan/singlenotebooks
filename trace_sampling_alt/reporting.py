"""Deterministic report assembly from sampled units and evaluation outputs."""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist

from .evaluation import EvaluationRun
from .metrics import (
    BinaryValue,
    CategoricalValue,
    LikertValue,
    MetricObservation,
    ScalarValue,
)
from .models import AgentKey, SampleBatch


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float


@dataclass(frozen=True)
class BinarySummary:
    selected_count: int
    submitted_count: int
    succeeded_count: int
    failed_count: int
    passes: int
    failures: int
    response_rate: float
    pass_rate: float | None
    population_eligible_count: int | None
    wilson_interval: Interval | None
    estimand_eligible: bool
    warning: str | None = None


@dataclass(frozen=True)
class LikertSummary:
    count: int
    distribution: tuple[tuple[int, int], ...]
    mean: float | None


@dataclass(frozen=True)
class ScalarSummary:
    count: int
    mean: float | None
    min_value: float | None
    max_value: float | None


@dataclass(frozen=True)
class CategoricalSummary:
    count: int
    counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class MetricSampleSummary:
    agent: AgentKey
    metric_id: str
    metric_version: str
    metric_kind: str
    sample_kind: str
    selected_count: int
    submitted_count: int
    succeeded_count: int
    failed_count: int
    response_rate: float
    binary: BinarySummary | None = None
    likert: LikertSummary | None = None
    scalar: ScalarSummary | None = None
    categorical: CategoricalSummary | None = None


@dataclass(frozen=True)
class EvaluationReport:
    version: str
    status: str
    summaries: tuple[MetricSampleSummary, ...]
    warnings: tuple[str, ...]
    ingest_issue_count: int
    notes: tuple[str, ...] = ()


def _wilson_interval(
    successes: int,
    n: int,
    confidence: float,
    population: int,
) -> Interval:
    if n <= 0:
        return Interval(lower=0.0, upper=1.0)
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1 - p) / n) + (z2 / (4 * n * n)))
    lower = center - half
    upper = center + half

    if population > 1 and n < population:
        fpc = math.sqrt((population - n) / (population - 1))
        lower = p - (p - lower) * fpc
        upper = p + (upper - p) * fpc

    return Interval(lower=max(0.0, lower), upper=min(1.0, upper))


def _obs_for(
    observations: tuple[MetricObservation, ...],
    agent: AgentKey,
    metric_id: str,
    metric_version: str,
    sample_kind: str,
) -> tuple[MetricObservation, ...]:
    return tuple(
        obs
        for obs in observations
        if obs.agent == agent
        and obs.metric.id == metric_id
        and obs.metric.version == metric_version
        and obs.sample_kind == sample_kind
    )


def build_report(
    batch: SampleBatch,
    run: EvaluationRun,
    ingest_issue_count: int = 0,
) -> EvaluationReport:
    summaries: list[MetricSampleSummary] = []
    warnings: list[str] = []
    notes: list[str] = []

    failures = run.failures
    observations = run.observations

    for agent_sample in batch.agents:
        agent = agent_sample.agent
        grouped_keys: set[tuple[str, str, str]] = set()
        metric_specs = {
            (metric.id, metric.version): metric for metric in run.metrics
        }
        for obs in observations:
            if obs.agent == agent:
                grouped_keys.add((obs.metric.id, obs.metric.version, obs.sample_kind))
                metric_specs[(obs.metric.id, obs.metric.version)] = obs.metric
        for failure in failures:
            if AgentKey(tenant_id=failure.tenant_id, agent_id=failure.agent_id) == agent:
                grouped_keys.add((failure.metric_id, failure.metric_version, failure.sample_kind))
        for metric in run.metrics:
            if agent_sample.core:
                grouped_keys.add((metric.id, metric.version, "core"))
            if agent_sample.diversity:
                grouped_keys.add((metric.id, metric.version, "diversity"))

        if "diversity_reserved_precision_shortfall" in agent_sample.plan.precision_status:
            notes.append(
                f"{agent.tenant_id}/{agent.agent_id} reserved diversity capacity reduced the statistical core allocation"
            )

        for metric_id, metric_version, sample_kind in sorted(grouped_keys):
            subset = _obs_for(observations, agent, metric_id, metric_version, sample_kind)
            matching_failures = [
                failure
                for failure in failures
                if failure.tenant_id == agent.tenant_id
                and failure.agent_id == agent.agent_id
                and failure.metric_id == metric_id
                and failure.metric_version == metric_version
                and failure.sample_kind == sample_kind
            ]
            metric = subset[0].metric if subset else metric_specs.get((metric_id, metric_version))
            if metric is None:
                warnings.append(
                    f"{agent.tenant_id}/{agent.agent_id}:{metric_id}:{metric_version} has no metric specification"
                )
                continue
            selected = len(agent_sample.core) if sample_kind == "core" else len(agent_sample.diversity)
            failed = len(matching_failures)
            succeeded = len(subset)
            submitted = succeeded + failed
            response_rate = (succeeded / selected) if selected else 0.0
            if metric.kind == "binary":
                passes = sum(1 for obs in subset if isinstance(obs.value, BinaryValue) and obs.value.passed)
                outcome_failures = succeeded - passes
                pass_rate = (passes / succeeded) if succeeded else None
                interval = None
                warning = None
                population = agent_sample.plan.population if sample_kind == "core" else None
                if sample_kind == "core" and succeeded > 0 and population is not None:
                    interval = _wilson_interval(
                        successes=passes,
                        n=succeeded,
                        confidence=batch.policy.confidence,
                        population=population,
                    )
                    if failed > 0:
                        warning = "Core metric has missing judge results; headline remains based on succeeded responses only"
                        warnings.append(
                            f"{agent.tenant_id}/{agent.agent_id}:{metric_id}:{metric_version} core has {failed} missing results"
                        )
                if sample_kind == "core" and succeeded == 0:
                    warning = "Core metric has no successful judge responses"
                    warnings.append(
                        f"{agent.tenant_id}/{agent.agent_id}:{metric_id}:{metric_version} core has no successful responses"
                    )

                summaries.append(
                    MetricSampleSummary(
                        agent=agent,
                        metric_id=metric_id,
                        metric_version=metric_version,
                        metric_kind="binary",
                        sample_kind=sample_kind,
                        selected_count=selected,
                        submitted_count=submitted,
                        succeeded_count=succeeded,
                        failed_count=failed,
                        response_rate=response_rate,
                        binary=BinarySummary(
                            selected_count=selected,
                            submitted_count=submitted,
                            succeeded_count=succeeded,
                            failed_count=failed,
                            passes=passes,
                            failures=outcome_failures,
                            response_rate=response_rate,
                            pass_rate=pass_rate,
                            population_eligible_count=population if sample_kind == "core" else None,
                            wilson_interval=interval if sample_kind == "core" else None,
                            estimand_eligible=(sample_kind == "core"),
                            warning=warning,
                        ),
                    )
                )
            elif not subset:
                warnings.append(
                    f"{agent.tenant_id}/{agent.agent_id}:{metric_id}:{metric_version} {sample_kind} has no successful responses"
                )
                summaries.append(
                    MetricSampleSummary(
                        agent=agent,
                        metric_id=metric_id,
                        metric_version=metric_version,
                        metric_kind=metric.kind,
                        sample_kind=sample_kind,
                        selected_count=selected,
                        submitted_count=submitted,
                        succeeded_count=0,
                        failed_count=failed,
                        response_rate=0.0,
                    )
                )
            elif metric.kind == "likert":
                values = [obs.value for obs in subset if isinstance(obs.value, LikertValue)]
                counts: dict[int, int] = {}
                for value in values:
                    counts[value.score] = counts.get(value.score, 0) + 1
                mean = (sum(value.score for value in values) / len(values)) if values else None
                summaries.append(
                    MetricSampleSummary(
                        agent=agent,
                        metric_id=metric_id,
                        metric_version=metric_version,
                        metric_kind="likert",
                        sample_kind=sample_kind,
                        selected_count=selected,
                        submitted_count=submitted,
                        succeeded_count=succeeded,
                        failed_count=failed,
                        response_rate=response_rate,
                        likert=LikertSummary(
                            count=len(values),
                            distribution=tuple(sorted(counts.items())),
                            mean=mean,
                        ),
                    )
                )
            elif metric.kind == "scalar":
                values = [obs.value.value for obs in subset if isinstance(obs.value, ScalarValue)]
                summaries.append(
                    MetricSampleSummary(
                        agent=agent,
                        metric_id=metric_id,
                        metric_version=metric_version,
                        metric_kind="scalar",
                        sample_kind=sample_kind,
                        selected_count=selected,
                        submitted_count=submitted,
                        succeeded_count=succeeded,
                        failed_count=failed,
                        response_rate=response_rate,
                        scalar=ScalarSummary(
                            count=len(values),
                            mean=(sum(values) / len(values)) if values else None,
                            min_value=min(values) if values else None,
                            max_value=max(values) if values else None,
                        ),
                    )
                )
            elif metric.kind == "categorical":
                values = [obs.value.category for obs in subset if isinstance(obs.value, CategoricalValue)]
                counts: dict[str, int] = {}
                for value in values:
                    counts[value] = counts.get(value, 0) + 1
                summaries.append(
                    MetricSampleSummary(
                        agent=agent,
                        metric_id=metric_id,
                        metric_version=metric_version,
                        metric_kind="categorical",
                        sample_kind=sample_kind,
                        selected_count=selected,
                        submitted_count=submitted,
                        succeeded_count=succeeded,
                        failed_count=failed,
                        response_rate=response_rate,
                        categorical=CategoricalSummary(
                            count=len(values),
                            counts=tuple(sorted(counts.items())),
                        ),
                    )
                )

    status = "completed"
    if run.status == "failed":
        status = "failed"
    elif run.status == "partial" or warnings:
        status = "partial"

    return EvaluationReport(
        version="alt-report-v1",
        status=status,
        summaries=tuple(
            sorted(
                summaries,
                key=lambda row: (
                    row.agent.tenant_id,
                    row.agent.agent_id,
                    row.metric_id,
                    row.metric_version,
                    row.sample_kind,
                ),
            )
        ),
        warnings=tuple(sorted(set(warnings))),
        ingest_issue_count=ingest_issue_count,
        notes=tuple(sorted(set(notes))),
    )
