"""Sampled-vs-census experiment harness for Agent 365 OTLP datasets."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

from .agent365_otel import NormalizationResult, normalize_agent365_records
from .evaluation import EvaluationFailure, EvaluationRun, EvaluationRunner, RunnerConfig
from .judge import AsyncJudge, JudgeDescriptor, JudgeRequest, JudgeResponse, JudgeResponseError
from .metrics import BinaryValue, MetricObservation, MetricSpec, TASK_COMPLETION_V1
from .models import (
    AgentKey,
    AgentSample,
    EvaluationUnit,
    EvaluationWindow,
    SampleBatch,
    SamplePlan,
    SamplePolicy,
    SampledUnit,
    StratumPlan,
)
from .reporting import EvaluationReport, build_report
from .sampling import SamplingEngine


_LABEL_INT_KEY = "evaluation.expected"
_LABEL_TEXT_KEY = "evaluation.expected.label"
_CASE_ID_KEY = "evaluation.source.case_id"
_CONVERSATION_ID_KEY = "gen_ai.conversation.id"


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _otlp_attr_map(attrs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(attrs, list):
        for row in attrs:
            if not isinstance(row, dict):
                continue
            key = row.get("key")
            if not isinstance(key, str) or not key:
                continue
            value = row.get("value")
            if isinstance(value, dict):
                if "stringValue" in value:
                    out[key] = value.get("stringValue")
                elif "intValue" in value:
                    out[key] = value.get("intValue")
                elif "boolValue" in value:
                    out[key] = value.get("boolValue")
                elif "doubleValue" in value:
                    out[key] = value.get("doubleValue")
                else:
                    out[key] = value
            else:
                out[key] = value
    elif isinstance(attrs, dict):
        out = dict(attrs)
    return out


def _coerce_label(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "pass", "passed", "completed"}:
            return True
        if token in {"0", "false", "fail", "failed", "not_completed", "not completed"}:
            return False
    return None


def _extract_labels_by_conversation(
    document: dict[str, Any],
) -> tuple[
    dict[tuple[str, str, str], bool],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, bool],
]:
    labels: dict[tuple[str, str, str], bool] = {}
    metadata: dict[tuple[str, str, str], dict[str, Any]] = {}
    by_conversation: dict[str, set[bool]] = {}

    resource_spans = document.get("resourceSpans")
    if not isinstance(resource_spans, list):
        raise ValueError("Expected strict OTLP top-level resourceSpans list")

    for resource_span in resource_spans:
        if not isinstance(resource_span, dict):
            continue
        resource_attrs = _otlp_attr_map(resource_span.get("resource", {}).get("attributes"))
        for scope_span in resource_span.get("scopeSpans") or []:
            if not isinstance(scope_span, dict):
                continue
            for span in scope_span.get("spans") or []:
                if not isinstance(span, dict):
                    continue
                name = str(span.get("name") or "")
                if "invoke_agent" not in name:
                    continue

                attrs = _otlp_attr_map(span.get("attributes"))
                merged = {**resource_attrs, **attrs}
                conversation_id = merged.get(_CONVERSATION_ID_KEY)
                if not isinstance(conversation_id, str) or not conversation_id.strip():
                    continue
                conversation_id = conversation_id.strip()
                tenant_id = str(
                    merged.get("microsoft.tenant.id")
                    or merged.get("microsoft.agent365.tenant.id")
                    or merged.get("tenant.id")
                    or merged.get("TenantId")
                    or ""
                ).strip()
                agent_id = str(
                    merged.get("gen_ai.agent.id")
                    or merged.get("AgentId")
                    or ""
                ).strip()
                if not tenant_id or not agent_id:
                    continue

                label_int = _coerce_label(merged.get(_LABEL_INT_KEY))
                label_text = _coerce_label(merged.get(_LABEL_TEXT_KEY))

                label: bool | None = None
                if label_int is not None and label_text is not None and label_int != label_text:
                    raise ValueError(f"Inconsistent labels for conversation {conversation_id}")
                if label_int is not None:
                    label = label_int
                elif label_text is not None:
                    label = label_text

                if label is None:
                    continue

                scoped_key = (tenant_id, agent_id, conversation_id)
                if scoped_key in labels and labels[scoped_key] != label:
                    raise ValueError(
                        f"Conflicting labels for conversation {conversation_id} under {tenant_id}/{agent_id}"
                    )
                labels[scoped_key] = label
                metadata.setdefault(scoped_key, {})
                metadata[scoped_key].update(
                    {
                        "case_id": merged.get(_CASE_ID_KEY),
                        "task": merged.get("evaluation.task") or merged.get("evaluation.task_name"),
                        "domain": merged.get("evaluation.domain"),
                        "difficulty": merged.get("evaluation.difficulty"),
                    }
                )
                by_conversation.setdefault(conversation_id, set()).add(label)

    unscoped: dict[str, bool] = {}
    for conversation_id, values in by_conversation.items():
        if len(values) == 1 and sum(1 for key in labels if key[2] == conversation_id) == 1:
            unscoped[conversation_id] = next(iter(values))

    return labels, metadata, unscoped


@dataclass(frozen=True)
class SyntheticDataset:
    document: dict[str, Any]
    normalization: NormalizationResult
    labels_by_unit: dict[str, bool]
    labels_by_conversation: dict[str, bool]
    labels_by_conversation_scoped: dict[tuple[str, str, str], bool]
    metadata_by_unit: dict[str, dict[str, Any]]
    evaluation_window: EvaluationWindow


def load_synthetic_a365_otel(path: str | Path) -> SyntheticDataset:
    source_path = Path(path)
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Expected top-level JSON object")
    if "resourceSpans" not in raw:
        raise ValueError("Expected strict OTLP payload with top-level resourceSpans")

    normalization = normalize_agent365_records([raw])
    labels_by_conversation_scoped, metadata_by_conversation_scoped, labels_by_conversation = _extract_labels_by_conversation(raw)

    labels_by_unit: dict[str, bool] = {}
    metadata_by_unit: dict[str, dict[str, Any]] = {}
    min_started: datetime | None = None
    max_ended: datetime | None = None

    for unit in normalization.units:
        if unit.started_at is not None:
            min_started = unit.started_at if min_started is None else min(min_started, unit.started_at)
        if unit.ended_at is not None:
            max_ended = unit.ended_at if max_ended is None else max(max_ended, unit.ended_at)

        matched = {
            labels_by_conversation_scoped[(unit.tenant_id, unit.agent_id, cid)]
            for cid in unit.conversation_ids
            if (unit.tenant_id, unit.agent_id, cid) in labels_by_conversation_scoped
        }
        if not matched:
            matched = {
                labels_by_conversation[cid]
                for cid in unit.conversation_ids
                if cid in labels_by_conversation
            }

        if len(matched) != 1:
            raise ValueError(f"Unit {unit.unit_id} does not map to exactly one consistent label")

        label = next(iter(matched))
        labels_by_unit[unit.unit_id or ""] = label

        combined_meta: dict[str, Any] = {}
        for cid in unit.conversation_ids:
            scoped_key = (unit.tenant_id, unit.agent_id, cid)
            if scoped_key in metadata_by_conversation_scoped:
                combined_meta.update(metadata_by_conversation_scoped[scoped_key])
        metadata_by_unit[unit.unit_id or ""] = combined_meta

    if min_started is None:
        min_started = _parse_time(raw.get("startTime")) or datetime.now(timezone.utc)
    if max_ended is None:
        max_ended = _parse_time(raw.get("endTime")) or min_started

    if max_ended <= min_started:
        max_ended = min_started + timedelta(microseconds=1)
    else:
        max_ended = max_ended + timedelta(microseconds=1)

    evaluation_window = EvaluationWindow(start_at=min_started, end_at=max_ended)

    return SyntheticDataset(
        document=raw,
        normalization=normalization,
        labels_by_unit=labels_by_unit,
        labels_by_conversation=labels_by_conversation,
        labels_by_conversation_scoped=labels_by_conversation_scoped,
        metadata_by_unit=metadata_by_unit,
        evaluation_window=evaluation_window,
    )


@dataclass(frozen=True)
class DatasetGroundTruthJudge(AsyncJudge):
    labels_by_unit: dict[str, bool]
    descriptor: JudgeDescriptor = JudgeDescriptor(provider="dataset", name="ground-truth", version="v1")

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        if request.unit_id not in self.labels_by_unit:
            raise JudgeResponseError(f"No dataset label for unit_id={request.unit_id}")
        return JudgeResponse(
            request_id=request.request_id,
            metric=request.metric,
            value=BinaryValue(passed=self.labels_by_unit[request.unit_id]),
            reasoning="dataset_label",
        )


def _group_by_agent(units: Iterable[EvaluationUnit]) -> dict[AgentKey, list[EvaluationUnit]]:
    grouped: dict[AgentKey, list[EvaluationUnit]] = {}
    for unit in units:
        key = AgentKey(tenant_id=unit.tenant_id, agent_id=unit.agent_id)
        grouped.setdefault(key, []).append(unit)
    for values in grouped.values():
        values.sort(key=lambda row: row.unit_id or "")
    return grouped


def build_census_batch(
    units: Iterable[EvaluationUnit],
    policy: SamplePolicy,
    evaluation_window: EvaluationWindow,
) -> SampleBatch:
    grouped = _group_by_agent(units)
    agents: list[AgentSample] = []

    for agent_key in sorted(grouped):
        population = grouped[agent_key]
        by_stratum: dict[str, int] = {}
        for unit in population:
            turn_count = len(unit.turns)
            if turn_count <= 1:
                band = "1"
            elif turn_count <= 3:
                band = "2-3"
            elif turn_count <= 7:
                band = "4-7"
            elif turn_count <= 15:
                band = "8-15"
            else:
                band = "16+"
            stratum_key = f"{band}|{(unit.channel or 'unknown').strip() or 'unknown'}"
            by_stratum[stratum_key] = by_stratum.get(stratum_key, 0) + 1

        units = tuple(
            SampledUnit(
                unit=unit,
                estimand_eligible=True,
                stratum_key="census",
                inclusion_probability=1.0,
                sampling_weight=1.0,
                selection_reason="census",
            )
            for unit in population
        )
        plan = SamplePlan(
            population=len(population),
            recommended=len(population),
            selected=len(population),
            capacity=None,
            census=True,
            precision_status="meets_statistical_recommendation",
            effective_rate=1.0 if population else 0.0,
        )
        strata = tuple(
            StratumPlan(key=key, population=count, selected=count)
            for key, count in sorted(by_stratum.items())
        )
        agents.append(
            AgentSample(
                agent=agent_key,
                plan=plan,
                strata=strata,
                units=units,
            )
        )

    run_material = [
        "census",
        policy.version,
        evaluation_window.start_at.isoformat(),
        evaluation_window.end_at.isoformat(),
    ]
    for agent_sample in agents:
        run_material.append(f"{agent_sample.agent.tenant_id}/{agent_sample.agent.agent_id}")
        run_material.append(
            ",".join(sampled.unit.unit_id or "" for sampled in agent_sample.units)
        )

    import hashlib

    run_id = hashlib.sha256("||".join(run_material).encode("utf-8")).hexdigest()[:16]
    return SampleBatch(
        policy=policy,
        version=policy.version,
        run_id=run_id,
        agents=tuple(agents),
        tenant_capacities=(),
        evaluation_window=evaluation_window,
    )


def _binary_value(observation: MetricObservation) -> bool | None:
    if isinstance(observation.value, BinaryValue):
        return observation.value.passed
    return None


def _wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)
    return (max(0.0, center - half), min(1.0, center + half))


def _wilson_with_fpc(successes: int, n: int, population: int, z: float) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    lower, upper = _wilson(successes, n, z=z)
    if population > 1 and n < population:
        fpc = math.sqrt((population - n) / (population - 1))
        lower = p - (p - lower) * fpc
        upper = p + (upper - p) * fpc
    return (max(0.0, lower), min(1.0, upper))


def _variance_with_wilson_adjustment(successes: int, n: int, population: int, z: float) -> float:
    if n <= 0:
        return 0.0
    if population <= 1 or n >= population:
        return 0.0
    z2 = z * z
    p_tilde = (successes + z2 / 2.0) / (n + z2)
    base = (p_tilde * (1.0 - p_tilde)) / n
    fpc = (population - n) / (population - 1)
    return max(0.0, base * fpc)


def compare_from_observations(
    dataset: SyntheticDataset,
    sampled_batch: SampleBatch,
    sampled_observations: tuple[MetricObservation, ...],
    sampled_failures: int,
    census_batch: SampleBatch,
    census_observations: tuple[MetricObservation, ...],
    census_failures: int,
    metric: MetricSpec = TASK_COMPLETION_V1,
) -> dict[str, Any]:
    if metric.kind != "binary":
        raise ValueError("compare_from_observations currently supports binary metrics")

    census_by_unit: dict[str, MetricObservation] = {}
    for obs in census_observations:
        if obs.metric.id == metric.id and obs.metric.version == metric.version:
            census_by_unit[obs.unit_id] = obs

    sampled_units = [
        obs
        for obs in sampled_observations
        if obs.metric.id == metric.id and obs.metric.version == metric.version
    ]

    census_binary = [_binary_value(obs) for obs in census_by_unit.values()]
    census_known = [value for value in census_binary if value is not None]
    observed_census_pass_rate = (
        (sum(1 for value in census_known if value) / len(census_known)) if census_known else None
    )

    confidence = sampled_batch.policy.confidence
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    weighted_num = 0.0
    weighted_den = 0.0
    weighted_variance = 0.0
    covered_population = 0
    total_population = sum(agent.plan.population for agent in sampled_batch.agents)
    per_agent_rows: list[dict[str, Any]] = []

    sampled_by_agent_unit: dict[tuple[str, str], list[MetricObservation]] = {}
    for obs in sampled_units:
        sampled_by_agent_unit.setdefault((obs.tenant_id, obs.agent_id), []).append(obs)

    census_by_agent_unit: dict[tuple[str, str], list[MetricObservation]] = {}
    for obs in census_by_unit.values():
        census_by_agent_unit.setdefault((obs.tenant_id, obs.agent_id), []).append(obs)

    for agent_sample in sampled_batch.agents:
        key = (agent_sample.agent.tenant_id, agent_sample.agent.agent_id)
        pop_n = agent_sample.plan.population

        agent_census_obs = census_by_agent_unit.get(key, [])
        agent_census_vals = [value for value in (_binary_value(o) for o in agent_census_obs) if value is not None]
        census_rate = (sum(1 for value in agent_census_vals if value) / len(agent_census_vals)) if agent_census_vals else None

        agent_sample_obs = sampled_by_agent_unit.get(key, [])
        sample_vals = [value for value in (_binary_value(o) for o in agent_sample_obs) if value is not None]
        sample_n = len(sample_vals)
        successes = sum(1 for value in sample_vals if value)
        sample_rate = (successes / sample_n) if sample_n > 0 else None

        if sample_n > 0:
            if agent_sample.plan.census or sample_n >= pop_n:
                ci_lower, ci_upper = (sample_rate, sample_rate)
            else:
                ci_lower, ci_upper = _wilson_with_fpc(successes, sample_n, pop_n, z)
        else:
            ci_lower, ci_upper = (None, None)

        if sample_rate is not None:
            weighted_num += pop_n * sample_rate
            weighted_den += pop_n
            covered_population += pop_n

            if pop_n > 1 and 0 < sample_n < pop_n:
                weight = pop_n
                weighted_variance += (weight * weight) * _variance_with_wilson_adjustment(
                    successes=successes,
                    n=sample_n,
                    population=pop_n,
                    z=z,
                )

        error = None
        covered = None
        if census_rate is not None and sample_rate is not None:
            error = sample_rate - census_rate
        if census_rate is not None and ci_lower is not None and ci_upper is not None:
            covered = ci_lower <= census_rate <= ci_upper

        per_agent_rows.append(
            {
                "tenant_id": agent_sample.agent.tenant_id,
                "agent_id": agent_sample.agent.agent_id,
                "population": pop_n,
                "is_census_agent": agent_sample.plan.census,
                "observed_census_pass_rate": census_rate,
                "census_pass_rate": None,
                "sample_n": sample_n,
                "observed_sample_pass_rate": sample_rate,
                "sample_pass_rate": None,
                "sample_ci_lower": ci_lower,
                "sample_ci_upper": ci_upper,
                "observed_signed_error": error,
                "signed_error": None,
                "observed_ci_contains_census": covered,
                "ci_contains_census": None,
            }
        )

    sampled_global_core_rate = (weighted_num / weighted_den) if weighted_den > 0 else None
    core_ci_lower = None
    core_ci_upper = None
    core_ci_method = "agent_weighted_normal_wilson_tilde_fpc"
    if sampled_global_core_rate is not None:
        if weighted_den <= 0:
            core_ci_lower, core_ci_upper = (None, None)
        elif weighted_variance <= 0:
            core_ci_lower, core_ci_upper = (sampled_global_core_rate, sampled_global_core_rate)
        else:
            se = math.sqrt(weighted_variance) / weighted_den
            core_ci_lower = max(0.0, sampled_global_core_rate - z * se)
            core_ci_upper = min(1.0, sampled_global_core_rate + z * se)

    core_population_coverage = (covered_population / total_population) if total_population > 0 else 0.0
    complete_core_population_coverage = bool(total_population > 0 and covered_population == total_population)

    observed_signed_error = None
    observed_absolute_error = None
    if sampled_global_core_rate is not None and observed_census_pass_rate is not None:
        observed_signed_error = sampled_global_core_rate - observed_census_pass_rate
        observed_absolute_error = abs(observed_signed_error)

    observed_census_truth_in_interval = (
        (core_ci_lower is not None and core_ci_upper is not None and observed_census_pass_rate is not None)
        and (core_ci_lower <= observed_census_pass_rate <= core_ci_upper)
    )

    census_complete = len(census_by_unit) == len(dataset.normalization.units) and census_failures == 0
    comparison_valid = complete_core_population_coverage and census_complete
    if comparison_valid:
        for row in per_agent_rows:
            row["census_pass_rate"] = row["observed_census_pass_rate"]
            row["sample_pass_rate"] = row["observed_sample_pass_rate"]
            row["signed_error"] = row["observed_signed_error"]
            row["ci_contains_census"] = row["observed_ci_contains_census"]

    calls_saved = max(0, len(census_batch.all_units()) - len(sampled_batch.all_units()))
    fraction_saved = (
        calls_saved / len(census_batch.all_units())
        if len(census_batch.all_units()) > 0
        else 0.0
    )

    observed_overall_accuracy = None
    sampled_core_accuracy = None
    if dataset.labels_by_unit:
        census_hits = 0
        census_total = 0
        for unit_id, obs in census_by_unit.items():
            predicted = _binary_value(obs)
            if predicted is None:
                continue
            expected = dataset.labels_by_unit.get(unit_id)
            if expected is None:
                continue
            census_total += 1
            if predicted == expected:
                census_hits += 1
        if census_total > 0:
            observed_overall_accuracy = census_hits / census_total

        core_hits = 0
        core_total = 0
        for obs in sampled_units:
            predicted = _binary_value(obs)
            if predicted is None:
                continue
            expected = dataset.labels_by_unit.get(obs.unit_id)
            if expected is None:
                continue
            core_total += 1
            if predicted == expected:
                core_hits += 1
        if core_total > 0:
            sampled_core_accuracy = core_hits / core_total

    return {
        "population_total": len(dataset.normalization.units),
        "census_evaluated": len(census_by_unit),
        "sampled_evaluated": len(sampled_observations),
        "sampled_units_evaluated": len(sampled_units),
        "observed_census_pass_rate": observed_census_pass_rate,
        "observed_sampled_units_pass_rate": sampled_global_core_rate,
        "census_pass_rate": observed_census_pass_rate if comparison_valid else None,
        "sampled_units_pass_rate": sampled_global_core_rate if comparison_valid else None,
        "units_population_covered": covered_population,
        "units_population_coverage": core_population_coverage,
        "complete_units_population_coverage": complete_core_population_coverage,
        "observed_absolute_error": observed_absolute_error,
        "observed_signed_error": observed_signed_error,
        "absolute_error": observed_absolute_error if comparison_valid else None,
        "signed_error": observed_signed_error if comparison_valid else None,
        "sample_ci_lower": core_ci_lower,
        "sample_ci_upper": core_ci_upper,
        "sample_ci_method": core_ci_method,
        "observed_census_truth_in_interval": observed_census_truth_in_interval,
        "census_truth_in_interval": observed_census_truth_in_interval if comparison_valid else None,
        "calls_saved": calls_saved,
        "calls_saved_kind": "counterfactual_if_sampled_only",
        "fraction_saved": fraction_saved,
        "census_complete": census_complete,
        "comparison_valid": comparison_valid,
        "comparison_note": (
            None
            if comparison_valid
            else (
                "Comparison invalid until census is complete and failure-free with full sample coverage"
                if not census_complete
                else "Estimator covers only agents with sampled responses; missing agents reduce validity"
            )
        ),
        "observed_judge_accuracy_census_overall": observed_overall_accuracy,
        "judge_accuracy_census_overall": observed_overall_accuracy if comparison_valid else None,
        "judge_accuracy_sampled_units": sampled_core_accuracy,
        "sampled_failure_count": sampled_failures,
        "census_failure_count": census_failures,
        "sampled_response_count": len(sampled_observations),
        "census_response_count": len(census_by_unit),
        "per_agent": sorted(
            per_agent_rows,
            key=lambda row: (row["tenant_id"], row["agent_id"]),
        ),
    }


@dataclass(frozen=True)
class ExperimentArmResult:
    batch: SampleBatch
    evaluation: EvaluationRun
    report: EvaluationReport


@dataclass(frozen=True)
class ExperimentComparison:
    sampled: ExperimentArmResult
    census: ExperimentArmResult
    comparison: dict[str, Any]


@dataclass(frozen=True)
class MultiPolicyComparison:
    by_policy: dict[str, ExperimentComparison]
    census: ExperimentArmResult


def _project_sampled_from_census_eval(
    sampled_batch: SampleBatch,
    census_eval: EvaluationRun,
    metric: MetricSpec,
) -> tuple[tuple[MetricObservation, ...], tuple[EvaluationFailure, ...], str]:
    sampled_lookup: dict[str, SampledUnit] = {}
    for sampled in sampled_batch.all_units():
        sampled_lookup[sampled.unit.unit_id or ""] = sampled

    sampled_obs: list[MetricObservation] = []
    for obs in census_eval.observations:
        sampled = sampled_lookup.get(obs.unit_id)
        if sampled is None:
            continue
        sampled_obs.append(
            MetricObservation(
                request_id=obs.request_id,
                agent=obs.agent,
                tenant_id=obs.tenant_id,
                agent_id=obs.agent_id,
                unit_id=obs.unit_id,
                session_id=obs.session_id,
                conversation_ids=obs.conversation_ids,
                metric=obs.metric,
                value=obs.value,
                estimand_eligible=sampled.estimand_eligible,
                judge=obs.judge,
                evidence_sha256=obs.evidence_sha256,
                reasoning=obs.reasoning,
            )
        )

    sampled_failures: list[EvaluationFailure] = []
    for failure in census_eval.failures:
        sampled = sampled_lookup.get(failure.unit_id)
        if sampled is None:
            continue
        sampled_failures.append(
            EvaluationFailure(
                request_id=failure.request_id,
                tenant_id=failure.tenant_id,
                agent_id=failure.agent_id,
                unit_id=failure.unit_id,
                session_id=failure.session_id,
                metric_id=failure.metric_id,
                metric_version=failure.metric_version,
                code=failure.code,
                attempts=failure.attempts,
                retryable=failure.retryable,
                message=failure.message,
            )
        )

    sampled_obs_sorted = tuple(sorted(sampled_obs, key=lambda row: row.request_id))
    sampled_failures_sorted = tuple(sorted(sampled_failures, key=lambda row: row.request_id))
    status = "completed"
    if sampled_failures_sorted and not sampled_obs_sorted:
        status = "failed"
    elif sampled_failures_sorted:
        status = "partial"
    return sampled_obs_sorted, sampled_failures_sorted, status


async def run_sampled_vs_census(
    dataset: SyntheticDataset,
    judge: AsyncJudge,
    sample_policy: SamplePolicy,
    runner_config: RunnerConfig,
    max_evidence_bytes: int,
    metric: MetricSpec = TASK_COMPLETION_V1,
) -> ExperimentComparison:
    units = dataset.normalization.units
    sampled_batch = SamplingEngine().sample(
        units=units,
        policy=sample_policy,
        evaluation_window=dataset.evaluation_window,
    )
    census_batch = build_census_batch(
        units=units,
        policy=sample_policy,
        evaluation_window=dataset.evaluation_window,
    )

    runner = EvaluationRunner(
        judge=judge,
        config=runner_config,
        max_evidence_bytes=max_evidence_bytes,
    )

    census_eval = await runner.run(census_batch, (metric,))
    census_report = build_report(
        census_batch,
        census_eval,
        ingest_issue_count=len(dataset.normalization.issues),
    )

    sampled_obs, sampled_failures, sampled_status = _project_sampled_from_census_eval(
        sampled_batch=sampled_batch,
        census_eval=census_eval,
        metric=metric,
    )
    sampled_eval = EvaluationRun(
        status=sampled_status,
        observations=sampled_obs,
        failures=sampled_failures,
        selected_count=len(sampled_batch.all_units()),
        request_count=len(sampled_batch.all_units()),
        judge_descriptor=census_eval.judge_descriptor,
        judge_prompt_schema_fingerprint=census_eval.judge_prompt_schema_fingerprint,
        metrics=(metric,),
    )
    sampled_report = build_report(
        sampled_batch,
        sampled_eval,
        ingest_issue_count=len(dataset.normalization.issues),
    )

    comparison = compare_from_observations(
        dataset=dataset,
        sampled_batch=sampled_batch,
        sampled_observations=sampled_eval.observations,
        sampled_failures=len(sampled_eval.failures),
        census_batch=census_batch,
        census_observations=census_eval.observations,
        census_failures=len(census_eval.failures),
        metric=metric,
    )

    return ExperimentComparison(
        sampled=ExperimentArmResult(
            batch=sampled_batch,
            evaluation=sampled_eval,
            report=sampled_report,
        ),
        census=ExperimentArmResult(
            batch=census_batch,
            evaluation=census_eval,
            report=census_report,
        ),
        comparison=comparison,
    )


async def run_policies_vs_census(
    dataset: SyntheticDataset,
    judge: AsyncJudge,
    policies: dict[str, SamplePolicy],
    runner_config: RunnerConfig,
    max_evidence_bytes: int,
    metric: MetricSpec = TASK_COMPLETION_V1,
) -> MultiPolicyComparison:
    if not policies:
        raise ValueError("policies must not be empty")

    units = dataset.normalization.units
    first_policy = next(iter(policies.values()))
    census_batch = build_census_batch(
        units=units,
        policy=first_policy,
        evaluation_window=dataset.evaluation_window,
    )

    runner = EvaluationRunner(
        judge=judge,
        config=runner_config,
        max_evidence_bytes=max_evidence_bytes,
    )
    census_eval = await runner.run(census_batch, (metric,))
    census_report = build_report(
        census_batch,
        census_eval,
        ingest_issue_count=len(dataset.normalization.issues),
    )
    census_arm = ExperimentArmResult(
        batch=census_batch,
        evaluation=census_eval,
        report=census_report,
    )

    by_policy: dict[str, ExperimentComparison] = {}
    for policy_name, policy in policies.items():
        sampled_batch = SamplingEngine().sample(
            units=units,
            policy=policy,
            evaluation_window=dataset.evaluation_window,
        )
        sampled_obs, sampled_failures, sampled_status = _project_sampled_from_census_eval(
            sampled_batch=sampled_batch,
            census_eval=census_eval,
            metric=metric,
        )
        sampled_eval = EvaluationRun(
            status=sampled_status,
            observations=sampled_obs,
            failures=sampled_failures,
            selected_count=len(sampled_batch.all_units()),
            request_count=len(sampled_batch.all_units()),
            judge_descriptor=census_eval.judge_descriptor,
            judge_prompt_schema_fingerprint=census_eval.judge_prompt_schema_fingerprint,
            metrics=(metric,),
        )
        sampled_report = build_report(
            sampled_batch,
            sampled_eval,
            ingest_issue_count=len(dataset.normalization.issues),
        )
        comparison = compare_from_observations(
            dataset=dataset,
            sampled_batch=sampled_batch,
            sampled_observations=sampled_eval.observations,
            sampled_failures=len(sampled_eval.failures),
            census_batch=census_batch,
            census_observations=census_eval.observations,
            census_failures=len(census_eval.failures),
            metric=metric,
        )
        comparison["census_complete"] = bool(
            comparison.get("census_evaluated") == comparison.get("population_total")
            and comparison.get("census_failure_count") == 0
        )
        comparison["comparison_valid"] = bool(
            comparison.get("complete_units_population_coverage", False)
            and comparison["census_complete"]
        )
        if not comparison["comparison_valid"]:
            comparison["comparison_note"] = (
                "Comparison invalid until census is complete and failure-free with full sample coverage"
            )
        by_policy[policy_name] = ExperimentComparison(
            sampled=ExperimentArmResult(
                batch=sampled_batch,
                evaluation=sampled_eval,
                report=sampled_report,
            ),
            census=census_arm,
            comparison=comparison,
        )

    return MultiPolicyComparison(by_policy=by_policy, census=census_arm)


async def resume_policies_vs_census(
    dataset: SyntheticDataset,
    judge: AsyncJudge,
    policies: dict[str, SamplePolicy],
    runner_config: RunnerConfig,
    max_evidence_bytes: int,
    prior_path: str | Path,
    metric: MetricSpec = TASK_COMPLETION_V1,
) -> MultiPolicyComparison:
    if not policies:
        raise ValueError("policies must not be empty")

    units = dataset.normalization.units
    first_policy = next(iter(policies.values()))
    census_batch = build_census_batch(
        units=units,
        policy=first_policy,
        evaluation_window=dataset.evaluation_window,
    )

    prior_observations, _prior_meta = load_prior_census_observations(
        path=prior_path,
        dataset=dataset,
        judge_descriptor=judge.descriptor,
        metric=metric,
        policy=first_policy,
    )

    expected_ids = {
        sampled.unit.unit_id
        for sampled in census_batch.all_units()
        if sampled.unit.unit_id is not None
    }
    prior_ids = {obs.unit_id for obs in prior_observations}
    pending_ids = expected_ids - prior_ids

    runner = EvaluationRunner(
        judge=judge,
        config=runner_config,
        max_evidence_bytes=max_evidence_bytes,
    )
    delta_eval = await runner.run(census_batch, (metric,), unit_ids=pending_ids)

    combined_by_unit: dict[str, MetricObservation] = {obs.unit_id: obs for obs in prior_observations}
    for obs in delta_eval.observations:
        combined_by_unit[obs.unit_id] = obs
    census_observations = tuple(sorted(combined_by_unit.values(), key=lambda row: row.unit_id))

    pending_id_set = set(pending_ids)
    unresolved_failures = tuple(
        failure
        for failure in delta_eval.failures
        if failure.unit_id in pending_id_set and failure.unit_id not in combined_by_unit
    )

    census_complete = len(census_observations) == len(expected_ids) and not unresolved_failures
    census_status = "completed" if census_complete else ("partial" if census_observations else "failed")
    census_eval = EvaluationRun(
        status=census_status,
        observations=census_observations,
        failures=tuple(sorted(unresolved_failures, key=lambda row: row.request_id)),
        selected_count=len(expected_ids),
        request_count=len(expected_ids) * len((metric,)),
        judge_descriptor=judge.descriptor,
        judge_prompt_schema_fingerprint=getattr(judge, "prompt_schema_fingerprint", None),
        metrics=(metric,),
    )
    census_report = build_report(
        census_batch,
        census_eval,
        ingest_issue_count=len(dataset.normalization.issues),
    )
    census_arm = ExperimentArmResult(
        batch=census_batch,
        evaluation=census_eval,
        report=census_report,
    )

    by_policy: dict[str, ExperimentComparison] = {}
    for policy_name, policy in policies.items():
        sampled_batch = SamplingEngine().sample(
            units=units,
            policy=policy,
            evaluation_window=dataset.evaluation_window,
        )
        sampled_obs, sampled_failures, sampled_status = _project_sampled_from_census_eval(
            sampled_batch=sampled_batch,
            census_eval=census_eval,
            metric=metric,
        )
        sampled_eval = EvaluationRun(
            status=sampled_status,
            observations=sampled_obs,
            failures=sampled_failures,
            selected_count=len(sampled_batch.all_units()),
            request_count=len(sampled_batch.all_units()),
            judge_descriptor=census_eval.judge_descriptor,
            judge_prompt_schema_fingerprint=census_eval.judge_prompt_schema_fingerprint,
            metrics=(metric,),
        )
        sampled_report = build_report(
            sampled_batch,
            sampled_eval,
            ingest_issue_count=len(dataset.normalization.issues),
        )
        comparison = compare_from_observations(
            dataset=dataset,
            sampled_batch=sampled_batch,
            sampled_observations=sampled_eval.observations,
            sampled_failures=len(sampled_eval.failures),
            census_batch=census_batch,
            census_observations=census_eval.observations,
            census_failures=len(census_eval.failures),
            metric=metric,
        )
        comparison["census_complete"] = census_complete
        comparison["comparison_valid"] = bool(
            comparison.get("complete_units_population_coverage", False)
            and comparison.get("census_evaluated") == comparison.get("population_total")
            and comparison.get("census_failure_count") == 0
        )
        if not comparison["comparison_valid"]:
            comparison["comparison_note"] = (
                "Comparison invalid until census is complete and failure-free with full sample coverage"
            )
        by_policy[policy_name] = ExperimentComparison(
            sampled=ExperimentArmResult(
                batch=sampled_batch,
                evaluation=sampled_eval,
                report=sampled_report,
            ),
            census=census_arm,
            comparison=comparison,
        )

    return MultiPolicyComparison(by_policy=by_policy, census=census_arm)


def _minimal_observations_rows(observations: Iterable[MetricObservation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obs in observations:
        if isinstance(obs.value, BinaryValue):
            value = int(obs.value.passed)
        else:
            continue
        rows.append(
            {
                "unit_id": obs.unit_id,
                "tenant_id": obs.tenant_id,
                "agent_id": obs.agent_id,
                "metric_id": obs.metric.id,
                "metric_version": obs.metric.version,
                "prediction": value,
            }
        )
    return sorted(rows, key=lambda row: (row["tenant_id"], row["agent_id"], row["unit_id"]))


def _failure_code_counts(failures: Iterable[EvaluationFailure]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        counts[failure.code] = counts.get(failure.code, 0) + 1
    return dict(sorted(counts.items()))


def load_prior_census_observations(
    path: str | Path,
    dataset: SyntheticDataset,
    judge_descriptor: JudgeDescriptor,
    metric: MetricSpec = TASK_COMPLETION_V1,
    policy: SamplePolicy | None = None,
    allow_legacy_judge: bool = False,
) -> tuple[tuple[MetricObservation, ...], dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected top-level JSON object in prior census artifact")
    if payload.get("version") != "random-multi-policy-sampled-vs-census-v1":
        raise ValueError("Unsupported prior artifact version")

    census = payload.get("census")
    if not isinstance(census, dict):
        raise ValueError("Prior artifact missing census object")

    artifact_judge = census.get("judge_descriptor")
    if not isinstance(artifact_judge, dict):
        raise ValueError("Prior census artifact missing judge_descriptor")
    artifact_provider = artifact_judge.get("provider")
    artifact_name = artifact_judge.get("name")
    artifact_version = artifact_judge.get("version")
    if not all(isinstance(value, str) and value.strip() for value in (artifact_provider, artifact_name, artifact_version)):
        raise ValueError("Prior census artifact has invalid judge_descriptor")

    artifact_descriptor = JudgeDescriptor(
        provider=artifact_provider.strip(),
        name=artifact_name.strip(),
        version=artifact_version.strip(),
    )
    if artifact_descriptor != judge_descriptor and not allow_legacy_judge:
        raise ValueError(
            "Prior census judge_descriptor mismatch: "
            f"expected {judge_descriptor.provider}/{judge_descriptor.name}/{judge_descriptor.version}, "
            f"found {artifact_descriptor.provider}/{artifact_descriptor.name}/{artifact_descriptor.version}"
        )

    effective_policy = policy if policy is not None else SamplePolicy(version="random-session-stratified-v1")
    expected_batch = build_census_batch(
        units=dataset.normalization.units,
        policy=effective_policy,
        evaluation_window=dataset.evaluation_window,
    )
    prior_run_id = census.get("run_id")
    if prior_run_id != expected_batch.run_id:
        raise ValueError(
            f"Prior census run_id mismatch: expected {expected_batch.run_id}, found {prior_run_id}"
        )

    rows = census.get("observations")
    if not isinstance(rows, list):
        raise ValueError("Prior census observations must be a list")

    expected_units: dict[str, EvaluationUnit] = {}
    for unit in dataset.normalization.units:
        if unit.unit_id is None:
            continue
        expected_units[unit.unit_id] = unit

    by_unit_id: dict[str, dict[str, Any]] = {}
    observations: list[MetricObservation] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Prior census observation at index {index} is not an object")

        unit_id = row.get("unit_id")
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise ValueError(f"Prior census observation at index {index} has invalid unit_id")
        unit_id = unit_id.strip()
        if unit_id in by_unit_id:
            raise ValueError(f"Duplicate prior census observation for unit_id={unit_id}")
        by_unit_id[unit_id] = row

        if unit_id not in expected_units:
            raise ValueError(f"Prior census contains unknown unit_id={unit_id}")
        unit = expected_units[unit_id]

        if row.get("metric_id") != metric.id or row.get("metric_version") != metric.version:
            raise ValueError(f"Prior census observation for unit_id={unit_id} has mismatched metric")

        prediction = row.get("prediction")
        if not isinstance(prediction, int) or prediction not in (0, 1):
            raise ValueError(f"Prior census observation for unit_id={unit_id} has invalid prediction")

        row_tenant_id = row.get("tenant_id")
        row_agent_id = row.get("agent_id")
        if row_tenant_id != unit.tenant_id or row_agent_id != unit.agent_id:
            raise ValueError(f"Prior census observation for unit_id={unit_id} has mismatched scoped identity")

        request_material = f"prior:{prior_run_id}:{metric.id}:{metric.version}:{unit_id}"
        request_id = __import__("hashlib").sha256(request_material.encode("utf-8")).hexdigest()[:24]
        observations.append(
            MetricObservation(
                request_id=request_id,
                agent=AgentKey(tenant_id=unit.tenant_id, agent_id=unit.agent_id),
                tenant_id=unit.tenant_id,
                agent_id=unit.agent_id,
                unit_id=unit_id,
                session_id=unit.session_id,
                conversation_ids=unit.conversation_ids,
                metric=metric,
                value=BinaryValue(passed=bool(prediction)),
                estimand_eligible=True,
                judge=judge_descriptor,
                evidence_sha256="not-persisted",
                reasoning=None,
            )
        )

    prior_meta = {
        "version": payload.get("version"),
        "run_id": prior_run_id,
        "status": census.get("status"),
        "selected_count": census.get("selected_count"),
        "planned_request_count": census.get("planned_request_count"),
        "delta_attempted_count": census.get("delta_attempted_count"),
        "failure_count": census.get("failure_count"),
        "judge_descriptor": {
            "provider": artifact_descriptor.provider,
            "name": artifact_descriptor.name,
            "version": artifact_descriptor.version,
        },
        "loaded_count": len(observations),
    }
    return tuple(sorted(observations, key=lambda row: row.unit_id)), prior_meta


def save_experiment_result(path: str | Path, result: ExperimentComparison) -> None:
    counterfactual_calls_saved = max(
        0,
        result.census.evaluation.selected_count - result.sampled.evaluation.selected_count,
    )
    counterfactual_fraction_saved = (
        counterfactual_calls_saved / result.census.evaluation.selected_count
        if result.census.evaluation.selected_count > 0
        else 0.0
    )
    destination = Path(path)
    payload = {
        "version": "random-sampled-vs-census-v1",
        "sampled": {
            "run_id": result.sampled.batch.run_id,
            "status": result.sampled.evaluation.status,
            "selected_count": result.sampled.evaluation.selected_count,
            "planned_request_count": result.sampled.evaluation.request_count,
            "judge_descriptor": {
                "provider": result.sampled.evaluation.judge_descriptor.provider,
                "name": result.sampled.evaluation.judge_descriptor.name,
                "version": result.sampled.evaluation.judge_descriptor.version,
            },
            "judge_prompt_schema_fingerprint": result.sampled.evaluation.judge_prompt_schema_fingerprint,
            "observations": _minimal_observations_rows(result.sampled.evaluation.observations),
            "failure_count": len(result.sampled.evaluation.failures),
            "counterfactual_calls_saved": counterfactual_calls_saved,
            "counterfactual_fraction_saved": counterfactual_fraction_saved,
        },
        "census": {
            "run_id": result.census.batch.run_id,
            "status": result.census.evaluation.status,
            "selected_count": result.census.evaluation.selected_count,
            "planned_request_count": result.census.evaluation.request_count,
            "judge_descriptor": {
                "provider": result.census.evaluation.judge_descriptor.provider,
                "name": result.census.evaluation.judge_descriptor.name,
                "version": result.census.evaluation.judge_descriptor.version,
            },
            "judge_prompt_schema_fingerprint": result.census.evaluation.judge_prompt_schema_fingerprint,
            "observations": _minimal_observations_rows(result.census.evaluation.observations),
            "failure_count": len(result.census.evaluation.failures),
        },
        "comparison": result.comparison,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_multi_policy_result(path: str | Path, result: MultiPolicyComparison) -> None:
    """Persist aggregate and minimal prediction lineage for multiple policy arms."""
    destination = Path(path)
    census = result.census
    payload = {
        "version": "random-multi-policy-sampled-vs-census-v1",
        "census": {
            "run_id": census.batch.run_id,
            "status": census.evaluation.status,
            "selected_count": census.evaluation.selected_count,
            "planned_request_count": census.evaluation.request_count,
            "judge_descriptor": {
                "provider": census.evaluation.judge_descriptor.provider,
                "name": census.evaluation.judge_descriptor.name,
                "version": census.evaluation.judge_descriptor.version,
            },
            "judge_prompt_schema_fingerprint": census.evaluation.judge_prompt_schema_fingerprint,
            "observations": _minimal_observations_rows(census.evaluation.observations),
            "failure_count": len(census.evaluation.failures),
            "failure_code_counts": _failure_code_counts(census.evaluation.failures),
        },
        "policies": {},
    }
    for policy_name, comparison in sorted(result.by_policy.items()):
        sampled = comparison.sampled
        payload["policies"][policy_name] = {
            "run_id": sampled.batch.run_id,
            "status": sampled.evaluation.status,
            "selected_count": sampled.evaluation.selected_count,
            "planned_request_count": sampled.evaluation.request_count,
            "judge_descriptor": {
                "provider": sampled.evaluation.judge_descriptor.provider,
                "name": sampled.evaluation.judge_descriptor.name,
                "version": sampled.evaluation.judge_descriptor.version,
            },
            "judge_prompt_schema_fingerprint": sampled.evaluation.judge_prompt_schema_fingerprint,
            "observations": _minimal_observations_rows(sampled.evaluation.observations),
            "failure_count": len(sampled.evaluation.failures),
            "failure_code_counts": _failure_code_counts(sampled.evaluation.failures),
            "counterfactual_calls_saved": comparison.comparison["calls_saved"],
            "counterfactual_fraction_saved": comparison.comparison["fraction_saved"],
            "comparison": comparison.comparison,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
