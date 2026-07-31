"""Top-level orchestration for the random sampling and evaluation pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from .agent365_otel import (
    NormalizationResult,
    filter_sessions_to_window,
    normalize_agent365_records,
)
from .evaluation import EvaluationRun, EvaluationRunner, RunnerConfig
from .judge import AsyncJudge
from .metrics import MetricSpec, TASK_COMPLETION_V1
from .models import AgentKey, EvaluationWindow, SampleBatch, SamplePolicy, SessionizationPolicy
from .reporting import EvaluationReport, build_report
from .sampling import SamplingEngine


@dataclass(frozen=True)
class PipelineRun:
    normalization: NormalizationResult
    sample_batch: SampleBatch
    evaluation: EvaluationRun
    report: EvaluationReport


@dataclass(frozen=True)
class RandomSamplingPipeline:
    judge: AsyncJudge
    sample_policy: SamplePolicy = SamplePolicy()
    metrics: tuple[MetricSpec, ...] = (TASK_COMPLETION_V1,)
    runner_config: RunnerConfig = RunnerConfig()
    max_evidence_bytes: int = 32768
    sessionization_policy: SessionizationPolicy = SessionizationPolicy()
    window_duration: timedelta = timedelta(hours=24)

    def __post_init__(self) -> None:
        if self.max_evidence_bytes <= 0:
            raise ValueError("max_evidence_bytes must be > 0")
        if self.window_duration < timedelta(seconds=1):
            raise ValueError("window_duration must be at least one second")
        if not self.metrics:
            raise ValueError("metrics must not be empty")

    async def run(
        self,
        records: Iterable[Any],
        capacities: Mapping[AgentKey, int] | None = None,
        tenant_capacities: Mapping[str, int] | None = None,
        agents: Iterable[AgentKey] | None = None,
        evaluation_window: EvaluationWindow | None = None,
        window_end: datetime | None = None,
    ) -> PipelineRun:
        if evaluation_window is not None and window_end is not None:
            raise ValueError("evaluation_window and window_end are mutually exclusive")

        normalization = normalize_agent365_records(
            records,
            sessionization_policy=self.sessionization_policy,
        )
        if evaluation_window is not None:
            window = evaluation_window
        else:
            end_at = (
                window_end
                if window_end is not None
                else self.default_evaluation_window(datetime.now(timezone.utc)).end_at
            )
            window = EvaluationWindow.ending_at(end_at, self.window_duration)

        filtered = filter_sessions_to_window(normalization, window)
        sample_batch = SamplingEngine().sample(
            units=filtered.units,
            policy=self.sample_policy,
            capacities=capacities,
            tenant_capacities=tenant_capacities,
            agents=agents,
            evaluation_window=window,
        )
        runner = EvaluationRunner(
            judge=self.judge,
            config=self.runner_config,
            max_evidence_bytes=self.max_evidence_bytes,
        )
        evaluation = await runner.run(sample_batch, self.metrics)
        report = build_report(
            sample_batch,
            evaluation,
            ingest_issue_count=len(filtered.issues),
        )
        return PipelineRun(
            normalization=filtered,
            sample_batch=sample_batch,
            evaluation=evaluation,
            report=report,
        )

    def default_evaluation_window(self, current_time: datetime) -> EvaluationWindow:
        """Return the most recently completed, UTC-aligned schedule window."""
        if current_time.tzinfo is None or current_time.utcoffset() is None:
            raise ValueError("current_time must be timezone-aware")
        current_utc = current_time.astimezone(timezone.utc)
        interval_seconds = int(self.window_duration.total_seconds())
        epoch_seconds = int(current_utc.timestamp())
        aligned_seconds = epoch_seconds - (epoch_seconds % interval_seconds)
        end_at = datetime.fromtimestamp(aligned_seconds, tz=timezone.utc)
        return EvaluationWindow.ending_at(end_at, self.window_duration)
