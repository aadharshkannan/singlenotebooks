"""Prototype for deterministic uniform sampling within each tenant/agent.

Sampling membership intentionally ignores token cost. Cost is used only after
selection to pace execution and identify requests that cannot fit the TPM
window. This preserves equal inclusion probability within every agent stratum.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from math import sqrt
from pathlib import Path
import os
import tempfile
from typing import Iterable


def _sha256(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class SessionCandidate:
    tenant_id: str
    agent_id: str
    session_id: str
    session_version: str
    estimated_tokens: int

    def __post_init__(self) -> None:
        for name in ("tenant_id", "agent_id", "session_id", "session_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        if self.estimated_tokens <= 0:
            raise ValueError("estimated_tokens must be positive")

    @property
    def stratum_key(self) -> str:
        return f"{self.tenant_id}/{self.agent_id}"

    @property
    def dedup_key(self) -> str:
        return f"{self.stratum_key}/{self.session_id}/{self.session_version}"


def stable_rank(*, seed: str, candidate: SessionCandidate) -> str:
    """Return a reproducible random ordering key that does not use cost."""
    if not seed.strip():
        raise ValueError("seed must not be blank")
    return _sha256(
        seed,
        candidate.tenant_id,
        candidate.agent_id,
        candidate.session_id,
        candidate.session_version,
    )


@dataclass(frozen=True)
class SampledSession:
    candidate: SessionCandidate
    rank_hash: str
    inclusion_probability: float
    population_size: int
    sample_size: int


@dataclass(frozen=True)
class AgentSample:
    tenant_id: str
    agent_id: str
    population_size: int
    sample_size: int
    inclusion_probability: float
    seed: str
    selected: tuple[SampledSession, ...]

    @property
    def stratum_key(self) -> str:
        return f"{self.tenant_id}/{self.agent_id}"


def uniformly_sample_by_agent(
    *,
    candidates: Iterable[SessionCandidate],
    sample_size_per_agent: int,
    seed: str,
) -> tuple[AgentSample, ...]:
    """Take a deterministic simple random sample without replacement per agent.

    Each candidate in an agent stratum has exactly ``min(n, N) / N`` inclusion
    probability. ``estimated_tokens`` is deliberately absent from all ranking
    and membership decisions.
    """
    if sample_size_per_agent <= 0:
        raise ValueError("sample_size_per_agent must be positive")
    if not seed.strip():
        raise ValueError("seed must not be blank")

    grouped: dict[tuple[str, str], list[SessionCandidate]] = {}
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.dedup_key in seen:
            raise ValueError(f"duplicate candidate: {candidate.dedup_key}")
        seen.add(candidate.dedup_key)
        grouped.setdefault((candidate.tenant_id, candidate.agent_id), []).append(candidate)

    samples: list[AgentSample] = []
    for tenant_id, agent_id in sorted(grouped):
        population = grouped[(tenant_id, agent_id)]
        population_size = len(population)
        sample_size = min(sample_size_per_agent, population_size)
        probability = sample_size / population_size
        ranked = sorted(
            ((stable_rank(seed=seed, candidate=candidate), candidate) for candidate in population),
            key=lambda item: (item[0], item[1].session_id, item[1].session_version),
        )
        selected = tuple(
            SampledSession(
                candidate=candidate,
                rank_hash=rank_hash,
                inclusion_probability=probability,
                population_size=population_size,
                sample_size=sample_size,
            )
            for rank_hash, candidate in ranked[:sample_size]
        )
        samples.append(
            AgentSample(
                tenant_id=tenant_id,
                agent_id=agent_id,
                population_size=population_size,
                sample_size=sample_size,
                inclusion_probability=probability,
                seed=seed,
                selected=selected,
            )
        )
    return tuple(samples)


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    COMPLETED = "COMPLETED"
    OVERSIZED = "OVERSIZED"
    DROPPED = "DROPPED"


@dataclass(frozen=True)
class QueueItem:
    request_id: str
    sampled: SampledSession
    sampling_seed: str
    status: ExecutionStatus
    scheduled_at_seconds: float | None = None
    score: float | None = None


class ExecutionQueue:
    """JSON-backed selected-sample queue with deterministic rolling-TPM pacing."""

    def __init__(
        self,
        path: str | Path,
        *,
        tpm_limit: int = 20_000,
        max_schedule_delay_seconds: float | None = None,
    ):
        if tpm_limit <= 0:
            raise ValueError("tpm_limit must be positive")
        if max_schedule_delay_seconds is not None and max_schedule_delay_seconds < 0:
            raise ValueError("max_schedule_delay_seconds must be non-negative")
        self.path = Path(path)
        self.tpm_limit = tpm_limit
        self.max_schedule_delay_seconds = max_schedule_delay_seconds
        if not self.path.exists():
            self._write(
                {
                    "schema_version": "agent-uniform-v1",
                    "tpm_limit": tpm_limit,
                    "max_schedule_delay_seconds": max_schedule_delay_seconds,
                    "items": {},
                    "sampling_runs": {},
                }
            )
        data = self._read()
        self._normalize_data(data)
        if data["tpm_limit"] != tpm_limit:
            raise ValueError("queue tpm_limit does not match existing queue")
        if data["max_schedule_delay_seconds"] != max_schedule_delay_seconds:
            raise ValueError("queue max_schedule_delay_seconds does not match existing queue")
        self._write(data)

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def _normalize_data(data: dict) -> None:
        data.setdefault("schema_version", "agent-uniform-v1")
        data.setdefault("max_schedule_delay_seconds", None)
        data.setdefault("items", {})
        data.setdefault("sampling_runs", {})

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def enqueue(self, samples: Iterable[AgentSample]) -> tuple[QueueItem, ...]:
        data = self._read()
        self._normalize_data(data)
        for agent_sample in samples:
            run_id = _sha256("sampling-run", agent_sample.stratum_key, agent_sample.seed)[:24]
            selected_request_ids = [
                _sha256("agent-uniform-v1", sampled.candidate.dedup_key, agent_sample.seed)[:24]
                for sampled in agent_sample.selected
            ]
            run = data["sampling_runs"].get(run_id)
            if run is None:
                run = {
                    "run_id": run_id,
                    "tenant_id": agent_sample.tenant_id,
                    "agent_id": agent_sample.agent_id,
                    "stratum_key": agent_sample.stratum_key,
                    "population_size": agent_sample.population_size,
                    "sample_size": agent_sample.sample_size,
                    "inclusion_probability": agent_sample.inclusion_probability,
                    "seed": agent_sample.seed,
                    "selected_request_ids": [],
                }
                data["sampling_runs"][run_id] = run
            elif (
                run["population_size"] != agent_sample.population_size
                or run["sample_size"] != agent_sample.sample_size
                or run["inclusion_probability"] != agent_sample.inclusion_probability
                or set(run["selected_request_ids"]) != set(selected_request_ids)
            ):
                raise ValueError("sampling run does not match existing queue metadata")
            for sampled, request_id in zip(agent_sample.selected, selected_request_ids, strict=True):
                if request_id not in run["selected_request_ids"]:
                    run["selected_request_ids"].append(request_id)
                data["items"].setdefault(
                    request_id,
                    {
                        "request_id": request_id,
                        "sampled": asdict(sampled),
                        "sampling_seed": agent_sample.seed,
                        "status": ExecutionStatus.PENDING.value,
                        "scheduled_at_seconds": None,
                        "score": None,
                    },
                )
        self._write(data)
        return self.items()

    def schedule_pending(self) -> tuple[QueueItem, ...]:
        data = self._read()
        self._normalize_data(data)
        events: list[tuple[float, int]] = []
        cursor = 0.0
        for item in data["items"].values():
            if item["status"] == ExecutionStatus.SCHEDULED.value:
                events.append((float(item["scheduled_at_seconds"]), int(item["sampled"]["candidate"]["estimated_tokens"])))
                cursor = max(cursor, float(item["scheduled_at_seconds"]))
        for request_id in sorted(data["items"]):
            item = data["items"][request_id]
            if item["status"] != ExecutionStatus.PENDING.value:
                continue
            estimate = int(item["sampled"]["candidate"]["estimated_tokens"])
            if estimate > self.tpm_limit:
                item["status"] = ExecutionStatus.OVERSIZED.value
                continue
            scheduled_at = cursor
            while sum(tokens for timestamp, tokens in events if scheduled_at - 60 < timestamp <= scheduled_at) + estimate > self.tpm_limit:
                active = sorted(timestamp for timestamp, _ in events if timestamp > scheduled_at - 60)
                scheduled_at = active[0] + 60
            if self.max_schedule_delay_seconds is not None and scheduled_at > self.max_schedule_delay_seconds:
                item["status"] = ExecutionStatus.DROPPED.value
                continue
            events.append((scheduled_at, estimate))
            item["status"] = ExecutionStatus.SCHEDULED.value
            item["scheduled_at_seconds"] = scheduled_at
            cursor = max(cursor, scheduled_at)
        self._write(data)
        return self.items()

    def complete(self, request_id: str, *, score: float) -> QueueItem:
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        data = self._read()
        self._normalize_data(data)
        item = data["items"][request_id]
        if item["status"] != ExecutionStatus.SCHEDULED.value:
            raise ValueError("only scheduled requests can be completed")
        item["status"] = ExecutionStatus.COMPLETED.value
        item["score"] = score
        self._write(data)
        return self._to_item(item)

    def items(self) -> tuple[QueueItem, ...]:
        data = self._read()
        self._normalize_data(data)
        return tuple(self._to_item(data["items"][request_id]) for request_id in sorted(data["items"]))

    @staticmethod
    def _to_item(data: dict) -> QueueItem:
        candidate = SessionCandidate(**data["sampled"]["candidate"])
        sampled = SampledSession(
            candidate=candidate,
            rank_hash=data["sampled"]["rank_hash"],
            inclusion_probability=float(data["sampled"]["inclusion_probability"]),
            population_size=int(data["sampled"]["population_size"]),
            sample_size=int(data["sampled"]["sample_size"]),
        )
        return QueueItem(
            request_id=data["request_id"],
            sampled=sampled,
            sampling_seed=data["sampling_seed"],
            status=ExecutionStatus(data["status"]),
            scheduled_at_seconds=data["scheduled_at_seconds"],
            score=data["score"],
        )


@dataclass(frozen=True)
class AgentScoreSummary:
    tenant_id: str
    agent_id: str
    population_size: int
    selected_count: int
    completed_count: int
    inclusion_probability: float
    mean_score: float | None
    confidence_interval_95: tuple[float, float] | None


def summarize_agent_scores(items: Iterable[QueueItem]) -> tuple[AgentScoreSummary, ...]:
    """Report sampled results per agent; no fleet-level aggregation is produced."""
    groups: dict[tuple[str, str], list[QueueItem]] = {}
    for item in items:
        candidate = item.sampled.candidate
        groups.setdefault((candidate.tenant_id, candidate.agent_id), []).append(item)

    summaries: list[AgentScoreSummary] = []
    for tenant_id, agent_id in sorted(groups):
        group = groups[(tenant_id, agent_id)]
        scores = [item.score for item in group if item.status == ExecutionStatus.COMPLETED and item.score is not None]
        mean = sum(scores) / len(scores) if scores else None
        interval: tuple[float, float] | None = None
        if len(scores) >= 2 and mean is not None:
            variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
            population_size = group[0].sampled.population_size
            finite_population_correction = sqrt((population_size - len(scores)) / (population_size - 1)) if population_size > 1 else 0.0
            margin = 1.96 * sqrt(variance / len(scores)) * finite_population_correction
            interval = (max(0.0, mean - margin), min(1.0, mean + margin))
        exemplar = group[0].sampled
        summaries.append(
            AgentScoreSummary(
                tenant_id=tenant_id,
                agent_id=agent_id,
                population_size=exemplar.population_size,
                selected_count=len(group),
                completed_count=len(scores),
                inclusion_probability=exemplar.inclusion_probability,
                mean_score=mean,
                confidence_interval_95=interval,
            )
        )
    return tuple(summaries)