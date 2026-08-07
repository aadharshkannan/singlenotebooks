"""Core immutable models for deterministic token-budget distribution."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Mapping


def _to_utc(name: str, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def stable_sha256_hex(*parts: str) -> str:
    material = "||".join(parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True, order=True)
class TenantAgentKey:
    tenant_id: str
    agent_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must not be blank")
        if not self.agent_id.strip():
            raise ValueError("agent_id must not be blank")


@dataclass(frozen=True)
class SessionDemand:
    """Structured deterministic session demand; labels/outcomes are excluded."""

    tenant_id: str
    agent_id: str
    session_id: str
    completed_at: datetime
    ingested_at: datetime
    estimated_input_tokens: int
    expected_output_tokens: int
    session_version: str = "v1"

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must not be blank")
        if not self.agent_id.strip():
            raise ValueError("agent_id must not be blank")
        if not self.session_id.strip():
            raise ValueError("session_id must not be blank")
        object.__setattr__(self, "completed_at", _to_utc("completed_at", self.completed_at))
        object.__setattr__(self, "ingested_at", _to_utc("ingested_at", self.ingested_at))
        if self.estimated_input_tokens < 0:
            raise ValueError("estimated_input_tokens must be non-negative")
        if self.expected_output_tokens < 0:
            raise ValueError("expected_output_tokens must be non-negative")
        if self.total_cost_tokens <= 0:
            raise ValueError("total session token cost must be positive")
        if not self.session_version.strip():
            raise ValueError("session_version must not be blank")

    @property
    def key(self) -> TenantAgentKey:
        return TenantAgentKey(tenant_id=self.tenant_id, agent_id=self.agent_id)

    @property
    def dedup_key(self) -> str:
        return f"{self.tenant_id}/{self.agent_id}/{self.session_id}/{self.session_version}"

    @property
    def total_cost_tokens(self) -> int:
        return self.estimated_input_tokens + self.expected_output_tokens


@dataclass(frozen=True)
class BatchWindow:
    previous_successful_watermark: datetime
    cutoff: datetime
    source_scan_start: datetime
    elapsed_minutes: float
    clamped_minutes: float
    max_catchup_minutes: float
    bootstrap_used: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "previous_successful_watermark", _to_utc("previous_successful_watermark", self.previous_successful_watermark))
        object.__setattr__(self, "cutoff", _to_utc("cutoff", self.cutoff))
        object.__setattr__(self, "source_scan_start", _to_utc("source_scan_start", self.source_scan_start))
        if self.previous_successful_watermark >= self.cutoff:
            raise ValueError("previous_successful_watermark must be before cutoff")
        if self.source_scan_start > self.previous_successful_watermark:
            raise ValueError("source_scan_start must be <= previous_successful_watermark")
        if self.elapsed_minutes <= 0:
            raise ValueError("elapsed_minutes must be positive")
        if self.clamped_minutes <= 0:
            raise ValueError("clamped_minutes must be positive")
        if self.max_catchup_minutes <= 0:
            raise ValueError("max_catchup_minutes must be positive")

    @property
    def window_start(self) -> datetime:
        return self.previous_successful_watermark

    @property
    def window_end(self) -> datetime:
        return self.cutoff


@dataclass(frozen=True)
class BatchBudget:
    tpm_limit: int
    nominal_tokens: int
    safety_deduction_tokens: int
    retry_deduction_tokens: int
    output_deduction_tokens: int
    effective_tokens: int

    def __post_init__(self) -> None:
        if self.tpm_limit <= 0:
            raise ValueError("tpm_limit must be positive")
        for name, value in (
            ("nominal_tokens", self.nominal_tokens),
            ("safety_deduction_tokens", self.safety_deduction_tokens),
            ("retry_deduction_tokens", self.retry_deduction_tokens),
            ("output_deduction_tokens", self.output_deduction_tokens),
            ("effective_tokens", self.effective_tokens),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class EligibleFrame:
    window_start: datetime
    window_end: datetime
    source_scan_start: datetime
    sessions: tuple[SessionDemand, ...]
    frame_hash: str
    membership_hash: str
    canonical_count: int
    lookback_admitted_count: int
    duplicate_count: int
    processed_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_start", _to_utc("window_start", self.window_start))
        object.__setattr__(self, "window_end", _to_utc("window_end", self.window_end))
        object.__setattr__(self, "source_scan_start", _to_utc("source_scan_start", self.source_scan_start))
        if self.window_start >= self.window_end:
            raise ValueError("window_start must be before window_end")
        if self.source_scan_start > self.window_start:
            raise ValueError("source_scan_start must be <= window_start")


@dataclass(frozen=True)
class AllocationNode:
    key: str
    demand_tokens: int
    floor_tokens: int
    grant_tokens: int
    deficit_priority: int


@dataclass(frozen=True)
class AllocationResult:
    total_budget_tokens: int
    tenant_nodes: tuple[AllocationNode, ...]
    agent_nodes: Mapping[TenantAgentKey, AllocationNode]
    unallocated_tokens: int
    zero_grant_agents: int
    fairness_input_hash: str
    fairness_output_hash: str


@dataclass(frozen=True)
class SelectionRecord:
    demand: SessionDemand
    rank_hash: str
    selected: bool
    reason: str


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[SelectionRecord, ...]
    unselected: tuple[SelectionRecord, ...]
    selected_total_tokens: int
    slack_tokens: int
    redistribution_rounds: int
    unserviceable_count: int
    too_large_for_agent_count: int
    selected_ids: tuple[str, ...]


@dataclass(frozen=True)
class FairnessState:
    tenant_deficit_tokens: Mapping[str, int] = field(default_factory=dict)
    agent_deficit_tokens: Mapping[str, int] = field(default_factory=dict)

    def tenant_deficit(self, tenant_id: str) -> int:
        return int(self.tenant_deficit_tokens.get(tenant_id, 0))

    def agent_deficit(self, key: TenantAgentKey) -> int:
        return int(self.agent_deficit_tokens.get(f"{key.tenant_id}/{key.agent_id}", 0))


@dataclass(frozen=True)
class ExecutionScheduleEntry:
    request_id: str
    tenant_id: str
    agent_id: str
    session_id: str
    reserved_tokens: int
    scheduled_offset_seconds: float


@dataclass(frozen=True)
class BatchPlan:
    pipeline_id: str
    batch_id: str
    seed: str
    window: BatchWindow
    budget: BatchBudget
    frame: EligibleFrame
    allocation: AllocationResult
    selection: SelectionResult
    schedule: tuple[ExecutionScheduleEntry, ...]
    planned_usage_tokens: int
    slack_tokens: int
    reallocation_rounds: int
    coverage_ratio: float
    zero_allocations: int
    fairness_state_out: FairnessState
    config_hash: str
    frame_hash: str
    membership_hash: str

    def __post_init__(self) -> None:
        if not self.pipeline_id.strip():
            raise ValueError("pipeline_id must not be blank")
        if not self.batch_id.strip():
            raise ValueError("batch_id must not be blank")
        if not self.seed.strip():
            raise ValueError("seed must not be blank")
        if self.planned_usage_tokens < 0:
            raise ValueError("planned_usage_tokens must be non-negative")
        if self.slack_tokens < 0:
            raise ValueError("slack_tokens must be non-negative")
