"""Batch window, budget, and eligible-frame construction."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math

from .allocator import AllocationConfig, allocate_hierarchical_tokens
from .models import BatchBudget, BatchWindow, EligibleFrame, SessionDemand, stable_sha256_hex
from .models import BatchPlan, ExecutionScheduleEntry, FairnessState
from .pacing import RollingTokenPacer
from .selection import AgentSelectionGrant, select_within_agent_grants


@dataclass(frozen=True)
class BudgetDeductions:
    safety_tokens: int = 0
    retry_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("safety_tokens", self.safety_tokens),
            ("retry_tokens", self.retry_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def resolve_batch_window(
    previous_successful_watermark: datetime | None,
    cutoff: datetime,
    *,
    bootstrap_watermark: datetime | None = None,
    lookback: timedelta = timedelta(0),
    max_catchup_minutes: float = 60.0,
) -> BatchWindow:
    """Resolve a half-open UTC canonical window [previous, cutoff)."""
    cutoff_utc = _utc(cutoff, "cutoff")
    if max_catchup_minutes <= 0:
        raise ValueError("max_catchup_minutes must be positive")
    if lookback < timedelta(0):
        raise ValueError("lookback must be non-negative")

    if previous_successful_watermark is None:
        if bootstrap_watermark is None:
            raise ValueError(
                "bootstrap_watermark is required when no previous successful watermark exists"
            )
        previous_utc = _utc(bootstrap_watermark, "bootstrap_watermark")
        bootstrap_used = True
    else:
        previous_utc = _utc(previous_successful_watermark, "previous_successful_watermark")
        bootstrap_used = False

    if previous_utc >= cutoff_utc:
        raise ValueError("previous watermark must be before cutoff")

    elapsed_minutes = (cutoff_utc - previous_utc).total_seconds() / 60.0
    clamped_minutes = min(elapsed_minutes, max_catchup_minutes)
    source_scan_start = previous_utc - lookback
    return BatchWindow(
        previous_successful_watermark=previous_utc,
        cutoff=cutoff_utc,
        source_scan_start=source_scan_start,
        elapsed_minutes=elapsed_minutes,
        clamped_minutes=clamped_minutes,
        max_catchup_minutes=max_catchup_minutes,
        bootstrap_used=bootstrap_used,
    )


def calculate_batch_budget(
    clamped_minutes: float | None = None,
    *,
    window: BatchWindow | None = None,
    tpm_limit: int = 20_000,
    deductions: BudgetDeductions | None = None,
) -> BatchBudget:
    """Convert clamped fractional minutes into an effective token budget."""
    if window is not None:
        minutes = window.clamped_minutes
        if clamped_minutes is not None:
            raise ValueError("provide either window or clamped_minutes, not both")
    else:
        if clamped_minutes is None:
            raise ValueError("clamped_minutes is required when window is not provided")
        minutes = clamped_minutes

    if minutes <= 0:
        raise ValueError("clamped_minutes must be positive")
    if tpm_limit <= 0:
        raise ValueError("tpm_limit must be positive")

    d = deductions or BudgetDeductions()
    nominal_tokens = int(math.floor(minutes * tpm_limit))
    effective_tokens = max(
        0,
        nominal_tokens - d.safety_tokens - d.retry_tokens - d.output_tokens,
    )
    return BatchBudget(
        tpm_limit=tpm_limit,
        nominal_tokens=nominal_tokens,
        safety_deduction_tokens=d.safety_tokens,
        retry_deduction_tokens=d.retry_tokens,
        output_deduction_tokens=d.output_tokens,
        effective_tokens=effective_tokens,
    )


def build_eligible_frame(
    *,
    source_sessions: list[SessionDemand],
    window: BatchWindow,
    processed_session_keys: set[str],
) -> EligibleFrame:
    """Build deterministic eligible frame from canonical window plus lookback scan."""
    canonical: list[SessionDemand] = []
    lookback_admitted = 0
    duplicate_count = 0
    processed_count = 0

    seen_dedup: set[str] = set()
    for session in source_sessions:
        in_scan = window.source_scan_start <= session.completed_at < window.cutoff
        if not in_scan:
            continue

        in_canonical = (
            window.previous_successful_watermark
            <= session.completed_at
            < window.cutoff
        )
        in_lookback = (
            window.source_scan_start
            <= session.completed_at
            < window.previous_successful_watermark
        )
        if not in_canonical and not in_lookback:
            continue

        if session.dedup_key in seen_dedup:
            duplicate_count += 1
            continue
        seen_dedup.add(session.dedup_key)

        if session.dedup_key in processed_session_keys:
            processed_count += 1
            continue

        if in_lookback:
            lookback_admitted += 1

        canonical.append(session)

    canonical.sort(
        key=lambda s: (
            s.completed_at,
            s.tenant_id,
            s.agent_id,
            s.session_id,
            s.session_version,
        )
    )

    membership_material = [
        f"{s.completed_at.isoformat()}|{s.tenant_id}|{s.agent_id}|{s.session_id}|{s.session_version}|{s.total_cost_tokens}"
        for s in canonical
    ]
    membership_hash = stable_sha256_hex(*membership_material) if membership_material else stable_sha256_hex("empty")
    frame_hash = stable_sha256_hex(
        window.previous_successful_watermark.isoformat(),
        window.cutoff.isoformat(),
        window.source_scan_start.isoformat(),
        membership_hash,
    )
    return EligibleFrame(
        window_start=window.previous_successful_watermark,
        window_end=window.cutoff,
        source_scan_start=window.source_scan_start,
        sessions=tuple(canonical),
        frame_hash=frame_hash,
        membership_hash=membership_hash,
        canonical_count=len(canonical),
        lookback_admitted_count=lookback_admitted,
        duplicate_count=duplicate_count,
        processed_count=processed_count,
    )


def build_batch_plan(
    *,
    pipeline_id: str,
    batch_id: str,
    seed: str,
    window: BatchWindow,
    budget: BatchBudget,
    frame: EligibleFrame,
    fairness_state: FairnessState,
    allocation_config: AllocationConfig | None = None,
) -> BatchPlan:
    if not pipeline_id.strip():
        raise ValueError("pipeline_id must not be blank")
    if not batch_id.strip():
        raise ValueError("batch_id must not be blank")
    if not seed.strip():
        raise ValueError("seed must not be blank")

    allocation, _ = allocate_hierarchical_tokens(
        sessions=frame.sessions,
        total_budget_tokens=budget.effective_tokens,
        fairness_state=fairness_state,
        config=allocation_config,
    )

    grants = tuple(
        AgentSelectionGrant(
            key=key,
            grant_tokens=node.grant_tokens,
            deficit_priority=node.deficit_priority,
        )
        for key, node in sorted(
            allocation.agent_nodes.items(),
            key=lambda pair: (pair[0].tenant_id, pair[0].agent_id),
        )
    )
    selection = select_within_agent_grants(
        sessions=frame.sessions,
        grants=grants,
        seed=seed,
        frame_hash=frame.frame_hash,
        effective_budget_tokens=budget.effective_tokens,
        max_reservable_tokens=budget.tpm_limit,
    )

    pacer = RollingTokenPacer(tpm_limit=budget.tpm_limit)
    schedule_items: list[ExecutionScheduleEntry] = []
    sorted_selected = sorted(
        selection.selected,
        key=lambda record: (
            record.demand.tenant_id,
            record.demand.agent_id,
            record.rank_hash,
            record.demand.session_id,
        ),
    )
    for record in sorted_selected:
        request_id = stable_sha256_hex(
            pipeline_id,
            batch_id,
            record.demand.tenant_id,
            record.demand.agent_id,
            record.demand.session_id,
            record.demand.session_version,
        )[:24]
        reservation = pacer.reserve(request_id, record.demand.total_cost_tokens)
        schedule_items.append(
            ExecutionScheduleEntry(
                request_id=request_id,
                tenant_id=record.demand.tenant_id,
                agent_id=record.demand.agent_id,
                session_id=record.demand.session_id,
                reserved_tokens=reservation.reserved_tokens,
                scheduled_offset_seconds=reservation.scheduled_at_seconds,
            )
        )

    planned_usage = selection.selected_total_tokens
    total_eligible = len(frame.sessions)
    coverage = len(selection.selected) / total_eligible if total_eligible else 0.0
    config_hash = stable_sha256_hex(
        f"tpm:{budget.tpm_limit}",
        f"nominal:{budget.nominal_tokens}",
        f"effective:{budget.effective_tokens}",
        f"safety:{budget.safety_deduction_tokens}",
        f"retry:{budget.retry_deduction_tokens}",
        f"output:{budget.output_deduction_tokens}",
        f"tenant_floor:{(allocation_config or AllocationConfig()).tenant_floor_tokens}",
        f"agent_floor:{(allocation_config or AllocationConfig()).agent_floor_tokens}",
    )

    cfg = allocation_config or AllocationConfig()
    served_by_tenant: dict[str, int] = {}
    served_by_agent: dict[str, int] = {}
    demand_by_tenant: dict[str, int] = {}
    demand_by_agent: dict[str, int] = {}
    for session in frame.sessions:
        demand_by_tenant[session.tenant_id] = demand_by_tenant.get(session.tenant_id, 0) + session.total_cost_tokens
        agent_key = f"{session.tenant_id}/{session.agent_id}"
        demand_by_agent[agent_key] = demand_by_agent.get(agent_key, 0) + session.total_cost_tokens
    for record in selection.selected:
        session = record.demand
        served_by_tenant[session.tenant_id] = served_by_tenant.get(session.tenant_id, 0) + session.total_cost_tokens
        agent_key = f"{session.tenant_id}/{session.agent_id}"
        served_by_agent[agent_key] = served_by_agent.get(agent_key, 0) + session.total_cost_tokens

    next_tenant_deficits: dict[str, int] = {}
    next_agent_deficits: dict[str, int] = {}
    for tenant_id in sorted(demand_by_tenant):
        shortfall = max(0, demand_by_tenant[tenant_id] - served_by_tenant.get(tenant_id, 0))
        carried = max(0, fairness_state.tenant_deficit(tenant_id) // 2)
        next_tenant_deficits[tenant_id] = min(cfg.fairness_deficit_cap_tokens, shortfall + carried)
    for agent_key in sorted(demand_by_agent):
        tenant_id, agent_id = agent_key.split("/", 1)
        shortfall = max(0, demand_by_agent[agent_key] - served_by_agent.get(agent_key, 0))
        carried = max(0, int(fairness_state.agent_deficit_tokens.get(agent_key, 0)) // 2)
        next_agent_deficits[agent_key] = min(cfg.fairness_deficit_cap_tokens, shortfall + carried)

    fairness_final = FairnessState(
        tenant_deficit_tokens=next_tenant_deficits,
        agent_deficit_tokens=next_agent_deficits,
    )

    return BatchPlan(
        pipeline_id=pipeline_id,
        batch_id=batch_id,
        seed=seed,
        window=window,
        budget=budget,
        frame=frame,
        allocation=allocation,
        selection=selection,
        schedule=tuple(schedule_items),
        planned_usage_tokens=planned_usage,
        slack_tokens=selection.slack_tokens,
        reallocation_rounds=selection.redistribution_rounds,
        coverage_ratio=coverage,
        zero_allocations=allocation.zero_grant_agents,
        fairness_state_out=fairness_final,
        config_hash=config_hash,
        frame_hash=frame.frame_hash,
        membership_hash=frame.membership_hash,
    )
