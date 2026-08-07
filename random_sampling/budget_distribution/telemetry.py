"""Telemetry helpers for deterministic budget-distribution planning."""
from __future__ import annotations

from dataclasses import dataclass

from .models import AllocationResult, SelectionResult


@dataclass(frozen=True)
class BatchTelemetry:
    utilization: float
    coverage: float
    slack_tokens: int
    selected_count: int
    fairness_jain: float
    zero_allocations: int
    tpm_compliance: bool


def jain_fairness(values: list[float]) -> float:
    if not values:
        return 1.0
    numerator = sum(values) ** 2
    denominator = len(values) * sum(value * value for value in values)
    if denominator == 0:
        return 1.0
    return numerator / denominator


def build_batch_telemetry(
    *,
    allocation: AllocationResult,
    selection: SelectionResult,
    total_eligible_sessions: int,
    tpm_compliance: bool,
) -> BatchTelemetry:
    selected_count = len(selection.selected)
    utilization = (
        selection.selected_total_tokens / allocation.total_budget_tokens
        if allocation.total_budget_tokens
        else 0.0
    )
    coverage = selected_count / total_eligible_sessions if total_eligible_sessions else 0.0
    grants = [float(node.grant_tokens) for node in allocation.agent_nodes.values()]
    return BatchTelemetry(
        utilization=utilization,
        coverage=coverage,
        slack_tokens=selection.slack_tokens,
        selected_count=selected_count,
        fairness_jain=jain_fairness(grants),
        zero_allocations=allocation.zero_grant_agents,
        tpm_compliance=tpm_compliance,
    )
