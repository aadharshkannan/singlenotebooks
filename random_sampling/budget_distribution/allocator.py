"""Deterministic hierarchical token allocation with capped Hamilton surplus."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .models import (
    AllocationNode,
    AllocationResult,
    FairnessState,
    SessionDemand,
    TenantAgentKey,
    stable_sha256_hex,
)


@dataclass(frozen=True)
class AllocationConfig:
    tenant_floor_tokens: int = 0
    agent_floor_tokens: int = 0
    fairness_deficit_cap_tokens: int = 5_000_000

    def __post_init__(self) -> None:
        if self.tenant_floor_tokens < 0:
            raise ValueError("tenant_floor_tokens must be non-negative")
        if self.agent_floor_tokens < 0:
            raise ValueError("agent_floor_tokens must be non-negative")
        if self.fairness_deficit_cap_tokens <= 0:
            raise ValueError("fairness_deficit_cap_tokens must be positive")


def _scaled_feasible_floors(
    keys: tuple[str, ...],
    floor_by_key: dict[str, int],
    budget: int,
    demand_by_key: dict[str, int],
    deficit_by_key: dict[str, int],
) -> dict[str, int]:
    if not keys or budget <= 0:
        return {key: 0 for key in keys}
    total_floor = sum(max(0, floor_by_key.get(key, 0)) for key in keys)
    if total_floor <= budget:
        return {
            key: min(max(0, floor_by_key.get(key, 0)), demand_by_key.get(key, 0))
            for key in keys
        }

    ideal = {
        key: (max(0, floor_by_key.get(key, 0)) * budget / total_floor)
        for key in keys
    }
    floors = {
        key: min(math.floor(value), demand_by_key.get(key, 0))
        for key, value in ideal.items()
    }
    remaining = budget - sum(floors.values())
    order = sorted(
        keys,
        key=lambda key: (
            -(ideal[key] - math.floor(ideal[key])),
            -deficit_by_key.get(key, 0),
            key,
        ),
    )
    for key in order:
        if remaining <= 0:
            break
        if floors[key] < demand_by_key.get(key, 0):
            floors[key] += 1
            remaining -= 1
    return floors


def _capped_hamilton_allocate(
    *,
    keys: tuple[str, ...],
    demand_by_key: dict[str, int],
    budget: int,
    floor_by_key: dict[str, int],
    deficit_by_key: dict[str, int],
) -> dict[str, int]:
    grants = _scaled_feasible_floors(
        keys=keys,
        floor_by_key=floor_by_key,
        budget=budget,
        demand_by_key=demand_by_key,
        deficit_by_key=deficit_by_key,
    )
    used = sum(grants.values())
    remaining_budget = max(0, budget - used)

    unmet = {
        key: max(0, demand_by_key.get(key, 0) - grants.get(key, 0))
        for key in keys
    }
    total_unmet = sum(unmet.values())
    if remaining_budget == 0 or total_unmet == 0:
        return grants

    ideal = {
        key: remaining_budget * unmet[key] / total_unmet if total_unmet else 0.0
        for key in keys
    }
    additional = {
        key: min(unmet[key], math.floor(ideal[key]))
        for key in keys
    }
    for key in keys:
        grants[key] = grants.get(key, 0) + additional[key]

    left = budget - sum(grants.values())
    if left <= 0:
        return grants

    order = sorted(
        keys,
        key=lambda key: (
            -(ideal[key] - math.floor(ideal[key])),
            -deficit_by_key.get(key, 0),
            key,
        ),
    )
    for key in order:
        if left <= 0:
            break
        if grants[key] < demand_by_key.get(key, 0):
            grants[key] += 1
            left -= 1
    return grants


def allocate_hierarchical_tokens(
    *,
    sessions: tuple[SessionDemand, ...],
    total_budget_tokens: int,
    fairness_state: FairnessState,
    config: AllocationConfig | None = None,
) -> tuple[AllocationResult, FairnessState]:
    cfg = config or AllocationConfig()
    if total_budget_tokens < 0:
        raise ValueError("total_budget_tokens must be non-negative")

    demand_by_tenant: dict[str, int] = {}
    demand_by_agent: dict[TenantAgentKey, int] = {}
    for session in sessions:
        demand_by_tenant[session.tenant_id] = (
            demand_by_tenant.get(session.tenant_id, 0) + session.total_cost_tokens
        )
        key = session.key
        demand_by_agent[key] = demand_by_agent.get(key, 0) + session.total_cost_tokens

    tenant_keys = tuple(sorted(demand_by_tenant))
    tenant_floor = {key: cfg.tenant_floor_tokens for key in tenant_keys}
    tenant_deficits = {
        key: fairness_state.tenant_deficit(key)
        for key in tenant_keys
    }
    tenant_grants = _capped_hamilton_allocate(
        keys=tenant_keys,
        demand_by_key=demand_by_tenant,
        budget=total_budget_tokens,
        floor_by_key=tenant_floor,
        deficit_by_key=tenant_deficits,
    )

    tenant_nodes: list[AllocationNode] = []
    agent_nodes: dict[TenantAgentKey, AllocationNode] = {}
    for tenant_id in tenant_keys:
        tenant_demand = demand_by_tenant.get(tenant_id, 0)
        tenant_grant = tenant_grants.get(tenant_id, 0)
        tenant_nodes.append(
            AllocationNode(
                key=tenant_id,
                demand_tokens=tenant_demand,
                floor_tokens=tenant_floor.get(tenant_id, 0),
                grant_tokens=tenant_grant,
                deficit_priority=tenant_deficits.get(tenant_id, 0),
            )
        )

        agent_keys = tuple(
            sorted(
                key
                for key in demand_by_agent
                if key.tenant_id == tenant_id
            )
        )
        agent_key_text = {
            key: f"{key.tenant_id}/{key.agent_id}"
            for key in agent_keys
        }
        demand_text = {
            agent_key_text[key]: demand_by_agent[key]
            for key in agent_keys
        }
        floor_text = {
            agent_key_text[key]: cfg.agent_floor_tokens
            for key in agent_keys
        }
        deficit_text = {
            agent_key_text[key]: fairness_state.agent_deficit(key)
            for key in agent_keys
        }

        agent_grants_text = _capped_hamilton_allocate(
            keys=tuple(sorted(demand_text)),
            demand_by_key=demand_text,
            budget=tenant_grant,
            floor_by_key=floor_text,
            deficit_by_key=deficit_text,
        )
        for key in agent_keys:
            text_key = agent_key_text[key]
            agent_nodes[key] = AllocationNode(
                key=text_key,
                demand_tokens=demand_by_agent[key],
                floor_tokens=floor_text[text_key],
                grant_tokens=agent_grants_text.get(text_key, 0),
                deficit_priority=deficit_text.get(text_key, 0),
            )

    total_granted = sum(node.grant_tokens for node in agent_nodes.values())
    unallocated = max(0, total_budget_tokens - total_granted)
    zero_grant_agents = sum(1 for node in agent_nodes.values() if node.grant_tokens == 0)

    next_tenant_deficits: dict[str, int] = {}
    next_agent_deficits: dict[str, int] = {}
    for node in tenant_nodes:
        shortfall = max(0, node.demand_tokens - node.grant_tokens)
        updated = shortfall + max(0, tenant_deficits.get(node.key, 0) // 2)
        next_tenant_deficits[node.key] = min(cfg.fairness_deficit_cap_tokens, updated)
    for key, node in agent_nodes.items():
        shortfall = max(0, node.demand_tokens - node.grant_tokens)
        updated = shortfall + max(0, fairness_state.agent_deficit(key) // 2)
        next_agent_deficits[f"{key.tenant_id}/{key.agent_id}"] = min(
            cfg.fairness_deficit_cap_tokens,
            updated,
        )

    fairness_in_hash = stable_sha256_hex(
        *(
            [f"tenant:{k}:{tenant_deficits.get(k, 0)}" for k in sorted(tenant_deficits)]
            + [f"agent:{k}:{v}" for k, v in sorted(fairness_state.agent_deficit_tokens.items())]
        )
    )
    fairness_out_hash = stable_sha256_hex(
        *(
            [f"tenant:{k}:{v}" for k, v in sorted(next_tenant_deficits.items())]
            + [f"agent:{k}:{v}" for k, v in sorted(next_agent_deficits.items())]
        )
    )

    result = AllocationResult(
        total_budget_tokens=total_budget_tokens,
        tenant_nodes=tuple(sorted(tenant_nodes, key=lambda n: n.key)),
        agent_nodes=agent_nodes,
        unallocated_tokens=unallocated,
        zero_grant_agents=zero_grant_agents,
        fairness_input_hash=fairness_in_hash,
        fairness_output_hash=fairness_out_hash,
    )
    next_state = FairnessState(
        tenant_deficit_tokens=next_tenant_deficits,
        agent_deficit_tokens=next_agent_deficits,
    )
    return result, next_state
