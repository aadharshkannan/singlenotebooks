"""Deterministic random ranking and whole-session token packing."""
from __future__ import annotations

from dataclasses import dataclass

from .models import (
    SelectionRecord,
    SelectionResult,
    SessionDemand,
    TenantAgentKey,
    stable_sha256_hex,
)


@dataclass(frozen=True)
class AgentSelectionGrant:
    key: TenantAgentKey
    grant_tokens: int
    deficit_priority: int = 0


def stable_rank_hash(
    *,
    seed: str,
    frame_hash: str,
    demand: SessionDemand,
) -> str:
    return stable_sha256_hex(
        seed,
        frame_hash,
        demand.tenant_id,
        demand.agent_id,
        demand.session_id,
        demand.session_version,
    )


def select_within_agent_grants(
    *,
    sessions: tuple[SessionDemand, ...],
    grants: tuple[AgentSelectionGrant, ...],
    seed: str,
    frame_hash: str,
    effective_budget_tokens: int,
    max_reservable_tokens: int | None = None,
) -> SelectionResult:
    grouped: dict[TenantAgentKey, list[SessionDemand]] = {}
    for session in sessions:
        grouped.setdefault(session.key, []).append(session)

    grant_map = {grant.key: grant.grant_tokens for grant in grants}
    deficit_map = {grant.key: grant.deficit_priority for grant in grants}

    selected_records: list[SelectionRecord] = []
    selected_keys: set[str] = set()
    selected_ids: list[str] = []
    unserviceable_keys: set[str] = set()

    remaining_by_agent: dict[TenantAgentKey, list[tuple[str, SessionDemand]]] = {}
    grant_used: dict[TenantAgentKey, int] = {key: 0 for key in grant_map}

    for key, items in grouped.items():
        ranked = sorted(
            ((stable_rank_hash(seed=seed, frame_hash=frame_hash, demand=item), item) for item in items),
            key=lambda pair: (pair[0], pair[1].session_id),
        )
        remaining_by_agent[key] = ranked

    for key in sorted(grant_map):
        grant = grant_map[key]
        ranked = remaining_by_agent.get(key, [])
        remaining_ranked: list[tuple[str, SessionDemand]] = []
        used = 0
        for rank_hash, demand in ranked:
            cost = demand.total_cost_tokens
            if cost > effective_budget_tokens or (max_reservable_tokens is not None and cost > max_reservable_tokens):
                unserviceable_keys.add(demand.dedup_key)
                continue
            if cost > grant:
                remaining_ranked.append((rank_hash, demand))
                continue
            if used + cost <= grant:
                used += cost
                if demand.dedup_key not in selected_keys:
                    selected_keys.add(demand.dedup_key)
                    selected_ids.append(demand.dedup_key)
                    selected_records.append(
                        SelectionRecord(demand=demand, rank_hash=rank_hash, selected=True, reason="selected_within_initial_grant")
                    )
            else:
                remaining_ranked.append((rank_hash, demand))
        remaining_by_agent[key] = remaining_ranked
        grant_used[key] = used

    unused_pool = sum(max(0, grant_map[key] - grant_used.get(key, 0)) for key in grant_map)
    redistribution_rounds = 0
    while unused_pool > 0:
        redistribution_rounds += 1
        progress = False
        priority_order = sorted(
            grant_map,
            key=lambda key: (-deficit_map.get(key, 0), key.tenant_id, key.agent_id),
        )
        for key in priority_order:
            ranked = remaining_by_agent.get(key, [])
            if not ranked:
                continue
            rank_hash, demand = ranked[0]
            cost = demand.total_cost_tokens
            if cost <= unused_pool:
                if demand.dedup_key not in selected_keys:
                    selected_keys.add(demand.dedup_key)
                    selected_ids.append(demand.dedup_key)
                    selected_records.append(
                        SelectionRecord(demand=demand, rank_hash=rank_hash, selected=True, reason="selected_after_redistribution")
                    )
                unused_pool -= cost
                remaining_by_agent[key] = ranked[1:]
                progress = True
            else:
                continue
        if not progress:
            break

    final_unselected: list[SelectionRecord] = []
    too_large_for_agent_count = 0
    for key in sorted(remaining_by_agent):
        grant = grant_map.get(key, 0)
        for rank_hash, demand in remaining_by_agent.get(key, []):
            if demand.dedup_key in selected_keys:
                continue
            reason = "too_large_for_agent_grant" if demand.total_cost_tokens > grant else "insufficient_remaining_agent_grant"
            if reason == "too_large_for_agent_grant":
                too_large_for_agent_count += 1
            final_unselected.append(
                SelectionRecord(demand=demand, rank_hash=rank_hash, selected=False, reason=reason)
            )

    for key, items in grouped.items():
        ranked = sorted(
            ((stable_rank_hash(seed=seed, frame_hash=frame_hash, demand=item), item) for item in items),
            key=lambda pair: (pair[0], pair[1].session_id),
        )
        for rank_hash, demand in ranked:
            if demand.dedup_key not in unserviceable_keys:
                continue
            if demand.dedup_key in selected_keys:
                continue
            final_unselected.append(
                SelectionRecord(demand=demand, rank_hash=rank_hash, selected=False, reason="unserviceable_batch_oversize")
            )

    final_unselected.sort(
        key=lambda record: (
            record.demand.tenant_id,
            record.demand.agent_id,
            record.rank_hash,
            record.demand.session_id,
            record.demand.session_version,
        )
    )

    selected_total_tokens = sum(record.demand.total_cost_tokens for record in selected_records)
    return SelectionResult(
        selected=tuple(selected_records),
        unselected=tuple(final_unselected),
        selected_total_tokens=selected_total_tokens,
        slack_tokens=unused_pool,
        redistribution_rounds=redistribution_rounds,
        unserviceable_count=len(unserviceable_keys),
        too_large_for_agent_count=too_large_for_agent_count,
        selected_ids=tuple(sorted(set(selected_ids))),
    )
