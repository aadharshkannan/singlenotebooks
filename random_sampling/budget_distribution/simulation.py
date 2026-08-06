"""Deterministic simulation harness comparing budget-distribution policies."""
from __future__ import annotations

from dataclasses import dataclass

from .allocator import AllocationConfig, allocate_hierarchical_tokens
from .models import FairnessState, SessionDemand, TenantAgentKey, stable_sha256_hex
from .pacing import RollingTokenPacer, UnpaceableReservationError
from .selection import AgentSelectionGrant, select_within_agent_grants
from .telemetry import jain_fairness


POLICIES = (
    "fcfs",
    "global_random",
    "equal_tenant",
    "equal_agent",
    "proportional",
    "hierarchical",
)

TPM_LIMIT = 20_000


@dataclass(frozen=True)
class SimulationScenario:
    name: str
    sessions_by_batch: tuple[tuple[SessionDemand, ...], ...]
    budget_tokens_per_batch: int


@dataclass(frozen=True)
class SimulationMetric:
    policy: str
    scenario: str
    utilization: float
    coverage: float
    selected: int
    starvation_proxy: float
    fairness_jain: float
    slack_tokens: int
    tpm_compliance_rate: float
    replay_match: bool


@dataclass(frozen=True)
class SimulationResult:
    metrics: tuple[SimulationMetric, ...]


def _agent_demands(sessions: tuple[SessionDemand, ...]) -> dict[TenantAgentKey, int]:
    result: dict[TenantAgentKey, int] = {}
    for session in sessions:
        result[session.key] = result.get(session.key, 0) + session.total_cost_tokens
    return result


def _capped_hamilton(total_budget: int, weights: dict[str, int], caps: dict[str, int] | None = None) -> dict[str, int]:
    if total_budget <= 0 or not weights:
        return {key: 0 for key in weights}

    cap_by_key = caps if caps is not None else weights
    keys = sorted(weights)
    alloc = {key: 0 for key in keys}

    global_cap = sum(max(0, int(cap_by_key.get(key, 0))) for key in keys)
    remaining = min(max(0, total_budget), global_cap)
    if remaining <= 0:
        return alloc

    while remaining > 0:
        active = [
            key
            for key in keys
            if max(0, int(cap_by_key.get(key, 0))) > alloc[key] and max(0, int(weights.get(key, 0))) > 0
        ]
        if not active:
            break

        total_weight = sum(max(0, int(weights[key])) for key in active)
        if total_weight <= 0:
            break

        fractional: list[tuple[float, str]] = []
        assigned_this_round = 0
        for key in active:
            cap = max(0, int(cap_by_key.get(key, 0)))
            room = max(0, cap - alloc[key])
            raw = (remaining * max(0, int(weights[key]))) / total_weight
            base = min(room, int(raw))
            if base > 0:
                alloc[key] += base
                assigned_this_round += base
            fractional.append((raw - int(raw), key))

        remaining -= assigned_this_round
        if remaining <= 0:
            break

        assigned_fractional = 0
        for _, key in sorted(fractional, key=lambda pair: (-pair[0], pair[1])):
            if remaining <= 0:
                break
            cap = max(0, int(cap_by_key.get(key, 0)))
            room = max(0, cap - alloc[key])
            if room <= 0:
                continue
            step = min(room, remaining)
            alloc[key] += step
            remaining -= step
            assigned_fractional += step

        if assigned_this_round == 0 and assigned_fractional == 0:
            break

    return alloc


def _simple_grants(policy: str, sessions: tuple[SessionDemand, ...], budget: int) -> tuple[AgentSelectionGrant, ...]:
    demand = _agent_demands(sessions)
    keys = sorted(demand)
    if not keys:
        return ()

    if policy == "equal_agent":
        raw = {f"{key.tenant_id}/{key.agent_id}": 1 for key in keys}
        caps = {f"{key.tenant_id}/{key.agent_id}": demand[key] for key in keys}
        alloc = _capped_hamilton(budget, raw, caps)
        return tuple(
            AgentSelectionGrant(
                key=key,
                grant_tokens=min(alloc[f"{key.tenant_id}/{key.agent_id}"], demand[key]),
            )
            for key in keys
        )

    if policy == "proportional":
        proportional_demands = {f"{key.tenant_id}/{key.agent_id}": demand[key] for key in keys}
        alloc = _capped_hamilton(budget, proportional_demands, proportional_demands)
        return tuple(
            AgentSelectionGrant(
                key=key,
                grant_tokens=min(alloc[f"{key.tenant_id}/{key.agent_id}"], demand[key]),
            )
            for key in keys
        )

    if policy == "equal_tenant":
        tenant_keys = sorted({key.tenant_id for key in keys})
        tenant_equal_weight = {tenant: 1 for tenant in tenant_keys}
        tenant_demand_caps = {
            tenant: sum(demand[key] for key in keys if key.tenant_id == tenant)
            for tenant in tenant_keys
        }
        tenant_budget = _capped_hamilton(budget, tenant_equal_weight, tenant_demand_caps)
        grants: list[AgentSelectionGrant] = []
        for tenant in tenant_keys:
            tenant_agents = [key for key in keys if key.tenant_id == tenant]
            tenant_demand = {
                f"{key.tenant_id}/{key.agent_id}": demand[key]
                for key in tenant_agents
            }
            per_agent_budget = _capped_hamilton(tenant_budget.get(tenant, 0), tenant_demand, tenant_demand)
            for key in tenant_agents:
                grants.append(
                    AgentSelectionGrant(
                        key=key,
                        grant_tokens=min(per_agent_budget[f"{key.tenant_id}/{key.agent_id}"], demand[key]),
                    )
                )
        return tuple(sorted(grants, key=lambda g: (g.key.tenant_id, g.key.agent_id)))

    if policy in ("fcfs", "global_random"):
        grants = [AgentSelectionGrant(key=key, grant_tokens=budget) for key in keys]
        return tuple(grants)

    raise ValueError(f"unsupported policy: {policy}")


def _simulate_policy_batch(
    *,
    policy: str,
    sessions: tuple[SessionDemand, ...],
    budget: int,
    fairness_state: FairnessState,
    seed: str,
) -> tuple[int, int, int, dict[str, int], bool, FairnessState]:
    ordered_sessions = sessions
    if policy == "fcfs":
        ordered_sessions = tuple(sorted(sessions, key=lambda s: (s.completed_at, s.dedup_key)))
    elif policy == "global_random":
        ordered_sessions = tuple(
            sorted(
                sessions,
                key=lambda s: stable_sha256_hex(seed, s.tenant_id, s.agent_id, s.session_id, s.session_version),
            )
        )

    if policy in ("fcfs", "global_random"):
        pacer = RollingTokenPacer(tpm_limit=TPM_LIMIT)
        selected_count = 0
        selected_tokens = 0
        served_by_agent: dict[str, int] = {}
        for demand in ordered_sessions:
            cost = demand.total_cost_tokens
            if cost > budget or cost > TPM_LIMIT:
                continue
            if selected_tokens + cost > budget:
                continue
            request_id = stable_sha256_hex(seed, policy, demand.dedup_key)[:24]
            try:
                pacer.reserve(request_id=request_id, estimated_tokens=cost)
                pacer.reconcile(request_id=request_id, actual_tokens=cost)
            except UnpaceableReservationError:
                continue
            selected_count += 1
            selected_tokens += cost
            key = f"{demand.tenant_id}/{demand.agent_id}"
            served_by_agent[key] = served_by_agent.get(key, 0) + cost

        return (
            selected_count,
            selected_tokens,
            max(0, budget - selected_tokens),
            served_by_agent,
            pacer.is_tpm_compliant(),
            fairness_state,
        )

    if policy == "hierarchical":
        allocation, next_state = allocate_hierarchical_tokens(
            sessions=sessions,
            total_budget_tokens=budget,
            fairness_state=fairness_state,
            config=AllocationConfig(tenant_floor_tokens=100, agent_floor_tokens=50),
        )
        grants = tuple(
            AgentSelectionGrant(
                key=key,
                grant_tokens=node.grant_tokens,
                deficit_priority=node.deficit_priority,
            )
            for key, node in allocation.agent_nodes.items()
        )
    else:
        grants = _simple_grants(policy=policy, sessions=sessions, budget=budget)
        next_state = fairness_state

    selection = select_within_agent_grants(
        sessions=ordered_sessions,
        grants=grants,
        seed=seed,
        frame_hash=stable_sha256_hex(policy, seed),
        effective_budget_tokens=budget,
        max_reservable_tokens=TPM_LIMIT,
    )

    pacer = RollingTokenPacer(tpm_limit=TPM_LIMIT)
    tpm_compliant = True
    served_by_agent: dict[str, int] = {}
    for record in selection.selected:
        request_id = stable_sha256_hex(seed, policy, record.demand.dedup_key)[:24]
        try:
            pacer.reserve(request_id=request_id, estimated_tokens=record.demand.total_cost_tokens)
            pacer.reconcile(request_id=request_id, actual_tokens=record.demand.total_cost_tokens)
            key = f"{record.demand.tenant_id}/{record.demand.agent_id}"
            served_by_agent[key] = served_by_agent.get(key, 0) + record.demand.total_cost_tokens
        except UnpaceableReservationError:
            tpm_compliant = False
            break
    tpm_compliant = tpm_compliant and pacer.is_tpm_compliant()

    return (
        len(selection.selected),
        selection.selected_total_tokens,
        selection.slack_tokens,
        served_by_agent,
        tpm_compliant,
        next_state,
    )


def simulate_policies(scenarios: tuple[SimulationScenario, ...], *, seed: str = "sim-v1") -> SimulationResult:
    def _run_once(scenario: SimulationScenario, policy: str) -> tuple[int, int, int, float, int, int, str]:
        fairness_state = FairnessState()
        selected_total = 0
        selected_tokens_total = 0
        eligible_total = 0
        active_agent_demands: dict[str, int] = {}
        served_by_agent_total: dict[str, int] = {}
        compliance_true = 0
        starvation_count = 0
        replay_signatures: list[str] = []

        for batch_index, batch_sessions in enumerate(scenario.sessions_by_batch):
            for demand in batch_sessions:
                key = f"{demand.tenant_id}/{demand.agent_id}"
                active_agent_demands[key] = active_agent_demands.get(key, 0) + demand.total_cost_tokens

            selected, selected_tokens, slack, served_by_agent, compliant, fairness_state = _simulate_policy_batch(
                policy=policy,
                sessions=batch_sessions,
                budget=scenario.budget_tokens_per_batch,
                fairness_state=fairness_state,
                seed=f"{seed}:{scenario.name}:{policy}:{batch_index}",
            )
            selected_total += selected
            selected_tokens_total += selected_tokens
            eligible_total += len(batch_sessions)
            for key, served in served_by_agent.items():
                served_by_agent_total[key] = served_by_agent_total.get(key, 0) + served
            compliance_true += 1 if compliant else 0
            if selected == 0 and len(batch_sessions) > 0:
                starvation_count += 1
            fairness_for_replay = jain_fairness(
                [float(served_by_agent_total.get(key, 0)) for key in sorted(active_agent_demands)]
            )
            replay_signatures.append(f"{selected}:{slack}:{fairness_for_replay:.6f}:{int(compliant)}")

        fairness_cumulative = jain_fairness(
            [float(served_by_agent_total.get(key, 0)) for key in sorted(active_agent_demands)]
        )

        replay_digest = stable_sha256_hex(*replay_signatures) if replay_signatures else stable_sha256_hex("empty")
        return (
            selected_total,
            selected_tokens_total,
            eligible_total,
            fairness_cumulative,
            compliance_true,
            starvation_count,
            replay_digest,
        )

    metrics: list[SimulationMetric] = []
    for scenario in scenarios:
        for policy in POLICIES:
            first = _run_once(scenario, policy)
            second = _run_once(scenario, policy)
            selected_total, selected_tokens_total, eligible_total, fairness_cumulative, compliance_true, starvation_count, replay_1 = first
            _, _, _, _, _, _, replay_2 = second

            budget_total = scenario.budget_tokens_per_batch * len(scenario.sessions_by_batch)
            coverage = selected_total / eligible_total if eligible_total else 0.0
            utilization = selected_tokens_total / budget_total if budget_total else 0.0
            slack_total = max(0, budget_total - selected_tokens_total)
            metrics.append(
                SimulationMetric(
                    policy=policy,
                    scenario=scenario.name,
                    utilization=utilization,
                    coverage=coverage,
                    selected=selected_total,
                    starvation_proxy=starvation_count / max(1, len(scenario.sessions_by_batch)),
                    fairness_jain=fairness_cumulative,
                    slack_tokens=slack_total,
                    tpm_compliance_rate=compliance_true / max(1, len(scenario.sessions_by_batch)),
                    replay_match=replay_1 == replay_2,
                )
            )

    return SimulationResult(metrics=tuple(metrics))
