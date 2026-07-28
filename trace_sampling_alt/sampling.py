"""Reference-compatible sample planning and exact stratum allocation."""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import replace
from statistics import NormalDist
from typing import Iterable, Mapping

from .models import (
    AgentKey,
    AgentSample,
    EvaluationWindow,
    EvaluationUnit,
    SampleBatch,
    SamplePlan,
    SamplePolicy,
    SampledUnit,
    StratumPlan,
    TenantCapacityPlan,
)


def z_for_confidence(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def cochran_sample_size(
    margin: float,
    confidence: float = 0.95,
    expected_rate: float = 0.5,
) -> int:
    """Return conservative infinite-population size for a proportion."""
    if not 0 < margin <= 1:
        raise ValueError(f"margin must be in (0, 1], got {margin}")
    if not 0 <= expected_rate <= 1:
        raise ValueError(
            f"expected_rate must be in [0, 1], got {expected_rate}"
        )
    z_score = z_for_confidence(confidence)
    return math.ceil(
        z_score**2 * expected_rate * (1 - expected_rate) / margin**2
    )


def finite_population_size(sample_size: int, population: int) -> int:
    if sample_size < 0:
        raise ValueError(f"sample_size must be non-negative, got {sample_size}")
    if population < 0:
        raise ValueError(f"population must be non-negative, got {population}")
    if population == 0:
        return 0
    return math.ceil(sample_size / (1 + (sample_size - 1) / population))


def plan_sample(
    population: int,
    margin: float = 0.10,
    confidence: float = 0.95,
    capacity: int | None = None,
) -> SamplePlan:
    """Plan a binary-proportion sample, optionally capped by tenant capacity."""
    if population < 0:
        raise ValueError(f"population must be non-negative, got {population}")
    if capacity is not None and capacity < 0:
        raise ValueError(f"capacity must be non-negative, got {capacity}")

    initial = cochran_sample_size(margin, confidence)
    recommended = min(finite_population_size(initial, population), population)
    selected = min(recommended, capacity) if capacity is not None else recommended
    precision_status = (
        "capacity_limited_precision_shortfall"
        if selected < recommended
        else "meets_statistical_recommendation"
    )
    return SamplePlan(
        population=population,
        recommended=recommended,
        selected=selected,
        capacity=capacity,
        census=selected == population,
        precision_status=precision_status,
        effective_rate=(selected / population) if population else 0.0,
        probability_selected=selected,
        diversity_selected=0,
    )


def allocate_strata(
    stratum_sizes: Mapping[str, int], target: int
) -> tuple[StratumPlan, ...]:
    """Allocate an exact target proportionately with capped Hamilton rounding."""
    if target < 0:
        raise ValueError(f"target must be non-negative, got {target}")
    if any(size < 0 for size in stratum_sizes.values()):
        raise ValueError("stratum populations must be non-negative")

    ordered_sizes = dict(sorted(stratum_sizes.items()))
    total = sum(ordered_sizes.values())
    if target > total:
        raise ValueError(f"target {target} exceeds population {total}")
    if total == 0:
        return tuple(
            StratumPlan(key=key, population=size, selected=0)
            for key, size in ordered_sizes.items()
        )

    ideal = {
        key: target * size / total for key, size in ordered_sizes.items()
    }
    selected = {
        key: min(math.floor(ideal[key]), size)
        for key, size in ordered_sizes.items()
    }
    remaining = target - sum(selected.values())
    candidates = sorted(
        (
            key
            for key, size in ordered_sizes.items()
            if selected[key] < size
        ),
        key=lambda key: (-(ideal[key] - selected[key]), key),
    )
    while remaining and candidates:
        for key in tuple(candidates):
            if remaining == 0:
                break
            if selected[key] < ordered_sizes[key]:
                selected[key] += 1
                remaining -= 1
        candidates = [
            key
            for key in candidates
            if selected[key] < ordered_sizes[key]
        ]

    if remaining:
        raise RuntimeError(f"could not allocate {remaining} sample units")
    return tuple(
        StratumPlan(key=key, population=size, selected=selected[key])
        for key, size in ordered_sizes.items()
    )


def _turn_count_band(unit: EvaluationUnit) -> str:
    turns = len(unit.turns)
    if turns <= 1:
        return "1"
    if turns <= 3:
        return "2-3"
    if turns <= 7:
        return "4-7"
    if turns <= 15:
        return "8-15"
    return "16+"


def _stratum_key(unit: EvaluationUnit) -> str:
    channel = (unit.channel or "unknown").strip() or "unknown"
    return f"{_turn_count_band(unit)}|{channel}"


def _unit_sort_key(unit: EvaluationUnit) -> tuple[str, str, str, str, str]:
    return (
        unit.tenant_id,
        unit.agent_id,
        unit.unit_id or "",
        unit.session_id or "",
        unit.started_at.isoformat() if unit.started_at is not None else "",
    )


def _stable_int_seed(*parts: str) -> int:
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _normalize_tokens(text: str) -> list[str]:
    normalized = "".join(ch if ch.isalnum() else " " for ch in text.casefold())
    return [tok for tok in normalized.split() if tok]


def _tagged_ngrams(tag: str, text: str, n: int) -> list[str]:
    tokens = _normalize_tokens(text)
    if not tokens:
        return []
    if len(tokens) < n:
        return [f"{tag}:{' '.join(tokens)}"]
    return [f"{tag}:{' '.join(tokens[i:i+n])}" for i in range(len(tokens) - n + 1)]


def _stable_hash64(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _mix_hash64(value: int, a: int, b: int) -> int:
    # 61-bit Mersenne prime for deterministic universal hashing.
    prime = (1 << 61) - 1
    return (a * value + b) % prime


def _unit_feature_strings(unit: EvaluationUnit, ngram_size: int) -> tuple[str, ...]:
    features: list[str] = []
    for turn in unit.turns:
        features.extend(_tagged_ngrams("user", turn.user_text, ngram_size))
        features.extend(_tagged_ngrams("assistant", turn.assistant_text, ngram_size))
    for call in unit.tool_calls:
        if call.name:
            features.extend(_tagged_ngrams("tool_name", str(call.name), ngram_size))
        if call.input_text:
            features.extend(_tagged_ngrams("tool_input", call.input_text, ngram_size))
        if call.output_text:
            features.extend(_tagged_ngrams("tool_output", call.output_text, ngram_size))
    if not features:
        features.append(f"unit:{(unit.unit_id or '').casefold()}")
    return tuple(sorted(set(features)))


def build_minhash_signature(unit: EvaluationUnit, policy: SamplePolicy) -> tuple[int, ...]:
    features = _unit_feature_strings(unit, policy.minhash_ngram_size)
    base_hashes = tuple(_stable_hash64(feature) for feature in features)
    if not base_hashes:
        return tuple()

    signature: list[int] = []
    for index in range(policy.minhash_permutations):
        seed_material = (
            f"{policy.version}|{policy.seed}|{policy.minhash_ngram_size}|"
            f"{policy.minhash_permutations}|perm|{index}"
        )
        a_seed = _stable_hash64(seed_material + "|a")
        b_seed = _stable_hash64(seed_material + "|b")
        a = (a_seed | 1)
        b = b_seed
        signature.append(min(_mix_hash64(value, a, b) for value in base_hashes))
    return tuple(signature)


def _signature_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right:
        return 1.0
    if len(left) != len(right):
        raise ValueError("signature lengths must match")
    matches = sum(1 for l_val, r_val in zip(left, right) if l_val == r_val)
    return 1.0 - (matches / len(left))


def _farthest_first(
    candidates: tuple[EvaluationUnit, ...],
    budget: int,
    seed: int,
    policy: SamplePolicy,
    core_signatures: tuple[tuple[int, ...], ...],
) -> tuple[EvaluationUnit, ...]:
    if budget <= 0 or not candidates:
        return ()
    if budget >= len(candidates):
        return candidates

    signatures = [build_minhash_signature(unit, policy) for unit in candidates]
    min_dist: list[float] = []
    if core_signatures:
        for sig in signatures:
            min_dist.append(
                min(_signature_distance(sig, core_sig) for core_sig in core_signatures)
            )
        chosen = []
        chosen_set: set[int] = set()
    else:
        rng = random.Random(seed)
        first_index = rng.randrange(len(candidates))
        chosen = [first_index]
        chosen_set = {first_index}
        min_dist = [_signature_distance(sig, signatures[first_index]) for sig in signatures]
        min_dist[first_index] = -1.0

    if core_signatures and budget > 0:
        first_index = max(
            range(len(candidates)),
            key=lambda idx: (min_dist[idx], _unit_sort_key(candidates[idx])),
        )
        chosen.append(first_index)
        chosen_set.add(first_index)
        first_signature = signatures[first_index]
        for idx, signature in enumerate(signatures):
            if idx in chosen_set:
                min_dist[idx] = -1.0
                continue
            min_dist[idx] = min(
                min_dist[idx],
                _signature_distance(first_signature, signature),
            )

    while len(chosen) < budget:
        next_index = max(
            range(len(candidates)),
            key=lambda idx: (
                min_dist[idx],
                _unit_sort_key(candidates[idx]),
            ),
        )
        if next_index in chosen_set or min_dist[next_index] < 0:
            break
        chosen.append(next_index)
        chosen_set.add(next_index)
        next_sig = signatures[next_index]
        for idx, sig in enumerate(signatures):
            if idx in chosen_set:
                min_dist[idx] = -1.0
                continue
            dist = _signature_distance(next_sig, sig)
            if dist < min_dist[idx]:
                min_dist[idx] = dist

    return tuple(candidates[idx] for idx in chosen)


def _agent_rng(policy: SamplePolicy, agent: AgentKey) -> random.Random:
    seed = _stable_int_seed(
        policy.version,
        str(policy.seed),
        agent.tenant_id,
        agent.agent_id,
    )
    return random.Random(seed)


class SamplingEngine:
    def sample(
        self,
        units: Iterable[EvaluationUnit],
        policy: SamplePolicy,
        capacities: Mapping[AgentKey, int] | None = None,
        tenant_capacities: Mapping[str, int] | None = None,
        agents: Iterable[AgentKey] | None = None,
        evaluation_window: EvaluationWindow | None = None,
    ) -> SampleBatch:
        if capacities is not None and tenant_capacities is not None:
            raise ValueError(
                "capacities and tenant_capacities are mutually exclusive"
            )
        grouped: dict[AgentKey, list[EvaluationUnit]] = {}
        for unit in sorted(units, key=_unit_sort_key):
            agent = AgentKey(tenant_id=unit.tenant_id, agent_id=unit.agent_id)
            grouped.setdefault(agent, []).append(unit)

        selected_agents = sorted(agents) if agents is not None else sorted(grouped)
        effective_capacities = dict(capacities or {})
        tenant_plans: list[TenantCapacityPlan] = []
        if tenant_capacities is not None:
            effective_capacities, tenant_plans = self._allocate_tenant_capacities(
                grouped=grouped,
                selected_agents=selected_agents,
                policy=policy,
                tenant_capacities=tenant_capacities,
            )
        samples: list[AgentSample] = []
        for agent in selected_agents:
            population = tuple(grouped.get(agent, []))
            if not population:
                continue
            capacity = effective_capacities.get(agent)
            samples.append(self.sample_agent(population, agent, policy, capacity=capacity))

        run_parts: list[str] = [
            policy.version,
            str(policy.seed),
            f"m{policy.margin}",
            f"c{policy.confidence}",
            f"d{policy.diversity_fraction}",
            f"ng{policy.minhash_ngram_size}",
            f"perm{policy.minhash_permutations}",
        ]
        if evaluation_window is not None:
            run_parts.append(f"window:{evaluation_window.start_at.isoformat()}->{evaluation_window.end_at.isoformat()}")
        for sample in samples:
            run_parts.append(f"{sample.agent.tenant_id}/{sample.agent.agent_id}")
            run_parts.append(
                ",".join(unit.unit.unit_id or "" for unit in sample.core)
            )
            run_parts.append(
                ",".join(unit.unit.unit_id or "" for unit in sample.diversity)
            )
        for tenant_plan in tenant_plans:
            run_parts.append(
                f"tenant-capacity:{tenant_plan.tenant_id}:"
                f"{tenant_plan.granted}:{tenant_plan.selected}"
            )
        run_id = hashlib.sha256("||".join(run_parts).encode("utf-8")).hexdigest()[:16]
        return SampleBatch(
            policy=policy,
            version=policy.version,
            run_id=run_id,
            agents=tuple(samples),
            tenant_capacities=tuple(tenant_plans),
            evaluation_window=evaluation_window,
        )

    def _allocate_tenant_capacities(
        self,
        grouped: Mapping[AgentKey, list[EvaluationUnit]],
        selected_agents: list[AgentKey],
        policy: SamplePolicy,
        tenant_capacities: Mapping[str, int],
    ) -> tuple[dict[AgentKey, int], list[TenantCapacityPlan]]:
        if any(capacity < 0 for capacity in tenant_capacities.values()):
            raise ValueError("tenant capacities must be non-negative")

        agents_by_tenant: dict[str, list[AgentKey]] = {}
        for agent in selected_agents:
            if grouped.get(agent):
                agents_by_tenant.setdefault(agent.tenant_id, []).append(agent)

        effective: dict[AgentKey, int] = {}
        plans: list[TenantCapacityPlan] = []
        for tenant_id, granted in sorted(tenant_capacities.items()):
            tenant_agents = sorted(agents_by_tenant.get(tenant_id, []))
            recommendations = {
                agent.agent_id: plan_sample(
                    population=len(grouped[agent]),
                    margin=policy.margin,
                    confidence=policy.confidence,
                ).recommended
                for agent in tenant_agents
            }
            statistical_recommended = sum(recommendations.values())
            selected = min(granted, statistical_recommended)
            allocation = allocate_strata(recommendations, selected)
            selected_by_agent = {row.key: row.selected for row in allocation}
            for agent in tenant_agents:
                effective[agent] = selected_by_agent[agent.agent_id]
            plans.append(
                TenantCapacityPlan(
                    tenant_id=tenant_id,
                    granted=granted,
                    statistical_recommended=statistical_recommended,
                    selected=selected,
                    unused=granted - selected,
                    precision_status=(
                        "capacity_limited_precision_shortfall"
                        if selected < statistical_recommended
                        else "meets_statistical_recommendation"
                    ),
                )
            )
        return effective, plans

    def sample_agent(
        self,
        units: Iterable[EvaluationUnit],
        agent: AgentKey,
        policy: SamplePolicy,
        capacity: int | None = None,
    ) -> AgentSample:
        population = tuple(sorted(units, key=_unit_sort_key))
        plan = plan_sample(
            population=len(population),
            margin=policy.margin,
            confidence=policy.confidence,
            capacity=capacity,
        )

        probability_selected = plan.selected
        diversity_selected = 0
        precision_status = plan.precision_status
        if not plan.census and policy.diversity_enabled and plan.selected > 0:
            reserved = int(math.floor((plan.selected * policy.diversity_fraction) + 0.5))
            reserved = min(reserved, plan.selected - 1) if plan.selected > 0 else 0
            diversity_selected = max(0, reserved)
            probability_selected = plan.selected - diversity_selected
            if diversity_selected > 0:
                status = "diversity_reserved_precision_shortfall"
                if precision_status != "meets_statistical_recommendation":
                    precision_status = f"{precision_status}+{status}"
                else:
                    precision_status = status

        plan = replace(
            plan,
            probability_selected=probability_selected,
            diversity_selected=diversity_selected,
            precision_status=precision_status,
        )

        if plan.census:
            strata_sizes: dict[str, int] = {}
            for unit in population:
                key = _stratum_key(unit)
                strata_sizes[key] = strata_sizes.get(key, 0) + 1
            strata = tuple(
                StratumPlan(key=key, population=size, selected=size)
                for key, size in sorted(strata_sizes.items())
            )
            core = tuple(
                SampledUnit(
                    unit=unit,
                    sample_kind="core",
                    estimand_eligible=True,
                    stratum_key=_stratum_key(unit),
                    inclusion_probability=1.0,
                    sampling_weight=1.0,
                    selection_reason="census",
                )
                for unit in population
            )
            return AgentSample(
                agent=agent,
                plan=plan,
                strata=strata,
                core=core,
                diversity=(),
            )

        by_stratum: dict[str, list[EvaluationUnit]] = {}
        for unit in population:
            by_stratum.setdefault(_stratum_key(unit), []).append(unit)

        for units_in_stratum in by_stratum.values():
            units_in_stratum.sort(key=_unit_sort_key)

        strata = allocate_strata(
            {key: len(values) for key, values in by_stratum.items()},
            plan.probability_selected,
        )
        strata_by_key = {plan_row.key: plan_row for plan_row in strata}

        rng = _agent_rng(policy, agent)
        core_records: list[SampledUnit] = []
        for plan_row in strata:
            pool = by_stratum.get(plan_row.key, [])
            if plan_row.selected <= 0 or not pool:
                continue
            indexes = sorted(rng.sample(range(len(pool)), plan_row.selected))
            for idx in indexes:
                unit = pool[idx]
                pi_i = plan_row.inclusion_probability
                core_records.append(
                    SampledUnit(
                        unit=unit,
                        sample_kind="core",
                        estimand_eligible=True,
                        stratum_key=plan_row.key,
                        inclusion_probability=pi_i,
                        sampling_weight=(1.0 / pi_i) if pi_i > 0 else None,
                        selection_reason=(
                            "stratified_random_selection"
                            f"(N_h={plan_row.population},n_h={plan_row.selected})"
                        ),
                    )
                )

        core = tuple(sorted(core_records, key=lambda row: _unit_sort_key(row.unit)))
        core_ids = {row.unit.unit_id for row in core}

        diversity_budget = plan.diversity_selected
        diversity: tuple[SampledUnit, ...] = ()
        if diversity_budget > 0:
            candidates = tuple(
                unit for unit in population if unit.unit_id not in core_ids
            )
            if candidates:
                core_signatures = tuple(
                    build_minhash_signature(row.unit, policy) for row in core
                )
                chosen = _farthest_first(
                    candidates=tuple(sorted(candidates, key=_unit_sort_key)),
                    budget=min(diversity_budget, len(candidates)),
                    seed=_stable_int_seed(
                        policy.version,
                        str(policy.seed),
                        agent.tenant_id,
                        agent.agent_id,
                        "diversity",
                    ),
                    policy=policy,
                    core_signatures=core_signatures,
                )
                diversity = tuple(
                    SampledUnit(
                        unit=unit,
                        sample_kind="diversity",
                        estimand_eligible=False,
                        stratum_key=_stratum_key(unit),
                        inclusion_probability=None,
                        sampling_weight=None,
                        selection_reason="diversity_minhash_farthest_first",
                    )
                    for unit in sorted(chosen, key=_unit_sort_key)
                )

        return AgentSample(
            agent=agent,
            plan=plan,
            strata=tuple(
                StratumPlan(
                    key=row.key,
                    population=row.population,
                    selected=row.selected,
                )
                for row in sorted(strata_by_key.values(), key=lambda row: row.key)
            ),
            core=core,
            diversity=diversity,
        )