from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .v4_idw import (
    IDWConfig,
    estimate_embedding_population,
    freeze_membership,
    validate_embedding_population,
)

SAMPLE_CAPS: tuple[int, ...] = (64, 128, 256, 512, 1024)
NOMINAL_TOKENS_PER_SESSION = 15_000
TRIAL_SEEDS: tuple[int, ...] = (13, 14, 15)
METHOD_IDS: dict[str, str] = {
    "arm1": "arm1_global_random",
    "arm2": "arm2_embedding_idw",
    "arm2_5": "arm2_5_embedding_idw_binary",
    "arm3": "arm3_agent_round_robin_floor",
    "arm4": "arm4_agent_round_robin",
    "arm5": "arm5_hajek_weighted",
    "arm6": "arm6_agent_use_case_hajek",
}
METHOD_ID_ORDER: tuple[str, ...] = tuple(METHOD_IDS[k] for k in ("arm1", "arm2", "arm2_5", "arm3", "arm4", "arm5", "arm6"))


def _trim_upper(value: Any) -> str:
    return str(value).strip().upper()


def _stable_sha256_hex(*values: Any) -> str:
    pieces = [str(value).encode("utf-8") for value in values if value is not None]
    if not pieces:
        pieces = [b""]
    return hashlib.sha256(b"\0".join(pieces)).hexdigest()


def _hash_float(seed: int, *parts: Any) -> float:
    payload = b"\0".join([str(seed).encode("utf-8")] + [str(part).encode("utf-8") for part in parts])
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return float(value) / float(1 << 64)


def _normalize_identity(value: Any) -> str:
    return _trim_upper(str(value))


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class SessionDescriptor:
    unit_id: str
    agent_id: str
    use_case_id: str
    concept_key: str
    label: bool
    business_use_case_guid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_id", str(self.unit_id))
        object.__setattr__(self, "agent_id", str(self.agent_id))
        object.__setattr__(self, "use_case_id", str(self.use_case_id))
        business_use_case_guid = str(self.business_use_case_guid).strip() if self.business_use_case_guid is not None else ""
        object.__setattr__(
            self,
            "business_use_case_guid",
            business_use_case_guid if business_use_case_guid else str(self.use_case_id),
        )
        object.__setattr__(self, "concept_key", str(self.concept_key))
        object.__setattr__(self, "label", bool(self.label))


@dataclass(frozen=True)
class SelectionRecord:
    unit_id: str
    method_id: str
    stratum: str
    inclusion_probability: float | None
    weight: float | None
    reason: str


@dataclass(frozen=True)
class TrialMetrics:
    method_id: str
    trial_seed: int
    window_id: str
    nominal_budget: int
    sample_size: int
    estimate: float
    census_pass_rate: float
    absolute_aggregate_mae: float
    actual_token_count: int
    concept_coverage: float
    use_case_coverage: float
    selected_label_rate: float
    agent_coverage: float
    top_five_agents: tuple[dict[str, Any], ...]
    selected_ids: tuple[str, ...]
    selected_only_rate: float | None = None
    selected_only_absolute_error: float | None = None
    idw_validation: dict[str, Any] | None = None
    idw_population_estimate: float | None = None
    arm4_membership_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectionOutcome:
    method_id: str
    selected_ids: tuple[str, ...]
    records: tuple[SelectionRecord, ...]
    per_agent_counts: dict[str, int] = field(default_factory=dict)
    inclusion_probability_by_unit: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def build_session_descriptors(rows: Sequence[Mapping[str, Any]]) -> tuple[SessionDescriptor, ...]:
    out: list[SessionDescriptor] = []
    for index, row in enumerate(rows):
        unit_id = str(row.get("unit_id") or row.get("id") or f"unit-{index}")
        agent_id = str(row.get("agent_id") or row.get("agent") or "agent-unknown")
        use_case_id = str(row.get("use_case_id") or row.get("maven_use_case_id") or row.get("use_case") or "use-case-unknown")
        business_use_case_guid = str(
            row.get("business_use_case_guid")
            or row.get("maven_business_use_case_guid")
            or row.get("maven_use_case_guid")
            or row.get("use_case_guid")
            or use_case_id
        )
        concept_key = str(row.get("concept_key") or row.get("concept") or f"concept-{index}")
        label = bool(row.get("label") or row.get("pass") or row.get("passed") or False)
        out.append(
            SessionDescriptor(
                unit_id=unit_id,
                agent_id=agent_id,
                use_case_id=use_case_id,
                business_use_case_guid=business_use_case_guid,
                concept_key=concept_key,
                label=label,
            )
        )
    return tuple(out)


def _window_identity(window_id: str, trial_seed: int, method_id: str) -> str:
    return _stable_sha256_hex(window_id, method_id, str(trial_seed))


def _agent_order_key(window_identity: str, agent_id: str) -> str:
    return _stable_sha256_hex("agent-order", _normalize_identity(agent_id), window_identity)


def _session_order_key(window_identity: str, session: SessionDescriptor, *, kind: str = "session-order") -> str:
    return _stable_sha256_hex(
        kind,
        window_identity,
        _normalize_identity(session.agent_id),
        _normalize_identity(session.unit_id),
        _normalize_identity(session.use_case_id),
    )


def _stratum_order_key(window_identity: str, agent_id: str, use_case_id: str) -> str:
    return _stable_sha256_hex(
        "stratum-order",
        window_identity,
        _normalize_identity(agent_id),
        _normalize_identity(use_case_id),
    )


def _round_robin_queue_key(window_identity: str, agent_id: str, unit_id: str) -> str:
    return _stable_sha256_hex("round-robin", _normalize_identity(agent_id), _normalize_identity(unit_id), window_identity)


def _sorted_population(descriptors: Sequence[SessionDescriptor], *, method_id: str, window_id: str, trial_seed: int) -> tuple[str, ...]:
    window_identity = _window_identity(window_id, trial_seed, method_id)
    ranked = sorted(
        descriptors,
        key=lambda session: (
            _session_order_key(window_identity, session, kind=f"{method_id}-global"),
            _normalize_identity(session.agent_id),
            _normalize_identity(session.unit_id),
        ),
    )
    return tuple(session.unit_id for session in ranked)


def _coerce_cap(population_size: int, cap: int) -> int:
    if cap < 0:
        raise ValueError("cap must be non-negative")
    return min(int(cap), population_size)


def _expected_agent_slots(agent_population: Mapping[str, int], total_cap: int) -> dict[str, float]:
    agents = sorted(agent_population.keys())
    count = {agent: int(max(0, agent_population[agent])) for agent in agents}
    if total_cap <= 0:
        return {agent: 0.0 for agent in agents}
    if not agents:
        return {}
    total_cap = min(int(total_cap), int(sum(count.values())))
    expected: dict[str, float] = {agent: 0.0 for agent in agents}
    remaining_capacity = dict(count)
    remaining_cap = total_cap
    while remaining_cap > 0:
        active = [agent for agent in agents if remaining_capacity[agent] > 0]
        if not active:
            break
        m_active = len(active)
        min_active_capacity = min(remaining_capacity[agent] for agent in active)
        full_rounds_available = remaining_cap // m_active
        if full_rounds_available >= min_active_capacity:
            for agent in active:
                expected[agent] += float(min_active_capacity)
                remaining_capacity[agent] -= min_active_capacity
            remaining_cap -= min_active_capacity * m_active
            continue

        if full_rounds_available > 0:
            for agent in active:
                expected[agent] += float(full_rounds_available)
                remaining_capacity[agent] -= full_rounds_available
            remaining_cap -= full_rounds_available * m_active
            continue

        fractional = float(remaining_cap) / float(m_active)
        for agent in active:
            expected[agent] += fractional
        remaining_cap = 0

    if abs(sum(expected.values()) - float(total_cap)) > 1e-9:
        raise AssertionError("expected agent slots must sum to exact cap")
    return expected


def _allocate_round_robin_counts(*, agent_capacity: Mapping[str, int], cap: int, agent_order: Sequence[str]) -> dict[str, int]:
    counts = {agent: 0 for agent in agent_order}
    cap = min(int(cap), int(sum(max(0, int(agent_capacity.get(agent, 0))) for agent in agent_order)))
    remaining = cap
    while remaining > 0:
        progress = False
        for agent in agent_order:
            if remaining <= 0:
                break
            if counts[agent] < int(agent_capacity.get(agent, 0)):
                counts[agent] += 1
                remaining -= 1
                progress = True
        if not progress:
            break
    return counts


def _build_agent_queues(*, descriptors: Sequence[SessionDescriptor], window_identity: str) -> tuple[dict[str, list[SessionDescriptor]], tuple[str, ...]]:
    by_agent: dict[str, list[SessionDescriptor]] = {}
    for descriptor in descriptors:
        by_agent.setdefault(descriptor.agent_id, []).append(descriptor)
    agent_order = tuple(sorted(by_agent, key=lambda agent: _agent_order_key(window_identity, agent)))
    queues: dict[str, list[SessionDescriptor]] = {}
    for agent in agent_order:
        queues[agent] = sorted(
            by_agent[agent],
            key=lambda session: (
                _round_robin_queue_key(window_identity, session.agent_id, session.unit_id),
                _normalize_identity(session.unit_id),
            ),
        )
    return queues, agent_order


def _systematic_stratum_assignments(pop_by_stratum: Mapping[str, int], desired_total: int, *, trial_seed: int, window_id: str, agent_id: str) -> dict[str, int]:
    window_identity = _window_identity(window_id, trial_seed, "stratum-systematic")
    total = sum(pop_by_stratum.values())
    if total <= 0 or desired_total <= 0:
        return {key: 0 for key in pop_by_stratum}
    desired_total = min(int(desired_total), int(total))
    ideal = {key: float(desired_total) * (float(count) / float(total)) for key, count in pop_by_stratum.items()}
    floors = {key: min(int(pop_by_stratum[key]), int(math.floor(ideal[key]))) for key in pop_by_stratum}
    remaining = desired_total - sum(floors.values())
    if remaining <= 0:
        return floors

    fracs = {
        key: max(0.0, min(1.0, ideal[key] - floors[key]))
        for key in pop_by_stratum
    }
    ordered = sorted(pop_by_stratum, key=lambda key: _stratum_order_key(window_identity, agent_id, key))
    counts: dict[str, int] = {key: floors[key] for key in ordered}
    if sum(fracs.values()) <= 0.0:
        return counts

    # Randomized systematic remainder rounding over [0, remaining).
    u = _hash_float(trial_seed, "stratum-systematic", window_identity, _normalize_identity(agent_id))
    target_points = [u + float(idx) for idx in range(remaining)]
    point_idx = 0
    cumulative = 0.0
    for key in ordered:
        frac = fracs[key]
        start = cumulative
        end = cumulative + frac
        while point_idx < remaining and target_points[point_idx] < end - 1e-12:
            if target_points[point_idx] >= start - 1e-12 and counts[key] < int(pop_by_stratum[key]):
                counts[key] += 1
            point_idx += 1
        cumulative = end
    if sum(counts.values()) != desired_total:
        raise AssertionError("systematic stratum assignments must produce the exact desired total")
    for key, value in counts.items():
        if value < 0 or value > int(pop_by_stratum[key]):
            raise AssertionError("systematic stratum assignments exceeded stratum capacity")
    return counts


def _select_within_stratum(agent_sessions: Sequence[SessionDescriptor], *, count: int, window_id: str, trial_seed: int, method_id: str) -> tuple[str, ...]:
    if count <= 0:
        return ()
    window_identity = _window_identity(window_id, trial_seed, method_id)
    ranked = sorted(
        agent_sessions,
        key=lambda session: _session_order_key(window_identity, session),
    )
    return tuple(session.unit_id for session in ranked[:count])


def _select_arm_records(
    *,
    descriptors: Sequence[SessionDescriptor],
    agent_count_targets: Mapping[str, int],
    method_id: str,
    trial_seed: int,
    window_id: str,
) -> tuple[SelectionRecord, ...]:
    by_agent: dict[str, list[SessionDescriptor]] = {}
    for session in descriptors:
        by_agent.setdefault(session.agent_id, []).append(session)

    selected_records: list[SelectionRecord] = []
    seen: set[str] = set()
    window_identity = _window_identity(window_id, trial_seed, method_id)
    for agent_id in sorted(agent_count_targets, key=lambda a: _agent_order_key(window_identity, a)):
        desired = int(agent_count_targets.get(agent_id, 0))
        if desired <= 0:
            continue
        agent_sessions = sorted(by_agent.get(agent_id, []), key=lambda session: _session_order_key(window_identity, session))
        grouped: dict[str, list[SessionDescriptor]] = {}
        for session in agent_sessions:
            grouped.setdefault(session.use_case_id, []).append(session)
        totals = {use_case_id: len(sessions) for use_case_id, sessions in grouped.items()}
        allocations = _systematic_stratum_assignments(totals, desired, trial_seed=trial_seed, window_id=window_id, agent_id=agent_id)
        for use_case_id in sorted(grouped, key=lambda key: _stratum_order_key(window_identity, agent_id, key)):
            count = int(allocations.get(use_case_id, 0))
            if count <= 0:
                continue
            for unit_id in _select_within_stratum(grouped[use_case_id], count=count, window_id=window_id, trial_seed=trial_seed, method_id=method_id):
                if unit_id in seen:
                    continue
                seen.add(unit_id)
                selected_records.append(
                    SelectionRecord(
                        unit_id=unit_id,
                        method_id=method_id,
                        stratum=use_case_id,
                        inclusion_probability=None,
                        weight=None,
                        reason=f"agent-round-robin:{method_id}",
                    )
                )
    return tuple(selected_records)


def _select_arm3_records(*, descriptors: Sequence[SessionDescriptor], cap: int, trial_seed: int, window_id: str) -> tuple[SelectionRecord, ...]:
    method_id = METHOD_IDS["arm3"]
    window_identity = _window_identity(window_id, trial_seed, method_id)
    queues, agent_order = _build_agent_queues(descriptors=descriptors, window_identity=window_identity)
    cap = min(cap, len(descriptors))
    if cap <= 0 or not agent_order:
        return ()

    floor_limit = {agent: min(3, len(queues[agent])) for agent in agent_order}
    taken_counts = {agent: 0 for agent in agent_order}
    floor_selected_ids: list[str] = []

    while len(floor_selected_ids) < cap:
        progressed = False
        for agent in agent_order:
            if len(floor_selected_ids) >= cap:
                break
            if taken_counts[agent] < floor_limit[agent]:
                floor_selected_ids.append(queues[agent][taken_counts[agent]].unit_id)
                taken_counts[agent] += 1
                progressed = True
        if not progressed:
            break

    if len(floor_selected_ids) >= cap:
        return tuple(
            SelectionRecord(
                unit_id=unit_id,
                method_id=method_id,
                stratum="floor-prefix",
                inclusion_probability=None,
                weight=None,
                reason="agent-round-robin-floor-prefix",
            )
            for unit_id in floor_selected_ids[:cap]
        )

    floor_set = set(floor_selected_ids)
    residual_by_agent: dict[str, list[SessionDescriptor]] = {
        agent: [session for session in queues[agent] if session.unit_id not in floor_set]
        for agent in agent_order
    }
    residual_capacities = {agent: len(residual_by_agent[agent]) for agent in agent_order}
    remaining_cap = cap - len(floor_selected_ids)
    additional_targets = _allocate_round_robin_counts(agent_capacity=residual_capacities, cap=remaining_cap, agent_order=agent_order)

    additional_records: list[SelectionRecord] = []
    for agent in agent_order:
        desired = int(additional_targets.get(agent, 0))
        if desired <= 0:
            continue
        grouped: dict[str, list[SessionDescriptor]] = {}
        for session in residual_by_agent[agent]:
            grouped.setdefault(session.use_case_id, []).append(session)
        totals = {use_case_id: len(sessions) for use_case_id, sessions in grouped.items()}
        allocations = _systematic_stratum_assignments(totals, desired, trial_seed=trial_seed, window_id=window_id, agent_id=agent)
        for use_case_id in sorted(grouped, key=lambda key: _stratum_order_key(window_identity, agent, key)):
            count = int(allocations.get(use_case_id, 0))
            if count <= 0:
                continue
            chosen = _select_within_stratum(
                grouped[use_case_id],
                count=count,
                window_id=window_id,
                trial_seed=trial_seed,
                method_id=method_id,
            )
            for unit_id in chosen:
                if unit_id in floor_set:
                    continue
                additional_records.append(
                    SelectionRecord(
                        unit_id=unit_id,
                        method_id=method_id,
                        stratum=use_case_id,
                        inclusion_probability=None,
                        weight=None,
                        reason="agent-round-robin-floor-residual",
                    )
                )

    all_records = [
        SelectionRecord(
            unit_id=unit_id,
            method_id=method_id,
            stratum="floor-prefix",
            inclusion_probability=None,
            weight=None,
            reason="agent-round-robin-floor-prefix",
        )
        for unit_id in floor_selected_ids
    ]
    all_records.extend(additional_records)
    unique_records: list[SelectionRecord] = []
    seen: set[str] = set()
    for record in all_records:
        if record.unit_id in seen:
            continue
        seen.add(record.unit_id)
        unique_records.append(record)
        if len(unique_records) >= cap:
            break
    return tuple(unique_records)


def _selection_by_method(method_id: str, *, descriptors: Sequence[SessionDescriptor], cap: int, trial_seed: int, window_id: str) -> SelectionOutcome:
    population_size = len(descriptors)
    cap = _coerce_cap(population_size, cap)
    if method_id == METHOD_IDS["arm1"]:
        selected_ids = _sorted_population(descriptors, method_id=method_id, window_id=window_id, trial_seed=trial_seed)[:cap]
        records = tuple(
            SelectionRecord(
                unit_id=unit_id,
                method_id=method_id,
                stratum="global",
                inclusion_probability=max(1e-12, float(cap) / max(1, population_size)),
                weight=1.0 / max(1e-12, float(cap) / max(1, population_size)),
                reason="external-3p-global-random",
            )
            for unit_id in selected_ids
        )
        return SelectionOutcome(
            method_id=method_id,
            selected_ids=tuple(selected_ids),
            records=records,
            per_agent_counts={},
            inclusion_probability_by_unit={record.unit_id: record.inclusion_probability for record in records},
            diagnostics={},
        )

    if method_id in (METHOD_IDS["arm3"], METHOD_IDS["arm4"]):
        if method_id == METHOD_IDS["arm3"]:
            records = _select_arm3_records(descriptors=descriptors, cap=cap, trial_seed=trial_seed, window_id=window_id)
            selected_ids = tuple(record.unit_id for record in records)
            per_agent: dict[str, int] = {}
            by_id = {descriptor.unit_id: descriptor for descriptor in descriptors}
            for unit_id in selected_ids:
                descriptor = by_id.get(unit_id)
                if descriptor is None:
                    continue
                per_agent[descriptor.agent_id] = per_agent.get(descriptor.agent_id, 0) + 1
            final_records = list(records)
            inclusion_map: dict[str, float] = {}
        else:
            window_identity = _window_identity(window_id, trial_seed, method_id)
            queues, agent_order = _build_agent_queues(descriptors=descriptors, window_identity=window_identity)
            capacities = {agent: len(queues[agent]) for agent in agent_order}
            agent_targets = _allocate_round_robin_counts(agent_capacity=capacities, cap=cap, agent_order=agent_order)
            records = _select_arm_records(
                descriptors=descriptors,
                agent_count_targets=agent_targets,
                method_id=method_id,
                trial_seed=trial_seed,
                window_id=window_id,
            )
            selected_ids = tuple(record.unit_id for record in records)
            pi_by_unit = arm4_inclusion_probabilities(descriptors=descriptors, cap=cap, trial_seed=trial_seed, window_id=window_id)
            final_records = []
            for record in records:
                pi = float(pi_by_unit.get(record.unit_id, 1.0))
                if not (0.0 < pi <= 1.0):
                    raise ValueError(f"inclusion probability out of range for unit {record.unit_id}: {pi}")
                final_records.append(
                    SelectionRecord(
                        unit_id=record.unit_id,
                        method_id=record.method_id,
                        stratum=record.stratum,
                        inclusion_probability=pi,
                        weight=1.0 / pi,
                        reason=record.reason,
                    )
                )
            per_agent = dict(agent_targets)
            inclusion_map = {record.unit_id: float(record.inclusion_probability or 0.0) for record in final_records}

        if len(set(selected_ids)) != len(selected_ids):
            raise AssertionError("selected ids must be unique")
        return SelectionOutcome(
            method_id=method_id,
            selected_ids=selected_ids,
            records=tuple(final_records),
            per_agent_counts=per_agent,
            inclusion_probability_by_unit=inclusion_map,
            diagnostics={},
        )

    raise ValueError(f"Unsupported method_id={method_id!r}")


def select_arm1(*, descriptors: Sequence[SessionDescriptor], cap: int, trial_seed: int = 13, window_id: str = "window-1") -> SelectionOutcome:
    return _selection_by_method(METHOD_IDS["arm1"], descriptors=descriptors, cap=cap, trial_seed=trial_seed, window_id=window_id)


def select_arm3(*, descriptors: Sequence[SessionDescriptor], cap: int, trial_seed: int = 13, window_id: str = "window-1") -> SelectionOutcome:
    return _selection_by_method(METHOD_IDS["arm3"], descriptors=descriptors, cap=cap, trial_seed=trial_seed, window_id=window_id)


def select_arm4(*, descriptors: Sequence[SessionDescriptor], cap: int, trial_seed: int = 13, window_id: str = "window-1") -> SelectionOutcome:
    return _selection_by_method(METHOD_IDS["arm4"], descriptors=descriptors, cap=cap, trial_seed=trial_seed, window_id=window_id)


def arm4_inclusion_probabilities(*, descriptors: Sequence[SessionDescriptor], cap: int, trial_seed: int = 13, window_id: str = "window-1") -> dict[str, float]:
    by_agent: dict[str, list[SessionDescriptor]] = {}
    for descriptor in descriptors:
        by_agent.setdefault(descriptor.agent_id, []).append(descriptor)
    window_identity = _window_identity(window_id, trial_seed, METHOD_IDS["arm4"])
    agent_order = tuple(sorted(by_agent, key=lambda agent: _agent_order_key(window_identity, agent)))
    if not agent_order:
        return {}
    agent_counts = {agent: len(by_agent[agent]) for agent in agent_order}
    cap = min(cap, sum(agent_counts.values()))
    expected_slots = _expected_agent_slots(agent_counts, cap)
    out: dict[str, float] = {}
    for agent, sessions in by_agent.items():
        N_a = len(sessions)
        if N_a <= 0:
            continue
        pi = float(expected_slots.get(agent, 0.0)) / float(N_a)
        pi = max(1e-12, min(1.0, pi))
        assert 0.0 < pi <= 1.0, f"invalid arm4 marginal inclusion probability for agent {agent}: {pi}"
        for session in sessions:
            out[session.unit_id] = pi
    return out


def select_arm5(*, descriptors: Sequence[SessionDescriptor], arm4_outcome: SelectionOutcome, labels_by_unit: Mapping[str, bool], trial_seed: int = 13, window_id: str = "window-1") -> SelectionOutcome:
    selected_ids = tuple(arm4_outcome.selected_ids)
    if not selected_ids:
        return SelectionOutcome(method_id=METHOD_IDS["arm5"], selected_ids=(), records=(), per_agent_counts={}, inclusion_probability_by_unit={}, diagnostics={})
    arm4_pi = dict(arm4_outcome.inclusion_probability_by_unit)
    if not arm4_pi:
        arm4_pi = {
            record.unit_id: float(record.inclusion_probability)
            for record in arm4_outcome.records
            if record.inclusion_probability is not None
        }
    records = tuple(
        SelectionRecord(
            unit_id=unit_id,
            method_id=METHOD_IDS["arm5"],
            stratum="arm4-membership",
            inclusion_probability=float(arm4_pi[unit_id]),
            weight=1.0 / max(1e-12, float(arm4_pi[unit_id])),
            reason="hajek-reuse-arm4",
        )
        for unit_id in selected_ids
    )
    return SelectionOutcome(
        method_id=METHOD_IDS["arm5"],
        selected_ids=selected_ids,
        records=records,
        per_agent_counts=dict(arm4_outcome.per_agent_counts),
        inclusion_probability_by_unit={record.unit_id: record.inclusion_probability for record in records},
        diagnostics={"design": "hajek-reuse-arm4"},
    )


def _extract_unit_estimate_value(row: Any) -> float:
    if hasattr(row, "value"):
        return float(getattr(row, "value"))
    if isinstance(row, Mapping) and "value" in row:
        return float(row["value"])
    raise ValueError("unit estimate row must expose a numeric 'value'")


def arm2_5_binary_estimate_from_rows(rows: Sequence[Any]) -> float:
    if not rows:
        return 0.0
    binary_values = [1.0 if _extract_unit_estimate_value(row) >= 0.5 else 0.0 for row in rows]
    return float(sum(binary_values) / len(binary_values))


def arm2_5_binary_estimate_from_population(estimated_population: Any) -> float:
    rows = tuple(getattr(estimated_population, "rows", ()) or ())
    return arm2_5_binary_estimate_from_rows(rows)


def run_arm2_5_binary_from_arm2_result(*, arm2_result: Mapping[str, Any]) -> dict[str, Any]:
    estimated_population = arm2_result.get("estimated_population")
    if estimated_population is None:
        raise ValueError("arm2_result must include estimated_population for arm2.5")
    binary_estimate = arm2_5_binary_estimate_from_population(estimated_population)
    validation_obj = arm2_result.get("validation")
    if isinstance(validation_obj, Mapping):
        census_rate = float(validation_obj.get("census_pass_rate", 0.0))
    elif validation_obj is not None and hasattr(validation_obj, "census_pass_rate"):
        census_rate = float(getattr(validation_obj, "census_pass_rate"))
    else:
        census_rate = 0.0
    selected_rate = float(arm2_result.get("selected_rate", 0.0))
    return {
        "method_id": METHOD_IDS["arm2_5"],
        "membership": arm2_result.get("membership"),
        "estimate": binary_estimate,
        "binary_estimate": binary_estimate,
        "continuous_estimate": float(arm2_result.get("estimate", 0.0)),
        "selected_rate": selected_rate,
        "selected_only_error": abs(selected_rate - census_rate),
        "validation": validation_obj,
        "estimated_population": estimated_population,
    }


def arm6_joint_cell_inclusion_probabilities(
    *,
    descriptors: Sequence[SessionDescriptor],
    selected_ids: Sequence[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    by_id = {descriptor.unit_id: descriptor for descriptor in descriptors}
    selected = tuple(str(uid) for uid in selected_ids)
    selected_set = set(selected)
    cells: dict[tuple[str, str], list[str]] = {}
    for descriptor in descriptors:
        key = (descriptor.agent_id, descriptor.business_use_case_guid)
        cells.setdefault(key, []).append(descriptor.unit_id)

    n_by_cell: dict[tuple[str, str], int] = {key: 0 for key in cells}
    for unit_id in selected_set:
        descriptor = by_id.get(unit_id)
        if descriptor is None:
            continue
        key = (descriptor.agent_id, descriptor.business_use_case_guid)
        if key in n_by_cell:
            n_by_cell[key] += 1

    represented_cells = {key for key, n_au in n_by_cell.items() if n_au > 0}
    zero_sample_cells = {key for key, n_au in n_by_cell.items() if n_au <= 0}

    pi_by_unit: dict[str, float] = {}
    weight_by_unit: dict[str, float] = {}
    reasons_by_unit: dict[str, str] = {}
    for unit_id in selected:
        descriptor = by_id.get(unit_id)
        if descriptor is None:
            continue
        key = (descriptor.agent_id, descriptor.business_use_case_guid)
        N_au = int(len(cells.get(key, ())))
        n_au = int(n_by_cell.get(key, 0))
        if N_au <= 0 or n_au <= 0:
            continue
        pi = float(n_au) / float(N_au)
        if not (0.0 < pi <= 1.0):
            raise ValueError(f"invalid arm6 represented-cell inclusion probability for unit {unit_id}: {pi}")
        pi_by_unit[unit_id] = pi
        weight_by_unit[unit_id] = float(N_au) / float(n_au)
        reasons_by_unit[unit_id] = "hajek-represented-joint-cell-poststratified-agent-use-case"

    total_population = int(len(descriptors))
    zero_sample_population = int(sum(len(cells[key]) for key in zero_sample_cells))
    represented_population = total_population - zero_sample_population
    weights = tuple(weight_by_unit.values())
    weight_sum = float(sum(weights))
    weight_sumsq = float(sum(weight * weight for weight in weights))
    diagnostics: dict[str, Any] = {
        "design": "represented_joint_cell_poststratified_hajek_agent_use_case",
        "total_cell_count": int(len(cells)),
        "represented_cell_count": int(len(represented_cells)),
        "zero_sample_cell_count": int(len(zero_sample_cells)),
        "population_count_in_zero_sample_cells": zero_sample_population,
        "represented_population_fraction": (float(represented_population) / float(total_population)) if total_population > 0 else 1.0,
        "weight_sum": weight_sum,
        "weight_ess": ((weight_sum * weight_sum) / weight_sumsq) if weight_sumsq > 0.0 else 0.0,
        "max_weight": max(weights) if weights else 0.0,
        "represented_reason": "post-stratified Hajek on represented (agent_id,use_case_guid) cells only",
        "cell_sizes": {
            f"{agent_id}|{use_case_guid}": {
                "N_au": int(len(unit_ids)),
                "n_au": int(n_by_cell[(agent_id, use_case_guid)]),
            }
            for (agent_id, use_case_guid), unit_ids in sorted(cells.items())
        },
    }
    return pi_by_unit, weight_by_unit, diagnostics


def select_arm6(*, descriptors: Sequence[SessionDescriptor], arm4_outcome: SelectionOutcome) -> SelectionOutcome:
    selected_ids = tuple(arm4_outcome.selected_ids)
    if not selected_ids:
        _pi, _weights, diagnostics = arm6_joint_cell_inclusion_probabilities(descriptors=descriptors, selected_ids=selected_ids)
        return SelectionOutcome(
            method_id=METHOD_IDS["arm6"],
            selected_ids=selected_ids,
            records=(),
            per_agent_counts=dict(arm4_outcome.per_agent_counts),
            inclusion_probability_by_unit={},
            diagnostics=diagnostics,
        )

    by_id = {descriptor.unit_id: descriptor for descriptor in descriptors}
    pi_by_unit, weight_by_unit, diagnostics = arm6_joint_cell_inclusion_probabilities(descriptors=descriptors, selected_ids=selected_ids)
    records: list[SelectionRecord] = []
    for unit_id in selected_ids:
        descriptor = by_id.get(unit_id)
        if descriptor is None:
            continue
        pi = float(pi_by_unit.get(unit_id, 0.0))
        if not (0.0 < pi <= 1.0):
            raise ValueError(f"missing represented-cell inclusion probability for selected unit {unit_id}")
        records.append(
            SelectionRecord(
                unit_id=unit_id,
                method_id=METHOD_IDS["arm6"],
                stratum=f"{descriptor.agent_id}|{descriptor.business_use_case_guid}",
                inclusion_probability=pi,
                weight=float(weight_by_unit[unit_id]),
                reason="hajek-represented-joint-cell-poststratified-agent-use-case",
            )
        )
    return SelectionOutcome(
        method_id=METHOD_IDS["arm6"],
        selected_ids=selected_ids,
        records=tuple(records),
        per_agent_counts=dict(arm4_outcome.per_agent_counts),
        inclusion_probability_by_unit={record.unit_id: float(record.inclusion_probability or 0.0) for record in records},
        diagnostics=diagnostics,
    )


def _hajek_estimate(selected_ids: Sequence[str], labels_by_unit: Mapping[str, bool], inclusion_probabilities: Mapping[str, float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for unit_id in selected_ids:
        pi = float(inclusion_probabilities.get(unit_id, 1.0))
        if not (0.0 < pi <= 1.0):
            raise ValueError(f"inclusion probability must satisfy 0 < pi <= 1 for {unit_id}, got {pi}")
        y = 1.0 if bool(labels_by_unit.get(unit_id, False)) else 0.0
        numerator += y / pi
        denominator += 1.0 / pi
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def run_arm2_idw(
    *,
    eligible_ids: Sequence[str],
    selected_ids: Sequence[str],
    agent_id_by_unit: Mapping[str, str],
    vector_by_unit: Mapping[str, Any],
    labels_by_unit: Mapping[str, bool],
    cell_id: str = "v6-arm2",
    config: IDWConfig = IDWConfig(),
) -> dict[str, Any]:
    eligible = tuple(sorted(set(str(uid) for uid in eligible_ids)))
    selected = tuple(sorted(set(str(uid) for uid in selected_ids)))
    expected_labels = {uid: float(1.0 if bool(labels_by_unit.get(uid, False)) else 0.0) for uid in eligible}
    membership = freeze_membership(cell_id=cell_id, eligible_ids=eligible, selected_ids=selected)
    estimates = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit={uid: str(agent_id_by_unit[uid]) for uid in eligible},
        vector_by_unit={uid: vector_by_unit[uid] for uid in eligible},
        judged_values_by_unit={uid: expected_labels[uid] for uid in selected if uid in expected_labels},
        config=config,
    )
    validation = validate_embedding_population(estimates, expected_labels)
    selected_rate = float(sum(expected_labels.get(uid, 0.0) for uid in selected) / max(1, len(selected))) if selected else 0.0
    census_rate = float(sum(expected_labels.values()) / max(1, len(expected_labels))) if expected_labels else 0.0
    return {
        "method_id": METHOD_IDS["arm2"],
        "membership": membership,
        "estimate": float(estimates.aggregate.estimated_pass_rate),
        "selected_rate": selected_rate,
        "selected_only_error": abs(selected_rate - census_rate),
        "validation": validation,
        "estimated_population": estimates,
    }


def _population_coverage(selected_values: Sequence[str], all_values: Sequence[str]) -> float:
    if not all_values:
        return 1.0
    if not selected_values:
        return 0.0
    return len(set(selected_values)) / len(set(all_values))


def compute_trial_metrics(
    *,
    descriptors: Sequence[SessionDescriptor],
    selected_ids: Sequence[str],
    method_id: str,
    trial_seed: int,
    window_id: str,
    nominal_budget: int,
    labels_by_unit: Mapping[str, bool],
    actual_token_count: int | None = None,
    idw_result: Mapping[str, Any] | None = None,
    arm4_outcome: SelectionOutcome | None = None,
    selection_outcome: SelectionOutcome | None = None,
) -> TrialMetrics:
    selected = tuple(sorted({str(uid) for uid in selected_ids}))
    pop_ids = tuple(sorted({descriptor.unit_id for descriptor in descriptors}))
    labels = {unit_id: 1.0 if bool(labels_by_unit.get(unit_id, False)) else 0.0 for unit_id in pop_ids}
    census_rate = float(sum(labels.values()) / max(1, len(labels))) if labels else 0.0
    if method_id == METHOD_IDS["arm2"] and idw_result is not None:
        estimate = float(idw_result.get("estimate", census_rate))
    elif method_id == METHOD_IDS["arm2_5"] and idw_result is not None:
        if "binary_estimate" in idw_result:
            estimate = float(idw_result.get("binary_estimate", census_rate))
        elif idw_result.get("estimated_population") is not None:
            estimate = arm2_5_binary_estimate_from_population(idw_result.get("estimated_population"))
        else:
            estimate = float(idw_result.get("estimate", census_rate))
    elif method_id == METHOD_IDS["arm5"] and arm4_outcome is not None:
        inclusion = arm4_inclusion_probabilities(descriptors=descriptors, cap=len(arm4_outcome.selected_ids), trial_seed=trial_seed, window_id=window_id)
        estimate = _hajek_estimate(selected, labels_by_unit, inclusion)
    elif method_id == METHOD_IDS["arm6"] and selection_outcome is not None:
        inclusion = dict(selection_outcome.inclusion_probability_by_unit)
        if not inclusion:
            inclusion = {
                record.unit_id: float(record.inclusion_probability)
                for record in selection_outcome.records
                if record.inclusion_probability is not None
            }
        estimate = _hajek_estimate(selected, labels_by_unit, inclusion) if selected else 0.0
    else:
        estimate = float(sum(labels.get(uid, 0.0) for uid in selected) / max(1, len(selected))) if selected else 0.0
    selected_label_rate = float(sum(labels.get(uid, 0.0) for uid in selected) / max(1, len(selected))) if selected else 0.0
    absolute_aggregate_mae = abs(estimate - census_rate)
    concept_population = sorted({descriptor.concept_key for descriptor in descriptors})
    concept_selected = sorted({descriptor.concept_key for descriptor in descriptors if descriptor.unit_id in set(selected)})
    use_case_population = sorted({descriptor.use_case_id for descriptor in descriptors})
    use_case_selected = sorted({descriptor.use_case_id for descriptor in descriptors if descriptor.unit_id in set(selected)})
    by_agent: dict[str, list[SessionDescriptor]] = {}
    for descriptor in descriptors:
        by_agent.setdefault(descriptor.agent_id, []).append(descriptor)
    ordered_agents = sorted(by_agent, key=lambda agent: (-len(by_agent[agent]), agent))[:5]
    top_five_agents: list[dict[str, Any]] = []
    for agent_id in ordered_agents:
        pop_ids_agent = tuple(sorted(session.unit_id for session in by_agent[agent_id]))
        selected_agent_ids = tuple(uid for uid in pop_ids_agent if uid in set(selected))
        selected_rate_agent = float(sum(labels.get(uid, 0.0) for uid in selected_agent_ids) / max(1, len(selected_agent_ids))) if selected_agent_ids else None
        census_rate_agent = float(sum(labels.get(uid, 0.0) for uid in pop_ids_agent) / max(1, len(pop_ids_agent))) if pop_ids_agent else 0.0
        abs_err = None if selected_rate_agent is None else abs(selected_rate_agent - census_rate_agent)
        concepts_agent_pop = {session.concept_key for session in by_agent[agent_id]}
        concepts_agent_sel = {session.concept_key for session in by_agent[agent_id] if session.unit_id in set(selected)}
        use_cases_agent_pop = {session.use_case_id for session in by_agent[agent_id]}
        use_cases_agent_sel = {session.use_case_id for session in by_agent[agent_id] if session.unit_id in set(selected)}
        top_five_agents.append(
            {
                "agent_id": agent_id,
                "N": len(pop_ids_agent),
                "n": len(selected_agent_ids),
                "selected_rate": selected_rate_agent,
                "census_rate": census_rate_agent,
                "absolute_error": abs_err,
                "concept_coverage": (len(concepts_agent_sel) / len(concepts_agent_pop)) if concepts_agent_pop else 1.0,
                "use_case_coverage": (len(use_cases_agent_sel) / len(use_cases_agent_pop)) if use_cases_agent_pop else 1.0,
            }
        )
    all_agents = sorted({descriptor.agent_id for descriptor in descriptors})
    selected_agents = sorted({descriptor.agent_id for descriptor in descriptors if descriptor.unit_id in set(selected)})
    return TrialMetrics(
        method_id=method_id,
        trial_seed=trial_seed,
        window_id=window_id,
        nominal_budget=nominal_budget,
        sample_size=len(selected),
        estimate=estimate,
        census_pass_rate=census_rate,
        absolute_aggregate_mae=absolute_aggregate_mae,
        actual_token_count=(int(actual_token_count) if actual_token_count is not None else len(selected) * NOMINAL_TOKENS_PER_SESSION),
        concept_coverage=_population_coverage(concept_selected, concept_population),
        use_case_coverage=_population_coverage(use_case_selected, use_case_population),
        selected_label_rate=selected_label_rate,
        agent_coverage=(len(set(selected_agents)) / len(set(all_agents))) if all_agents else 1.0,
        top_five_agents=tuple(top_five_agents),
        selected_ids=selected,
        selected_only_rate=selected_label_rate,
        selected_only_absolute_error=abs(selected_label_rate - census_rate),
        idw_validation=(
            dict(idw_result.get("validation"))
            if idw_result and isinstance(idw_result.get("validation"), Mapping)
            else (asdict(idw_result.get("validation")) if idw_result and is_dataclass(idw_result.get("validation")) else None)
        ),
        idw_population_estimate=(float(idw_result.get("estimate")) if idw_result and isinstance(idw_result.get("estimate"), (int, float)) else None),
        arm4_membership_ids=tuple(),
    )


def arm4_and_arm5_membership_identity(*, descriptors: Sequence[SessionDescriptor], cap: int, trial_seed: int = 13, window_id: str = "window-1") -> tuple[tuple[str, ...], tuple[str, ...]]:
    arm4 = select_arm4(descriptors=descriptors, cap=cap, trial_seed=trial_seed, window_id=window_id)
    labels_by_unit = {descriptor.unit_id: descriptor.label for descriptor in descriptors}
    arm5 = select_arm5(descriptors=descriptors, arm4_outcome=arm4, labels_by_unit=labels_by_unit, trial_seed=trial_seed, window_id=window_id)
    return arm4.selected_ids, arm5.selected_ids


def _make_dense_fixture() -> tuple[tuple[SessionDescriptor, ...], dict[str, bool]]:
    out: list[SessionDescriptor] = []
    labels: dict[str, bool] = {}
    for agent_idx in range(1, 6):
        for idx in range(1, 9):
            unit_id = f"agent-{agent_idx}-u{idx}"
            descriptor = SessionDescriptor(
                unit_id=unit_id,
                agent_id=f"agent-{agent_idx}",
                use_case_id=f"use-case-{(idx % 3) + 1}",
                business_use_case_guid=f"use-case-{(idx % 3) + 1}",
                concept_key=f"concept-{(agent_idx + idx) % 4}",
                label=(idx + agent_idx) % 2 == 0,
            )
            out.append(descriptor)
            labels[unit_id] = descriptor.label
    return tuple(out), labels


def validate_selection_exactness(*, outcome: SelectionOutcome, population: Sequence[SessionDescriptor], cap: int) -> None:
    if len(outcome.selected_ids) != min(cap, len(population)):
        raise ValueError("selected count mismatch")
    if len(set(outcome.selected_ids)) != len(outcome.selected_ids):
        raise ValueError("selected ids are not unique")
    population_ids = {descriptor.unit_id for descriptor in population}
    if not set(outcome.selected_ids).issubset(population_ids):
        raise ValueError("selected ids are not a subset of the population")


__all__ = [
    "SAMPLE_CAPS",
    "NOMINAL_TOKENS_PER_SESSION",
    "TRIAL_SEEDS",
    "METHOD_IDS",
    "METHOD_ID_ORDER",
    "SessionDescriptor",
    "SelectionRecord",
    "SelectionOutcome",
    "TrialMetrics",
    "build_session_descriptors",
    "select_arm1",
    "select_arm3",
    "select_arm4",
    "select_arm5",
    "select_arm6",
    "run_arm2_idw",
    "run_arm2_5_binary_from_arm2_result",
    "arm2_5_binary_estimate_from_rows",
    "arm2_5_binary_estimate_from_population",
    "compute_trial_metrics",
    "arm4_inclusion_probabilities",
    "arm6_joint_cell_inclusion_probabilities",
    "_stable_sha256_hex",
    "_hash_float",
    "arm4_and_arm5_membership_identity",
    "_make_dense_fixture",
    "validate_selection_exactness",
]
