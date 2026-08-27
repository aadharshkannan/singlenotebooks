from __future__ import annotations

import math
from collections import defaultdict

import pytest

from sampling_comparison.v6_experiment import (
    METHOD_IDS,
    SAMPLE_CAPS,
    TrialMetrics,
    _allocate_round_robin_counts,
    _build_agent_queues,
    _expected_agent_slots,
    _systematic_stratum_assignments,
    _window_identity,
    arm4_and_arm5_membership_identity,
    arm4_inclusion_probabilities,
    build_session_descriptors,
    compute_trial_metrics,
    run_arm2_idw,
    select_arm1,
    select_arm3,
    select_arm4,
    validate_selection_exactness,
)


def _make_fixture() -> tuple[list[dict[str, object]], dict[str, bool]]:
    out: list[dict[str, object]] = []
    labels: dict[str, bool] = {}
    for agent_idx in range(1, 6):
        agent_id = f"agent-{agent_idx}"
        for idx in range(1, 9):
            unit_id = f"{agent_id}-u{idx}"
            use_case_id = f"use-case-{(idx % 3) + 1}"
            concept_key = f"concept-{(agent_idx + idx) % 4}"
            label = (idx + agent_idx) % 2 == 0
            out.append(
                {
                    "unit_id": unit_id,
                    "agent_id": agent_id,
                    "use_case_id": use_case_id,
                    "concept_key": concept_key,
                    "label": label,
                }
            )
            labels[unit_id] = label
    return out, labels


def test_v6_exact_caps_and_no_label_leakage():
    rows, labels = _make_fixture()
    descriptors = build_session_descriptors(rows)
    for cap in SAMPLE_CAPS:
        outcome = select_arm1(descriptors=descriptors, cap=cap)
        assert len(outcome.selected_ids) == min(cap, len(descriptors))
        assert len(set(outcome.selected_ids)) == len(outcome.selected_ids)
        assert set(outcome.selected_ids).issubset({d.unit_id for d in descriptors})
        permuted = [{**row, "label": not row["label"], "use_case_id": row["use_case_id"]} for row in rows]
        permuted_desc = build_session_descriptors(permuted)
        same = select_arm1(descriptors=permuted_desc, cap=cap)
        assert set(same.selected_ids) == set(outcome.selected_ids)


def test_v6_arm3_arm4_and_arm5_membership_identity_and_replayability():
    rows, _labels = _make_fixture()
    descriptors = build_session_descriptors(rows)
    arm3 = select_arm3(descriptors=descriptors, cap=64)
    arm4 = select_arm4(descriptors=descriptors, cap=64)
    labels_by_unit = {d.unit_id: d.label for d in descriptors}
    from sampling_comparison.v6_experiment import select_arm5

    arm5 = select_arm5(descriptors=descriptors, arm4_outcome=arm4, labels_by_unit=labels_by_unit)
    assert len(arm3.selected_ids) == min(64, len(descriptors))
    assert arm4.selected_ids == arm5.selected_ids
    arm4_pi = {record.unit_id: record.inclusion_probability for record in arm4.records}
    arm5_pi = {record.unit_id: record.inclusion_probability for record in arm5.records}
    assert arm4_pi == arm5_pi
    replay = select_arm4(descriptors=descriptors, cap=64)
    assert replay.selected_ids == arm4.selected_ids


def test_v6_arm3_floor_cap_behavior_and_exact_selection_counts():
    rows, _labels = _make_fixture()
    descriptors = build_session_descriptors(rows)
    method_id = METHOD_IDS["arm3"]
    window_identity = _window_identity("window-1", 13, method_id)
    queues, agent_order = _build_agent_queues(descriptors=descriptors, window_identity=window_identity)
    floor_prefix: list[str] = []
    floor_limit = {agent: min(3, len(queues[agent])) for agent in agent_order}
    taken = {agent: 0 for agent in agent_order}
    while True:
        progressed = False
        for agent in agent_order:
            if taken[agent] < floor_limit[agent]:
                floor_prefix.append(queues[agent][taken[agent]].unit_id)
                taken[agent] += 1
                progressed = True
        if not progressed:
            break

    total_floor = len(floor_prefix)
    for cap in [1, 2, 3, 4, 5, 10, 20, 30]:
        arm3 = select_arm3(descriptors=descriptors, cap=cap)
        validate_selection_exactness(outcome=arm3, population=descriptors, cap=cap)
        assert len(arm3.selected_ids) == min(cap, len(descriptors))
        expected_prefix = tuple(floor_prefix[: min(cap, total_floor)])
        assert arm3.selected_ids[: len(expected_prefix)] == expected_prefix
        if cap <= len(agent_order):
            selected_agents = {d.agent_id for d in descriptors if d.unit_id in arm3.selected_ids}
            assert len(selected_agents) == cap


def test_v6_stratum_proportional_allocation():
    rows = [
        {"unit_id": f"u{i}", "agent_id": "agent-a", "use_case_id": "use-case-1", "concept_key": f"c{i}", "label": i % 2 == 0}
        for i in range(1, 11)
    ]
    rows += [
        {"unit_id": f"u{i}", "agent_id": "agent-b", "use_case_id": "use-case-2", "concept_key": f"c{i}", "label": i % 2 == 1}
        for i in range(11, 21)
    ]
    descriptors = build_session_descriptors(rows)
    out = select_arm4(descriptors=descriptors, cap=8)
    assert len(out.selected_ids) == 8
    by_agent = defaultdict(lambda: defaultdict(int))
    by_id = {d.unit_id: d for d in descriptors}
    for unit_id in out.selected_ids:
        descriptor = by_id[unit_id]
        by_agent[descriptor.agent_id][descriptor.use_case_id] += 1
    assert sum(by_agent["agent-a"].values()) == 4
    assert sum(by_agent["agent-b"].values()) == 4


def test_v6_arm4_arm5_same_membership_and_hajek_estimator():
    rows, labels = _make_fixture()
    descriptors = build_session_descriptors(rows)
    arm4 = select_arm4(descriptors=descriptors, cap=16, trial_seed=15)
    from sampling_comparison.v6_experiment import select_arm5

    arm5 = select_arm5(
        descriptors=descriptors,
        arm4_outcome=arm4,
        labels_by_unit=labels,
        trial_seed=15,
    )
    assert arm4.selected_ids == arm5.selected_ids
    ids, ids2 = arm4_and_arm5_membership_identity(descriptors=descriptors, cap=16, trial_seed=15)
    assert ids == ids2
    probs = arm4_inclusion_probabilities(descriptors=descriptors, cap=16, trial_seed=15)
    for uid, pi in probs.items():
        assert 0.0 < pi <= 1.0
    estimate = sum((1.0 if labels[uid] else 0.0) / probs[uid] for uid in arm4.selected_ids) / sum(1.0 / probs[uid] for uid in arm4.selected_ids)
    assert math.isfinite(estimate)
    assert 0.0 <= estimate <= 1.0


def test_v6_uneven_capacities_and_cap_below_agent_count_inclusion_probabilities():
    rows = [
        {"unit_id": f"u{i}", "agent_id": agent, "use_case_id": "use-case-1", "concept_key": f"c{i}", "label": i % 2 == 0}
        for i, agent in enumerate(["a", "a", "a", "b", "b", "c"]) if i < 6
    ]
    descriptors = build_session_descriptors(rows)
    probs = arm4_inclusion_probabilities(descriptors=descriptors, cap=2, trial_seed=13)
    assert probs
    for pi in probs.values():
        assert 0.0 < pi <= 1.0


def test_v6_expected_agent_slots_capacity_aware_sums_to_cap():
    expected = _expected_agent_slots({"a": 1, "b": 100}, 50)
    assert expected["a"] == pytest.approx(1.0)
    assert expected["b"] == pytest.approx(49.0)
    assert sum(expected.values()) == pytest.approx(50.0)


def test_v6_systematic_assignments_exact_total_capacity_and_marginals():
    pop = {"s1": 1, "s2": 2, "s3": 7}
    for desired in range(0, 11):
        counts = _systematic_stratum_assignments(pop, desired, trial_seed=13, window_id="w", agent_id="a")
        assert sum(counts.values()) == min(desired, sum(pop.values()))
        for key, value in counts.items():
            assert 0 <= value <= pop[key]

    desired = 4
    runs = 2000
    extra_counts = {"s1": 0, "s2": 0, "s3": 0}
    ideal = {k: desired * (v / sum(pop.values())) for k, v in pop.items()}
    floors = {k: math.floor(ideal[k]) for k in pop}
    frac = {k: ideal[k] - floors[k] for k in pop}
    for seed in range(runs):
        counts = _systematic_stratum_assignments(pop, desired, trial_seed=seed, window_id="w", agent_id="a")
        for key in pop:
            extra_counts[key] += int(counts[key] - floors[key])
    for key in pop:
        empirical = extra_counts[key] / runs
        assert empirical == pytest.approx(frac[key], abs=0.06)


def test_v6_arm4_empirical_inclusion_matches_pi_small_uneven_fixture():
    rows = []
    for idx in range(1, 2):
        rows.append({"unit_id": f"a-{idx}", "agent_id": "a", "use_case_id": "u1", "concept_key": f"c{idx}", "label": False})
    for idx in range(1, 6):
        rows.append({"unit_id": f"b-{idx}", "agent_id": "b", "use_case_id": "u1" if idx <= 3 else "u2", "concept_key": f"c{idx}", "label": False})
    descriptors = build_session_descriptors(rows)
    cap = 3
    target_pi = arm4_inclusion_probabilities(descriptors=descriptors, cap=cap, trial_seed=13, window_id="base")

    observed = {d.unit_id: 0 for d in descriptors}
    runs = 500
    for seed in range(runs):
        out = select_arm4(descriptors=descriptors, cap=cap, trial_seed=13, window_id=f"trial-{seed}")
        for unit_id in out.selected_ids:
            observed[unit_id] += 1
    for unit_id, count in observed.items():
        empirical = count / runs
        assert empirical == pytest.approx(target_pi[unit_id], abs=0.08)


def test_v6_trial_variation_and_no_label_leakage_for_arm3_arm4():
    rows, _labels = _make_fixture()
    descriptors = build_session_descriptors(rows)
    mutated = [{**row, "label": not bool(row["label"])} for row in rows]
    descriptors_mutated = build_session_descriptors(mutated)

    arm3_a = select_arm3(descriptors=descriptors, cap=12, trial_seed=13, window_id="w-a")
    arm3_b = select_arm3(descriptors=descriptors, cap=12, trial_seed=13, window_id="w-b")
    assert arm3_a.selected_ids != arm3_b.selected_ids
    assert select_arm3(descriptors=descriptors_mutated, cap=12, trial_seed=13, window_id="w-a").selected_ids == arm3_a.selected_ids

    arm4_a = select_arm4(descriptors=descriptors, cap=12, trial_seed=13, window_id="w-a")
    arm4_b = select_arm4(descriptors=descriptors, cap=12, trial_seed=13, window_id="w-b")
    assert arm4_a.selected_ids != arm4_b.selected_ids
    assert select_arm4(descriptors=descriptors_mutated, cap=12, trial_seed=13, window_id="w-a").selected_ids == arm4_a.selected_ids


def test_v6_concepts_use_cases_and_metrics():
    rows, labels = _make_fixture()
    descriptors = build_session_descriptors(rows)
    selected = select_arm1(descriptors=descriptors, cap=12, trial_seed=13)
    metrics = compute_trial_metrics(
        descriptors=descriptors,
        selected_ids=selected.selected_ids,
        method_id=METHOD_IDS["arm1"],
        trial_seed=13,
        window_id="window-1",
        nominal_budget=12,
        labels_by_unit=labels,
        actual_token_count=12345,
    )
    assert isinstance(metrics, TrialMetrics)
    assert metrics.sample_size == 12
    assert metrics.actual_token_count == 12345
    assert 0.0 <= metrics.concept_coverage <= 1.0
    assert 0.0 <= metrics.use_case_coverage <= 1.0
    assert 0.0 <= metrics.agent_coverage <= 1.0
    assert metrics.top_five_agents


def test_v6_arm2_idw_population_aggregation_from_full_population():
    rows, labels = _make_fixture()
    descriptors = build_session_descriptors(rows)
    eligible_ids = [d.unit_id for d in descriptors]
    agent_ids = {d.unit_id: d.agent_id for d in descriptors}
    vector_by_unit = {d.unit_id: [float((ord(ch) % 7) + 1) for ch in d.unit_id] for d in descriptors}
    result = run_arm2_idw(
        eligible_ids=eligible_ids,
        selected_ids=eligible_ids[:10],
        agent_id_by_unit=agent_ids,
        vector_by_unit=vector_by_unit,
        labels_by_unit=labels,
    )
    assert result["estimate"] >= 0.0
    assert result["selected_only_error"] >= 0.0
    assert result["validation"].absolute_aggregate_rate_error >= 0.0


def test_v6_arm5_reuses_exact_arm4_probabilities_and_ids():
    rows, labels = _make_fixture()
    descriptors = build_session_descriptors(rows)
    arm4 = select_arm4(descriptors=descriptors, cap=14, trial_seed=15, window_id="same")
    from sampling_comparison.v6_experiment import select_arm5

    arm5 = select_arm5(descriptors=descriptors, arm4_outcome=arm4, labels_by_unit=labels, trial_seed=111, window_id="different")
    assert arm5.selected_ids == arm4.selected_ids
    assert arm5.inclusion_probability_by_unit == arm4.inclusion_probability_by_unit
    arm4_record_pi = {record.unit_id: record.inclusion_probability for record in arm4.records}
    arm5_record_pi = {record.unit_id: record.inclusion_probability for record in arm5.records}
    assert arm5_record_pi == arm4_record_pi
