from __future__ import annotations

from dataclasses import replace
from math import acos, pi

import numpy as np
import pytest

from sampling_comparison.v4_idw import (
    FrozenMembership,
    IDWConfig,
    estimate_embedding_population,
    freeze_membership,
    leave_one_out_donor_diagnostics,
    validate_embedding_population,
)


def _vec(*coords: float) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n == 0.0:
        return arr
    return arr / n


def _base_fixture():
    eligible_ids = ("a_s1", "a_s2", "a_u1", "a_u2", "b_u1")
    selected_ids = ("a_s1", "a_s2")
    membership = freeze_membership(cell_id="cell-1", eligible_ids=eligible_ids, selected_ids=selected_ids)

    agent_id_by_unit = {
        "a_s1": "t|a",
        "a_s2": "t|a",
        "a_u1": "t|a",
        "a_u2": "t|a",
        "b_u1": "t|b",
    }

    vector_by_unit = {
        "a_s1": _vec(1.0, 0.0, 0.0),
        "a_s2": _vec(0.0, 1.0, 0.0),
        "a_u1": _vec(0.98, 0.2, 0.0),
        "a_u2": _vec(0.4, 0.9, 0.0),
        "b_u1": _vec(1.0, 0.0, 0.0),
    }

    judged_values_by_unit = {
        "a_s1": 1.0,
        "a_s2": 0.0,
    }
    return membership, agent_id_by_unit, vector_by_unit, judged_values_by_unit


def _rows_by_id(est):
    return {row.unit_id: row for row in est.rows}


def test_exact_selected_values_preserved():
    membership, agent, vecs, judged = _base_fixture()
    est = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vecs,
        judged_values_by_unit=judged,
    )
    rows = _rows_by_id(est)
    assert rows["a_s1"].value == pytest.approx(1.0)
    assert rows["a_s2"].value == pytest.approx(0.0)
    assert rows["a_s1"].provenance == "observed"
    assert rows["a_s2"].provenance == "observed"


def test_idw_favors_nearest_and_formula_matches():
    membership, agent, vecs, judged = _base_fixture()
    est = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vecs,
        judged_values_by_unit=judged,
        config=IDWConfig(k=2, power=2.0, eps=1e-6),
    )
    row = _rows_by_id(est)["a_u1"]
    assert row.provenance == "idw"
    assert row.donor_ids[0] == "a_s1"

    d1 = acos(float(np.clip(np.dot(vecs["a_u1"], vecs["a_s1"]), -1.0, 1.0))) / pi
    d2 = acos(float(np.clip(np.dot(vecs["a_u1"], vecs["a_s2"]), -1.0, 1.0))) / pi
    w1 = 1.0 / ((d1 + 1e-6) ** 2)
    w2 = 1.0 / ((d2 + 1e-6) ** 2)
    expected = w1 / (w1 + w2)
    assert row.value == pytest.approx(expected, rel=1e-8, abs=1e-8)


def test_exact_match_averaging_ignores_non_exact():
    membership = freeze_membership(
        cell_id="cell-exact",
        eligible_ids=("d1", "d2", "d3", "u"),
        selected_ids=("d1", "d2", "d3"),
    )
    agent = {"d1": "t|a", "d2": "t|a", "d3": "t|a", "u": "t|a"}
    vecs = {
        "d1": _vec(1.0, 0.0, 0.0),
        "d2": _vec(1.0, 0.0, 0.0),
        "d3": _vec(0.0, 1.0, 0.0),
        "u": _vec(1.0, 0.0, 0.0),
    }
    judged = {"d1": 1.0, "d2": 0.0, "d3": 1.0}

    est = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vecs,
        judged_values_by_unit=judged,
    )
    row = _rows_by_id(est)["u"]
    assert row.provenance == "exact_match"
    assert set(row.donor_ids) == {"d1", "d2"}
    assert row.value == pytest.approx(0.5)


def test_no_cross_agent_donors():
    membership, agent, vecs, judged = _base_fixture()
    # Make the other-agent unit close to a_s1 but it must not use cross-agent donors.
    vecs["b_u1"] = _vec(1.0, 0.0, 0.0)
    est = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vecs,
        judged_values_by_unit=judged,
    )
    row = _rows_by_id(est)["b_u1"]
    assert row.provenance in {"global_mean", "prior"}
    assert row.donor_ids == ()


def test_zero_donor_agent_global_mean_and_prior_without_global():
    membership, agent, vecs, judged = _base_fixture()
    est = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vecs,
        judged_values_by_unit=judged,
    )
    row = _rows_by_id(est)["b_u1"]
    assert row.provenance == "global_mean"
    assert row.value == pytest.approx(0.5)

    membership2 = freeze_membership(
        cell_id="cell-prior",
        eligible_ids=("x", "y"),
        selected_ids=(),
    )
    agent2 = {"x": "t|x", "y": "t|y"}
    vecs2 = {"x": _vec(1, 0), "y": _vec(0, 1)}
    est2 = estimate_embedding_population(
        membership=membership2,
        agent_id_by_unit=agent2,
        vector_by_unit=vecs2,
        judged_values_by_unit={},
    )
    for row2 in est2.rows:
        assert row2.provenance == "prior"
        assert row2.value == pytest.approx(0.5)


def test_invalid_vector_with_same_agent_donors_uses_agent_mean():
    membership, agent, vecs, judged = _base_fixture()
    vecs["a_u1"] = np.asarray([np.nan, 0.0, 0.0], dtype=np.float64)
    est = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vecs,
        judged_values_by_unit=judged,
    )
    row = _rows_by_id(est)["a_u1"]
    assert row.provenance == "agent_mean"
    assert row.value == pytest.approx(0.5)


def test_valid_target_with_only_invalid_same_agent_donors_uses_agent_mean():
    membership = freeze_membership(
        cell_id="cell-invalid-donors",
        eligible_ids=("a_s1", "a_u1", "b_s1"),
        selected_ids=("a_s1", "b_s1"),
    )
    agent = {"a_s1": "t|a", "a_u1": "t|a", "b_s1": "t|b"}
    vecs = {
        "a_s1": np.asarray([np.nan, 0.0], dtype=np.float64),
        "a_u1": _vec(1.0, 0.0),
        "b_s1": _vec(0.0, 1.0),
    }
    judged = {"a_s1": 1.0, "b_s1": 0.0}

    est = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vecs,
        judged_values_by_unit=judged,
    )

    row = _rows_by_id(est)["a_u1"]
    assert row.provenance == "agent_mean"
    assert row.value == pytest.approx(1.0)


def test_k_and_tie_deterministic():
    membership = freeze_membership(
        cell_id="cell-tie",
        eligible_ids=("d1", "d2", "d3", "u"),
        selected_ids=("d1", "d2", "d3"),
    )
    agent = {"d1": "t|a", "d2": "t|a", "d3": "t|a", "u": "t|a"}
    vecs = {
        "d1": _vec(1.0, 0.0),
        "d2": _vec(0.0, 1.0),
        "d3": _vec(-1.0, 0.0),
        "u": _vec(1.0, 1.0),
    }
    judged = {"d1": 0.0, "d2": 1.0, "d3": 1.0}

    est = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vecs,
        judged_values_by_unit=judged,
        config=IDWConfig(k=2),
    )
    row = _rows_by_id(est)["u"]
    # d1 and d2 are equidistant to u, tie must break by unit id.
    assert row.donor_ids == ("d1", "d2")


def test_membership_hash_tamper_and_judged_outside_membership_reject():
    membership, agent, vecs, judged = _base_fixture()
    with pytest.raises(ValueError, match="membership_hash"):
        FrozenMembership(
            cell_id=membership.cell_id,
            membership_hash="bad",
            population_hash=membership.population_hash,
            eligible_ids=membership.eligible_ids,
            selected_ids=membership.selected_ids,
        )

    judged_bad = dict(judged)
    judged_bad["a_u1"] = 0.2
    with pytest.raises(ValueError, match="outside selected_ids"):
        estimate_embedding_population(
            membership=membership,
            agent_id_by_unit=agent,
            vector_by_unit=vecs,
            judged_values_by_unit=judged_bad,
        )


def test_estimator_without_labels_and_label_permutation_only_changes_validation():
    membership, agent, vecs, judged = _base_fixture()
    est = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vecs,
        judged_values_by_unit=judged,
    )

    labels_a = {
        "a_s1": 1.0,
        "a_s2": 0.0,
        "a_u1": 1.0,
        "a_u2": 0.0,
        "b_u1": 1.0,
    }
    labels_b = {
        "a_s1": 0.0,
        "a_s2": 1.0,
        "a_u1": 0.0,
        "a_u2": 1.0,
        "b_u1": 0.0,
    }

    va = validate_embedding_population(est, labels_a)
    vb = validate_embedding_population(est, labels_b)

    assert va.per_unit_mae != vb.per_unit_mae
    assert [row.value for row in est.rows] == [row.value for row in est.rows]


def test_aggregate_uses_observed_plus_imputed_and_differs_from_judged_only_fixture():
    membership, agent, vecs, judged = _base_fixture()
    est = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vecs,
        judged_values_by_unit=judged,
    )
    judged_only = float(np.mean(np.asarray(list(judged.values()), dtype=np.float64)))
    assert est.aggregate.population_count == len(membership.eligible_ids)
    assert est.aggregate.observed_count == len(judged)
    assert est.aggregate.imputed_count == len(membership.eligible_ids) - len(judged)
    assert est.aggregate.estimated_pass_rate != pytest.approx(judged_only)


def test_validation_metrics_macro_agent_mae_and_ece_shape():
    membership, agent, vecs, judged = _base_fixture()
    est = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vecs,
        judged_values_by_unit=judged,
    )
    labels = {
        "a_s1": 1.0,
        "a_s2": 0.0,
        "a_u1": 1.0,
        "a_u2": 0.0,
        "b_u1": 1.0,
    }
    val = validate_embedding_population(est, labels, calibration_bin_count=5)

    assert 0.0 <= val.census_pass_rate <= 1.0
    assert val.macro_per_agent_mae >= 0.0
    assert val.judged_only_absolute_rate_error == pytest.approx(
        abs(val.judged_only_pass_rate - val.census_pass_rate)
    )
    assert len(val.calibration_bins) == 5
    total = sum(int(bin_row["count"]) for bin_row in val.calibration_bins)
    assert total == len(est.rows)


def test_no_prior_if_any_global_judgment_and_all_estimates_finite_probabilities():
    membership, agent, vecs, judged = _base_fixture()
    est = estimate_embedding_population(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vecs,
        judged_values_by_unit=judged,
    )
    assert est.aggregate.prior_count == 0
    for row in est.rows:
        assert np.isfinite(row.value)
        assert 0.0 <= row.value <= 1.0


def test_leave_one_out_excludes_self():
    membership, agent, vecs, judged = _base_fixture()
    loo = leave_one_out_donor_diagnostics(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vecs,
        judged_values_by_unit=judged,
    )
    for uid, row in loo.per_unit_predictions.items():
        assert uid not in row.donor_ids
    assert loo.mae >= 0.0
    assert loo.brier_score >= 0.0


def test_leave_one_out_does_not_run_full_population_estimator(monkeypatch):
    membership = freeze_membership(
        cell_id="cell-loo-no-full-pass",
        eligible_ids=("d1", "d2", "u"),
        selected_ids=("d1", "d2"),
    )
    agent = {"d1": "t|a", "d2": "t|a", "u": "t|a"}
    vectors = {"d1": _vec(1.0, 0.0), "d2": _vec(0.0, 1.0), "u": _vec(1.0, 1.0)}

    import sampling_comparison.v4_idw as module

    monkeypatch.setattr(
        module,
        "estimate_embedding_population",
        lambda **kwargs: pytest.fail("LOO must not run a redundant full-population estimate"),
    )

    loo = leave_one_out_donor_diagnostics(
        membership=membership,
        agent_id_by_unit=agent,
        vector_by_unit=vectors,
        judged_values_by_unit={"d1": 0.0, "d2": 1.0},
    )

    assert loo.mae == pytest.approx(1.0)
