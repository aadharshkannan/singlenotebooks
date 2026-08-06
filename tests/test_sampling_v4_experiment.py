from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sampling_comparison.v4_experiment import (
    V4_EMBEDDING_METHOD,
    V4_OUTCOME_VERSION,
    augment_v3_outcome_with_idw,
)
from sampling_comparison.v4_idw import IDWConfig


class _Trace:
    def __init__(self, trace_id: int, agent_id: str):
        self.trace_id = trace_id
        self.agent_id = agent_id


class _Data:
    def __init__(self):
        self.unit_ids = ("u1", "u2", "u3", "u4")
        self.trace_by_unit_id = {
            "u1": _Trace(11, "tenant|a"),
            "u2": _Trace(12, "tenant|a"),
            "u3": _Trace(13, "tenant|a"),
            "u4": _Trace(14, "tenant|b"),
        }
        self.labels_by_unit = {
            "u1": True,
            "u2": False,
            "u3": True,
            "u4": False,
        }


class _Runtime:
    def __init__(self):
        self.embedding_vector_by_trace_id = {
            11: np.asarray([1.0, 0.0], dtype=np.float32),
            12: np.asarray([0.0, 1.0], dtype=np.float32),
            13: np.asarray([0.8, 0.2], dtype=np.float32),
            14: np.asarray([1.0, 0.0], dtype=np.float32),
        }


def _base_v3_outcome() -> dict[str, Any]:
    return {
        "version": "sampling-v3-outcome-v1",
        "runtime_version": "sampling-v3",
        "population_count": 4,
        "eligible_token_mass": 100,
        "pairing": {"paired_order_manifest": [{"repetition": 0, "order_hash": "abc", "unit_count": 4}]},
        "runs": [
            {
                "method": "random_sampling_token_priority",
                "budget_tokens": 50,
                "legacy_tier_pct": 10,
                "repetition": 0,
                "order_hash": "order-hash-1",
                "selected_ids": ["u1", "u2"],
                "selected_count": 2,
                "selected_pass_rate": 0.5,
                "census_pass_rate": 0.5,
                "absolute_error": 0.0,
            },
            {
                "method": "adaptive_minhash_32x4_token",
                "budget_tokens": 50,
                "legacy_tier_pct": 10,
                "repetition": 0,
                "order_hash": "order-hash-1",
                "selected_ids": ["u1", "u2"],
                "selected_count": 2,
                "selected_pass_rate": 0.5,
                "census_pass_rate": 0.5,
                "absolute_error": 0.0,
            },
            {
                "method": V4_EMBEDDING_METHOD,
                "budget_tokens": 50,
                "legacy_tier_pct": 10,
                "repetition": 0,
                "order_hash": "order-hash-1",
                "selected_ids": ["u1", "u2"],
                "selected_count": 2,
                "selected_pass_rate": 0.5,
                "census_pass_rate": 0.5,
                "absolute_error": 0.0,
            },
            {
                "method": V4_EMBEDDING_METHOD,
                "budget_tokens": 50,
                "legacy_tier_pct": 20,
                "repetition": 1,
                "order_hash": "order-hash-2",
                "selected_ids": ["u1", "u3"],
                "selected_count": 2,
                "selected_pass_rate": 1.0,
                "census_pass_rate": 0.5,
                "absolute_error": 0.5,
            },
        ],
    }


def _walk(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def test_non_embedding_rows_selected_only_and_no_idw():
    out = augment_v3_outcome_with_idw(_Data(), _Runtime(), _base_v3_outcome(), IDWConfig())
    assert out["version"] == V4_OUTCOME_VERSION

    for row in out["runs"]:
        if row["method"] in {"random_sampling_token_priority", "adaptive_minhash_32x4_token"}:
            assert row["estimation_mode"] == "selected_only"
            assert row["model_assisted"] is None
            assert row["selected_only_pass_rate"] == pytest.approx(row["selected_pass_rate"])
            assert row["selected_only_absolute_error"] == pytest.approx(row["absolute_error"])


def test_only_selected_labels_reach_estimator(monkeypatch):
    seen: dict[str, Any] = {"judged_sets": []}

    def _fake_estimate(*, membership, judged_values_by_unit, **kwargs):
        seen["judged_sets"].append(set(judged_values_by_unit.keys()))
        return SimpleNamespace(
            membership=membership,
            aggregate=SimpleNamespace(
                estimated_pass_rate=0.6,
                population_count=4,
                observed_count=2,
                imputed_count=2,
                provenance_counts={"observed": 2, "idw": 2},
                provenance_population_weighted_rates={"observed": 0.25, "idw": 0.35},
                zero_donor_agent_count=0,
                prior_count=0,
            ),
            rows=(),
            agent_id_by_unit={"u1": "a", "u2": "a", "u3": "a", "u4": "b"},
            judged_unit_ids=frozenset(judged_values_by_unit.keys()),
        )

    def _fake_validate(estimates, expected_labels):
        assert set(expected_labels) == {"u1", "u2", "u3", "u4"}
        return SimpleNamespace(
            census_pass_rate=0.5,
            absolute_aggregate_rate_error=0.1,
            per_unit_mae=0.2,
            brier_score=0.15,
            macro_per_agent_mae=0.25,
            judged_only_pass_rate=0.5,
            judged_only_absolute_rate_error=0.0,
            unjudged_only_mae=0.3,
            unjudged_only_brier=0.2,
            calibration_bins=(),
            expected_calibration_error=0.05,
        )

    def _fake_loo(**kwargs):
        return SimpleNamespace(per_unit_predictions={}, mae=0.12, brier_score=0.08)

    import sampling_comparison.v4_experiment as mod

    monkeypatch.setattr(mod, "estimate_embedding_population", _fake_estimate)
    monkeypatch.setattr(mod, "validate_embedding_population", _fake_validate)
    monkeypatch.setattr(mod, "leave_one_out_donor_diagnostics", _fake_loo)

    out = augment_v3_outcome_with_idw(_Data(), _Runtime(), _base_v3_outcome(), IDWConfig())
    emb_rows = [r for r in out["runs"] if r["method"] == V4_EMBEDDING_METHOD]
    assert emb_rows
    assert len(seen["judged_sets"]) == len(emb_rows)
    for judged_set in seen["judged_sets"]:
        assert judged_set.issubset({"u1", "u2", "u3", "u4"})

    expected_sets = sorted([set(row["selected_ids"]) for row in emb_rows], key=lambda s: sorted(s))
    observed_sets = sorted([set(s) for s in seen["judged_sets"]], key=lambda s: sorted(s))
    assert observed_sets == expected_sets


def test_all_embedding_rows_augmented_and_hashes_deterministic():
    base = _base_v3_outcome()
    out1 = augment_v3_outcome_with_idw(_Data(), _Runtime(), base, IDWConfig(k=1))
    out2 = augment_v3_outcome_with_idw(_Data(), _Runtime(), base, IDWConfig(k=1))

    rows1 = [r for r in out1["runs"] if r["method"] == V4_EMBEDDING_METHOD]
    rows2 = [r for r in out2["runs"] if r["method"] == V4_EMBEDDING_METHOD]
    assert len(rows1) == 2
    assert len(rows2) == 2
    for a, b in zip(rows1, rows2, strict=True):
        assert a["estimation_mode"] == "model_assisted_idw"
        assert a["model_assisted"] is not None
        assert a["model_assisted"]["membership"]["membership_hash"] == b["model_assisted"]["membership"]["membership_hash"]
        assert a["model_assisted"]["membership"]["population_hash"] == b["model_assisted"]["membership"]["population_hash"]
        assert len(a["model_assisted"]["metrics"]["nearest_distance_error_bins"]) == 10
        assert a["model_assisted"]["metrics"]["per_agent"]
        assert sum(row["population_count"] for row in a["model_assisted"]["metrics"]["per_agent"]) == 4


def test_aggregate_math_and_embedding_summaries_present():
    out = augment_v3_outcome_with_idw(_Data(), _Runtime(), _base_v3_outcome(), IDWConfig(k=1))
    agg_rows = [row for row in out["aggregate"] if row["method"] == V4_EMBEDDING_METHOD and row["budget_tokens"] == 50]
    assert len(agg_rows) == 1
    agg = agg_rows[0]
    assert agg["replays"] == 2
    assert agg["selected_only_mae"]["mean"] == pytest.approx((0.0 + 0.5) / 2.0)
    assert "idw_absolute_error" in agg
    assert "idw_delta_vs_selected_only" in agg
    assert "idw_unjudged_only_mae" in agg
    assert "idw_unjudged_only_brier" in agg
    assert "idw_expected_calibration_error" in agg
    assert "model_assisted_counts_sum" in agg
    assert "model_assisted_provenance_counts_sum" in agg
    assert "model_assisted_provenance_population_weighted_rates_sum" in agg


def test_sensitive_fields_absent_recursively():
    out = augment_v3_outcome_with_idw(_Data(), _Runtime(), _base_v3_outcome(), IDWConfig())
    forbidden = {
        "vector_by_unit",
        "labels_by_unit",
        "canonical_json",
        "donor_ids",
        "distances",
        "normalized_weights",
        "per_unit_predictions",
        "rows",
        "packet_text",
    }
    keys = {k for k, _ in _walk(out)}
    for key in forbidden:
        assert key not in keys


def test_malformed_population_or_selection_rejected():
    data = _Data()
    runtime = _Runtime()

    bad_pop = _base_v3_outcome()
    bad_pop["population_count"] = 3
    with pytest.raises(ValueError, match="population_count"):
        augment_v3_outcome_with_idw(data, runtime, bad_pop, IDWConfig())

    bad_sel = _base_v3_outcome()
    bad_sel["runs"][0]["selected_ids"] = ["u1", "u1"]
    with pytest.raises(ValueError, match="duplicates"):
        augment_v3_outcome_with_idw(data, runtime, bad_sel, IDWConfig())

    bad_member = _base_v3_outcome()
    bad_member["runs"][0]["selected_ids"] = ["u1", "uX"]
    with pytest.raises(ValueError, match="subset"):
        augment_v3_outcome_with_idw(data, runtime, bad_member, IDWConfig())
