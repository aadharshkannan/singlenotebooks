from __future__ import annotations

import copy

import pytest

from sampling_comparison.idw_threshold_report import build_report


def metric(value):
    return {
        "n": 10, "positive_count": 5, "predicted_positive_count": 4,
        "accuracy": value, "precision": value, "recall": value, "f1": value,
        "tp": 4, "fp": 0, "fn": 1, "tn": 5, "reason": None,
    }


def fixture():
    selectors = []
    rows = []
    cell_id = 0
    for dataset in ("historical_300", "dense_2500", "cosmos_otel"):
        for dimension in (1536, 8):
            for rate in (0.01, 0.02, 0.05, 0.1, 0.2):
                grid = [{"threshold": i / 100, **metric(0.7)} for i in range(101)]
                grid.append({"threshold": 1.0000000000000002, **metric(0.5)})
                selectors.append({
                    "selector_id": f"{dataset}|{dimension}|{rate:g}", "dataset_id": dataset,
                    "dimension": dimension, "rate": rate, "cell_count": 150,
                    "pooled_unjudged_n": 1000, "pooled_eligible_n": 800,
                    "point_auc": 0.75, "point_auc_reason": None,
                    "lower_auc": 0.70, "lower_auc_reason": None,
                    "point_roc_display": {"fpr": [0, 1], "tpr": [0, 1], "exact_point_count": 2, "display_point_count": 2},
                    "lower_roc_display": {"fpr": [0, 1], "tpr": [0, 1], "exact_point_count": 2, "display_point_count": 2},
                    "point_grid": grid, "lower_grid": grid, "all_unjudged_point_at_0_5": metric(0.7),
                })
                for seed in range(13, 18):
                    rows.append({
                        "cell_id": cell_id, "dataset_id": dataset, "dimension": dimension,
                        "seed": seed, "schedule": "evenly_spaced", "rate": rate,
                        "unjudged_count": 10, "eligible_count": 8,
                        "point_auc": 0.75, "lower_auc": 0.70,
                        "paired_point_at_0_5": metric(0.8), "paired_lower_at_0_5": metric(0.6),
                    })
                    cell_id += 1
    # Validation count is contract-level; fixture need not materialize all rows.
    profiles = []
    for dataset, n, agents, positive in (
        ("historical_300", 300, 100, 188), ("dense_2500", 2500, 5, 1805), ("cosmos_otel", 205, 7, 69)
    ):
        profiles.append({
            "dataset_id": dataset, "sessions": n, "agents": agents, "positive_count": positive,
            "pass_rate": positive / n, "label_source": "expected labels",
            "representation_sources": {"linked_raw_spans": 195} if dataset == "cosmos_otel" else {},
            "truncated_sessions": 186 if dataset == "cosmos_otel" else 0, "excluded_sessions": 0,
        })
    agents = []
    for dataset in ("historical_300", "dense_2500", "cosmos_otel"):
        for dimension in (1536, 8):
            for method, recall in (("paired_point", 0.8), ("paired_lower", 0.5)):
                agents.append({
                    "dataset_id": dataset, "dimension": dimension, "agent_id": "agent-a",
                    "method": method, **metric(recall),
                })
    return {
        "version": "idw-threshold-envelope-v1", "status": "complete", "run_id": "fixture",
        "source_revision": "abc123", "rows": rows, "selectors": selectors, "datasets": profiles,
        "equal_cell_summary": [
            {
                "dataset_id": selector["dataset_id"], "dimension": selector["dimension"],
                "rate": selector["rate"], "point_auc_mean": 0.75,
                "point_auc_seed_t95": [0.72, 0.78], "lower_auc_mean": 0.70,
                "lower_auc_seed_t95": [0.66, 0.74],
            }
            for selector in selectors
        ],
        "agent_summary": agents,
        "validation": {
            "actual_cells": 4500, "target_occurrences": 12000, "eligible_target_occurrences": 9600,
            "baseline_accuracy_mismatches": 0, "baseline_max_mae_absolute_error": 0,
            "baseline_max_accuracy_absolute_error": 0,
            "threshold_dominance": "passed_all_cells",
        },
        "preregistration": {"acceptance": {"amended_execution_accuracy_absolute_tolerance": 0.001}},
        "baseline": {
            "aggregate": "baseline/aggregate.json", "aggregate_sha256": "a" * 64,
            "memberships": "baseline/memberships.jsonl.gz", "memberships_sha256": "b" * 64,
        },
        "environment": {"python": "3.11", "platform": "test"},
        "costs": {"wall_seconds": 2.0, "cpu_seconds": 1.0},
        "code_hashes": {"source.py": "c" * 64},
        "files": {
            "target_evidence": "run/target_evidence.npz", "exact_roc_curves": "run/exact_roc_curves.npz",
            "preregistration": "run/preregistration.json",
        },
    }


def test_report_is_standalone_accessible_and_has_five_charts():
    html = build_report(fixture(), "run/aggregate.json")
    assert "<title>Full-session IDW ROC and threshold experiment</title>" in html
    assert html.count("<svg") == 5
    assert html.count('class="chart-scroll"') == 5
    assert "min-width:620px" in html
    assert "Illustrative IDW example—not a run result" in html
    assert "not a confidence interval" in html
    assert "normalized angular geometry" in html
    assert "configured sparse-calibration fallback" in html
    assert "empirical target-time L" in html
    assert "global-mean/prior" in html
    assert "Tau2 remains excluded" in html
    assert "195 linked raw-span" in html
    assert "186 sessions were truncated" in html
    assert "accuracy / precision / recall / F1" in html
    assert "exact_roc_curves.npz" in html
    assert "src=\"http" not in html.lower()
    assert "<script src=" not in html.lower()


def test_report_has_native_controls_and_explicit_denominators():
    html = build_report(fixture(), "run/aggregate.json")
    assert '<select id="dataset">' in html
    assert '<select id="dimension">' in html
    assert '<select id="rate">' in html
    assert "paired eligible /" in html
    assert "repeated occurrences" in html
    assert "not independent sessions" in html
    assert "Precision" in html and "Defined as 0 when none are predicted" in html
    assert "95% seed-sensitivity interval" in html


def test_report_rejects_incomplete_or_wrong_version():
    payload = fixture()
    payload["status"] = "partial"
    with pytest.raises(ValueError, match="incomplete"):
        build_report(payload, "run/aggregate.json")
    payload = fixture()
    payload["version"] = "other"
    with pytest.raises(ValueError, match="requires"):
        build_report(payload, "run/aggregate.json")


def test_report_escapes_paths_and_agent_ids():
    payload = fixture()
    payload["run_id"] = "<script>alert(1)</script>"
    payload["agent_summary"][0]["agent_id"] = "<img src=x>"
    html = build_report(payload, "<bad>")
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x>" not in html
    assert "&lt;bad&gt;" in html


@pytest.mark.parametrize("metric", ["mae", "accuracy"])
def test_canonical_report_does_not_apply_superseded_runtime_tolerances(metric):
    payload = fixture()
    payload["protocol"] = {"canonical_runtime_required": True}
    payload["preregistration"]["acceptance"].update(
        canonical_runtime_mae_absolute_tolerance=0.0,
        canonical_runtime_accuracy_absolute_tolerance=0.0,
        amended_execution_mae_absolute_tolerance=0.0001,
    )
    payload["validation"][f"baseline_max_{metric}_absolute_error"] = 0.00001
    with pytest.raises(ValueError, match="tolerance"):
        build_report(payload, "run/aggregate.json")


def test_fresh_report_describes_retained_precision_without_inventing_failed_runs():
    html = build_report(fixture(), "run/aggregate.json")
    assert "retained as float32" in html
    assert "float32 ties can affect ROC breakpoints" in html
    assert "<code>failed_runs.json</code>" not in html
