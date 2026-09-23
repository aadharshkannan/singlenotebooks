"""Report fixtures only; none of these synthetic numbers are study results."""
import copy
import json

import pytest

from sampling_comparison.imdb_report import build_report, public_payload, summarize_pca, summarize_rows


def report_fixture(dimensions=(1536, 8)):
    rows = []
    for seed in (13, 14):
        for schedule in ("uniformly_random", "bursty"):
            for rate in (.05, .2):
                for dimension in dimensions:
                    for family in ("reference", "pca"):
                        mae = {(1536, "reference"): .2, (8, "reference"): .4,
                               (1536, "pca"): .21, (8, "pca"): .25}.get(
                                   (dimension, family), .3 if family == "reference" else .28)
                        metrics = {
                            "n": int(2000 * (1 - rate)), "mae": mae, "accuracy": .8,
                            "precision": .8, "recall": .8, "f1": .8, "auc": .85,
                        }
                        row = {
                            "dimension": dimension, "seed": seed, "schedule": schedule, "rate": rate,
                            **{cohort: dict(metrics) for cohort in (
                                "all_unselected", "eligible_point", "eligible_lower", "novel_source",
                            )},
                            "roc": {method: {"auc": .85, "fpr": [0, .3, 1], "tpr": [0, 1, 1]}
                                    for method in ("point", "lower")},
                        }
                        if family == "pca":
                            row.update(representation="pca", representation_id=f"pca_{dimension}")
                        rows.append(row)
    return {
        "version": "imdb-sampling-v1", "status": "completed",
        "dataset": {"dataset_id": "imdb_50000", "sessions": 2000, "agents": 1},
        "protocol": {"dimensions": list(dimensions), "repetitions": 2, "rates": [.05, .2],
                     "schedules": ["uniformly_random", "bursty"], "planned_cells": len(rows)},
        "rows": rows,
        "pca_study": {
            "baseline_aggregate_sha256": "a" * 64, "baseline_manifest_sha256": "b" * 64,
            "pca_manifest_sha256": "c" * 64, "reused_cells": len(rows) // 2, "added_cells": len(rows) // 2,
            "dimensions": list(dimensions), "baseline_rows_unchanged": True, "replay_pairing_exact": True,
            "embedding_calls": 0, "judge_calls": 0, "fit_scope": "full_unlabeled_source_population",
            "fit_source_count": 2000, "solver": "full", "whiten": False, "fit_seed": 13,
            "fit_once": True, "native_1536_control_note": "Centered control, not original native geometry.",
            "private_information": "DO NOT PUBLISH",
        },
        "pca_preparation": {
            "explained_variance_by_dimension": {str(d): d / 1536 for d in dimensions},
            "fit_seconds": 1.2, "solver": "full", "whiten": False, "fit_seed": 13,
            "private_information": "DO NOT PUBLISH",
        },
    }


def test_pca_summaries_use_real_native_and_same_dimension_prefix_references():
    data = report_fixture()
    reference = [r for r in data["rows"] if r.get("representation") != "pca"]
    pca = [r for r in data["rows"] if r.get("representation") == "pca"]
    result = summarize_pca(pca, reference)
    full = next(r for r in result if r["dimension"] == 1536 and r["rate"] == .05 and r["schedule"] == "all")
    small = next(r for r in result if r["dimension"] == 8 and r["rate"] == .05 and r["schedule"] == "all")
    assert full["paired_mae_delta_native"]["mean"] == pytest.approx(.01)
    assert full["paired_mae_delta_prefix"]["mean"] == pytest.approx(.01)
    assert small["paired_mae_delta_native"]["mean"] == pytest.approx(.05)
    assert small["paired_mae_delta_prefix"]["mean"] == pytest.approx(-.15)
    assert small["paired_mae_delta_prefix"]["replays"] == 2


def test_pca_rows_do_not_replace_reference_or_leak_private_fields(tmp_path):
    data = report_fixture()
    source = tmp_path / "fixture.json"
    source.write_text(json.dumps(data))
    payload = public_payload(data, source)
    reference = [r for r in data["rows"] if r.get("representation") != "pca"]
    assert payload["summaries"] == summarize_rows(reference)
    assert len(payload["pca_summaries"]) == 12
    assert payload["protocol"]["planned_cells"] == 32
    assert "DO NOT PUBLISH" not in json.dumps(payload)
    output = tmp_path / "report"
    build_report(source, output)
    assert "PCA comparison" in (output / "report.html").read_text()
    assert "DO NOT PUBLISH" not in (output / "report.html").read_text()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "relabel", "whitening", "refit", "calls"])
def test_pca_report_fails_closed_on_incomplete_or_changed_protocol(tmp_path, mutation):
    data = copy.deepcopy(report_fixture())
    if mutation == "missing":
        data["rows"].pop()
    elif mutation == "duplicate":
        data["rows"][-1] = dict(data["rows"][1])
    elif mutation == "relabel":
        data["rows"][1]["representation_id"] = "prefix_1536"
    elif mutation == "whitening":
        data["pca_study"]["whiten"] = True
    elif mutation == "refit":
        data["pca_study"]["fit_once"] = False
    else:
        data["pca_study"]["embedding_calls"] = 1
    source = tmp_path / "fixture.json"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        public_payload(data, source)


def test_pca_tab_filters_roc_and_numeric_parity(tmp_path):
    pytest.importorskip("playwright.sync_api")
    from scripts.validate_imdb_report import validate_report

    source = tmp_path / "fixture.json"
    source.write_text(json.dumps(report_fixture((1536, 256, 128, 64, 32, 24, 16, 12, 8))))
    output = tmp_path / "report"
    build_report(source, output)
    result = validate_report(output / "report.html", output / "validation_screenshots")
    assert result["ok"], result["issues"]
    assert sum(check.get("pca_filter_combinations", 0) for check in result["checks"]) == 108
