import json

import numpy as np
import pytest

from sampling_comparison.imdb_report import build_report, public_payload, summarize_rows
from sampling_comparison.matryoshka_experiment import sha256_file


def test_readiness_never_fabricates_results(tmp_path):
    source = tmp_path / "readiness.json"
    source.write_text(json.dumps({
        "dataset_id": "imdb_50000", "status": "prepared_without_embeddings",
        "sessions": 50_000, "positive_count": 25_000, "negative_count": 25_000,
        "raw_review": "DO NOT PUBLISH", "endpoint": "DO NOT PUBLISH",
    }))
    before = source.read_bytes()
    output = tmp_path / "report"
    build_report(source, output)
    summary = json.loads((output / "summary.json").read_text())
    html = (output / "report.html").read_text()
    assert summary["status"] == "results_pending"
    assert summary["summaries"] == []
    assert summary["protocol"]["planned_cells"] == 2400
    assert "DO NOT PUBLISH" not in html
    assert "No performance conclusion is justified yet" in html
    assert "No embedding calls or benchmark replays were performed" in html
    assert "cdn." not in html
    assert source.read_bytes() == before


def test_completed_claim_requires_complete_grid(tmp_path):
    path = tmp_path / "aggregate.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="measured IMDb"):
        public_payload({"status": "completed", "version": "imdb-sampling-v1", "rows": []}, path)
    with pytest.raises(ValueError, match="missing or repeats"):
        public_payload({
            "status": "completed", "version": "imdb-sampling-v1", "dataset": {"dataset_id": "imdb_50000"},
            "rows": [{"dimension": 1536, "rate": .05, "schedule": "bursty", "seed": 1}],
        }, path)


def test_schedule_averaging_and_paired_seed_deltas():
    rows = []
    for seed in (1, 2):
        for schedule, base in (("uniformly_random", .2), ("bursty", .4)):
            for dimension, delta in ((1536, 0), (8, .1)):
                rows.append({
                    "dimension": dimension, "rate": .05, "schedule": schedule, "seed": seed,
                    "all_unselected": {"mae": base + delta, "n": 90},
                    "roc": {"point": {"auc": .75, "fpr": [0, .5, 1], "tpr": [0, 1, 1]}},
                })
    cell = next(r for r in summarize_rows(rows) if r["schedule"] == "all" and r["dimension"] == 8)
    assert cell["cohorts"]["all_unselected"]["mae"]["mean"] == pytest.approx(.4)
    assert cell["cohorts"]["all_unselected"]["mae"]["replays"] == 2
    assert cell["paired_mae_delta_native"]["mean"] == pytest.approx(.1)
    assert cell["roc"]["point"]["replays"] == 2
    assert cell["roc"]["lower"]["tpr"] == []


def test_completed_report_browser_numeric_and_controls(tmp_path):
    pytest.importorskip("playwright.sync_api")
    from scripts.validate_imdb_report import validate_report

    dimensions = [1536, 256, 128, 64, 32, 24, 16, 12, 8]
    rows = []
    for dimension in dimensions:
        for seed in (13, 14):
            for schedule in ("uniformly_random", "bursty"):
                for rate in (.01, .05):
                    metrics = {
                        "n": 90, "mae": .2 + rate, "accuracy": .8, "precision": .8,
                        "recall": .8, "f1": .8, "auc": .85,
                    }
                    rows.append({
                        "dimension": dimension, "seed": seed, "schedule": schedule, "rate": rate,
                        **{cohort: metrics for cohort in (
                            "all_unselected", "eligible_point", "eligible_lower", "novel_source",
                        )},
                        "roc": {method: {"auc": .85, "fpr": [0, .3, 1], "tpr": [0, 1, 1]}
                                for method in ("point", "lower")},
                    })
    source = tmp_path / "fixture-aggregate.json"
    source.write_text(json.dumps({
        "version": "imdb-sampling-v1", "status": "completed",
        "dataset": {"dataset_id": "imdb_50000", "sessions": 100, "agents": 1},
        "protocol": {
            "dimensions": dimensions, "repetitions": 2, "rates": [.01, .05],
            "schedules": ["uniformly_random", "bursty"],
        }, "rows": rows,
        "extension": {
            "baseline_aggregate_sha256": "a" * 64, "baseline_manifest_sha256": "b" * 64,
            "reused_cells": 48, "added_cells": 24,
            "reused_dimensions": [1536, 32, 24, 16, 12, 8], "added_dimensions": [256, 128, 64],
            "baseline_rows_unchanged": True, "replay_pairing_exact": True, "embedding_calls": 0,
        },
        "execution": {
            "workers": 3, "blas_threads_per_worker": 4, "checkpoint_cells_preserved_at_switch": 2,
            "completed_checkpoints_unchanged": True, "scientific_binding_unchanged": True,
            "transition_sha256": "c" * 64,
        },
    }))
    report = tmp_path / "fixture-report"
    build_report(source, report)
    result = validate_report(report / "report.html", report / "validation_screenshots")
    assert result["ok"], result["issues"]


def test_report_extension_metadata_matches_grid(tmp_path):
    source = tmp_path / "extended.json"
    dimensions = [1536, 256, 128, 64, 32, 24, 16, 12, 8]
    rows = [
        {"dimension": dimension, "seed": 13, "schedule": "bursty", "rate": .05,
         "all_unselected": {"n": 95, "mae": .3}}
        for dimension in dimensions
    ]
    extension = {
        "baseline_aggregate_sha256": "a" * 64, "baseline_manifest_sha256": "b" * 64,
        "reused_cells": 6, "added_cells": 3, "reused_dimensions": [1536, 32, 24, 16, 12, 8],
        "added_dimensions": [256, 128, 64], "baseline_rows_unchanged": True,
        "replay_pairing_exact": True, "embedding_calls": 0, "private_path": "DO NOT PUBLISH",
    }
    data = {
        "version": "imdb-sampling-v1", "status": "completed",
        "dataset": {"dataset_id": "imdb_50000", "sessions": 100},
        "protocol": {"dimensions": dimensions, "repetitions": 1, "rates": [.05], "schedules": ["bursty"]},
        "rows": rows, "extension": extension,
    }
    source.write_text(json.dumps(data))
    payload = public_payload(data, source)
    assert payload["protocol"]["planned_cells"] == 9
    assert payload["extension"]["added_cells"] == 3
    assert "private_path" not in payload["extension"]
    extension["baseline_rows_unchanged"] = False
    with pytest.raises(ValueError, match="extension provenance"):
        public_payload(data, source)


def test_real_engine_to_report_contract_on_offline_fixture(tmp_path):
    from sampling_comparison.imdb_experiment import run_experiment
    from scripts.validate_imdb_experiment import validate_scores

    vectors = np.random.default_rng(13).normal(size=(40, 1536)).astype(np.float32)
    labels = np.arange(40) % 2
    output = tmp_path / "fixture-run"
    aggregate = run_experiment(
        vectors, labels, output,
        profile={"dataset_id": "imdb_50000", "sessions": 40, "model": "text-embedding-3-small",
                 "fixture_only": True},
        dimensions=(1536, 8), repetitions=2, rates=(.1,), schedules=("uniformly_random",),
    )
    row = aggregate["rows"][0]
    with np.load(output / row["evidence"], allow_pickle=False) as arrays:
        evidence = {name: arrays[name] for name in arrays.files}
    assert validate_scores(row, evidence, labels)["max_metric_error"] <= 1e-12
    evidence["label"][0] = 1 - evidence["label"][0]
    with pytest.raises(ValueError, match="original source labels"):
        validate_scores(row, evidence, labels)
    evidence["label"][0] = 1 - evidence["label"][0]
    evidence["max_donor_position"][0] = evidence["position"][0]
    with pytest.raises(ValueError, match="noncausal donor"):
        validate_scores(row, evidence, labels)
    report = tmp_path / "fixture-report"
    build_report(output / "aggregate.json", report)
    summary = json.loads((report / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert len(summary["summaries"]) == 4
    for row in summary["summaries"]:
        assert row["cohorts"]["all_unselected"]["n"]["mean"] == 36
        assert row["cohorts"]["eligible_point"]["n"] == row["cohorts"]["eligible_lower"]["n"]
        assert row["diagnostics"]["idw"]["mean"] is not None
    audit = tmp_path / "fixture-validation.json"
    audit.write_text(json.dumps({
        "ok": True, "cells_checked": 4,
        "aggregate_sha256": sha256_file(output / "aggregate.json"),
        "maximum_absolute_metric_difference": 0,
        "native_5pct_equal_cell_diagnostics": {"lower_zero_fraction": .5, "mean_envelope_width": .8},
    }))
    build_report(output / "aggregate.json", report, numerical_validation=audit)
    audited = json.loads((report / "summary.json").read_text())
    assert audited["score_validation"]["cells_checked"] == 4
    assert audited["score_validation"]["sha256"] == sha256_file(audit)
    audit.write_text(json.dumps({"ok": True, "aggregate_sha256": "different-run", "cells_checked": 4}))
    with pytest.raises(ValueError, match="does not match"):
        build_report(output / "aggregate.json", report, numerical_validation=audit)
