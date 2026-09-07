from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess

import pytest

from scripts.build_sampling_v7_distance_report import main


def _aggregate_payload():
    return {
        "summary": {
            "run_count": 6,
            "dataset_count": 3,
            "seed_count": 1,
            "budget_count": 5,
            "repetition_count": 10,
            "base_seed": 13,
        },
        "datasets": [
            {
                "dataset_id": "historical_300",
                "population": 300,
                "pca": {
                    "explained_variance_ratio": [0.35, 0.22, 0.18, 0.12, 0.08, 0.04, 0.01, 0.00],
                    "explained_variance_total": 0.77,
                },
                "shared_preprocessing": {
                    "embedding_model_id": "foundry-text-embedding-3-large",
                    "embedding_deployment_id": "foundry-embeddings-prod",
                },
            },
            {
                "dataset_id": "dense_2500",
                "population": 2500,
                "pca": {
                    "explained_variance_ratio": [0.41, 0.19, 0.14, 0.11, 0.08, 0.05, 0.02, 0.00],
                    "explained_variance_total": 0.85,
                },
                "shared_preprocessing": {
                    "embedding_model_id": "foundry-text-embedding-3-large",
                    "embedding_deployment_id": "foundry-embeddings-prod",
                },
            },
        ],
        "runs": [
            {
                "dataset_id": "historical_300",
                "repetition_index": 1,
                "budget_pct": 10,
                "method_id": "random_sampling",
                "aggregate_pass_rate_mae": 0.5,
            },
            {
                "dataset_id": "historical_300",
                "repetition_index": 1,
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_cosine",
                "estimator_type": "binary_idw",
                "aggregate_pass_rate_mae": 0.12,
                "selected_only_pass_rate_mae": 0.18,
                "replay_aggregate_pass_rate_mae": 0.14,
                "coverage": {"concept": {"coverage_ratio": 0.61}},
                "actual_token_count": 220,
                "latency_seconds": {"per_method_total": 1.4},
                "embedding_metrics": {
                    "imputed_only": {"accuracy": 0.72, "precision": 0.76, "recall": 0.70, "f1": 0.73},
                    "judged_plus_imputed": {"accuracy": 0.77, "precision": 0.8, "recall": 0.75, "f1": 0.78},
                },
                "search_evidence": {"index_name": "trace-clusters-sampling-v7-cosine"},
            },
            {
                "dataset_id": "historical_300",
                "repetition_index": 1,
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_euclidean",
                "estimator_type": "binary_idw",
                "aggregate_pass_rate_mae": 0.15,
                "selected_only_pass_rate_mae": 0.2,
                "replay_aggregate_pass_rate_mae": 0.17,
                "coverage": {"concept": {"coverage_ratio": 0.58}},
                "actual_token_count": 230,
                "latency_seconds": {"per_method_total": 1.6},
                "embedding_metrics": {
                    "imputed_only": {"accuracy": 0.69, "precision": 0.72, "recall": 0.67, "f1": 0.69},
                    "judged_plus_imputed": {"accuracy": 0.74, "precision": 0.77, "recall": 0.72, "f1": 0.75},
                },
                "search_evidence": {"index_name": "trace-clusters-sampling-v7-euclidean"},
            },
            {
                "dataset_id": "dense_2500",
                "repetition_index": 2,
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_cosine",
                "estimator_type": "binary_idw",
                "aggregate_pass_rate_mae": 0.09,
                "selected_only_pass_rate_mae": 0.11,
                "replay_aggregate_pass_rate_mae": 0.11,
                "coverage": {"concept": {"coverage_ratio": 0.63}},
                "actual_token_count": 750,
                "latency_seconds": {"per_method_total": 2.2},
                "embedding_metrics": {
                    "imputed_only": {"accuracy": 0.81, "precision": 0.83, "recall": 0.79, "f1": 0.81},
                    "judged_plus_imputed": {"accuracy": 0.84, "precision": 0.87, "recall": 0.82, "f1": 0.84},
                },
                "search_evidence": {"index_name": "trace-clusters-sampling-v7-cosine"},
            },
            {
                "dataset_id": "dense_2500",
                "repetition_index": 2,
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_euclidean",
                "estimator_type": "binary_idw",
                "aggregate_pass_rate_mae": 0.1,
                "selected_only_pass_rate_mae": 0.13,
                "replay_aggregate_pass_rate_mae": 0.12,
                "coverage": {"concept": {"coverage_ratio": 0.6}},
                "actual_token_count": 760,
                "latency_seconds": {"per_method_total": 2.4},
                "embedding_metrics": {
                    "imputed_only": {"accuracy": 0.78, "precision": 0.8, "recall": 0.77, "f1": 0.79},
                    "judged_plus_imputed": {"accuracy": 0.81, "precision": 0.84, "recall": 0.8, "f1": 0.82},
                },
                "search_evidence": {"index_name": "trace-clusters-sampling-v7-euclidean"},
            },
            {
                "dataset_id": "cosmos_otel",
                "repetition_index": 3,
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_cosine",
                "estimator_type": "binary_idw",
                "aggregate_pass_rate_mae": 0.19,
                "selected_only_pass_rate_mae": 0.22,
                "replay_aggregate_pass_rate_mae": 0.2,
                "coverage": {"concept": {"coverage_ratio": 0.52}},
                "actual_token_count": 440,
                "latency_seconds": {"per_method_total": 1.9},
                "embedding_metrics": {
                    "imputed_only": {"accuracy": 0.67, "precision": 0.7, "recall": 0.64, "f1": 0.67},
                    "judged_plus_imputed": {"accuracy": 0.7, "precision": 0.73, "recall": 0.68, "f1": 0.7},
                },
                "search_evidence": {"index_name": "trace-clusters-sampling-v7-cosine"},
            },
            {
                "dataset_id": "cosmos_otel",
                "repetition_index": 3,
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_euclidean",
                "estimator_type": "binary_idw",
                "aggregate_pass_rate_mae": 0.21,
                "selected_only_pass_rate_mae": 0.24,
                "replay_aggregate_pass_rate_mae": 0.22,
                "coverage": {"concept": {"coverage_ratio": 0.49}},
                "actual_token_count": 450,
                "latency_seconds": {"per_method_total": 2.0},
                "embedding_metrics": {
                    "imputed_only": {"accuracy": 0.65, "precision": 0.68, "recall": 0.61, "f1": 0.64},
                    "judged_plus_imputed": {"accuracy": 0.68, "precision": 0.71, "recall": 0.65, "f1": 0.68},
                },
                "search_evidence": {"index_name": "trace-clusters-sampling-v7-euclidean"},
            },
        ],
        "aggregate_metrics": {
            "rows": [
                {
                    "dataset_id": "historical_300",
                    "budget_pct": 10,
                    "method_id": "pca8_idw_binary_cosine",
                    "metric_id": "aggregate_pass_rate_mae",
                    "left_method_id": "pca8_idw_binary_cosine",
                    "right_method_id": "pca8_idw_binary_euclidean",
                    "direction": "lower_better",
                    "n": 1,
                    "mean": 0.12,
                    "p05": 0.12,
                    "p95": 0.12,
                    "mean_ci95_lower": 0.11,
                    "mean_ci95_upper": 0.13,
                },
                {
                    "dataset_id": "historical_300",
                    "budget_pct": 10,
                    "method_id": "pca8_idw_binary_euclidean",
                    "metric_id": "aggregate_pass_rate_mae",
                    "direction": "lower_better",
                    "n": 1,
                    "mean": 0.15,
                    "p05": 0.15,
                    "p95": 0.15,
                    "mean_ci95_lower": 0.14,
                    "mean_ci95_upper": 0.16,
                },
                {
                    "dataset_id": "historical_300",
                    "budget_pct": 10,
                    "method_id": "pca8_idw_binary_cosine",
                    "metric_id": "coverage_concept_ratio",
                    "n": 1,
                    "mean": 0.61,
                    "p05": 0.61,
                    "p95": 0.61,
                    "mean_ci95_lower": 0.60,
                    "mean_ci95_upper": 0.62,
                },
                {
                    "dataset_id": "historical_300",
                    "budget_pct": 10,
                    "method_id": "pca8_idw_binary_euclidean",
                    "metric_id": "coverage_concept_ratio",
                    "n": 1,
                    "mean": 0.58,
                    "p05": 0.58,
                    "p95": 0.58,
                    "mean_ci95_lower": 0.57,
                    "mean_ci95_upper": 0.59,
                },
            ]
        },
        "paired_differences": {
            "rows": [
                {
                    "dataset_id": "historical_300",
                    "budget_pct": 10,
                    "metric_id": "aggregate_pass_rate_mae",
                    "left_method_id": "pca8_idw_binary_cosine",
                    "right_method_id": "pca8_idw_binary_euclidean",
                    "wins": 1,
                    "ties": 0,
                    "losses": 0,
                    "n": 1,
                    "mean_delta": -0.03,
                    "win_rate": 1.0,
                },
                {
                    "dataset_id": "historical_300",
                    "budget_pct": 10,
                    "metric_id": "coverage_concept_ratio",
                    "left_method_id": "pca8_idw_binary_cosine",
                    "right_method_id": "pca8_idw_binary_euclidean",
                    "wins": 0,
                    "ties": 1,
                    "losses": 0,
                    "n": 1,
                    "mean_delta": 0.0,
                    "win_rate": 0.0,
                }
            ],
            "cells": [
                {
                    "dataset_id": "historical_300",
                    "budget_pct": 10,
                    "repetition_index": 1,
                    "metric_id": "aggregate_pass_rate_mae",
                    "delta_cosine_minus_euclidean": -0.03,
                },
                {
                    "dataset_id": "historical_300",
                    "budget_pct": 10,
                    "repetition_index": 1,
                    "metric_id": "coverage_concept_ratio",
                    "delta_cosine_minus_euclidean": 0.0,
                }
            ],
        },
    }


def _euclidean_strong_payload():
    return {
        "summary": {"dataset_count": 1, "repetition_count": 10, "budget_count": 1, "run_count": 2, "base_seed": 13},
        "datasets": [{"dataset_id": "cosmos_otel", "population": 205, "pca": {"explained_variance_total": 0.73}}],
        "runs": [
            {
                "dataset_id": "cosmos_otel",
                "repetition_index": 1,
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_cosine",
                "latency_seconds": {"selection": 0.8, "idw": 0.5},
                "embedding_metrics": {"imputed_only": {"f1": 0.72}},
            },
            {
                "dataset_id": "cosmos_otel",
                "repetition_index": 1,
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_euclidean",
                "latency_seconds": {"selection": 0.9, "idw": 0.7},
                "embedding_metrics": {"imputed_only": {"f1": 0.79}},
            },
        ],
        "aggregate_metrics": {"rows": []},
        "paired_differences": {
            "rows": [
                {
                    "dataset_id": "cosmos_otel",
                    "budget_pct": 10,
                    "metric_id": "imputed_only_f1",
                    "left_method_id": "pca8_idw_binary_cosine",
                    "right_method_id": "pca8_idw_binary_euclidean",
                    "wins": 3,
                    "ties": 0,
                    "losses": 7,
                    "n": 10,
                    "mean_delta": -0.077,
                    "win_rate": 0.3,
                }
            ],
            "cells": [
                {"dataset_id": "cosmos_otel", "budget_pct": 10, "repetition_index": idx, "metric_id": "imputed_only_f1", "delta_cosine_minus_euclidean": -0.077 if idx < 7 else 0.01}
                for idx in range(10)
            ],
        },
    }


def test_build_sampling_v7_distance_report(tmp_path, monkeypatch):
    aggregate = _aggregate_payload()
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"files": {"interactive_report": "stale.html"}, "hashes": {"interactive_report": "stale"}}), encoding="utf-8")
    output_path = tmp_path / "pca8-distance-report.html"

    monkeypatch.setattr("sys.argv", ["build_sampling_v7_distance_report.py", "--input", str(aggregate_path), "--output", str(output_path)])
    main()

    html = output_path.read_text(encoding="utf-8")
    assert "Full-session embeddings -> PCA-8 -> Binary IDW: Cosine vs Euclidean" in html


def test_distance_report_uses_favored_arm_rate_and_runtime_exclusion():
    html = main.__globals__["build_distance_report_html"](_euclidean_strong_payload())
    assert "favoredWinRate" in html
    assert "effLosses" in html
    assert "rawIncremental: true" in html
    assert "metricIds: ['latency_per_method_total_seconds']" not in html
    assert "Incremental selection+IDW runtime" in html
    assert "const perCorpus = datasetList.map" in html
    assert "return `${datasetId}: ${merged.verdict}" in html
    assert "No pooled winner is claimed" in html
    assert "Mixed / no clear winner" in html
    assert "Incremental selection+IDW runtime" in html


def test_distance_report_runtime_no_paired_latency_objective_map(tmp_path, monkeypatch):
    aggregate = _aggregate_payload()
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"files": {"pca8_distance_report": "stale.html"}, "hashes": {"pca8_distance_report": "stale"}}), encoding="utf-8")
    output_path = tmp_path / "pca8-distance-report.html"
    monkeypatch.setattr("sys.argv", ["build_sampling_v7_distance_report.py", "--input", str(aggregate_path), "--output", str(output_path)])
    main()
    html = output_path.read_text(encoding="utf-8")
    assert "Raw per_method_total (contains shared Search)" in html
    assert "rawIncremental: true" in html
    assert "objectiveVerdict(latency_per_method_total_seconds" not in html
    assert "metricIds: ['latency_per_method_total_seconds']" not in html
    assert "pca8_idw_binary_cosine" in html
    assert "pca8_idw_binary_euclidean" in html
    assert "random_sampling" not in html
    assert "minhash_lsh" not in html
    assert "aggregate_pass_rate_mae" in html
    assert "selected_only_pass_rate_mae" in html
    assert "replay_aggregate_pass_rate_mae" in html
    assert "coverage_concept_ratio" in html
    assert "imputed_only_accuracy" in html
    assert "imputed_only_precision" in html
    assert "imputed_only_recall" in html
    assert "imputed_only_f1" in html
    assert "actual_token_count" in html
    assert "const PAIRED_SUPPORTED_METRICS" in html
    paired_metrics_block = html.split("const PAIRED_SUPPORTED_METRICS", 1)[1].split(";", 1)[0]
    assert "judged_plus_imputed_accuracy" not in paired_metrics_block
    assert "judged_plus_imputed_precision" not in paired_metrics_block
    assert "judged_plus_imputed_recall" not in paired_metrics_block
    assert "judged_plus_imputed_f1" not in paired_metrics_block
    assert "latency_per_method_total_seconds" in html
    assert "Hero comparison and answer" in html
    assert "Holding embeddings, one transductive PCA-8 fit per corpus, paired replay, budget" in html
    assert "This evidence is objective-specific and does not claim a universal winner." in html
    assert "Decision summary / What we learned" in html
    assert "Experiment design / What stayed constant and what changed" in html
    assert "Corpus conditions / Why results may differ" in html
    assert "max input tokens = 8191" in html
    assert "k=8" in html
    assert "power=2" in html
    assert "eps=1e-6" in html
    assert "cutoff >= 0.5" in html
    assert "No-donor fallback prior is" in html
    assert "within-agent donors first" in html
    assert "Token cost signal first" in html
    assert "cache_hit" in html
    assert "per-session truncation incidence is not recorded" in html
    assert "Paired delta" in html
    assert "paired replay" in html
    assert "Budget trend" in html or "Budget Trend" in html
    assert "Paired delta" in html or "Paired Delta" in html
    assert 'id="pca8DistanceReport"' in html
    assert 'id="pairedDeltaPlot"' in html
    assert 'id="winLossChart"' in html
    assert 'id="classificationMatrix"' in html
    assert 'id="conceptCoverageChart"' in html
    assert 'id="tokensAndLatency"' in html
    assert 'id="pcaScreeChart"' in html
    assert 'id="replayFrequencyContext"' in html
    assert 'id="provenanceGrid"' in html
    assert 'id="datasetProfiles"' in html
    assert 'id="executiveAnswer"' in html
    assert 'id="decisionScorecard"' in html
    assert 'id="storyConclusion"' in html
    assert 'id="budgetCapCaption"' in html
    assert 'id="winLossCounts"' in html
    assert 'id="budgetTrendStory"' in html
    assert 'id="pairedDeltaStory"' in html
    assert 'id="winLossStory"' in html
    assert 'id="classificationStory"' in html
    assert 'id="coverageStory"' in html
    assert 'id="tokensLatencyStory"' in html
    assert 'id="uncertaintyStory"' in html
    assert 'id="replayFrequencyStory"' in html
    assert 'id="pcaStory"' in html
    assert ".paired-count-card" in html
    assert ".classification-card" in html
    assert ".story-block" in html
    assert "What this shows:" in html
    assert "How to read it:" in html
    assert "Current takeaway:" in html
    assert "<table><thead><tr><th>Scope</th><th>Method</th>" not in html
    assert "const yPct = (v) => top + (1 - v) * (height - top - bottom);" in html
    assert "const h = yPct(0) - yPct(value);" in html
    assert "viewBox=\"0 0 760 250\"" in html
    assert "blue bars = per-component explained variance" in html
    assert "amber line/dots = cumulative explained variance" in html
    assert "No paired-difference evidence exists for" in html
    assert "run-level summaries but no paired-difference evidence" in html
    assert "Shared Search excluded; no paired CI winner claim." in html
    assert "the evidence is objective-specific and should be read by corpus" in html
    assert "choose per corpus and budget rather than pooling win counts" in html
    assert "Foundry" in html
    assert "Azure AI Search" in html
    assert "negative favors cosine; positive favors Euclidean" in html
    assert "positive favors cosine; negative favors Euclidean" in html
    assert "filteredAggregate" in html
    assert "rowsForFilter().filter" in html
    assert "renderConclusion" in html
    assert "renderDecisionScorecard" in html
    assert "verdictFromPaired" in html
    assert "combineObjectiveVerdicts" in html
    assert "function scopeVerdictSummary(metricId)" in html
    assert "Per-corpus verdicts:" in html
    assert "No pooled winner is claimed" in html
    assert "objectiveVerdict(metricId, activeDataset() === 'all' ? 'all' : activeDataset())" not in html
    assert ".plot svg { width: 640px; min-width: 640px; max-width: none; }" in html
    assert "Mixed / no clear winner" in html
    assert "Cosine leads" in html
    assert "Euclidean leads" in html
    assert "Exact tie" in html
    assert "renderExecutiveAnswer" in html
    assert "renderDatasetProfiles" in html
    assert "renderBudgetCapCaption" in html
    assert "incrementalLatency" in html
    assert "sharedSearchLatency" in html
    assert "Incremental selection+IDW runtime" in html
    assert "not interpreted as per-replay incremental runtime" in html
    assert "relative diff" in html
    assert "paired wins/ties/losses" in html
    assert "wins/ties/losses" in html
    assert "0-100%" in html
    assert "Historical-style low retention caution" in html
    assert "Absolute budget caps by corpus" in html
    assert "Cells with cap < 10 are flagged as low-N" in html
    assert "sample_size" in html
    assert "Shared Search reference" in html
    assert "No pooled winner is claimed; the evidence is objective-specific" in html
    assert "Judged+imputed cards are secondary and observed-inflated" in html
    assert "choose cosine when judged-token efficiency is the priority" in html
    assert "prefer or consider Euclidean when imputed classification is the priority" in html
    assert "choose per corpus and budget" in html
    assert "overflow-x: hidden" in html
    assert "min-width: 0" in html
    assert "@media print" in html
    assert '<html lang="en">' in html
    assert 'lang=\\"en\\"' not in html
    assert "<script src=" not in html
    assert "<link rel=" not in html

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"]["pca8_distance_report"].endswith("pca8-distance-report.html")
    assert manifest["hashes"]["pca8_distance_report"] == hashlib.sha256(output_path.read_bytes()).hexdigest()

    script_match = re.search(r"<script>([\s\S]*)</script>", html)
    assert script_match, "HTML must contain an inline script block"
    if shutil.which("node"):
        script_path = tmp_path / "distance-report-inline.js"
        script_path.write_text(script_match.group(1), encoding="utf-8")
        subprocess.run(["node", "--check", str(script_path)], check=True)


def test_distance_report_separates_coverage_and_token_panels_in_source():
    html = main.__globals__["build_distance_report_html"](_aggregate_payload())
    assert 'const coveragePanels = selectedDatasets().map' in html
    assert 'const tokenPanels = selectedDatasets().map' in html
    assert 'conceptChart.innerHTML = coveragePanels.join(\'\')' in html
    assert 'tokenContainer.innerHTML = tokenPanels.join(\'\')' in html
    assert 'class="panel token-cost-panel' in html
    assert 'class="panel coverage-panel' in html
    assert 'tokenContainer.innerHTML = datasetPanels.join(\'\')' not in html
    assert 'conceptChart.innerHTML = datasetPanels.join(\'\')' not in html


def test_distance_report_exact_tie_and_single_metric_verdicts_are_preserved():
    html = main.__globals__["build_distance_report_html"](_aggregate_payload())
    assert 'if (nonEmpty.length === 1) {' in html
    assert 'Exact tie' in html
    assert 'VERDICT_WORDS.tie' in html


def test_distance_report_runtime_language_and_weak_pca_caveats_are_compact():
    html = main.__globals__["build_distance_report_html"](_aggregate_payload())
    assert 'Lower observed raw mean: Euclidean' in html
    assert 'Lower observed raw mean: Cosine' in html or 'Lower observed raw mean: Euclidean' in html
    assert 'not a paired winner claim' in html
    assert 'PCA-8 retains' in html
    assert 'interpret arm verdicts cautiously' in html
    assert 'paired n=0 / no paired CI' in html


def test_distance_report_browser_regression_playwright(tmp_path):
    sync_api = pytest.importorskip('playwright.sync_api', reason='Python Playwright is not installed')
    sync_playwright = sync_api.sync_playwright

    payload = _aggregate_payload()
    report_path = tmp_path / 'pca8-distance-report.html'
    report_path.write_text(main.__globals__['build_distance_report_html'](payload), encoding='utf-8')

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel='msedge', headless=True)
        except Exception as exc:
            pytest.skip(f'Playwright could not launch Edge: {exc}')
        try:
            page = browser.new_page(viewport={'width': 390, 'height': 1200})
            page.goto(report_path.as_uri(), wait_until='networkidle')
            page.select_option('#datasetFilter', 'all')
            page.select_option('#metricFilter', 'aggregate_pass_rate_mae')
            assert 'No pooled winner is claimed' in page.locator('#executiveAnswer').inner_text()
            assert 'historical_300' in page.locator('#executiveAnswer').inner_text()
            assert 'dense_2500' in page.locator('#executiveAnswer').inner_text()
            assert page.locator('#storyConclusion .conclusion-card').count() > 0
            assert page.locator('#storyConclusion table').count() == 0
            assert page.locator('#storyConclusion .verdict').first.is_visible()
            assert page.locator('#decisionScorecard details').count() > 0
            assert page.locator('#decisionScorecard details[open]').count() == 0
            token_html = page.locator('#tokensAndLatency').inner_html()
            coverage_html = page.locator('#conceptCoverageChart').inner_html()
            assert token_html != coverage_html
            assert 'token-cost-panel' in token_html
            assert 'coverage-panel' in coverage_html
            assert 'token-cost-panel' not in coverage_html
            assert 'coverage-panel' not in token_html
            assert page.locator('.story-block').count() == 9
            story_texts = page.locator('.story-block').all_inner_texts()
            assert all('What this shows:' in text for text in story_texts)
            assert all('How to read it:' in text for text in story_texts)
            assert all('Current takeaway:' in text for text in story_texts)
            page.select_option('#metricFilter', 'coverage_concept_ratio')
            assert 'Exact tie' in page.locator('#coverageStory').inner_text() or 'Exact tie' in page.locator('#decisionScorecard').inner_text()
            strong_payload = _euclidean_strong_payload()
            strong_report = tmp_path / 'euclidean-strong.html'
            strong_report.write_text(main.__globals__['build_distance_report_html'](strong_payload), encoding='utf-8')
            page.goto(strong_report.as_uri(), wait_until='networkidle')
            page.select_option('#datasetFilter', 'cosmos_otel')
            page.select_option('#metricFilter', 'imputed_only_f1')
            assert 'Euclidean leads' in page.locator('#decisionScorecard').inner_text()
            runtime_text = page.locator('#decisionScorecard').inner_text()
            assert 'Mixed / no clear winner' in runtime_text
            assert 'not a paired winner claim' in runtime_text
            assert 'per_method_total' not in runtime_text.lower()
            assert page.evaluate('document.documentElement.scrollWidth === document.documentElement.clientWidth') is True
            assert page.evaluate("[...document.querySelectorAll('.plot')].some(el => el.scrollWidth > el.clientWidth)") is True
        finally:
            browser.close()
