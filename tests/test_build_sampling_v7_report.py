from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess

from scripts.build_sampling_v7_report import main


def test_build_sampling_v7_report_renders_required_sections(tmp_path, monkeypatch):
    aggregate = {
        "summary": {
            "run_count": 4,
            "dataset_count": 1,
            "seed_count": 2,
            "budget_count": 1,
            "repetition_count": 2,
            "base_seed": 13,
            "expected_default_shape": 4,
            "configured_shape": 4,
            "is_live": True,
            "generated_at": "2026-09-03T13:47:12Z",
        },
        "datasets": [
            {
                "dataset_id": "historical_300",
                "population": 300,
                "agent_count": 100,
                "concept_count": 300,
                "domain_count": 100,
                "task_count": 300,
                "label_rate": 0.627,
                "representation_source_counts": {"combined_normalized": 300},
                "linkage_path_counts": {"linked_raw_spans": 42, "synthetic_label_document": 13},
                "pca": {"explained_variance_total": 0.72, "explained_variance_ratio": [0.42, 0.18, 0.07, 0.03, 0.02, 0.01, 0.01, 0.01]},
                "shared_preprocessing": {"embedding_model_id": "foundry-text-embedding-3-large", "embedding_deployment_id": "foundry-embeddings-prod"},
                "runtime_ledger": {
                    "embedding_model_id": "foundry-text-embedding-3-large",
                    "embedding_deployment_id": "foundry-embeddings-prod",
                    "cache_hit": True,
                    "cache_rows": 300,
                    "cache_provenance": {
                        "source_hashes": {"historical_300": "abc123"},
                    },
                },
            }
        ],
        "runs": [
            {
                "dataset_id": "historical_300",
                "seed": 13,
                "repetition_index": 1,
                "replay_frequency_summary": {"unique_source_fraction": 0.8, "max_frequency": 2, "duplicate_event_count": 2},
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_cosine",
                "estimator_type": "pca_binary_imputed_population",
                "aggregate_pass_rate_mae": 0.08,
                "selected_only_pass_rate_mae": 0.21,
                "actual_token_count": 100,
                "coverage": {"agent": 1.0, "domain": 1.0, "task": 1.0, "concept": {"selected_distinct": 3, "population_distinct": 5, "coverage_ratio": 0.6}},
                "search_evidence": {"neighbor_hits": 2, "index_name": "trace-clusters-sampling-v7-cosine", "latency_seconds": 0.42, "query_count": 1},
                "latency_seconds": {"per_method_total": 1.1, "selection": 0.5, "idw": 0.18, "search_evidence": 0.42},
                "embedding_metrics": {
                    "imputed_only": {"accuracy": 0.8, "precision": 0.82, "recall": 0.79, "f1": 0.804},
                    "judged_plus_imputed": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9},
                },
            },
            {
                "dataset_id": "historical_300",
                "seed": 14,
                "repetition_index": 2,
                "replay_frequency_summary": {"unique_source_fraction": 0.81, "max_frequency": 2, "duplicate_event_count": 2},
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_cosine",
                "estimator_type": "pca_binary_imputed_population",
                "aggregate_pass_rate_mae": 0.082,
                "selected_only_pass_rate_mae": 0.211,
                "actual_token_count": 98,
                "coverage": {"agent": 1.0, "domain": 1.0, "task": 1.0, "concept": {"selected_distinct": 3, "population_distinct": 5, "coverage_ratio": 0.6}},
                "search_evidence": {"neighbor_hits": 2, "index_name": "trace-clusters-sampling-v7-cosine", "latency_seconds": 0.42, "query_count": 1},
                "latency_seconds": {"per_method_total": 1.12, "selection": 0.52, "idw": 0.18, "search_evidence": 0.42},
                "embedding_metrics": {
                    "imputed_only": {"accuracy": 0.81, "precision": 0.83, "recall": 0.8, "f1": 0.811},
                    "judged_plus_imputed": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9},
                },
            },
            {
                "dataset_id": "historical_300",
                "seed": 13,
                "repetition_index": 1,
                "replay_frequency_summary": {"unique_source_fraction": 0.8, "max_frequency": 2, "duplicate_event_count": 2},
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_euclidean",
                "estimator_type": "pca_binary_imputed_population",
                "aggregate_pass_rate_mae": 0.09,
                "selected_only_pass_rate_mae": 0.2,
                "actual_token_count": 110,
                "coverage": {"agent": 1.0, "domain": 1.0, "task": 1.0, "concept": {"selected_distinct": 2, "population_distinct": 5, "coverage_ratio": 0.4}},
                "search_evidence": {"neighbor_hits": 2, "index_name": "trace-clusters-sampling-v7-euclidean", "latency_seconds": 0.4, "query_count": 1},
                "latency_seconds": {"per_method_total": 1.2, "selection": 0.55, "idw": 0.25, "search_evidence": 0.4},
                "embedding_metrics": {
                    "imputed_only": {"accuracy": 0.79, "precision": 0.8, "recall": 0.78, "f1": 0.789},
                    "judged_plus_imputed": {"accuracy": 0.88, "precision": 0.89, "recall": 0.87, "f1": 0.879},
                },
            },
            {
                "dataset_id": "historical_300",
                "seed": 14,
                "repetition_index": 2,
                "replay_frequency_summary": {"unique_source_fraction": 0.81, "max_frequency": 2, "duplicate_event_count": 2},
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_euclidean",
                "estimator_type": "pca_binary_imputed_population",
                "aggregate_pass_rate_mae": 0.091,
                "selected_only_pass_rate_mae": 0.201,
                "actual_token_count": 109,
                "coverage": {"agent": 1.0, "domain": 1.0, "task": 1.0, "concept": {"selected_distinct": 2, "population_distinct": 5, "coverage_ratio": 0.4}},
                "search_evidence": {"neighbor_hits": 2, "index_name": "trace-clusters-sampling-v7-euclidean", "latency_seconds": 0.4, "query_count": 1},
                "latency_seconds": {"per_method_total": 1.22, "selection": 0.57, "idw": 0.25, "search_evidence": 0.4},
                "embedding_metrics": {
                    "imputed_only": {"accuracy": 0.79, "precision": 0.8, "recall": 0.78, "f1": 0.79},
                    "judged_plus_imputed": {"accuracy": 0.88, "precision": 0.89, "recall": 0.87, "f1": 0.88},
                },
            },
        ],
        "aggregate_metrics": {
            "rows": [
                {
                    "dataset_id": "historical_300",
                    "budget_pct": 10,
                    "method_id": "pca8_idw_binary_cosine",
                    "metric_id": "aggregate_pass_rate_mae",
                    "n": 2,
                    "mean": 0.081,
                    "median": 0.08,
                    "sample_std": 0.001,
                    "standard_error": 0.0007,
                    "min": 0.08,
                    "max": 0.082,
                    "p05": 0.08,
                    "p25": 0.08,
                    "p75": 0.082,
                    "p95": 0.082,
                    "mean_ci95_lower": 0.08,
                    "mean_ci95_upper": 0.082,
                }
            ]
        },
        "paired_differences": {
            "rows": [
                {
                    "dataset_id": "historical_300",
                    "budget_pct": 10,
                    "metric_id": "aggregate_pass_rate_mae",
                    "win_rate": 1.0,
                    "wins": 2,
                    "ties": 0,
                    "losses": 0,
                    "n": 2,
                    "mean_delta": -0.01,
                }
            ],
            "cells": [
                {
                    "dataset_id": "historical_300",
                    "budget_pct": 10,
                    "metric_id": "aggregate_pass_rate_mae",
                    "delta_cosine_minus_euclidean": -0.01,
                    "repetition_index": 1,
                }
                ,
                {
                    "dataset_id": "historical_300",
                    "budget_pct": 10,
                    "metric_id": "aggregate_pass_rate_mae",
                    "delta_cosine_minus_euclidean": -0.009,
                    "repetition_index": 2,
                }
            ],
        },
    }
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": {"interactive_report": str(tmp_path / "interactive_report.html")},
                "hashes": {"interactive_report": "stale"},
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "interactive_report.html"
    monkeypatch.setattr("sys.argv", ["build_sampling_v7_report.py", "--input", str(aggregate_path), "--output", str(output_path)])
    main()

    html = output_path.read_text(encoding="utf-8")
    final_path = tmp_path / "final_report.json"
    print_path = tmp_path / "print_report.html"
    final_payload = json.loads(final_path.read_text(encoding="utf-8"))
    assert "Introduction" in html
    assert "Data &amp; Cloud Provenance" in html
    assert "Methodology" in html
    assert "Classification Quality" in html
    assert "Coverage &amp; Cost" in html
    assert "Paired Cosine vs Euclidean" in html
    assert "Analysis" in html
    assert "Analysis and Conclusion" in html
    assert "imputed_only_f1" in html
    assert "0.804" in html
    assert "Accuracy" in html
    assert "Precision" in html
    assert "Recall" in html
    assert "F1" in html
    assert "imputed_only" in html
    assert "judged_plus_imputed" in html
    assert "Concept coverage" in html
    assert "Foundry embedding model" in html
    assert "Azure AI Search" in html
    assert "trace-clusters-sampling-v7-cosine" in html
    assert "trace-clusters-sampling-v7-euclidean" in html
    assert "Five parallel methods" in html
    assert "Agent-Balanced Weighted Sampling" in html
    assert "agent_balanced_weighted_sampling" in html
    assert "Source method ID: arm5_hajek_weighted" in html or "arm5_hajek_weighted" in html
    assert "Inverse-probability-weighted Hajek ratio" in html or "Hajek ratio" in html
    assert "min-width:0" in html
    assert "overflow-wrap:anywhere" in html
    assert "grid-template-columns:1fr" in html
    assert "max-width:100%" in html
    assert ".controls-wrap{position:sticky" not in html
    assert ".controls-wrap{position:relative;top:auto;z-index:1;" in html
    assert "<h3>Random Sampling</h3><div class=\"method-id\">" in html
    assert "<h3>MinHash LSH</h3><div class=\"method-id\">" in html
    assert "<h3>PCA-8 Cosine + Binary IDW</h3><div class=\"method-id\">" in html
    assert "<h3>PCA-8 Euclidean + Binary IDW</h3><div class=\"method-id\">" in html
    assert "<h3>Agent-Balanced Weighted Sampling</h3><div class=\"method-id\">" in html
    assert "<h3>Random Sampling <span" not in html
    assert "<h3>Agent-Balanced Weighted Sampling <span" not in html
    assert "Hajek ratio" in html
    assert "What this visual shows" in html
    assert "What these visuals show" in html
    assert "What this section represents" in html
    assert "ARM5" not in html
    assert "ARM4" not in html
    assert "arm5_hajek_weighted" in html or "Source method ID: arm5_hajek_weighted" in html
    assert "V6" not in html
    assert "This is a parallel comparison, not a sequential flow or chain of methods" in html
    assert "Budget Trends" in html
    assert 'id="activeWinner"' in html
    assert 'id="activeRanking"' in html
    assert "BEST FOR THIS VIEW" in html
    assert "margin" in html
    assert "Gap from leader" in html
    assert "leads at the selected" in html
    assert "CI95 overlaps the runner-up" in html
    assert "metric-cell ${isBest?'best':''}" in html
    assert "PCA-8 Context" in html
    assert "Replay Uncertainty" in html
    assert "Replay Frequency Profile" in html
    assert "Repetition" in html
    assert "Per-replay signed deltas" in html
    assert '"aggregateMetrics":[{' in html
    assert '"pairedRows":[{' in html
    assert '"pairedCells":[{' in html
    assert '"mean":0.081' in html
    assert '"win_rate":1.0' in html
    assert "0.6" in html
    assert "Incremental replay runtime (seconds)" in html
    assert "Shared search evidence reference (seconds)" in html
    assert "Per-method runtime (seconds)" not in html
    assert "const LABELS={random_sampling:'Random Sampling'" in html
    assert "source method ID: arm5_hajek_weighted" in html
    assert "Agent-Balanced Weighted Sampling (agent_balanced_weighted_sampling; source method ID: arm5_hajek_weighted)" not in html
    assert "Metric reading guide" in html
    assert "Mean Absolute Error (MAE)" in html
    assert "inverse-distance weighting (IDW)" in html
    assert "Base pass rate (source census)" in html
    assert "drives fixed-source MAE denominator" in html
    assert "Selected-rate mean" in html
    assert "Binary IDW imputed-population estimate" in html
    assert "Hajek-weighted ratio estimate" in html
    assert "datasetSlices=()" in html
    assert "classification denominator uses imputed-only unsampled rows" in html
    assert "coverage denominators are corpus-specific" in html
    assert "pooled directional summary; absolute denominators differ by corpus" in html
    assert "fill=\"#5f7580\"" in html
    assert "Color is neutral and has no method semantics" in html
    assert "repeat(auto-fit,minmax(250px,1fr))" in html
    assert "<div class=\"table-wrap\"><div id=\"activeRanking\" class=\"method-table\"></div></div>" in html
    assert "<div class=\"table-wrap\"><div id=\"classificationMatrix\" class=\"matrix\"></div></div>" in html
    assert "<div class=\"table-wrap\"><div id=\"provenanceTable\"></div></div>" in html
    assert "function renderClassification(){const host=document.getElementById('classificationMatrix');" in html
    assert "classification-grid" in html
    assert "classification-card" in html
    assert "function renderProvenance(){const host=document.getElementById('provenanceTable');" in html
    assert "provenance-grid" in html
    assert "provenance-card" in html
    assert "Agents / Concepts / Domains / Tasks" in html
    assert "_countFromProfile(p,'agent_count'" in html
    assert "_countFromProfile(p,'concept_count'" in html
    assert "_countFromProfile(p,'domain_count'" in html
    assert "_countFromProfile(p,'task_count'" in html
    assert "const baseDirect=_num(p.label_rate);" in html
    assert "Base pass rate (source census)" in html
    assert "Base pass rate (source census)'" in html or "Base pass rate (source census)" in html
    assert "pct(base)" in html
    assert "function renderCoverage(){const coverageHost=document.getElementById('coverageCharts');" in html
    assert "coverage denominators are corpus-specific" in html
    assert "cost panels remain unit-specific" in html
    assert "<article class=\"dataset-block\"><h3>${esc(slice.dataset_id)}</h3>" in html
    assert "@media print" in html
    assert "<script src=" not in html
    assert "<link rel=" not in html

    assert print_path.exists()
    print_html = print_path.read_text(encoding="utf-8")
    assert "Sampling V7 Final Report (Print)" in print_html
    assert "unweighted means over dataset-budget cell means" in print_html
    assert "Agent-Balanced Weighted Sampling" in print_html
    assert "Best fixed-source MAE:" in print_html
    assert "Best concept coverage:" in print_html
    assert "Cosine paired win rates:" in print_html
    assert "100.0%" in print_html
    assert "60.0%" in print_html or "0.6000" in print_html
    assert "{'method_id':" not in print_html
    assert "source_method_id" not in print_html or "Source method ID" in print_html
    assert "report_method_id" not in print_html or "Report alias" in print_html

    methods = final_payload["canonical_methods"]
    agent = next(entry for entry in methods if entry.get("report_method_id") == "agent_balanced_weighted_sampling")
    assert agent["source_method_id"] == "arm5_hajek_weighted"
    assert agent["report_method_id"] == "agent_balanced_weighted_sampling"
    assert agent["method_id"] == "agent_balanced_weighted_sampling"

    assert final_payload["schema_version"] == "sampling-v7-final-report-v1"
    assert final_payload["run_configuration"]["observed_run_rows"] == 4
    assert final_payload["run_configuration"]["expected_shape_value"] == 4
    assert final_payload["cost_accounting"]["guard"]
    assert final_payload["embedding_cache_provenance"]["all_cache_hit"] is True
    shared_rows = final_payload["cost_accounting"]["shared_search_reference_diagnostics"]["rows"]
    assert any(row["is_shared_reference_candidate"] for row in shared_rows)
    assert "legacy_raw_per_method_total_note" in final_payload["cost_accounting"]
    assert "shared search probe" in final_payload["cost_accounting"]["legacy_raw_per_method_total_note"].lower()
    assert "final_report" not in final_payload["provenance"]["artifacts"]["generated"]
    assert "cache hits" in html
    assert "empirical replay sensitivity under perturbations of the same labeled corpus" in html
    assert "duplicated reference probe latency" in html or "do not interpret duplicated reference probe latency" in html

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["hashes"]["interactive_report"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert manifest["hashes"]["final_report"] == hashlib.sha256(final_path.read_bytes()).hexdigest()
    assert manifest["hashes"]["print_report"] == hashlib.sha256(print_path.read_bytes()).hexdigest()
    assert "final_report" not in final_payload["provenance"]["artifacts"]["generated"]


def test_build_sampling_v7_report_inline_script_node_check(tmp_path, monkeypatch):
    aggregate = {
        "summary": {
            "run_count": 4,
            "dataset_count": 1,
            "seed_count": 2,
            "budget_count": 1,
            "repetition_count": 2,
            "base_seed": 13,
            "expected_default_shape": 4,
            "configured_shape": 4,
            "is_live": True,
            "generated_at": "2026-09-03T13:47:12Z",
        },
        "datasets": [
            {
                "dataset_id": "historical_300",
                "population": 300,
                "agent_count": 100,
                "concept_count": 300,
                "domain_count": 100,
                "task_count": 300,
                "label_rate": 0.627,
                "pca": {"explained_variance_total": 0.72, "explained_variance_ratio": [0.42, 0.18, 0.07, 0.03, 0.02, 0.01, 0.01, 0.01]},
                "shared_preprocessing": {"embedding_model_id": "foundry-text-embedding-3-large", "embedding_deployment_id": "foundry-embeddings-prod"},
                "runtime_ledger": {"cache_hit": True, "cache_rows": 300, "cache_provenance": {"source_hashes": {"historical_300": "abc123"}}},
            }
        ],
        "runs": [
            {
                "dataset_id": "historical_300",
                "seed": 13,
                "repetition_index": 1,
                "replay_frequency_summary": {"unique_source_fraction": 0.8, "max_frequency": 2, "duplicate_event_count": 2},
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_cosine",
                "estimator_type": "pca_binary_imputed_population",
                "aggregate_pass_rate_mae": 0.08,
                "selected_only_pass_rate_mae": 0.21,
                "actual_token_count": 100,
                "coverage": {"agent": 1.0, "domain": 1.0, "task": 1.0, "concept": {"selected_distinct": 3, "population_distinct": 5, "coverage_ratio": 0.6}},
                "search_evidence": {"neighbor_hits": 2, "index_name": "trace-clusters-sampling-v7-cosine", "latency_seconds": 0.42, "query_count": 1},
                "latency_seconds": {"per_method_total": 1.1, "selection": 0.5, "idw": 0.18, "search_evidence": 0.42},
                "embedding_metrics": {"imputed_only": {"accuracy": 0.8, "precision": 0.82, "recall": 0.79, "f1": 0.804}, "judged_plus_imputed": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9}},
            },
            {
                "dataset_id": "historical_300",
                "seed": 14,
                "repetition_index": 2,
                "replay_frequency_summary": {"unique_source_fraction": 0.81, "max_frequency": 2, "duplicate_event_count": 2},
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_cosine",
                "estimator_type": "pca_binary_imputed_population",
                "aggregate_pass_rate_mae": 0.082,
                "selected_only_pass_rate_mae": 0.211,
                "actual_token_count": 98,
                "coverage": {"agent": 1.0, "domain": 1.0, "task": 1.0, "concept": {"selected_distinct": 3, "population_distinct": 5, "coverage_ratio": 0.6}},
                "search_evidence": {"neighbor_hits": 2, "index_name": "trace-clusters-sampling-v7-cosine", "latency_seconds": 0.42, "query_count": 1},
                "latency_seconds": {"per_method_total": 1.12, "selection": 0.52, "idw": 0.18, "search_evidence": 0.42},
                "embedding_metrics": {"imputed_only": {"accuracy": 0.81, "precision": 0.83, "recall": 0.8, "f1": 0.811}, "judged_plus_imputed": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9}},
            },
            {
                "dataset_id": "historical_300",
                "seed": 13,
                "repetition_index": 1,
                "replay_frequency_summary": {"unique_source_fraction": 0.8, "max_frequency": 2, "duplicate_event_count": 2},
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_euclidean",
                "estimator_type": "pca_binary_imputed_population",
                "aggregate_pass_rate_mae": 0.09,
                "selected_only_pass_rate_mae": 0.2,
                "actual_token_count": 110,
                "coverage": {"agent": 1.0, "domain": 1.0, "task": 1.0, "concept": {"selected_distinct": 2, "population_distinct": 5, "coverage_ratio": 0.4}},
                "search_evidence": {"neighbor_hits": 2, "index_name": "trace-clusters-sampling-v7-euclidean", "latency_seconds": 0.4, "query_count": 1},
                "latency_seconds": {"per_method_total": 1.2, "selection": 0.55, "idw": 0.25, "search_evidence": 0.4},
                "embedding_metrics": {"imputed_only": {"accuracy": 0.79, "precision": 0.8, "recall": 0.78, "f1": 0.789}, "judged_plus_imputed": {"accuracy": 0.88, "precision": 0.89, "recall": 0.87, "f1": 0.879}},
            },
            {
                "dataset_id": "historical_300",
                "seed": 14,
                "repetition_index": 2,
                "replay_frequency_summary": {"unique_source_fraction": 0.81, "max_frequency": 2, "duplicate_event_count": 2},
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_euclidean",
                "estimator_type": "pca_binary_imputed_population",
                "aggregate_pass_rate_mae": 0.091,
                "selected_only_pass_rate_mae": 0.201,
                "actual_token_count": 109,
                "coverage": {"agent": 1.0, "domain": 1.0, "task": 1.0, "concept": {"selected_distinct": 2, "population_distinct": 5, "coverage_ratio": 0.4}},
                "search_evidence": {"neighbor_hits": 2, "index_name": "trace-clusters-sampling-v7-euclidean", "latency_seconds": 0.4, "query_count": 1},
                "latency_seconds": {"per_method_total": 1.22, "selection": 0.57, "idw": 0.25, "search_evidence": 0.4},
                "embedding_metrics": {"imputed_only": {"accuracy": 0.79, "precision": 0.8, "recall": 0.78, "f1": 0.79}, "judged_plus_imputed": {"accuracy": 0.88, "precision": 0.89, "recall": 0.87, "f1": 0.88}},
            },
        ],
        "aggregate_metrics": {"rows": [{"dataset_id": "historical_300", "budget_pct": 10, "method_id": "pca8_idw_binary_cosine", "metric_id": "aggregate_pass_rate_mae", "n": 2, "mean": 0.081, "median": 0.08, "sample_std": 0.001, "standard_error": 0.0007, "min": 0.08, "max": 0.082, "p05": 0.08, "p25": 0.08, "p75": 0.082, "p95": 0.082, "mean_ci95_lower": 0.08, "mean_ci95_upper": 0.082}]},
        "paired_differences": {"rows": [{"dataset_id": "historical_300", "budget_pct": 10, "metric_id": "aggregate_pass_rate_mae", "win_rate": 1.0, "wins": 2, "ties": 0, "losses": 0, "n": 2, "mean_delta": -0.01}], "cells": [{"dataset_id": "historical_300", "budget_pct": 10, "metric_id": "aggregate_pass_rate_mae", "delta_cosine_minus_euclidean": -0.01, "repetition_index": 1}, {"dataset_id": "historical_300", "budget_pct": 10, "metric_id": "aggregate_pass_rate_mae", "delta_cosine_minus_euclidean": -0.009, "repetition_index": 2}]},
    }
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    output_path = tmp_path / "interactive_report.html"
    monkeypatch.setattr("sys.argv", ["build_sampling_v7_report.py", "--input", str(aggregate_path), "--output", str(output_path)])
    main()

    html = output_path.read_text(encoding="utf-8")
    script_match = re.search(r"<script>([\s\S]*)</script>", html)
    assert script_match, "HTML must contain an inline script block"
    if shutil.which("node"):
        script_path = tmp_path / "v7-inline.js"
        script_path.write_text(script_match.group(1), encoding="utf-8")
        subprocess.run(["node", "--check", str(script_path)], check=True)

    assert "arm5_hajek_weighted" in html or "Source method ID: arm5_hajek_weighted" in html
    assert "pca8_idw_binary_cosine" in html
    assert "pca8_idw_binary_euclidean" in html
    assert "CHART_LABELS" in script_match.group(1) or "short" in script_match.group(1).lower()


def test_build_sampling_v7_report_validates_summary_shape(tmp_path, monkeypatch):
    aggregate = {
        "summary": {"expected_default_shape": 2, "configured_shape": 2},
        "datasets": [{"dataset_id": "d1"}],
        "runs": [
            {"dataset_id": "d1", "method_id": "m1", "budget_pct": 1, "repetition_index": 1},
        ],
    }
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["build_sampling_v7_report.py", "--input", str(aggregate_path)])

    try:
        main()
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "expected_default_shape" in str(exc)


def test_build_sampling_v7_report_rejects_duplicate_or_missing_cartesian_keys(tmp_path, monkeypatch):
    aggregate = {
        "summary": {"expected_default_shape": 4, "configured_shape": 4},
        "datasets": [{"dataset_id": "d1"}],
        "runs": [
            {"dataset_id": "d1", "method_id": "m1", "budget_pct": 10, "repetition_index": 1},
            {"dataset_id": "d1", "method_id": "m1", "budget_pct": 10, "repetition_index": 1},
            {"dataset_id": "d1", "method_id": "m2", "budget_pct": 10, "repetition_index": 1},
            {"dataset_id": "d1", "method_id": "m2", "budget_pct": 10, "repetition_index": 2},
        ],
    }
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["build_sampling_v7_report.py", "--input", str(aggregate_path)])

    try:
        main()
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "duplicate" in str(exc).lower() or "missing" in str(exc).lower()
