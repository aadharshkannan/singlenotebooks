from __future__ import annotations

import json
from pathlib import Path

import pytest

from sampling_comparison.v2_report import (
    V2ReportInputs,
    load_v2_artifacts,
    render_v2_html_report,
    validate_v2_artifacts,
    write_v2_html_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def _sample_snapshot(method: str, run: str, sampled: int) -> dict:
    return {
        "runId": run,
        "agentId": "agent-1",
        "date": "2026-07-30",
        "granularity": "day",
        "createdAtUtc": "2026-07-30T12:00:00Z",
        "completedAtUtc": "2026-07-30T12:00:00Z",
        "totalConversationsCount": 14,
        "totalSampledCount": sampled,
        "avgScore": 0.6,
        "results": [
            {
                "conversationId": f"conv-{method}-1",
                "conversationEndTimeUtc": "2026-07-30T11:59:00Z",
                "metrics": [
                    {
                        "name": "task_completion",
                        "displayName": "Task Completion",
                        "model": "dataset-expected-label",
                        "scoreScale": "binary",
                        "score": 1,
                        "passed": True,
                        "threshold": 1,
                        "scoreMin": 0,
                        "scoreMax": 1,
                    }
                ],
            },
            {
                "conversationId": f"conv-{method}-2",
                "conversationEndTimeUtc": "2026-07-30T11:58:00Z",
                "metrics": [
                    {
                        "name": "task_completion",
                        "displayName": "Task Completion",
                        "model": "dataset-expected-label",
                        "scoreScale": "binary",
                        "score": 0,
                        "passed": False,
                        "threshold": 1,
                        "scoreMin": 0,
                        "scoreMax": 1,
                    }
                ],
            },
        ],
    }


def _make_fixture_bundle(tmp_path: Path) -> V2ReportInputs:
    out = tmp_path / "outputs_sampling_v2" / "v2"

    aggregate = {
        "version": "sampling-v2-bundle-v1",
        "population_count": 2800,
        "outcome": {
            "census_baseline": {
                "selected_pass_rate": 0.57,
            },
            "aggregate_means": {
                "random_sampling_stratified": {
                    "5": {"mean_absolute_error": 0.021, "mean_fraction_saved": 0.95, "mean_concept_coverage": 0.41},
                    "10": {"mean_absolute_error": 0.017, "mean_fraction_saved": 0.90, "mean_concept_coverage": 0.53},
                    "20": {"mean_absolute_error": 0.011, "mean_fraction_saved": 0.80, "mean_concept_coverage": 0.67},
                    "30": {"mean_absolute_error": 0.009, "mean_fraction_saved": 0.70, "mean_concept_coverage": 0.74},
                    "50": {"mean_absolute_error": 0.006, "mean_fraction_saved": 0.50, "mean_concept_coverage": 0.85},
                },
                "adaptive_minhash_32x4": {
                    "5": {"mean_absolute_error": 0.024, "mean_fraction_saved": 0.955, "mean_concept_coverage": 0.45},
                    "10": {"mean_absolute_error": 0.019, "mean_fraction_saved": 0.905, "mean_concept_coverage": 0.58},
                    "20": {"mean_absolute_error": 0.012, "mean_fraction_saved": 0.805, "mean_concept_coverage": 0.71},
                    "30": {"mean_absolute_error": 0.010, "mean_fraction_saved": 0.705, "mean_concept_coverage": 0.79},
                    "50": {"mean_absolute_error": 0.007, "mean_fraction_saved": 0.505, "mean_concept_coverage": 0.88},
                },
                "adaptive_embedding_fullsession": {
                    "5": {"mean_absolute_error": 0.028, "mean_fraction_saved": 0.96, "mean_concept_coverage": 0.49},
                    "10": {"mean_absolute_error": 0.022, "mean_fraction_saved": 0.91, "mean_concept_coverage": 0.61},
                    "20": {"mean_absolute_error": 0.014, "mean_fraction_saved": 0.81, "mean_concept_coverage": 0.76},
                    "30": {"mean_absolute_error": 0.011, "mean_fraction_saved": 0.71, "mean_concept_coverage": 0.83},
                    "50": {"mean_absolute_error": 0.008, "mean_fraction_saved": 0.51, "mean_concept_coverage": 0.91},
                },
            },
            "aggregate_budget_diagnostics": {},
            "aggregate_per_agent": {},
        },
    }

    # build agent fixtures from census and 20% methods
    for idx in range(1, 106):
        aid = f"tenant-a|agent-{idx:03d}"
        pop = 3 + (idx % 52)
        aggregate["outcome"]["aggregate_per_agent"][f"census|b100|{aid}"] = {
            "method": "census",
            "budget_pct": 100,
            "agent_id": aid,
            "mean_sampled_count": float(pop),
            "mean_absolute_error": 0.0,
        }
        aggregate["outcome"]["aggregate_per_agent"][f"random_sampling_stratified|b20|{aid}"] = {
            "method": "random_sampling_stratified",
            "budget_pct": 20,
            "agent_id": aid,
            "mean_sampled_count": float(max(1, int(round(pop * 0.20)))),
            "mean_absolute_error": 0.010 + (idx % 7) * 0.001,
        }
        aggregate["outcome"]["aggregate_per_agent"][f"adaptive_minhash_32x4|b20|{aid}"] = {
            "method": "adaptive_minhash_32x4",
            "budget_pct": 20,
            "agent_id": aid,
            "mean_sampled_count": float(max(1, int(round(pop * 0.19)))),
            "mean_absolute_error": 0.012 + (idx % 6) * 0.001,
        }
        aggregate["outcome"]["aggregate_per_agent"][f"adaptive_embedding_fullsession|b20|{aid}"] = {
            "method": "adaptive_embedding_fullsession",
            "budget_pct": 20,
            "agent_id": aid,
            "mean_sampled_count": float(max(1, int(round(pop * 0.21)))),
            "mean_absolute_error": 0.013 + (idx % 5) * 0.001,
        }

    # budget diagnostics include per-source keeps
    for method, bump in (
        ("random_sampling_stratified", 0.0),
        ("adaptive_minhash_32x4", -0.01),
        ("adaptive_embedding_fullsession", 0.01),
    ):
        for budget in (5, 10, 20, 30, 50):
            nominal = budget / 100.0
            realized = max(0.0, min(1.0, nominal + bump))
            aggregate["outcome"]["aggregate_budget_diagnostics"][f"{method}|b{budget}"] = {
                "method": method,
                "budget_pct": budget,
                "nominal_keep_rate": nominal,
                "realized_keep_rate_mean": realized,
                "deviation_from_nominal_pp": (realized - nominal) * 100.0,
                "per_corpus": {
                    "historical_300": {"mean_keep_rate": max(0.0, min(1.0, realized + 0.02))},
                    "dense_2500": {"mean_keep_rate": max(0.0, min(1.0, realized - 0.01))},
                },
            }

    corpus_audit = {
        "version": "sampling-v2-corpus-audit-v1",
        "source_files": {
            "historical_300": {
                "path": "h.json",
                "sha256": "a" * 64,
                "counts": {"units": 300, "labels": 300, "agents": 100, "concepts": 120},
                "label_pass_rate": 0.61,
            },
            "dense_2500": {
                "path": "d.json",
                "sha256": "b" * 64,
                "counts": {"units": 2500, "labels": 2500, "agents": 105, "concepts": 400},
                "label_pass_rate": 0.56,
            },
        },
        "combined": {
            "counts": {"units": 2800, "labels": 2800, "agents": 105, "concepts": 520},
            "label_pass_rate": 0.57,
        },
    }

    quadrant = {
        "version": "sampling-v2-actual-quadrant-v1",
        "config": {"budgets": [15, 30], "replay_count": 2},
        "quadrants": {
            "counts": {"total_units": 2800},
            "axis_summary_by_corpus": {
                "variety": {
                    "historical_300": {"low": 150, "high": 150},
                    "dense_2500": {"low": 1250, "high": 1250},
                },
                "velocity": {
                    "historical_300": {"low": 150, "high": 150},
                    "dense_2500": {"low": 1250, "high": 1250},
                },
            },
            "quadrant_summary": {
                "high_variety_high_velocity": {"unit_count": 700, "agent_count": 60, "corpus_counts": {"historical_300": 75, "dense_2500": 625}},
                "high_variety_low_velocity": {"unit_count": 700, "agent_count": 60, "corpus_counts": {"historical_300": 75, "dense_2500": 625}},
                "low_variety_high_velocity": {"unit_count": 700, "agent_count": 55, "corpus_counts": {"historical_300": 75, "dense_2500": 625}},
                "low_variety_low_velocity": {"unit_count": 700, "agent_count": 55, "corpus_counts": {"historical_300": 75, "dense_2500": 625}},
            },
        },
        "aggregate_groups": {},
    }
    for method in ("random_online_admission", "adaptive_minhash_32x4", "adaptive_embedding_fullsession"):
        for q in (
            "high_variety_high_velocity",
            "high_variety_low_velocity",
            "low_variety_high_velocity",
            "low_variety_low_velocity",
        ):
            for b in (15, 30):
                quadrant["aggregate_groups"][f"{method}|{q}|b{b}"] = {
                    "method": method,
                    "quadrant": q,
                    "budget_pct": b,
                    "representation_mean": 0.55,
                    "budget_utilization_mean": 0.92,
                    "zero_selection_agent_rate_mean": 0.08,
                }

    throughput = {
        "version": "sampling-v2-throughput-v1",
        "config": {
            "budgets": [15, 30],
            "arrival_rates": [0.25, 1.0, 4.0, 16.0],
            "eval_throughputs": [0.25, 1.0, 4.0, 16.0],
            "replay_count": 2,
        },
        "aggregate_grid": {},
    }
    for method in ("random_online_admission", "adaptive_minhash_32x4", "adaptive_embedding_fullsession"):
        for ar in throughput["config"]["arrival_rates"]:
            for ev in throughput["config"]["eval_throughputs"]:
                for b in throughput["config"]["budgets"]:
                    throughput["aggregate_grid"][f"{method}|a{ar}|e{ev}|b{b}"] = {
                        "method": method,
                        "arrival_rate_per_second": ar,
                        "eval_throughput_per_second": ev,
                        "budget_pct": b,
                        "representation_mean": 0.50,
                        "budget_utilization_mean": 0.90,
                        "zero_selection_agent_rate_mean": 0.11,
                        "decision_latency_p95_mean": 2.4,
                    }

    membership = {
        "version": "sampling-v2-representative-comparison-membership-v2",
        "budget_pct": 20,
        "methods": {
            "census": {"declared_budget": "100%", "selected_count": 2800, "selected_ids": ["id"]},
            "random_sampling_stratified": {"declared_budget": "20% cap", "selected_count": 560, "selected_ids": ["id"]},
            "adaptive_minhash_32x4": {"declared_budget": "20% cap", "selected_count": 540, "selected_ids": ["id"]},
            "adaptive_embedding_fullsession": {"declared_budget": "20% cap", "selected_count": 575, "selected_ids": ["id"]},
        },
    }

    storage = {
        "version": "sampling-v2-production-storage-manifest-v1",
        "scope": {"implemented": False},
        "authoritative_state": {"source": "ESP/Cosmos"},
        "proposed_logical_model": {
            "containers": [
                {"name": "evaluationRuns", "partitionKey": "/tenantId/agentId/date", "query_paths": ["tenantId", "agentId"]},
                {"name": "selectionMembership", "partitionKey": "/tenantId/agentId/runId", "query_paths": ["runId", "method"]},
                {"name": "evaluationFacts", "partitionKey": "/tenantId/agentId/date", "query_paths": ["conversationId"]},
                {"name": "similarityState", "partitionKey": "/tenantId/agentId/profileId", "query_paths": ["profileId", "expiresAt"]},
            ]
        },
        "ppapi_contract_requirements": {
            "route": "POST /evals/service/results?api-version=1",
            "tenant_handling": "tenant derived from route/auth, not request body",
        },
        "azure_ai_search_assessment": {
            "inspection": {
                "service": "stangoodwin-ai-search",
                "index": "maven-session-sampling-v1",
                "vector_field_present": False,
            }
        },
    }

    snapshots_dir = out / "external_eval_snapshots"
    methods_rows = {
        "census": [_sample_snapshot("census", "v2-census-2026-07-30-aaaa", 14)],
        "random_sampling_stratified": [_sample_snapshot("random", "v2-random-2026-07-30-bbbb", 5)],
        "adaptive_minhash_32x4": [_sample_snapshot("minhash", "v2-minhash-2026-07-30-cccc", 5)],
        "adaptive_embedding_fullsession": [_sample_snapshot("embedding", "v2-embedding-2026-07-30-dddd", 5)],
    }
    methods_files = {}
    for method, rows in methods_rows.items():
        path = snapshots_dir / f"{method}.jsonl"
        _write_jsonl(path, rows)
        methods_files[method] = {
            "path": str(path),
            "sha256": "f" * 64,
            "line_count": len(rows),
        }

    external_manifest = {
        "version": "sampling-v2-external-snapshots-manifest-v1",
        "grouping": "method x tenant x agent x utc_day",
        "route_template": "POST /evals/service/results?api-version=1",
        "not_posted": True,
        "methods_files": methods_files,
    }

    _write_json(out / "aggregate.json", aggregate)
    _write_json(out / "corpus_audit.json", corpus_audit)
    _write_json(out / "quadrant.json", quadrant)
    _write_json(out / "throughput.json", throughput)
    _write_json(out / "selected_membership_20pct.json", membership)
    _write_json(out / "production_storage_manifest.json", storage)
    _write_json(out / "external_eval_snapshots" / "manifest.json", external_manifest)

    return V2ReportInputs(
        aggregate=out / "aggregate.json",
        corpus_audit=out / "corpus_audit.json",
        quadrant=out / "quadrant.json",
        throughput=out / "throughput.json",
        selected_membership_20pct=out / "selected_membership_20pct.json",
        production_storage_manifest=out / "production_storage_manifest.json",
        external_eval_manifest=out / "external_eval_snapshots" / "manifest.json",
    )


def test_v2_report_renders_required_tabs_and_content(tmp_path: Path) -> None:
    inputs = _make_fixture_bundle(tmp_path)
    artifacts = load_v2_artifacts(inputs)
    html = render_v2_html_report(artifacts)

    # 11 required concerns + provenance extra
    assert html.count('class="tab-button"') >= 11
    assert "Overview" in html
    assert "Input Data" in html
    assert "Metrics" in html
    assert "Outcomes" in html
    assert "Agents" in html
    assert "Quadrants" in html
    assert "Throughput" in html
    assert "Methods" in html
    assert "Production Storage" in html
    assert "Output/API" in html
    assert "Methodology/Limits" in html

    # Accessibility and keyboard controls
    assert 'role="tablist"' in html
    assert 'role="tab"' in html
    assert "aria-controls" in html
    assert "ArrowLeft" in html
    assert "ArrowRight" in html
    assert "Home" in html
    assert "End" in html

    # Print button and self-contained assets
    assert "id=\"printButton\"" in html
    assert "window.print()" in html
    assert "http://" not in html
    assert "https://" not in html
    assert "cdn" not in html.lower()

    # Architecture diagrams
    assert "Census architecture" in html
    assert "Native stratified random architecture" in html
    assert "MinHash LSH 32x4 architecture" in html
    assert "Full-session embedding architecture" in html
    assert "Production architecture flow" in html

    # Formula / explanatory content
    assert "MAE(m,b)" in html
    assert "corpus|domain|task|difficulty" in html
    assert "Adaptive selections are nonprobability mechanisms" in html

    # Azure/PPAPI caveats and required statements
    assert "stangoodwin-ai-search" in html
    assert "maven-session-sampling-v1" in html
    assert "without a vector field" in html
    assert "local experiment did not use Azure resources" in html
    assert "POST /evals/service/results?api-version=1" in html
    assert "not_posted=true" in html

    # Responsive/mobile chart containment and legends
    assert "chart-scroll" in html
    assert "chart-legend" in html
    assert "@media (max-width: 980px)" in html
    assert "@media (max-width: 680px)" in html

    # Escaped sample payload and data-derived value checks
    assert "&quot;runId&quot;" in html
    assert "&quot;conversationId&quot;" in html
    assert "80.0%" in html  # representative 20% random saved

    # richness threshold
    assert len(html) > 60000


def test_v2_report_write_and_artifact_validation(tmp_path: Path) -> None:
    inputs = _make_fixture_bundle(tmp_path)
    out_html = tmp_path / "outputs_sampling_v2" / "v2" / "agent365-sampling-v2-report.html"

    output = write_v2_html_report(output_path=out_html, inputs=inputs)
    assert output == out_html
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "Agent365 Sampling V2 Self-Contained Report" in html
    assert "No experiment execution occurs during report generation" in html


def test_v2_report_integrity_failure_on_population(tmp_path: Path) -> None:
    inputs = _make_fixture_bundle(tmp_path)
    payload = json.loads(inputs.aggregate.read_text(encoding="utf-8"))
    payload["population_count"] = 2799
    _write_json(inputs.aggregate, payload)

    artifacts = load_v2_artifacts(inputs)
    with pytest.raises(ValueError, match="population_count"):
        validate_v2_artifacts(artifacts)


def test_v2_report_integrity_failure_on_throughput_shape(tmp_path: Path) -> None:
    inputs = _make_fixture_bundle(tmp_path)
    payload = json.loads(inputs.throughput.read_text(encoding="utf-8"))
    payload["config"]["arrival_rates"] = [1.0, 2.0]
    _write_json(inputs.throughput, payload)

    artifacts = load_v2_artifacts(inputs)
    with pytest.raises(ValueError, match="throughput config"):
        validate_v2_artifacts(artifacts)
