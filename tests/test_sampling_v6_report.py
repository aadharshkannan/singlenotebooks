from __future__ import annotations

import json
from pathlib import Path

import pytest

from sampling_comparison.v6_report import (
    DEFAULT_OUTPUT_NAME,
    V6ReportInputs,
    _normalize_aggregate_rows,
    _paired_seed_comparison_summary,
    compose_pdf_command,
    load_v6_artifacts,
    render_v6_html_report,
    validate_pdf_file,
    write_v6_html_report,
)
from tests.test_sampling_v6_runner import _run_bundle


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def test_normalize_aggregate_rows_enriches_missing_descriptive_stats():
    runs = [
        {"method_id": "arm1_global_random", "cap": 64, "seed": 1, "absolute_aggregate_mae": 0.12, "concept_coverage": 0.4, "use_case_coverage": 0.5, "agent_coverage": 0.6, "selected_count": 10, "actual_token_count": 1000, "nominal_budget": 1500},
        {"method_id": "arm1_global_random", "cap": 64, "seed": 2, "absolute_aggregate_mae": 0.18, "concept_coverage": 0.6, "use_case_coverage": 0.7, "agent_coverage": 0.8, "selected_count": 11, "actual_token_count": 1100, "nominal_budget": 1600},
        {"method_id": "arm1_global_random", "cap": 64, "seed": 3, "absolute_aggregate_mae": 0.22, "concept_coverage": 0.5, "use_case_coverage": 0.6, "agent_coverage": 0.7, "selected_count": 12, "actual_token_count": 1200, "nominal_budget": 1700},
    ]
    aggregate = {"methods": [{"method_id": "arm1_global_random", "cap": 64, "selected_count": 10, "absolute_aggregate_mae": {"mean": 0.17, "min": 0.12, "max": 0.22}, "concept_coverage": {"mean": 0.5, "min": 0.4, "max": 0.6}, "actual_token_count": {"mean": 1100.0, "min": 1000.0, "max": 1200.0}}]}
    rows = _normalize_aggregate_rows(aggregate, runs)
    assert rows[0]["metrics"]["absolute_aggregate_mae"]["median"] == pytest.approx(0.18)
    assert rows[0]["metrics"]["absolute_aggregate_mae"]["p05"] == pytest.approx(0.126)
    assert rows[0]["metrics"]["absolute_aggregate_mae"]["sample_std"] == pytest.approx(0.050332229568471665)


def test_paired_seed_comparisons_use_all_matching_seeds_and_group_arm5_vs_arm4():
    runs = []
    for seed, arm1_mae, arm4_mae, arm5_mae in (
        (13, 0.10, 0.12, 0.08),
        (14, 0.20, 0.15, 0.18),
        (15, 0.30, 0.35, 0.25),
    ):
        base = {
            "cap": 64,
            "seed": seed,
            "concept_coverage": 0.5,
            "use_case_coverage": 0.5,
            "agent_coverage": 0.5,
        }
        runs.append({**base, "method_id": "arm1_global_random", "absolute_aggregate_mae": arm1_mae, "estimate": 0.70})
        runs.append({**base, "method_id": "arm4_agent_round_robin", "absolute_aggregate_mae": arm4_mae, "estimate": 0.65})
        runs.append({**base, "method_id": "arm5_hajek_weighted", "absolute_aggregate_mae": arm5_mae, "estimate": 0.72})

    comparisons = _paired_seed_comparison_summary(runs)
    arm4 = next(row for row in comparisons if row["method_id"] == "arm4_agent_round_robin")
    arm5_vs_arm4 = next(row for row in comparisons if row["method_id"] == "arm5_vs_arm4")

    assert arm4["n_pairs"] == 3
    assert arm4["mae_delta"]["mean"] == pytest.approx((0.02 - 0.05 + 0.05) / 3)
    assert arm4["mae_win_rate"] == pytest.approx(1 / 3)
    assert arm5_vs_arm4["n_pairs"] == 3
    assert arm5_vs_arm4["mae_delta"]["mean"] == pytest.approx((-0.04 + 0.03 - 0.10) / 3)
    assert arm5_vs_arm4["estimate_delta"]["mean"] == pytest.approx(0.07)
    assert arm5_vs_arm4["mae_win_rate"] == pytest.approx(2 / 3)


def test_normalized_runs_preserve_estimator_and_selected_only_fields(tmp_path: Path):
    inputs = _build_fixture_bundle(tmp_path)
    artifacts = load_v6_artifacts(inputs)
    arm2 = next(row for row in artifacts.runs if row["method_id"] == "arm2_embedding_idw")

    assert arm2["estimate"] is not None
    assert arm2["census_pass_rate"] is not None
    assert arm2["selected_rate"] is not None
    assert arm2["selected_only_absolute_error"] is not None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_canonical(row) for row in rows) + "\n", encoding="utf-8")


def test_render_v6_html_report_contains_descriptive_stability_section_and_caveat(tmp_path: Path):
    bundle = _build_fixture_bundle(tmp_path)
    html = render_v6_html_report(bundle)
    assert "30-Trial Descriptive Stability and paired deltas/win rates" in html
    assert "empirical 5th-95th percentile descriptive spread" in html
    assert "no p-values" in html.lower() or "no p-values/general-population inference" in html.lower()
    assert "Medians and percentile bars" in html or "MAE median" in html
    assert "ARM3 Agent Round Robin Floor vs ARM1" in html
    assert "ARM4 Agent Round Robin vs ARM1" in html
    assert "ARM5 vs ARM4" in html
    assert "Estimate Delta" in html
    assert "ARM1 Global Random" in html


def _build_fixture_bundle(tmp_path: Path) -> V6ReportInputs:
    root = tmp_path / "artifact_bundle"
    root.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "version": "sampling-v6-bundle-v1",
        "generated_at": "2026-08-21T00:00:00Z",
        "status": "complete",
        "run_status": "complete",
        "population_audit": {"unit_count": 1200, "agent_count": 350},
        "methods": [],
    }
    caps = [64, 128, 256]
    method_rows = [
        ("arm1_global_random", 0.14, 0.55, 0.52, 0.47, 1300),
        ("arm2_embedding_idw", 0.11, 0.61, 0.58, 0.54, 1330),
        ("arm3_agent_round_robin_floor", 0.105, 0.63, 0.60, 0.58, 1280),
        ("arm4_agent_round_robin", 0.09, 0.69, 0.67, 0.64, 1250),
        ("arm5_hajek_weighted", 0.086, 0.72, 0.70, 0.67, 1210),
    ]
    for method_id, mae, c_cov, uc_cov, a_cov, tok in method_rows:
        for cap in caps:
            aggregate["methods"].append(
                {
                    "method_id": method_id,
                    "cap": cap,
                    "selected_count": cap,
                    "absolute_aggregate_mae": {"mean": mae - (0.005 if cap > 64 else 0.0), "min": mae - 0.01, "max": mae + 0.01},
                    "concept_coverage": {"mean": c_cov + (0.02 if cap > 64 else 0.0), "min": c_cov - 0.02, "max": c_cov + 0.03},
                    "use_case_coverage": {"mean": uc_cov + (0.02 if cap > 64 else 0.0), "min": uc_cov - 0.02, "max": uc_cov + 0.03},
                    "agent_coverage": {"mean": a_cov + (0.02 if cap > 64 else 0.0), "min": a_cov - 0.02, "max": a_cov + 0.03},
                    "actual_token_count": cap * tok,
                    "nominal_budget": cap * 1500,
                    "cap_replays": 3,
                }
            )
    _write_json(root / "aggregate.json", aggregate)

    runs: list[dict] = []
    for seed in (1, 2, 3):
        for method_id, base_mae, c_cov, uc_cov, a_cov, tok in method_rows:
            for cap in caps:
                run = {
                    "method_id": method_id,
                    "cap": cap,
                    "seed": seed,
                    "estimate": 0.70 + (0.01 if method_id == "arm5_hajek_weighted" else 0.0),
                    "census_pass_rate": 0.69,
                    "selected_only_rate": 0.68,
                    "selected_count": cap,
                    "nominal_budget": cap * 1500,
                    "actual_tokens": cap * (tok + (seed * 3)),
                    "mae": base_mae + (seed * 0.001),
                    "concept_coverage": c_cov + (0.02 if cap > 64 else 0.0),
                    "maven_coverage": uc_cov + (0.02 if cap > 64 else 0.0),
                    "agent_coverage": a_cov + (0.02 if cap > 64 else 0.0),
                    "selected_only_absolute_error": (base_mae + 0.01) if method_id == "arm2_embedding_idw" else None,
                    "top_five_agents": [
                        {
                            "agent_id": "A-102",
                            "N": 400,
                            "n": 12 + seed,
                            "selected_rate": 0.81,
                            "census_rate": 0.78,
                            "absolute_error": 0.05,
                            "concept_coverage": 0.76,
                            "use_case_coverage": 0.74,
                        },
                        {
                            "agent_id": "A-<script>alert(1)</script>",
                            "N": 300,
                            "n": 9 + seed,
                            "selected_rate": 0.66,
                            "census_rate": 0.62,
                            "absolute_error": 0.08,
                            "concept_coverage": 0.64,
                            "use_case_coverage": 0.59,
                        },
                        {
                            "agent_id": "A-null-agent",
                            "N": 50,
                            "n": 2,
                            "selected_rate": None,
                            "census_rate": 0.12,
                            "absolute_error": None,
                            "concept_coverage": 0.1,
                            "use_case_coverage": 0.2,
                        },
                    ],
                }
                if method_id == "arm2_embedding_idw":
                    run["idw_provenance"] = {
                        "observed_count": 140 + cap + seed,
                        "imputed_count": 30 + cap // 4 + seed,
                        "provenance_count": 260 + cap,
                        "provenance_counts": {
                            "observed": 140 + cap + seed,
                            "idw": 30 + cap // 4 + seed,
                            "exact_match": 2 + seed,
                            "agent_mean": 10 + seed,
                            "global_mean": 8 + seed,
                            "prior": 6 + seed,
                        },
                    }
                    run["idw_validation"] = {
                        "quality": "high",
                        "mae": 0.06,
                        "mape": 0.08,
                        "coverage": 0.91,
                    }
                    run["idw_quality"] = {
                        "absolute_aggregate_rate_error": 0.018 + (cap / 10000),
                        "per_unit_mae": 0.041 + (cap / 20000),
                        "brier_score": 0.11 + (cap / 50000),
                        "macro_per_agent_mae": 0.052,
                        "unjudged_only_mae": 0.06 + (cap / 30000),
                        "unjudged_only_brier": 0.125,
                        "expected_calibration_error": 0.022 + (cap / 40000),
                    }
                runs.append(run)

    _write_jsonl(root / "runs.jsonl", runs)

    memberships = []
    for cap in caps:
        memberships.append(
            {
                "method_id": "arm3_agent_round_robin_floor",
                "cap": cap,
                "selected_agent_count": 60 + cap // 2,
                "eligible_agents_with_at_least_3": 80 + cap // 3,
                "agents_with_at_least_3": 55 + cap // 4,
                "represented_strata": 22 + cap // 8,
                "agent_coverage": 0.50 + (cap / 1000),
                "total_floor_target": 120 + cap,
                "floor_prefix_count": cap,
                "floor_complete": False,
                "arm3_floor": {
                    "total_floor_target": 120 + cap,
                    "floor_prefix_count": cap,
                    "floor_complete": False,
                },
                "arm3_floor_min_per_agent": 3,
            }
        )
        memberships.append(
            {
                "method_id": "arm4_agent_round_robin",
                "cap": cap,
                "selected_agent_count": 88 + cap // 2,
                "eligible_agents_with_at_least_3": 84 + cap // 3,
                "agents_with_at_least_3": 60 + cap // 5,
                "represented_strata": 26 + cap // 7,
                "agent_coverage": 0.54 + (cap / 900),
            }
        )
    _write_jsonl(root / "memberships.jsonl", memberships)

    classifications = [
        {
            "use_case_guid": "UC-01",
            "domain": "operations",
            "segment": "enterprise",
            "category": "procurement",
            "sub_category": "approval",
            "sub_subcategory": "tier2",
            "business_task": "contract_review",
            "status": "resolved",
            "confidence_level": "level-1",
            "combined_cosine_similarity": 0.92,
            "agent_id": "A-102",
            "concept_key": "concept-11",
            "corpus_id": "corp-001",
        },
        {
            "use_case_guid": "UC-02",
            "domain": "knowledge",
            "segment": "enterprise",
            "category": "workspace",
            "sub_category": "search",
            "sub_subcategory": "faq",
            "business_task": "answer_generation",
            "status": "ambiguous",
            "confidence_level": "level-1",
            "combined_cosine_similarity": 0.84,
            "agent_id": "A-205",
            "concept_key": "concept-12",
            "corpus_id": "corp-002",
        },
        {
            "use_case_guid": "undetermined",
            "domain": "undetermined",
            "segment": "undetermined",
            "category": "undetermined",
            "sub_category": "undetermined",
            "sub_subcategory": "undetermined",
            "business_task": "undetermined",
            "status": "ambiguous",
            "confidence_level": "level-1",
            "combined_cosine_similarity": 0.55,
            "agent_id": "A-301",
            "concept_key": "concept-13",
            "corpus_id": "corp-003",
        },
        {
            "use_case_guid": "UC-03",
            "domain": "sales",
            "segment": "midmarket",
            "category": "triage",
            "sub_category": "routing",
            "sub_subcategory": "lead",
            "business_task": "handoff",
            "status": "resolved",
            "confidence_level": "level-2",
            "combined_cosine_similarity": 0.74,
            "agent_id": "A-401",
            "concept_key": "concept-15",
            "corpus_id": "corp-002",
        },
        {
            "use_case_guid": "UC-04",
            "domain": "support",
            "segment": "enterprise",
            "category": "incident",
            "sub_category": "diagnostic",
            "sub_subcategory": "network",
            "business_task": "root_cause",
            "status": "ambiguous",
            "confidence_level": "level-1",
            "combined_cosine_similarity": 0.64,
            "agent_id": "A-402",
            "concept_key": "concept-16",
            "corpus_id": "corp-003",
        },
    ]
    _write_jsonl(root / "classifications.jsonl", classifications)

    dataset = {
        "examples": [
            {
                "corpus_id": "corp-001",
                "agent": "tenant|A-102",
                "source": {
                    "corpus_id": "corp-001",
                    "is_synthetic": True,
                    "source_hash": "abc123",
                },
                "shape": {
                    "turn_count": 1,
                    "tool_call_count": 2,
                    "had_error": False,
                },
                "expected_label": True,
                "metadata": {"source_hash": "abc123", "pass_rate": 0.93, "count": 211, "task": "contract_review", "domain": "procurement"},
                "snippet": {
                    "user": "<script>alert('x')</script> user asks for contract review",
                    "assistant": "assistant returns bounded recommendation",
                },
            },
            {
                "corpus_id": "corp-002",
                "agent": "tenant|A-205",
                "source": {"corpus_id": "corp-002", "is_synthetic": False, "source_hash": "def456"},
                "shape": {"turn_count": 2, "tool_call_count": 1, "had_error": True},
                "expected_label": False,
                "metadata": {"source_hash": "def456", "pass_rate": 0.88, "count": 122, "task": "answer_generation", "domain": "knowledge"},
                "snippet": {"user": "user requests knowledge search fallback", "assistant": "assistant asks clarifying question"},
            },
            {
                "corpus_id": "corp-003",
                "agent": "tenant|A-301",
                "source": {"corpus_id": "corp-003", "is_synthetic": True, "source_hash": "ghi789"},
                "shape": {"turn_count": 3, "tool_call_count": 0, "had_error": False},
                "expected_label": True,
                "metadata": {"source_hash": "ghi789", "pass_rate": 0.79, "count": 98, "task": "root_cause", "domain": "support"},
                "snippet": {"user": "user reports intermittent outage", "assistant": "assistant proposes diagnostics checklist"},
            },
        ],
        "source_summary": {
            "schema": {
                "description": "Combined synthetic evaluation corpus built from source corpora.",
                "expected_label_field": "labels_by_unit",
                "snippet_policy": "bounded first-turn preview",
            },
            "overall": {"unit_count": 431, "pass_count": 368, "pass_rate": 0.85},
            "by_corpus": [
                {"corpus_id": "corp-001", "unit_count": 211, "pass_count": 196, "pass_rate": 0.93, "source_hash": "abc123"},
                {"corpus_id": "corp-002", "unit_count": 122, "pass_count": 106, "pass_rate": 0.87, "source_hash": "def456"},
                {"corpus_id": "corp-003", "unit_count": 98, "pass_count": 66, "pass_rate": 0.67, "source_hash": "ghi789"},
            ],
        },
        "synthesized_fields": {
            "source_synthetic": ["expected_label", "shape.turn_count"],
            "report_derived": ["snippet.user", "snippet.assistant"],
        },
        "schema_explanation": "Examples are synthesized from stratified use-case packets and constrained by taxonomy schema.",
    }
    _write_json(root / "dataset_examples.json", dataset)

    (root / "methodology.md").write_text(
        "# Methodology\\n- Derived from TrialMetrics and aggregate aliases\\n## Limits\\nNo secrets; bounded synthetic rows.",
        encoding="utf-8",
    )

    _write_json(
        root / "manifest.json",
        {
            "version": "sampling-v6-manifest-v1",
            "generated_at": "2026-08-21T00:00:00Z",
            "status": "complete",
        },
    )

    return V6ReportInputs(
        aggregate=root / "aggregate.json",
        runs_jsonl=root / "runs.jsonl",
        memberships=root / "memberships.jsonl",
        classifications=root / "classifications.jsonl",
        dataset_examples=root / "dataset_examples.json",
        methodology=root / "methodology.md",
        manifest=root / "manifest.json",
    )


def test_v6_report_renders_interactive_sections_and_schema_content(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    output_path = tmp_path / DEFAULT_OUTPUT_NAME

    html_path = write_v6_html_report(output_path=output_path, inputs=inputs, pdf=False)
    html = html_path.read_text(encoding="utf-8")

    assert html_path.exists()
    assert "Choosing a Sampling Strategy Under Evaluation Budget Constraints" in html
    assert "Population Units" in html
    assert "1200" in html
    assert "Methods" in html and "5 expected" in html
    assert "Trials" in html and "3" in html
    assert "64-256" in html
    assert "2026-08-21 00:00 UTC" in html

    assert "cap-select" in html
    assert "metric-tab" in html
    assert "method-toggle" in html
    assert "mode-tab" in html
    assert "renderAll" in html
    assert "interactive-chart" in html
    assert "interactive-table" in html
    assert "<div class='chart-shell screen-only'>" in html
    assert "<div id='top-five-table' class='screen-only'>" in html
    assert "@page { size: A4 landscape; margin: 10mm; }" in html
    assert "table-layout:fixed" in html
    assert "overflow-wrap:anywhere" in html
    assert "word-break:break-word" in html
    assert "break-inside:avoid-page !important" in html
    assert "height:125mm !important" in html
    assert "screen-only" in html
    assert "print-only" in html
    assert "print-compact" in html

    assert "arm1_global_random" in html
    assert "arm2_embedding_idw" in html
    assert "ARM2 Embedding IDW" in html

    assert "Token Diagnostics" in html
    assert "Actual/Nominal" in html
    assert "15k planning conversion" in html
    assert "Five Largest Agents: Sentinel Drilldown" in html
    assert "Maven Business Use-Case Assignment" in html
    assert "Below-0.30 fallback" in html
    assert "provisional" in html
    assert "Compact taxonomy mapping table" in html
    assert "Dataset and Ground Truth" in html
    assert "Method Guide" in html
    assert "Generated Methodology Narrative (Artifact-Driven)" in html
    assert "Core Equations" in html
    assert "Hajek estimator" in html
    assert "Cap from token budget" in html
    assert "15000" in html
    assert "Worked Example: ARM2 IDW Row" in html
    assert "Worked Example: ARM3 Membership Floor" in html
    assert "Comparative Results" in html
    assert html.count("<svg") >= 9
    assert "mae-vs-cap-chart" in html
    assert "coverage-concept-chart" in html
    assert "coverage-maven-chart" in html
    assert "coverage-agent-chart" in html
    assert "mae-vs-maven-frontier" in html
    assert "token-ratio-chart" in html
    assert "arm3-floor-chart" in html
    assert "idw-provenance-chart" in html
    assert "idw-quality-chart" in html
    assert "0.1113" in html
    assert "maven-status-confidence-chart" in html
    assert "maven-similarity-histogram" in html
    assert "top-five-error-heatmap" in html
    assert html.count("id='mae-vs-cap-chart'") == 1
    assert html.count("id='coverage-concept-chart'") == 1
    assert html.count("id='coverage-maven-chart'") == 1
    assert html.count("id='coverage-agent-chart'") == 1
    assert "Decision guide:" in html
    assert "near-tie" in html
    assert "Mean MAE winners" in html
    assert "Median MAE winners" in html
    assert "ARM5 is not broadly competitive at small caps" in html
    assert "classification-derived and provisional" in html
    assert "worse in" in html
    assert "not an accuracy benchmark" in html
    assert "Ambiguous 3 (60.0%)" in html
    assert "because 60.0% of assignments are Ambiguous" in html
    assert html.count("60.0% of assignments are Ambiguous") == 2
    assert "because 0.0% of assignments are Ambiguous" not in html
    assert "91.8%" not in html
    assert "0.30" in html and "0.70" in html
    assert "Top Use Cases (Human-readable labels, GUID secondary)" in html
    assert "classification-derived/provisional" in html

    # Navigation and section ids
    assert "<nav class='toc screen-only'" in html
    assert "<nav class='toc print-only'" in html
    assert "href='#executive-recommendations'" in html
    assert "href='#purpose-and-decision'" in html
    assert "href='#dataset-ground-truth'" in html
    assert "href='#maven-use-case-assignment'" in html
    assert "href='#experiment-design-metrics'" in html
    assert "href='#method-guide'" in html
    assert "href='#comparative-results'" in html
    assert "href='#five-largest-agents'" in html
    assert "href='#limitations-validity'" in html
    assert "href='#recommendations-next-steps'" in html
    assert "href='#interactive-explorer'" in html
    assert "href='#appendix'" in html

    # Required order: purpose -> recommendations -> dataset -> maven -> design -> methods -> results -> sentinel -> limitations -> recs -> interactive -> appendix
    assert html.index("id='purpose-and-decision'") < html.index("id='executive-recommendations'")
    assert html.index("id='executive-recommendations'") < html.index("id='dataset-ground-truth'")
    assert html.index("id='dataset-ground-truth'") < html.index("id='maven-use-case-assignment'")
    assert html.index("id='maven-use-case-assignment'") < html.index("id='experiment-design-metrics'")
    assert html.index("id='experiment-design-metrics'") < html.index("id='method-guide'")
    assert html.index("id='method-guide'") < html.index("id='comparative-results'")
    assert html.index("id='comparative-results'") < html.index("id='five-largest-agents'")
    assert html.index("id='five-largest-agents'") < html.index("id='limitations-validity'")
    assert html.index("id='limitations-validity'") < html.index("id='recommendations-next-steps'")
    assert html.index("id='recommendations-next-steps'") < html.index("id='interactive-explorer'")
    assert html.index("id='interactive-explorer'") < html.index("id='appendix'")

    # Header wording fix: no legacy "Conclusion:" pointer
    assert "<strong>Conclusion:</strong>" not in html

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "\\u003cscript\\u003e" in html
    assert "N/A" in html
    assert "https://" not in html.lower()
    assert "http://" not in html.lower()


def test_v6_report_infers_complete_status_for_legacy_finished_bundle(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    aggregate = json.loads(inputs.aggregate.read_text(encoding="utf-8"))
    aggregate.pop("status", None)
    aggregate.pop("run_status", None)
    _write_json(inputs.aggregate, aggregate)
    manifest = json.loads(inputs.manifest.read_text(encoding="utf-8"))
    manifest.pop("status", None)
    _write_json(inputs.manifest, manifest)

    output_path = tmp_path / "legacy-complete.html"
    html = write_v6_html_report(output_path=output_path, inputs=inputs, pdf=False).read_text(encoding="utf-8")

    assert html.count(">complete<") >= 2


def test_v6_report_rejects_wrong_bundle_version(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    _write_json(inputs.aggregate, {"version": "sampling-v5-bundle-v1", "methods": []})
    with pytest.raises(ValueError, match="version"):
        write_v6_html_report(output_path=tmp_path / "bad.html", inputs=inputs, pdf=False)


def test_v6_report_compose_pdf_command_uses_file_uri_and_validate_size(tmp_path: Path) -> None:
    html_path = tmp_path / "report.html"
    pdf_path = tmp_path / "report.pdf"
    cmd = compose_pdf_command("/fake/browser", html_path, pdf_path)

    assert cmd[0] == "/fake/browser"
    assert "--no-pdf-header-footer" in cmd
    pdf_tokens = [token for token in cmd if token.startswith("--print-to-pdf=")]
    assert len(pdf_tokens) == 1
    assert pdf_tokens[0] == f"--print-to-pdf={pdf_path.resolve()}"
    assert cmd[-1].startswith("file://")

    pdf_path.write_bytes(b"%PDF" + b"x" * 1300)
    assert validate_pdf_file(pdf_path)

    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF" + b"x" * 100)
    with pytest.raises(ValueError, match="1KB"):
        validate_pdf_file(bad)


def test_v6_report_normalizes_runner_shaped_dicts_and_new_membership_idw_fields(tmp_path: Path) -> None:
    root = tmp_path / "artifact_bundle"
    root.mkdir(parents=True, exist_ok=True)

    _write_json(
        root / "aggregate.json",
        {
            "version": "sampling-v6-bundle-v1",
            "generated_at": "2026-08-21T00:00:00Z",
            "status": "complete",
            "run_status": "complete",
            "population_audit": {"unit_count": 12, "agent_count": 4},
            "methods": [
                {
                    "method_id": "arm2_embedding_idw",
                    "cap": 64,
                    "selected_count": {"mean": 64, "min": 64, "max": 64},
                    "absolute_aggregate_mae": {"mean": 0.11, "min": 0.1, "max": 0.13},
                    "concept_coverage": {"mean": 0.61, "min": 0.58, "max": 0.65},
                    "use_case_coverage": {"mean": 0.58, "min": 0.55, "max": 0.62},
                    "agent_coverage": {"mean": 0.54, "min": 0.5, "max": 0.56},
                    "actual_token_count": {"mean": 86000, "min": 83000, "max": 90000},
                    "nominal_budget": {"mean": 97000, "min": 95000, "max": 98000},
                }
            ],
        },
    )
    _write_jsonl(
        root / "runs.jsonl",
        [
            {
                "method_id": "arm2_embedding_idw",
                "cap": 64,
                "seed": 1,
                "selected_count": 64,
                "actual_token_count": 87500,
                "nominal_budget": 96000,
                "absolute_aggregate_mae": 0.12,
                "concept_coverage": 0.62,
                "use_case_coverage": 0.59,
                "agent_coverage": 0.55,
                "idw_provenance": {
                    "observed_count": 200,
                    "imputed_count": 80,
                    "provenance_counts": {
                        "observed": 200,
                        "idw": 40,
                        "global_mean": 35,
                        "agent_mean": 5,
                    },
                },
                "idw_quality": {
                    "absolute_aggregate_rate_error": 0.021,
                    "per_unit_mae": 0.044,
                    "brier_score": 0.118,
                    "macro_per_agent_mae": 0.053,
                    "unjudged_only_mae": 0.061,
                    "unjudged_only_brier": 0.126,
                    "expected_calibration_error": 0.028,
                },
            }
        ],
    )
    _write_jsonl(
        root / "memberships.jsonl",
        [
            {
                "method_id": "arm3_agent_round_robin_floor",
                "cap": 64,
                "selected_agent_count": 5,
                "eligible_agents_with_at_least_3": 10,
                "agents_with_at_least_3": 4,
                "represented_strata": ["uc-1", "uc-2", "uc-3"],
                "agent_coverage": 0.75,
                "total_floor_target": 20,
                "floor_prefix_count": 15,
                "floor_complete": True,
                "arm3_floor": {
                    "total_floor_target": 20,
                    "floor_prefix_count": 15,
                    "floor_complete": True,
                },
            }
        ],
    )
    _write_jsonl(root / "classifications.jsonl", [])
    _write_json(root / "dataset_examples.json", {"examples": [], "source_summary": {}, "schema_explanation": ""})
    (root / "methodology.md").write_text("# Methodology", encoding="utf-8")
    _write_json(root / "manifest.json", {"version": "sampling-v6-manifest-v1", "generated_at": "2026-08-21T00:00:00Z", "status": "complete"})

    inputs = V6ReportInputs(
        aggregate=root / "aggregate.json",
        runs_jsonl=root / "runs.jsonl",
        memberships=root / "memberships.jsonl",
        classifications=root / "classifications.jsonl",
        dataset_examples=root / "dataset_examples.json",
        methodology=root / "methodology.md",
        manifest=root / "manifest.json",
    )
    artifacts = load_v6_artifacts(inputs)

    row = artifacts.aggregate["aggregate_rows"][0]
    assert row["actual_token_count"] == 86000.0
    assert row["nominal_budget"] == 97000.0
    assert row["selected_count"] == 64

    membership = artifacts.memberships[0]
    assert membership["represented_strata"] == 3
    assert membership["eligible_agents_with_at_least_3"] == 10
    assert membership["agents_with_at_least_3"] == 4
    assert membership["total_floor_target"] == 20
    assert membership["floor_prefix_count"] == 15
    assert membership["arm3_floor_completion"] == pytest.approx(0.75)
    assert membership["floor_complete"] is False

    html_path = write_v6_html_report(output_path=tmp_path / "schema.html", inputs=inputs, pdf=False)
    html = html_path.read_text(encoding="utf-8")
    assert "86000" in html
    assert "97000" in html
    assert "Provenance Categories" in html
    assert "Abs Aggregate Error" in html
    assert "Per-Unit MAE" in html
    assert "Brier" in html
    assert "ECE" in html
    assert "Arm3 Floor Target" in html
    assert "Arm3 Floor Prefix Selected" in html
    assert "Eligible Agents >=3" in html
    assert "Selected Agents >=3" in html
    assert "Floor Complete" in html
    assert "No" in html
    assert "Cap</th><th>Observed</th><th>Imputed</th><th>Aggregate Error</th><th>Brier</th><th>ECE</th>" in html
    assert "Selected/Census rate" in html
    assert "Membership Coverage (Print)" in html
    assert "Arm3 Floor Status (Print)" in html


def test_idw_provenance_chart_accounts_for_canonical_population_categories(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    html = render_v6_html_report(inputs)

    assert "global_mean" in html
    assert "agent_mean" in html
    assert "idw-provenance-chart" in html


def test_runner_to_report_integration_renders_nonzero_fields(tmp_path: Path) -> None:
    result, _ = _run_bundle(tmp_path, out_name="integration")
    paths = result["output_paths"]
    inputs = V6ReportInputs(
        aggregate=Path(paths["aggregate"]),
        runs_jsonl=Path(paths["runs"]),
        memberships=Path(paths["memberships"]),
        classifications=Path(paths["classifications"]),
        dataset_examples=Path(paths["dataset_examples"]),
        methodology=Path(paths["methodology"]),
        manifest=Path(paths["manifest"]),
    )
    html_path = write_v6_html_report(output_path=tmp_path / "integration.html", inputs=inputs, pdf=False)
    html = html_path.read_text(encoding="utf-8")

    assert "Token Diagnostics" in html
    assert "Actual/Nominal" in html
    assert "g. ARM3 floor behavior and ARM4/5 relationship" in html
    assert "Arm3 Floor Target" in html
    assert "Arm3 Floor Prefix Selected" in html
    assert "f. ARM2 imputation/calibration" in html
    assert "Provenance Categories" in html
    assert "Abs Aggregate Error" in html
    assert "Per-Unit MAE" in html
    assert "Brier" in html
    assert "Recommendations and Next Steps" in html
