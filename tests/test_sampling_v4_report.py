from __future__ import annotations

import json
from pathlib import Path

import pytest

from sampling_comparison.v4_report import (
    DEFAULT_OUTPUT_NAME,
    V4ReportInputs,
    load_v4_artifacts,
    render_v4_html_report,
    validate_report_manifest,
    validate_v4_artifacts,
    write_v4_html_report,
)


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_canonical(row) for row in rows) + "\n", encoding="utf-8")


def _build_fixture_bundle(tmp_path: Path) -> V4ReportInputs:
    root = tmp_path / "outputs_sampling_v4" / "runs" / "fixture"
    root.mkdir(parents=True, exist_ok=True)

    source_v3_dir = root / "source_v3"
    source_v3_dir.mkdir(parents=True, exist_ok=True)

    source_v3_aggregate = {
        "version": "sampling-v3-bundle-v1",
        "generated_at": "2026-08-05T01:00:00Z",
        "runtime_seconds": 8.25,
        "config": {
            "runtime": {
                "embedding_ledger": {
                    "embedding_calls": 12,
                    "embedding_input_tokens": 24000,
                    "embedding_model_id": "text-embedding-3-small",
                    "embedding_deployment_id": "emb-v3",
                }
            }
        },
    }

    source_v3_token_inventory = [
        {"unit_id": "u1", "original_tokens": 1000, "emitted_tokens": 920, "truncated": False},
        {"unit_id": "u2", "original_tokens": 1100, "emitted_tokens": 1020, "truncated": True},
        {"unit_id": "u3", "original_tokens": 1300, "emitted_tokens": 1200, "truncated": False},
        {"unit_id": "u4", "original_tokens": 900, "emitted_tokens": 860, "truncated": False},
    ]

    source_v3_quadrant = {
        "version": "sampling-v3-quadrant-v1",
        "config": {"legacy_quadrant_tiers_pct": [20, 30], "methods": [
            "random_sampling_token_priority",
            "adaptive_minhash_32x4_token",
            "adaptive_embedding_fullsession_token",
        ], "replay_count": 1},
        "quadrants": {
            "quadrant_summary": {
                "high_variety_high_velocity": {"unit_count": 2, "agent_count": 2, "corpus_counts": {"synth": 2}},
                "high_variety_low_velocity": {"unit_count": 1, "agent_count": 1, "corpus_counts": {"synth": 1}},
                "low_variety_high_velocity": {"unit_count": 1, "agent_count": 1, "corpus_counts": {"synth": 1}},
                "low_variety_low_velocity": {"unit_count": 0, "agent_count": 0, "corpus_counts": {"synth": 0}},
            }
        },
        "aggregate_groups": [
            {"method": "random_sampling_token_priority", "budget_tokens": 4500, "legacy_tier_pct_provenance": [20], "quadrant": "high_variety_high_velocity", "representation_mean": 0.4, "concept_coverage_mean": 0.5, "zero_selection_agent_rate_mean": 0.0},
            {"method": "adaptive_minhash_32x4_token", "budget_tokens": 4500, "legacy_tier_pct_provenance": [20], "quadrant": "high_variety_high_velocity", "representation_mean": 0.3, "concept_coverage_mean": 0.4, "zero_selection_agent_rate_mean": 0.25},
            {"method": "adaptive_embedding_fullsession_token", "budget_tokens": 4500, "legacy_tier_pct_provenance": [20], "quadrant": "high_variety_high_velocity", "representation_mean": 0.5, "concept_coverage_mean": 0.6, "zero_selection_agent_rate_mean": 0.0},
            {"method": "random_sampling_token_priority", "budget_tokens": 9000, "legacy_tier_pct_provenance": [30], "quadrant": "high_variety_high_velocity", "representation_mean": 0.7, "concept_coverage_mean": 0.8, "zero_selection_agent_rate_mean": 0.0},
            {"method": "adaptive_minhash_32x4_token", "budget_tokens": 9000, "legacy_tier_pct_provenance": [30], "quadrant": "high_variety_high_velocity", "representation_mean": 0.6, "concept_coverage_mean": 0.7, "zero_selection_agent_rate_mean": 0.1},
            {"method": "adaptive_embedding_fullsession_token", "budget_tokens": 9000, "legacy_tier_pct_provenance": [30], "quadrant": "high_variety_high_velocity", "representation_mean": 0.8, "concept_coverage_mean": 0.9, "zero_selection_agent_rate_mean": 0.0},
        ],
    }

    source_v3_throughput = {
        "version": "sampling-v3-throughput-v1",
        "config": {
            "arrival_rates_sessions_per_second": [0.25, 1.0],
            "eval_capacity_rates_sessions_per_second": [0.25, 1.0],
            "queue_capacity_policy": "bounded",
            "methods": [
                "random_sampling_token_priority",
                "adaptive_minhash_32x4_token",
                "adaptive_embedding_fullsession_token",
            ],
            "legacy_budget_tiers_pct": [20, 30],
        },
        "aggregate_grid": [
            {"arrival_rate_sessions_per_second": 0.25, "eval_capacity_sessions_per_second": 0.25, "method": "random_sampling_token_priority", "budget_tokens": 7500, "token_pressure_ratio_mean": 0.8, "decision_runtime_ms_p95_mean": 120},
            {"arrival_rate_sessions_per_second": 1.0, "eval_capacity_sessions_per_second": 0.25, "method": "random_sampling_token_priority", "budget_tokens": 7500, "token_pressure_ratio_mean": 1.0, "decision_runtime_ms_p95_mean": 260},
            {"arrival_rate_sessions_per_second": 0.25, "eval_capacity_sessions_per_second": 0.25, "method": "adaptive_minhash_32x4_token", "budget_tokens": 7500, "token_pressure_ratio_mean": 0.7, "decision_runtime_ms_p95_mean": 180},
            {"arrival_rate_sessions_per_second": 1.0, "eval_capacity_sessions_per_second": 0.25, "method": "adaptive_minhash_32x4_token", "budget_tokens": 7500, "token_pressure_ratio_mean": 0.95, "decision_runtime_ms_p95_mean": 300},
            {"arrival_rate_sessions_per_second": 0.25, "eval_capacity_sessions_per_second": 0.25, "method": "adaptive_embedding_fullsession_token", "budget_tokens": 7500, "token_pressure_ratio_mean": 0.85, "decision_runtime_ms_p95_mean": 200},
            {"arrival_rate_sessions_per_second": 1.0, "eval_capacity_sessions_per_second": 0.25, "method": "adaptive_embedding_fullsession_token", "budget_tokens": 7500, "token_pressure_ratio_mean": 1.0, "decision_runtime_ms_p95_mean": 420},
            {"arrival_rate_sessions_per_second": 0.25, "eval_capacity_sessions_per_second": 1.0, "method": "adaptive_embedding_fullsession_token", "budget_tokens": 15000, "token_pressure_ratio_mean": 0.75, "decision_runtime_ms_p95_mean": 90},
            {"arrival_rate_sessions_per_second": 1.0, "eval_capacity_sessions_per_second": 1.0, "method": "adaptive_embedding_fullsession_token", "budget_tokens": 15000, "token_pressure_ratio_mean": 0.95, "decision_runtime_ms_p95_mean": 160},
            {"arrival_rate_sessions_per_second": 0.25, "eval_capacity_sessions_per_second": 1.0, "method": "random_sampling_token_priority", "budget_tokens": 15000, "token_pressure_ratio_mean": 0.65, "decision_runtime_ms_p95_mean": 80},
            {"arrival_rate_sessions_per_second": 1.0, "eval_capacity_sessions_per_second": 1.0, "method": "random_sampling_token_priority", "budget_tokens": 15000, "token_pressure_ratio_mean": 0.9, "decision_runtime_ms_p95_mean": 140},
            {"arrival_rate_sessions_per_second": 0.25, "eval_capacity_sessions_per_second": 1.0, "method": "adaptive_minhash_32x4_token", "budget_tokens": 15000, "token_pressure_ratio_mean": 0.6, "decision_runtime_ms_p95_mean": 100},
            {"arrival_rate_sessions_per_second": 1.0, "eval_capacity_sessions_per_second": 1.0, "method": "adaptive_minhash_32x4_token", "budget_tokens": 15000, "token_pressure_ratio_mean": 0.88, "decision_runtime_ms_p95_mean": 170},
        ],
    }

    _write_json(source_v3_dir / "aggregate.json", source_v3_aggregate)
    _write_jsonl(source_v3_dir / "token_inventory.jsonl", source_v3_token_inventory)
    _write_json(source_v3_dir / "quadrant.json", source_v3_quadrant)
    _write_json(source_v3_dir / "throughput.json", source_v3_throughput)

    source_v3_manifest = {
        "version": "sampling-v3-manifest-v1",
        "generated_at": "2026-08-05T01:00:05Z",
        "artifacts": {
            "aggregate": {
                "path": "aggregate.json",
                "bytes": int((source_v3_dir / "aggregate.json").stat().st_size),
                "sha256": _sha(source_v3_dir / "aggregate.json"),
            },
            "token_inventory": {
                "path": "token_inventory.jsonl",
                "bytes": int((source_v3_dir / "token_inventory.jsonl").stat().st_size),
                "sha256": _sha(source_v3_dir / "token_inventory.jsonl"),
            },
            "quadrant": {
                "path": "quadrant.json",
                "bytes": int((source_v3_dir / "quadrant.json").stat().st_size),
                "sha256": _sha(source_v3_dir / "quadrant.json"),
            },
            "throughput": {
                "path": "throughput.json",
                "bytes": int((source_v3_dir / "throughput.json").stat().st_size),
                "sha256": _sha(source_v3_dir / "throughput.json"),
            },
        },
    }
    _write_json(source_v3_dir / "manifest.json", source_v3_manifest)

    runs = [
        {
            "method": "random_sampling_token_priority",
            "budget_tokens": 6000,
            "legacy_tier_pct": 20,
            "repetition": 0,
            "order_hash": "ord-a",
            "selected_ids": ["u1", "u2"],
            "selected_count": 2,
            "selected_pass_rate": 0.55,
            "selected_only_pass_rate": 0.55,
            "census_pass_rate": 0.60,
            "absolute_error": 0.05,
            "selected_only_absolute_error": 0.05,
            "fraction_saved": 0.5,
            "concept_coverage": 0.5,
            "representation": 0.5,
            "estimation_mode": "selected_only",
            "model_assisted": None,
            "telemetry": {},
        },
        {
            "method": "adaptive_minhash_32x4_token",
            "budget_tokens": 6000,
            "legacy_tier_pct": 20,
            "repetition": 0,
            "order_hash": "ord-a",
            "selected_ids": ["u1", "u3"],
            "selected_count": 2,
            "selected_pass_rate": 0.52,
            "selected_only_pass_rate": 0.52,
            "census_pass_rate": 0.60,
            "absolute_error": 0.08,
            "selected_only_absolute_error": 0.08,
            "fraction_saved": 0.5,
            "concept_coverage": 0.4,
            "representation": 0.4,
            "estimation_mode": "selected_only",
            "model_assisted": None,
            "telemetry": {"no_candidate_novel": 3, "full_scan_fallbacks": 0},
        },
        {
            "method": "adaptive_embedding_fullsession_token",
            "budget_tokens": 6000,
            "legacy_tier_pct": 20,
            "repetition": 0,
            "order_hash": "ord-a",
            "selected_ids": ["u1", "u2"],
            "selected_count": 2,
            "selected_pass_rate": 0.50,
            "selected_only_pass_rate": 0.50,
            "census_pass_rate": 0.60,
            "absolute_error": 0.10,
            "selected_only_absolute_error": 0.10,
            "fraction_saved": 0.5,
            "concept_coverage": 0.6,
            "representation": 0.6,
            "estimation_mode": "model_assisted_idw",
            "telemetry": {},
            "model_assisted": {
                "rates": {
                    "estimated_pass_rate": 0.48,
                    "absolute_aggregate_rate_error": 0.12,
                    "delta_vs_selected_only_absolute_error": 0.02,
                    "provenance_population_weighted_rates": {"observed": 0.50, "idw": 0.40, "prior": 0.10},
                },
                "counts": {
                    "population_count": 4,
                    "observed_count": 2,
                    "imputed_count": 2,
                    "zero_donor_agent_count": 0,
                    "prior_count": 1,
                    "provenance_counts": {"observed": 2, "idw": 1, "prior": 1},
                },
                "metrics": {
                    "per_unit_mae": 0.12,
                    "brier_score": 0.15,
                    "expected_calibration_error": 0.06,
                    "macro_per_agent_mae": 0.09,
                    "unjudged_only_mae": 0.13,
                    "unjudged_only_brier": 0.16,
                    "calibration_bins": [
                        {"bin_index": 0.0, "lower": 0.0, "upper": 0.5, "count": 2.0, "avg_prediction": 0.42, "avg_label": 0.50, "abs_gap": 0.08},
                        {"bin_index": 1.0, "lower": 0.5, "upper": 1.0, "count": 2.0, "avg_prediction": 0.58, "avg_label": 0.50, "abs_gap": 0.08},
                    ],
                    "nearest_distance_error_bins": [
                        {"bin_index": 0, "lower": 0.0, "upper": 0.2, "count": 1, "avg_distance": 0.12, "mae": 0.07},
                        {"bin_index": 1, "lower": 0.2, "upper": 0.4, "count": 1, "avg_distance": 0.25, "mae": 0.11},
                    ],
                    "per_agent": [
                        {"agent_id": "tenant|a", "population_count": 3, "observed_count": 2, "imputed_count": 1, "estimated_pass_rate": 0.54, "census_pass_rate": 0.67, "absolute_error": 0.13},
                        {"agent_id": "tenant|b", "population_count": 1, "observed_count": 0, "imputed_count": 1, "estimated_pass_rate": 0.31, "census_pass_rate": 0.00, "absolute_error": 0.31},
                    ],
                    "leave_one_out": {"judged_count": 2, "mae": 0.19, "brier_score": 0.18},
                },
            },
        },
        {
            "method": "random_sampling_token_priority",
            "budget_tokens": 12000,
            "legacy_tier_pct": 30,
            "repetition": 0,
            "order_hash": "ord-b",
            "selected_ids": ["u1", "u2", "u4"],
            "selected_count": 3,
            "selected_pass_rate": 0.60,
            "selected_only_pass_rate": 0.60,
            "census_pass_rate": 0.60,
            "absolute_error": 0.00,
            "selected_only_absolute_error": 0.00,
            "fraction_saved": 0.25,
            "concept_coverage": 0.8,
            "representation": 0.8,
            "estimation_mode": "selected_only",
            "model_assisted": None,
            "telemetry": {},
        },
        {
            "method": "adaptive_minhash_32x4_token",
            "budget_tokens": 12000,
            "legacy_tier_pct": 30,
            "repetition": 0,
            "order_hash": "ord-b",
            "selected_ids": ["u1", "u3", "u4"],
            "selected_count": 3,
            "selected_pass_rate": 0.58,
            "selected_only_pass_rate": 0.58,
            "census_pass_rate": 0.60,
            "absolute_error": 0.02,
            "selected_only_absolute_error": 0.02,
            "fraction_saved": 0.25,
            "concept_coverage": 0.7,
            "representation": 0.7,
            "estimation_mode": "selected_only",
            "model_assisted": None,
            "telemetry": {"no_candidate_novel": 2, "full_scan_fallbacks": 0},
        },
        {
            "method": "adaptive_embedding_fullsession_token",
            "budget_tokens": 12000,
            "legacy_tier_pct": 30,
            "repetition": 0,
            "order_hash": "ord-b",
            "selected_ids": ["u1", "u2", "u3"],
            "selected_count": 3,
            "selected_pass_rate": 0.66,
            "selected_only_pass_rate": 0.66,
            "census_pass_rate": 0.60,
            "absolute_error": 0.06,
            "selected_only_absolute_error": 0.06,
            "fraction_saved": 0.25,
            "concept_coverage": 0.9,
            "representation": 0.9,
            "estimation_mode": "model_assisted_idw",
            "telemetry": {},
            "model_assisted": {
                "rates": {
                    "estimated_pass_rate": 0.62,
                    "absolute_aggregate_rate_error": 0.02,
                    "delta_vs_selected_only_absolute_error": -0.04,
                    "provenance_population_weighted_rates": {"observed": 0.75, "idw": 0.25, "prior": 0.0},
                },
                "counts": {
                    "population_count": 4,
                    "observed_count": 3,
                    "imputed_count": 1,
                    "zero_donor_agent_count": 0,
                    "prior_count": 0,
                    "provenance_counts": {"observed": 3, "idw": 1, "prior": 0},
                },
                "metrics": {
                    "per_unit_mae": 0.08,
                    "brier_score": 0.09,
                    "expected_calibration_error": 0.04,
                    "macro_per_agent_mae": 0.07,
                    "unjudged_only_mae": 0.10,
                    "unjudged_only_brier": 0.12,
                    "calibration_bins": [
                        {"bin_index": 0.0, "lower": 0.0, "upper": 0.5, "count": 1.0, "avg_prediction": 0.35, "avg_label": 0.00, "abs_gap": 0.35},
                        {"bin_index": 1.0, "lower": 0.5, "upper": 1.0, "count": 3.0, "avg_prediction": 0.69, "avg_label": 0.67, "abs_gap": 0.02},
                    ],
                    "nearest_distance_error_bins": [
                        {"bin_index": 0, "lower": 0.0, "upper": 0.2, "count": 1, "avg_distance": 0.08, "mae": 0.05},
                        {"bin_index": 1, "lower": 0.2, "upper": 0.4, "count": 1, "avg_distance": 0.22, "mae": 0.09},
                    ],
                    "per_agent": [
                        {"agent_id": "tenant|a", "population_count": 3, "observed_count": 3, "imputed_count": 0, "estimated_pass_rate": 0.67, "census_pass_rate": 0.67, "absolute_error": 0.00},
                        {"agent_id": "tenant|b", "population_count": 1, "observed_count": 0, "imputed_count": 1, "estimated_pass_rate": 0.40, "census_pass_rate": 0.00, "absolute_error": 0.40},
                    ],
                    "leave_one_out": {"judged_count": 3, "mae": 0.11, "brier_score": 0.10},
                },
            },
        },
    ]

    aggregate = {
        "version": "sampling-v4-bundle-v1",
        "generated_at": "2026-08-05T02:00:00Z",
        "population_count": 4,
        "runtime_seconds": 8.25,
        "outcome": {
            "version": "sampling-v4-outcome-v1",
            "aggregate": [
                {
                    "method": "random_sampling_token_priority",
                    "budget_tokens": 6000,
                    "replays": 1,
                    "selected_only_mae": {"mean": 0.05, "empirical_low": 0.05, "empirical_high": 0.05},
                },
                {
                    "method": "adaptive_minhash_32x4_token",
                    "budget_tokens": 6000,
                    "replays": 1,
                    "selected_only_mae": {"mean": 0.08, "empirical_low": 0.08, "empirical_high": 0.08},
                },
                {
                    "method": "adaptive_embedding_fullsession_token",
                    "budget_tokens": 6000,
                    "replays": 1,
                    "selected_only_mae": {"mean": 0.10, "empirical_low": 0.10, "empirical_high": 0.10},
                    "idw_absolute_error": {"mean": 0.12, "empirical_low": 0.12, "empirical_high": 0.12},
                    "model_assisted_counts_sum": {"population_count": 4, "observed_count": 2, "imputed_count": 2},
                },
                {
                    "method": "random_sampling_token_priority",
                    "budget_tokens": 12000,
                    "replays": 1,
                    "selected_only_mae": {"mean": 0.00, "empirical_low": 0.00, "empirical_high": 0.00},
                },
                {
                    "method": "adaptive_minhash_32x4_token",
                    "budget_tokens": 12000,
                    "replays": 1,
                    "selected_only_mae": {"mean": 0.02, "empirical_low": 0.02, "empirical_high": 0.02},
                },
                {
                    "method": "adaptive_embedding_fullsession_token",
                    "budget_tokens": 12000,
                    "replays": 1,
                    "selected_only_mae": {"mean": 0.06, "empirical_low": 0.06, "empirical_high": 0.06},
                    "idw_absolute_error": {"mean": 0.02, "empirical_low": 0.02, "empirical_high": 0.02},
                    "model_assisted_counts_sum": {"population_count": 4, "observed_count": 3, "imputed_count": 1},
                },
            ],
        },
        "source_v3": {
            "subdir": "source_v3",
            "bundle_version": "sampling-v3-bundle-v1",
            "manifest_relative_path": "source_v3/manifest.json",
            "manifest": {
                "path": "source_v3/manifest.json",
                "bytes": int((source_v3_dir / "manifest.json").stat().st_size),
                "sha256": _sha(source_v3_dir / "manifest.json"),
                "version": "sampling-v3-manifest-v1",
            },
        },
    }

    idw_config = {
        "version": "sampling-v4-idw-config-v1",
        "idw_config": {"k": 8, "power": 2.0, "eps": 1e-6, "exact_cosine_eps": 1e-8, "prior": 0.5},
    }

    methodology_delta = (
        "# V4 Methodology Delta\n\n"
        "- Outcome cells preserve exact token budgets from selection-stage bundles.\n"
        "- Random and MinHash remain selected-only.\n"
        "- Embedding adds model-assisted IDW estimates.\n"
        "- No Cochran/FPC.\n"
        "- Same-agent donor only.\n"
    )

    source_lineage = {
        "version": "sampling-v4-source-lineage-v1",
        "source_bundle_subdir": "source_v3",
        "source_bundle_version": "sampling-v3-bundle-v1",
        "source_outcome_version": "sampling-v3-outcome-v1",
        "source_manifest": {
            "path": "source_v3/manifest.json",
            "bytes": int((source_v3_dir / "manifest.json").stat().st_size),
            "sha256": _sha(source_v3_dir / "manifest.json"),
            "version": "sampling-v3-manifest-v1",
        },
        "selection_rerun": False,
        "augmentation": "augment_v3_outcome_with_idw",
    }

    files = {
        "aggregate": root / "aggregate.json",
        "runs_jsonl": root / "runs.jsonl",
        "idw_config": root / "idw_config.json",
        "methodology_delta": root / "methodology_delta.md",
        "source_lineage": root / "source_lineage.json",
    }

    _write_json(files["aggregate"], aggregate)
    _write_jsonl(files["runs_jsonl"], runs)
    _write_json(files["idw_config"], idw_config)
    files["methodology_delta"].write_text(methodology_delta, encoding="utf-8")
    _write_json(files["source_lineage"], source_lineage)

    manifest = {
        "version": "sampling-v4-manifest-v1",
        "generated_at": "2026-08-05T02:00:01Z",
        "artifacts": {},
        "source": {
            "source_subdir": "source_v3",
            "source_manifest_version": "sampling-v3-manifest-v1",
            "source_manifest_sha256": _sha(source_v3_dir / "manifest.json"),
            "source_manifest_relative_path": "source_v3/manifest.json",
        },
    }

    for key in ("aggregate", "runs_jsonl", "idw_config", "methodology_delta", "source_lineage"):
        path = files[key]
        manifest["artifacts"][key] = {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": int(path.stat().st_size),
            "sha256": _sha(path),
        }
    source_manifest_path = source_v3_dir / "manifest.json"
    manifest["artifacts"]["source_v3_manifest"] = {
        "path": str(source_manifest_path.relative_to(root)).replace("\\", "/"),
        "bytes": int(source_manifest_path.stat().st_size),
        "sha256": _sha(source_manifest_path),
    }

    _write_json(root / "manifest.json", manifest)

    return V4ReportInputs(
        aggregate=files["aggregate"],
        runs_jsonl=files["runs_jsonl"],
        idw_config=files["idw_config"],
        methodology_delta=files["methodology_delta"],
        source_lineage=files["source_lineage"],
        manifest=root / "manifest.json",
    )


def test_v4_report_renders_redesign_tabs_and_required_sections(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    artifacts = load_v4_artifacts(inputs)
    html = render_v4_html_report(artifacts)

    assert html.count('class="tab-button"') == 8

    expected_order = [
        "Executive Summary",
        "Outcomes",
        "Sampling Methods",
        "Quadrant Behavior",
        "Throughput",
        "Embedding Diagnostics",
        "Lineage &amp; Integrity",
        "Reproducibility &amp; Caveats",
    ]
    positions = [html.index(name) for name in expected_order]
    assert positions == sorted(positions)

    assert 'id="tab-executive"' in html
    assert 'aria-selected="true"' in html

    assert "Selected-only absolute aggregate rate error = |selected pass rate - census|" in html
    assert "IDW aggregate error = |observed+imputed population mean - census|" in html
    assert "Deterministic expected labels are pseudo-judge outputs" in html
    assert "Practical Recommendation" in html
    assert "Random selected-only is the accuracy default for this run" in html
    assert "MinHash won" in html
    assert "Embedding led concept coverage in" in html
    assert "should remain diagnostic or budget-specific" in html
    assert '<section class="print-keep"><h3>Metric Definitions</h3>' in html
    assert ".method-panel, .print-keep" in html

    assert html.count('<figure class="chart"><figcaption>Mean absolute error versus census by exact token budget') == 1
    assert html.count('<figure class="chart"><figcaption>Fraction saved by exact token budget') == 1
    assert html.count('<figure class="chart"><figcaption>Concept coverage by exact token budget') == 1
    outcomes = html[html.index('id="panel-outcomes"'):html.index('id="panel-methods"')]
    mae_chart = outcomes[:outcomes.index('</figure>')]
    assert mae_chart.count('<rect ') == 6
    assert '<circle ' not in mae_chart
    assert 'class="value-label"' in mae_chart
    assert outcomes.count('<figure class="chart">') == 3
    assert "Embedding provenance shares" not in outcomes
    assert "Embedding selected-only" not in outcomes
    assert "Full-session embedding + IDW" in outcomes
    assert "[low, high]" not in outcomes
    assert "confidence interval" not in outcomes.lower()

    assert "Representation ratio by quadrant, exact budget, and method" in html
    assert "Zero-selection agent rate by quadrant, budget, and method" in html
    assert "Exact Quadrant Results" in html
    assert "4,500" in html and "9,000" in html
    assert "128 values; 32 bands x 4 rows" in html
    assert "prior probability" in html
    assert "Token pressure heatmap" in html
    assert "P95 decision latency heatmap" in html
    assert "15,000" in html

    assert "Token and Embedding Ledger" in html
    assert "Selection-stage manifest" in html

    assert html.count('<figure class="diagram"><figcaption>') >= 5
    assert "Overall selection and estimation flow" in html
    assert "Random token-priority explainer" in html
    assert "MinHash LSH explainer (128 values, 32x4)" in html
    assert "Embedding selection explainer" in html
    assert "Post-freeze same-agent IDW explainer" in html
    assert "role=\"img\" aria-label=\"Overall flow from persisted source bundle through selection parity to method-specific estimation\"" in html
    assert "<b>What this shows:</b>" in html
    assert "<b>How to interpret it:</b>" in html

    forbidden = [
        "consult V2",
        "consult V3",
        "Source V3 runtime",
        "V3 did not already perform IDW; V4 augments V3",
    ]
    for phrase in forbidden:
        assert phrase not in html

    assert "Print Report" in html
    assert "@media print" in html
    assert "@page { size: A4 portrait; margin: 10mm; }" in html
    assert ".diagram svg" in html
    assert "overflow-wrap: anywhere;" in html
    assert "http://" not in html
    assert "https://" not in html
    assert "cdn" not in html.lower()


def test_v4_report_writes_sidecar_manifest_and_validates_integrity(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    out = tmp_path / "outputs_sampling_v4" / "runs" / "fixture" / DEFAULT_OUTPUT_NAME
    written = write_v4_html_report(output_path=out, inputs=inputs)
    assert written == out
    assert out.exists()

    payload = json.loads(out.with_name("report_manifest.json").read_text(encoding="utf-8"))
    assert payload["version"] == "sampling-v4-report-manifest-v1"
    assert payload["report_filename"] == out.name
    assert len(payload["report_sha256"]) == 64
    assert "source_input_hashes" in payload
    assert "source_v3_manifest" in payload["source_input_hashes"]
    assert "source_v3_quadrant" in payload["source_input_hashes"]
    assert "source_v3_throughput" in payload["source_input_hashes"]
    validate_report_manifest(report_path=out, manifest_path=out.with_name("report_manifest.json"))


def test_v4_summary_report_contains_expanded_requested_sections(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    out = tmp_path / "summary.html"
    written = write_v4_html_report(
        output_path=out,
        inputs=inputs,
        section_ids=(
            "executive",
            "outcomes",
            "methods",
            "quadrant",
            "throughput",
            "embedding",
            "lineage",
            "repro",
        ),
        report_title="Agent365 Sampling V4 Summary Report",
        manifest_name="summary_report_manifest.json",
    )

    html = written.read_text(encoding="utf-8")
    assert html.count('class="tab-button"') == 8
    assert "Executive Summary" in html
    assert "Outcomes" in html
    assert "Sampling Methods" in html
    assert "Quadrant Behavior" in html
    assert "Throughput" in html
    assert "Embedding Diagnostics" in html
    assert "Lineage &amp; Integrity" in html
    assert "Reproducibility &amp; Caveats" in html
    validate_report_manifest(
        report_path=written,
        manifest_path=written.with_name("summary_report_manifest.json"),
    )


def test_v4_report_rejects_tampering_and_source_linkage_mismatch(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)

    runs_path = inputs.runs_jsonl
    runs_path.write_text(runs_path.read_text(encoding="utf-8") + _canonical({"x": 1}) + "\n", encoding="utf-8")
    artifacts = load_v4_artifacts(inputs)
    with pytest.raises(ValueError, match="manifest hash mismatch for runs_jsonl"):
        validate_v4_artifacts(artifacts)

    inputs = _build_fixture_bundle(tmp_path)
    src_lineage = json.loads(inputs.source_lineage.read_text(encoding="utf-8"))
    src_lineage["source_manifest"]["sha256"] = "f" * 64
    inputs.source_lineage.write_text(_canonical(src_lineage) + "\n", encoding="utf-8")

    manifest = json.loads(inputs.manifest.read_text(encoding="utf-8"))
    manifest["artifacts"]["source_lineage"]["sha256"] = _sha(inputs.source_lineage)
    manifest["artifacts"]["source_lineage"]["bytes"] = int(inputs.source_lineage.stat().st_size)
    _write_json(inputs.manifest, manifest)

    artifacts = load_v4_artifacts(inputs)
    with pytest.raises(ValueError, match="source_lineage source manifest hash mismatch"):
        validate_v4_artifacts(artifacts)


def test_v4_report_rejects_tampered_source_v3_secondary_artifact(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    source_aggregate = inputs.manifest.parent / "source_v3" / "aggregate.json"
    source_aggregate.write_text(source_aggregate.read_text(encoding="utf-8") + " ", encoding="utf-8")

    artifacts = load_v4_artifacts(inputs)
    with pytest.raises(ValueError, match="manifest hash mismatch for source_v3.aggregate"):
        validate_v4_artifacts(artifacts)


def test_v4_report_rejects_tampered_source_v3_quadrant_throughput(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    source_quadrant = inputs.manifest.parent / "source_v3" / "quadrant.json"
    source_quadrant.write_text(source_quadrant.read_text(encoding="utf-8") + " ", encoding="utf-8")

    artifacts = load_v4_artifacts(inputs)
    with pytest.raises(ValueError, match="manifest hash mismatch for source_v3.quadrant"):
        validate_v4_artifacts(artifacts)

    inputs = _build_fixture_bundle(tmp_path)
    source_throughput = inputs.manifest.parent / "source_v3" / "throughput.json"
    source_throughput.write_text(source_throughput.read_text(encoding="utf-8") + " ", encoding="utf-8")

    artifacts = load_v4_artifacts(inputs)
    with pytest.raises(ValueError, match="manifest hash mismatch for source_v3.throughput"):
        validate_v4_artifacts(artifacts)


def test_v4_report_rejects_invalid_method_mode_combinations(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    rows = [
        json.loads(line)
        for line in inputs.runs_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if row["method"] == "random_sampling_token_priority":
            row["estimation_mode"] = "model_assisted_idw"
            break
    _write_jsonl(inputs.runs_jsonl, rows)

    manifest = json.loads(inputs.manifest.read_text(encoding="utf-8"))
    manifest["artifacts"]["runs_jsonl"]["sha256"] = _sha(inputs.runs_jsonl)
    manifest["artifacts"]["runs_jsonl"]["bytes"] = int(inputs.runs_jsonl.stat().st_size)
    _write_json(inputs.manifest, manifest)

    artifacts = load_v4_artifacts(inputs)
    with pytest.raises(ValueError, match="random/minhash must be selected_only"):
        validate_v4_artifacts(artifacts)
