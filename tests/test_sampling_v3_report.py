from __future__ import annotations

import json
from pathlib import Path

import pytest

from sampling_comparison.v3_report import (
    DEFAULT_OUTPUT_NAME,
    V3ReportInputs,
    validate_report_manifest,
    load_v3_artifacts,
    render_v3_html_report,
    validate_v3_artifacts,
    write_v3_html_report,
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
    text = "\n".join(_canonical(row) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def _build_fixture_bundle(tmp_path: Path, *, skip_quadrant: bool = False, skip_throughput: bool = False) -> V3ReportInputs:
    out = tmp_path / "outputs_sampling_v3" / "runs" / "fixture"
    out.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "version": "sampling-v3-bundle-v1",
        "generated_at": "2026-08-04T01:02:03Z",
        "population_count": 120,
        "runtime_seconds": 12.5,
        "config": {
            "search": {
                "schema_validation": {
                    "index": "v3-index",
                    "key_field": "cluster_id",
                }
            }
        },
        "outcome": {
            "version": "sampling-v3-outcome-v1",
            "eligible_token_mass": 120000,
            "aggregate": [
                {
                    "method": "random_sampling_token_priority",
                    "budget_tokens": 6000,
                    "replays": 2,
                    "mae": {"mean": 0.031, "empirical_low": 0.029, "empirical_high": 0.033},
                    "concept_coverage": {"mean": 0.41, "empirical_low": 0.39, "empirical_high": 0.43},
                    "fraction_saved": {"mean": 0.83, "empirical_low": 0.82, "empirical_high": 0.84},
                    "token_utilization": {"mean": 0.97, "empirical_low": 0.96, "empirical_high": 0.98},
                    "native_count": {"mean": 12.0, "empirical_low": 11.0, "empirical_high": 13.0},
                    "fill_count": {"mean": 0.0, "empirical_low": 0.0, "empirical_high": 0.0},
                },
                {
                    "method": "adaptive_embedding_fullsession_token",
                    "budget_tokens": 6000,
                    "replays": 2,
                    "mae": {"mean": 0.022, "empirical_low": 0.021, "empirical_high": 0.023},
                    "concept_coverage": {"mean": 0.55, "empirical_low": 0.54, "empirical_high": 0.56},
                    "fraction_saved": {"mean": 0.82, "empirical_low": 0.81, "empirical_high": 0.83},
                    "token_utilization": {"mean": 0.99, "empirical_low": 0.98, "empirical_high": 1.0},
                    "native_count": {"mean": 9.5, "empirical_low": 9.0, "empirical_high": 10.0},
                    "fill_count": {"mean": 2.5, "empirical_low": 2.0, "empirical_high": 3.0},
                },
            ],
        },
        "quadrant": None if skip_quadrant else {
            "version": "sampling-v3-quadrant-v1",
            "aggregate_groups": [
                {
                    "method": "adaptive_embedding_fullsession_token",
                    "quadrant": "high_variety_high_velocity",
                    "budget_tokens": 4500,
                    "representation_mean": 0.66,
                    "budget_utilization_tokens_mean": 0.95,
                    "zero_selection_agent_rate_mean": 0.08,
                    "mae_mean": 0.024,
                }
            ],
        },
        "throughput": None if skip_throughput else {
            "version": "sampling-v3-throughput-v1",
            "aggregate_grid": [
                {
                    "method": "adaptive_embedding_fullsession_token",
                    "arrival_rate_sessions_per_second": 1.0,
                    "eval_capacity_sessions_per_second": 1.0,
                    "budget_tokens": 4500,
                    "queue_admitted_tokens_mean": 4200.0,
                    "queue_max_tokens_mean": 4490.0,
                    "token_pressure_ratio_mean": 1.15,
                    "budget_utilization_tokens_mean": 0.93,
                }
            ],
            "eval_tokens_per_second_map": {"1.0": 188.0},
        },
        "provenance": {
            "code_hashes": {
                "sampling_comparison/v3_outputs.py": "a" * 64,
                "sampling_comparison/v3_experiment.py": "b" * 64,
                "scripts/run_sampling_v3.py": "c" * 64,
            },
            "source_hashes": {},
        },
    }

    runs_jsonl = [
        {
            "method": "adaptive_embedding_fullsession_token",
            "legacy_tier_pct": 20,
            "budget_tokens": 6000,
            "selected_tokens": 5970,
            "selected_count": 12,
            "absolute_error": 0.02,
            "concept_coverage": 0.54,
            "fraction_saved": 0.82,
            "budget_utilization_tokens": 0.995,
            "native_count": 9,
            "fill_count": 3,
        }
    ]

    quadrant = (
        {"version": "sampling-v3-quadrant-v1", "skipped": True}
        if skip_quadrant
        else {
            "version": "sampling-v3-quadrant-v1",
            "aggregate_groups": [
                {
                    "method": "adaptive_embedding_fullsession_token",
                    "quadrant": "high_variety_high_velocity",
                    "budget_tokens": 4500,
                    "representation_mean": 0.66,
                    "budget_utilization_tokens_mean": 0.95,
                    "zero_selection_agent_rate_mean": 0.08,
                    "mae_mean": 0.024,
                }
            ],
        }
    )

    throughput = (
        {"version": "sampling-v3-throughput-v1", "skipped": True}
        if skip_throughput
        else {
            "version": "sampling-v3-throughput-v1",
            "config": {
                "eval_tokens_per_second_map": {"1.0": 188.0, "4.0": 752.0},
            },
            "aggregate_grid": [
                {
                    "method": "adaptive_embedding_fullsession_token",
                    "arrival_rate_sessions_per_second": 1.0,
                    "eval_capacity_sessions_per_second": 1.0,
                    "budget_tokens": 4500,
                    "queue_admitted_tokens_mean": 4200.0,
                    "queue_max_tokens_mean": 4490.0,
                    "token_pressure_ratio_mean": 1.15,
                    "budget_utilization_tokens_mean": 0.93,
                }
            ],
        }
    )

    corpus_audit = {
        "version": "sampling-v3-corpus-audit-v1",
        "source_files": {
            "historical_300": {
                "path": "h.json",
                "sha256": "1" * 64,
                "counts": {"units": 20, "labels": 20, "agents": 8, "concepts": 10},
                "label_pass_rate": 0.65,
            },
            "dense_2500": {
                "path": "d.json",
                "sha256": "2" * 64,
                "counts": {"units": 100, "labels": 100, "agents": 12, "concepts": 24},
                "label_pass_rate": 0.58,
            },
        },
        "combined": {
            "counts": {"units": 120, "labels": 120, "agents": 20, "concepts": 34},
            "label_pass_rate": 0.59,
        },
    }

    token_inventory = [
        {
            "unit_id": "u-1",
            "content_sha256": "3" * 64,
            "original_tokens": 240,
            "emitted_tokens": 220,
            "truncated": False,
        },
        {
            "unit_id": "u-2",
            "content_sha256": "4" * 64,
            "original_tokens": 9000,
            "emitted_tokens": 8191,
            "truncated": True,
        },
    ]

    budget_manifest = {
        "outcome": {"eligible_token_mass": 120000, "legacy_outcome_tiers_pct": [5, 10, 20, 30, 50]},
        "quadrant": {"legacy_quadrant_tiers_pct": [15, 30]},
        "throughput": {
            "arrival_rates_sessions_per_second": [0.25, 1.0],
            "capacity_rates_sessions_per_second": [0.25, 1.0],
        },
    }

    embedding_ledger = {
        "packet_builds": 120,
        "packet_cache_hits": 0,
        "embedding_calls": 4,
        "embedding_inputs": 120,
        "embedding_input_tokens": 40000,
        "embedding_latency_seconds": 2.25,
        "embedding_content_hash_count": 120,
        "embedding_model_id": "text-embedding-3-small",
        "embedding_deployment_id": "embed-dep",
        "embedding_embedder_class": "AzureOpenAIEmbedder",
    }

    selected_membership = {
        "version": "sampling-v3-selected-membership-v1",
        "legacy_tier_pct_provenance": 20,
        "budget_tokens": 24000,
        "membership": {
            "method": "adaptive_embedding_fullsession_token",
            "budget_utilization_tokens": 0.992,
            "selected_count": 24,
        },
    }

    methodology = """
# V3 Methodology Delta
- token_profile_id: token-profile-v3
- minhash_profile_id: v3-token-minhash-v1
- embedding_profile_id: embedding-profile-v3
- embedding_semantic_scope: full-session-packet-v3
- A 4096-entry exact recent-leader buffer resolves many decisions before HNSW lookup and shields index lag effects.
- Packet cap binding check from runtime inventory: 1/2 packets truncated; max emitted tokens 8191; cap is binding.
""".strip() + "\n"

    files = {
        "aggregate": out / "aggregate.json",
        "runs_jsonl": out / "runs.jsonl",
        "quadrant": out / "quadrant.json",
        "throughput": out / "throughput.json",
        "corpus_audit": out / "corpus_audit.json",
        "token_inventory": out / "token_inventory.jsonl",
        "budget_manifest": out / "budget_manifest.json",
        "embedding_ledger": out / "embedding_ledger.json",
        "selected_membership": out / "selected_membership.json",
        "methodology_delta": out / "methodology_delta.md",
    }

    _write_json(files["aggregate"], aggregate)
    _write_jsonl(files["runs_jsonl"], runs_jsonl)
    _write_json(files["quadrant"], quadrant)
    _write_json(files["throughput"], throughput)
    _write_json(files["corpus_audit"], corpus_audit)
    _write_jsonl(files["token_inventory"], token_inventory)
    _write_json(files["budget_manifest"], budget_manifest)
    _write_json(files["embedding_ledger"], embedding_ledger)
    _write_json(files["selected_membership"], selected_membership)
    files["methodology_delta"].write_text(methodology, encoding="utf-8")

    manifest = {
        "version": "sampling-v3-manifest-v1",
        "generated_at": "2026-08-04T01:02:03Z",
        "artifacts": {},
        "notes": [
            "No raw packet text or embedding vectors are persisted.",
            "V3 bundle intentionally omits V2 ExternalEvalSnapshot artifacts.",
            "Legacy percent tiers are provenance only; exact token budgets are primary axes.",
        ],
    }
    for key, path in files.items():
        manifest["artifacts"][key] = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha(path),
        }

    run_source_manifest = {
        "version": "sampling-v3-run-source-manifest-v1",
        "captured_at": "2026-08-04T01:02:10Z",
        "branch": "stangoodwin/sampling-experiment-v3",
        "note": "fixture",
        "source_hashes": {
            "sampling_comparison/v3_outputs.py": "a" * 64,
            "sampling_comparison/v3_report.py": "b" * 64,
        },
    }
    _write_json(out / "run_source_manifest.json", run_source_manifest)
    manifest["artifacts"]["run_source_manifest"] = {
        "path": str(out / "run_source_manifest.json"),
        "bytes": int((out / "run_source_manifest.json").stat().st_size),
        "sha256": _sha(out / "run_source_manifest.json"),
    }

    search_cleanup_audit = {
        "version": "sampling-v3-search-cleanup-audit-v1",
        "tenant_id": "sampling-v3-experiment",
        "checked_at": "2026-08-04T01:02:11Z",
        "remaining_count": 0,
        "scopes": {
            "tenant_id": "sampling-v3-experiment",
        },
        "allow_nonzero": False,
    }
    _write_json(out / "search_cleanup_audit.json", search_cleanup_audit)
    manifest["artifacts"]["search_cleanup_audit"] = {
        "path": str(out / "search_cleanup_audit.json"),
        "bytes": int((out / "search_cleanup_audit.json").stat().st_size),
        "sha256": _sha(out / "search_cleanup_audit.json"),
    }

    _write_json(out / "manifest.json", manifest)

    return V3ReportInputs(
        aggregate=files["aggregate"],
        runs_jsonl=files["runs_jsonl"],
        quadrant=files["quadrant"],
        throughput=files["throughput"],
        corpus_audit=files["corpus_audit"],
        token_inventory=files["token_inventory"],
        budget_manifest=files["budget_manifest"],
        embedding_ledger=files["embedding_ledger"],
        selected_membership=files["selected_membership"],
        methodology_delta=files["methodology_delta"],
        manifest=out / "manifest.json",
        run_source_manifest=out / "run_source_manifest.json",
        search_cleanup_audit=out / "search_cleanup_audit.json",
    )


def test_v3_report_renders_tabs_and_required_caveats(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    artifacts = load_v3_artifacts(inputs)
    html = render_v3_html_report(artifacts)

    assert html.count('class="tab-button"') >= 9
    assert "Overview" in html
    assert "What Changed from V2" in html
    assert "Methods" in html
    assert "Outcomes" in html
    assert "Token and Embedding" in html
    assert "Quadrant" in html
    assert "Throughput Queue" in html
    assert "Reproducibility and Storage" in html
    assert "Caveats and Conclusions" in html

    assert "Token-mass percentages are provenance only" in html
    assert "does not use Cochran sample sizing or finite-population correction" in html
    assert "No Cochran sizing or finite-population correction is applied" in html
    assert "Random method results are descriptive" in html
    assert "Adaptive rates are diagnostic mechanism outputs" in html
    assert "Tau 0.55 is an uncalibrated assumption" in html
    assert "Live Azure Search HNSW behavior and latency are environment-specific" in html
    assert "4096-entry exact recent-leader buffer" in html
    assert "Labels are joined only post-membership" in html
    assert "packet-cap check from token inventory" in html.lower()
    assert "No V2 external snapshot artifacts are present" in html
    assert "no universal best row is declared across budgets" in html
    assert "Lowest observed MAE per exact budget" in html
    assert "descriptive within-budget only" in html
    assert "Optional post-run artifacts" in html

    assert 'role="tablist"' in html
    assert 'role="tab"' in html
    assert "ArrowLeft" in html
    assert "ArrowRight" in html
    assert "Home" in html
    assert "End" in html
    assert "chart-scroll" in html
    assert "@media (max-width: 980px)" in html
    assert "@media (max-width: 680px)" in html

    assert "https://" not in html
    assert "http://" not in html
    assert "cdn" not in html.lower()


def test_v3_report_handles_skipped_planes(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path, skip_quadrant=True, skip_throughput=True)
    artifacts = load_v3_artifacts(inputs)
    html = render_v3_html_report(artifacts)

    assert "Skipped in this run. Quadrant artifact exists and hash validation passed." in html
    assert "Skipped in this run. Throughput artifact exists and hash validation passed." in html


def test_v3_report_manifest_tamper_detection(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    runs_path = inputs.runs_jsonl
    runs_path.write_text(runs_path.read_text(encoding="utf-8") + _canonical({"x": 1}) + "\n", encoding="utf-8")

    artifacts = load_v3_artifacts(inputs)
    with pytest.raises(ValueError, match="manifest hash mismatch for runs_jsonl"):
        validate_v3_artifacts(artifacts)


def test_v3_report_write_output(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    output = tmp_path / "outputs_sampling_v3" / "runs" / "fixture" / DEFAULT_OUTPUT_NAME
    written = write_v3_html_report(output_path=output, inputs=inputs)
    assert written == output
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "Agent365 Sampling V3 Operational Report" in text
    assert "Self-contained output with inline CSS, JS, and SVG only." in text

    report_manifest = output.with_name("report_manifest.json")
    assert report_manifest.exists()
    payload = json.loads(report_manifest.read_text(encoding="utf-8"))
    assert payload["aggregate_generated_at"] == "2026-08-04T01:02:03Z"
    assert payload["report_filename"] == output.name
    assert len(payload["report_sha256"]) == 64
    assert len(payload["bundle_manifest_sha256"]) == 64
    assert len(payload["report_generator_source_sha256"]) == 64
    validate_report_manifest(report_path=output, manifest_path=report_manifest)


def test_v3_report_manifest_tamper_detection_for_report_sidecar(tmp_path: Path) -> None:
    inputs = _build_fixture_bundle(tmp_path)
    output = tmp_path / "outputs_sampling_v3" / "runs" / "fixture" / DEFAULT_OUTPUT_NAME
    write_v3_html_report(output_path=output, inputs=inputs)

    report_manifest = output.with_name("report_manifest.json")
    payload = json.loads(report_manifest.read_text(encoding="utf-8"))
    payload["report_sha256"] = "f" * 64
    report_manifest.write_text(_canonical(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="report manifest sha256 mismatch"):
        validate_report_manifest(report_path=output, manifest_path=report_manifest)


def test_v3_report_regeneration_is_stable_with_same_inputs(tmp_path: Path) -> None:
    import hashlib

    inputs = _build_fixture_bundle(tmp_path)
    output = tmp_path / "outputs_sampling_v3" / "runs" / "fixture" / DEFAULT_OUTPUT_NAME

    first = write_v3_html_report(output_path=output, inputs=inputs)
    first_manifest = first.with_name("report_manifest.json")
    first_html_sha = hashlib.sha256(first.read_bytes()).hexdigest()
    first_manifest_sha = hashlib.sha256(first_manifest.read_bytes()).hexdigest()

    second = write_v3_html_report(output_path=output, inputs=inputs)
    second_manifest = second.with_name("report_manifest.json")
    second_html_sha = hashlib.sha256(second.read_bytes()).hexdigest()
    second_manifest_sha = hashlib.sha256(second_manifest.read_bytes()).hexdigest()

    assert first_html_sha == second_html_sha
    assert first_manifest_sha == second_manifest_sha
