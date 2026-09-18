from __future__ import annotations

import json
import math
import re
from pathlib import Path

from sampling_comparison.matryoshka_report import build_report
from scripts.build_matryoshka_report import main


def test_low_dimension_protocol_is_used_in_all_detailed_report_surfaces():
    data = _build_fixture()
    dimensions = [1536, *range(32, 1, -2)]
    data["protocol"]["dimensions"] = dimensions
    data["rows"] = [
        {**row, "dimension": dimension, "mae": 0.2 + dimension / 10000}
        for row in data["rows"] if row["dimension"] == 1536
        for dimension in dimensions
    ]
    data["summary"] = []
    html = build_report(data, "low/aggregate.json")
    assert "left is 2d, right is native 1536d" in html
    assert "2d point is shown explicitly" in html
    assert "Accuracy by dimension (1536→2)" in html
    assert all(f"<th>{d}d</th>" in html for d in dimensions)
    assert "<th>512d</th>" not in html
    assert "Takeaway: 2d differs from native" in html


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _build_fixture() -> dict:
    dimensions = [1536, 512, 256, 128, 64, 32, 16, 8]
    modes = ["end_to_end", "fixed_membership"]
    schedules = ["evenly_spaced", "uniformly_random"]
    rates = [0.01, 0.10]
    seeds = [13, 14]
    rows = []

    for mode in modes:
        base = 0.78 if mode == "end_to_end" else 0.80
        for dim in dimensions:
            dim_penalty = max(0.0, math.log2(1536 / dim)) * 0.008
            for seed in seeds:
                for schedule in schedules:
                    schedule_penalty = 0.002 if schedule == "uniformly_random" else 0.0
                    for rate in rates:
                        rate_penalty = 0.001 if rate == 0.10 else 0.0
                        native_reference = base - schedule_penalty - rate_penalty
                        accuracy = native_reference - dim_penalty
                        unjudged = 300 - max(1, int(math.floor(300 * rate)))
                        positive_unjudged = int(round(unjudged * 0.68))
                        tp = int(round(positive_unjudged * 0.7))
                        fn = max(0, positive_unjudged - tp)
                        rows.append(
                            {
                                "dataset_id": "historical_300",
                                "mode": mode,
                                "dimension": dim,
                                "seed": seed,
                                "schedule": schedule,
                                "rate": rate,
                                "selected_count": max(1, int(math.floor(300 * rate))),
                                "unjudged_count": unjudged,
                                "accuracy": accuracy,
                                "mae": 0.21 + dim_penalty,
                                "f1": max(0.0, accuracy - 0.03),
                                "brier": 0.16 + (dim_penalty * 0.4),
                                "tp": tp,
                                "fn": fn,
                                "macro_agent_accuracy": max(0.0, accuracy - 0.025),
                                "combined_accuracy": min(1.0, accuracy + 0.11),
                                "aggregate_rate_error": 0.05 + dim_penalty,
                                "concept_coverage": max(0.0, 0.74 - dim_penalty * 1.4),
                                "agent_coverage": 0.88,
                                "provenance_counts": {
                                    "idw": 80,
                                    "exact_match": 7,
                                    "global_mean": 4 + int(dim_penalty * 100),
                                    "prior": 3 + int(dim_penalty * 100),
                                },
                                "membership_jaccard_native": 0.99 if mode == "fixed_membership" else 0.92 - dim_penalty,
                                "accuracy_delta_native": -dim_penalty,
                                "per_agent": {
                                    "agent_alpha": {
                                        "n": 12,
                                        "selected_count": 2,
                                        "unjudged_count": 10,
                                        "accuracy": max(0.0, accuracy - 0.04),
                                        "mae": 0.24 + dim_penalty,
                                    },
                                    "agent_beta": {
                                        "n": 20,
                                        "selected_count": 3,
                                        "unjudged_count": 17,
                                        "accuracy": min(1.0, accuracy + 0.02),
                                        "mae": 0.20 + dim_penalty,
                                    },
                                },
                            }
                        )

    for mode in modes:
        for dim in [1536, 512, 256]:
            rows.append(
                {
                    "dataset_id": "dense_2500",
                    "mode": mode,
                    "dimension": dim,
                    "seed": 13,
                    "schedule": "evenly_spaced",
                    "rate": 0.20,
                    "selected_count": max(1, int(math.floor(2500 * 0.20))),
                    "unjudged_count": 2000,
                    "accuracy": 0.81 - max(0.0, math.log2(1536 / dim)) * 0.005,
                    "mae": 0.17 + max(0.0, math.log2(1536 / dim)) * 0.004,
                    "f1": 0.79 - max(0.0, math.log2(1536 / dim)) * 0.004,
                    "brier": 0.12 + max(0.0, math.log2(1536 / dim)) * 0.003,
                    "tp": 1000,
                    "fn": 444,
                    "macro_agent_accuracy": 0.78 - max(0.0, math.log2(1536 / dim)) * 0.004,
                    "combined_accuracy": 0.89 - max(0.0, math.log2(1536 / dim)) * 0.003,
                    "aggregate_rate_error": 0.04 + max(0.0, math.log2(1536 / dim)) * 0.002,
                    "concept_coverage": 0.69 - max(0.0, math.log2(1536 / dim)) * 0.01,
                    "agent_coverage": 0.92,
                    "provenance_counts": {"idw": 120, "exact_match": 10, "global_mean": 8, "prior": 6},
                    "accuracy_delta_native": -max(0.0, math.log2(1536 / dim)) * 0.005,
                    "per_agent": {},
                }
            )

    grouped = {}
    for row in rows:
        key = (row["dataset_id"], row["mode"], row["dimension"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (dataset_id, mode, dimension), values in sorted(grouped.items()):
        ci = None if dataset_id == "dense_2500" else [-0.02, 0.02]
        summary.append(
            {
                "dataset_id": dataset_id,
                "mode": mode,
                "dimension": dimension,
                "accuracy": _mean([item["accuracy"] for item in values]),
                "mae": _mean([item["mae"] for item in values]),
                "f1": _mean([item["f1"] for item in values]),
                "brier": _mean([item["brier"] for item in values]),
                "macro_agent_accuracy": _mean([item["macro_agent_accuracy"] for item in values]),
                "combined_accuracy": _mean([item["combined_accuracy"] for item in values]),
                "aggregate_rate_error": _mean([item["aggregate_rate_error"] for item in values]),
                "concept_coverage": _mean([item["concept_coverage"] for item in values]),
                "agent_coverage": _mean([item["agent_coverage"] for item in values]),
                "accuracy_delta_native": _mean([item["accuracy_delta_native"] for item in values]),
                "accuracy_delta_seed_ci95": ci,
            }
        )

    return {
        "version": "matryoshka-cutoff-v1",
        "run_id": "matryoshka-demo",
        "generated_at": "2026-09-14T10:38:32Z",
        "status": "partial",
        "source_revision": "abc1234",
        "protocol": {
            "dimensions": [1536, 512, 256, 128, 64, 32, 16, 8],
            "seeds": list(range(13, 43)),
            "rates": [0.01, 0.02, 0.05, 0.10, 0.20],
            "schedules": ["evenly_spaced", "uniformly_random", "bursty", "front_loaded", "agent_blocked"],
            "modes": ["end_to_end", "fixed_membership"],
            "idw": {"k": 8, "power": 2, "eps": 1e-6},
            "tau": 0.55,
            "accuracy_tolerance": 0.01,
        },
        "datasets": [
            {
                "dataset_id": "historical_300",
                "status": "completed",
                "n": 300,
                "agents": 48,
                "positive_count": 189,
                "pass_rate": 0.63,
                "provenance": {
                    "source": "fixture",
                    "input_manifest": "C:/fixture/historical/manifest.json",
                    "embedding_preparation": {"embedding_api_calls": 300, "provider": "azure"},
                    "notes": ["paired labels"],
                },
                "source_hashes": {"labels": "hash-historical"},
            },
            {
                "dataset_id": "dense_2500",
                "status": "completed",
                "n": 2500,
                "agents": 220,
                "positive_count": 1680,
                "pass_rate": 0.672,
                "provenance": {"source": "fixture", "input_manifest": "C:/fixture/dense/manifest.json", "notes": ["dense cohort"]},
                "source_hashes": {"labels": "hash-dense"},
            },
            {
                "dataset_id": "cosmos_otel",
                "status": "blocked",
                "reason": "access pending",
                "n": 0,
                "agents": 0,
                "positive_count": 0,
                "pass_rate": None,
                "provenance": {"source": "fixture"},
                "source_hashes": {"labels": "hash-cosmos"},
            },
        ],
        "rows": rows,
        "summary": summary,
        "agent_summary": [
            {
                "dataset_id": "historical_300",
                "mode": "end_to_end",
                "dimension": 1536,
                "agent_id": "agent_gamma",
                "accuracy": 0.71,
                "mae": 0.24,
                "n": 30.25,
                "selected_count": 3.5,
                "unjudged_count": 26.75,
                "provenance_counts": {"idw": 19.2, "global_mean": 5.1, "exact_match": 2.7},
                "cell_count": 20,
                "cells_with_unjudged_targets": 20,
            },
            {
                "dataset_id": "historical_300",
                "mode": "fixed_membership",
                "dimension": 1536,
                "agent_id": "agent_gamma",
                "accuracy": 0.73,
                "mae": 0.23,
                "n": 30.25,
                "selected_count": 3.5,
                "unjudged_count": 26.75,
                "provenance_counts": {"idw": 18.4, "global_mean": 5.8, "exact_match": 2.5},
                "cell_count": 20,
                "cells_with_unjudged_targets": 20,
            },
        ],
        "decisions": [
            {
                "dataset_id": "historical_300",
                "mode": "end_to_end",
                "smallest_non_degrading_dimension": 256,
                "one_pp_candidate_dimension": 128,
                "reason": "Within 1pp tolerance in paired replay.",
            }
        ],
        "files": {"aggregate_json": "C:/tmp/aggregate.json"},
        "validation": {"scientific_checks": "parent-owned"},
    }


def test_build_report_renders_required_datasets_and_curves() -> None:
    html = build_report(_build_fixture(), "C:/tmp/aggregate.json")
    for dataset_id in ("historical_300", "dense_2500", "cosmos_otel", "tau2_bench"):
        assert dataset_id in html
    for dim in ("8", "16", "32", "64", "128", "256", "512", "1536"):
        assert f">{dim}<" in html
    assert "Missing from aggregate datasets list" in html
    assert "Unjudged-only accuracy (fraction)" in html
    assert "unjudged-only MAE by dimension" in html
    assert "MAE difference (fraction)" in html
    assert "MAE (lower is better):" in html
    assert "before thresholding" in html
    assert "Existing cutoff-candidate tables use accuracy, not an agreed MAE tolerance" in html
    assert "Rate-specific paired accuracy delta (pp) vs native" in html
    assert "N/A (single seed or unavailable)" in html
    assert "provenance</code> is taken from input-manifest profile metadata" in html
    assert "Embedding preparation vs offline sweep boundary" in html
    assert "embedding_preparation.embedding_api_calls=300" in html
    assert "Do not infer zero API usage for the run as a whole" in html
    assert "Curve shape and cutoff interpretation" in html
    assert "Class-imbalance reference baseline" in html
    assert "Do not claim absolute IDW classification utility from compression parity alone." in html
    assert "Always-pass baseline" in html
    assert "72.200000%" in html
    assert "Qualified takeaway:" in html
    assert html.index("Qualified takeaway:") < html.index("<section id=\"source-status\">")
    assert "grid-template-columns:minmax(0,1fr)" in html
    assert ".small-multiples>*{min-width:0;}" in html
    assert "figure{margin:0;padding:.5rem;border:1px solid var(--line);border-radius:8px;background:#fff;min-width:0;max-width:100%;}" in html
    assert "chart-scroll{max-width:100%;width:100%;min-width:0;overflow-x:auto;overflow-y:hidden;}" in html
    assert ".chart{width:100%;min-width:720px;max-width:960px;height:auto;display:block;}" in html
    assert "figcaption{font-size:.88rem;color:var(--muted);margin-top:.45rem;white-space:normal;overflow-wrap:anywhere;}" in html
    assert '<div class="chart-scroll">' in html
    assert '<svg class="chart" role="img" aria-label="' in html
    assert "-0.120" not in html and "1.120" not in html
    assert html.count("<svg") >= 8
    assert "table-wrap" in html


def test_missing_data_not_zero_and_no_false_winner_claim() -> None:
    html = build_report(_build_fixture(), "C:/tmp/aggregate.json")
    row_match = re.search(r"<tr data-dataset=\"tau2_bench\">.*?</tr>", html, re.S)
    assert row_match is not None
    tau_row = row_match.group(0)
    assert "Unavailable" in tau_row
    assert ">0.00<" not in tau_row
    assert "No universal winner is claimed" in html
    assert "all larger tested prefixes to pass" in html
    assert "Expected production output is all four datasets" in html


def test_report_escapes_xss_and_is_static_standalone() -> None:
    payload = _build_fixture()
    payload["run_id"] = "<script>alert(1)</script>"
    payload["datasets"][0]["reason"] = "<img src=x onerror=alert(1)>"
    html = build_report(payload, "C:/tmp/aggregate.json")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<script src=" not in html.lower()
    assert "cdn.jsdelivr" not in html.lower()
    assert "src=\"http" not in html.lower()


def test_agent_summary_semantics_and_unjudged_accuracy_wording() -> None:
    html = build_report(_build_fixture(), "C:/tmp/aggregate.json")
    assert "agent_gamma" in html
    assert "Accuracy (unjudged)" in html
    assert "N total" in html
    assert "n</code> is total agent sessions; <code>unjudged</code> is the denominator" in html
    assert "3.50" in html
    assert "26.75" in html
    assert "Mean provenance mix" in html


def test_input_readiness_auth_blocker_section_reads_sibling_file(tmp_path: Path) -> None:
    payload = _build_fixture()
    payload["files"] = {
        "aggregate": str(tmp_path / "aggregate.json"),
        "input_readiness": "input_readiness.json",
    }
    readiness = {
        "authentication_blocker": (
            "Embedding API calls authorized, but Azure CLI uses a lab-tenant account. "
            "Corporate-tenant token acquisition fails with AADSTS50020."
        ),
        "boundary": "Live embedding preparation is authorized; sweep is offline and uses oracle labels without LLM judges.",
        "datasets": [
            {
                "dataset_id": "historical_300",
                "source": "outputs_matryoshka/cache/historical_300/readiness.json",
                "preparation": {
                    "status": "prepared_without_embeddings",
                    "reason": "AADSTS50020 wrong-tenant authentication",
                    "sessions": 300,
                    "agents": 100,
                    "positive_count": 188,
                    "truncated_sessions": 0,
                    "label_source": "Retained synthetic expected labels; no new judges.",
                },
            },
            {
                "dataset_id": "dense_2500",
                "source": "outputs_matryoshka/cache/dense_2500/manifest.json",
                "preparation": {"status": "ready_with_cached_embeddings", "sessions": 2500},
            },
            {
                "dataset_id": "cosmos_otel",
                "source": "outputs_matryoshka/cache/cosmos_otel/readiness.json",
                "preparation": {
                    "status": "prepared_without_embeddings",
                    "sessions": 205,
                    "agents": 7,
                    "positive_count": 69,
                    "representation_sources": {"linked_raw_spans": 195, "synthetic_label_document": 10},
                    "truncated_sessions": 186,
                    "label_source": "Expected outcome mapping only; no fresh LLM judgments.",
                },
            },
            {
                "dataset_id": "tau2_bench",
                "source": "outputs_matryoshka/cache/tau2_bench/readiness.json",
                "preparation": {
                    "status": "blocked_expected_labels",
                    "reason": "No verified explicit expected labels and auth still blocked.",
                    "sessions": 388,
                    "agents": 1,
                    "label_source": "reward_info.reward is observed benchmark result, not explicit expected label.",
                },
            },
        ],
    }
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "input_readiness.json").write_text(json.dumps(readiness), encoding="utf-8")
    html = build_report(payload, str(aggregate_path))
    assert "Input readiness and blockers" in html
    assert "AADSTS50020" in html
    assert "operational authentication blocker" in html
    assert "not a policy refusal" in html
    assert "prepared_without_embeddings" in html
    assert "ready_with_cached_embeddings" in html
    assert "historical_300" in html and "300" in html and "188" in html
    assert "cosmos_otel" in html and "205" in html and "195" in html and "10" in html and "186" in html
    assert "tau2_bench uses observed benchmark rewards (simulations[].reward_info.reward)" in html
    assert "blocked on both auth and expected-label mapping" in html


def test_nonmonotonic_curve_callout_and_8d_visibility() -> None:
    payload = _build_fixture()
    dense_rows = []
    for row in payload["summary"]:
        if row["dataset_id"] == "dense_2500" and row["mode"] == "end_to_end":
            dense_rows.append(row)
    target = {row["dimension"]: row for row in dense_rows}
    baseline = 0.721014
    for dim, value in [(1536, 0.721014), (512, 0.721080), (256, 0.717860), (128, 0.708970), (64, 0.703849), (32, 0.701878), (16, 0.714452), (8, 0.723936)]:
        row = target.get(dim)
        if row is None:
            row = {
                "dataset_id": "dense_2500",
                "mode": "end_to_end",
                "dimension": dim,
                "accuracy": None,
                "mae": None,
                "f1": None,
                "brier": None,
                "macro_agent_accuracy": None,
                "combined_accuracy": None,
                "aggregate_rate_error": None,
                "concept_coverage": None,
                "agent_coverage": None,
                "accuracy_delta_native": None,
                "accuracy_delta_seed_ci95": None,
            }
            payload["summary"].append(row)
            target[dim] = row
        row["accuracy"] = value
        row["accuracy_delta_native"] = value - baseline
    payload["decisions"].append(
        {
            "dataset_id": "dense_2500",
            "mode": "end_to_end",
            "smallest_non_degrading_dimension": 512,
            "one_pp_candidate_dimension": 256,
            "reason": "test decision",
        }
    )
    html = build_report(payload, "C:/tmp/aggregate.json")
    assert "dense_2500 / end_to_end" in html
    assert "non-monotonic" in html
    assert "native 72.10%, 512d 72.11%, 8d 72.39%" in html
    assert "Run status partial" in html
    assert "8d point is shown explicitly" in html
    assert "This is a small pooled shift." in html
    assert "smallest contiguous pooled non-degrading prefix = <strong>512</strong>" in html
    assert "exploratory 1pp seed-interval candidate = <strong>256</strong>" in html
    assert "1536:72.1014%" in html
    assert "8:72.3936%" in html


def test_cli_builds_html_file(tmp_path: Path, monkeypatch) -> None:
    payload = _build_fixture()
    aggregate_path = tmp_path / "aggregate.json"
    output_path = tmp_path / "matryoshka.html"
    aggregate_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["build_matryoshka_report.py", "--input", str(aggregate_path), "--output", str(output_path)],
    )
    main()
    assert output_path.exists()
    rendered = output_path.read_text(encoding="utf-8")
    assert "Matryoshka Prefix Truncation Report" in rendered
    assert "matryoshka-demo" in rendered
