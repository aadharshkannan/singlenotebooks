from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from sampling_comparison.matryoshka_experiment import sha256_file, write_json
from sampling_comparison.matryoshka_summary_report import build_report, summarize_evidence, write_report


def fixture():
    profiles = []
    rows = []
    for dataset, n, agents, base in (
        ("historical_300", 300, 100, 0.48),
        ("dense_2500", 2500, 5, 0.35),
        ("cosmos_otel", 205, 7, 0.32),
    ):
        profiles.append({
            "dataset_id": dataset, "status": "completed", "n": n, "agents": agents, "pass_rate": 0.6,
            "provenance": {
                "embedding_model_id": "text-embedding-3-small", "embedding_dimensions": 1536,
                "live_judge_calls": 0, "label_source": "Synthetic fixture expected labels.",
                "truncated_sessions": 0,
                "representation_sources": {"linked_raw_spans": 195, "synthetic_label_document": 10},
            },
        })
        for dimension in (1536, 512, 8):
            for seed in (13, 14):
                for schedule in ("evenly_spaced", "bursty"):
                    for rate in (0.01, 0.2):
                        delta = 0 if dimension == 1536 else -0.004
                        if dataset == "cosmos_otel" and dimension == 8:
                            delta = 0.005 if rate == 0.2 else -0.02
                        selected = max(1, math.floor(n * rate))
                        row = {
                            "dataset_id": dataset, "dimension": dimension, "seed": seed,
                            "schedule": schedule, "rate": rate, "mode": "end_to_end",
                            "mae": base + delta, "accuracy": 0.7, "brier": 0.2,
                            "selected_count": selected, "unjudged_count": n - selected,
                            "provenance_counts": {"idw": n - selected},
                        }
                        rows.extend([row, {**row, "mode": "fixed_membership"}])
    profiles.append({"dataset_id": "tau2_bench", "status": "blocked", "n": 388})
    return {
        "version": "matryoshka-cutoff-v1", "run_id": "test-fixture",
        "protocol": {
            "dimensions": [1536, 512, 8], "seeds": [13, 14],
            "rates": [0.01, 0.2], "schedules": ["evenly_spaced", "bursty"],
        },
        "datasets": profiles, "rows": rows,
    }


def render(data=None):
    return build_report(data or fixture(), aggregate_path="source/aggregate.json", aggregate_hash="abc", detailed_url="../../runs/report.html")


def test_three_graphs_and_exact_method_are_explicit():
    html, evidence = render()
    assert html.count("<svg ") == 3
    assert html.count("data-graph=") == 3
    assert html.count('class="pipeline"') == 1
    assert evidence["graph_count"] + evidence["method_diagram_count"] <= 5
    for text in ("text-embedding-3-small", "0 through 7", "full_embedding[:d]",
                 "np.linalg.norm(short)", "No PCA, SVD, Gaussian random projection",
                 "not a new LLM judge", "not an independent deployment holdout"):
        assert text in html


def test_conclusion_is_scoped_and_does_not_hide_budget_regression():
    html, evidence = render()
    assert evidence["no_average_mae_drop"] is True
    assert "did not notice any meaningful performance drop-off" in html
    assert "average MAE across the 3 datasets tested" in html
    assert "at a 20% sampling budget in Cosmos" in html
    assert "0.3200 to 0.3250" in html
    assert "not a claim that every setting or metric is unchanged" in html
    assert "No formal acceptable-loss threshold was pre-specified for MAE" in html
    assert "Tau2 bench" in html and "Not included in the results" in html
    assert "not silently substituted" in html


def test_negative_results_cannot_receive_requested_positive_conclusion():
    data = fixture()
    for row in data["rows"]:
        if row["dataset_id"] == "dense_2500" and row["dimension"] == 8:
            row["mae"] = 0.7
    html, evidence = render(data)
    assert evidence["no_average_mae_drop"] is False
    assert "do not support a blanket no-drop-off conclusion" in html
    assert "did not notice any meaningful performance drop-off" not in html
    assert "promising candidate" not in html


def test_primary_metrics_do_not_mix_fixed_membership_or_selected_labels():
    data = fixture()
    for row in data["rows"]:
        if row["mode"] == "fixed_membership":
            row["mae"] = 0.99
    evidence = summarize_evidence(data)
    dense = next(r for r in evidence["datasets"] if r["dataset_id"] == "dense_2500")
    assert dense["native_mae"] == pytest.approx(0.35)
    assert dense["eight_mae"] == pytest.approx(0.346)
    assert dense["change_pct"] == pytest.approx(100 * (0.346 / 0.35 - 1))
    assert evidence["primary_cells"] * 2 == evidence["retained_cells"]


@pytest.mark.parametrize("change", ["duplicate", "missing", "nan", "counts", "model", "judge"])
def test_invalid_or_mismatched_evidence_fails(change):
    data = fixture()
    if change == "duplicate":
        data["rows"].append(deepcopy(data["rows"][0]))
    elif change == "missing":
        data["rows"].pop(0)
    elif change == "nan":
        data["rows"][0]["mae"] = float("nan")
    elif change == "counts":
        data["rows"][0]["selected_count"] = 9999
    elif change == "model":
        data["datasets"][0]["provenance"]["embedding_model_id"] = "text-embedding-3-large"
    else:
        data["datasets"][0]["provenance"]["live_judge_calls"] = 1
    with pytest.raises(ValueError):
        render(data)


def test_source_text_is_escaped_and_details_are_native_interactions():
    data = fixture()
    data["datasets"][0]["provenance"]["label_source"] = "<script>alert('x')</script>"
    html, _ = render(data)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert html.count("<details") == 3
    assert 'id="budget-detail"' in html
    assert 'role="region"' in html and 'tabindex="0"' in html


def test_derivative_preserves_source_and_records_numeric_provenance(tmp_path: Path):
    source = tmp_path / "run" / "aggregate.json"
    write_json(source, fixture())
    source_manifest = source.parent / "manifest.json"
    write_json(source_manifest, {"files": {"aggregate": {"sha256": sha256_file(source)}}})
    before = source.read_bytes(), source_manifest.read_bytes()
    output = tmp_path / "summary" / "report.html"
    write_report(source, output)
    assert before == (source.read_bytes(), source_manifest.read_bytes())
    metadata = json.loads((output.parent / "manifest.json").read_text())
    assert metadata["graph_count"] == 3
    assert metadata["files"]["report.html"]["sha256"] == sha256_file(output)
    assert json.loads((output.parent / "summary.json").read_text())["no_average_mae_drop"] is True
    with pytest.raises(FileExistsError):
        write_report(source, output)
    with pytest.raises(FileExistsError):
        write_report(source, output.parent / "another.html")
    with pytest.raises(ValueError, match="separate directory"):
        write_report(source, source.parent / "report.html")
    source.write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        write_report(source, output, overwrite=True)
