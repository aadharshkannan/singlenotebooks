from copy import deepcopy
import json
import math
from pathlib import Path
import re
from xml.etree import ElementTree

import pytest

from sampling_comparison.matryoshka_experiment import sha256_file, write_json
from sampling_comparison.matryoshka_summary_report import build_report, summarize_evidence, trend_chart, trend_figure, write_report


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


def test_normalization_note_explains_magnitude_and_cosine_without_an_extra_graph():
    html, _ = render()
    for text in ("we are not dividing by 8", "[0.3, 0.4]", "[0.6, 0.8]",
                 "same direction and proportions, but length 1",
                 "cosine calculation normalizes internally",
                 "not another reduction technique or an accuracy-enhancing trick"):
        assert text in html
    assert html.index('id="normalization-note"') > html.index("np.linalg.norm(short)")
    assert html.count("data-graph=") == 3


def test_negative_results_cannot_receive_requested_positive_conclusion():
    data = fixture()
    for row in data["rows"]:
        if row["dataset_id"] == "dense_2500" and row["dimension"] == 8:
            row["mae"] = 0.7
    html, evidence = render(data)
    assert evidence["no_average_mae_drop"] is False
    assert "do not support a blanket no-drop-off conclusion" in html
    assert "did not notice any meaningful performance drop-off" not in html
    assert min(evidence["grid_no_average_increase"]) == 512
    assert "512 dimensions was the smallest tested size" in html
    assert "not a validated cutoff" in html
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


def test_fallback_dependency_is_visible_with_the_main_results():
    html, _ = render()
    start = html.index('<section id="results">')
    end = html.index('<section id="variation">')
    assert "Not IDW-only" in html[start:end]
    assert "dilute the effect of changing embedding dimensions" in html[start:end]


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


def low_dimension_fixture():
    data = fixture()
    dimensions = [1536, *range(32, 1, -2)]
    templates = [r for r in data["rows"] if r["dimension"] == 1536]
    data["protocol"]["dimensions"] = dimensions
    data["rows"] = [
        {**row, "dimension": dimension, "mae": row["mae"] + (0.03 if dimension == 2 else 0)}
        for row in templates for dimension in dimensions
    ]
    return data


def test_two_dimension_report_uses_actual_endpoint_and_full_grid():
    data = low_dimension_fixture()
    html, evidence = build_report(
        data, aggregate_path="low/aggregate.json", aggregate_hash="abc",
        detailed_url="../../run/report.html", focus_dimension=2,
    )
    assert evidence["focus_dimension"] == 2
    assert evidence["no_average_mae_drop"] is False
    assert len(evidence["datasets"][0]["curve"]) == 17
    assert evidence["datasets"][0]["short_mae"] == pytest.approx(0.51)
    assert "eight_mae" not in evidence["datasets"][0]
    for text in ("0 through 1", "768&times; fewer values", "MAE at 2",
                 "2 dimensions versus native", "versus 8 bytes at 2",
                 "higher average MAE at 2 dimensions"):
        assert text in html
    assert "192&times;" not in html
    assert "MAE at 8" not in html
    assert html.count("data-graph=") == 3
    assert "did not notice any meaningful performance drop-off" not in html
    assert min(evidence["grid_no_average_increase"]) == 4
    assert "4 dimensions was the smallest tested size" in html
    assert "not a validated cutoff" in html


@pytest.mark.parametrize("dimension", [1536, 3, 0, True, 2.0])
def test_focus_dimension_must_be_a_tested_reduced_integer(dimension):
    with pytest.raises(ValueError, match="focus dimension"):
        summarize_evidence(low_dimension_fixture(), focus_dimension=dimension)


def test_focus_dimension_is_preserved_in_derivative_manifest(tmp_path: Path):
    source = tmp_path / "run" / "aggregate.json"
    write_json(source, low_dimension_fixture())
    write_json(source.parent / "manifest.json", {"files": {"aggregate": {"sha256": sha256_file(source)}}})
    output = tmp_path / "summary" / "report.html"
    write_report(source, output, focus_dimension=2)
    metadata = json.loads((output.parent / "manifest.json").read_text())
    assert metadata["focus_dimension"] == 2
    assert "'--focus-dimension' '2'" in metadata["generation_command"]
    assert json.loads((output.parent / "summary.json").read_text())["focus_dimension"] == 2


def test_trend_dropdown_defaults_to_percentage_and_has_two_self_contained_views():
    html, evidence = render()
    match = re.search(r'<script type="application/json" id="trend-variants">(.*?)</script>', html, re.S)
    variants = json.loads(match.group(1))
    assert set(variants) == {"relative", "mae"}
    assert '<label for="trend-scale">Y-axis</label>' in html
    assert '<option value="relative" selected>Relative MAE change (%)</option>' in html
    assert '<option value="mae">MAE</option>' in html
    assert 'aria-controls="trend-chart"' in html
    assert 'id="trend-caption" aria-live="polite"' in html
    assert "below zero is better" in variants["relative"]["caption"]
    assert "zero-based MAE axis" in variants["mae"]["caption"]
    assert "including mean/prior fallbacks" in variants["mae"]["caption"]
    assert html.count("<svg ") == 3
    assert html.count("data-graph=") == 3
    assert evidence["graph_count"] == 3
    assert "<svg " not in match.group(1)
    assert "fetch(" not in html and "<script src=" not in html


@pytest.mark.parametrize("scale", ["relative", "mae"])
def test_trend_plot_uses_exact_measured_mae_or_relative_values_at_every_dimension(scale):
    evidence = summarize_evidence(low_dimension_fixture(), focus_dimension=2)
    svg = ElementTree.fromstring(trend_chart(evidence, scale=scale))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert svg.tag == "{http://www.w3.org/2000/svg}svg"
    points = svg.findall("svg:circle", namespace)
    assert len(points) == 3 * 17
    by_dataset = {row["dataset_id"]: row for row in evidence["datasets"]}
    for point in points:
        row = by_dataset[point.attrib["data-dataset"]]
        mae = row["curve"][point.attrib["data-dimension"]]
        expected = 100 * (mae / row["native_mae"] - 1) if scale == "relative" else mae
        assert float(point.attrib["data-value"]) == pytest.approx(expected)
        tooltip = point.find("svg:title", namespace).text
        if scale == "mae":
            assert f"MAE {mae:.6f}" in tooltip and "%" not in tooltip
            assert float(point.attrib["cy"]) == pytest.approx(72 + (0.6 - mae) * 280 / 0.6, abs=0.001)
    text = [node.text for node in svg.findall("svg:text", namespace)]
    if scale == "mae":
        assert "0.000" in text and "0.600" in text
        assert not any("%" in value for value in text)
        assert "Mean absolute error (MAE, 0-1 units): lower is better" in text


def test_zero_native_mae_is_missing_only_in_relative_view():
    evidence = summarize_evidence(fixture())
    row = evidence["datasets"][0]
    row["native_mae"] = 0.0
    row["curve"]["1536"] = 0.0
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    relative = ElementTree.fromstring(trend_chart(evidence)).findall("svg:circle", namespace)
    absolute = ElementTree.fromstring(trend_chart(evidence, scale="mae")).findall("svg:circle", namespace)
    assert not any(point.attrib["data-dataset"] == row["dataset_id"] for point in relative)
    assert sum(point.attrib["data-dataset"] == row["dataset_id"] for point in absolute) == 3


def test_trend_variants_escape_script_terminators_in_all_dynamic_content():
    evidence = summarize_evidence(fixture())
    injected = "</script><script>alert('private')</script>"
    evidence["datasets"][0]["label"] = injected
    html = trend_figure(evidence, injected)
    assert injected not in html
    data = re.search(r'id="trend-variants">(.*?)</script>', html, re.S).group(1)
    assert "</script>" not in data
    assert "&lt;/script&gt;" in json.loads(data)["mae"]["svg"]
    with pytest.raises(ValueError, match="trend scale"):
        trend_chart(evidence, scale="accuracy")
