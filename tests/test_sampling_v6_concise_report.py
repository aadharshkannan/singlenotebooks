from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

from sampling_comparison.v6_concise_report import (
    DEFAULT_OUTPUT_NAME,
    DEFAULT_PDF_NAME,
    _build_recommendation,
    compose_pdf_command,
    render_v6_concise_html_report,
    write_v6_concise_report,
)
from sampling_comparison.v6_report import load_v6_artifacts
from tests.test_sampling_v6_report import _build_fixture_bundle


def test_concise_report_renders_sections_and_constraints(tmp_path: Path) -> None:
    bundle = _build_fixture_bundle(tmp_path)
    html = render_v6_concise_html_report(bundle)

    required_sections = [
        "Decision at a glance",
        "What is measured",
        "Five techniques",
        "Results story",
        "Interactive explorer",
        "Conclusion",
    ]
    for section in required_sections:
        assert section in html, f"missing required section: {section}"

    assert "proportion of distinct source concepts" in html.lower()
    assert "ARM2" in html and "ARM3" in html and "ARM4" in html and "ARM5" in html
    assert html.count("<svg") >= 3
    assert "artifact-json" in html
    assert "method-toggle" in html
    assert "https://" not in html.lower() and "http://" not in html.lower()

    bodies = re.findall(r"<tbody>(.*?)</tbody>", html, re.S)
    assert bodies, "expected at least one table body"
    assert all(len(re.findall(r"<tr>", body)) <= 5 for body in bodies), "summary tables exceeded the 5-row cap"


def test_concise_report_uses_concise_output_names_and_pdf_command() -> None:
    assert DEFAULT_OUTPUT_NAME.endswith("sampling-v6-concise-report.html")
    assert DEFAULT_PDF_NAME.endswith("sampling-v6-concise-report.pdf")
    cmd = compose_pdf_command(None, Path("C:/tmp/sampling-v6-concise-report.html"), Path("C:/tmp/sampling-v6-concise-report.pdf"))
    assert "print-to-pdf=C:/tmp/sampling-v6-concise-report.pdf" in cmd[0:10] or any("print-to-pdf=" in part for part in cmd)
    assert "sampling-v6-concise-report.pdf" in " ".join(cmd)


def test_write_concise_report_exports_safe_files(tmp_path: Path, monkeypatch) -> None:
    bundle = _build_fixture_bundle(tmp_path)
    out_dir = tmp_path / "report-output"

    def fake_run(command, check, stdout=None, stderr=None):
        assert command[0].lower().endswith(("msedge", "chrome", "chrome.exe")) or "msedge" in command[0].lower() or "chrome" in command[0].lower()
        pdf_path = Path(command[3].split("=", 1)[1])
        pdf_path.write_bytes(b"%PDF-1.4\n" + (b"x" * 2048))
        return None

    monkeypatch.setattr("sampling_comparison.v6_concise_report.subprocess.run", fake_run)

    result = write_v6_concise_report(out_dir / "sampling-v6-concise-report.html", bundle, pdf=True, browser_path="C:/Program Files/Microsoft/Edge/Application/msedge.exe")
    assert result.name == "sampling-v6-concise-report.html"
    assert (out_dir / "sampling-v6-concise-report.pdf").exists()
    assert "<script>" in result.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in result.read_text(encoding="utf-8")


def test_concise_report_has_dynamic_context_and_labeled_cards(tmp_path: Path) -> None:
    bundle = _build_fixture_bundle(tmp_path)
    html = render_v6_concise_html_report(bundle)

    assert "2,800 sessions" in html
    assert "105 agents" in html
    assert "dense_2500" in html and "historical_300" in html
    assert "Ambiguous" in html and "provisional" in html.lower()
    assert "Steps" in html and "Estimator" in html and "Strength" in html and "Risk" in html and "Choose when" in html
    assert "Actual tokens" in html and "Actual / nominal" in html
    assert "seed-select" in html and "Trial mode" in html
    assert "if (typeof stats === 'number') return stats" in html
    assert "Cap-by-cap leaders" in html
    assert "lower MAE and increase concept coverage" in html
    assert "0%" in html or "25%" in html or "50%" in html
    assert "ARM1" in html and "ARM2" in html and "ARM3" in html and "ARM4" in html and "ARM5" in html


def test_concise_report_tied_leaders_are_named_and_seed_filter_applies(tmp_path: Path) -> None:
    bundle = _build_fixture_bundle(tmp_path)
    artifacts = load_v6_artifacts(bundle)
    aggregate_rows = deepcopy(artifacts.aggregate.get("aggregate_rows") or [])
    for row in aggregate_rows:
        if row.get("method_id") in {
            "arm2_embedding_idw",
            "arm3_agent_round_robin_floor",
            "arm4_agent_round_robin",
            "arm5_hajek_weighted",
        } and row.get("cap") == 64:
            row["metrics"]["concept_coverage"]["mean"] = 0.72
    recommendation = _build_recommendation(aggregate_rows)
    html = render_v6_concise_html_report(bundle)

    assert "tied for broadest concept coverage" in recommendation.lower()
    assert "ARM2" in html and "ARM3" in html and "ARM4" in html and "ARM5" in html

    js = html.split("<script>", 1)[1]
    assert "seed-select" in js.lower()
    assert "selectedSeed" in js or "state.seed" in js or "trialRows" in js or "filter" in js.lower()
    assert "slice(0, 5)" not in js


def test_concise_report_high_cap_ties_match_exact_leaders(tmp_path: Path) -> None:
    bundle = _build_fixture_bundle(tmp_path)
    aggregate_rows = deepcopy(load_v6_artifacts(bundle).aggregate.get("aggregate_rows") or [])
    for row in aggregate_rows:
        if row.get("cap") == 256:
            if row.get("method_id") == "arm5_hajek_weighted":
                row["metrics"]["concept_coverage"]["mean"] = 0.80
            elif row.get("method_id") == "arm4_agent_round_robin":
                row["metrics"]["concept_coverage"]["mean"] = 0.80
            elif row.get("method_id") == "arm3_agent_round_robin_floor":
                row["metrics"]["concept_coverage"]["mean"] = 0.7996
    recommendation = _build_recommendation(aggregate_rows)
    assert "best combined MAE/coverage balance" in recommendation
    assert "ARM3 Agent Round Robin Floor" not in recommendation.split("At the high-cap end", 1)[1]
