from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_sampling_v4_report import _build_parser, _resolve_input_dir, _resolve_output, main


def test_build_v4_parser_defaults_are_empty_for_resolve_logic() -> None:
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.input_dir == ""
    assert args.output == ""
    assert args.summary is False


def test_build_v4_resolve_rejects_v2_v3_paths() -> None:
    with pytest.raises(ValueError, match="must not target V2/V3"):
        _resolve_input_dir("outputs_sampling_v3/runs/x")

    with pytest.raises(ValueError, match="must not target V2/V3"):
        _resolve_output("outputs_sampling_v2/runs/report.html", Path("outputs_sampling_v4/runs/x"))


def test_build_v4_resolve_defaults_to_v4_output_name() -> None:
    input_dir = _resolve_input_dir("")
    out = _resolve_output("", input_dir)
    assert out.as_posix().endswith("outputs_sampling_v4/runs/agent365-sampling-v4-report.html")

    summary_out = _resolve_output("", input_dir, summary=True)
    assert summary_out.as_posix().endswith("outputs_sampling_v4/runs/agent365-sampling-v4-summary-report.html")


def test_build_v4_main_invokes_writer(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: dict[str, object] = {}

    def _fake_write_v4_html_report(*, output_path, inputs, section_ids, report_title, manifest_name):
        calls["output_path"] = output_path
        calls["inputs"] = inputs
        calls["section_ids"] = section_ids
        calls["report_title"] = report_title
        calls["manifest_name"] = manifest_name
        return output_path

    monkeypatch.setattr("scripts.build_sampling_v4_report.write_v4_html_report", _fake_write_v4_html_report)
    monkeypatch.setattr("sys.argv", ["build_sampling_v4_report.py"])

    main()

    assert "output_path" in calls
    assert str(calls["output_path"]).replace("\\", "/").endswith("outputs_sampling_v4/runs/agent365-sampling-v4-report.html")
    assert calls["section_ids"] is None
    assert calls["report_title"] == "Agent365 Sampling V4 Report"
    assert calls["manifest_name"] == "report_manifest.json"
    assert capsys.readouterr().out.strip().endswith("agent365-sampling-v4-report.html")


def test_build_v4_main_builds_expanded_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake_write_v4_html_report(*, output_path, inputs, section_ids, report_title, manifest_name):
        calls.update(
            output_path=output_path,
            section_ids=section_ids,
            report_title=report_title,
            manifest_name=manifest_name,
        )
        return output_path

    monkeypatch.setattr("scripts.build_sampling_v4_report.write_v4_html_report", _fake_write_v4_html_report)
    monkeypatch.setattr("sys.argv", ["build_sampling_v4_report.py", "--summary"])

    main()

    assert str(calls["output_path"]).replace("\\", "/").endswith("agent365-sampling-v4-summary-report.html")
    assert len(calls["section_ids"]) == 8
    assert calls["report_title"] == "Agent365 Sampling V4 Summary Report"
    assert calls["manifest_name"] == "summary_report_manifest.json"
