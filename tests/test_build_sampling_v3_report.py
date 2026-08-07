from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_sampling_v3_report import _build_parser, _resolve_input_dir, _resolve_output


def test_build_v3_parser_defaults_are_empty_for_resolve_logic() -> None:
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.input_dir == ""
    assert args.output == ""


def test_build_v3_resolve_rejects_v2_paths() -> None:
    with pytest.raises(ValueError, match="must not target V2"):
        _resolve_input_dir("outputs_sampling_v2/v2")

    with pytest.raises(ValueError, match="must not target V2"):
        _resolve_output("outputs_sampling_v2/runs/report.html", Path("outputs_sampling_v3/runs/x"))


def test_build_v3_resolve_defaults_to_v3_output_name() -> None:
    input_dir = _resolve_input_dir("")
    out = _resolve_output("", input_dir)
    assert out.as_posix().endswith("outputs_sampling_v3/runs/agent365-sampling-v3-report.html")
