from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.v4_report import (  # noqa: E402
    DEFAULT_OUTPUT_NAME,
    V4ReportInputs,
    default_inputs,
    write_v4_html_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build self-contained HTML report from persisted sampling v4 artifacts"
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="Directory containing v4 artifact outputs",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output HTML file path. Defaults to input-dir/agent365-sampling-v4-report.html",
    )
    return parser


def _inputs_from_dir(input_dir: Path) -> V4ReportInputs:
    return default_inputs(input_dir)


def _resolve_input_dir(input_arg: str) -> Path:
    if input_arg.strip():
        out = Path(input_arg)
    else:
        out = Path("outputs_sampling_v4") / "runs"
    posix = out.as_posix()
    if "outputs_sampling_v2" in posix or "outputs_sampling_v3" in posix:
        raise ValueError("V4 report input path must not target V2/V3 output paths")
    return out


def _resolve_output(output_arg: str, input_dir: Path) -> Path:
    if output_arg.strip():
        out = Path(output_arg)
    else:
        out = input_dir / DEFAULT_OUTPUT_NAME
    posix = out.as_posix()
    if "outputs_sampling_v2" in posix or "outputs_sampling_v3" in posix:
        raise ValueError("V4 report output path must not target V2/V3 output paths")
    return out


def main() -> None:
    args = _build_parser().parse_args()
    input_dir = _resolve_input_dir(str(args.input_dir))
    output_path = _resolve_output(str(args.output), input_dir)

    output = write_v4_html_report(
        output_path=output_path,
        inputs=_inputs_from_dir(input_dir),
    )
    print(str(output))


if __name__ == "__main__":
    main()
