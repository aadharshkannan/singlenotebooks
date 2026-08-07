from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.v3_report import (  # noqa: E402
    DEFAULT_OUTPUT_NAME,
    V3ReportInputs,
    default_inputs,
    write_v3_html_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build self-contained HTML report from persisted sampling v3 artifacts"
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="Directory containing v3 artifact JSON/JSONL outputs",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output HTML file path. Defaults to input-dir/agent365-sampling-v3-report.html",
    )
    return parser


def _inputs_from_dir(input_dir: Path) -> V3ReportInputs:
    return default_inputs(input_dir)


def _resolve_input_dir(input_arg: str) -> Path:
    if input_arg.strip():
        out = Path(input_arg)
    else:
        out = Path("outputs_sampling_v3") / "runs"
    posix = out.as_posix()
    if "outputs_sampling_v2" in posix:
        raise ValueError("V3 report input path must not target V2 output paths")
    return out


def _resolve_output(output_arg: str, input_dir: Path) -> Path:
    if output_arg.strip():
        out = Path(output_arg)
    else:
        out = input_dir / DEFAULT_OUTPUT_NAME
    if "outputs_sampling_v2" in out.as_posix():
        raise ValueError("V3 report output path must not target V2 output paths")
    return out


def main() -> None:
    args = _build_parser().parse_args()
    input_dir = _resolve_input_dir(str(args.input_dir))
    output_path = _resolve_output(str(args.output), input_dir)

    output = write_v3_html_report(
        output_path=output_path,
        inputs=_inputs_from_dir(input_dir),
    )
    print(str(output))


if __name__ == "__main__":
    main()
