from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.v6_report import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    V6ReportInputs,
    default_inputs,
    write_v6_html_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a self-contained Sampling V6 HTML report.")
    parser.add_argument("--input-dir", default="", help="Directory containing aggregate.json, runs.jsonl, memberships.jsonl, and other artifacts.")
    parser.add_argument("--output", default="", help="Output HTML report path. Defaults to the input directory and DEFAULT_OUTPUT_NAME.")
    parser.add_argument("--browser", default="", help="Optional Chrome/Edge executable override for PDF export.")
    parser.add_argument("--html-only", action="store_true", help="Generate only HTML and skip PDF export.")
    return parser


def _resolve_input_dir(input_arg: str) -> Path:
    if input_arg.strip():
        return Path(input_arg)
    return DEFAULT_INPUT_DIR


def _resolve_output(output_arg: str, input_dir: Path) -> Path:
    if output_arg.strip():
        return Path(output_arg)
    return input_dir / DEFAULT_OUTPUT_NAME


def _inputs_from_dir(input_dir: Path) -> V6ReportInputs:
    return default_inputs(input_dir)


def main() -> None:
    args = _build_parser().parse_args()
    input_dir = _resolve_input_dir(args.input_dir)
    output_path = _resolve_output(args.output, input_dir)
    write_v6_html_report(output_path=output_path, inputs=_inputs_from_dir(input_dir), pdf=not bool(args.html_only), browser_path=args.browser or None)
    print(str(output_path))


if __name__ == "__main__":
    main()
