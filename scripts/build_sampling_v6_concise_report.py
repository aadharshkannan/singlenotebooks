from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.v6_concise_report import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_PDF_NAME,
    V6ReportInputs,
    default_inputs,
    write_v6_concise_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a concise Sampling V6 HTML/PDF report from the canonical bundle.")
    parser.add_argument("--input-dir", default="", help="Directory containing aggregate.json, runs.jsonl, memberships.jsonl, and other V6 artifacts.")
    parser.add_argument("--output", default="", help="Output HTML path. Defaults to the canonical bundle output folder and sampling-v6-concise-report.html.")
    parser.add_argument("--browser", default="", help="Optional browser executable override for PDF export.")
    parser.add_argument("--html-only", action="store_true", help="Generate only HTML and skip PDF export.")
    return parser


def _resolve_input_dir(arg: str) -> Path:
    if arg.strip():
        return Path(arg)
    return DEFAULT_INPUT_DIR


def _resolve_output(arg: str, input_dir: Path) -> Path:
    if arg.strip():
        return Path(arg)
    return input_dir / DEFAULT_OUTPUT_NAME


def _inputs_from_dir(input_dir: Path):
    return default_inputs(input_dir)


def main() -> None:
    args = _build_parser().parse_args()
    input_dir = _resolve_input_dir(args.input_dir)
    output_path = _resolve_output(args.output, input_dir)
    write_v6_concise_report(output_path=output_path, inputs=_inputs_from_dir(input_dir), pdf=not args.html_only, browser_path=args.browser or None)
    print(str(output_path))
    if not args.html_only:
        print(str(output_path.with_name(DEFAULT_PDF_NAME)))


if __name__ == "__main__":
    main()
