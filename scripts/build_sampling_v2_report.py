from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.v2_report import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    V2ReportInputs,
    default_inputs,
    write_v2_html_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build self-contained HTML report from persisted sampling v2 artifacts"
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing v2 artifact JSON/JSONL outputs",
    )
    parser.add_argument(
        "--output",
        default=str(Path("outputs_sampling_v2") / "runs" / "agent365-sampling-v2-report.html"),
        help="Output HTML file path; the reviewed v2 reference is not overwritten by default",
    )
    return parser


def _inputs_from_dir(input_dir: Path) -> V2ReportInputs:
    defaults = default_inputs(input_dir)
    return V2ReportInputs(
        aggregate=defaults.aggregate,
        corpus_audit=defaults.corpus_audit,
        quadrant=defaults.quadrant,
        throughput=defaults.throughput,
        selected_membership_20pct=defaults.selected_membership_20pct,
        production_storage_manifest=defaults.production_storage_manifest,
        external_eval_manifest=defaults.external_eval_manifest,
    )


def main() -> None:
    args = _build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    output = write_v2_html_report(
        output_path=output_path,
        inputs=_inputs_from_dir(input_dir),
    )
    print(str(output))


if __name__ == "__main__":
    main()
