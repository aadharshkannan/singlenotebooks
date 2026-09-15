from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_report import build_report  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build static Matryoshka prefix truncation HTML report from aggregate JSON."
    )
    parser.add_argument("--input", required=True, help="Path to aggregate JSON artifact.")
    parser.add_argument(
        "--output",
        default="",
        help="Output HTML path. Defaults to sibling matryoshka_report.html beside --input.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if str(args.output).strip() else input_path.with_name("matryoshka_report.html")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    html = build_report(payload, str(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
