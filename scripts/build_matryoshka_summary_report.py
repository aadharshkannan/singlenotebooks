from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_summary_report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a three-graph, plain-language Matryoshka report from retained artifacts only.")
    parser.add_argument("--input", required=True, help="Exact Matryoshka aggregate.json path.")
    parser.add_argument("--output", required=True, help="HTML path in a separate derivative directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing derivative, never the source run.")
    args = parser.parse_args()
    print(write_report(Path(args.input), Path(args.output), overwrite=args.overwrite))


if __name__ == "__main__":
    main()
