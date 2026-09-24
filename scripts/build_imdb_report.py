from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.imdb_report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standalone IMDb report from explicit readiness or completed replay evidence.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numerical-validation", type=Path,
                        help="Optional successful source-bound retained-score audit JSON.")
    args = parser.parse_args()
    result = build_report(args.input, args.output, numerical_validation=args.numerical_validation)
    print(f"{result['status']}: {args.output / 'report.html'}")


if __name__ == "__main__":
    main()
