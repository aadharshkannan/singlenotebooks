from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.idw_threshold_public_report import write_public_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish aggregate-only IDW threshold/envelope results.")
    parser.add_argument("--private-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(write_public_report(args.private_run, args.output))


if __name__ == "__main__":
    main()
