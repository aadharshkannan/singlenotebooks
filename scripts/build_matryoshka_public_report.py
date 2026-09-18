from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_public_report import write_public_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish aggregate-only reports from a protected Matryoshka run.")
    parser.add_argument("--private-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--focus-dimension", type=int, default=2)
    parser.add_argument("--snapshot-readiness", type=Path, help="Verified count/hash-only snapshot preflight; no raw documents.")
    args = parser.parse_args()
    print(write_public_report(
        args.private_run, args.output, focus_dimension=args.focus_dimension,
        snapshot_readiness=args.snapshot_readiness,
    ))


if __name__ == "__main__":
    main()
