"""Continue a checkpointed IMDb extension with three independent replay workers."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.imdb_extension import DEFAULT_BASELINE, DEFAULT_OUTPUT
from sampling_comparison.imdb_parallel import run_parallel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs_imdb/cache/imdb-50000/manifest.json"))
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("--workers must be between one and three")
    result = run_parallel(args.input, baseline=args.baseline, output=args.output, workers=args.workers)
    print(f"{result['status']}: {args.output / 'aggregate.json'}")


if __name__ == "__main__":
    main()
