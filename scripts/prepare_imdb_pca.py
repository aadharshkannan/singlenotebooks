"""Prepare one offline, source-bound PCA fit in a dedicated process (no API calls)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.imdb_pca import DEFAULT_INPUT, DEFAULT_PREPARATION, DIMENSIONS, prepare_pca


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--dimensions", type=int, nargs="+", default=DIMENSIONS)
    parser.add_argument("--blas-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = prepare_pca(args.input, args.output, dimensions=args.dimensions,
                         blas_threads=args.blas_threads, resume=args.resume)
    print(f"{result['status']}: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
