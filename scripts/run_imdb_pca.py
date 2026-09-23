"""Replay PCA only, preserving every native/prefix cell; preparation must already exist."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.imdb_extension import DEFAULT_BASELINE as ORIGINAL
from sampling_comparison.imdb_pca import (
    DEFAULT_BASELINE, DEFAULT_INPUT, DEFAULT_OUTPUT, DEFAULT_PREPARATION, run_pca,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pca-manifest", type=Path, default=DEFAULT_PREPARATION / "manifest.json")
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_pca(args.input, pca_manifest=args.pca_manifest, original=args.original,
                     baseline=args.baseline, output=args.output, workers=args.workers,
                     blas_threads=args.blas_threads, resume=args.resume)
    print(f"{result['status']}: {args.output / 'aggregate.json'}")


if __name__ == "__main__":
    main()
