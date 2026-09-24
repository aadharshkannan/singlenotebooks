"""Continue existing PCA checkpoints with six disjoint workers; never refit PCA."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison import imdb_pca as pca
from sampling_comparison.imdb_pca_scale import run_scaled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=pca.DEFAULT_INPUT)
    parser.add_argument("--pca-manifest", type=Path, default=pca.DEFAULT_PREPARATION / "manifest.json")
    parser.add_argument("--original", type=Path, default=pca.extension.DEFAULT_BASELINE)
    parser.add_argument("--baseline", type=Path, default=pca.DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=pca.DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    result = run_scaled(args.input, pca_manifest=args.pca_manifest, original=args.original,
                        baseline=args.baseline, output=args.output, workers=args.workers)
    print(f"{result['status']}: {args.output / 'aggregate.json'}")


if __name__ == "__main__":
    main()
