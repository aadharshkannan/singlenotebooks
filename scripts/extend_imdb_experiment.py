"""Add only new normalized prefixes to a hash-verified completed IMDb study."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.imdb_extension import DEFAULT_BASELINE, DEFAULT_OUTPUT, DIMENSIONS, run_extension
from sampling_comparison.imdb_inputs import load_input


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs_imdb") / "cache" / "imdb-50000" / "manifest.json")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(DIMENSIONS))
    parser.add_argument("--blas-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="reuse only hash-compatible committed extension work")
    args = parser.parse_args(argv)
    if args.blas_threads < 1:
        parser.error("--blas-threads must be positive")
    if not args.input.is_file():
        parser.error("complete native input manifest is required; this command makes no embedding calls")
    vectors, labels, profile = load_input(args.input)
    result = run_extension(
        vectors, labels, args.output, profile=profile, baseline=args.baseline,
        dimensions=tuple(args.dimensions), blas_threads=args.blas_threads, resume=args.resume,
    )
    print(f"{result['status']}: {args.output / 'aggregate.json'}")


if __name__ == "__main__":
    main()
