from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threadpoolctl import threadpool_limits

from sampling_comparison.imdb_experiment import run_experiment
from sampling_comparison.imdb_inputs import load_input


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline paired IMDb sampling/IDW/envelope replay; requires real prepared embeddings.")
    parser.add_argument("--input", type=Path, default=Path("outputs_imdb/cache/imdb-50000/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs_imdb/private_runs/imdb-40-replay"))
    parser.add_argument("--repetitions", type=int, default=40)
    parser.add_argument("--base-seed", type=int, default=13)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1536, 32, 24, 16, 12, 8])
    parser.add_argument("--rates", type=float, nargs="+", default=[.01, .02, .05, .10, .20])
    parser.add_argument("--schedules", nargs="+", choices=["uniformly_random", "bursty"],
                        default=["uniformly_random", "bursty"])
    parser.add_argument("--blas-threads", type=int, default=4)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("complete embedding manifest is missing; run prepare_imdb_input.py --live first")
    if args.blas_threads < 1:
        parser.error("--blas-threads must be positive")
    vectors, labels, profile = load_input(args.input)
    with threadpool_limits(limits=args.blas_threads):
        result = run_experiment(
            vectors, labels, args.output, profile=profile, dimensions=tuple(args.dimensions),
            repetitions=args.repetitions, base_seed=args.base_seed,
            rates=tuple(args.rates), schedules=tuple(args.schedules),
        )
    print(f"{result['status']}: {args.output / 'aggregate.json'}")


if __name__ == "__main__":
    main()
