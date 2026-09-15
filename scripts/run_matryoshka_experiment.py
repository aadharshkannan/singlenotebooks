from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_experiment import DATASETS, DIMENSIONS, RATES, SCHEDULES, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline prefix-truncation + causal IDW experiment; never calls embedding or judge APIs.")
    parser.add_argument("--input", action="append", default=[], metavar="DATASET=MANIFEST", help=f"Repeat for {', '.join(DATASETS)}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dimensions", default=",".join(map(str, DIMENSIONS)))
    parser.add_argument("--seeds", default=",".join(map(str, range(13, 43))))
    parser.add_argument("--rates", default=",".join(map(str, RATES)))
    parser.add_argument("--schedules", default=",".join(SCHEDULES))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    inputs = {}
    for value in args.input:
        name, path = value.split("=", 1)
        if name in inputs:
            parser.error(f"duplicate dataset input: {name}")
        inputs[name] = Path(path)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = run_experiment(
        inputs, Path(args.output), source_revision=revision,
        dimensions=tuple(int(x) for x in args.dimensions.split(",")),
        seeds=tuple(int(x) for x in args.seeds.split(",")),
        rates=tuple(float(x) for x in args.rates.split(",")),
        schedules=tuple(args.schedules.split(",")), resume=args.resume,
    )
    print(f"{result['status']}: {len(result['rows'])} result cells; {args.output}")


if __name__ == "__main__":
    main()
