from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.v2_experiment import (  # noqa: E402
    DENSE_2500_PATH,
    HISTORICAL_300_PATH,
    run_v2_experiment_bundle,
)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    cleaned = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(int(part) for part in cleaned)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run combined-corpus sampling v2 experiment bundle")
    parser.add_argument(
        "--output",
        default=str(Path("outputs_sampling_v2") / "runs" / "latest"),
        help="Output directory for rerun artifacts; the reviewed v2 reference is not overwritten by default",
    )
    parser.add_argument(
        "--historical-path",
        default=HISTORICAL_300_PATH,
        help="Path to historical 300-session source",
    )
    parser.add_argument(
        "--dense-path",
        default=DENSE_2500_PATH,
        help="Path to dense 2500-session source",
    )
    parser.add_argument(
        "--budget-pcts",
        default="5,10,20,30,50",
        help="Comma-separated outcome budget percents",
    )
    parser.add_argument(
        "--outcome-repetitions",
        type=int,
        default=3,
        help="Paired repetitions for outcome comparison",
    )
    parser.add_argument(
        "--quadrant-replays",
        type=int,
        default=3,
        help="Paired replays per quadrant cell",
    )
    parser.add_argument(
        "--throughput-replays",
        type=int,
        default=2,
        help="Paired replays per throughput cell",
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_v2_experiment_bundle(
        historical_path=args.historical_path,
        dense_path=args.dense_path,
        budget_pcts=_parse_int_csv(args.budget_pcts),
        outcome_repetitions=int(args.outcome_repetitions),
        quadrant_replays=int(args.quadrant_replays),
        throughput_replays=int(args.throughput_replays),
        seed=int(args.seed),
        output_dir=Path(args.output),
    )

    summary = {
        "version": result["aggregate"]["version"],
        "population_count": result["aggregate"]["population_count"],
        "runtime_seconds": result["aggregate"]["runtime_seconds"],
        "output_paths": result["output_paths"],
        "notes": [
            "Label-only scoring; no LLM calls.",
            "Deterministic offline embeddings for full-session arm.",
            "MinHash is 32x4 with 128 permutations.",
            "No Azure resources required for local run.",
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
