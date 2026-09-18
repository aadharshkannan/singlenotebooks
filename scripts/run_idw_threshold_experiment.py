from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.idw_threshold_experiment import BASELINE, DATASETS, INPUTS, run_experiment


def resolve_inputs(overrides: list[str]) -> dict[str, Path]:
    inputs = dict(INPUTS)
    seen = set()
    for override in overrides:
        name, separator, path = override.partition("=")
        if not separator or name not in DATASETS or not path or name in seen:
            raise ValueError("input overrides require a unique recognized DATASET=MANIFEST")
        seen.add(name)
        inputs[name] = Path(path)
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline replay of full-session IDW thresholds and conditional lower envelopes."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", default=str(BASELINE))
    parser.add_argument("--input", action="append", default=[], metavar="DATASET=MANIFEST", help="Override a retained input; requires a matching baseline for those exact input hashes.")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = run_experiment(
        Path(args.output), baseline_dir=Path(args.baseline), input_manifests=resolve_inputs(args.input),
        source_revision=revision, require_canonical_runtime=True,
    )
    print(
        f"{result['status']}: {len(result['rows'])} cells, "
        f"{result['validation']['target_occurrences']} target occurrences; {args.output}"
    )


if __name__ == "__main__":
    main()
