from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.idw_threshold_experiment import refresh_manifest, sha256_file, write_json
from sampling_comparison.idw_threshold_report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standalone IDW threshold/envelope HTML report.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    aggregate = json.loads(input_path.read_text(encoding="utf-8"))
    aggregate["files"]["report"] = str(output_path)
    aggregate["code_hashes"]["sampling_comparison/idw_threshold_report.py"] = sha256_file(
        Path("sampling_comparison/idw_threshold_report.py")
    )
    aggregate["code_hashes"]["scripts/build_idw_threshold_report.py"] = sha256_file(Path(__file__))
    write_json(input_path, aggregate)
    html = build_report(aggregate, str(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    refresh_manifest(input_path)
    print(str(output_path))


if __name__ == "__main__":
    main()
