from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_experiment import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach local input readiness and report evidence without changing measured cells.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--authentication-blocker", default="")
    args = parser.parse_args()
    root, inputs = Path(args.run), Path(args.input_root)
    aggregate_path = root / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    readiness = []
    for profile in aggregate["datasets"]:
        dataset = profile["dataset_id"]
        if profile["status"] == "completed":
            source = inputs / dataset / "manifest.json"
            if not source.exists():
                source = Path(profile["provenance"]["input_manifest"])
        else:
            source = inputs / dataset / "readiness.json"
        if source.exists():
            prepared = json.loads(source.read_text(encoding="utf-8"))
            readiness.append({"dataset_id": dataset, "source": str(source),
                              "sha256": sha256_file(source), "preparation": prepared})
            if profile["status"] != "completed":
                profile.update(
                    n=prepared["sessions"], agents=prepared["agents"],
                    positive_count=prepared["positive_count"],
                    pass_rate=prepared["positive_count"] / prepared["sessions"] if prepared["positive_count"] is not None else None,
                    provenance=prepared, source_hashes=prepared["source_hashes"],
                )
        if profile["status"] != "completed":
            if source.exists() and prepared.get("status") == "blocked_expected_labels":
                profile["reason"] = prepared["reason"]
            elif args.authentication_blocker:
                profile["reason"] = args.authentication_blocker
    readiness_path = root / "input_readiness.json"
    write_json(readiness_path, {
        "datasets": readiness, "authentication_blocker": args.authentication_blocker,
        "boundary": "Prepared source/label counts are not measured sampling accuracy. Live embedding preparation is authorized; no new judge calls. Successful completed-dataset embedding totals are recorded in each preparation manifest.",
    })
    aggregate["files"]["input_readiness"] = str(readiness_path)
    write_json(aggregate_path, aggregate)
    for name, path in aggregate["files"].items():
        manifest["files"][name] = {"path": path, "sha256": sha256_file(Path(path))}
    for name in (
        "report.html", "scientific_validation.json", "browser_validation.json",
        "visual_validation.json", "protocol_parity.json", "environment-requirements.txt",
        "unit_tests.xml", "cache_reuse_validation.json",
    ):
        path = root / name
        if path.exists():
            manifest["files"][name] = {"path": str(path), "sha256": sha256_file(path)}
    manifest["source_revision_note"] = "Git HEAD at execution; new experiment code was uncommitted then. Exact executed implementation hashes, not HEAD alone, identify the run."
    manifest["artifact_generator_hashes"] = {
        str(path): sha256_file(path) for path in (
            Path("sampling_comparison/matryoshka_report.py"),
            Path("sampling_comparison/matryoshka_inputs.py"),
            Path("scripts/prepare_matryoshka_input.py"),
            Path("scripts/prepare_matryoshka_live_input.py"),
            Path("scripts/inspect_matryoshka_tau2.py"),
            Path("scripts/build_matryoshka_report.py"),
            Path("scripts/finalize_matryoshka_run.py"),
            Path("scripts/validate_matryoshka_run.py"),
        )
    }
    write_json(manifest_path, manifest)
    print("Readiness and retained artifact hashes attached; measured result rows unchanged.")


if __name__ == "__main__":
    main()
