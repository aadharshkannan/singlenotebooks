from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_experiment import load_input, sha256_file, write_json


KEY_FIELDS = ("dataset_id", "dimension", "seed", "schedule", "rate", "mode")


def _keyed(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    result = {tuple(row[field] for field in KEY_FIELDS): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate replay cell")
    return result


def compare_aggregates(
    current: Mapping[str, Any], baseline: Mapping[str, Any], *,
    refreshed_datasets: tuple[str, ...] = (),
) -> dict[str, Any]:
    for aggregate in (current, baseline):
        if aggregate["version"] != "matryoshka-cutoff-v1":
            raise ValueError("comparison requires matching Matryoshka experiment versions")
    new_protocol = {key: value for key, value in current["protocol"].items() if key != "dimensions"}
    old_protocol = {key: value for key, value in baseline["protocol"].items() if key != "dimensions"}
    if new_protocol != old_protocol:
        raise ValueError("protocol changed beyond the requested dimensions")
    if current.get("environment") != baseline.get("environment"):
        raise ValueError("numerical runtime differs from baseline")
    profiles = [
        {row["dataset_id"]: row for row in aggregate["datasets"] if row["status"] == "completed"}
        for aggregate in (current, baseline)
    ]
    if profiles[0].keys() != profiles[1].keys():
        raise ValueError("completed datasets differ from baseline")
    refreshed = set(refreshed_datasets)
    if len(refreshed) != len(refreshed_datasets) or not refreshed.issubset(profiles[0]):
        raise ValueError("refreshed datasets must be unique completed cohorts")
    unchanged = set(profiles[0]) - refreshed
    if not unchanged:
        raise ValueError("at least one unchanged dataset is required for parity comparison")
    refresh_records = {}
    for dataset, profile in profiles[0].items():
        previous = profiles[1][dataset]
        if dataset in refreshed:
            if profile["provenance"]["input_manifest_sha256"] == previous["provenance"]["input_manifest_sha256"]:
                raise ValueError("declared refreshed dataset has an unchanged input manifest")
            refresh_records[dataset] = {
                "previous_units": previous["n"], "current_units": profile["n"],
                "previous_positive_count": previous["positive_count"], "current_positive_count": profile["positive_count"],
                "previous_input_sha256": previous["provenance"]["input_manifest_sha256"],
                "current_input_sha256": profile["provenance"]["input_manifest_sha256"],
                "boundary": "Changed population: exact-result parity is not asserted.",
            }
            continue
        for field in ("n", "agents", "positive_count", "pass_rate", "source_hashes"):
            if profile[field] != previous[field]:
                raise ValueError(f"dataset profile changed: {dataset}/{field}")
        if profile["provenance"]["input_manifest_sha256"] != previous["provenance"]["input_manifest_sha256"]:
            raise ValueError(f"input manifest changed: {dataset}")
    common = sorted(set(current["protocol"]["dimensions"]) & set(baseline["protocol"]["dimensions"]))
    if 1536 not in common:
        raise ValueError("native reference missing")
    new_rows, old_rows = (_keyed(aggregate["rows"]) for aggregate in (current, baseline))
    shared = {key: row for key, row in new_rows.items() if row["dimension"] in common and row["dataset_id"] in unchanged}
    expected = {key: row for key, row in old_rows.items() if row["dimension"] in common and row["dataset_id"] in unchanged}
    if shared.keys() != expected.keys() or not shared:
        raise ValueError("overlapping replay grid is incomplete")
    for key, row in shared.items():
        if row != expected[key]:
            raise ValueError(f"overlapping measured cell changed: {key}")
    return {
        "status": "passed",
        "baseline_run": baseline["run_id"],
        "current_run": current["run_id"],
        "only_protocol_change": "dimensions" if current["protocol"]["dimensions"] != baseline["protocol"]["dimensions"] else "none",
        "current_dimensions": current["protocol"]["dimensions"],
        "identical_overlap_dimensions": common,
        "identical_overlap_result_cells": len(shared),
        "input_manifests_identical": sorted(unchanged),
        "refreshed_datasets": refresh_records,
        "comparison_tolerance": 0,
        "runtime_identical": current.get("environment"),
    }


def validate_comparison(
    run: Path, baseline: Path, *, refreshed_datasets: tuple[str, ...] = (),
) -> dict[str, Any]:
    aggregates, manifests = [], []
    for root in (run, baseline):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for key, filename in (("aggregate", "aggregate.json"), ("memberships", "memberships.jsonl.gz")):
            if sha256_file(root / filename) != manifest["files"][key]["sha256"]:
                raise ValueError(f"source artifact hash mismatch: {root / filename}")
        aggregates.append(json.loads((root / "aggregate.json").read_text(encoding="utf-8")))
        manifests.append(manifest)
    result = compare_aggregates(*aggregates, refreshed_datasets=refreshed_datasets)
    code_hashes = [
        {Path(path.replace("\\", "/")).name: digest for path, digest in manifest["code_hashes"].items()}
        for manifest in manifests
    ]
    if code_hashes[0] != code_hashes[1]:
        raise ValueError("controlling experiment implementation changed")
    memberships = []
    for root in (run, baseline):
        with gzip.open(root / "memberships.jsonl.gz", "rt", encoding="utf-8") as stream:
            records = [
                row for line in stream
                if (row := json.loads(line))["dimension"] in result["identical_overlap_dimensions"]
                and row["dataset_id"] in result["input_manifests_identical"]
            ]
        memberships.append(_keyed(records))
    if memberships[0] != memberships[1]:
        raise ValueError("overlapping membership or order evidence changed")
    result["identical_overlap_memberships"] = len(memberships[0])
    result["controlling_code_hashes_unchanged"] = code_hashes[0]
    result["verified_cache_hashes"] = {}
    for profile in aggregates[0]["datasets"]:
        if profile["status"] != "completed":
            continue
        loaded = load_input(Path(profile["provenance"]["input_manifest"]))
        if loaded.profile["input_manifest_sha256"] != profile["provenance"]["input_manifest_sha256"]:
            raise ValueError("cached input changed during the experiment")
        result["verified_cache_hashes"][loaded.dataset_id] = {
            "manifest": loaded.profile["input_manifest_sha256"], **loaded.profile["hashes"],
        }
    result["artifact_hashes"] = {
        label: {name: sha256_file(root / name) for name in ("aggregate.json", "memberships.jsonl.gz")}
        for label, root in (("current", run), ("baseline", baseline))
    }
    result["boundary"] = "Exact retained-input and overlapping-cell reproducibility, not independent-label or production validation."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a dimension-only rerun against its exact retained baseline.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--refreshed-dataset", action="append", default=[], help="Explicit changed-input cohort excluded from exact-result parity; other cohorts must remain identical.")
    args = parser.parse_args()
    result = validate_comparison(args.run, args.baseline, refreshed_datasets=tuple(args.refreshed_dataset))
    write_json(args.run / "protocol_parity.json", result)
    print(f"Passed: {result['identical_overlap_result_cells']} identical overlapping cells and memberships.")


if __name__ == "__main__":
    main()
