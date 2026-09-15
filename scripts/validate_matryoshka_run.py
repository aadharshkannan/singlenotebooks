from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_experiment import canonical, sha256_file, summarize, write_json


def validate(root: Path) -> dict:
    aggregate_path = root / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"].values():
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"artifact hash mismatch: {record['path']}")
    for path, digest in {**manifest["code_hashes"], **manifest.get("artifact_generator_hashes", {})}.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"executed source differs from current source: {path}")
    rows = aggregate["rows"]
    protocol = aggregate["protocol"]
    completed = {r["dataset_id"]: r for r in aggregate["datasets"] if r["status"] == "completed"}
    expected = len(completed)
    for name in ("dimensions", "seeds", "schedules", "rates", "modes"):
        expected *= len(protocol[name])
    key_fields = ("dataset_id", "dimension", "seed", "schedule", "rate", "mode")
    by_key = {tuple(row[name] for name in key_fields): row for row in rows}
    if len(rows) != expected or len(by_key) != expected:
        raise ValueError("result grid has missing or duplicate cells")
    for row in rows:
        n = completed[row["dataset_id"]]["n"]
        selected = max(1, math.floor(n * row["rate"]))
        if row["selected_count"] != selected or row["unjudged_count"] != n - selected:
            raise ValueError("sample cap/target denominator mismatch")
        if sum(row["provenance_counts"].values()) != n - selected:
            raise ValueError("IDW/fallback counts do not sum to unjudged targets")
        if row["representation_fallbacks"] != 0:
            raise ValueError("representation fallback contaminated experiment")
        if row["tp"] + row["tn"] + row["fp"] + row["fn"] != n - selected:
            raise ValueError("confusion matrix denominator mismatch")
        np.testing.assert_allclose(row["accuracy"], (row["tp"] + row["tn"]) / (n - selected), atol=1e-14)
        np.testing.assert_allclose(row["combined_accuracy"], (selected + row["accuracy"] * (n - selected)) / n, atol=1e-14)
        native_key = tuple(1536 if name == "dimension" else "end_to_end" if name == "mode" else row[name] for name in key_fields)
        np.testing.assert_allclose(row["accuracy_delta_native"], row["accuracy"] - by_key[native_key]["accuracy"], atol=1e-14)
    membership_keys = set()
    replay_orders = {}
    with gzip.open(root / "memberships.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            membership = json.loads(line)
            key = tuple(membership[name] for name in key_fields)
            if key not in by_key or key in membership_keys:
                raise ValueError("missing or duplicate membership key")
            membership_keys.add(key)
            row = by_key[key]
            replay_key = (row["dataset_id"], row["seed"], row["schedule"])
            order_hash = membership["order_sha256"]
            if replay_key in replay_orders and replay_orders[replay_key] != order_hash:
                raise ValueError("arrival order is not paired across dimensions, modes or budgets")
            replay_orders[replay_key] = order_hash
            indices = membership["selected_indices"]
            n = completed[row["dataset_id"]]["n"]
            if len(indices) != row["selected_count"] or len(set(indices)) != len(indices) or not set(indices).issubset(range(n)):
                raise ValueError("invalid selected row indices")
            digest = hashlib.sha256(canonical(sorted(indices)).encode()).hexdigest()
            if digest != row["membership_sha256"] or digest != membership["membership_sha256"]:
                raise ValueError("selected membership checksum mismatch")
            if row["mode"] == "fixed_membership":
                native_key = tuple(1536 if name == "dimension" else "end_to_end" if name == "mode" else row[name] for name in key_fields)
                if digest != by_key[native_key]["membership_sha256"]:
                    raise ValueError("fixed membership differs from native")
    if membership_keys != set(by_key):
        raise ValueError("membership grid is incomplete")
    unique_order_counts = {}
    for dataset_id in completed:
        unique_order_counts[dataset_id] = {}
        for schedule in protocol["schedules"]:
            orders = {digest for (dataset, _, kind), digest in replay_orders.items()
                      if dataset == dataset_id and kind == schedule}
            unique_order_counts[dataset_id][schedule] = len(orders)
            if len(orders) != len(protocol["seeds"]):
                raise ValueError("configured seeds did not produce distinct randomized source orders")
    summary, decisions = summarize(rows, protocol["accuracy_tolerance"])
    if summary != aggregate["summary"] or decisions != aggregate["decisions"]:
        raise ValueError("summary/decision does not match machine-readable cells")
    return {
        "status": "passed", "aggregate_sha256": sha256_file(aggregate_path),
        "result_rows": expected, "membership_records": len(membership_keys),
        "unique_orders_per_dataset_schedule": unique_order_counts,
        "checks": [
            "artifact and controlling source hashes", "complete unique configured grid",
            "budget/target/provenance/confusion denominators", "combined versus unjudged-only accuracy",
            "paired native deltas", "membership indices and checksums", "fixed-membership control",
            "summary and candidate recomputation", "distinct randomized orders and dimension/budget/mode pairing",
        ],
        "scope": "Artifact consistency, not independent production validation or proof of zero accuracy loss.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    result = validate(root)
    write_json(root / "scientific_validation.json", result)
    print(canonical(result))


if __name__ == "__main__":
    main()
