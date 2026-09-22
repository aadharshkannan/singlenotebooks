"""Source-bound, additive IMDb prefix replay; never regenerate baseline cells.

``run_extension(vectors, labels, output, profile=profile, baseline=baseline)``
consumes the same native cache and complete profile as the frozen engine. Only
new dimensions are ranked/evaluated. The baseline, cache and scientific methods
remain read-only. One writer per output directory is required.

The combined aggregate keeps imdb-sampling-v1. Its ``extension`` record and
aggregate-only ``extension_validation.json`` attest to exact baseline row parity
(except relocated evidence paths), inherited artifact integrity and replay
pairing. Both directories must remain in their recorded relative locations.
The compact aggregate ``extension`` metadata is repeated in the audit, which
adds ``ok`` and the final aggregate SHA256 without a circular aggregate hash.
Finalization rechecks every baseline artifact and input-array hash. Partial
work is explicitly resumable; incompatible or corrupt committed work fails
closed. No embedding or judge interface is used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from sampling_comparison.imdb_experiment import (
    VERSION, _checkpoint, _digest, _load_checkpoint, _method_hashes,
    _positive_integer, _read_record, _runtime, _write_record, array_sha256,
    canonical, cell_metrics, evaluate_replay, normalized_prefix, rank_membership,
    sha256_file, summarize, write_json,
)


EXTENSION_VERSION = "imdb-extension-v1"
DIMENSIONS = (256, 128, 64)
DEFAULT_BASELINE = Path("outputs_imdb") / "private_runs" / "imdb-40-replay"
DEFAULT_OUTPUT = Path("outputs_imdb") / "private_runs" / "imdb-40-replay-extended"
CellKey = tuple[int, str, int, float]


def _cell_key(row: dict[str, Any]) -> CellKey:
    return row["seed"], row["schedule"], row["dimension"], row["rate"]


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def _local_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError(f"baseline artifact escapes its original bundle: {relative}")
    return resolved


def _verify_files(files: dict[Path, str]) -> None:
    for path, expected in files.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"artifact hash mismatch: {path}")


def _input_hashes(vectors: np.ndarray, labels: np.ndarray) -> dict[str, str]:
    return {"vectors": array_sha256(vectors), "labels": array_sha256(labels)}


def _extension_code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (Path(__file__).resolve(), root / "scripts" / "extend_imdb_experiment.py")
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}


def _same_fingerprints(binding: dict[str, Any], vectors: np.ndarray, labels: np.ndarray,
                       profile: dict[str, Any]) -> None:
    for name, current in (
        ("method_code_hashes", _method_hashes()), ("runtime", _runtime()),
        ("dataset", profile), ("input_hashes", _input_hashes(vectors, labels)),
    ):
        if canonical(binding[name]) != canonical(current):
            raise ValueError(f"baseline {name} mismatch; extension requires the exact frozen source/runtime/input")


@dataclass
class _Baseline:
    root: Path
    aggregate: dict[str, Any]
    binding: dict[str, Any]
    binding_sha256: str
    files: dict[Path, str]
    replays: dict[tuple[int, str], dict[str, Any]]


def _replay_arrays(root: Path, seed: int, schedule: str, record: dict[str, Any],
                   source_count: int) -> dict[str, np.ndarray]:
    path = root / "replays" / f"s{seed}-{schedule}.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {"source_id", "order", "timestamps", "occurrence_id"}
    if set(arrays) != required or {k: array_sha256(v) for k, v in arrays.items()} != record["hashes"]:
        raise ValueError("baseline replay array hash/shape mismatch")
    if any(a.shape != (source_count,) for a in arrays.values()):
        raise ValueError("baseline replay count mismatch")
    for name in ("source_id", "order", "occurrence_id"):
        if arrays[name].dtype.kind not in "iu":
            raise ValueError("baseline replay indices must be integers")
    if (not np.array_equal(arrays["occurrence_id"], np.arange(source_count))
            or not np.array_equal(np.sort(arrays["order"]), np.arange(source_count))
            or np.any((arrays["source_id"] < 0) | (arrays["source_id"] >= source_count))
            or not np.isfinite(arrays["timestamps"]).all()
            or np.any(np.diff(arrays["timestamps"]) < 0)):
        raise ValueError("invalid baseline occurrence mapping/order/timestamps")
    return arrays


def _verify_baseline(vectors: np.ndarray, labels: np.ndarray, root: Path,
                     profile: dict[str, Any]) -> _Baseline:
    """Caller must enter the intended threadpool_limits context first."""
    root = root.resolve()
    manifest_path = root / "manifest.json"
    prereg_path = root / "preregistration.json"
    aggregate_path = root / "aggregate.json"
    manifest = _read_record(manifest_path)
    prereg = _read_record(prereg_path)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    binding = prereg["binding"]
    binding_hash = _digest(binding)
    if (manifest["version"] != VERSION or aggregate["version"] != VERSION
            or manifest["status"] != "completed" or aggregate["status"] != "completed"
            or "extension" in aggregate):
        raise ValueError("baseline must be a completed original IMDb engine bundle")
    if any(value != binding_hash for value in (
        prereg["binding_sha256"], manifest["binding_sha256"], aggregate["binding_sha256"],
    )):
        raise ValueError("baseline binding fingerprint mismatch")
    if (canonical(aggregate["provenance"]) != canonical(binding)
            or canonical(aggregate["dataset"]) != canonical(binding["dataset"])
            or canonical(aggregate["protocol"]) != canonical(binding["protocol"])):
        raise ValueError("baseline aggregate differs from its preregistered source binding")
    _same_fingerprints(binding, vectors, labels, profile)
    protocol = aggregate["protocol"]
    seeds, schedules, dimensions, rates = (
        protocol[key] for key in ("seeds", "schedules", "dimensions", "rates")
    )
    if (not all((seeds, schedules, dimensions, rates))
            or any(len(set(values)) != len(values) for values in (seeds, schedules, dimensions, rates))
            or dimensions[0] != 1536 or protocol["repetitions"] != len(seeds)
            or protocol["source_count"] != len(labels)
            or protocol["occurrences_per_replay"] != len(labels)):
        raise ValueError("invalid baseline protocol grid")
    expected = {(s, a, d, r) for s in seeds for a in schedules for d in dimensions for r in rates}
    rows = aggregate["rows"]
    if (len(rows) != len(expected) or {_cell_key(row) for row in rows} != expected
            or manifest["completed_cells"] != len(expected) or protocol["planned_cells"] != len(expected)):
        raise ValueError("baseline does not contain its complete unique cell grid")
    files = {_local_file(root, name): digest for name, digest in manifest["files"].items()}
    if len(files) != len(manifest["files"]) or not {aggregate_path, prereg_path} <= files.keys():
        raise ValueError("baseline manifest has aliases or missing required artifacts")
    files[manifest_path] = sha256_file(manifest_path)
    _verify_files(files)

    def checkpoint(relative: Path, identity: dict[str, Any]) -> dict[str, Any]:
        path = root / relative
        if path not in files or path.with_suffix(".npz") not in files:
            raise ValueError(f"baseline checkpoint is not manifest-bound: {relative}")
        record = _read_record(path)
        if (record["binding_sha256"] != binding_hash or record["identity"] != identity
                or record["evidence_sha256"] != files[path.with_suffix(".npz")]):
            raise ValueError(f"baseline checkpoint binding mismatch: {relative}")
        return record

    replays = {}
    memberships = {}
    for seed in seeds:
        draw_hash = None
        for schedule in schedules:
            identity = {"kind": "replay", "seed": seed, "schedule": schedule}
            record = checkpoint(Path("replays") / f"s{seed}-{schedule}.json", identity)
            _replay_arrays(root, seed, schedule, record, len(labels))
            if draw_hash is not None and record["hashes"]["source_id"] != draw_hash:
                raise ValueError("baseline schedules do not share the frozen bootstrap draw")
            draw_hash = record["hashes"]["source_id"]
            replays[seed, schedule] = record
            for dimension in dimensions:
                rank_identity = {**identity, "kind": "membership", "dimension": dimension}
                memberships[seed, schedule, dimension] = checkpoint(
                    Path("memberships") / f"s{seed}-{schedule}-d{dimension}.json", rank_identity,
                )
    for row in rows:
        seed, schedule, dimension, rate = _cell_key(row)
        budget = max(1, math.floor(len(labels) * rate))
        relative = Path("cells") / f"s{seed}-{schedule}-d{dimension}-r{rates.index(rate)}.json"
        record = checkpoint(relative, {
            "kind": "cell", "seed": seed, "schedule": schedule, "dimension": dimension,
            "rate": rate, "budget": budget,
        })
        membership = memberships[seed, schedule, dimension]
        if (row["budget"] != budget
                or row["replay_hashes"] != replays[seed, schedule]["hashes"]
                or membership["replay_hashes"] != row["replay_hashes"]
                or membership["ranking_sha256"] != row["ranking_sha256"]
                or membership["selector_counts"] != row["selector_counts"]
                or row["evidence_sha256"] != record["evidence_sha256"]
                or _local_file(root, row["evidence"]) != (root / relative).with_suffix(".npz")
                or canonical(record["row"]) != canonical({
                    k: v for k, v in row.items() if k not in ("paired_delta_native", "evidence_sha256")
                })):
            raise ValueError("baseline measured row differs from its retained checkpoint/replay/membership")
    return _Baseline(root, aggregate, binding, binding_hash, files, replays)


def _progress(path: Path, event: str, **fields: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical({
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **fields,
        }) + "\n")


def _parity_and_pairing(rows: list[dict[str, Any]], source: _Baseline, output: Path,
                        dimensions: tuple[int, ...]) -> None:
    protocol = source.aggregate["protocol"]
    old = {_cell_key(row): row for row in source.aggregate["rows"]}
    current = {_cell_key(row): row for row in rows}
    added = {(s, a, d, r) for s in protocol["seeds"] for a in protocol["schedules"]
             for d in dimensions for r in protocol["rates"]}
    if len(rows) != len(current) or current.keys() != old.keys() | added:
        raise ValueError("combined row grid has missing, extra or duplicate cells")
    for key, original in old.items():
        row = current[key]
        if canonical({k: v for k, v in row.items() if k != "evidence"}) != canonical({
            k: v for k, v in original.items() if k != "evidence"
        }):
            raise ValueError("baseline row parity failed; metrics or metadata changed")
        if (output / row["evidence"]).resolve() != (source.root / original["evidence"]).resolve():
            raise ValueError("baseline evidence path does not reference original artifact")
    for row in rows:
        if row["replay_hashes"] != source.replays[row["seed"], row["schedule"]]["hashes"]:
            raise ValueError("extension replay pairing differs from baseline")


def run_extension(
    vectors: np.ndarray, labels: np.ndarray, output: Path = DEFAULT_OUTPUT, *,
    profile: dict[str, Any], baseline: Path = DEFAULT_BASELINE,
    dimensions: Sequence[int] = DIMENSIONS, resume: bool = False, blas_threads: int = 4,
) -> dict[str, Any]:
    """Extend only disjoint prefixes; reuse every baseline seed/schedule/rate.

    All scientific configuration (including block sizes and reservoir capacity)
    is copied from the verified baseline. ``blas_threads`` defaults to four and
    must reproduce the recorded runtime exactly. No seed/budget overrides,
    draws, embeddings, judgments or baseline algorithms are run.
    """
    dimensions = tuple(dimensions)
    if (not dimensions or len(set(dimensions)) != len(dimensions)
            or any(isinstance(d, bool) or not isinstance(d, int) or not 1 <= d < 1536 for d in dimensions)):
        raise ValueError("new dimensions must be unique integer prefixes in [1,1535]")
    dimensions = tuple(sorted(dimensions, reverse=True))
    blas_threads = _positive_integer(blas_threads, "blas_threads")
    baseline, output = Path(baseline).resolve(), Path(output).resolve()
    if output.is_relative_to(baseline) or baseline.is_relative_to(output):
        raise ValueError("extension output and baseline must be separate, non-nested directories")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("nonempty extension output requires explicit resume=True")
    with threadpool_limits(limits=blas_threads):
        source = _verify_baseline(vectors, labels, baseline, profile)
        if set(dimensions) & set(source.aggregate["protocol"]["dimensions"]):
            raise ValueError("new dimensions overlap baseline dimensions")
        return _extend(vectors, labels, output, source, dimensions, resume)


def _extend(vectors: np.ndarray, labels: np.ndarray, output: Path, source: _Baseline,
            dimensions: tuple[int, ...], resume: bool) -> dict[str, Any]:
    original = source.aggregate
    protocol = {
        **original["protocol"],
        "dimensions": sorted(original["protocol"]["dimensions"] + list(dimensions), reverse=True),
    }
    added_count = len(dimensions) * len(protocol["seeds"]) * len(protocol["schedules"]) * len(protocol["rates"])
    protocol["planned_cells"] = len(original["rows"]) + added_count
    code_hashes = _extension_code_hashes()
    extension = {
        "version": EXTENSION_VERSION,
        "baseline_directory": _relative(source.root, output),
        "baseline_aggregate_sha256": source.files[source.root / "aggregate.json"],
        "baseline_manifest_sha256": source.files[source.root / "manifest.json"],
        "baseline_binding_sha256": source.binding_sha256,
        "reused_dimensions": original["protocol"]["dimensions"], "added_dimensions": list(dimensions),
        "reused_cells": len(original["rows"]), "added_cells": added_count,
        "extension_code_sha256": sha256_file(Path(__file__)),
        "extension_code_hashes": code_hashes,
        "method_code_hashes_sha256": _digest(source.binding["method_code_hashes"]),
        "runtime_sha256": _digest(source.binding["runtime"]),
        "input_hashes": source.binding["input_hashes"],
    }
    binding_payload = {**source.binding, "protocol": protocol, "extension": extension}
    binding = _digest(binding_payload)
    prereg = output / "preregistration.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not prereg.is_file():
            raise ValueError("cannot resume extension without matching preregistration")
        previous = _read_record(prereg)
        if previous["binding_sha256"] != binding or canonical(previous["binding"]) != canonical(binding_payload):
            raise ValueError("incompatible extension resume fingerprints/configuration")
    else:
        # Validate only the requested new prefixes, never regenerate the native
        # or previously measured representations.
        for dimension in dimensions:
            normalized_prefix(vectors, dimension)
        output.mkdir(parents=True, exist_ok=True)
        _write_record(prereg, {
            "binding_sha256": binding, "binding": binding_payload,
            "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = _read_record(manifest_path)
        if (manifest["binding_sha256"] != binding or manifest["status"] != "completed"
                or manifest["completed_cells"] != protocol["planned_cells"]):
            raise ValueError("incompatible extension final manifest")
        _verify_files({(output / name).resolve(): digest for name, digest in manifest["files"].items()})
        aggregate = json.loads((output / "aggregate.json").read_text(encoding="utf-8"))
        if (aggregate["binding_sha256"] != binding or aggregate["protocol"] != protocol
                or aggregate["status"] != "completed"):
            raise ValueError("incompatible extension aggregate")
        _parity_and_pairing(aggregate["rows"], source, output, dimensions)
        return aggregate

    log = output / "progress.jsonl"
    _progress(log, "extension_started", added_cells=added_count, reused_cells=len(original["rows"]))
    retained = {prereg, log}
    rows = [{**row, "evidence": _relative(source.root / row["evidence"], output)} for row in original["rows"]]
    completed = 0
    for seed in protocol["seeds"]:
        unit_ids = tuple(f"seed-{seed}:occ-{i:012d}" for i in range(len(labels)))
        for schedule in protocol["schedules"]:
            started = time.perf_counter()
            computed = reused = 0
            _progress(log, "replay_started", seed=seed, schedule=schedule, completed_new_cells=completed)
            replay_record = source.replays[seed, schedule]
            replay = _replay_arrays(source.root, seed, schedule, replay_record, len(labels))
            sources, order, times = replay["source_id"], replay["order"], replay["timestamps"]
            for dimension in dimensions:
                identity = {"kind": "membership", "seed": seed, "schedule": schedule, "dimension": dimension}
                rank_path = output / "memberships" / f"s{seed}-{schedule}-d{dimension}.json"
                rank_record = _load_checkpoint(rank_path, binding, identity)
                prefix = None
                if rank_record is None:
                    prefix = normalized_prefix(vectors, dimension)
                    ranking, selector_counts = rank_membership(
                        unit_ids, ("imdb",) * len(sources), (("imdb-review",),) * len(sources),
                        prefix[sources], order, times, seed,
                    )
                    rank_record = _checkpoint(rank_path, binding, identity, {"ranking": ranking}, {
                        "ranking_sha256": array_sha256(ranking), "selector_counts": selector_counts,
                        "replay_hashes": replay_record["hashes"], "label_blind": True,
                    })
                with np.load(rank_path.with_suffix(".npz"), allow_pickle=False) as archive:
                    ranking = archive["ranking"]
                if (rank_record["replay_hashes"] != replay_record["hashes"]
                        or rank_record["ranking_sha256"] != array_sha256(ranking)
                        or not np.array_equal(np.sort(ranking), np.arange(len(sources)))):
                    raise ValueError("extension membership ranking/pairing mismatch")
                retained.update((rank_path, rank_path.with_suffix(".npz")))
                for rate_index, rate in enumerate(protocol["rates"]):
                    budget = max(1, math.floor(len(sources) * rate))
                    selected = ranking[:budget]
                    cell_path = output / "cells" / f"s{seed}-{schedule}-d{dimension}-r{rate_index}.json"
                    cell_identity = {**identity, "kind": "cell", "rate": rate, "budget": budget}
                    cell = _load_checkpoint(cell_path, binding, cell_identity)
                    if cell is None:
                        cell_started = time.perf_counter()
                        if prefix is None:
                            prefix = normalized_prefix(vectors, dimension)
                        evidence = evaluate_replay(
                            prefix, labels, sources, order, selected, seed=seed,
                            target_block_size=protocol["distance_blocks"]["targets"],
                            donor_block_size=protocol["distance_blocks"]["donors"],
                            calibration_reservoir_size=protocol["lipschitz"]["reservoir_size"],
                        )
                        row = {
                            "dimension": dimension, "seed": seed, "schedule": schedule,
                            "rate": rate, "budget": budget,
                            **cell_metrics(evidence, labels[sources[selected]]),
                            "replay_hashes": replay_record["hashes"],
                            "ranking_sha256": rank_record["ranking_sha256"],
                            "membership_sha256": array_sha256(np.sort(selected)),
                            "selector_counts": rank_record["selector_counts"],
                            "evidence": _relative(cell_path.with_suffix(".npz"), output),
                            "elapsed_seconds": time.perf_counter() - cell_started,
                        }
                        cell = _checkpoint(cell_path, binding, cell_identity, evidence, {"row": row})
                        computed += 1
                    else:
                        reused += 1
                    row = {**cell["row"], "evidence_sha256": cell["evidence_sha256"]}
                    if (_cell_key(row) != (seed, schedule, dimension, rate)
                            or row["budget"] != budget
                            or row["replay_hashes"] != replay_record["hashes"]
                            or row["ranking_sha256"] != rank_record["ranking_sha256"]
                            or row["membership_sha256"] != array_sha256(np.sort(selected))
                            or row["selector_counts"] != rank_record["selector_counts"]
                            or row["evidence"] != _relative(cell_path.with_suffix(".npz"), output)):
                        raise ValueError("extension cell row differs from its membership/replay")
                    rows.append(row)
                    completed += 1
                    retained.update((cell_path, cell_path.with_suffix(".npz")))
            _progress(log, "replay_completed", seed=seed, schedule=schedule,
                      computed_cells=computed, reused_cells=reused, completed_new_cells=completed,
                      elapsed_seconds=time.perf_counter() - started)

    if completed != added_count:
        raise AssertionError("cannot finalize an incomplete extension")
    summaries = summarize(rows)
    _parity_and_pairing(rows, source, output, dimensions)
    _verify_files(source.files)
    _same_fingerprints(source.binding, vectors, labels, source.binding["dataset"])
    if _extension_code_hashes() != code_hashes:
        raise ValueError("extension code changed during execution")
    proven = {
        "baseline_rows_unchanged": True, "replay_pairing_exact": True, "embedding_calls": 0,
    }
    extension = {**extension, **proven}
    public_extension = {
        **{key: extension[key] for key in (
            "baseline_aggregate_sha256", "baseline_manifest_sha256",
            "reused_cells", "added_cells", "reused_dimensions", "added_dimensions",
        )},
        **proven,
    }
    audit = {
        "version": "imdb-extension-validation-v1", "status": "passed", "ok": True,
        **public_extension,
        "baseline_artifacts_unchanged": True, "baseline_artifacts_checked": len(source.files),
        "input_arrays_unchanged": True, "method_code_hashes_unchanged": True, "runtime_unchanged": True,
        "baseline_before": {
            "aggregate_sha256": extension["baseline_aggregate_sha256"],
            "manifest_sha256": extension["baseline_manifest_sha256"],
        },
        "baseline_after": {
            "aggregate_sha256": sha256_file(source.root / "aggregate.json"),
            "manifest_sha256": sha256_file(source.root / "manifest.json"),
        },
        "combined_cells": len(rows),
        "replay_count": len(source.replays), "repetitions": protocol["repetitions"],
        "rates_per_replay": len(protocol["rates"]), "source_count": protocol["source_count"],
        "extension_code_sha256": extension["extension_code_sha256"],
    }
    aggregate = {
        "version": VERSION, "status": "completed", "dataset": original["dataset"],
        "protocol": protocol, "binding_sha256": binding, "provenance": binding_payload,
        "rows": rows, "summaries": summaries, "extension": public_extension,
    }
    aggregate_path = output / "aggregate.json"
    write_json(aggregate_path, aggregate)
    audit_path = output / "extension_validation.json"
    write_json(audit_path, {**audit, "aggregate_sha256": sha256_file(aggregate_path)})
    _progress(log, "extension_completed", computed_grid_cells=completed, combined_cells=len(rows))
    retained.update((audit_path, aggregate_path))
    files = {_relative(p, output): sha256_file(p) for p in sorted(retained)}
    files.update({_relative(p, output): digest for p, digest in source.files.items()})
    _write_record(manifest_path, {
        "version": VERSION, "status": "completed", "binding_sha256": binding,
        "completed_cells": len(rows), "files": files, "extension": extension,
    })
    return aggregate
