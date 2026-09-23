"""Parallel scheduling of an existing IMDb extension, without changing its methods."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Iterator

import numpy as np
from threadpoolctl import threadpool_limits

from sampling_comparison import imdb_extension as extension
from sampling_comparison.imdb_experiment import (
    _checkpoint, _digest, _load_checkpoint, _positive_integer, _read_record,
    _write_record, array_sha256, canonical, cell_metrics, evaluate_replay,
    normalized_prefix, rank_membership, sha256_file, write_json,
)
from sampling_comparison.imdb_inputs import load_input


VERSION = "imdb-parallel-execution-v1"
Job = tuple[int, str]


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """OS-owned locks release on process exit; a leftover file is not a lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"another coordinator or worker owns {path.name}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), root / "scripts" / "run_imdb_parallel.py"]
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in files}


def _job_lock(output: Path, job: Job) -> Path:
    seed, schedule = job
    return output / "execution_locks" / f"s{seed}-{schedule}.lock"


def _verify_registration(source: extension._Baseline, output: Path) -> tuple[dict, tuple[int, ...]]:
    registration = _read_record(output / "preregistration.json")
    binding = registration["binding"]
    if registration["binding_sha256"] != _digest(binding):
        raise ValueError("extension preregistration binding mismatch")
    recorded = binding["extension"]
    dimensions = tuple(recorded["added_dimensions"])
    if (not dimensions or len(set(dimensions)) != len(dimensions)
            or tuple(sorted(dimensions, reverse=True)) != dimensions
            or any(isinstance(d, bool) or not isinstance(d, int) or not 1 <= d < 1536 for d in dimensions)
            or set(dimensions) & set(source.aggregate["protocol"]["dimensions"])):
        raise ValueError("extension added dimension grid is invalid")
    protocol = {
        **source.aggregate["protocol"],
        "dimensions": sorted(source.aggregate["protocol"]["dimensions"] + list(dimensions), reverse=True),
    }
    new_cells = len(dimensions) * len(protocol["seeds"]) * len(protocol["schedules"]) * len(protocol["rates"])
    protocol["planned_cells"] = len(source.aggregate["rows"]) + new_cells
    expected_extension = {
        "version": extension.EXTENSION_VERSION,
        "baseline_directory": extension._relative(source.root, output),
        "baseline_aggregate_sha256": source.files[source.root / "aggregate.json"],
        "baseline_manifest_sha256": source.files[source.root / "manifest.json"],
        "baseline_binding_sha256": source.binding_sha256,
        "reused_dimensions": source.aggregate["protocol"]["dimensions"], "added_dimensions": list(dimensions),
        "reused_cells": len(source.aggregate["rows"]), "added_cells": new_cells,
        "extension_code_sha256": sha256_file(Path(extension.__file__)),
        "extension_code_hashes": extension._extension_code_hashes(),
        "method_code_hashes_sha256": _digest(source.binding["method_code_hashes"]),
        "runtime_sha256": _digest(source.binding["runtime"]),
        "input_hashes": source.binding["input_hashes"],
    }
    expected_binding = {**source.binding, "protocol": protocol, "extension": expected_extension}
    if canonical(binding) != canonical(expected_binding):
        raise ValueError("parallel scheduling requires the unchanged original extension binding")
    return registration, dimensions


def _inventory(output: Path, registration: dict, dimensions: tuple[int, ...]) -> dict:
    """Verify committed checkpoints and choose disjoint incomplete replay groups."""
    protocol = registration["binding"]["protocol"]
    binding = registration["binding_sha256"]
    artifacts: dict[str, str] = {}
    pending: list[Job] = []
    cells = 0
    for seed in protocol["seeds"]:
        for schedule in protocol["schedules"]:
            missing = False
            for dimension in dimensions:
                identity = {"kind": "membership", "seed": seed, "schedule": schedule, "dimension": dimension}
                rank_path = output / "memberships" / f"s{seed}-{schedule}-d{dimension}.json"
                rank = _load_checkpoint(rank_path, binding, identity)
                if rank is None:
                    missing = True
                else:
                    for path in (rank_path, rank_path.with_suffix(".npz")):
                        artifacts[extension._relative(path, output)] = sha256_file(path)
                for index, rate in enumerate(protocol["rates"]):
                    budget = max(1, int(protocol["source_count"] * rate))
                    path = output / "cells" / f"s{seed}-{schedule}-d{dimension}-r{index}.json"
                    cell = _load_checkpoint(path, binding, {**identity, "kind": "cell", "rate": rate, "budget": budget})
                    if cell is None:
                        missing = True
                    else:
                        if rank is None:
                            raise ValueError("committed cell has no committed membership")
                        cells += 1
                        for artifact in (path, path.with_suffix(".npz")):
                            artifacts[extension._relative(artifact, output)] = sha256_file(artifact)
            if missing:
                pending.append((seed, schedule))
    expected_json = set(artifacts)
    for directory in ("cells", "memberships"):
        for path in (output / directory).glob("*.json"):
            if extension._relative(path, output) not in expected_json:
                raise ValueError("unexpected committed checkpoint outside the registered grid")
    return {"files": artifacts, "pending": pending, "completed_cells": cells}


def _transition(output: Path, registration: dict, inventory: dict, workers: int) -> dict:
    path = output / "parallel_transition.json"
    execution = {
        "version": VERSION, "scientific_binding_sha256": registration["binding_sha256"],
        "preregistration_sha256": sha256_file(output / "preregistration.json"),
        "scheduler_code_hashes": _code_hashes(), "workers": workers, "blas_threads_per_worker": 4,
        "job_unit": "exclusive seed/schedule; dimensions and budgets sequential within each group",
    }
    if path.exists():
        transition = _read_record(path)
        if transition["execution"] != execution:
            raise ValueError("parallel resume execution fingerprint changed")
        extension._verify_files({output / name: digest for name, digest in transition["preserved_files"].items()})
    else:
        transition = {
            "execution": execution, "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preserved_cells": inventory["completed_cells"], "preserved_files": inventory["files"],
        }
        _write_record(path, transition)
    return transition


@dataclass
class _WorkerState:
    vectors: np.ndarray
    labels: np.ndarray
    baseline: Path
    output: Path
    binding: str
    protocol: dict[str, Any]
    dimensions: tuple[int, ...]
    replays: dict[Job, dict[str, Any]]
    scheduler_code_hashes: dict[str, str]


_WORKER: _WorkerState | None = None
_THREAD_LIMITER: Any = None


def _initialize_worker(input_manifest: str, baseline: str, output: str, registration: dict,
                       dimensions: tuple[int, ...], replays: dict, scheduler_hashes: dict) -> None:
    global _WORKER, _THREAD_LIMITER
    _THREAD_LIMITER = threadpool_limits(limits=4)
    if _code_hashes() != scheduler_hashes:
        raise ValueError("worker scheduler code changed")
    vectors, labels, profile = load_input(Path(input_manifest))
    extension._same_fingerprints(registration["binding"], vectors, labels, profile)
    _WORKER = _WorkerState(
        vectors, labels, Path(baseline), Path(output), registration["binding_sha256"],
        registration["binding"]["protocol"], dimensions, replays, scheduler_hashes,
    )


def _run_group(job: Job) -> dict[str, Any]:
    state = _WORKER
    if state is None:
        raise RuntimeError("replay worker was not initialized")
    with exclusive_lock(_job_lock(state.output, job)):
        if _code_hashes() != state.scheduler_code_hashes:
            raise ValueError("scheduler code changed during execution")
        seed, schedule = job
        started = time.perf_counter()
        record = state.replays[job]
        replay = extension._replay_arrays(state.baseline, seed, schedule, record, len(state.labels))
        sources, order, times = replay["source_id"], replay["order"], replay["timestamps"]
        ids = tuple(f"seed-{seed}:occ-{i:012d}" for i in range(len(state.labels)))
        computed = reused = 0
        for dimension in state.dimensions:
            identity = {"kind": "membership", "seed": seed, "schedule": schedule, "dimension": dimension}
            rank_path = state.output / "memberships" / f"s{seed}-{schedule}-d{dimension}.json"
            rank = _load_checkpoint(rank_path, state.binding, identity)
            prefix = None
            if rank is None:
                prefix = normalized_prefix(state.vectors, dimension)
                ranking, counts = rank_membership(
                    ids, ("imdb",) * len(sources), (("imdb-review",),) * len(sources),
                    prefix[sources], order, times, seed,
                )
                rank = _checkpoint(rank_path, state.binding, identity, {"ranking": ranking}, {
                    "ranking_sha256": array_sha256(ranking), "selector_counts": counts,
                    "replay_hashes": record["hashes"], "label_blind": True,
                })
            with np.load(rank_path.with_suffix(".npz"), allow_pickle=False) as arrays:
                ranking = arrays["ranking"]
            if (rank["replay_hashes"] != record["hashes"]
                    or rank["ranking_sha256"] != array_sha256(ranking)
                    or not np.array_equal(np.sort(ranking), np.arange(len(sources)))):
                raise ValueError("worker membership ranking/replay mismatch")
            for index, rate in enumerate(state.protocol["rates"]):
                budget = max(1, int(len(sources) * rate))
                selected = ranking[:budget]
                cell_path = state.output / "cells" / f"s{seed}-{schedule}-d{dimension}-r{index}.json"
                cell_identity = {**identity, "kind": "cell", "rate": rate, "budget": budget}
                cell = _load_checkpoint(cell_path, state.binding, cell_identity)
                if cell is not None:
                    reused += 1
                    continue
                cell_started = time.perf_counter()
                if prefix is None:
                    prefix = normalized_prefix(state.vectors, dimension)
                evidence = evaluate_replay(
                    prefix, state.labels, sources, order, selected, seed=seed,
                    target_block_size=state.protocol["distance_blocks"]["targets"],
                    donor_block_size=state.protocol["distance_blocks"]["donors"],
                    calibration_reservoir_size=state.protocol["lipschitz"]["reservoir_size"],
                )
                row = {
                    "dimension": dimension, "seed": seed, "schedule": schedule, "rate": rate, "budget": budget,
                    **cell_metrics(evidence, state.labels[sources[selected]]),
                    "replay_hashes": record["hashes"], "ranking_sha256": rank["ranking_sha256"],
                    "membership_sha256": array_sha256(np.sort(selected)),
                    "selector_counts": rank["selector_counts"],
                    "evidence": extension._relative(cell_path.with_suffix(".npz"), state.output),
                    "elapsed_seconds": time.perf_counter() - cell_started,
                }
                _checkpoint(cell_path, state.binding, cell_identity, evidence, {"row": row})
                computed += 1
        return {
            "seed": seed, "schedule": schedule, "worker_pid": os.getpid(),
            "computed_cells": computed, "reused_cells": reused,
            "elapsed_seconds": time.perf_counter() - started,
        }


def _replace_from_stage(stage: Path, target: Path) -> None:
    temporary = target.with_suffix(".publication.tmp")
    temporary.write_bytes(stage.read_bytes())
    temporary.replace(target)


def _recover_publication(output: Path, transition: dict) -> None:
    journal_path = output / "parallel_publication.json"
    if not journal_path.exists():
        return
    journal = _read_record(journal_path)
    if (journal["scientific_binding_sha256"] != transition["execution"]["scientific_binding_sha256"]
            or journal["transition_sha256"] != sha256_file(output / "parallel_transition.json")
            or journal["scheduler_code_hashes"] != _code_hashes()
            or set(journal["targets"]) != {"aggregate.json", "extension_validation.json", "manifest.json"}):
        raise ValueError("parallel publication recovery fingerprint mismatch")
    for name, record in journal["targets"].items():
        stage, target = output / "execution_publication" / name, output / name
        if (sha256_file(stage) != record["sha256"]
                or sha256_file(target) not in (record["previous_sha256"], record["sha256"])):
            raise ValueError("parallel publication recovery artifact changed")
    for name in ("aggregate.json", "extension_validation.json", "manifest.json"):
        record = journal["targets"][name]
        if sha256_file(output / name) != record["sha256"]:
            _replace_from_stage(output / "execution_publication" / name, output / name)


def _publish_execution(output: Path, aggregate: dict, transition: dict, computed: int,
                       workers: int, log: Path) -> dict:
    transition_path = output / "parallel_transition.json"
    execution = {
        "version": VERSION, "workers": workers, "blas_threads_per_worker": 4,
        "checkpoint_cells_preserved_at_switch": transition["preserved_cells"],
        "completed_checkpoints_unchanged": True,
        "scientific_binding_unchanged": True,
        "transition_sha256": sha256_file(transition_path),
        "scheduler_code_hashes": transition["execution"]["scheduler_code_hashes"],
    }
    aggregate["execution"] = execution
    stage_root = output / "execution_publication"
    aggregate_path = stage_root / "aggregate.json"
    write_json(aggregate_path, aggregate)
    audit_path = output / "extension_validation.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    write_json(stage_root / audit_path.name, {
        **audit, "aggregate_sha256": sha256_file(aggregate_path), "execution": execution,
    })
    extension._progress(log, "parallel_completed", computed_cells_this_invocation=computed,
                        preserved_cells_at_switch=transition["preserved_cells"], combined_cells=len(aggregate["rows"]))
    manifest_path = output / "manifest.json"
    manifest = _read_record(manifest_path)
    for name in ("aggregate.json", "extension_validation.json"):
        manifest["files"][name] = sha256_file(stage_root / name)
    for path in (transition_path, log):
        manifest["files"][extension._relative(path, output)] = sha256_file(path)
    manifest["execution"] = execution
    _write_record(stage_root / manifest_path.name, manifest)
    _write_record(output / "parallel_publication.json", {
        "scientific_binding_sha256": transition["execution"]["scientific_binding_sha256"],
        "transition_sha256": execution["transition_sha256"], "scheduler_code_hashes": _code_hashes(),
        "targets": {
            name: {"previous_sha256": sha256_file(output / name), "sha256": sha256_file(stage_root / name)}
            for name in ("aggregate.json", "extension_validation.json", "manifest.json")
        },
    })
    _recover_publication(output, transition)
    return aggregate


def run_parallel(
    input_manifest: Path, *, baseline: Path = extension.DEFAULT_BASELINE,
    output: Path = extension.DEFAULT_OUTPUT, workers: int = 3,
) -> dict[str, Any]:
    """Resume an existing source-bound extension with one coordinator and N workers."""
    workers = _positive_integer(workers, "workers")
    if workers > 3:
        raise ValueError("this continuation is capped at three workers")
    output, baseline, input_manifest = Path(output).resolve(), Path(baseline).resolve(), Path(input_manifest).resolve()
    if not (output / "preregistration.json").is_file():
        raise ValueError("parallel continuation requires the existing serial extension preregistration")
    with exclusive_lock(output / "execution_locks" / "coordinator.lock"), threadpool_limits(limits=4):
        vectors, labels, profile = load_input(input_manifest)
        source = extension._verify_baseline(vectors, labels, baseline, profile)
        registration, dimensions = _verify_registration(source, output)
        # Workers from an interrupted coordinator must finish or be stopped first.
        jobs = [(s, a) for s in registration["binding"]["protocol"]["seeds"]
                for a in registration["binding"]["protocol"]["schedules"]]
        for job in jobs:
            with exclusive_lock(_job_lock(output, job)):
                pass
        inventory = _inventory(output, registration, dimensions)
        transition = _transition(output, registration, inventory, workers)
        _recover_publication(output, transition)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = _read_record(manifest_path)
            if manifest.get("execution", {}).get("transition_sha256") == sha256_file(output / "parallel_transition.json"):
                return extension._extend(vectors, labels, output, source, dimensions, True)
        log = output / "parallel_progress.jsonl"
        extension._progress(log, "parallel_started", workers=workers,
                            completed_cells=inventory["completed_cells"], pending_groups=len(inventory["pending"]))
        computed = 0
        if inventory["pending"]:
            executor = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(str(input_manifest), str(baseline), str(output), registration,
                          dimensions, source.replays, _code_hashes()),
            )
            futures = {}
            try:
                futures = {executor.submit(_run_group, job): job for job in inventory["pending"]}
                for future in as_completed(futures):
                    result = future.result()
                    computed += result["computed_cells"]
                    extension._progress(log, "group_completed", **result,
                                        completed_cells=inventory["completed_cells"] + computed)
                    print(f"New-dimension checkpoints: {inventory['completed_cells'] + computed}/"
                          f"{registration['binding']['extension']['added_cells']} "
                          f"(worker {result['worker_pid']}, seed {result['seed']}, {result['schedule']})", flush=True)
            except BaseException as exc:
                for future in futures:
                    future.cancel()
                extension._progress(log, "parallel_failed", error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        if _code_hashes() != transition["execution"]["scheduler_code_hashes"]:
            raise ValueError("parallel scheduling code changed during execution")
        extension._verify_files({output / name: digest for name, digest in transition["preserved_files"].items()})
        # The unchanged serial coordinator validates every cell and is the only final publisher.
        aggregate = extension._extend(vectors, labels, output, source, dimensions, True)
        extension._verify_files({output / name: digest for name, digest in transition["preserved_files"].items()})
        return _publish_execution(output, aggregate, transition, computed, workers, log)
