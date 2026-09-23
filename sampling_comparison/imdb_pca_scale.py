"""Six-worker continuation of a registered PCA study without changing its code binding."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

from threadpoolctl import threadpool_limits

from sampling_comparison import imdb_experiment as engine
from sampling_comparison import imdb_extension as extension
from sampling_comparison import imdb_parallel as parallel
from sampling_comparison import imdb_pca as pca
from sampling_comparison.imdb_inputs import load_input


VERSION = "imdb-pca-worker-scaling-v1"


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {path.relative_to(root).as_posix(): engine.sha256_file(path) for path in (
        Path(__file__).resolve(), root / "scripts" / "scale_imdb_pca.py",
    )}


def _preflight(input_manifest: Path, pca_manifest: Path, original: Path, baseline: Path, output: Path):
    vectors, labels, profile = load_input(input_manifest)
    source = pca._verify_sources(vectors, labels, profile, original, baseline)
    prepared = pca.load_pca(input_manifest, pca_manifest)
    dims = prepared.manifest["binding"]["dimensions"]
    protocol = {
        **source.aggregate["protocol"],
        "dimensions": sorted(set(source.aggregate["protocol"]["dimensions"]) | set(dims), reverse=True),
        "representations": pca._representations(source.aggregate["protocol"]["dimensions"], dims),
    }
    added = len(dims) * len(protocol["seeds"]) * len(protocol["schedules"]) * len(protocol["rates"])
    protocol["planned_cells"] = len(source.aggregate["rows"]) + added
    source_files = {**source.files, **prepared.files, **pca._input_files(input_manifest, profile)}
    binding = {
        "version": pca.STUDY_VERSION, "original_binding": source.original.binding,
        "protocol": protocol, "pca_dimensions": dims,
        "pca_manifest_sha256": engine.sha256_file(pca_manifest),
        "baseline_aggregate_sha256": source.files[baseline / "aggregate.json"],
        "baseline_manifest_sha256": source.files[baseline / "manifest.json"],
        "source_files": {extension._relative(path, output): h for path, h in source_files.items()},
        "code_hashes": pca._code_hashes(), "blas_threads": 4,
    }
    registration = engine._read_record(output / "preregistration.json")
    if registration != {"binding_sha256": engine._digest(binding), "binding": binding}:
        raise ValueError("scaling requires the unchanged PCA scientific registration")
    return vectors, labels, profile, source, prepared, source_files, registration


def _archive_uncommitted(output: Path, registration: dict) -> None:
    """Preserve only recognized uncommitted fragments after every job lock is checked."""
    protocol = registration["binding"]["protocol"]
    expected = set()
    for seed in protocol["seeds"]:
        for schedule in protocol["schedules"]:
            for dimension in registration["binding"]["pca_dimensions"]:
                job = (seed, schedule, dimension)
                expected.add(pca._paths(output, job))
                expected.update(pca._paths(output, job, i) for i in range(len(protocol["rates"])))
    candidates = []
    for path in expected:
        if not path.exists() and path.with_suffix(".npz").exists():
            candidates.append(path.with_suffix(".npz"))
        for suffix in (".json.tmp", ".npz.tmp"):
            temporary = path.with_suffix(suffix)
            if temporary.exists():
                candidates.append(temporary)
    if not candidates:
        return
    archive = output / "scaling_recovery" / str(time.time_ns())
    files = {extension._relative(path, output): engine.sha256_file(path) for path in candidates}
    engine.write_json(archive / "manifest.json", {
        "reason": "uncommitted fragments at an explicitly stopped worker boundary",
        "binding_sha256": registration["binding_sha256"], "files": files,
    })
    for path in candidates:
        target = archive / path.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)


def _transition(output: Path, registration: dict, checkpoint_files: dict, rows: list, workers: int) -> dict:
    path = output / "scaling_transition.json"
    execution = {
        "version": VERSION, "workers": workers, "blas_threads_per_worker": 4,
        "scientific_binding_sha256": registration["binding_sha256"],
        "preregistration_sha256": engine.sha256_file(output / "preregistration.json"),
        "scheduler_code_hashes": _code_hashes(),
    }
    if path.exists():
        record = engine._read_record(path)
        if record["execution"] != execution:
            raise ValueError("scaling execution fingerprint changed")
        extension._verify_files({output / name: h for name, h in record["preserved_files"].items()})
        return record
    record = {
        "execution": execution, "preserved_cells": len(rows),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preserved_files": {extension._relative(path, output): h for path, h in checkpoint_files.items()},
    }
    engine._write_record(path, record)
    return record


def _publish(output, source, prepared, source_files, registration, rows, checkpoint_files, transition, workers):
    binding, protocol = registration["binding_sha256"], registration["binding"]["protocol"]
    meta = prepared.public_metadata
    study = {
        "baseline_aggregate_sha256": registration["binding"]["baseline_aggregate_sha256"],
        "baseline_manifest_sha256": registration["binding"]["baseline_manifest_sha256"],
        "pca_manifest_sha256": registration["binding"]["pca_manifest_sha256"],
        "reused_cells": len(source.aggregate["rows"]), "added_cells": len(rows),
        "dimensions": registration["binding"]["pca_dimensions"],
        "baseline_rows_unchanged": True, "replay_pairing_exact": True, "embedding_calls": 0, "judge_calls": 0,
        **{key: meta[key] for key in ("fit_scope", "fit_source_count", "solver", "whiten", "fit_seed",
                                    "fit_once", "fit_count", "labels_used_for_fit", "deduplicated",
                                    "transductive", "native_1536_control_note")},
    }
    before = {key: study[key] for key in (
        "baseline_aggregate_sha256", "baseline_manifest_sha256", "pca_manifest_sha256",
    )}
    scaling = {
        "transition_sha256": engine.sha256_file(output / "scaling_transition.json"),
        "preserved_cells": transition["preserved_cells"], "preserved_files": transition["preserved_files"],
        "scheduler_code_hashes": transition["execution"]["scheduler_code_hashes"],
        "scientific_binding_unchanged": True, "completed_checkpoints_unchanged": True,
    }
    combined = pca._combined_rows(source, output, rows)
    aggregate = {
        "version": engine.VERSION, "status": "completed", "dataset": source.aggregate["dataset"],
        "protocol": protocol, "binding_sha256": binding, "provenance": registration["binding"],
        "rows": combined, "pca_study": study, "pca_preparation": meta,
        "summaries": source.aggregate["summaries"], "summaries_scope": "native_and_prefix_only",
        "source_provenance": {key: source.aggregate[key] for key in ("extension", "execution") if key in source.aggregate},
        "pca_execution": {
            "workers": workers, "blas_threads_per_worker": 4, "start_method": "spawn",
            "job_unit": "seed/schedule/dimension; all budgets", "one_finalizer": True,
            "scaling": scaling,
        },
    }
    stage, log = output / "publication", output / "progress.jsonl"
    engine.write_json(stage / "aggregate.json", aggregate)
    engine.write_json(stage / "pca_study_validation.json", {
        "version": "imdb-pca-study-validation-v1", "status": "passed", "ok": True, **study,
        "baseline_before": before, "baseline_after": before,
        "source_artifacts_unchanged": True, "source_artifacts_checked": len(source_files),
        "native_cache_unchanged": True, "pca_cache_unchanged": True, "input_arrays_unchanged": True,
        "method_code_hashes_unchanged": True, "runtime_unchanged": True,
        "fit_artifact_integrity_verified": True, "fit_verification": meta["verification"],
        "combined_cells": len(combined), "replay_count": len(source.original.replays),
        "protocol": protocol, "aggregate_sha256": engine.sha256_file(stage / "aggregate.json"),
        "scaling": scaling, "workers": workers,
    })
    extension._progress(log, "pca_completed", combined_cells=len(combined), workers=workers)
    files = {extension._relative(path, output): h for path, h in {**source_files, **checkpoint_files}.items()}
    files.update({extension._relative(path, output): engine.sha256_file(path)
                  for path in (output / "preregistration.json", log)})
    for name in ("aggregate.json", "pca_study_validation.json"):
        files[name] = engine.sha256_file(stage / name)
    engine._write_record(stage / "manifest.json", {
        "version": engine.VERSION, "status": "completed", "binding_sha256": binding,
        "completed_cells": len(combined), "files": files, "pca_study": study,
    })
    engine._write_record(output / "publication.json", {
        "binding_sha256": binding, "targets": {
            name: engine.sha256_file(stage / name)
            for name in ("aggregate.json", "pca_study_validation.json", "manifest.json")
        },
    })
    pca._recover_publication(output, binding)
    return aggregate


def run_scaled(
    input_manifest: Path = pca.DEFAULT_INPUT, *, pca_manifest: Path = pca.DEFAULT_PREPARATION / "manifest.json",
    original: Path = extension.DEFAULT_BASELINE, baseline: Path = pca.DEFAULT_BASELINE,
    output: Path = pca.DEFAULT_OUTPUT, workers: int = 6,
) -> dict:
    workers = engine._positive_integer(workers, "workers")
    if not 1 <= workers <= 6:
        raise ValueError("worker count must be between one and six")
    input_manifest, pca_manifest, original, baseline, output = map(
        lambda path: Path(path).resolve(), (input_manifest, pca_manifest, original, baseline, output))
    if not (output / "preregistration.json").is_file():
        raise ValueError("scaling requires an existing PCA registration")
    roots = (input_manifest.parent, pca_manifest.parent, original, baseline)
    pca._separate(output, roots)
    with parallel.exclusive_lock(output / "execution_locks/coordinator.lock"), threadpool_limits(limits=4):
        vectors, labels, profile, source, prepared, source_files, registration = _preflight(
            input_manifest, pca_manifest, original, baseline, output)
        protocol = registration["binding"]["protocol"]
        jobs = [(s, a, d) for s in protocol["seeds"] for a in protocol["schedules"]
                for d in registration["binding"]["pca_dimensions"]]
        for job in jobs:
            with parallel.exclusive_lock(pca._lock_path(output, job)):
                pass
        _archive_uncommitted(output, registration)
        pending, rows, files = pca._inventory(output, registration, source.original.replays)
        transition = _transition(output, registration, files, rows, workers)
        if (output / "publication.json").exists():
            if pending:
                raise ValueError("publication exists for an incomplete PCA grid")
            pca._recover_publication(output, registration["binding_sha256"])
        if (output / "manifest.json").exists():
            manifest = engine._read_record(output / "manifest.json")
            retained = pca._bound_files(output, manifest["files"], (*roots, output))
            required = {**source_files, **files}
            required.update({path: engine.sha256_file(path) for path in (
                output / "preregistration.json", output / "progress.jsonl",
                output / "aggregate.json", output / "pca_study_validation.json",
            )})
            if retained != required:
                raise ValueError("scaled PCA manifest omits or adds retained artifacts")
            aggregate = json.loads((output / "aggregate.json").read_text())
            if (pending or aggregate["rows"] != pca._combined_rows(source, output, rows)
                    or aggregate["binding_sha256"] != registration["binding_sha256"]
                    or aggregate["protocol"] != protocol or aggregate["pca_execution"]["workers"] != workers
                    or aggregate["status"] != "completed" or manifest["status"] != "completed"
                    or manifest["completed_cells"] != protocol["planned_cells"]):
                raise ValueError("completed scaled PCA study differs from retained evidence")
            return aggregate
        log = output / "progress.jsonl"
        extension._progress(log, "pca_scaled_started", workers=workers, pending_jobs=len(pending),
                            preserved_cells=transition["preserved_cells"])
        if pending:
            executor = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=pca._initialize_worker,
                initargs=(str(input_manifest), str(pca_manifest), str(original), str(output),
                          registration, source.original.replays),
            )
            futures = []
            try:
                futures = [executor.submit(pca._run_job, job) for job in pending]
                for future in as_completed(futures):
                    event = future.result()
                    extension._progress(log, "pca_job_completed", **event)
                    print(f"PCA seed={event['seed']} {event['schedule']} d={event['dimension']}: "
                          f"{event['computed_cells']} new cells (pid {event['worker_pid']})", flush=True)
            except BaseException as exc:
                for future in futures:
                    future.cancel()
                extension._progress(log, "pca_failed", error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        pending, rows, checkpoint_files = pca._inventory(output, registration, source.original.replays)
        expected_added = protocol["planned_cells"] - len(source.aggregate["rows"])
        if pending or len(rows) != expected_added:
            raise ValueError("cannot publish incomplete scaled PCA study")
        extension._verify_files({output / name: h for name, h in transition["preserved_files"].items()})
        extension._verify_files(source_files)
        extension._same_fingerprints(source.original.binding, vectors, labels, profile)
        if (pca._code_hashes() != registration["binding"]["code_hashes"]
                or _code_hashes() != transition["execution"]["scheduler_code_hashes"]):
            raise ValueError("PCA scientific or scaling code changed")
        return _publish(output, source, prepared, source_files, registration, rows, checkpoint_files, transition, workers)
