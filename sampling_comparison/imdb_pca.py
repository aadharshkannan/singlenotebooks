"""Offline, additive PCA study over frozen IMDb replays.

Preparation is a separate process: only ``prepare_pca`` imports sklearn.
One unlabeled, transductive fit retains every source row (including duplicates).
Production uses all 1536 learned components; small fixtures may request fewer.
Full SVD intentionally replaces the reference study's randomized solver.
Neither preparation nor replay has an embedding or judge interface.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from importlib.metadata import version
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from sampling_comparison import imdb_experiment as engine
from sampling_comparison import imdb_extension as extension
from sampling_comparison import imdb_parallel as parallel
from sampling_comparison.imdb_inputs import load_input


DIMENSIONS = (1536, 256, 128, 64, 32, 24, 16, 12, 8)
DEFAULT_INPUT = Path("outputs_imdb/cache/imdb-50000/manifest.json")
DEFAULT_PREPARATION = Path("outputs_imdb/cache/imdb-pca-full")
DEFAULT_BASELINE = extension.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path("outputs_imdb/private_runs/imdb-pca-40-replay")
PREPARATION_VERSION = "imdb-pca-preparation-v1"
STUDY_VERSION = "imdb-pca-study-v1"
CONTROL_NOTE = (
    "PCA1536 is a centered full-component control, not native1536: centering "
    "followed by row normalization changes angular geometry. PCA prefixes are "
    "leading learned components, not native coordinates."
)
NORMALIZATION = (
    "frozen normalized_prefix(native,1536): float32 row normalization; "
    "unwhitened centered PCA scores stored float32; normalized_prefix(scores,d); "
    "unchanged evaluate_replay float64 normalized-angular geometry"
)
ARRAYS = (
    "mean", "components", "singular_values", "explained_variance",
    "explained_variance_ratio", "projected",
)
Job = tuple[int, str, int]


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "scripts/prepare_imdb_pca.py", root / "scripts/run_imdb_pca.py",
             Path(extension.__file__), Path(parallel.__file__), root / "sampling_comparison/imdb_inputs.py"]
    return {p.relative_to(root).as_posix(): engine.sha256_file(p) for p in paths}


def _versions() -> dict[str, str]:
    # Metadata inspection does not load sklearn or its extra OpenMP runtime.
    return {"python": sys.version, **{name: version(name) for name in ("numpy", "scipy", "scikit-learn")}}


def _dimensions(values: Sequence[int], maximum: int) -> tuple[int, ...]:
    values = tuple(values)
    if (not values or len(set(values)) != len(values)
            or any(isinstance(d, bool) or not isinstance(d, (int, np.integer))
                   or not 1 <= d <= maximum for d in values)):
        raise ValueError(f"dimensions must be unique integers in [1,{maximum}]")
    return tuple(sorted(map(int, values), reverse=True))


def _separate(output: Path, roots: Sequence[Path]) -> None:
    for root in roots:
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("output and source bundles must be separate non-nested directories")


def _bound_files(root: Path, files: dict[str, str], allowed: Sequence[Path]) -> dict[Path, str]:
    result = {}
    for name, digest in files.items():
        path = (root / name).resolve()
        if Path(name).is_absolute() or not any(path.is_relative_to(base) for base in allowed):
            raise ValueError(f"artifact outside allowed source bundles: {name}")
        if path in result:
            raise ValueError("artifact path aliases are not allowed")
        result[path] = digest
    extension._verify_files(result)
    return result


def _input_files(manifest: Path, profile: dict) -> dict[Path, str]:
    files = _bound_files(manifest.parent, {
        profile["files"][key]: profile["hashes"][key] for key in ("vectors", "units")
    }, (manifest.parent,))
    files[manifest] = engine.sha256_file(manifest)
    return files


def _prep_binding(input_manifest: Path, vectors: np.ndarray, labels: np.ndarray, profile: dict,
                  dimensions: tuple[int, ...], threads: int) -> dict:
    return {
        "version": PREPARATION_VERSION, "source_manifest_sha256": engine.sha256_file(input_manifest),
        "source_files": {extension._relative(p, input_manifest.parent): h
                         for p, h in _input_files(input_manifest, profile).items()},
        "input_hashes": extension._input_hashes(vectors, labels),
        "input_shape": list(vectors.shape), "dimensions": list(dimensions),
        "n_components": max(dimensions), "fit_source_count": len(vectors),
        "fit_scope": "full_unlabeled_source_population", "fit_seed": 13,
        "solver": "full", "whiten": False, "fit_once": True,
        "labels_used_for_fit": False, "deduplicated": False, "transductive": True,
        "normalization": NORMALIZATION, "versions": _versions(),
        "blas_threads": threads, "code_hashes": _code_hashes(), "method_code_hashes": engine._method_hashes(),
    }


def _fit_arrays(native: np.ndarray, components: int, *, blas_threads: int = 4
                ) -> tuple[dict[str, np.ndarray], dict]:
    """Only this preparation-only function imports sklearn; no labels argument."""
    from sklearn.decomposition import PCA

    # Apply limits AFTER importing sklearn, which may load another native pool.
    with threadpool_limits(limits=blas_threads):
        started = time.perf_counter()
        reducer = PCA(n_components=components, svd_solver="full", whiten=False, random_state=13)
        reducer.fit(native)
        arrays = {
            "mean": reducer.mean_, "components": reducer.components_,
            "singular_values": reducer.singular_values_, "explained_variance": reducer.explained_variance_,
            "explained_variance_ratio": reducer.explained_variance_ratio_,
            "projected": reducer.transform(native),
        }
        return {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}, {
            "fit_count": 1, "elapsed_seconds": time.perf_counter() - started,
            "runtime": engine._runtime(), "sklearn_version": version("scikit-learn"),
            "normalized_input_sha256": engine.array_sha256(native),
        }


def _verify_geometry(native: np.ndarray, arrays: dict[str, np.ndarray]) -> dict:
    """Blockwise independent transform and full-component reconstruction checks."""
    n, width = native.shape
    components, scores, mean = arrays["components"], arrays["projected"], arrays["mean"]
    k = len(components)
    expected = {"mean": (width,), "components": (k, width), "projected": (n, k),
                **{name: (k,) for name in ARRAYS[2:5]}}
    for name, shape in expected.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"invalid PCA array: {name}")
    if not np.allclose(mean, native.mean(axis=0), atol=2e-6, rtol=2e-5):
        raise ValueError("PCA mean does not match all source rows")
    if not np.allclose(components @ components.T, np.eye(k), atol=1e-5, rtol=1e-5):
        raise ValueError("PCA components are not orthonormal (whitening is forbidden)")
    transform_error = reconstruction_error = 0.0
    for start in range(0, n, 512):
        centered = native[start:start + 512].astype(np.float64) - mean
        retained = scores[start:start + 512]
        transformed = centered @ components.astype(np.float64).T
        transform_error = max(transform_error, float(np.max(np.abs(transformed - retained))))
        if not np.allclose(transformed, retained, atol=1e-5, rtol=5e-5):
            raise ValueError("retained PCA scores differ from unwhitened transform")
        if k == width:
            reconstructed = retained.astype(np.float64) @ components
            reconstruction_error = max(reconstruction_error, float(np.max(np.abs(reconstructed - centered))))
            if not np.allclose(reconstructed, centered, atol=1e-5, rtol=5e-5):
                raise ValueError("full-component PCA does not reconstruct centered input")
    variance = np.var(scores.astype(np.float64), axis=0, ddof=1)
    total = np.var(native.astype(np.float64), axis=0, ddof=1).sum()
    if (not np.allclose(variance, arrays["explained_variance"], atol=2e-7, rtol=1e-4)
            or not np.allclose(arrays["singular_values"] ** 2 / (n - 1),
                               arrays["explained_variance"], atol=2e-7, rtol=1e-4)
            or not np.allclose(arrays["explained_variance"] / total,
                               arrays["explained_variance_ratio"], atol=2e-7, rtol=1e-4)):
        raise ValueError("PCA variance/singular values disagree with unwhitened scores")
    return {
        "transform_verified": True, "transform_max_abs_error": transform_error,
        "no_whitening_verified": True, "full_component_centered_identity_verified": k == width,
        "centered_reconstruction_max_abs_error": reconstruction_error if k == width else None,
    }


def _save_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


@dataclass
class PreparedPCA:
    root: Path
    manifest: dict
    arrays: dict[str, np.ndarray]
    files: dict[Path, str]

    @property
    def public_metadata(self) -> dict:
        binding, fit = self.manifest["binding"], self.manifest["fit"]
        variance_by_dimension = {
            str(d): float(self.arrays["explained_variance_ratio"][:d].astype(float).sum())
            for d in binding["dimensions"]
        }
        return {
            **{key: binding[key] for key in (
                "dimensions", "n_components", "fit_source_count", "fit_scope", "fit_seed",
                "solver", "whiten", "fit_once", "normalization",
                "labels_used_for_fit", "deduplicated", "transductive",
            )},
            "fit_count": fit["fit_count"], "sklearn_version": fit["sklearn_version"],
            "elapsed_seconds": fit["elapsed_seconds"], "verification": fit["verification"],
            "fit_seconds": fit["elapsed_seconds"],
            "solver_deviation": "Deterministic full SVD, not the reference randomized solver.",
            "native_1536_control_note": CONTROL_NOTE,
            "explained_variance": self.arrays["explained_variance"].tolist(),
            "explained_variance_ratio": self.arrays["explained_variance_ratio"].tolist(),
            "explained_variance_by_dimension": variance_by_dimension,
            "cumulative_explained_variance_ratio": dict(variance_by_dimension),
        }


def _read_prepared(root: Path, binding: dict, *, completed: bool = True) -> PreparedPCA:
    path = root / ("manifest.json" if completed else "fit.json")
    record = engine._read_record(path)
    if (record["binding"] != binding or record["binding_sha256"] != engine._digest(binding)
            or record["status"] != "completed" or record["fit"]["fit_count"] != 1
            or not record["fit"]["verification"]["transform_verified"]
            or not record["fit"]["verification"]["no_whitening_verified"]):
        raise ValueError("incompatible PCA preparation fingerprints/parameters/code")
    required = {f"{name}.npy" for name in ARRAYS} | {"preregistration.json", "fit_started.json"}
    if completed:
        required.add("fit.json")
    if set(record["files"]) != required:
        raise ValueError("PCA manifest is missing required numerical artifacts")
    files = _bound_files(root, record["files"], (root,))
    files[path] = engine.sha256_file(path)
    arrays = {name: np.load(root / f"{name}.npy", allow_pickle=False, mmap_mode="r") for name in ARRAYS}
    if {name: engine.array_sha256(value) for name, value in arrays.items()} != record["array_hashes"]:
        raise ValueError("PCA input array hashes differ")
    if arrays["projected"].shape != (binding["fit_source_count"], binding["n_components"]):
        raise ValueError("PCA score shape differs")
    return PreparedPCA(root, record, arrays, files)


def load_pca(input_manifest: Path, pca_manifest: Path, *,
             dimensions: Sequence[int] | None = None) -> PreparedPCA:
    """Verify source, code, versions and all retained fit bytes; never import sklearn."""
    input_manifest, pca_manifest = Path(input_manifest).resolve(), Path(pca_manifest).resolve()
    if pca_manifest.name != "manifest.json":
        raise ValueError("PCA replay requires the completed preparation manifest.json")
    vectors, labels, profile = load_input(input_manifest)
    record = engine._read_record(pca_manifest)
    dims = _dimensions(record["binding"]["dimensions"] if dimensions is None else dimensions,
                       min(vectors.shape))
    binding = _prep_binding(input_manifest, vectors, labels, profile, dims, record["binding"]["blas_threads"])
    return _read_prepared(pca_manifest.parent, binding)


def prepare_pca(input_manifest: Path = DEFAULT_INPUT, output: Path = DEFAULT_PREPARATION, *,
                dimensions: Sequence[int] = DIMENSIONS, resume: bool = False,
                blas_threads: int = 4) -> dict:
    """Run in a dedicated process. Never silently refit an interrupted/corrupt fit."""
    input_manifest, output = Path(input_manifest).resolve(), Path(output).resolve()
    _separate(output, (input_manifest.parent,))
    threads = engine._positive_integer(blas_threads, "blas_threads")
    with parallel.exclusive_lock(output / "execution_locks/coordinator.lock"), threadpool_limits(limits=threads):
        vectors, labels, profile = load_input(input_manifest)
        dims = _dimensions(dimensions, min(vectors.shape))
        binding = _prep_binding(input_manifest, vectors, labels, profile, dims, threads)
        source_files = _input_files(input_manifest, profile)
        registration = output / "preregistration.json"
        existing = [p for p in output.iterdir() if p.name != "execution_locks"]
        if existing:
            if not resume:
                raise FileExistsError("nonempty PCA preparation requires explicit resume=True")
            if not registration.exists() or engine._read_record(registration)["binding"] != binding:
                raise ValueError("incompatible PCA preparation source/parameters/code")
        else:
            engine._write_record(registration, {"binding": binding, "binding_sha256": engine._digest(binding)})
        if (output / "manifest.json").exists():
            return _read_prepared(output, binding).manifest
        if (output / "fit.json").exists():
            prepared = _read_prepared(output, binding, completed=False)
            record = prepared.manifest
        else:
            if (output / "fit_started.json").exists() or any(output.glob("*.npy*")):
                raise ValueError("partial PCA fit has no verified commit; refusing to refit or fall back")
            engine._write_record(output / "fit_started.json", {"binding_sha256": engine._digest(binding)})
            native = engine.normalized_prefix(vectors, 1536)
            arrays, fit = _fit_arrays(native, max(dims), blas_threads=threads)
            fit["verification"] = _verify_geometry(native, arrays)
            for d in dims:
                engine.normalized_prefix(arrays["projected"], d)
            extension._verify_files(source_files)
            if _prep_binding(input_manifest, vectors, labels, profile, dims, threads) != binding:
                raise ValueError("PCA preparation source/code changed during fit")
            for name, value in arrays.items():
                _save_npy(output / f"{name}.npy", value)
            record = {
                "version": PREPARATION_VERSION, "status": "completed", "binding": binding,
                "binding_sha256": engine._digest(binding), "fit": fit,
                "array_hashes": {name: engine.array_sha256(value) for name, value in arrays.items()},
                "files": {name: engine.sha256_file(output / name)
                          for name in [*(f"{key}.npy" for key in ARRAYS),
                                       "preregistration.json", "fit_started.json"]},
            }
            engine._write_record(output / "fit.json", record)
        extension._verify_files(source_files)
        engine._write_record(output / "manifest.json", {
            **record, "files": {**record["files"], "fit.json": engine.sha256_file(output / "fit.json")},
        })
        return _read_prepared(output, binding).manifest


@dataclass
class _Sources:
    original: extension._Baseline
    root: Path
    aggregate: dict
    files: dict[Path, str]


def _verify_sources(vectors: np.ndarray, labels: np.ndarray, profile: dict,
                    original: Path, baseline: Path) -> _Sources:
    source = extension._verify_baseline(vectors, labels, original, profile)
    _separate(baseline, (original,))
    registration, added = parallel._verify_registration(source, baseline)
    manifest = engine._read_record(baseline / "manifest.json")
    aggregate = json.loads((baseline / "aggregate.json").read_text(encoding="utf-8"))
    if (manifest["status"] != "completed" or aggregate["status"] != "completed"
            or manifest["version"] != engine.VERSION or aggregate["version"] != engine.VERSION
            or manifest["binding_sha256"] != registration["binding_sha256"]
            or aggregate["binding_sha256"] != registration["binding_sha256"]
            or aggregate["protocol"] != registration["binding"]["protocol"]
            or aggregate["provenance"] != registration["binding"] or aggregate["dataset"] != profile
            or manifest["completed_cells"] != aggregate["protocol"]["planned_cells"]):
        raise ValueError("prefix baseline must be a complete source-bound extension")
    files = _bound_files(baseline, manifest["files"], (original, baseline))
    for path, digest in source.files.items():
        if files.get(path) != digest:
            raise ValueError("prefix baseline omitted or changed original inherited artifacts")
    if not {baseline / name for name in ("aggregate.json", "preregistration.json",
                                        "extension_validation.json")} <= files.keys():
        raise ValueError("prefix manifest omits required artifacts")
    files[baseline / "manifest.json"] = engine.sha256_file(baseline / "manifest.json")
    extension._parity_and_pairing(aggregate["rows"], source, baseline, added)
    inventory = parallel._inventory(baseline, registration, added)
    if inventory["pending"]:
        raise ValueError("prefix baseline has incomplete checkpoints")
    for relative, digest in inventory["files"].items():
        if files.get(baseline / relative) != digest:
            raise ValueError("prefix checkpoint is not manifest-bound")
    # Bind added rows to their retained cell and membership, not merely to a JSON grid.
    rates = aggregate["protocol"]["rates"]
    for row in aggregate["rows"]:
        if row["dimension"] not in added:
            continue
        s, a, d, r = extension._cell_key(row)
        stem = f"s{s}-{a}-d{d}"
        cell_path = baseline / "cells" / f"{stem}-r{rates.index(r)}.json"
        cell = engine._read_record(cell_path)
        rank = engine._read_record(baseline / "memberships" / f"{stem}.json")
        if (engine.canonical(cell["row"]) != engine.canonical({
                k: v for k, v in row.items() if k not in ("paired_delta_native", "evidence_sha256")})
                or row["evidence_sha256"] != cell["evidence_sha256"]
                or (baseline / row["evidence"]).resolve() != cell_path.with_suffix(".npz")
                or row["replay_hashes"] != rank["replay_hashes"]
                or row["ranking_sha256"] != rank["ranking_sha256"]
                or row["selector_counts"] != rank["selector_counts"]):
            raise ValueError("prefix measured row differs from retained checkpoint")
    return _Sources(source, baseline, aggregate, files)


def _representations(old_dimensions: Sequence[int], dimensions: Sequence[int]) -> list[dict]:
    return [
        {"id": "native_1536" if d == 1536 else f"prefix_{d}",
         "family": "native" if d == 1536 else "prefix", "dimension": d} for d in old_dimensions
    ] + [{"id": f"pca_{d}", "family": "pca", "dimension": d} for d in dimensions]


def _identity(job: Job, rate: float | None = None, count: int = 0) -> dict:
    seed, schedule, dimension = job
    result = {"kind": "membership", "seed": seed, "schedule": schedule, "dimension": dimension,
              "representation": "pca", "representation_id": f"pca_{dimension}"}
    if rate is not None:
        result.update(kind="cell", rate=rate, budget=max(1, math.floor(count * rate)))
    return result


def _paths(output: Path, job: Job, index: int | None = None) -> Path:
    s, a, d = job
    stem = f"s{s}-{a}-pca{d}"
    return output / ("memberships" if index is None else "cells") / (
        f"{stem}.json" if index is None else f"{stem}-r{index}.json")


def _lock_path(output: Path, job: Job) -> Path:
    return output / "execution_locks" / (_paths(output, job).stem + ".lock")


def _load(path: Path, binding: str, identity: dict) -> dict | None:
    # Unlike the older helper, reject orphan evidence rather than silently overwrite it.
    if not path.exists() and (path.with_suffix(".npz").exists() or path.with_suffix(".npz.tmp").exists()):
        raise ValueError(f"partial checkpoint has no committed metadata: {path}")
    return engine._load_checkpoint(path, binding, identity)


def _ranking(path: Path, rank: dict, replay: dict, count: int) -> np.ndarray:
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
        ranking = data["ranking"]
    if (rank["replay_hashes"] != replay["hashes"] or not rank["label_blind"]
            or rank["ranking_sha256"] != engine.array_sha256(ranking)
            or ranking.dtype.kind not in "iu" or not np.array_equal(np.sort(ranking), np.arange(count))):
        raise ValueError("PCA membership differs from paired replay")
    return ranking


def _row(cell: dict, path: Path, output: Path, identity: dict, rank: dict,
         ranking: np.ndarray, replay: dict) -> dict:
    row = {**cell["row"], "evidence_sha256": cell["evidence_sha256"]}
    if (any(row.get(k) != v for k, v in identity.items() if k != "kind")
            or row["replay_hashes"] != replay["hashes"] or row["ranking_sha256"] != rank["ranking_sha256"]
            or row["selector_counts"] != rank["selector_counts"]
            or row["membership_sha256"] != engine.array_sha256(np.sort(ranking[:identity["budget"]]))
            or row["evidence"] != extension._relative(path.with_suffix(".npz"), output)):
        raise ValueError("PCA cell differs from paired replay/membership")
    return row


def _inventory(output: Path, registration: dict, replays: dict) -> tuple[list[Job], list[dict], dict[Path, str]]:
    binding, protocol = registration["binding_sha256"], registration["binding"]["protocol"]
    pending, rows, files = [], [], {}
    expected_paths = set()
    for s in protocol["seeds"]:
        for a in protocol["schedules"]:
            for d in registration["binding"]["pca_dimensions"]:
                job = (s, a, d)
                path = _paths(output, job)
                expected_paths.add(path)
                rank = _load(path, binding, _identity(job))
                ranking = None if rank is None else _ranking(path, rank, replays[s, a], protocol["source_count"])
                missing = rank is None
                if rank is not None:
                    files.update({p: engine.sha256_file(p) for p in (path, path.with_suffix(".npz"))})
                for i, rate in enumerate(protocol["rates"]):
                    path = _paths(output, job, i)
                    expected_paths.add(path)
                    identity = _identity(job, rate, protocol["source_count"])
                    cell = _load(path, binding, identity)
                    if cell is None:
                        missing = True
                    else:
                        if rank is None:
                            raise ValueError("PCA cell has no membership checkpoint")
                        rows.append(_row(cell, path, output, identity, rank, ranking, replays[s, a]))
                        files.update({p: engine.sha256_file(p) for p in (path, path.with_suffix(".npz"))})
                if missing:
                    pending.append(job)
    for folder in ("memberships", "cells"):
        for path in (output / folder).glob("*"):
            if path.suffix not in (".json", ".npz") or path.with_suffix(".json") not in expected_paths:
                raise ValueError("unexpected or partial PCA checkpoint outside registered grid")
    return pending, rows, files


_WORKER: dict | None = None
_THREAD_LIMITER: Any = None


def _initialize_worker(input_manifest: str, pca_manifest: str, original: str, output: str,
                       registration: dict, replays: dict) -> None:
    global _WORKER, _THREAD_LIMITER
    _THREAD_LIMITER = threadpool_limits(limits=registration["binding"]["blas_threads"])
    vectors, labels, profile = load_input(Path(input_manifest))
    binding = registration["binding"]
    runtime = engine._runtime()
    if runtime != binding["original_binding"]["runtime"]:
        raise ValueError("PCA worker runtime mismatch: " + engine.canonical({
            "expected": binding["original_binding"]["runtime"], "actual": runtime,
        }))
    extension._same_fingerprints(binding["original_binding"], vectors, labels, profile)
    if _code_hashes() != binding["code_hashes"]:
        raise ValueError("PCA worker code changed")
    prepared = load_pca(Path(input_manifest), Path(pca_manifest), dimensions=binding["pca_dimensions"])
    if engine.sha256_file(Path(pca_manifest)) != binding["pca_manifest_sha256"]:
        raise ValueError("PCA worker preparation changed")
    _WORKER = {"labels": labels, "scores": prepared.arrays["projected"], "original": Path(original),
               "output": Path(output), "registration": registration, "replays": replays}


def _run_job(job: Job) -> dict:
    if _WORKER is None:
        raise RuntimeError("PCA worker is not initialized")
    state = _WORKER
    output, registration, labels = state["output"], state["registration"], state["labels"]
    protocol, binding = registration["binding"]["protocol"], registration["binding_sha256"]
    with parallel.exclusive_lock(_lock_path(output, job)):
        if _code_hashes() != registration["binding"]["code_hashes"]:
            raise ValueError("PCA code changed during replay")
        s, a, d = job
        replay_record = state["replays"][s, a]
        replay = extension._replay_arrays(state["original"], s, a, replay_record, len(labels))
        sources, order, times = replay["source_id"], replay["order"], replay["timestamps"]
        rank_path = _paths(output, job)
        rank = _load(rank_path, binding, _identity(job))
        prefix = None
        if rank is None:
            prefix = engine.normalized_prefix(state["scores"], d)
            ids = tuple(f"seed-{s}:occ-{i:012d}" for i in range(len(labels)))
            ranking, counts = engine.rank_membership(
                ids, ("imdb",) * len(labels), (("imdb-review",),) * len(labels),
                prefix[sources], order, times, s,
            )
            rank = engine._checkpoint(rank_path, binding, _identity(job), {"ranking": ranking}, {
                "ranking_sha256": engine.array_sha256(ranking), "selector_counts": counts,
                "replay_hashes": replay_record["hashes"], "label_blind": True,
            })
        ranking = _ranking(rank_path, rank, replay_record, len(labels))
        computed = 0
        for i, rate in enumerate(protocol["rates"]):
            identity, path = _identity(job, rate, len(labels)), _paths(output, job, i)
            cell = _load(path, binding, identity)
            if cell is None:
                started = time.perf_counter()
                if prefix is None:
                    prefix = engine.normalized_prefix(state["scores"], d)
                selected = ranking[:identity["budget"]]
                evidence = engine.evaluate_replay(
                    prefix, labels, sources, order, selected, seed=s,
                    target_block_size=protocol["distance_blocks"]["targets"],
                    donor_block_size=protocol["distance_blocks"]["donors"],
                    calibration_reservoir_size=protocol["lipschitz"]["reservoir_size"],
                )
                row = {
                    **{k: v for k, v in identity.items() if k != "kind"},
                    **engine.cell_metrics(evidence, labels[sources[selected]]),
                    "replay_hashes": replay_record["hashes"], "ranking_sha256": rank["ranking_sha256"],
                    "membership_sha256": engine.array_sha256(np.sort(selected)),
                    "selector_counts": rank["selector_counts"],
                    "evidence": extension._relative(path.with_suffix(".npz"), output),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                cell = engine._checkpoint(path, binding, identity, evidence, {"row": row})
                computed += 1
            _row(cell, path, output, identity, rank, ranking, replay_record)
        return {"seed": s, "schedule": a, "dimension": d, "computed_cells": computed, "worker_pid": os.getpid()}


def _delta(row: dict, reference: dict) -> dict:
    return {population: {
        metric: row[population][metric] - reference[population][metric]
        if row[population][metric] is not None and reference[population][metric] is not None else None
        for metric in engine.METRICS
    } for population in engine.POPULATIONS}


def _combined_rows(source: _Sources, output: Path, new_rows: list[dict]) -> list[dict]:
    old = source.aggregate["rows"]
    rows = [{**r, "evidence": extension._relative((source.root / r["evidence"]).resolve(), output)} for r in old]
    native = {(r["seed"], r["schedule"], r["rate"]): r for r in old if r["dimension"] == 1536}
    prefixes = {extension._cell_key(r): r for r in old if r["dimension"] != 1536}
    for row in new_rows:
        row = {**row, "paired_delta_native": _delta(row, native[row["seed"], row["schedule"], row["rate"]])}
        if extension._cell_key(row) in prefixes:
            row["paired_delta_prefix"] = _delta(row, prefixes[extension._cell_key(row)])
        rows.append(row)
    for original, current in zip(old, rows):
        if ({k: v for k, v in original.items() if k != "evidence"}
                != {k: v for k, v in current.items() if k != "evidence"}
                or (source.root / original["evidence"]).resolve() != (output / current["evidence"]).resolve()):
            raise ValueError("old representation row parity failed")
    for row in rows:
        if row["replay_hashes"] != source.original.replays[row["seed"], row["schedule"]]["hashes"]:
            raise ValueError("PCA exact replay pairing failed")
    return rows


def _replace_from_stage(stage: Path, target: Path) -> None:
    temporary = target.with_suffix(".publication.tmp")
    temporary.write_bytes(stage.read_bytes())
    temporary.replace(target)


def _recover_publication(output: Path, binding: str) -> None:
    path = output / "publication.json"
    if not path.exists():
        return
    journal = engine._read_record(path)
    names = ("aggregate.json", "pca_study_validation.json", "manifest.json")
    if journal["binding_sha256"] != binding or set(journal["targets"]) != set(names):
        raise ValueError("PCA publication journal binding mismatch")
    for name in names:
        digest = journal["targets"][name]
        stage, target = output / "publication" / name, output / name
        if engine.sha256_file(stage) != digest or (target.exists() and engine.sha256_file(target) != digest):
            raise ValueError("PCA publication artifacts changed")
    for name in names:  # Manifest is the last commit; incomplete work never has a completed manifest.
        if not (output / name).exists():
            _replace_from_stage(output / "publication" / name, output / name)


def run_pca(input_manifest: Path = DEFAULT_INPUT, *, pca_manifest: Path = DEFAULT_PREPARATION / "manifest.json",
            original: Path = extension.DEFAULT_BASELINE, baseline: Path = DEFAULT_BASELINE,
            output: Path = DEFAULT_OUTPUT, workers: int = 3, blas_threads: int = 4,
            resume: bool = False) -> dict:
    """Add only PCA cells; one coordinator, spawn workers, one job per seed/schedule/d."""
    workers = engine._positive_integer(workers, "workers")
    if workers > 3:
        raise ValueError("PCA replay is capped at three workers")
    threads = engine._positive_integer(blas_threads, "blas_threads")
    input_manifest, pca_manifest, original, baseline, output = map(
        lambda p: Path(p).resolve(), (input_manifest, pca_manifest, original, baseline, output))
    roots = (input_manifest.parent, pca_manifest.parent, original, baseline)
    _separate(output, roots)
    with parallel.exclusive_lock(output / "execution_locks/coordinator.lock"), threadpool_limits(limits=threads):
        vectors, labels, profile = load_input(input_manifest)
        source = _verify_sources(vectors, labels, profile, original, baseline)
        prepared = load_pca(input_manifest, pca_manifest)
        dims = prepared.manifest["binding"]["dimensions"]
        protocol = {
            **source.aggregate["protocol"],
            "dimensions": sorted(set(source.aggregate["protocol"]["dimensions"]) | set(dims), reverse=True),
            "representations": _representations(source.aggregate["protocol"]["dimensions"], dims),
        }
        added = len(dims) * len(protocol["seeds"]) * len(protocol["schedules"]) * len(protocol["rates"])
        protocol["planned_cells"] = len(source.aggregate["rows"]) + added
        source_files = {**source.files, **prepared.files, **_input_files(input_manifest, profile)}
        binding_payload = {
            "version": STUDY_VERSION, "original_binding": source.original.binding,
            "protocol": protocol, "pca_dimensions": dims,
            "pca_manifest_sha256": engine.sha256_file(pca_manifest),
            "baseline_aggregate_sha256": source.files[baseline / "aggregate.json"],
            "baseline_manifest_sha256": source.files[baseline / "manifest.json"],
            "source_files": {extension._relative(p, output): h for p, h in source_files.items()},
            "code_hashes": _code_hashes(), "blas_threads": threads,
        }
        binding = engine._digest(binding_payload)
        registration = {"binding_sha256": binding, "binding": binding_payload}
        prereg = output / "preregistration.json"
        if any(p.name != "execution_locks" for p in output.iterdir()):
            if not resume:
                raise FileExistsError("nonempty PCA replay output requires explicit resume=True")
            if not prereg.exists() or engine._read_record(prereg) != registration:
                raise ValueError("incompatible PCA replay source/parameters/code")
        else:
            engine._write_record(prereg, registration)
        jobs = [(s, a, d) for s in protocol["seeds"] for a in protocol["schedules"] for d in dims]
        for job in jobs:
            with parallel.exclusive_lock(_lock_path(output, job)):
                pass
        pending, new_rows, checkpoint_files = _inventory(output, registration, source.original.replays)
        preserved = dict(checkpoint_files)
        # Source integrity and complete checkpoint grid must precede publication recovery.
        if (output / "publication.json").exists():
            if pending:
                raise ValueError("publication journal exists for incomplete PCA grid")
            _recover_publication(output, binding)
        if (output / "manifest.json").exists():
            manifest = engine._read_record(output / "manifest.json")
            if (manifest["binding_sha256"] != binding or manifest["status"] != "completed"
                    or manifest["completed_cells"] != protocol["planned_cells"] or pending):
                raise ValueError("incompatible completed PCA manifest")
            retained = _bound_files(output, manifest["files"], (*roots, output))
            required = {**source_files, **checkpoint_files}
            required.update({p: engine.sha256_file(p) for p in (
                prereg, output / "progress.jsonl", output / "aggregate.json",
                output / "pca_study_validation.json",
            )})
            if retained != required:
                raise ValueError("completed PCA manifest omits or adds retained artifacts")
            aggregate = json.loads((output / "aggregate.json").read_text(encoding="utf-8"))
            if (aggregate["rows"] != _combined_rows(source, output, new_rows)
                    or aggregate["protocol"] != protocol or aggregate["binding_sha256"] != binding
                    or aggregate.get("summaries") != source.aggregate["summaries"]
                    or aggregate.get("summaries_scope") != "native_and_prefix_only"
                    or aggregate["status"] != "completed"):
                raise ValueError("completed PCA aggregate differs from retained rows")
            return aggregate
        log = output / "progress.jsonl"
        extension._progress(log, "pca_started", workers=workers, pending_jobs=len(pending))
        if pending:
            # Even workers=1 uses spawn, keeping preparation libraries out of scientific replay.
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(str(input_manifest), str(pca_manifest), str(original), str(output),
                          registration, source.original.replays),
            ) as executor:
                futures = [executor.submit(_run_job, job) for job in pending]
                try:
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
        pending, new_rows, checkpoint_files = _inventory(output, registration, source.original.replays)
        if pending or len(new_rows) != added:
            raise ValueError("cannot publish incomplete PCA study")
        extension._verify_files(preserved)
        extension._verify_files(source_files)
        extension._same_fingerprints(source.original.binding, vectors, labels, profile)
        if _code_hashes() != binding_payload["code_hashes"]:
            raise ValueError("PCA code changed during execution")
        rows = _combined_rows(source, output, new_rows)
        meta = prepared.public_metadata
        study = {
            "baseline_aggregate_sha256": binding_payload["baseline_aggregate_sha256"],
            "baseline_manifest_sha256": binding_payload["baseline_manifest_sha256"],
            "pca_manifest_sha256": binding_payload["pca_manifest_sha256"],
            "reused_cells": len(source.aggregate["rows"]), "added_cells": added, "dimensions": dims,
            "baseline_rows_unchanged": True, "replay_pairing_exact": True, "embedding_calls": 0, "judge_calls": 0,
            **{key: meta[key] for key in ("fit_scope", "fit_source_count", "solver", "whiten",
                                        "fit_seed", "fit_once", "fit_count", "labels_used_for_fit",
                                        "deduplicated", "transductive", "native_1536_control_note")},
        }
        aggregate = {
            "version": engine.VERSION, "status": "completed", "dataset": source.aggregate["dataset"],
            "protocol": protocol, "binding_sha256": binding, "provenance": binding_payload,
            "rows": rows, "pca_study": study, "pca_preparation": meta,
            # Preserve the published Results-tab summaries verbatim. Never run
            # the dimension-only summarizer on the combined family-aware grid.
            "summaries": source.aggregate["summaries"],
            "summaries_scope": "native_and_prefix_only",
            "source_provenance": {key: source.aggregate[key] for key in ("extension", "execution")
                                  if key in source.aggregate},
            "pca_execution": {"workers": workers, "blas_threads_per_worker": threads, "start_method": "spawn",
                              "job_unit": "seed/schedule/dimension; all budgets", "one_finalizer": True},
        }
        before = {key: study[key] for key in (
            "baseline_aggregate_sha256", "baseline_manifest_sha256", "pca_manifest_sha256")}
        after = {
            "baseline_aggregate_sha256": engine.sha256_file(baseline / "aggregate.json"),
            "baseline_manifest_sha256": engine.sha256_file(baseline / "manifest.json"),
            "pca_manifest_sha256": engine.sha256_file(pca_manifest),
        }
        if before != after:
            raise ValueError("source manifests changed during PCA replay")
        stage = output / "publication"
        engine.write_json(stage / "aggregate.json", aggregate)
        engine.write_json(stage / "pca_study_validation.json", {
            "version": "imdb-pca-study-validation-v1", "status": "passed", "ok": True,
            **study, "baseline_before": before, "baseline_after": after,
            "source_artifacts_unchanged": True, "source_artifacts_checked": len(source_files),
            "native_cache_unchanged": True, "pca_cache_unchanged": True,
            "input_arrays_unchanged": True, "method_code_hashes_unchanged": True, "runtime_unchanged": True,
            "fit_artifact_integrity_verified": True, "fit_verification": meta["verification"],
            "fit_count": meta["fit_count"],
            "combined_cells": len(rows), "replay_count": len(source.original.replays),
            "protocol": protocol, "aggregate_sha256": engine.sha256_file(stage / "aggregate.json"),
        })
        extension._progress(log, "pca_completed", combined_cells=len(rows))
        files = {extension._relative(p, output): h for p, h in {**source_files, **checkpoint_files}.items()}
        files.update({extension._relative(p, output): engine.sha256_file(p) for p in (prereg, log)})
        for name in ("aggregate.json", "pca_study_validation.json"):
            files[name] = engine.sha256_file(stage / name)
        engine._write_record(stage / "manifest.json", {
            "version": engine.VERSION, "status": "completed", "binding_sha256": binding,
            "completed_cells": len(rows), "files": files, "pca_study": study,
        })
        engine._write_record(output / "publication.json", {
            "binding_sha256": binding, "targets": {
                name: engine.sha256_file(stage / name)
                for name in ("aggregate.json", "pca_study_validation.json", "manifest.json")
            },
        })
        _recover_publication(output, binding)
        return aggregate
