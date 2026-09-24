"""Tiny real PCA fits and frozen replay checks only; never launch production work."""
from __future__ import annotations

from concurrent.futures import Future
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from sampling_comparison import imdb_experiment as engine
from sampling_comparison import imdb_extension as extension
from sampling_comparison import imdb_pca as pca
from sampling_comparison.imdb_inputs import MODEL, POLICY, VERSION, load_input


ROOT = Path(__file__).resolve().parents[1]


def snapshot(root):
    return {p.relative_to(root).as_posix(): engine.sha256_file(p) for p in root.rglob("*") if p.is_file()}


def make_input(root, *, flip=False):
    root.mkdir()
    vectors = np.random.default_rng(314).normal(size=(24, 1536)).astype(np.float32)
    vectors[-1] = vectors[0]  # Two distinct source rows; do not deduplicate before fitting.
    np.save(root / "vectors.npy", vectors)
    units = [{"unit_id": f"fixture-{i}", "label": (i + int(flip)) % 2} for i in range(len(vectors))]
    engine.write_json(root / "units.json", units)
    engine.write_json(root / "manifest.json", {
        "version": VERSION, "status": "complete", "model": MODEL, "dimensions": 1536,
        "sessions": len(vectors), "dataset_id": "imdb_50000", "agents": 1,
        "representation_policy": POLICY, "label_source": "Offline PCA test fixture only",
        "files": {"vectors": "vectors.npy", "units": "units.json"},
        "hashes": {name: engine.sha256_file(root / filename)
                   for name, filename in (("vectors", "vectors.npy"), ("units", "units.json"))},
        "units_sha256": engine._digest(units),
    })
    return root / "manifest.json"


def fit(manifest, output, *extra):
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/prepare_imdb_pca.py"), "--input", str(manifest),
        "--output", str(output), "--dimensions", "16", "8", *extra,
    ], cwd=ROOT, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("pca-fixture")
    manifest = make_input(root / "native")
    prepared = root / "prepared"
    fit(manifest, prepared)
    vectors, labels, profile = load_input(manifest)
    original, baseline = root / "original", root / "prefix"
    with threadpool_limits(limits=4):
        engine.run_experiment(
            vectors, labels, original, profile=profile, dimensions=(1536, 8), repetitions=2,
            schedules=engine.SCHEDULES, rates=(.05, .1, .25, .5, .75),
            target_block_size=4, donor_block_size=3, calibration_reservoir_size=4,
        )
        extension.run_extension(vectors, labels, baseline, profile=profile,
                                baseline=original, dimensions=(16,))
    return root


def run(root, output, **kwargs):
    return pca.run_pca(root / "native/manifest.json", pca_manifest=root / "prepared/manifest.json",
                       original=root / "original", baseline=root / "prefix", output=output, **kwargs)


class SerialExecutor:
    """Test oracle only: execute the identical worker synchronously."""
    def __init__(self, *, initializer, initargs, **kwargs):
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def test_real_fit_transform_norm_distance_and_label_blindness(bundles, tmp_path):
    before = snapshot(bundles)
    prepared = pca.load_pca(bundles / "native/manifest.json", bundles / "prepared/manifest.json")
    vectors, _, _ = load_input(bundles / "native/manifest.json")
    native = engine.normalized_prefix(vectors, 1536)
    arrays = prepared.arrays
    independent = (native.astype(float) - arrays["mean"]) @ arrays["components"].astype(float).T
    np.testing.assert_allclose(arrays["projected"], independent, atol=1e-5, rtol=5e-5)
    np.testing.assert_allclose(arrays["mean"], native.mean(axis=0), atol=1e-7)
    np.testing.assert_allclose(arrays["projected"][0], arrays["projected"][-1], atol=1e-7)
    for d in (16, 8):
        actual = engine.normalized_prefix(arrays["projected"], d)
        expected = independent[:, :d] / np.linalg.norm(independent[:, :d], axis=1, keepdims=True)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=2e-7)
        a = actual.astype(float)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        _, angles = engine._angular_block(a[:3], a[3:6])
        np.testing.assert_allclose(angles, np.arccos(np.clip(expected[:3] @ expected[3:6].T, -1, 1)) / np.pi,
                                   atol=2e-7)
    assert all(isinstance(a, np.memmap) and not a.flags.writeable for a in arrays.values())
    assert prepared.manifest["fit"]["fit_count"] == 1
    assert prepared.manifest["binding"]["fit_source_count"] == 24
    assert prepared.manifest["binding"]["labels_used_for_fit"] is False
    assert prepared.manifest["fit"]["normalized_input_sha256"] == engine.array_sha256(native)
    assert prepared.public_metadata["labels_used_for_fit"] is False
    assert prepared.public_metadata["deduplicated"] is False
    metadata = prepared.public_metadata
    assert metadata["explained_variance_by_dimension"] == {
        str(d): float(arrays["explained_variance_ratio"][:d].astype(float).sum()) for d in (16, 8)
    }
    assert metadata["explained_variance_by_dimension"] == metadata["cumulative_explained_variance_ratio"]
    assert metadata["fit_seconds"] == prepared.manifest["fit"]["elapsed_seconds"]
    assert metadata["fit_seconds"] > 0
    assert metadata["solver"] == "full" and metadata["whiten"] is False
    assert metadata["fit_scope"] == "full_unlabeled_source_population"
    assert metadata["native_1536_control_note"] == pca.CONTROL_NOTE
    assert "labels" not in inspect.signature(pca._fit_arrays).parameters
    changed_input = make_input(tmp_path / "flipped", flip=True)
    fit(changed_input, tmp_path / "flipped-pca")
    flipped = pca.load_pca(changed_input, tmp_path / "flipped-pca/manifest.json")
    for name in pca.ARRAYS:
        np.testing.assert_array_equal(arrays[name], flipped.arrays[name])
    assert snapshot(bundles) == before
    public = json.dumps(prepared.public_metadata)
    assert not any(word in public for word in ("unit_id", "packet_sha256", "source_manifest", "endpoint"))


def test_full_rank_centered_control_is_not_native():
    # A genuine small full-feature fit tests the same verifier without a 1536d fit.
    code = """
import numpy as np
from threadpoolctl import threadpool_limits
from sampling_comparison import imdb_pca as p
from sampling_comparison import imdb_experiment as e
from sklearn.decomposition import PCA
native = e.normalized_prefix(np.random.default_rng(91).normal(size=(24,8)) + 2, 8)
original = PCA.fit
calls = []
def spy(self, X, y=None):
    assert y is None and X.shape == (24,8) and X.dtype == np.float32
    assert self.svd_solver == 'full' and not self.whiten and self.random_state == 13
    calls.append(1)
    return original(self, X)
PCA.fit = spy
with threadpool_limits(limits=4):
    arrays, meta = p._fit_arrays(native, 8)
    proof = p._verify_geometry(native, arrays)
assert len(calls) == 1 and meta['fit_count'] == 1
assert proof['full_component_centered_identity_verified']
center = native.astype(float) - arrays['mean']
center /= np.linalg.norm(center,axis=1,keepdims=True)
scores = e.normalized_prefix(arrays['projected'],8)
np.testing.assert_allclose(scores @ scores.T, center @ center.T, atol=2e-6)
assert not np.allclose(scores @ scores.T, native @ native.T, atol=.01)
bad = dict(arrays, projected=arrays['projected'] / np.sqrt(arrays['explained_variance']))
try:
    p._verify_geometry(native,bad)
except ValueError:
    pass
else:
    raise AssertionError('whitened scores accepted')
print('centered control verified')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "centered control verified" in result.stdout


def test_preparation_resume_is_fit_once_and_immutable(bundles, monkeypatch):
    before = snapshot(bundles / "prepared")
    monkeypatch.setattr(pca, "_fit_arrays", lambda *a: pytest.fail("refitted prepared PCA"))
    result = pca.prepare_pca(bundles / "native/manifest.json", bundles / "prepared",
                             dimensions=(16, 8), resume=True)
    assert result["fit"]["fit_count"] == 1
    assert snapshot(bundles / "prepared") == before
    with pytest.raises(FileExistsError, match="resume"):
        pca.prepare_pca(bundles / "native/manifest.json", bundles / "prepared", dimensions=(16, 8))
    with pytest.raises(ValueError, match="dimensions"):
        pca.prepare_pca(bundles / "native/manifest.json", bundles / "prepared", resume=True)


@pytest.mark.parametrize("change", ["source", "array", "parameters", "code", "partial"])
def test_preparation_fails_closed(bundles, tmp_path, monkeypatch, change):
    root = tmp_path / "copy"
    shutil.copytree(bundles, root)
    if change == "source":
        with (root / "native/vectors.npy").open("ab") as stream:
            stream.write(b"changed")
    elif change == "array":
        with (root / "prepared/projected.npy").open("ab") as stream:
            stream.write(b"changed")
    elif change == "code":
        monkeypatch.setattr(pca, "_code_hashes", lambda: {"changed": "hash"})
    elif change == "partial":
        (root / "prepared/manifest.json").unlink()
        (root / "prepared/fit.json").unlink()
    monkeypatch.setattr(pca, "_fit_arrays", lambda *a: pytest.fail("silently refitted corrupt cache"))
    with pytest.raises(ValueError):
        pca.prepare_pca(root / "native/manifest.json", root / "prepared",
                         dimensions=(12, 8) if change == "parameters" else (16, 8), resume=True)


def test_preparation_final_manifest_recovery_without_refit(bundles, tmp_path, monkeypatch):
    root = tmp_path / "copy"
    shutil.copytree(bundles, root)
    before = (root / "prepared/manifest.json").read_bytes()
    (root / "prepared/manifest.json").unlink()
    monkeypatch.setattr(pca, "_fit_arrays", lambda *a: pytest.fail("refitted committed cache"))
    pca.prepare_pca(root / "native/manifest.json", root / "prepared", dimensions=(16, 8), resume=True)
    assert (root / "prepared/manifest.json").read_bytes() == before


def test_runner_import_never_loads_sklearn():
    result = subprocess.run([sys.executable, "-c", (
        "import sys; from sampling_comparison import imdb_pca; "
        "imdb_pca._versions(); assert not any(k.startswith('sklearn') for k in sys.modules)"
    )], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_three_spawn_workers_match_serial_and_preserve_all_old_bytes(bundles, tmp_path, monkeypatch):
    before = snapshot(bundles)
    output = tmp_path / "parallel"
    result = run(bundles, output)
    old = json.loads((bundles / "prefix/aggregate.json").read_text())
    assert result["status"] == "completed" and len(result["rows"]) == 100
    assert result["pca_study"]["reused_cells"] == 60
    assert result["pca_study"]["added_cells"] == 40
    assert result["pca_study"]["baseline_rows_unchanged"] is True
    assert result["pca_study"]["replay_pairing_exact"] is True
    assert result["pca_study"]["fit_count"] == 1 and result["pca_study"]["fit_once"] is True
    assert result["pca_study"]["solver"] == "full" and result["pca_study"]["whiten"] is False
    assert result["pca_study"]["labels_used_for_fit"] is False
    assert result["pca_study"]["fit_scope"] == "full_unlabeled_source_population"
    assert result["pca_study"]["transductive"] is True and result["pca_study"]["deduplicated"] is False
    assert result["pca_study"]["baseline_aggregate_sha256"] == engine.sha256_file(bundles / "prefix/aggregate.json")
    assert result["pca_study"]["baseline_manifest_sha256"] == engine.sha256_file(bundles / "prefix/manifest.json")
    assert result["pca_execution"]["workers"] == 3
    assert result["dataset"] == old["dataset"]
    assert "extension" not in result and "execution" not in result
    assert result["summaries"] == old["summaries"]
    assert result["summaries_scope"] == "native_and_prefix_only"
    assert result["source_provenance"]["extension"] == old["extension"]
    for original, reused in zip(old["rows"], result["rows"]):
        assert "representation" not in reused and "representation_id" not in reused
        assert {k: v for k, v in original.items() if k != "evidence"} == {
            k: v for k, v in reused.items() if k != "evidence"}
        assert (output / reused["evidence"]).resolve() == (bundles / "prefix" / original["evidence"]).resolve()
    native = {(r["seed"], r["schedule"], r["rate"]): r for r in old["rows"] if r["dimension"] == 1536}
    for row in result["rows"][60:]:
        assert row["representation"] == "pca" and row["representation_id"] == f"pca_{row['dimension']}"
        ref = native[row["seed"], row["schedule"], row["rate"]]
        assert row["replay_hashes"] == ref["replay_hashes"]
        assert row["paired_delta_native"] == pca._delta(row, ref)
        with np.load(output / row["evidence"], allow_pickle=False) as evidence:
            assert evidence["score"].dtype == np.float64
    events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    assert len({e["worker_pid"] for e in events if e["event"] == "pca_job_completed"}) >= 2
    assert snapshot(bundles) == before
    manifest = engine._read_record(output / "manifest.json")
    for name, digest in manifest["files"].items():
        assert engine.sha256_file(output / name) == digest
    audit = json.loads((output / "pca_study_validation.json").read_text())
    assert audit["baseline_before"] == audit["baseline_after"]
    assert audit["fit_artifact_integrity_verified"] is True
    assert audit["aggregate_sha256"] == engine.sha256_file(output / "aggregate.json")
    ranks = evaluations = 0
    original_rank, original_evaluate = engine.rank_membership, engine.evaluate_replay

    def rank(*args, **kwargs):
        nonlocal ranks
        ranks += 1
        assert args[3].shape[1] in (16, 8)
        return original_rank(*args, **kwargs)

    def evaluate(*args, **kwargs):
        nonlocal evaluations
        evaluations += 1
        assert args[0].shape[1] in (16, 8)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(pca, "ProcessPoolExecutor", SerialExecutor)
    monkeypatch.setattr(engine, "rank_membership", rank)
    monkeypatch.setattr(engine, "evaluate_replay", evaluate)
    monkeypatch.setattr(engine, "bootstrap_replay", lambda *a: pytest.fail("regenerated draws"))
    monkeypatch.setattr(engine, "summarize", lambda *a: pytest.fail("dimension-only combined summary"))
    monkeypatch.setattr(extension, "rank_membership", lambda *a: pytest.fail("recomputed old membership"))
    monkeypatch.setattr(extension, "evaluate_replay", lambda *a: pytest.fail("recomputed old estimates"))
    serial_output = tmp_path / "serial"
    serial = run(bundles, serial_output, workers=1)
    assert ranks == 8 and evaluations == 40
    for row, other in zip(result["rows"], serial["rows"]):
        assert {k: v for k, v in row.items() if k not in ("elapsed_seconds", "evidence")} == {
            k: v for k, v in other.items() if k not in ("elapsed_seconds", "evidence")}
        with np.load(output / row["evidence"]) as first, np.load(serial_output / other["evidence"]) as second:
            assert first.files == second.files
            for name in first.files:
                np.testing.assert_array_equal(first[name], second[name])
    output_before = snapshot(output)
    monkeypatch.setattr(pca, "ProcessPoolExecutor", lambda **kw: pytest.fail("recomputed completed run"))
    assert run(bundles, output, resume=True) == result
    assert snapshot(output) == output_before and snapshot(bundles) == before


def test_worker_failure_preserves_checkpoints_and_resumes(bundles, tmp_path, monkeypatch):
    output = tmp_path / "run"
    monkeypatch.setattr(pca, "ProcessPoolExecutor", SerialExecutor)
    evaluate = engine.evaluate_replay
    calls = 0

    def fail(*a, **kw):
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise RuntimeError("injected worker failure")
        return evaluate(*a, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "evaluate_replay", fail)
        with pytest.raises(RuntimeError, match="injected worker"):
            run(bundles, output)
    assert not (output / "manifest.json").exists() and not (output / "aggregate.json").exists()
    cells = snapshot(output / "cells")
    assert len(cells) == 4
    result = run(bundles, output, resume=True)
    assert result["status"] == "completed"
    for name, digest in cells.items():
        assert engine.sha256_file(output / "cells" / name) == digest


def test_failed_checkpoint_commit_is_not_silently_overwritten(bundles, tmp_path, monkeypatch):
    output = tmp_path / "run"
    monkeypatch.setattr(pca, "ProcessPoolExecutor", SerialExecutor)
    write = engine._write_record

    def fail(path, value):
        if path.parent.name == "cells":
            raise RuntimeError("injected checkpoint failure")
        return write(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_write_record", fail)
        with pytest.raises(RuntimeError, match="checkpoint failure"):
            run(bundles, output)
    assert not (output / "manifest.json").exists()
    before = snapshot(output / "cells")
    with pytest.raises(ValueError, match="partial checkpoint"):
        run(bundles, output, resume=True)
    assert snapshot(output / "cells") == before


def test_failure_after_checkpoint_commit_resumes_without_rewriting(bundles, tmp_path, monkeypatch):
    output = tmp_path / "run"
    monkeypatch.setattr(pca, "ProcessPoolExecutor", SerialExecutor)
    checkpoint = engine._checkpoint
    failed = False

    def fail_after_commit(path, *args, **kwargs):
        nonlocal failed
        result = checkpoint(path, *args, **kwargs)
        if path.parent.name == "cells" and not failed:
            failed = True
            raise RuntimeError("injected failure after checkpoint commit")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_checkpoint", fail_after_commit)
        with pytest.raises(RuntimeError, match="after checkpoint commit"):
            run(bundles, output)
    before = snapshot(output / "cells")
    assert before and not (output / "manifest.json").exists()
    assert run(bundles, output, resume=True)["status"] == "completed"
    for name, digest in before.items():
        assert engine.sha256_file(output / "cells" / name) == digest


def test_completed_manifest_cannot_omit_source_fit_arrays(bundles, tmp_path, monkeypatch):
    output = tmp_path / "run"
    monkeypatch.setattr(pca, "ProcessPoolExecutor", SerialExecutor)
    run(bundles, output)
    manifest = engine._read_record(output / "manifest.json")
    name = next(k for k in manifest["files"] if k.endswith("/components.npy"))
    manifest["files"].pop(name)
    # Keep the publication journal consistent too; required inventory is independently checked.
    engine._write_record(output / "manifest.json", manifest)
    engine._write_record(output / "publication/manifest.json", manifest)
    journal = engine._read_record(output / "publication.json")
    journal["targets"]["manifest.json"] = engine.sha256_file(output / "manifest.json")
    engine._write_record(output / "publication.json", journal)
    with pytest.raises(ValueError, match="omits or adds"):
        run(bundles, output, resume=True)


def test_interrupted_publication_recovers_without_replay(bundles, tmp_path, monkeypatch):
    output = tmp_path / "run"
    monkeypatch.setattr(pca, "ProcessPoolExecutor", SerialExecutor)
    replace = pca._replace_from_stage

    def interrupt(stage, target):
        if target.name == "pca_study_validation.json":
            raise RuntimeError("injected publication failure")
        return replace(stage, target)

    with monkeypatch.context() as patch:
        patch.setattr(pca, "_replace_from_stage", interrupt)
        with pytest.raises(RuntimeError, match="publication failure"):
            run(bundles, output)
    assert not (output / "manifest.json").exists()
    before = snapshot(output / "cells")
    monkeypatch.setattr(pca, "ProcessPoolExecutor", lambda **kw: pytest.fail("replayed during finalization"))
    assert run(bundles, output, resume=True)["status"] == "completed"
    assert snapshot(output / "cells") == before


@pytest.mark.parametrize("change", ["inherited", "escape", "pairing", "original", "pca"])
def test_source_verification_precedes_any_new_replay(bundles, tmp_path, monkeypatch, change):
    root = tmp_path / "copy"
    shutil.copytree(bundles, root)
    if change == "original":
        with (root / "original/aggregate.json").open("ab") as stream:
            stream.write(b" ")
    elif change == "pca":
        with (root / "prepared/components.npy").open("ab") as stream:
            stream.write(b"x")
    elif change == "pairing":
        # Hash-consistent aggregate tamper still fails retained-row/replay verification.
        path = root / "prefix/aggregate.json"
        aggregate = json.loads(path.read_text())
        aggregate["rows"][-1]["replay_hashes"]["source_id"] = "changed"
        engine.write_json(path, aggregate)
        manifest = engine._read_record(root / "prefix/manifest.json")
        manifest["files"]["aggregate.json"] = engine.sha256_file(path)
        engine._write_record(root / "prefix/manifest.json", manifest)
    else:
        manifest = engine._read_record(root / "prefix/manifest.json")
        if change == "inherited":
            name = next(k for k in manifest["files"] if k.startswith("../original/"))
            manifest["files"].pop(name)
        else:
            outside = tmp_path / "outside"
            outside.write_text("not a source artifact")
            manifest["files"]["../../outside"] = engine.sha256_file(outside)
        engine._write_record(root / "prefix/manifest.json", manifest)
    monkeypatch.setattr(pca, "ProcessPoolExecutor", lambda **kw: pytest.fail("launched on invalid sources"))
    with pytest.raises(ValueError):
        run(root, tmp_path / "run")
    assert not (tmp_path / "run/preregistration.json").exists()


def test_coordinator_and_job_exclusion(bundles, tmp_path, monkeypatch):
    output = tmp_path / "run"
    monkeypatch.setattr(pca, "ProcessPoolExecutor", lambda **kw: pytest.fail("launched with another owner"))
    with pca.parallel.exclusive_lock(output / "execution_locks/coordinator.lock"):
        with pytest.raises(RuntimeError, match="another coordinator"):
            run(bundles, output)
    with pca.parallel.exclusive_lock(pca._lock_path(output, (13, "bursty", 8))):
        with pytest.raises(RuntimeError, match="another coordinator"):
            run(bundles, output)
    assert not (output / "manifest.json").exists()


def test_production_family_grid_separates_native_and_pca_control():
    reps = pca._representations(pca.DIMENSIONS, pca.DIMENSIONS)
    assert len(reps) == 18 and len({r["id"] for r in reps}) == 18
    assert [r for r in reps if r["dimension"] == 1536] == [
        {"id": "native_1536", "family": "native", "dimension": 1536},
        {"id": "pca_1536", "family": "pca", "dimension": 1536},
    ]
    assert sum(r["family"] == "prefix" for r in reps) == 8
    assert len(reps) * 40 * 2 * 5 == 7200
    assert "not native1536" in pca.CONTROL_NOTE


def test_pca_1536_delta_uses_original_native_not_itself(bundles, tmp_path):
    vectors, labels, profile = load_input(bundles / "native/manifest.json")
    with threadpool_limits(limits=4):
        source = pca._verify_sources(vectors, labels, profile, bundles / "original", bundles / "prefix")
    native = next(row for row in source.aggregate["rows"] if row["dimension"] == 1536)
    control = json.loads(json.dumps(native))
    control.update(representation="pca", representation_id="pca_1536")
    control["all_unselected"]["mae"] += .125
    rows = pca._combined_rows(source, tmp_path, [control])
    assert rows[-1]["paired_delta_native"]["all_unselected"]["mae"] == pytest.approx(.125)
    assert rows[-1]["representation_id"] == "pca_1536"
    assert "paired_delta_prefix" not in rows[-1]
    assert source.aggregate["rows"][0] == native


@pytest.mark.parametrize("workers", [0, -1, 4, True, 1.5])
def test_invalid_worker_count_rejected_before_io(tmp_path, workers):
    with pytest.raises(ValueError):
        pca.run_pca(tmp_path / "missing", workers=workers)
