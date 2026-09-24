"""Spawn real worker processes on tiny fixtures; no synthetic benchmark results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from sampling_comparison import imdb_experiment as engine
from sampling_comparison import imdb_extension as extension
from sampling_comparison import imdb_parallel as parallel
from sampling_comparison.imdb_inputs import MODEL, POLICY, VERSION, load_input


def create_fixture(tmp_path, monkeypatch, *, stop_after=2):
    cache = tmp_path / "cache"
    cache.mkdir()
    vectors = np.random.default_rng(314).normal(size=(24, 1536)).astype(np.float32)
    np.save(cache / "vectors.npy", vectors)
    units = [{"unit_id": f"fixture-{i}", "label": i % 2} for i in range(len(vectors))]
    engine.write_json(cache / "units.json", units)
    profile = {
        "version": VERSION, "status": "complete", "model": MODEL, "dimensions": 1536,
        "sessions": len(vectors), "dataset_id": "imdb_50000", "agents": 1,
        "representation_policy": POLICY, "label_source": "Offline test fixture only",
        "files": {"vectors": "vectors.npy", "units": "units.json"},
        "hashes": {name: engine.sha256_file(cache / filename)
                   for name, filename in (("vectors", "vectors.npy"), ("units", "units.json"))},
        "units_sha256": engine._digest(units),
    }
    manifest = cache / "manifest.json"
    engine.write_json(manifest, profile)
    matrix, labels, prepared = load_input(manifest)
    baseline, output = tmp_path / "baseline", tmp_path / "extension"
    with threadpool_limits(limits=4):
        engine.run_experiment(
            matrix, labels, baseline, profile=prepared, dimensions=(1536, 8),
            repetitions=2, schedules=("uniformly_random", "bursty"), rates=(.25, .5),
            target_block_size=4, donor_block_size=3, calibration_reservoir_size=4,
        )
    calls = 0
    original = extension.evaluate_replay

    def interrupt(*args, **kwargs):
        nonlocal calls
        if calls == stop_after:
            raise RuntimeError("test serial interruption")
        calls += 1
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(extension, "evaluate_replay", interrupt)
        with pytest.raises(RuntimeError, match="test serial interruption"):
            extension.run_extension(matrix, labels, output, profile=prepared, baseline=baseline)
    return manifest, baseline, output


def snapshot(root):
    return {str(p.relative_to(root)): engine.sha256_file(p)
            for p in root.rglob("*") if p.is_file()}


def run(fixture, **kwargs):
    manifest, baseline, output = fixture
    return parallel.run_parallel(manifest, baseline=baseline, output=output, **kwargs)


def test_three_workers_preserve_serial_progress_and_exact_results(tmp_path, monkeypatch):
    fixture = create_fixture(tmp_path, monkeypatch)
    manifest, baseline, output = fixture
    source_snapshot = snapshot(baseline)
    cache_snapshot = snapshot(manifest.parent)
    preserved = snapshot(output)
    result = run(fixture)
    assert result["status"] == "completed"
    assert result["execution"]["workers"] == 3
    assert result["execution"]["checkpoint_cells_preserved_at_switch"] == 2
    assert result["execution"]["scientific_binding_unchanged"] is True
    assert snapshot(baseline) == source_snapshot
    assert snapshot(manifest.parent) == cache_snapshot
    for relative, digest in preserved.items():
        if relative != "progress.jsonl":
            assert engine.sha256_file(output / relative) == digest
    assert len(result["rows"]) == 40
    audit = json.loads((output / "extension_validation.json").read_text())
    assert audit["aggregate_sha256"] == engine.sha256_file(output / "aggregate.json")
    final_manifest = engine._read_record(output / "manifest.json")
    for name, digest in final_manifest["files"].items():
        assert engine.sha256_file(output / name) == digest
    events = [json.loads(line) for line in (output / "parallel_progress.jsonl").read_text().splitlines()]
    completed = [event for event in events if event["event"] == "group_completed"]
    assert len({(event["seed"], event["schedule"]) for event in completed}) == 4
    assert sum(event["computed_cells"] for event in completed) == 22
    assert sum(event["reused_cells"] for event in completed) == 2
    assert len({event["worker_pid"] for event in completed}) >= 2
    assert events[-1]["event"] == "parallel_completed"
    source, labels, profile = load_input(manifest)
    oracle = tmp_path / "serial-oracle"
    expected = extension.run_extension(source, labels, oracle, profile=profile, baseline=baseline)
    original_rows = {extension._cell_key(row): row for row in expected["rows"]}
    for row in result["rows"]:
        other = original_rows[extension._cell_key(row)]
        fields = {key: value for key, value in row.items() if key not in ("evidence", "elapsed_seconds")}
        other_fields = {key: value for key, value in other.items() if key not in ("evidence", "elapsed_seconds")}
        assert engine.canonical(fields) == engine.canonical(other_fields)
    before_resume = snapshot(output)
    rerun = run(fixture)
    assert engine.canonical(rerun) == engine.canonical(result)
    assert snapshot(output) == before_resume
    from scripts.validate_imdb_experiment import validate_execution, validate_extension

    assert validate_extension(result, final_manifest, output)["reused_cells"] == 16
    assert validate_execution(result, final_manifest, output)["checkpoint_cells_preserved_at_switch"] == 2


def test_failed_pool_does_not_rewrite_progress_and_can_resume(tmp_path, monkeypatch):
    fixture = create_fixture(tmp_path, monkeypatch)
    output = fixture[2]
    before = snapshot(output)

    class CannotStart:
        def __init__(self, **kwargs):
            raise RuntimeError("test worker startup failure")

    with monkeypatch.context() as patch:
        patch.setattr(parallel, "ProcessPoolExecutor", CannotStart)
        with pytest.raises(RuntimeError, match="worker startup"):
            run(fixture)
    for relative, digest in before.items():
        assert engine.sha256_file(output / relative) == digest
    transition = engine._read_record(output / "parallel_transition.json")
    assert transition["preserved_cells"] == 2
    assert run(fixture)["status"] == "completed"


def test_interrupted_final_publication_recovers_without_recomputing(tmp_path, monkeypatch):
    fixture = create_fixture(tmp_path, monkeypatch)
    original = parallel._replace_from_stage
    calls = 0

    def fail_second_replacement(stage, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("test interruption publishing audit")
        original(stage, target)

    with monkeypatch.context() as patch:
        patch.setattr(parallel, "_replace_from_stage", fail_second_replacement)
        with pytest.raises(RuntimeError, match="publishing audit"):
            run(fixture)
    output = fixture[2]
    checkpoint_snapshot = snapshot(output / "cells")
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("recomputed finished jobs"))
    result = run(fixture)
    assert result["status"] == "completed"
    assert snapshot(output / "cells") == checkpoint_snapshot
    manifest = engine._read_record(output / "manifest.json")
    for name, digest in manifest["files"].items():
        assert engine.sha256_file(output / name) == digest


def test_global_and_group_locks_prevent_duplicate_owners(tmp_path, monkeypatch):
    path = tmp_path / "lock"
    with parallel.exclusive_lock(path):
        with pytest.raises(RuntimeError, match="another coordinator or worker"):
            with parallel.exclusive_lock(path):
                pytest.fail("second owner acquired lock")
    with parallel.exclusive_lock(path):
        pass
    fixture = create_fixture(tmp_path, monkeypatch)
    output = fixture[2]
    with parallel.exclusive_lock(output / "execution_locks" / "coordinator.lock"):
        with pytest.raises(RuntimeError, match="coordinator"):
            run(fixture)
    with parallel.exclusive_lock(parallel._job_lock(output, (13, "bursty"))):
        with pytest.raises(RuntimeError, match="another coordinator or worker"):
            run(fixture)
    assert not (output / "parallel_transition.json").exists()


@pytest.mark.parametrize("change", ["checkpoint", "scheduler", "extension", "workers"])
def test_transition_fails_closed_on_drift(tmp_path, monkeypatch, change):
    fixture = create_fixture(tmp_path, monkeypatch)
    output = fixture[2]

    class CannotStart:
        def __init__(self, **kwargs):
            raise RuntimeError("pause before workers")

    with monkeypatch.context() as patch:
        patch.setattr(parallel, "ProcessPoolExecutor", CannotStart)
        with pytest.raises(RuntimeError, match="pause"):
            run(fixture)
    if change == "checkpoint":
        path = next((output / "cells").glob("*.npz"))
        path.write_bytes(b"tampered")
        message = "evidence hash mismatch"
    elif change == "scheduler":
        monkeypatch.setattr(parallel, "_code_hashes", lambda: {"changed.py": "different"})
        message = "execution fingerprint changed"
    elif change == "extension":
        monkeypatch.setattr(extension, "_extension_code_hashes", lambda: {"changed.py": "different"})
        message = "unchanged original extension binding"
    else:
        message = "execution fingerprint changed"
    with pytest.raises(ValueError, match=message):
        run(fixture, workers=2 if change == "workers" else 3)


@pytest.mark.parametrize("workers", [0, -1, 4, True, 1.5])
def test_worker_limit_rejects_invalid_values(tmp_path, workers):
    with pytest.raises(ValueError):
        parallel.run_parallel(tmp_path / "missing.json", output=tmp_path, workers=workers)


def test_requires_existing_registration(tmp_path):
    with pytest.raises(ValueError, match="existing serial extension preregistration"):
        parallel.run_parallel(tmp_path / "missing.json", output=tmp_path)
