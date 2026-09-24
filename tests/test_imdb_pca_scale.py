"""Scale real tiny-fixture work without rewriting committed scientific evidence."""
from __future__ import annotations

import json

import pytest

from sampling_comparison import imdb_pca as pca
from sampling_comparison import imdb_pca_scale as scale
from sampling_comparison.imdb_experiment import _read_record, canonical, sha256_file
from sampling_comparison.imdb_parallel import exclusive_lock
from test_imdb_pca import SerialExecutor, bundles, run as original_run, snapshot


def interrupt_study(root, output, monkeypatch):
    original = pca.engine.evaluate_replay
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        if calls >= 2:
            raise RuntimeError("test interruption after two cells")
        calls += 1
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pca, "ProcessPoolExecutor", SerialExecutor)
        patch.setattr(pca.engine, "evaluate_replay", interrupted)
        with pytest.raises(RuntimeError, match="after two cells"):
            original_run(root, output)
    assert len(list((output / "cells").glob("*.json"))) == 2


def run(root, output, **kwargs):
    return scale.run_scaled(
        root / "native/manifest.json", pca_manifest=root / "prepared/manifest.json",
        original=root / "original", baseline=root / "prefix", output=output, **kwargs,
    )


def test_six_workers_reuse_every_checkpoint_and_match_original_pipeline(bundles, tmp_path, monkeypatch):
    output = tmp_path / "scaled"
    interrupt_study(bundles, output, monkeypatch)
    preserved = snapshot(output)
    sources = snapshot(bundles)
    code_hashes = pca._code_hashes()
    result = run(bundles, output)
    assert result["pca_execution"]["workers"] == 6
    scaling = result["pca_execution"]["scaling"]
    assert scaling["preserved_cells"] == 2
    assert scaling["completed_checkpoints_unchanged"] is True
    assert scaling["scientific_binding_unchanged"] is True
    for name, digest in preserved.items():
        if name != "progress.jsonl":
            assert sha256_file(output / name) == digest
    assert snapshot(bundles) == sources
    assert pca._code_hashes() == code_hashes
    oracle = original_run(bundles, tmp_path / "oracle", workers=1)
    expected = {(r.get("representation"), r["seed"], r["schedule"], r["dimension"], r["rate"]): r
                for r in oracle["rows"]}
    for row in result["rows"]:
        other = expected[row.get("representation"), row["seed"], row["schedule"], row["dimension"], row["rate"]]
        assert canonical({k: v for k, v in row.items() if k not in ("evidence", "elapsed_seconds")}) == canonical({
            k: v for k, v in other.items() if k not in ("evidence", "elapsed_seconds")
        })
    manifest = _read_record(output / "manifest.json")
    for name, digest in manifest["files"].items():
        assert sha256_file(output / name) == digest
    events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    assert any(event["event"] == "pca_scaled_started" and event["workers"] == 6 for event in events)
    assert len({e["worker_pid"] for e in events if e["event"] == "pca_job_completed"}) >= 2
    before_resume = snapshot(output)
    assert run(bundles, output) == result
    assert snapshot(output) == before_resume
    # The frozen three-worker entry point still recognizes the completed scientific bundle.
    assert original_run(bundles, output, resume=True) == result
    assert snapshot(output) == before_resume
    from scripts.validate_imdb_experiment import validate_pca_scaling

    assert validate_pca_scaling(result, output)["preserved_cells"] == 2


def test_checkpoint_orphans_are_archived_not_deleted(bundles, tmp_path, monkeypatch):
    output = tmp_path / "scaled"
    interrupt_study(bundles, output, monkeypatch)
    orphan = output / "cells/s13-uniformly_random-pca16-r2.npz"
    temporary = output / "cells/s13-uniformly_random-pca16-r2.json.tmp"
    orphan.write_bytes(b"uncommitted fit evidence retained for audit")
    temporary.write_text("uncommitted row")
    hashes = (sha256_file(orphan), sha256_file(temporary))
    run(bundles, output, workers=2)
    archived = list((output / "scaling_recovery").rglob("manifest.json"))
    assert len(archived) == 1
    audit = json.loads(archived[0].read_text())
    assert set(audit["files"].values()) == set(hashes)
    for name, digest in audit["files"].items():
        assert sha256_file(archived[0].parent / name) == digest
    assert not temporary.exists()


def test_scaling_refuses_active_writer(bundles, tmp_path, monkeypatch):
    output = tmp_path / "scaled"
    interrupt_study(bundles, output, monkeypatch)
    with exclusive_lock(output / "execution_locks/coordinator.lock"):
        with pytest.raises(RuntimeError, match="another coordinator"):
            run(bundles, output)
    with exclusive_lock(pca._lock_path(output, (13, "uniformly_random", 16))):
        with pytest.raises(RuntimeError, match="another coordinator"):
            run(bundles, output)
    assert not (output / "scaling_transition.json").exists()


def test_publication_recovery_preserves_finished_cells(bundles, tmp_path, monkeypatch):
    output = tmp_path / "scaled"
    interrupt_study(bundles, output, monkeypatch)
    original = pca._replace_from_stage
    replacements = 0

    def interrupt(stage, target):
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise RuntimeError("test publication interrupted")
        original(stage, target)

    with monkeypatch.context() as patch:
        patch.setattr(pca, "_replace_from_stage", interrupt)
        with pytest.raises(RuntimeError, match="publication interrupted"):
            run(bundles, output, workers=2)
    cells = snapshot(output / "cells")
    monkeypatch.setattr(scale, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("repeated finished work"))
    result = run(bundles, output, workers=2)
    assert result["status"] == "completed"
    assert snapshot(output / "cells") == cells


@pytest.mark.parametrize("change", ["checkpoint", "scheduler", "workers"])
def test_corruption_or_incompatible_scaling_fails_closed(bundles, tmp_path, monkeypatch, change):
    output = tmp_path / "scaled"
    interrupt_study(bundles, output, monkeypatch)

    def fail_start(**kwargs):
        raise RuntimeError("test stop before pool starts")

    with monkeypatch.context() as patch:
        patch.setattr(scale, "ProcessPoolExecutor", fail_start)
        with pytest.raises(RuntimeError, match="pool starts"):
            run(bundles, output)
    if change == "checkpoint":
        next((output / "cells").glob("*.npz")).write_bytes(b"corrupted")
        message = "evidence hash mismatch"
    elif change == "scheduler":
        monkeypatch.setattr(scale, "_code_hashes", lambda: {"changed.py": "not-original"})
        message = "execution fingerprint changed"
    else:
        message = "execution fingerprint changed"
    with pytest.raises(ValueError, match=message):
        run(bundles, output, workers=5 if change == "workers" else 6)


@pytest.mark.parametrize("workers", [0, 7, True, 1.5])
def test_worker_limit(tmp_path, workers):
    with pytest.raises(ValueError):
        scale.run_scaled(tmp_path / "missing.json", output=tmp_path, workers=workers)
