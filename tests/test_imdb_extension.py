"""Real engine executions on tiny offline fixtures, never benchmark results."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from sampling_comparison import imdb_experiment as engine
from sampling_comparison import imdb_extension as extension
from scripts import extend_imdb_experiment as cli


def make_source(tmp_path, *, repetitions=1, schedules=("uniformly_random",), rates=(.25,)):
    cache = tmp_path / "cache"
    cache.mkdir()
    vector_path = cache / "vectors.npy"
    np.save(vector_path, np.random.default_rng(67).normal(size=(16, 1536)).astype(np.float32))
    vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    labels = (np.arange(16) % 2).astype(np.uint8)
    profile = {
        "dataset_id": "offline-extension-fixture", "sessions": 16, "agents": 1,
        "label_source": "fixture binary labels", "model": "text-embedding-3-small",
        "representation_policy": "random fixture vectors; NOT measured IMDb embeddings",
        "input_manifest_sha256": "offline-fixture-manifest",
    }
    baseline = tmp_path / "baseline"
    with threadpool_limits(limits=4):
        aggregate = engine.run_experiment(
            vectors, labels, baseline, profile=profile, dimensions=(1536, 8),
            repetitions=repetitions, schedules=schedules, rates=rates,
            target_block_size=4, donor_block_size=3, calibration_reservoir_size=4,
        )
    return {
        "vectors": vectors, "labels": labels, "profile": profile,
        "baseline": baseline, "output": tmp_path / "extended",
        "aggregate": aggregate, "cache": vector_path,
    }


@pytest.fixture
def source(tmp_path):
    return make_source(tmp_path)


def run(source, **kwargs):
    args = {
        "profile": source["profile"], "baseline": source["baseline"],
        "dimensions": (256, 128, 64),
    }
    args.update(kwargs)
    return extension.run_extension(source["vectors"], source["labels"], source["output"], **args)


def snapshots(root):
    return {str(p.relative_to(root)): engine.sha256_file(p) for p in root.rglob("*") if p.is_file()}


def without_path(row):
    return {k: v for k, v in row.items() if k != "evidence"}


def test_adds_only_requested_algorithms_preserves_all_baseline_rows_and_files(tmp_path, monkeypatch):
    source = make_source(tmp_path, repetitions=2, schedules=engine.SCHEDULES, rates=(.25, .5))
    before = snapshots(source["baseline"])
    cache_hash = engine.sha256_file(source["cache"])
    method_hashes = engine._method_hashes()
    observed = {"prefix": [], "rank": [], "evaluate": []}
    original_prefix = extension.normalized_prefix
    original_rank = extension.rank_membership
    original_evaluate = extension.evaluate_replay

    def prefix(vectors, dimension):
        observed["prefix"].append(dimension)
        assert dimension in (256, 128, 64)
        return original_prefix(vectors, dimension)

    def rank(ids, agents, signatures, vectors, order, times, seed):
        observed["rank"].append(vectors.shape[1])
        assert vectors.shape[1] in (256, 128, 64)
        return original_rank(ids, agents, signatures, vectors, order, times, seed)

    def evaluate(vectors, labels, sources, order, selected, **kwargs):
        observed["evaluate"].append(vectors.shape[1])
        assert vectors.shape[1] in (256, 128, 64)
        assert kwargs["target_block_size"] == 4
        assert kwargs["donor_block_size"] == 3
        assert kwargs["calibration_reservoir_size"] == 4
        return original_evaluate(vectors, labels, sources, order, selected, **kwargs)

    monkeypatch.setattr(extension, "normalized_prefix", prefix)
    monkeypatch.setattr(extension, "rank_membership", rank)
    monkeypatch.setattr(extension, "evaluate_replay", evaluate)
    monkeypatch.setattr(engine, "bootstrap_replay", lambda *a, **k: pytest.fail("regenerated a bootstrap draw"))
    monkeypatch.setattr(engine, "run_experiment", lambda *a, **k: pytest.fail("reran the baseline engine"))
    result = run(source)
    assert result["version"] == engine.VERSION and result["status"] == "completed"
    assert result["dataset"] == source["profile"]
    assert result["protocol"]["dimensions"] == [1536, 256, 128, 64, 8]
    expected_protocol = {
        **source["aggregate"]["protocol"], "dimensions": [1536, 256, 128, 64, 8], "planned_cells": 40,
    }
    assert result["protocol"] == expected_protocol
    assert len(observed["rank"]) == 12
    assert len(observed["evaluate"]) == 24
    assert len(result["rows"]) == 40
    assert snapshots(source["baseline"]) == before
    assert engine.sha256_file(source["cache"]) == cache_hash
    assert engine._method_hashes() == method_hashes
    original = {extension._cell_key(r): r for r in source["aggregate"]["rows"]}
    current = {extension._cell_key(r): r for r in result["rows"]}
    for key, row in original.items():
        assert engine.canonical(without_path(row)) == engine.canonical(without_path(current[key]))
        assert (source["output"] / current[key]["evidence"]).resolve() == (source["baseline"] / row["evidence"]).resolve()
    for row in result["rows"]:
        native = original[row["seed"], row["schedule"], 1536, row["rate"]]
        assert row["replay_hashes"] == native["replay_hashes"]
        assert row["paired_delta_native"]["all_unselected"]["mae"] == (
            row["all_unselected"]["mae"] - native["all_unselected"]["mae"]
        )
    manifest = engine._read_record(source["output"] / "manifest.json")
    assert manifest["completed_cells"] == 40
    for path, digest in manifest["files"].items():
        assert engine.sha256_file(source["output"] / path) == digest
    for relative, digest in before.items():
        inherited = extension._relative(source["baseline"] / relative, source["output"])
        assert manifest["files"][inherited] == digest
    for path in source["output"].rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            assert path.relative_to(source["output"]).as_posix() in manifest["files"]
    assert not (source["output"] / "replays").exists()
    audit = json.loads((source["output"] / "extension_validation.json").read_text())
    assert audit["baseline_before"] == audit["baseline_after"]
    assert audit["reused_cells"] == 16 and audit["added_cells"] == 24 and audit["combined_cells"] == 40
    assert audit["embedding_calls"] == 0
    assert audit["baseline_rows_unchanged"] and audit["replay_pairing_exact"] and audit["input_arrays_unchanged"]
    assert audit["ok"] is True
    assert audit["aggregate_sha256"] == engine.sha256_file(source["output"] / "aggregate.json")
    assert set(result["extension"]) == {
        "baseline_aggregate_sha256", "baseline_manifest_sha256",
        "reused_cells", "added_cells", "reused_dimensions", "added_dimensions",
        "baseline_rows_unchanged", "replay_pairing_exact", "embedding_calls",
    }
    assert all(audit[key] == value for key, value in result["extension"].items())
    assert set(audit) == {
        "version", "status", "baseline_rows_unchanged", "replay_pairing_exact", "embedding_calls",
        "baseline_artifacts_unchanged", "baseline_artifacts_checked", "input_arrays_unchanged",
        "method_code_hashes_unchanged", "runtime_unchanged", "baseline_before", "baseline_after",
        "reused_dimensions", "added_dimensions", "reused_cells", "added_cells", "combined_cells",
        "replay_count", "repetitions", "rates_per_replay", "source_count", "extension_code_sha256",
        "ok", "aggregate_sha256", "baseline_aggregate_sha256", "baseline_manifest_sha256",
    }
    progress = [json.loads(line) for line in (source["output"] / "progress.jsonl").read_text().splitlines()]
    assert len([event for event in progress if event["event"] == "replay_completed"]) == 4
    assert progress[-1]["event"] == "extension_completed"
    # Match the parent's public extension validation contract, without editing it.
    from scripts.validate_imdb_experiment import validate_extension
    assert validate_extension(result, manifest, source["output"])["added_cells"] == 24


def test_new_measurements_match_independent_full_fixture_sweep(source):
    actual = run(source)
    with threadpool_limits(limits=4):
        reference = engine.run_experiment(
            source["vectors"], source["labels"], source["output"].parent / "oracle",
            profile=source["profile"], dimensions=(1536, 256, 128, 64, 8), repetitions=1,
            schedules=("uniformly_random",), rates=(.25,), target_block_size=4,
            donor_block_size=3, calibration_reservoir_size=4,
        )
    expected = {extension._cell_key(row): row for row in reference["rows"]}
    for row in actual["rows"]:
        if row["dimension"] not in (256, 128, 64):
            continue
        other = expected[extension._cell_key(row)]
        assert {k: v for k, v in row.items() if k != "elapsed_seconds"} == {
            k: v for k, v in other.items() if k != "elapsed_seconds"
        }


def test_success_audit_is_written_after_verified_aggregate_without_circular_hash(source, monkeypatch):
    original_write = extension.write_json
    writes = []

    def observe_write(path, payload):
        writes.append(path.name)
        if path.name == "extension_validation.json":
            aggregate_path = source["output"] / "aggregate.json"
            assert aggregate_path.exists()
            assert payload["aggregate_sha256"] == engine.sha256_file(aggregate_path)
            aggregate = json.loads(aggregate_path.read_text())
            assert "aggregate_sha256" not in aggregate
            assert "aggregate_sha256" not in aggregate["extension"]
            assert payload["ok"] is True
            assert all(payload[key] == value for key, value in aggregate["extension"].items())
        original_write(path, payload)

    monkeypatch.setattr(extension, "write_json", observe_write)
    run(source)
    assert writes == ["aggregate.json", "extension_validation.json"]


@pytest.mark.parametrize("change", ["labels", "vectors", "profile", "method", "runtime"])
def test_baseline_source_method_runtime_input_drift_fails_closed(source, monkeypatch, change):
    if change == "labels":
        source["labels"] = 1 - source["labels"]
    elif change == "vectors":
        source["vectors"] = source["vectors"].copy()
        source["vectors"][0, 0] += 1
    elif change == "profile":
        source["profile"] = {**source["profile"], "label_source": "changed"}
    elif change == "method":
        monkeypatch.setattr(extension, "_method_hashes", lambda: {"changed.py": "different"})
    else:
        monkeypatch.setattr(extension, "_runtime", lambda: {"python": "different"})
    monkeypatch.setattr(extension, "rank_membership", lambda *a, **k: pytest.fail("ranking before source verification"))
    with pytest.raises(ValueError, match="mismatch"):
        run(source)
    assert not source["output"].exists()


def test_actual_blas_thread_count_must_match_baseline(source):
    with pytest.raises(ValueError, match="runtime mismatch"):
        run(source, blas_threads=1)
    assert not source["output"].exists()


@pytest.mark.parametrize("dimensions", [(), (256, 256), (8,), (1536,), (0,), (1537,), (True,), (128.5,)])
def test_duplicate_invalid_or_overlapping_dimensions_refused(source, dimensions):
    with pytest.raises(ValueError, match="dimensions"):
        run(source, dimensions=dimensions)
    assert not source["output"].exists()


@pytest.mark.parametrize("artifact", ["aggregate.json", "manifest.json", "preregistration.json", "replay", "cell", "membership"])
def test_every_baseline_artifact_is_verified(source, artifact):
    root = source["baseline"]
    if artifact in ("replay", "cell", "membership"):
        directory = {"replay": "replays", "cell": "cells", "membership": "memberships"}[artifact]
        path = next((root / directory).glob("*.npz"))
    else:
        path = root / artifact
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError):
        run(source)
    assert not source["output"].exists()


def test_incomplete_or_missing_grid_cannot_be_source(source):
    manifest_path = source["baseline"] / "manifest.json"
    manifest = engine._read_record(manifest_path)
    manifest["status"] = "in_progress"
    engine._write_record(manifest_path, manifest)
    with pytest.raises(ValueError, match="completed"):
        run(source)
    manifest["status"] = "completed"
    engine._write_record(manifest_path, manifest)
    aggregate_path = source["baseline"] / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["rows"].pop()
    engine.write_json(aggregate_path, aggregate)
    with pytest.raises(ValueError, match="complete unique cell grid"):
        run(source)


def interrupt_after_one_cell(source, monkeypatch):
    original = extension.evaluate_replay
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        assert (source["output"] / "preregistration.json").exists()
        assert not (source["output"] / "manifest.json").exists()
        assert not (source["output"] / "aggregate.json").exists()
        assert not (source["output"] / "extension_validation.json").exists()
        calls += 1
        if calls == 2:
            raise RuntimeError("fixture interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(extension, "evaluate_replay", interrupted)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        run(source)
    monkeypatch.setattr(extension, "evaluate_replay", original)


def test_partial_and_completed_resume_reuse_committed_new_work(source, monkeypatch):
    interrupt_after_one_cell(source, monkeypatch)
    committed = next((source["output"] / "cells").glob("*.json"))
    before = committed.read_bytes()
    computed, ranked = [], []
    original_evaluate, original_rank = extension.evaluate_replay, extension.rank_membership

    def evaluate(vectors, *args, **kwargs):
        computed.append(vectors.shape[1])
        return original_evaluate(vectors, *args, **kwargs)

    def rank(ids, agents, signatures, vectors, *args, **kwargs):
        ranked.append(vectors.shape[1])
        return original_rank(ids, agents, signatures, vectors, *args, **kwargs)

    monkeypatch.setattr(extension, "evaluate_replay", evaluate)
    monkeypatch.setattr(extension, "rank_membership", rank)
    result = run(source, resume=True)
    assert computed == [128, 64]
    assert ranked == [64]
    assert committed.read_bytes() == before
    frozen_output = snapshots(source["output"])
    monkeypatch.setattr(extension, "evaluate_replay", lambda *a, **k: pytest.fail("recomputed a completed cell"))
    monkeypatch.setattr(extension, "rank_membership", lambda *a, **k: pytest.fail("reranked a completed dimension"))
    monkeypatch.setattr(extension, "normalized_prefix", lambda *a, **k: pytest.fail("rebuilt a completed prefix"))
    assert run(source, resume=True) == result
    assert snapshots(source["output"]) == frozen_output
    with pytest.raises(FileExistsError):
        run(source)


@pytest.mark.parametrize("artifact", ["cell_npz", "cell_json", "membership_npz", "membership_json"])
def test_partial_resume_rejects_checkpoint_tampering(source, monkeypatch, artifact):
    interrupt_after_one_cell(source, monkeypatch)
    directory, suffix = artifact.split("_")
    directory = "cells" if directory == "cell" else "memberships"
    path = next((source["output"] / directory).glob(f"*.{suffix}"))
    if suffix == "json":
        record = json.loads(path.read_text())
        record["identity"]["dimension"] = 99
        path.write_text(json.dumps(record), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        run(source, resume=True)
    assert not (source["output"] / "aggregate.json").exists()


@pytest.mark.parametrize("artifact", ["aggregate.json", "extension_validation.json", "progress.jsonl", "manifest.json"])
def test_completed_extension_resume_rejects_artifact_tampering(source, artifact):
    run(source)
    path = source["output"] / artifact
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash|format"):
        run(source, resume=True)


@pytest.mark.parametrize("change", ["dimensions", "extension_code", "source_manifest"])
def test_resume_is_bound_to_extension_code_source_and_configuration(source, monkeypatch, change):
    interrupt_after_one_cell(source, monkeypatch)
    kwargs = {"resume": True}
    if change == "dimensions":
        kwargs["dimensions"] = (256, 128)
    elif change == "extension_code":
        monkeypatch.setattr(extension, "_extension_code_hashes", lambda: {"extension.py": "changed"})
    else:
        path = source["baseline"] / "manifest.json"
        manifest = engine._read_record(path)
        manifest["annotation"] = "changed after registration"
        engine._write_record(path, manifest)
    with pytest.raises(ValueError, match="fingerprints"):
        run(source, **kwargs)


def test_old_aggregate_without_registration_is_not_reused(source):
    source["output"].mkdir()
    engine.write_json(source["output"] / "aggregate.json", source["aggregate"])
    with pytest.raises(ValueError, match="preregistration"):
        run(source, resume=True)


@pytest.mark.parametrize("location", ["same", "inside", "ancestor"])
def test_output_cannot_write_into_baseline_or_its_ancestor(source, location):
    source["output"] = {
        "same": source["baseline"], "inside": source["baseline"] / "nested",
        "ancestor": source["baseline"].parent,
    }[location]
    before = snapshots(source["baseline"])
    with pytest.raises(ValueError, match="non-nested"):
        run(source, resume=True)
    assert snapshots(source["baseline"]) == before


@pytest.mark.parametrize("change", ["baseline", "inputs", "method"])
def test_finalization_rechecks_source_input_and_method_immutability(source, monkeypatch, change):
    original = extension.evaluate_replay
    if change == "inputs":
        source["labels"] = source["labels"].copy()
    once = False

    def alter_after_evaluation(*args, **kwargs):
        nonlocal once
        result = original(*args, **kwargs)
        if not once:
            once = True
            if change == "baseline":
                path = next((source["baseline"] / "memberships").glob("*.npz"))
                path.write_bytes(path.read_bytes() + b"concurrent drift")
            elif change == "inputs":
                source["labels"][0] = 1 - source["labels"][0]
            else:
                monkeypatch.setattr(extension, "_method_hashes", lambda: {"changed.py": "different"})
        return result

    monkeypatch.setattr(extension, "evaluate_replay", alter_after_evaluation)
    with pytest.raises(ValueError, match="mismatch"):
        run(source)
    assert not (source["output"] / "manifest.json").exists()
    assert not (source["output"] / "aggregate.json").exists()


def test_cli_defaults_and_explicit_resume_wire_real_fixture_engine(source, monkeypatch, capsys):
    calls = []

    def load(path):
        calls.append(path)
        return source["vectors"], source["labels"], source["profile"]

    monkeypatch.setattr(cli, "load_input", load)
    argv = [
        "--input", str(source["cache"]), "--baseline", str(source["baseline"]),
        "--output", str(source["output"]),
    ]
    cli.main(argv)
    aggregate = json.loads((source["output"] / "aggregate.json").read_text())
    assert aggregate["extension"]["added_dimensions"] == [256, 128, 64]
    assert all(pool["num_threads"] == 4 for pool in aggregate["provenance"]["runtime"]["threadpools"])
    cli.main(argv + ["--resume"])
    assert calls == [source["cache"], source["cache"]]
    assert "completed:" in capsys.readouterr().out


def test_cli_help_is_available_without_input_or_credentials():
    path = Path(__file__).resolve().parents[1] / "scripts" / "extend_imdb_experiment.py"
    result = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True, check=True)
    assert "--resume" in result.stdout and "--blas-threads" in result.stdout
    assert "--repetitions" not in result.stdout and "--rates" not in result.stdout
