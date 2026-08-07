from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sampling_comparison.v4_experiment import IDWConfig
from sampling_comparison.v4_outputs import run_v4_experiment_bundle


class _DummyRuntime:
    token_profile_id = "token-prof"
    minhash_profile_id = "minhash-prof"
    embedding_profile_id = "embed-prof"
    embedding_semantic_scope = "scope-a"


class _DummyData:
    unit_ids = ["u1", "u2", "u3"]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_v3_bundle(source_dir: Path) -> dict:
    source_dir.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "version": "sampling-v3-bundle-v1",
        "runtime_seconds": 1.25,
        "config": {
            "selection_budget_policy": {
                "unit": "tokens",
                "cochran_sample_sizing": False,
                "finite_population_correction": False,
                "sessions_are_indivisible": True,
                "selection_rule": "maximal_feasible_greedy_pack",
            },
            "runtime": {
                "embedding_ledger": {
                    "embedding_calls": 12,
                    "embedding_input_tokens": 300,
                }
            },
        },
    }

    outcome = {
        "version": "sampling-v3-outcome-v1",
        "runtime_version": "sampling-v3",
        "population_count": 3,
        "eligible_token_mass": 99,
        "aggregate": [{"method": "random_sampling_token_priority", "budget_tokens": 20}],
        "runs": [
            {
                "method": "random_sampling_token_priority",
                "budget_tokens": 20,
                "legacy_tier_pct": 20,
                "repetition": 0,
                "order_hash": "abc",
                "selected_ids": ["u1", "u2"],
                "selected_count": 2,
                "selected_pass_rate": 0.5,
                "census_pass_rate": 0.4,
                "absolute_error": 0.1,
            }
        ],
    }

    manifest = {
        "version": "sampling-v3-manifest-v1",
        "generated_at": "2026-01-01T00:00:00Z",
        "artifacts": {
            "aggregate": {"path": "aggregate.json", "bytes": 1, "sha256": "a" * 64}
        },
    }

    (source_dir / "aggregate.json").write_text(json.dumps(aggregate) + "\n", encoding="utf-8")
    (source_dir / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    return {
        "aggregate": aggregate,
        "outcome": outcome,
        "embedding_ledger": aggregate["config"]["runtime"]["embedding_ledger"],
        "output_paths": {
            "aggregate": str(source_dir / "aggregate.json"),
            "manifest": str(source_dir / "manifest.json"),
        },
    }


def test_v4_bundle_calls_v3_once_and_forwards_parameters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, dict] = {}

    def _fake_v3(**kwargs):
        calls["v3"] = kwargs
        return _fake_v3_bundle(Path(kwargs["output_dir"]))

    def _fake_augment(*, data, runtime, v3_outcome, idw_config):
        calls["augment"] = {
            "data": data,
            "runtime": runtime,
            "v3_outcome": v3_outcome,
            "idw_config": idw_config,
        }
        return {
            "version": "sampling-v4-outcome-v1",
            "population_count": 3,
            "eligible_token_mass": 99,
            "aggregate": [{"method": "adaptive_embedding_fullsession_token", "budget_tokens": 20}],
            "runs": [
                {
                    "method": "adaptive_embedding_fullsession_token",
                    "budget_tokens": 20,
                    "legacy_tier_pct": 20,
                    "repetition": 0,
                    "selected_ids": ["u1"],
                    "selected_count": 1,
                    "selected_pass_rate": 1.0,
                    "labels": [1, 0],
                    "vectors": [[0.1, 0.2]],
                    "packet_text": "secret",
                    "donor_ids": ["u2"],
                    "donor_distances": [0.2],
                    "donor_weights": [0.8],
                    "per_unit_estimators": [{"u": "u3"}],
                }
            ],
        }

    monkeypatch.setattr("sampling_comparison.v4_outputs.run_v3_experiment_bundle", _fake_v3)
    monkeypatch.setattr("sampling_comparison.v4_outputs.augment_v3_outcome_with_idw", _fake_augment)

    runtime = _DummyRuntime()
    data = _DummyData()
    idw_config = IDWConfig(k=8)

    out = run_v4_experiment_bundle(
        runtime=runtime,
        data=data,
        output_dir=tmp_path / "v4",
        vector_store_factory=None,
        outcome_repetitions=1,
        quadrant_replays=2,
        throughput_replays=3,
        legacy_outcome_tiers_pct=(20,),
        legacy_quadrant_tiers_pct=(15,),
        throughput_arrival_rates_sessions_per_second=(1.0,),
        throughput_capacity_rates_sessions_per_second=(2.0,),
        seed=21,
        tenant_id="tenant-x",
        cleanup_max_attempts=7,
        cleanup_settle_seconds=0.1,
        skip_quadrant=True,
        skip_throughput=True,
        idw_config=idw_config,
        aggregate_config={"user_flag": True},
    )

    assert "v3" in calls
    assert "augment" in calls
    assert calls["v3"]["output_dir"] == (tmp_path / "v4" / "source_v3")
    assert calls["v3"]["runtime"] is runtime
    assert calls["v3"]["data"] is data
    assert calls["v3"]["vector_store_factory"] is None
    assert calls["v3"]["outcome_repetitions"] == 1
    assert calls["v3"]["quadrant_replays"] == 2
    assert calls["v3"]["throughput_replays"] == 3
    assert calls["v3"]["legacy_outcome_tiers_pct"] == (20,)
    assert calls["v3"]["legacy_quadrant_tiers_pct"] == (15,)
    assert calls["v3"]["throughput_arrival_rates_sessions_per_second"] == (1.0,)
    assert calls["v3"]["throughput_capacity_rates_sessions_per_second"] == (2.0,)
    assert calls["v3"]["seed"] == 21
    assert calls["v3"]["tenant_id"] == "tenant-x"
    assert calls["v3"]["cleanup_max_attempts"] == 7
    assert calls["v3"]["cleanup_settle_seconds"] == 0.1
    assert calls["v3"]["skip_quadrant"] is True
    assert calls["v3"]["skip_throughput"] is True
    assert calls["v3"]["aggregate_config"] == {"user_flag": True}

    assert calls["augment"]["data"] is data
    assert calls["augment"]["runtime"] is runtime
    assert calls["augment"]["v3_outcome"]["version"] == "sampling-v3-outcome-v1"
    assert calls["augment"]["idw_config"] == idw_config

    assert out["source_v3"]["outcome"]["version"] == "sampling-v3-outcome-v1"
    assert out["outcome"]["version"] == "sampling-v4-outcome-v1"


def test_v4_bundle_persists_required_root_artifacts_and_manifest_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_v3(**kwargs):
        return _fake_v3_bundle(Path(kwargs["output_dir"]))

    def _fake_augment(*, data, runtime, v3_outcome, idw_config):
        return {
            "version": "sampling-v4-outcome-v1",
            "population_count": 3,
            "eligible_token_mass": 99,
            "aggregate": [{"method": "adaptive_embedding_fullsession_token", "budget_tokens": 20}],
            "runs": [{"method": "adaptive_embedding_fullsession_token", "budget_tokens": 20, "selected_ids": ["u1"]}],
        }

    monkeypatch.setattr("sampling_comparison.v4_outputs.run_v3_experiment_bundle", _fake_v3)
    monkeypatch.setattr("sampling_comparison.v4_outputs.augment_v3_outcome_with_idw", _fake_augment)

    out = run_v4_experiment_bundle(
        runtime=_DummyRuntime(),
        data=_DummyData(),
        output_dir=tmp_path / "v4",
        vector_store_factory=None,
    )

    paths = out["output_paths"]
    for key in ("aggregate", "runs_jsonl", "idw_config", "methodology_delta", "source_lineage", "manifest"):
        assert Path(paths[key]).exists(), key

    manifest = _read_json(Path(paths["manifest"]))
    assert manifest["version"] == "sampling-v4-manifest-v1"
    assert manifest["source"]["source_subdir"] == "source_v3"
    assert manifest["source"]["source_manifest_version"] == "sampling-v3-manifest-v1"

    artifact_keys = {
        "aggregate",
        "runs_jsonl",
        "idw_config",
        "methodology_delta",
        "source_lineage",
        "source_v3_manifest",
    }
    assert set(manifest["artifacts"]) == artifact_keys
    for key in artifact_keys:
        entry = manifest["artifacts"][key]
        assert len(str(entry["sha256"])) == 64
        assert int(entry["bytes"]) > 0
        if key != "source_v3_manifest":
            assert Path(out["output_paths"]["manifest"]).parent.joinpath(entry["path"]).exists()

    source_manifest_path = Path(paths["source_v3_manifest"])
    assert manifest["artifacts"]["source_v3_manifest"]["sha256"] == _sha(source_manifest_path)

    aggregate = _read_json(Path(paths["aggregate"]))
    assert aggregate["version"] == "sampling-v4-bundle-v1"
    assert aggregate["source_v3"]["manifest_relative_path"] == "source_v3/manifest.json"
    assert aggregate["source_v3"]["manifest"]["version"] == "sampling-v3-manifest-v1"
    assert aggregate["runtime"]["embedding_ledger"]["embedding_calls"] == 12
    assert aggregate["provenance"]["token_budget_policy"]["cochran_sample_sizing"] is False


def test_v4_bundle_rejects_missing_or_malformed_source_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_missing(**kwargs):
        source_dir = Path(kwargs["output_dir"])
        source_dir.mkdir(parents=True, exist_ok=True)
        return {
            "aggregate": {"version": "sampling-v3-bundle-v1", "runtime_seconds": 0.1},
            "outcome": {"version": "sampling-v3-outcome-v1", "population_count": 3, "eligible_token_mass": 1, "aggregate": [], "runs": []},
        }

    monkeypatch.setattr("sampling_comparison.v4_outputs.run_v3_experiment_bundle", _fake_missing)

    with pytest.raises(FileNotFoundError, match="source V3 manifest not found"):
        run_v4_experiment_bundle(
            runtime=_DummyRuntime(),
            data=_DummyData(),
            output_dir=tmp_path / "v4_missing",
            vector_store_factory=None,
        )

    def _fake_bad_version(**kwargs):
        bundle = _fake_v3_bundle(Path(kwargs["output_dir"]))
        bad_manifest = {
            "version": "sampling-v3-manifest-v0",
            "artifacts": {},
        }
        Path(bundle["output_paths"]["manifest"]).write_text(json.dumps(bad_manifest) + "\n", encoding="utf-8")
        return bundle

    monkeypatch.setattr("sampling_comparison.v4_outputs.run_v3_experiment_bundle", _fake_bad_version)

    with pytest.raises(ValueError, match="source V3 manifest version must be sampling-v3-manifest-v1"):
        run_v4_experiment_bundle(
            runtime=_DummyRuntime(),
            data=_DummyData(),
            output_dir=tmp_path / "v4_bad",
            vector_store_factory=None,
        )


def test_v4_runs_jsonl_and_methodology_enforce_policy_requirements(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_v3(**kwargs):
        return _fake_v3_bundle(Path(kwargs["output_dir"]))

    def _fake_augment(*, data, runtime, v3_outcome, idw_config):
        return {
            "version": "sampling-v4-outcome-v1",
            "population_count": 3,
            "eligible_token_mass": 99,
            "aggregate": [],
            "runs": [
                {
                    "method": "adaptive_embedding_fullsession_token",
                    "budget_tokens": 20,
                    "selected_ids": ["u1", "u2"],
                    "labels": [1, 0],
                    "vectors": [[0.1, 0.2]],
                    "packet_text": "secret",
                    "donor_ids": ["u3"],
                    "donor_distances": [0.3],
                    "donor_weights": [0.7],
                    "per_unit_estimators": [{"unit_id": "u3"}],
                    "nested": {
                        "labels_by_unit": {"u1": True},
                        "donor_ids": ["u2"],
                    },
                }
            ],
        }

    monkeypatch.setattr("sampling_comparison.v4_outputs.run_v3_experiment_bundle", _fake_v3)
    monkeypatch.setattr("sampling_comparison.v4_outputs.augment_v3_outcome_with_idw", _fake_augment)

    out = run_v4_experiment_bundle(
        runtime=_DummyRuntime(),
        data=_DummyData(),
        output_dir=tmp_path / "v4",
        vector_store_factory=None,
    )

    runs_lines = [
        json.loads(line)
        for line in Path(out["output_paths"]["runs_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(runs_lines) == 1
    payload = json.dumps(runs_lines[0])
    for forbidden in (
        '"labels"',
        '"labels_by_unit"',
        '"vectors"',
        '"packet_text"',
        '"donor_ids"',
        '"donor_distances"',
        '"donor_weights"',
        '"per_unit_estimators"',
    ):
        assert forbidden not in payload
    assert "selected_ids" in runs_lines[0]

    methodology = Path(out["output_paths"]["methodology_delta"]).read_text(encoding="utf-8")
    assert "exact token mass" in methodology
    assert "no Cochran sample sizing" in methodology
    assert "no finite-population correction" in methodology
    assert "whole-session maximal packing" in methodology
    assert "Random and MinHash arms remain selected-only" in methodology
    assert "bucket miss is treated as novelty/no-candidate" in methodology
    assert "selected-only metrics plus judged+IDW" in methodology
    assert "same-agent k=8 angular neighbors" in methodology
    assert "Deterministic expected labels" in methodology
    assert "no design-unbiasedness claim" in methodology

    lineage = _read_json(Path(out["output_paths"]["source_lineage"]))
    assert lineage["selection_rerun"] is False
    assert lineage["source_bundle_subdir"] == "source_v3"
