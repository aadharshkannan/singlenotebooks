from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sampling_comparison.v2_experiment import load_combined_dataset, slice_dataset
from sampling_comparison.v3_experiment import build_v3_runtime, Deterministic1536Embedder
from sampling_comparison.v3_outputs import _code_hashes
from sampling_comparison.v3_outputs import register_manifest_artifact
from sampling_comparison.v3_outputs import run_v3_experiment_bundle
from sampling_comparison.v3_outputs import write_run_source_manifest
from sampling_comparison.v3_outputs import write_search_cleanup_audit
from trace_sampling.vector_store import InMemoryVectorStore


class _FakeTokenizer:
    model_name = "text-embedding-3-small"
    name = "cl100k_base"
    version = "fake-1"

    def count(self, text: str) -> int:
        return max(1, len(text.encode("utf-8")) // 8)


def _slice(limit: int = 80):
    return slice_dataset(load_combined_dataset(), limit=limit)


def test_v3_bundle_writes_required_artifacts_and_manifest_hashes(tmp_path: Path):
    data = _slice(60)
    runtime = build_v3_runtime(
        data,
        tokenizer=_FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=5),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )

    out = run_v3_experiment_bundle(
        runtime=runtime,
        data=data,
        output_dir=tmp_path / "v3",
        vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
        outcome_repetitions=1,
        quadrant_replays=1,
        throughput_replays=1,
        legacy_outcome_tiers_pct=(20,),
        legacy_quadrant_tiers_pct=(15,),
        throughput_arrival_rates_sessions_per_second=(1.0,),
        throughput_capacity_rates_sessions_per_second=(1.0,),
        seed=13,
    )

    paths = out["output_paths"]
    for key in (
        "aggregate",
        "runs_jsonl",
        "quadrant",
        "throughput",
        "corpus_audit",
        "token_inventory",
        "budget_manifest",
        "embedding_ledger",
        "selected_membership",
        "methodology_delta",
        "manifest",
    ):
        assert Path(paths[key]).exists(), key

    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["version"] == "sampling-v3-manifest-v1"
    for _, entry in manifest["artifacts"].items():
        assert len(entry["sha256"]) == 64
        assert int(entry["bytes"]) > 0

    aggregate = json.loads(Path(paths["aggregate"]).read_text(encoding="utf-8"))
    policy = aggregate["config"]["selection_budget_policy"]
    assert policy == {
        "unit": "tokens",
        "cochran_sample_sizing": False,
        "finite_population_correction": False,
        "sessions_are_indivisible": True,
        "selection_rule": "maximal_feasible_greedy_pack",
    }


def test_v3_bundle_persists_no_raw_text_or_vectors(tmp_path: Path):
    data = _slice(40)
    runtime = build_v3_runtime(
        data,
        tokenizer=_FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=8),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )

    out = run_v3_experiment_bundle(
        runtime=runtime,
        data=data,
        output_dir=tmp_path / "v3",
        vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
        outcome_repetitions=1,
        quadrant_replays=1,
        throughput_replays=1,
        legacy_outcome_tiers_pct=(20,),
        legacy_quadrant_tiers_pct=(15,),
        throughput_arrival_rates_sessions_per_second=(1.0,),
        throughput_capacity_rates_sessions_per_second=(1.0,),
        seed=13,
        skip_quadrant=True,
        skip_throughput=True,
    )

    token_lines = [
        json.loads(line)
        for line in Path(out["output_paths"]["token_inventory"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert token_lines
    assert all("canonical_json" not in row for row in token_lines)
    assert all("vector" not in row for row in token_lines)

    selected_payload = json.loads(Path(out["output_paths"]["selected_membership"]).read_text(encoding="utf-8"))
    serialized = json.dumps(selected_payload)
    assert "canonical_json" not in serialized
    assert '"vector"' not in serialized


def test_methodology_delta_mentions_required_v3_changes(tmp_path: Path):
    data = _slice(30)
    runtime = build_v3_runtime(
        data,
        tokenizer=_FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=11),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )

    out = run_v3_experiment_bundle(
        runtime=runtime,
        data=data,
        output_dir=tmp_path / "v3",
        vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
        outcome_repetitions=1,
        quadrant_replays=1,
        throughput_replays=1,
        legacy_outcome_tiers_pct=(20,),
        legacy_quadrant_tiers_pct=(15,),
        throughput_arrival_rates_sessions_per_second=(1.0,),
        throughput_capacity_rates_sessions_per_second=(1.0,),
        seed=13,
        skip_quadrant=True,
        skip_throughput=True,
    )

    text = Path(out["output_paths"]["methodology_delta"]).read_text(encoding="utf-8")
    assert "1536" in text
    assert "HNSW" in text
    assert "4096-entry exact recent-leader buffer" in text
    assert "8191" in text
    assert "exact token-mass" in text
    assert "does not use Cochran sample sizing or finite-population correction" in text
    assert "Legacy percent tiers" in text
    assert "Packet cap binding check from runtime inventory" in text


def test_code_hashes_cover_required_contributing_modules() -> None:
    hashes = _code_hashes()
    expected = {
        "sampling_comparison/v3_experiment.py",
        "sampling_comparison/v3_outputs.py",
        "sampling_comparison/v3_report.py",
        "sampling_comparison/v2_experiment.py",
        "scripts/run_sampling_v3.py",
        "scripts/build_sampling_v3_report.py",
        "trace_sampling/token_representation.py",
        "trace_sampling/samplers.py",
        "trace_sampling/stats.py",
        "trace_sampling/reservoir.py",
        "trace_sampling/backpressure.py",
        "trace_sampling/variety.py",
        "trace_sampling/cluster_index.py",
        "trace_sampling/vector_store.py",
        "trace_sampling/model.py",
        "trace_sampling/session_embedding.py",
        "trace_sampling/embedding.py",
        "trace_sampling/azure_config.py",
        "minhash_sampling/config.py",
        "minhash_sampling/index.py",
        "minhash_sampling/signature.py",
        "random_sampling/datasets.py",
        "random_sampling/agent365_otel.py",
        "random_sampling/models.py",
    }
    assert set(hashes) == expected
    assert all(value is None or len(str(value)) == 64 for value in hashes.values())


def test_register_manifest_artifact_and_optional_artifacts_roundtrip(tmp_path: Path) -> None:
    data = _slice(30)
    runtime = build_v3_runtime(
        data,
        tokenizer=_FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=4),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )
    out = run_v3_experiment_bundle(
        runtime=runtime,
        data=data,
        output_dir=tmp_path / "v3",
        vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
        outcome_repetitions=1,
        quadrant_replays=1,
        throughput_replays=1,
        legacy_outcome_tiers_pct=(20,),
        legacy_quadrant_tiers_pct=(15,),
        throughput_arrival_rates_sessions_per_second=(1.0,),
        throughput_capacity_rates_sessions_per_second=(1.0,),
        seed=13,
        skip_quadrant=True,
        skip_throughput=True,
    )
    manifest_path = Path(out["output_paths"]["manifest"])

    extra = Path(out["output_paths"]["aggregate"]).parent / "extra_note.json"
    extra.write_text('{"ok":true}\n', encoding="utf-8")
    entry = register_manifest_artifact(manifest_path=manifest_path, key="extra_note", artifact_path=extra)
    assert entry["bytes"] == int(extra.stat().st_size)
    assert entry["sha256"] == hashlib.sha256(extra.read_bytes()).hexdigest()

    run_source = write_run_source_manifest(
        output_dir=extra.parent,
        pre_run_source_hashes=_code_hashes(),
        branch="test-branch",
        captured_at="2026-08-04T01:02:03Z",
        note="unit-test",
        manifest_path=manifest_path,
    )
    cleanup = write_search_cleanup_audit(
        output_dir=extra.parent,
        tenant_id="tenant-a",
        checked_at="2026-08-04T01:02:04Z",
        remaining_count=0,
        scopes={"tenant_id": "tenant-a"},
        manifest_path=manifest_path,
        allow_nonzero=False,
    )

    assert Path(run_source["path"]).exists()
    assert Path(cleanup["path"]).exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "run_source_manifest" in manifest["artifacts"]
    assert "search_cleanup_audit" in manifest["artifacts"]


def test_search_cleanup_audit_fails_when_nonzero_without_override(tmp_path: Path) -> None:
    data = _slice(20)
    runtime = build_v3_runtime(
        data,
        tokenizer=_FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=5),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )
    out = run_v3_experiment_bundle(
        runtime=runtime,
        data=data,
        output_dir=tmp_path / "v3",
        vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
        outcome_repetitions=1,
        quadrant_replays=1,
        throughput_replays=1,
        legacy_outcome_tiers_pct=(20,),
        legacy_quadrant_tiers_pct=(15,),
        throughput_arrival_rates_sessions_per_second=(1.0,),
        throughput_capacity_rates_sessions_per_second=(1.0,),
        seed=13,
        skip_quadrant=True,
        skip_throughput=True,
    )

    manifest_path = Path(out["output_paths"]["manifest"])
    with pytest.raises(ValueError, match="remaining_count == 0"):
        write_search_cleanup_audit(
            output_dir=Path(out["output_paths"]["aggregate"]).parent,
            tenant_id="tenant-a",
            checked_at="2026-08-04T01:02:04Z",
            remaining_count=2,
            scopes={"tenant_id": "tenant-a"},
            manifest_path=manifest_path,
            allow_nonzero=False,
        )
