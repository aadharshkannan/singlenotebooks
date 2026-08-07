from __future__ import annotations

import numpy as np
import pytest

from sampling_comparison.v2_experiment import (
    load_combined_dataset,
    slice_dataset,
    with_permuted_labels,
)
from sampling_comparison.v3_experiment import (
    Deterministic1536Embedder,
    V3_OUTCOME_METHODS,
    V3ReadonlyEmbeddingCache,
    build_exact_token_budget_manifest,
    build_v3_offline_runtime,
    build_v3_runtime,
    _prepare_replay,
    run_v3_quadrant_experiment,
    run_v3_throughput_grid_experiment,
    run_v3_outcome_comparison,
    select_v3_membership,
)
from trace_sampling.session_embedding import TiktokenTokenizer
from trace_sampling.vector_store import InMemoryVectorStore


class FakeTokenizer:
    model_name = "text-embedding-3-small"
    name = "cl100k_base"
    version = "fake-1"

    def count(self, text: str) -> int:
        return max(1, len(text.encode("utf-8")) // 8)


class RecordingEmbedder(Deterministic1536Embedder):
    def __init__(self, seed: int = 13):
        super().__init__(seed=seed)
        self.batch_sizes: list[int] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.batch_sizes.append(len(texts))
        return super().embed(texts)


class ZeroTokenFakeTokenizer(FakeTokenizer):
    def count(self, text: str) -> int:
        return 0


def _slice(limit: int = 120):
    return slice_dataset(load_combined_dataset(), limit=limit)


def test_runtime_inventory_respects_hard_token_limit_and_hash_dedup():
    data = _slice(90)
    runtime = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=3),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
        max_session_packet_tokens=500,
        embedding_batch_size=16,
    )

    assert len(runtime.packet_records_by_unit_id) == len(data.unit_ids)
    assert len(runtime.embedding_vector_by_trace_id) == len(data.unit_ids)
    assert all(rec.emitted_tokens <= 500 for rec in runtime.packet_records_by_unit_id.values())

    unique_hashes = {rec.content_sha256 for rec in runtime.packet_records_by_unit_id.values()}
    assert set(runtime.embedding_records_by_content_sha256.keys()) == unique_hashes
    assert runtime.ledger.embedding_inputs == len(unique_hashes)


def test_runtime_embedding_dedup_batches_and_vectors_are_unit_norm():
    data = _slice(80)
    embedder = RecordingEmbedder(seed=17)
    runtime = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=embedder,
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
        embedding_batch_size=7,
    )

    assert embedder.calls == runtime.ledger.embedding_calls
    assert sum(embedder.batch_sizes) == runtime.ledger.embedding_inputs
    assert all(size <= 7 for size in embedder.batch_sizes)

    for vec in runtime.embedding_records_by_content_sha256.values():
        assert vec.dimensions == 1536
        assert np.isfinite(vec.vector).all()
        assert np.linalg.norm(vec.vector) == np.float32(1.0) or np.isclose(np.linalg.norm(vec.vector), 1.0)


def test_exact_budget_manifest_conversion_uses_floor_of_token_mass():
    data = _slice(70)
    runtime = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=4),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )
    manifest = build_exact_token_budget_manifest(runtime, eligible_unit_ids=data.unit_ids)

    mass = manifest["eligible_token_mass"]
    by_tier = {int(row["legacy_tier_pct"]): int(row["budget_tokens"]) for row in manifest["outcome"]}
    assert by_tier[5] == int((5 / 100.0) * mass)
    assert by_tier[50] == int((50 / 100.0) * mass)


def test_maximal_pack_invariant_and_selected_token_accounting_random():
    data = _slice(110)
    runtime = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=5),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )
    manifest = build_exact_token_budget_manifest(runtime, eligible_unit_ids=data.unit_ids)
    budget_tokens = int(next(row["budget_tokens"] for row in manifest["outcome"] if int(row["legacy_tier_pct"]) == 20))

    row = select_v3_membership(
        data,
        runtime=runtime,
        method="random_sampling_token_priority",
        eligible_unit_ids=data.unit_ids,
        budget_tokens=budget_tokens,
        seed=19,
    )

    assert row["selected_tokens"] <= row["budget_tokens"]
    if row["min_unselected_token_cost"] is not None:
        assert row["slack_tokens"] < row["min_unselected_token_cost"]

    recounted = sum(runtime.token_cost_by_unit_id[uid] for uid in row["selected_ids"])
    assert recounted == row["selected_tokens"]


def test_adaptive_methods_fill_to_budget_and_report_native_fill_telemetry():
    data = _slice(120)
    runtime = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=11),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )
    manifest = build_exact_token_budget_manifest(runtime, eligible_unit_ids=data.unit_ids)
    budget_tokens = int(next(row["budget_tokens"] for row in manifest["outcome"] if int(row["legacy_tier_pct"]) == 30))

    for method in ("adaptive_minhash_32x4_token", "adaptive_embedding_fullsession_token"):
        row = select_v3_membership(
            data,
            runtime=runtime,
            method=method,
            eligible_unit_ids=data.unit_ids,
            budget_tokens=budget_tokens,
            seed=23,
            run_scope=f"test-{method}",
            vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
        )
        assert row["native_count"] + row["fill_count"] == row["selected_count"]
        assert row["native_tokens"] + row["fill_tokens"] == row["selected_tokens"]
        assert row["telemetry"]["proposal_mode"] == "adaptive_native_then_fill"
        assert row["selected_tokens"] <= budget_tokens
        assert isinstance(row["native_proposed_ids"], list)
        if method == "adaptive_minhash_32x4_token":
            assert row["telemetry"]["no_candidate_novel"] > 0
            assert row["telemetry"]["full_scan_fallbacks"] == 0
            assert row["telemetry"]["candidate_lookups"] > 0
        if row["min_unselected_token_cost"] is not None:
            assert row["slack_tokens"] < row["min_unselected_token_cost"]


def test_selection_is_invariant_to_label_permutation():
    data = _slice(90)
    permuted = with_permuted_labels(data, seed=101)
    runtime = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=7),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )

    methods = (
        "random_sampling_token_priority",
        "adaptive_minhash_32x4_token",
        "adaptive_embedding_fullsession_token",
    )
    manifest = build_exact_token_budget_manifest(runtime, eligible_unit_ids=data.unit_ids)
    budget_tokens = int(next(row["budget_tokens"] for row in manifest["outcome"] if int(row["legacy_tier_pct"]) == 20))

    for method in methods:
        left = select_v3_membership(
            data,
            runtime=runtime,
            method=method,
            eligible_unit_ids=data.unit_ids,
            budget_tokens=budget_tokens,
            seed=31,
            run_scope=f"base-{method}",
            vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
        )
        right = select_v3_membership(
            permuted,
            runtime=runtime,
            method=method,
            eligible_unit_ids=permuted.unit_ids,
            budget_tokens=budget_tokens,
            seed=31,
            run_scope=f"perm-{method}",
            vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
        )
        assert left["selected_ids"] == right["selected_ids"]


def test_embedding_cache_is_readonly_and_hash_scoped():
    data = _slice(40)
    runtime = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=9),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )
    cache = V3ReadonlyEmbeddingCache(runtime.embedding_profile_id, runtime.embedding_vector_by_trace_id)
    trace = data.trace_by_unit_id[data.unit_ids[0]]

    assert cache.contains_trace(trace)
    vec = cache.get_trace(trace)
    assert vec.shape == (1536,)


def test_embedding_selector_scoped_cleanup_and_cross_run_isolation():
    data = _slice(100)
    runtime = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=12),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )
    shared_store = InMemoryVectorStore()

    manifest = build_exact_token_budget_manifest(runtime, eligible_unit_ids=data.unit_ids)
    budget_tokens = int(next(row["budget_tokens"] for row in manifest["outcome"] if int(row["legacy_tier_pct"]) == 20))

    first = select_v3_membership(
        data,
        runtime=runtime,
        method="adaptive_embedding_fullsession_token",
        eligible_unit_ids=data.unit_ids,
        budget_tokens=budget_tokens,
        seed=41,
        run_scope="cell-a",
        vector_store_factory=lambda _tenant, _scope: shared_store,
    )
    second = select_v3_membership(
        data,
        runtime=runtime,
        method="adaptive_embedding_fullsession_token",
        eligible_unit_ids=data.unit_ids,
        budget_tokens=budget_tokens,
        seed=41,
        run_scope="cell-b",
        vector_store_factory=lambda _tenant, _scope: shared_store,
    )

    assert first["selected_ids"] == second["selected_ids"]
    assert first["telemetry"]["cleanup_deleted"] >= 0
    assert second["telemetry"]["cleanup_deleted"] >= 0


def test_outcome_shape_is_deterministic_and_token_first():
    data = _slice(120)
    runtime = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=15),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )

    out = run_v3_outcome_comparison(
        data,
        runtime=runtime,
        methods=V3_OUTCOME_METHODS,
        legacy_outcome_tiers_pct=(5, 10),
        repetitions=2,
        seed=13,
        vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
    )

    assert out["version"] == "sampling-v3-outcome-v1"
    assert out["population_count"] == len(data.unit_ids)
    assert len(out["runs"]) == 2 * 2 * len(V3_OUTCOME_METHODS)
    assert isinstance(out["aggregate"], list)
    assert out["aggregate"]

    sample = out["runs"][0]
    for key in (
        "budget_tokens",
        "selected_tokens",
        "slack_tokens",
        "budget_utilization_tokens",
        "selected_count",
        "native_count",
        "native_tokens",
        "fill_count",
        "fill_tokens",
        "selected_ids",
        "legacy_tier_pct",
    ):
        assert key in sample
    assert "budget_pct" not in sample


def test_outcome_forwards_tenant_scope_to_all_adaptive_cells(monkeypatch):
    data = _slice(30)
    runtime = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=15),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )
    seen_tenants: list[str] = []

    from sampling_comparison import v3_experiment as module

    original = module.select_v3_membership

    def _recording_select(*args, **kwargs):
        seen_tenants.append(str(kwargs["tenant_id"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "select_v3_membership", _recording_select)
    run_v3_outcome_comparison(
        data,
        runtime=runtime,
        methods=("random_sampling_token_priority", "adaptive_minhash_32x4_token"),
        legacy_outcome_tiers_pct=(20,),
        repetitions=1,
        tenant_id="sampling-v4-experiment",
        vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
    )

    assert seen_tenants == ["sampling-v4-experiment", "sampling-v4-experiment"]


def test_quadrant_experiment_uses_exact_token_budgets_and_measured_runtime_fields():
    data = _slice(100)
    runtime = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=17),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )

    result = run_v3_quadrant_experiment(
        data,
        runtime=runtime,
        legacy_quadrant_tiers_pct=(15, 30),
        replay_count=1,
        seed=13,
        vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
    )

    assert result["version"] == "sampling-v3-quadrant-v1"
    assert result["runs"]
    row = result["runs"][0]
    assert "budget_tokens" in row
    assert "decision_runtime_ms_p50" in row
    assert "decision_runtime_ms_p95" in row
    assert "zero_selection_agent_rate" in row
    assert row["selected_tokens"] <= row["budget_tokens"]


def test_throughput_mapping_uses_median_packet_tokens_and_queue_metrics_present():
    data = _slice(90)
    runtime = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=19),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )

    out = run_v3_throughput_grid_experiment(
        data,
        runtime=runtime,
        legacy_budget_tiers_pct=(15,),
        arrival_rates_sessions_per_second=(0.25, 1.0),
        eval_capacity_rates_sessions_per_second=(0.25, 1.0),
        replay_count=1,
        seed=13,
        vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
    )

    m = out["high_variety_population"]["median_packet_tokens"]
    mapped = out["config"]["eval_tokens_per_second_map"]
    assert float(mapped["0.25"]) == pytest.approx(m * 0.25)
    assert float(mapped["1.0"]) == pytest.approx(m * 1.0)
    sample = out["runs"][0]
    for key in (
        "queue_proposed_count",
        "queue_proposed_tokens",
        "queue_admitted_count",
        "queue_admitted_tokens",
        "queue_max_tokens",
        "queue_final_tokens",
        "token_pressure_ratio",
    ):
        assert key in sample


def test_cleanup_settle_seconds_threads_through_selection():
    data = _slice(60)
    runtime = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=3),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )
    manifest = build_exact_token_budget_manifest(runtime, eligible_unit_ids=data.unit_ids)
    budget_tokens = int(next(row["budget_tokens"] for row in manifest["outcome"] if int(row["legacy_tier_pct"]) == 20))

    row = select_v3_membership(
        data,
        runtime=runtime,
        method="adaptive_embedding_fullsession_token",
        eligible_unit_ids=data.unit_ids,
        budget_tokens=budget_tokens,
        seed=1,
        run_scope="cleanup-test",
        vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
        cleanup_settle_seconds=0.0,
    )
    assert row["selected_tokens"] <= budget_tokens


def test_replay_reconstructs_monotonic_timestamps_in_processing_order():
    data = _slice(30)
    order = tuple(reversed(data.unit_ids[:20]))
    replay = _prepare_replay(data, order)
    observed = [trace.timestamp for _, trace in replay]
    assert observed == [float(i) for i in range(len(order))]
    assert all(observed[i] < observed[i + 1] for i in range(len(observed) - 1))


def test_build_runtime_requires_explicit_embedder():
    data = _slice(10)
    with pytest.raises(ValueError, match="embedder is required"):
        build_v3_runtime(
            data,
            tokenizer=FakeTokenizer(),
            embedder=None,  # type: ignore[arg-type]
            embedding_model_id="text-embedding-3-small",
            embedding_deployment_id="dep-a",
        )


def test_build_runtime_embedding_identity_changes_scope_profile_and_ledger():
    data = _slice(30)
    base = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=1),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )
    changed_model = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=1),
        embedding_model_id="text-embedding-3-large",
        embedding_deployment_id="dep-a",
    )
    changed_deployment = build_v3_runtime(
        data,
        tokenizer=FakeTokenizer(),
        embedder=Deterministic1536Embedder(seed=1),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-b",
    )

    assert base.embedding_profile_id != changed_model.embedding_profile_id
    assert base.embedding_profile_id != changed_deployment.embedding_profile_id
    assert base.token_profile_id != changed_model.token_profile_id
    assert base.token_profile_id != changed_deployment.token_profile_id
    assert base.embedding_semantic_scope != changed_model.embedding_semantic_scope
    assert base.embedding_semantic_scope != changed_deployment.embedding_semantic_scope
    assert base.ledger.embedding_model_id == "text-embedding-3-small"
    assert base.ledger.embedding_deployment_id == "dep-a"
    assert base.ledger.embedding_embedder_class == "Deterministic1536Embedder"


def test_build_v3_offline_runtime_helper_uses_explicit_offline_embedder():
    data = _slice(20)
    runtime = build_v3_offline_runtime(data, tokenizer=FakeTokenizer(), seed=21)
    assert runtime.ledger.embedding_embedder_class == "Deterministic1536Embedder"


def test_non_positive_token_cost_rejected_at_runtime_build():
    data = _slice(8)
    with pytest.raises(ValueError, match="non-positive emitted token cost"):
        build_v3_runtime(
            data,
            tokenizer=ZeroTokenFakeTokenizer(),
            embedder=Deterministic1536Embedder(seed=3),
            embedding_model_id="text-embedding-3-small",
            embedding_deployment_id="dep-a",
            max_session_packet_tokens=20,
        )


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("tiktoken") is None,
    reason="tiktoken not installed",
)
def test_tiktoken_tokenizer_exposes_model_and_encoding_identity():
    tok = TiktokenTokenizer(model_name="text-embedding-3-small", encoding_name="cl100k_base")
    assert tok.model_name == "text-embedding-3-small"
    assert tok.encoding_name == "cl100k_base"
    assert tok.encoding_id.startswith("cl100k_base:")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("tiktoken") is None,
    reason="tiktoken not installed",
)
def test_tiktoken_identity_affects_token_profile_id():
    data = _slice(20)
    tok_a = TiktokenTokenizer(model_name="text-embedding-3-small", encoding_name="cl100k_base")
    tok_b = TiktokenTokenizer(model_name="text-embedding-3-small", encoding_name="o200k_base")
    rt_a = build_v3_runtime(
        data,
        tokenizer=tok_a,
        embedder=Deterministic1536Embedder(seed=2),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )
    rt_b = build_v3_runtime(
        data,
        tokenizer=tok_b,
        embedder=Deterministic1536Embedder(seed=2),
        embedding_model_id="text-embedding-3-small",
        embedding_deployment_id="dep-a",
    )
    assert rt_a.token_profile_id != rt_b.token_profile_id


def test_outcome_comparison_requires_explicit_runtime():
    data = _slice(15)
    with pytest.raises(ValueError, match="runtime is required"):
        run_v3_outcome_comparison(
            data,
            runtime=None,
            methods=V3_OUTCOME_METHODS,
            legacy_outcome_tiers_pct=(5,),
            repetitions=1,
            seed=13,
            vector_store_factory=lambda _tenant, _scope: InMemoryVectorStore(),
        )
