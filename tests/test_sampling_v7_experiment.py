from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np

from sampling_comparison.v6_experiment import SessionDescriptor, select_arm4, select_arm5
from sampling_comparison.v7_dataset import (
    COSMOS_LABELS_FILE,
    COSMOS_SPANS_FILE,
    load_cosmos_otel_expected_dataset,
    map_expected_outcome_label,
)
from sampling_comparison.v7_experiment import (
    V7_DEFAULT_BASE_SEED,
    V7_DEFAULT_REPETITIONS,
    V7_METHOD_IDS,
    build_replay_plan,
    derive_budget_caps,
    embedding_binary_metrics,
    fit_pca8_embeddings,
    idw_binary_population,
    run_v7_core,
    select_minhash_exact_cap,
    run_v7_experiment,
    select_diverse_exact_cap,
)
from sampling_comparison.v7_search import FakeV7SearchAdapter
from sampling_comparison.v7_search import AzureV7SearchAdapter, decode_search_cluster_id, encode_search_cluster_id
from trace_sampling.model import SessionEvent, Trace


def _vector_for_uid(uid: str, dim: int = 16) -> np.ndarray:
    seed = sum(ord(ch) for ch in uid) % 997
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim)
    norm = float(np.linalg.norm(vec))
    return np.asarray(vec / (norm or 1.0), dtype=np.float32)


def _trace_for_uid(uid: str, agent: str) -> Trace:
    return Trace(
        trace_id=int(uid.split("-")[-1]),
        agent_id=agent,
        timestamp=0.0,
        signature=("search", "read"),
        span_count=2,
        duration_ms=1.0,
        status="ok",
        concept_id=0,
        events=(SessionEvent(role="assistant", text=f"trace for {uid}"),),
    )


def test_v7_pca8_determinism_and_dimensions() -> None:
    unit_ids = [f"u-{i}" for i in range(12)]
    vectors = {uid: _vector_for_uid(uid) for uid in unit_ids}

    first = fit_pca8_embeddings(ordered_unit_ids=unit_ids, full_vectors_by_unit=vectors)
    second = fit_pca8_embeddings(ordered_unit_ids=unit_ids, full_vectors_by_unit=vectors)

    assert first["n_components"] == 8
    assert len(first["explained_variance_ratio"]) == 8
    assert first["input_dimensions"] == 16
    for uid in unit_ids:
        a = first["reduced_vectors_by_unit"][uid]
        b = second["reduced_vectors_by_unit"][uid]
        assert a.shape == (8,)
        assert np.allclose(a, b)


def test_v7_selector_metric_path_cosine_vs_euclidean() -> None:
    unit_ids = ["a", "b", "c", "d"]
    vectors = {
        "a": np.asarray([1.0, 0.0]),
        "b": np.asarray([100.0, 0.0]),
        "c": np.asarray([-1.0, 0.0]),
        "d": np.asarray([0.0, 1.0]),
    }
    cosine = select_diverse_exact_cap(
        ordered_unit_ids=unit_ids,
        vectors_by_unit=vectors,
        cap=2,
        metric="cosine",
        seed=13,
        anchor_unit_id="a",
    )
    euclidean = select_diverse_exact_cap(
        ordered_unit_ids=unit_ids,
        vectors_by_unit=vectors,
        cap=2,
        metric="euclidean",
        seed=13,
        anchor_unit_id="a",
    )
    assert cosine["selected_ids"][0] == "a"
    assert euclidean["selected_ids"][0] == "a"
    assert cosine["selected_ids"][1] == "c"
    assert euclidean["selected_ids"][1] == "b"


def test_v7_binary_imputation_and_metrics_scopes() -> None:
    unit_ids = ["u1", "u2", "u3", "u4"]
    labels = {"u1": 1, "u2": 0, "u3": 1, "u4": 0}
    vectors = {
        "u1": np.asarray([1.0, 0.0]),
        "u2": np.asarray([0.0, 1.0]),
        "u3": np.asarray([0.95, 0.05]),
        "u4": np.asarray([0.05, 0.95]),
    }
    selected = ["u1", "u2"]
    idw = idw_binary_population(
        ordered_unit_ids=unit_ids,
        selected_ids=selected,
        labels_by_unit=labels,
        vectors_by_unit=vectors,
        metric="cosine",
        k=1,
    )

    rows = {row["unit_id"]: row for row in idw.rows}
    assert all(row["binary"] in (0, 1) for row in rows.values())
    assert rows["u1"]["observed"] and rows["u1"]["binary"] == 1
    assert rows["u2"]["observed"] and rows["u2"]["binary"] == 0
    assert not rows["u3"]["observed"] and not rows["u4"]["observed"]

    metrics = embedding_binary_metrics(ordered_unit_ids=unit_ids, labels_by_unit=labels, idw_result=idw)
    assert metrics["counts"]["population"] == 4
    assert metrics["counts"]["imputed"] == 2
    assert metrics["judged_plus_imputed"]["accuracy"] == 1.0
    assert metrics["imputed_only"]["accuracy"] == 1.0


def test_v7_budget_caps() -> None:
    assert derive_budget_caps(300) == {1: 3, 3: 9, 5: 15, 10: 30, 20: 60}
    assert derive_budget_caps(2500) == {1: 25, 3: 75, 5: 125, 10: 250, 20: 500}
    assert derive_budget_caps(7) == {1: 1, 3: 1, 5: 1, 10: 1, 20: 1}


def test_v7_arm5_semantics_match_v6_membership() -> None:
    unit_ids = [f"u-{i}" for i in range(20)]
    labels = {uid: int(i % 2 == 0) for i, uid in enumerate(unit_ids)}
    agent_id_by_unit = {uid: f"agent-{(i % 4) + 1}" for i, uid in enumerate(unit_ids)}
    use_case_id_by_unit = {uid: f"task-{(i % 3) + 1}" for i, uid in enumerate(unit_ids)}
    business_use_case_guid_by_unit = {uid: use_case_id_by_unit[uid] for uid in unit_ids}
    concept_key_by_unit = {uid: f"concept-{(i % 5) + 1}" for i, uid in enumerate(unit_ids)}
    metadata_by_unit = {
        uid: {
            "task_id": use_case_id_by_unit[uid],
            "domain": f"domain-{(i % 2) + 1}",
        }
        for i, uid in enumerate(unit_ids)
    }
    traces = {uid: _trace_for_uid(uid, agent_id_by_unit[uid]) for uid in unit_ids}
    vectors = {uid: _vector_for_uid(uid) for uid in unit_ids}

    result = run_v7_core(
        ordered_unit_ids=unit_ids,
        labels_by_unit=labels,
        full_vectors_by_unit=vectors,
        traces_by_unit_id=traces,
        agent_id_by_unit=agent_id_by_unit,
        concept_key_by_unit=concept_key_by_unit,
        use_case_id_by_unit=use_case_id_by_unit,
        business_use_case_guid_by_unit=business_use_case_guid_by_unit,
        token_count_by_unit={uid: 100 for uid in unit_ids},
        metadata_by_unit=metadata_by_unit,
        budget_levels=(20,),
        seed=13,
        window_id="v7-test",
    )
    arm5_row = next(row for row in result["runs"] if row["method_id"] == V7_METHOD_IDS["arm5"])
    assert arm5_row["coverage"]["concept"]["coverage_ratio"] >= 0.0
    assert arm5_row["coverage"]["concept"]["selected_distinct"] >= 1

    descriptors = tuple(
        SessionDescriptor(
            unit_id=uid,
            agent_id=agent_id_by_unit[uid],
            use_case_id=use_case_id_by_unit[uid],
            business_use_case_guid=business_use_case_guid_by_unit[uid],
            concept_key=concept_key_by_unit[uid],
            label=bool(labels[uid]),
        )
        for uid in unit_ids
    )
    cap = 4
    arm4 = select_arm4(descriptors=descriptors, cap=cap, trial_seed=13, window_id="v7-test")
    arm5 = select_arm5(descriptors=descriptors, arm4_outcome=arm4, labels_by_unit={uid: bool(v) for uid, v in labels.items()})

    assert arm5_row["cap"] == cap
    assert arm5_row["selected_ids"] == list(arm5.selected_ids)


def test_v7_cosmos_expected_label_mapping_and_fallback_representation(tmp_path) -> None:
    labels_path = tmp_path / COSMOS_LABELS_FILE
    spans_path = tmp_path / COSMOS_SPANS_FILE

    labels = [
        {
            "id": "label-1",
            "sessionId": "session-1",
            "traceIds": ["trace-1"],
            "expected_outcome": "good",
            "task_id": "task-alpha",
            "domain": "ops",
            "euw_guid": "guid-1",
            "required_tools": ["search"],
            "available_tools": ["search", "read"],
        },
        {
            "id": "label-2",
            "sessionId": "session-2",
            "traceIds": ["trace-missing"],
            "expected_outcome": "partial",
            "task_id": "task-beta",
            "domain": "sales",
            "euw_guid": "guid-2",
            "required_tools": ["query"],
            "available_knowledge": ["doc-a"],
        },
    ]
    spans = [
        {
            "traceId": "trace-session",
            "spanId": "b",
            "name": "invoke_agent",
            "agentId": "agent-1",
            "attributes": {
                "gen_ai.response.id": "trace-1",
                "genesis.session.id": "session-1",
                "startTime": "2",
                "endTime": "3",
            },
            "events": [{"name": "step"}],
        },
        {
            "traceId": "trace-session-2",
            "spanId": "a",
            "name": "tool",
            "agentId": "agent-1",
            "attributes": {
                "genesis.session.id": "session-1",
                "startTime": "1",
                "endTime": "1.5",
            },
            "events": [{"name": "step"}],
        }
    ]
    labels_path.write_text("\n".join(json.dumps(row) for row in labels) + "\n", encoding="utf-8")
    spans_path.write_text("\n".join(json.dumps(row) for row in spans) + "\n", encoding="utf-8")

    dataset = load_cosmos_otel_expected_dataset(tmp_path)

    uid1 = "cosmos_otel:label-1"
    uid2 = "cosmos_otel:label-2"
    assert dataset.labels_by_unit[uid1] == 1
    assert dataset.labels_by_unit[uid2] == 0
    assert dataset.representation_source_by_unit[uid1] == "linked_raw_spans"
    assert dataset.representation_source_by_unit[uid2] == "synthetic_label_document"
    assert dataset.metadata_by_unit[uid1]["task_id"] == "task-alpha"
    assert dataset.metadata_by_unit[uid1]["domain"] == "ops"
    assert dataset.metadata_by_unit[uid1]["euw_guid"] == "guid-1"
    assert dataset.metadata_by_unit[uid1]["linkage_path"] == "response_id_to_session"
    assert dataset.metadata_by_unit[uid1]["linked_session_ids"] == ["session-1"]
    assert dataset.metadata_by_unit[uid1]["linked_span_count"] == 2
    assert dataset.representation_text_by_unit[uid1].splitlines()[0].find("tool") != -1
    assert dataset.metadata_by_unit[uid2]["task_design_expected_label"] == 0
    fallback_text = dataset.representation_text_by_unit[uid2]
    assert "expected_outcome" not in fallback_text
    assert "\"good\"" not in fallback_text
    assert "\"bad\"" not in fallback_text
    assert "\"partial\"" not in fallback_text
    assert map_expected_outcome_label("partial", partial_label=1) == 1


def test_v7_cosmos_span_chronology_sort_uses_top_level_iso8601(tmp_path) -> None:
    labels_path = tmp_path / COSMOS_LABELS_FILE
    spans_path = tmp_path / COSMOS_SPANS_FILE
    labels = [
        {
            "id": "label-1",
            "sessionId": "session-chron",
            "traceIds": ["resp-1"],
            "expected_outcome": "good",
            "task_id": "task-alpha",
            "domain": "ops",
            "required_tools": ["search"],
        }
    ]
    spans = [
        {
            "traceId": "t-late",
            "spanId": "s2",
            "startTime": "2026-08-28T10:11:04.226263+01:00",
            "endTime": "2026-08-28T10:11:04.326263+01:00",
            "attributes": {"gen_ai.response.id": "resp-1", "genesis.session.id": "session-chron"},
            "events": [{"name": "later"}],
        },
        {
            "traceId": "t-early",
            "spanId": "s1",
            "startTime": "2026-08-28T10:11:03.126263+01:00",
            "endTime": "2026-08-28T10:11:03.226263+01:00",
            "attributes": {"gen_ai.response.id": "resp-1", "genesis.session.id": "session-chron"},
            "events": [{"name": "earlier"}],
        },
    ]
    labels_path.write_text("\n".join(json.dumps(row) for row in labels) + "\n", encoding="utf-8")
    spans_path.write_text("\n".join(json.dumps(row) for row in spans) + "\n", encoding="utf-8")

    dataset = load_cosmos_otel_expected_dataset(tmp_path)
    text = dataset.representation_text_by_unit["cosmos_otel:label-1"]
    first_line = text.splitlines()[0]
    second_line = text.splitlines()[1]
    assert "earlier" in first_line
    assert "later" in second_line


def test_v7_search_key_encoding_round_trip_and_original_unit_retrieval() -> None:
    uid = "historical_300:inactivity:foo/bar?baz"
    encoded = encode_search_cluster_id(uid)
    assert ":" not in encoded
    assert decode_search_cluster_id(encoded) == uid

    adapter = FakeV7SearchAdapter()
    written = adapter.sync_documents(
        index_name="idx",
        dimensions=2,
        metric="cosine",
        documents=[
            {
                "cluster_id": uid,
                "original_unit_id": uid,
                "agent_id": "a",
                "run_scope": "r",
                "semantic_scope": "s",
                "last_seen": 0.0,
                "vector": [1.0, 0.0],
            }
        ],
    )
    assert written == 1
    hits = adapter.query_neighbors(
        index_name="idx",
        vector=np.asarray([1.0, 0.0], dtype=np.float32),
        k=1,
        metric="cosine",
        run_scope="r",
        semantic_scope="s",
    )
    assert hits
    assert hits[0]["original_unit_id"] == uid
    assert hits[0]["cluster_id"].startswith("b64_")


def test_v7_azure_sync_documents_chunks_and_validates_success(monkeypatch) -> None:
    calls: list[int] = []

    class _FakeSearchClient:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

        def merge_or_upload_documents(self, docs):
            calls.append(len(docs))
            return [SimpleNamespace(succeeded=True, key=row.get("cluster_id")) for row in docs]

    fake_docs_module = ModuleType("azure.search.documents")
    fake_docs_module.SearchClient = _FakeSearchClient
    monkeypatch.setitem(sys.modules, "azure.search.documents", fake_docs_module)

    cfg = SimpleNamespace(search_endpoint="https://example", search_api_key=None)
    adapter = AzureV7SearchAdapter(azure_config=cfg, cosine_index_name="idx-c", euclidean_index_name="idx-e")
    monkeypatch.setattr(adapter, "ensure_index", lambda **kwargs: None)
    monkeypatch.setattr(adapter, "_credential", lambda _cfg: object())

    documents = []
    for i in range(1205):
        documents.append(
            {
                "cluster_id": f"historical_300:item:{i}",
                "agent_id": "a",
                "run_scope": "r",
                "semantic_scope": "s",
                "last_seen": 0.0,
                "vector": [0.1] * 8,
            }
        )
    written = adapter.sync_documents(index_name="idx-c", documents=documents, dimensions=8, metric="cosine")
    assert written == 1205
    assert calls == [500, 500, 205]


def test_v7_azure_sync_documents_raises_on_failed_result(monkeypatch) -> None:
    class _FakeSearchClient:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

        def merge_or_upload_documents(self, docs):
            rows = [SimpleNamespace(succeeded=True, key=row.get("cluster_id")) for row in docs]
            rows[0] = SimpleNamespace(succeeded=False, key=docs[0].get("cluster_id"), status_code=400, error_message="bad")
            return rows

    fake_docs_module = ModuleType("azure.search.documents")
    fake_docs_module.SearchClient = _FakeSearchClient
    monkeypatch.setitem(sys.modules, "azure.search.documents", fake_docs_module)

    cfg = SimpleNamespace(search_endpoint="https://example", search_api_key=None)
    adapter = AzureV7SearchAdapter(azure_config=cfg, cosine_index_name="idx-c", euclidean_index_name="idx-e")
    monkeypatch.setattr(adapter, "ensure_index", lambda **kwargs: None)
    monkeypatch.setattr(adapter, "_credential", lambda _cfg: object())

    try:
        adapter.sync_documents(
            index_name="idx-c",
            dimensions=8,
            metric="cosine",
            documents=[
                {
                    "cluster_id": "historical_300:item:1",
                    "agent_id": "a",
                    "run_scope": "r",
                    "semantic_scope": "s",
                    "last_seen": 0.0,
                    "vector": [0.1] * 8,
                }
            ],
        )
    except ValueError as exc:
        assert "failed" in str(exc)
    else:
        raise AssertionError("expected sync failure to raise ValueError")


def test_v7_default_matrix_and_aggregate_bundle(tmp_path) -> None:
    from sampling_comparison.v7_experiment import build_v7_method_matrix

    datasets = {}
    for dataset_id, size in (("historical_300", 8), ("dense_2500", 8), ("cosmos_otel", 8)):
        unit_ids = [f"{dataset_id}-{i}" for i in range(size)]
        vectors = {uid: _vector_for_uid(uid, dim=16) for uid in unit_ids}
        labels = {uid: int(i % 2 == 0) for i, uid in enumerate(unit_ids)}
        traces = {uid: _trace_for_uid(uid, "agent-1") for uid in unit_ids}
        agent_id_by_unit = {uid: "agent-1" for uid in unit_ids}
        concept_key_by_unit = {uid: f"concept-{i % 5}" for i, uid in enumerate(unit_ids)}
        use_case_id_by_unit = {uid: f"task-{i % 3}" for i, uid in enumerate(unit_ids)}
        business_use_case_guid_by_unit = {uid: f"guid-{i % 4}" for i, uid in enumerate(unit_ids)}
        metadata_by_unit = {uid: {"domain": "ops", "task_id": use_case_id_by_unit[uid]} for uid in unit_ids}
        datasets[dataset_id] = {
            "ordered_unit_ids": unit_ids,
            "labels_by_unit": labels,
            "full_vectors_by_unit": vectors,
            "traces_by_unit_id": traces,
            "agent_id_by_unit": agent_id_by_unit,
            "concept_key_by_unit": concept_key_by_unit,
            "use_case_id_by_unit": use_case_id_by_unit,
            "business_use_case_guid_by_unit": business_use_case_guid_by_unit,
            "metadata_by_unit": metadata_by_unit,
            "token_count_by_unit": {uid: 120 for uid in unit_ids},
        }

    result = run_v7_experiment(datasets=datasets, output_dir=tmp_path, search_adapter=FakeV7SearchAdapter())
    assert len(result["runs"]) == 750
    assert len(result["replays"]) == 30
    assert build_v7_method_matrix()[0] == (13, 1)
    assert result["summary"]["dataset_count"] == 3
    assert result["summary"]["seed_count"] == 10
    assert result["summary"]["repetition_count"] == 10
    assert result["summary"]["base_seed"] == 13
    assert result["summary"]["paired_replay"] is True
    assert result["summary"]["randomized_order"] is True
    assert result["summary"]["randomized_frequency"] is True
    assert result["summary"]["budget_count"] == 5
    assert all("label" not in row for row in result["memberships"])
    assert all("selected_rate" not in row for row in result["memberships"])
    assert all("selected_ids" not in row for row in result["memberships"])
    assert all("shared_pca" not in (row.get("latency_seconds") or {}) for row in result["runs"])
    assert all("expected_outcome" not in row for row in result["memberships"])
    assert (tmp_path / "aggregate.json").exists()
    assert (tmp_path / "replays.jsonl").exists()
    assert (tmp_path / "aggregate_metrics.json").exists()
    assert (tmp_path / "paired_differences.json").exists()
    assert (tmp_path / "memberships.jsonl").exists()
    assert (tmp_path / "imputed_predictions.jsonl").exists()
    assert (tmp_path / "interactive_report.html").exists()
    assert (tmp_path / "manifest.json").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "manifest" not in manifest.get("hashes", {})
    assert len(manifest.get("embedding_ledger") or []) == 3
    for key, path_text in manifest["files"].items():
        path = tmp_path / Path(path_text).name
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert manifest["hashes"][key] == actual_hash

    profile = json.loads((tmp_path / "dataset_profile.json").read_text(encoding="utf-8"))
    profile_rows = profile["datasets"]
    assert len(profile_rows) == 3
    first_profile = profile_rows[0]
    assert "label_rate" in first_profile
    assert "representation_source_counts" in first_profile
    assert "runtime_ledger" in first_profile
    assert len(first_profile["pca"]["explained_variance_ratio"]) == 8


def test_v7_replay_plan_determinism_and_frequency_properties() -> None:
    ordered = tuple(f"u-{i}" for i in range(8))
    first = build_replay_plan(dataset_id="d", ordered_source_unit_ids=ordered, repetition_index=1, base_seed=13)
    second = build_replay_plan(dataset_id="d", ordered_source_unit_ids=ordered, repetition_index=1, base_seed=13)
    third = build_replay_plan(dataset_id="d", ordered_source_unit_ids=ordered, repetition_index=2, base_seed=13)

    assert first.occurrence_ids == second.occurrence_ids
    assert first.source_unit_ids == second.source_unit_ids
    assert first.order_sha256 == second.order_sha256
    assert first.frequency_sha256 == second.frequency_sha256
    assert first.order_sha256 != third.order_sha256 or first.frequency_sha256 != third.frequency_sha256
    assert len(first.occurrence_ids) == len(ordered)
    assert len(set(first.occurrence_ids)) == len(ordered)
    assert sum(first.frequency_by_source.values()) == len(ordered)
    assert set(first.frequency_by_source.keys()) == set(ordered)

    has_duplicates = False
    has_zeros = False
    for rep in range(1, 21):
        plan = build_replay_plan(dataset_id="d", ordered_source_unit_ids=ordered, repetition_index=rep, base_seed=13)
        if any(v > 1 for v in plan.frequency_by_source.values()):
            has_duplicates = True
        if any(v == 0 for v in plan.frequency_by_source.values()):
            has_zeros = True
        if has_duplicates and has_zeros:
            break
    assert has_duplicates is True
    assert has_zeros is True


def test_v7_minhash_observes_the_replay_order(monkeypatch) -> None:
    from sampling_comparison import v7_experiment_repaired as v7r

    observed_trace_ids: list[int] = []

    class _RecordingIndex:
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

        def observe(self, trace):
            observed_trace_ids.append(int(trace.trace_id))
            return SimpleNamespace(novelty=1.0, rarity=0.0)

        def telemetry(self):
            return {}

    monkeypatch.setattr(v7r, "BandedMinHashLSHIndex", _RecordingIndex)
    ordered = ["u-3", "u-1", "u-2"]
    traces = {uid: _trace_for_uid(uid, "agent-1") for uid in ordered}

    select_minhash_exact_cap(
        ordered_unit_ids=ordered,
        traces_by_unit_id=traces,
        cap=2,
        seed=13,
    )

    assert observed_trace_ids == [3, 1, 2]


def test_v7_headline_mae_uses_fixed_source_census() -> None:
    unit_ids = [f"u-{i}" for i in range(8)]
    labels = {uid: 1 for uid in unit_ids}
    vectors = {uid: _vector_for_uid(uid, dim=16) for uid in unit_ids}
    traces = {uid: _trace_for_uid(uid, "agent-1") for uid in unit_ids}
    result = run_v7_core(
        ordered_unit_ids=unit_ids,
        labels_by_unit=labels,
        full_vectors_by_unit=vectors,
        traces_by_unit_id=traces,
        agent_id_by_unit={uid: "agent-1" for uid in unit_ids},
        concept_key_by_unit={uid: "concept" for uid in unit_ids},
        use_case_id_by_unit={uid: "task" for uid in unit_ids},
        business_use_case_guid_by_unit={uid: "guid" for uid in unit_ids},
        token_count_by_unit={uid: 1 for uid in unit_ids},
        metadata_by_unit={uid: {"domain": "ops", "task_id": "task"} for uid in unit_ids},
        budget_levels=(20,),
        seed=13,
        source_corpus_census_pass_rate=0.0,
    )
    random_row = next(row for row in result["runs"] if row["method_id"] == V7_METHOD_IDS["random"])

    assert random_row["replay_census_pass_rate"] == 1.0
    assert random_row["source_corpus_census_pass_rate"] == 0.0
    assert random_row["aggregate_pass_rate_mae"] == 1.0
    assert random_row["replay_aggregate_pass_rate_mae"] == 0.0


def test_v7_replay_ci_uses_student_t_for_small_samples() -> None:
    from sampling_comparison import v7_experiment_repaired as v7r

    stats = v7r._metric_summary([0.0, 1.0, 2.0, 3.0, 4.0])
    normal_half_width = 1.96 * float(stats["standard_error"])
    actual_half_width = float(stats["mean_ci95_upper"]) - float(stats["mean"])

    assert actual_half_width > normal_half_width


def test_v7_replay_hashes_paired_within_repetition_and_different_between_repetitions(tmp_path) -> None:
    unit_ids = [f"historical_300-{i}" for i in range(10)]
    vectors = {uid: _vector_for_uid(uid, dim=16) for uid in unit_ids}
    labels = {uid: int(i % 2 == 0) for i, uid in enumerate(unit_ids)}
    traces = {uid: _trace_for_uid(uid, "agent-1") for uid in unit_ids}
    dataset = {
        "ordered_unit_ids": unit_ids,
        "labels_by_unit": labels,
        "full_vectors_by_unit": vectors,
        "traces_by_unit_id": traces,
        "agent_id_by_unit": {uid: "agent-1" for uid in unit_ids},
        "concept_key_by_unit": {uid: f"concept-{i % 4}" for i, uid in enumerate(unit_ids)},
        "use_case_id_by_unit": {uid: f"task-{i % 3}" for i, uid in enumerate(unit_ids)},
        "business_use_case_guid_by_unit": {uid: f"guid-{i % 2}" for i, uid in enumerate(unit_ids)},
        "metadata_by_unit": {uid: {"domain": "ops", "task_id": "task"} for uid in unit_ids},
        "token_count_by_unit": {uid: 20 for uid in unit_ids},
    }
    result = run_v7_experiment(
        datasets={"historical_300": dataset},
        output_dir=tmp_path,
        repetitions=2,
        base_seed=13,
        search_adapter=FakeV7SearchAdapter(),
    )
    by_rep = {}
    for row in result["runs"]:
        rep = int(row["repetition_index"])
        by_rep.setdefault(rep, set()).add((row["replay_order_sha256"], row["replay_frequency_sha256"]))
    assert len(by_rep[1]) == 1
    assert len(by_rep[2]) == 1
    assert next(iter(by_rep[1])) != next(iter(by_rep[2]))
    assert all("source_corpus_census_pass_rate" in row for row in result["runs"])
    assert all("replay_census_pass_rate" in row for row in result["runs"])


def test_v7_pca_fit_once_per_dataset_not_per_repetition(monkeypatch, tmp_path) -> None:
    calls = {"n": 0}
    from sampling_comparison import v7_experiment_repaired as v7r

    original = v7r.fit_pca8_embeddings

    def _counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(v7r, "fit_pca8_embeddings", _counted)

    unit_ids = [f"historical_300-{i}" for i in range(8)]
    vectors = {uid: _vector_for_uid(uid, dim=16) for uid in unit_ids}
    labels = {uid: int(i % 2 == 0) for i, uid in enumerate(unit_ids)}
    traces = {uid: _trace_for_uid(uid, "agent-1") for uid in unit_ids}
    dataset = {
        "ordered_unit_ids": unit_ids,
        "labels_by_unit": labels,
        "full_vectors_by_unit": vectors,
        "traces_by_unit_id": traces,
        "agent_id_by_unit": {uid: "agent-1" for uid in unit_ids},
        "concept_key_by_unit": {uid: "c" for uid in unit_ids},
        "use_case_id_by_unit": {uid: "t" for uid in unit_ids},
        "business_use_case_guid_by_unit": {uid: "g" for uid in unit_ids},
        "metadata_by_unit": {uid: {"domain": "ops", "task_id": "task"} for uid in unit_ids},
        "token_count_by_unit": {uid: 1 for uid in unit_ids},
    }
    _ = run_v7_experiment(
        datasets={"historical_300": dataset},
        output_dir=tmp_path,
        repetitions=3,
        base_seed=13,
        search_adapter=FakeV7SearchAdapter(),
    )
    assert calls["n"] == 1


def test_v7_search_writes_not_multiplied_by_repetitions(tmp_path) -> None:
    adapter = FakeV7SearchAdapter()
    unit_ids = [f"historical_300-{i}" for i in range(8)]
    vectors = {uid: _vector_for_uid(uid, dim=16) for uid in unit_ids}
    labels = {uid: int(i % 2 == 0) for i, uid in enumerate(unit_ids)}
    traces = {uid: _trace_for_uid(uid, "agent-1") for uid in unit_ids}
    dataset = {
        "ordered_unit_ids": unit_ids,
        "labels_by_unit": labels,
        "full_vectors_by_unit": vectors,
        "traces_by_unit_id": traces,
        "agent_id_by_unit": {uid: "agent-1" for uid in unit_ids},
        "concept_key_by_unit": {uid: "c" for uid in unit_ids},
        "use_case_id_by_unit": {uid: "t" for uid in unit_ids},
        "business_use_case_guid_by_unit": {uid: "g" for uid in unit_ids},
        "metadata_by_unit": {uid: {"domain": "ops", "task_id": "task"} for uid in unit_ids},
        "token_count_by_unit": {uid: 1 for uid in unit_ids},
    }
    _ = run_v7_experiment(
        datasets={"historical_300": dataset},
        output_dir=tmp_path,
        repetitions=4,
        base_seed=13,
        search_adapter=adapter,
    )
    writes = adapter.telemetry
    assert writes["trace-clusters-sampling-v7-cosine"]["writes"] == 8
    assert writes["trace-clusters-sampling-v7-euclidean"]["writes"] == 8


def test_v7_aggregate_and_paired_outputs_have_expected_fields(tmp_path) -> None:
    unit_ids = [f"historical_300-{i}" for i in range(8)]
    vectors = {uid: _vector_for_uid(uid, dim=16) for uid in unit_ids}
    labels = {uid: int(i % 2 == 0) for i, uid in enumerate(unit_ids)}
    traces = {uid: _trace_for_uid(uid, "agent-1") for uid in unit_ids}
    dataset = {
        "ordered_unit_ids": unit_ids,
        "labels_by_unit": labels,
        "full_vectors_by_unit": vectors,
        "traces_by_unit_id": traces,
        "agent_id_by_unit": {uid: "agent-1" for uid in unit_ids},
        "concept_key_by_unit": {uid: "c" for uid in unit_ids},
        "use_case_id_by_unit": {uid: "t" for uid in unit_ids},
        "business_use_case_guid_by_unit": {uid: "g" for uid in unit_ids},
        "metadata_by_unit": {uid: {"domain": "ops", "task_id": "task"} for uid in unit_ids},
        "token_count_by_unit": {uid: 1 for uid in unit_ids},
    }
    result = run_v7_experiment(
        datasets={"historical_300": dataset},
        output_dir=tmp_path,
        repetitions=3,
        base_seed=13,
        search_adapter=FakeV7SearchAdapter(),
    )
    agg = json.loads((tmp_path / "aggregate_metrics.json").read_text(encoding="utf-8"))
    rows = agg["rows"]
    assert rows
    first = rows[0]
    for field in ("n", "mean", "median", "sample_std", "standard_error", "min", "max", "p05", "p25", "p75", "p95", "mean_ci95_lower", "mean_ci95_upper"):
        assert field in first
        assert np.isfinite(float(first[field]))
    assert any(row["n"] == 3 for row in rows if row["metric_id"] == "aggregate_pass_rate_mae")

    paired = json.loads((tmp_path / "paired_differences.json").read_text(encoding="utf-8"))
    assert paired["rows"]
    mae_pair = next(row for row in paired["rows"] if row["metric_id"] == "aggregate_pass_rate_mae")
    assert mae_pair["n"] == 3
    assert mae_pair["wins"] + mae_pair["ties"] + mae_pair["losses"] == mae_pair["n"]
    assert 0.0 <= float(mae_pair["win_rate"]) <= 1.0


def test_v7_replays_and_memberships_are_label_free(tmp_path) -> None:
    unit_ids = [f"historical_300-{i}" for i in range(8)]
    vectors = {uid: _vector_for_uid(uid, dim=16) for uid in unit_ids}
    labels = {uid: int(i % 2 == 0) for i, uid in enumerate(unit_ids)}
    traces = {uid: _trace_for_uid(uid, "agent-1") for uid in unit_ids}
    dataset = {
        "ordered_unit_ids": unit_ids,
        "labels_by_unit": labels,
        "full_vectors_by_unit": vectors,
        "traces_by_unit_id": traces,
        "agent_id_by_unit": {uid: "agent-1" for uid in unit_ids},
        "concept_key_by_unit": {uid: "c" for uid in unit_ids},
        "use_case_id_by_unit": {uid: "t" for uid in unit_ids},
        "business_use_case_guid_by_unit": {uid: "g" for uid in unit_ids},
        "metadata_by_unit": {uid: {"domain": "ops", "task_id": "task"} for uid in unit_ids},
        "token_count_by_unit": {uid: 1 for uid in unit_ids},
    }
    _ = run_v7_experiment(
        datasets={"historical_300": dataset},
        output_dir=tmp_path,
        repetitions=2,
        base_seed=13,
        search_adapter=FakeV7SearchAdapter(),
    )
    replay_lines = [json.loads(line) for line in (tmp_path / "replays.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert replay_lines
    for row in replay_lines:
        assert "label" not in row
        assert "expected_outcome" not in row
        assert "source_unit_ids_in_order" in row
        assert "frequency_by_source" in row

    membership_lines = [json.loads(line) for line in (tmp_path / "memberships.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert membership_lines
    for row in membership_lines:
        assert "label" not in row
        assert "replay_id" in row
        assert "replay_order_sha256" in row
        assert "replay_frequency_sha256" in row


def test_v7_report_includes_required_sections_and_values(tmp_path) -> None:
    from sampling_comparison.v7_experiment import build_v7_report_html

    payload = {
        "runs": [
            {
                "dataset_id": "historical_300",
                "seed": 13,
                "budget_pct": 10,
                "method_id": "random_sampling",
                "estimator_type": "selected_rate",
                "estimate": 0.55,
                "aggregate_pass_rate_mae": 0.1,
                "selected_only_pass_rate_mae": 0.2,
                "actual_token_count": 180,
                "coverage": {"agent": 1.0, "domain": 1.0, "task": 1.0},
                "search_evidence": {"neighbor_hits": 1},
                "latency_seconds": {"per_method_total": 1.2},
                "embedding_metrics": {
                    "imputed_only": {"accuracy": 0.75, "precision": 0.8, "recall": 0.7, "f1": 0.7467},
                    "judged_plus_imputed": {"accuracy": 0.9, "precision": 0.92, "recall": 0.88, "f1": 0.8996},
                },
            },
            {
                "dataset_id": "historical_300",
                "seed": 13,
                "budget_pct": 10,
                "method_id": "pca8_idw_binary_cosine",
                "estimator_type": "pca_binary_imputed_population",
                "estimate": 0.6,
                "aggregate_pass_rate_mae": 0.08,
                "selected_only_pass_rate_mae": 0.21,
                "actual_token_count": 200,
                "coverage": {"agent": 1.0, "domain": 1.0, "task": 1.0},
                "search_evidence": {"neighbor_hits": 2},
                "latency_seconds": {"per_method_total": 1.1},
                "embedding_metrics": {
                    "imputed_only": {"accuracy": 0.8, "precision": 0.82, "recall": 0.79, "f1": 0.804},
                    "judged_plus_imputed": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9},
                },
            }
        ],
        "datasets": [
            {
                "dataset_id": "historical_300",
                "population": 300,
                "representation_source_counts": {"combined_normalized": 300},
                "pca": {"explained_variance_total": 0.73},
            }
        ],
        "summary": {"dataset_count": 1, "seed_count": 1, "budget_count": 1, "run_count": 1},
    }
    html = build_v7_report_html(payload)
    assert "Executive Readout" in html
    assert "Study Design" in html
    assert "Five parallel methods" in html
    assert "Budget Trend" in html
    assert "Replay Uncertainty" in html
    assert "Classification Quality" in html
    assert "Coverage &amp; Cost" in html
    assert "Paired Cosine vs Euclidean" in html
    assert "PCA-8 Context" in html
    assert "Analysis" in html
    assert "Limitations" in html
    assert "Budget %" in html
    assert "Methodology" in html
    assert "Metric" in html
    assert "imputed_only_f1" in html
    assert "0.55" in html
    assert "MAE" in html
    assert "0.7467" in html
    assert "parallel-study-graph" in html
    assert "pairedCells" in html
    assert "explained_variance_ratio" in html
    assert "not a sequential flow" in html.lower()


def test_v7_search_evidence_retries_until_hits_without_sleep_in_tests(tmp_path) -> None:
    class _RetryAdapter(FakeV7SearchAdapter):
        def __init__(self) -> None:
            super().__init__()
            self._seen: dict[tuple[str, str, str], int] = {}

        def query_neighbors(self, *, index_name, vector, k, metric, run_scope, semantic_scope):
            key = (str(index_name), str(run_scope), str(semantic_scope))
            count = self._seen.get(key, 0)
            self._seen[key] = count + 1
            if count == 0:
                return []
            return super().query_neighbors(
                index_name=index_name,
                vector=vector,
                k=k,
                metric=metric,
                run_scope=run_scope,
                semantic_scope=semantic_scope,
            )

    unit_ids = [f"historical_300-{i}" for i in range(8)]
    vectors = {uid: _vector_for_uid(uid, dim=16) for uid in unit_ids}
    labels = {uid: int(i % 2 == 0) for i, uid in enumerate(unit_ids)}
    traces = {uid: _trace_for_uid(uid, "agent-1") for uid in unit_ids}
    dataset = {
        "ordered_unit_ids": unit_ids,
        "labels_by_unit": labels,
        "full_vectors_by_unit": vectors,
        "traces_by_unit_id": traces,
        "agent_id_by_unit": {uid: "agent-1" for uid in unit_ids},
        "concept_key_by_unit": {uid: f"concept-{i % 4}" for i, uid in enumerate(unit_ids)},
        "use_case_id_by_unit": {uid: f"task-{i % 3}" for i, uid in enumerate(unit_ids)},
        "business_use_case_guid_by_unit": {uid: f"guid-{i % 2}" for i, uid in enumerate(unit_ids)},
        "metadata_by_unit": {uid: {"domain": "ops", "task_id": "task"} for uid in unit_ids},
        "token_count_by_unit": {uid: 10 for uid in unit_ids},
    }
    result = run_v7_experiment(
        datasets={"historical_300": dataset},
        output_dir=tmp_path,
        repetitions=1,
        seed_values=(13,),
        budget_levels=(10,),
        method_ids=(V7_METHOD_IDS["pca8_cosine"],),
        search_adapter=_RetryAdapter(),
        search_probe_attempts=3,
        search_probe_delay_seconds=0.0,
    )
    assert len(result["runs"]) == 1
    evidence = result["runs"][0]["search_evidence"]
    assert evidence["attempt_count"] >= 1
    assert evidence["neighbor_hits"] > 0


def test_v7_idw_within_agent_then_global_fallback() -> None:
    unit_ids = ["a-1", "a-2", "b-1", "b-2"]
    labels = {"a-1": 1, "a-2": 0, "b-1": 0, "b-2": 1}
    vectors = {
        "a-1": np.asarray([1.0, 0.0]),
        "a-2": np.asarray([0.0, 1.0]),
        "b-1": np.asarray([0.95, 0.05]),
        "b-2": np.asarray([0.05, 0.95]),
    }
    agent_ids = {"a-1": "A", "a-2": "A", "b-1": "B", "b-2": "B"}
    result = idw_binary_population(
        ordered_unit_ids=unit_ids,
        selected_ids=["a-1"],
        labels_by_unit=labels,
        vectors_by_unit=vectors,
        metric="cosine",
        agent_id_by_unit=agent_ids,
        k=1,
    )
    rows = {row["unit_id"]: row for row in result.rows}
    assert rows["a-2"]["donor_scope"] == "within_agent"
    assert rows["b-1"]["donor_scope"] == "global_fallback"
