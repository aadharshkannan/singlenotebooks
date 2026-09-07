from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from minhash_sampling.signature import MinHashRecord
from sampling_comparison.v7_dataset import V7Dataset
from trace_sampling.model import SessionEvent, Trace


def _dataset(dataset_id: str) -> V7Dataset:
    unit_ids = tuple(f"{dataset_id}-{i}" for i in range(8))
    traces = {
        uid: Trace(
            trace_id=int(uid.split("-")[-1]),
            agent_id="agent-1",
            timestamp=0.0,
            signature=("a", "b"),
            span_count=2,
            duration_ms=1.0,
            status="ok",
            concept_id=0,
            events=(SessionEvent(role="assistant", text=uid),),
        )
        for uid in unit_ids
    }
    return V7Dataset(
        dataset_id=dataset_id,
        ordered_unit_ids=unit_ids,
        labels_by_unit={uid: int(i % 2 == 0) for i, uid in enumerate(unit_ids)},
        traces_by_unit_id=traces,
        agent_id_by_unit={uid: "agent-1" for uid in unit_ids},
        concept_key_by_unit={uid: "concept" for uid in unit_ids},
        use_case_id_by_unit={uid: "task" for uid in unit_ids},
        business_use_case_guid_by_unit={uid: "guid" for uid in unit_ids},
        metadata_by_unit={uid: {"task_id": "task", "domain": "ops"} for uid in unit_ids},
        representation_source_by_unit={uid: "combined_normalized" for uid in unit_ids},
        representation_text_by_unit={uid: "" for uid in unit_ids},
        source_paths={},
    )


def test_run_sampling_v7_cli_builds_real_payload_and_passes_runtime_outputs(monkeypatch, tmp_path):
    import scripts.run_sampling_v7 as cli

    datasets = {
        "historical_300": _dataset("historical_300"),
        "dense_2500": _dataset("dense_2500"),
        "cosmos_otel": _dataset("cosmos_otel"),
    }
    monkeypatch.setattr(
        "sampling_comparison.v7_dataset.load_v7_datasets",
        lambda **kwargs: datasets,
    )

    runtime_vectors = {i: np.asarray([float(i + 1)] * 1536, dtype=np.float32) for i in range(8)}

    def _runtime_with_cache(data, **kwargs):
        _ = data
        vectors_by_trace = {i: runtime_vectors[i] for i in range(8)}
        minhash = {
            uid: MinHashRecord(
                content_sha256=f"h-{uid}",
                profile_id="p",
                signature=(1, 2, 3, 4),
                shingle_count=4,
                representation_truncated=False,
            )
            for uid in data.unit_ids
        }
        runtime = SimpleNamespace(
            embedding_vector_by_trace_id=vectors_by_trace,
            token_cost_by_unit_id={uid: 321 for uid in data.unit_ids},
            minhash_records_by_unit_id=minhash,
            ledger=SimpleNamespace(
                embedding_model_id="text-embedding-3-small",
                embedding_deployment_id="dep",
                embedding_calls=3,
                embedding_inputs=8,
                embedding_input_tokens=999,
                embedding_latency_seconds=1.25,
                packet_cache_hits=0,
            ),
        )
        cache_meta = {"cache_hit": False, "cache_rows": 8, "cache_packet_hash_count": 8, "provenance": {"k": "v"}}
        return runtime, cache_meta

    monkeypatch.setattr(cli, "_build_runtime_with_cache", _runtime_with_cache)
    monkeypatch.setattr(cli, "AzureConfig", SimpleNamespace(from_env=lambda: SimpleNamespace(embedding_deployment="dep")))
    monkeypatch.setattr(cli, "AzureOpenAIEmbedder", lambda _cfg: object())
    monkeypatch.setattr(cli, "TiktokenTokenizer", lambda **kwargs: object())

    observed = {}

    def _run_v7(**kwargs):
        observed.update(kwargs)
        return {"output_dir": str(tmp_path), "summary": {"run_count": 750}, "report_path": str(tmp_path / "interactive_report.html")}

    monkeypatch.setattr(cli, "run_v7_experiment", _run_v7)
    monkeypatch.setattr(
        "sys.argv",
        ["run_sampling_v7.py", "--output", str(tmp_path)],
    )

    cli.main()

    assert "datasets" in observed
    assert observed["azure_config"] is not None
    assert observed["repetitions"] == 10
    assert observed["base_seed"] == 13
    assert observed["seed_values"] is None
    for dataset_payload in observed["datasets"].values():
        assert dataset_payload["full_vectors_by_unit"]
        assert set(dataset_payload["token_count_by_unit"].values()) == {321}
        assert dataset_payload["representation_source_by_unit"]
        assert "minhash_records_by_unit_id" in dataset_payload
        assert dataset_payload["runtime_ledger"]["embedding_input_tokens"] == 999


def test_run_sampling_v7_cli_seed_override_sets_repetitions_from_seeds(monkeypatch, tmp_path):
    import scripts.run_sampling_v7 as cli

    datasets = {
        "historical_300": _dataset("historical_300"),
        "dense_2500": _dataset("dense_2500"),
        "cosmos_otel": _dataset("cosmos_otel"),
    }
    monkeypatch.setattr("sampling_comparison.v7_dataset.load_v7_datasets", lambda **kwargs: datasets)
    monkeypatch.setattr(cli, "_build_runtime_with_cache", lambda data, **kwargs: (SimpleNamespace(
        embedding_vector_by_trace_id={int(uid.split("-")[-1]): np.asarray([1.0] * 1536, dtype=np.float32) for uid in data.unit_ids},
        token_cost_by_unit_id={uid: 1 for uid in data.unit_ids},
        minhash_records_by_unit_id={uid: MinHashRecord(content_sha256=uid, profile_id="p", signature=(1, 2), shingle_count=2, representation_truncated=False) for uid in data.unit_ids},
        ledger=SimpleNamespace(
            embedding_model_id="m",
            embedding_deployment_id="d",
            embedding_calls=1,
            embedding_inputs=1,
            embedding_input_tokens=1,
            embedding_latency_seconds=0.1,
            packet_cache_hits=0,
        ),
    ), {"cache_hit": False, "cache_rows": 0, "cache_packet_hash_count": 0, "provenance": {}}))
    monkeypatch.setattr(cli, "AzureConfig", SimpleNamespace(from_env=lambda: SimpleNamespace(embedding_deployment="dep")))
    monkeypatch.setattr(cli, "AzureOpenAIEmbedder", lambda _cfg: object())
    monkeypatch.setattr(cli, "TiktokenTokenizer", lambda **kwargs: object())

    observed = {}
    monkeypatch.setattr(cli, "run_v7_experiment", lambda **kwargs: observed.update(kwargs) or {"output_dir": str(tmp_path), "summary": {"run_count": 1}, "report_path": str(tmp_path / "interactive_report.html")})
    monkeypatch.setattr("sys.argv", ["run_sampling_v7.py", "--output", str(tmp_path), "--seeds", "21,34"])
    cli.main()
    assert observed["seed_values"] == (21, 34)


def test_run_sampling_v7_cli_rejects_conflicting_seed_contract(monkeypatch, tmp_path):
    import scripts.run_sampling_v7 as cli

    monkeypatch.setattr("sampling_comparison.v7_dataset.load_v7_datasets", lambda **kwargs: {})
    monkeypatch.setattr("sys.argv", ["run_sampling_v7.py", "--output", str(tmp_path), "--seeds", "21,34", "--repetitions", "5"])
    try:
        cli.main()
    except ValueError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("expected conflicting seed contract to raise")
