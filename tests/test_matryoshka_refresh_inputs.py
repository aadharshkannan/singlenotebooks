from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sampling_comparison.matryoshka_refresh_inputs as refresh
from sampling_comparison.matryoshka_experiment import canonical, load_input, sha256_file, write_json
from sampling_comparison.matryoshka_inputs import AzureBatchEmbedder, prepare_live_input
from sampling_comparison.v7_dataset import V7Dataset
from scripts.prepare_matryoshka_refresh_input import main
from trace_sampling.model import SessionEvent, Trace
from trace_sampling.session_embedding import TiktokenTokenizer
from trace_sampling.token_representation import CanonicalizationOptions, TokenSessionEvidencePacketBuilder


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    @staticmethod
    def vector(text):
        seed = np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8).astype(np.float32) + 1
        return np.tile(seed, 48)

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.asarray([self.vector(text) for text in texts]), 10 * len(texts)


def packets(data):
    builder = TokenSessionEvidencePacketBuilder(
        options=CanonicalizationOptions(
            tokenizer=TiktokenTokenizer(model_name=refresh.MODEL, encoding_name="cl100k_base"), max_tokens=8191,
        ), max_size=max(4096, len(data.ordered_unit_ids)),
    )
    return {uid: builder.build(data.traces_by_unit_id[uid]).canonical_json for uid in data.ordered_unit_ids}


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    rows = []
    for name in sorted(refresh.CONTAINERS):
        path = root / f"genesis_observability__{name}.jsonl"
        path.write_text('{"fixture":true}\n', encoding="utf-8")
        rows.append({"container": name, **refresh._file_state(path, count_lines=True)})
    for name in ("manifest.json", "refresh_report.json"):
        write_json(root / name, {"fixture": True})
    ids = tuple(f"SENSITIVE_UNIT_{i}" for i in range(7))
    traces = {
        uid: Trace(
            trace_id=i, agent_id="SENSITIVE_AGENT", timestamp=float(i), signature=("test-tool",),
            span_count=1, duration_ms=1, status="ok", concept_id=-1,
            events=(SessionEvent(role="user", text=f"PRIVATE_PAYLOAD test goal {i}"),
                    SessionEvent(role="assistant", text="test response")),
        ) for i, uid in enumerate(ids)
    }
    data = V7Dataset(
        dataset_id="cosmos_otel", ordered_unit_ids=ids,
        labels_by_unit={uid: int(i % 2 == 0) for i, uid in enumerate(ids)},
        traces_by_unit_id=traces, agent_id_by_unit={uid: "SENSITIVE_AGENT" for uid in ids},
        concept_key_by_unit={uid: "SENSITIVE_TASK" for uid in ids},
        use_case_id_by_unit={}, business_use_case_guid_by_unit={},
        metadata_by_unit={uid: {"expected_outcome": "good" if i % 2 == 0 else "bad"} for i, uid in enumerate(ids)},
        representation_source_by_unit={uid: "linked_raw_spans" for uid in ids},
        representation_text_by_unit={},
        source_paths={name: str(root / f"genesis_observability__{name}.jsonl") for name in ("labels", "spans")},
    )
    text = packets(data)
    sorted_ids = sorted(ids, key=lambda uid: hashlib.sha256(text[uid].encode()).hexdigest())
    # Fully cached first batch; remaining batches exercise API-only rows.
    old_ids = tuple(sorted_ids[:2])
    old = replace(
        data, ordered_unit_ids=old_ids,
        # Label changes must not invalidate identical embedding input.
        labels_by_unit={uid: 1 - data.labels_by_unit[uid] for uid in old_ids},
        agent_id_by_unit={uid: data.agent_id_by_unit[uid] for uid in old_ids},
    )
    old_out = tmp_path / "old"
    prepare_live_input(
        old, old_out, embedder=FakeEmbedder(), endpoint=refresh.ENDPOINT,
        deployment=refresh.MODEL, tenant_id=None, label_source=refresh.LABEL_SOURCE, batch_size=2,
    )
    preflight = {
        "containers": rows, "required_model": refresh.MODEL, "required_dimensions": 1536,
        "snapshot_bytes": sum(r["bytes"] for r in rows), "snapshot_documents": 10,
        "metadata_sha256": {name: sha256_file(root / name) for name in ("manifest.json", "refresh_report.json")},
        "eligible_units": 7, "agents": 1, "unique_canonical_packets": 7, "cached_packet_matches": 2,
        "missing_unique_native_embeddings": 5, "outcome_counts": {"good": 4, "bad": 3},
        "truncated_packets": 0, "representation_sources": {"linked_raw_spans": 7},
        "snapshot_cutoff_utc": "2026-09-15T13:40:42+00:00",
        "snapshot_export_completed_at": "2026-09-15T13:41:33+00:00", "point_in_time_snapshot": False,
    }
    preflight_path = tmp_path / "preflight.json"
    write_json(preflight_path, preflight)
    return SimpleNamespace(
        root=root, data=data, preflight=preflight, preflight_path=preflight_path,
        reuse=old_out / "manifest.json", output=tmp_path / "refresh", text=text,
        old_ids=old_ids, sorted_ids=sorted_ids,
    )


def run(case, embedder=None, **kwargs):
    return refresh.prepare_refresh_input(
        case.root, case.output, preflight_path=case.preflight_path, reuse_manifest=case.reuse,
        embedder=embedder, loader=lambda root, partial_label: case.data, batch_size=2, **kwargs,
    )


def tree_hashes(path):
    return {str(p.relative_to(path)): sha256_file(p) for p in path.rglob("*") if p.is_file()}


def test_exact_reuse_label_changes_order_native_vectors_and_usage(case):
    before = tree_hashes(case.reuse.parent), tree_hashes(case.root)
    api = FakeEmbedder()
    result = run(case, api)
    assert result["stored_embedding_batches"] == 4
    assert result["logical_requests"] == result["live_embedding_calls"] == len(api.calls) == 3
    assert result["reused_vectors"] == 2
    assert result["requested_vectors"] == result["fresh_requested_vectors_this_invocation"] == 5
    assert result["api_input_tokens"] == 50
    assert result["http_attempts"] == result["http_successes"] == 3
    assert result["resumed_embedding_batches_this_invocation"] == 0
    assert result["live_judge_calls"] == 0
    requested = [text for batch in api.calls for text in batch]
    assert all(case.text[uid] not in requested for uid in case.old_ids)
    data = load_input(case.output / "manifest.json")
    assert data.unit_ids == case.data.ordered_unit_ids
    np.testing.assert_array_equal(data.labels, list(case.data.labels_by_unit.values()))
    for uid, vector in zip(data.unit_ids, data.vectors, strict=True):
        assert vector.tobytes() == FakeEmbedder.vector(case.text[uid]).tobytes()
    assert before == (tree_hashes(case.reuse.parent), tree_hashes(case.root))
    # Exact parity with the existing canonical preparation, not a new representation.
    comparison = case.output.parent / "reference"
    prepare_live_input(
        case.data, comparison, embedder=None, endpoint=refresh.ENDPOINT, deployment=refresh.MODEL,
        tenant_id=None, label_source=refresh.LABEL_SOURCE, batch_size=2,
    )
    assert (comparison / "units.json").read_bytes() == (case.output / "units.json").read_bytes()


def test_partial_cache_batches_restore_original_order(case):
    # Keep one row from each of the first two sorted batches.
    units = json.loads((case.reuse.parent / "units.json").read_text())
    new_uid = case.sorted_ids[2]
    units[1]["unit_id"] = new_uid
    units[1]["packet_hash"] = hashlib.sha256(case.text[new_uid].encode()).hexdigest()
    write_json(case.reuse.parent / "units.json", units)
    np.savez_compressed(
        case.reuse.parent / "vectors.npz", unit_ids=np.asarray([r["unit_id"] for r in units]),
        vectors=np.asarray([FakeEmbedder.vector(case.text[r["unit_id"]]) for r in units]),
    )
    manifest = json.loads(case.reuse.read_text())
    manifest["hashes"] = {k: sha256_file(case.reuse.parent / v) for k, v in manifest["files"].items()}
    write_json(case.reuse, manifest)
    api = FakeEmbedder()
    result = run(case, api)
    assert [len(batch) for batch in api.calls] == [1, 1, 2, 1]
    assert result["requested_vectors"] == 5
    restored = load_input(case.output / "manifest.json")
    for uid, vector in zip(restored.unit_ids, restored.vectors, strict=True):
        assert vector.tobytes() == FakeEmbedder.vector(case.text[uid]).tobytes()


def test_offline_preflight_completed_reentry_is_byte_immutable(case):
    result = run(case)
    assert result["status"] == "prepared_without_embeddings"
    assert not (case.output / "manifest.json").exists()
    assert not (case.output / "batches").exists()
    api = FakeEmbedder()
    run(case, api)
    before = tree_hashes(case.output)
    calls = len(api.calls)
    # Completed reentry must not even rebuild canonical packets.
    case.data = None
    result = run(case, api)
    assert len(api.calls) == calls
    assert result["fresh_embedding_batches_this_invocation"] == 0
    assert result["fresh_requested_vectors_this_invocation"] == 0
    assert result["fresh_api_input_tokens_this_invocation"] == 0
    assert result["resumed_requested_vectors_this_invocation"] == 5
    assert before == tree_hashes(case.output)


@pytest.mark.parametrize("artifact", ["manifest.json", "units.json", "vectors.npz"])
def test_reuse_sha_tamper_rejected_before_api(case, artifact):
    run(case)
    path = case.reuse.parent / artifact
    path.write_bytes(path.read_bytes() + b" ")
    api = FakeEmbedder()
    with pytest.raises(ValueError):
        run(case, api)
    assert api.calls == []


def test_valid_but_changed_reuse_manifest_rejected_on_resume(case):
    run(case)
    manifest = json.loads(case.reuse.read_text())
    manifest["additional_provenance"] = "different"
    write_json(case.reuse, manifest)
    api = FakeEmbedder()
    with pytest.raises(refresh.RefreshError, match="fingerprint"):
        run(case, api)
    assert api.calls == []


@pytest.mark.parametrize("field,value", [
    ("embedding_model_id", "wrong-model"), ("embedding_dimensions", 512), ("embedding_model_id", None),
])
def test_wrong_or_missing_reuse_model_dimensions(case, field, value):
    manifest = json.loads(case.reuse.read_text())
    if value is None:
        manifest.pop(field)
    else:
        manifest[field] = value
    write_json(case.reuse, manifest)
    api = FakeEmbedder()
    with pytest.raises((ValueError, KeyError)):
        run(case, api)
    assert not api.calls


@pytest.mark.parametrize("name", ["genesis_observability__spans.jsonl", "manifest.json"])
def test_snapshot_tamper_blocks_before_loading_or_api(case, name):
    run(case)
    path = case.root / name
    path.write_bytes(path.read_bytes() + b" ")
    api = FakeEmbedder()
    with pytest.raises(refresh.RefreshError, match="mismatch"):
        run(case, api)
    assert api.calls == []


def test_source_changes_during_preparation_block_api(case):
    api = FakeEmbedder()

    def loader(*args, **kwargs):
        path = case.root / "genesis_observability__labels.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
        return case.data

    with pytest.raises(refresh.RefreshError, match="mismatch"):
        refresh.prepare_refresh_input(
            case.root, case.output, preflight_path=case.preflight_path, reuse_manifest=case.reuse,
            embedder=api, loader=loader, batch_size=2,
        )
    assert api.calls == []


def test_source_changes_during_api_never_publish_success(case):
    class Mutating(FakeEmbedder):
        def embed(self, texts):
            path = case.root / "genesis_observability__labels.jsonl"
            path.write_bytes(path.read_bytes() + b" ")
            return super().embed(texts)

    with pytest.raises(refresh.RefreshError, match="changed|mismatch"):
        run(case, Mutating())
    assert not (case.output / "manifest.json").exists()


def test_empty_population_and_nonempty_unbound_output_block_cloud(case):
    case.data = replace(case.data, ordered_unit_ids=())
    api = FakeEmbedder()
    with pytest.raises(refresh.RefreshError, match="two unique"):
        run(case, api)
    assert api.calls == []
    case.output.mkdir()
    (case.output / "unbound").write_text("fixture")
    with pytest.raises(refresh.RefreshError, match="Nonempty"):
        run(case, api)


def test_completed_batch_tamper_and_wrong_order(case):
    run(case, FakeEmbedder())
    path = case.output / "batches" / "000000.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    api = FakeEmbedder()
    with pytest.raises(refresh.RefreshError, match="checksum"):
        run(case, api)
    assert not api.calls


@pytest.mark.parametrize("failure", ["exception", "dimensions", "tokens"])
def test_api_failure_is_sanitized_never_publishes_or_silently_resends(case, failure, capsys):
    class Failing:
        def embed(self, texts):
            if failure == "exception":
                raise RuntimeError("SENSITIVE_CREDENTIAL PRIVATE_PAYLOAD SENSITIVE_UNIT")
            if failure == "dimensions":
                return np.ones((len(texts), 512), dtype=np.float32), 10
            return np.ones((len(texts), 1536), dtype=np.float32), None

    with pytest.raises(refresh.RefreshError, match="explicit recovery") as error:
        run(case, Failing())
    assert "SENSITIVE" not in str(error.value)
    assert not (case.output / "manifest.json").exists()
    api = FakeEmbedder()
    with pytest.raises(refresh.RefreshError, match="explicit recovery"):
        run(case, api)
    assert api.calls == []
    assert "PRIVATE_PAYLOAD" not in capsys.readouterr().out
    for path in (case.output / "batches").glob("*.json"):
        assert "SENSITIVE" not in path.read_text()


def test_resume_successful_batches_fresh_vs_retained(case):
    def interrupt(message):
        if message.startswith("Vectors available: 4/"):
            raise InterruptedError("test interruption AFTER durable success")

    first = FakeEmbedder()
    with pytest.raises(InterruptedError):
        run(case, first, progress=interrupt)
    assert len(first.calls) == 1
    retained = tree_hashes(case.output / "batches")
    second = FakeEmbedder()
    result = run(case, second)
    assert len(second.calls) == 2
    assert result["api_input_tokens"] == 50
    assert result["resumed_embedding_batches_this_invocation"] == 2
    assert result["fresh_embedding_batches_this_invocation"] == 2
    assert result["resumed_requested_vectors_this_invocation"] == 2
    assert result["fresh_requested_vectors_this_invocation"] == 3
    assert result["fresh_api_input_tokens_this_invocation"] == 30
    for name, digest in retained.items():
        assert sha256_file(case.output / "batches" / name) == digest


def test_duplicate_new_canonical_packet_is_requested_once(case):
    uid = "SENSITIVE_DUPLICATE"
    original = case.sorted_ids[-1]
    data = case.data
    case.data = replace(
        data, ordered_unit_ids=(*data.ordered_unit_ids, uid),
        traces_by_unit_id={**data.traces_by_unit_id, uid: data.traces_by_unit_id[original]},
        labels_by_unit={**data.labels_by_unit, uid: 1},
        agent_id_by_unit={**data.agent_id_by_unit, uid: "SENSITIVE_AGENT"},
        concept_key_by_unit={**data.concept_key_by_unit, uid: "SENSITIVE_TASK"},
        metadata_by_unit={**data.metadata_by_unit, uid: {"expected_outcome": "good"}},
        representation_source_by_unit={**data.representation_source_by_unit, uid: "linked_raw_spans"},
    )
    case.preflight.update(
        eligible_units=8, outcome_counts={"good": 5, "bad": 3},
        representation_sources={"linked_raw_spans": 8},
    )
    write_json(case.preflight_path, case.preflight)
    api = FakeEmbedder()
    result = run(case, api)
    assert result["sessions"] == 8
    assert result["unique_packets"] == 7
    assert sum(len(batch) for batch in api.calls) == 5
    restored = load_input(case.output / "manifest.json")
    assert restored.vectors[-1].tobytes() == restored.vectors[restored.unit_ids.index(original)].tobytes()


def test_later_batch_tamper_blocks_before_any_earlier_missing_api(case):
    run(case, FakeEmbedder())
    (case.output / "manifest.json").unlink()
    for suffix in (".npz", ".json"):
        (case.output / "batches" / ("000002" + suffix)).unlink()
    path = case.output / "batches" / "000004.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    api = FakeEmbedder()
    with pytest.raises(refresh.RefreshError, match="checksum"):
        run(case, api)
    assert api.calls == []


def test_safe_summary_excludes_sensitive_metadata(case, capsys):
    result = run(case, FakeEmbedder(), progress=print)
    result["credentials"] = "SENSITIVE_CREDENTIAL"
    result["representation_sources"]["SENSITIVE_AGENT"] = 1
    summary = canonical(refresh.safe_summary(result))
    output = capsys.readouterr().out
    assert all(s not in summary + output for s in (
        "PRIVATE_PAYLOAD", "SENSITIVE_UNIT", "SENSITIVE_AGENT", "SENSITIVE_TASK", "SENSITIVE_CREDENTIAL",
        "packet_hash", "source_paths", "reuse_source",
    ))


@pytest.mark.parametrize("endpoint,model", [
    ("https://another.invalid", refresh.MODEL), (refresh.ENDPOINT, "text-embedding-3-large"),
])
def test_target_guard_blocks_cloud(case, endpoint, model):
    api = FakeEmbedder()
    with pytest.raises(refresh.RefreshError, match="authorized"):
        run(case, api, endpoint=endpoint, deployment=model)
    assert not api.calls


def test_http_meter_counts_retries_and_blocks_other_targets():
    meter = refresh.MeteredAzureEmbedder.__new__(refresh.MeteredAzureEmbedder)
    meter.http_attempts = meter.http_successes = 0
    for status in (429, 200):
        meter._request(SimpleNamespace(url=refresh.ENDPOINT + "/openai/v1/embeddings"))
        meter._response(SimpleNamespace(status_code=status))
    assert (meter.http_attempts, meter.http_successes) == (2, 1)
    with pytest.raises(refresh.RefreshError, match="unauthorized"):
        meter._request(SimpleNamespace(url="https://another.invalid/embeddings"))
    assert meter.http_attempts == 2


@pytest.mark.parametrize("model,width", [("wrong-model", 1536), (refresh.MODEL, 512)])
def test_existing_azure_response_validation_remains_active(model, width):
    api = AzureBatchEmbedder.__new__(AzureBatchEmbedder)
    api.deployment = refresh.MODEL
    response = SimpleNamespace(
        model=model, data=[SimpleNamespace(index=0, embedding=[1.0] * width)],
        usage=SimpleNamespace(total_tokens=10),
    )
    api.client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs: response))
    with pytest.raises(ValueError):
        api.embed(["test-only"])


def test_cli_live_requires_explicit_env_and_offline_does_not_construct_client(case, monkeypatch, capsys):
    monkeypatch.setattr(
        "scripts.prepare_matryoshka_refresh_input.MeteredAzureEmbedder",
        lambda **kwargs: pytest.fail("Must not initialize cloud without --live"),
    )
    args = [
        "--cosmos-root", str(case.root), "--preflight", str(case.preflight_path),
        "--reuse-manifest", str(case.reuse), "--output", str(case.output),
    ]
    assert main(args) == 1  # External/nonignored output rejected before cloud.
    with pytest.raises(SystemExit):
        main([*args, "--live"])
    assert "SENSITIVE" not in capsys.readouterr().err
