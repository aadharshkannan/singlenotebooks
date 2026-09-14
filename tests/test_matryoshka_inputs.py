from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sampling_comparison.matryoshka_experiment import load_input
from sampling_comparison.matryoshka_inputs import AzureBatchEmbedder, load_tau2_expected_dataset, prepare_live_input
from sampling_comparison.v7_dataset import V7Dataset
from scripts.inspect_matryoshka_tau2 import inspect
from scripts.prepare_matryoshka_live_input import read_settings
from trace_sampling.model import SessionEvent, Trace


def dataset(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"source":"test-only"}')
    ids = ("one", "two", "three")
    traces = {
        uid: Trace(
            trace_id=i, agent_id="tenant|agent", timestamp=float(i), signature=("test-tool",),
            span_count=1, duration_ms=1, status="ok", concept_id=-1,
            events=(SessionEvent(role="user", text=f"TEST session goal {uid}"),
                    SessionEvent(role="assistant", text="TEST response")),
        ) for i, uid in enumerate(ids)
    }
    return V7Dataset(
        dataset_id="cosmos_otel", ordered_unit_ids=ids, labels_by_unit=dict(zip(ids, (0, 1, 0))),
        traces_by_unit_id=traces, agent_id_by_unit={uid: "tenant|agent" for uid in ids},
        concept_key_by_unit={uid: uid for uid in ids}, use_case_id_by_unit={}, business_use_case_guid_by_unit={},
        metadata_by_unit={uid: {"expected_outcome": "MUST_NOT_EMBED_THIS_LABEL"} for uid in ids},
        representation_source_by_unit={uid: "TEST fixture" for uid in ids},
        representation_text_by_unit={}, source_paths={"test": str(source)},
    )


class FixtureEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return np.ones((len(texts), 1536), dtype=np.float32), 10 * len(texts)


def options():
    return dict(endpoint="https://test.invalid", deployment="text-embedding-3-small",
                tenant_id="test-tenant", label_source="TEST existing references only", batch_size=2)


def test_preflight_then_cached_live_batches_exclude_labels(tmp_path):
    data = dataset(tmp_path)
    out = tmp_path / "cache"
    preflight = prepare_live_input(data, out, embedder=None, **options())
    assert preflight["status"] == "prepared_without_embeddings"
    assert preflight["positive_count"] == 1
    assert not (out / "manifest.json").exists()
    embedder = FixtureEmbedder()
    result = prepare_live_input(data, out, embedder=embedder, **options())
    assert len(embedder.calls) == 2
    assert all("MUST_NOT_EMBED_THIS_LABEL" not in text for batch in embedder.calls for text in batch)
    assert result["live_embedding_calls"] == 2
    assert result["api_input_tokens"] == 30
    assert result["live_judge_calls"] == 0
    restored = load_input(out / "manifest.json")
    np.testing.assert_array_equal(restored.labels, [0, 1, 0])
    result = prepare_live_input(data, out, embedder=embedder, **options())
    assert len(embedder.calls) == 2
    assert result["fresh_embedding_batches_this_invocation"] == 0
    assert result["live_embedding_calls"] == 2


def test_resume_source_label_or_endpoint_mismatch_is_error(tmp_path):
    data = dataset(tmp_path)
    out = tmp_path / "cache"
    prepare_live_input(data, out, embedder=None, **options())
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_live_input(replace(data, labels_by_unit={"one": 1, "two": 1, "three": 0}), out, embedder=None, **options())
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_live_input(data, out, embedder=None, **{**options(), "endpoint": "https://another.invalid"})


def test_corrupt_batch_and_partial_response_fail(tmp_path):
    data = dataset(tmp_path)
    out = tmp_path / "cache"
    prepare_live_input(data, out, embedder=FixtureEmbedder(), **options())
    with (out / "batches" / "000000.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        prepare_live_input(data, out, embedder=FixtureEmbedder(), **options())


@pytest.mark.parametrize("model,indices,width", [
    ("text-embedding-3-large", [0, 1], 1536),
    ("text-embedding-3-small", [0, 0], 1536),
    ("text-embedding-3-small", [0, 1], 512),
])
def test_api_model_dimensions_and_order_are_verified(model, indices, width):
    embedder = AzureBatchEmbedder.__new__(AzureBatchEmbedder)
    embedder.deployment = "test-deployment"
    response = SimpleNamespace(
        model=model, data=[SimpleNamespace(index=i, embedding=[1.0] * width) for i in indices],
        usage=SimpleNamespace(total_tokens=10),
    )
    embedder.client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs: response))
    with pytest.raises(ValueError):
        embedder.embed(["a", "b"])


def test_failed_api_does_not_publish_success_manifest(tmp_path):
    class FailingEmbedder:
        def embed(self, texts):
            raise RuntimeError("authentication failed")
    with pytest.raises(RuntimeError, match="authentication failed"):
        prepare_live_input(dataset(tmp_path), tmp_path / "cache", embedder=FailingEmbedder(), **options())
    assert not (tmp_path / "cache" / "manifest.json").exists()


def test_tau2_inspection_keeps_observed_rewards_separate_from_expected_labels(tmp_path):
    import json

    source = tmp_path / "tau2.json"
    source.write_text(json.dumps({
        "tasks": [{"id": "task-a", "evaluation_criteria": {"reward_basis": ["DB", "NL_ASSERTION"]}}],
        "simulations": [
            {"id": "simulation-a", "task_id": "task-a", "messages": [],
             "reward_info": {"reward": 1.0, "reward_basis": ["DB", "NL_ASSERTION"]}},
            {"id": "simulation-b", "task_id": "task-a", "messages": [],
             "reward_info": {"reward": 0.0, "reward_basis": ["DB"]}},
        ],
    }))
    result = inspect(source)
    assert result["sessions"] == 2
    assert result["positive_count"] is None
    assert result["status"] == "blocked_expected_labels"
    assert result["tasks_with_nl_assertion_basis"] == ["task-a"]
    assert result["live_judge_calls"] == 0
    assert result["recognized_explicit_expected_label_fields"] == {}


def test_tau2_inspection_rejects_unmatched_tasks(tmp_path):
    import json

    source = tmp_path / "tau2.json"
    source.write_text(json.dumps({"tasks": [], "simulations": [{"id": "a", "task_id": "missing"}]}))
    with pytest.raises(ValueError, match="task IDs"):
        inspect(source)


def test_foundry_api_key_uses_existing_v1_client_factory(monkeypatch):
    import openai

    captured = {}
    client = SimpleNamespace(with_options=lambda **kwargs: client)

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return client

    def no_classic_client(**kwargs):
        pytest.fail("modern Foundry must not use the legacy AzureOpenAI endpoint path")

    monkeypatch.setattr(openai, "OpenAI", fake_openai)
    monkeypatch.setattr(openai, "AzureOpenAI", no_classic_client)
    embedder = AzureBatchEmbedder(
        endpoint="https://bugboss-foundry.services.ai.azure.com",
        deployment="text-embedding-3-small", api_key="test-only-not-a-real-key", api_version="2025-08-07",
    )
    assert embedder.client is client
    assert captured == {"base_url": "https://bugboss-foundry.services.ai.azure.com/openai/v1/",
                        "api_key": "test-only-not-a-real-key"}


def test_dotenv_is_explicit_and_process_values_take_precedence(tmp_path):
    env = tmp_path / ".env"
    env.write_text('AZURE_OPENAI_ENDPOINT="https://fixture.invalid"\nAZURE_OPENAI_API_KEY="test-only"\n')
    settings = read_settings(env, {"AZURE_OPENAI_ENDPOINT": "https://override.invalid"})
    assert settings["AZURE_OPENAI_ENDPOINT"] == "https://override.invalid"
    assert settings["AZURE_OPENAI_API_KEY"] == "test-only"
    child = tmp_path / "isolated"
    child.mkdir()
    assert not read_settings(child / ".env", {})["AZURE_OPENAI_API_KEY"]


def test_tau2_adapter_uses_explicit_labels_not_rewards_and_preserves_tools(tmp_path):
    import json
    from trace_sampling.session_embedding import TiktokenTokenizer
    from trace_sampling.token_representation import CanonicalizationOptions, TokenSessionEvidencePacketBuilder

    source = tmp_path / "simulations.json"
    expected = tmp_path / "expected.json"
    source.write_text(json.dumps({
        "tasks": [{"id": "task-a"}, {"id": "task-b"}],
        "simulations": [{
            "id": "simulation-one", "task_id": "task-b",
            "reward_info": {"reward": 1, "review": "REWARD_SENTINEL"},
            "messages": [
                {"role": "user", "content": "goal", "raw_data": "RAW_SENTINEL",
                 "tool_calls": [{"id": "call-a", "name": "lookup", "arguments": {"query": "x"}, "requestor": "user"}]},
                {"tool_messages": [{"role": "tool", "id": "call-a", "content": "result", "requestor": "user", "error": False}]},
                {"role": "assistant", "content": "finished"},
            ],
        }],
    }))
    expected.write_text('{"simulation-one": 0}')
    data = load_tau2_expected_dataset(source, expected)
    assert data.labels_by_unit == {"tau2_bench:simulation-one": 0}
    assert data.concept_key_by_unit == {"tau2_bench:simulation-one": "task:task-b"}
    trace = data.traces_by_unit_id["tau2_bench:simulation-one"]
    assert [event.role for event in trace.events] == ["user", "tool", "tool", "assistant"]
    assert trace.events[1].arguments["requestor"] == "user"
    assert trace.events[2].tool_name == "lookup"
    packet = TokenSessionEvidencePacketBuilder(
        options=CanonicalizationOptions(tokenizer=TiktokenTokenizer(model_name="text-embedding-3-small"), max_tokens=8191),
    ).build(trace).canonical_json
    assert "REWARD_SENTINEL" not in packet
    assert "RAW_SENTINEL" not in packet
    expected.write_text('{"a-different-simulation": 0}')
    with pytest.raises(ValueError, match="explicit expected label"):
        load_tau2_expected_dataset(source, expected)
