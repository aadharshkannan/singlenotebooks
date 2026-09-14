from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from sampling_comparison.matryoshka_experiment import canonical, load_input, normalized_prefix, sha256_file, write_json
from sampling_comparison.v7_dataset import V7Dataset
from trace_sampling.model import SessionEvent, Trace
from trace_sampling.session_embedding import TiktokenTokenizer
from trace_sampling.token_representation import CanonicalizationOptions, TokenSessionEvidencePacketBuilder


class BatchEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> tuple[np.ndarray, int]: ...


def load_tau2_expected_dataset(source: Path, expected_labels_path: Path) -> V7Dataset:
    """Require explicit simulation-ID labels; never use reward_info or invoke evaluators."""
    import ijson

    labels = json.loads(expected_labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict) or any(value not in (0, 1) for value in labels.values()):
        raise ValueError("Tau2 expected labels must map raw simulation IDs to binary 0/1 values")
    with source.open("rb") as stream:
        tasks = {str(task["id"]) for task in ijson.items(stream, "tasks.item")}
    traces = {}
    target_labels = {}
    concepts = {}
    metadata = {}
    seen = set()
    agent_id = "tau2_bench|single-model-run"

    def text(value: Any) -> str:
        return value if isinstance(value, str) else canonical(value)

    with source.open("rb") as stream:
        for ordinal, simulation in enumerate(ijson.items(stream, "simulations.item", use_float=True)):
            simulation_id = str(simulation["id"])
            if simulation_id in seen or simulation_id not in labels:
                raise ValueError("Tau2 simulation has duplicate ID or lacks an explicit expected label")
            seen.add(simulation_id)
            task_id = str(simulation["task_id"])
            if task_id not in tasks:
                raise ValueError("Tau2 simulation references a task absent from the embedded snapshot")
            events = []
            tools = []
            call_names = {}

            def append_message(message: Mapping[str, Any]) -> None:
                wrapped = message.get("tool_messages")
                if wrapped is not None:
                    if message.get("content") or message.get("tool_calls"):
                        raise ValueError("ambiguous Tau2 tool wrapper contains its own message content")
                    for child in wrapped:
                        append_message(child)
                    return
                role = message["role"]
                if role not in ("system", "user", "assistant", "tool"):
                    raise ValueError(f"unsupported Tau2 message role: {role}")
                content = message.get("content")
                if role == "tool":
                    call_id = str(message["id"])
                    if call_id not in call_names:
                        raise ValueError(f"Tau2 tool result has no preceding matching call: {call_id}")
                    events.append(SessionEvent(
                        role="tool", tool_name=call_names[call_id],
                        output=text({
                            "call_id": call_id, "content": content,
                            "requestor": message.get("requestor"), "error": message.get("error"),
                        }),
                    ))
                elif content is not None:
                    events.append(SessionEvent(role=role, text=text(content)))
                for call in message.get("tool_calls") or []:
                    call_id, name = str(call["id"]), str(call["name"])
                    if call_id in call_names:
                        raise ValueError("duplicate Tau2 tool call ID")
                    call_names[call_id] = name
                    tools.append(name)
                    events.append(SessionEvent(
                        role="tool", tool_name=name,
                        arguments={"call_id": call_id, "requestor": call.get("requestor"),
                                   "arguments": call.get("arguments")},
                    ))

            for message in simulation["messages"]:
                append_message(message)
            if not events:
                raise ValueError("Tau2 trajectory has no session evidence")
            uid = f"tau2_bench:{simulation_id}"
            traces[uid] = Trace(
                trace_id=ordinal, agent_id=agent_id, timestamp=float(ordinal),
                signature=tuple(tools) or ("no-tool",), span_count=max(1, len(tools)),
                duration_ms=0.0, status="ok", concept_id=-1, events=tuple(events),
            )
            target_labels[uid] = int(labels[simulation_id])
            concepts[uid] = f"task:{task_id}"
            metadata[uid] = {"task_id": task_id, "trial": simulation.get("trial"), "seed": simulation.get("seed")}
    if seen != set(labels):
        raise ValueError("Tau2 expected label IDs do not exactly match the simulation population")
    ids = tuple(sorted(traces))
    return V7Dataset(
        dataset_id="tau2_bench", ordered_unit_ids=ids, labels_by_unit=target_labels,
        traces_by_unit_id=traces, agent_id_by_unit={uid: agent_id for uid in ids},
        concept_key_by_unit=concepts, use_case_id_by_unit=concepts, business_use_case_guid_by_unit={},
        metadata_by_unit=metadata,
        representation_source_by_unit={uid: "tau2_recorded_messages" for uid in ids},
        representation_text_by_unit={},
        source_paths={"simulations": str(source), "expected_labels": str(expected_labels_path)},
    )


class AzureBatchEmbedder:
    def __init__(
        self, *, endpoint: str, deployment: str, tenant_id: str | None = None,
        api_key: str | None = None, api_version: str = "2024-02-01",
    ):
        from trace_sampling.azure_config import AzureConfig
        from trace_sampling.embedding import _is_modern_foundry_endpoint, _openai_base_url, build_openai_embedding_client

        if api_key or not tenant_id:
            config = AzureConfig(
                openai_endpoint=endpoint, openai_api_version=api_version,
                embedding_deployment=deployment, openai_api_key=api_key,
                search_endpoint="", search_index="",
            )
            client = build_openai_embedding_client(config)
        else:
            from azure.identity import AzureCliCredential, get_bearer_token_provider
            from openai import AzureOpenAI, OpenAI

            provider = get_bearer_token_provider(
                AzureCliCredential(tenant_id=tenant_id), "https://cognitiveservices.azure.com/.default",
            )
            client = (
                OpenAI(base_url=_openai_base_url(endpoint), api_key=provider)
                if _is_modern_foundry_endpoint(endpoint)
                else AzureOpenAI(azure_endpoint=endpoint, api_version=api_version, azure_ad_token_provider=provider)
            )
        self.client = client.with_options(timeout=120.0, max_retries=3)
        self.deployment = deployment

    def embed(self, texts: Sequence[str]) -> tuple[np.ndarray, int]:
        response = self.client.embeddings.create(
            model=self.deployment, input=list(texts), dimensions=1536, encoding_format="float",
        )
        if response.model != "text-embedding-3-small":
            raise ValueError(f"endpoint returned unexpected embedding model: {response.model}")
        rows = sorted(response.data, key=lambda row: row.index)
        if [row.index for row in rows] != list(range(len(texts))):
            raise ValueError("embedding response indices do not align with the requested batch")
        matrix = np.asarray([row.embedding for row in rows], dtype=np.float32)
        if matrix.shape != (len(texts), 1536):
            raise ValueError("API response must contain one full 1536-dimensional vector per input")
        normalized_prefix(matrix, 1536)
        return matrix, response.usage.total_tokens


def prepare_live_input(
    data: V7Dataset, output: Path, *, embedder: BatchEmbedder | None,
    endpoint: str, deployment: str, tenant_id: str | None, label_source: str,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Stage canonical packets offline, then resume only missing embedding batches."""
    if batch_size <= 0 or batch_size > 32:
        raise ValueError("batch_size must be between 1 and 32")
    source_hashes = {key: sha256_file(Path(path)) for key, path in data.source_paths.items()}
    tokenizer = TiktokenTokenizer(model_name="text-embedding-3-small", encoding_name="cl100k_base")
    builder = TokenSessionEvidencePacketBuilder(
        options=CanonicalizationOptions(tokenizer=tokenizer, max_tokens=8191),
        max_size=max(4096, len(data.ordered_unit_ids)),
    )
    records = []
    text_by_hash = {}
    for uid in data.ordered_unit_ids:
        trace = data.traces_by_unit_id[uid]
        packet = builder.build(trace)
        packet_hash = hashlib.sha256(packet.canonical_json.encode()).hexdigest()
        text_by_hash[packet_hash] = packet.canonical_json
        label = data.labels_by_unit[uid]
        if label not in (0, 1):
            raise ValueError(f"missing or non-binary task-completion reference for {uid}")
        records.append({
            "unit_id": uid, "agent_id": data.agent_id_by_unit[uid], "signature": list(trace.signature),
            "concept_key": data.concept_key_by_unit[uid], "label": label, "packet_hash": packet_hash,
            "original_tokens": packet.original_tokens, "emitted_tokens": packet.emitted_tokens,
            "truncated": packet.audit.truncated, "representation_source": data.representation_source_by_unit[uid],
        })
    if len(records) < 2 or len({row["unit_id"] for row in records}) != len(records):
        raise ValueError("live input requires unique IDs and at least two sessions")
    compatibility = {
        "version": "matryoshka-live-input-v1", "dataset_id": data.dataset_id,
        "source_hashes": source_hashes, "units_sha256": hashlib.sha256(canonical(records).encode()).hexdigest(),
        "endpoint": endpoint, "deployment": deployment, "tenant_id": tenant_id, "batch_size": batch_size,
        "embedding_model_id": "text-embedding-3-small", "embedding_dimensions": 1536,
        "label_source": label_source,
    }
    config_path = output / "preparation.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != compatibility:
            raise ValueError("preparation fingerprint differs; use a new input directory")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an input directory without matching preparation provenance")
    write_json(config_path, compatibility)
    write_json(output / "units.json", records)
    unique_hashes = sorted(text_by_hash)
    plan = {
        **compatibility, "status": "prepared_without_embeddings" if embedder is None else "embedding",
        "sessions": len(records), "agents": len(set(data.agent_id_by_unit.values())),
        "positive_count": sum(r["label"] for r in records),
        "unique_packets": len(unique_hashes),
        "input_tokens_before_api": sum(tokenizer.count(text_by_hash[h]) for h in unique_hashes),
        "truncated_sessions": sum(r["truncated"] for r in records),
        "representation_sources": dict(Counter(r["representation_source"] for r in records)),
        "source_paths": data.source_paths, "live_judge_calls": 0,
    }
    write_json(output / "readiness.json", plan)
    if embedder is None:
        return plan
    vectors_by_hash = {}
    total_tokens = 0
    total_batches = 0
    fresh_batches = 0
    for start in range(0, len(unique_hashes), batch_size):
        hashes = unique_hashes[start:start + batch_size]
        batch_path = output / "batches" / f"{start:06}.npz"
        ledger_path = batch_path.with_suffix(".json")
        if batch_path.exists() and ledger_path.exists():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            if sha256_file(batch_path) != ledger["sha256"]:
                raise ValueError(f"cached embedding batch checksum mismatch: {batch_path}")
            with np.load(batch_path, allow_pickle=False) as arrays:
                if arrays["hashes"].tolist() != hashes:
                    raise ValueError("cached batch does not match canonical session hashes")
                vectors = arrays["vectors"].copy()
            tokens = ledger["api_input_tokens"]
        elif batch_path.exists() or ledger_path.exists():
            raise RuntimeError(f"incomplete batch artifact needs explicit recovery: {batch_path}")
        else:
            vectors, tokens = embedder.embed([text_by_hash[h] for h in hashes])
            if vectors.shape != (len(hashes), 1536):
                raise ValueError("embedding batch shape mismatch")
            normalized_prefix(vectors, 1536)
            batch_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = batch_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, hashes=np.asarray(hashes), vectors=vectors)
            tmp.replace(batch_path)
            write_json(ledger_path, {
                "sha256": sha256_file(batch_path), "api_input_tokens": tokens,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            fresh_batches += 1
        normalized_prefix(vectors, 1536)
        total_tokens += tokens
        total_batches += 1
        vectors_by_hash.update(zip(hashes, vectors, strict=True))
        print(f"{data.dataset_id}: {min(start + batch_size, len(unique_hashes))}/{len(unique_hashes)} vectors available", flush=True)
    np.savez_compressed(
        output / "vectors.npz", unit_ids=np.asarray([r["unit_id"] for r in records]),
        vectors=np.asarray([vectors_by_hash[r["packet_hash"]] for r in records]),
    )
    manifest = {
        **plan, "version": "matryoshka-input-v1", "status": "complete",
        "representation_policy": "V3 weighted canonical full-session packet, cl100k_base, max 8191 tokens; prefixes taken only after full-vector API generation.",
        "concept_definition": "Dataset loader's cohort-relative concept key; not a validated business-use-case classifier.",
        "label_source": label_source, "excluded_sessions": 0,
        "files": {"units": "units.json", "vectors": "vectors.npz"},
        "hashes": {key: sha256_file(output / value) for key, value in (("units", "units.json"), ("vectors", "vectors.npz"))},
        "live_embedding_calls": total_batches, "fresh_embedding_batches_this_invocation": fresh_batches,
        "api_input_tokens": total_tokens, "live_judge_calls": 0,
    }
    write_json(output / "manifest.json", manifest)
    load_input(output / "manifest.json")
    write_json(output / "readiness.json", {**plan, "status": "complete"})
    return manifest
