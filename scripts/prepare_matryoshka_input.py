from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sampling_comparison.matryoshka_experiment import load_input, sha256_file, write_json
from sampling_comparison.v2_experiment import _concept_key, load_combined_dataset
from trace_sampling.session_embedding import TiktokenTokenizer
from trace_sampling.token_representation import CanonicalizationOptions, TokenSessionEvidencePacketBuilder


def prepare(dataset_id: str, cache: Path, output: Path) -> Path:
    """Bind retained V6 packet-hash vectors to current, source-hash-checked synthetic sessions."""
    if dataset_id not in ("historical_300", "dense_2500"):
        raise ValueError("this adapter accepts only retained V6 synthetic dataset caches")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("input output is nonempty; preserve it and use a fresh path")
    source_manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    provenance = source_manifest["provenance"]
    if source_manifest["dimensions"] != 1536 or provenance["embedding_model_id"] != "text-embedding-3-small":
        raise ValueError("source cache is not full text-embedding-3-small")
    data = load_combined_dataset()
    source_hashes = {key: sha256_file(Path(value)) for key, value in data.source_paths.items()}
    for key, value in source_hashes.items():
        if provenance["source_hashes"][key] != value:
            raise ValueError(f"{key} source hash does not match retained embeddings")
    with np.load(cache / "vectors.npz", allow_pickle=False) as arrays:
        hashes = arrays["hashes"].tolist()
        vectors = arrays["vectors"].copy()
    if vectors.shape != (len(hashes), 1536) or len(set(hashes)) != len(hashes):
        raise ValueError("source cache vectors/hash mapping is invalid")
    vector_by_hash = dict(zip(hashes, vectors, strict=True))
    tokenizer = TiktokenTokenizer(model_name="text-embedding-3-small", encoding_name="cl100k_base")
    builder = TokenSessionEvidencePacketBuilder(
        options=CanonicalizationOptions(tokenizer=tokenizer, max_tokens=8191), max_size=4096,
    )
    units = sorted(
        (unit for unit in data.units if data.corpus_id_by_unit[unit.unit_id] == dataset_id),
        key=lambda unit: (f"{unit.tenant_id}|{unit.agent_id}", unit.ended_at.isoformat() if unit.ended_at else "", unit.unit_id),
    )
    records = []
    matrix = []
    for unit in units:
        packet = builder.build(data.trace_by_unit_id[unit.unit_id])
        packet_hash = hashlib.sha256(packet.canonical_json.encode("utf-8")).hexdigest()
        if packet_hash not in vector_by_hash:
            raise ValueError(f"missing retained full-session embedding for {unit.unit_id}; no API fallback")
        matrix.append(vector_by_hash[packet_hash])
        records.append({
            "unit_id": unit.unit_id, "agent_id": f"{unit.tenant_id}|{unit.agent_id}",
            "signature": [call.name.strip() for call in unit.tool_calls if call.name and call.name.strip()] or ["no-tool"],
            "label": int(data.labels_by_unit[unit.unit_id]), "concept_key": _concept_key(data.metadata_by_unit[unit.unit_id]),
            "packet_hash": packet_hash, "original_tokens": packet.original_tokens,
            "emitted_tokens": packet.emitted_tokens, "truncated": packet.audit.truncated,
        })
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "units.json", records)
    np.savez_compressed(output / "vectors.npz", unit_ids=np.asarray([r["unit_id"] for r in records]), vectors=np.asarray(matrix))
    manifest = {
        "version": "matryoshka-input-v1", "dataset_id": dataset_id,
        "embedding_model_id": "text-embedding-3-small", "embedding_dimensions": 1536,
        "label_source": "Retained synthetic expected binary task-completion labels, not fresh LLM judgments.",
        "representation_policy": "V3 weighted canonical full-session packet, cl100k_base, max 8191 tokens; byte-for-byte packet SHA256 cache match.",
        "concept_definition": "V2 cohort-relative domain + task + difficulty key; not an external taxonomy guarantee.",
        "source_paths": data.source_paths, "source_hashes": source_hashes,
        "source_cache": {
            "path": str(cache), "manifest_sha256": sha256_file(cache / "manifest.json"),
            "vectors_sha256": sha256_file(cache / "vectors.npz"), "provenance": provenance,
        },
        "files": {"units": "units.json", "vectors": "vectors.npz"},
        "hashes": {key: sha256_file(output / value) for key, value in (("units", "units.json"), ("vectors", "vectors.npz"))},
        "sessions": len(records), "agent_counts": dict(Counter(r["agent_id"] for r in records)),
        "truncated_sessions": sum(r["truncated"] for r in records),
        "representation_fallbacks": 0, "excluded_sessions": 0,
        "token_summary": {
            "emitted_min": min(r["emitted_tokens"] for r in records),
            "emitted_median": float(np.median([r["emitted_tokens"] for r in records])),
            "emitted_max": max(r["emitted_tokens"] for r in records),
        },
        "live_embedding_calls": 0, "live_judge_calls": 0,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    load_input(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a verified, portable input using retained V6 embeddings only.")
    parser.add_argument("--dataset", required=True, choices=("historical_300", "dense_2500"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(prepare(args.dataset, Path(args.cache), Path(args.output)))


if __name__ == "__main__":
    main()
