"""Local IMDb provenance and resumable, explicitly enabled embedding preparation."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import tarfile
from typing import Any

import numpy as np
import tiktoken

from sampling_comparison.matryoshka_experiment import (
    canonical, normalized_prefix, sha256_file, write_json,
)
from sampling_comparison.matryoshka_inputs import BatchEmbedder


SOURCE_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
SOURCE_PAGE = "https://ai.stanford.edu/~amaas/data/sentiment/"
MODEL = "text-embedding-3-small"
VERSION = "imdb-input-v1"
POLICY = (
    "One whole review is one single-message session. Decode UTF-8, replace HTML "
    "br tags with newlines, unescape HTML entities, strip outer whitespace. "
    "Embed only review text, never sentiment, filename, rating or split. "
    "Keep the first 8191 cl100k_base tokens if necessary; no summarization. "
    "Generate native 1536 coordinates once; shorten only offline."
)
_REVIEW = re.compile(r"aclImdb/(train|test)/(pos|neg)/(\d+)_(\d+)\.txt")


def read_reviews(archive: Path, *, expected_count: int = 50_000) -> tuple[list[dict], dict[str, str]]:
    """Read only recognized regular members, without extracting untrusted paths."""
    encoding = tiktoken.get_encoding("cl100k_base")
    records: list[dict] = []
    texts: dict[str, str] = {}
    seen: set[str] = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            match = _REVIEW.fullmatch(member.name)
            if match is None:
                continue
            if not member.isfile() or member.size > 5_000_000 or member.name in seen:
                raise ValueError("invalid, oversized or duplicate labeled review member")
            seen.add(member.name)
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError("labeled review cannot be read")
            raw = stream.read().decode("utf-8")
            split, sentiment, _, rating_text = match.groups()
            rating = int(rating_text)
            label = int(sentiment == "pos")
            if not ((label == 1 and 7 <= rating <= 10) or (label == 0 and 1 <= rating <= 4)):
                raise ValueError("IMDb rating disagrees with sentiment directory")
            text = html.unescape(re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)).strip()
            tokens = encoding.encode(text, disallowed_special=())
            if not tokens:
                raise ValueError("empty review text")
            emitted = encoding.decode(tokens[:8191])
            # Re-encoding also guards a token boundary inside a Unicode character.
            while len(encoding.encode(emitted, disallowed_special=())) > 8191:
                emitted = emitted[:-1]
            packet_hash = hashlib.sha256(emitted.encode("utf-8")).hexdigest()
            texts[packet_hash] = emitted
            records.append({
                "unit_id": hashlib.sha256(member.name.encode()).hexdigest(),
                "split": split, "label": label, "packet_sha256": packet_hash,
                "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "original_tokens": len(tokens),
                "emitted_tokens": len(encoding.encode(emitted, disallowed_special=())),
                "truncated": len(tokens) > 8191,
            })
    records.sort(key=lambda row: row["unit_id"])
    if len(records) != expected_count:
        raise ValueError(f"expected {expected_count} labeled reviews, found {len(records)}")
    if expected_count == 50_000:
        counts = Counter((r["split"], r["label"]) for r in records)
        if counts != Counter({(split, label): 12_500 for split in ("train", "test") for label in (0, 1)}):
            raise ValueError("official 50K balanced train/test composition does not match")
    return records, texts


def _batches(hashes: list[str], tokens: dict[str, int], batch_size: int) -> list[list[str]]:
    result: list[list[str]] = []
    batch: list[str] = []
    total = 0
    for key in hashes:
        if batch and (len(batch) == batch_size or total + tokens[key] > 100_000):
            result.append(batch)
            batch, total = [], 0
        batch.append(key)
        total += tokens[key]
    if batch:
        result.append(batch)
    return result


def prepare_input(
    archive: Path, output: Path, *, embedder: BatchEmbedder | None = None,
    provider_identity: str | None = None, batch_size: int = 64, expected_count: int = 50_000,
) -> dict[str, Any]:
    if not 1 <= batch_size <= 128:
        raise ValueError("batch_size must be in [1,128]")
    if embedder is not None and not provider_identity:
        raise ValueError("live preparation requires a stable provider identity")
    records, texts = read_reviews(archive, expected_count=expected_count)
    fingerprint = {
        "version": VERSION, "source_sha256": sha256_file(archive),
        "units_sha256": hashlib.sha256(canonical(records).encode()).hexdigest(),
        "representation_policy": POLICY, "model": MODEL, "dimensions": 1536,
        "tokenizer": f"cl100k_base:{tiktoken.__version__}", "batch_size": batch_size,
    }
    config = output / "preparation.json"
    if config.exists():
        if json.loads(config.read_text(encoding="utf-8")) != fingerprint:
            raise ValueError("input preparation differs; use a new cache directory")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("input directory lacks preparation provenance")
    binding_path = output / "provider.json"
    binding = (
        {"provider_identity_sha256": hashlib.sha256(provider_identity.encode()).hexdigest()}
        if provider_identity else None
    )
    if binding is not None and binding_path.exists():
        if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
            raise ValueError("embedding provider changed; use a new cache directory")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        _, _, profile = load_input(manifest_path)
        return {**profile, "fresh_embedding_calls": 0}
    write_json(config, fingerprint)
    write_json(output / "units.json", records)
    positive = sum(row["label"] for row in records)
    lengths = np.asarray([row["original_tokens"] for row in records])
    labels_per_text: dict[str, set[int]] = {}
    for row in records:
        labels_per_text.setdefault(row["packet_sha256"], set()).add(row["label"])
    token_counts = {row["packet_sha256"]: row["emitted_tokens"] for row in records}
    profile = {
        **fingerprint, "dataset_id": "imdb_50000", "sessions": len(records), "agents": 1,
        "source_url": SOURCE_URL, "source_page": SOURCE_PAGE,
        "citation": "Maas et al. (2011), Learning Word Vectors for Sentiment Analysis, ACL.",
        "source_archive": str(archive),
        "source_choice": "Original Stanford IMDb corpus; missing dataset link in request inferred from 50K movie-review description.",
        "label_source": "Published review-rating sentiment: positive >=7/10 is 1, negative <=4/10 is 0. No LLM judge.",
        "split_policy": "Combine the 25K train and 25K test reviews as one agent; not a held-out benchmark test.",
        "positive_count": positive, "negative_count": len(records) - positive,
        "split_counts": dict(Counter(row["split"] for row in records)),
        "unique_texts": len(texts), "duplicate_text_rows": len(records) - len(texts),
        "conflicting_label_text_groups": sum(len(values) > 1 for values in labels_per_text.values()),
        "deduplication": "Retain every labeled source row; identical emitted text shares an embedding only.",
        "excluded_unlabeled_reviews": 50_000 if expected_count == 50_000 else None,
        "truncated_sessions": sum(row["truncated"] for row in records),
        "token_length": {
            "min": int(lengths.min()), "median": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, .95)), "max": int(lengths.max()),
        },
        "unique_embedding_input_tokens": sum(token_counts.values()),
        "live_judge_calls": 0, "status": "prepared_without_embeddings",
    }
    write_json(output / "readiness.json", profile)
    if embedder is None:
        return profile
    write_json(binding_path, binding)
    hashes = sorted(texts)
    vector_by_hash: dict[str, np.ndarray] = {}
    calls, fresh_calls, api_tokens = 0, 0, 0
    for index, keys in enumerate(_batches(hashes, token_counts, batch_size)):
        batch_path = output / "batches" / f"{index:06d}.npz"
        ledger_path = batch_path.with_suffix(".json")
        if batch_path.exists() and ledger_path.exists():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            if sha256_file(batch_path) != ledger["sha256"]:
                raise ValueError("embedding batch hash mismatch")
            with np.load(batch_path, allow_pickle=False) as batch:
                if batch["hashes"].tolist() != keys:
                    raise ValueError("embedding batch is not bound to these review texts")
                matrix = batch["vectors"].copy()
            tokens = ledger["api_input_tokens"]
        elif batch_path.exists() or ledger_path.exists():
            raise RuntimeError(f"incomplete embedding batch requires recovery: {batch_path}")
        else:
            matrix, tokens = embedder.embed([texts[key] for key in keys])
            matrix = np.asarray(matrix, dtype=np.float32)
            if matrix.shape != (len(keys), 1536):
                raise ValueError("embedding response must have native shape [batch,1536]")
            normalized_prefix(matrix, 1536)
            batch_path.parent.mkdir(parents=True, exist_ok=True)
            temp = batch_path.with_suffix(".tmp.npz")
            np.savez_compressed(temp, hashes=np.asarray(keys), vectors=matrix)
            temp.replace(batch_path)
            write_json(ledger_path, {
                "sha256": sha256_file(batch_path), "api_input_tokens": int(tokens),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            fresh_calls += 1
        if matrix.shape != (len(keys), 1536):
            raise ValueError("cached vector shape mismatch")
        normalized_prefix(matrix, 1536)
        vector_by_hash.update(zip(keys, matrix, strict=True))
        calls += 1
        api_tokens += tokens
        print(f"IMDb embeddings: {len(vector_by_hash)}/{len(hashes)} unique texts available", flush=True)
    vectors = np.stack([vector_by_hash[row["packet_sha256"]] for row in records])
    vector_path = output / "vectors.npy"
    temporary = output / "vectors.tmp.npy"
    np.save(temporary, vectors, allow_pickle=False)
    temporary.replace(vector_path)
    manifest = {
        **profile, "status": "complete", "embedding_calls": calls,
        "fresh_embedding_calls": fresh_calls, "api_input_tokens": int(api_tokens),
        "files": {"vectors": "vectors.npy", "units": "units.json"},
        "hashes": {"vectors": sha256_file(vector_path), "units": sha256_file(output / "units.json")},
    }
    write_json(manifest_path, manifest)
    load_input(manifest_path)
    write_json(output / "readiness.json", manifest)
    return manifest


def load_input(manifest_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    profile = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (profile["version"] != VERSION or profile["status"] != "complete"
            or profile["model"] != MODEL or profile["dimensions"] != 1536):
        raise ValueError("not a complete native IMDb embedding cache")
    paths = {}
    for name in ("vectors", "units"):
        paths[name] = manifest_path.parent / profile["files"][name]
        if sha256_file(paths[name]) != profile["hashes"][name]:
            raise ValueError(f"IMDb input {name} hash mismatch")
    units = json.loads(paths["units"].read_text(encoding="utf-8"))
    if hashlib.sha256(canonical(units).encode()).hexdigest() != profile["units_sha256"]:
        raise ValueError("IMDb units differ from prepared source metadata")
    vectors = np.load(paths["vectors"], allow_pickle=False, mmap_mode="r")
    labels = np.asarray([row["label"] for row in units])
    if vectors.shape != (len(labels), 1536) or len(labels) != profile["sessions"]:
        raise ValueError("IMDb vector/label row alignment mismatch")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("IMDb labels must be binary")
    labels = labels.astype(np.uint8)
    normalized_prefix(vectors, 1536)
    return vectors, labels, {**profile, "input_manifest_sha256": sha256_file(manifest_path)}
