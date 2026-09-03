from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from random_sampling.models import EvaluationUnit
from sampling_comparison.v2_experiment import CombinedDataset, load_combined_dataset
from sampling_comparison.v3_experiment import (
    V3_DEFAULT_EMBEDDING_BATCH_SIZE,
    V3_DEFAULT_EMBEDDING_TAU,
    V3_EMBEDDING_DIMENSIONS,
    V3_EMBEDDING_ENCODING,
    V3_EMBEDDING_MODEL,
    V3_MAX_SESSION_PACKET_TOKENS,
    V3EmbeddingSelector,
    V3Runtime,
    build_v3_runtime,
)
from sampling_comparison.v4_idw import IDWConfig
from sampling_comparison.v6_business_use_case import (
    LOW_CONFIDENCE_FALLBACK_GUID,
    MAX_INPUT_TOKENS,
    BusinessUseCaseClassifier,
    SessionSelection,
    SessionStepText,
    load_business_use_case_artifacts,
)
from sampling_comparison.v6_experiment import (
    METHOD_IDS,
    METHOD_ID_ORDER,
    NOMINAL_TOKENS_PER_SESSION,
    SAMPLE_CAPS,
    TRIAL_SEEDS,
    SelectionOutcome,
    SelectionRecord,
    SessionDescriptor,
    TrialMetrics,
    build_session_descriptors,
    compute_trial_metrics,
    run_arm2_5_binary_from_arm2_result,
    run_arm2_idw,
    select_arm1,
    select_arm3,
    select_arm4,
    select_arm5,
    select_arm6,
    validate_selection_exactness,
)
from trace_sampling.azure_config import AzureConfig
from trace_sampling.embedding import AzureOpenAIEmbedder
from trace_sampling.model import Trace
from trace_sampling.samplers import AdaptiveSampler, SamplerConfig
from trace_sampling.session_embedding import TiktokenTokenizer
from trace_sampling.token_representation import CanonicalizationOptions, TokenSessionEvidencePacketBuilder
from trace_sampling.vector_store import AzureSearchVectorStore, VectorStore


DEFAULT_MAVEN_ROOT = Path(
    r"C:\Users\stangoodwin\mvn-mavenservice\src\MVN\Kairo\MachineLearning\BusinessUseCase\data"
)
DEFAULT_MAVEN_CENTROIDS_DB = str(DEFAULT_MAVEN_ROOT / "centroids_v6.db")
DEFAULT_MAVEN_TAXONOMY_DB = str(DEFAULT_MAVEN_ROOT / "taxonomy_v6.db")
V6_BUNDLE_VERSION = "sampling-v6-bundle-v2"
V6_MANIFEST_VERSION = "sampling-v6-manifest-v2"
V6_CHECKPOINT_VERSION = "sampling-v6-checkpoint-v2"
V6_AGENT_METRICS_VERSION = "sampling-v6-agent-metrics-v1"
UNDETERMINED_SENTINEL = "undetermined:none"

_SELECTION_CONTROLLING_CODE_PATHS: tuple[Path, ...] = (
    Path("sampling_comparison") / "v6_experiment.py",
    Path("sampling_comparison") / "v6_business_use_case.py",
    Path("sampling_comparison") / "v3_experiment.py",
    Path("trace_sampling") / "cluster_index.py",
    Path("trace_sampling") / "vector_store.py",
    Path("trace_sampling") / "samplers.py",
    Path("trace_sampling") / "reservoir.py",
)

_PUBLISHER_CODE_PATHS: tuple[Path, ...] = (
    Path("sampling_comparison") / "v6_runner.py",
    Path("scripts") / "run_sampling_v6.py",
    *_SELECTION_CONTROLLING_CODE_PATHS,
)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _iso_now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(_canonical_json(dict(payload)) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + "\n")
    os.replace(tmp, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


ProgressCallback = Callable[[Mapping[str, Any]], None]


def _safe_progress_error(exc: BaseException) -> tuple[str, str]:
    kind = type(exc).__name__
    message = str(exc).strip() or "unknown error"
    return kind, message


def _progress_event(
    *,
    version: str,
    status: str,
    phase: str,
    message: str,
    current_seed: Any | None,
    current_cap: Any | None,
    current_method: Any | None,
    completed_replays: int,
    total_replays: int,
    completed_cells: int,
    total_cells: int,
    replay_session_current: int = 0,
    replay_session_total: int = 0,
    percent: float = 0.0,
    error_type: str | None = None,
    error_message: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": version,
        "status": status,
        "phase": phase,
        "message": message,
        "current_seed": current_seed,
        "current_cap": current_cap,
        "current_method": current_method,
        "completed_replays": int(completed_replays),
        "total_replays": int(total_replays),
        "completed_cells": int(completed_cells),
        "total_cells": int(total_cells),
        "replay_session_current": int(replay_session_current),
        "replay_session_total": int(replay_session_total),
        "percent": float(percent),
        "elapsed_seconds": 0.0,
        "updated_at": _iso_now_utc(),
    }
    if error_type is not None:
        payload["error_type"] = error_type
    if error_message is not None:
        payload["error_message"] = error_message
    if extra:
        for key, value in dict(extra).items():
            payload[key] = value
    return payload


def _global_progress_percent(*, phase: str, current_replay: int | None, total_replays: int, replay_session_current: int | None, replay_session_total: int | None, completed_cells: int, total_cells: int, previous_percent: float | None = None) -> float:
    phase_key = str(phase or "progress").lower()
    if phase_key in {"complete"}:
        return 100.0
    if phase_key in {"pre-cleanup", "replay-setup"}:
        replay = max(1, int(current_replay or 1))
        base = ((replay - 1) / max(1, int(total_replays))) * 100.0
        if previous_percent is not None:
            return max(float(previous_percent), base)
        return base
    if phase_key.startswith("replay") or phase_key == "search-replay":
        replay = max(1, int(current_replay or 1))
        current = int(replay_session_current or 0)
        total = max(1, int(replay_session_total or 1))
        replay_fraction = (current / total) if total > 0 else 0.0
        base = (((replay - 1) + replay_fraction) / max(1, int(total_replays))) * 100.0
        if previous_percent is not None:
            return max(float(previous_percent), base)
        return base
    if phase_key == "post-cleanup":
        replay = max(1, int(current_replay or 1))
        base = (replay / max(1, int(total_replays))) * 100.0
        if previous_percent is not None:
            return max(float(previous_percent), base)
        return base
    if phase_key == "method-evaluation":
        completed = max(0, int(completed_cells or 0))
        total = max(1, int(total_cells or 1))
        base = (completed / total) * 100.0
        if previous_percent is not None:
            return max(float(previous_percent), base)
        return base
    if phase_key == "failed":
        return float(previous_percent if previous_percent is not None else 0.0)
    return float(previous_percent if previous_percent is not None else 0.0)


def _emit_progress_event(
    *,
    output_dir: Path | None,
    progress_callback: ProgressCallback | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    event = dict(payload)
    event["updated_at"] = _iso_now_utc()
    if output_dir is not None:
        path = Path(output_dir) / "progress.json"
        last_error: OSError | None = None
        for attempt in range(5):
            try:
                _write_json_atomic(path, event)
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                if attempt < 4:
                    time.sleep(0.05 * (attempt + 1))
        # Progress is observational. A transient file lock must never invalidate
        # a completed experiment cell; the next event will retry the snapshot.
        if last_error is not None:
            event["progress_write_error"] = type(last_error).__name__
    if progress_callback is not None:
        try:
            progress_callback(dict(event))
        except Exception:
            pass
    return dict(event)


def _artifact_meta(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _stable_hash(seed: int, *parts: Any) -> str:
    data = "|".join([str(seed)] + [str(p) for p in parts])
    return _sha256_text(data)


def _stable_order(unit_ids: Sequence[str], *, seed: int, token: str) -> tuple[str, ...]:
    ranked = [(_stable_hash(seed, token, unit_id), str(unit_id)) for unit_id in unit_ids]
    ranked.sort(key=lambda row: (row[0], row[1]))
    return tuple(uid for _, uid in ranked)


def _to_trace_with_timestamp(base: Trace, idx: int) -> Trace:
    return Trace(
        trace_id=base.trace_id,
        agent_id=base.agent_id,
        timestamp=float(idx),
        signature=base.signature,
        span_count=base.span_count,
        duration_ms=base.duration_ms,
        status=base.status,
        concept_id=base.concept_id,
        events=base.events,
    )


def _scrub_endpoint(endpoint: str) -> str:
    clean = str(endpoint or "").strip()
    if "?" in clean:
        clean = clean.split("?", 1)[0]
    return clean


def _basename_or_empty(path_like: Any) -> str:
    value = str(path_like or "").strip()
    if not value:
        return ""
    return Path(value).name


def _safe_preview(text: str | None, *, max_len: int = 120) -> str:
    value = (text or "").replace("\r", " ").replace("\n", " ").strip()
    if len(value) <= max_len:
        return value
    return value[:max_len] + "..."


def _serialize_selection_record(record: SelectionRecord) -> dict[str, Any]:
    return {
        "unit_id": record.unit_id,
        "method_id": record.method_id,
        "stratum": record.stratum,
        "inclusion_probability": record.inclusion_probability,
        "weight": record.weight,
        "reason": record.reason,
    }


def _sanitize_for_run_row(metrics: TrialMetrics) -> dict[str, Any]:
    payload = asdict(metrics)
    payload.pop("selected_ids", None)
    return payload


def _strip_label_fields_from_run_row(row: Mapping[str, Any]) -> dict[str, Any]:
    def _clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            cleaned: dict[str, Any] = {}
            for key, raw in value.items():
                if "label" in str(key).lower():
                    continue
                cleaned[str(key)] = _clean(raw)
            return cleaned
        if isinstance(value, list):
            return [_clean(item) for item in value]
        if isinstance(value, tuple):
            return [_clean(item) for item in value]
        return value

    return dict(_clean(dict(row)))


def _sanitize_idw_validation(value: Any | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        source = dict(value)
    elif is_dataclass(value):
        source = dict(asdict(value))
    elif hasattr(value, "__dict__"):
        source = dict(getattr(value, "__dict__") or {})
    else:
        return None
    banned = {
        "donor_ids",
        "distances",
        "normalized_weights",
        "rows",
        "per_unit_predictions",
        "per_unit_rows",
    }
    out: dict[str, Any] = {}
    for key, raw in source.items():
        k = str(key)
        if k in banned or k.startswith("donor_"):
            continue
        if isinstance(raw, Mapping):
            out[k] = dict(_sanitize_idw_validation(raw) or {})
        elif isinstance(raw, (list, tuple)):
            clean_items: list[Any] = []
            for item in raw:
                if isinstance(item, Mapping):
                    clean_items.append(dict(_sanitize_idw_validation(item) or {}))
                else:
                    clean_items.append(item)
            out[k] = clean_items
        else:
            out[k] = raw
    return out


class EmbedderLike(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class SessionClassifierLike(Protocol):
    def enumerate_unique_clean_texts(
        self,
        *,
        steps: Sequence[SessionStepText],
        token_counter: Callable[[str], int] | None = None,
        max_input_tokens: int = MAX_INPUT_TOKENS,
    ) -> tuple[str, ...]: ...

    def classify_sessions_from_text_embeddings(
        self,
        *,
        sessions_by_unit_id: Mapping[str, Sequence[SessionStepText]],
        embeddings_by_text: Mapping[str, Sequence[float]],
        token_counter: Callable[[str], int] | None = None,
        max_input_tokens: int = MAX_INPUT_TOKENS,
    ) -> dict[str, SessionSelection | None]: ...


class _PacketHashEmbedder:
    def __init__(self, vectors_by_hash: Mapping[str, np.ndarray]) -> None:
        self._vectors = {str(k): np.asarray(v, dtype=np.float32) for k, v in vectors_by_hash.items()}

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            h = _sha256_text(text)
            vec = self._vectors.get(h)
            if vec is None:
                raise KeyError(f"packet hash not found in cache: {h}")
            rows.append(vec)
        return np.asarray(rows, dtype=np.float32)


@dataclass(frozen=True)
class _EmbeddingsCacheMaterial:
    provenance: dict[str, Any]
    packet_hashes: tuple[str, ...]
    vectors_by_hash: dict[str, np.ndarray]


@dataclass(frozen=True)
class _TextEmbeddingsCacheMaterial:
    provenance: dict[str, Any]
    text_hashes: tuple[str, ...]
    vectors_by_hash: dict[str, np.ndarray]


@dataclass(frozen=True)
class _Arm2SelectionResult:
    selected_ids: tuple[str, ...]
    records: tuple[SelectionRecord, ...]
    telemetry: dict[str, Any]


class _ResilientVectorStore:
    """Retry transient live-store operations without changing selection semantics."""

    def __init__(self, inner: VectorStore, *, max_attempts: int = 3, retry_seconds: float = 0.25) -> None:
        self._inner = inner
        self._max_attempts = max(1, int(max_attempts))
        self._retry_seconds = max(0.0, float(retry_seconds))
        self._purged_cluster_ids: set[str] = set()
        self.retried_operations = 0
        self.stale_nearest_ignores = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 409, 429} or (isinstance(status_code, int) and status_code >= 500):
            return True
        return type(exc).__name__ in {"ServiceRequestError", "ServiceResponseError", "ServiceResponseTimeoutError"}

    def _call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                return getattr(self._inner, method_name)(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                if not self._is_retryable(exc):
                    raise
                if attempt < self._max_attempts - 1:
                    self.retried_operations += 1
                    if self._retry_seconds > 0.0:
                        time.sleep(self._retry_seconds)
        assert last_error is not None
        raise last_error

    def nearest(self, vec, agent_id=None, semantic_scope=None, tenant_id=None, run_scope=None):
        nearest = self._call(
            "nearest",
            vec,
            agent_id=agent_id,
            semantic_scope=semantic_scope,
            tenant_id=tenant_id,
            run_scope=run_scope,
        )
        if nearest is not None and str(nearest[0]) in self._purged_cluster_ids:
            self.stale_nearest_ignores += 1
            return None
        return nearest

    def upsert(self, doc):
        return self._call("upsert", doc)

    def touch(self, cluster_id: str, now: float) -> None:
        self._call("touch", cluster_id, now)

    def purge_stale(self, now: float, ttl: float, semantic_scope=None, tenant_id=None, run_scope=None):
        purged = self._call(
            "purge_stale",
            now=now,
            ttl=ttl,
            semantic_scope=semantic_scope,
            tenant_id=tenant_id,
            run_scope=run_scope,
        )
        self._purged_cluster_ids.update(str(cluster_id) for cluster_id in purged)
        return purged

    def delete_scope(self, tenant_id: str, run_scope: str, semantic_scope=None):
        return self._call("delete_scope", tenant_id, run_scope, semantic_scope=semantic_scope)

    def delete_scope_settled(self, tenant_id: str, run_scope: str, semantic_scope=None, *, max_attempts=3, settle_seconds=0.0):
        return self._call(
            "delete_scope_settled",
            tenant_id,
            run_scope,
            semantic_scope=semantic_scope,
            max_attempts=max_attempts,
            settle_seconds=settle_seconds,
        )


@dataclass(frozen=True)
class _ClassificationRow:
    unit_id: str
    agent_id: str
    concept_key: str
    corpus_id: str
    use_case_guid: str
    domain: str
    segment: str
    category: str
    sub_category: str
    sub_subcategory: str
    business_task: str
    status: str
    confidence_level: int
    combined_cosine_similarity: float | None
    selected_step_index: int | None
    selected_step_provenance: str | None


def _build_packet_hashes(
    data: CombinedDataset,
    *,
    tokenizer: TiktokenTokenizer,
    max_session_packet_tokens: int,
) -> dict[str, str]:
    builder = TokenSessionEvidencePacketBuilder(
        options=CanonicalizationOptions(tokenizer=tokenizer, max_tokens=max_session_packet_tokens),
        max_size=max(4096, len(data.unit_ids)),
    )
    out: dict[str, str] = {}
    for unit_id in data.unit_ids:
        trace = data.trace_by_unit_id[unit_id]
        packet = builder.build(trace)
        out[unit_id] = _sha256_text(packet.canonical_json)
    return out


def _chunked(values: Sequence[str], size: int) -> Iterable[list[str]]:
    size = max(1, int(size))
    for idx in range(0, len(values), size):
        yield list(values[idx : idx + size])


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _cache_npz_paths(base_path: Path) -> tuple[Path, Path]:
    if base_path.suffix.lower() == ".npz":
        npz_path = base_path
        manifest_path = base_path.with_suffix(".manifest.json")
    else:
        base_path.mkdir(parents=True, exist_ok=True)
        npz_path = base_path / "vectors.npz"
        manifest_path = base_path / "manifest.json"
    return npz_path, manifest_path


def _load_embeddings_cache(base_path: Path) -> _EmbeddingsCacheMaterial | None:
    npz_path, manifest_path = _cache_npz_paths(base_path)
    if not npz_path.exists() or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet_hashes = tuple(str(x) for x in (manifest.get("packet_hashes") or []))
    payload = np.load(npz_path, allow_pickle=False)
    hashes = tuple(str(x) for x in payload["hashes"].tolist())
    vectors = np.asarray(payload["vectors"], dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("embedding cache vectors must be a 2D matrix")
    if len(hashes) != vectors.shape[0]:
        raise ValueError("embedding cache hash/vector row mismatch")
    vectors_by_hash = {h: vectors[idx] for idx, h in enumerate(hashes)}
    return _EmbeddingsCacheMaterial(
        provenance=dict(manifest.get("provenance") or {}),
        packet_hashes=packet_hashes,
        vectors_by_hash=vectors_by_hash,
    )


def _persist_embeddings_cache(base_path: Path, *, provenance: Mapping[str, Any], vectors_by_hash: Mapping[str, np.ndarray], packet_hashes: Sequence[str]) -> None:
    npz_path, manifest_path = _cache_npz_paths(base_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_hashes = sorted(vectors_by_hash)
    matrix = np.asarray([np.asarray(vectors_by_hash[h], dtype=np.float32) for h in ordered_hashes], dtype=np.float32)

    tmp_npz = npz_path.with_name(
        f".{npz_path.stem}.{os.getpid()}.{time.time_ns()}.tmp.npz"
    )
    np.savez_compressed(tmp_npz, hashes=np.asarray(ordered_hashes, dtype="<U64"), vectors=matrix)
    os.replace(tmp_npz, npz_path)

    manifest = {
        "version": "sampling-v6-runtime-embeddings-cache-v1",
        "generated_at": _iso_now_utc(),
        "provenance": dict(provenance),
        "packet_hashes": list(packet_hashes),
        "rows": len(ordered_hashes),
        "dimensions": int(matrix.shape[1]) if matrix.size else 0,
    }
    _write_json_atomic(manifest_path, manifest)


def _load_text_embeddings_cache(base_path: Path) -> _TextEmbeddingsCacheMaterial | None:
    npz_path, manifest_path = _cache_npz_paths(base_path)
    if not npz_path.exists() or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = np.load(npz_path, allow_pickle=False)
    hashes = tuple(str(x) for x in payload["hashes"].tolist())
    vectors = np.asarray(payload["vectors"], dtype=np.float32)
    vectors_by_hash = {h: vectors[idx] for idx, h in enumerate(hashes)}
    return _TextEmbeddingsCacheMaterial(
        provenance=dict(manifest.get("provenance") or {}),
        text_hashes=hashes,
        vectors_by_hash=vectors_by_hash,
    )


def _persist_text_embeddings_cache(base_path: Path, *, provenance: Mapping[str, Any], vectors_by_hash: Mapping[str, np.ndarray]) -> None:
    npz_path, manifest_path = _cache_npz_paths(base_path)
    ordered_hashes = sorted(vectors_by_hash)
    matrix = np.asarray([np.asarray(vectors_by_hash[h], dtype=np.float32) for h in ordered_hashes], dtype=np.float32)

    tmp_npz = npz_path.with_name(
        f".{npz_path.stem}.{os.getpid()}.{time.time_ns()}.tmp.npz"
    )
    np.savez_compressed(tmp_npz, hashes=np.asarray(ordered_hashes, dtype="<U64"), vectors=matrix)
    os.replace(tmp_npz, npz_path)

    manifest = {
        "version": "sampling-v6-classification-embeddings-cache-v1",
        "generated_at": _iso_now_utc(),
        "provenance": dict(provenance),
        "rows": len(ordered_hashes),
        "dimensions": int(matrix.shape[1]) if matrix.size else 0,
    }
    _write_json_atomic(manifest_path, manifest)


def _build_runtime_with_cache(
    data: CombinedDataset,
    *,
    tokenizer: TiktokenTokenizer,
    embedder: EmbedderLike,
    embedding_model_id: str,
    embedding_deployment_id: str,
    embedding_batch_size: int,
    embedding_dimensions: int,
    max_session_packet_tokens: int,
    cache_base_path: Path,
    source_hashes: Mapping[str, str],
) -> tuple[V3Runtime, dict[str, Any]]:
    packet_hashes_by_unit = _build_packet_hashes(
        data,
        tokenizer=tokenizer,
        max_session_packet_tokens=max_session_packet_tokens,
    )
    expected_packet_hashes = tuple(sorted(set(packet_hashes_by_unit.values())))
    provenance = {
        "version": "sampling-v6-runtime-cache-provenance-v1",
        "source_hashes": dict(source_hashes),
        "embedding_model_id": embedding_model_id,
        "embedding_deployment_id": embedding_deployment_id,
        "embedding_dimensions": int(embedding_dimensions),
        "max_session_packet_tokens": int(max_session_packet_tokens),
        "tokenizer_encoding": tokenizer.encoding_name,
    }

    cache_hit = False
    cache = _load_embeddings_cache(cache_base_path)
    if cache is not None:
        if cache.provenance == provenance and cache.packet_hashes == expected_packet_hashes:
            cached_embedder = _PacketHashEmbedder(cache.vectors_by_hash)
            runtime = build_v3_runtime(
                data,
                tokenizer=tokenizer,
                embedder=cached_embedder,
                embedding_model_id=embedding_model_id,
                embedding_deployment_id=embedding_deployment_id,
                embedding_batch_size=embedding_batch_size,
                embedding_dimensions=embedding_dimensions,
                max_session_packet_tokens=max_session_packet_tokens,
            )
            cache_hit = True
            return runtime, {
                "cache_hit": cache_hit,
                "cache_rows": len(cache.vectors_by_hash),
                "cache_packet_hash_count": len(expected_packet_hashes),
                "provenance": provenance,
            }

    runtime = build_v3_runtime(
        data,
        tokenizer=tokenizer,
        embedder=embedder,
        embedding_model_id=embedding_model_id,
        embedding_deployment_id=embedding_deployment_id,
        embedding_batch_size=embedding_batch_size,
        embedding_dimensions=embedding_dimensions,
        max_session_packet_tokens=max_session_packet_tokens,
    )
    vectors_by_hash = {
        content_hash: np.asarray(record.vector, dtype=np.float32)
        for content_hash, record in runtime.embedding_records_by_content_sha256.items()
    }
    _persist_embeddings_cache(
        cache_base_path,
        provenance=provenance,
        vectors_by_hash=vectors_by_hash,
        packet_hashes=expected_packet_hashes,
    )
    return runtime, {
        "cache_hit": cache_hit,
        "cache_rows": len(vectors_by_hash),
        "cache_packet_hash_count": len(expected_packet_hashes),
        "provenance": provenance,
    }


def _session_steps_for_unit(unit: EvaluationUnit) -> tuple[SessionStepText, ...]:
    steps: list[SessionStepText] = []
    for turn in unit.turns:
        steps.append(SessionStepText(request=turn.user_text, response=turn.assistant_text))
    return tuple(steps)


def _classify_units_with_cache(
    *,
    data: CombinedDataset,
    classifier: SessionClassifierLike,
    embedder: EmbedderLike,
    tokenizer: TiktokenTokenizer,
    embedding_batch_size: int,
    cache_base_path: Path,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, _ClassificationRow], dict[str, Any]]:
    sessions_by_unit: dict[str, tuple[SessionStepText, ...]] = {}
    unit_by_id: dict[str, EvaluationUnit] = {}
    for unit in data.units:
        uid = str(unit.unit_id or "")
        sessions_by_unit[uid] = _session_steps_for_unit(unit)
        unit_by_id[uid] = unit

    text_to_hash: dict[str, str] = {}
    for unit_id in data.unit_ids:
        unique = classifier.enumerate_unique_clean_texts(
            steps=sessions_by_unit[unit_id],
            token_counter=tokenizer.count,
            max_input_tokens=MAX_INPUT_TOKENS,
        )
        for text in unique:
            text_to_hash[text] = _sha256_text(text)

    cache = _load_text_embeddings_cache(cache_base_path)
    vectors_by_hash: dict[str, np.ndarray] = {}
    if cache is not None and cache.provenance == dict(provenance):
        vectors_by_hash.update(cache.vectors_by_hash)

    missing_texts = [text for text, h in sorted(text_to_hash.items(), key=lambda row: row[1]) if h not in vectors_by_hash]
    cache_hit_rows_before = len(text_to_hash) - len(missing_texts)
    embedding_calls = 0
    embedding_inputs = 0
    embedding_input_tokens = 0
    embedding_latency_seconds = 0.0
    if missing_texts:
        embedding_input_tokens = int(sum(int(tokenizer.count(text)) for text in missing_texts))
        for batch in _chunked(missing_texts, max(1, int(embedding_batch_size))):
            started = time.perf_counter()
            vectors = np.asarray(embedder.embed(batch), dtype=np.float32)
            embedding_latency_seconds += float(time.perf_counter() - started)
            embedding_calls += 1
            embedding_inputs += len(batch)
            if vectors.ndim != 2:
                raise ValueError("classification embedder returned non-matrix output")
            if vectors.shape[0] != len(batch):
                raise ValueError(
                    f"classification embedder row mismatch: expected {len(batch)}, got {vectors.shape[0]}"
                )
            if vectors.shape[1] != V3_EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"classification embedder dimensions mismatch: expected {V3_EMBEDDING_DIMENSIONS}, got {vectors.shape[1]}"
                )
            if not np.all(np.isfinite(vectors)):
                raise ValueError("classification embedder returned non-finite vector values")
            for idx, text in enumerate(batch):
                vectors_by_hash[text_to_hash[text]] = np.asarray(vectors[idx], dtype=np.float32)

    _persist_text_embeddings_cache(cache_base_path, provenance=provenance, vectors_by_hash=vectors_by_hash)

    embeddings_by_text = {text: vectors_by_hash[text_to_hash[text]] for text in text_to_hash}
    selections = classifier.classify_sessions_from_text_embeddings(
        sessions_by_unit_id=sessions_by_unit,
        embeddings_by_text=embeddings_by_text,
        token_counter=tokenizer.count,
        max_input_tokens=MAX_INPUT_TOKENS,
    )

    rows: dict[str, _ClassificationRow] = {}
    for unit_id in data.unit_ids:
        unit = unit_by_id[unit_id]
        selection = selections.get(unit_id)
        if selection is None:
            rows[unit_id] = _ClassificationRow(
                unit_id=unit_id,
                agent_id=f"{unit.tenant_id}|{unit.agent_id}",
                concept_key="|".join(
                    [
                        str((data.metadata_by_unit.get(unit_id) or {}).get("corpus_id") or "unknown"),
                        str((data.metadata_by_unit.get(unit_id) or {}).get("domain") or "unknown"),
                        str((data.metadata_by_unit.get(unit_id) or {}).get("task") or "unknown"),
                        str((data.metadata_by_unit.get(unit_id) or {}).get("difficulty") or "unknown"),
                    ]
                ),
                corpus_id=str((data.metadata_by_unit.get(unit_id) or {}).get("corpus_id") or "unknown"),
                use_case_guid=UNDETERMINED_SENTINEL,
                domain="",
                segment="",
                category="",
                sub_category="",
                sub_subcategory="",
                business_task="",
                status="Undetermined",
                confidence_level=0,
                combined_cosine_similarity=None,
                selected_step_index=None,
                selected_step_provenance=None,
            )
            continue

        determination = selection.determination
        info = determination.use_case
        guid = determination.guid
        guid_text = UNDETERMINED_SENTINEL if guid is None else str(guid)
        rows[unit_id] = _ClassificationRow(
            unit_id=unit_id,
            agent_id=f"{unit.tenant_id}|{unit.agent_id}",
            concept_key="|".join(
                [
                    str((data.metadata_by_unit.get(unit_id) or {}).get("corpus_id") or "unknown"),
                    str((data.metadata_by_unit.get(unit_id) or {}).get("domain") or "unknown"),
                    str((data.metadata_by_unit.get(unit_id) or {}).get("task") or "unknown"),
                    str((data.metadata_by_unit.get(unit_id) or {}).get("difficulty") or "unknown"),
                ]
            ),
            corpus_id=str((data.metadata_by_unit.get(unit_id) or {}).get("corpus_id") or "unknown"),
            use_case_guid=guid_text,
            domain="" if info is None else info.domain,
            segment="" if info is None else info.segment,
            category="" if info is None else info.category,
            sub_category="" if info is None else info.sub_category,
            sub_subcategory="" if info is None else info.sub_subcategory,
            business_task="" if info is None else info.business_task,
            status=determination.status,
            confidence_level=int(determination.confidence_level),
            combined_cosine_similarity=_finite_float_or_none(determination.combined_cosine_similarity),
            selected_step_index=int(selection.step_index),
            selected_step_provenance=str(selection.provenance),
        )

    stats = {
        "unique_clean_text_count": len(text_to_hash),
        "cache_rows": len(vectors_by_hash),
        "cache_hit_rows": cache_hit_rows_before,
        "missing_embedded_rows": len(missing_texts),
        "embedding_calls": int(embedding_calls),
        "embedding_inputs": int(embedding_inputs),
        "embedding_input_tokens": int(embedding_input_tokens),
        "embedding_latency_seconds": float(embedding_latency_seconds),
        "elapsed_seconds": float(embedding_latency_seconds),
        "provenance": dict(provenance),
    }
    return rows, stats


def _arm3_membership_summary(
    *,
    selected_ids: Sequence[str],
    descriptors_by_unit: Mapping[str, SessionDescriptor],
    descriptor_population: Sequence[SessionDescriptor],
) -> dict[str, Any]:
    agent_capacity: dict[str, int] = {}
    for descriptor in descriptor_population:
        agent_capacity[descriptor.agent_id] = agent_capacity.get(descriptor.agent_id, 0) + 1

    selected_agents = {
        descriptors_by_unit[unit_id].agent_id
        for unit_id in selected_ids
        if unit_id in descriptors_by_unit
    }
    represented_strata = sorted(
        {
            descriptors_by_unit[unit_id].use_case_id
            for unit_id in selected_ids
            if unit_id in descriptors_by_unit
        }
    )
    floor_targets = {agent_id: min(3, count) for agent_id, count in agent_capacity.items()}
    total_floor_target = int(sum(floor_targets.values()))
    floor_prefix_limit = min(int(total_floor_target), len(selected_ids))
    floor_prefix_count = 0
    per_agent_prefix: dict[str, int] = {}
    for unit_id in list(selected_ids)[:floor_prefix_limit]:
        descriptor = descriptors_by_unit.get(unit_id)
        if descriptor is None:
            continue
        agent_id = descriptor.agent_id
        cur = per_agent_prefix.get(agent_id, 0)
        if cur < floor_targets.get(agent_id, 0):
            per_agent_prefix[agent_id] = cur + 1
            floor_prefix_count += 1

    floor_complete = floor_prefix_count >= min(total_floor_target, len(selected_ids))
    return {
        "selected_agent_count": int(len(selected_agents)),
        "agents_with_at_least_3": int(sum(1 for count in agent_capacity.values() if count >= 3)),
        "represented_strata": represented_strata,
        "total_floor_target": int(total_floor_target),
        "floor_complete": bool(floor_complete),
        "floor_prefix_count": int(floor_prefix_count),
        "arm3_floor": {
            "total_floor_target": int(total_floor_target),
            "floor_complete": bool(floor_complete),
            "floor_prefix_count": int(floor_prefix_count),
        },
    }


def _sanitize_arm2_idw_provenance(idw_result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not idw_result:
        return None
    estimated_population = idw_result.get("estimated_population")
    aggregate = getattr(estimated_population, "aggregate", None)
    if aggregate is None:
        return None
    return {
        "population_count": int(getattr(aggregate, "population_count", 0)),
        "observed_count": int(getattr(aggregate, "observed_count", 0)),
        "imputed_count": int(getattr(aggregate, "imputed_count", 0)),
        "provenance_counts": dict(getattr(aggregate, "provenance_counts", {}) or {}),
        "zero_donor_agent_count": int(getattr(aggregate, "zero_donor_agent_count", 0)),
        "prior_count": int(getattr(aggregate, "prior_count", 0)),
        "estimated_pass_rate": _finite_float_or_none(getattr(aggregate, "estimated_pass_rate", None)),
    }


def _sanitize_arm2_idw_quality(validation: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if validation is None:
        return None
    return {
        "absolute_aggregate_rate_error": _finite_float_or_none(validation.get("absolute_aggregate_rate_error")),
        "per_unit_mae": _finite_float_or_none(validation.get("per_unit_mae")),
        "brier_score": _finite_float_or_none(validation.get("brier_score")),
        "macro_per_agent_mae": _finite_float_or_none(validation.get("macro_per_agent_mae")),
        "unjudged_only_mae": _finite_float_or_none(validation.get("unjudged_only_mae")),
        "unjudged_only_brier": _finite_float_or_none(validation.get("unjudged_only_brier")),
        "expected_calibration_error": _finite_float_or_none(validation.get("expected_calibration_error")),
    }


def _arm2_5_estimator_diagnostics(binary_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "threshold": 0.5,
        "rule": "value < 0.5 => 0; value >= 0.5 => 1",
        "continuous_estimate": _finite_float_or_none(binary_result.get("continuous_estimate")),
        "binary_estimate": _finite_float_or_none(binary_result.get("binary_estimate")),
        "note": "binary estimate is derived by thresholding the continuous donor model outputs",
    }


def _sanitize_arm6_diagnostics_for_run_row(diagnostics: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(diagnostics or {})
    return {
        "design": str(payload.get("design") or "represented_joint_cell_poststratified_hajek_agent_use_case"),
        "total_cell_count": int(payload.get("total_cell_count") or 0),
        "represented_cell_count": int(payload.get("represented_cell_count") or 0),
        "zero_sample_cell_count": int(payload.get("zero_sample_cell_count") or 0),
        "population_count_in_zero_sample_cells": int(payload.get("population_count_in_zero_sample_cells") or 0),
        "represented_population_fraction": _finite_float_or_none(payload.get("represented_population_fraction")),
        "weight_sum": _finite_float_or_none(payload.get("weight_sum")),
        "weight_ess": _finite_float_or_none(payload.get("weight_ess")),
        "max_weight": _finite_float_or_none(payload.get("max_weight")),
        "estimator_limitation": "realized represented-cell post-stratification; zero-sample cells are not recovered",
    }


def _sanitize_arm6_diagnostics_for_membership_row(diagnostics: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(diagnostics or {})
    return {
        "total_cell_count": int(payload.get("total_cell_count") or 0),
        "represented_cell_count": int(payload.get("represented_cell_count") or 0),
        "zero_sample_cell_count": int(payload.get("zero_sample_cell_count") or 0),
        "population_count_in_zero_sample_cells": int(payload.get("population_count_in_zero_sample_cells") or 0),
        "represented_population_fraction": _finite_float_or_none(payload.get("represented_population_fraction")),
        "weight_sum": _finite_float_or_none(payload.get("weight_sum")),
        "weight_ess": _finite_float_or_none(payload.get("weight_ess")),
        "max_weight": _finite_float_or_none(payload.get("max_weight")),
    }


def _unit_estimate_fields(row: Any) -> tuple[str, float] | None:
    if isinstance(row, Mapping):
        unit_id = str(row.get("unit_id") or "").strip()
        value = _finite_float_or_none(row.get("value"))
    else:
        unit_id = str(getattr(row, "unit_id", "") or "").strip()
        value = _finite_float_or_none(getattr(row, "value", None))
    if not unit_id or value is None:
        return None
    return unit_id, float(value)


def _hajek_ratio_estimate(selected_ids: Sequence[str], labels_by_unit: Mapping[str, bool], inclusion_probability_by_unit: Mapping[str, float]) -> float | None:
    if not selected_ids:
        return None
    numerator = 0.0
    denominator = 0.0
    for unit_id in selected_ids:
        pi = _finite_float_or_none(inclusion_probability_by_unit.get(unit_id))
        if pi is None or pi <= 0.0:
            continue
        y = 1.0 if bool(labels_by_unit.get(unit_id, False)) else 0.0
        numerator += y / pi
        denominator += 1.0 / pi
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _scope_count(store: Any, *, tenant_id: str, run_scope: str, semantic_scope: str) -> int | None:
    if hasattr(store, "count_scope"):
        return int(store.count_scope(tenant_id=tenant_id, run_scope=run_scope, semantic_scope=semantic_scope))
    build_filter = getattr(store, "_build_filter", None)
    search_ids = getattr(store, "_search_ids", None)
    if callable(build_filter) and callable(search_ids):
        flt = build_filter(tenant_id=tenant_id, run_scope=run_scope, semantic_scope=semantic_scope)
        ids = search_ids(filter_expr=flt)
        return len(ids)
    return None


def _select_arm2_exact_count(
    *,
    data: CombinedDataset,
    runtime: V3Runtime,
    cap: int,
    seed: int,
    ordered_unit_ids: Sequence[str],
    tenant_id: str,
    run_scope: str,
    vector_store_factory: Callable[[str, str], VectorStore],
    cleanup_max_attempts: int,
    cleanup_settle_seconds: float,
    progress_callback: ProgressCallback | None = None,
) -> _Arm2SelectionResult:
    total_sessions = max(1, len(ordered_unit_ids))
    replay_step = max(1, total_sessions // 100)
    start_time = time.monotonic()

    def emit(event: Mapping[str, Any]) -> None:
        payload = {
            "version": "sampling-v6-progress-v1",
            "status": str(event.get("status", "running")),
            "phase": str(event.get("phase", "replay")),
            "message": str(event.get("message", "arm2 replay")),
            "current_seed": seed,
            "current_cap": cap,
            "current_method": METHOD_IDS["arm2"],
            "completed_replays": 0,
            "total_replays": 1,
            "completed_cells": 0,
            "total_cells": 5,
            "replay_session_current": int(event.get("replay_session_current", 0)),
            "replay_session_total": total_sessions,
            "percent": float(event.get("percent", 0.0)),
            "elapsed_seconds": round(time.monotonic() - start_time, 6),
            "updated_at": _iso_now_utc(),
        }
        for key, value in dict(event).items():
            if key not in {"status", "phase", "message", "percent", "replay_session_current", "replay_session_total"}:
                payload[key] = value
        if progress_callback is not None:
            try:
                progress_callback(dict(payload))
            except Exception:
                pass

    emit({
        "status": "running",
        "phase": "pre-cleanup",
        "message": "Cleaning Azure Search scope before Arm2 replay",
        "replay_session_current": 0,
        "replay_session_total": total_sessions,
        "percent": 0.0,
    })

    selector = V3EmbeddingSelector(
        runtime,
        vector_store_factory=vector_store_factory,
        tau=V3_DEFAULT_EMBEDDING_TAU,
        cleanup_max_attempts=cleanup_max_attempts,
        cleanup_settle_seconds=cleanup_settle_seconds,
    )
    index, store = selector.build_index(tenant_id=tenant_id, run_scope=run_scope)

    _, _ = store.delete_scope_settled(
        tenant_id,
        run_scope,
        semantic_scope=runtime.embedding_semantic_scope,
        max_attempts=cleanup_max_attempts,
        settle_seconds=cleanup_settle_seconds,
    )
    remaining_before = _scope_count(
        store,
        tenant_id=tenant_id,
        run_scope=run_scope,
        semantic_scope=runtime.embedding_semantic_scope,
    )
    if remaining_before is not None and remaining_before != 0:
        raise RuntimeError("arm2 pre-run cleanup did not settle to zero remaining docs")

    sampler = AdaptiveSampler(
        SamplerConfig(
            llm_throughput=1_000_000.0,
            agent_floor=0.0,
            enforce_keep_one_floor=False,
        ),
        seed=seed,
        variety_index=index,
        use_novelty=True,
    )

    native: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for idx, unit_id in enumerate(ordered_unit_ids):
        if idx == 0 or idx % replay_step == 0 or idx == total_sessions - 1:
            emit({
                "status": "running",
                "phase": "replay",
                "message": f"Arm2 replay at session {idx + 1}/{total_sessions}",
                "replay_session_current": idx + 1,
                "replay_session_total": total_sessions,
                "percent": round((idx + 1) / total_sessions * 100.0, 4),
                "native_count": len(native),
                "rejected_count": len(rejected),
            })
        base = data.trace_by_unit_id[unit_id]
        trace = _to_trace_with_timestamp(base, idx)
        _ = sampler.decide(trace, admit_keep=True)
        obs = sampler.last_observation
        row = {
            "unit_id": unit_id,
            "replay_idx": idx,
            "novelty": float(getattr(obs, "novelty", 0.0) or 0.0),
            "rarity": float(getattr(obs, "rarity", 0.0) or 0.0),
            "rank": _stable_hash(seed, "arm2", unit_id),
        }
        kind = str(getattr(getattr(obs, "key", None), "kind", ""))
        if "fallback" in kind:
            raise RuntimeError("arm2 encountered fallback-signature behavior")
        if bool(sampler.last_proposed_keep):
            native.append(row)
        else:
            rejected.append(row)

    if int(getattr(index, "n_fallbacks", 0)) > 0:
        raise RuntimeError("arm2 encountered fallback behavior; refusing to continue")

    cap = min(int(cap), len(ordered_unit_ids))
    selected_rows: list[dict[str, Any]] = []
    reasons: dict[str, str] = {}

    if len(native) > cap:
        ranked_native = sorted(native, key=lambda row: (-row["novelty"], -row["rarity"], row["rank"]))
        selected_rows = ranked_native[:cap]
        for row in selected_rows:
            reasons[row["unit_id"]] = "arm2-native-overflow-ranked"
    else:
        selected_rows = sorted(native, key=lambda row: row["replay_idx"])
        for row in selected_rows:
            reasons[row["unit_id"]] = "arm2-native-replay-order"
        remaining = cap - len(selected_rows)
        if remaining > 0:
            ranked_fill = sorted(rejected, key=lambda row: (-row["novelty"], -row["rarity"], row["rank"]))
            for row in ranked_fill:
                if remaining <= 0:
                    break
                selected_rows.append(row)
                reasons[row["unit_id"]] = "arm2-fill-novelty-rarity"
                remaining -= 1

    selected_ids = tuple(row["unit_id"] for row in selected_rows)
    if len(selected_ids) != cap:
        raise RuntimeError(f"arm2 selected {len(selected_ids)} sessions, expected exact cap {cap}")
    if len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("arm2 selected duplicate unit IDs")

    records = tuple(
        SelectionRecord(
            unit_id=unit_id,
            method_id=METHOD_IDS["arm2"],
            stratum="embedding-native-fill",
            inclusion_probability=None,
            weight=None,
            reason=reasons[unit_id],
        )
        for unit_id in selected_ids
    )

    _, _ = store.delete_scope_settled(
        tenant_id,
        run_scope,
        semantic_scope=runtime.embedding_semantic_scope,
        max_attempts=cleanup_max_attempts,
        settle_seconds=cleanup_settle_seconds,
    )
    remaining_after = _scope_count(
        store,
        tenant_id=tenant_id,
        run_scope=run_scope,
        semantic_scope=runtime.embedding_semantic_scope,
    )
    if remaining_after is not None and remaining_after != 0:
        raise RuntimeError("arm2 post-run cleanup did not settle to zero remaining docs")

    resilient_store = getattr(store, "_inner", None)
    telemetry = {
        "native_proposed_count": len(native),
        "rejected_count": len(rejected),
        "fallbacks": int(getattr(index, "n_fallbacks", 0)),
        "search_queries": int(getattr(store, "search_queries", 0)),
        "writes": int(getattr(store, "writes", 0)),
        "cleanup_deleted": int(getattr(store, "cleanup_deleted", 0)),
        "search_retried_operations": int(getattr(resilient_store, "retried_operations", 0)),
        "search_stale_nearest_ignores": int(getattr(resilient_store, "stale_nearest_ignores", 0)),
    }
    emit({
        "status": "running",
        "phase": "post-cleanup",
        "message": "Cleaning Azure Search scope after Arm2 replay",
        "replay_session_current": total_sessions,
        "replay_session_total": total_sessions,
        "percent": 100.0,
    })
    return _Arm2SelectionResult(selected_ids=selected_ids, records=records, telemetry=telemetry)


def _linear_interpolate_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    if quantile <= 0.0:
        return ordered[0]
    if quantile >= 1.0:
        return ordered[-1]
    position = quantile * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    if lower_index == upper_index:
        return lower_value
    fraction = position - lower_index
    return lower_value + (upper_value - lower_value) * fraction


def _descriptive_stat_block(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "sample_std": 0.0, "p05": 0.0, "p95": 0.0, "count": 0, "min": 0.0, "max": 0.0}
    values_float = [float(v) for v in values]
    count = len(values_float)
    std_dev = 0.0 if count < 2 else float(np.std(values_float, ddof=1))
    block = {
        "mean": float(mean(values_float)),
        "median": float(_linear_interpolate_quantile(values_float, 0.50)),
        "sample_std": std_dev,
        "p05": float(_linear_interpolate_quantile(values_float, 0.05)),
        "p95": float(_linear_interpolate_quantile(values_float, 0.95)),
        "count": int(count),
        "min": float(min(values_float)),
        "max": float(max(values_float)),
    }
    return block


def _aggregate_trial_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method_id"]), int(row["cap"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (method_id, cap), bucket in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        maes = [float(r["absolute_aggregate_mae"]) for r in bucket]
        concept = [float(r["concept_coverage"]) for r in bucket]
        maven_cov = [float(r["use_case_coverage"]) for r in bucket]
        agent_cov = [float(r["agent_coverage"]) for r in bucket]
        sel_cnt = [int(r["sample_size"]) for r in bucket]
        tokens = [int(r["actual_token_count"]) for r in bucket]
        nominal_budget = [int(r["nominal_budget"]) for r in bucket]

        mae_block = _descriptive_stat_block(maes)
        concept_block = _descriptive_stat_block(concept)
        maven_block = _descriptive_stat_block(maven_cov)
        agent_block = _descriptive_stat_block(agent_cov)
        selected_block = _descriptive_stat_block([float(v) for v in sel_cnt])
        tokens_block = _descriptive_stat_block([float(v) for v in tokens])
        nominal_block = _descriptive_stat_block([float(v) for v in nominal_budget])

        out.append(
            {
                "method_id": method_id,
                "cap": int(cap),
                "replays": len(bucket),
                "trial_count": len(bucket),
                "mae": mae_block,
                "absolute_aggregate_mae": mae_block,
                "concept_coverage": concept_block,
                "maven_coverage": maven_block,
                "use_case_coverage": maven_block,
                "agent_coverage": agent_block,
                "selected_count": {**selected_block, "mean": float(selected_block["mean"]), "min": float(selected_block["min"]), "max": float(selected_block["max"])},
                "actual_tokens": {**tokens_block, "mean": float(tokens_block["mean"]), "min": float(tokens_block["min"]), "max": float(tokens_block["max"])},
                "actual_token_count": {**tokens_block, "mean": float(tokens_block["mean"]), "min": float(tokens_block["min"]), "max": float(tokens_block["max"])},
                "nominal_budget": {**nominal_block, "mean": float(nominal_block["mean"]), "min": float(nominal_block["min"]), "max": float(nominal_block["max"])},
            }
        )
    return out


def _aggregate_top_agents(run_metrics: Sequence[TrialMetrics | Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for metric in run_metrics:
        entries = (
            metric.get("top_five_agents") or ()
            if isinstance(metric, Mapping)
            else metric.top_five_agents
        )
        for entry in entries:
            key = str(entry.get("agent_id") or "")
            rows.setdefault(key, []).append(dict(entry))

    out: list[dict[str, Any]] = []
    for agent_id, bucket in rows.items():
        out.append(
            {
                "agent_id": agent_id,
                "rows": len(bucket),
                "n_mean": float(mean(float(r.get("n") or 0.0) for r in bucket)),
                "absolute_error_mean": float(
                    mean(float(r.get("absolute_error") or 0.0) for r in bucket)
                ),
                "concept_coverage_mean": float(
                    mean(float(r.get("concept_coverage") or 0.0) for r in bucket)
                ),
                "use_case_coverage_mean": float(
                    mean(float(r.get("use_case_coverage") or 0.0) for r in bucket)
                ),
            }
        )
    out.sort(key=lambda row: (-row["rows"], row["agent_id"]))
    return out[:5]


def _default_cache_paths(output_dir: Path) -> tuple[Path, Path]:
    reusable = Path("outputs_sampling_v6") / "cache"
    if output_dir.is_relative_to(Path("outputs_sampling_v6")):
        base = output_dir / "cache"
    else:
        base = reusable
    return base / "embeddings", base / "classifications"


def _source_hashes(data: CombinedDataset) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in sorted(data.source_paths.items()):
        path = Path(value)
        out[key] = _sha256_file(path) if path.exists() else ""
    return out


def _build_dataset_examples(data: CombinedDataset, source_hashes: Mapping[str, str]) -> dict[str, Any]:
    by_corpus: dict[str, list[EvaluationUnit]] = {}
    for unit in data.units:
        uid = str(unit.unit_id or "")
        corpus = str(data.corpus_id_by_unit.get(uid) or "unknown")
        by_corpus.setdefault(corpus, []).append(unit)

    examples: list[dict[str, Any]] = []
    source_summary: dict[str, Any] = {
        "schema": {
            "description": "Combined synthetic evaluation corpus built from source corpora and normalized into EvaluationUnit rows.",
            "expected_label_field": "labels_by_unit (boolean pass/fail)",
            "snippet_policy": "bounded first-turn preview from source text",
        },
        "corpora": [],
    }
    labels = [bool(data.labels_by_unit.get(str(unit.unit_id or ""), False)) for unit in data.units]
    source_summary["overall"] = {
        "unit_count": len(data.units),
        "pass_count": int(sum(1 for value in labels if value)),
        "pass_rate": float(sum(1 for value in labels if value) / max(1, len(labels))),
    }
    for corpus, units in sorted(by_corpus.items()):
        corpus_labels = [bool(data.labels_by_unit.get(str(unit.unit_id or ""), False)) for unit in units]
        source_summary["corpora"].append(
            {
                "corpus_id": corpus,
                "unit_count": len(units),
                "pass_count": int(sum(1 for value in corpus_labels if value)),
                "pass_rate": float(sum(1 for value in corpus_labels if value) / max(1, len(corpus_labels))),
                "source_hash": str(source_hashes.get(corpus, "")),
            }
        )
        for unit in units[:2]:
            uid = str(unit.unit_id or "")
            first_turn = unit.turns[0] if unit.turns else None
            examples.append(
                {
                    "corpus_id": corpus,
                    "unit_id": uid,
                    "source": {
                        "corpus_id": corpus,
                        "is_synthetic": True,
                        "source_hash": str(source_hashes.get(corpus, "")),
                    },
                    "expected_label": bool(data.labels_by_unit.get(uid, False)),
                    "agent": f"{unit.tenant_id}|{unit.agent_id}",
                    "metadata": dict(data.metadata_by_unit.get(uid) or {}),
                    "shape": {
                        "turn_count": len(unit.turns),
                        "tool_call_count": len(unit.tool_calls),
                        "had_error": bool(unit.had_error),
                    },
                    "snippet": {
                        "user": _safe_preview(None if first_turn is None else first_turn.user_text),
                        "assistant": _safe_preview(None if first_turn is None else first_turn.assistant_text),
                    },
                }
            )

    return {
        "version": "sampling-v6-dataset-examples-v1",
        "generated_at": _iso_now_utc(),
        "source_hashes": dict(source_hashes),
        "source_summary": source_summary,
        "synthesized_fields": {
            "source_synthetic": [
                "expected_label",
                "shape.turn_count",
                "shape.tool_call_count",
                "shape.had_error",
            ],
            "report_derived": [
                "snippet.user",
                "snippet.assistant",
            ],
        },
        "examples": examples,
    }


def _methodology_md() -> str:
    return "\n".join(
        [
            "# Sampling V6 Methodology",
            "",
            "## Scope",
            "- Population is the exact combined 2800-session retained synthetic corpus unless canary slicing is explicitly requested.",
            "- Expected labels are joined after membership freeze for every arm.",
            "",
            "## Arms",
            "- Arm1: global deterministic random membership.",
            "- Arm2: embedding-based adaptive selection replayed against the real vector index with exact-count cap and IDW estimation over the full population.",
            "- Arm2.5: thresholded binary estimator that reuses Arm2 membership exactly and maps continuous IDW per-unit values to binary via value < 0.5 => 0 and value >= 0.5 => 1.",
            "- Arm3: agent round-robin with floor prefix.",
            "- Arm4: agent round-robin probability design.",
            "- Arm5: Hajek-weighted reuse of Arm4 membership and inclusion probabilities.",
            "- Arm6: Hajek estimator on realized represented (agent,use-case-guid) cells, reusing Arm4 membership exactly.",
            "",
            "## Arm2 Exact-Count Logic",
            "- Replays a deterministic seed order through AdaptiveSampler + AzureClusterIndex via V3EmbeddingSelector.",
            "- Native proposals are accepted in replay order if native_count <= cap.",
            "- If native_count > cap, native rows are ranked by novelty desc, rarity desc, stable hash and truncated to cap.",
            "- If native_count < cap, fill rows are ranked by novelty desc, rarity desc, stable hash and appended until exact cap.",
            "- Any fallback-signature behavior causes hard failure.",
            "- Azure scope is cleaned before and after each arm2 replay; settled zero remaining docs is required when the store supports scope counting.",
            "",
            "## Token Accounting",
            "- Nominal budget per run is cap * avg_tokens_per_session.",
            "- Actual token count is the sum of runtime token costs for selected unit IDs.",
            "",
            "## Maven Classification Parity",
            "- Each turn is mapped in order to SessionStepText(request=user_text, response=assistant_text).",
            "- Cleaning and token limits use classifier-native logic and cl100k_base counting.",
            "- Classification applies threshold early-stop then max-similarity fallback exactly as classifier implementation defines.",
            "- classification artifacts store no raw session text.",
            "",
            "## Estimands",
            "- Aggregate estimand is pass-rate over full population.",
            "- Arm2 uses observed+IDW imputation and reports validation diagnostics.",
            "- Arm2.5 uses the Arm2 donor-model population and computes binary aggregate estimate after thresholding.",
            "- Arm5 uses Hajek ratio estimator with Arm4 inclusion probabilities.",
            "- Arm6 uses realized represented-cell post-stratification; unconditional design marginals are not used.",
            "",
            "## Limitations",
            "- Arm2 exact-count replay introduces deterministic ranking choices when proposals exceed cap.",
            "- IDW is model-assisted and should be treated as an approximation, not a design-unbiased guarantee.",
            "- Arm6 does not recover zero-sample joint cells; represented-cell weights only adjust within observed support.",
            "- Existing dataset metadata and expected labels are synthetic and non-production.",
            "",
        ]
    )


def _code_hashes(paths: Sequence[Path]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for rel in paths:
        p = Path(rel)
        key = str(rel).replace("\\", "/")
        out[key] = _sha256_file(p) if p.exists() else None
    return out


def _selection_code_hashes() -> dict[str, str | None]:
    return _code_hashes(_SELECTION_CONTROLLING_CODE_PATHS)


def _publisher_code_hashes() -> dict[str, str | None]:
    return _code_hashes(_PUBLISHER_CODE_PATHS)


def _hash_unit_ids(unit_ids: Sequence[str]) -> str:
    return _sha256_text(_canonical_json([str(x) for x in unit_ids]))


def _compatibility_payload(
    *,
    source_hashes: Mapping[str, str],
    unit_ids_hash: str,
    population_count: int,
    caps: Sequence[int],
    avg_tokens_per_session: int,
    embedding_model: str,
    embedding_deployment: str,
    embedding_dimensions: int,
    embedding_tau: float,
    search_endpoint_host: str,
    search_index: str,
    maven_meta: Mapping[str, Any],
    idw_config: IDWConfig,
    selection_code_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "version": "sampling-v6-compatibility-v1",
        "dataset_source_hashes": dict(source_hashes),
        "population": {
            "count": int(population_count),
            "unit_ids_hash": str(unit_ids_hash),
        },
        "caps": [int(x) for x in caps],
        "avg_tokens_per_session": int(avg_tokens_per_session),
        "embedding": {
            "model": str(embedding_model),
            "deployment": str(embedding_deployment),
            "dimensions": int(embedding_dimensions),
            "tau": float(embedding_tau),
        },
        "search": {
            "endpoint_host": str(search_endpoint_host),
            "index": str(search_index),
        },
        "maven": {
            "taxonomy_version": str(maven_meta.get("taxonomy_version") or ""),
            "taxonomy_db_sha256": str(maven_meta.get("taxonomy_db_sha256") or ""),
            "centroids_db_sha256": str(maven_meta.get("centroids_db_sha256") or ""),
        },
        "idw": asdict(idw_config),
        "selection_code_hashes": dict(selection_code_hashes),
    }


def _compatibility_fingerprint(payload: Mapping[str, Any]) -> str:
    return _sha256_text(_canonical_json(dict(payload)))


def _checkpoint_payload_hash(payload: Mapping[str, Any]) -> str:
    raw = {k: v for k, v in dict(payload).items() if k != "payload_hash"}
    return _sha256_text(_canonical_json(raw))


def _required_text(value: Any) -> str:
    out = str(value or "").strip()
    if not out:
        raise ValueError("required string value is missing")
    return out


def _extract_cell_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return (int(row["seed"]), int(row["cap"]), str(row["method_id"]))


def _validate_unique_cell_grid(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[tuple[int, int, str]] = set()
    for row in rows:
        key = _extract_cell_key(row)
        if key in seen:
            raise ValueError(f"duplicate run row key detected: {key}")
        seen.add(key)


def _arm2_zero_fallbacks_or_fail(run_rows: Sequence[Mapping[str, Any]]) -> None:
    for row in run_rows:
        if str(row.get("method_id")) != METHOD_IDS["arm2"]:
            continue
        telemetry = dict(row.get("arm2_telemetry") or {})
        if int(telemetry.get("fallbacks") or 0) != 0:
            raise ValueError("arm2 checkpoint/baseline row reports non-zero fallbacks")


def _find_arm_membership(rows: Sequence[Mapping[str, Any]], *, seed: int, cap: int, method_id: str) -> Mapping[str, Any] | None:
    for row in rows:
        if int(row.get("seed", -1)) == int(seed) and int(row.get("cap", -1)) == int(cap) and str(row.get("method_id")) == method_id:
            return row
    return None


def _validate_arm4_arm5_identity(membership_rows: Sequence[Mapping[str, Any]], *, seed: int, cap: int) -> None:
    arm4 = _find_arm_membership(membership_rows, seed=seed, cap=cap, method_id=METHOD_IDS["arm4"])
    arm5 = _find_arm_membership(membership_rows, seed=seed, cap=cap, method_id=METHOD_IDS["arm5"])
    if arm4 is None or arm5 is None:
        raise ValueError("missing arm4/arm5 membership rows")
    if tuple(arm4.get("selected_ids") or []) != tuple(arm5.get("selected_ids") or []):
        raise ValueError("arm4 and arm5 selected_ids must match")
    if dict(arm4.get("inclusion_probability_by_unit") or {}) != dict(arm5.get("inclusion_probability_by_unit") or {}):
        raise ValueError("arm4 and arm5 inclusion probabilities must match")


def _sanitize_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    blob = _canonical_json(dict(payload)).lower()
    banned = (
        "openai_api_key",
        "search_api_key",
        '"api_key"',
        '"password"',
        '"secret"',
        "user 0",
        "asst 0",
        '"vector"',
        '"vectors"',
    )
    for marker in banned:
        if marker in blob:
            raise ValueError(f"checkpoint payload contains forbidden marker: {marker}")


def _validate_checkpoint_rows(
    *,
    seed: int,
    cap: int,
    run_rows: Sequence[Mapping[str, Any]],
    membership_rows: Sequence[Mapping[str, Any]],
) -> None:
    expected_method_count = len(METHOD_ID_ORDER)
    if len(run_rows) != expected_method_count or len(membership_rows) != expected_method_count:
        raise ValueError("checkpoint must contain exactly one run row and one membership row per method")
    run_keys = {_extract_cell_key(row) for row in run_rows}
    mem_keys = {_extract_cell_key(row) for row in membership_rows}
    expected = {(int(seed), int(cap), method_id) for method_id in METHOD_ID_ORDER}
    if run_keys != expected:
        raise ValueError("checkpoint run rows do not match expected method set")
    if mem_keys != expected:
        raise ValueError("checkpoint membership rows do not match expected method set")
    for row in run_rows:
        if int(row.get("sample_size") or 0) != int(cap):
            raise ValueError("checkpoint run row sample_size does not equal cap")
    _arm2_zero_fallbacks_or_fail(run_rows)
    _validate_arm4_arm5_identity(membership_rows, seed=seed, cap=cap)


def _normalize_baseline_membership_row(
    *,
    row: Mapping[str, Any],
    descriptors_by_unit: Mapping[str, SessionDescriptor],
    descriptor_population: Sequence[SessionDescriptor],
) -> dict[str, Any]:
    out = dict(row)
    selected_ids = tuple(str(x) for x in (row.get("selected_ids") or []))
    represented = sorted(
        {
            descriptors_by_unit[unit_id].use_case_id
            for unit_id in selected_ids
            if unit_id in descriptors_by_unit
        }
    )
    selected_agents = {
        descriptors_by_unit[unit_id].agent_id
        for unit_id in selected_ids
        if unit_id in descriptors_by_unit
    }
    total_agent_count = len({descriptor.agent_id for descriptor in descriptor_population})
    agent_capacity: dict[str, int] = {}
    for descriptor in descriptor_population:
        agent_capacity[descriptor.agent_id] = agent_capacity.get(descriptor.agent_id, 0) + 1
    eligible_agents_with_3 = int(sum(1 for count in agent_capacity.values() if count >= 3))
    out["selected_agent_count"] = int(len(selected_agents))
    out["total_agent_count"] = int(total_agent_count)
    out["agents_with_at_least_3"] = int(sum(1 for agent in selected_agents if int(agent_capacity.get(agent, 0)) >= 3))
    out["eligible_agents_with_at_least_3"] = int(eligible_agents_with_3)
    out["agent_coverage"] = float((len(selected_agents) / total_agent_count) if total_agent_count else 1.0)
    out["represented_strata"] = represented
    if str(out.get("method_id")) == METHOD_IDS["arm3"]:
        out.update(
            _arm3_membership_summary(
                selected_ids=selected_ids,
                descriptors_by_unit=descriptors_by_unit,
                descriptor_population=descriptor_population,
            )
        )
        floor = dict(out.get("arm3_floor") or {})
        floor["floor_completion_ratio"] = float(
            (float(floor.get("floor_prefix_count") or 0.0) / max(1.0, float(floor.get("total_floor_target") or 1.0)))
        )
        out["arm3_floor"] = floor
        out["floor_completion_ratio"] = float(floor.get("floor_completion_ratio") or 0.0)
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(dict(json.loads(line)))
    return rows


def _baseline_paths(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        root / "aggregate.json",
        root / "runs.jsonl",
        root / "memberships.jsonl",
        root / "manifest.json",
        root / "classifications.jsonl",
    )


def _validate_baseline_classifications(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_population: int,
) -> list[dict[str, Any]]:
    cleaned = [dict(row) for row in rows]
    if len(cleaned) != expected_population:
        raise ValueError(f"baseline classifications row count mismatch: expected {expected_population}, got {len(cleaned)}")

    required_fields = {
        "version",
        "unit_id",
        "agent_id",
        "concept_key",
        "corpus_id",
        "use_case_guid",
        "domain",
        "segment",
        "category",
        "sub_category",
        "sub_subcategory",
        "business_task",
        "status",
        "confidence_level",
        "combined_cosine_similarity",
        "selected_step_index",
        "selected_step_provenance",
    }
    raw_text_fields = {"request", "response", "user_text", "assistant_text", "text", "raw_text", "request_text", "response_text"}
    seen_unit_ids: set[str] = set()
    for idx, row in enumerate(cleaned):
        if not isinstance(row, dict):
            raise ValueError(f"baseline classifications row {idx} is not an object")
        if str(row.get("version") or "") != "sampling-v6-classification-v1":
            raise ValueError(f"baseline classifications version mismatch at row {idx}")
        missing = sorted(field for field in required_fields if field not in row)
        if missing:
            raise ValueError(f"baseline classifications missing required fields at row {idx}: {missing}")
        if set(row).intersection(raw_text_fields):
            raise ValueError(f"baseline classifications row {idx} contains raw text fields")
        unit_id = str(row.get("unit_id") or "").strip()
        if not unit_id:
            raise ValueError(f"baseline classifications row {idx} has missing unit_id")
        if unit_id in seen_unit_ids:
            raise ValueError(f"baseline classifications duplicate unit_id: {unit_id}")
        seen_unit_ids.add(unit_id)
        similarity = row.get("combined_cosine_similarity")
        if similarity is not None:
            try:
                value = float(similarity)
            except (TypeError, ValueError):
                raise ValueError(f"baseline classifications similarity must be finite or null for unit_id={unit_id}")
            if not math.isfinite(value):
                raise ValueError(f"baseline classifications similarity must be finite or null for unit_id={unit_id}")
        selected_step_index = row.get("selected_step_index")
        if selected_step_index is not None:
            try:
                int(selected_step_index)
            except (TypeError, ValueError):
                raise ValueError(f"baseline classifications selected_step_index must be int or null for unit_id={unit_id}")
    if len(seen_unit_ids) != len(cleaned):
        raise ValueError("baseline classifications must contain unique unit IDs")
    return cleaned


def _coerce_baseline_classification_row(row: Mapping[str, Any]) -> _ClassificationRow:
    unit_id = str(row.get("unit_id") or "")
    if not unit_id:
        raise ValueError("baseline classifications contain blank unit_id")
    selected_step_index = row.get("selected_step_index")
    selected_step_provenance = row.get("selected_step_provenance")
    return _ClassificationRow(
        unit_id=unit_id,
        agent_id=str(row.get("agent_id") or ""),
        concept_key=str(row.get("concept_key") or ""),
        corpus_id=str(row.get("corpus_id") or ""),
        use_case_guid=str(row.get("use_case_guid") or UNDETERMINED_SENTINEL),
        domain=str(row.get("domain") or ""),
        segment=str(row.get("segment") or ""),
        category=str(row.get("category") or ""),
        sub_category=str(row.get("sub_category") or ""),
        sub_subcategory=str(row.get("sub_subcategory") or ""),
        business_task=str(row.get("business_task") or ""),
        status=str(row.get("status") or "Undetermined"),
        confidence_level=int(row.get("confidence_level") or 0),
        combined_cosine_similarity=_finite_float_or_none(row.get("combined_cosine_similarity")),
        selected_step_index=None if selected_step_index is None else int(selected_step_index),
        selected_step_provenance=None if selected_step_provenance in (None, "") else str(selected_step_provenance),
    )


def _load_baseline_bundle(root: Path) -> dict[str, Any]:
    aggregate_path, runs_path, memberships_path, manifest_path, classifications_path = _baseline_paths(root)
    for path in (aggregate_path, runs_path, memberships_path, manifest_path, classifications_path):
        if not path.exists():
            raise ValueError(f"baseline artifact missing: {path}")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    runs = _read_jsonl(runs_path)
    memberships = _read_jsonl(memberships_path)
    classifications = _read_jsonl(classifications_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(aggregate.get("version")) != V6_BUNDLE_VERSION:
        raise ValueError("baseline aggregate version mismatch")
    if str(manifest.get("version")) != V6_MANIFEST_VERSION:
        raise ValueError("baseline manifest version mismatch")
    expected_population = int((aggregate.get("population_audit") or {}).get("unit_count") or 0)
    classifications = _validate_baseline_classifications(classifications, expected_population=expected_population)
    _validate_unique_cell_grid(runs)
    return {
        "aggregate": aggregate,
        "runs": runs,
        "memberships": memberships,
        "classifications": classifications,
        "manifest": manifest,
        "aggregate_path": aggregate_path,
        "manifest_path": manifest_path,
        "classifications_path": classifications_path,
    }


def _validate_baseline_compatibility(
    *,
    baseline: Mapping[str, Any],
    expected_compatibility_payload: Mapping[str, Any],
    requested_new_seeds: Sequence[int],
) -> tuple[list[int], str | None]:
    aggregate = dict(baseline.get("aggregate") or {})
    runs = [dict(row) for row in (baseline.get("runs") or [])]
    memberships = [dict(row) for row in (baseline.get("memberships") or [])]

    baseline_cfg = dict(aggregate.get("config") or {})
    baseline_caps = [int(x) for x in (baseline_cfg.get("caps") or [])]
    expected_caps = [int(x) for x in (expected_compatibility_payload.get("caps") or [])]
    if baseline_caps != expected_caps:
        raise ValueError("baseline caps mismatch")
    if int(baseline_cfg.get("avg_tokens_per_session") or 0) != int(expected_compatibility_payload.get("avg_tokens_per_session") or 0):
        raise ValueError("baseline avg_tokens_per_session mismatch")

    baseline_population = dict(aggregate.get("population_audit") or {})
    expected_population = dict(expected_compatibility_payload.get("population") or {})
    if int(baseline_population.get("unit_count") or 0) != int(expected_population.get("count") or 0):
        raise ValueError("baseline population count mismatch")
    if str(baseline_population.get("unit_ids_hash") or "") and str(baseline_population.get("unit_ids_hash")) != str(expected_population.get("unit_ids_hash") or ""):
        raise ValueError("baseline population id hash mismatch")

    if dict(baseline_population.get("source_hashes") or {}) != dict(expected_compatibility_payload.get("dataset_source_hashes") or {}):
        raise ValueError("baseline source hashes mismatch")

    baseline_azure = dict(aggregate.get("azure") or {})
    expected_embedding = dict(expected_compatibility_payload.get("embedding") or {})
    expected_search = dict(expected_compatibility_payload.get("search") or {})
    if str(baseline_azure.get("embedding_model") or "") != str(expected_embedding.get("model") or ""):
        raise ValueError("baseline embedding model mismatch")
    if str(baseline_azure.get("embedding_deployment") or "") != str(expected_embedding.get("deployment") or ""):
        raise ValueError("baseline embedding deployment mismatch")
    if int(baseline_azure.get("embedding_dimensions") or 0) != int(expected_embedding.get("dimensions") or 0):
        raise ValueError("baseline embedding dimensions mismatch")
    if float(baseline_azure.get("embedding_tau") or 0.0) != float(expected_embedding.get("tau") or 0.0):
        raise ValueError("baseline embedding tau mismatch")
    if str(baseline_azure.get("search_endpoint_host") or "") != str(expected_search.get("endpoint_host") or ""):
        raise ValueError("baseline search endpoint mismatch")
    if str(baseline_azure.get("search_index") or "") != str(expected_search.get("index") or ""):
        raise ValueError("baseline search index mismatch")

    baseline_maven = dict(aggregate.get("maven_artifacts") or {})
    expected_maven = dict(expected_compatibility_payload.get("maven") or {})
    if str(baseline_maven.get("taxonomy_version") or "") != str(expected_maven.get("taxonomy_version") or ""):
        raise ValueError("baseline maven taxonomy version mismatch")
    if str(baseline_maven.get("taxonomy_db_sha256") or "") != str(expected_maven.get("taxonomy_db_sha256") or ""):
        raise ValueError("baseline taxonomy db hash mismatch")
    if str(baseline_maven.get("centroids_db_sha256") or "") != str(expected_maven.get("centroids_db_sha256") or ""):
        raise ValueError("baseline centroids db hash mismatch")

    baseline_idw = dict((aggregate.get("config") or {}).get("idw") or {})
    if baseline_idw != dict(expected_compatibility_payload.get("idw") or {}):
        raise ValueError("baseline idw config mismatch")

    expected_grid = len(baseline_caps) * len([int(x) for x in (baseline_cfg.get("seeds") or [])]) * len(METHOD_ID_ORDER)
    if len(runs) != expected_grid or len(memberships) != expected_grid:
        raise ValueError("baseline row counts do not match baseline seed x cap grid")

    baseline_seeds = sorted({int(row["seed"]) for row in runs})
    requested = sorted({int(x) for x in requested_new_seeds})
    if len(requested) != len([int(x) for x in requested_new_seeds]):
        raise ValueError("requested new seeds contain duplicates")
    overlap = sorted(set(baseline_seeds).intersection(requested))
    if overlap:
        raise ValueError(f"requested seeds overlap baseline seeds: {overlap}")

    _arm2_zero_fallbacks_or_fail(runs)
    method_set = set(METHOD_ID_ORDER)
    if {str(row.get("method_id")) for row in runs} != method_set:
        raise ValueError("baseline run rows do not include the full seven-arm method set")
    if {str(row.get("method_id")) for row in memberships} != method_set:
        raise ValueError("baseline membership rows do not include the full seven-arm method set")
    for seed in baseline_seeds:
        for cap in baseline_caps:
            _validate_arm4_arm5_identity(memberships, seed=seed, cap=cap)

    ext_fingerprint = str(((aggregate.get("provenance") or {}).get("extension_compatibility_fingerprint") or "")).strip() or None
    return baseline_seeds, ext_fingerprint


def _checkpoint_cell_path(checkpoint_root: Path, *, seed: int, cap: int) -> Path:
    return checkpoint_root / "cells" / f"seed-{int(seed)}-cap-{int(cap)}.json"


def _write_cell_checkpoint(
    *,
    checkpoint_root: Path,
    seed: int,
    cap: int,
    window_id: str,
    run_scope: str,
    run_rows: Sequence[Mapping[str, Any]],
    membership_rows: Sequence[Mapping[str, Any]],
    compatibility_payload: Mapping[str, Any],
    compatibility_fingerprint: str,
    source_hashes: Mapping[str, str],
    selection_code_hashes: Mapping[str, Any],
    publisher_code_hashes: Mapping[str, Any],
    arm2_telemetry: Mapping[str, Any],
    agent_metric_rows: Sequence[Mapping[str, Any]],
) -> None:
    payload: dict[str, Any] = {
        "version": V6_CHECKPOINT_VERSION,
        "status": "complete",
        "seed": int(seed),
        "cap": int(cap),
        "window_id": str(window_id),
        "run_scope": str(run_scope),
        "run_rows": [dict(row) for row in run_rows],
        "membership_rows": [dict(row) for row in membership_rows],
        "compatibility_fingerprint": str(compatibility_fingerprint),
        "compatibility_payload": dict(compatibility_payload),
        "source_hashes": dict(source_hashes),
        "selection_code_hashes": dict(selection_code_hashes),
        "producer_code_hashes": dict(publisher_code_hashes),
        "arm2_telemetry": dict(arm2_telemetry),
        "agent_metric_rows": [dict(row) for row in agent_metric_rows],
        "created_at": _iso_now_utc(),
    }
    _validate_checkpoint_rows(seed=seed, cap=cap, run_rows=payload["run_rows"], membership_rows=payload["membership_rows"])
    _sanitize_checkpoint_payload(payload)
    payload["payload_hash"] = _checkpoint_payload_hash(payload)
    _write_json_atomic(_checkpoint_cell_path(checkpoint_root, seed=seed, cap=cap), payload)


def _load_resume_checkpoint(
    *,
    checkpoint_root: Path,
    seed: int,
    cap: int,
    compatibility_fingerprint: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    path = _checkpoint_cell_path(checkpoint_root, seed=seed, cap=cap)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if str(raw.get("version")) != V6_CHECKPOINT_VERSION:
        raise ValueError(f"incompatible checkpoint version for seed={seed} cap={cap}")
    if str(raw.get("status")) != "complete":
        raise ValueError(f"checkpoint status is not complete for seed={seed} cap={cap}")
    if int(raw.get("seed", -1)) != int(seed) or int(raw.get("cap", -1)) != int(cap):
        raise ValueError(f"checkpoint seed/cap mismatch for seed={seed} cap={cap}")
    if str(raw.get("compatibility_fingerprint") or "") != str(compatibility_fingerprint):
        raise ValueError(f"checkpoint compatibility fingerprint mismatch for seed={seed} cap={cap}")
    expected_hash = _checkpoint_payload_hash(raw)
    if str(raw.get("payload_hash") or "") != expected_hash:
        raise ValueError(f"checkpoint payload hash mismatch for seed={seed} cap={cap}")
    run_rows = [dict(row) for row in (raw.get("run_rows") or [])]
    membership_rows = [dict(row) for row in (raw.get("membership_rows") or [])]
    agent_metric_rows = [dict(row) for row in (raw.get("agent_metric_rows") or [])]
    _validate_checkpoint_rows(seed=seed, cap=cap, run_rows=run_rows, membership_rows=membership_rows)
    _sanitize_checkpoint_payload(raw)
    return run_rows, membership_rows, agent_metric_rows


def run_sampling_v6_bundle(
    *,
    output_dir: str | Path,
    caps: Sequence[int] = SAMPLE_CAPS,
    seeds: Sequence[int] = TRIAL_SEEDS,
    avg_tokens_per_session: int = NOMINAL_TOKENS_PER_SESSION,
    embedding_batch_size: int = V3_DEFAULT_EMBEDDING_BATCH_SIZE,
    cleanup_max_attempts: int = 10,
    cleanup_settle_seconds: float = 0.0,
    ensure_search_index: bool = False,
    idw_config: IDWConfig = IDWConfig(),
    centroids_db_path: str = DEFAULT_MAVEN_CENTROIDS_DB,
    taxonomy_db_path: str = DEFAULT_MAVEN_TAXONOMY_DB,
    embeddings_cache_path: str | Path | None = None,
    classifications_cache_path: str | Path | None = None,
    skip_report: bool = False,
    data: CombinedDataset | None = None,
    azure_config: AzureConfig | None = None,
    tokenizer: TiktokenTokenizer | None = None,
    embedder: EmbedderLike | None = None,
    runtime: V3Runtime | None = None,
    classifier: SessionClassifierLike | None = None,
    use_case_artifacts: Any | None = None,
    vector_store_factory: Callable[[str, str], VectorStore] | None = None,
    arm2_selector: Callable[..., _Arm2SelectionResult] | None = None,
    enforce_integrity_counts: bool = True,
    progress_callback: ProgressCallback | None = None,
    baseline_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    requested_new_seeds = [int(x) for x in seeds]
    caps_list = [int(x) for x in caps]
    total_replays = max(1, len(requested_new_seeds) * len(caps_list))
    total_cells = max(1, total_replays * len(METHOD_ID_ORDER))
    progress_state: dict[str, Any] = {
        "version": "sampling-v6-progress-v1",
        "status": "running",
        "phase": "preprocessing",
        "message": "Preparing Sampling V6 bundle",
        "current_seed": None,
        "current_cap": None,
        "current_method": None,
        "completed_replays": 0,
        "total_replays": total_replays,
        "completed_cells": 0,
        "total_cells": total_cells,
        "baseline_trials": 0,
        "final_trials": len(seeds),
        "baseline_rows": 0,
        "final_rows": total_cells,
        "replay_session_current": 0,
        "replay_session_total": 0,
        "percent": 0.0,
        "elapsed_seconds": 0.0,
        "updated_at": _iso_now_utc(),
    }

    def emit(event: Mapping[str, Any]) -> None:
        merged = dict(progress_state)
        merged.update(event)
        merged["version"] = "sampling-v6-progress-v1"
        merged["updated_at"] = _iso_now_utc()
        merged["elapsed_seconds"] = round(time.monotonic() - start_time, 6)

        phase_name = str(merged.get("phase") or "progress")
        prior_percent = float(progress_state.get("percent", 0.0) or 0.0)
        merged["percent"] = _global_progress_percent(
            phase=phase_name,
            current_replay=merged.get("current_replay"),
            total_replays=int(merged.get("total_replays") or total_replays),
            replay_session_current=merged.get("replay_session_current"),
            replay_session_total=merged.get("replay_session_total"),
            completed_cells=int(merged.get("completed_cells", 0) or 0),
            total_cells=int(merged.get("total_cells") or total_cells),
            previous_percent=prior_percent,
        )
        if str(merged.get("status") or "running") == "complete":
            merged["percent"] = 100.0
        elif str(merged.get("status") or "running") == "failed":
            merged["percent"] = max(float(prior_percent), float(merged.get("percent", prior_percent)))
        else:
            merged["percent"] = max(float(prior_percent), float(merged.get("percent", prior_percent)))
        progress_state.update({key: value for key, value in merged.items() if key in progress_state or key in {"version", "status", "phase", "message", "current_seed", "current_cap", "current_method", "completed_replays", "total_replays", "completed_cells", "total_cells", "replay_session_current", "replay_session_total", "percent", "elapsed_seconds", "updated_at"}})
        _emit_progress_event(output_dir=root, progress_callback=progress_callback, payload=merged)

    try:
        emit({"phase": "preprocessing", "message": "Loading dataset and build runtime"})

        dataset = data or load_combined_dataset(enforce_integrity_counts=enforce_integrity_counts)
        if enforce_integrity_counts and len(dataset.unit_ids) != 2800:
            raise ValueError(f"Expected exact 2800-session corpus, got {len(dataset.unit_ids)}")

        source_hashes = _source_hashes(dataset)
        unit_ids_hash = _hash_unit_ids(dataset.unit_ids)

        cfg = azure_config or AzureConfig.from_env()
        if "AZURE_SEARCH_INDEX" not in os.environ:
            cfg = replace(cfg, search_index="trace-clusters-sampling-v6")

        tok = tokenizer or TiktokenTokenizer(model_name=V3_EMBEDDING_MODEL, encoding_name=V3_EMBEDDING_ENCODING)
        emb = embedder or AzureOpenAIEmbedder(cfg)

        if embeddings_cache_path is None or classifications_cache_path is None:
            d_emb, d_cls = _default_cache_paths(root)
            emb_cache = d_emb if embeddings_cache_path is None else Path(embeddings_cache_path)
            cls_cache = d_cls if classifications_cache_path is None else Path(classifications_cache_path)
        else:
            emb_cache = Path(embeddings_cache_path)
            cls_cache = Path(classifications_cache_path)

        if runtime is None:
            runtime, runtime_cache_stats = _build_runtime_with_cache(
                dataset,
                tokenizer=tok,
                embedder=emb,
                embedding_model_id=V3_EMBEDDING_MODEL,
                embedding_deployment_id=cfg.embedding_deployment,
                embedding_batch_size=max(1, int(embedding_batch_size)),
                embedding_dimensions=V3_EMBEDDING_DIMENSIONS,
                max_session_packet_tokens=V3_MAX_SESSION_PACKET_TOKENS,
                cache_base_path=emb_cache,
                source_hashes=source_hashes,
            )
        else:
            runtime_cache_stats = {
                "cache_hit": True,
                "cache_rows": len(runtime.embedding_records_by_content_sha256),
                "cache_packet_hash_count": len(runtime.embedding_records_by_content_sha256),
                "provenance": {"provided_runtime": True},
            }

        if use_case_artifacts is None:
            use_case_artifacts = load_business_use_case_artifacts(
                centroids_db_path=centroids_db_path,
                taxonomy_db_path=taxonomy_db_path,
            )
        clf = classifier or BusinessUseCaseClassifier(use_case_artifacts)

        classification_provenance = {
            "version": "sampling-v6-classification-cache-provenance-v1",
            "source_hashes": dict(source_hashes),
            "embedding_model_id": V3_EMBEDDING_MODEL,
            "embedding_deployment_id": cfg.embedding_deployment,
            "taxonomy_db_sha256": str(getattr(use_case_artifacts.metadata, "taxonomy_db_sha256", "")),
            "centroids_db_sha256": str(getattr(use_case_artifacts.metadata, "centroids_db_sha256", "")),
            "max_input_tokens": int(MAX_INPUT_TOKENS),
        }

        baseline_bundle: dict[str, Any] | None = None
        baseline_runs: list[dict[str, Any]] = []
        baseline_memberships: list[dict[str, Any]] = []
        baseline_seeds: list[int] = []
        baseline_trial_count = 0
        baseline_manifest: dict[str, Any] | None = None
        baseline_manifest_path: Path | None = None
        baseline_manifest_hash: str | None = None
        baseline_producer_hashes: dict[str, Any] | None = None
        baseline_agent_metrics: list[dict[str, Any]] = []
        classifications_by_unit: dict[str, _ClassificationRow] = {}
        classification_cache_stats: dict[str, Any] = {}

        vstore_factory = vector_store_factory or (
            lambda _tenant, _scope: _ResilientVectorStore(
                AzureSearchVectorStore(
                    cfg,
                    dim=V3_EMBEDDING_DIMENSIONS,
                    ensure_index=bool(ensure_search_index),
                ),
                max_attempts=3,
                retry_seconds=0.25,
            )
        )
        arm2_fn = arm2_selector or _select_arm2_exact_count

        selection_code_hashes = _selection_code_hashes()
        publisher_code_hashes = _publisher_code_hashes()
        compatibility_payload = _compatibility_payload(
            source_hashes=source_hashes,
            unit_ids_hash=unit_ids_hash,
            population_count=len(dataset.unit_ids),
            caps=caps_list,
            avg_tokens_per_session=int(avg_tokens_per_session),
            embedding_model=V3_EMBEDDING_MODEL,
            embedding_deployment=cfg.embedding_deployment,
            embedding_dimensions=V3_EMBEDDING_DIMENSIONS,
            embedding_tau=V3_DEFAULT_EMBEDDING_TAU,
            search_endpoint_host=_scrub_endpoint(cfg.search_endpoint),
            search_index=cfg.search_index,
            maven_meta={
                "taxonomy_version": getattr(use_case_artifacts.metadata, "taxonomy_version", ""),
                "taxonomy_db_sha256": str(getattr(use_case_artifacts.metadata, "taxonomy_db_sha256", "")),
                "centroids_db_sha256": str(getattr(use_case_artifacts.metadata, "centroids_db_sha256", "")),
            },
            idw_config=idw_config,
            selection_code_hashes=selection_code_hashes,
        )
        compatibility_fingerprint = _compatibility_fingerprint(compatibility_payload)

        if baseline_dir is not None:
            baseline_bundle = _load_baseline_bundle(Path(baseline_dir))
            baseline_seeds, _ = _validate_baseline_compatibility(
                baseline=baseline_bundle,
                expected_compatibility_payload=compatibility_payload,
                requested_new_seeds=requested_new_seeds,
            )
            baseline_runs = [_strip_label_fields_from_run_row(row) for row in baseline_bundle["runs"]]
            baseline_trial_count = len(baseline_seeds)
            baseline_manifest = dict(baseline_bundle["manifest"])
            baseline_manifest_path = Path(baseline_bundle["manifest_path"])
            baseline_manifest_hash = _sha256_file(baseline_manifest_path)
            baseline_producer_hashes = dict(((baseline_manifest.get("provenance") or {}).get("code_hashes") or {}))
            baseline_agent_metrics_path = Path(baseline_dir) / "agent_metrics.jsonl"
            if not baseline_agent_metrics_path.exists():
                raise ValueError("baseline artifact missing: agent_metrics.jsonl")
            baseline_agent_metrics = _read_jsonl(baseline_agent_metrics_path)
            baseline_classification_rows = {
                str(row.get("unit_id") or ""): _coerce_baseline_classification_row(row)
                for row in baseline_bundle["classifications"]
            }
            expected_unit_ids = {str(uid) for uid in dataset.unit_ids}
            if set(baseline_classification_rows) != expected_unit_ids:
                missing = sorted(expected_unit_ids - set(baseline_classification_rows))
                extra = sorted(set(baseline_classification_rows) - expected_unit_ids)
                raise ValueError(f"baseline classifications missing IDs or unexpected IDs: missing={missing[:5]} extra={extra[:5]}")
            classifications_by_unit = dict(baseline_classification_rows)
            classification_cache_stats = {
                "cache_hit": True,
                "source": "baseline",
                "rows": int(len(classifications_by_unit)),
                "embedding_calls": 0,
                "embedding_inputs": 0,
                "embedding_input_tokens": 0,
                "embedding_latency_seconds": 0.0,
                "elapsed_seconds": 0.0,
                "provenance": {
                    "version": classification_provenance["version"],
                    "source": "baseline",
                    "source_hashes": dict(classification_provenance["source_hashes"]),
                    "embedding_model_id": classification_provenance["embedding_model_id"],
                    "embedding_deployment_id": classification_provenance["embedding_deployment_id"],
                    "taxonomy_db_sha256": classification_provenance["taxonomy_db_sha256"],
                    "centroids_db_sha256": classification_provenance["centroids_db_sha256"],
                    "max_input_tokens": int(classification_provenance.get("max_input_tokens", 0) or 0),
                    "compatibility_fingerprint": compatibility_fingerprint,
                },
            }
        else:
            classifications_by_unit, classification_cache_stats = _classify_units_with_cache(
                data=dataset,
                classifier=clf,
                embedder=emb,
                tokenizer=tok,
                embedding_batch_size=max(1, int(embedding_batch_size)),
                cache_base_path=cls_cache,
                provenance=classification_provenance,
            )

        descriptor_rows: list[dict[str, Any]] = []
        for unit in dataset.units:
            unit_id = str(unit.unit_id or "")
            meta = dict(dataset.metadata_by_unit.get(unit_id) or {})
            cls_row = classifications_by_unit[unit_id]
            concept_key = "|".join(
                [
                    str(meta.get("corpus_id") or "unknown"),
                    str(meta.get("domain") or "unknown"),
                    str(meta.get("task") or "unknown"),
                    str(meta.get("difficulty") or "unknown"),
                ]
            )
            descriptor_rows.append(
                {
                    "unit_id": unit_id,
                    "agent_id": f"{unit.tenant_id}|{unit.agent_id}",
                    "use_case_id": cls_row.use_case_guid,
                    "concept_key": concept_key,
                    "label": bool(dataset.labels_by_unit[unit_id]),
                }
            )

        descriptors = build_session_descriptors(descriptor_rows)
        labels_by_unit = {d.unit_id: d.label for d in descriptors}
        agent_id_by_unit = {d.unit_id: d.agent_id for d in descriptors}
        descriptor_by_id = {d.unit_id: d for d in descriptors}
        agent_population_sizes: dict[str, int] = {}
        for descriptor in descriptors:
            agent_population_sizes[descriptor.agent_id] = agent_population_sizes.get(descriptor.agent_id, 0) + 1
        agents_with_at_least_3_count = int(sum(1 for count in agent_population_sizes.values() if count >= 3))
        total_floor_target = int(sum(min(3, count) for count in agent_population_sizes.values()))
        vector_by_unit = {
            unit_id: runtime.embedding_vector_by_trace_id[int(dataset.trace_by_unit_id[unit_id].trace_id)]
            for unit_id in dataset.unit_ids
        }

        if baseline_dir is not None:
            baseline_memberships = [
                _normalize_baseline_membership_row(
                    row=row,
                    descriptors_by_unit=descriptor_by_id,
                    descriptor_population=descriptors,
                )
                for row in baseline_bundle["memberships"]
            ]

        if len(set(requested_new_seeds)) != len(requested_new_seeds):
            raise ValueError("requested seeds contain duplicates")
        combined_seeds = sorted(set(baseline_seeds).union(requested_new_seeds))
        expected_final_rows = len(combined_seeds) * len(caps_list) * len(METHOD_ID_ORDER)
        progress_state["baseline_trials"] = int(baseline_trial_count)
        progress_state["final_trials"] = int(len(combined_seeds))
        progress_state["baseline_rows"] = int(len(baseline_runs))
        progress_state["final_rows"] = int(expected_final_rows)

        checkpoints_root = Path(checkpoint_dir) if checkpoint_dir is not None else (root / "checkpoints")

        run_rows: list[dict[str, Any]] = [dict(row) for row in baseline_runs]
        membership_rows: list[dict[str, Any]] = [dict(row) for row in baseline_memberships]
        agent_metric_rows: list[dict[str, Any]] = [dict(row) for row in baseline_agent_metrics]
        trial_metrics_rows: list[TrialMetrics] = []
        checkpoint_reused_cells = 0
        checkpoint_new_cells = 0

        for seed_index, seed in enumerate(requested_new_seeds):
            order = _stable_order(dataset.unit_ids, seed=seed, token="v6-order")
            for cap_index, cap in enumerate(caps_list):
                cap_eff = min(cap, len(descriptors))
                window_id = f"sampling-v6|seed={seed}|cap={cap_eff}"
                nominal_budget = int(cap_eff) * int(avg_tokens_per_session)
                current_replay = 1 + (seed_index * len(caps_list)) + cap_index
                progress_state["current_seed"] = seed
                progress_state["current_cap"] = cap_eff
                progress_state["current_method"] = METHOD_IDS["arm2"]
                progress_state["current_replay"] = current_replay
                emit({
                    "phase": "replay-setup",
                    "message": f"Running replay {current_replay}/{total_replays} for seed {seed} cap {cap_eff}",
                    "current_seed": seed,
                    "current_cap": cap_eff,
                    "current_method": METHOD_IDS["arm2"],
                    "replay_session_current": 0,
                    "replay_session_total": len(order),
                    "percent": round(((current_replay - 1) / max(1, total_replays)) * 100.0, 4),
                })

                if resume:
                    resumed = _load_resume_checkpoint(
                        checkpoint_root=checkpoints_root,
                        seed=seed,
                        cap=cap_eff,
                        compatibility_fingerprint=compatibility_fingerprint,
                    )
                    if resumed is not None:
                        resumed_runs, resumed_memberships, resumed_agent_metrics = resumed
                        run_rows.extend(resumed_runs)
                        membership_rows.extend(resumed_memberships)
                        agent_metric_rows.extend(resumed_agent_metrics)
                        checkpoint_reused_cells += 1
                        progress_state["completed_replays"] = current_replay
                        progress_state["completed_cells"] = int(progress_state.get("completed_cells", 0)) + len(METHOD_ID_ORDER)
                        emit({
                            "phase": "checkpoint-resume",
                            "message": f"Reused checkpoint for seed {seed} cap {cap_eff}",
                            "current_seed": seed,
                            "current_cap": cap_eff,
                            "current_method": None,
                            "completed_cells": int(progress_state.get("completed_cells", 0)),
                            "total_cells": total_cells,
                            "completed_replays": int(progress_state.get("completed_replays", 0)),
                            "total_replays": total_replays,
                            "baseline_trials": int(baseline_trial_count),
                            "final_trials": int(len(combined_seeds)),
                            "baseline_rows": int(len(baseline_runs)),
                            "final_rows": int(expected_final_rows),
                        })
                        continue

                arm1 = select_arm1(descriptors=descriptors, cap=cap_eff, trial_seed=seed, window_id=window_id)
                validate_selection_exactness(outcome=arm1, population=descriptors, cap=cap_eff)

                nested_progress: ProgressCallback | None = None
                if arm2_selector is None:
                    nested_progress = lambda payload: emit({
                        "phase": str(payload.get("phase", "replay")),
                        "message": str(payload.get("message", "Arm2 replay")),
                        "current_seed": seed,
                        "current_cap": cap_eff,
                        "current_method": METHOD_IDS["arm2"],
                        "current_replay": current_replay,
                        "replay_session_current": int(payload.get("replay_session_current", 0)),
                        "replay_session_total": int(payload.get("replay_session_total") or len(order)),
                        "percent": float(payload.get("percent", 0.0)),
                        "status": str(payload.get("status", "running")),
                    })
                else:
                    try:
                        sig = __import__("inspect").signature(arm2_selector)
                        params = sig.parameters
                        if "progress_callback" in params or any(p.kind == __import__("inspect").Parameter.VAR_KEYWORD for p in params.values()):
                            nested_progress = lambda payload, _seed=seed, _cap=cap_eff, _curr=current_replay: emit({
                                "phase": str(payload.get("phase", "replay")),
                                "message": str(payload.get("message", "Arm2 replay")),
                                "current_seed": _seed,
                                "current_cap": _cap,
                                "current_method": METHOD_IDS["arm2"],
                                "current_replay": _curr,
                                "replay_session_current": int(payload.get("replay_session_current", 0)),
                                "replay_session_total": int(payload.get("replay_session_total") or len(order)),
                                "percent": float(payload.get("percent", 0.0)),
                                "status": str(payload.get("status", "running")),
                            })
                    except (TypeError, ValueError):
                        nested_progress = None

                arm2_kwargs: dict[str, Any] = {
                    "data": dataset,
                    "runtime": runtime,
                    "cap": cap_eff,
                    "seed": seed,
                    "ordered_unit_ids": order,
                    "tenant_id": "sampling-v6-experiment",
                    "run_scope": f"seed-{seed}-cap-{cap_eff}",
                    "vector_store_factory": vstore_factory,
                    "cleanup_max_attempts": cleanup_max_attempts,
                    "cleanup_settle_seconds": cleanup_settle_seconds,
                }
                if nested_progress is not None:
                    arm2_kwargs["progress_callback"] = nested_progress
                arm2 = arm2_fn(**arm2_kwargs)
                progress_state["completed_replays"] = current_replay
                if len(arm2.selected_ids) != cap_eff:
                    raise AssertionError("arm2 must produce exact cap selection")

                arm2_idw = run_arm2_idw(
                    eligible_ids=dataset.unit_ids,
                    selected_ids=arm2.selected_ids,
                    agent_id_by_unit=agent_id_by_unit,
                    vector_by_unit=vector_by_unit,
                    labels_by_unit=labels_by_unit,
                    cell_id=f"v6-arm2-{seed}-{cap_eff}",
                    config=idw_config,
                )

                arm3 = select_arm3(descriptors=descriptors, cap=cap_eff, trial_seed=seed, window_id=window_id)
                validate_selection_exactness(outcome=arm3, population=descriptors, cap=cap_eff)

                arm4 = select_arm4(descriptors=descriptors, cap=cap_eff, trial_seed=seed, window_id=window_id)
                validate_selection_exactness(outcome=arm4, population=descriptors, cap=cap_eff)

                arm5 = select_arm5(
                    descriptors=descriptors,
                    arm4_outcome=arm4,
                    labels_by_unit=labels_by_unit,
                    trial_seed=seed,
                    window_id=window_id,
                )
                arm6 = select_arm6(descriptors=descriptors, arm4_outcome=arm4)
                if arm4.selected_ids != arm5.selected_ids:
                    raise AssertionError("arm4 and arm5 membership must be identical")
                if arm4.selected_ids != arm6.selected_ids:
                    raise AssertionError("arm4 and arm6 membership must be identical")

                arm2_outcome = SelectionOutcome(
                    method_id=METHOD_IDS["arm2"],
                    selected_ids=arm2.selected_ids,
                    records=arm2.records,
                )
                arm2_5 = run_arm2_5_binary_from_arm2_result(arm2_result=arm2_idw)
                arm2_5_outcome = SelectionOutcome(
                    method_id=METHOD_IDS["arm2_5"],
                    selected_ids=arm2.selected_ids,
                    records=tuple(
                        SelectionRecord(
                            unit_id=rec.unit_id,
                            method_id=METHOD_IDS["arm2_5"],
                            stratum=rec.stratum,
                            inclusion_probability=rec.inclusion_probability,
                            weight=rec.weight,
                            reason="arm2_5-thresholded-reuse-arm2-membership",
                        )
                        for rec in arm2.records
                    ),
                )
                if tuple(arm2_outcome.selected_ids) != tuple(arm2_5_outcome.selected_ids):
                    raise AssertionError("arm2 and arm2.5 membership must be identical")

                arm_map: list[tuple[str, SelectionOutcome, dict[str, Any] | None]] = [
                    (METHOD_IDS["arm1"], arm1, None),
                    (METHOD_IDS["arm2"], arm2_outcome, arm2_idw),
                    (METHOD_IDS["arm2_5"], arm2_5_outcome, arm2_5),
                    (METHOD_IDS["arm3"], arm3, None),
                    (METHOD_IDS["arm4"], arm4, None),
                    (METHOD_IDS["arm5"], arm5, None),
                    (METHOD_IDS["arm6"], arm6, None),
                ]

                cell_run_rows: list[dict[str, Any]] = []
                cell_membership_rows: list[dict[str, Any]] = []
                cell_agent_metric_rows: list[dict[str, Any]] = []
                arm2_run_row: dict[str, Any] | None = None
                for method_id, outcome, idw_result in arm_map:
                    progress_state["current_method"] = method_id
                    progress_state["completed_cells"] = int(progress_state.get("completed_cells", 0)) + 1
                    emit({
                        "phase": "method-evaluation",
                        "message": f"Evaluating {method_id} for seed {seed} cap {cap_eff}",
                        "current_seed": seed,
                        "current_cap": cap_eff,
                        "current_method": method_id,
                        "completed_cells": int(progress_state.get("completed_cells", 0)),
                        "total_cells": total_cells,
                        "percent": round((int(progress_state.get("completed_cells", 0)) / max(1, total_cells)) * 100.0, 4),
                    })
                    actual_tokens = int(sum(runtime.token_cost_by_unit_id[uid] for uid in outcome.selected_ids))
                    metric = compute_trial_metrics(
                        descriptors=descriptors,
                        selected_ids=outcome.selected_ids,
                        method_id=method_id,
                        trial_seed=seed,
                        window_id=window_id,
                        nominal_budget=nominal_budget,
                        labels_by_unit=labels_by_unit,
                        actual_token_count=actual_tokens,
                        idw_result=idw_result,
                        arm4_outcome=arm4 if method_id == METHOD_IDS["arm5"] else None,
                        selection_outcome=outcome if method_id == METHOD_IDS["arm6"] else None,
                    )
                    trial_metrics_rows.append(metric)

                    membership_hash = _sha256_text(
                        _canonical_json(
                            {
                                "method_id": method_id,
                                "seed": seed,
                                "cap": cap_eff,
                                "selected_ids": list(outcome.selected_ids),
                            }
                        )
                    )

                    selection_records = [_serialize_selection_record(rec) for rec in outcome.records]
                    selection_records = [
                        {
                            **record,
                            "agent_id": descriptor_by_id[record["unit_id"]].agent_id,
                            "use_case_id": descriptor_by_id[record["unit_id"]].use_case_id,
                        }
                        for record in selection_records
                    ]
                    represented_strata = sorted(
                        {
                            descriptor_by_id[unit_id].use_case_id
                            for unit_id in outcome.selected_ids
                            if unit_id in descriptor_by_id
                        }
                    )
                    selected_agent_count = len(
                        {
                            descriptor_by_id[unit_id].agent_id
                            for unit_id in outcome.selected_ids
                            if unit_id in descriptor_by_id
                        }
                    )
                    if method_id == METHOD_IDS["arm3"]:
                        selected_agent_counts = {
                            agent_id: 0 for agent_id in sorted({descriptor.agent_id for descriptor in descriptors})
                        }
                        for unit_id in outcome.selected_ids:
                            descriptor = descriptor_by_id.get(unit_id)
                            if descriptor is None:
                                continue
                            selected_agent_counts[descriptor.agent_id] = selected_agent_counts.get(descriptor.agent_id, 0) + 1
                        selected_agent_count = len([agent_id for agent_id, count in selected_agent_counts.items() if count > 0])
                        agents_with_at_least_3 = int(sum(1 for count in selected_agent_counts.values() if count >= 3))
                        total_agent_count = len(agent_population_sizes)
                        agent_coverage = (selected_agent_count / total_agent_count) if total_agent_count else 1.0
                    else:
                        agents_with_at_least_3 = int(sum(1 for count in agent_population_sizes.values() if count >= 3))
                        total_agent_count = len(agent_population_sizes)
                        agent_coverage = (selected_agent_count / total_agent_count) if total_agent_count else 1.0

                    membership_row = {
                        "version": "sampling-v6-membership-v1",
                        "method_id": method_id,
                        "seed": seed,
                        "cap": cap_eff,
                        "membership_hash": membership_hash,
                        "selected_ids": list(outcome.selected_ids),
                        "selection_records": selection_records,
                        "selected_agent_count": int(selected_agent_count),
                        "total_agent_count": int(total_agent_count),
                        "agents_with_at_least_3": int(agents_with_at_least_3),
                        "eligible_agents_with_at_least_3": int(sum(1 for count in agent_population_sizes.values() if count >= 3)),
                        "agent_coverage": float(agent_coverage),
                        "represented_strata": represented_strata,
                    }
                    if method_id == METHOD_IDS["arm3"]:
                        floor_prefix_limit = min(int(total_floor_target), len(outcome.selected_ids))
                        floor_prefix_count = 0
                        per_agent_prefix: dict[str, int] = {}
                        for unit_id in list(outcome.selected_ids)[:floor_prefix_limit]:
                            descriptor = descriptor_by_id.get(unit_id)
                            if descriptor is None:
                                continue
                            agent_id = descriptor.agent_id
                            target = min(3, int(agent_population_sizes.get(agent_id, 0)))
                            cur = per_agent_prefix.get(agent_id, 0)
                            if cur < target:
                                per_agent_prefix[agent_id] = cur + 1
                                floor_prefix_count += 1

                        floor_complete = floor_prefix_count >= int(total_floor_target)
                        floor_completion_ratio = (floor_prefix_count / int(total_floor_target)) if int(total_floor_target) > 0 else 1.0
                        membership_row.update(
                            {
                                "total_floor_target": int(total_floor_target),
                                "floor_complete": bool(floor_complete),
                                "floor_prefix_count": int(floor_prefix_count),
                                "floor_completion_ratio": float(floor_completion_ratio),
                                "arm3_floor": {
                                    "total_floor_target": int(total_floor_target),
                                    "floor_complete": bool(floor_complete),
                                    "floor_prefix_count": int(floor_prefix_count),
                                    "floor_completion_ratio": float(floor_completion_ratio),
                                },
                            }
                        )
                    if method_id == METHOD_IDS["arm4"]:
                        membership_row["inclusion_probability_by_unit"] = {
                            str(rec.unit_id): _finite_float_or_none(rec.inclusion_probability)
                            for rec in arm4.records
                        }
                    if method_id == METHOD_IDS["arm5"]:
                        membership_row["inclusion_probability_by_unit"] = {
                            str(rec.unit_id): _finite_float_or_none(rec.inclusion_probability)
                            for rec in arm5.records
                        }
                    if method_id == METHOD_IDS["arm6"]:
                        membership_row["inclusion_probability_by_unit"] = {
                            str(rec.unit_id): _finite_float_or_none(rec.inclusion_probability)
                            for rec in arm6.records
                        }
                        membership_row["arm6_diagnostics"] = _sanitize_arm6_diagnostics_for_membership_row(outcome.diagnostics)
                        membership_row["arm6_estimator_limitation"] = "realized represented-cell post-stratification; zero-sample cells are not recovered"
                    membership_rows.append(membership_row)
                    cell_membership_rows.append(dict(membership_row))

                    run_row = {
                        "version": "sampling-v6-run-v1",
                        "method_id": method_id,
                        "seed": seed,
                        "cap": cap_eff,
                        "window_id": window_id,
                        "nominal_budget": nominal_budget,
                        "membership_hash": membership_hash,
                        "membership_ref": "memberships.jsonl",
                        **_sanitize_for_run_row(metric),
                    }
                    if method_id == METHOD_IDS["arm2"]:
                        run_row["arm2_telemetry"] = dict(arm2.telemetry)
                        run_row["idw_validation"] = _sanitize_idw_validation(run_row.get("idw_validation"))
                        run_row["idw_provenance"] = _sanitize_arm2_idw_provenance(idw_result)
                        run_row["idw_quality"] = _sanitize_arm2_idw_quality(run_row.get("idw_validation"))
                    if method_id == METHOD_IDS["arm2_5"]:
                        run_row["arm2_telemetry"] = dict(arm2.telemetry)
                        run_row["idw_provenance"] = _sanitize_arm2_idw_provenance(arm2_idw)
                        run_row["continuous_idw_diagnostic"] = _sanitize_arm2_idw_quality(_sanitize_idw_validation(arm2_idw.get("validation")))
                        run_row["estimator_diagnostics"] = _arm2_5_estimator_diagnostics(arm2_5)
                        run_row["idw_validation"] = {
                            "continuous_donor_model_diagnostic": _sanitize_arm2_idw_quality(_sanitize_idw_validation(arm2_idw.get("validation")))
                        }
                    if method_id == METHOD_IDS["arm6"]:
                        run_row["estimator_diagnostics"] = _sanitize_arm6_diagnostics_for_run_row(outcome.diagnostics)
                        run_row["joint_cell_probability_by_unit"] = {
                            str(rec.unit_id): _finite_float_or_none(rec.inclusion_probability)
                            for rec in arm6.records
                        }
                    run_row = _strip_label_fields_from_run_row(run_row)
                    run_rows.append(run_row)
                    cell_run_rows.append(dict(run_row))
                    if method_id == METHOD_IDS["arm2"]:
                        arm2_run_row = dict(run_row)

                    selected_set = set(outcome.selected_ids)
                    all_population_use_cases_by_agent: dict[str, set[str]] = {}
                    all_population_concepts_by_agent: dict[str, set[str]] = {}
                    represented_use_cases_by_agent: dict[str, set[str]] = {}
                    represented_concepts_by_agent: dict[str, set[str]] = {}
                    selected_ids_by_agent: dict[str, list[str]] = {}
                    all_ids_by_agent: dict[str, list[str]] = {}
                    for descriptor in descriptors:
                        aid = descriptor.agent_id
                        all_population_use_cases_by_agent.setdefault(aid, set()).add(descriptor.use_case_id)
                        all_population_concepts_by_agent.setdefault(aid, set()).add(descriptor.concept_key)
                        all_ids_by_agent.setdefault(aid, []).append(descriptor.unit_id)
                        if descriptor.unit_id in selected_set:
                            represented_use_cases_by_agent.setdefault(aid, set()).add(descriptor.use_case_id)
                            represented_concepts_by_agent.setdefault(aid, set()).add(descriptor.concept_key)
                            selected_ids_by_agent.setdefault(aid, []).append(descriptor.unit_id)

                    unit_estimate_by_unit: dict[str, float] = {}
                    if method_id in {METHOD_IDS["arm2"], METHOD_IDS["arm2_5"]}:
                        source = arm2_idw if method_id == METHOD_IDS["arm2"] else arm2_5
                        estimated_population = source.get("estimated_population")
                        for estimate_row in tuple(getattr(estimated_population, "rows", ()) or ()):
                            parsed = _unit_estimate_fields(estimate_row)
                            if parsed is None:
                                continue
                            uid, value = parsed
                            unit_estimate_by_unit[uid] = (1.0 if value >= 0.5 else 0.0) if method_id == METHOD_IDS["arm2_5"] else float(value)

                    for agent_id in sorted(all_ids_by_agent):
                        population_agent_ids = all_ids_by_agent[agent_id]
                        selected_agent_ids = selected_ids_by_agent.get(agent_id, [])
                        N = len(population_agent_ids)
                        n = len(selected_agent_ids)
                        census_rate = float(
                            sum(1.0 if bool(labels_by_unit.get(uid, False)) else 0.0 for uid in population_agent_ids)
                            / max(1, N)
                        )

                        estimator_label = "selected_mean"
                        estimate: float | None
                        represented_population_fraction = 1.0
                        if method_id in {METHOD_IDS["arm1"], METHOD_IDS["arm3"], METHOD_IDS["arm4"]}:
                            estimate = None if n == 0 else float(
                                sum(1.0 if bool(labels_by_unit.get(uid, False)) else 0.0 for uid in selected_agent_ids) / n
                            )
                            represented_population_fraction = 1.0 if n > 0 else 0.0
                        elif method_id == METHOD_IDS["arm2"]:
                            estimator_label = "idw_continuous_mean_full_population"
                            vals = [unit_estimate_by_unit[uid] for uid in population_agent_ids if uid in unit_estimate_by_unit]
                            estimate = None if len(vals) != N else float(sum(vals) / max(1, len(vals)))
                            represented_population_fraction = 1.0
                        elif method_id == METHOD_IDS["arm2_5"]:
                            estimator_label = "idw_thresholded_binary_mean_full_population"
                            vals = [unit_estimate_by_unit[uid] for uid in population_agent_ids if uid in unit_estimate_by_unit]
                            estimate = None if len(vals) != N else float(sum(vals) / max(1, len(vals)))
                            represented_population_fraction = 1.0
                        elif method_id == METHOD_IDS["arm5"]:
                            estimator_label = "hajek_with_arm4_probabilities"
                            pi = {str(rec.unit_id): float(rec.inclusion_probability or 0.0) for rec in arm5.records}
                            estimate = _hajek_ratio_estimate(selected_agent_ids, labels_by_unit, pi)
                            represented_population_fraction = 1.0 if n > 0 else 0.0
                        elif method_id == METHOD_IDS["arm6"]:
                            estimator_label = "hajek_represented_joint_agent_use_case_cells"
                            pi = {str(rec.unit_id): float(rec.inclusion_probability or 0.0) for rec in arm6.records}
                            estimate = _hajek_ratio_estimate(selected_agent_ids, labels_by_unit, pi)
                            represented_cells = {
                                (descriptor_by_id[uid].agent_id, descriptor_by_id[uid].business_use_case_guid)
                                for uid in outcome.selected_ids
                                if uid in descriptor_by_id
                            }
                            represented_pop = 0
                            for uid in population_agent_ids:
                                descriptor = descriptor_by_id[uid]
                                if (descriptor.agent_id, descriptor.business_use_case_guid) in represented_cells:
                                    represented_pop += 1
                            represented_population_fraction = float(represented_pop / max(1, N))
                        else:
                            estimate = None if n == 0 else float(
                                sum(1.0 if bool(labels_by_unit.get(uid, False)) else 0.0 for uid in selected_agent_ids) / n
                            )
                            represented_population_fraction = 1.0 if n > 0 else 0.0

                        concept_cov = float(
                            len(represented_concepts_by_agent.get(agent_id, set())) / max(1, len(all_population_concepts_by_agent.get(agent_id, set())))
                        )
                        use_case_cov = float(
                            len(represented_use_cases_by_agent.get(agent_id, set())) / max(1, len(all_population_use_cases_by_agent.get(agent_id, set())))
                        )
                        abs_error = None if estimate is None else float(abs(float(estimate) - census_rate))

                        agent_row = {
                            "version": V6_AGENT_METRICS_VERSION,
                            "method_id": method_id,
                            "seed": int(seed),
                            "cap": int(cap_eff),
                            "agent_id": agent_id,
                            "N": int(N),
                            "n": int(n),
                            "estimate": estimate,
                            "census_rate": census_rate,
                            "absolute_error": abs_error,
                            "concept_coverage": concept_cov,
                            "use_case_coverage": use_case_cov,
                            "estimator": estimator_label,
                            "represented_population_fraction": float(represented_population_fraction),
                        }
                        agent_row = _strip_label_fields_from_run_row(agent_row)
                        agent_metric_rows.append(agent_row)
                        cell_agent_metric_rows.append(dict(agent_row))

                arm4_membership = next((row for row in cell_membership_rows if row.get("method_id") == METHOD_IDS["arm4"]), None)
                arm5_membership = next((row for row in cell_membership_rows if row.get("method_id") == METHOD_IDS["arm5"]), None)
                arm6_membership = next((row for row in cell_membership_rows if row.get("method_id") == METHOD_IDS["arm6"]), None)
                if arm4_membership and arm5_membership:
                    if arm4_membership["method_id"] == METHOD_IDS["arm4"] and arm5_membership["method_id"] == METHOD_IDS["arm5"]:
                        if tuple(arm4_membership["selected_ids"]) != tuple(arm5_membership["selected_ids"]):
                            raise AssertionError("arm4 and arm5 selected_ids must match exactly")
                        if dict(arm4_membership.get("inclusion_probability_by_unit") or {}) != dict(
                            arm5_membership.get("inclusion_probability_by_unit") or {}
                        ):
                            raise AssertionError("arm4 and arm5 inclusion probabilities must match exactly")
                if arm4_membership and arm6_membership:
                    if tuple(arm4_membership["selected_ids"]) != tuple(arm6_membership["selected_ids"]):
                        raise AssertionError("arm4 and arm6 selected_ids must match exactly")

                _write_cell_checkpoint(
                    checkpoint_root=checkpoints_root,
                    seed=seed,
                    cap=cap_eff,
                    window_id=window_id,
                    run_scope=f"seed-{seed}-cap-{cap_eff}",
                    run_rows=cell_run_rows,
                    membership_rows=cell_membership_rows,
                    compatibility_payload=compatibility_payload,
                    compatibility_fingerprint=compatibility_fingerprint,
                    source_hashes=source_hashes,
                    selection_code_hashes=selection_code_hashes,
                    publisher_code_hashes=publisher_code_hashes,
                    arm2_telemetry=dict((arm2_run_row or {}).get("arm2_telemetry") or {}),
                    agent_metric_rows=cell_agent_metric_rows,
                )
                checkpoint_new_cells += 1

        expected_agent_metric_rows = expected_final_rows * len(agent_population_sizes)
        if len(run_rows) != expected_final_rows:
            raise AssertionError("run row count mismatch")
        if len(membership_rows) != expected_final_rows:
            raise AssertionError("membership row count mismatch")
        if len(agent_metric_rows) != expected_agent_metric_rows:
            raise AssertionError("agent metric row count mismatch")
        _validate_unique_cell_grid(run_rows)

        classifications_rows = [
            {
                "version": "sampling-v6-classification-v1",
                "unit_id": row.unit_id,
                "agent_id": row.agent_id,
                "concept_key": row.concept_key,
                "corpus_id": row.corpus_id,
                "use_case_guid": row.use_case_guid,
                "domain": row.domain,
                "segment": row.segment,
                "category": row.category,
                "sub_category": row.sub_category,
                "sub_subcategory": row.sub_subcategory,
                "business_task": row.business_task,
                "status": row.status,
                "confidence_level": row.confidence_level,
                "combined_cosine_similarity": row.combined_cosine_similarity,
                "selected_step_index": row.selected_step_index,
                "selected_step_provenance": row.selected_step_provenance,
            }
            for row in [classifications_by_unit[uid] for uid in dataset.unit_ids]
        ]

        aggregate_rows = _aggregate_trial_rows(run_rows)
        ranking = sorted(
            [
                {
                    "method_id": row["method_id"],
                    "cap": row["cap"],
                    "mae_mean": row["mae"]["mean"],
                    "concept_coverage_mean": row["concept_coverage"]["mean"],
                    "maven_coverage_mean": row["maven_coverage"]["mean"],
                    "agent_coverage_mean": row["agent_coverage"]["mean"],
                }
                for row in aggregate_rows
            ],
            key=lambda row: (row["mae_mean"], -row["concept_coverage_mean"], row["method_id"], row["cap"]),
        )

        aggregate = {
            "version": V6_BUNDLE_VERSION,
            "generated_at": _iso_now_utc(),
            "population_audit": {
                "unit_count": len(dataset.unit_ids),
                "label_count": len(dataset.labels_by_unit),
                "agent_count": len({d.agent_id for d in descriptors}),
                "source_hashes": dict(source_hashes),
                "unit_ids_hash": unit_ids_hash,
                "integrity_enforced": bool(enforce_integrity_counts),
            },
            "config": {
                "caps": [int(x) for x in caps_list],
                "seeds": [int(x) for x in combined_seeds],
                "requested_extension_seeds": [int(x) for x in requested_new_seeds],
                "baseline_seeds": [int(x) for x in baseline_seeds],
                "avg_tokens_per_session": int(avg_tokens_per_session),
                "embedding_batch_size": int(embedding_batch_size),
                "cleanup_max_attempts": int(cleanup_max_attempts),
                "cleanup_settle_seconds": float(cleanup_settle_seconds),
                "ensure_search_index": bool(ensure_search_index),
                "skip_report": bool(skip_report),
                "idw": asdict(idw_config),
            },
            "azure": {
                "openai_endpoint_host": str(_scrub_endpoint(cfg.openai_endpoint)),
                "search_endpoint_host": str(_scrub_endpoint(cfg.search_endpoint)),
                "search_index": cfg.search_index,
                "embedding_deployment": cfg.embedding_deployment,
                "embedding_model": V3_EMBEDDING_MODEL,
                "embedding_dimensions": V3_EMBEDDING_DIMENSIONS,
                "embedding_tau": V3_DEFAULT_EMBEDDING_TAU,
            },
            "maven_artifacts": {
                "taxonomy_version": getattr(use_case_artifacts.metadata, "taxonomy_version", ""),
                "dimensions": int(getattr(use_case_artifacts.metadata, "dimensions", 0)),
                "taxonomy_count": int(getattr(use_case_artifacts.metadata, "taxonomy_count", 0)),
                "request_centroid_count": int(getattr(use_case_artifacts.metadata, "request_centroid_count", 0)),
                "response_centroid_count": int(getattr(use_case_artifacts.metadata, "response_centroid_count", 0)),
                "taxonomy_db_path": _basename_or_empty(getattr(use_case_artifacts.metadata, "taxonomy_db_path", "")),
                "taxonomy_db_source_kind": "local-file" if str(getattr(use_case_artifacts.metadata, "taxonomy_db_path", "")).strip() else "unknown",
                "taxonomy_db_sha256": str(getattr(use_case_artifacts.metadata, "taxonomy_db_sha256", "")),
                "centroids_db_path": _basename_or_empty(getattr(use_case_artifacts.metadata, "centroids_db_path", "")),
                "centroids_db_source_kind": "local-file" if str(getattr(use_case_artifacts.metadata, "centroids_db_path", "")).strip() else "unknown",
                "centroids_db_sha256": str(getattr(use_case_artifacts.metadata, "centroids_db_sha256", "")),
            },
            "embedding_ledgers": {
                "runtime": {
                    "packet_builds": int(runtime.ledger.packet_builds),
                    "packet_cache_hits": int(runtime.ledger.packet_cache_hits),
                    "embedding_calls": int(runtime.ledger.embedding_calls),
                    "embedding_inputs": int(runtime.ledger.embedding_inputs),
                    "embedding_input_tokens": int(runtime.ledger.embedding_input_tokens),
                    "embedding_latency_seconds": float(runtime.ledger.embedding_latency_seconds),
                    "embedding_content_hash_count": len(runtime.ledger.embedding_content_hashes),
                    "embedding_model_id": runtime.ledger.embedding_model_id,
                    "embedding_deployment_id": runtime.ledger.embedding_deployment_id,
                    "embedding_embedder_class": runtime.ledger.embedding_embedder_class,
                },
                "runtime_cache": dict(runtime_cache_stats),
                "classification_cache": dict(classification_cache_stats),
            },
            "provenance": {
                "extension_compatibility_fingerprint": compatibility_fingerprint,
                "selection_code_hashes": selection_code_hashes,
                "publisher_code_hashes": publisher_code_hashes,
                "code_hashes": publisher_code_hashes,
                "trial_cohorts": [
                    {
                        "cohort": "baseline",
                        "seeds": [int(x) for x in baseline_seeds],
                        "trial_count": int(baseline_trial_count),
                        "row_count": int(len(baseline_runs)),
                        "parent_manifest_path": None if baseline_manifest_path is None else str(baseline_manifest_path),
                        "parent_manifest_sha256": baseline_manifest_hash,
                        "producer_code_hashes": dict(baseline_producer_hashes or {}),
                    },
                    {
                        "cohort": "extension",
                        "seeds": [int(x) for x in requested_new_seeds],
                        "trial_count": int(len(requested_new_seeds)),
                        "row_count": int(total_cells),
                        "compatibility_fingerprint": compatibility_fingerprint,
                        "producer_code_hashes": dict(publisher_code_hashes),
                    },
                ],
                "checkpoint_summary": {
                    "total_cells": int(total_replays),
                    "reused_cells": int(checkpoint_reused_cells),
                    "new_cells": int(checkpoint_new_cells),
                },
            },
            "aggregate_rows": aggregate_rows,
            "rankings": ranking,
            "top_five_agent_aggregate": _aggregate_top_agents(run_rows),
        }

        dataset_examples = _build_dataset_examples(dataset, source_hashes)

        methodology = _methodology_md()

        aggregate_path = root / "aggregate.json"
        runs_path = root / "runs.jsonl"
        memberships_path = root / "memberships.jsonl"
        classifications_path = root / "classifications.jsonl"
        agent_metrics_path = root / "agent_metrics.jsonl"
        examples_path = root / "dataset_examples.json"
        methodology_path = root / "methodology.md"

        _write_json_atomic(aggregate_path, aggregate)
        _write_jsonl_atomic(runs_path, run_rows)
        _write_jsonl_atomic(memberships_path, membership_rows)
        _write_jsonl_atomic(classifications_path, classifications_rows)
        _write_jsonl_atomic(agent_metrics_path, agent_metric_rows)
        _write_json_atomic(examples_path, dataset_examples)
        _write_text_atomic(methodology_path, methodology)

        manifest_path = root / "manifest.json"
        manifest = {
            "version": V6_MANIFEST_VERSION,
            "generated_at": _iso_now_utc(),
            "artifacts": {
                "aggregate": _artifact_meta(aggregate_path, root),
                "runs": _artifact_meta(runs_path, root),
                "memberships": _artifact_meta(memberships_path, root),
                "classifications": _artifact_meta(classifications_path, root),
                "agent_metrics": _artifact_meta(agent_metrics_path, root),
                "dataset_examples": _artifact_meta(examples_path, root),
                "methodology": _artifact_meta(methodology_path, root),
            },
            "provenance": {
                "publisher_code_hashes": publisher_code_hashes,
                "code_hashes": publisher_code_hashes,
                "trial_cohorts": aggregate["provenance"]["trial_cohorts"],
                "checkpoint_summary": aggregate["provenance"]["checkpoint_summary"],
                "compatibility_fingerprint": compatibility_fingerprint,
                "dataset_source_hashes": source_hashes,
                "runtime_cache_provenance": runtime_cache_stats.get("provenance"),
                "classification_cache_provenance": classification_cache_stats.get("provenance"),
            },
        }
        _write_json_atomic(manifest_path, manifest)

        emit({
            "status": "complete",
            "phase": "complete",
            "message": "Sampling V6 bundle complete",
            "current_seed": None,
            "current_cap": None,
            "current_method": None,
            "completed_replays": total_replays,
            "total_replays": total_replays,
            "completed_cells": total_cells,
            "total_cells": total_cells,
            "baseline_trials": int(baseline_trial_count),
            "final_trials": int(len(combined_seeds)),
            "baseline_rows": int(len(baseline_runs)),
            "final_rows": int(expected_final_rows),
            "replay_session_current": 0,
            "replay_session_total": 0,
            "percent": 100.0,
        })

        return {
            "aggregate": aggregate,
            "runs": run_rows,
            "memberships": membership_rows,
            "classifications": classifications_rows,
            "agent_metrics": agent_metric_rows,
            "output_paths": {
                "aggregate": str(aggregate_path),
                "runs": str(runs_path),
                "memberships": str(memberships_path),
                "classifications": str(classifications_path),
                "agent_metrics": str(agent_metrics_path),
                "dataset_examples": str(examples_path),
                "methodology": str(methodology_path),
                "manifest": str(manifest_path),
            },
        }
    except Exception as exc:
        kind, message = _safe_progress_error(exc)
        emit({
            "status": "failed",
            "phase": progress_state.get("phase", "failed"),
            "message": f"Sampling V6 bundle failed: {message}",
            "error_type": kind,
            "error_message": message,
            "percent": min(100.0, max(0.0, float(progress_state.get("percent", 0.0)))),
        })
        raise


def default_output_dir() -> Path:
    return Path("outputs_sampling_v6") / "runs" / _utc_stamp()
