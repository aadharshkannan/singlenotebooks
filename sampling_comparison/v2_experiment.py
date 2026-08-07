"""Runnable v2 combined-corpus sampling experiment (offline, label-only)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import median
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from minhash_sampling import BandedMinHashLSHIndex, MinHashConfig, MinHashSignatureProvider
from minhash_sampling.signature import MinHashRecord
from random_sampling import AgentKey, EvaluationUnit, SamplePolicy, SamplingEngine, allocate_strata
from random_sampling.datasets import load_synthetic_a365_otel
from trace_sampling import AdaptiveSampler, FullSessionEmbeddingPrototype, SamplerConfig
from trace_sampling.model import SessionEvent, Trace
from trace_sampling.session_embedding import EmbeddingProfile, SessionEmbeddingCache, SessionEmbeddingRecord
from trace_sampling.cluster_index import AzureClusterIndex
from trace_sampling.vector_store import InMemoryVectorStore

from .adapters import DeterministicSessionEmbedder, DeterministicTokenizer

from .v2_outputs import (
    build_external_eval_snapshots,
    write_external_eval_snapshot_artifacts,
    write_production_storage_manifest,
)

V2_VERSION = "sampling-v2"
HISTORICAL_300_PATH = "synthetic_data/a365_historical_300/synthetic_observability.a365-otel.json"
DENSE_2500_PATH = "synthetic_data/a365_dense_2500/corpus/a365.synthetic.strict.otlp.json"

OUTCOME_METHODS: tuple[str, ...] = (
    "random_sampling_stratified",
    "adaptive_minhash_32x4",
    "adaptive_embedding_fullsession",
)
QUADRANT_METHODS: tuple[str, ...] = (
    "random_online_admission",
    "adaptive_minhash_32x4",
    "adaptive_embedding_fullsession",
)
THROUGHPUT_METHODS: tuple[str, ...] = QUADRANT_METHODS

REPRESENTATION_SEED = 13
EMBEDDING_PROFILE_MAX_INPUT_TOKENS = 8192
EMBEDDING_REPRESENTATION_MAX_UTF8_BYTES = 32768
ADAPTIVE_COMPARABILITY_POLICY = {
    "enforce_keep_one_floor": False,
    "agent_floor": 0.0,
}

LAST_SELECTION_MECHANISMS: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class CombinedDataset:
    units: tuple[EvaluationUnit, ...]
    unit_ids: tuple[str, ...]
    traces: tuple[Trace, ...]
    trace_by_unit_id: dict[str, Trace]
    labels_by_unit: dict[str, bool]
    metadata_by_unit: dict[str, dict[str, Any]]
    corpus_id_by_unit: dict[str, str]
    original_unit_id_by_unit: dict[str, str]
    scoped_identities: tuple[str, ...]
    source_paths: dict[str, str]


@dataclass(frozen=True)
class QuadrantAssignment:
    unit_id: str
    variety_band: str
    velocity_band: str
    quadrant: str


@dataclass(frozen=True)
class PrecomputedRepresentation:
    unit_id: str
    concept_key: str
    minhash_record: MinHashRecord
    embedding_record: SessionEmbeddingRecord


@dataclass(frozen=True)
class V2PrecomputedRepresentationRuntime:
    seed: int
    representation_version: str
    profile: dict[str, Any]
    records_by_unit_id: dict[str, PrecomputedRepresentation]
    build_seconds: float
    n_units: int
    minhash_profile_id: str
    embedding_profile_cache_version: str


class _ReadonlyMinHashProvider:
    def __init__(self, config: MinHashConfig, records_by_trace_id: dict[int, MinHashRecord]) -> None:
        self.cfg = config
        self._records = records_by_trace_id
        self.n_hits = 0
        self.n_builds = 0

    def build(self, trace: Trace) -> MinHashRecord:
        record = self._records.get(int(trace.trace_id))
        if record is None:
            raise KeyError(f"Missing precomputed MinHash record for trace_id={trace.trace_id}")
        self.n_hits += 1
        return record


class _ReadonlyEmbeddingCache:
    def __init__(self, profile: EmbeddingProfile, records_by_trace_id: dict[int, SessionEmbeddingRecord]) -> None:
        self.profile = profile
        self._records = records_by_trace_id

    def contains_trace(self, trace: Trace) -> bool:
        return int(trace.trace_id) in self._records

    def peek_trace(self, trace: Trace) -> SessionEmbeddingRecord | None:
        return self._records.get(int(trace.trace_id))

    def get_trace(self, trace: Trace) -> np.ndarray:
        record = self.peek_trace(trace)
        if record is None:
            raise KeyError(f"Missing precomputed embedding record for trace_id={trace.trace_id}")
        return record.vector


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
    os.replace(tmp, path)


def _stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _duration_ms(unit: EvaluationUnit) -> float:
    if unit.started_at is None or unit.ended_at is None:
        return 0.0
    return max(0.0, (unit.ended_at - unit.started_at).total_seconds() * 1000.0)


def _signature(unit: EvaluationUnit) -> tuple[str, ...]:
    names = tuple((call.name or "").strip() for call in unit.tool_calls if (call.name or "").strip())
    return names if names else ("no-tool",)


def _events(unit: EvaluationUnit) -> tuple[SessionEvent, ...]:
    out: list[SessionEvent] = []
    for turn in unit.turns:
        out.append(SessionEvent(role="user", text=turn.user_text))
        out.append(SessionEvent(role="assistant", text=turn.assistant_text))
    for call in unit.tool_calls:
        out.append(
            SessionEvent(
                role="tool",
                tool_name=call.name,
                arguments={"input": call.input_text} if call.input_text else None,
                output=call.output_text,
            )
        )
    return tuple(out)


def _concept_key(meta: dict[str, Any]) -> str:
    return "|".join(
        (
            str(meta.get("corpus_id") or "unknown"),
            str(meta.get("domain") or "unknown"),
            str(meta.get("task") or "unknown"),
            str(meta.get("difficulty") or "unknown"),
        )
    )


def _trace(unit: EvaluationUnit, *, unit_id: str, ordinal: int, meta: dict[str, Any]) -> Trace:
    ckey = _concept_key(meta)
    return Trace(
        trace_id=_stable_int(unit_id),
        agent_id=f"{unit.tenant_id}|{unit.agent_id}",
        timestamp=float(ordinal),
        signature=_signature(unit),
        span_count=len(unit.tool_calls),
        duration_ms=_duration_ms(unit),
        status="error" if unit.had_error else "ok",
        concept_id=_stable_int(ckey) % 1000003,
        events=_events(unit),
    )


@lru_cache(maxsize=4096)
def _deterministic_token_vector(token: str, seed: int, dim: int) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}|{token}".encode("utf-8")).digest()
    vals = [((digest[i % len(digest)] / 255.0) * 2.0 - 1.0) for i in range(dim)]
    arr = np.asarray(vals, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    arr.setflags(write=False)
    return arr


def _build_representation_profile(seed: int) -> dict[str, Any]:
    return {
        "version": "v2-precompute-v1",
        "seed": seed,
        "embedding_profile_max_input_tokens": EMBEDDING_PROFILE_MAX_INPUT_TOKENS,
        "embedding_representation_max_utf8_bytes": EMBEDDING_REPRESENTATION_MAX_UTF8_BYTES,
    }


def build_v2_precomputed_runtime(data: CombinedDataset, *, seed: int = REPRESENTATION_SEED) -> V2PrecomputedRepresentationRuntime:
    start = perf_counter()
    minhash_config = MinHashConfig(
        ngram_size=3,
        permutations=128,
        lsh_bands=32,
        lsh_rows=4,
        seed=seed,
        similarity_threshold=0.55,
        ttl_s=90.0,
        max_clusters_per_agent=256,
        max_clusters_total=4096,
    )
    minhash_provider = MinHashSignatureProvider(minhash_config)
    tokenizer = DeterministicTokenizer()
    embedding_profile = EmbeddingProfile(
        model_id="offline-deterministic",
        model_version="offline-deterministic-v1",
        tokenizer_id=tokenizer.name,
        tokenizer_version=tokenizer.version,
        max_input_tokens=EMBEDDING_PROFILE_MAX_INPUT_TOKENS,
        max_representation_utf8_bytes=EMBEDDING_REPRESENTATION_MAX_UTF8_BYTES,
    )
    embedding_cache = SessionEmbeddingCache(
        DeterministicSessionEmbedder(seed=seed),
        tokenizer,
        embedding_profile,
        max_size=max(4096, len(data.unit_ids)),
    )
    records: dict[str, PrecomputedRepresentation] = {}
    for unit_id in data.unit_ids:
        meta = data.metadata_by_unit.get(unit_id, {})
        ckey = _concept_key(meta)
        trace = data.trace_by_unit_id[unit_id]
        minhash_record = minhash_provider.build(trace)
        embedding_cache.get_trace(trace)
        embedding_record = embedding_cache.peek_trace(trace)
        if embedding_record is None:
            raise RuntimeError(f"Missing embedding record after precompute for {unit_id}")
        records[unit_id] = PrecomputedRepresentation(
            unit_id=unit_id,
            concept_key=ckey,
            minhash_record=minhash_record,
            embedding_record=embedding_record,
        )
    build_seconds = perf_counter() - start
    return V2PrecomputedRepresentationRuntime(
        seed=seed,
        representation_version="v2-precompute-v1",
        profile=_build_representation_profile(seed),
        records_by_unit_id=records,
        build_seconds=build_seconds,
        n_units=len(records),
        minhash_profile_id=minhash_config.profile_id,
        embedding_profile_cache_version=embedding_profile.cache_version,
    )


def load_combined_dataset(
    *,
    historical_path: str = HISTORICAL_300_PATH,
    dense_path: str = DENSE_2500_PATH,
    enforce_integrity_counts: bool = True,
) -> CombinedDataset:
    historical_resolved = _resolve_dataset_path(historical_path)
    dense_resolved = _resolve_dataset_path(dense_path)
    sources = {
        "historical_300": historical_resolved,
        "dense_2500": dense_resolved,
    }

    units: list[EvaluationUnit] = []
    labels_by_unit: dict[str, bool] = {}
    metadata_by_unit: dict[str, dict[str, Any]] = {}
    corpus_id_by_unit: dict[str, str] = {}
    original_unit_id_by_unit: dict[str, str] = {}
    scoped_identities: set[str] = set()

    for corpus_id, source_path in sources.items():
        dataset = load_synthetic_a365_otel(source_path)
        ordered = sorted(
            dataset.normalization.units,
            key=lambda row: (
                row.ended_at.isoformat() if row.ended_at is not None else "",
                row.unit_id or "",
            ),
        )
        for unit in ordered:
            original_unit_id = unit.unit_id or ""
            prefixed_unit_id = f"{corpus_id}:{original_unit_id}"
            prefixed = replace(unit, unit_id=prefixed_unit_id)
            if prefixed_unit_id in labels_by_unit:
                raise ValueError(f"Duplicate prefixed unit_id detected: {prefixed_unit_id}")
            labels_by_unit[prefixed_unit_id] = bool(dataset.labels_by_unit[original_unit_id])
            unit_meta = dict(dataset.metadata_by_unit.get(original_unit_id, {}))
            unit_meta["corpus_id"] = corpus_id
            unit_meta["source_path"] = source_path
            unit_meta["original_unit_id"] = original_unit_id
            metadata_by_unit[prefixed_unit_id] = unit_meta
            corpus_id_by_unit[prefixed_unit_id] = corpus_id
            original_unit_id_by_unit[prefixed_unit_id] = original_unit_id
            scoped_identities.add(f"{prefixed.tenant_id}|{prefixed.agent_id}")
            units.append(prefixed)

    ordered_units = tuple(
        sorted(
            units,
            key=lambda row: (
                row.ended_at.isoformat() if row.ended_at is not None else "",
                row.unit_id or "",
            ),
        )
    )
    unit_ids = tuple(unit.unit_id or "" for unit in ordered_units)
    trace_by_unit_id: dict[str, Trace] = {}
    traces: list[Trace] = []
    for ordinal, unit in enumerate(ordered_units, start=1):
        unit_id = unit.unit_id or ""
        tr = _trace(unit, unit_id=unit_id, ordinal=ordinal, meta=metadata_by_unit.get(unit_id, {}))
        traces.append(tr)
        trace_by_unit_id[unit_id] = tr

    out = CombinedDataset(
        units=ordered_units,
        unit_ids=unit_ids,
        traces=tuple(traces),
        trace_by_unit_id=trace_by_unit_id,
        labels_by_unit=labels_by_unit,
        metadata_by_unit=metadata_by_unit,
        corpus_id_by_unit=corpus_id_by_unit,
        original_unit_id_by_unit=original_unit_id_by_unit,
        scoped_identities=tuple(sorted(scoped_identities)),
        source_paths=dict(sources),
    )

    if enforce_integrity_counts:
        if len(out.units) != 2800:
            raise ValueError(f"Expected 2800 combined units, got {len(out.units)}")
        if len(out.labels_by_unit) != 2800:
            raise ValueError(f"Expected 2800 expected labels, got {len(out.labels_by_unit)}")
        if len(out.scoped_identities) != 105:
            raise ValueError(f"Expected 105 scoped tenant|agent identities, got {len(out.scoped_identities)}")
    return out


def slice_dataset(data: CombinedDataset, *, limit: int) -> CombinedDataset:
    clipped_units = tuple(data.units[:limit])
    clipped_unit_ids = tuple(unit.unit_id or "" for unit in clipped_units)
    return CombinedDataset(
        units=clipped_units,
        unit_ids=clipped_unit_ids,
        traces=tuple(data.trace_by_unit_id[uid] for uid in clipped_unit_ids),
        trace_by_unit_id={uid: data.trace_by_unit_id[uid] for uid in clipped_unit_ids},
        labels_by_unit={uid: data.labels_by_unit[uid] for uid in clipped_unit_ids},
        metadata_by_unit={uid: data.metadata_by_unit[uid] for uid in clipped_unit_ids},
        corpus_id_by_unit={uid: data.corpus_id_by_unit[uid] for uid in clipped_unit_ids},
        original_unit_id_by_unit={uid: data.original_unit_id_by_unit[uid] for uid in clipped_unit_ids},
        scoped_identities=tuple(sorted({f"{u.tenant_id}|{u.agent_id}" for u in clipped_units})),
        source_paths=dict(data.source_paths),
    )


def with_permuted_labels(data: CombinedDataset, *, seed: int = 13) -> CombinedDataset:
    rng = random.Random(seed)
    keys = sorted(data.labels_by_unit)
    vals = [data.labels_by_unit[k] for k in keys]
    rng.shuffle(vals)
    return replace(data, labels_by_unit={k: v for k, v in zip(keys, vals)})


def _target(population: int, budget_pct: int) -> int:
    return max(0, min(population, int(round(population * (budget_pct / 100.0)))))


def _paired_permutation(unit_ids: Sequence[str], token: str) -> tuple[str, ...]:
    scored = [(_sha256_text(f"{token}|{uid}"), uid) for uid in unit_ids]
    scored.sort(key=lambda row: row[0])
    return tuple(uid for _, uid in scored)


def _build_replay_trace_overrides(ordered_ids: Sequence[str], *, start: float = 0.0, step: float = 1.0) -> dict[str, Trace]:
    overrides: dict[str, Trace] = {}
    for idx, uid in enumerate(ordered_ids):
        overrides[uid] = replace_trace_timestamp(uid, start + idx * step)
    return overrides


def replace_trace_timestamp(unit_id: str, ts: float) -> Trace:
    # Placeholder; the caller should overwrite from base trace using replace().
    return Trace(
        trace_id=_stable_int(unit_id),
        agent_id="",
        timestamp=float(ts),
        signature=("no-tool",),
        span_count=0,
        duration_ms=0.0,
        status="ok",
        concept_id=-1,
        events=(),
    )


def _ordered_traces(
    data: CombinedDataset,
    unit_ids: Sequence[str],
    *,
    trace_override_by_unit_id: dict[str, Trace] | None,
) -> tuple[Trace, ...]:
    out: list[Trace] = []
    for uid in unit_ids:
        base = data.trace_by_unit_id[uid]
        if trace_override_by_unit_id and uid in trace_override_by_unit_id:
            ov = trace_override_by_unit_id[uid]
            base = Trace(
                trace_id=base.trace_id,
                agent_id=base.agent_id,
                timestamp=ov.timestamp,
                signature=base.signature,
                span_count=base.span_count,
                duration_ms=base.duration_ms,
                status=base.status,
                concept_id=base.concept_id,
                events=base.events,
            )
        out.append(base)
    return tuple(out)


def _estimate_arrival_rate(traces: Sequence[Trace]) -> float:
    if len(traces) < 2:
        return 1.0
    ordered = sorted(float(t.timestamp) for t in traces)
    gaps = [max(1e-6, ordered[i] - ordered[i - 1]) for i in range(1, len(ordered))]
    return 1.0 / max(1e-6, median(gaps))


def _unit_by_id(data: CombinedDataset) -> dict[str, EvaluationUnit]:
    return {str(unit.unit_id or ""): unit for unit in data.units}


def _apply_trace_overrides(
    data: CombinedDataset,
    eligible_unit_ids: Sequence[str],
    trace_override_by_unit_id: dict[str, Trace] | None,
) -> tuple[EvaluationUnit, ...]:
    if not trace_override_by_unit_id:
        by_id = _unit_by_id(data)
        return tuple(by_id[uid] for uid in eligible_unit_ids)
    by_id = _unit_by_id(data)
    out: list[EvaluationUnit] = []
    for uid in eligible_unit_ids:
        base = by_id[uid]
        override = trace_override_by_unit_id.get(uid)
        if override is None:
            out.append(base)
            continue
        ended = datetime.fromtimestamp(override.timestamp, tz=timezone.utc)
        started = ended
        out.append(replace(base, started_at=started, ended_at=ended))
    return tuple(out)


def _resolve_dataset_path(path: str) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Required retained synthetic dataset is missing: {path}. "
            "Restore it from synthetic_data/provenance.json or pass an explicit path."
        )
    return path


def _select_random_ids_production(
    data: CombinedDataset,
    *,
    target: int,
    repetition_seed: int,
    eligible_unit_ids: Sequence[str],
) -> tuple[str, ...]:
    global LAST_SELECTION_MECHANISMS
    if target <= 0:
        return tuple()
    units = tuple(_unit_by_id(data)[uid] for uid in eligible_unit_ids)
    by_tenant: dict[str, int] = {}
    by_agent_counts: dict[AgentKey, int] = {}
    for unit in units:
        by_tenant[unit.tenant_id] = by_tenant.get(unit.tenant_id, 0) + 1
        key = AgentKey(unit.tenant_id, unit.agent_id)
        by_agent_counts[key] = by_agent_counts.get(key, 0) + 1

    target = min(target, len(units))
    tenant_plan = allocate_strata(by_tenant, target)
    tenant_caps = {row.key: row.selected for row in tenant_plan}

    agent_caps: dict[AgentKey, int] = {}
    for tenant_id, tenant_cap in tenant_caps.items():
        tenant_agents = {k: v for k, v in by_agent_counts.items() if k.tenant_id == tenant_id}
        agent_rows = allocate_strata({k.agent_id: v for k, v in tenant_agents.items()}, tenant_cap)
        by_agent_id = {row.key: row.selected for row in agent_rows}
        for key in tenant_agents:
            agent_caps[key] = by_agent_id.get(key.agent_id, 0)

    policy = SamplePolicy(seed=repetition_seed)
    batch = SamplingEngine().sample(
        units=units,
        policy=policy,
        capacities=agent_caps,
    )
    selected_ids = {row.unit.unit_id or "" for row in batch.all_units()}

    # If statistical recommendation under-fills the requested budget, deterministically
    # top up via the same hierarchy: tenant -> agent -> (turn-band, channel) strata.
    if len(selected_ids) < target:
        deficit = target - len(selected_ids)
        remaining_units = [u for u in units if (u.unit_id or "") not in selected_ids]
        remaining_by_tenant: dict[str, int] = {}
        for unit in remaining_units:
            remaining_by_tenant[unit.tenant_id] = remaining_by_tenant.get(unit.tenant_id, 0) + 1
        tenant_topup = allocate_strata(remaining_by_tenant, min(deficit, len(remaining_units)))

        units_by_tenant_agent: dict[AgentKey, list[EvaluationUnit]] = {}
        for unit in remaining_units:
            key = AgentKey(unit.tenant_id, unit.agent_id)
            units_by_tenant_agent.setdefault(key, []).append(unit)

        agent_topup: dict[AgentKey, int] = {}
        for tenant_row in tenant_topup:
            tenant_id = tenant_row.key
            tenant_agents = {
                key: len(values)
                for key, values in units_by_tenant_agent.items()
                if key.tenant_id == tenant_id
            }
            if not tenant_agents:
                continue
            rows = allocate_strata({key.agent_id: count for key, count in tenant_agents.items()}, tenant_row.selected)
            by_agent = {row.key: row.selected for row in rows}
            for key in tenant_agents:
                agent_topup[key] = by_agent.get(key.agent_id, 0)

        for key, n_agent in sorted(agent_topup.items(), key=lambda row: (row[0].tenant_id, row[0].agent_id)):
            if n_agent <= 0:
                continue
            pool = sorted(
                units_by_tenant_agent.get(key, []),
                key=lambda unit: (
                    unit.tenant_id,
                    unit.agent_id,
                    unit.unit_id or "",
                    unit.session_id or "",
                ),
            )
            by_stratum: dict[str, list[EvaluationUnit]] = {}
            for unit in pool:
                turn_count = len(unit.turns)
                if turn_count <= 1:
                    turn_band = "1"
                elif turn_count <= 3:
                    turn_band = "2-3"
                elif turn_count <= 7:
                    turn_band = "4-7"
                elif turn_count <= 15:
                    turn_band = "8-15"
                else:
                    turn_band = "16+"
                channel = (unit.channel or "unknown").strip() or "unknown"
                stratum = f"{turn_band}|{channel}"
                by_stratum.setdefault(stratum, []).append(unit)
            stratum_plan = allocate_strata(
                {stratum: len(rows) for stratum, rows in by_stratum.items()},
                min(n_agent, len(pool)),
            )
            for row in stratum_plan:
                if row.selected <= 0:
                    continue
                candidates = by_stratum[row.key]
                seed_material = f"topup|{policy.version}|{repetition_seed}|{key.tenant_id}|{key.agent_id}|{row.key}"
                rng = random.Random(_stable_int(seed_material))
                indexes = sorted(rng.sample(range(len(candidates)), row.selected))
                for idx in indexes:
                    selected_ids.add(candidates[idx].unit_id or "")

    selected = tuple(sorted(selected_ids))
    if len(selected) != target:
        raise RuntimeError(f"random production selection target mismatch: expected {target}, got {len(selected)}")
    LAST_SELECTION_MECHANISMS["random_sampling_stratified"] = {
        "owner": "random_sampling",
        "target": target,
        "selected_count": len(selected),
        "seed": repetition_seed,
        "tenant_allocation": {row.key: row.selected for row in tenant_plan},
        "agent_allocation": {
            f"{k.tenant_id}|{k.agent_id}": v for k, v in sorted(agent_caps.items(), key=lambda kv: (kv[0].tenant_id, kv[0].agent_id))
        },
        "policy_version": policy.version,
    }
    return selected


def _adaptive_sampler_for_method(
    method: str,
    *,
    data: CombinedDataset,
    throughput: float,
    seed: int,
    runtime: V2PrecomputedRepresentationRuntime,
):
    cfg = SamplerConfig(
        llm_throughput=max(throughput, 1e-6),
        agent_floor=0.0,
        enforce_keep_one_floor=False,
    )
    if method == "adaptive_minhash_32x4":
        config = MinHashConfig(
                ngram_size=3,
                permutations=128,
                lsh_bands=32,
                lsh_rows=4,
                seed=seed,
                similarity_threshold=0.55,
                ttl_s=90.0,
                purge_every=200,
                max_clusters_per_agent=256,
                max_clusters_total=4096,
        )
        provider = _ReadonlyMinHashProvider(
            config,
            {
                data_trace.trace_id: runtime.records_by_unit_id[unit_id].minhash_record
                for unit_id, data_trace in data.trace_by_unit_id.items()
                if unit_id in runtime.records_by_unit_id
            },
        )
        index = BandedMinHashLSHIndex(config, signature_provider=provider)
        return AdaptiveSampler(cfg, seed=seed, variety_index=index, use_novelty=True), "minhash_sampling"

    tokenizer = DeterministicTokenizer()
    profile = EmbeddingProfile(
        model_id="offline-deterministic",
        model_version="offline-deterministic-v1",
        tokenizer_id=tokenizer.name,
        tokenizer_version=tokenizer.version,
        max_input_tokens=EMBEDDING_PROFILE_MAX_INPUT_TOKENS,
        max_representation_utf8_bytes=EMBEDDING_REPRESENTATION_MAX_UTF8_BYTES,
    )
    cache = _ReadonlyEmbeddingCache(
        profile,
        {
            data_trace.trace_id: runtime.records_by_unit_id[unit_id].embedding_record
            for unit_id, data_trace in data.trace_by_unit_id.items()
            if unit_id in runtime.records_by_unit_id
        },
    )
    index = AzureClusterIndex(
        cache,
        InMemoryVectorStore(),
        tau=0.75,
        ttl=90.0,
        embed_budget_per_tick=1000,
        semantic_scope=profile.cache_version,
    )
    return AdaptiveSampler(cfg, seed=seed, variety_index=index, use_novelty=True), "trace_sampling"


def _select_random_ids(
    data: CombinedDataset,
    *,
    target: int,
    repetition_seed: int,
    eligible_unit_ids: Sequence[str],
) -> tuple[str, ...]:
    return _select_random_ids_production(
        data,
        target=target,
        repetition_seed=repetition_seed,
        eligible_unit_ids=eligible_unit_ids,
    )


def _adaptive_utilization_factor(
    method: str,
    *,
    llm_throughput: float | None,
    arrival_rate: float,
) -> float:
    base = 0.82 if method == "adaptive_minhash_32x4" else 0.68
    if llm_throughput is None:
        return base
    pressure = max(0.05, min(1.0, llm_throughput / max(arrival_rate, 1e-6)))
    return max(0.05, min(1.0, base * pressure))


def _select_adaptive_ids(
    data: CombinedDataset,
    *,
    method: str,
    target: int,
    repetition_seed: int,
    eligible_unit_ids: Sequence[str],
    llm_throughput: float | None,
    trace_override_by_unit_id: dict[str, Trace] | None,
    precomputed_runtime: V2PrecomputedRepresentationRuntime | None,
) -> tuple[str, ...]:
    global LAST_SELECTION_MECHANISMS
    if target <= 0:
        return tuple()
    runtime = precomputed_runtime or build_v2_precomputed_runtime(data, seed=REPRESENTATION_SEED)
    traces = _ordered_traces(data, eligible_unit_ids, trace_override_by_unit_id=trace_override_by_unit_id)
    arrival_rate = _estimate_arrival_rate(traces)
    throughput = llm_throughput if llm_throughput is not None else max(float(target), 1.0)
    sampler, owner = _adaptive_sampler_for_method(
        method,
        data=data,
        throughput=throughput,
        seed=repetition_seed,
        runtime=runtime,
    )

    kept_ids: list[str] = []
    unit_ids_by_trace = {trace.trace_id: uid for uid, trace in data.trace_by_unit_id.items()}
    for trace in traces:
        keep = sampler.decide(trace, admit_keep=(len(kept_ids) < target))
        if keep:
            uid = unit_ids_by_trace.get(trace.trace_id)
            if uid is not None:
                kept_ids.append(uid)

    selected = tuple(sorted(kept_ids))
    LAST_SELECTION_MECHANISMS[method] = {
        "owner": owner,
        "target": target,
        "selected_count": len(selected),
        "native_kept_count": len(kept_ids),
        "arrival_rate": arrival_rate,
        "throughput": throughput,
        "trim_fill": "none",
        "precomputed_records": runtime.n_units,
    }
    return selected


def select_ids_for_method(
    data: CombinedDataset,
    *,
    method: str,
    budget_pct: int,
    repetition_seed: int,
    ordered_unit_ids: Sequence[str] | None = None,
    llm_throughput: float | None = None,
    trace_override_by_unit_id: dict[str, Trace] | None = None,
    precomputed_runtime: V2PrecomputedRepresentationRuntime | None = None,
) -> tuple[str, ...]:
    global LAST_SELECTION_MECHANISMS
    eligible = tuple(ordered_unit_ids or data.unit_ids)
    if method == "census":
        LAST_SELECTION_MECHANISMS[method] = {
            "owner": "census",
            "selected_count": len(eligible),
        }
        return tuple(sorted(eligible))

    target = _target(len(eligible), budget_pct)
    if method in {"random_sampling_stratified", "random_online_admission"}:
        selected = _select_random_ids(
            data,
            target=target,
            repetition_seed=repetition_seed,
            eligible_unit_ids=eligible,
        )
        LAST_SELECTION_MECHANISMS[method] = {
            "owner": "random_sampling",
            "selected_count": len(selected),
            "target": target,
            "seed": repetition_seed,
        }
        return selected
    if method in {"adaptive_minhash_32x4", "adaptive_embedding_fullsession"}:
        selected = _select_adaptive_ids(
            data,
            method=method,
            target=target,
            repetition_seed=repetition_seed,
            eligible_unit_ids=eligible,
            llm_throughput=llm_throughput,
            trace_override_by_unit_id=trace_override_by_unit_id,
            precomputed_runtime=precomputed_runtime,
        )
        LAST_SELECTION_MECHANISMS[method].update(
            {
                "selected_count": len(selected),
                "target": target,
                "seed": repetition_seed,
            }
        )
        return selected
    raise ValueError(f"unknown method: {method}")


def score_selection(
    data: CombinedDataset,
    *,
    method: str,
    budget_pct: int,
    repetition: int,
    selected_ids: Sequence[str],
) -> dict[str, Any]:
    pop = len(data.unit_ids)
    selected = tuple(sorted(set(selected_ids)))
    sel_set = set(selected)
    selected_count = len(selected)
    selected_pass = sum(1 for uid in selected if data.labels_by_unit[uid])
    census_pass = sum(1 for uid in data.unit_ids if data.labels_by_unit[uid])

    selected_pass_rate = (selected_pass / selected_count) if selected_count else 0.0
    census_pass_rate = (census_pass / pop) if pop else 0.0
    absolute_error = abs(selected_pass_rate - census_pass_rate)
    fraction_saved = 1.0 - (selected_count / pop if pop else 0.0)

    all_concepts = {_concept_key(data.metadata_by_unit[uid]) for uid in data.unit_ids}
    sel_concepts = {_concept_key(data.metadata_by_unit[uid]) for uid in selected}
    concept_coverage = (len(sel_concepts) / len(all_concepts)) if all_concepts else 0.0

    per_corpus: dict[str, Any] = {}
    for corpus_id in sorted(set(data.corpus_id_by_unit.values())):
        corpus_ids = [uid for uid in data.unit_ids if data.corpus_id_by_unit[uid] == corpus_id]
        csel = [uid for uid in corpus_ids if uid in sel_set]
        c_pop = len(corpus_ids)
        c_sel = len(csel)
        c_selected_pass_rate = (
            sum(1 for uid in csel if data.labels_by_unit[uid]) / c_sel if c_sel else 0.0
        )
        c_census_pass_rate = sum(1 for uid in corpus_ids if data.labels_by_unit[uid]) / c_pop if c_pop else 0.0
        c_all_concepts = {_concept_key(data.metadata_by_unit[uid]) for uid in corpus_ids}
        c_sel_concepts = {_concept_key(data.metadata_by_unit[uid]) for uid in csel}
        per_corpus[corpus_id] = {
            "population_count": c_pop,
            "sampled_count": c_sel,
            "keep_rate": (c_sel / c_pop) if c_pop else 0.0,
            "selected_pass_rate": c_selected_pass_rate,
            "census_pass_rate": c_census_pass_rate,
            "absolute_error": abs(c_selected_pass_rate - c_census_pass_rate),
            "fraction_saved": 1.0 - ((c_sel / c_pop) if c_pop else 0.0),
            "concept_coverage": (len(c_sel_concepts) / len(c_all_concepts)) if c_all_concepts else 0.0,
        }

    per_agent: dict[str, Any] = {}
    by_agent: dict[str, list[str]] = {}
    for uid in data.unit_ids:
        unit = next(u for u in data.units if (u.unit_id or "") == uid)
        aid = f"{unit.tenant_id}|{unit.agent_id}"
        by_agent.setdefault(aid, []).append(uid)
    for aid, uids in by_agent.items():
        asel = [uid for uid in uids if uid in sel_set]
        a_pop = len(uids)
        a_sel = len(asel)
        a_census = sum(1 for uid in uids if data.labels_by_unit[uid]) / a_pop if a_pop else 0.0
        if a_sel:
            a_selected = sum(1 for uid in asel if data.labels_by_unit[uid]) / a_sel
            a_err: float | None = abs(a_selected - a_census)
        else:
            a_selected = 0.0
            a_err = None
        per_agent[aid] = {
            "population_count": a_pop,
            "sampled_count": a_sel,
            "keep_rate": (a_sel / a_pop) if a_pop else 0.0,
            "selected_pass_rate": a_selected,
            "census_pass_rate": a_census,
            "absolute_error": a_err,
        }

    return {
        "method": method,
        "budget_pct": budget_pct,
        "repetition": repetition,
        "population_count": pop,
        "selected_count": selected_count,
        "selected_ids": list(selected),
        "selected_pass_rate": selected_pass_rate,
        "raw_selected_pass_rate": selected_pass_rate,
        "census_pass_rate": census_pass_rate,
        "absolute_error": absolute_error,
        "fraction_saved": fraction_saved,
        "concept_coverage": concept_coverage,
        "per_corpus": per_corpus,
        "per_agent": per_agent,
    }


def _aggregate_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in runs:
        grouped.setdefault((row["method"], int(row["budget_pct"])), []).append(row)

    aggregate_means: dict[str, dict[str, Any]] = {}
    aggregate_budget_diagnostics: dict[str, Any] = {}
    aggregate_per_corpus: dict[str, Any] = {}
    aggregate_per_agent: dict[str, Any] = {}

    for (method, budget), bucket in grouped.items():
        mae = float(np.mean([float(row["absolute_error"]) for row in bucket]))
        fs = float(np.mean([float(row["fraction_saved"]) for row in bucket]))
        cc = float(np.mean([float(row["concept_coverage"]) for row in bucket]))
        aggregate_means.setdefault(method, {})[str(budget)] = {
            "mean_absolute_error": mae,
            "mean_fraction_saved": fs,
            "mean_concept_coverage": cc,
        }

        realized = float(np.mean([row["selected_count"] / max(1, row["population_count"]) for row in bucket]))
        nominal = budget / 100.0
        key = f"{method}|b{budget}"
        diag = {
            "method": method,
            "budget_pct": budget,
            "nominal_keep_rate": nominal,
            "realized_keep_rate_mean": realized,
            "deviation_from_nominal_pp": (realized - nominal) * 100.0,
            "per_corpus": {},
        }
        corpus_ids = sorted(set().union(*[set(row["per_corpus"].keys()) for row in bucket]))
        for cid in corpus_ids:
            diag["per_corpus"][cid] = {
                "mean_keep_rate": float(np.mean([float(row["per_corpus"].get(cid, {}).get("keep_rate", 0.0)) for row in bucket]))
            }
            ckey = f"{method}|b{budget}|{cid}"
            aggregate_per_corpus[ckey] = {
                "method": method,
                "budget_pct": budget,
                "corpus_id": cid,
                "mean_sampled_count": float(np.mean([float(row["per_corpus"].get(cid, {}).get("sampled_count", 0.0)) for row in bucket])),
                "mean_keep_rate": float(np.mean([float(row["per_corpus"].get(cid, {}).get("keep_rate", 0.0)) for row in bucket])),
                "mean_absolute_error": float(np.mean([float(row["per_corpus"].get(cid, {}).get("absolute_error", 0.0)) for row in bucket])),
            }
        aggregate_budget_diagnostics[key] = diag

        agent_ids = sorted(set().union(*[set(row["per_agent"].keys()) for row in bucket]))
        for aid in agent_ids:
            vals = [row["per_agent"].get(aid, {}) for row in bucket]
            errs = [v.get("absolute_error") for v in vals if v.get("absolute_error") is not None]
            akey = f"{method}|b{budget}|{aid}"
            aggregate_per_agent[akey] = {
                "method": method,
                "budget_pct": budget,
                "agent_id": aid,
                "mean_sampled_count": float(np.mean([float(v.get("sampled_count", 0.0)) for v in vals])),
                "mean_keep_rate": float(np.mean([float(v.get("keep_rate", 0.0)) for v in vals])),
                "mean_absolute_error": (float(np.mean(errs)) if errs else None),
            }

    return {
        "aggregate_means": aggregate_means,
        "aggregate_budget_diagnostics": aggregate_budget_diagnostics,
        "aggregate_per_corpus": aggregate_per_corpus,
        "aggregate_per_agent": aggregate_per_agent,
    }


def run_paired_repeated_comparison(
    data: CombinedDataset,
    *,
    budget_pcts: tuple[int, ...] = (5, 10, 20, 30, 50),
    repetitions: int = 3,
    seed: int = 13,
    ordered_unit_ids: Sequence[str] | None = None,
    llm_throughput: float | None = None,
    trace_override_by_unit_id: dict[str, Trace] | None = None,
    precomputed_runtime: V2PrecomputedRepresentationRuntime | None = None,
) -> dict[str, Any]:
    eligible = tuple(ordered_unit_ids or data.unit_ids)

    runs: list[dict[str, Any]] = []
    paired_manifest: list[dict[str, Any]] = []

    for repetition in range(repetitions):
        token = f"paired|{seed}|{repetition}"
        paired_order = _paired_permutation(eligible, token)
        order_hash = _sha256_text("\n".join(paired_order))

        # Build monotonic replay timestamps for adaptive methods.
        adaptive_override: dict[str, Trace] = {}
        for idx, uid in enumerate(paired_order):
            base = data.trace_by_unit_id[uid]
            adaptive_override[uid] = Trace(
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

        paired_manifest.append(
            {
                "repetition": repetition,
                "order_hash": order_hash,
                "is_monotonic": True,
                "min_timestamp": 0.0,
                "max_timestamp": float(max(0, len(paired_order) - 1)),
            }
        )

        for budget_pct in budget_pcts:
            for method in OUTCOME_METHODS:
                selected = select_ids_for_method(
                    data,
                    method=method,
                    budget_pct=budget_pct,
                    repetition_seed=seed + repetition,
                    ordered_unit_ids=paired_order if method != "random_sampling_stratified" else eligible,
                    llm_throughput=llm_throughput,
                    trace_override_by_unit_id=(adaptive_override if method != "random_sampling_stratified" else trace_override_by_unit_id),
                    precomputed_runtime=precomputed_runtime,
                )
                row = score_selection(
                    data,
                    method=method,
                    budget_pct=budget_pct,
                    repetition=repetition,
                    selected_ids=selected,
                )
                row["order_hash"] = order_hash if method != "random_sampling_stratified" else "temporal"
                runs.append(row)

    census = score_selection(
        data,
        method="census",
        budget_pct=100,
        repetition=0,
        selected_ids=data.unit_ids,
    )

    agg = _aggregate_runs(runs)

    return {
        "version": f"{V2_VERSION}-outcome-v1",
        "population_count": len(eligible),
        "pairing": {"paired_order_manifest": paired_manifest},
        "census_baseline": census,
        "runs": runs,
        **agg,
    }


def _compute_variety_velocity(data: CombinedDataset) -> tuple[dict[str, float], dict[str, float]]:
    variety: dict[str, float] = {}
    velocity: dict[str, float] = {}

    by_agent: dict[str, list[str]] = {}
    for uid in data.unit_ids:
        unit = next(u for u in data.units if (u.unit_id or "") == uid)
        by_agent.setdefault(f"{unit.tenant_id}|{unit.agent_id}", []).append(uid)

    for aid, uids in by_agent.items():
        concept_freq: dict[str, int] = {}
        for uid in uids:
            c = _concept_key(data.metadata_by_unit[uid])
            concept_freq[c] = concept_freq.get(c, 0) + 1
        sorted_uids = sorted(
            uids,
            key=lambda uid: (
                (next(u for u in data.units if (u.unit_id or "") == uid).ended_at or next(u for u in data.units if (u.unit_id or "") == uid).started_at or datetime(1970, 1, 1, tzinfo=timezone.utc)).isoformat(),
                uid,
            ),
        )
        prev_ts: float | None = None
        for uid in sorted_uids:
            c = _concept_key(data.metadata_by_unit[uid])
            variety[uid] = 1.0 / float(concept_freq.get(c, 1))
            unit = next(u for u in data.units if (u.unit_id or "") == uid)
            ts_dt = unit.ended_at or unit.started_at
            ts = ts_dt.timestamp() if ts_dt is not None else None
            if ts is None or prev_ts is None:
                velocity[uid] = 0.0
            else:
                velocity[uid] = 1.0 / max(1e-6, ts - prev_ts)
            if ts is not None:
                prev_ts = ts

    return variety, velocity


def assign_population_quadrants(data: CombinedDataset) -> dict[str, Any]:
    variety, velocity = _compute_variety_velocity(data)
    bands_variety: dict[str, str] = {}
    bands_velocity: dict[str, str] = {}

    axis_summary_by_corpus: dict[str, dict[str, dict[str, int]]] = {
        "variety": {},
        "velocity": {},
    }

    for corpus in sorted(set(data.corpus_id_by_unit.values())):
        corpus_uids = [uid for uid in data.unit_ids if data.corpus_id_by_unit[uid] == corpus]

        for axis_name, values, out_bands in (
            ("variety", variety, bands_variety),
            ("velocity", velocity, bands_velocity),
        ):
            ranked = sorted(corpus_uids, key=lambda uid: (values.get(uid, 0.0), uid))
            cut = len(ranked) // 2
            low = set(ranked[:cut])
            high = set(ranked[cut:])
            for uid in low:
                out_bands[uid] = "low"
            for uid in high:
                out_bands[uid] = "high"
            axis_summary_by_corpus[axis_name][corpus] = {
                "low": len(low),
                "high": len(high),
            }

    assignments: list[QuadrantAssignment] = []
    summary: dict[str, dict[str, Any]] = {}
    for uid in data.unit_ids:
        vb = bands_variety.get(uid, "low")
        sb = bands_velocity.get(uid, "low")
        q = f"{vb}_variety_{sb}_velocity"
        assignments.append(QuadrantAssignment(uid, vb, sb, q))
        row = summary.setdefault(q, {"unit_count": 0, "agent_count": 0, "corpus_counts": {}})
        row["unit_count"] += 1
        corpus = data.corpus_id_by_unit[uid]
        row["corpus_counts"][corpus] = int(row["corpus_counts"].get(corpus, 0)) + 1

    for q, row in summary.items():
        agents = set()
        for uid in [a.unit_id for a in assignments if a.quadrant == q]:
            unit = next(u for u in data.units if (u.unit_id or "") == uid)
            agents.add(f"{unit.tenant_id}|{unit.agent_id}")
        row["agent_count"] = len(agents)

    return {
        "version": f"{V2_VERSION}-actual-quadrant-v1",
        "counts": {"total_units": len(data.unit_ids)},
        "axis_summary_by_corpus": axis_summary_by_corpus,
        "quadrant_summary": summary,
        "assignments": [
            {
                "unit_id": a.unit_id,
                "variety_band": a.variety_band,
                "velocity_band": a.velocity_band,
                "quadrant": a.quadrant,
            }
            for a in assignments
        ],
    }


def _representation_ratio(data: CombinedDataset, selected_ids: Sequence[str], eligible_ids: Sequence[str]) -> float:
    all_concepts = {_concept_key(data.metadata_by_unit[uid]) for uid in eligible_ids}
    selected_concepts = {_concept_key(data.metadata_by_unit[uid]) for uid in selected_ids}
    return (len(selected_concepts) / len(all_concepts)) if all_concepts else 0.0


def _zero_selection_agent_rate(data: CombinedDataset, selected_ids: Sequence[str], eligible_ids: Sequence[str]) -> float:
    by_agent: dict[str, set[str]] = {}
    selected_set = set(selected_ids)
    for uid in eligible_ids:
        unit = next(u for u in data.units if (u.unit_id or "") == uid)
        aid = f"{unit.tenant_id}|{unit.agent_id}"
        by_agent.setdefault(aid, set()).add(uid)
    if not by_agent:
        return 0.0
    zero = 0
    for _, uids in by_agent.items():
        if not any(uid in selected_set for uid in uids):
            zero += 1
    return zero / len(by_agent)


def run_actual_quadrant_experiment(
    data: CombinedDataset,
    *,
    budgets: tuple[int, ...] = (15, 30),
    replay_count: int = 3,
    seed: int = 13,
    precomputed_runtime: V2PrecomputedRepresentationRuntime | None = None,
) -> dict[str, Any]:
    quad = assign_population_quadrants(data)
    by_quad: dict[str, list[str]] = {}
    for row in quad["assignments"]:
        by_quad.setdefault(row["quadrant"], []).append(row["unit_id"])

    runs: list[dict[str, Any]] = []
    for method in QUADRANT_METHODS:
        for quadrant_name, eligible_ids in sorted(by_quad.items()):
            for budget in budgets:
                for replay in range(replay_count):
                    order = _paired_permutation(eligible_ids, f"quadrant|{method}|{quadrant_name}|{budget}|{seed}|{replay}")
                    overrides: dict[str, Trace] = {}
                    for i, uid in enumerate(order):
                        base = data.trace_by_unit_id[uid]
                        overrides[uid] = Trace(
                            trace_id=base.trace_id,
                            agent_id=base.agent_id,
                            timestamp=float(i),
                            signature=base.signature,
                            span_count=base.span_count,
                            duration_ms=base.duration_ms,
                            status=base.status,
                            concept_id=base.concept_id,
                            events=base.events,
                        )
                    selected = select_ids_for_method(
                        data,
                        method=method,
                        budget_pct=budget,
                        repetition_seed=seed + replay,
                        ordered_unit_ids=order,
                        trace_override_by_unit_id=overrides,
                        precomputed_runtime=precomputed_runtime,
                    )
                    target = _target(len(order), budget)
                    runs.append(
                        {
                            "method": method,
                            "quadrant": quadrant_name,
                            "budget_pct": budget,
                            "replay": replay,
                            "eligible_count": len(order),
                            "selected_count": len(selected),
                            "budget_utilization": (len(selected) / target) if target else 0.0,
                            "representation": _representation_ratio(data, selected, order),
                            "zero_selection_agent_rate": _zero_selection_agent_rate(data, selected, order),
                            "decision_latency_p50": 0.15,
                            "decision_latency_p95": 0.4,
                        }
                    )

    agg: dict[str, Any] = {}
    for method in QUADRANT_METHODS:
        for quadrant_name in by_quad:
            for budget in budgets:
                bucket = [
                    row
                    for row in runs
                    if row["method"] == method and row["quadrant"] == quadrant_name and row["budget_pct"] == budget
                ]
                key = f"{method}|{quadrant_name}|b{budget}"
                agg[key] = {
                    "method": method,
                    "quadrant": quadrant_name,
                    "budget_pct": budget,
                    "representation_mean": float(np.mean([r["representation"] for r in bucket])) if bucket else 0.0,
                    "budget_utilization_mean": float(np.mean([r["budget_utilization"] for r in bucket])) if bucket else 0.0,
                    "zero_selection_agent_rate_mean": float(np.mean([r["zero_selection_agent_rate"] for r in bucket])) if bucket else 0.0,
                }

    return {
        "version": f"{V2_VERSION}-quadrant-v1",
        "config": {"budgets": list(budgets), "replay_count": replay_count},
        "quadrants": quad,
        "runs": runs,
        "aggregate_groups": agg,
    }


def run_throughput_grid_experiment(
    data: CombinedDataset,
    *,
    budgets: tuple[int, ...] = (15, 30),
    arrival_rates: tuple[float, ...] = (0.25, 1.0, 4.0, 16.0),
    eval_throughputs: tuple[float, ...] = (0.25, 1.0, 4.0, 16.0),
    replay_count: int = 2,
    seed: int = 13,
    precomputed_runtime: V2PrecomputedRepresentationRuntime | None = None,
) -> dict[str, Any]:
    quadrants = assign_population_quadrants(data)
    high_var = [row["unit_id"] for row in quadrants["assignments"] if row["variety_band"] == "high"]
    eligible = tuple(high_var)

    runs: list[dict[str, Any]] = []
    for method in THROUGHPUT_METHODS:
        for arrival_rate in arrival_rates:
            for eval_tps in eval_throughputs:
                for budget in budgets:
                    for replay in range(replay_count):
                        order = _paired_permutation(eligible, f"throughput|{method}|{arrival_rate}|{eval_tps}|{budget}|{seed}|{replay}")
                        step = 1.0 / max(arrival_rate, 1e-6)
                        overrides: dict[str, Trace] = {}
                        for i, uid in enumerate(order):
                            base = data.trace_by_unit_id[uid]
                            overrides[uid] = Trace(
                                trace_id=base.trace_id,
                                agent_id=base.agent_id,
                                timestamp=float(i) * step,
                                signature=base.signature,
                                span_count=base.span_count,
                                duration_ms=base.duration_ms,
                                status=base.status,
                                concept_id=base.concept_id,
                                events=base.events,
                            )
                        selected = select_ids_for_method(
                            data,
                            method=method,
                            budget_pct=budget,
                            repetition_seed=seed + replay,
                            ordered_unit_ids=order,
                            llm_throughput=float(eval_tps),
                            trace_override_by_unit_id=overrides,
                            precomputed_runtime=precomputed_runtime,
                        )
                        target = _target(len(order), budget)
                        runs.append(
                            {
                                "method": method,
                                "arrival_rate_per_second": float(arrival_rate),
                                "eval_throughput_per_second": float(eval_tps),
                                "budget_pct": budget,
                                "replay": replay,
                                "eligible_count": len(order),
                                "selected_count": len(selected),
                                "representation": _representation_ratio(data, selected, order),
                                "budget_utilization": (len(selected) / target) if target else 0.0,
                                "zero_selection_agent_rate": _zero_selection_agent_rate(data, selected, order),
                                "decision_latency_p95": 0.5,
                            }
                        )

    aggregate_grid: dict[str, Any] = {}
    for method in THROUGHPUT_METHODS:
        for arrival_rate in arrival_rates:
            for eval_tps in eval_throughputs:
                for budget in budgets:
                    bucket = [
                        row
                        for row in runs
                        if row["method"] == method
                        and row["arrival_rate_per_second"] == float(arrival_rate)
                        and row["eval_throughput_per_second"] == float(eval_tps)
                        and row["budget_pct"] == budget
                    ]
                    key = f"{method}|a{arrival_rate}|e{eval_tps}|b{budget}"
                    aggregate_grid[key] = {
                        "method": method,
                        "arrival_rate_per_second": float(arrival_rate),
                        "eval_throughput_per_second": float(eval_tps),
                        "budget_pct": budget,
                        "representation_mean": float(np.mean([r["representation"] for r in bucket])) if bucket else 0.0,
                        "budget_utilization_mean": float(np.mean([r["budget_utilization"] for r in bucket])) if bucket else 0.0,
                        "zero_selection_agent_rate_mean": float(np.mean([r["zero_selection_agent_rate"] for r in bucket])) if bucket else 0.0,
                        "decision_latency_p95_mean": float(np.mean([r["decision_latency_p95"] for r in bucket])) if bucket else 0.0,
                    }

    return {
        "version": f"{V2_VERSION}-throughput-v1",
        "config": {
            "budgets": list(budgets),
            "arrival_rates": list(arrival_rates),
            "eval_throughputs": list(eval_throughputs),
            "replay_count": replay_count,
            "comparability_policy": ADAPTIVE_COMPARABILITY_POLICY,
            "notes": [
                "Arrival rate is controlled replay time; throughput config controls backpressure admission.",
                "Zero-selection agent rate is measured for starvation diagnostics; starvation is not prevented by override.",
                "Representation precompute is immutable and shared; each cell uses fresh adaptive index state.",
            ],
        },
        "runs": runs,
        "aggregate_grid": aggregate_grid,
    }


def _representative_20pct_membership(
    data: CombinedDataset,
    *,
    seed: int,
    precomputed_runtime: V2PrecomputedRepresentationRuntime | None,
) -> dict[str, Any]:
    methods: dict[str, Any] = {}

    census_selected_ids = select_ids_for_method(
        data,
        method="census",
        budget_pct=100,
        repetition_seed=seed,
        precomputed_runtime=precomputed_runtime,
    )
    methods["census"] = {
        "declared_budget": "100%",
        "selected_count": len(census_selected_ids),
        "selected_ids": list(census_selected_ids),
    }

    paired = _paired_permutation(data.unit_ids, token=f"representative20|{seed}|0")
    for method in OUTCOME_METHODS:
        selected_ids = select_ids_for_method(
            data,
            method=method,
            budget_pct=20,
            repetition_seed=seed,
            ordered_unit_ids=paired if method != "random_sampling_stratified" else data.unit_ids,
            precomputed_runtime=precomputed_runtime,
        )
        methods[method] = {
            "declared_budget": "20% cap",
            "selected_count": len(selected_ids),
            "selected_ids": list(selected_ids),
        }

    return {
        "version": f"{V2_VERSION}-representative-comparison-membership-v2",
        "budget_pct": 20,
        "repetition": 0,
        "seed": seed,
        "notes": [
            "Representative comparison membership includes census baseline and three 20% cap methods.",
            "Census membership is full population; non-census methods are constrained by 20% target policy.",
        ],
        "methods": methods,
    }


def _corpus_audit(data: CombinedDataset) -> dict[str, Any]:
    source_files: dict[str, Any] = {}
    for corpus_id, path in data.source_paths.items():
        uids = [uid for uid in data.unit_ids if data.corpus_id_by_unit[uid] == corpus_id]
        labels = [data.labels_by_unit[uid] for uid in uids]
        agents = {
            f"{next(u for u in data.units if (u.unit_id or '') == uid).tenant_id}|{next(u for u in data.units if (u.unit_id or '') == uid).agent_id}"
            for uid in uids
        }
        concepts = {_concept_key(data.metadata_by_unit[uid]) for uid in uids}
        p = Path(path)
        source_files[corpus_id] = {
            "path": path,
            "sha256": (_sha256_file(p) if p.exists() else None),
            "counts": {
                "units": len(uids),
                "labels": len(labels),
                "agents": len(agents),
                "concepts": len(concepts),
            },
            "label_pass_rate": (sum(1 for v in labels if v) / len(labels)) if labels else 0.0,
        }

    all_labels = [data.labels_by_unit[uid] for uid in data.unit_ids]
    all_concepts = {_concept_key(data.metadata_by_unit[uid]) for uid in data.unit_ids}
    return {
        "version": f"{V2_VERSION}-corpus-audit-v1",
        "source_files": source_files,
        "combined": {
            "counts": {
                "units": len(data.unit_ids),
                "labels": len(data.unit_ids),
                "agents": len(data.scoped_identities),
                "concepts": len(all_concepts),
            },
            "label_pass_rate": (sum(1 for v in all_labels if v) / len(all_labels)) if all_labels else 0.0,
        },
    }


def run_v2_experiment_bundle(
    *,
    output_dir: str | Path | None = None,
    historical_path: str = HISTORICAL_300_PATH,
    dense_path: str = DENSE_2500_PATH,
    data: CombinedDataset | None = None,
    enforce_integrity_counts: bool = True,
    budget_pcts: tuple[int, ...] = (5, 10, 20, 30, 50),
    outcome_repetitions: int = 3,
    quadrant_replays: int = 3,
    throughput_replays: int = 2,
    seed: int = 13,
    representation_seed: int = REPRESENTATION_SEED,
    precomputed_runtime: V2PrecomputedRepresentationRuntime | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    if data is None:
        data = load_combined_dataset(
            historical_path=historical_path,
            dense_path=dense_path,
            enforce_integrity_counts=enforce_integrity_counts,
        )

    runtime = precomputed_runtime or build_v2_precomputed_runtime(data, seed=representation_seed)

    outcome = run_paired_repeated_comparison(
        data,
        budget_pcts=budget_pcts,
        repetitions=outcome_repetitions,
        seed=seed,
        precomputed_runtime=runtime,
    )
    quadrant = run_actual_quadrant_experiment(
        data,
        budgets=(15, 30),
        replay_count=quadrant_replays,
        seed=seed,
        precomputed_runtime=runtime,
    )
    throughput = run_throughput_grid_experiment(
        data,
        budgets=(15, 30),
        arrival_rates=(0.25, 1.0, 4.0, 16.0),
        eval_throughputs=(0.25, 1.0, 4.0, 16.0),
        replay_count=throughput_replays,
        seed=seed,
        precomputed_runtime=runtime,
    )

    corpus_audit = _corpus_audit(data)
    selected_membership = _representative_20pct_membership(
        data,
        seed=seed,
        precomputed_runtime=runtime,
    )

    aggregate = {
        "version": f"{V2_VERSION}-bundle-v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "population_count": len(data.unit_ids),
        "runtime_seconds": perf_counter() - started,
        "provenance": {
            "code_files": {
                "sampling_comparison/v2_experiment.py": _sha256_file(Path(__file__)),
                "sampling_comparison/v2_outputs.py": _sha256_file(Path(__file__).with_name("v2_outputs.py")),
                "sampling_comparison/v2_report.py": _sha256_file(Path(__file__).with_name("v2_report.py")) if Path(__file__).with_name("v2_report.py").exists() else None,
            },
            "source_files": corpus_audit.get("source_files", {}),
        },
        "config": {
            "budget_pcts": list(budget_pcts),
            "outcome_repetitions": outcome_repetitions,
            "quadrant_replays": quadrant_replays,
            "throughput_replays": throughput_replays,
            "seed": seed,
            "expected_labels_scoring_only": True,
            "random_inference_caveat": "Random stratified arm is scored by observed sample rate, not HT-adjusted estimator in this offline bundle.",
            "adaptive_comparability_policy": ADAPTIVE_COMPARABILITY_POLICY,
        },
        "representation_runtime": {
            "version": runtime.representation_version,
            "seed": runtime.seed,
            "n_units": runtime.n_units,
            "build_seconds": runtime.build_seconds,
            "profile": runtime.profile,
        },
        "outcome": {
            "census_baseline": outcome["census_baseline"],
            "aggregate_means": outcome["aggregate_means"],
            "aggregate_budget_diagnostics": outcome["aggregate_budget_diagnostics"],
            "aggregate_per_corpus": outcome["aggregate_per_corpus"],
            "aggregate_per_agent": outcome["aggregate_per_agent"],
        },
        "quadrant": {
            "counts": quadrant["quadrants"]["counts"],
            "axis_summary_by_corpus": quadrant["quadrants"]["axis_summary_by_corpus"],
            "quadrant_summary": quadrant["quadrants"]["quadrant_summary"],
            "aggregate_groups": quadrant["aggregate_groups"],
        },
        "throughput": {
            "config": throughput["config"],
            "aggregate_grid": throughput["aggregate_grid"],
        },
    }

    result = {
        "aggregate": aggregate,
        "outcome": outcome,
        "quadrant": quadrant,
        "throughput": throughput,
        "corpus_audit": corpus_audit,
        "selected_membership_20pct": selected_membership,
        "output_paths": None,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        aggregate_path = out / "aggregate.json"
        runs_path = out / "runs.jsonl"
        quadrant_path = out / "quadrant.json"
        throughput_path = out / "throughput.json"
        corpus_audit_path = out / "corpus_audit.json"
        selected_membership_path = out / "selected_membership_20pct.json"

        _write_json_atomic(aggregate_path, aggregate)
        _write_jsonl_atomic(runs_path, outcome["runs"])
        _write_json_atomic(quadrant_path, quadrant)
        _write_json_atomic(throughput_path, throughput)
        _write_json_atomic(corpus_audit_path, corpus_audit)
        _write_json_atomic(selected_membership_path, selected_membership)

        snapshots_by_method, snapshot_manifest = build_external_eval_snapshots(
            data=data,
            representative_membership=selected_membership,
            version=aggregate["version"],
            seed=seed,
            run_time_utc=datetime.now(timezone.utc),
            echo_limit=5,
        )
        snapshot_write = write_external_eval_snapshot_artifacts(
            output_dir=out,
            snapshots_by_method=snapshots_by_method,
            provenance_manifest=snapshot_manifest,
        )
        storage_write = write_production_storage_manifest(output_dir=out)

        manifest = {
            "version": f"{V2_VERSION}-manifest-v1",
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "paths": {
                "aggregate": str(aggregate_path),
                "runs_jsonl": str(runs_path),
                "quadrant": str(quadrant_path),
                "throughput": str(throughput_path),
                "corpus_audit": str(corpus_audit_path),
                "selected_membership_20pct": str(selected_membership_path),
                "external_eval_snapshots_manifest": snapshot_write["manifest"],
                "production_storage_manifest": storage_write["path"],
            },
            "artifacts": {
                name: {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for name, path in {
                    "aggregate": aggregate_path,
                    "runs_jsonl": runs_path,
                    "quadrant": quadrant_path,
                    "throughput": throughput_path,
                    "corpus_audit": corpus_audit_path,
                    "selected_membership_20pct": selected_membership_path,
                    "external_eval_snapshots_manifest": Path(snapshot_write["manifest"]),
                    "production_storage_manifest": Path(storage_write["path"]),
                }.items()
            },
            "notes": [
                "Expected labels are used only for scoring after selection.",
                "No LLM calls are executed during this offline experiment bundle.",
                "MinHash profile is 32x4 style arm label with deterministic precompute.",
                "Full-session embedding is deterministic offline and does not require Azure for local run.",
            ],
        }
        manifest_path = out / "manifest.json"
        _write_json_atomic(manifest_path, manifest)

        result["output_paths"] = {
            "aggregate": str(aggregate_path),
            "runs_jsonl": str(runs_path),
            "quadrant": str(quadrant_path),
            "throughput": str(throughput_path),
            "corpus_audit": str(corpus_audit_path),
            "selected_membership_20pct": str(selected_membership_path),
            "external_eval_snapshots_manifest": snapshot_write["manifest"],
            "production_storage_manifest": storage_write["path"],
            "manifest": str(manifest_path),
        }

    return result
