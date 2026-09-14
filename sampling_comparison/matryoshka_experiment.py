from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from sampling_comparison.v4_idw import IDWConfig
from trace_sampling.cluster_index import AzureClusterIndex
from trace_sampling.model import Trace
from trace_sampling.samplers import AdaptiveSampler, SamplerConfig
from trace_sampling.vector_store import InMemoryVectorStore


VERSION = "matryoshka-cutoff-v1"
DATASETS = ("historical_300", "dense_2500", "cosmos_otel", "tau2_bench")
DIMENSIONS = (1536, 512, 256, 128, 64, 32, 16, 8)
SCHEDULES = ("evenly_spaced", "uniformly_random", "bursty", "front_loaded", "agent_blocked")
RATES = (0.01, 0.02, 0.05, 0.10, 0.20)
MODES = ("end_to_end", "fixed_membership")
SUMMARY_METRICS = (
    "accuracy", "mae", "f1", "brier", "macro_agent_accuracy", "combined_accuracy",
    "aggregate_rate_error", "concept_coverage", "agent_coverage", "accuracy_delta_native",
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(canonical(value) + "\n", encoding="utf-8")
    tmp.replace(path)


def stable_rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{int(seed)}|{key}".encode("utf-8")).hexdigest()


def normalized_prefix(matrix: np.ndarray, dimension: int) -> np.ndarray:
    source = np.asarray(matrix)
    if source.ndim != 2 or source.shape[0] == 0:
        raise ValueError("embeddings must be a nonempty matrix")
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
        raise ValueError("dimension must be an integer")
    if not 1 <= dimension <= source.shape[1]:
        raise ValueError("dimension is outside the source vector")
    if not np.isfinite(source).all():
        raise ValueError("embeddings contain non-finite values")
    prefix = np.array(source[:, :dimension], dtype=np.float32, copy=True)
    norms = np.linalg.norm(prefix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError(f"invalid or zero-norm prefix at dimension {dimension}")
    return prefix / norms


@dataclass(frozen=True)
class InputDataset:
    dataset_id: str
    unit_ids: tuple[str, ...]
    agents: tuple[str, ...]
    signatures: tuple[tuple[str, ...], ...]
    concepts: tuple[str, ...]
    labels: np.ndarray
    vectors: np.ndarray
    profile: dict[str, Any]

    def __post_init__(self) -> None:
        n = len(self.unit_ids)
        if self.dataset_id not in DATASETS:
            raise ValueError(f"unknown dataset: {self.dataset_id}")
        if n < 2 or len(set(self.unit_ids)) != n or any(not uid for uid in self.unit_ids):
            raise ValueError("at least two unique nonempty unit IDs are required")
        if any(len(values) != n for values in (self.agents, self.signatures, self.concepts, self.labels)):
            raise ValueError("dataset row counts do not align")
        if any(not agent for agent in self.agents):
            raise ValueError("agent IDs must include a nonempty cohort identity")
        if np.asarray(self.labels).shape != (n,) or not np.isin(self.labels, (0, 1)).all():
            raise ValueError("labels must be explicitly mapped binary outcomes")
        if self.vectors.shape != (n, 1536):
            raise ValueError("full text-embedding-3-small vectors must have shape [N,1536]")
        normalized_prefix(self.vectors, 1536)


def load_input(manifest_path: Path) -> InputDataset:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["version"] != "matryoshka-input-v1":
        raise ValueError("unsupported input manifest version")
    if manifest["embedding_model_id"] != "text-embedding-3-small":
        raise ValueError("input must identify text-embedding-3-small, not a surrogate embedder")
    if manifest["embedding_dimensions"] != 1536:
        raise ValueError("input requires original 1536-dimensional embeddings")
    if not manifest.get("label_source") or not manifest.get("representation_policy"):
        raise ValueError("label and representation provenance are required")
    artifacts: dict[str, Path] = {}
    for name in ("units", "vectors"):
        path = manifest_path.parent / manifest["files"][name]
        if sha256_file(path) != manifest["hashes"][name]:
            raise ValueError(f"{name} hash mismatch")
        artifacts[name] = path
    units = json.loads(artifacts["units"].read_text(encoding="utf-8"))
    with np.load(artifacts["vectors"], allow_pickle=False) as arrays:
        ids = tuple(str(x) for x in arrays["unit_ids"].tolist())
        vectors = arrays["vectors"].copy()
    if ids != tuple(row["unit_id"] for row in units):
        raise ValueError("vector IDs/order must exactly match unit metadata")
    return InputDataset(
        dataset_id=manifest["dataset_id"],
        unit_ids=ids,
        agents=tuple(row["agent_id"] for row in units),
        signatures=tuple(tuple(row["signature"]) for row in units),
        concepts=tuple(row["concept_key"] for row in units),
        labels=np.asarray([row["label"] for row in units], dtype=np.float64),
        vectors=vectors,
        profile={**manifest, "input_manifest": str(manifest_path),
                 "input_manifest_sha256": sha256_file(manifest_path)},
    )


def build_schedule(unit_ids: Sequence[str], agents: Sequence[str], schedule: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule: {schedule}")
    n = len(unit_ids)
    ranked = sorted(range(n), key=lambda i: (stable_rank(seed, f"order|{schedule}|{unit_ids[i]}"), unit_ids[i]))
    if schedule == "agent_blocked":
        by_agent: dict[str, list[int]] = {}
        for i in ranked:
            by_agent.setdefault(agents[i], []).append(i)
        ranked = []
        for agent in sorted(by_agent, key=lambda a: stable_rank(seed, f"agent_block|{schedule}|{a}")):
            ranked.extend(sorted(by_agent[agent], key=lambda i: (stable_rank(seed, f"agent_local|{schedule}|{unit_ids[i]}"), unit_ids[i])))
    horizon = 10_000.0
    if schedule in ("evenly_spaced", "agent_blocked"):
        times = np.linspace(0.0, horizon, n, endpoint=False)
    elif schedule == "uniformly_random":
        times = np.random.default_rng(seed + 10).uniform(0.0, horizon, n)
    elif schedule == "front_loaded":
        times = np.linspace(0.0, 1.0, n, endpoint=False) ** 3 * horizon
    else:
        rng = np.random.default_rng(seed + 20)
        bursts = min(25, n)
        width = max(horizon * 0.002, horizon / max(2000.0, n * 8.0))
        times = np.concatenate([
            np.clip(center + rng.uniform(-width, width, n // bursts + (i < n % bursts)), 0.0, horizon)
            for i, center in enumerate(np.linspace(horizon * 0.02, horizon * 0.98, bursts))
        ])
    rows = sorted(
        ((float(times[j]), stable_rank(seed, f"tie|{schedule}|{unit_ids[i]}"), unit_ids[i], i)
         for j, i in enumerate(ranked)),
        key=lambda row: row[:3],
    )
    return np.asarray([r[3] for r in rows]), np.asarray([r[0] for r in rows])


class _ArrayCache:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    def contains_trace(self, trace: Trace) -> bool:
        return 0 <= trace.trace_id < len(self.vectors)

    def get_trace(self, trace: Trace) -> np.ndarray:
        return self.vectors[trace.trace_id]


def rank_membership(
    unit_ids: Sequence[str], agents: Sequence[str], signatures: Sequence[tuple[str, ...]],
    vectors: np.ndarray, order: np.ndarray, times: np.ndarray, seed: int, tau: float = 0.55,
) -> tuple[np.ndarray, dict[str, int]]:
    """Replay the earlier ARM2 selector without giving it outcomes or concepts."""
    store = InMemoryVectorStore()
    index = AzureClusterIndex(
        _ArrayCache(vectors), store, tau=tau, ttl=90.0, purge_every=200,
        embed_budget_per_tick=10_000_000, recent_buffer_size=4096,
        semantic_scope="idw-dimensionality-sweep-v1", tenant_id="dense-only", run_scope=f"seed-{seed}",
    )
    sampler = AdaptiveSampler(
        SamplerConfig(llm_throughput=1_000_000.0, agent_floor=0.0, enforce_keep_one_floor=False),
        seed=seed, variety_index=index, use_novelty=True,
    )
    native: list[tuple[float, float, str, int]] = []
    rejected: list[tuple[float, float, str, int]] = []
    for i, timestamp in zip(order, times, strict=True):
        trace = Trace(
            trace_id=int(i), agent_id=agents[i], timestamp=float(timestamp),
            signature=signatures[i], span_count=1, duration_ms=0.0, status="ok", concept_id=-1,
        )
        sampler.decide(trace, admit_keep=True)
        obs = sampler.last_observation
        if obs is None:
            raise RuntimeError("ARM2 selector did not produce an observation")
        score = (-float(obs.novelty), -float(obs.rarity), stable_rank(seed, f"native|{unit_ids[i]}"), int(i))
        (native if sampler.last_proposed_keep else rejected).append(score)
    if index.n_fallbacks:
        raise RuntimeError(f"ARM2 representation fallback: {index.n_fallbacks}")
    ranked = np.asarray([row[3] for row in sorted(native) + sorted(rejected)], dtype=np.int64)
    if len(ranked) != len(unit_ids) or len(set(ranked)) != len(unit_ids):
        raise RuntimeError("ARM2 ranking must contain every session exactly once")
    return ranked, {"native_candidate_count": len(native), "representation_fallbacks": index.n_fallbacks}


def distance_blocks(data: InputDataset, vectors: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    blocks = {}
    for agent in sorted(set(data.agents)):
        ids = np.asarray([i for i, value in enumerate(data.agents) if value == agent], dtype=np.int64)
        # Match the previous sweep's float32, re-normalized per-agent distance cache.
        matrix = normalized_prefix(vectors[ids], vectors.shape[1])
        cosine = np.clip(matrix @ matrix.T, -1.0, 1.0).astype(np.float32)
        angular = (np.arccos(cosine) / math.pi).astype(np.float32)
        blocks[agent] = (ids, cosine, angular)
    return blocks


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    if len(labels) == 0:
        return {"n": 0, "accuracy": None, "mae": None, "f1": None, "brier": None}
    predicted = predictions >= 0.5
    true = labels == 1
    tp = int(np.sum(predicted & true))
    fp = int(np.sum(predicted & ~true))
    fn = int(np.sum(~predicted & true))
    return {
        "n": len(labels), "accuracy": float(np.mean(predicted == true)),
        "mae": float(np.mean(np.abs(labels - predictions))),
        "brier": float(np.mean((labels - predictions) ** 2)),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": int(np.sum(~predicted & ~true)),
    }


def evaluate(
    data: InputDataset, blocks: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    order: np.ndarray, selected_ids: np.ndarray, config: IDWConfig = IDWConfig(),
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    n = len(data.unit_ids)
    if len(order) != n or set(order) != set(range(n)):
        raise ValueError("replay order must be a permutation")
    if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids).issubset(range(n)):
        raise ValueError("selected IDs must be unique in-range row indices")
    selected = np.zeros(n, dtype=bool)
    selected[selected_ids] = True
    positions = np.empty(n, dtype=np.int64)
    positions[order] = np.arange(n)
    ordered_selected = selected[order]
    earlier_count = np.cumsum(ordered_selected) - ordered_selected
    ordered_values = np.where(ordered_selected, data.labels[order], 0.0)
    earlier_sum = np.cumsum(ordered_values) - ordered_values
    predictions = np.empty(n, dtype=np.float64)
    predictions[order] = np.divide(
        earlier_sum, earlier_count, out=np.full(n, config.prior), where=earlier_count > 0,
    )
    provenance = np.empty(n, dtype="<U12")
    provenance[order] = np.where(earlier_count > 0, "global_mean", "prior")
    for ids, cosine, angular in blocks.values():
        donor_local = np.asarray(sorted(
            np.flatnonzero(selected[ids]), key=lambda j: data.unit_ids[ids[j]],
        ), dtype=np.int64)
        if not len(donor_local):
            continue
        donors = ids[donor_local]
        valid = positions[donors][None, :] < positions[ids][:, None]
        exact = valid & ((1.0 - cosine[:, donor_local]) <= config.exact_cosine_eps)
        exact_count = exact.sum(axis=1)
        exact_values = exact @ data.labels[donors]
        distances = angular[:, donor_local].astype(np.float64)
        nearest = np.argsort(np.where(valid, distances, np.inf), axis=1, kind="stable")[:, :config.k]
        d = np.take_along_axis(distances, nearest, axis=1)
        active = np.take_along_axis(valid, nearest, axis=1)
        weights = np.where(active, 1.0 / (d + config.eps) ** config.power, 0.0)
        denominator = weights.sum(axis=1)
        supported = denominator > 0
        weighted_values = (weights * data.labels[donors][nearest]).sum(axis=1)
        estimates = np.divide(weighted_values, denominator, out=np.zeros(len(ids)), where=supported)
        predictions[ids[supported]] = estimates[supported]
        provenance[ids[supported]] = "idw"
        exact_supported = exact_count > 0
        predictions[ids[exact_supported]] = exact_values[exact_supported] / exact_count[exact_supported]
        provenance[ids[exact_supported]] = "exact_match"
    predictions[selected] = data.labels[selected]
    provenance[selected] = "observed"
    if not np.isfinite(predictions).all() or np.any((predictions < -1e-12) | (predictions > 1 + 1e-12)):
        raise RuntimeError("invalid IDW probability")
    predictions = np.clip(predictions, 0.0, 1.0)
    unjudged = ~selected
    result = binary_metrics(data.labels[unjudged], predictions[unjudged])
    per_agent = {}
    for agent, (ids, _, _) in blocks.items():
        targets = ids[unjudged[ids]]
        per_agent[agent] = {
            **binary_metrics(data.labels[targets], predictions[targets]),
            "n": len(ids), "selected_count": int(selected[ids].sum()), "unjudged_count": len(targets),
            "provenance_counts": dict(Counter(provenance[targets].tolist())),
        }
    selected_concepts = {data.concepts[i] for i in selected_ids if data.concepts[i]}
    all_concepts = {key for key in data.concepts if key}
    result.update(
        selected_count=int(selected.sum()), unjudged_count=int(unjudged.sum()),
        macro_agent_accuracy=float(np.mean([r["accuracy"] for r in per_agent.values() if r["accuracy"] is not None])),
        combined_accuracy=float(np.mean((predictions >= 0.5) == data.labels)),
        aggregate_rate_error=float(abs(predictions.mean() - data.labels.mean())),
        concept_coverage=len(selected_concepts) / len(all_concepts) if all_concepts else None,
        agent_coverage=len({data.agents[i] for i in selected_ids}) / len(set(data.agents)),
        provenance_counts=dict(Counter(provenance[unjudged].tolist())), per_agent=per_agent,
    )
    return result, predictions, provenance


def summarize(rows: Sequence[dict[str, Any]], tolerance: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset_id"], row["mode"], row["dimension"]), []).append(row)
    summary = []
    for (dataset, mode, dimension), values in sorted(groups.items()):
        item: dict[str, Any] = {"dataset_id": dataset, "mode": mode, "dimension": dimension}
        for metric in SUMMARY_METRICS:
            available = [r[metric] for r in values if r[metric] is not None]
            item[metric] = float(np.mean(available)) if available else None
        seed_deltas = [
            np.mean([r["accuracy_delta_native"] for r in values if r["seed"] == seed])
            for seed in sorted({r["seed"] for r in values})
        ]
        center = float(np.mean(seed_deltas))
        if len(seed_deltas) > 1:
            half = float(student_t.ppf(0.975, len(seed_deltas) - 1) * np.std(seed_deltas, ddof=1) / math.sqrt(len(seed_deltas)))
            item["accuracy_delta_seed_ci95"] = [center - half, center + half]
        else:
            item["accuracy_delta_seed_ci95"] = None
        summary.append(item)
    decisions = []
    for dataset, mode in sorted({key[:2] for key in groups}):
        cells = [row for row in summary if row["dataset_id"] == dataset and row["mode"] == mode]
        reduced = [row for row in cells if row["dimension"] != 1536]
        # Require the candidate and every larger tested prefix to pass, avoiding a lucky isolated point.
        non_degrading = [
            r["dimension"] for r in reduced
            if all(x["accuracy_delta_native"] >= 0 for x in reduced if x["dimension"] >= r["dimension"])
        ]
        candidates = [
            r["dimension"] for r in reduced
            if all(x["accuracy_delta_seed_ci95"] is not None and x["accuracy_delta_seed_ci95"][0] >= -tolerance
                   for x in reduced if x["dimension"] >= r["dimension"])
        ]
        decisions.append({
            "dataset_id": dataset, "mode": mode,
            "smallest_non_degrading_dimension": min(non_degrading) if non_degrading else None,
            "one_pp_candidate_dimension": min(candidates) if candidates else None,
            "reason": (
                "Exploratory grid decision: all larger tested prefixes must also pass. "
                "Non-degrading means pooled observed accuracy delta >= 0; candidate means the "
                f"pointwise 95% seed-sensitivity interval lower bound >= -{tolerance:g}. "
                "Neither is a losslessness guarantee, multiplicity-adjusted test, independently "
                "held-out cutoff validation, or protection for every budget, schedule, or agent."
            ),
        })
    return summary, decisions


def run_experiment(
    inputs: Mapping[str, Path], output: Path, *, source_revision: str,
    dimensions: Sequence[int] = DIMENSIONS, seeds: Sequence[int] = tuple(range(13, 43)),
    rates: Sequence[float] = RATES, schedules: Sequence[str] = SCHEDULES,
    resume: bool = False, accuracy_tolerance: float = 0.01,
) -> dict[str, Any]:
    if not inputs or set(inputs) - set(DATASETS):
        raise ValueError("provide at least one recognized dataset input")
    if not dimensions or dimensions[0] != 1536 or len(set(dimensions)) != len(dimensions):
        raise ValueError("dimensions must be unique and begin with native 1536")
    if any(isinstance(d, bool) or not isinstance(d, int) or d < 1 or d > 1536 for d in dimensions):
        raise ValueError("dimensions must be integers in [1,1536]")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    if not rates or len(set(rates)) != len(rates) or any(not 0 < r < 1 for r in rates):
        raise ValueError("rates must be unique fractions between zero and one")
    if not schedules or len(set(schedules)) != len(schedules) or set(schedules) - set(SCHEDULES):
        raise ValueError("schedules must be unique known schedule names")
    if not math.isfinite(accuracy_tolerance) or not 0 <= accuracy_tolerance < 1:
        raise ValueError("accuracy tolerance must be a fraction in [0,1)")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("output is nonempty; use a fresh run path or explicit --resume")
    loaded = {name: load_input(path) for name, path in inputs.items()}
    if any(name != data.dataset_id for name, data in loaded.items()):
        raise ValueError("dataset option and input manifest ID disagree")
    code_paths = [
        Path(__file__), Path("sampling_comparison/v4_idw.py"),
        Path("trace_sampling/cluster_index.py"), Path("trace_sampling/samplers.py"),
        Path("trace_sampling/vector_store.py"), Path("trace_sampling/reservoir.py"),
        Path("trace_sampling/stats.py"), Path("trace_sampling/backpressure.py"),
    ]
    protocol = {
        "dimensions": list(dimensions), "seeds": list(seeds), "rates": list(rates),
        "schedules": list(schedules), "modes": list(MODES), "idw": asdict(IDWConfig()),
        "tau": 0.55, "accuracy_tolerance": accuracy_tolerance,
        "budget_unit": "sessions", "budget_rule": "max(1,floor(N*rate))",
        "causal_donors": True, "causal_membership": False, "bootstrap": False,
        "live_embedding_calls": 0, "live_judge_calls": 0, "search_calls": 0,
    }
    compatibility = {
        "version": VERSION, "protocol": protocol,
        "inputs": {name: data.profile["input_manifest_sha256"] for name, data in loaded.items()},
        "code_hashes": {str(p): sha256_file(p) for p in code_paths},
    }
    fingerprint = hashlib.sha256(canonical(compatibility).encode()).hexdigest()
    config_path = output / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise ValueError("resume fingerprint differs: input, protocol or implementation changed")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("cannot resume without run_config.json")
    write_json(config_path, {**compatibility, "fingerprint": fingerprint})
    profiles = []
    all_rows = []
    membership_records = []
    agent_accumulators: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for dataset_id in DATASETS:
        if dataset_id not in loaded:
            profiles.append({
                "dataset_id": dataset_id, "status": "blocked",
                "reason": "No verifiable full-session 1536-dimensional embedding input bundle supplied; not run. See input readiness evidence for preparation blockers.",
            })
            continue
        data = loaded[dataset_id]
        n = len(data.unit_ids)
        profiles.append({
            "dataset_id": dataset_id, "status": "completed", "n": n, "agents": len(set(data.agents)),
            "positive_count": int(data.labels.sum()), "pass_rate": float(data.labels.mean()),
            "provenance": data.profile, "source_hashes": data.profile.get("source_hashes", {}),
            "modeled_float32_bytes": {str(d): n * d * 4 for d in dimensions},
        })
        native_memberships: dict[tuple[int, str, float], np.ndarray] = {}
        native_accuracy: dict[tuple[int, str, float], float] = {}
        for dimension in dimensions:
            vectors = normalized_prefix(data.vectors, dimension)
            blocks = distance_blocks(data, vectors)
            for seed in seeds:
                for schedule in schedules:
                    checkpoint = output / "checkpoints" / f"{dataset_id}-{dimension}-{seed}-{schedule}.json"
                    if checkpoint.exists():
                        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                        if payload["fingerprint"] != fingerprint:
                            raise ValueError(f"checkpoint fingerprint mismatch: {checkpoint}")
                    else:
                        order, times = build_schedule(data.unit_ids, data.agents, schedule, seed)
                        ranking, telemetry = rank_membership(
                            data.unit_ids, data.agents, data.signatures, vectors, order, times, seed,
                        )
                        rows = []
                        memberships = []
                        for rate in rates:
                            count = max(1, math.floor(n * rate))
                            key = (seed, schedule, rate)
                            native = ranking[:count] if dimension == 1536 else native_memberships[key]
                            for mode in MODES:
                                selected = ranking[:count] if mode == "end_to_end" else native
                                metrics, _, _ = evaluate(data, blocks, order, selected)
                                base_accuracy = metrics["accuracy"] if dimension == 1536 else native_accuracy[key]
                                intersection = len(set(selected) & set(native))
                                row = {
                                    "dataset_id": dataset_id, "dimension": dimension, "seed": seed,
                                    "schedule": schedule, "rate": rate, "mode": mode, **metrics,
                                    "accuracy_delta_native": metrics["accuracy"] - base_accuracy,
                                    "membership_jaccard_native": intersection / len(set(selected) | set(native)),
                                    "membership_sha256": hashlib.sha256(canonical(sorted(selected.tolist())).encode()).hexdigest(),
                                    **telemetry,
                                }
                                rows.append(row)
                                memberships.append({
                                    "dataset_id": dataset_id, "dimension": dimension, "seed": seed,
                                    "schedule": schedule, "rate": rate, "mode": mode,
                                    "selected_indices": selected.tolist(), "membership_sha256": row["membership_sha256"],
                                    "order_sha256": hashlib.sha256(canonical(order.tolist()).encode()).hexdigest(),
                                })
                        payload = {"fingerprint": fingerprint, "rows": rows, "memberships": memberships}
                        write_json(checkpoint, payload)
                    if len(payload["rows"]) != len(rates) * len(MODES):
                        raise ValueError(f"checkpoint has incomplete rows: {checkpoint}")
                    if dimension == 1536:
                        for record in payload["memberships"]:
                            if record["mode"] == "end_to_end":
                                native_memberships[(seed, schedule, record["rate"])] = np.asarray(record["selected_indices"])
                        for row in payload["rows"]:
                            if row["mode"] == "end_to_end":
                                native_accuracy[(seed, schedule, row["rate"])] = row["accuracy"]
                    for row in payload["rows"]:
                        for agent, metrics in row["per_agent"].items():
                            agent_key = (dataset_id, row["mode"], dimension, agent)
                            acc = agent_accumulators.setdefault(agent_key, {
                                "count": 0, "metric_count": 0, "accuracy": 0.0, "mae": 0.0,
                                "n": 0, "selected_count": 0, "unjudged_count": 0, "provenance_counts": Counter(),
                            })
                            acc["count"] += 1
                            if metrics["accuracy"] is not None:
                                acc["metric_count"] += 1
                                acc["accuracy"] += metrics["accuracy"]
                                acc["mae"] += metrics["mae"]
                            for name in ("n", "selected_count", "unjudged_count"):
                                acc[name] += metrics[name]
                            acc["provenance_counts"].update(metrics["provenance_counts"])
                        all_rows.append({key: value for key, value in row.items() if key != "per_agent"})
                    membership_records.extend(payload["memberships"])
                print(f"{dataset_id}: dimension={dimension} seed={seed} complete", flush=True)
    expected = len(loaded) * len(dimensions) * len(seeds) * len(schedules) * len(rates) * len(MODES)
    keys = [(r["dataset_id"], r["dimension"], r["seed"], r["schedule"], r["rate"], r["mode"]) for r in all_rows]
    if len(all_rows) != expected or len(set(keys)) != expected:
        raise RuntimeError("missing or duplicate result cells")
    summary, decisions = summarize(all_rows, accuracy_tolerance)
    agent_summary = []
    for (dataset_id, mode, dimension, agent), acc in sorted(agent_accumulators.items()):
        agent_summary.append({
            "dataset_id": dataset_id, "mode": mode, "dimension": dimension, "agent_id": agent,
            **{name: acc[name] / acc["metric_count"] if acc["metric_count"] else None for name in ("accuracy", "mae")},
            **{name: acc[name] / acc["count"] for name in ("n", "selected_count", "unjudged_count")},
            "provenance_counts": {name: count / acc["count"] for name, count in acc["provenance_counts"].items()},
            "cell_count": acc["count"], "cells_with_unjudged_targets": acc["metric_count"],
        })
    membership_path = output / "memberships.jsonl.gz"
    with gzip.open(membership_path, "wt", encoding="utf-8") as stream:
        for record in membership_records:
            stream.write(canonical(record) + "\n")
    result = {
        "version": VERSION, "run_id": output.name, "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if len(loaded) == len(DATASETS) else "partial",
        "source_revision": source_revision, "protocol": protocol, "datasets": profiles,
        "rows": all_rows, "summary": summary, "decisions": decisions, "agent_summary": agent_summary,
        "files": {"aggregate": str(output / "aggregate.json"), "memberships": str(membership_path)},
        "validation": {"expected_rows": expected, "actual_rows": len(all_rows), "unique_rows": len(set(keys)),
                       "representation_fallbacks": sum(r["representation_fallbacks"] for r in all_rows)},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    write_json(output / "aggregate.json", result)
    write_json(output / "manifest.json", {
        "version": VERSION, "status": result["status"], "fingerprint": fingerprint,
        "source_revision": source_revision, "code_hashes": compatibility["code_hashes"],
        "files": {name: {"path": path, "sha256": sha256_file(Path(path))} for name, path in result["files"].items()},
    })
    return result
