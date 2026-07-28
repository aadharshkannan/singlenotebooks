from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Iterable, Optional

import pandas as pd

from trace_sampling.model import SessionEvent, Trace
from trace_sampling.samplers import AdaptiveSampler, SamplerConfig
from trace_sampling.variety import ExactSignatureIndex
from trace_sampling.variety_metrics import (
    cluster_agreement,
    concept_coverage,
    novel_concept_latency,
    novel_concept_latency_traces,
    redundancy_per_concept,
)

from .config import MinHashConfig
from .index import MinHashClusterIndex
from .signature import MinHashSignatureProvider
from .signature import MinHashBuildError


def _events(text: str, tool: str, output: str) -> tuple[SessionEvent, ...]:
    return (
        SessionEvent(role="system", text="follow policy and solve task"),
        SessionEvent(role="user", text=text),
        SessionEvent(role="assistant", text="working"),
        SessionEvent(role="tool", tool_name=tool, arguments={"q": text}, output=output),
        SessionEvent(role="assistant", text=output),
    )


def make_minhash_demo_stream(seed: int = 13) -> list[Trace]:
    # Deterministic stream with exact-signature collisions across concepts.
    concept_templates = {
        0: [
            ("reset account password quickly", "search", "password reset completed"),
            ("recover access credentials", "search", "access restored"),
            ("unlock user account", "search", "account unlocked"),
        ],
        1: [
            ("book customer refund", "search", "refund posted"),
            ("issue reimbursement", "search", "reimbursement confirmed"),
            ("cancel and refund order", "search", "order refunded"),
        ],
        2: [
            ("schedule deployment to production", "deploy", "deployment complete"),
            ("release service update", "deploy", "release complete"),
            ("promote build artifact", "deploy", "promotion complete"),
        ],
        3: [
            ("query invoice status", "lookup", "invoice located"),
            ("fetch billing record", "lookup", "billing record found"),
            ("inspect payment history", "lookup", "payment history loaded"),
        ],
    }

    traces: list[Trace] = []
    tid = 0
    ts = 0.0

    # Burst of concept 0 duplicates and lexical near duplicates.
    for _ in range(6):
        text, tool, out = concept_templates[0][0]
        traces.append(Trace(tid, "agent-a", ts, (tool,), 1, 10.0, "ok", concept_id=0, events=_events(text, tool, out)))
        tid += 1
        ts += 0.4
    for idx in (1, 2):
        text, tool, out = concept_templates[0][idx]
        traces.append(Trace(tid, "agent-a", ts, (tool,), 1, 10.0, "ok", concept_id=0, events=_events(text, tool, out)))
        tid += 1
        ts += 0.6

    # Interleaved concepts with deliberate exact-signature collisions:
    # concept 0 and 1 both use tool signature ("search",) but represent different tasks.
    for round_idx in range(5):
        for concept_id in (1, 2, 3):
            v = round_idx % 3
            text, tool, out = concept_templates[concept_id][v]
            signature = (tool,)
            traces.append(
                Trace(tid, "agent-a" if concept_id != 3 else "agent-b", ts, signature, 1, 12.0, "ok", concept_id=concept_id, events=_events(text, tool, out))
            )
            tid += 1
            ts += 0.9

    # Additional near-duplicate collision block where exact baseline must map both
    # concepts into one key, while MinHash can still split by lexical evidence.
    collision_block = [
        ("search", "reset enterprise password now", "password reset complete", 0),
        ("search", "refund duplicate charge now", "refund complete", 1),
        ("search", "reset account credentials", "access restored", 0),
        ("search", "issue customer reimbursement", "reimbursement confirmed", 1),
    ]
    for tool, text, out, cid in collision_block:
        traces.append(Trace(tid, "agent-a", ts, (tool,), 1, 8.0, "ok", concept_id=cid, events=_events(text, tool, out)))
        tid += 1
        ts += 0.5

    # Long gap then returning behavior for TTL novelty re-flag.
    ts += 120.0
    text, tool, out = concept_templates[0][1]
    traces.append(Trace(tid, "agent-a", ts, (tool,), 1, 11.0, "ok", concept_id=0, events=_events(text, tool, out)))
    tid += 1
    ts += 0.5

    # Truly empty evidence exercises exact-signature fallback.
    traces.append(Trace(tid, "agent-a", ts, (), 0, 9.0, "ok", concept_id=0, events=()))

    return traces


@dataclass(frozen=True)
class ArmMetrics:
    keep_count: int
    concept_coverage: float
    ari: float
    v_measure: float
    redundancy_mean: float
    novelty_latency_s_mean: float
    novelty_latency_traces_mean: float
    decision_latency_ms_p50: float
    decision_latency_ms_p95: float
    pair_separation_accuracy: float
    cluster_purity: float
    cluster_count: int
    clusters_per_concept_mean: float


@dataclass(frozen=True)
class MinHashExperimentResult:
    config: dict
    exact: ArmMetrics
    minhash: ArmMetrics
    signature_jaccard_mae: float
    telemetry: dict[str, dict]
    separation_gain: float
    purity_gain: float

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "exact": asdict(self.exact),
            "minhash": asdict(self.minhash),
            "signature_jaccard_mae": self.signature_jaccard_mae,
            "telemetry": self.telemetry,
            "separation_gain": self.separation_gain,
            "purity_gain": self.purity_gain,
        }


def _latency_summary(latencies_ms: list[float]) -> tuple[float, float]:
    if not latencies_ms:
        return 0.0, 0.0
    ordered = sorted(latencies_ms)
    p50 = ordered[int(0.50 * (len(ordered) - 1))]
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    return float(p50), float(p95)


def _pair_separation_accuracy(log: pd.DataFrame) -> float:
    rows = log[["concept_id", "variety_key"]].to_dict("records")
    correct = 0
    total = 0
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            same_concept = left["concept_id"] == right["concept_id"]
            same_cluster = left["variety_key"] == right["variety_key"]
            correct += int(same_concept == same_cluster)
            total += 1
    return correct / total if total else 0.0


def _cluster_purity(log: pd.DataFrame) -> float:
    if log.empty:
        return 0.0
    majority = 0
    for _, group in log.groupby("variety_key"):
        majority += int(group["concept_id"].value_counts().iloc[0])
    return majority / len(log)


def _arm_metrics(log: pd.DataFrame, latencies_ms: list[float]) -> ArmMetrics:
    kept = log[log["kept"]]
    red = redundancy_per_concept(log)
    lat_s = novel_concept_latency(log)
    lat_t = novel_concept_latency_traces(log)
    finite_s = [v for v in lat_s.values() if v != float("inf")]
    finite_t = [v for v in lat_t.values() if v != float("inf")]
    p50, p95 = _latency_summary(latencies_ms)
    # Agreement focuses on kept traces only.
    ari, vm = cluster_agreement(kept if not kept.empty else log)
    clusters_per_concept = log.groupby("concept_id")["variety_key"].nunique()
    return ArmMetrics(
        keep_count=int(kept.shape[0]),
        concept_coverage=float(concept_coverage(log)),
        ari=float(ari),
        v_measure=float(vm),
        redundancy_mean=float(statistics.mean(red.values()) if red else 0.0),
        novelty_latency_s_mean=float(statistics.mean(finite_s) if finite_s else float("inf")),
        novelty_latency_traces_mean=float(statistics.mean(finite_t) if finite_t else float("inf")),
        decision_latency_ms_p50=p50,
        decision_latency_ms_p95=p95,
        pair_separation_accuracy=float(_pair_separation_accuracy(log)),
        cluster_purity=float(_cluster_purity(log)),
        cluster_count=int(log["variety_key"].nunique()),
        clusters_per_concept_mean=float(clusters_per_concept.mean() if len(clusters_per_concept) else 0.0),
    )


def _run_sampler(stream: Iterable[Trace], sampler: AdaptiveSampler) -> tuple[pd.DataFrame, list[float]]:
    rows: list[dict] = []
    latencies_ms: list[float] = []
    for trace in stream:
        started = time.perf_counter_ns()
        kept = sampler.decide(trace)
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        obs = sampler.last_observation
        rows.append(
            {
                "trace_id": trace.trace_id,
                "timestamp": trace.timestamp,
                "agent_id": trace.agent_id,
                "concept_id": trace.concept_id,
                "signature": trace.signature,
                "variety_key": str(obs.key.value),
                "key_kind": obs.key.kind,
                "kept": bool(kept),
            }
        )
    return pd.DataFrame(rows), latencies_ms


def _signature_calibration_mae(stream: list[Trace], provider: MinHashSignatureProvider) -> float:
    # Compare exact shingle Jaccard vs MinHash estimate across nearby pairs.
    errors: list[float] = []
    for i in range(len(stream) - 1):
        a = stream[i]
        b = stream[i + 1]
        try:
            rec_a = provider.build(a)
            rec_b = provider.build(b)
        except MinHashBuildError:
            continue
        est = sum(1 for x, y in zip(rec_a.signature, rec_b.signature) if x == y) / len(rec_a.signature)
        exact = provider.shingle_jaccard(a, b)
        errors.append(abs(est - exact))
    return float(statistics.mean(errors) if errors else 0.0)


def run_minhash_experiment(
    *,
    ngram_size: int = 3,
    permutations: int = 128,
    threshold: float = 0.50,
    ttl_s: float = 60.0,
) -> MinHashExperimentResult:
    stream = make_minhash_demo_stream(seed=13)
    # Keep all traces for clustering-quality evaluation while exercising the real sampler seam.
    sampler_cfg = SamplerConfig(
        llm_throughput=1000.0,
        active_window=20.0,
        max_signatures_per_agent=256,
        agent_floor=1.0,
    )

    exact_index = ExactSignatureIndex(max_signatures_per_agent=256)
    exact_sampler = AdaptiveSampler(sampler_cfg, seed=13, variety_index=exact_index, use_novelty=True)
    exact_log, exact_lat = _run_sampler(stream, exact_sampler)

    mh_cfg = MinHashConfig(
        ngram_size=ngram_size,
        permutations=permutations,
        similarity_threshold=threshold,
        ttl_s=ttl_s,
        purge_every=1,
        seed=13,
        retain_debug_shingles=True,
    )
    provider = MinHashSignatureProvider(mh_cfg)
    mh_index = MinHashClusterIndex(mh_cfg, provider)
    mh_sampler = AdaptiveSampler(sampler_cfg, seed=13, variety_index=mh_index, use_novelty=True)
    mh_log, mh_lat = _run_sampler(stream, mh_sampler)

    exact_metrics = _arm_metrics(exact_log, exact_lat)
    mh_metrics = _arm_metrics(mh_log, mh_lat)

    calibration_mae = _signature_calibration_mae(stream, provider)

    return MinHashExperimentResult(
        config={
            "ngram_size": ngram_size,
            "permutations": permutations,
            "similarity_threshold": threshold,
            "ttl_s": ttl_s,
        },
        exact=exact_metrics,
        minhash=mh_metrics,
        signature_jaccard_mae=calibration_mae,
        telemetry={
            "minhash_index": mh_index.telemetry(),
            "signature_provider": {
                "builds": provider.n_builds,
                "hits": provider.n_hits,
                "truncations": provider.n_truncations,
            },
        },
        separation_gain=mh_metrics.pair_separation_accuracy - exact_metrics.pair_separation_accuracy,
        purity_gain=mh_metrics.cluster_purity - exact_metrics.cluster_purity,
    )


def sweep_minhash_experiments() -> list[MinHashExperimentResult]:
    out = []
    for n in (2, 3, 4):
        for perms in (64, 128):
            for thr in (0.3, 0.5, 0.7):
                out.append(run_minhash_experiment(ngram_size=n, permutations=perms, threshold=thr))
    return out


def save_experiment_result(path: str | Path, result: MinHashExperimentResult) -> None:
    payload = result.to_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_experiment_sweep(path: str | Path, results: list[MinHashExperimentResult]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([r.to_dict() for r in results], indent=2, sort_keys=True),
        encoding="utf-8",
    )
