"""Offline driver that measures how well snap IDW-imputed evals recover the
true value of the traces the adaptive sampler DROPS (never judges).

The adaptive sampler keeps a biased minority of traces (rare / novel); in
production the expensive eval (LLM judge) runs only on that kept minority. Every
DROPPED trace therefore has no eval -- unless we impute one. `ValuePipeline` /
`ClusterValueReservoir` give each dropped trace a snap inverse-distance-weighted
(IDW) imputed eval from nearby judged neighbours of its own agent-scoped cluster.

This module attaches a synthetic ground-truth value field (a smooth function of
embedding position + observation noise) to a completed `adaptive_cluster` run,
replays that run through the REAL `ValuePipeline` with a synchronous judge, and
scores, over the DROPPED traces, the squared error of two imputers against each
trace's true value:

  * agent-mean fill -- predict the causal running mean of the agent's judged
    evals so far (the embedding-free predictor)
  * idw (pipeline)  -- the reservoir's neighbourhood imputation, with its
    agent/global/prior fallback for cold clusters

Errors are aggregated per agent and overall. Because there are thousands of
dropped traces (not a handful of per-agent means), the comparison is
statistically stable. It reuses the run's already-cached embeddings, so it makes
NO new embedding calls. See
docs/superpowers/specs/2026-07-09-snap-imputed-evals-cluster-value-reservoir-design.md
section 5.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .variety import VarietyKey, VarietyObservation
from .value_reservoir import ClusterValueReservoir, _Running
from .value_pipeline import ValuePipeline

ValueField = Callable[[np.ndarray], float]


def make_smooth_value_field(dim: int, seed: int = 20260710,
                            n_dirs: int = 8, scale: float = 6.0) -> ValueField:
    """A deterministic scalar field over the unit sphere that varies smoothly
    with embedding position, so nearby embeddings have similar values (exactly
    the structure IDW exploits) while distinct behaviour clusters get distinct
    values. Value is a sinusoidal mixture of a few fixed random projections,
    squashed into [0, 1]; `scale` controls how strongly value depends on which
    behaviour cluster a trace belongs to."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_dirs, dim))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    freqs = rng.uniform(0.5, 1.5, size=n_dirs)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_dirs)
    amps = rng.uniform(size=n_dirs)
    amps /= amps.sum()

    def field(vec: np.ndarray) -> float:
        u = np.asarray(vec, dtype=np.float64)
        n = np.linalg.norm(u)
        if n:
            u = u / n
        s = float(np.sum(amps * np.sin(freqs * (W @ u) * scale + phases)))
        return 0.5 + 0.5 * s  # smooth, in [0, 1]

    return field


class ReplaySampler:
    """Replays a completed run's per-trace keep/drop decision and variety key
    without re-running the real sampler (so no state drift, no Azure calls). Its
    `decide`/`last_observation` surface matches what `ValuePipeline` reads."""

    def __init__(self, kept: List[bool], kinds: List[str], keys: List[str]):
        self._kept = list(kept)
        self._kinds = list(kinds)
        self._keys = list(keys)
        self._i = 0
        self.last_observation: Optional[VarietyObservation] = None

    def decide(self, trace) -> bool:
        i = self._i
        self._i += 1
        self.last_observation = VarietyObservation(
            VarietyKey(self._kinds[i], self._keys[i]), rarity=0.0, novelty=0.0)
        return bool(self._kept[i])


@dataclass(frozen=True)
class ValueSimResult:
    per_agent: pd.DataFrame   # per agent: n_dropped, idw_covered, agent_fill_mse, idw_mse, mse_reduction_pct
    agent_fill_mse: float     # overall MSE of the agent-mean-fill imputer over dropped traces
    idw_mse: float            # overall MSE of the pipeline's IDW imputer over dropped traces
    idw_coverage: float       # fraction of dropped traces served by the IDW path (vs a mean fallback)
    n_dropped: int            # dropped embedded traces analysed
    per_trace: pd.DataFrame   # one row per dropped trace: agent_id, true, idw_pred, agent_fill_pred, provenance


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def run_value_simulation(stream, cluster_result, *, field: Optional[ValueField] = None,
                         scale: float = 8.0, k: int = 64, power: float = 2.0,
                         sigma: float = 0.05, seed: int = 20260710) -> ValueSimResult:
    """Replay `cluster_result` (a RunResult from an `adaptive_cluster*` arm) over
    the real `ValuePipeline`, judging kept traces with a synthetic value field and
    imputing dropped traces, then score the pipeline's IDW imputation of each
    DROPPED trace against an agent-mean-fill baseline.

    `scale` controls how strongly the (auto-built) value field varies across
    behaviour clusters -- i.e. how much genuine, behaviour-specific value there is
    for IDW to recover (ignored if an explicit `field` is passed). `sigma` is the
    judge observation noise.

    Only traces that received an embedding (cluster-assigned; signature already in
    the run's cache) are analysed -- rare fallback-signature traces have no
    embedding and are excluded. No new embeddings are computed."""
    cache = cluster_result.index._cache
    log = cluster_result.log
    kept_flags = log["kept"].tolist()
    kinds = log["key_kind"].tolist()
    keys = log["variety_key"].tolist()
    if len(kept_flags) != len(stream):
        raise ValueError("stream length and run log length differ; pass the same "
                         "stream the run consumed")

    if field is None:
        dim = None
        for trace in stream:
            if trace.signature in cache:
                dim = cache.get(trace.signature).shape[0]
                break
        if dim is None:
            raise ValueError("no embedded traces in this run; cannot build a value field")
        field = make_smooth_value_field(dim, seed=seed, scale=scale)

    def judge_noise(trace_id: int) -> float:
        return float(np.random.default_rng([seed, int(trace_id)]).normal(0.0, sigma))

    sampler = ReplaySampler(kept_flags, kinds, keys)
    reservoir = ClusterValueReservoir(k=k, power=power)

    # Causal agent-mean-fill baseline: the running mean of each agent's judged
    # evals seen so far (the embedding-free predictor for a dropped trace).
    agent_running: Dict[str, _Running] = {}
    global_running = _Running()
    drop_rows: List[dict] = []
    pending_drop: Dict[int, tuple] = {}   # trace_id -> (agent_id, true_value) awaiting its imputation

    def sink(tv) -> None:
        if tv.provenance == "pending":
            return
        row = pending_drop.pop(tv.trace_id, None)
        if row is None:
            return  # a judged (kept) trace; not scored here
        agent_id, true_v = row
        am = agent_running.get(agent_id)
        fill = am.mean if (am and am.count) else (
            global_running.mean if global_running.count else reservoir.prior)
        drop_rows.append(dict(
            agent_id=agent_id, true=true_v, idw_pred=tv.value,
            agent_fill_pred=fill, provenance=tv.provenance))

    def sync_judge(trace, on_done) -> None:
        # Judge only embedded traces so no new embedding is ever triggered.
        if trace.signature in cache:
            v = _clip01(field(cache.get(trace.signature)) + judge_noise(trace.trace_id))
            agent_running.setdefault(trace.agent_id, _Running()).update(v)
            global_running.update(v)
            on_done(v)

    pipe = ValuePipeline(sampler, cache, reservoir,
                         submit_judge=sync_judge, on_value=sink)

    for i, trace in enumerate(stream):
        embedded = trace.signature in cache
        if embedded and not kept_flags[i]:
            # Register the dropped trace's ground truth before process() imputes it.
            pending_drop[trace.trace_id] = (trace.agent_id, field(cache.get(trace.signature)))
        pipe.process(trace)

    per_trace = pd.DataFrame(drop_rows)
    if per_trace.empty:
        raise ValueError("no dropped embedded traces to score")

    def _mse(sub: pd.DataFrame, col: str) -> float:
        return float(np.mean((sub[col] - sub["true"]) ** 2))

    rows = []
    for agent_id, sub in per_trace.groupby("agent_id"):
        a_mse = _mse(sub, "agent_fill_pred")
        i_mse = _mse(sub, "idw_pred")
        rows.append(dict(
            agent_id=agent_id, n_dropped=len(sub),
            idw_covered=float((sub["provenance"] == "idw").mean()),
            agent_fill_mse=a_mse, idw_mse=i_mse,
            mse_reduction_pct=100.0 * (1.0 - i_mse / a_mse) if a_mse else float("nan")))
    per_agent = pd.DataFrame(rows).set_index("agent_id")

    return ValueSimResult(
        per_agent=per_agent,
        agent_fill_mse=_mse(per_trace, "agent_fill_pred"),
        idw_mse=_mse(per_trace, "idw_pred"),
        idw_coverage=float((per_trace["provenance"] == "idw").mean()),
        n_dropped=len(per_trace),
        per_trace=per_trace)
