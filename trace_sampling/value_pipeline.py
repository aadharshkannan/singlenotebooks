"""Per-trace wrapper wiring the sampler, embedding cache, ClusterValueReservoir
and an async judge together.

For each trace: run the sampler's keep/drop, read the cluster id and cached
embedding off last_observation (never triggering a new embed), impute a snap value
for drops, and submit the async judge for keeps — recording the returned eval back
into the reservoir causally (only already-returned evals are ever stored).

Requires a sampler that populates `last_observation` on every `decide` (satisfied
by AdaptiveSampler). See docs/superpowers/specs/2026-07-09-snap-imputed-evals-cluster-value-reservoir-design.md
"""
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .model import Trace
from .value_reservoir import ClusterValueReservoir

# on_done(value) may fire on any thread (asyncio task, thread pool, or synchronously).
SubmitJudge = Callable[[Trace, Callable[[float], None]], None]


@dataclass(frozen=True)
class TraceValue:
    trace_id: int
    kept: bool
    value: Optional[float]   # imputed now (dropped); None until judge returns (kept)
    provenance: str          # "idw"|"agent_mean"|"global_mean"|"prior"|"pending"|"judged"


class ValuePipeline:
    """Wires sampler + cache + reservoir + async judge per trace.

    process() must be called from a single hot-path thread: it reads
    sampler.last_observation immediately after decide(), and the n_* counters
    are non-atomic. Only judge callbacks (_done) may run on other threads.
    """

    def __init__(self, sampler, cache, reservoir: ClusterValueReservoir,
                 submit_judge: Optional[SubmitJudge],
                 on_value: "Optional[Callable[[TraceValue], None]]" = None,
                 purge_every: int = 200):
        """
        Args:
            on_value: optional sink called for every emitted TraceValue. May be
                invoked from both the process() thread ("pending"/imputed emissions)
                and a judge thread ("judged" emissions), so it must be thread-safe.
            purge_every: call reservoir.purge_stale() every this many traces to
                reclaim stale cluster buffers (must be >= 1).
        """
        if purge_every < 1:
            raise ValueError(f"purge_every must be >= 1, got {purge_every}")
        self.sampler = sampler
        self.cache = cache
        self.reservoir = reservoir
        self.submit_judge = submit_judge
        self.on_value = on_value
        self.purge_every = purge_every
        self.n_submit_failures = 0
        self.n_rejected_evals = 0
        self._since_purge = 0

    def _emit(self, tv: TraceValue) -> TraceValue:
        if self.on_value is not None:
            self.on_value(tv)
        return tv

    def process(self, trace: Trace) -> TraceValue:
        kept = self.sampler.decide(trace)
        obs = self.sampler.last_observation
        cid = obs.key.value if obs.key.kind == "cluster" else None
        vec = self.cache.get(trace.signature) if trace.signature in self.cache else None
        if kept:
            def _done(v: float) -> None:
                if self.reservoir.record_eval(cid, trace.agent_id, vec, v):
                    self._emit(TraceValue(trace.trace_id, True, v, "judged"))
                else:
                    self.n_rejected_evals += 1
            # Emit "pending" BEFORE submitting so a synchronous judge (one that
            # calls _done inline) still emits "judged" AFTER "pending" — the
            # natural-order contract holds for sync and async judges alike.
            pending = self._emit(TraceValue(trace.trace_id, True, None, "pending"))
            try:
                if self.submit_judge is not None:
                    self.submit_judge(trace, _done)
                # NOTE: for a synchronous judge, an exception escaping the judge
                # callback (including from _done / record_eval / on_value) is caught
                # here and counted as a submission failure; async judges run _done on
                # their own thread and must handle callback exceptions themselves.
            except Exception:
                self.n_submit_failures += 1
            tv = pending
        else:
            imp = self.reservoir.impute(cid, trace.agent_id, vec)
            tv = self._emit(TraceValue(trace.trace_id, False, imp.value, imp.provenance))

        self._since_purge += 1
        if self._since_purge >= self.purge_every:
            self.reservoir.purge_stale(now=time.monotonic())
            self._since_purge = 0
        return tv
