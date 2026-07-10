"""Per-trace wrapper wiring the sampler, embedding cache, ClusterValueReservoir
and an async judge together.

For each trace: run the sampler's keep/drop, read the cluster id and cached
embedding off last_observation (never triggering a new embed), impute a snap value
for drops, and submit the async judge for keeps — recording the returned eval back
into the reservoir causally (only already-returned evals are ever stored).

Requires a sampler that populates `last_observation` on every `decide` (satisfied
by AdaptiveSampler). See docs/superpowers/specs/2026-07-09-snap-imputed-evals-cluster-value-reservoir-design.md
"""
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
    def __init__(self, sampler, cache, reservoir: ClusterValueReservoir,
                 submit_judge: Optional[SubmitJudge],
                 on_value: "Optional[Callable[[TraceValue], None]]" = None):
        self.sampler = sampler
        self.cache = cache
        self.reservoir = reservoir
        self.submit_judge = submit_judge
        self.on_value = on_value

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
            # Emit "pending" BEFORE submitting so a synchronous judge (one that
            # calls _done inline) still emits "judged" AFTER "pending" — the
            # natural-order contract holds for sync and async judges alike.
            pending = self._emit(TraceValue(trace.trace_id, True, None, "pending"))
            try:
                if self.submit_judge is not None:
                    self.submit_judge(trace, _done)
            except Exception:
                pass  # submission failed: treat as never-judged, no reservoir update
            return pending
        imp = self.reservoir.impute(cid, trace.agent_id, vec)
        return self._emit(TraceValue(trace.trace_id, False, imp.value, imp.provenance))
