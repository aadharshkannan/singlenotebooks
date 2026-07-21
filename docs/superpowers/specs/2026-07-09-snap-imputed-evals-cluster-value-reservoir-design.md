# Snap Imputed Evals via a Cluster Value Reservoir — Design

**Date:** 2026-07-09
**Status:** Approved (spec review passed), pending user review
**Builds on:** `2026-07-09-similarity-calibrated-keep-signal-design.md` (agent-scoped embedding clusters + `AzureClusterIndex`)

---

## 1. Problem

In production the adaptive sampler keeps only a biased minority of traces (rare / novel / stale
behaviors), and the expensive eval (LLM-judge) runs **only on kept traces** and returns
**asynchronously**. Two consequences:

- A naive "average the judged (kept) traces' evals" is a **biased** estimate of the true population
  eval, because the kept set is non-random with respect to behavior — and behavior correlates with
  eval.
- The dropped majority never gets an eval at all, so there is no real-time, full-population view of
  quality.

**Goal:** give **every** trace a value in real time — the true judge eval for the kept minority, and
a **snap imputed eval** for the dropped majority — so downstream gets a continuous, de-biased,
full-population estimate. The imputed value for a dropped candidate is an **inverse-distance-weighted
(IDW) average of the evals of nearby, already-judged members of the same agent-scoped cluster**.

## 2. Constraints & boundaries

- **Online, causal, snap.** A dropped trace's imputation must be produced at decision time in
  microseconds, using only evals that have **already returned** (no future leakage).
- **Do not touch variety scoring.** `AzureClusterIndex` scores *variety*; this component estimates
  *value*. They share the `cluster_id` the index already assigns, but neither's correctness depends
  on the other. The keep/drop decision is **never** influenced by the imputed value.
- **Bounded memory**, `O(#clusters · k) + O(#agents)`.
- **No extra embeddings on the hot path.** Imputation reads the embedding the index already cached
  during `decide`; it never triggers a new embed.
- **Async-framework-agnostic.** The judge is wired through a submit-with-callback seam so the same
  code works with asyncio, a thread pool, a queue worker, or a synchronous judge in tests.
- **Agent-scoped neighborhoods.** Clusters are already agent-scoped (`store.nearest(agent_id=...)`),
  so a dropped candidate borrows only from kept members of its own agent's matched cluster — no
  cross-agent leakage.

## 3. Design

### 3.1 Where it sits in the per-trace flow

1. `kept = sampler.decide(trace)` — unchanged.
2. `obs = sampler.last_observation` → `cluster_id = obs.key.value if obs.key.kind == "cluster" else
   None`; `vec = cache.get(trace.signature) if trace.signature in cache else None`.
3. **Dropped** → `reservoir.impute(cluster_id, agent_id, vec)` returns `(value, provenance, …)`
   immediately.
4. **Kept** → enqueue the real judge; on completion `reservoir.record_eval(cluster_id, agent_id,
   vec, value)`. The reservoir only ever holds already-returned evals → causal by construction.

Two failure modes handled up front:
- `kind == "fallback-signature"` (circuit-breaker / embed-budget exhaustion): no cluster and
  possibly no cached vector → `cluster_id`/`vec` are `None`, imputation routes to the mean fallbacks.
- Empty cluster reservoir (cold start): the fallback chain (§3.3) supplies a value.

### 3.2 Component interface & state (`trace_sampling/value_reservoir.py`)

```python
@dataclass(frozen=True)
class Imputation:
    value: float
    provenance: str      # "idw" | "agent_mean" | "global_mean" | "prior"
    n_donors: int        # reservoir members used (0 for non-idw)
    nearest_dist: float  # min (1 - cosine) to a donor; NaN if none

class ClusterValueReservoir:
    def __init__(self, k: int = 64, power: float = 2.0,
                 eps: float = 1e-6, prior: float = 0.5, ttl: float = 60.0):
        # validate: k >= 1, power > 0, eps > 0, ttl > 0
        self._members: dict[str, deque]       = {}   # cluster_id -> deque[(vec, value)] maxlen=k
        self._last_seen: dict[str, float]     = {}   # cluster_id -> last activity ts (self-TTL)
        self._agent_mean: dict[str, _Running] = {}   # agent_id  -> running mean of judged evals
        self._global: _Running                = _Running()
        self._lock = threading.Lock()

    def record_eval(self, cluster_id: str | None, agent_id: str,
                    vec: np.ndarray | None, value: float,
                    now: float | None = None) -> bool: ...   # returns True iff value accepted
    def impute(self, cluster_id: str | None, agent_id: str,
               vec: np.ndarray | None) -> Imputation: ...
    def purge_stale(self, now: float, ttl: float | None = None) -> list[str]: ...
    def evict(self, cluster_ids: Iterable[str]) -> None: ...
```

State notes:
- `_members[cluster_id]` is a `deque(maxlen=k)` — a ring buffer of the last `k` judged members,
  giving bounded memory and `O(k)` imputation. Recency bias is acceptable and desirable.
- `_Running` is a tiny incremental `(count, mean)` accumulator (Welford mean), `O(1)` update,
  `O(#agents)` memory.
- `record_eval` **always** updates the agent + global running means for any **finite** eval (even
  when `vec is None`), so the fallbacks keep improving under circuit-breaker conditions.
- The reservoir stores each **member's own embedding**, not the cluster centroid — IDW needs the
  actual points to weight by distance. This is the one piece of per-member state deliberately kept
  here (in a separate component) rather than in the index.

### 3.3 IDW + fallback semantics

`impute` returns the first applicable of:

```
1. IDW      — cluster_id present, vec present, reservoir non-empty:
                d_i   = max(0.0, 1 - cosine(vec, member_i))
                w_i   = 1 / (d_i + eps) ** power
                value = Σ w_i · v_i / Σ w_i
                provenance="idw", n_donors=len, nearest_dist=min d_i
2. agent_mean  — elif agent has a running mean:  value = _agent_mean[agent_id].mean
3. global_mean — elif global has any evals:       value = _global.mean
4. prior       — else:                            value = prior (default 0.5)
```

- **Locality via `power`** (single knob): `power=2` makes near neighbors dominate; `power→0`
  approaches an unweighted cluster mean.
- **Near-duplicate donor** (`d_i → 0`): the `eps` floor caps its weight at `1/eps**power` (huge but
  finite) → the estimate converges to that donor's value, no divide-by-zero.
- Cosine reuses the index's `_cos` helper (zero-norm → 0).
- Provenance is emitted on every call for observability (§3.5).

`record_eval` first **validates `value`** (non-finite / NaN → discard, no mean or member update, so
a bad judge response can never poison the estimators). For a finite value it always
`_global.update(value)` and `_agent_mean[agent_id].update(value)`; it appends `(vec, value)` to
`_members[cluster_id]` (created lazily) and refreshes `_last_seen[cluster_id]` **only when both
`cluster_id is not None` and `vec is not None`** — otherwise there is no valid cluster ring buffer to
attach the point to (fallback-signature / error paths still update the means but contribute no IDW
donor, and `_members[None]` is never created). Each judged trace is recorded exactly once by the
pipeline (no idempotency requirement). **`record_eval` returns `True` iff the value was accepted**
(finite) and `False` if it was discarded, so the pipeline can decide whether to emit `"judged"`.

### 3.4 Async-eval lifecycle & integration wrapper (`trace_sampling/value_pipeline.py`)

```python
SubmitJudge = Callable[[Trace, Callable[[float], None]], None]  # on_done(value) fires on any thread

@dataclass(frozen=True)
class TraceValue:
    trace_id: int
    kept: bool
    value: float | None       # imputed now (dropped); None until judge returns (kept)
    provenance: str           # "idw"|"agent_mean"|"global_mean"|"prior"|"pending"|"judged"

class ValuePipeline:
    def __init__(self, sampler, cache, reservoir, submit_judge,
                 on_value: Callable[[TraceValue], None] | None = None): ...

    def process(self, trace) -> TraceValue:
        kept = self.sampler.decide(trace)
        obs  = self.sampler.last_observation
        cid  = obs.key.value if obs.key.kind == "cluster" else None
        vec  = self.cache.get(trace.signature) if trace.signature in self.cache else None
        if kept:
            def _done(v):
                if self.reservoir.record_eval(cid, trace.agent_id, vec, v):   # True iff v accepted
                    self._emit(TraceValue(trace.trace_id, True, v, "judged"))
                # rejected (non-finite) eval: no member/mean update, no "judged" emission
            try:
                self.submit_judge(trace, _done)
            except Exception:
                pass  # submission failed: treat as never-judged, no reservoir update
            return self._emit(TraceValue(trace.trace_id, True, None, "pending"))
        imp = self.reservoir.impute(cid, trace.agent_id, vec)
        return self._emit(TraceValue(trace.trace_id, False, imp.value, imp.provenance))
```

- **No hot-path embed:** `vec` is read only if already cached; non-fallback traces were embedded
  inside `sampler.decide`, so this is a guaranteed cache hit; fallback-signature traces stay
  `vec=None`.
- **Sampler requirement:** `ValuePipeline` requires a sampler that populates `last_observation` on
  every `decide` (satisfied by `AdaptiveSampler`); this is a documented precondition of the wrapper,
  not a generic-sampler guarantee.
- **Judge failure handling:** the judge seam is best-effort and must not corrupt state or stall the
  hot path. `submit_judge` is invoked inside a guard — if submission itself raises, the trace is
  emitted as `"pending"` and treated as never-judged (no reservoir update). `_done` validates its
  value via `record_eval` (non-finite → discarded), so a bad or exceptional eval never poisons the
  estimators. A callback that never fires simply leaves that kept trace un-judged: no member is added
  and imputations fall back through the mean chain — correctness degrades gracefully rather than
  breaking. Timeouts/retries are the responsibility of the injected `submit_judge`, not the pipeline.
- **Value stream:** `on_value` fires at most twice per trace, in natural order — once immediately
  (imputed for drops, `"pending"` for keeps) and again with `"judged"` when a kept trace's real eval
  returns. `_emit(tv)` calls `on_value(tv)` if a sink is set and returns `tv`; because the second
  (`"judged"`) emission originates on a judge thread, the sink must be thread-safe (documented
  contract; the reservoir's own updates are already lock-guarded).
- **Eviction wiring:** the pipeline calls `reservoir.purge_stale(now, ttl)` on the same cadence the
  index purges (self-TTL keeps the index untouched); `evict(cluster_ids)` is also exposed for a
  caller that prefers to forward the index's purge output.

### 3.5 Memory, concurrency, observability

- **Memory:** reservoirs `O(#clusters · k)`; fallback means `O(#agents)`; self-TTL sweeps stale
  reservoirs (monotonic, never-reused `cluster_id`s → at worst bounded garbage until swept).
- **Concurrency:** a single `threading.Lock` guards `_members`/means; `record_eval` (judge threads)
  vs `impute` (hot path) critical sections are `O(k)` → negligible contention; near-free for
  single-thread asyncio.
- **Observability:** `provenance`, `n_donors`, `nearest_dist` emitted per imputation (not used in
  control flow). A rise in `"prior"`/`"global_mean"` signals cold clusters; large `nearest_dist`
  flags candidates far from their donors.

## 4. Testing

`tests/test_value_reservoir.py`:
1. **IDW math:** hand-computed 2–3-donor weighted mean; higher `power` → closer to nearest donor.
2. **eps floor:** a near-duplicate donor makes the estimate ≈ that donor's value; no divide-by-zero.
3. **Fallback chain order:** empty cluster → `agent_mean` → `global_mean` → `prior`;
   `record_eval` updates means even when `vec is None`; a `cluster_id=None` or `vec=None` eval
   creates **no** `_members` entry.
4. **Bad-value rejection:** a non-finite / NaN eval passed to `record_eval` updates neither the means
   nor any ring buffer.
5. **Ring-buffer cap:** more than `k` evals retains only the last `k`; imputation uses `k`.
6. **Self-TTL:** `purge_stale` drops a stale cluster reservoir and returns its id.
7. **Concurrency (light):** interleaved `record_eval`/`impute` under the lock stays consistent.

`tests/test_value_pipeline.py`:
8. **Drop path:** dropped trace returns an immediate imputed value + provenance.
9. **Keep path & causality:** a kept trace emits `"pending"` then `"judged"` with the true value;
   its eval affects only imputations produced **after** `_done` fires.
10. **No hot-path embed:** spy on the cache — a dropped fallback-signature trace yields `vec=None`,
    routes to a mean fallback, and triggers **zero** embed calls.
11. **Judge-submit failure:** if `submit_judge` raises, `process` still returns `"pending"` and no
    reservoir member is added.

## 5. Integration & simulation note

- The component is standalone; the eval harness / notebook can drive it by running the existing
  `adaptive_cluster` arm and feeding each trace through `ValuePipeline` with a synthetic
  `submit_judge` (optionally with a simulated delay) and a spatial value field, to demonstrate that
  IDW-imputed per-agent means track ground truth better than the naive kept-only mean. This offline
  driver is behaviorally identical to the production path (same causal reservoir) and is the natural
  place to validate estimator quality, but it is **not required** for the component to ship.

## 6. Out of scope (YAGNI)

- Distributed / Redis-shared reservoirs — design-noted for the sharding story only (route by
  `agent_id` for locality; shared tier as fallback). Not built.
- Age-decay weighting of donors — the ring-buffer's recency bias is sufficient.
- Live audit-sampling (occasionally judging a dropped trace to measure imputation error) — an
  observability follow-up.
- Feeding the imputed value back into the keep/drop decision — deliberately excluded to preserve the
  variety/value separation.
