# Snap Imputed Evals via a Cluster Value Reservoir — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every trace a real-time value — the true async judge eval for kept traces and an IDW-imputed snap eval for dropped traces — via a standalone `ClusterValueReservoir` plus a thin `ValuePipeline` wrapper.

**Architecture:** `ClusterValueReservoir` holds bounded per-cluster ring buffers of `(embedding, judged_value)` plus per-agent and global running means. On a drop it computes an inverse-distance-weighted (IDW) average of nearby judged members of the same agent-scoped cluster, falling back to agent-mean → global-mean → prior. `ValuePipeline` reads the cluster id and cached embedding off the sampler's `last_observation` (no hot-path embed), imputes instantly for drops, and submits the async judge for keeps, recording the returned eval causally. Neither touches variety scoring or the keep/drop decision.

**Tech Stack:** Python 3, numpy, `collections.deque`, `threading.Lock`, pytest. Spec: `docs/superpowers/specs/2026-07-09-snap-imputed-evals-cluster-value-reservoir-design.md`.

---

## File Structure

- Create `trace_sampling/value_reservoir.py` — `Imputation` dataclass, `_Running` mean accumulator, `ClusterValueReservoir`. Pure/offline, no Azure. One responsibility: store judged values and impute a snap value.
- Create `trace_sampling/value_pipeline.py` — `TraceValue` dataclass, `SubmitJudge` type alias, `ValuePipeline`. One responsibility: wire sampler + cache + reservoir + async judge together per trace.
- Create `tests/test_value_reservoir.py` — unit tests for reservoir math, fallbacks, ring buffer, TTL, concurrency.
- Create `tests/test_value_pipeline.py` — unit tests for keep/drop paths, causality, no-embed guarantee, judge-submit failure.

Cosine reuses `trace_sampling.vector_store._cosine` (existing helper). No existing files are modified.

---

## Chunk 1: ClusterValueReservoir

### Task 1: `_Running` mean accumulator + `Imputation` dataclass + reservoir constructor/validation

**Files:**
- Create: `trace_sampling/value_reservoir.py`
- Test: `tests/test_value_reservoir.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_value_reservoir.py
import math
import time
import numpy as np
import pytest
from trace_sampling.value_reservoir import ClusterValueReservoir, Imputation, _Running


def test_running_mean_updates_incrementally():
    r = _Running()
    assert r.count == 0
    r.update(1.0); r.update(3.0)
    assert r.count == 2
    assert r.mean == pytest.approx(2.0)


def test_ctor_defaults_and_validation():
    res = ClusterValueReservoir()
    assert res.k == 64 and res.power == 2.0 and res.eps == 1e-6
    assert res.prior == 0.5 and res.ttl == 60.0
    for bad in dict(k=0), dict(power=0.0), dict(eps=0.0), dict(ttl=0.0):
        with pytest.raises(ValueError):
            ClusterValueReservoir(**bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_reservoir.py -q`
Expected: FAIL with `ModuleNotFoundError: trace_sampling.value_reservoir`

- [ ] **Step 3: Write minimal implementation**

```python
# trace_sampling/value_reservoir.py
"""Snap IDW-imputed eval values for dropped traces.

ClusterValueReservoir stores the judged eval of kept traces per agent-scoped
cluster (bounded ring buffer of (embedding, value)) plus per-agent and global
running means. When a trace is dropped, impute() returns an inverse-distance-
weighted average of nearby judged members of the same cluster, degrading through
agent-mean -> global-mean -> prior. It never influences keep/drop or variety
scoring; see docs/superpowers/specs/2026-07-09-snap-imputed-evals-cluster-value-reservoir-design.md
"""
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .vector_store import _cosine


@dataclass(frozen=True)
class Imputation:
    value: float
    provenance: str      # "idw" | "agent_mean" | "global_mean" | "prior"
    n_donors: int        # reservoir members used (0 for non-idw)
    nearest_dist: float  # min (1 - cosine) to a donor; NaN if none


class _Running:
    """Incremental (count, mean) accumulator. O(1) update, O(1) memory."""

    __slots__ = ("count", "mean")

    def __init__(self):
        self.count = 0
        self.mean = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.mean += (value - self.mean) / self.count


class ClusterValueReservoir:
    def __init__(self, k: int = 64, power: float = 2.0,
                 eps: float = 1e-6, prior: float = 0.5, ttl: float = 60.0):
        if not k >= 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not power > 0:
            raise ValueError(f"power must be > 0, got {power}")
        if not eps > 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if not ttl > 0:
            raise ValueError(f"ttl must be > 0, got {ttl}")
        self.k = k
        self.power = power
        self.eps = eps
        self.prior = prior
        self.ttl = ttl
        self._members: Dict[str, Deque[Tuple[np.ndarray, float]]] = {}
        self._last_seen: Dict[str, float] = {}
        self._agent_mean: Dict[str, _Running] = {}
        self._global = _Running()
        self._lock = threading.Lock()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_reservoir.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/value_reservoir.py tests/test_value_reservoir.py
git commit -m "feat(value): reservoir scaffolding — _Running, Imputation, ctor validation"
```

### Task 2: `record_eval` — running means + guarded ring-buffer append + accept flag

**Files:**
- Modify: `trace_sampling/value_reservoir.py`
- Test: `tests/test_value_reservoir.py`

- [ ] **Step 1: Write the failing test**

```python
def _vec(*xs):
    return np.array(xs, dtype=np.float64)


def test_record_eval_updates_means_and_returns_accept_flag():
    res = ClusterValueReservoir()
    assert res.record_eval("c1", "a", _vec(1.0, 0.0), 0.8) is True
    # means updated
    assert res._global.mean == pytest.approx(0.8)
    assert res._agent_mean["a"].mean == pytest.approx(0.8)
    # member stored under c1
    assert len(res._members["c1"]) == 1


def test_record_eval_no_member_when_cluster_or_vec_missing():
    res = ClusterValueReservoir()
    assert res.record_eval(None, "a", _vec(1.0), 0.4) is True      # no cluster
    assert res.record_eval("c1", "a", None, 0.6) is True           # no vec
    assert res._members == {}                                       # nothing stored
    assert res._agent_mean["a"].count == 2                          # both means updated


def test_record_eval_rejects_non_finite():
    res = ClusterValueReservoir()
    assert res.record_eval("c1", "a", _vec(1.0), float("nan")) is False
    assert res.record_eval("c1", "a", _vec(1.0), float("inf")) is False
    assert res._global.count == 0 and res._members == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_reservoir.py -q`
Expected: FAIL with `AttributeError: 'ClusterValueReservoir' object has no attribute 'record_eval'`

- [ ] **Step 3: Write minimal implementation**

Add to `ClusterValueReservoir`:

```python
    def record_eval(self, cluster_id: Optional[str], agent_id: str,
                    vec: Optional[np.ndarray], value: float,
                    now: Optional[float] = None) -> bool:
        """Record a judged eval. Returns True iff the value was accepted (finite).

        Always updates the agent + global running means for a finite value.
        Appends an IDW donor (vec, value) to the cluster ring buffer only when
        BOTH cluster_id and vec are present; otherwise no _members entry is made.
        `now` defaults to a monotonic-clock reading (time.monotonic) so a donor
        recorded without an explicit timestamp is fresh, not immediately purgeable."""
        if not math.isfinite(value):
            return False
        ts = time.monotonic() if now is None else now
        with self._lock:
            self._global.update(value)
            self._agent_mean.setdefault(agent_id, _Running()).update(value)
            if cluster_id is not None and vec is not None:
                buf = self._members.get(cluster_id)
                if buf is None:
                    buf = self._members[cluster_id] = deque(maxlen=self.k)
                buf.append((np.asarray(vec, dtype=np.float64), float(value)))
                self._last_seen[cluster_id] = ts
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_reservoir.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/value_reservoir.py tests/test_value_reservoir.py
git commit -m "feat(value): record_eval — means always, IDW donor only with cluster+vec"
```

### Task 3: `impute` — IDW + fallback chain

**Files:**
- Modify: `trace_sampling/value_reservoir.py`
- Test: `tests/test_value_reservoir.py`

- [ ] **Step 1: Write the failing test**

```python
def test_impute_idw_weighted_average():
    res = ClusterValueReservoir(power=2.0, eps=1e-6)
    # two orthogonal-ish donors; query closer to the first
    res.record_eval("c1", "a", _vec(1.0, 0.0), 0.0)
    res.record_eval("c1", "a", _vec(0.0, 1.0), 1.0)
    imp = res.impute("c1", "a", _vec(1.0, 0.2))   # nearer donor-1 (value 0.0)
    assert imp.provenance == "idw"
    assert imp.n_donors == 2
    assert 0.0 <= imp.value < 0.5                  # pulled toward the near donor
    assert imp.nearest_dist == pytest.approx(0.0, abs=0.05)


def test_impute_higher_power_favors_nearest_donor():
    donors = [(_vec(1.0, 0.0), 0.0), (_vec(0.0, 1.0), 1.0)]
    q = _vec(1.0, 0.3)
    lo = ClusterValueReservoir(power=1.0)
    hi = ClusterValueReservoir(power=6.0)
    for r in (lo, hi):
        for v, val in donors:
            r.record_eval("c1", "a", v, val)
    assert hi.impute("c1", "a", q).value < lo.impute("c1", "a", q).value


def test_impute_eps_floor_near_duplicate_donor():
    res = ClusterValueReservoir(power=2.0, eps=1e-6)
    res.record_eval("c1", "a", _vec(1.0, 0.0), 0.3)   # value 0.3
    res.record_eval("c1", "a", _vec(0.0, 1.0), 0.9)
    imp = res.impute("c1", "a", _vec(1.0, 0.0))       # exact duplicate of donor-1
    assert imp.value == pytest.approx(0.3, abs=1e-3)  # near-dup dominates, no div0


def test_impute_fallback_chain():
    res = ClusterValueReservoir(prior=0.5)
    # cold: nothing anywhere -> prior
    assert res.impute("c1", "a", _vec(1.0, 0.0)).provenance == "prior"
    # global-only (recorded with no cluster/vec): agent "b" query with no members -> agent_mean
    res.record_eval(None, "b", None, 0.2)
    assert res.impute("cX", "b", None).provenance == "agent_mean"
    # unknown agent falls to global_mean
    assert res.impute("cX", "zzz", None).provenance == "global_mean"


def test_impute_no_vec_skips_idw():
    res = ClusterValueReservoir()
    res.record_eval("c1", "a", _vec(1.0, 0.0), 0.7)
    imp = res.impute("c1", "a", None)                 # no query vec -> cannot IDW
    assert imp.provenance == "agent_mean"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_reservoir.py -q`
Expected: FAIL with `AttributeError: ... 'impute'`

- [ ] **Step 3: Write minimal implementation**

Add to `ClusterValueReservoir`:

```python
    def impute(self, cluster_id: Optional[str], agent_id: str,
               vec: Optional[np.ndarray]) -> Imputation:
        """Snap value for a dropped trace: IDW over the cluster's judged members,
        falling back to agent-mean -> global-mean -> prior (first applicable)."""
        with self._lock:
            members = self._members.get(cluster_id) if cluster_id is not None else None
            if vec is not None and members:
                q = np.asarray(vec, dtype=np.float64)
                num = 0.0
                den = 0.0
                nearest = math.inf
                for mvec, mval in members:
                    d = max(0.0, 1.0 - _cosine(q, mvec))
                    nearest = min(nearest, d)
                    w = 1.0 / (d + self.eps) ** self.power
                    num += w * mval
                    den += w
                return Imputation(num / den, "idw", len(members), nearest)
            am = self._agent_mean.get(agent_id)
            if am is not None and am.count > 0:
                return Imputation(am.mean, "agent_mean", 0, math.nan)
            if self._global.count > 0:
                return Imputation(self._global.mean, "global_mean", 0, math.nan)
            return Imputation(self.prior, "prior", 0, math.nan)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_reservoir.py -q`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/value_reservoir.py tests/test_value_reservoir.py
git commit -m "feat(value): impute — IDW average with agent/global/prior fallback"
```

### Task 4: Ring-buffer cap, `purge_stale`, `evict`

**Files:**
- Modify: `trace_sampling/value_reservoir.py`
- Test: `tests/test_value_reservoir.py`

- [ ] **Step 1: Write the failing test**

```python
def test_ring_buffer_caps_at_k():
    res = ClusterValueReservoir(k=3)
    for i in range(5):
        res.record_eval("c1", "a", _vec(float(i), 0.0), float(i))
    assert len(res._members["c1"]) == 3          # only last k retained
    vals = [v for _, v in res._members["c1"]]
    assert vals == [2.0, 3.0, 4.0]


def test_purge_stale_drops_and_returns_ids():
    res = ClusterValueReservoir(ttl=10.0)
    res.record_eval("c1", "a", _vec(1.0), 0.5, now=0.0)
    res.record_eval("c2", "a", _vec(1.0), 0.5, now=100.0)
    dropped = res.purge_stale(now=105.0)          # c1 is stale (>10s), c2 fresh
    assert dropped == ["c1"]
    assert "c1" not in res._members and "c1" not in res._last_seen
    assert "c2" in res._members


def test_evict_removes_named_clusters():
    res = ClusterValueReservoir()
    res.record_eval("c1", "a", _vec(1.0), 0.5, now=0.0)
    res.record_eval("c2", "a", _vec(1.0), 0.5, now=0.0)
    res.evict(["c1"])
    assert "c1" not in res._members and "c1" not in res._last_seen
    assert "c2" in res._members


def test_record_eval_without_now_is_fresh_not_immediately_purgeable():
    res = ClusterValueReservoir(ttl=3600.0)
    res.record_eval("c1", "a", _vec(1.0), 0.5)          # no explicit now -> monotonic()
    dropped = res.purge_stale(now=time.monotonic())      # same clock, just recorded
    assert "c1" not in dropped
    assert "c1" in res._members
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_reservoir.py -q`
Expected: FAIL — `test_ring_buffer_caps_at_k` passes (deque maxlen already caps), `purge_stale`/`evict` fail with `AttributeError`

- [ ] **Step 3: Write minimal implementation**

Add to `ClusterValueReservoir`:

```python
    def purge_stale(self, now: float, ttl: Optional[float] = None) -> List[str]:
        """Drop cluster reservoirs untouched for longer than ttl. Returns dropped ids."""
        ttl = self.ttl if ttl is None else ttl
        with self._lock:
            stale = [cid for cid, ts in self._last_seen.items() if now - ts > ttl]
            for cid in stale:
                self._members.pop(cid, None)
                self._last_seen.pop(cid, None)
        return stale

    def evict(self, cluster_ids: Iterable[str]) -> None:
        """Forget the named cluster reservoirs (e.g. mirroring the index's purge)."""
        with self._lock:
            for cid in cluster_ids:
                self._members.pop(cid, None)
                self._last_seen.pop(cid, None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_reservoir.py -q`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/value_reservoir.py tests/test_value_reservoir.py
git commit -m "feat(value): ring-buffer cap, self-TTL purge_stale, evict"
```

### Task 5: Concurrency smoke test

**Files:**
- Modify: `tests/test_value_reservoir.py`

- [ ] **Step 1: Write the failing test**

```python
import threading


def test_concurrent_record_and_impute_stays_consistent():
    res = ClusterValueReservoir(k=128)
    errors = []

    def writer():
        try:
            for i in range(500):
                res.record_eval("c1", "a", _vec(float(i % 7), 1.0), (i % 10) / 10.0)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def reader():
        try:
            for _ in range(500):
                imp = res.impute("c1", "a", _vec(1.0, 1.0))
                assert math.isfinite(imp.value)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=writer) for _ in range(2)] + \
         [threading.Thread(target=reader) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    assert res._global.count == 1000
```

- [ ] **Step 2: Run test to verify it fails/passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_reservoir.py::test_concurrent_record_and_impute_stays_consistent -q`
Expected: PASS (the lock added in Tasks 2–3 already guards this). If it fails intermittently, the lock scope is wrong — fix before committing.

- [ ] **Step 3: Run the full reservoir suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_reservoir.py -q`
Expected: PASS (15 tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_value_reservoir.py
git commit -m "test(value): concurrency smoke for record_eval/impute under the lock"
```

---

## Chunk 2: ValuePipeline

### Task 6: `TraceValue`, `SubmitJudge`, `ValuePipeline` — drop path

**Files:**
- Create: `trace_sampling/value_pipeline.py`
- Test: `tests/test_value_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_value_pipeline.py
import numpy as np
import pytest
from trace_sampling.model import Trace
from trace_sampling.variety import VarietyKey, VarietyObservation
from trace_sampling.value_reservoir import ClusterValueReservoir
from trace_sampling.value_pipeline import ValuePipeline, TraceValue


class _FakeSampler:
    """Minimal sampler stub: decide() returns a queued verdict and sets
    last_observation to a preset VarietyObservation."""

    def __init__(self):
        self._verdicts = []
        self._obs = []
        self.last_observation = None

    def queue(self, keep: bool, key: VarietyKey):
        self._verdicts.append(keep)
        self._obs.append(VarietyObservation(key, rarity=0.0, novelty=0.0))

    def decide(self, trace):
        self.last_observation = self._obs.pop(0)
        return self._verdicts.pop(0)


class _Cache:
    """dict-backed stand-in for EmbeddingCache: supports `sig in cache` and get()."""

    def __init__(self, mapping=None):
        self._m = dict(mapping or {})
        self.get_calls = []

    def __contains__(self, sig):
        return sig in self._m

    def get(self, sig):
        self.get_calls.append(sig)
        return self._m[sig]


def _trace(tid=1, agent="a", sig=("search",)):
    return Trace(tid, agent, 0.0, sig, len(sig), 1.0, "ok")


def test_drop_path_returns_immediate_imputed_value():
    sig = ("search",)
    vec = np.array([1.0, 0.0])
    sampler = _FakeSampler()
    sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    res.record_eval("c1", "a", vec, 0.7)          # one judged donor
    cache = _Cache({sig: vec})
    emitted = []
    pipe = ValuePipeline(sampler, cache, res, submit_judge=None, on_value=emitted.append)

    tv = pipe.process(_trace(sig=sig))
    assert tv.kept is False
    assert tv.provenance == "idw"
    assert tv.value == pytest.approx(0.7, abs=1e-3)
    assert emitted == [tv]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_pipeline.py -q`
Expected: FAIL with `ModuleNotFoundError: trace_sampling.value_pipeline`

- [ ] **Step 3: Write minimal implementation**

```python
# trace_sampling/value_pipeline.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_pipeline.py -q`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/value_pipeline.py tests/test_value_pipeline.py
git commit -m "feat(value): ValuePipeline scaffolding + drop path imputation"
```

### Task 7: Keep path, causality, and judged emission

**Files:**
- Modify: `tests/test_value_pipeline.py`

- [ ] **Step 1: Write the characterization tests** (behavior implemented in Task 6; these lock it in)

```python
def test_keep_path_pending_then_judged_and_is_causal():
    sig = ("edit",)
    vec = np.array([0.0, 1.0])
    sampler = _FakeSampler()
    # first a keep, then a later drop in the same cluster
    sampler.queue(keep=True, key=VarietyKey("cluster", "c1"))
    sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    cache = _Cache({sig: vec})

    pending_judge = {}

    def submit_judge(trace, on_done):
        pending_judge[trace.trace_id] = on_done   # defer -> async

    emitted = []
    pipe = ValuePipeline(sampler, cache, res, submit_judge, on_value=emitted.append)

    tv_keep = pipe.process(_trace(tid=1, sig=sig))
    assert tv_keep.kept is True and tv_keep.value is None and tv_keep.provenance == "pending"

    # BEFORE the judge returns, a drop cannot use the pending eval -> not idw
    tv_drop_early = pipe.process(_trace(tid=2, sig=sig))
    assert tv_drop_early.provenance != "idw"

    # judge returns -> recorded, "judged" emitted
    pending_judge[1](0.9)
    assert emitted[-1] == TraceValue(1, True, 0.9, "judged")

    # a NEW drop now sees the donor -> idw
    sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    tv_drop_late = pipe.process(_trace(tid=3, sig=sig))
    assert tv_drop_late.provenance == "idw"
    assert tv_drop_late.value == pytest.approx(0.9, abs=1e-3)


def test_synchronous_judge_emits_pending_before_judged():
    sig = ("fetch",)
    vec = np.array([1.0, 0.0])
    sampler = _FakeSampler()
    sampler.queue(keep=True, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    cache = _Cache({sig: vec})
    emitted = []

    def sync_judge(trace, on_done):
        on_done(0.6)                              # fire inline (synchronous judge)

    pipe = ValuePipeline(sampler, cache, res, sync_judge, on_value=emitted.append)
    pipe.process(_trace(tid=1, sig=sig))
    # natural order preserved even for a synchronous callback
    assert [tv.provenance for tv in emitted] == ["pending", "judged"]


def test_judge_returning_non_finite_does_not_emit_judged():
    sig = ("run",)
    vec = np.array([1.0, 0.0])
    sampler = _FakeSampler()
    sampler.queue(keep=True, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    cache = _Cache({sig: vec})
    captured = {}
    pipe = ValuePipeline(sampler, cache, res,
                         submit_judge=lambda t, cb: captured.setdefault("cb", cb))
    emitted = []
    pipe.on_value = emitted.append
    pipe.process(_trace(tid=1, sig=sig))
    captured["cb"](float("nan"))                  # bad eval
    assert all(tv.provenance != "judged" for tv in emitted)
    assert res._global.count == 0                 # nothing recorded
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_pipeline.py -q`
Expected: PASS (4 tests) — the Task 6 implementation already covers these paths. If any fails, fix the implementation (not the test) before committing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_value_pipeline.py
git commit -m "test(value): keep-path causality + non-finite judge eval handling"
```

### Task 8: No-hot-path-embed guarantee and judge-submit failure

**Files:**
- Modify: `tests/test_value_pipeline.py`

- [ ] **Step 1: Write the characterization tests** (behavior implemented in Task 6; these lock it in)

```python
def test_fallback_signature_yields_no_embed_and_mean_fallback():
    sig = ("plan",)
    sampler = _FakeSampler()
    sampler.queue(keep=False, key=VarietyKey("fallback-signature", sig))
    res = ClusterValueReservoir()
    res.record_eval(None, "a", None, 0.4)         # seed agent mean
    cache = _Cache({})                            # sig NOT cached
    pipe = ValuePipeline(sampler, cache, res, submit_judge=None)

    tv = pipe.process(_trace(sig=sig))
    assert cache.get_calls == []                  # NEVER embedded on the hot path
    assert tv.provenance == "agent_mean"
    assert tv.value == pytest.approx(0.4)


def test_uncached_cluster_trace_does_not_embed():
    sig = ("test",)
    sampler = _FakeSampler()
    sampler.queue(keep=False, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    cache = _Cache({})                            # cluster kind but sig not cached
    pipe = ValuePipeline(sampler, cache, res, submit_judge=None)
    pipe.process(_trace(sig=sig))
    assert cache.get_calls == []                  # guard prevents embed


def test_submit_judge_failure_still_returns_pending_no_record():
    sig = ("write",)
    vec = np.array([1.0, 0.0])
    sampler = _FakeSampler()
    sampler.queue(keep=True, key=VarietyKey("cluster", "c1"))
    res = ClusterValueReservoir()
    cache = _Cache({sig: vec})

    def boom(trace, on_done):
        raise RuntimeError("judge queue full")

    pipe = ValuePipeline(sampler, cache, res, submit_judge=boom)
    tv = pipe.process(_trace(sig=sig))
    assert tv.kept is True and tv.provenance == "pending"
    assert res._global.count == 0 and res._members == {}
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_value_pipeline.py -q`
Expected: PASS (7 tests) — implementation already guards the cache read and wraps `submit_judge`. Fix implementation if any fail.

- [ ] **Step 3: Commit**

```bash
git add tests/test_value_pipeline.py
git commit -m "test(value): no-hot-path-embed + judge-submit-failure guarantees"
```

### Task 9: Full-suite regression

**Files:** none (verification only)

- [ ] **Step 1: Run the whole non-Azure suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q -m "not azure"`
Expected: PASS — previous 78 + new reservoir/pipeline tests, 0 failures. Confirm no existing test regressed.

- [ ] **Step 2: If green, no commit needed.** If any pre-existing test broke, investigate with superpowers:systematic-debugging before proceeding.

---

## Notes for the implementer

- Run pytest with the repo venv: `.\.venv\Scripts\python.exe -m pytest ...` (Windows).
- `_cosine` lives in `trace_sampling/vector_store.py` and returns 0.0 for a zero-norm vector — reuse it; do not reimplement cosine.
- The reservoir is deliberately Azure-free and pure-Python for fast, deterministic unit tests.
- Do NOT modify `AzureClusterIndex`, `AdaptiveSampler`, or `EmbeddingCache` — the pipeline only reads their existing public surface (`decide`, `last_observation`, `sig in cache`, `cache.get`).
- Section 5 of the spec (offline notebook driver / estimator-quality demo) is explicitly optional and out of scope for this plan.
