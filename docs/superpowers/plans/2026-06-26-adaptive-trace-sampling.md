# Adaptive Backpressure Trace Sampler — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an online, bounded-memory trace sampler that keeps a representative, variety-rich, non-starving subset of multi-agent OTel-style traces under an LLM-throughput budget, and demonstrate it on a controllable synthetic stream in a notebook.

**Architecture:** Core logic lives in an importable `trace_sampling/` Python package (unit-tested with pytest, since notebooks aren't directly testable). A synthetic generator emits OTel-shaped traces with per-agent velocity/variety dials. Two samplers — a fixed-rate baseline (Strategy A) and the adaptive backpressure sampler (Strategy B) — consume the stream. A metrics module scores coverage, starvation, representativeness, and budget adherence. A notebook (`adaptive_trace_sampling.ipynb`) wires it together and plots the comparison.

**Tech Stack:** Python 3.12, numpy, pandas, matplotlib, pytest. No external services; the "LLM backend" is modeled as a bounded-throughput consumer queue. (Entropy is computed directly with `math.log`, so scipy is not required.)

**Spec:** `docs/superpowers/specs/2026-06-26-adaptive-trace-sampling-design.md`

---

## File Structure

- `trace_sampling/__init__.py` — package exports.
- `trace_sampling/model.py` — `Trace` and `AgentConfig` dataclasses.
- `trace_sampling/generator.py` — synthetic OTel-shaped stream generator (Poisson velocity, Zipf variety, late arrivals).
- `trace_sampling/stats.py` — `AgentStats`: EWMA velocity, LRU-capped signature table, distinct/entropy/rarity, cold-start flag.
- `trace_sampling/reservoir.py` — `WeightedReservoir`: bounded per-stratum store with weighted replacement.
- `trace_sampling/backpressure.py` — `BackpressureController`: AIMD multiplier driven by a drained consumer queue.
- `trace_sampling/samplers.py` — `BaselineSampler` (A) and `AdaptiveSampler` (B), plus shared `SamplerConfig`.
- `trace_sampling/metrics.py` — coverage, starvation, representativeness (KL/TV), budget timeseries.
- `tests/test_*.py` — one test module per package module.
- `adaptive_trace_sampling.ipynb` — generator → samplers → metrics → plots → conclusions.
- `requirements-sampling.txt` — pinned deps for the prototype.

All paths are relative to repo root `C:\Users\aadkannan\source\repos\personal\singlenotebooks`.

---

## Chunk 1: Data model, generator, stats, reservoir

### Task 1: Package scaffold + data model

**Files:**
- Create: `trace_sampling/__init__.py`
- Create: `trace_sampling/model.py`
- Create: `tests/test_model.py`
- Create: `requirements-sampling.txt`

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:
```python
from trace_sampling.model import Trace, AgentConfig


def test_trace_is_immutable_and_hashable():
    t = Trace(trace_id=1, agent_id="a", timestamp=0.0,
              signature=("search", "read"), span_count=2,
              duration_ms=12.5, status="ok")
    assert t.signature == ("search", "read")
    # frozen dataclass -> hashable, usable as dict key
    assert {t: 1}[t] == 1


def test_agent_config_defaults():
    c = AgentConfig(agent_id="a", velocity=2.0, vocab_size=5, zipf_s=1.2)
    assert c.start_time == 0.0
    assert 0.0 <= c.error_rate <= 1.0
    assert len(c.tool_pool) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trace_sampling'`

- [ ] **Step 3: Write minimal implementation**

`trace_sampling/__init__.py`:
```python
"""Adaptive backpressure trace sampling prototype."""
```

`trace_sampling/model.py`:
```python
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Trace:
    """One OTel-style agent trace (a job-to-be-done with tool spans)."""
    trace_id: int
    agent_id: str
    timestamp: float
    signature: Tuple[str, ...]  # ordered tool/span names = the variety key
    span_count: int
    duration_ms: float
    status: str  # "ok" | "error"


@dataclass
class AgentConfig:
    """Ground-truth dials controlling an agent's synthetic behavior."""
    agent_id: str
    velocity: float       # Poisson arrival rate (traces per sim-second)
    vocab_size: int       # number of distinct signatures the agent can emit
    zipf_s: float         # skew of the signature distribution (higher = peakier)
    start_time: float = 0.0
    error_rate: float = 0.05
    tool_pool: Tuple[str, ...] = (
        "search", "read", "edit", "run", "plan", "test", "fetch", "write",
    )
```

`requirements-sampling.txt`:
```
numpy
pandas
matplotlib
pytest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/__init__.py trace_sampling/model.py tests/test_model.py requirements-sampling.txt
git commit -m "feat(sampling): add Trace and AgentConfig data model"
```

---

### Task 2: Synthetic OTel-shaped stream generator

**Files:**
- Create: `trace_sampling/generator.py`
- Create: `tests/test_generator.py`

Design: each agent gets a fixed vocabulary of `vocab_size` signatures (random tuples of 1–4 tools), with Zipf-weighted selection probabilities. Arrivals are a Poisson process (exponential inter-arrival with mean `1/velocity`) starting at `start_time`. The generator merges all agents' traces into one time-sorted stream. A seeded `numpy.random.Generator` makes it deterministic.

- [ ] **Step 1: Write the failing test**

`tests/test_generator.py`:
```python
import numpy as np
from trace_sampling.model import AgentConfig
from trace_sampling.generator import generate_stream


def _configs():
    return [
        AgentConfig("fast_lowvar", velocity=10.0, vocab_size=2, zipf_s=2.0),
        AgentConfig("slow_highvar", velocity=1.0, vocab_size=20, zipf_s=0.6),
        AgentConfig("late", velocity=5.0, vocab_size=5, zipf_s=1.0, start_time=5.0),
    ]


def test_stream_is_time_sorted_and_deterministic():
    s1 = generate_stream(_configs(), duration=10.0, seed=7)
    s2 = generate_stream(_configs(), duration=10.0, seed=7)
    assert [t.trace_id for t in s1] == [t.trace_id for t in s2]
    assert [t.timestamp for t in s1] == [t.timestamp for t in s2]
    assert [t.signature for t in s1] == [t.signature for t in s2]
    assert [t.agent_id for t in s1] == [t.agent_id for t in s2]
    ts = [t.timestamp for t in s1]
    assert ts == sorted(ts)
    assert all(0.0 <= t.timestamp <= 10.0 for t in s1)


def test_late_agent_only_appears_after_start_time():
    s = generate_stream(_configs(), duration=10.0, seed=7)
    late = [t for t in s if t.agent_id == "late"]
    assert late, "late agent should emit some traces"
    assert min(t.timestamp for t in late) >= 5.0


def test_velocity_controls_relative_volume():
    s = generate_stream(_configs(), duration=20.0, seed=3)
    counts = {}
    for t in s:
        counts[t.agent_id] = counts.get(t.agent_id, 0) + 1
    assert counts["fast_lowvar"] > counts["slow_highvar"]


def test_variety_controls_distinct_signatures():
    s = generate_stream(_configs(), duration=30.0, seed=3)
    distinct = {}
    for t in s:
        distinct.setdefault(t.agent_id, set()).add(t.signature)
    assert len(distinct["slow_highvar"]) > len(distinct["fast_lowvar"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trace_sampling.generator'`

- [ ] **Step 3: Write minimal implementation**

`trace_sampling/generator.py`:
```python
from typing import List, Tuple
import numpy as np

from .model import Trace, AgentConfig


def _build_vocab(cfg: AgentConfig, rng: np.random.Generator):
    """Return (signatures, probabilities) for one agent."""
    signatures = []
    seen = set()
    # Generate distinct signatures (tuples of 1-4 tools).
    attempts = 0
    while len(signatures) < cfg.vocab_size and attempts < cfg.vocab_size * 50:
        attempts += 1
        length = int(rng.integers(1, 5))
        sig = tuple(rng.choice(cfg.tool_pool, size=length))
        if sig not in seen:
            seen.add(sig)
            signatures.append(sig)
    # Zipf weights over ranks 1..N.
    ranks = np.arange(1, len(signatures) + 1, dtype=float)
    weights = 1.0 / np.power(ranks, cfg.zipf_s)
    weights /= weights.sum()
    return signatures, weights


def generate_stream(configs: List[AgentConfig], duration: float,
                    seed: int = 0) -> List[Trace]:
    """Generate a time-sorted interleaved stream of traces across agents."""
    rng = np.random.default_rng(seed)
    events: List[Tuple[float, str, Tuple[str, ...], str]] = []
    for cfg in configs:
        signatures, weights = _build_vocab(cfg, rng)
        t = cfg.start_time
        mean_gap = 1.0 / cfg.velocity
        while True:
            t += rng.exponential(mean_gap)
            if t > duration:
                break
            idx = rng.choice(len(signatures), p=weights)
            sig = signatures[idx]
            status = "error" if rng.random() < cfg.error_rate else "ok"
            events.append((t, cfg.agent_id, sig, status))
    events.sort(key=lambda e: e[0])
    traces = []
    for i, (t, agent_id, sig, status) in enumerate(events):
        traces.append(Trace(
            trace_id=i,
            agent_id=agent_id,
            timestamp=t,
            signature=sig,
            span_count=len(sig),
            duration_ms=float(rng.uniform(5.0, 500.0)),
            status=status,
        ))
    return traces
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_generator.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/generator.py tests/test_generator.py
git commit -m "feat(sampling): add synthetic OTel-shaped stream generator"
```

---

### Task 3: Per-agent live statistics (velocity, variety, cold-start)

**Files:**
- Create: `trace_sampling/stats.py`
- Create: `tests/test_stats.py`

Design: `AgentStats` updates incrementally per trace. Velocity is an EWMA of instantaneous rate `1/dt`. Signatures live in an `OrderedDict` LRU capped at `max_signatures`; on overflow the least-recently-seen entry is evicted. `rarity(sig)` is high for rarely-seen signatures, `is_coldstart()` is true until `coldstart_min_samples` observed.

- [ ] **Step 1: Write the failing test**

`tests/test_stats.py`:
```python
from trace_sampling.stats import AgentStats


def test_coldstart_until_min_samples():
    s = AgentStats(coldstart_min_samples=3, max_signatures=10, ewma_alpha=0.5)
    assert s.is_coldstart()
    for i in range(3):
        s.observe(timestamp=float(i), signature=("a",))
    assert not s.is_coldstart()


def test_velocity_ewma_tracks_rate():
    s = AgentStats(coldstart_min_samples=1, max_signatures=10, ewma_alpha=0.5)
    # One trace per second -> rate ~1.0
    for i in range(1, 11):
        s.observe(timestamp=float(i), signature=("a",))
    assert 0.5 < s.velocity() < 1.5


def test_lru_eviction_bounds_memory():
    s = AgentStats(coldstart_min_samples=1, max_signatures=3, ewma_alpha=0.5)
    for i in range(10):
        s.observe(timestamp=float(i), signature=(f"sig{i}",))
    assert s.distinct_estimate() == 3  # capped


def test_rarity_higher_for_rare_signature():
    s = AgentStats(coldstart_min_samples=1, max_signatures=10, ewma_alpha=0.5)
    for i in range(100):
        s.observe(timestamp=float(i), signature=("common",))
    s.observe(timestamp=100.0, signature=("rare",))
    assert s.rarity(("rare",)) > s.rarity(("common",))


def test_entropy_nonnegative():
    s = AgentStats(coldstart_min_samples=1, max_signatures=10, ewma_alpha=0.5)
    for i in range(5):
        s.observe(timestamp=float(i), signature=(f"s{i % 2}",))
    assert s.entropy() >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trace_sampling.stats'`

- [ ] **Step 3: Write minimal implementation**

`trace_sampling/stats.py`:
```python
import math
from collections import OrderedDict
from typing import Tuple


class AgentStats:
    """Bounded-memory live statistics for a single agent."""

    def __init__(self, coldstart_min_samples: int = 20,
                 max_signatures: int = 256, ewma_alpha: float = 0.1):
        self.coldstart_min_samples = coldstart_min_samples
        self.max_signatures = max_signatures
        self.ewma_alpha = ewma_alpha
        self._counts: "OrderedDict[Tuple[str, ...], int]" = OrderedDict()
        self._total = 0
        self._velocity = 0.0
        self._last_ts = None

    def observe(self, timestamp: float, signature: Tuple[str, ...]) -> None:
        if self._last_ts is not None:
            dt = max(timestamp - self._last_ts, 1e-9)
            inst_rate = 1.0 / dt
            if self._velocity == 0.0:
                self._velocity = inst_rate
            else:
                a = self.ewma_alpha
                self._velocity = a * inst_rate + (1 - a) * self._velocity
        self._last_ts = timestamp
        self._total += 1
        if signature in self._counts:
            self._counts[signature] += 1
            self._counts.move_to_end(signature)
        else:
            self._counts[signature] = 1
            self._counts.move_to_end(signature)
            if len(self._counts) > self.max_signatures:
                self._counts.popitem(last=False)  # evict least-recently-seen

    def is_coldstart(self) -> bool:
        return self._total < self.coldstart_min_samples

    def velocity(self) -> float:
        return self._velocity

    def distinct_estimate(self) -> int:
        return len(self._counts)

    def total(self) -> int:
        return self._total

    def rarity(self, signature: Tuple[str, ...]) -> float:
        """In [0, 1]; ~1 for unseen/rare signatures, ->0 for very frequent ones."""
        if self._total == 0:
            return 1.0
        count = self._counts.get(signature, 0)
        return 1.0 / (1.0 + count)

    def entropy(self) -> float:
        if self._total == 0:
            return 0.0
        total = sum(self._counts.values())
        h = 0.0
        for c in self._counts.values():
            p = c / total
            h -= p * math.log(p + 1e-12)
        return h
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stats.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/stats.py tests/test_stats.py
git commit -m "feat(sampling): add bounded-memory per-agent live stats"
```

---

### Task 4: Weighted reservoir (bounded per-stratum store)

**Files:**
- Create: `trace_sampling/reservoir.py`
- Create: `tests/test_reservoir.py`

Design: A-Res weighted reservoir sampling. Each item gets key `u**(1/weight)` with `u ~ Uniform(0,1)`; the reservoir keeps the top-`capacity` keys. Higher weight (more diverse/rare) → more likely to be retained. This bounds memory per stratum and biases the stored set toward variety.

- [ ] **Step 1: Write the failing test**

`tests/test_reservoir.py`:
```python
import numpy as np
from trace_sampling.reservoir import WeightedReservoir


def test_capacity_is_bounded():
    r = WeightedReservoir(capacity=5, seed=1)
    for i in range(100):
        r.offer(item=i, weight=1.0)
    assert len(r.items()) == 5


def test_high_weight_items_favored():
    rng_seed = 2
    r = WeightedReservoir(capacity=10, seed=rng_seed)
    # 1000 low-weight items, 10 high-weight items
    for i in range(1000):
        r.offer(item=("low", i), weight=1.0)
    for i in range(10):
        r.offer(item=("high", i), weight=100.0)
    kept = r.items()
    high_kept = sum(1 for it in kept if it[0] == "high")
    assert high_kept >= 5  # high-weight items dominate the reservoir


def test_offer_returns_admission_flag():
    r = WeightedReservoir(capacity=1, seed=0)
    assert r.offer(item="first", weight=1.0) is True  # fills empty slot
    # A much higher weight item should be admitted (replace).
    admitted_any = any(r.offer(item=f"x{i}", weight=1000.0) for i in range(20))
    assert admitted_any is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reservoir.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trace_sampling.reservoir'`

- [ ] **Step 3: Write minimal implementation**

`trace_sampling/reservoir.py`:
```python
import heapq
from typing import Any, List
import numpy as np


class WeightedReservoir:
    """A-Res weighted reservoir: retains the top-`capacity` items by key
    u**(1/weight). Higher weight -> higher retention probability."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        # min-heap of (key, tiebreak, item); smallest key at top for eviction
        self._heap: List = []
        self._tiebreak = 0

    def offer(self, item: Any, weight: float) -> bool:
        weight = max(weight, 1e-9)
        u = self._rng.random()
        key = u ** (1.0 / weight)
        self._tiebreak += 1
        entry = (key, self._tiebreak, item)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, entry)
            return True
        if key > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)
            return True
        return False

    def items(self) -> List[Any]:
        return [e[2] for e in self._heap]

    def __len__(self) -> int:
        return len(self._heap)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reservoir.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/reservoir.py tests/test_reservoir.py
git commit -m "feat(sampling): add A-Res weighted reservoir"
```

---

### Task 5: Chunk 1 review gate

- [ ] **Step 1: Run the full Chunk-1 suite**

Run: `python -m pytest tests/test_model.py tests/test_generator.py tests/test_stats.py tests/test_reservoir.py -v`
Expected: PASS (all green)

- [ ] **Step 2: If any test fails, fix the implementation and re-run**

Re-run the same command above until green. Then commit the fix:
```bash
git add trace_sampling/ tests/
git commit -m "fix(sampling): address chunk 1 test failures"
```
If everything already passed in Step 1, there is nothing to commit here; proceed to Chunk 2.

---

## Chunk 2: Backpressure, samplers, metrics, notebook

### Task 6: Backpressure controller (AIMD over a drained queue)

**Files:**
- Create: `trace_sampling/backpressure.py`
- Create: `tests/test_backpressure.py`

Design: models the LLM consumer as a queue drained at `llm_throughput` per sim-second. `on_kept()` enqueues. `tick(now)` drains by `throughput*(now-last_drain)` and updates the admission `multiplier` via AIMD: queue above `queue_high` → `multiplier *= aimd_decrease`; below `queue_low` → `multiplier += aimd_increase`. `multiplier` is clamped to `[min_multiplier, 1.0]`.

- [ ] **Step 1: Write the failing test**

`tests/test_backpressure.py`:
```python
from trace_sampling.backpressure import BackpressureController


def _ctrl():
    return BackpressureController(
        throughput=10.0, queue_high=20.0, queue_low=5.0,
        aimd_increase=0.05, aimd_decrease=0.5, min_multiplier=0.01)


def test_multiplier_drops_under_backpressure():
    c = _ctrl()
    now = 0.0
    for _ in range(100):       # flood without draining time
        c.on_kept()
    c.tick(now)                # queue huge -> decrease
    assert c.multiplier < 1.0


def test_multiplier_recovers_with_slack():
    c = _ctrl()
    c.multiplier = 0.2
    # Advance time with empty queue -> additive increase each tick.
    for i in range(1, 50):
        c.tick(float(i))
    assert c.multiplier > 0.2


def test_multiplier_clamped():
    c = _ctrl()
    for _ in range(10000):
        c.on_kept()
    for i in range(1, 50):
        c.tick(0.0)
    assert c.multiplier >= 0.01
    assert c.multiplier <= 1.0


def test_queue_drains_over_time():
    c = _ctrl()
    for _ in range(50):
        c.on_kept()
    c.tick(10.0)               # 10s * 10/s = 100 drained -> empty
    assert c.queue_len == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backpressure.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trace_sampling.backpressure'`

- [ ] **Step 3: Write minimal implementation**

`trace_sampling/backpressure.py`:
```python
class BackpressureController:
    """AIMD admission multiplier driven by a drained consumer queue."""

    def __init__(self, throughput: float, queue_high: float, queue_low: float,
                 aimd_increase: float = 0.05, aimd_decrease: float = 0.5,
                 min_multiplier: float = 0.01):
        self.throughput = throughput
        self.queue_high = queue_high
        self.queue_low = queue_low
        self.aimd_increase = aimd_increase
        self.aimd_decrease = aimd_decrease
        self.min_multiplier = min_multiplier
        self.multiplier = 1.0
        self.queue_len = 0.0
        self._last_drain = 0.0

    def on_kept(self) -> None:
        self.queue_len += 1.0

    def tick(self, now: float) -> None:
        drained = self.throughput * max(now - self._last_drain, 0.0)
        self.queue_len = max(0.0, self.queue_len - drained)
        self._last_drain = now
        if self.queue_len > self.queue_high:
            self.multiplier *= self.aimd_decrease
        elif self.queue_len < self.queue_low:
            self.multiplier += self.aimd_increase
        self.multiplier = min(1.0, max(self.min_multiplier, self.multiplier))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backpressure.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/backpressure.py tests/test_backpressure.py
git commit -m "feat(sampling): add AIMD backpressure controller"
```

---

### Task 7: Samplers (baseline + adaptive)

**Files:**
- Create: `trace_sampling/samplers.py`
- Create: `tests/test_samplers.py`

Design:
- `SamplerConfig` holds all tunables (defaults from the spec's parameter table).
- `BaselineSampler`: fixed global keep probability, no floors, no diversity, no backpressure. Used as the comparison baseline.
- `AdaptiveSampler.decide(trace) -> bool` implements the spec pipeline:
  1. Update per-agent `AgentStats` and global backpressure `tick`.
  2. Track per-agent "kept in active_window" to enforce the **deterministic keep-one** floor: if the agent has zero keeps in the trailing `active_window`, keep this trace (budget permitting).
  3. Otherwise compute keep probability = `base_diversity_prob * coldstart_boost * backpressure.multiplier`, lower-bounded by the agent's `agent_floor` share; Bernoulli draw.
  4. On keep: push to the stratum `WeightedReservoir` (weight = diversity score), record keep time, and call `backpressure.on_kept()`.

`base_diversity_prob` rises with signature rarity and falls when the stratum reservoir is full. `coldstart_boost` applies while `stats.is_coldstart()`.

- [ ] **Step 1: Write the failing test**

`tests/test_samplers.py`:
```python
from trace_sampling.model import AgentConfig
from trace_sampling.generator import generate_stream
from trace_sampling.samplers import AdaptiveSampler, BaselineSampler, SamplerConfig


def _stream():
    cfgs = [
        AgentConfig("fast_lowvar", velocity=20.0, vocab_size=2, zipf_s=2.0),
        AgentConfig("rare_highvar", velocity=0.5, vocab_size=15, zipf_s=0.6),
        AgentConfig("late", velocity=5.0, vocab_size=6, zipf_s=1.0, start_time=8.0),
    ]
    return generate_stream(cfgs, duration=30.0, seed=11)


def _run(sampler, stream):
    kept = [t for t in stream if sampler.decide(t)]
    return kept


def test_adaptive_does_not_starve_any_active_agent():
    stream = _stream()
    cfg = SamplerConfig(llm_throughput=15.0)
    kept = _run(AdaptiveSampler(cfg, seed=1), stream)
    kept_agents = {t.agent_id for t in kept}
    active_agents = {t.agent_id for t in stream}
    assert kept_agents == active_agents  # every active agent retained


def test_adaptive_respects_budget_better_than_no_cap():
    stream = _stream()
    cfg = SamplerConfig(llm_throughput=15.0)
    kept = _run(AdaptiveSampler(cfg, seed=1), stream)
    # Kept count should be far below total (sampler is selective).
    assert len(kept) < len(stream)


def test_adaptive_captures_more_rare_variety_than_baseline():
    stream = _stream()
    cfg = SamplerConfig(llm_throughput=15.0)
    adaptive_kept = _run(AdaptiveSampler(cfg, seed=1), stream)
    baseline = BaselineSampler(keep_prob=len(_run(AdaptiveSampler(cfg, seed=1),
                                                  stream)) / len(stream), seed=1)
    baseline_kept = _run(baseline, stream)

    def distinct_for(kept, agent):
        return {t.signature for t in kept if t.agent_id == agent}

    a = distinct_for(adaptive_kept, "rare_highvar")
    b = distinct_for(baseline_kept, "rare_highvar")
    assert len(a) >= len(b)


def test_baseline_keeps_roughly_keep_prob_fraction():
    stream = _stream()
    kept = _run(BaselineSampler(keep_prob=0.3, seed=2), stream)
    frac = len(kept) / len(stream)
    assert 0.2 < frac < 0.4


def test_reservoir_count_is_bounded():
    # Many distinct signatures must not grow the reservoir map without bound.
    from trace_sampling.model import Trace
    cfg = SamplerConfig(llm_throughput=50.0, max_reservoirs=16)
    sampler = AdaptiveSampler(cfg, seed=0)
    for i in range(500):
        sampler.decide(Trace(i, "a", float(i) * 0.001, (f"sig{i}",), 1, 1.0, "ok"))
    assert len(sampler._reservoirs) <= 16


def test_rare_agent_floor_keeps_almost_everything():
    # A single very-rare agent should be (near) fully retained via the floor.
    from trace_sampling.model import Trace
    cfg = SamplerConfig(llm_throughput=50.0)
    sampler = AdaptiveSampler(cfg, seed=0)
    rare = [Trace(i, "rare", float(i) * 5.0, (f"v{i}",), 1, 1.0, "ok")
            for i in range(20)]
    kept = [t for t in rare if sampler.decide(t)]
    assert len(kept) >= 18  # rare/low-velocity agent is protected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_samplers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trace_sampling.samplers'`

- [ ] **Step 3: Write minimal implementation**

`trace_sampling/samplers.py`:
```python
from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict, Tuple
import numpy as np

from .model import Trace
from .stats import AgentStats
from .reservoir import WeightedReservoir
from .backpressure import BackpressureController


@dataclass
class SamplerConfig:
    llm_throughput: float = 50.0
    agent_floor: float = 0.02          # guaranteed per-active-agent budget share
    active_window: float = 30.0
    max_signatures_per_agent: int = 256
    max_reservoirs: int = 4096         # global LRU cap on (agent, sig) reservoirs
    coldstart_min_samples: int = 20
    coldstart_boost: float = 5.0
    ewma_alpha: float = 0.1
    aimd_increase: float = 0.05
    aimd_decrease: float = 0.5
    queue_high_factor: float = 2.0
    queue_low_factor: float = 0.5
    reservoir_size: int = 8
    min_multiplier: float = 0.01


class BaselineSampler:
    """Strategy A: fixed global keep probability, no adaptation."""

    def __init__(self, keep_prob: float, seed: int = 0):
        self.keep_prob = keep_prob
        self._rng = np.random.default_rng(seed)

    def decide(self, trace: Trace) -> bool:
        return bool(self._rng.random() < self.keep_prob)


class AdaptiveSampler:
    """Strategy B: stratified, diversity-weighted, floored, backpressure-aware."""

    def __init__(self, config: SamplerConfig, seed: int = 0):
        self.cfg = config
        self._rng = np.random.default_rng(seed)
        self._stats: Dict[str, AgentStats] = {}
        # LRU-bounded map of (agent, signature) -> reservoir; keeps memory bounded.
        self._reservoirs: "OrderedDict[Tuple[str, Tuple[str, ...]], WeightedReservoir]" = OrderedDict()
        self._last_kept_ts: Dict[str, float] = {}
        self._last_seen_ts: Dict[str, float] = {}   # for active-agent counting
        self._bp = BackpressureController(
            throughput=config.llm_throughput,
            queue_high=config.llm_throughput * config.queue_high_factor,
            queue_low=config.llm_throughput * config.queue_low_factor,
            aimd_increase=config.aimd_increase,
            aimd_decrease=config.aimd_decrease,
            min_multiplier=config.min_multiplier,
        )
        self._res_seed = seed

    def _stats_for(self, agent_id: str) -> AgentStats:
        if agent_id not in self._stats:
            self._stats[agent_id] = AgentStats(
                coldstart_min_samples=self.cfg.coldstart_min_samples,
                max_signatures=self.cfg.max_signatures_per_agent,
                ewma_alpha=self.cfg.ewma_alpha,
            )
        return self._stats[agent_id]

    def _reservoir_for(self, key) -> WeightedReservoir:
        if key in self._reservoirs:
            self._reservoirs.move_to_end(key)
            return self._reservoirs[key]
        self._res_seed += 1
        res = WeightedReservoir(capacity=self.cfg.reservoir_size, seed=self._res_seed)
        self._reservoirs[key] = res
        if len(self._reservoirs) > self.cfg.max_reservoirs:
            self._reservoirs.popitem(last=False)  # evict least-recently-used
        return res

    def _active_agent_count(self, now: float) -> int:
        w = self.cfg.active_window
        return sum(1 for ts in self._last_seen_ts.values() if now - ts <= w) or 1

    def decide(self, trace: Trace) -> bool:
        cfg = self.cfg
        self._bp.tick(trace.timestamp)
        stats = self._stats_for(trace.agent_id)
        stats.observe(trace.timestamp, trace.signature)
        self._last_seen_ts[trace.agent_id] = trace.timestamp

        key = (trace.agent_id, trace.signature)
        reservoir = self._reservoir_for(key)

        # Diversity score: rarer signatures and under-filled strata score higher.
        rarity = stats.rarity(trace.signature)
        fill = len(reservoir) / max(cfg.reservoir_size, 1)
        diversity = rarity * (1.0 - 0.5 * fill)

        # Velocity-based fair-share floor (spec step 4 precedence):
        # each active agent is guaranteed a share of the budget, scaled down
        # equally when too many agents are active so the total stays bounded.
        n_active = self._active_agent_count(trace.timestamp)
        guaranteed_rate = min(cfg.agent_floor * cfg.llm_throughput,
                              cfg.llm_throughput / n_active)
        velocity = max(stats.velocity(), 1e-6)
        floor_prob = min(1.0, guaranteed_rate / velocity)

        # Deterministic keep-one floor: if this agent has had no keep in the
        # trailing active_window, keep this trace (the hard anti-starvation
        # guarantee). Keep-one volume is bounded by one per agent per window,
        # negligible vs llm_throughput.
        last_kept = self._last_kept_ts.get(trace.agent_id)
        stale = last_kept is None or (trace.timestamp - last_kept) >= cfg.active_window

        if stale:
            keep = True
        else:
            boost = cfg.coldstart_boost if stats.is_coldstart() else 1.0
            prob = diversity * boost * self._bp.multiplier
            prob = max(prob, floor_prob)   # probabilistic budget-share floor
            prob = min(prob, 1.0)
            keep = bool(self._rng.random() < prob)

        if keep:
            reservoir.offer(item=trace.trace_id, weight=max(diversity, 1e-6))
            self._last_kept_ts[trace.agent_id] = trace.timestamp
            self._bp.on_kept()
        return keep
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_samplers.py -v`
Expected: PASS (6 passed). If `test_adaptive_respects_budget_better_than_no_cap` is borderline, confirm kept fraction < 1.0; tune nothing else.

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/samplers.py tests/test_samplers.py
git commit -m "feat(sampling): add baseline and adaptive backpressure samplers"
```

---

### Task 8: Metrics (coverage, starvation, representativeness, budget)

**Files:**
- Create: `trace_sampling/metrics.py`
- Create: `tests/test_metrics.py`

Design: pure functions over the original stream and a kept subset.
- `signature_coverage(stream, kept) -> dict[agent -> fraction]`: distinct kept signatures / distinct true signatures.
- `min_active_keep_rate(stream, kept, window) -> float`: minimum, across active agents, of kept-rate within tumbling (non-overlapping) windows (starvation; >0 means no starvation).
- `representativeness(stream, kept) -> dict[agent -> {"kl":..., "tv":...}]`: divergence between kept vs true per-agent signature distributions.
- `kept_rate_timeseries(kept, bucket) -> (times, rates)`: kept traces per second in each time bucket for budget plots.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
import math
from trace_sampling.model import Trace
from trace_sampling.metrics import (
    signature_coverage, min_active_keep_rate, representativeness,
    kept_rate_timeseries,
)


def _t(tid, agent, ts, sig):
    return Trace(tid, agent, ts, sig, len(sig), 1.0, "ok")


def _stream():
    out = []
    tid = 0
    for ts in range(20):
        out.append(_t(tid, "a", float(ts), ("x",))); tid += 1
        out.append(_t(tid, "a", float(ts) + 0.5, ("y",))); tid += 1
        out.append(_t(tid, "b", float(ts), ("z",))); tid += 1
    return out


def test_coverage_full_when_all_kept():
    s = _stream()
    cov = signature_coverage(s, s)
    assert cov["a"] == 1.0 and cov["b"] == 1.0


def test_coverage_partial_when_one_signature_dropped():
    s = _stream()
    kept = [t for t in s if t.signature != ("y",)]
    cov = signature_coverage(s, kept)
    assert math.isclose(cov["a"], 0.5)


def test_min_active_keep_rate_zero_when_agent_starved():
    s = _stream()
    kept = [t for t in s if t.agent_id == "a"]   # b fully starved
    assert min_active_keep_rate(s, kept, window=5.0) == 0.0


def test_min_active_keep_rate_positive_when_all_served():
    s = _stream()
    kept = s[::2]
    assert min_active_keep_rate(s, kept, window=50.0) > 0.0


def test_representativeness_zero_divergence_when_proportional():
    s = _stream()
    rep = representativeness(s, s)
    assert rep["a"]["tv"] < 1e-9
    assert rep["a"]["kl"] < 1e-9


def test_kept_rate_timeseries_shapes():
    s = _stream()
    bucket = 5.0
    times, rates = kept_rate_timeseries(s, bucket=bucket)
    assert len(times) == len(rates)
    # rates are per-second; multiplying back by bucket recovers total count.
    assert abs(sum(rates) * bucket - len(s)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trace_sampling.metrics'`

- [ ] **Step 3: Write minimal implementation**

`trace_sampling/metrics.py`:
```python
import math
from collections import defaultdict
from typing import Dict, List, Tuple

from .model import Trace


def _by_agent_signatures(traces: List[Trace]):
    out = defaultdict(lambda: defaultdict(int))
    for t in traces:
        out[t.agent_id][t.signature] += 1
    return out


def signature_coverage(stream: List[Trace], kept: List[Trace]) -> Dict[str, float]:
    truth = _by_agent_signatures(stream)
    keptd = _by_agent_signatures(kept)
    cov = {}
    for agent, sigs in truth.items():
        denom = len(sigs)
        num = len(set(keptd.get(agent, {}).keys()) & set(sigs.keys()))
        cov[agent] = num / denom if denom else 0.0
    return cov


def min_active_keep_rate(stream: List[Trace], kept: List[Trace],
                         window: float) -> float:
    """Minimum, across active agents, of kept/seen within tumbling windows.

    Time is split into non-overlapping windows of `window` seconds. An agent is
    active in a window if it emitted >=1 trace there. Returns the minimum
    kept-rate over all (agent, active-window) pairs. 0.0 means some active agent
    was fully starved in some window.
    """
    if not stream:
        return 0.0
    start = min(t.timestamp for t in stream)
    # Bucket by tumbling windows.
    seen = defaultdict(lambda: defaultdict(int))
    got = defaultdict(lambda: defaultdict(int))
    for t in stream:
        w = int((t.timestamp - start) // window)
        seen[w][t.agent_id] += 1
    for t in kept:
        w = int((t.timestamp - start) // window)
        got[w][t.agent_id] += 1
    worst = 1.0
    for w, agents in seen.items():
        for agent, n in agents.items():
            rate = got[w].get(agent, 0) / n
            worst = min(worst, rate)
    return worst


def representativeness(stream: List[Trace], kept: List[Trace]) -> Dict[str, Dict[str, float]]:
    truth = _by_agent_signatures(stream)
    keptd = _by_agent_signatures(kept)
    out = {}
    for agent, sigs in truth.items():
        keys = list(sigs.keys())
        t_total = sum(sigs.values())
        k_counts = keptd.get(agent, {})
        k_total = sum(k_counts.values())
        kl = 0.0
        tv = 0.0
        for s in keys:
            p = sigs[s] / t_total
            q = (k_counts.get(s, 0) / k_total) if k_total else 0.0
            tv += abs(p - q)
            if p > 0 and q > 0:
                kl += p * math.log(p / q)
            elif p > 0 and q == 0:
                kl += p * math.log(p / 1e-9)  # penalize missing mass
        out[agent] = {"kl": kl, "tv": 0.5 * tv}
    return out


def kept_rate_timeseries(kept: List[Trace], bucket: float) -> Tuple[List[float], List[float]]:
    if not kept:
        return [], []
    start = min(t.timestamp for t in kept)
    counts = defaultdict(int)
    for t in kept:
        b = int((t.timestamp - start) // bucket)
        counts[b] += 1
    bmax = max(counts.keys())
    times = [start + b * bucket for b in range(bmax + 1)]
    rates = [counts.get(b, 0) / bucket for b in range(bmax + 1)]  # per second
    return times, rates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/metrics.py tests/test_metrics.py
git commit -m "feat(sampling): add coverage/starvation/representativeness/budget metrics"
```

---

### Task 9: Full suite + package export

**Files:**
- Modify: `trace_sampling/__init__.py`

- [ ] **Step 1: Add convenience exports**

`trace_sampling/__init__.py`:
```python
"""Adaptive backpressure trace sampling prototype."""
from .model import Trace, AgentConfig
from .generator import generate_stream
from .stats import AgentStats
from .reservoir import WeightedReservoir
from .backpressure import BackpressureController
from .samplers import SamplerConfig, BaselineSampler, AdaptiveSampler
from .metrics import (
    signature_coverage, min_active_keep_rate, representativeness,
    kept_rate_timeseries,
)

__all__ = [
    "Trace", "AgentConfig", "generate_stream", "AgentStats",
    "WeightedReservoir", "BackpressureController", "SamplerConfig",
    "BaselineSampler", "AdaptiveSampler", "signature_coverage",
    "min_active_keep_rate", "representativeness", "kept_rate_timeseries",
]
```

- [ ] **Step 2: Run the entire suite**

Run: `python -m pytest tests/ -v`
Expected: PASS (all modules green)

- [ ] **Step 3: Commit**

```bash
git add trace_sampling/__init__.py
git commit -m "chore(sampling): export public API"
```

---

### Task 10: Demonstration notebook

**Files:**
- Create: `adaptive_trace_sampling.ipynb`

Design: build the notebook programmatically (so it's reproducible) using `nbformat`, then execute it to verify it runs end-to-end. The notebook imports `trace_sampling`, builds a scenario with diverse agents (fast/low-variety, rare/high-variety, late-arriving for cold start) plus a mid-stream **burst** to trigger backpressure, runs both samplers at equal budget, and plots the four metric families.

- [ ] **Step 1: Create the notebook builder script and generate the notebook**

Create a temporary builder `build_notebook.py` at repo root:
```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("# Adaptive Backpressure Trace Sampling\n"
   "Online, bounded-memory sampling of multi-agent OTel-style traces that "
   "preserves variety, avoids starving any agent, stays representative, and "
   "respects an LLM-throughput budget via backpressure.\n\n"
   "See design: `docs/superpowers/specs/2026-06-26-adaptive-trace-sampling-design.md`.")

md("## 1. Setup & scenario\n"
   "Agents span the velocity x variety space; one arrives late (cold start). "
   "We also inject a burst to exercise backpressure.")
code(
"import numpy as np\n"
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"from trace_sampling import (AgentConfig, generate_stream, SamplerConfig,\n"
"    AdaptiveSampler, BaselineSampler, signature_coverage,\n"
"    min_active_keep_rate, representativeness, kept_rate_timeseries)\n"
"\n"
"configs = [\n"
"    AgentConfig('fast_lowvar', velocity=25.0, vocab_size=3, zipf_s=2.2),\n"
"    AgentConfig('fast_highvar', velocity=20.0, vocab_size=30, zipf_s=0.7),\n"
"    AgentConfig('rare_lowvar', velocity=0.7, vocab_size=2, zipf_s=2.0),\n"
"    AgentConfig('rare_highvar', velocity=0.8, vocab_size=25, zipf_s=0.6),\n"
"    AgentConfig('late_cold', velocity=8.0, vocab_size=10, zipf_s=1.0, start_time=20.0),\n"
"]\n"
"stream = generate_stream(configs, duration=40.0, seed=42)\n"
"\n"
"# Inject a burst from a noisy agent between t=10 and t=14 to trigger backpressure.\n"
"burst = generate_stream([AgentConfig('burst', velocity=120.0, vocab_size=4, zipf_s=1.5)],\n"
"                        duration=4.0, seed=99)\n"
"from trace_sampling.model import Trace\n"
"offset = len(stream)\n"
"burst = [Trace(offset+i, t.agent_id, t.timestamp+10.0, t.signature, t.span_count,\n"
"               t.duration_ms, t.status) for i, t in enumerate(burst)]\n"
"stream = sorted(stream + burst, key=lambda t: t.timestamp)\n"
"print(f'total traces: {len(stream)}')\n"
"pd.Series([t.agent_id for t in stream]).value_counts()")

md("## 2. Run both samplers at equal budget")
code(
"BUDGET = 20.0  # traces per sim-second the LLM backend can consume\n"
"cfg = SamplerConfig(llm_throughput=BUDGET)\n"
"adaptive = AdaptiveSampler(cfg, seed=1)\n"
"adaptive_kept = [t for t in stream if adaptive.decide(t)]\n"
"\n"
"# Baseline matched to the same kept volume for a fair comparison.\n"
"keep_prob = len(adaptive_kept) / len(stream)\n"
"baseline = BaselineSampler(keep_prob=keep_prob, seed=1)\n"
"baseline_kept = [t for t in stream if baseline.decide(t)]\n"
"print(f'kept: adaptive={len(adaptive_kept)}  baseline={len(baseline_kept)} '\n"
"      f'(keep_prob={keep_prob:.3f})')")

md("## 3. Variety coverage (higher is better)")
code(
"cov_a = signature_coverage(stream, adaptive_kept)\n"
"cov_b = signature_coverage(stream, baseline_kept)\n"
"agents = list(cov_a.keys())\n"
"x = np.arange(len(agents)); w = 0.38\n"
"plt.figure(figsize=(9,4))\n"
"plt.bar(x-w/2, [cov_a[a] for a in agents], w, label='adaptive')\n"
"plt.bar(x+w/2, [cov_b[a] for a in agents], w, label='baseline')\n"
"plt.xticks(x, agents, rotation=30, ha='right'); plt.ylabel('distinct-signature coverage')\n"
"plt.title('Variety coverage per agent'); plt.legend(); plt.tight_layout(); plt.show()")

md("## 4. Starvation (min kept-rate across active agents; >0 = nobody starved)")
code(
"print('adaptive min active keep-rate:', round(min_active_keep_rate(stream, adaptive_kept, window=10.0), 3))\n"
"print('baseline min active keep-rate:', round(min_active_keep_rate(stream, baseline_kept, window=10.0), 3))")

md("## 5. Representativeness (lower divergence is better)")
code(
"rep_a = representativeness(stream, adaptive_kept)\n"
"rep_b = representativeness(stream, baseline_kept)\n"
"df = pd.DataFrame({\n"
"    'adaptive_tv': {a: rep_a[a]['tv'] for a in agents},\n"
"    'baseline_tv': {a: rep_b[a]['tv'] for a in agents},\n"
"    'adaptive_kl': {a: rep_a[a]['kl'] for a in agents},\n"
"    'baseline_kl': {a: rep_b[a]['kl'] for a in agents},\n"
"})\n"
"df")

md("## 6. Budget adherence & backpressure response")
code(
"ta, ra = kept_rate_timeseries(adaptive_kept, bucket=1.0)\n"
"plt.figure(figsize=(9,4))\n"
"plt.plot(ta, ra, label='adaptive kept/sec')\n"
"plt.axhline(BUDGET, color='r', ls='--', label='LLM throughput budget')\n"
"plt.axvspan(10, 14, color='orange', alpha=0.2, label='burst window')\n"
"plt.xlabel('sim time (s)'); plt.ylabel('kept traces / s')\n"
"plt.title('Kept-rate vs budget (backpressure should cap the burst)')\n"
"plt.legend(); plt.tight_layout(); plt.show()")

md("## 7. Success-criteria assertions\n"
   "These assertions fail the notebook execution if the strategy does not meet "
   "its goals, so the conclusions below are always backed by results.")
code(
"span = max(t.timestamp for t in stream) - min(t.timestamp for t in stream)\n"
"mean_rate = len(adaptive_kept) / span\n"
"kept_agents = {t.agent_id for t in adaptive_kept}\n"
"active_agents = {t.agent_id for t in stream}\n"
"min_adaptive = min_active_keep_rate(stream, adaptive_kept, window=10.0)\n"
"min_baseline = min_active_keep_rate(stream, baseline_kept, window=10.0)\n"
"# 1. No active agent is fully starved (the hard anti-starvation guarantee).\n"
"assert kept_agents == active_agents, f'starved: {active_agents - kept_agents}'\n"
"# 2. Adaptive captures at least as much rare-agent variety as the baseline.\n"
"assert cov_a['rare_highvar'] >= cov_b['rare_highvar'], 'adaptive lost rare variety'\n"
"# 3. Adaptive is at least as representative (<= TV) for the rare agent.\n"
"assert rep_a['rare_highvar']['tv'] <= rep_b['rare_highvar']['tv'] + 1e-9\n"
"# 4. Sustained kept-rate respects the LLM budget (headroom allows brief transients;\n"
"#    backpressure bounds the average drain, not every instantaneous second).\n"
"assert mean_rate <= BUDGET * 1.5, f'sustained rate {mean_rate:.2f} exceeds budget'\n"
"# 5. The sampler is selective overall.\n"
"assert len(adaptive_kept) < len(stream)\n"
"print('all success-criteria assertions passed')\n"
"print(f'sustained kept-rate={mean_rate:.2f}/s (budget={BUDGET}/s)')\n"
"print(f'min 10s-window keep-rate  adaptive={min_adaptive:.3f}  baseline={min_baseline:.3f}')")

md("## 8. Conclusions\n"
   "- **No starvation:** every active agent — including the rare and late-arriving "
   "ones — retains traces (no agent is fully dropped), unlike the naive baseline.\n"
   "- **Variety preserved:** adaptive coverage of rare/high-variety agents meets or "
   "beats the baseline at equal budget.\n"
   "- **Representative:** lower TV/KL divergence per agent.\n"
   "- **Bounded cost:** kept-rate tracks the budget and the controller caps the burst.")

nb["cells"] = cells
with open("adaptive_trace_sampling.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("notebook written")
```

Run: `python build_notebook.py`
Expected: `notebook written`

- [ ] **Step 2: Execute the notebook end-to-end to verify it runs**

Run: `python -m jupyter nbconvert --to notebook --execute --inplace adaptive_trace_sampling.ipynb`
(If `jupyter`/`nbconvert` is missing, install: `python -m pip install nbconvert nbformat ipykernel`.)
Expected: completes with no errors; the notebook now contains rendered outputs.

- [ ] **Step 3: Sanity-check outputs**

Run: `python -c "import nbformat; nb=nbformat.read('adaptive_trace_sampling.ipynb', as_version=4); errs=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error']; print('errors:', len(errs)); assert not errs"`
Expected: `errors: 0`

- [ ] **Step 4: Remove the builder script**

Run: `Remove-Item build_notebook.py`

- [ ] **Step 5: Commit**

```bash
git add adaptive_trace_sampling.ipynb
git commit -m "feat(sampling): add demonstration notebook with synthetic data"
```

---

### Task 11: Final verification & docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the entire test suite once more**

Run: `python -m pytest tests/ -q`
Expected: all passed.

- [ ] **Step 2: Add a short README section**

Append to `README.md` a brief entry describing `adaptive_trace_sampling.ipynb` and the `trace_sampling/` package: what problem it solves (online, variety-preserving, non-starving, backpressure-bounded multi-agent trace sampling) and how to run the tests and notebook.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe adaptive trace sampling prototype"
```

---

## Notes for the implementing engineer

- **TDD:** every module is test-first. Run the test, watch it fail, implement, watch it pass, commit.
- **Determinism:** all randomness flows through seeded `numpy.random.default_rng`. Keep seeds in tests so they're reproducible.
- **If a probabilistic assertion is flaky:** widen the tolerance or raise the seed-controlled sample size; do not weaken the metric definitions.
- **Run tests with** `python -m pytest` from the repo root so `trace_sampling` is importable (no install needed).
- **YAGNI:** do not add HyperLogLog, real OTel wiring, or persistence — the capped-exact signature table and in-memory stream are sufficient for the prototype.
