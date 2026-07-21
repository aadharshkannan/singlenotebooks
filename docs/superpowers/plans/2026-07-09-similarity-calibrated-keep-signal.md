# Similarity-Calibrated Keep-Signal Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `AzureClusterIndex`'s mis-scaled keep-signal (`max(count-rarity, 1−cosine)`) with a calibrated signal: binary novelty anchored at `tau` plus cadence-normalized temporal staleness, so redundant cluster members correctly score ~0 and keep-volume is governed by one dimensionless knob `k`.

**Architecture:** Only what `AzureClusterIndex.observe` *writes* into `VarietyObservation.novelty`/`rarity` changes. Novelty becomes binary (1.0 new cluster / 0.0 join). Rarity becomes `1 − 0.5^(dt/half_life)` where `half_life = k · EWMA(per-cluster inter-arrival)`, computed from the *prior* cadence before folding in the current gap. The `VarietyObservation` interface, the sampler's `base = max(rarity, novelty)` fusion, the `ExactSignatureIndex` baseline, and cluster assignment (hence ARI/V-measure) are all unchanged.

**Tech Stack:** Python, numpy, pytest. Files: `trace_sampling/cluster_index.py`, `tests/test_cluster_index.py`, `trace_sampling/eval_harness.py`, `tests/test_eval_harness.py`, `adaptive_trace_sampling.ipynb`.

**Spec:** `docs/superpowers/specs/2026-07-09-similarity-calibrated-keep-signal-design.md`

---

## Chunk 1: Cluster-index keep-signal redesign

### Task 1: Constructor — new params, validation, state; remove `decay_half_life`

**Files:**
- Modify: `trace_sampling/cluster_index.py:54-73` (`__init__`)
- Test: `tests/test_cluster_index.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cluster_index.py`:

```python
def test_ctor_rejects_bad_k():
    with pytest.raises(ValueError):
        _index(k=0.0)

def test_ctor_rejects_bad_iat_alpha():
    with pytest.raises(ValueError):
        _index(iat_alpha=0.0)
    with pytest.raises(ValueError):
        _index(iat_alpha=1.5)

def test_ctor_defaults_present():
    idx = _index()
    assert idx.k == 8.0
    assert idx.iat_alpha == 0.3
    assert not hasattr(idx, "_decay_half_life")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cluster_index.py -k "ctor" -q`
Expected: FAIL (`k`/`iat_alpha` are unexpected kwargs / attributes missing).

- [ ] **Step 3: Edit the constructor**

In `cluster_index.py`, add a module constant near the top (after imports):

```python
_EPS = 1e-9
```

Replace the `__init__` signature line:

```python
    def __init__(self, cache: EmbeddingCache, store: VectorStore, tau: float = 0.85,
                 ttl: float = 60.0, purge_every: int = 200,
                 embed_budget_per_tick: int = 8, recent_buffer_size: int = 64,
                 breaker=None, decay_half_life: float = 30.0):
```

with:

```python
    def __init__(self, cache: EmbeddingCache, store: VectorStore, tau: float = 0.85,
                 ttl: float = 60.0, purge_every: int = 200,
                 embed_budget_per_tick: int = 8, recent_buffer_size: int = 64,
                 breaker=None, k: float = 8.0, iat_alpha: float = 0.3):
        if not k > 0:
            raise ValueError(f"k must be > 0, got {k}")
        if not (0.0 < iat_alpha <= 1.0):
            raise ValueError(f"iat_alpha must be in (0, 1], got {iat_alpha}")
```

In the constructor body, replace these three lines:

```python
        self._counts = {}
        self._last_decay_ts = {}
        self._decay_half_life = decay_half_life
```

with:

```python
        self.k = k
        self.iat_alpha = iat_alpha
        self._last_seen = {}        # cluster_id -> last observe timestamp
        self._iat = {}              # cluster_id -> EWMA of inter-observe gap (seeded on first join)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cluster_index.py -k "ctor" -q`
Expected: PASS. (Other tests in the file will fail until Task 2/3 — that is expected; do not fix them here.)

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/cluster_index.py tests/test_cluster_index.py
git commit -m "refactor(cluster-index): add k/iat_alpha ctor params, drop decay_half_life"
```

---

### Task 2: Binary novelty anchored at `tau`

**Files:**
- Modify: `trace_sampling/cluster_index.py` (`observe`, the join branch)
- Test: `tests/test_cluster_index.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cluster_index.py`:

```python
def test_novelty_is_binary():
    idx = _index()
    new = idx.observe(_t(("search",), agent="a", cid=0))
    assert new.novelty == 1.0                      # new cluster
    join = idx.observe(_t(("query",), agent="a", cid=0))
    assert join.novelty == 0.0                     # joins existing cluster -> exactly 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cluster_index.py::test_novelty_is_binary -q`
Expected: FAIL — current join novelty is `max(0, 1−score)` ≈ 0.1, not 0.0.

- [ ] **Step 3: Edit the join branch of `observe`**

In `cluster_index.py`, in the `if near is not None and near[1] >= self.tau:` branch, replace:

```python
                cluster_id, score = near
                novelty = max(0.0, 1.0 - score)
                self._store.touch(cluster_id, trace.timestamp)
                self._touch_recent(cluster_id, trace.timestamp)
```

with:

```python
                cluster_id, score = near
                novelty = 0.0
                self._store.touch(cluster_id, trace.timestamp)
                self._touch_recent(cluster_id, trace.timestamp)
```

(The new-cluster branch already sets `novelty = 1.0`; leave it unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cluster_index.py::test_novelty_is_binary -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/cluster_index.py tests/test_cluster_index.py
git commit -m "feat(cluster-index): binary novelty anchored at tau (join -> 0.0)"
```

---

### Task 3: Cadence-normalized staleness rarity (replace `_bump`)

**Files:**
- Modify: `trace_sampling/cluster_index.py` (remove `_bump` at ~110-116; add `_staleness`; rewrite creation + join timestamp bookkeeping and the final `rarity = ...` line in `observe`)
- Test: `tests/test_cluster_index.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cluster_index.py`:

```python
def test_creation_has_zero_staleness_rarity():
    idx = _index()
    new = idx.observe(_t(("search",), agent="a", ts=0.0, cid=0))
    assert new.rarity == 0.0                        # nothing stale on first sight

def test_zero_gap_join_has_zero_staleness():
    # Two joins at the identical timestamp -> dt=0 -> staleness exactly 0, no div-by-zero.
    idx = _index()
    idx.observe(_t(("search",), agent="a", ts=5.0, cid=0))   # create
    idx.observe(_t(("query",),  agent="a", ts=6.0, cid=0))   # first join seeds iat
    same_ts = idx.observe(_t(("find",), agent="a", ts=6.0, cid=0))  # dt = 0
    assert same_ts.rarity == 0.0

def test_staleness_grows_with_gap():
    # Two indices, identical except the returning gap; larger gap -> larger rarity.
    def run(gap):
        idx = _index(k=8.0)
        idx.observe(_t(("search",), agent="a", ts=0.0, cid=0))   # create
        idx.observe(_t(("query",),  agent="a", ts=1.0, cid=0))   # first join, seeds iat=1
        return idx.observe(_t(("find",), agent="a", ts=1.0 + gap, cid=0)).rarity
    assert run(50.0) > run(1.0) > 0.0

def test_cadence_normalization_converges_across_speeds():
    # A regularly-hit cluster reaches steady-state staleness ~ 1 - 2^(-1/k),
    # independent of absolute cadence. Compare a fast and a slow cluster.
    import math
    k = 8.0
    expected = 1 - 2 ** (-1.0 / k)
    def steady(step):
        idx = _index(k=k)
        idx.observe(_t(("search",), agent="a", ts=0.0, cid=0))    # create
        r = 0.0
        for i in range(1, 6):
            r = idx.observe(_t(("query",), agent="a", ts=i * step, cid=0)).rarity
        return r
    fast, slow = steady(0.5), steady(50.0)
    assert abs(fast - expected) < 0.05
    assert abs(slow - expected) < 0.05
    assert abs(fast - slow) < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cluster_index.py -k "staleness or cadence or creation_has_zero or zero_gap" -q`
Expected: FAIL (rarity still count-based; `test_creation_has_zero_staleness_rarity` currently returns 0.5).

- [ ] **Step 3: Implement staleness and rewire `observe`**

Delete the `_bump` method (`cluster_index.py:110-116`) and add in its place:

```python
    def _staleness(self, cluster_id: str, now: float) -> float:
        """Cadence-normalized temporal under-representation in [0, 1).

        Uses the PRIOR inter-arrival EWMA to size the half-life, THEN folds the
        current gap into the EWMA. Reading the prior cadence first is what lets a
        long-absent cluster score ~1 instead of cancelling its own staleness."""
        dt = max(0.0, now - self._last_seen.get(cluster_id, now))
        if cluster_id not in self._iat:                      # first join after creation -> seed
            iat_ref = max(dt, _EPS)
            self._iat[cluster_id] = iat_ref
        else:
            iat_ref = max(self._iat[cluster_id], _EPS)
            self._iat[cluster_id] = self.iat_alpha * dt + (1.0 - self.iat_alpha) * self._iat[cluster_id]
        half_life = max(self.k * iat_ref, _EPS)
        self._last_seen[cluster_id] = now
        return 1.0 - 0.5 ** (dt / half_life)
```

In `observe`, in the **new-cluster** branch, add last-seen bookkeeping (no `_iat` seed) right after `cluster_id = self._new_id()`:

```python
                cluster_id = self._new_id()
                novelty = 1.0
                self._last_seen[cluster_id] = trace.timestamp
                self._store.upsert(VectorDoc(cluster_id, vec, trace.agent_id, trace.timestamp))
```

Finally, replace the last line of `observe`:

```python
        rarity = self._bump(cluster_id, trace.timestamp)
        return VarietyObservation(VarietyKey("cluster", cluster_id), rarity, novelty)
```

with:

```python
        if novelty == 1.0:                    # new cluster: nothing is stale yet
            rarity = 0.0
        else:
            rarity = self._staleness(cluster_id, trace.timestamp)
        return VarietyObservation(VarietyKey("cluster", cluster_id), rarity, novelty)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cluster_index.py -k "staleness or cadence or creation_has_zero or zero_gap" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/cluster_index.py tests/test_cluster_index.py
git commit -m "feat(cluster-index): cadence-normalized staleness rarity (replaces count-rarity)"
```

---

### Task 4: Purge cleanup of per-cluster state

**Files:**
- Modify: `trace_sampling/cluster_index.py` (the `purge_stale` block in `observe`, ~131-134)
- Test: `tests/test_cluster_index.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cluster_index.py`:

```python
def test_purge_drops_per_cluster_state():
    idx = _index(ttl=10.0, purge_every=1)
    first = idx.observe(_t(("search",), agent="a", ts=0.0, cid=0))
    cid = first.key.value
    idx.observe(_t(("query",), agent="a", ts=1.0, cid=0))     # join -> seeds _iat[cid]
    assert cid in idx._iat and cid in idx._last_seen
    # far-future trace triggers purge of the now-stale cluster
    idx.observe(_t(("search",), agent="a", ts=100.0, cid=0))
    assert cid not in idx._iat
    assert cid not in idx._last_seen
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cluster_index.py::test_purge_drops_per_cluster_state -q`
Expected: FAIL — purge still pops `_counts`/`_last_decay_ts`, which no longer exist (AttributeError) or leaves `_iat`/`_last_seen` populated.

- [ ] **Step 3: Edit the purge block**

In `observe`, replace:

```python
                for cid in self._store.purge_stale(now=trace.timestamp, ttl=self.ttl):
                    self._counts.pop(cid, None)
                    self._last_decay_ts.pop(cid, None)
```

with:

```python
                for cid in self._store.purge_stale(now=trace.timestamp, ttl=self.ttl):
                    self._iat.pop(cid, None)
                    self._last_seen.pop(cid, None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cluster_index.py::test_purge_drops_per_cluster_state -q`
Expected: PASS.

- [ ] **Step 5: Run the full cluster-index suite (no regressions)**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cluster_index.py -q`
Expected: PASS (all non-`@pytest.mark.azure` tests). Existing tests `test_same_concept_joins_same_cluster` and `test_recent_low_sim...` (asserting `novelty < 0.5`) still pass since join novelty is now `0.0`.

- [ ] **Step 6: Commit**

```bash
git add trace_sampling/cluster_index.py tests/test_cluster_index.py
git commit -m "fix(cluster-index): purge drops _iat/_last_seen for bounded per-cluster state"
```

---

## Chunk 2: Eval regression guard + notebook

### Task 5: Offline regression — cluster keeps ≤ exact keeps

**Files:**
- Modify: `tests/test_eval_harness.py`
- Reference (no change expected): `trace_sampling/eval_harness.py` (arms `adaptive_exact`, `adaptive_cluster_offline`)

- [ ] **Step 1: Write the failing/guard test**

Add to `tests/test_eval_harness.py` (reuse the existing module `_stream()` and `_sm()` helpers and the real `run_arm(stream, cfg, arm=..., ...)` signature):

```python
def test_cluster_arm_does_not_inflate_keeps_offline():
    # Calibrated keep-signal must NOT keep more than the exact arm on the same stream.
    stream = _stream()
    cfg = SamplerConfig(llm_throughput=20.0)
    exact = run_arm(stream, cfg, arm="adaptive_exact", seed=0)
    cluster = run_arm(stream, cfg, arm="adaptive_cluster_offline", seed=0, synonym_map=_sm())
    assert cluster.ledger["kept"] <= exact.ledger["kept"]
    # cluster unification still collapses signature variants into fewer distinct keys
    assert cluster.log["variety_key"].nunique() < exact.log["variety_key"].nunique()
```

`_stream()`, `_sm()`, `run_arm`, and `SamplerConfig` are already imported/defined at the top of `tests/test_eval_harness.py` — do not redefine them.

- [ ] **Step 2: Run test**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_eval_harness.py::test_cluster_arm_does_not_inflate_keeps_offline -q`
Expected: PASS with the calibrated signal. If it FAILS because `kept` inflation remains, stop and re-check Task 2/3 — do not weaken the assertion.

- [ ] **Step 3: Run the whole harness + variety suites**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_eval_harness.py tests/test_cluster_index.py tests/test_variety.py -q`
Expected: PASS (all non-azure). Confirms no arm renamed/broke.

- [ ] **Step 4: Commit**

```bash
git add tests/test_eval_harness.py
git commit -m "test(eval): guard cluster arm against keep inflation offline"
```

---

### Task 6: Full test sweep

**Files:** none (verification task)

- [ ] **Step 1: Run the entire non-azure suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q -m "not azure"`
Expected: all pass (prior baseline was 68 passed / 2 skipped; the new tests add to the passed count). If anything fails, fix the offending task before proceeding.

---

### Task 7: Live Azure re-run + notebook Section 8 caveat rewrite

> Requires live Azure resources (`RUN_AZURE_TESTS=1`, `.env` present). This spends money; confirm with the user before running. The redundancy assertion `med_red_cluster <= med_red_exact` in cell 25 should now PASS.

**Files:**
- Modify: `adaptive_trace_sampling.ipynb` cell 24 (caveat markdown) — rewrite from "honest failure" to "calibration fix + win"
- Re-run: whole notebook live

- [ ] **Step 1: Rewrite the Section 8 caveat markdown (cell 24)**

Open the notebook (nbformat or editor) and replace the "Live-arm calibration & an honest caveat" markdown with text describing: (a) the original inflation root cause (novelty floor), (b) the calibrated fix (binary novelty + cadence-normalized staleness, knob `k=8`), and (c) the resulting win — cluster arm keeps fewer traces than exact at equal-or-lower median redundancy while retaining ~2× ARI. Do not assert numbers you have not just reproduced; word it to reference the live cell output below it.

- [ ] **Step 2: Execute the notebook live (saves even on assertion, so you can inspect)**

Run:
```powershell
$env:RUN_AZURE_TESTS=1; .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --allow-errors --ExecutePreprocessor.kernel_name=singlenotebooks-venv --ExecutePreprocessor.timeout=900 adaptive_trace_sampling.ipynb
```
Expected: Section 8 cells populate; cell 25's `med_red_cluster <= med_red_exact` assertion PASSES (no AssertionError in output).

- [ ] **Step 3: Verify the assertion actually passed**

Inspect cell 25's output in the executed notebook. Confirm NO `AssertionError`. If it still fails, capture the printed kept/redundancy numbers and stop — the calibration needs review (do NOT re-add `--allow-errors` as a way to hide a real failure in the committed artifact; only use it to inspect, then fix).

- [ ] **Step 4: If clean, re-execute without `--allow-errors` for a truthful committed run**

Run:
```powershell
$env:RUN_AZURE_TESTS=1; .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=singlenotebooks-venv --ExecutePreprocessor.timeout=900 adaptive_trace_sampling.ipynb
```
Expected: exit 0 (all assertions pass).

- [ ] **Step 5: Commit**

```bash
git add adaptive_trace_sampling.ipynb
git commit -m "docs(notebook): rewrite Section 8 caveat as calibration win; re-run live"
```

---

### Task 8: Push to PR #6

- [ ] **Step 1: Push the branch**

Run: `git push origin embedding-variety`
Expected: updates PR #6. (Ambient credential may be `aadkannan_microsoft` → 403; the active gh github.com account is `aadharshkannan`. If push 403s, resolve auth before retrying.)

- [ ] **Step 2: Remind the user** about Azure cost/deprovision (`az group delete -n aadkannan-trace-sampling`).

---

## Notes for the implementer

- **Interface is frozen:** do not touch `VarietyObservation`, `AdaptiveSampler.decide`, `ExactSignatureIndex`, or `variety_metrics.py`. ARI/V-measure are invariant by construction (cluster assignment is unchanged).
- **Ordering is load-bearing** in `_staleness`: compute the return value from the *prior* `iat_ref`, then update the EWMA. Getting this backwards silently cancels the returning-cluster staleness.
- **`eval_harness._make_index`** does not pass `decay_half_life`, so removing it needs no harness edit — but grep to confirm before committing Task 1.
