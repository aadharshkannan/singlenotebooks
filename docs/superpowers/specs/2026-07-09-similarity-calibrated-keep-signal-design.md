# Similarity-Calibrated Keep-Signal for AzureClusterIndex — Design

**Date:** 2026-07-09
**Status:** Approved design, pending spec review
**Builds on:** `2026-07-07-embedding-variety-design.md` (embedding-based variety index)

---

## 1. Problem

`AzureClusterIndex.observe` emits two diversity signals that the `AdaptiveSampler` fuses via
`base = max(rarity, novelty)` (when `use_novelty=True`):

- `rarity = 1 / (1 + decayed_count)` — a **count**-based signal per cluster, range `[0, 0.5]`.
- `novelty = 1 − cosine_to_nearest_cluster` — a **distance**-based signal, range `~[0.2, 1.0]`.

On **real** embeddings these two terms are on incompatible scales, and `novelty` has **no zero
point at the clustering decision boundary `tau`**: a trace that confidently *joins* an existing
cluster (cosine ≈ 0.77–0.81) still reports `novelty ≈ 0.19–0.23`. Because `max()` picks whichever
term has the higher floor, this ~0.2 novelty floor dominates for essentially every trace and acts
as a flat keep-probability floor.

### Measured evidence (1093-trace concept stream, live `text-embedding-3-small`, tau=0.50)

| arm | kept | coverage | median redundancy | ARI | V-measure |
|---|---|---|---|---|---|
| baseline (random) | 64 | 1.00 | 16.5 | 0.27 | 0.49 |
| adaptive+exact | 79 | 1.00 | 17.5 | 0.27 | 0.49 |
| **adaptive+cluster (current)** | **116** | 1.00 | 18.5 | **0.53** | **0.62** |

Per-trace decomposition of the current cluster arm:

- 8 real clusters formed; 1085 of 1093 traces were **joins**.
- `novelty` on joins: mean **0.187**, median **0.234** (never approaches 0).
- cluster keep-signal `max(rarity, novelty)`: median **0.256**; the `base` was **set by novelty in
  66% of traces**.
- exact keep-signal `rarity`: median **0.031** (collapses as the ~46 signatures repeat).
- mean `cluster_base − exact_base = +0.125`.

**Root cause:** the cluster arm keeps ~1.5× more traces than the exact arm purely because the
mis-scaled `novelty` term never drops to zero for redundant cluster members. With only 8 concepts
and coverage already saturated at 1.0, those extra keeps add **redundancy, not coverage**.

**Goal:** re-derive both diversity terms so they share one `[0, 1]` "under-representation" scale, so
that redundant cluster members correctly score ~0 and the keep-volume is controlled by a single
principled, scale-invariant knob — without regressing clustering quality (ARI/V-measure).

## 2. Constraints

- **Online / streaming, bounded memory.** Per-trace decision; no unbounded per-cluster state.
- **Clean index/sampler boundary.** The signal must be computable inside `AzureClusterIndex.observe`
  from the trace stream alone (timestamps + embeddings). The index does **not** receive keep/drop
  outcomes from the sampler, and this design does not change that.
- **No interface change.** `VarietyObservation` keeps its `rarity` and `novelty` fields; the sampler
  keeps `base = max(rarity, novelty)`. Only what the cluster index *writes* into those fields
  changes. The `ExactSignatureIndex` baseline is untouched.
- **Clustering quality is invariant.** `cluster_agreement` (ARI/V-measure) is computed over the
  cluster assignment of **all** observed rows, independent of keep/drop. The keep-signal redesign
  must not alter cluster assignment, so ARI/V-measure are unchanged by construction.
- **Graceful degradation** (circuit-breaker fallback to exact signature) is preserved.

## 3. Design

### 3.1 Novelty (spatial) — binary, anchored at the boundary `tau`

Crossing `tau` *is* the "is this a new behavior?" decision:

```
novelty = 1.0  if a new cluster is created (score < tau)
novelty = 0.0  if the trace joins an existing cluster (score >= tau)
```

A cluster **member** has zero spatial novelty. This removes the flat floor. (A graded novelty above
`tau` is not possible — joins are always `score >= tau` — and graded novelty is intentionally *not*
introduced; the temporal term below carries the "sample a known cluster again" responsibility. This
separation of concerns is deliberate: novelty answers "new cluster?", staleness answers "known
cluster overdue?".)

### 3.2 Rarity (temporal) — cadence-normalized staleness (Scheme A)

Replace the count-based rarity with a **time-normalized staleness** on the same `[0, 1]` scale, whose
half-life is **dynamic**, scaled to each cluster's own observed cadence:

```
dt          = now − last_seen[cluster]                       # gap since this cluster was last observed
iat[cluster]= EWMA(dt, alpha=iat_alpha)                      # per-cluster mean inter-observe interval
half_life   = max(k * iat[cluster], eps)                     # dimensionless knob k
staleness   = 1 − 0.5 ** (dt / half_life)                    # in [0, 1)
```

- **New cluster:** `staleness = 0` (novelty=1 dominates anyway).
- **Regularly-hit cluster:** `dt ≈ iat`, so `staleness ≈ 1 − 2^(−1/k)` — a **uniform, tunable
  steady-state floor** (k=1→0.50, k=3→0.21, k=8→0.08), *identical across clusters regardless of their
  absolute velocity*.
- **Returning rare cluster:** `dt ≫ iat`, so `staleness → 1` — the trace is re-sampled (the temporal
  win: known-but-rare behaviors do not fade out).

**Why dynamic (Scheme A) over a fixed half-life:** a fixed half-life in absolute seconds is brittle —
it means something completely different on a 20-second vs a 20-minute stream, and it lets the fastest
agent dominate the keep budget. Normalizing to per-cluster cadence makes staleness **scale-invariant**
(independent of overall stream speed) and **fair across the per-agent velocity spread** (which was a
root contributor to the original inflation). The knob becomes a dimensionless `k` ("sample roughly
once per `k` arrivals of a cluster") rather than a fragile seconds value.

### 3.3 Fusion (unchanged, now principled)

The sampler still computes `base = max(rarity, novelty)`. It is now well-posed because both terms are
`[0, 1]` under-representation estimates and are **regime-disjoint**: new cluster → `novelty = 1`;
known cluster → `staleness`. No mismatched scales, no arbitrary term winning on floor height.

### 3.4 State changes inside `AzureClusterIndex`

- Keep the existing per-cluster last-seen timestamp map (currently `_last_decay_ts`) to supply
  `last_seen[cluster]`; read it *before* updating to compute `dt`.
- Add a per-cluster `iat` EWMA map (`_iat`), pruned alongside clusters on `purge_stale` (bounded
  memory).
- Remove the now-vestigial count-based rarity: `_counts` and `_bump`. `purge_stale` cleanup drops the
  `_iat`/last-seen entries instead of `_counts`.
- New constructor params: `k: float = 8.0` (dimensionless cadence multiplier) and
  `iat_alpha: float = 0.3` (EWMA smoothing). The existing `decay_half_life` param is removed (or
  retained only as a deprecated no-op if any caller passes it — resolve during implementation).

### 3.5 Default knob

Default **`k = 8.0`**. From the validated spike this yields, on the live-representative stream: cluster
arm **65 kept** (fewer than exact's 79) at **median redundancy 17.5** (equal to exact) with **ARI
0.53** (≈2× exact's 0.27) and coverage 1.0 — i.e. *equal-or-better redundancy, fewer total keeps,
far better variety fidelity*. `k` is the single tuning knob for keep-volume vs. redundancy.

## 4. Spike validation (already run, cached real embeddings, tau=0.50)

| design | kept | coverage | median redundancy | ARI |
|---|---|---|---|---|
| current (`max(rarity, 1−cos)`) | 116 | 1.00 | 18.5 | 0.531 |
| Scheme A, k=1 | 164 | 1.00 | 37.0 | 0.531 |
| Scheme A, k=3 | 117 | 1.00 | 30.5 | 0.531 |
| Scheme A, k=5 | 84 | 1.00 | 20.0 | 0.531 |
| **Scheme A, k=8 (default)** | **65** | 1.00 | **17.5** | **0.531** |

ARI/V-measure are identical across every cluster design (0.531/0.616), confirming the keep-signal is
decoupled from clustering quality. The old design behaves like Scheme A at k≈3 but via a broken,
non-uniform, non-scale-invariant floor.

## 5. Impact on the notebook and evals

- The Section 8 live assertion `med_red_cluster <= med_red_exact` — currently a documented *honest
  failure* — becomes a **genuine pass** (redundancy no longer inflated). The "Live-arm calibration &
  an honest caveat" markdown is rewritten to describe the calibration fix and the win, and the
  notebook is re-run live.
- `eval_harness` arms are unchanged in name/shape; only the cluster index's scoring differs.
- The offline `FakeEmbedder` arm (tau=0.9) collapses variants to cosine ≈ 0.99, so joins are
  `score ≥ tau` → `novelty = 0`, and staleness governs — the fix behaves correctly offline too.

## 6. Testing

1. **novelty is binary:** a first-seen signature (new cluster) → `novelty == 1.0`; an immediate
   identical repeat that joins → `novelty == 0.0`.
2. **staleness grows with the gap:** the same cluster observed at `t` and again at `t + Δ` yields
   `staleness` that increases with Δ, and `staleness == 0` on cluster creation.
3. **cadence normalization:** two clusters with very different absolute inter-arrival intervals but
   regular cadence converge to comparable steady-state staleness (`≈ 1 − 2^(−1/k)`).
4. **regression (offline `run_arm`):** the cluster arm's keep-count is `<=` the adaptive+exact
   keep-count (no inflation), while distinct cluster keys `<` distinct signatures (unification
   preserved). Existing `test_eval_harness` tests continue to pass.
5. **live smoke test** (`RUN_AZURE_TESTS=1`) remains green.

## 7. Out of scope (YAGNI)

- Budget-tracking / keep-feedback controllers on the half-life (Scheme B) — rejected: crosses the
  index/sampler boundary and overlaps the existing backpressure throttle.
- Graded (non-binary) novelty above `tau`.
- k-NN / density-based coverage signals (Level 2) — the unified TTL-geometry approach was explored
  and set aside; its temporal elegance is invisible on realistic short streams and it trades away the
  direct, interpretable `k` knob.
