# Adaptive Backpressure Sampler for Agent Traces — Design

Date: 2026-06-26
Status: Approved (design), pending implementation

## Problem

Many agents each emit OpenTelemetry-style traces. A trace represents one agent
attempting a job-to-be-done, emitting tool-use spans as it works. The agents
differ along two independent dimensions:

- **Velocity** — how frequently an agent emits traces (some frequent, some rare).
- **Variety** — how diverse an agent's traces are (some always do the same thing;
  some have a large, skewed vocabulary of distinct behaviors).

We need an **online** sampler (a tail-sampling processor) that decides keep-or-drop
per trace as it streams in, under bounded memory. The kept subset feeds an
**observability/monitoring** use case (dashboards & debugging), and each kept trace
is consumed by an **LLM backend** whose throughput is the binding cost constraint.

### Requirements

1. **No agent starved** — every active agent retains a guaranteed minimum share.
2. **Preserve variety** — rare/distinct trace signatures must survive, not be
   drowned out by high-frequency repetitive ones.
3. **Representativeness** — the kept distribution should not deviate too far from
   the true per-agent distribution.
4. **Bounded cost** — total kept-rate must respect the LLM backend throughput via
   a **backpressure** mechanism.
5. **Automatic adaptation** — per-agent sampling adjusts automatically to that
   agent's live *variety* and *velocity*.
6. **Cold start** — new/unknown agents (and unknown signatures) are handled until
   enough samples exist to estimate their statistics.
7. **Efficient** — O(1) work per trace, bounded memory (sketches/reservoirs).

### Non-goals

- No real distributed deployment, no real OTel collector wiring, no persistence
  layer. This is a single-notebook prototype on a controllable synthetic stream.
- No LLM is actually called; the "LLM backend" is modeled as a bounded-throughput
  consumer to exercise backpressure.

## Data model (OTel-shaped synthetic stream)

Each trace:

| field        | meaning                                                        |
|--------------|----------------------------------------------------------------|
| `trace_id`   | unique id                                                      |
| `agent_id`   | which agent emitted it (known at decision time)                |
| `timestamp`  | emission time (simulated clock)                                |
| `signature`  | ordered tuple of tool/span names — the **variety key**, cheap to compute at decision time |
| `span_count` | number of spans                                                |
| `duration_ms`| trace duration                                                 |
| `status`     | `ok` / `error`                                                 |

Per-agent generator dials (ground truth, so outcomes are measurable):

- `velocity` — Poisson arrival rate λ.
- `variety` — signature-vocabulary size and skew (Zipf exponent).
- `start_time` — when the agent first appears (late arrivals exercise cold start).

The generator yields an interleaved, time-ordered stream across all agents.

## Sampler design (Strategy B — adaptive)

Per-trace decision pipeline, all O(1) and bounded memory:

1. **Stratify.** Decision key = `(agent_id, signature_bucket)`. Each stratum owns
   a small **weighted reservoir**. Stratification is what prevents a high-velocity
   signature from starving a rare one within the same agent.

2. **Live per-agent statistics** (bounded memory):
   - *Velocity* — EWMA of inter-arrival rate.
   - *Variety* — distinct-signature estimate via a sketch (HyperLogLog or a
     capped exact set) plus a streaming entropy estimate.

3. **Diversity-weighted keep score.** A trace's base keep probability rises when
   its signature is rare or its stratum is under-filled, and falls for the Nth
   identical repeat. This simultaneously preserves variety and keeps the kept
   distribution close to the true one.

4. **Anti-starvation floor.** Every active agent is guaranteed retention through a
   **deterministic keep-one rule**: when an agent has had zero kept traces in the
   trailing `active_window`, its next arriving trace is kept unconditionally (budget
   permitting per step 6 precedence), independent of its diversity score. Above that
   guarantee, the agent also receives a minimum admission *share* of the budget.
   Together these ensure both an observed floor (≥1 keep per active agent per
   window) and a probabilistic floor, so the success metric is satisfiable for even
   single-trace low-volume agents. Signatures are tracked in a **bounded, LRU-evicted table**
   (cap `max_signatures_per_agent`, default 256); when full, the least-recently-seen
   signature is evicted so memory stays bounded. A signature floor applies only
   while its entry is resident; a re-appearing evicted signature is treated as
   cold-start (step 5), which restores its protection. This keeps the floor set
   finite even though the true signature space is unbounded.

   **Precedence under contention.** Floors are expressed as a guaranteed share of
   the *budget*, not an absolute admit. The controller allocates the LLM budget in
   priority order: (a) per-active-agent floors first, (b) cold-start exploration
   boosts second, (c) remaining budget distributed by diversity weight. If the sum
   of agent floors would exceed `llm_throughput` (too many simultaneously active
   agents), floors are scaled down **proportionally and equally** across agents so
   fairness is preserved while the total stays within budget — i.e. the budget cap
   in step 6 always wins, but it degrades every agent equally rather than starving
   any single one. The floor is therefore `min(configured_floor,
   fair_share_of_budget)`.

5. **Cold-start exploration.** An unknown agent or signature enters a temporary
   boosted-admission "exploration" state until its velocity/variety estimates
   stabilize (minimum sample count reached), then decays to steady state.

6. **Global backpressure controller.** A token-bucket sized to a configurable
   `llm_throughput`, with an **AIMD** admission multiplier:
   - The modeled LLM consumer drains kept traces at `llm_throughput`; a queue
     models lag.
   - When the queue grows (backpressure), the multiplier multiplicatively
     decreases, shedding the **most redundant** traces first (low diversity score).
   - When slack appears, it additively increases.
   - This keeps the **total** kept-rate bounded while the floors (step 4) and
     diversity weighting (step 3) decide *which* traces survive the squeeze.

### Default parameters (tunable in the notebook)

| parameter | default | meaning |
|-----------|---------|---------|
| `llm_throughput` | 50 traces / sim-sec | LLM consumer drain rate (the budget) |
| `agent_floor` | 0.02 of budget | guaranteed per-active-agent share |
| `active_window` | 30 sim-sec | an agent/signature is "active" if seen within this window |
| `max_signatures_per_agent` | 256 | LRU cap on tracked signatures per agent |
| `coldstart_min_samples` | 20 | samples before an agent/signature leaves exploration |
| `coldstart_boost` | 5× base rate | exploration admission multiplier |
| `ewma_alpha` | 0.1 | velocity EWMA smoothing |
| `aimd_increase` | +0.05 | additive increase of admission multiplier per slack tick |
| `aimd_decrease` | ×0.5 | multiplicative decrease per backpressure tick |
| `queue_high / queue_low` | 2× / 0.5× throughput | backpressure trigger thresholds |
| `reservoir_size` | 8 per stratum | weighted reservoir capacity |

### Baseline for comparison (Strategy A)

Fixed per-agent keep-probabilities, same total budget. Cannot adapt to
velocity/variety changes and ignores backpressure — used only as a baseline to
quantify Strategy B's improvement.

## Experiment & metrics

Run the same synthetic stream (including a deliberate **burst** segment to trigger
backpressure, and a **late-arriving agent** to trigger cold start) through both
strategies at the same LLM budget. Measure:

- **Coverage / variety retention** — fraction of distinct signatures captured per
  agent, with emphasis on rare-tail signatures.
- **Starvation** — minimum kept-rate across agents over time (must stay > 0;
  Strategy B should dominate Strategy A).
- **Representativeness** — KL divergence and total-variation distance between kept
  vs true signature distribution, per agent.
- **Budget adherence & backpressure response** — kept-rate vs `llm_throughput`
  over time across the burst.

Plots: per-agent coverage bars, kept-rate-under-backpressure time series, per-agent
starvation comparison, representativeness (divergence) comparison.

## Deliverable

A single notebook `adaptive_trace_sampling.ipynb` at the repo root (matching the
repo's notebook convention), with sections mirroring this design:

1. Synthetic OTel-shaped generator (configurable per-agent velocity/variety).
2. Sampler implementation (Strategy A baseline + Strategy B adaptive).
3. Experiment harness (interleaved stream, burst + cold-start scenarios).
4. Metrics & plots.
5. Conclusions.

Stack: pure Python + numpy / pandas / matplotlib (the repo's existing stack). No
external services or network calls.

## Success criteria

The prototype is successful if, at equal budget, Strategy B vs Strategy A shows:

- Higher rare-signature coverage and no starved agent — every agent that is
  **active** (emitted ≥1 trace within the trailing `active_window`) has kept-rate
  > 0 measured over that same window (assuming a feasible budget, i.e. the sum of
  active-agent floors does not exceed `llm_throughput`; otherwise floors degrade
  equally per step 4 precedence), **and**
- Lower per-agent distributional divergence (more representative), **while**
- Holding total kept-rate within the configured `llm_throughput` budget through the
  burst (demonstrated backpressure response).
