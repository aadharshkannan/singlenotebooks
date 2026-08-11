# Full-Session Embedding Concept-Coverage Tuning

## Status

Completed on 2026-08-10 using the persisted V4/V3 exact budgets, paired replay orders, token inventory, and a deterministic reconstructed embedding cache.

**Decision:** retain the existing selector configuration for authoritative use. The nominal tuned candidate improved coverage on the tuning replays, but the gain did not reproduce consistently on the held-out replay.

## Question

Can the full-session embedding selector improve concept coverage under the same exact token budgets by tuning its clustering behavior?

This experiment tuned selection, not IDW estimation. IDW parameters such as donor `k`, power, distance epsilon, and prior probability cannot change concept coverage after membership has been frozen.

## Inputs and Provenance

- Population: 2,800 sessions.
- Source bundle: `outputs_sampling_v4/runs/full-20260805/source_v3`.
- Exact budgets: 65,949; 131,898; 263,797; 395,695; and 659,492 tokens.
- Repetitions: 0 and 1 for tuning; repetition 2 held out.
- Source manifest SHA-256: `c7a393ea68d7a222c566803a0c90bf84c5e085c25ebc341cc538dd3ed24251cd`.
- Source runs SHA-256: `a7f33581effef6b50a458d48d5fd0b384ba73e6d6ce3252e660e72e3407c4652`.
- Vector cache: `idw-vectors.deterministic-seed13.npz`, seed 13, 1,536 dimensions.
- Machine-readable results: [coverage-parameter-sweep.deterministic.json](coverage-parameter-sweep.deterministic.json).
- Tuning implementation: [../../../../scripts/tune_sampling_v4_coverage.py](../../../../scripts/tune_sampling_v4_coverage.py).

The vector cache uses the repository's deterministic embedder. It does **not** reproduce the original Azure `text-embedding-3-small` vectors, which were not retained in the V4 artifact bundle.

## Label Boundary

No outcome labels were used by selection or candidate ranking. The selector used only:

- deterministic replay order;
- exact token budgets and per-session token costs;
- agent identifiers;
- cached embedding vectors; and
- adaptive novelty and rarity signals.

Concept metadata (`corpus_id|domain|task|difficulty`) was used only after selection to score concept coverage. No expected labels, pass rates, MAE values, or LLM judge outputs were used.

## Fixed Selection Semantics

The experiment preserved the V4/V3 comparison contract:

- maximal whole-session token packing;
- native adaptive proposals followed by deterministic fill;
- `agent_floor=0.0`;
- `enforce_keep_one_floor=False`;
- fixed TTL of 90 seconds;
- purge interval of 200 observations;
- recent buffer size of 4,096; and
- the remaining `SamplerConfig` defaults.

Budget utilization remained approximately 99.9% for both baseline and tuned configurations.

## Parameter Search

Stage 1 varied the cosine cluster threshold:

```text
tau = 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85
```

Stage 2 took the two best tuning-only thresholds and varied:

```text
cluster staleness k = 4, 8, 16, 32
IAT EWMA alpha      = 0.1, 0.3, 0.5
```

Candidates were ranked using repetitions 0 and 1 by:

1. higher mean concept coverage;
2. lower zero-selection-agent rate;
3. lower p95 selection-decision latency; and
4. proximity to the existing `tau=0.55` baseline.

Repetition 2 was evaluated only after ranking and did not influence the winner.

## Configurations

| Parameter | Existing baseline | Nominal tuning winner |
|---|---:|---:|
| Cosine threshold `tau` | 0.55 | 0.65 |
| Staleness multiplier `k` | 16 | 8 |
| IAT EWMA alpha | 0.3 | 0.5 |
| Recent buffer | 4,096 | 4,096 |

## Overall Result

| Split | Baseline coverage | Tuned coverage | Tuned minus baseline |
|---|---:|---:|---:|
| Tuning, repetitions 0-1 | 63.44% | 64.88% | +1.44 pp |
| Held out, repetition 2 | 64.17% | 64.21% | +0.04 pp |

Only about 3% of the tuning-set gain appeared in the held-out replay. The result is therefore consistent with replay-order overfitting rather than a robust selector improvement.

The tuned candidate also increased the overall mean zero-selection-agent rate:

| Split | Baseline | Tuned |
|---|---:|---:|
| Tuning | 28.48% | 31.71% |
| Held out | 26.29% | 29.14% |

This means the coverage objective favored broader concept coverage while leaving more agents with no selected sessions, particularly at lower budgets.

## Held-Out Results by Budget

| Legacy tier | Exact tokens | Baseline coverage | Tuned coverage | Difference |
|---:|---:|---:|---:|---:|
| 5% | 65,949 | 20.83% | 21.46% | +0.63 pp |
| 10% | 131,898 | 38.13% | 37.50% | -0.63 pp |
| 20% | 263,797 | 69.58% | 70.21% | +0.63 pp |
| 30% | 395,695 | 93.96% | 92.29% | -1.67 pp |
| 50% | 659,492 | 98.33% | 99.58% | +1.25 pp |

The tuned candidate improved three held-out budgets and regressed two. The largest regression occurred at the 30% budget, while the largest gain occurred at 50% where deterministic baseline coverage was already near saturation.

## Authoritative Cross-Method Context

The valid Random, MinHash, and embedding comparison remains the persisted Azure V4 result:

| Legacy tier | Exact tokens | Random | MinHash | Original Azure embedding | Coverage winner |
|---:|---:|---:|---:|---:|---|
| 5% | 65,949 | 23.2% | 21.8% | 19.4% | Random |
| 10% | 131,898 | 35.4% | 35.9% | 34.8% | MinHash |
| 20% | 263,797 | 48.3% | 50.5% | 65.1% | Embedding |
| 30% | 395,695 | 55.4% | 59.1% | 78.7% | Embedding |
| 50% | 659,492 | 68.5% | 75.0% | 87.9% | Embedding |

The deterministic tuned values must not be inserted into this table as if they came from the same embedding regime. Even the unchanged selector baseline produced materially different deterministic coverage, especially at the 30% and 50% budgets. That gap is caused by the reconstructed vector geometry, not by parameter tuning.

## Interpretation

- The original Azure full-session embedding method remains the authoritative coverage winner at budgets of 20% and above.
- The nominal deterministic winner was `tau=0.65`, `k=8`, and `iat_alpha=0.5`.
- Its tuning gain did not replicate on the held-out replay and came with worse agent representation at lower budgets.
- The deterministic embedding regime systematically produced higher medium/high-budget coverage than the retained Azure run, so it cannot establish that the tuned selector beats Random, MinHash, or the original embedding selector.
- Concept metadata is synthetic scoring ground truth. Optimizing directly against it can overfit this corpus even without outcome-label leakage.

## Decision and Next Step

Keep the existing authoritative selector configuration:

```text
tau=0.55
k=16
iat_alpha=0.3
```

Treat `tau=0.65`, `k=8`, `iat_alpha=0.5` as a candidate only. Before changing the default, rerun the staged sweep using the original Azure embedding profile and validate it with more replay repetitions or a genuinely new population. A production change should require a consistent held-out coverage gain without materially increasing zero-selection-agent rates.
