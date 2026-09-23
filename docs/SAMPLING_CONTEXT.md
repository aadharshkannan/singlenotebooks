# Agent Evaluation Sampling Context

Updated: 2026-09-23. This is the repo's working context, combining the team's
stated direction with separately identified implementation and experiment evidence.
It is not a production rollout claim. Preserve these distinctions in future reports.

## Goal and team direction

We sample agent-user **sessions**, reconstructed from telemetry spans/traces,
to evaluate agents affordably while representing the different tasks and semantic
conversations each agent handles. Traces, sessions, agents, and tenant populations
are different counting units; do not interchange their denominators.

The team's preferred direction, as stated on 2026-09-07, is:

1. Build a canonical full-session evidence representation and embedding.
2. Select semantically diverse sessions within the appropriate tenant/agent scope.
3. Judge selected sessions with an LLM, initially for **task completion**.
4. Estimate unjudged session scores through **inverse distance weighting (IDW)**.
5. Attach conditional **Lipschitz sensitivity bounds** where their assumptions and
   calibration are supported.
6. Use **PCA to reduce 1,536-dimensional embeddings to eight dimensions**, subject
   to validation of coverage, prediction error, and the intended deployment geometry.

The aim is a score record for every session, not a claim that every session was
judged. Records must identify direct judgments, imputations, pending states, and
fallbacks. Per-agent summaries must disclose that mix and their aggregation rule.
Task adherence and tool-call success are intended extensions, not validated
outcomes of the current task-completion experiment.

### Proposed weekly policy

The user reports that, in some extreme tenants, **less than 1% of agents produce
more than 90% of traces**. This motivates agent-aware allocation; it is a planning
observation, not a statistic established by the retained experiment datasets.

The proposed policy is to judge all sessions for an agent with **at most 20
sessions per week**, and introduce sampling above that level. The repository does
not yet implement or validate that weekly gate. The threshold is not a fixed
20-judgment budget for high-volume agents and is not V7's 20% budget tier.

Before implementation, specify the tenant/agent/week key, calendar and timezone,
late-arrival handling, whether the first 20 remain guaranteed once traffic crosses
the gate, and the sampling budget and calibration policy above the threshold.

## What exists today

| Area | Implementation and evidence boundary |
| --- | --- |
| Canonical full-session evidence | [session_embedding.py](../trace_sampling/session_embedding.py) and [full_session_prototype.py](../trace_sampling/full_session_prototype.py) build bounded, content-addressed session representations. "Full session" does not mean unlimited verbatim input; report compression/truncation and token policy. |
| PCA-8 binary IDW experiment | [v7_experiment.py](../sampling_comparison/v7_experiment.py) is a compatibility facade; [v7_experiment_repaired.py](../sampling_comparison/v7_experiment_repaired.py) owns selection, PCA, replay and scoring. PCA is fitted once on each entire unlabeled corpus, then reused across replays. |
| V7 IDW | Nearest eight donors, power 2, epsilon 1e-6, same-agent preference with global fallback; exact matches have special handling. Threshold 0.5 produces binary predictions. Keep probability averages distinct from thresholded pass-rate estimates. |
| Live value-imputation prototype | [value_pipeline.py](../trace_sampling/value_pipeline.py), [value_reservoir.py](../trace_sampling/value_reservoir.py), and [lipschitz.py](../trace_sampling/lipschitz.py) provide dropped-trace value imputation and conditional geodesic bounds. This is a separate configuration/path, not the V7 experiment. |
| Bounds | Empirical agent-scoped calibration and donor distances produce deterministic sensitivity envelopes under assumptions. They are **not confidence intervals** or a universal guarantee. Missing calibration/fallback states must be visible. |
| Integration gap | The retained V7 report does not test one integrated PCA-8 + IDW + Lipschitz + weekly-policy system. Reduced-space bounds and new judge metrics require their own calibration and validation. |

### Method lineage

- [random_sampling](../random_sampling/README.md): stratified probability-sampling
  prototype. Its guarantees do not automatically transfer to later token-priority
  or fixed-budget experimental arms.
- [minhash_sampling](../minhash_sampling/README.md): lexical novelty using MinHash
  LSH; distinguishes text similarity from semantic task coverage.
- [trace_sampling](../trace_sampling/README.md): semantic full-session novelty,
  canonical evidence, and the separate value/bounds prototype.
- [agent_uniform_sampling](../agent_uniform_sampling/README.md): uniform sampling
  within agents, with sample membership separated from token execution pacing.
- [sampling_comparison](../sampling_comparison/README.md): versioned comparative
  experiments. V2 establishes the baseline; V3 compares exact token budgets; V4
  explores IDW; V6 broadens the arms; V7 compares PCA-8 distance arms. Preserve
  negative results and version-specific budget/estimator differences.

## Retained V7 evidence

Reference bundle: [v7-live-repeated-20260903](../outputs_sampling_v7/runs/v7-live-repeated-20260903/manifest.json).
This is a named historical run, not an automatically refreshed "latest" result.

| Dataset | Sessions | Agents | Representation and label provenance |
| --- | ---: | ---: | --- |
| `historical_300` | 300 | 100 | Retained synthetic historical-shape Agent365 corpus, expected labels. |
| `dense_2500` | 2,500 | 5 | Retained synthetic dense corpus, expected labels. |
| `cosmos_otel` | 205 | 7 | External `synth-data/data/cosmos_otel` label/span export; 195 linked-span representations and 10 synthetic label-document fallbacks. Linked spans alone do not establish representative real production traffic. |

Source of these counts: [dataset_profile.json](../outputs_sampling_v7/runs/v7-live-repeated-20260903/dataset_profile.json).
The external Cosmos export is not a self-contained checked-in input. Record its
availability and provenance before claiming a clean-checkout rerun.

The run crosses **three datasets x five methods x five budgets x ten paired
bootstrap replays = 750 rows**. Methods are `random_sampling`,
`minhash_lsh`, `pca8_idw_binary_cosine`, `pca8_idw_binary_euclidean`,
and `arm5_hajek_weighted`. Budgets are **1%, 3%, 5%, 10%, 20% of session
occurrences**, not V3's token budgets or the proposed weekly gate.

Membership selection is label-blind; expected outcome labels provide the
evaluation reference. A "live" embedding/Search run does not mean fresh LLM
judgments supplied the labels. Ten replays resample the same corpus with
replacement, changing order and frequency, not collecting ten independent sets
of labels. Replay intervals describe that perturbation, not production-population
confidence or judge reliability.

PCA-8 retains about 37.1%, 66.3%, and 73.2% of embedding variance for the three
corpora respectively, per the run's
[embedding ledger](../outputs_sampling_v7/runs/v7-live-repeated-20260903/embedding_ledger.json).
Variance retained is not task-completion accuracy. The full vectors are generated
before PCA; eight-dimensional storage/search does not itself imply proportionally
lower embedding API cost. A per-corpus transductive fit needs time-forward and
out-of-corpus validation before deployment.

### Measured tradeoff, not a universal winner

The run's [final_report.json](../outputs_sampling_v7/runs/v7-live-repeated-20260903/final_report.json)
records these findings:

- Random sampling has the lowest **unweighted mean of dataset-budget cell means**
  for fixed-source aggregate absolute error: about **0.1022**, or **10.22
  percentage points** on a 0-1 pass-rate scale.
- PCA-8 cosine has the highest concept-coverage ratio on the same cell-average
  basis: about **0.3440 (34.40%)**. This is coverage of the corpus's defined concept
  proxy, not proof that every agent's semantic activities are covered.
- Cosine beats Euclidean on aggregate error in **66%** of paired comparisons,
  but its win rates for imputed-only accuracy and F1 are about **42.67%** and
  **33.33%** respectively. Metric choice matters.

These pooled summaries are neither tenant-traffic-weighted nor per-agent macro
scores. Reports must show dataset/budget breakdowns and unjudged-only quality,
not hide those distinctions in one headline. Directly observed labels mixed into
judged-plus-imputed metrics make that view easier than unjudged-only prediction.

## Initial Matryoshka follow-up (local evidence)

The initial 2026-09-15 study
narrows the previous prefix study to **32, 30, ..., 2 dimensions**, retaining
1,536 dimensions as the native reference. This is prefix truncation plus L2
normalization of `text-embedding-3-small`, **not PCA**. The
local bundle `outputs_matryoshka/runs/mrl-low-dimensions-30-seed-20260915-paired/`
contains 76,500 cells: three datasets, 17 representations, 30 paired seeds,
five schedules, five session budgets and both membership modes. The original
data, labels, selection and causal IDW rules are unchanged. Tau2 is excluded
for the same unavailable expected-label mapping.

The native/32/16/8 overlap reproduces all 18,000 previous result cells and
memberships exactly. This required the original Python 3.13.13 x64 environment
and 12 OpenBLAS threads: an initial ARM64 diagnostic changed a few threshold-edge
predictions despite identical selected memberships. The matched-runtime bundle,
not that preliminary diagnostic, is the retained local evidence. Original
identifier-bearing bundles are not added to Git; see the refreshed
aggregate-only reports below for the published deliverables.

Dataset-average **end-to-end, unselected-only MAE**, including explicitly
identified mean/prior fallbacks, gives the following endpoint comparison:

| Dataset | Native MAE | 2d MAE | Relative MAE change |
| --- | ---: | ---: | ---: |
| Historical | 0.482226 | 0.484837 | +0.54% |
| Dense | 0.347605 | 0.393629 | +13.24% |
| Cosmos | 0.316814 | 0.322648 | +1.84% |

Six dimensions is the smallest tested size whose dataset-average MAE does
not exceed native in all three datasets. That is a descriptive grid result,
not a validated cutoff: the curves are non-monotonic, budgets/agents can
regress, and repeated orders reuse the same labels. Historical also has an
83.3% native mean/prior fallback share of unselected target occurrences,
so its headline average is not an IDW-only quality measure.

## Initial IDW threshold/envelope follow-up (local evidence)

The initial fresh-start verified threshold run, retained locally at
`outputs_matryoshka/runs/idw-threshold-envelope-20260915-verified/`,
replays 4,500 native/8d end-to-end settings with exact baseline MAE/accuracy
parity. ROC compares recall with false-positive rate; accuracy, precision,
recall and F1 are separate threshold curves. Point and lower-envelope methods
share 3,744,223 eligible repeated target occurrences; 420,977 mean/prior fallback
occurrences have no claimed envelope and are reported separately.

At threshold 0.5, native lower-envelope classification raises equal-cell mean
precision from 66.09% to 70.02%, but lowers recall from 70.52% to 40.13% and
F1 from 65.54% to 41.85%. Mean eligible-cell AUROC falls from 0.644 to 0.599;
8d shows the same broad tradeoff. These are not independent-session or
production-population estimates.

The envelope uses the empirical 90th percentile of earlier same-agent selected
label slopes in normalized angular geometry, with a distance floor and
explicit sparse-calibration fallback. It is a conditional sensitivity
construction, not calibrated confidence. This separate prefix-space experiment
does not validate the integrated value pipeline, PCA-V7 geometry or weekly
policy. See the [scientific record](IDW_THRESHOLD_EXPERIMENT.md) for precision,
calibration, eligibility and reproducibility boundaries.

## Refreshed Cosmos evidence

The refreshed [dimension report](../outputs_matryoshka/reports/mrl-cosmos-refresh-20260915/report.html)
uses the September 15 export with catch-up cutoff 13:40:42 UTC and **755**
expected-label units across seven agents. Historical and Dense are unchanged.
All 51,000 Historical/Dense cells and memberships reproduce the earlier
low-dimensional study exactly; the changed Cosmos population is deliberately
excluded from that same-input parity assertion.

For the refreshed Cosmos cohort, native versus 2d end-to-end unselected-only
MAE is **0.276282 versus 0.293814** (+6.35% relative), and accuracy at 0.5 falls
from **74.92% to 70.64%**. Mean MAE at 6d is **0.271500** and at 8d is
**0.273882**. Six dimensions remains the smallest tested size with no increase
in dataset-average MAE across all three datasets; this is not a validated
cutoff or a guarantee for every budget/agent.

The [threshold/envelope follow-up](../outputs_matryoshka/reports/idw-threshold-cosmos-refresh-20260915/report.html)
uses the same refreshed inputs and verified native/8d reference memberships.
It completes 4,500 settings with 4,492,885 matched eligible repeated target
occurrences and 434,915 ordinary fallback occurrences. Across equal-weight
eligible replay cells, native point versus lower-envelope AUROC is
**0.656 versus 0.578**. At threshold 0.5, mean precision rises from
**66.93% to 73.51%**, while mean recall falls from **66.05% to 32.16%**.
Dataset/budget-specific behavior can differ from these cell averages.
Only aggregate reports are published. Original identifiers and per-target
evidence stay in protected local storage. Snapshot documents are not the
evaluation denominator; alternate label containers are not silently combined.

## Completed IMDb sentiment follow-up

The [IMDb report](../outputs_imdb/reports/imdb-40-replay/report.html) evaluates
50,000 public movie reviews as one agent, with sentiment labels replacing
selected judge observations. All vectors are real native Azure
`text-embedding-3-small` embeddings; 1536, 256, 128, 64, 32, 24, 16, 12 and 8 dimensions use
prefix slicing and re-normalization, not PCA. Forty paired bootstrap seeds,
two arrival schedules and five label budgets produce 3,600 completed cells.
The 256/128/64 extension adds 1,200 cells using the exact recorded source draws
and arrivals, preserving all 2,400 original rows and making no additional
embedding calls. Three workers completed the extension after preserving
465 already-completed added cells; numerical methods and bindings did not change.
Unlike the earlier unique-session permutations, this replay also changes
source-review frequency.

At 5% label budget, native/32d/8d unselected MAE is 0.2054/0.2449/0.3226,
and accuracy is 89.60%/83.03%/74.37%. The added 256/128/64 MAEs at 5% are
0.2242/0.2358/0.3028, with accuracy 87.91%/87.09%/80.35%.
Native has the lowest mean MAE at four of five budget averages. At 20%,
128d mean MAE is lower (0.1645 versus 0.1703), but accuracy is lower too
(90.55% versus 91.35%); its paired MAE replay range crosses zero.
Prefix curves are not globally monotonic, and an MAE improvement is not
automatically a classification improvement.
The native lower-envelope classifier at threshold 0.5 raises eligible precision
from 87.93% to 99.17% while reducing recall from 91.50% to 3.04%;
AUROC changes from 0.9605 to 0.7580. Most lower scores are clipped to zero,
and average full-envelope width is 0.9094, so high label coverage does not
establish informative or calibrated bounds.

Selection remains label-blind full-schedule ranking, not online admission.
IDW and calibration use only earlier selected observations. Scaling to 50K
uses blocked float64 distances and a 128-donor causal calibration reservoir,
not the prior small-corpus all-pair estimator. The report separates novel-source
targets from repeated-review reuse. Train/test reviews are combined, so these
are neither held-out IMDb benchmark results nor agent-task-completion evidence.
Independent retained-score checks cover every cell, verify all source/artifact
hashes and reproduce the requested metrics with zero discrepancy.

## Reporting contract

Use [the reporting skill](../.github/skills/sampling-experiment-report/SKILL.md)
for HTML/PDF work. Every report should identify the decision, explain the methods
visually and in plain language, disclose the exact data and label provenance,
show comparable results with denominators and uncertainty meaning, and conclude
with evidence-backed analysis and limitations. Do not write the conclusion first
and select only supporting results.

Use the existing generators and preserve run provenance. Browser QA and actual
PDF export are separate from scientific validation. See
[repository layout and retention](REPOSITORY_LAYOUT.md) for generated evidence,
archival cleanup, and compatibility-preserving organization.