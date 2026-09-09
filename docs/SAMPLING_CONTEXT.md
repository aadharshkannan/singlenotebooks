# Agent Evaluation Sampling Context

Updated: 2026-09-07. This is the repo's working context, combining the team's
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