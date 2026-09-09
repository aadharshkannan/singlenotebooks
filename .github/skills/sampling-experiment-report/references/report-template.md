# Report Template

Use this structure for any experiment generation; replace placeholders with
verified values. Missing evidence is "not measured" or "unavailable", never zero.

## 1. Decision and Source

Title: `[Experiment/version]: [literal comparison]`.
Identify audience, decision question, primary success measure, run ID/date,
aggregate/manifest paths, source revision/hashes when available, and report status.
Give one qualified result summary, written after analysis, with the main tradeoff.

## 2. How the Methods Work

For **every** tested method, provide:

| Method ID / readable label | Representation | Selection rule and scope | Score estimator | Budget unit | Assumption / failure mode |
| --- | --- | --- | --- | --- | --- |
| `[from run]` | `[actual input]` | `[actual rule]` | `[actual estimator]` | `[sessions/tokens/etc.]` | `[specific limitation]` |

Draw a compact pipeline: session evidence -> representation -> selection ->
observed labels/judgments -> imputation -> agent/population summary. Mark what
is shared and what differs between arms. Distinguish lexical MinHash novelty,
semantic embedding distance, random selection, and weighted estimation. Render
diagrams into standalone HTML; do not leave raw Mermaid syntax in the deliverable.

For full-session/PCA/IDW, explain evidence ordering and truncation, embedding model,
original/reduced dimensions, reducer fit scope, distance metric, neighbor count,
same-agent rules, exact matches, missing donors, threshold, and aggregation.

**Illustrative IDW calculation, not a run result:** an unjudged target has two
donors at distances 1 and 2 with binary labels 1 and 0. Ignoring epsilon for this
example, inverse-square weights are 1 and 1/4; normalized weights are 0.8 and 0.2.
The estimated probability is 0.8; applying threshold 0.5 gives binary prediction 1.
Show the distinction between that model estimate and a direct judgment. For the
actual experiment, state epsilon and exact-match handling rather than borrowing
the illustration's simplified rule.

## 3. Data and Experiment Design

| Dataset | Origin | Eligible sessions / agents | Label source | Representation fallback | Included/excluded records |
| --- | --- | --- | --- | --- | --- |
| `[dataset ID]` | `[synthetic/export/etc.]` | `[N / agents]` | `[expected/human/judge+rubric]` | `[counts]` | `[counts and reasons]` |

Include source locations/hashes, domains, observed concept definition, lengths,
pass-rate reference and class imbalance, preprocessing/deduplication/missing-data
rules, and external-input availability. "Linked spans" does not establish real
production representativeness. For live judges disclose model, rubric/version,
retry/error treatment and whether repeated labels were independently collected.

State budget units and absolute caps per corpus; seeds/replays and their pairing;
time-forward versus transductive fitting; label-blind selection boundary; and
fixed-source versus replay-relative targets. Show selected, actually judged,
unjudged/imputed, prior/fallback and unavailable counts without conflating them.

## 4. Visual Results

Choose one clear view per question, consistently labeled across methods:

- Coverage/cost: budget-response curves or a paired coverage/cost panel. Name the
	concept proxy and denominator; show per-agent coverage or mark it unavailable.
- Aggregate error: dataset-by-budget small multiples or a compact heatmap. State
	fraction versus percentage-point units and whether lower is better.
- Imputation: unjudged-only MAE/Brier or accuracy/F1; confusion matrix where useful.
	Keep judged-plus-imputed metrics separate and identify observed-label inflation.
- Distance comparisons: paired deltas with zero reference and uncertainty meaning;
	distinguish win frequency from effect size and include ties.
- Resource costs: selected sessions, tokens, model calls, cache hits, latency and
	memory only where measured. Separate projected savings from observed costs.

Each visual includes **What it shows**, **How to read it**, and **Takeaway** in
short domain-specific sentences. Use consistent method colors, accessible contrast,
shared comparable axes, explicit units/n, and legends that work without color alone.
No 3D chart when a 2D plot answers the question. Avoid unexplained metric walls.
Do not chart missing data as zero or silently change chart scope through filters.

## 5. Analysis and Limits

Explain the mechanism consistent with the observed results, without claiming an
untested cause. Show tradeoffs, counterexamples, weak cohorts, sample variability,
fallback behavior, and whether a gain matters for the decision. Disclose how
pooled summaries are weighted; do not substitute them for per-agent guarantees.

Separate **measured**, **interpretation**, and **proposed** claims. Name the next
discriminating experiment (for example time-forward PCA, actual judge calibration,
new metrics, rare concepts, reduced-space bounds, or the weekly policy gate).

## 6. Reproducibility and Validation

Include source artifact paths, exact generation command, environment, code revision,
artifact hashes and known provenance gaps. Include focused test and browser results,
viewport sizes, print format, output links, and manual screenshot/PDF inspection.
Keep the narrative and numeric claims identical in HTML and PDF; print must include
all required scopes rather than only the current interactive filter selection.
