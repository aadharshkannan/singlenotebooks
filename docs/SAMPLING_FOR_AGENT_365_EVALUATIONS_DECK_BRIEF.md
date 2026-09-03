# Sampling for Agent 365 Evaluations

## CoWork presentation brief

Create a polished 16:9 presentation with exactly 12 slides. The audience is non-technical business and product stakeholders. The story should move from the scale problem, through the experiments and production decision, to the research path that makes richer semantic sampling practical.

### Presentation style

- Use a clean Microsoft enterprise visual language: white or very light neutral background, dark charcoal text, Agent 365 teal/blue accents, and one warm highlight color for decisions.
- Keep each slide to one message. Prefer diagrams, point clouds, funnels, and direct-label charts over paragraphs, legends, or dense tables.
- Use short headlines that state the conclusion, not generic section labels.
- Use progressive visual storytelling: reveal the scale problem, compare the options, show the evidence, then show the path forward.
- Keep technical terms off the main canvas unless they are immediately translated into plain language.
- Do not show file paths, citations, caveats, or formulas as body copy unless this brief explicitly places them on the slide. Put supporting detail in speaker notes.
- Recreate charts from the supplied data so they use one consistent visual style. Do not paste screenshots of tables.

### Visual system

- **Random Sampling:** blue
- **MinHash LSH:** amber
- **Full-Session Embedding:** teal
- **IDW estimated scores:** blue with a hollow treatment
- **LLM-as-a-Judge calls / cost pressure:** coral
- **Observed judge score:** solid dot
- **Imputed score:** hollow dot

---

## Slide 1 — Sampling for Agent 365 Evaluations

### What goes on the slide

**Sampling for Agent 365 Evaluations**

*Evaluate every agent without sending every session to an LLM judge*

### Visual direction

Use a sparse opening visual: many Agent 365 session traces flow toward a much smaller set of highlighted evaluation calls, while a scorecard still appears for the full population. Make the title the dominant element.

### Speaker notes

Agent 365 provides an enterprise control plane for observing, governing, and securing agents. Evaluation turns that observability into an answer to a business question: are agents completing work successfully and behaving as intended?

The challenge is scale. We want evaluation coverage across every agent and every session, but LLM judges have real cost and capacity constraints. This presentation explains how sampling lets us control those constraints, what our experiments taught us, what is implemented today, and the path toward a more intelligent approach.

---

## Slide 2 — The ambition multiplies faster than capacity

### What goes on the slide

**Every session. Every agent. Every evaluation dimension.**

`Agent sessions × Evaluators = LLM judge calls`

**1 evaluator today → 8 evaluation dimensions planned**

Bottom-line statement:

**Sampling turns an unbounded evaluation bill into a controlled capacity decision.**

### Visual direction

Build a multiplication visual rather than a conventional chart:

1. A stream labeled **Every session from every agent**.
2. One current evaluator tile labeled **Task Completion**.
3. That tile expands into eight evaluator tiles: **Task Completion**, **Intent Resolution**, **Task Adherence**, **Tool Call Accuracy**, **Tool Call Success**, **Tool Input Accuracy**, **Tool Output Utilization**, and **Tool Selection**.
4. Show the resulting LLM-call stream rapidly widening, then passing through a **Sampling** valve that brings it inside a fixed **Cost + LLM capacity** container.

Avoid fabricated dollar amounts or traffic volumes.

### Speaker notes

Our goal is to bring LLM-as-a-Judge evaluation to Agent 365 for all agents. We are starting with Task Completion and expect to expand across seven additional dimensions: Intent Resolution, Task Adherence, Tool Call Accuracy, Tool Call Success, Tool Input Accuracy, Tool Output Utilization, and Tool Selection.

Without sampling, each added evaluator multiplies the work: one LLM call per session, per evaluator. Even before attaching a dollar amount, this creates two constraints: what we can afford and how much LLM throughput is available. Sampling is therefore not merely an optimization. It is the control that makes broad evaluation operationally possible.

---

## Slide 3 — Two simple ways to choose what gets judged

### What goes on the slide

Use two equal visual panels with very little text.

**Random Sampling**

- Every session gets an equal chance
- Simple, predictable, defensible

**MinHash LSH**

- Give each session a compact content fingerprint
- Favor fingerprints we have seen less often

Footer:

**Random represents fairly. MinHash searches for lexical novelty.**

### Visual direction

**Left panel — Random:** Show 20 mixed session cards entering a transparent tumbler. Five cards emerge, preserving the overall mix. Use a small equal-chance icon on every card.

**Right panel — MinHash LSH:** Use a two-stage visual. First, show three recognizable session phrases becoming compact fingerprints. Two phrases share highlighted words such as "reset" and "password" and receive nearly matching fingerprints; a different phrase receives a visibly different fingerprint. Second, show the two similar fingerprints entering one **Similar wording** neighborhood while the unmatched fingerprint lands in **No close match** and is highlighted **Prioritize**. This should make the lexical nature of MinHash visible before showing the LSH grouping result.

Do not show hash formulas, permutations, bands, or rows on the slide.

### Speaker notes

Random sampling is the benchmark and the simplest production option. It gives every eligible session the same chance of selection. Over a sufficiently large population, the sample should resemble the whole.

MinHash LSH is a fast way to recognize sessions that use similar words, phrases, and tool patterns. Each session becomes a compact fingerprint. Similar fingerprints are likely to land in the same buckets, while a bucket miss signals something less familiar. That lets the sampler prioritize lexical novelty without comparing every session with every other session.

MinHash does not understand meaning in the way an embedding model does. Two sessions can use different words for the same task, or similar words for different tasks. That limitation motivated the semantic approach on the next slide.

---

## Slide 4 — Map meaning, sample broadly, estimate locally

### What goes on the slide

Use a three-step sequence:

1. **Map** — A whole session becomes one point based on meaning
2. **Sample** — Judge points that broaden behavioral coverage
3. **Estimate** — Nearby judged points inform unjudged points

Small labels:

- **Embedding = the semantic map**
- **IDW = closer evidence counts more**

### Visual direction

Create a large two-dimensional embedding point cloud with several behavioral clusters. Use an irregular, realistic distribution rather than evenly spaced decorative dots: overlapping dense clusters, sparse bridges, small detached groups, and a few isolated outliers.

- Dense central clusters represent common behavior.
- Small detached clusters and isolated points represent rare or novel sessions.
- Ring a subset of points to show sessions selected for LLM judging, including points from the rare regions.
- In a magnified cluster, show one hollow unjudged point connected to several solid judged neighbors. Use thicker lines for closer neighbors and thinner lines for more distant neighbors.
- In the HTML deck, let points drift only 1–3 pixels on slow, seeded paths, with a gentle pulse around selected outliers. Motion must preserve cluster membership, stop under reduced-motion preferences, and be disabled in print.

The point cloud should dominate the slide. Avoid mathematical notation.

### Speaker notes

A full-session embedding converts the entire interaction — the request, response, and tool-use context — into a numerical representation of meaning. Sessions with similar behavior appear near one another; semantically unusual sessions appear farther away.

The embedding selector uses that map to seek broader semantic coverage. This is the selection step.

Inverse Distance Weighting, or IDW, is a separate estimation step. After selected sessions receive observed LLM-judge scores, an unjudged session can borrow information from nearby judged sessions from the same agent. Closer neighbors receive more weight. The result is a model-assisted estimate, not a new LLM judgment.

---

## Slide 5 — We tested against a known answer

### What goes on the slide

Use four large numeric callouts:

- **2,800** eval-ready sessions
- **105** agents
- **480** behavior concepts
- **5** matched capacity levels

Then one short line:

**Expected labels were hidden from selection and used only to score the result.**

### Visual direction

Show a population map composed of two clearly labeled groups:

- **Dense agents:** 5 agents × 500 sessions = 2,500 sessions
- **Broad agent mix:** 100 agents × 300 sessions total

Around the map, use small icons to show what each record contained:

- OpenTelemetry session trace
- Expected pass/fail label
- Behavior concept
- Difficulty level
- Token cost

Use a closed-eye or sealed-envelope treatment around the expected labels to show they were not visible to the selectors.

### Speaker notes

The v4 experiment used a controlled, labeled population of 2,800 normalized Agent 365-style sessions across 105 agents. Five dense agents contributed 500 sessions each, giving us super-agent-like populations with enough behavioral variety to stress the samplers. A broader 100-agent corpus contributed another 300 sessions.

Each evaluation unit contained a structured OpenTelemetry trace, an expected pass/fail outcome, concept metadata, difficulty, and a token cost. A concept was defined by the combination of corpus, domain, task, and difficulty, producing 480 distinct concepts.

The expected outcome and concept metadata were not used to choose sessions. They were revealed only after selection to calculate performance. This is what lets the experiment compare the sample against a known full-population answer.

The population pass rate was 71.18%. The five matched token budgets were 65,949; 131,898; 263,797; 395,695; and 659,492 tokens, corresponding to the legacy 5%, 10%, 20%, 30%, and 50% tiers.

---

## Slide 6 — Success means accuracy and breadth

### What goes on the slide

Use two side-by-side metric cards.

**Outcome error (MAE)**

*How close is the sampled pass rate to the full-population pass rate?*

**Lower is better**

Illustrative example: `|69% − 72%| = 3 percentage points`

**Concept coverage**

*How much of the population's behavioral variety did the sample touch?*

**Higher is better**

Illustrative example: `120 concepts ÷ 480 concepts = 25%`

Footer:

**Same sessions, same whole-session token budgets, repeated deterministic replays.**

### Visual direction

For outcome error, show two nearly aligned thermometers or gauges labeled **Full population** and **Sample**, with the gap highlighted.

For concept coverage, show a 12-color behavior palette: the population has all 12 colors and the sample has a subset. Avoid a Venn diagram.

### Speaker notes

We measured two complementary questions.

First, selected-only outcome-rate MAE is the absolute gap between the pass rate in the selected sample and the known pass rate of all eligible sessions. It is reported in percentage points. A smaller gap means the sample reproduces the population-level outcome more accurately.

Second, concept coverage is the number of distinct behavior concepts touched by the selected sessions divided by the 480 concepts in the full population. A higher value means the sample exposes the judge to more of the system's behavioral variety.

These metrics can move differently. A sample can estimate the overall pass rate accurately while missing rare behaviors. Conversely, a novelty-seeking sample can cover many behaviors while distorting the headline pass rate. That is why we evaluated both.

The comparison preserved whole sessions and used the same exact token budgets for every method. Selection used no outcome labels.

---

## Slide 7 — Semantics won coverage; simplicity won the first release

### What goes on the slide

Headline callout:

**At 50% capacity, semantic sampling covered 87.9% of concepts — 19.4 points more than Random.**

Subordinate label:

**Coverage view shown; Random had lower headline pass-rate error at 4 of 5 budgets.**

Decision ribbon:

**Now: deterministic Random Sampling**  →  **Next: Full-Session Embedding + IDW**

Use the exact data below to create the chart. **The table is source data, not an on-slide element.**

| Capacity tier | Random | MinHash LSH | Full-Session Embedding |
|---:|---:|---:|---:|
| 5% | 23.2% | 21.8% | 19.4% |
| 10% | 35.4% | 35.9% | 34.8% |
| 20% | 48.3% | 50.5% | 65.1% |
| 30% | 55.4% | 59.1% | 78.7% |
| 50% | 68.5% | 75.0% | 87.9% |

### Visual direction

Create a direct-label three-line chart:

- X-axis: **Evaluation capacity used**
- Y-axis: **Behavior concepts covered**
- Directly label line endpoints; do not use a detached legend.
- Add a subtle shaded region from 20% to 50% with the annotation **Semantic coverage separates**.
- Under the chart, show the decision ribbon. Random should look ready and operational; Embedding + IDW should look like the next-stage research path, not a failed option.

### Speaker notes

Random and MinHash produced similar concept coverage at lower budgets. At the 5% tier Random led; at 10% MinHash led by less than one percentage point.

From the 20% tier onward, full-session embedding separated clearly. At 30% capacity it covered 78.7% of concepts versus 55.4% for Random. At 50%, it reached 87.9% versus 68.5% for Random and 75.0% for MinHash.

Coverage was not the only result. On selected-only outcome-rate error, Random won four of five budget cells and MinHash won one. The initial IDW-augmented population estimate also did not beat the simpler selected-only estimate in this v4 study. Those results reinforce a staged decision rather than an immediate semantic rollout.

Random was far simpler to ship, explain, and operate. We therefore implemented deterministic Random Sampling as an immediate capacity control while continuing the work needed to make semantic selection and model-assisted IDW estimates production-ready.

The chart reports embedding selection coverage. IDW does not change which concepts were selected; it estimates outcomes after selection.

### Evidence / asset

Use the authoritative Azure-embedding values in:

- `outputs_sampling_v4/runs/full-20260805/COVERAGE_TUNING_EXPERIMENT.md`
- `outputs_sampling_v4/runs/full-20260805/agent365-sampling-v4-summary-report.html`

Do not substitute the later deterministic reconstructed-embedding tuning values; they are not directly comparable.

---

## Slide 8 — The same session gets the same answer everywhere

### What goes on the slide

**Production containers do not share local memory. Sampling does not ask them to.**

Use this four-step flow:

1. **Session identity** — Agent ID + Session ID
2. **Stable hash** — SHA-256
3. **Bucket** — 0 to 99
4. **Decision** — Include when bucket is below the configured percentage

Bottom-line statement:

**Same session + same rate = same decision in every container and retry**

Small footer:

**Then apply per-agent minimums and run/capacity limits.**

### Visual direction

Show three separate production-container icons receiving the same session identity. Each independently computes the same bucket and reaches the same include/exclude result. Cross out a central in-memory cache and a Redis coordination path; the message is that neither is required for the sampling decision.

To the right, show selected sessions fan out to the configured evaluators.

### Speaker notes

The current External3p session-completion path makes a deterministic percentage decision. It normalizes the agent and session identities, hashes them with SHA-256, maps the result into one of 100 buckets, and includes the session when its bucket is below the configured percentage.

This matters in production because each container has separate memory and requests or retries can reach different containers. An in-memory cache cannot coordinate them reliably. A shared Redis cache would add state, network calls, locking, and failure modes. Deterministic hashing lets each container recompute the same decision independently.

The final number of sessions can differ from the nominal percentage because production guardrails run after the percentage decision: a per-agent minimum can add sessions, while per-run and throughput limits can constrain them. When a positive TPM budget governs session mode, percentage sampling is intentionally bypassed in favor of capacity scheduling.

The cross-container statement applies to the default SHA-256 session-completion path. A legacy turn-level path uses process-seeded `.NET HashCode.Combine`; it is repeatable inside one process but is not guaranteed to make the same decision after a restart or in another container.

### Evidence / source

- `C:\Users\stangoodwin\BIC-Evaluations-Service\src\BicEvalsService.BusinessLogic\Services\Jobs\BackgroundJobs\SessionCompletionSelector.cs`
- `C:\Users\stangoodwin\BIC-Evaluations-Service\src\BicEvalsService.BusinessLogic\Services\Jobs\BackgroundJobs\ExternalEvaluationParentJob.cs`
- `C:\Users\stangoodwin\BIC-Evaluations-Service\src\BicEvals.Tests\Services\Jobs\ExternalEvaluationParentJobExecutionTests.cs`

---

## Slide 9 — Judge the evidence-rich sample; score the full picture

### What goes on the slide

**Semantic selection helps find:**

- Novel and rare sessions
- Unique and long-tail tasks
- Hidden regions inside super agents

**IDW extends the result:**

- Observed score for judged sessions
- Model-assisted estimate for every eligible unjudged session

Bottom-line statement:

**Broader discovery + population-wide estimated scores**

### Visual direction

Show a wide semantic map for a super agent with many distinct task clusters.

- Select and ring points across common and rare clusters.
- Send only those points to an LLM judge.
- Return solid score dots to judged points.
- Spread hollow estimated score dots to the remaining points using weighted links to nearby judged examples.
- Add a tiny provenance key: **solid = judged**, **hollow = estimated**.
- Use uneven cluster sizes, overlapping task regions, sparse bridges, detached long-tail groups, and isolated outliers so the map resembles a real projected embedding population. Apply the same restrained, reduced-motion-aware drift used on slide 4.

### Speaker notes

Random sampling is a strong baseline for estimating an overall rate, but it spends most of its budget where most traffic already exists. Semantic selection is designed to seek broader coverage, making it attractive for rare, novel, and unique sessions.

That matters especially for super agents. An agent that handles many kinds of work can have a healthy average while failing in a small but important task region. A semantic map makes those regions visible and gives the sampler a way to allocate judge capacity beyond the traffic center.

IDW then uses the judged sessions as same-agent donors for unjudged sessions. Instead of leaving the majority of sessions blank, it can assign every eligible session a model-assisted probability of passing. This supports richer population views and expected-value calculations without pretending every estimate was directly observed.

The production contract should always retain provenance: judged versus estimated, the donors used, representation version, and an uncertainty signal. The initial v4 IDW result was mixed, so this remains a validated research direction rather than a claim of production accuracy.

---

## Slide 10 — Confidence depends on how far we reached

### What goes on the slide

**Nearby evidence → narrower range**

**Distant evidence → wider range**

Plain-language formula:

`Estimated score ± (behavior sensitivity × weighted neighbor distance)`

Worked example:

**Estimated pass probability: 65%**

**Conditional range: 55%–75%**

Required footer:

**A conditional geometric sensitivity range — not a statistical confidence interval**

### Visual direction

Use two mirrored mini-scenes.

- **Left:** an unjudged point surrounded by close judged neighbors; show a narrow horizontal range.
- **Right:** an unjudged point borrowing from distant neighbors; show a wide horizontal range.

Between them, use a single horizontal ruler labeled **Distance across the behavior map**. Keep the mathematical formula in speaker notes, not on the main canvas.

### Speaker notes

An IDW estimate alone does not tell us whether it was supported by nearly identical sessions or whether it had to borrow across a sparse region. A Lipschitz bound turns that distance into a confidence signal.

The method assumes that the expected evaluation score changes smoothly across the semantic map. The Lipschitz value, $L$, describes how quickly that score can change per unit of angular distance. For an unjudged session, we calculate the weighted average angle to its judged donors and use:

$$
u_{\mathrm{Lip}} = L\bar{\theta}
$$

The displayed range is the IDW estimate plus or minus that allowance, clamped to 0–1. In the illustrated example, an estimate of 0.65, weighted angle of 0.20, and $L=0.50$ produce an allowance of 0.10 and a range of 0.55–0.75.

This is conditional model-assisted uncertainty. It measures geometric sensitivity under the smoothness assumption. It is not a 95% confidence interval and does not cover evaluator bias, embedding drift, binary outcome noise, or selection bias. Those need separate controls and calibration.

### Evidence / source

- `C:\Users\stangoodwin\lipschitz\MATH.md`
- `C:\Users\stangoodwin\lipschitz\src\lipschitz_uncertainty\core.py`

---

## Slide 11 — A science experiment attacked the storage blocker

### What goes on the slide

**Engineering blocker**

- Native embedding: **1,536 values per session**
- Embeddings are customer data
- Similarity search appeared to require a compliant vector store

**Research question**

**How small can the representation become without harming the end-to-end result?**

Methods tested:

**PCA · SVD · GRP · Random coordinate mean**

Study design:

**2,500 sessions · 30 replay seeds · 5 traffic patterns · 5 judge budgets**

### Visual direction

Create a compression-lab visual:

`1,536 dimensions` → `1,024` → `512` → `128` → `32` → `16` → `8` → `2`

Show four labeled scientific instruments or lanes for PCA, SVD, GRP, and Random Coordinate Mean feeding the same end-to-end test rig:

**reduce → select → impute with IDW → compare with expected labels**

Do not attempt to draw 1,536 literal bars. Use a dense matrix that visibly compresses into eight clear components.

### Speaker notes

The semantic approach introduced an engineering blocker. Native `text-embedding-3-small` representations contain 1,536 values per session. Because embeddings are customer data, storage must meet the corresponding compliance requirements. The initial architecture assumed a vector-capable store that could retain these representations and run cosine-similarity operations.

We investigated whether dimensionality reduction could remove that dependency without damaging the actual sampling-and-imputation result. The experiment tested Principal Component Analysis (PCA), Truncated Singular Value Decomposition (SVD), Gaussian Random Projection (GRP), and a low-cost random-coordinate-mean baseline across dimensions from 1,024 down to 2. A 2D t-SNE fixed-membership diagnostic was also explored but is not deployable and should not be shown as a production candidate.

The intended v6 study used 2,500 synthetic sessions from five dense agents and 30 replay seeds. It exercised five traffic schedules and five judge-budget levels from 1% to 20%, producing 54,750 raw result rows. The reducer itself was fitted once, with seed 13, on the full unlabeled population; the 30 replay seeds varied the downstream sampling, schedule, and IDW behavior rather than refitting the reducer.

Crucially, the experiment did not judge a representation only by how well it reconstructed the original vector. It tested the end-to-end behavior we care about: selection, IDW imputation, and error against known expected labels.

---

## Slide 12 — Eight dimensions can unlock the semantic path

### What goes on the slide

Use three dominant callouts:

**8 dimensions**

**23% lower end-to-end MAE**

**~119× smaller stored representation**

Create the visual comparison from the data below. **The table is source data, not an on-slide element.**

| Representation | End-to-end MAE |
|---|---:|
| Native 1,536D | 0.3476 |
| PCA 8D | 0.2682 |

Decision statement:

**Proposed engineering unlock: store eight components in ESP and compute similarity directly — no dedicated vector-store dependency.**

Small qualifier:

**Synthetic-label research result; confirm reducer choice, ESP schema/compliance, and production-scale performance.**

### Visual direction

Use a presentation-scaled version of the report's **actual imputation error by representation dimension** chart:

- Plot end-to-end mean MAE for PCA, SVD, GRP, and Random Coordinate Mean across the tested dimensions.
- Add a dashed native 1,536D reference at MAE 0.3476.
- Clearly highlight PCA-8 as the minimum at MAE 0.2682.
- Use direct axes and a compact legend. Omit the 95% bootstrap intervals at slide scale and state that the full report retains them.
- Keep the three dominant callouts and the proposed ESP engineering unlock visible beside and below the chart.

### Speaker notes

The dimensionality sweep produced a counterintuitive result. Across the intended 30-seed v6 run, PCA at eight dimensions achieved end-to-end MAE of 0.2682 compared with 0.3476 for the native 1,536-dimensional embedding. That is a 0.0794 absolute reduction, or approximately 23% lower error. PCA-8 also achieved end-to-end F1 of 0.8589, a 0.0396 improvement over native. The modeled stored-representation comparison was 0.129 MB versus 15.36 MB in the experiment, about 119 times smaller; this was not a measurement of total process memory.

PCA-8 was the leading deployable candidate within this artifact. Lower-dimensional representations were at least as accurate here, which is consistent with a useful denoising effect. However, this was a synthetic, transductive study using one fitted reducer realization, not a production accuracy measurement. The robust conclusion is that **eight dimensions are highly promising**; production data and an out-of-sample reducer validation are still required.

The lower-dimensional geometry materially changed which sessions were selected: PCA-8 and native selection had only 18.9% overlap in the end-to-end comparison. That is not automatically bad — the resulting aggregate MAE and F1 improved — but it means a production rollout must validate the selected sessions themselves, not only the final average metrics.

Eight scalar components create a plausible engineering path to store the representation alongside the session in ESP and calculate cosine similarity directly, removing the need for a dedicated vector index. The experiment proves the accuracy and memory result; ESP schema fit, compliance approval, query ergonomics, and production-scale performance are engineering validations still to complete.

Do not claim that the experiment measured a speedup. It found no persuasive runtime separation, and reducer-fit time was outside the reported evaluation runtime.

### Evidence / assets

- `C:\Users\stangoodwin\singlenotebooks-idw-dimensionality-sweep\outputs_sampling_v6\runs\idw-dimensionality-full-30-seed-20260827\idw-dimensionality-report.html`
- `C:\Users\stangoodwin\singlenotebooks-idw-dimensionality-sweep\outputs_sampling_v6\runs\idw-dimensionality-full-30-seed-20260827\idw-dimensionality-report.pdf`
- `C:\Users\stangoodwin\singlenotebooks-idw-dimensionality-sweep\outputs_sampling_v6\runs\idw-dimensionality-full-30-seed-20260827\screenshots\groundtruth-final-desktop-fig1.png`
- `C:\Users\stangoodwin\singlenotebooks-idw-dimensionality-sweep\outputs_sampling_v6\runs\idw-dimensionality-full-30-seed-20260827\screenshots\groundtruth-final-desktop-fig4.png`
- `C:\Users\stangoodwin\singlenotebooks-idw-dimensionality-sweep\outputs_sampling_v6\runs\idw-dimensionality-full-30-seed-20260827\screenshots\groundtruth-final-desktop-fig5.png`

---

## CoWork chart data and artifact handoff

### Slide 7 — concept coverage CSV

```csv
capacity_tier,random,minhash_lsh,full_session_embedding
5,23.19,21.81,19.38
10,35.42,35.90,34.79
20,48.33,50.49,65.07
30,55.42,59.10,78.68
50,68.54,75.00,87.92
```

### Slide 12 — dimensionality headline CSV

```csv
representation,dimensions,end_to_end_mae,representation_memory_mb
native,1536,0.3476,15.360
pca,8,0.2682,0.129
```

### Recommended source artifacts to provide with this brief

1. `outputs_sampling_v4/runs/full-20260805/agent365-sampling-v4-summary-report.html`
2. `outputs_sampling_v4/runs/full-20260805/agent365-sampling-v4-summary-report.pdf`
3. `outputs_sampling_v4/runs/full-20260805/COVERAGE_TUNING_EXPERIMENT.md`
4. `C:\Users\stangoodwin\singlenotebooks-idw-dimensionality-sweep\outputs_sampling_v6\runs\idw-dimensionality-full-30-seed-20260827\idw-dimensionality-report.html`
5. `C:\Users\stangoodwin\singlenotebooks-idw-dimensionality-sweep\outputs_sampling_v6\runs\idw-dimensionality-full-30-seed-20260827\idw-dimensionality-report.pdf`
6. `C:\Users\stangoodwin\singlenotebooks-idw-dimensionality-sweep\outputs_sampling_v6\runs\idw-dimensionality-full-30-seed-20260827\screenshots\groundtruth-final-desktop-fig1.png`
7. `C:\Users\stangoodwin\singlenotebooks-idw-dimensionality-sweep\outputs_sampling_v6\runs\idw-dimensionality-full-30-seed-20260827\screenshots\groundtruth-final-desktop-fig4.png`
8. `C:\Users\stangoodwin\singlenotebooks-idw-dimensionality-sweep\outputs_sampling_v6\runs\idw-dimensionality-full-30-seed-20260827\screenshots\groundtruth-final-desktop-fig5.png`

### Accuracy guardrails for deck generation

- Do not say IDW caused the concept-coverage lift. Embedding selection caused the lift; IDW estimates outcomes after selection.
- Do not label the Lipschitz range as a confidence interval, margin of error, or 95% interval.
- Do not imply that an IDW estimate is an observed LLM-judge score.
- Do not imply embedding sampling won every budget. Random won at 5%, MinHash at 10%, and embedding at 20% and above.
- Do not claim PCA-8 is universally optimal. It led this synthetic, transductive 30-seed replay artifact; production and out-of-sample reducer validation remain necessary.
- Do not claim a measured runtime improvement from dimensionality reduction.
- Describe direct ESP storage as the proposed engineering unlock pending schema, compliance, and production validation, not as an already deployed capability.
- Keep the default SHA-256 session-completion implementation separate from the legacy turn-level hash path when discussing cross-container determinism.

### Internal evidence provenance

- Agent 365 platform and evaluation context: `C:\Users\stangoodwin\the-vault\wiki\concepts\agent-365-platform.md`, `concepts\observability.md`, `topics\agent-evaluation.md`, and `topics\agent-evaluation-sampling.md`.
- Sampling statistics and architecture context: `C:\Users\stangoodwin\the-vault\wiki\concepts\evaluation-sampling-statistics.md`, `projects\a365-span-sampling.md`, and `projects\eval-sampling-poc.md`.
- V4 experiment design and results: `outputs_sampling_v4/runs/full-20260805/` and `sampling_comparison/v4_idw.py`.
- Production External3p sampling: `C:\Users\stangoodwin\BIC-Evaluations-Service\src\BicEvalsService.BusinessLogic\Services\Jobs\BackgroundJobs\` and `Services\Sampling\`.
- Dimensionality research: `C:\Users\stangoodwin\singlenotebooks-idw-dimensionality-sweep\outputs_sampling_v6\runs\idw-dimensionality-full-30-seed-20260827\`.
- Lipschitz uncertainty: `C:\Users\stangoodwin\lipschitz\MATH.md` and `src\lipschitz_uncertainty\core.py`.