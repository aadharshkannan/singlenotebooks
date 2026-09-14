# singlenotebooks
Agent-evaluation sampling experiments, reporting tools, and supporting analysis notebooks.

## Start Here

| Need | Start with |
| --- | --- |
| Current goal, team direction, and evidence limits | [Sampling context](docs/SAMPLING_CONTEXT.md) |
| Consistent methods/data/results reports in HTML or PDF | [Sampling experiment reporting skill](.github/skills/sampling-experiment-report/SKILL.md) |
| Source, notebooks, retained results, and safe cleanup | [Repository layout](docs/REPOSITORY_LAYOUT.md) |
| Latest retained repeated V7 comparison | [Interactive report](outputs_sampling_v7/runs/v7-live-repeated-20260903/interactive_report.html) and [PCA-8 distance report](outputs_sampling_v7/runs/v7-live-repeated-20260903/pca8-distance-report.html) |

The preferred direction is full-session semantic sampling with IDW score
imputation, conditional Lipschitz bounds, and PCA-8 representations. These are
not yet one validated production pipeline: V7 tests PCA-8 binary IDW, while the
value/bounds prototype is separate. The proposed 20-session-per-agent weekly
sampling gate is planning context, not implemented behavior. Read the context
before interpreting the historical experiments below.

For report work, ask an agent to use `sampling-experiment-report` and supply a
specific run's aggregate path, audience, question, and HTML/PDF requirements.
The skill includes a report template, acceptance checklist, and local browser
quality checker. Existing notebooks and run paths are retained for compatibility.

## Matryoshka prefix-cutoff experiment

The new [experiment runner](sampling_comparison/matryoshka_experiment.py) compares
the full 1,536-dimensional `text-embedding-3-small` vector with its first
512, 256, 128, 64, 32, 16 and 8 coordinates, re-normalized to unit length.
It replays the earlier dimensionality study's ARM2 selection and causal IDW
protocol, with both end-to-end selection and a native-fixed-membership diagnostic.
There is no PCA/SVD/GRP fitting and no new LLM judge.

The [2026-09-14 report](outputs_matryoshka/runs/mrl-cutoff-20260914/report.html)
is **partial**: `dense_2500` completed 30 paired seeds, five arrival schedules,
five session-budget rates and eight dimensions (12,000 cells across both modes).
The other requested datasets, `historical_300`, `cosmos_otel` and `tau2_bench`,
must not be interpreted as measured until their embedding inputs are available.
Fresh embedding API calls were authorized. The user identified
`https://bugboss-foundry.services.ai.azure.com` as the correct resource; its
API-key `.env` is not present in this isolated worktree. The preparer supports
that modern Foundry endpoint through the existing `/openai/v1/` client factory.
Earlier corporate-tenant authentication failures involved the superseded
endpoint, not a verified access failure against this corrected resource.
Existing labels, not a new LLM judge, remain the scoring reference.
The local Tau2 source has 388 trajectories but no verified expected-label
mapping; recorded benchmark rewards are not silently substituted.
See the [input and execution instructions](sampling_comparison/README.md#matryoshka-prefix-cutoff).

## Agent365 Sampling V2

The retained sampling work has four production prototypes:

- [`random_sampling/`](random_sampling/) — deterministic stratified random sampling.
- [`minhash_sampling/`](minhash_sampling/) — lexical novelty sampling with 32x4 MinHash LSH.
- [`trace_sampling/`](trace_sampling/) — compressed full-session embeddings, vector clustering,
  and the matching compressed evidence path for optional LLM judging.
- [`agent_uniform_sampling/`](agent_uniform_sampling/) — deterministic, agent-stratified uniform
  sampling that separates sample membership from token-budget execution pacing.

The expected-label-only experiment and report harness lives in
[`sampling_comparison/`](sampling_comparison/). Start with the interactive
[`sampling_v2_runbook.ipynb`](sampling_v2_runbook.ipynb), or open the retained
[`Agent365 Sampling V2 report`](outputs_sampling_v2/v2/agent365-sampling-v2-report.html).
The exact synthetic sources are retained under [`synthetic_data/`](synthetic_data/).

## Sampling V7

The v7 experiment is a transductive PCA-8 evaluation over the existing three corpora:
`historical_300`, `dense_2500`, and `cosmos_otel`. Every dataset is evaluated with ten paired
bootstrap replays by default. Each replay draws $N$ session occurrences with replacement from
the $N$ source sessions, randomizing both event order and source-session frequency while using
the same replay for every method and budget. This is a sensitivity analysis, not ten sets of
independent labels.

To run it locally without Azure calls, use the deterministic offline test fixtures or a fake
embedder/search adapter. The default live CLI is:

```powershell
.\.venv-v3\Scripts\python.exe scripts\run_sampling_v7.py `
  --output outputs_sampling_v7\runs\<name> `
  --repetitions 10 `
  --base-seed 13
```

This CLI is cloud-backed by default: it loads `AzureConfig.from_env()`, builds canonical v3 full-session runtime packets/embeddings, and syncs PCA-8 evidence vectors into dedicated Search indexes (`trace-clusters-sampling-v7-cosine` and `trace-clusters-sampling-v7-euclidean`). It intentionally fails if live Search sync is disabled outside debug/test usage. Use `--skip-search-sync` only for tests or debug-only runs.

The output includes label-free replay manifests, Student-t replay uncertainty summaries, and
paired cosine-versus-Euclidean differences in addition to the per-run metrics and interactive
report.

Each package README documents its production contract. The V2 runbook is the
single retained interactive experiment and keeps all LLM/network paths disabled
by default.

Run the full offline test suite with:

```powershell
py -3.11 -m pytest -q
```

## Agent-Uniform Sampling (`agent_uniform_sampling_walkthrough.ipynb`)

**Problem:** Token-constrained selection can bias reporting toward short
sessions. This prototype separates **membership** from **execution pacing** so
token cost cannot affect which sessions are included.

For engineering handoff, start with
[`agent_uniform_sampling/README.md`](agent_uniform_sampling/README.md), then read
the language-neutral
[`bounded-evidence design`](docs/AGENT_UNIFORM_BOUNDED_EVIDENCE_DESIGN.md) and
the concrete
[`BIC Evaluations Service implementation map`](docs/BIC_EVALUATIONS_SERVICE_HANDOFF.md).
The standalone generated overview is available as
[`HTML`](outputs_agent_uniform_sampling/agent-uniform-sampling-overview.html) and
[`Markdown`](outputs_agent_uniform_sampling/agent-uniform-sampling-overview.md).

**What it does:**

- Performs deterministic simple random sampling without replacement inside each
  tenant/agent stratum.
- Persists per-stratum metadata in the queue (`N_a`, `n_a`, `p_a`, seed,
  selected request IDs) plus per-item status transitions.
- Applies rolling TPM pacing only after sampling membership is fixed.
- Marks selected items that exceed TPM as `OVERSIZED` instead of replacing them.
- Optionally materializes deterministic token-bounded evidence after membership
  is fixed. Enable it with `BoundedEvidenceConfig(enabled=True)`, call
  `materialize_bounded_evidence(...)`, then schedule using the persisted request
  reservation instead of the raw session estimate. The default remains off.
- Optionally drops pending items whose earliest budget-compliant start exceeds a
  configured schedule-delay limit, while retaining their selected-sample record.
- Reports per-agent selected vs completed counts, mean score, and a
  finite-population-corrected normal-approximation 95% interval when enough
  completed scores exist.

Bounded mode reuses the weighted full-session representation policy, retaining
task goals, final outcomes, and tool results before lower-priority context. It
uses `tiktoken` by default and records the tokenizer identity with the immutable
packet hash. The queue JSON contains the bounded canonical session evidence so
dispatch and retries can send exactly the same artifact; store that file under
the same access and retention controls as source telemetry.

**Run the focused test:**
```bash
python -m pytest tests/test_agent_uniform_sampling.py -q
```

**Notebook execution note:** In this environment, execute notebook code cells
directly from notebook JSON/editor cell execution rather than in-place
`nbconvert --execute --inplace`, which has previously produced empty notebook
files.

## Snap Value Imputation and Lipschitz Bounds

`trace_sampling/value_pipeline.py` gives dropped traces an immediate value from
the same-cluster judged donors retained by `ClusterValueReservoir`. For IDW
imputations in `[0, 1]`, `TraceValue.conditional_geodesic_bounds` contains a
deterministic Lipschitz envelope around the imputed value. The allowance is the
agent-scoped empirical Lipschitz estimate multiplied by the IDW-weighted angular
distance to the donors; display bounds are clamped to `[0, 1]` while raw bounds
remain available for audit.

Calibration uses only completed judge results, keeps all-observation summary
statistics outside the bounded IDW ring, and is cached until judged data changes.
Kept, pending, mean-fallback, prior, and out-of-range values do not claim a band.
The envelope is a sensitivity bound, explicitly **not a confidence interval**.
Configure the estimator with `LipschitzEstimatorConfig` when constructing
`ValuePipeline`; sparse calibration uses its `conservative_fallback`.

> **💸 Cost caveat:** The Azure AI Search Basic tier bills continuously
> (~$75/month) and Azure OpenAI embeddings are usage-based. **Delete the
> resource group when you are done** to stop all charges:
> ```bash
> az group delete -n aadkannan-trace-sampling
> ```
