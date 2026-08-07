# singlenotebooks
Repo where I have single notebooks for one off analysis

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
