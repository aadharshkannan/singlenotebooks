# singlenotebooks
Repo where I have single notebooks for one off analysis

## Adaptive Trace Sampling (`adaptive_trace_sampling.ipynb`)

**Problem:** When multiple AI agents produce OTel-style traces at high and
unequal rates, a naive random sampler either overloads the LLM backend (no
budget control), starves rare agents (they disappear from the kept stream), or
collapses variety (only the most-common signatures survive). This prototype
solves all three problems simultaneously.

**What it does:** The `trace_sampling/` package implements an online,
variety-preserving, non-starving, backpressure-bounded multi-agent trace
sampler. Key properties:

- **Anti-starvation** — every active agent keeps at least one trace per window,
  regardless of how dominant faster agents are.
- **Variety preservation** — per-agent signature coverage (distinct
  signature fraction) meets or beats a budget-matched uniform baseline.
- **Representativeness** — lower TV / KL divergence vs. the stream's true
  signature distribution, measured per agent.
- **Backpressure** — an AIMD rate controller (additive-increase /
  multiplicative-decrease over a drained consumer queue) caps the sustained
  kept-rate to the configured LLM throughput budget, absorbing short bursts
  without exceeding the average limit.

**Demo notebook:** `adaptive_trace_sampling.ipynb` runs a synthetic scenario
with fast/slow and low/high-variety agents plus a mid-stream burst, compares
`AdaptiveSampler` against a budget-matched `BaselineSampler`, plots four metric
families, and finishes with embedded assertions that fail the notebook if any
guarantee is violated.

**Run the tests:**
```bash
python -m pytest tests/ -q
```

**Run the notebook:**
```bash
python -m jupyter nbconvert --to notebook --execute --inplace adaptive_trace_sampling.ipynb
```

## Embedding-Based Variety Comparison (`adaptive_trace_sampling.ipynb`, Section 8)

**Problem:** The base sampler measures variety by *exact* tool-call signature.
That over-counts trivially different signatures as distinct, treats synonym
vocabulary (`search` vs. `query`) as unrelated behavior, and has no notion of
*novel vs. seen-before* beyond first sight. This upgrade makes the variety
comparison production-grade by clustering **semantically similar** tool
sequences with embeddings.

**What it does:** A swappable `VarietyIndex` interface (`trace_sampling/variety.py`)
lets the sampler score variety by either the exact-signature baseline
(`ExactSignatureIndex`) or an embedding-cluster treatment
(`AzureClusterIndex`, `trace_sampling/cluster_index.py`). The treatment:

- **Embeds** each tool-call signature by default (Azure OpenAI
  `text-embedding-3-small`), behind an LRU `EmbeddingCache` so repeated
  signatures cost nothing. Set `ENABLE_FULL_SESSION_EMBEDDINGS=TRUE` to embed
  ordered user/assistant/tool session events instead, with token-aware chunking
  and normalized token-weighted pooling for oversized sessions.
- **Leader-clusters** embeddings with a cosine threshold `tau` over a vector
  store (Azure AI Search vector NN, with an in-process recent-centroid
  fast-path), joining an existing cluster or creating a new one.
- **TTL-purges** stale centroids so returning behavior is re-flagged as fresh
  and memory stays bounded.
- **Degrades gracefully** to exact-signature scoring when a per-tick embed
  budget is exhausted or when Azure calls fail (a `CircuitBreaker` trips and
  the index falls back, then recovers after a cooldown).

**Evaluation:** `trace_sampling/eval_harness.py` runs baseline vs. treatment
arms over a latent-concept synthetic stream (`generate_concept_stream`) and
`trace_sampling/variety_metrics.py` scores concept coverage, ARI / V-measure
vs. the ground-truth concept labels, per-concept redundancy, novel-concept
latency, and cross-agent unification. The treatment recovers more distinct
concepts at a fixed keep budget, unifies synonym vocabulary across agents, and
lowers redundant keeps — all with a >99% embedding-cache hit rate. This
comparison lives in **Section 8** of `adaptive_trace_sampling.ipynb`, alongside
the base adaptive-sampling demo, for a single combined view.

### Azure setup (live arm)

The live treatment arm uses two Azure resources in resource group
`aadkannan-trace-sampling`:

- **Azure OpenAI** account `aadkannan-trace-aoai` with a
  `text-embedding-3-small` deployment.
- **Azure AI Search** service `aadkannan-trace-search` (Basic tier) holding the
  `trace-clusters` vector index.

Authentication is **Entra ID only** (no keys in code); the signed-in principal
needs these RBAC roles:

- **Cognitive Services OpenAI User** (on the OpenAI account) — to call embeddings.
- **Search Index Data Contributor** (on the Search service) — to read/write documents.
- **Search Service Contributor** (on the Search service) — to create the index.

Configure and run:

```bash
cp .env.example .env          # then fill in your resource endpoints
az login                      # Entra auth for the DefaultAzureCredential chain

# run the opt-in live Azure tests (embedding roundtrip + end-to-end cluster smoke)
RUN_AZURE_TESTS=1 python -m pytest -m azure -v

# run the combined notebook against live Azure (omit RUN_AZURE_TESTS for the
# deterministic offline fallback that uses FakeEmbedder + InMemoryVectorStore)
RUN_AZURE_TESTS=1 python -m jupyter nbconvert --to notebook --execute --inplace adaptive_trace_sampling.ipynb
```

In `.env`, leave `ENABLE_FULL_SESSION_EMBEDDINGS=FALSE` (or omit it) for the
original `search -> read -> edit` signature embeddings. Set it to `TRUE` to use
full-session embeddings; the accompanying `SESSION_EMBEDDING_*` values select
the model, immutable model version, tokenizer, and per-chunk token limit.

On Windows PowerShell, set the variable with `$env:RUN_AZURE_TESTS=1` instead of
the inline `RUN_AZURE_TESTS=1` prefix.

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
