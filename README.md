# singlenotebooks
Repo where I have single notebooks for one off analysis

## Random Agent Evaluation Sampling

`random_sampling/` is a separate, batch-oriented Agent 365 evaluation
sampling prototype. It normalizes observed OTLP, ESP, and Kusto trace shapes,
reconstructs Agent 365 sessions, and evaluates one completed UTC-aligned window
per run. The default window is the previous completed 24 hours; one-hour or
other positive durations are configurable. Cochran/FPC planning runs per agent
over that window's eligible sessions.

All selected sessions come from a deterministic proportionate stratified random
sample without replacement. Long sessions become deterministic UTF-8-bounded
evidence packets that preserve first-task, final-outcome, and latest-tool
evidence while auditing omitted middle content.

The included judge is a deterministic stub. The provider-neutral async/sync
judge contract is the swap point for a future CAPI or Foundry resource. See
[`random_sampling/README.md`](random_sampling/README.md) for architecture,
usage, schema support, privacy boundaries, and current limitations.

Run the end-to-end comparison in
[`random_sampling_experiments.ipynb`](random_sampling_experiments.ipynb).
The notebook compares deterministic random seeds against one reused labeled
census pass.

## MinHash Adaptive Trace Sampling

`minhash_sampling/` adds a deterministic lexical MinHash `VarietyIndex` plugin
for the existing online `trace_sampling.AdaptiveSampler`. It hashes bounded,
field-tagged n-grams from session messages and tool evidence, then performs
per-agent TTL/LRU leader clustering without an embedding model or network call.

The package includes an exact-signature comparison and an 18-configuration
sweep. Results show a real tradeoff: aggressive settings improve purity and
separate concepts that share a tool signature, but can over-fragment one
concept into many lexical clusters. See
[`minhash_sampling/README.md`](minhash_sampling/README.md) and
`minhash_sampling_experiments.ipynb` for the calibrated results.

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
  ordered user/assistant/tool session events instead. Oversized sessions become
  one deterministic, UTF-8-bounded evidence packet that preserves task and
  outcome evidence; the same packet is sent to the live judge for kept traces.
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
the model, immutable model version, tokenizer, and model token limit.
`SESSION_REPRESENTATION_MAX_UTF8_BYTES` bounds the canonical packet before any
embedding or judge call. Packet policy/version, byte budget, model, tokenizer,
and pooling identity jointly scope embedding caches and semantic clusters.

Live judge adapters receive a `LiveEvaluationRequest`, not the raw `Trace`.
They must judge `canonical_representation_json`; identity fields are provided
only for routing, logging, and idempotency. Normalization failures propagate
before sampling, external calls, or donor mutation.

On Windows PowerShell, set the variable with `$env:RUN_AZURE_TESTS=1` instead of
the inline `RUN_AZURE_TESTS=1` prefix.

## MinHash Variety Comparison (`minhash_sampling/`)

`minhash_sampling/` adds a plugin-style lexical MinHash variety index for
`trace_sampling.AdaptiveSampler` without forking the sampler itself.

- `MinHashSignatureProvider` builds deterministic field-tagged n-gram signatures
  from canonicalized session evidence.
- `MinHashClusterIndex` implements `VarietyIndex.observe(trace)` with per-agent
  immutable leader clusters, staleness-based rarity, novelty on new clusters,
  TTL purge, and LRU memory bounds.
- Known build-time MinHash failures fall back to exact signature scoring under
  `key.kind="fallback-signature"`; canonical representation failures still
  propagate.

Important limits:

- MinHash here is lexical, not semantic.
- No raw shingles are persisted by the runtime index state.
- Profiles are versioned by seed, n-gram size, permutations, and
  representation policy/version/max-byte settings.
- Complexity is approximately `O(k * shingles + clusters * k)` per trace,
  bounded by configured caps.

Use `minhash_sampling/experiments.py` for deterministic baseline-vs-minhash
comparisons and parameter sweeps (`n`, `permutations`, threshold). Production
usage still requires calibration on real workloads.

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
