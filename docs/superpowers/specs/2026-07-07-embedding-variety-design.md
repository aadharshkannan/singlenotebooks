# Embedding-Based Variety Comparison for Adaptive Trace Sampling — Design

**Date:** 2026-07-07
**Status:** Approved design, pending spec review
**Builds on:** `2026-06-26-adaptive-trace-sampling-design.md` (the online adaptive tail-sampler)

---

## 1. Problem

The current adaptive trace sampler measures an agent's behavioral **variety** using an
**exact-match signature** — the ordered tuple of tool/span names (`Trace.signature`). Rarity is the
inverse count of that exact tuple (`AgentStats.rarity`). This has three production-grade limitations:

1. **Over-counting variety.** `search→read→edit` and `search→read→edit→test` are treated as 100%
   distinct, so near-duplicate behaviors inflate perceived variety and waste budget on redundant
   traces.
2. **Missed novelty.** A genuinely new behavior whose exact tuple happens to collide with a seen one
   is not recognized as novel.
3. **Vocab mismatch.** Two agents expressing the *same* underlying behavior with different tool
   names (`search` vs `query`) never share a variety bucket.

**Goal:** replace exact-match variety with an **embedding-based similarity space** that unifies
over-counting, novelty detection, and vocab mismatch — and **prove** it beats the current
implementation with a controlled evaluation on synthetic data with known ground truth.

## 2. Constraints

* **Online / streaming.** Per-trace keep/drop decision, bounded memory, gates a live budget (from
  the base design). The variety mechanism must not break these properties.
* **Live Azure end-to-end.** Embeddings via **Azure OpenAI**; nearest-neighbor via **Azure AI
  Search** vector search. All evals hit live Azure (user decision); cost/latency is controlled by
  aggressive signature-level caching, not by stubbing.
* **Entra-only auth.** The subscription policy *disables API keys* on both Azure OpenAI and Azure AI
  Search. All access uses `DefaultAzureCredential` (Entra ID). No keys anywhere in code or config.
* **Graceful degradation.** An Azure slowdown or outage must never stall the keep/drop decision or
  the judge path; the sampler falls back to exact-signature behavior.

### Provisioned infrastructure (already stood up and verified live)

| Resource | Name | Region | Notes |
|---|---|---|---|
| Resource group | `aadkannan-trace-sampling` | eastus2 | sub `Cortex-BIC-Evaluation-Service-Local` (`f1f12908-cf33-4c2f-9fc3-ec4f879defc9`) |
| Azure OpenAI | `aadkannan-trace-aoai` | eastus2 | deployment `text-embedding-3-small`, 1536-dim, `disableLocalAuth=true` |
| Azure AI Search | `aadkannan-trace-search` | eastus | `basic` SKU, `disableLocalAuth=true` |

Data-plane RBAC required (assigned to the running identity): `Cognitive Services OpenAI User` on the
AOAI account; `Search Index Data Contributor` + `Search Service Contributor` on the Search service.
Both data planes verified: embeddings return 1536 dims; the Search index API responds 200.

## 3. Architecture

The variety mechanism becomes **swappable** so today's exact-match logic is the baseline and the
Azure embedding approach is a drop-in alternative — this is what enables a clean ablation eval.

```
Trace ─▶ AdaptiveSampler.decide()
             │  obs = variety_index.observe(trace)   # -> VarietyObservation(key, rarity, novelty)
             │  self.last_observation = obs           # for the eval harness (see §6)
             ▼
        VarietyIndex (protocol)
          ├── ExactSignatureIndex     (baseline; wraps today's AgentStats)
          └── AzureClusterIndex        (treatment)
                   ├── Embedder ──────────▶ AzureOpenAIEmbedder (text-embedding-3-small)
                   ├── EmbeddingCache (LRU, keyed by signature tuple)
                   └── VectorStore ───────▶ AzureSearchVectorStore (HNSW cosine index)
```

### Components (new files, each single-purpose)

* **`trace_sampling/variety.py`** — `VarietyIndex` protocol + `ExactSignatureIndex` (the baseline).
  * **Single atomic call:** `observe(trace) -> VarietyObservation`. `rarity` and `novelty` are
    computed **inside** `observe` and returned together, replacing the current
    "`stats.observe()` then `stats.rarity()`" two-step.
  * `VarietyObservation` = `(key: VarietyKey, rarity: float, novelty: float)`, both scores in `[0,1]`.
  * **Exact baseline scoring semantics (explicit):** `ExactSignatureIndex.observe` **preserves the
    current sampler's rarity behavior exactly** — it increments the signature count first and returns
    `rarity = 1 / (1 + count_after)`, so a first-seen signature yields `rarity = 0.5` (identical to
    today's `AgentStats.observe()`→`rarity()` sequence; the ablation baseline must not silently change
    this). `novelty` is a **new** signal the base sampler never had: `1.0` when the signature was
    unseen **before** this trace (`count_before == 0`), else `0.0`. Thus "no behavior change" refers to
    rarity/stratum-key parity with today; novelty is strictly additive. (The base sampler ignored
    novelty, so surfacing it does not alter baseline decisions unless the diversity formula uses it —
    which it does only in the treatment path; for a faithful A/B, the baseline run uses the *original*
    `diversity = rarity * (1 - 0.5*fill)`, and the treatment run uses `max(rarity, novelty) * ...`.
    This is stated in §5.)
  * **`VarietyKey`** is a **tagged** value — `VarietyKey(kind, value)` where `kind ∈ {"signature",
    "cluster", "fallback-signature"}`. `"cluster"` = a real cluster id from vector NN; `"signature"` =
    the baseline exact tuple; `"fallback-signature"` = treatment degraded to the exact tuple (embed
    budget exhausted or circuit-breaker open). The sampler treats the key opaquely; the tag is for
    metrics/diagnostics (e.g. measuring fallback rate).
  * Baseline `novelty` is a degenerate 0/1: `1.0` for the first sight of a signature (pre-observe
    count `== 0`), else `0.0`.
* **`trace_sampling/embedding.py`** — `Embedder` protocol, `AzureOpenAIEmbedder`, `EmbeddingCache`,
  and a `FakeEmbedder` (unit tests only).
  * `Embedder.embed(texts: list[str]) -> np.ndarray` (batch-friendly).
  * `AzureOpenAIEmbedder` uses `DefaultAzureCredential` + a bearer-token provider; calls the
    `text-embedding-3-small` deployment.
  * `EmbeddingCache` — bounded LRU keyed by the signature tuple → vector. Only novel signatures miss
    and pay an Azure call.
* **`trace_sampling/vector_store.py`** — `VectorStore` protocol, `AzureSearchVectorStore`, and an
  `InMemoryVectorStore` (unit tests only).
  * `nearest(vec, agent_id=None) -> (cluster_id, score) | None`.
  * `upsert(doc)`, `purge_stale(now) -> int`.
* **`trace_sampling/cluster_index.py`** — `AzureClusterIndex` implementing `VarietyIndex` via
  TTL leader-clustering over the vector store, plus the in-memory rarity counters, recent-centroids
  buffer, embed budget, and circuit-breaker.
* **`trace_sampling/azure_config.py`** — reads endpoints/deployment/index names from env (`.env`),
  builds `DefaultAzureCredential`, and the token providers for OpenAI and Search.
* **`trace_sampling/samplers.py`** (modified) — `AdaptiveSampler` depends on an injected
  `VarietyIndex` instead of hard-coded signature/`AgentStats` variety logic.

## 4. TTL clustering + vector search mechanics

`AzureClusterIndex.observe(trace)` performs a streaming **leader clustering** with time-decaying
centroids:

1. **Render** the trace's tool sequence to text, e.g. `"search -> read -> edit"`.
2. **Cache lookup** by signature tuple.
   * Hit → reuse the cached vector (no Azure call).
   * Miss → if the per-tick **embed budget** remains, call `AzureOpenAIEmbedder`, cache the vector;
     else **fall back** to the exact-signature key for this trace (graceful degradation).
3. **Recent-centroids buffer** — check a small in-process buffer of just-created centroids first, to
   cluster back-to-back near-identical traces correctly before Azure Search has indexed the new doc
   (indexing latency is sub-second but nonzero).
4. **Vector NN query** — `VectorStore.nearest(vec, agent_id=...)` returns top-1 by cosine. For the
   **online stratum key, the query is scoped to the same `agent_id`** (an agent's strata are its own
   clusters), matching the base sampler's per-agent stratification. Cross-agent unification is **not**
   done online — it is a separate offline metric-only pass (see §6), so agent-scoped strata and the
   vocab-mismatch metric do not conflict. `nn_scope` remains a config knob for experimentation, but
   the headline eval runs agent-scoped.
5. **Assign:**
   * **Hit** (cosine ≥ `tau`): join that `cluster_id`; refresh its `last_seen`, increment `hits`
     (merge/update the doc).
   * **Miss** (< `tau` or empty): allocate a **new** `cluster_id`, `upsert` a new doc; add it to the
     recent-centroids buffer. This is maximal novelty.
6. **Scores:**
   * `novelty = 1 - cosine_to_nearest` (new cluster → `1.0`).
   * `rarity` = inverse **time-decayed** cluster frequency, held in a bounded in-memory
     `cluster_id -> decayed_count` map (mirrors today's `AgentStats` counting, LRU-capped).

**Staleness / TTL.** Every doc carries `last_seen`. On a cadence (every N traces or T seconds),
`purge_stale(now)` deletes docs with `now - last_seen > ttl` and evicts their in-memory counters.
Effects: (a) bounds index size and cost; (b) a behavior that stops and later returns re-registers as
a **new** cluster → re-sampled as fresh variety (captures drift / regression re-emergence).

**Search index schema** (fields on the single index):

| field | type | notes |
|---|---|---|
| `cluster_id` | `Edm.String` (key) | leader/centroid id |
| `vector` | `Collection(Edm.Single)`, dim 1536 | HNSW, cosine |
| `agent_id` | `Edm.String` (filterable) | stratum scoping |
| `last_seen` | `Edm.Double` (filterable/sortable) | sim-time of last hit |
| `hits` | `Edm.Int64` | merge count (diagnostics) |

**Parameters (config):** `tau` (cosine threshold, start ~0.85), `ttl`, purge cadence,
`embed_budget_per_tick`, cache size, `nn_scope` (agent vs global; default **agent**), and the
circuit-breaker settings: `azure_timeout_s` (default 2.0s per embed/query call), `breaker_fail_threshold`
(default 5 consecutive failures → open), `breaker_cooldown_s` (default 30s open before half-open retry).

**Determinism.** Live embeddings + HNSW ANN are approximate/non-deterministic (accepted per the
"live for everything" choice). Reproducibility comes from fixed **generator** seeds and reporting
metrics as **means over multiple runs**, not from deterministic embeddings.

## 5. Sampler integration + backpressure

`AdaptiveSampler.decide()` delegates variety to the injected `VarietyIndex`:

1. `obs = variety_index.observe(trace)` → one atomic call returning `(key, rarity, novelty)`. The
   stratum key becomes `(agent_id, obs.key)` (the cluster id, or a signature tuple under fallback).
   The reservoir map and fair-share floor are **unchanged**; they simply key on clusters. `decide()`
   also assigns `self.last_observation = obs` so the eval harness can record the assignment (see §6).
2. `rarity, novelty = obs.rarity, obs.novelty` (both from the single `observe` call; no separate
   lookups, preserving observation-time semantics — baseline rarity is intentionally post-increment,
   novelty pre-increment, both fixed inside `observe`).
3. **Diversity score (differs by arm):** the **baseline** run uses the original
   `diversity = rarity * (1 - 0.5 * fill)` unchanged. The **treatment** run uses
   `diversity = max(rarity, novelty) * (1 - 0.5 * fill)`, so a brand-new cluster scores high even when
   its cluster count is tiny and genuinely novel behavior is preferentially kept. Keeping the baseline
   formula untouched makes the ablation clean (only the variety mechanism + its novelty contribution
   change).
4. **Unchanged guarantees:** the deterministic keep-one floor stays **per `agent_id`** exactly as
   today (`_last_kept_ts: Dict[str, float]`, keyed by `trace.agent_id`) — it is **not** re-keyed to
   `(agent_id, cluster_id)`, because doing so would raise the guaranteed-keep volume to one-per-cluster
   -per-window and change the base sampler's budget semantics. Cluster-level variety is driven by the
   diversity score (step 3), not by keep-one. Velocity fair-share floor, cold-start boost, and AIMD
   `bp_multiplier` are likewise unchanged.

**Two decoupled budgets:**

* **Judge budget** (existing) — AIMD on kept volume. Unchanged.
* **Embedding budget** (new) — Azure OpenAI calls. Bounded structurally by the signature cache (only
  novel signatures embed) plus the per-tick embed cap. Hitting the cap → exact-signature fallback for
  those traces (never blocks).

Embeddings gate *how well we resolve variety*; AIMD gates *how much we keep*. A failure/timeout in
Azure (embeddings or Search) trips a **circuit-breaker** → the index transparently falls back to
`ExactSignatureIndex` behavior for a cooldown, so the sampler always makes progress.

**Memory.** In-memory maps (`cluster_id -> decayed_count`, recent-centroids buffer, cache) stay
bounded by LRU + TTL, preserving the bounded-memory property. Azure Search holds the durable
centroid set, itself TTL-bounded via purge.

## 6. Evaluation

### Ground-truth generator (latent concepts)

Extend the synthetic generator with `K` hidden **behavior concepts**:

* Each concept owns a *canonical* short tool subsequence (its "meaning") plus a **synonym map** over
  the tool pool (e.g. `search~query~find~lookup`, `edit~modify~patch`).
* Emitting a trace: pick a concept (per-agent Zipf), realize the canonical sequence, substitute tools
  via the synonym map, and apply light edits (insert/drop/reorder one step). Two traces from the same
  concept are **ground-truth-equivalent** despite different surface tokens.
* Each `Trace` gains a hidden `concept_id: int` field, populated by the generator and used **only**
  for scoring — it is never read by any `VarietyIndex` or the sampler. (`Trace` is a frozen
  dataclass; add the field with a default so existing construction sites/tests stay valid.)
* **Vocab mismatch:** different agents draw synonyms from different regions of the map, so the same
  concept surfaces as different tuples across agents.

### Recording assignments for scoring

`AdaptiveSampler.decide()` returns only `bool`, which is insufficient to compute the metrics. The
**eval harness** (in the notebook, not the sampler) therefore records one row per trace into an
in-memory **assignment log** (a list of dicts / a small SQLite or DataFrame):

`(timestamp, agent_id, concept_id, variety_key, key_kind, kept: bool)`

To obtain `variety_key`/`key_kind` per trace without changing `decide()`'s return type, the sampler
sets a public `last_observation` attribute (the `VarietyObservation` for the most recent `decide`
call). The harness reads it after each `decide()`. All metrics below are computed from this
assignment log after the run.

### Ablation

Same generator stream, same sampler config, same budget; swap only the `VarietyIndex`:

* **Baseline:** `ExactSignatureIndex` (today).
* **Treatment:** `AzureClusterIndex` (live Azure).

### Metrics

All computed from the assignment log (+ the treatment's kept-trace embeddings for the offline pass):

* **Headline — concept coverage at fixed budget:** fraction of active ground-truth concepts (distinct
  `concept_id`) with ≥1 kept trace (per-second and cumulative). Treatment should cover more concepts
  per dollar.
* **Redundancy:** kept traces per concept (over-counting; lower is better).
* **Cluster agreement:** V-measure / Adjusted Rand Index between assigned `variety_key` and true
  `concept_id`, over all observed traces.
* **Novel-concept detection latency:** traces/seconds from a concept's first appearance to its first
  kept trace.
* **Cross-agent unification (offline metric-only pass):** because online strata are agent-scoped,
  this is computed **after** the run by globally clustering the **kept** traces' embeddings (single
  global leader-clustering at the same `tau`, or agglomerative at `1-tau`) and measuring the fraction
  of ground-truth concepts whose resulting global cluster spans ≥2 agents. This isolates "can the
  embedding space unify vocab-divergent expressions of one concept" from the online stratum key.
* **Cost/latency ledger:** embedding calls, Search queries, cache hit-rate, p50/p95 added latency,
  and a $ estimate.

Reporting: multi-seed runs (generator seeds fixed; Azure non-determinism averaged), mean ± spread,
baseline vs treatment side by side.

## 7. Testing

* **Unit (offline, deterministic)** — `FakeEmbedder` (concept_id → fixed base vector + noise) +
  `InMemoryVectorStore` exercise the clustering logic without Azure: `tau` join vs new-cluster, TTL
  purge/eviction, recent-centroids buffer, rarity/novelty scoring, cache hit/miss + embed-budget
  fallback, circuit-breaker → exact-signature fallback.
* **Generator tests** — concept labels stable; synonym substitution preserves `concept_id`;
  cross-agent vocab divergence actually occurs.
* **Contract tests** — both `ExactSignatureIndex` and `AzureClusterIndex` satisfy the `VarietyIndex`
  protocol surface.
* **Live smoke (opt-in, `@pytest.mark.azure`)** — one real embed + one index upsert/query/delete
  against the provisioned resources; skipped unless `RUN_AZURE_TESTS=1`. CI stays green offline while
  the live path is proven.

*The `FakeEmbedder` / `InMemoryVectorStore` exist for unit tests only; the headline eval runs live
Azure.*

## 8. Deliverables

* New modules: `variety.py`, `embedding.py`, `vector_store.py`, `cluster_index.py`,
  `azure_config.py`.
* Generator extension for latent concepts + synonym maps (with `concept_id` on `Trace`).
* Tests (unit + generator + contract + opt-in live smoke).
* Notebook `embedding_variety.ipynb` — problem recap; example synonym-variant traces; baseline vs
  Azure treatment at fixed budget (live); plots (coverage-over-time, redundancy, cluster agreement,
  novel-concept latency, cross-agent unification, cost/latency ledger); embedded success assertions
  (treatment ≥ baseline on coverage; redundancy lower; ARI above threshold).
* `.env.example` (endpoints/deployment/index names — no secrets).
* `requirements` additions: `openai`, `azure-identity`, `azure-search-documents`, `scikit-learn`.
* README section: Azure setup, Entra-only auth, required RBAC roles, how to run the eval.

## 9. Non-goals

* Replacing the base sampler's keep-one / fair-share / AIMD design (reused unchanged).
* A production .NET port (this stays a Python reference/prototype).
* Real production OTel trace ingestion (synthetic generator only).
* Judge integration / scoring (out of scope, as in the base design).
* Async/micro-batch embedding on the critical path (documented as future hardening; the prototype
  uses cache + sync-embed-on-novel-only with a per-tick budget).

## 10. Key decisions & rationale

| Decision | Why | Alternative rejected |
|---|---|---|
| Swappable `VarietyIndex` | Clean ablation baseline vs treatment; isolation/testability. | Fork the sampler (two code paths drift). |
| Azure AI Search for NN | Managed ANN, scales past in-memory; user directive. | In-memory leader clustering (doesn't scale, was the first draft). |
| Entra-only auth | Subscription policy disables keys; production-correct. | API keys (blocked by policy). |
| TTL centroids | Bounds memory/cost; re-flags returning behavior as novel. | Unbounded centroids (memory blow-up; stale variety). |
| Cache + sync-embed-on-novel-only | Bounds embedding QPS/cost to the novel-signature rate; deterministic-ish. | Embed every trace (cost/latency); async back-fill (complexity). |
| Circuit-breaker → exact-signature | Never stall the decision/judge path on Azure issues. | Block on Azure (violates online constraint). |
| Latent-concept generator | Clean, controllable ground truth for the eval. | Hand-labeled fixtures (tiny); real traces (out of scope). |
| Concept coverage @ budget (headline) | Directly ties to "don't lose variety" under cost. | Keep-rate parity (penalizes correct subsampling). |
