# Embedding-Based Variety Comparison — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the exact-match tool-signature variety mechanism with an embedding-based similarity space (Azure OpenAI embeddings + Azure AI Search vector NN, TTL leader-clustering) behind a swappable interface, and prove it beats the baseline with a ground-truth evaluation.

**Architecture:** A `VarietyIndex` protocol abstracts variety scoring; `ExactSignatureIndex` reproduces today's behavior (baseline) and `AzureClusterIndex` implements TTL leader-clustering over Azure AI Search vector search (treatment). `AdaptiveSampler` depends on an injected `VarietyIndex`. A latent-concept synthetic generator supplies ground-truth `concept_id`s so an ablation notebook can compare arms on concept coverage at a fixed budget.

**Tech Stack:** Python 3.12, numpy, pandas, matplotlib, pytest, scikit-learn (ARI/V-measure), `openai`, `azure-identity`, `azure-search-documents`. Entra-only auth via `DefaultAzureCredential` (API keys disabled by policy).

**Spec:** `docs/superpowers/specs/2026-07-07-embedding-variety-design.md`

**Provisioned infra (live):** RG `aadkannan-trace-sampling` (sub `f1f12908-cf33-4c2f-9fc3-ec4f879defc9`) — Azure OpenAI `aadkannan-trace-aoai` (deployment `text-embedding-3-small`, 1536-dim, eastus2) + Azure AI Search `aadkannan-trace-search` (basic, eastus). Both Entra-only.

**Branch:** create `embedding-variety` off `adaptive-trace-sampling`.

---

## Chunk 1: Latent-concept generator + model change

Adds `concept_id` to `Trace` and a concept/synonym-aware generator that produces ground-truth-labeled traces where the same concept surfaces through different tool vocabularies.

### File structure
- Modify: `trace_sampling/model.py` (add `concept_id` field to `Trace`; add concept structures to `AgentConfig` or a new `ConceptSpec`).
- Create: `trace_sampling/concepts.py` (concept definitions + synonym map + realization).
- Modify: `trace_sampling/generator.py` (add `generate_concept_stream`).
- Test: `tests/test_concepts.py`, extend `tests/test_generator.py`.

### Task 1: Add `concept_id` to `Trace`

**Files:**
- Modify: `trace_sampling/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py  (add)
def test_trace_has_concept_id_default():
    from trace_sampling.model import Trace
    t = Trace(0, "a", 0.0, ("search",), 1, 1.0, "ok")
    assert t.concept_id == -1  # default = unknown/unlabeled

def test_trace_accepts_concept_id():
    from trace_sampling.model import Trace
    t = Trace(0, "a", 0.0, ("search",), 1, 1.0, "ok", concept_id=3)
    assert t.concept_id == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py -k concept -v`
Expected: FAIL (unexpected keyword argument `concept_id`).

- [ ] **Step 3: Implement**

```python
# trace_sampling/model.py — add field to the frozen Trace dataclass, LAST with a default
@dataclass(frozen=True)
class Trace:
    trace_id: int
    agent_id: str
    timestamp: float
    signature: Tuple[str, ...]
    span_count: int
    duration_ms: float
    status: str
    concept_id: int = -1  # ground-truth latent concept; -1 = unlabeled. Scoring only.
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_model.py -v`
Expected: PASS (existing positional-construction tests still pass — new field has a default).

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/model.py tests/test_model.py
git commit -m "feat: add concept_id to Trace for ground-truth variety labeling"
```

### Task 2: Concept + synonym model

**Files:**
- Create: `trace_sampling/concepts.py`
- Test: `tests/test_concepts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_concepts.py
import numpy as np
from trace_sampling.concepts import ConceptSpec, SynonymMap, realize_concept

def test_synonym_map_groups_tokens():
    sm = SynonymMap([["search", "query", "find"], ["edit", "modify"]])
    assert sm.canonical("query") == "search"
    assert sm.canonical("modify") == "edit"
    assert sm.canonical("run") == "run"  # ungrouped -> itself

def test_realize_concept_preserves_canonical_sequence():
    sm = SynonymMap([["search", "query", "find"], ["edit", "modify"]])
    spec = ConceptSpec(concept_id=1, canonical=("search", "edit"))
    rng = np.random.default_rng(0)
    # realize with an agent that prefers the 2nd synonym in each group.
    # edit_prob=0.0 makes the surface deterministic (no drop/duplicate) so the
    # canonical subsequence is guaranteed to be recoverable.
    surface = realize_concept(spec, sm, rng, vocab_bias={"search": "query", "edit": "modify"},
                              edit_prob=0.0)
    # surface tokens may differ, but canonicalizing recovers the concept sequence
    canon = tuple(sm.canonical(t) for t in surface if sm.canonical(t) in ("search", "edit"))
    assert canon[:2] == ("search", "edit")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_concepts.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# trace_sampling/concepts.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


class SynonymMap:
    """Maps surface tool tokens to a canonical token per synonym group."""

    def __init__(self, groups: List[List[str]]):
        self.groups = groups
        self._to_canonical: Dict[str, str] = {}
        for group in groups:
            canonical = group[0]
            for tok in group:
                self._to_canonical[tok] = canonical

    def canonical(self, token: str) -> str:
        return self._to_canonical.get(token, token)

    def synonyms(self, canonical: str) -> List[str]:
        for group in self.groups:
            if group[0] == canonical:
                return list(group)
        return [canonical]


@dataclass
class ConceptSpec:
    """A latent behavior concept: a canonical (synonym-normalized) tool subsequence."""
    concept_id: int
    canonical: Tuple[str, ...]


def realize_concept(spec: ConceptSpec, sm: SynonymMap, rng: np.random.Generator,
                    vocab_bias: Optional[Dict[str, str]] = None,
                    edit_prob: float = 0.15) -> Tuple[str, ...]:
    """Turn a concept's canonical sequence into a surface tool sequence.

    * substitute each canonical token with one of its synonyms (biased per-agent
      via ``vocab_bias`` so different agents express the concept differently);
    * apply light edits (drop/duplicate a step) with probability ``edit_prob`` so
      surface sequences vary within a concept.
    """
    vocab_bias = vocab_bias or {}
    out: List[str] = []
    for canon_tok in spec.canonical:
        if canon_tok in vocab_bias:
            surface = vocab_bias[canon_tok]
        else:
            choices = sm.synonyms(canon_tok)
            surface = choices[int(rng.integers(0, len(choices)))]
        if rng.random() < edit_prob:
            continue  # drop this step
        out.append(surface)
        if rng.random() < edit_prob:
            out.append(surface)  # duplicate this step
    if not out:  # never emit an empty sequence
        out.append(vocab_bias.get(spec.canonical[0], spec.canonical[0]))
    return tuple(out)
```

> **Surface-variation scope:** the generator deliberately models two surface-edit
> operations — **drop** and **duplicate** a step — plus **synonym substitution**.
> This is a prototype simplification of the spec's broader edit set (insert/reorder
> are out of scope); drop+duplicate+synonym is enough to exercise the embedding
> variety comparison against exact-match. Do not add insert/reorder unless a later
> task requires it.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_concepts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/concepts.py tests/test_concepts.py
git commit -m "feat: add concept/synonym model for ground-truth variety"
```

### Task 3: Concept-aware generator

**Files:**
- Modify: `trace_sampling/generator.py`
- Test: `tests/test_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator.py  (add)
def test_concept_stream_labels_and_vocab_divergence():
    import numpy as np
    from trace_sampling.concepts import ConceptSpec, SynonymMap
    from trace_sampling.generator import generate_concept_stream, ConceptAgentConfig

    sm = SynonymMap([["search", "query", "find"], ["edit", "modify"], ["run", "exec"]])
    concepts = [ConceptSpec(0, ("search", "edit")), ConceptSpec(1, ("run", "search"))]
    agents = [
        ConceptAgentConfig("agent_a", velocity=10.0, concept_ids=(0, 1), zipf_s=1.0,
                           vocab_bias={"search": "search", "edit": "edit", "run": "run"}),
        ConceptAgentConfig("agent_b", velocity=10.0, concept_ids=(0, 1), zipf_s=1.0,
                           vocab_bias={"search": "query", "edit": "modify", "run": "exec"}),
    ]
    stream = generate_concept_stream(agents, concepts, sm, duration=5.0, seed=3)
    assert len(stream) > 0
    # every trace carries a valid concept label
    assert all(t.concept_id in (0, 1) for t in stream)
    # the SAME concept produces DIFFERENT surface signatures across agents
    sig_a = {t.signature for t in stream if t.agent_id == "agent_a" and t.concept_id == 0}
    sig_b = {t.signature for t in stream if t.agent_id == "agent_b" and t.concept_id == 0}
    assert sig_a and sig_b
    assert sig_a.isdisjoint(sig_b)  # vocab mismatch is real
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_generator.py -k concept -v`
Expected: FAIL (`generate_concept_stream` / `ConceptAgentConfig` not defined).

- [ ] **Step 3: Implement**

```python
# trace_sampling/generator.py  (add imports + new code; keep existing generate_stream)
from dataclasses import dataclass, field
from typing import Dict, Optional
from .concepts import ConceptSpec, SynonymMap, realize_concept


@dataclass
class ConceptAgentConfig:
    agent_id: str
    velocity: float
    concept_ids: Tuple[int, ...]
    zipf_s: float = 1.0
    start_time: float = 0.0
    error_rate: float = 0.05
    vocab_bias: Optional[Dict[str, str]] = None


def generate_concept_stream(agents, concepts, synonym_map: SynonymMap,
                            duration: float, seed: int = 0):
    """Interleaved, time-sorted stream of concept-labeled traces across agents."""
    rng = np.random.default_rng(seed)
    by_id = {c.concept_id: c for c in concepts}
    events = []
    for a in agents:
        ids = list(a.concept_ids)
        ranks = np.arange(1, len(ids) + 1, dtype=float)
        weights = 1.0 / np.power(ranks, a.zipf_s)
        weights /= weights.sum()
        t = a.start_time
        mean_gap = 1.0 / a.velocity
        while True:
            t += rng.exponential(mean_gap)
            if t > duration:
                break
            cid = ids[int(rng.choice(len(ids), p=weights))]
            sig = realize_concept(by_id[cid], synonym_map, rng, vocab_bias=a.vocab_bias)
            status = "error" if rng.random() < a.error_rate else "ok"
            events.append((t, a.agent_id, sig, status, cid))
    events.sort(key=lambda e: e[0])
    traces = []
    for i, (t, agent_id, sig, status, cid) in enumerate(events):
        traces.append(Trace(
            trace_id=i, agent_id=agent_id, timestamp=t, signature=sig,
            span_count=len(sig), duration_ms=float(rng.uniform(5.0, 500.0)),
            status=status, concept_id=cid,
        ))
    return traces
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_generator.py -v`
Expected: PASS (both existing and new tests).

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/generator.py tests/test_generator.py
git commit -m "feat: concept-aware trace generator with vocab divergence"
```

---

## Chunk 2: VarietyIndex interface + baseline + sampler refactor

Introduces the swappable variety abstraction and refactors `AdaptiveSampler` to depend on it, with the baseline (`ExactSignatureIndex`) preserving today's behavior exactly.

### File structure
- Create: `trace_sampling/variety.py` (`VarietyKey`, `VarietyObservation`, `VarietyIndex`, `ExactSignatureIndex`).
- Modify: `trace_sampling/samplers.py` (`AdaptiveSampler` takes an injected `VarietyIndex`, sets `last_observation`, per-arm diversity).
- Test: `tests/test_variety.py`, extend `tests/test_samplers.py`.

### Task 4: `VarietyKey`, `VarietyObservation`, `ExactSignatureIndex`

**Files:**
- Create: `trace_sampling/variety.py`
- Test: `tests/test_variety.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_variety.py
from trace_sampling.model import Trace
from trace_sampling.variety import VarietyKey, ExactSignatureIndex


def _t(sig, agent="a", ts=0.0):
    return Trace(0, agent, ts, sig, len(sig), 1.0, "ok")


def test_exact_index_first_seen_rarity_is_half():
    idx = ExactSignatureIndex()
    obs = idx.observe(_t(("search",)))
    # matches today's AgentStats: count_after=1 -> rarity=1/(1+1)=0.5
    assert obs.rarity == 0.5
    assert obs.novelty == 1.0            # first sight -> novel
    assert obs.key == VarietyKey("signature", ("search",))


def test_exact_index_rarity_decreases_with_repeats():
    idx = ExactSignatureIndex()
    idx.observe(_t(("search",)))
    obs = idx.observe(_t(("search",)))
    assert obs.rarity == 1.0 / 3.0       # count_after=2
    assert obs.novelty == 0.0            # seen before


def test_exact_index_key_is_tagged():
    idx = ExactSignatureIndex()
    obs = idx.observe(_t(("a", "b")))
    assert obs.key.kind == "signature"
    assert obs.key.value == ("a", "b")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_variety.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# trace_sampling/variety.py
from dataclasses import dataclass
from typing import Any, Protocol, Tuple

from .model import Trace
from .stats import AgentStats


@dataclass(frozen=True)
class VarietyKey:
    kind: str   # "signature" | "cluster" | "fallback-signature"
    value: Any  # tuple[str,...] for signatures, str for cluster ids


@dataclass(frozen=True)
class VarietyObservation:
    key: VarietyKey
    rarity: float    # in [0,1]
    novelty: float   # in [0,1]


class VarietyIndex(Protocol):
    def observe(self, trace: Trace) -> VarietyObservation: ...


class ExactSignatureIndex:
    """Baseline: exact tool-signature stratification. Preserves current sampler
    scoring exactly — rarity is post-increment (first-seen == 0.5)."""

    def __init__(self, max_signatures_per_agent: int = 256):
        self.max_signatures_per_agent = max_signatures_per_agent
        self._stats = {}

    def _stats_for(self, agent_id: str) -> AgentStats:
        if agent_id not in self._stats:
            self._stats[agent_id] = AgentStats(max_signatures=self.max_signatures_per_agent)
        return self._stats[agent_id]

    def observe(self, trace: Trace) -> VarietyObservation:
        stats = self._stats_for(trace.agent_id)
        seen_before = trace.signature in stats._counts  # pre-observe state
        stats.observe(trace.timestamp, trace.signature)  # increments first (as today)
        rarity = stats.rarity(trace.signature)           # post-increment: first-seen=0.5
        novelty = 0.0 if seen_before else 1.0
        return VarietyObservation(
            key=VarietyKey("signature", trace.signature),
            rarity=rarity, novelty=novelty,
        )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_variety.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/variety.py tests/test_variety.py
git commit -m "feat: VarietyIndex interface + ExactSignatureIndex baseline"
```

### Task 5: Refactor `AdaptiveSampler` to use an injected `VarietyIndex`

**Files:**
- Modify: `trace_sampling/samplers.py`
- Test: extend `tests/test_samplers.py`

- [ ] **Step 1: Write the failing test** (behavior parity + new plumbing)

```python
# tests/test_samplers.py  (add)
def test_adaptive_sets_last_observation():
    from trace_sampling.model import Trace
    from trace_sampling.samplers import AdaptiveSampler, SamplerConfig
    s = AdaptiveSampler(SamplerConfig(llm_throughput=50.0), seed=0)
    s.decide(Trace(0, "a", 0.0, ("search",), 1, 1.0, "ok"))
    assert s.last_observation is not None
    assert s.last_observation.key.kind == "signature"


def test_adaptive_default_variety_matches_prior_behavior():
    # With the default ExactSignatureIndex, anti-starvation still holds.
    stream = _stream()
    cfg = SamplerConfig(llm_throughput=15.0)
    kept = _run(AdaptiveSampler(cfg, seed=1), stream)
    assert {t.agent_id for t in kept} == {t.agent_id for t in stream}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_samplers.py -k "last_observation or prior_behavior" -v`
Expected: FAIL (`last_observation` missing).

- [ ] **Step 3: Implement** — inject the index; default to `ExactSignatureIndex`. Keep the existing `AgentStats` per agent for **velocity + coldstart + active-agent** tracking (unchanged), and only swap **rarity + stratum key** to come from the injected `VarietyIndex`. This is a minimal, behavior-preserving diff to `AdaptiveSampler`.

Change the constructor signature and add two fields:

```python
# trace_sampling/samplers.py
from .variety import VarietyIndex, VarietyObservation, ExactSignatureIndex

class AdaptiveSampler:
    def __init__(self, config: SamplerConfig, seed: int = 0,
                 variety_index: "VarietyIndex | None" = None,
                 use_novelty: bool = False):
        self.cfg = config
        self._rng = np.random.default_rng(seed)
        self._stats: Dict[str, AgentStats] = {}   # velocity/coldstart/active only
        self._variety: VarietyIndex = variety_index or ExactSignatureIndex(
            max_signatures_per_agent=config.max_signatures_per_agent)
        self._use_novelty = use_novelty           # True only for the treatment arm
        self.last_observation: "VarietyObservation | None" = None
        self._reservoirs: "OrderedDict" = OrderedDict()
        self._last_kept_ts: Dict[str, float] = {}
        self._last_seen_ts: Dict[str, float] = {}
        self._bp = BackpressureController(
            throughput=config.llm_throughput,
            queue_high=config.llm_throughput * config.queue_high_factor,
            queue_low=config.llm_throughput * config.queue_low_factor,
            aimd_increase=config.aimd_increase,
            aimd_decrease=config.aimd_decrease,
            min_multiplier=config.min_multiplier,
        )
        self._res_seed = seed
```

Keep `_stats_for`, `_reservoir_for`, `_active_agent_count` exactly as they are today. Replace `decide()` with this version (the ONLY changes vs today are the three commented lines):

```python
    def decide(self, trace: Trace) -> bool:
        cfg = self.cfg
        self._bp.tick(trace.timestamp)
        stats = self._stats_for(trace.agent_id)
        stats.observe(trace.timestamp, trace.signature)   # velocity/coldstart only
        self._last_seen_ts[trace.agent_id] = trace.timestamp

        obs = self._variety.observe(trace)                # CHANGED: variety via index
        self.last_observation = obs                       # NEW: expose for eval harness
        key = (trace.agent_id, obs.key)                   # CHANGED: stratify by VarietyKey
        reservoir = self._reservoir_for(key)

        fill = len(reservoir) / max(cfg.reservoir_size, 1)
        base = max(obs.rarity, obs.novelty) if self._use_novelty else obs.rarity
        diversity = base * (1.0 - 0.5 * fill)

        n_active = self._active_agent_count(trace.timestamp)
        guaranteed_rate = min(cfg.agent_floor * cfg.llm_throughput,
                              cfg.llm_throughput / n_active)
        velocity = max(stats.velocity(), 1e-6)
        floor_prob = min(1.0, guaranteed_rate / velocity)

        last_kept = self._last_kept_ts.get(trace.agent_id)   # PER-AGENT keep-one (unchanged)
        stale = last_kept is None or (trace.timestamp - last_kept) >= cfg.active_window

        if stale:
            keep = True
        else:
            boost = cfg.coldstart_boost if stats.is_coldstart() else 1.0
            prob = diversity * boost * self._bp.multiplier
            prob = max(prob, floor_prob)
            prob = min(prob, 1.0)
            keep = bool(self._rng.random() < prob)

        if keep:
            reservoir.offer(item=trace.trace_id, weight=max(diversity, 1e-6))
            self._last_kept_ts[trace.agent_id] = trace.timestamp
            self._bp.on_kept()
        return keep
```

**Why baseline parity holds:** with the default `ExactSignatureIndex`, `obs.rarity` equals the old `stats.rarity(signature)` (both are post-increment `1/(1+count)` over the same signature counts), `obs.key.value` equals `trace.signature` so the reservoir key `(agent_id, VarietyKey("signature", sig))` is 1:1 with the old `(agent_id, sig)`, and `use_novelty=False` means `base == rarity`. Velocity/coldstart still come from the sampler's own `AgentStats`. The scoring is therefore identical.

- [ ] **Step 4: Run the full sampler suite (parity check)**

Run: `python -m pytest tests/test_samplers.py -v`
Expected: PASS — all pre-existing tests (anti-starvation, budget, rare-variety, bounded reservoirs, rare-agent floor) still pass, proving the refactor is behavior-preserving.

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/samplers.py tests/test_samplers.py
git commit -m "refactor: AdaptiveSampler depends on injected VarietyIndex"
```

---

## Chunk 3: Embedder + VectorStore (Azure + fakes) + config

Adds the embedding client, the vector store, the signature cache, and the Entra-auth config — each with a deterministic in-memory/fake sibling for offline unit tests plus opt-in live smoke tests.

### File structure
- Create: `trace_sampling/azure_config.py` (env + `DefaultAzureCredential` + token providers).
- Create: `trace_sampling/embedding.py` (`Embedder`, `AzureOpenAIEmbedder`, `FakeEmbedder`, `EmbeddingCache`).
- Create: `trace_sampling/vector_store.py` (`VectorStore`, `InMemoryVectorStore`, `AzureSearchVectorStore`).
- Test: `tests/test_embedding.py`, `tests/test_vector_store.py` (offline + `@pytest.mark.azure`).
- Modify: `requirements-sampling.txt`, `.env.example`, `conftest.py` (marker + skip logic).

### Task 6: Azure config + Entra auth

**Files:**
- Create: `trace_sampling/azure_config.py`
- Test: `tests/test_azure_config.py`

- [ ] **Step 1: Write the failing test** (pure env parsing; no network)

```python
# tests/test_azure_config.py
import os
from trace_sampling.azure_config import AzureConfig

def test_azure_config_from_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://y.search.windows.net")
    monkeypatch.setenv("AZURE_SEARCH_INDEX", "trace-clusters")
    cfg = AzureConfig.from_env()
    assert cfg.openai_endpoint.endswith("openai.azure.com/")
    assert cfg.embedding_deployment == "text-embedding-3-small"
    assert cfg.search_index == "trace-clusters"
```

- [ ] **Step 2: Run** → FAIL. `python -m pytest tests/test_azure_config.py -v`
  Expected: collection error `ModuleNotFoundError: No module named 'trace_sampling.azure_config'`.

- [ ] **Step 3: Implement**

```python
# trace_sampling/azure_config.py
import os
from dataclasses import dataclass


@dataclass
class AzureConfig:
    openai_endpoint: str
    openai_api_version: str
    embedding_deployment: str
    search_endpoint: str
    search_index: str

    @classmethod
    def from_env(cls) -> "AzureConfig":
        # Load a local .env if present (no-op if python-dotenv isn't installed
        # or the file is absent). Real env vars always win over .env values.
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except Exception:
            pass
        return cls(
            openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            embedding_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            search_index=os.environ.get("AZURE_SEARCH_INDEX", "trace-clusters"),
        )


def get_credential():
    """Entra-only credential (API keys are policy-disabled)."""
    from azure.identity import DefaultAzureCredential
    return DefaultAzureCredential()


def openai_token_provider():
    from azure.identity import get_bearer_token_provider
    return get_bearer_token_provider(
        get_credential(), "https://cognitiveservices.azure.com/.default")
```

- [ ] **Step 4: Run** → PASS. `python -m pytest tests/test_azure_config.py -v`

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/azure_config.py tests/test_azure_config.py
git commit -m "feat: Entra-only Azure config loader"
```

### Task 7: Embedder + EmbeddingCache (+ FakeEmbedder)

**Files:**
- Create: `trace_sampling/embedding.py`
- Test: `tests/test_embedding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedding.py
import numpy as np
from trace_sampling.embedding import FakeEmbedder, EmbeddingCache, sequence_to_text
from trace_sampling.model import Trace


def test_sequence_to_text():
    assert sequence_to_text(("search", "read", "edit")) == "search -> read -> edit"


def test_fake_embedder_groups_synonyms_by_canonical():
    from trace_sampling.concepts import SynonymMap
    sm = SynonymMap([["search", "query", "find"], ["run", "exec"]])
    fe = FakeEmbedder(dim=64, synonym_map=sm, noise=0.0, seed=0)

    def cos(x, y): return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    v_search = fe.embed(["search"])[0]
    v_query = fe.embed(["query"])[0]     # synonym of "search"
    v_run = fe.embed(["run"])[0]         # different concept
    # synonyms canonicalize to the same token -> (near-)identical vectors
    assert cos(v_search, v_query) > 0.99
    # unrelated tokens are far apart
    assert cos(v_search, v_run) < 0.5


def test_fake_embedder_is_deterministic():
    fe = FakeEmbedder(dim=32, noise=0.0, seed=0)
    a = fe.embed(["search -> read"])[0]
    b = fe.embed(["search -> read"])[0]
    assert np.allclose(a, b)


def test_embedding_cache_hits_by_signature():
    calls = {"n": 0}
    class Counting:
        def embed(self, texts):
            calls["n"] += len(texts)
            return np.ones((len(texts), 4), dtype=np.float32)
    cache = EmbeddingCache(Counting(), max_size=8)
    cache.get(("search", "read"))
    cache.get(("search", "read"))   # cache hit -> no new call
    assert calls["n"] == 1
    assert cache.n_calls == 1 and cache.n_hits == 1
```

- [ ] **Step 2: Run** → FAIL.

Run: `python -m pytest tests/test_embedding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trace_sampling.embedding'`.

- [ ] **Step 3: Implement**

```python
# trace_sampling/embedding.py
from collections import OrderedDict
from typing import List, Optional, Protocol, Tuple
import numpy as np


def sequence_to_text(signature: Tuple[str, ...]) -> str:
    return " -> ".join(signature)


class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray: ...


class FakeEmbedder:
    """Deterministic, offline embedder for unit tests.

    Builds each text's vector as the SUM of per-token basis vectors, where every
    token is first mapped to its canonical synonym (via an optional ``SynonymMap``).
    Because synonyms collapse to the same canonical token, two surface sequences
    that express the same concept with different vocabulary embed to (near-)identical
    vectors — mimicking a real semantic embedder — while unrelated tokens are far
    apart. This is what lets the offline unit tests exercise concept unification
    deterministically, with no network."""

    def __init__(self, dim: int = 64, synonym_map=None, noise: float = 0.01, seed: int = 0):
        self.dim = dim
        self.synonym_map = synonym_map
        self.noise = noise
        self.seed = seed

    def _token_vec(self, token: str) -> np.ndarray:
        h = abs(hash(token)) % (2**31)
        v = np.random.default_rng(h).normal(size=self.dim)
        n = np.linalg.norm(v)
        return v / (n or 1.0)

    def _canonical(self, token: str) -> str:
        return self.synonym_map.canonical(token) if self.synonym_map is not None else token

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, txt in enumerate(texts):
            tokens = [t.strip() for t in txt.split("->") if t.strip()]
            vec = np.zeros(self.dim, dtype=np.float64)
            for tok in tokens:
                vec += self._token_vec(self._canonical(tok))
            if self.noise:
                h = abs(hash(txt)) % (2**31)
                vec += np.random.default_rng(h).normal(scale=self.noise, size=self.dim)
            out[i] = vec.astype(np.float32)
        return out


class AzureOpenAIEmbedder:
    """Live Azure OpenAI embeddings via Entra ID (no keys)."""

    def __init__(self, config, batch: int = 16):
        from openai import AzureOpenAI
        from .azure_config import openai_token_provider
        self._deployment = config.embedding_deployment
        self._client = AzureOpenAI(
            azure_endpoint=config.openai_endpoint,
            api_version=config.openai_api_version,
            azure_ad_token_provider=openai_token_provider(),
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._deployment, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


class EmbeddingCache:
    """Bounded LRU cache keyed by the signature tuple -> vector. Tracks call/hit
    counters and per-miss embed latency for the eval cost/latency ledger."""

    def __init__(self, embedder: Embedder, max_size: int = 4096):
        self._embedder = embedder
        self._max = max_size
        self._cache: "OrderedDict[Tuple[str, ...], np.ndarray]" = OrderedDict()
        self.n_calls = 0                 # underlying embed() invocations (cache misses)
        self.n_hits = 0                  # cache hits
        self.embed_latencies_ms = []     # wall-clock ms per miss (for p50/p95)

    def get(self, signature: Tuple[str, ...]) -> np.ndarray:
        if signature in self._cache:
            self._cache.move_to_end(signature)
            self.n_hits += 1
            return self._cache[signature]
        import time
        t0 = time.perf_counter()
        vec = self._embedder.embed([sequence_to_text(signature)])[0]
        self.embed_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        self.n_calls += 1
        self._cache[signature] = vec
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return vec

    def __contains__(self, signature) -> bool:
        return signature in self._cache
```

*(The live `AzureClusterIndex` embeds signature text through this same cache; the
canonicalizing `FakeEmbedder` is a deterministic offline stand-in for the real
semantic embedder used only in unit tests.)*

- [ ] **Step 4: Run** → PASS. `python -m pytest tests/test_embedding.py -v`

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/embedding.py tests/test_embedding.py
git commit -m "feat: Embedder (Azure + fake) and signature-keyed EmbeddingCache"
```

### Task 8: VectorStore (InMemory + Azure) + live smoke marker

**Files:**
- Create: `trace_sampling/vector_store.py`
- Create/Modify: `conftest.py`
- Test: `tests/test_vector_store.py`

- [ ] **Step 1: Write the failing tests** (offline InMemory; live Azure marked)

```python
# tests/test_vector_store.py
import numpy as np
import pytest
from trace_sampling.vector_store import InMemoryVectorStore, VectorDoc


def test_inmemory_nearest_and_upsert():
    vs = InMemoryVectorStore()
    assert vs.nearest(np.array([1.0, 0.0]), agent_id="a") is None
    vs.upsert(VectorDoc("c0", np.array([1.0, 0.0]), "a", last_seen=0.0))
    cid, score = vs.nearest(np.array([0.9, 0.1]), agent_id="a")
    assert cid == "c0" and score > 0.9


def test_inmemory_agent_scoping():
    vs = InMemoryVectorStore()
    vs.upsert(VectorDoc("c0", np.array([1.0, 0.0]), "a", last_seen=0.0))
    assert vs.nearest(np.array([1.0, 0.0]), agent_id="b") is None  # different agent


def test_inmemory_purge_stale():
    vs = InMemoryVectorStore()
    vs.upsert(VectorDoc("c0", np.array([1.0, 0.0]), "a", last_seen=0.0))
    vs.upsert(VectorDoc("c1", np.array([0.0, 1.0]), "a", last_seen=100.0))
    removed = vs.purge_stale(now=100.0, ttl=50.0)
    assert removed == ["c0"]                       # returns the purged ids
    res = vs.nearest(np.array([1.0, 0.0]), agent_id="a")
    assert res is not None and res[0] == "c1"      # c0 purged; only c1 remains


@pytest.mark.azure
def test_azure_search_roundtrip():
    # opt-in: requires RUN_AZURE_TESTS=1 and live resources
    from trace_sampling.azure_config import AzureConfig
    from trace_sampling.vector_store import AzureSearchVectorStore
    vs = AzureSearchVectorStore(AzureConfig.from_env(), dim=1536, ensure_index=True)
    v = np.random.default_rng(0).normal(size=1536).astype("float32")
    vs.upsert(VectorDoc("smoke-c0", v, "smoke-agent", last_seen=0.0))
    import time; time.sleep(2)
    res = vs.nearest(v, agent_id="smoke-agent")
    assert res is not None and res[0] == "smoke-c0"
    vs.delete("smoke-c0")
```

- [ ] **Step 2: Run offline** → FAIL. `python -m pytest tests/test_vector_store.py -k inmemory -v`
  Expected: collection error `ModuleNotFoundError: No module named 'trace_sampling.vector_store'`.

- [ ] **Step 3: Implement** the protocol + both stores.

```python
# trace_sampling/vector_store.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple
import numpy as np


@dataclass
class VectorDoc:
    cluster_id: str
    vector: np.ndarray
    agent_id: str
    last_seen: float
    hits: int = 1


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


class VectorStore(Protocol):
    def nearest(self, vec: np.ndarray, agent_id: Optional[str] = None): ...
    def upsert(self, doc: VectorDoc) -> None: ...
    def touch(self, cluster_id: str, now: float) -> None: ...
    def purge_stale(self, now: float, ttl: float) -> "List[str]": ...


class InMemoryVectorStore:
    """Deterministic exact-NN store for unit tests / offline metric passes."""

    def __init__(self):
        self._docs: Dict[str, VectorDoc] = {}
        self.n_queries = 0   # nearest() calls (ledger telemetry)

    def nearest(self, vec, agent_id=None):
        self.n_queries += 1
        best = None
        for doc in self._docs.values():
            if agent_id is not None and doc.agent_id != agent_id:
                continue
            s = _cosine(vec, doc.vector)
            if best is None or s > best[1]:
                best = (doc.cluster_id, s)
        return best

    def upsert(self, doc: VectorDoc) -> None:
        self._docs[doc.cluster_id] = doc

    def touch(self, cluster_id: str, now: float) -> None:
        # refresh TTL only; the centroid vector is intentionally left unchanged
        d = self._docs.get(cluster_id)
        if d:
            d.last_seen = now

    def delete(self, cluster_id: str) -> None:
        self._docs.pop(cluster_id, None)

    def purge_stale(self, now: float, ttl: float) -> List[str]:
        stale = [cid for cid, d in self._docs.items() if now - d.last_seen > ttl]
        for cid in stale:
            del self._docs[cid]
        return stale


class AzureSearchVectorStore:
    """Live Azure AI Search vector store (Entra-only). HNSW + cosine."""

    def __init__(self, config, dim: int = 1536, ensure_index: bool = False):
        from azure.search.documents import SearchClient
        from .azure_config import get_credential
        self._index = config.search_index
        self._endpoint = config.search_endpoint
        self._cred = get_credential()
        self._dim = dim
        self.n_queries = 0   # nearest() calls (ledger telemetry)
        if ensure_index:
            self._ensure_index(config)
        self._client = SearchClient(self._endpoint, self._index, self._cred)

    def _ensure_index(self, config):
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            SearchIndex, SimpleField, SearchField, SearchFieldDataType,
            VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
        )
        ic = SearchIndexClient(self._endpoint, self._cred)
        fields = [
            SimpleField(name="cluster_id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="agent_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="last_seen", type=SearchFieldDataType.Double,
                        filterable=True, sortable=True),
            SimpleField(name="hits", type=SearchFieldDataType.Int64),
            SearchField(name="vector",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=self._dim,
                        vector_search_profile_name="hnsw-cosine"),
        ]
        vs = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="hnsw-cosine", algorithm_configuration_name="hnsw")],
        )
        ic.create_or_update_index(SearchIndex(name=self._index, fields=fields, vector_search=vs))

    def nearest(self, vec, agent_id=None):
        from azure.search.documents.models import VectorizedQuery
        self.n_queries += 1
        vq = VectorizedQuery(vector=vec.tolist(), k_nearest_neighbors=1, fields="vector")
        flt = f"agent_id eq '{agent_id}'" if agent_id is not None else None
        results = self._client.search(search_text=None, vector_queries=[vq], filter=flt, top=1)
        for r in results:
            return (r["cluster_id"], float(r["@search.score"]))
        return None

    def upsert(self, doc: VectorDoc) -> None:
        self._client.merge_or_upload_documents([{
            "cluster_id": doc.cluster_id, "agent_id": doc.agent_id,
            "last_seen": doc.last_seen, "hits": doc.hits,
            "vector": doc.vector.tolist(),
        }])

    def touch(self, cluster_id: str, now: float) -> None:
        # merge updates only the provided fields, leaving the centroid vector intact
        self._client.merge_or_upload_documents([{
            "cluster_id": cluster_id, "last_seen": now,
        }])

    def delete(self, cluster_id: str) -> None:
        self._client.delete_documents([{"cluster_id": cluster_id}])

    def purge_stale(self, now: float, ttl: float) -> List[str]:
        flt = f"last_seen lt {now - ttl}"
        stale = list(self._client.search(search_text=None, filter=flt, select=["cluster_id"], top=1000))
        ids = [r["cluster_id"] for r in stale]
        if ids:
            self._client.delete_documents([{"cluster_id": cid} for cid in ids])
        return ids
```

Note: Azure Search vector `@search.score` for cosine is a rescaled similarity; the `AzureClusterIndex` compares it against a tuned `tau`. The `InMemoryVectorStore` returns raw cosine — the treatment uses whichever store it's given, so `tau` is set appropriately per store (documented in Chunk 4). On a **join**, the index calls `store.touch(cluster_id, now)` (refresh TTL, keep centroid stable); on a **new** cluster it calls `store.upsert(doc)`.

> **Design note — `hits` (supersedes spec §4 "increment hits"):** `touch()` deliberately refreshes `last_seen` only. The `hits` field stays a creation-time telemetry value because Azure Search `merge` cannot atomically increment a field without a read-modify-write round-trip, and per-cluster frequency is not needed at the store: rarity is driven by `AzureClusterIndex._counts` (an in-process **time-decayed** counter), which is a better rarity signal than a raw lifetime hit count. Keeping both stores `last_seen`-only on `touch` also makes the in-memory and Azure paths behaviorally identical.

- [ ] **Step 4: Add the `azure` marker + skip logic**

```python
# conftest.py  (create or extend)
import os
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "azure: live Azure tests (require RUN_AZURE_TESTS=1)")

def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_AZURE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_AZURE_TESTS=1 to run live Azure tests")
    for item in items:
        if "azure" in item.keywords:
            item.add_marker(skip)
```

- [ ] **Step 5: Run offline → PASS; run live smoke opt-in**

Run offline: `python -m pytest tests/test_vector_store.py -v` (azure test shows as skipped)
Expected: InMemory tests PASS, `test_azure_search_roundtrip` SKIPPED.

Run live (after `.env` populated + `az login`): `$env:RUN_AZURE_TESTS=1; python -m pytest tests/test_vector_store.py -k azure -v`
Expected: PASS (real index create/upsert/query/delete).

- [ ] **Step 6: Update deps + env template + commit**

```
# requirements-sampling.txt (add)
scikit-learn
openai
azure-identity
azure-search-documents
python-dotenv
nbformat
nbconvert
ipykernel
```

```
# .env.example — ensure these are present (AZURE_OPENAI_ENDPOINT /
# AZURE_OPENAI_API_VERSION already exist from the prior gpt-5 config; keep them
# and ADD the embedding + search vars below). AZURE_OPENAI_DEPLOYMENT (gpt-5) is
# unused by this feature and may stay or be removed.
AZURE_OPENAI_ENDPOINT="https://aadkannan-trace-aoai.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2024-02-01"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
AZURE_SEARCH_ENDPOINT="https://aadkannan-trace-search.search.windows.net"
AZURE_SEARCH_INDEX="trace-clusters"
```

`AzureConfig.from_env()` loads this `.env` automatically via `python-dotenv` (real
environment variables always take precedence). Never commit a real `.env` — only
`.env.example`.

```bash
git add trace_sampling/vector_store.py tests/test_vector_store.py conftest.py requirements-sampling.txt .env.example
git commit -m "feat: VectorStore (InMemory + Azure AI Search) with live smoke marker"
```

---

## Chunk 4: AzureClusterIndex (TTL leader-clustering + resilience)

Implements the treatment `VarietyIndex`: embed (cached) → recent-centroids/vector NN → τ-threshold join-or-create → TTL purge, with an embed budget and circuit-breaker fallback to exact-signature.

### File structure
- Create: `trace_sampling/cluster_index.py` (`AzureClusterIndex`, `CircuitBreaker`).
- Test: `tests/test_cluster_index.py` (all offline via `FakeEmbedder` + `InMemoryVectorStore`; one `@pytest.mark.azure` end-to-end).

### Task 9: CircuitBreaker

**Files:**
- Create: `trace_sampling/cluster_index.py` (start with the breaker)
- Test: `tests/test_cluster_index.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_cluster_index.py
from trace_sampling.cluster_index import CircuitBreaker

def test_breaker_opens_after_threshold_and_recovers():
    b = CircuitBreaker(fail_threshold=2, cooldown_s=10.0)
    assert b.allow(now=0.0)
    b.on_failure(now=0.0); b.on_failure(now=0.0)     # 2 failures -> open
    assert not b.allow(now=1.0)                       # within cooldown
    assert b.allow(now=11.0)                          # half-open after cooldown
    b.on_success()                                    # closes
    assert b.allow(now=12.0)
```

- [ ] **Step 2: Run** → FAIL.

Run: `python -m pytest tests/test_cluster_index.py -k breaker -v`
Expected: FAIL — `ImportError: cannot import name 'CircuitBreaker' from 'trace_sampling.cluster_index'` (module/class not yet defined).

- [ ] **Step 3: Implement**

```python
# trace_sampling/cluster_index.py  (part 1)
class CircuitBreaker:
    def __init__(self, fail_threshold: int = 5, cooldown_s: float = 30.0):
        self.fail_threshold = fail_threshold
        self.cooldown_s = cooldown_s
        self._fails = 0
        self._opened_at = None

    def allow(self, now: float) -> bool:
        if self._opened_at is None:
            return True
        if now - self._opened_at >= self.cooldown_s:
            return True  # half-open: allow a trial
        return False

    def on_success(self) -> None:
        self._fails = 0
        self._opened_at = None

    def on_failure(self, now: float) -> None:
        self._fails += 1
        if self._fails >= self.fail_threshold:
            self._opened_at = now
```

- [ ] **Step 4: Run** → PASS. `python -m pytest tests/test_cluster_index.py -k breaker -v`
- [ ] **Step 5: Commit**

```bash
git add trace_sampling/cluster_index.py tests/test_cluster_index.py
git commit -m "feat: CircuitBreaker for Azure resilience"
```

### Task 10: AzureClusterIndex clustering behavior

**Files:**
- Modify: `trace_sampling/cluster_index.py`
- Test: `tests/test_cluster_index.py`

- [ ] **Step 1: Failing tests** (offline, deterministic)

```python
# tests/test_cluster_index.py  (add)
import numpy as np
from trace_sampling.model import Trace
from trace_sampling.embedding import FakeEmbedder, EmbeddingCache
from trace_sampling.vector_store import InMemoryVectorStore
from trace_sampling.cluster_index import AzureClusterIndex
from trace_sampling.variety import VarietyKey


def _index(**kw):
    from trace_sampling.concepts import SynonymMap
    # synonyms collapse to a canonical token, so "search"/"query" embed together
    # while "deploy" is far away — deterministic concept unification, no network.
    sm = SynonymMap([["search", "query", "find"], ["edit", "modify"], ["deploy", "release"]])
    fe = FakeEmbedder(dim=64, synonym_map=sm, noise=0.005, seed=0)
    return AzureClusterIndex(EmbeddingCache(fe), InMemoryVectorStore(), tau=0.9, **kw)


def _t(sig, agent="a", ts=0.0, cid=0):
    return Trace(0, agent, ts, sig, len(sig), 1.0, "ok", concept_id=cid)


def test_first_trace_is_new_cluster_and_novel():
    idx = _index()
    obs = idx.observe(_t(("search",), cid=0))
    assert obs.key.kind == "cluster"
    assert obs.novelty == 1.0

def test_same_concept_joins_same_cluster():
    idx = _index()
    a = idx.observe(_t(("search",), agent="a", cid=0))
    b = idx.observe(_t(("query",), agent="a", cid=0))   # different surface, same concept
    assert a.key == b.key                                 # unified
    assert b.novelty < 0.5

def test_different_concept_makes_new_cluster():
    idx = _index()
    idx.observe(_t(("search",), agent="a", cid=0))
    obs = idx.observe(_t(("deploy",), agent="a", cid=2))
    assert obs.novelty > 0.5

def test_ttl_purge_reflags_returning_behavior_as_new():
    idx = _index(ttl=10.0, purge_every=1)
    first = idx.observe(_t(("search",), agent="a", ts=0.0, cid=0))
    # long gap beyond ttl -> old centroid purged -> returns as a NEW cluster
    later = idx.observe(_t(("search",), agent="a", ts=100.0, cid=0))
    assert later.key != first.key
    assert later.novelty == 1.0

def test_embed_budget_exhaustion_falls_back_to_signature():
    idx = _index(embed_budget_per_tick=0)   # no embeds allowed
    obs = idx.observe(_t(("search",), cid=0))
    assert obs.key.kind == "fallback-signature"

def test_fallback_preserves_exact_signature_rarity():
    idx = _index(embed_budget_per_tick=0)   # everything falls back
    a = idx.observe(_t(("search",), agent="a", cid=0))
    b = idx.observe(_t(("search",), agent="a", cid=0))
    assert a.key.kind == "fallback-signature" and b.key.kind == "fallback-signature"
    assert a.rarity == 0.5              # post-increment 1/(1+1)
    assert b.rarity == 1.0 / 3.0        # exact-signature counting continues in fallback
```

- [ ] **Step 2: Run** → FAIL.

Run: `python -m pytest tests/test_cluster_index.py -k "concept or cluster or ttl or budget or fallback" -v`
Expected: FAIL — `ImportError: cannot import name 'AzureClusterIndex' from 'trace_sampling.cluster_index'`.

- [ ] **Step 3: Implement** the index.

```python
# trace_sampling/cluster_index.py  (part 2)
import itertools
import numpy as np

from .model import Trace
from .variety import VarietyObservation, VarietyKey, ExactSignatureIndex
from .embedding import EmbeddingCache
from .vector_store import VectorStore, VectorDoc


def _cos(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


class AzureClusterIndex:
    """Treatment VarietyIndex: TTL leader-clustering over a vector store.

    Per trace: embed (cached) -> nearest centroid (recent in-process buffer first,
    then the vector store) -> if cosine >= tau join that cluster else create a new
    one. Stale centroids (now - last_seen > ttl) are purged so returning behavior
    is re-flagged as fresh and memory stays bounded. Degrades to exact-signature
    scoring when the per-tick embed budget is exhausted or Azure calls fail
    (circuit breaker)."""

    def __init__(self, cache: EmbeddingCache, store: VectorStore, tau: float = 0.85,
                 ttl: float = 60.0, purge_every: int = 200,
                 embed_budget_per_tick: int = 8, recent_buffer_size: int = 64,
                 breaker=None, decay_half_life: float = 30.0):
        self._cache = cache
        self._store = store
        self.tau = tau
        self.ttl = ttl
        self.purge_every = purge_every
        self.embed_budget_per_tick = embed_budget_per_tick
        self._recent = []                      # list[(cluster_id, agent_id, vector, last_seen)]
        self._recent_max = recent_buffer_size
        self._breaker = breaker
        self._ids = itertools.count()
        self._counts = {}                      # cluster_id -> decayed count
        self._last_decay_ts = {}
        self._decay_half_life = decay_half_life
        self._since_purge = 0
        self._embeds_this_tick = 0
        self._last_tick = None
        self._fallback_index = ExactSignatureIndex()  # exact-signature degrade path
        self.n_fallbacks = 0                   # ledger counter

    def _new_id(self) -> str:
        return f"c{next(self._ids)}"

    def _tick(self, ts: float):
        if ts != self._last_tick:
            self._embeds_this_tick = 0
            self._last_tick = ts

    def _fallback(self, trace: Trace) -> VarietyObservation:
        # exact-signature scoring, re-tagged so the sampler/eval can see it degraded
        self.n_fallbacks += 1
        obs = self._fallback_index.observe(trace)
        return VarietyObservation(
            key=VarietyKey("fallback-signature", trace.signature),
            rarity=obs.rarity, novelty=obs.novelty)

    def _evict_recent(self, now: float):
        self._recent = [e for e in self._recent if now - e[3] <= self.ttl]

    def _touch_recent(self, cluster_id: str, now: float):
        for i, (cid, aid, v, _) in enumerate(self._recent):
            if cid == cluster_id:
                self._recent[i] = (cid, aid, v, now)
                return

    def _recent_nearest(self, agent_id, vec):
        best = None
        for cid, aid, v, _ in self._recent:
            if aid != agent_id:
                continue
            s = _cos(vec, v)
            if best is None or s > best[1]:
                best = (cid, s)
        return best

    def _bump(self, cluster_id: str, ts: float) -> float:
        prev_ts = self._last_decay_ts.get(cluster_id, ts)
        decay = 0.5 ** ((ts - prev_ts) / self._decay_half_life) if self._decay_half_life else 1.0
        c = self._counts.get(cluster_id, 0.0) * decay + 1.0
        self._counts[cluster_id] = c
        self._last_decay_ts[cluster_id] = ts
        return 1.0 / (1.0 + c)

    def observe(self, trace: Trace) -> VarietyObservation:
        self._tick(trace.timestamp)
        if self._breaker and not self._breaker.allow(trace.timestamp):
            return self._fallback(trace)

        # embed budget: only NOVEL signatures cost an embed; overflow -> fallback
        if trace.signature not in self._cache:
            if self._embeds_this_tick >= self.embed_budget_per_tick:
                return self._fallback(trace)
            self._embeds_this_tick += 1

        try:
            # TTL maintenance FIRST so stale centroids can't be matched this tick
            self._evict_recent(trace.timestamp)
            self._since_purge += 1
            if self._since_purge >= self.purge_every:
                for cid in self._store.purge_stale(now=trace.timestamp, ttl=self.ttl):
                    self._counts.pop(cid, None)
                    self._last_decay_ts.pop(cid, None)
                self._since_purge = 0
            vec = self._cache.get(trace.signature)
            near = self._recent_nearest(trace.agent_id, vec)
            if near is None:
                near = self._store.nearest(vec, agent_id=trace.agent_id)
            # assign (store WRITES are inside the try so a failed write also degrades)
            if near is not None and near[1] >= self.tau:
                cluster_id, score = near
                novelty = max(0.0, 1.0 - score)
                self._store.touch(cluster_id, trace.timestamp)   # keep centroid, refresh TTL
                self._touch_recent(cluster_id, trace.timestamp)
            else:
                cluster_id = self._new_id()
                novelty = 1.0
                self._store.upsert(VectorDoc(cluster_id, vec, trace.agent_id, trace.timestamp))
                self._recent.append((cluster_id, trace.agent_id, vec, trace.timestamp))
                if len(self._recent) > self._recent_max:
                    self._recent.pop(0)
            if self._breaker:
                self._breaker.on_success()
        except Exception:
            if self._breaker:
                self._breaker.on_failure(trace.timestamp)
            return self._fallback(trace)

        rarity = self._bump(cluster_id, trace.timestamp)
        return VarietyObservation(VarietyKey("cluster", cluster_id), rarity, novelty)
```

Key correctness points (each maps to a test above):
- **TTL maintenance runs before matching** — with `purge_every=1, ttl=10`, the second `("search",)` at `ts=100` finds the stale `c0` already evicted from both the recent buffer (`_evict_recent`) and the store (`purge_stale`), so it becomes a fresh cluster (`test_ttl_purge_reflags_returning_behavior_as_new`).
- **Embed + NN + purge are all inside the `try`** so any Azure failure trips the breaker and degrades cleanly (no exception escapes).
- **Fallback delegates to a real `ExactSignatureIndex`**, so `fallback-signature` rarity/novelty follow exact-match counting (`test_fallback_preserves_exact_signature_rarity`).
- **Join uses `store.touch`** (refresh TTL, keep the centroid stable); only new clusters `upsert` a vector.

- [ ] **Step 4: Run** → PASS. `python -m pytest tests/test_cluster_index.py -v`

- [ ] **Step 5: Commit**

```bash
git add trace_sampling/cluster_index.py tests/test_cluster_index.py
git commit -m "feat: AzureClusterIndex TTL leader-clustering with fallback"
```

### Task 11: Live end-to-end cluster smoke (opt-in)

**Files:**
- Test: `tests/test_cluster_index.py`

- [ ] **Step 1: Add the marked live test**

```python
# tests/test_cluster_index.py  (add)
import os, pytest

@pytest.mark.azure
def test_azure_cluster_index_end_to_end():
    from trace_sampling.azure_config import AzureConfig
    from trace_sampling.embedding import AzureOpenAIEmbedder, EmbeddingCache
    from trace_sampling.vector_store import AzureSearchVectorStore
    from trace_sampling.cluster_index import AzureClusterIndex
    cfg = AzureConfig.from_env()
    idx = AzureClusterIndex(
        EmbeddingCache(AzureOpenAIEmbedder(cfg)),
        AzureSearchVectorStore(cfg, dim=1536, ensure_index=True),
        tau=0.55)   # Azure cosine score scaling differs from raw cosine
    a = idx.observe(_t(("search", "read"), agent="live", cid=0))
    b = idx.observe(_t(("query", "read"), agent="live", cid=0))   # synonym variant
    assert a.key.kind == "cluster" and b.key.kind == "cluster"
    # the synonym variant should UNIFY into the same cluster as the original,
    # which is the whole point of the embedding approach:
    assert a.key == b.key
    c = idx.observe(_t(("deploy", "release"), agent="live", cid=1))  # unrelated concept
    assert c.key != a.key
```

- [ ] **Step 2: Run offline** → SKIPPED. `python -m pytest tests/test_cluster_index.py -k end_to_end -v`
- [ ] **Step 3: Run live** (after `.env` + `az login`): `$env:RUN_AZURE_TESTS=1; python -m pytest tests/test_cluster_index.py -k end_to_end -v`
  Expected: PASS. Tune `tau` if the synonym variant does not join (inspect scores).
- [ ] **Step 4: Commit**

```bash
git add tests/test_cluster_index.py
git commit -m "test: live Azure end-to-end cluster smoke (opt-in)"
```

---

## Chunk 5: Evaluation harness, metrics & notebook

Builds the assignment-log harness, the metrics module, and the ablation notebook comparing baseline vs treatment on live Azure.

### File structure
- Create: `trace_sampling/eval_harness.py` (run an arm, produce an assignment DataFrame + cost ledger).
- Create: `trace_sampling/variety_metrics.py` (coverage, redundancy, ARI/V-measure, novel latency, cross-agent offline pass).
- Create: `embedding_variety.ipynb`.
- Modify: `README.md`.
- Test: `tests/test_variety_metrics.py`, `tests/test_eval_harness.py`.

### Task 12: variety_metrics

**Files:**
- Create: `trace_sampling/variety_metrics.py`
- Test: `tests/test_variety_metrics.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_variety_metrics.py
import numpy as np
import pandas as pd
from trace_sampling.variety_metrics import (
    concept_coverage, redundancy_per_concept, cluster_agreement,
    novel_concept_latency, novel_concept_latency_traces, cross_agent_unification)

def _log():
    return pd.DataFrame([
        # ts, agent, concept_id, variety_key, key_kind, kept
        dict(timestamp=0.0, agent_id="a", concept_id=0, variety_key="c0", key_kind="cluster", kept=True),
        dict(timestamp=1.0, agent_id="a", concept_id=0, variety_key="c0", key_kind="cluster", kept=False),
        dict(timestamp=2.0, agent_id="a", concept_id=1, variety_key="c1", key_kind="cluster", kept=True),
        dict(timestamp=3.0, agent_id="b", concept_id=2, variety_key="c2", key_kind="cluster", kept=False),
    ])

def test_concept_coverage():
    cov = concept_coverage(_log())
    # concepts present: {0,1,2}; concepts with >=1 kept: {0,1} -> 2/3
    assert abs(cov - 2/3) < 1e-9

def test_redundancy_per_concept():
    r = redundancy_per_concept(_log())
    assert r[0] == 1  # one kept trace for concept 0

def test_cluster_agreement_perfect():
    # variety_key perfectly aligns with concept_id here
    ari, v = cluster_agreement(_log())
    assert ari == 1.0 and v == 1.0

def test_novel_concept_latency():
    lat = novel_concept_latency(_log())
    assert lat[0] == 0.0   # concept 0 kept at its first appearance (seconds)

def test_novel_concept_latency_traces():
    lat = novel_concept_latency_traces(_log())
    assert lat[0] == 0     # concept 0 kept on its 1st appearance -> 0 traces waited
    assert lat[2] == float("inf")  # concept 2 never kept

def test_cross_agent_unification():
    # two agents each keep the SAME concept; give them near-identical embeddings
    kept = pd.DataFrame([
        dict(agent_id="a", concept_id=0),
        dict(agent_id="b", concept_id=0),
        dict(agent_id="a", concept_id=1),
    ])
    emb = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=float)
    frac = cross_agent_unification(kept, emb, tau=0.9)
    # concept 0's global cluster spans agents a & b -> unified; concept 1 does not
    assert abs(frac - 0.5) < 1e-9
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**

```python
# trace_sampling/variety_metrics.py
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, v_measure_score


def concept_coverage(log: pd.DataFrame) -> float:
    present = set(log["concept_id"].unique())
    kept = set(log.loc[log["kept"], "concept_id"].unique())
    return len(kept & present) / max(1, len(present))


def redundancy_per_concept(log: pd.DataFrame) -> Dict[int, int]:
    kept = log[log["kept"]]
    return kept.groupby("concept_id").size().to_dict()


def cluster_agreement(log: pd.DataFrame) -> Tuple[float, float]:
    labels_true = log["concept_id"].to_numpy()
    labels_pred = log["variety_key"].astype("category").cat.codes.to_numpy()
    return (float(adjusted_rand_score(labels_true, labels_pred)),
            float(v_measure_score(labels_true, labels_pred)))


def novel_concept_latency(log: pd.DataFrame) -> Dict[int, float]:
    """Wall-clock latency (seconds) from a concept's first appearance to its first
    kept trace. inf if the concept is never kept."""
    out = {}
    for cid, grp in log.sort_values("timestamp").groupby("concept_id"):
        first_seen = grp["timestamp"].iloc[0]
        kept = grp[grp["kept"]]
        out[cid] = float(kept["timestamp"].iloc[0] - first_seen) if len(kept) else float("inf")
    return out


def novel_concept_latency_traces(log: pd.DataFrame) -> Dict[int, float]:
    """Number of traces of a concept observed BEFORE its first kept trace (0 if
    kept on first appearance). inf if never kept. Complements the seconds-based
    latency with a volume-based one, as the spec requires both units."""
    out = {}
    for cid, grp in log.sort_values("timestamp").reset_index(drop=True).groupby("concept_id"):
        grp = grp.reset_index(drop=True)
        kept_positions = grp.index[grp["kept"]].tolist()
        out[cid] = float(kept_positions[0]) if kept_positions else float("inf")
    return out


def cross_agent_unification(kept_log: pd.DataFrame, embeddings: np.ndarray,
                            tau: float) -> float:
    """Offline metric-only pass: globally leader-cluster KEPT traces' embeddings
    (ignoring agent scope) and report the fraction of concepts whose resulting
    global cluster spans >=2 agents."""
    assign = []
    centers = []  # (vector, cluster_idx)
    for i in range(len(kept_log)):
        v = embeddings[i]
        best = None
        for cvec, cidx in centers:
            s = float(v @ cvec / ((np.linalg.norm(v)*np.linalg.norm(cvec)) or 1.0))
            if best is None or s > best[1]:
                best = (cidx, s)
        if best is not None and best[1] >= tau:
            assign.append(best[0])
        else:
            assign.append(len(centers))
            centers.append((v, len(centers)))
    df = kept_log.copy()
    df["gcluster"] = assign
    ok = 0
    concepts = df["concept_id"].unique()
    for cid in concepts:
        sub = df[df["concept_id"] == cid]
        # concept "unified" if its dominant global cluster spans >=2 agents
        dom = sub["gcluster"].mode().iloc[0]
        spanning = sub[sub["gcluster"] == dom]["agent_id"].nunique()
        ok += 1 if spanning >= 2 else 0
    return ok / max(1, len(concepts))
```

- [ ] **Step 4: Run** → PASS. `python -m pytest tests/test_variety_metrics.py -v`
- [ ] **Step 5: Commit**

```bash
git add trace_sampling/variety_metrics.py tests/test_variety_metrics.py
git commit -m "feat: variety metrics (coverage, redundancy, ARI/V, latency, cross-agent)"
```

### Task 13: eval_harness

**Files:**
- Create: `trace_sampling/eval_harness.py`
- Test: `tests/test_eval_harness.py`

- [ ] **Step 1: Failing test** (offline, using FakeEmbedder + InMemoryVectorStore)

```python
# tests/test_eval_harness.py
from trace_sampling.eval_harness import run_arm
from trace_sampling.samplers import SamplerConfig
from trace_sampling.concepts import ConceptSpec, SynonymMap
from trace_sampling.generator import generate_concept_stream, ConceptAgentConfig


def _stream():
    # 3 synonym groups; two concepts that are NOT bag-identical so they form
    # distinct clusters. Default edit_prob in the generator produces drop/duplicate
    # surface variants of each concept -> multiple raw signatures per concept.
    sm = SynonymMap([["search", "query"], ["edit", "modify"], ["run", "exec"]])
    concepts = [ConceptSpec(0, ("search", "edit")), ConceptSpec(1, ("search", "run"))]
    agents = [ConceptAgentConfig("a", 10.0, (0, 1),
                                 vocab_bias={"search": "search", "edit": "edit", "run": "run"}),
              ConceptAgentConfig("b", 10.0, (0, 1),
                                 vocab_bias={"search": "query", "edit": "modify", "run": "exec"})]
    return generate_concept_stream(agents, concepts, sm, duration=8.0, seed=1)


def _sm():
    return SynonymMap([["search", "query"], ["edit", "modify"], ["run", "exec"]])


def test_run_arm_produces_result_with_log_and_ledger():
    stream = _stream()
    result = run_arm(stream, SamplerConfig(llm_throughput=20.0), arm="baseline", seed=0)
    log = result.log
    assert set(["timestamp", "agent_id", "concept_id", "signature",
                "variety_key", "key_kind", "kept"]).issubset(log.columns)
    assert len(log) == len(stream)
    assert log["kept"].sum() > 0
    # ledger present; baseline never calls the embedder
    assert set(["embed_calls", "cache_hits", "cache_hit_rate", "search_queries",
                "embed_latency_p50_ms", "embed_latency_p95_ms",
                "added_latency_p50_ms", "added_latency_p95_ms", "est_cost_usd",
                "fallbacks", "kept"]).issubset(result.ledger)
    assert result.ledger["embed_calls"] == 0
    assert result.ledger["est_cost_usd"] == 0.0


def test_run_arm_offline_treatment_unifies_variants():
    stream = _stream()
    result = run_arm(stream, SamplerConfig(llm_throughput=20.0),
                     arm="treatment_offline", seed=0, synonym_map=_sm())
    assert result.ledger["embed_calls"] > 0        # the offline treatment embeds
    log = result.log
    # UNIFICATION: within each agent, surface/order/edit variants of a concept
    # collapse into FEWER embedding clusters than there are distinct raw signatures.
    distinct_sigs = log.groupby("agent_id")["signature"].nunique().sum()
    clustered = log[log["key_kind"] == "cluster"]
    distinct_keys = clustered.groupby("agent_id")["variety_key"].nunique().sum()
    assert distinct_keys < distinct_sigs
```

- [ ] **Step 2: Run** → FAIL.

Run: `python -m pytest tests/test_eval_harness.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_arm' from 'trace_sampling.eval_harness'`.

- [ ] **Step 3: Implement**

```python
# trace_sampling/eval_harness.py
from dataclasses import dataclass
from typing import Optional
import time
import numpy as np
import pandas as pd

from .samplers import AdaptiveSampler, SamplerConfig
from .variety import ExactSignatureIndex


@dataclass
class RunResult:
    log: pd.DataFrame     # per-trace assignment log
    ledger: dict          # cost/latency counters (embed_calls, cache_hits, fallbacks, kept)
    index: object         # the VarietyIndex used (exposes _cache for treatment arms)


def _make_index(arm: str, cfg: SamplerConfig, synonym_map=None):
    if arm == "baseline":
        return ExactSignatureIndex(max_signatures_per_agent=cfg.max_signatures_per_agent), False
    if arm == "treatment_offline":
        from .embedding import FakeEmbedder, EmbeddingCache
        from .vector_store import InMemoryVectorStore
        from .cluster_index import AzureClusterIndex
        # canonicalizing fake embedder -> deterministic concept unification offline
        fe = FakeEmbedder(dim=64, synonym_map=synonym_map, noise=0.01, seed=0)
        return AzureClusterIndex(EmbeddingCache(fe), InMemoryVectorStore(), tau=0.9), True
    if arm == "treatment_azure":
        from .azure_config import AzureConfig
        from .embedding import AzureOpenAIEmbedder, EmbeddingCache
        from .vector_store import AzureSearchVectorStore
        from .cluster_index import AzureClusterIndex, CircuitBreaker
        c = AzureConfig.from_env()
        return AzureClusterIndex(
            EmbeddingCache(AzureOpenAIEmbedder(c)),
            AzureSearchVectorStore(c, dim=1536, ensure_index=True),
            tau=0.55, breaker=CircuitBreaker()), True
    raise ValueError(arm)


def _ledger(index, kept_count: int, decide_latencies_ms=None) -> dict:
    cache = getattr(index, "_cache", None)
    store = getattr(index, "_store", None)
    lat = list(getattr(cache, "embed_latencies_ms", []) or [])
    dlat = list(decide_latencies_ms or [])
    calls = getattr(cache, "n_calls", 0)
    hits = getattr(cache, "n_hits", 0)
    total = calls + hits
    # text-embedding-3-small ~= $0.02 / 1M tokens; assume ~10 tokens/call (rough).
    cost_per_call = 10.0 / 1_000_000.0 * 0.02
    return dict(
        embed_calls=calls,
        cache_hits=hits,
        cache_hit_rate=(hits / total) if total else 0.0,
        search_queries=getattr(store, "n_queries", 0),
        # embed_latency_* = miss-only wall-clock of the embedder call.
        embed_latency_p50_ms=float(np.percentile(lat, 50)) if lat else 0.0,
        embed_latency_p95_ms=float(np.percentile(lat, 95)) if lat else 0.0,
        # added_latency_* = TOTAL per-decision overhead this arm adds end-to-end
        # (embed miss/hit path + vector search/NN + bookkeeping). This is the
        # headline "added latency" the spec asks for; embed_latency_* is a subset.
        added_latency_p50_ms=float(np.percentile(dlat, 50)) if dlat else 0.0,
        added_latency_p95_ms=float(np.percentile(dlat, 95)) if dlat else 0.0,
        est_cost_usd=calls * cost_per_call,
        fallbacks=getattr(index, "n_fallbacks", 0),
        kept=kept_count,
    )


def run_arm(stream, cfg: SamplerConfig, arm: str, seed: int = 0,
            synonym_map=None) -> RunResult:
    index, use_novelty = _make_index(arm, cfg, synonym_map)
    sampler = AdaptiveSampler(cfg, seed=seed, variety_index=index, use_novelty=use_novelty)
    rows = []
    kept_count = 0
    decide_latencies_ms = []
    for t in stream:
        t0 = time.perf_counter()
        kept = sampler.decide(t)
        decide_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        kept_count += int(kept)
        obs = sampler.last_observation
        rows.append(dict(
            timestamp=t.timestamp, agent_id=t.agent_id, concept_id=t.concept_id,
            signature=t.signature, variety_key=str(obs.key.value),
            key_kind=obs.key.kind, kept=kept,
        ))
    return RunResult(pd.DataFrame(rows), _ledger(index, kept_count, decide_latencies_ms), index)


def kept_embeddings(result: RunResult) -> Optional[np.ndarray]:
    """Re-embed the KEPT traces' signatures via the run's cache (treatment arms
    only; returns None for baseline). Row order matches result.log[kept] order,
    so it lines up with cross_agent_unification's kept_log argument."""
    cache = getattr(result.index, "_cache", None)
    if cache is None:
        return None
    kept = result.log[result.log["kept"]]
    return np.array([cache.get(tuple(sig)) for sig in kept["signature"]], dtype=float)
```

*Note: the offline treatment arm becomes decisive only when a `synonym_map` is
passed (so surface-vocabulary variants of a concept collapse together). The
headline result is produced by the **live Azure** arm (`treatment_azure`) per the
spec's "live for everything" choice; the offline arm exists for deterministic CI.*

- [ ] **Step 4: Run** → PASS. `python -m pytest tests/test_eval_harness.py -v`
- [ ] **Step 5: Commit**

```bash
git add trace_sampling/eval_harness.py tests/test_eval_harness.py
git commit -m "feat: eval harness producing per-trace assignment log"
```

### Task 14: Notebook + README + full suite

**Files:**
- Create: `scripts/build_embedding_variety_notebook.py` (temporary `nbformat` builder — run, then delete, mirroring the prior `adaptive_trace_sampling.ipynb` build pattern).
- Create (via the builder): `embedding_variety.ipynb`.
- Modify: `README.md`.

- [ ] **Step 1: Write the notebook builder script.** Create `scripts/build_embedding_variety_notebook.py` that assembles the notebook with `nbformat`. It must emit these cells in order:

```python
# scripts/build_embedding_variety_notebook.py
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Embedding-Based Variety Comparison\n"
    "Baseline exact-signature variety vs. embedding-cluster variety, evaluated on\n"
    "a latent-concept synthetic stream. Shows the treatment recovers more distinct\n"
    "concepts at a fixed keep budget, unifies synonym vocab, and flags novelty."))

# 1. Problem recap
cells.append(nbf.v4.new_markdown_cell(
    "## 1. The problem with exact-match variety\n"
    "Exact tool-signature matching (a) over-counts trivially different signatures as\n"
    "distinct, (b) has no notion of *novelty vs. seen-before* beyond first sight, and\n"
    "(c) treats synonym vocab (`search` vs `query`) as unrelated. Embeddings fix all\n"
    "three by clustering semantically similar tool sequences."))

# 2. Concept generator demo
cells.append(nbf.v4.new_code_cell(
    "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
    "from trace_sampling.concepts import ConceptSpec, SynonymMap\n"
    "from trace_sampling.generator import generate_concept_stream, ConceptAgentConfig\n"
    "\n"
    "sm = SynonymMap([['search','query','find'],['edit','modify'],['run','exec'],\n"
    "                 ['deploy','release']])\n"
    "concepts = [ConceptSpec(0,('search','edit')), ConceptSpec(1,('run','search')),\n"
    "            ConceptSpec(2,('deploy',)), ConceptSpec(3,('edit','run','search'))]\n"
    "agents = [\n"
    "  ConceptAgentConfig('agent_a', 12.0, (0,1,2,3), zipf_s=1.2,\n"
    "     vocab_bias={'search':'search','edit':'edit','run':'run','deploy':'deploy'}),\n"
    "  ConceptAgentConfig('agent_b', 40.0, (0,1), zipf_s=0.6,\n"
    "     vocab_bias={'search':'query','edit':'modify','run':'exec','deploy':'release'}),\n"
    "  ConceptAgentConfig('agent_c', 3.0, (2,3), zipf_s=1.0,\n"
    "     vocab_bias={'search':'find','edit':'edit','run':'run','deploy':'release'}),\n"
    "]\n"
    "stream = generate_concept_stream(agents, concepts, sm, duration=120.0, seed=7)\n"
    "print('traces:', len(stream))\n"
    "same = {(t.agent_id, t.signature) for t in stream if t.concept_id==0}\n"
    "pd.Series([str(s) for s in sorted({str(x) for x in same})]).head(10)"))

# 3. Run both arms at a fixed budget
cells.append(nbf.v4.new_code_cell(
    "import os\n"
    "from trace_sampling.samplers import SamplerConfig\n"
    "from trace_sampling.eval_harness import run_arm, kept_embeddings\n"
    "\n"
    "cfg = SamplerConfig(llm_throughput=8.0)\n"
    "AZURE = os.environ.get('RUN_AZURE_TESTS') == '1'\n"
    "treatment_arm = 'treatment_azure' if AZURE else 'treatment_offline'\n"
    "if not AZURE:\n"
    "    print('WARNING: RUN_AZURE_TESTS!=1 -> using deterministic offline treatment.')\n"
    "base = run_arm(stream, cfg, arm='baseline', seed=0)\n"
    "treat = run_arm(stream, cfg, arm=treatment_arm, seed=0, synonym_map=sm)\n"
    "print('baseline ledger:', base.ledger)\n"
    "print('treatment ledger:', treat.ledger)"))

# 4. Metrics + plots
cells.append(nbf.v4.new_code_cell(
    "from trace_sampling.variety_metrics import (concept_coverage,\n"
    "    redundancy_per_concept, cluster_agreement, novel_concept_latency,\n"
    "    novel_concept_latency_traces, cross_agent_unification)\n"
    "\n"
    "cov_b = concept_coverage(base.log); cov_t = concept_coverage(treat.log)\n"
    "ari_b, v_b = cluster_agreement(base.log); ari_t, v_t = cluster_agreement(treat.log)\n"
    "red_b = redundancy_per_concept(base.log); red_t = redundancy_per_concept(treat.log)\n"
    "import numpy as np\n"
    "med_red_b = float(np.median(list(red_b.values()))) if red_b else 0.0\n"
    "med_red_t = float(np.median(list(red_t.values()))) if red_t else 0.0\n"
    "print(dict(cov_b=cov_b, cov_t=cov_t, ari_b=ari_b, ari_t=ari_t,\n"
    "           med_red_b=med_red_b, med_red_t=med_red_t))"))

cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots(1, 4, figsize=(19, 4))\n"
    "ax[0].bar(['baseline','treatment'], [cov_b, cov_t]); ax[0].set_title('Concept coverage')\n"
    "ax[1].bar(['baseline','treatment'], [ari_b, ari_t]); ax[1].set_title('ARI vs ground truth')\n"
    "ax[2].bar(['baseline','treatment'], [v_b, v_t]); ax[2].set_title('V-measure vs ground truth')\n"
    "ax[3].bar(['baseline','treatment'], [med_red_b, med_red_t]); ax[3].set_title('Median redundancy/concept')\n"
    "plt.tight_layout(); plt.show()"))

# 4b. Cumulative concept-coverage over time (the headline story)
cells.append(nbf.v4.new_code_cell(
    "def coverage_curve(log):\n"
    "    kept = log[log['kept']].sort_values('timestamp')\n"
    "    seen = set(); xs = []; ys = []\n"
    "    for _, r in kept.iterrows():\n"
    "        seen.add(r['concept_id']); xs.append(r['timestamp']); ys.append(len(seen))\n"
    "    return xs, ys\n"
    "xb, yb = coverage_curve(base.log); xt, yt = coverage_curve(treat.log)\n"
    "n_concepts = base.log['concept_id'].nunique()\n"
    "plt.figure(figsize=(8,4))\n"
    "plt.step(xb, yb, where='post', label='baseline')\n"
    "plt.step(xt, yt, where='post', label='treatment')\n"
    "plt.axhline(n_concepts, ls='--', c='grey', label='all concepts')\n"
    "plt.xlabel('time'); plt.ylabel('distinct concepts kept')\n"
    "plt.title('Cumulative concept coverage at fixed budget'); plt.legend(); plt.show()"))

# 4c. Novel-concept latency (traces waited before first keep, per concept)
cells.append(nbf.v4.new_code_cell(
    "lat_b = novel_concept_latency_traces(base.log)\n"
    "lat_t = novel_concept_latency_traces(treat.log)\n"
    "cids = sorted(set(lat_b) | set(lat_t))\n"
    "def _finite(d, k): \n"
    "    v = d.get(k, float('inf')); return v if np.isfinite(v) else np.nan\n"
    "xb = [_finite(lat_b, c) for c in cids]; xt = [_finite(lat_t, c) for c in cids]\n"
    "x = np.arange(len(cids)); w = 0.35\n"
    "plt.figure(figsize=(8,4))\n"
    "plt.bar(x - w/2, xb, w, label='baseline')\n"
    "plt.bar(x + w/2, xt, w, label='treatment')\n"
    "plt.xticks(x, [f'c{c}' for c in cids]); plt.ylabel('traces before first keep')\n"
    "plt.title('Novel-concept latency (lower is better; NaN = never kept)')\n"
    "plt.legend(); plt.show()"))

# 4d. Cost / latency ledger
cells.append(nbf.v4.new_code_cell(
    "ledger_df = pd.DataFrame([\n"
    "    dict(arm='baseline', **base.ledger),\n"
    "    dict(arm='treatment', **treat.ledger),\n"
    "]).set_index('arm')\n"
    "display(ledger_df)\n"
    "fig, ax = plt.subplots(1, 3, figsize=(15, 4))\n"
    "ledger_df[['embed_calls','cache_hits','search_queries','fallbacks']].plot.bar(ax=ax[0])\n"
    "ax[0].set_title('Call counts')\n"
    "ledger_df[['embed_latency_p50_ms','embed_latency_p95_ms',\n"
    "           'added_latency_p50_ms','added_latency_p95_ms']].plot.bar(ax=ax[1])\n"
    "ax[1].set_title('Latency (ms): embed-miss vs total added')\n"
    "ledger_df[['est_cost_usd']].plot.bar(ax=ax[2])\n"
    "ax[2].set_title('Estimated cost (USD)')\n"
    "plt.tight_layout(); plt.show()"))

cells.append(nbf.v4.new_code_cell(
    "# cross-agent unification (offline metric-only pass) — treatment arms only\n"
    "emb = kept_embeddings(treat)\n"
    "if emb is not None:\n"
    "    kept_log = treat.log[treat.log['kept']][['agent_id','concept_id']].reset_index(drop=True)\n"
    "    frac = cross_agent_unification(kept_log, emb, tau=(0.55 if AZURE else 0.9))\n"
    "    print('cross-agent unification fraction:', frac)\n"
    "    plt.figure(figsize=(4,4))\n"
    "    plt.bar(['treatment'], [frac], color='tab:green')\n"
    "    plt.ylim(0, 1); plt.ylabel('fraction of concepts shared across agents')\n"
    "    plt.title('Cross-agent unification (higher = more shared concepts merged)')\n"
    "    plt.show()\n"
    "else:\n"
    "    print('no embeddings for baseline arm')"))

# 5. Success assertions (only strict on the live arm)
cells.append(nbf.v4.new_code_cell(
    "if AZURE:\n"
    "    assert cov_t >= cov_b, (cov_t, cov_b)\n"
    "    assert med_red_t <= med_red_b, (med_red_t, med_red_b)\n"
    "    assert ari_t >= 0.5, ari_t\n"
    "    print('all embedding-variety success criteria passed (LIVE)')\n"
    "else:\n"
    "    print('offline run complete (assertions are only enforced on the live Azure arm)')"))

nb['cells'] = cells
with open('embedding_variety.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('wrote embedding_variety.ipynb')
```

- [ ] **Step 2: Build, execute, then delete the builder**

```powershell
.\.venv\Scripts\python.exe scripts\build_embedding_variety_notebook.py
$env:RUN_AZURE_TESTS=1   # omit to run the deterministic offline fallback
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace embedding_variety.ipynb
Remove-Item scripts\build_embedding_variety_notebook.py
```
Expected: `nbconvert` exits 0; the final cell prints the success line. (Offline path prints the "assertions enforced only on live arm" message and still exits 0.)

- [ ] **Step 3: README section** — document: what the embedding variety does, the Azure resources (`aadkannan-trace-aoai`, `aadkannan-trace-search` in RG `aadkannan-trace-sampling`) + Entra-only auth + the 3 RBAC roles (Cognitive Services OpenAI User, Search Index Data Contributor, Search Service Contributor), how to populate `.env` from `.env.example`, `az login`, how to run the eval (`RUN_AZURE_TESTS=1; python -m pytest -m azure`) and the notebook, and a **cost caveat** (Basic Search ~$75/mo + usage-based OpenAI — delete the RG when done: `az group delete -n aadkannan-trace-sampling`).

- [ ] **Step 4: Run the FULL offline suite (regression gate)**

Run: `python -m pytest -q`
Expected: all offline tests PASS; azure-marked tests SKIPPED. Confirm the count includes the pre-existing 30 tests plus the new ones.

- [ ] **Step 5: Commit**

```bash
git add embedding_variety.ipynb README.md
git commit -m "docs: embedding-variety ablation notebook + README"
```

---

## Final verification checklist

- [ ] `python -m pytest -q` → all offline tests green, azure tests skipped.
- [ ] `$env:RUN_AZURE_TESTS=1; python -m pytest -m azure -v` → live smoke tests green (needs `.env` + `az login`).
- [ ] Notebook executes end-to-end (live) with success assertions passing.
- [ ] Baseline sampler behavior unchanged — run `python -m pytest tests/test_samplers.py -v` and confirm every pre-existing test passes (anti-starvation, budget, rare-variety, bounded reservoirs, rare-agent floor).
- [ ] `.env.example` documents all required vars; no secrets committed.
- [ ] README documents Azure setup, RBAC roles, and the cost caveat.
