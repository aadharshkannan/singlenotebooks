# MinHash Variety Index for Adaptive Trace Sampling

`minhash_sampling` owns the production lexical-novelty prototype. It provides
both exhaustive MinHash clustering for compatibility and a true 32-band x 4-row
LSH candidate index used by Sampling V2.

## What It Is

- A deterministic, bounded-memory alternative to exact signature matching.
- Clusters traces per agent using MinHash similarity over normalized session evidence.
- Emits `VarietyObservation` with:
  - `key.kind="cluster"` for MinHash clusters.
  - `novelty=1.0` for new cluster, `0.0` for joins.
  - `rarity` driven by cadence-normalized staleness for joins.

## What It Is Not

- This is lexical similarity, not semantic understanding.
- It can be vulnerable to dictionary-level paraphrase misses and token manipulation.
- Production deployments need dataset-specific threshold calibration.

## Privacy and Persistence

- Raw shingles are neither persisted nor retained in production memory by default. Calibration runs can opt into bounded in-memory shingle retention.
- Runtime record identity is content-addressed (`sha256`) canonical representation plus profile.
- Content hashes exclude trace IDs, timestamps, and agent identity, so duplicate session content reuses signatures across different trace envelopes.
- Profiles are versioned (`profile_id`) with representation policy/version/max-bytes and MinHash parameters.

## Core Components

- `config.py`: frozen `MinHashConfig` with validation and profile identity.
- `signature.py`: canonical representation -> field-tagged shingles -> MinHash signature.
- `index.py`: `MinHashClusterIndex` and `BandedMinHashLSHIndex` implementing
  `VarietyIndex.observe(trace)`.

## Fallback Behavior

If MinHash build fails with known build-time issues (for example empty evidence), the index falls back to a private `ExactSignatureIndex` and emits `key.kind="fallback-signature"`.

For `BandedMinHashLSHIndex`, candidate matching is intentionally strict: if an
agent has live clusters but LSH bucket lookup yields no valid candidate leaders,
the trace is treated as completely novel and a new cluster is created. The index
does not run an exhaustive leader scan in this case. This means LSH false
negatives intentionally split clusters to preserve sublinear candidate lookup cost.

`RepresentationError` from canonicalization is propagated and not swallowed.

## Complexity

For one trace with `k=permutations` and `s=shingle_count`, comparison against `c` candidate clusters is approximately:

- Build: `O(k * s)`
- Match scan: `O(c * k)`

Total is bounded in practice by:

- `max_shingles`
- `max_clusters_per_agent`
- `max_clusters_total`

TTL and LRU caps ensure hard memory bounds.

## Usage

```python
from trace_sampling.samplers import AdaptiveSampler, SamplerConfig
from minhash_sampling import BandedMinHashLSHIndex, MinHashConfig

cfg = SamplerConfig(llm_throughput=10.0)
mh = BandedMinHashLSHIndex(
  MinHashConfig(
    similarity_threshold=0.55,
    permutations=128,
    lsh_bands=32,
    lsh_rows=4,
  )
)
sampler = AdaptiveSampler(cfg, variety_index=mh, use_novelty=True)
```

New clusters emit both `novelty=1.0` and first-seen `rarity=0.5`, so the index also works with `AdaptiveSampler`'s default `use_novelty=False` mode. Enabling novelty remains recommended when cluster creation should receive the strongest admission signal.

The interactive Sampling V2 experiment, metrics, and retained benchmark evidence
are documented in [`sampling_v2_runbook.ipynb`](../sampling_v2_runbook.ipynb).
