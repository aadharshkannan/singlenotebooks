# MinHash Variety Index for Adaptive Trace Sampling

`minhash_sampling` adds a lexical MinHash-backed `VarietyIndex` implementation that plugs into `trace_sampling.samplers.AdaptiveSampler`.

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
- `index.py`: `MinHashClusterIndex` implementing `VarietyIndex.observe(trace)`.
- `experiments.py`: deterministic baseline/treatment harness and sweeps.

## Fallback Behavior

If MinHash build fails with known build-time issues (for example empty evidence), the index falls back to a private `ExactSignatureIndex` and emits `key.kind="fallback-signature"`.

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
from minhash_sampling import MinHashClusterIndex, MinHashConfig

cfg = SamplerConfig(llm_throughput=10.0)
mh = MinHashClusterIndex(MinHashConfig(similarity_threshold=0.5, permutations=128))
sampler = AdaptiveSampler(cfg, variety_index=mh, use_novelty=True)
```

New clusters emit both `novelty=1.0` and first-seen `rarity=0.5`, so the index also works with `AdaptiveSampler`'s default `use_novelty=False` mode. Enabling novelty remains recommended when cluster creation should receive the strongest admission signal.

## Experiment Findings

`minhash_sampling.experiments` compares the existing exact-signature index with MinHash over the same eventful stream and adaptive sampler seam. The 18-point sweep covers n-grams `{2,3,4}`, permutations `{64,128}`, and thresholds `{0.3,0.5,0.7}`.

The default `3 / 128 / 0.5` configuration improves cluster purity and pairwise concept separation, but over-fragments concepts and therefore lowers ARI and V-measure. It must not be described as a universal quality improvement.

On the checked-in synthetic workload, `4 / 64 / 0.3` is the best calibrated point:

- pair-separation accuracy: +10.8 percentage points versus exact signatures;
- cluster purity: +24.1 points;
- ARI: 0.678 versus 0.534;
- V-measure: 0.820 versus 0.771;
- MinHash Jaccard MAE: 0.029;
- decision p95: about 0.2 ms on the development machine.

These values are deterministic except wall-clock latency. They calibrate this synthetic workload only. Production thresholds must be selected on representative Agent 365 traces, with fragmentation, latency, TTL, and lexical paraphrase misses reported alongside purity/separation gains.
