# Full-Session Embedding Sampling Prototype

`trace_sampling` owns the production semantic-novelty prototype used by Sampling V2.

## Production flow

1. Build one bounded canonical `SessionEvidencePacket` from ordered messages, tool names, arguments, outputs, and structural evidence.
2. Use `SessionEmbeddingCache` to obtain one content-addressed, unit-normalized full-session vector.
3. Use `FullSessionEmbeddingPrototype` to cluster against same-agent leaders in a `VectorStore`.
4. Feed semantic novelty and cadence-normalized rarity into `AdaptiveSampler` with backpressure and an external budget cap.
5. For selected sessions, reuse the exact canonical packet as auditable LLM-judge evidence. Numeric vectors are retrieval metadata and are included in local judge payloads only when explicitly requested.

The local V2 experiment uses deterministic offline embeddings and `InMemoryVectorStore`. Production adapters for Azure AI Search or Cosmos are resource-specific projections; the canonical packet/profile and tenant-agent scope remain the source contract.

```python
prototype = FullSessionEmbeddingPrototype(cache)
prepared = prototype.prepare(trace)
compact_judge_payload = prototype.build_judge_payload(prepared)
vector_debug_payload = prototype.build_judge_payload(prepared, include_vector=True)
```

No method in this package submits an LLM request by itself. See [`../sampling_v2_runbook.ipynb`](../sampling_v2_runbook.ipynb) for the expected-label-only V2 experiment.
