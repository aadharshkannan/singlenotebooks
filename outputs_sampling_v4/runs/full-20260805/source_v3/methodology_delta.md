# V3 Methodology Delta

- Live embeddings are 1536-dimensional Foundry/Azure OpenAI vectors with explicit deployment provenance.
- Embedding-cell novelty uses live scoped Azure AI Search HNSW filtered by tenant, run scope, and semantic scope.
- A 4096-entry exact recent-leader buffer resolves many decisions before HNSW lookup and shields index lag effects.
- Token representation is token-v3 packetized evidence with max packet tokens of 8191.
- Budgets are exact token-mass floors derived from eligible population token mass.
- V3 does not use Cochran sample sizing or finite-population correction; every arm packs as many whole sessions as its exact token budget permits.
- Sessions are indivisible token units; membership is maximal feasible under exact budget slack constraints.
- Adaptive policy is native proposal followed by deterministic maximal fill under remaining budget.
- Random arm is descriptive only; no probability-based confidence interval is reported.
- Expected labels are joined after label-blind membership selection for diagnostics/scoring only.
- Legacy percent tiers are stored only as provenance for conversion to exact token budgets.
- Live resource and threshold limits are explicit: embedding/vector dim 1536, packet max 8191, tau 0.55.
- Packet cap binding check from runtime inventory: 0/2800 packets truncated; max emitted tokens 3899; cap is non-binding.

## Runtime Provenance

- token_profile_id: token-profile-v3|model=text-embedding-3-small|encoding=cl100k_base|encoding_id=cl100k_base:0.13.0|version=0.13.0|max_tokens=8191|embedding_model_id=text-embedding-3-small|embedding_deployment_id=text-embedding-3-small
- minhash_profile_id: v3-token-minhash-v1|token_profile=token-profile-v3|model=text-embedding-3-small|encoding=cl100k_base|encoding_id=cl100k_base:0.13.0|version=0.13.0|max_tokens=8191|embedding_model_id=text-embedding-3-small|embedding_deployment_id=text-embedding-3-small|seed=13|n=3|perms=128|bands=32|rows=4|max_shingles=4096
- embedding_profile_id: 5c4af6abb63737859b4fd72777a3f1d2ec3d51a9c7a34a18d6609a8c279121df
- embedding_semantic_scope: 7c8226708710df0cf240cf0fd4f3f36a68190e2a268478a90645dde2aae80c01
