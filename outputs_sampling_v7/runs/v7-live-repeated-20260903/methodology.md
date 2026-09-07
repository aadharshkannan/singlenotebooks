# Sampling V7 Methodology

## Arms
1. random_sampling: deterministic exact-cap baseline.
2. minhash_lsh: stream-order MinHash novelty+rarity scoring; exact-cap top selection.
3. pca8_idw_binary_cosine: corpus-level transductive PCA-8 fit once on unique source corpus, cosine diversity selection on replay occurrences, then binary IDW with threshold 0.5.
4. pca8_idw_binary_euclidean: same shared source-corpus PCA-8 projection with euclidean distance for both selection and IDW on replay occurrences.
5. arm5_hajek_weighted: v6 Arm5 Hajek-weighted estimator.

## Replay Method
- For each dataset and repetition, N replay occurrences are sampled with replacement from N source sessions.
- Replay order and source frequency are randomized per repetition and paired across all methods/budgets.
- Repeated occurrences are a sensitivity analysis of frequency/order perturbations, not independent new labels.

## Imputation and Labeling
- Binary threshold is fixed at 0.5 for embedding IDW probabilities.
- IDW donor selection uses within-agent neighbors first, then global fallback if no within-agent judged donors exist.
- Expected label mapping is good=1, bad=0, partial defaults to 0 unless overridden.
- expected_outcome and correlated fields are excluded from representation/embedding packets.

## Search Evidence
- Search sync writes reduced vectors and validates successful indexing responses.
- Search evidence checks non-empty neighbor retrieval on selected probes and retries with bounded attempts for eventual consistency.
- Search evidence is validation-only; it does not alter deterministic selection outputs.

## Estimator Comparability
- Headline aggregate and selected-only MAE use the fixed source-corpus census target.
- Replay-relative MAE is retained separately to expose sensitivity to each bootstrap frequency draw.
- Aggregate MAE spans heterogeneous estimators and is reported with estimator_type per row.
- Selected-only MAE is reported for apples-to-apples selection quality comparison.

## Shared Cloud Cost
- Embedding and PCA preprocessing are shared corpus costs and recorded once per dataset in shared_preprocessing/embedding_ledger.
- Per-row latency excludes additive duplication of shared preprocessing overhead.
- Search writes are synced per unique source corpus and metric, then shared across replay repetitions.
