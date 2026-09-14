# Sampling V2 Experiment Harness

This package is the experiment boundary around three production prototypes:

| Method | Owning package |
|---|---|
| Stratified random | `random_sampling` |
| MinHash LSH 32x4 | `minhash_sampling` |
| Full-session embedding | `trace_sampling` |

The harness combines the retained 300-session and 2,500-session synthetic
Agent365 OTLP sources, keeps expected labels separate from selectors, runs
paired outcome/quadrant/throughput comparisons, emits production-shaped
ExternalEvalSnapshot JSONL, and renders the self-contained V2 HTML report.

## Run

Use [`../sampling_v2_runbook.ipynb`](../sampling_v2_runbook.ipynb) for the
step-by-step artifact-first workflow. Command-line entry points are:

```powershell
py -3.11 scripts/run_sampling_v2.py --output outputs_sampling_v2/runs/<name>
py -3.11 scripts/build_sampling_v2_report.py --input-dir outputs_sampling_v2/runs/<name>
```

The reviewed reference bundle is retained under `outputs_sampling_v2/v2/`.
New CLI runs should target `outputs_sampling_v2/runs/`; do not overwrite the
reference bundle without an explicit review.

## Label Boundary

Expected labels are loaded from the synthetic sources and used only after
selection to calculate pass-rate MAE, fraction saved, and concept coverage.
No LLM judge is called by the V2 experiment. The optional compressed-evidence
judge path remains in `trace_sampling`, disabled by default.

## Matryoshka prefix-cutoff

`matryoshka_experiment.py` tests full-session first-coordinate truncation and
re-normalization against native 1536 dimensions. Defaults match the earlier
dense-only dimensionality study: seeds 13-42, rates 1%, 2%, 5%, 10%, 20%,
and evenly spaced, uniformly random, bursty, front-loaded and agent-blocked
arrivals. Every replay uses each source session once; these are order
perturbations, not bootstrap samples or independent labels.

The selector reuses `AdaptiveSampler`, `AzureClusterIndex` and the local
`InMemoryVectorStore`, including the earlier tau=0.55 and TTL=90 settings.
Full-schedule novelty/rarity ranking enforces exact session caps; membership
is label-blind but **not strictly online admission**. IDW predictions use only
earlier selected same-agent labels, then an earlier global mean or prior 0.5.
Distances are angular, k=8, power=2, epsilon=1e-6; all exact donors are averaged
when 1-cosine <= 1e-8. The primary metric excludes directly observed labels.
Expected labels serve as an oracle for selected sessions, not actual new judge
responses. Business-use-case classification, Lipschitz bounds and the weekly
20-session policy are not measured.

### Inputs and live preparation

For a retained V6 cache, prepare source-hash- and canonical-packet-verified input:

```powershell
.\.venv-v3\Scripts\python.exe scripts\prepare_matryoshka_input.py `
  --dataset dense_2500 --cache "<retained-cache-or-snapshot-directory>" `
  --output outputs_matryoshka\cache\dense_2500
```

For fresh API vectors, use the explicit live preparer. Omitting `--live` is a
zero-API preflight that counts eligible records, tokens and representation
fallbacks. Existing expected labels remain separate from canonical packet text.
The endpoint and deployment must be approved and accessible. The preparer reads
the explicitly named `--env-file` (default `.env` in this worktree), with process
environment values taking precedence. It never searches another checkout's
configuration or prints/persists the key. API-key authentication reuses the
repository's client factory, including `/openai/v1/` for modern Foundry endpoints;
keyless authentication can optionally select an explicit Azure CLI tenant.
Neither secrets nor Azure default account changes are written by these scripts.

```powershell
.\.venv-v3\Scripts\python.exe scripts\prepare_matryoshka_live_input.py `
  --dataset historical_300 --output outputs_matryoshka\cache\historical_300 `
  --env-file .env --live
```

For the user-specified `bugboss-foundry.services.ai.azure.com` resource, the
dotenv must supply `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` and
`AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small`. The supplied
`AZURE_OPENAI_API_VERSION` is accepted; modern Foundry uses the versioned
`/openai/v1/` route rather than a classic Azure deployment route.
Search and chat deployment settings are not used by this experiment.

For `cosmos_otel`, also supply `--cosmos-root "<export-directory>"`.
The existing V7 loader maps good=1, bad=0, partial=0, explicitly retaining
linked-span versus label-document representation provenance. Those labels are
task-design expectations, not proof of actual observed task completion.
API batches are checkpointed and checksum-verified. Resume the same preparation
command after an authentication interruption; a changed source or endpoint
requires a fresh input directory. No embeddings are requested for each cutoff:
all prefixes come from one set of full-dimensional API vectors.
Once a complete cache is published, re-entering preparation validates its
checksums and returns without rewriting vectors, manifests or batch files.
The `.npz` files retain the original 1,536 coordinates, not only one shortened
representation, so later dimensions and randomized replays can reuse them.

The portable input contract also accepts `tau2_bench`: `manifest.json` version
`matryoshka-input-v1` names the dataset, embedding model/dimensions, label source,
representation policy, source hashes, and SHA256-verified `units.json` and
`vectors.npz`. The NPZ contains non-object `unit_ids` and an N-by-1536 `vectors`
matrix in exactly the JSON row order. Every unit explicitly supplies `unit_id`,
tenant/agent-scoped `agent_id`, `signature`, `concept_key` (empty if unavailable),
and a binary `label`. Missing labels, wrong shapes, non-finite vectors, zero
prefixes and mismatched IDs are errors, not synthetic substitutions.

The available Tau2 banking simulation file contains recorded
`reward_info.reward`, not a verified explicit expected-label mapping. Its
embedded task criteria include `NL_ASSERTION` (legacy LLM-scored checks).
Those rewards are not silently relabeled as expected outcomes, and the
preparer does not invoke Tau2 evaluators. Until an appropriate label source
is supplied, Tau2 stays blocked even after embedding authentication is fixed.
`scripts\inspect_matryoshka_tau2.py --source <simulation-json> --output <readiness-json>`
streams counts, source hashes and reward provenance without running a judge
or allocating embeddings.
Once explicit labels are available, the live preparer accepts
`--dataset tau2_bench --tau2-source <simulation-json> --tau2-labels <labels-json>`.
The label file must map every raw simulation ID to 0 or 1 exactly once. It
preserves each trial's ordered messages and tool calls/results, excludes rewards,
evaluation criteria and duplicated provider payloads, and uses task ID only as
a cohort-relative coverage proxy. The adapter treats this single-model run as
one agent cohort, not 97 artificial agents.

### Execution and interpretation

```powershell
.\.venv-v3\Scripts\python.exe scripts\run_matryoshka_experiment.py `
  --input dense_2500=outputs_matryoshka\cache\dense_2500\manifest.json `
  --output outputs_matryoshka\runs\<fresh-run>
.\.venv-v3\Scripts\python.exe scripts\build_matryoshka_report.py `
  --input outputs_matryoshka\runs\<fresh-run>\aggregate.json `
  --output outputs_matryoshka\runs\<fresh-run>\report.html
```

Repeat `--input DATASET=MANIFEST` for each verified dataset. Missing requested
datasets are explicitly blocked, never zero-filled. `--resume` accepts only the
same input hashes, protocol and controlling code. To add newly available
datasets, use a fresh four-input run rather than relabeling an incomplete one.
The default four-dataset experiment has 48,000 cells.

The completed three-dataset run uses these retained local caches. To run another
30 different seeds without embedding or judge calls, use a fresh output name:

```powershell
$seeds = (43..72) -join ','
.\.venv-v3\Scripts\python.exe scripts\run_matryoshka_experiment.py `
  --input historical_300=outputs_matryoshka\cache\live-20260914\historical_300\manifest.json `
  --input dense_2500=outputs_matryoshka\cache\dense_2500\manifest.json `
  --input cosmos_otel=outputs_matryoshka\cache\live-20260914\cosmos_otel\manifest.json `
  --seeds $seeds --output outputs_matryoshka\runs\<fresh-30-seed-rerun>
```

`scripts\validate_matryoshka_run.py --run <run-directory>` checks every membership
hash and verifies both distinct seed orders and order pairing across dimensions,
budgets and modes. The report plots continuous unjudged-only MAE and its
dimension-minus-native difference alongside thresholded accuracy. MAE is not
`1 - accuracy`; the existing accuracy-based candidate table does not establish
a MAE non-inferiority threshold.

The report uses equal-weight cell means and pointwise Student-t intervals over
30 seed means of paired dimension-minus-native accuracy differences. Those
intervals quantify replay sensitivity, not independent production generalization.
The exploratory one-percentage-point tolerance is not an agreed production SLA.
A cutoff candidate must pass at every larger tested prefix too, so an isolated
good result at eight dimensions cannot by itself establish a stable elbow.
The grid is also the evaluation set for this exploratory choice: independent
time-forward confirmation and per-agent/budget checks are still needed.

Keep input caches (including staged packet-derived metadata) under the source
data's access and retention controls. Source vectors and checkpoints stay local;
retain the run's aggregate, membership evidence, input provenance and report.
Float32 storage calculations count N*dimensions*4 bytes, not process memory or
API price savings. Embedding API billing is based on input tokens.

```powershell
.\.venv-v3\Scripts\python.exe -m pytest tests\test_matryoshka_experiment.py `
  tests\test_matryoshka_inputs.py tests\test_matryoshka_report.py `
  tests\test_matryoshka_chart_layout.py -q
```
