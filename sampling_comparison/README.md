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

## IMDb 50K sentiment follow-up

`imdb_inputs.py`, `imdb_experiment.py` and `imdb_report.py` provide a separate,
single-agent full-review experiment. The request did not include a dataset
URL; its 50K positive/negative movie-review description is interpreted as
Stanford's original **Large Movie Review Dataset v1.0** (Maas et al., ACL 2011).
The labeled train/test partitions are deliberately combined. This is an
imputation/replay study, **not held-out IMDb benchmark accuracy**.

Current evidence: 50,000 labeled reviews, 25,000 per class, 49,581 distinct
normalized embedding inputs, 419 duplicate-text rows, no conflicting-label
text groups, and no reviews exceeding the 8,191-token limit. Unique inputs
contain 14,166,270 `cl100k_base` tokens. The archive SHA-256 is
`c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe`.
These are observed **input statistics**, not model results or billed API tokens.
The local input cache and archive are not checked into Git.

The authorized Azure preparation has now completed: **775 successful requests,
14,166,270 reported API input tokens, zero judge calls**, with a verified
`[50000,1536]` native `text-embedding-3-small` matrix. Vector-file SHA-256:
`7218d733c1fafea421feb4f513f3a578cd3d2ad39c4146242764c52eb51733ba`.
All **2,400 real-vector replay cells** subsequently completed. The aggregate-only
[report](../outputs_imdb/reports/imdb-40-replay/report.html) contains every budget,
dimension and schedule. At 5% budget (both schedules averaged within seeds):

| Dimensions | Unselected MAE | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1536 | 0.2054 | 89.60% | 87.93% | 91.50% | 0.8966 |
| 32 | 0.2449 | 83.03% | 88.14% | 75.83% | 0.8149 |
| 24 | 0.2585 | 81.39% | 87.46% | 72.88% | 0.7947 |
| 16 | 0.2832 | 78.98% | 81.85% | 74.21% | 0.7782 |
| 12 | 0.2984 | 77.16% | 78.89% | 73.97% | 0.7633 |
| 8 | 0.3226 | 74.37% | 75.74% | 71.55% | 0.7357 |

Native has lower mean MAE than every prefix at each of the five budget averages.
The native/8d novel-source MAEs at 5% are 0.2104/0.3307, so earlier observations
of the same review do not explain away the gap. Prefix quality is not globally
monotonic: at 1% budget, 8d MAE is 0.3475 versus 32d 0.3565.

On matched native envelope-eligible targets at 5%, lower-envelope thresholding
raises precision from 87.93% to 99.17%, but recall falls from 91.50% to 3.04%;
mean exact AUROC falls from 0.9605 to 0.7580. High precision with very low recall
is not an overall classification improvement. These findings concern causal
imputation under the replay protocol, not a deployable calibrated classifier.
The retained-score audit finds 73.42% of native eligible lower scores clipped
to zero at 5%; the mean full-envelope width is 0.9094 on the 0-1 scale.
Broad envelopes can have high observed label coverage without being informative.

The executed design uses 40 paired bootstrap seeds, two arrival schedules (uniform
and bursty), five occurrence-label budgets (1%, 2%, 5%, 10%, 20%) and six
dimensions (1536, 32, 24, 16, 12, 8): **2,400 cells**. Draw 50K occurrences
with replacement per seed to change both order and review frequency. Every
dimension/budget sees the same stream for that seed/schedule. Selection reuses
the existing ARM2 label-blind full-schedule ranking; only IDW and calibration
are causal. This is not strictly online membership selection.

Point estimation uses normalized angular distance, eight earlier selected
donors, inverse-square weights and epsilon `1e-6`; exact matches average all
earlier matching donors. A source-repeat diagnostic excludes targets with an
earlier selected occurrence of that source. Distinct source rows containing
identical text can still match. Primary metrics exclude directly observed
labels. Eligible-point and eligible-lower metrics/ROC use identical targets.
All-unselected includes any prior/fallback scores separately accounted for.

To avoid 50K-by-50K distance matrices, the replay implementation evaluates
bounded target/donor blocks. The causal Lipschitz calibration uses a deterministic
uniform reservoir of up to 128 earlier selected occurrences, q90 pair slopes,
angular denominator floor 0.01 and sparse fallback L=1. This is a documented
scale adaptation from the previous all-pair study, not baseline parity.
The envelope is a conditional sensitivity construction, not a confidence
interval. Replay percentile ranges reuse labels and are not population CIs.

Prepare the public source locally (no extraction of arbitrary archive members):

```powershell
New-Item -ItemType Directory -Force external_data\imdb | Out-Null
curl.exe --fail --location https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz `
  --output external_data\imdb\aclImdb_v1.tar.gz
.\.venv-v3\Scripts\python.exe scripts\prepare_imdb_input.py
```

One review is one single-message session. Normalize HTML line breaks/entities
and outer whitespace; embed the whole review text without labels, scores or
filenames. Cap at 8,191 tokens if necessary and record truncation. Request
native 1536 vectors once; never use PCA/SVD or a reduced-dimension API request.
Retain all source rows while sharing embeddings for identical emitted text.

The following command explicitly enables paid embedding requests using only
this worktree's `.env` (or an explicit `--env-file`) and process environment.
It uses the existing Azure embedding client, checks the returned model/dimension,
and resumes only source-bound, hash-verified batches. It never calls a judge.
Supply `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` and the
appropriate API key or existing Entra credentials. Do not paste secrets into
reports or source files. No Azure Search resource is needed.

```powershell
.\.venv-v3\Scripts\python.exe scripts\prepare_imdb_input.py --live --env-file .env
.\.venv-v3\Scripts\python.exe scripts\run_imdb_experiment.py
.\.venv-v3\Scripts\python.exe scripts\validate_imdb_experiment.py
.\.venv-v3\Scripts\python.exe scripts\build_imdb_report.py `
  --input outputs_imdb\private_runs\imdb-40-replay\aggregate.json `
  --output outputs_imdb\reports\imdb-40-replay `
  --numerical-validation outputs_imdb\reports\imdb-40-replay\numerical_validation.json
```

The runner defaults to four BLAS threads and records its runtime. Offline
replays checkpoint per-cell memberships/scores and bind their configuration,
input and method hashes. Rerun the same command to resume; incompatible evidence
must use a new output directory. Private score arrays and expensive caches are
retained, not disposable files. A complete aggregate is written only after
all planned cells finish.

Without embeddings, build a clearly marked preparation report instead:

```powershell
.\.venv-v3\Scripts\python.exe scripts\build_imdb_report.py `
  --input outputs_imdb\cache\imdb-50000\readiness.json `
  --output outputs_imdb\reports\imdb-40-replay
.\.venv-v3\Scripts\python.exe .github\skills\sampling-experiment-report\quality_check.py `
  --html outputs_imdb\reports\imdb-40-replay\report.html `
  --screenshots outputs_imdb\reports\imdb-40-replay\validation_screenshots
.\.venv-v3\Scripts\python.exe scripts\validate_imdb_report.py `
  --html outputs_imdb\reports\imdb-40-replay\report.html `
  --screenshots outputs_imdb\reports\imdb-40-replay\validation_screenshots `
  --output outputs_imdb\reports\imdb-40-replay\interaction_validation.json
.\.venv-v3\Scripts\python.exe scripts\validate_imdb_experiment.py
.\.venv-v3\Scripts\python.exe -m pytest tests\test_imdb_inputs.py `
  tests\test_imdb_experiment.py tests\test_imdb_report.py -q
```

The HTML uses five tabs and no external assets. The independent numeric checker
hash-verifies the full retained run and real input cache, then recomputes all
requested metrics in five cohorts from every cell's retained scores. It also
checks source-label alignment, earlier-only donor/calibration positions and
point/lower nesting. Its aggregate-only record is `numerical_validation.json`.
Numerical summaries average
schedules within paired seeds before taking empirical 2.5/97.5 percentiles.
AUROC uses exact per-cell tied scores; displayed ROC curves interpolate onto
a shared FPR grid. Mean AUROC is not asserted equal to area under the mean
display curve. No winner or validated dimension cutoff is inferred in advance.

### Adding 256, 128 and 64 dimensions without rerunning existing cells

The additive runner uses the original **recorded** bootstrap draws, order and
timestamps, all 40 seeds, both schedules and all five budgets. It evaluates
only the three new normalized prefixes: **1,200 additional cells**. No native
or previous low-dimensional cell is recomputed, and there are no embedding or
judge calls. The preserved native embedding cache supplies all three prefixes.

The original run remains immutable at `outputs_imdb/private_runs/imdb-40-replay`.
The extension writes `outputs_imdb/private_runs/imdb-40-replay-extended`,
containing new evidence and a combined 3,600-cell aggregate. Old rows reference
their original evidence files; only relative evidence paths are rewritten.
Input, method and four-thread runtime hashes must match the original run.
The extension fails closed on mismatched inputs, incompatible runtime, or
altered retained artifacts, and `--resume` explicitly enables checkpoint reuse.

```powershell
.\.venv-v3\Scripts\python.exe scripts\extend_imdb_experiment.py `
  --baseline outputs_imdb\private_runs\imdb-40-replay `
  --output outputs_imdb\private_runs\imdb-40-replay-extended --resume
.\.venv-v3\Scripts\python.exe scripts\validate_imdb_experiment.py `
  --run outputs_imdb\private_runs\imdb-40-replay-extended
.\.venv-v3\Scripts\python.exe scripts\build_imdb_report.py `
  --input outputs_imdb\private_runs\imdb-40-replay-extended\aggregate.json `
  --output outputs_imdb\reports\imdb-40-replay `
  --numerical-validation outputs_imdb\reports\imdb-40-replay\numerical_validation.json
```

The same report URL is updated only after the full extension completes. Its
controls, plots, tables, paired deltas and conclusions then include nine
dimensions. The numeric checker verifies that every original measured row is
unchanged and every added row uses the exact original replay, before recomputing
metrics from the complete retained evidence.

## Matryoshka prefix-cutoff (earlier datasets)

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

For a shorter, user-friendly HTML report, generate a separate read-only derivative:

```powershell
.\.venv-v3\Scripts\python.exe scripts\build_matryoshka_summary_report.py `
  --input outputs_matryoshka\runs\mrl-three-datasets-30-seed-20260914\aggregate.json `
  --output outputs_matryoshka\reports\mrl-eight-dimensions-20260914\report.html
```

This view has three graphs: native versus 8d MAE, the full dimensionality sweep,
and a collapsible budget-level comparison. The second graph's **Y-axis**
dropdown switches between relative MAE change (%) and the measured MAE values
on a shared zero-based axis. Percentage is the default; changing the view
updates labels, tooltips and the caption without changing the experiment data.
It describes the native
`text-embedding-3-small` model and exact prefix-plus-L2-normalization operation.
Its no-drop-off conclusion is restricted to dataset-average end-to-end MAE;
budget, agent and other-metric regressions and untested datasets remain explicit.
The builder verifies source hashes, refuses to write inside the source run, and
does not call any cloud API. Use `--overwrite` only to rebuild this derivative.

### Fine-grained low-dimensional follow-up

The follow-up grid is **32, 30, 28, ..., 2**, plus the unchanged native
1,536-dimensional reference. It uses the same verified input manifests,
seeds 13-42, five schedules, five budgets, selection rules and both membership
modes as `mrl-three-datasets-30-seed-20260914`. The native reference is a
comparison control, not an extra shortened dimension. This is **76,500 cells**
over the same three available datasets; Tau2 remains excluded because its
expected-label mapping is still unavailable.

```powershell
$env:OPENBLAS_NUM_THREADS = "12"
$env:OMP_NUM_THREADS = "12"
$env:MKL_NUM_THREADS = "12"
.\.venv-v3\Scripts\python.exe scripts\run_matryoshka_experiment.py `
  --input historical_300=outputs_matryoshka\cache\live-20260914\historical_300\manifest.json `
  --input dense_2500=outputs_matryoshka\cache\dense_2500\manifest.json `
  --input cosmos_otel=outputs_matryoshka\cache\live-20260914\cosmos_otel\manifest.json `
  --dimensions 1536,32,30,28,26,24,22,20,18,16,14,12,10,8,6,4,2 `
  --output outputs_matryoshka\runs\mrl-low-dimensions-30-seed-20260915-paired
.\.venv-v3\Scripts\python.exe scripts\build_matryoshka_report.py `
  --input outputs_matryoshka\runs\mrl-low-dimensions-30-seed-20260915-paired\aggregate.json `
  --output outputs_matryoshka\runs\mrl-low-dimensions-30-seed-20260915-paired\report.html
.\.venv-v3\Scripts\python.exe scripts\build_matryoshka_summary_report.py `
  --input outputs_matryoshka\runs\mrl-low-dimensions-30-seed-20260915-paired\aggregate.json `
  --focus-dimension 2 `
  --output outputs_matryoshka\reports\mrl-low-dimensions-20260915\report.html
.\.venv-v3\Scripts\python.exe scripts\validate_matryoshka_run.py `
  --run outputs_matryoshka\runs\mrl-low-dimensions-30-seed-20260915-paired
.\.venv-v3\Scripts\python.exe scripts\validate_matryoshka_comparison.py `
  --run outputs_matryoshka\runs\mrl-low-dimensions-30-seed-20260915-paired `
  --baseline outputs_matryoshka\runs\mrl-three-datasets-30-seed-20260914
```

The concise report's `--focus-dimension` chooses a tested shortened endpoint
for its native comparison and budget breakdown; it defaults to 8 for older
reports. The trend chart always contains the complete configured grid.
Detailed chart axes, metric tables, replay intervals and budget tables also
follow that grid rather than the original eight dimension choices.

The comparison validator checks that **only dimensions changed** in the
protocol, inputs and controlling implementation hashes match, and all
overlapping 1,536/32/16/8-dimensional result cells, memberships and order hashes
reproduce the retained baseline exactly. This is a reproducibility check, not
independent evidence of prediction accuracy.

For exact baseline parity, use **Python 3.13.13 x64**, its retained pinned
`environment-requirements.txt`, and **12 OpenBLAS threads**. An initial
ARM64/one-thread numerical diagnostic preserved membership but changed some
float32 distances and a few predictions exactly at the 0.5 cutoff; it is not
the canonical result bundle. Thread count is therefore part of numerical
reproduction, not a performance-only setting.

### IDW threshold and conditional-envelope follow-up

The [threshold experiment](../docs/IDW_THRESHOLD_EXPERIMENT.md) replays the
native and 8d end-to-end memberships on the same three datasets and 4,500
settings. The [published refreshed report](../outputs_matryoshka/reports/idw-threshold-cosmos-refresh-20260915/report.html)
compares point-score and conditional-lower-envelope ROC curves, with separate
accuracy, precision, recall and F1 threshold plots. The primary comparison
uses identical envelope-eligible targets; mean/prior fallback scores are
shown separately. Calibration uses only earlier selected same-agent labels.
The envelope is an empirical sensitivity construction, not a confidence
interval or a validated threshold-selection rule.

### Refreshing a sensitive Cosmos snapshot

Use top-level `manifest.json` and `refresh_report.json` to bind the selected
local export; never silently read an older `_snapshots` backup. The September
15 export with catch-up cutoff `13:40:42 UTC` has 103,373 documents across ten
containers, but the unchanged experiment adapter uses **755 expected-label
units** from `labels`, linked to `spans`. Other evaluation/smoke label
containers are not pooled into this protocol.

The refreshed 755-unit input has 745 linked-span representations and ten
label-document fallbacks. All 205 earlier canonical packets and reference
labels are unchanged; only 550 new native vectors require preparation.
Reuse requires an exact canonical-packet hash and verified model/vector
provenance, not merely the same session identifier. Live preparation requires
explicit authorization and the same `text-embedding-3-small` model; never
substitute a different local model to make a nominally offline rerun succeed.

Keep full results under `outputs_matryoshka\private_runs\` and publish
aggregate-only reports. Native/8d reference runs can establish fresh
memberships for the ROC experiment while the 32-to-2 sweep runs independently.
Compare the full refreshed sweep against that same-input reference for exact
native/8d parity. When comparing with an older corpus, explicitly pass
`--refreshed-dataset cosmos_otel` to `validate_matryoshka_comparison.py`:
the other two datasets must still reproduce exactly, and the changed Cosmos
population is not subject to a false same-population parity claim.

```powershell
.\.venv-v3\Scripts\python.exe scripts\build_matryoshka_public_report.py `
  --private-run outputs_matryoshka\private_runs\cosmos-refresh-20260915-134042\dimension-sweep `
  --snapshot-readiness outputs_matryoshka\cache\cosmos-refresh-20260915-preflight\readiness.json `
  --focus-dimension 2 `
  --output outputs_matryoshka\reports\mrl-cosmos-refresh-20260915
```

This publication preserves plotted metrics and aggregate agent-regression
counts without retaining original identifiers. The protected source remains
unchanged, and the publication records its whole-artifact hashes. Comparisons
with earlier reports also reflect a changed cohort and label prevalence,
not only the embedding representation.

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
