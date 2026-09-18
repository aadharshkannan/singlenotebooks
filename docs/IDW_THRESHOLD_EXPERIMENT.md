# Full-session IDW threshold and conditional-envelope experiment

This document is the pre-registration and compact scientific record for
`idw-threshold-envelope-20260915`. Its initial clean-start bundle is retained
locally at `outputs_matryoshka/runs/idw-threshold-envelope-20260915-verified/`,
not published because it includes identifiers and per-target evidence.
The current published deliverable is the
[refreshed aggregate-only report](../outputs_matryoshka/reports/idw-threshold-cosmos-refresh-20260915/report.html).

## Pre-registration (recorded before the confirmatory run)

- **Decision/question.** On the existing three expected-label corpora, describe
  ordinary binary classification of causal full-session IDW scores across
  thresholds, and compare it with thresholding the lower side of a conditional
  Lipschitz sensitivity envelope.
- **Cohort.** Historical 300, Dense 2,500, and Cosmos OTEL 205. Tau2 is excluded
  because it has no approved expected-label mapping. Selected/direct observations
  are excluded from primary quality metrics.
- **Fixed design.** Reuse and hash-verify the `end_to_end` memberships from
  `mrl-three-datasets-30-seed-20260914` for dimensions 1,536 and 8, seeds 13–42,
  five schedules, and session rates 1%, 2%, 5%, 10%, and 20%. Budgets use
  `max(1, floor(N * rate))`. Prefixes are re-normalized; there is no PCA.
- **Point estimator.** Existing `distance_blocks` and `evaluate` implement angular
  same-agent causal IDW (`k=8`, power 2, epsilon `1e-6`), averaging all exact
  earlier donors when `1-cosine <= 1e-8`; otherwise the earlier global observed
  mean or 0.5 prior is used. Selection is label-blind full-schedule ranking, not
  strictly online admission.
- **Envelope.** At each target, calibrate an empirical 90th-percentile Lipschitz
  slope from pairs of earlier selected labels for that target's agent in the same
  normalized angular geometry. Distance denominators have a 0.01 floor and sparse
  calibration uses configured `L=1`. The lower value is
  `clamp(IDW - L * IDW-weighted donor distance, 0, 1)`. Fallback point estimates
  have no envelope. This observed-label oracle calibration is a deterministic
  sensitivity analysis, not a confidence interval or deployable calibration.
- **Confirmatory hypotheses and acceptance thresholds.**
  1. Baseline replay succeeds only if every cell has identical threshold-0.5
     accuracy and absolute MAE difference at most `1e-12`.
  2. On every matched eligible cohort and threshold, lower-bound positives are a
     subset of point-score positives; therefore its false-positive rate and recall
     cannot exceed the point score (floating tolerance `1e-12`).
  3. Source membership/order hashes and all input-cache artifact hashes must match
     before and after the run.
- **Exploratory analyses.** AUROC, accuracy, precision, recall, and F1 differences
  across datasets, dimensions, budgets, schedules, agents, and thresholds. No
  threshold selected on these labels is claimed deployable. Eight-dimensional
  superiority or non-inferiority was not pre-specified.
- **Metrics.** Exact tied-score ROC with all-negative/all-positive endpoints and
  AUROC when both classes exist; threshold-grid accuracy, precision, recall, and
  F1 for `score >= threshold`. Precision is recorded as zero when no positives are
  predicted; F1 is zero in that case when the positive class exists. Empty and
  one-class undefined metrics are null with reasons.
- **Retained-score precision.** Point scores and envelope fields are serialized
  as float32, and the ROC/threshold analysis uses those retained values.
  Exact ROC means exact tied-score breakpoints for this representation, not
  unrounded float64 estimator values. Baseline estimator replay is checked before
  serialization; all-unjudged classifications at 0.5 were also compared with
  the baseline after serialization, with no mismatching cells.
- **Aggregation and uncertainty.** Primary point-versus-envelope comparisons use
  the identical envelope-eligible target occurrences. All-unjudged point scores,
  including global-mean/prior fallbacks, are separate. Pooled curves treat replay
  occurrences as repeated observations; equal-cell summaries weight each
  dataset/dimension/budget/seed/schedule cell equally. Student-t intervals over 30
  seed means are replay-sensitivity summaries, not population confidence
  intervals and not independent-label uncertainty.
- **Stopping rule and budget.** Stop successfully after all 4,500 planned cells
  and validations complete; fail closed on hash, temporal, baseline-replay, or
  cohort-pairing violations. Maximum recorded budget: two CPU-hours, 500 MiB of
  new run artifacts, zero network/API calls, and $0 external cost.

### Pre-analysis amendment after failed technical replay

The second startup attempt stopped before producing result artifacts when one
Historical cell differed from the retained baseline MAE by
`1.407873817527161e-12`, while accuracy was exactly equal. The executed
`matryoshka_experiment.py` SHA256 matched the baseline manifest. This is just over
the registered `1e-12` bound and is consistent with float32 matrix-product
rounding under a different BLAS execution configuration. Before examining any
threshold or envelope result, the execution compatibility tolerance was first
amended to `1e-10`. A third startup attempt stopped at `2.0293569047424853e-9`
for the same native seed/schedule at the 20% budget. A later Dense cell exposed
a larger `4.6127053828337594e-5` MAE difference despite identical accuracy,
F1, confusion counts, and provenance counts. This is consistent with
platform/BLAS-sensitive float32 dot products changing continuous IDW weights
without changing classifications. The final execution tolerance was set to
`1e-4` at `2026-09-15T09:54:24.9145696Z`. A subsequent Dense cell had one of
2,450 unjudged predictions cross the exact 0.5 boundary (accuracy difference
`0.0004081633`) while MAE differed by only `5.15e-9`. The final compatibility
rule therefore allows absolute accuracy difference at most `0.001`, while
recording every mismatched cell and the original exact criterion as failed.
This was fixed at `2026-09-15T09:58:55.7710295Z`, still before any result
artifact was written or threshold/envelope result inspected. All startup
failures remain recorded separately.

### Runtime correction before finalization

After the first complete run, the baseline environment record was identified as
Python 3.13.13 AMD64 with NumPy 2.5.3, while that run used Python 3.11 ARM64.
The Python 3.11 output was retained under
`runtime_variants/py311-arm64/` rather than discarded. A targeted check with the
original Python 3.13 environment and default BLAS threading exactly reproduced
the previously discrepant baseline cells; forcing one thread still moved one
score across 0.5. The root cause was subsequently confirmed: the baseline used
OpenBLAS with 12 threads. The Python 3.13 single-thread output is retained under
`runtime_variants/py313-amd64-single-thread/`, and the final canonical run uses
Python 3.13.13 AMD64, NumPy 2.5.3 and 12 threads for OpenBLAS, OMP, MKL, and
NumExpr. `threadpool_info()` is recorded in the aggregate. The canonical run
restores the original exact MAE/accuracy acceptance criteria; earlier numerical
tolerance amendments remain visible only as part of the failed/superseded-run
audit trail.

### Clean-start reproduction check

Final code inspection found that canonical acceptance keys were available in
the amended run's pre-registration but missing from the defaults for a fresh
output directory. The defaults now explicitly require zero MAE and accuracy
replay differences in the canonical runtime. A fresh-directory regression test
and the separate `-verified` run exercise startup without copying or manually
editing an existing pre-registration. This correction does not change the
point estimator, calibration, threshold metrics or cohort membership.

Superseded runtime bundles remain local at the original
`outputs_matryoshka/runs/idw-threshold-envelope-20260915/` path. Their compact
failure/correction and archive metadata are also retained locally under
`outputs_matryoshka/reports/idw-runtime-audit-20260915/`; their duplicate large
arrays and identifier-bearing bundles are not committed.

## Reproduction

From the repository root with cached inputs already present:

```powershell
$env:OPENBLAS_NUM_THREADS="12"
$env:OMP_NUM_THREADS="12"
$env:MKL_NUM_THREADS="12"
$env:NUMEXPR_NUM_THREADS="12"
.\.venv-v3\Scripts\python.exe scripts\run_idw_threshold_experiment.py `
  --output outputs_matryoshka\runs\idw-threshold-envelope-20260915-verified
.\.venv-v3\Scripts\python.exe scripts\build_idw_threshold_report.py `
  --input outputs_matryoshka\runs\idw-threshold-envelope-20260915-verified\aggregate.json `
  --output outputs_matryoshka\runs\idw-threshold-envelope-20260915-verified\report.html
```

The environment must match the retained Python 3.13.13 AMD64 requirements; the
Python 3.11 ARM64 interpreter is not baseline-compatible for this numerical run.

The HTML report contains the measured results, limitations, artifact hashes,
commands, and validation status. It does not validate the separate integrated
value pipeline or the proposed weekly policy.

## September 15 refreshed Cosmos snapshot

The refreshed [aggregate-only report](../outputs_matryoshka/reports/idw-threshold-cosmos-refresh-20260915/report.html)
uses the export with catch-up cutoff **2026-09-15 13:40:42 UTC**. The unchanged
labels-plus-spans adapter yields **755 expected-label units**, not 103,373
evaluated sessions: the larger number counts documents across all ten exported
containers. The 755 units comprise 745 linked-span representations and ten
label-document fallbacks; 725 bounded packets are truncated. Good=1 gives
179 positive references, with bad/partial mapped to zero. Other evaluation
label containers are not pooled into the reference.

The user authorized the one-time Azure embedding preparation: 205 existing
canonical-packet/vector matches were reused bit-for-bit, and 550 new native
`text-embedding-3-small` vectors were generated using 48 successful HTTP
requests and 4,494,400 reported input tokens. This cost belongs to preparation,
not the offline replay; no new judge labels were collected.

Use the restored Python 3.13.13 x64 / NumPy 2.5.3 environment and 12 threads
for OpenBLAS, OMP, MKL and NumExpr. First establish the native/8d reference
using the existing Matryoshka runner with the refreshed Cosmos input and
unchanged Historical/Dense inputs. The threshold CLI accepts an explicit
input override and rejects a baseline bound to a different input manifest.
The validator loads input and baseline paths from the run's recorded provenance,
not global defaults.

```powershell
$env:OPENBLAS_NUM_THREADS="12"; $env:OMP_NUM_THREADS="12"
$env:MKL_NUM_THREADS="12"; $env:NUMEXPR_NUM_THREADS="12"
.\.venv-v3\Scripts\python.exe scripts\run_idw_threshold_experiment.py `
  --input cosmos_otel=outputs_matryoshka\cache\cosmos-refresh-20260915-134042\manifest.json `
  --baseline outputs_matryoshka\private_runs\cosmos-refresh-20260915-134042\reference-1536-8 `
  --output outputs_matryoshka\private_runs\cosmos-refresh-20260915-134042\threshold-envelope
.\.venv-v3\Scripts\python.exe scripts\validate_idw_threshold_experiment.py `
  --input outputs_matryoshka\private_runs\cosmos-refresh-20260915-134042\threshold-envelope\aggregate.json `
  --output outputs_matryoshka\private_runs\cosmos-refresh-20260915-134042\threshold-envelope\score_consistency_validation.json
.\.venv-v3\Scripts\python.exe scripts\build_idw_threshold_public_report.py `
  --private-run outputs_matryoshka\private_runs\cosmos-refresh-20260915-134042\threshold-envelope `
  --output outputs_matryoshka\reports\idw-threshold-cosmos-refresh-20260915
```

Run commands require fresh output directories. Full memberships, per-target
score/ROC arrays, original agent identifiers and input manifests remain in
ignored private storage. Public HTML/JSON allowlists numeric summaries and
whole-artifact hashes; per-agent changes appear only as aggregate counts and
gap statistics, not identifier-bearing rows. Historical and Dense are
unchanged controls. Differences from the earlier Cosmos results reflect
a different cohort and label prevalence as well as the tested representation;
they are not a causal improvement attributed to the embedding model.
