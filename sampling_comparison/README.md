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
