---
name: sampling-experiment-report
description: 'Produce or review sampling experiment HTML and PDF reports in this repository. Use for method comparisons, dataset provenance, intuitive visual results, scientific analysis, report generation, and browser/print quality checks.'
---

# Sampling Experiment Report

## Start Here

Read [sampling context](../../../docs/SAMPLING_CONTEXT.md) for the team's goal,
method lineage, measured evidence, and unresolved policy decisions. Follow
[the layout policy](../../../docs/REPOSITORY_LAYOUT.md) for retained artifacts.
Use [the report template](./references/report-template.md) and
[acceptance checklist](./references/acceptance-checklist.md) as the deliverable contract.

## Workflow

1. Establish the audience, decision, exact run/version, primary metric, and required
   formats. If unspecified, use a technical team audience and HTML, explicitly
   naming the chosen run. Ask only when missing inputs block honest analysis.
2. Inspect the run manifest, aggregate, dataset profile, label provenance, and
   owning method code. Record missing artifacts, source hashes, dimensions,
   budget units, selected/imputed counts, and repeat design before drafting.
3. Explain every compared method, not just the preferred one. Separate evidence
   representation, selection, judging, imputation, and aggregation. Include one
   method diagram and a clearly illustrative IDW example when relevant.
4. Build the narrative using the template: question, source, methods, data, results,
   analysis, limitations, and validation. Use comparable charts with denominators,
   units, metric direction, and a short takeaway next to each.
5. Reuse the matching version's generator and tests. Do not feed older artifacts
   to V7 builders. For another generation, locate its corresponding
   `scripts/build_sampling_*_report.py` and `sampling_comparison/*_report.py`.
6. Validate numeric claims against machine-readable artifacts, render HTML, run
   the local checker, inspect screenshots, and check all interactive controls.
   Export and inspect actual PDF pages only when PDF is requested.
7. Deliver artifact links, the observed tradeoff, the exact validation performed,
   and unresolved limitations. A failed required gate is not a completed report.

## Scientific Boundaries

- Direct judgments, expected/reference labels, imputed values, and fallback/prior
  scores must be distinguishable. A live embedding/Search run is not a live judge run.
- V7 tests transductive per-corpus PCA-8 binary IDW, not integrated Lipschitz bounds.
  The separate value prototype's conditional bounds are sensitivity envelopes,
  not confidence intervals. Recalibrate when geometry or evaluator changes.
- Bootstrap replays reuse the same labels. Replay intervals are not evidence of
  production generalization or independent judge reliability.
- Semantic/concept coverage is a defined cohort-relative proxy, not guaranteed
  coverage of all activities. Show per-agent and per-dataset gaps when available.
- Separate token budgets from session budgets, macro from micro aggregation,
  probability means from binary predictions, and imputed-only from combined metrics.
- The <=20 sessions/agent/week gate, high-traffic tenant anecdote, and additional
  evaluator metrics are planning context unless the chosen run actually tests them.
- Do not suppress negative results or announce a universal winner. Coverage,
  aggregate accuracy, and individual-session prediction can favor different methods.

## Safe Generation

Both V7 builders can update the manifest beside the **input aggregate**, even
when `--output` points elsewhere. The rich builder also writes `final_report.json`
and `print_report.html` beside its output. Preserve existing bundles by default.

This PowerShell example uses a specific historical run, not an implicit latest
default, and a fresh scratch directory. Existing manifest references may still
point to the original evidence: preserve that provenance and identify generated
artifacts as a derivative. Do not silently rewrite source hashes.

```powershell
$run = "outputs_sampling_v7/runs/v7-live-repeated-20260903"
$scratch = Join-Path $env:TEMP ("sampling-report-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $scratch | Out-Null
Copy-Item -Recurse "$run\*" $scratch

.\.venv-v3\Scripts\python.exe scripts\build_sampling_v7_report.py `
  --input "$scratch\aggregate.json" --output "$scratch\interactive_report.html"
.\.venv-v3\Scripts\python.exe scripts\build_sampling_v7_distance_report.py `
  --input "$scratch\aggregate.json" --output "$scratch\pca8-distance-report.html"
```

Adapt the rendered narrative in the owning generator, not only in generated HTML.
Use the corresponding report tests before generating the final derivative bundle.
Do not rerun embeddings, Search sync, or LLM judging just to rebuild a report.

## Browser and PDF Gate

Use [quality_check.py](./quality_check.py) with the chosen interpreter. The existing
Windows environment is `.venv-v3`; on another machine use its configured Python.
If absent, install Playwright in that environment (`python -m pip install playwright`)
and a browser (`python -m playwright install chromium`). Windows auto mode tries
Edge first. HTML-only checks do not require a print source.

```powershell
.\.venv-v3\Scripts\python.exe .github\skills\sampling-experiment-report\quality_check.py `
  --html "$scratch\interactive_report.html" `
  --screenshots "$scratch\validation_screenshots"

.\.venv-v3\Scripts\python.exe .github\skills\sampling-experiment-report\quality_check.py `
  --html "$scratch\interactive_report.html" --print-html "$scratch\print_report.html" `
  --pdf "$scratch\sampling-report.pdf"
```

The checker blocks external network access by default and checks desktop/mobile
HTML for content, broken assets, page errors, horizontal overflow, and text clipping.
It allows genuine scrollable tables, not clipped content. PDF requires an explicit
static print source, fixed A4 portrait/10mm margins, and a print-content-width gate.
Existing PDFs are protected unless `--overwrite-pdf` is explicitly authorized.

These checks do not prove scientific correctness or pixel-perfect pagination.
Inspect desktop/mobile screenshots and PDF pages for chart/label overlap, page
breaks, missing charts, legibility, and numeric parity. Custom page formats require
an explicitly adapted validation/export contract. Never treat PDF magic bytes as
proof of content quality. Keep new captures in `validation_screenshots/` and
preserve canonical manifest-referenced evidence.