# Repository layout

This repository is organized around reproducible experiment outputs, source code, and a small set of tracked review artifacts. The root remains intentionally flat for compatibility with existing notebooks and scripts that assume relative paths from the repo root.

## Root-level structure

- `scripts/` — operational automation and report builders for each sampling generation pass.
- `tests/` — focused pytest coverage for report generation and data-pipeline contracts.
- `docs/` — design notes and repository layout guidance.
- `random_sampling/`, `minhash_sampling/`, `sampling_comparison/`, `agent_uniform_sampling/` — the implementation packages for the historical experiments.
- `trace_sampling/` — full-session and Lipschitz-oriented trace sampling pipelines, analysis helpers, and related experimentation code.
- `outputs_sampling_v*/` — local output trees for each sampling iteration; they contain report bundles, cache data, and local evidence. Reviewed deliverables stay tracked, while local rerun artifacts remain untracked or curated via explicit ignore rules.
- `outputs_agent_uniform_sampling/` — canonical audit artifacts and reported evidence for the agent-uniform run. Keep the tracked JSON/HTML/PNG evidence here when it is referenced by the report manifest.
- `external_data/` and `test-results/` — downloaded benchmark inputs and browser-test evidence. These remain local-only.
- `synthetic_data/` and `outputs_assertions/` — supporting datasets and assertion plots. Only canonical plots and referenced outputs are committed.

## Canonical output expectations

- `outputs_agent_uniform_sampling/` stores final report outputs, manifest JSON, and the screenshot sets referenced by the final evidence bundle.
- `outputs_sampling_v6/runs/` keeps only reviewed, tracked deliverables. Raw run directories and cache-heavy scratch artifacts remain local.
- `outputs_sampling_v7/runs/` separates canonical HTML/JSON/JSONL and manifest-referenced evidence from disposable captures. Put new captures in `validation_screenshots/`. Existing run paths remain stable.
- `outputs_sampling_v*/cache/` and similar expensive caches remain in place; the cleanup utility never removes them.
- `outputs_matryoshka/runs/` retains prefix-cutoff aggregates, membership evidence,
  manifests and HTML reports. Per-run `checkpoints/` and `validation_screenshots/`
  are local scratch. `outputs_matryoshka/cache/` retains expensive vectors and
  source-bound preparation evidence locally; it is not a disposable screenshot
  directory and is never covered by blanket cleanup.
- `outputs_matryoshka/reports/` holds concise derivative HTML reports, plotted
  summaries, source/output hash manifests and validation records. Their builders
  leave canonical run bundles untouched. New screenshots stay in each report's
  local `validation_screenshots/` directory.
- IDW threshold/envelope follow-ups also use `outputs_matryoshka/runs/`.
  Retain the report, threshold/ROC summaries, compressed per-target score
  evidence and provenance/validation manifests; these are reproducibility
  artifacts, not disposable screenshots. Keep input embedding caches local
  and distinguish them from retained derived score arrays.
  The pre-refresh `idw-threshold-envelope-20260915-verified` and
  `mrl-low-dimensions-30-seed-20260915-paired` bundles are preserved locally
  and Git-ignored, along with their superseded report/runtime-audit derivatives.
  They are not published because they include identifiers or per-target evidence.
  Existing historical artifacts already tracked before this branch remain
  unchanged; no history rewriting or blanket deletion is performed.
- **Sensitive refreshed snapshots use a different publication boundary.**
  `outputs_matryoshka/private_runs/` is Git-ignored and retains original
  identifiers, memberships, per-target predictions, private aggregates and
  complete scientific evidence locally. Native vectors and source-bound
  preparation metadata stay in the ignored cache tree. These are retained
  evidence, not disposable scratch.
- For those runs, `outputs_matryoshka/reports/` contains only allowlisted
  aggregate HTML/JSON, numerical validation summaries and whole-artifact
  hashes. Do not publish raw telemetry, credentials, endpoint configuration,
  session/task/agent identifiers, per-unit hashes or per-target score arrays.
  Aggregate agent-regression counts may be published without identities.
  Keep browser captures inside each publication's `validation_screenshots/`.
  Authorization for necessary Azure embedding requests does not authorize
  uploading the raw snapshot or private run bundle to GitHub.

## Screenshot cleanup policy

The cleanup utility is deliberately conservative:

- It only archives untracked, unreferenced PNGs under `outputs_sampling_v7/`: the `validation_screenshots/` and `storytelling-validation/` directories, plus filenames starting with `report-`, `pca8-audit-`, `pca8-story-`, `pca8-final-`, or `pca8-distance-report-`. Canonical `pca8-distance-audit-*` and `pca8-section-*` images are retained.
- It refuses to delete tracked files and also refuses to delete any PNG referenced by eligible repository text sources (for example tracked root notebooks, Python modules, docs, and output manifests).
- It refuses symlinks and Windows reparse-point paths (including guarded parent ancestry checks for source and archive paths).
- It requires an archive directory outside the repo before it removes anything. The archive path must be absolute and pass ancestor reparse-point checks before and after resolution. The archive preserves relative paths so the files can be restored later.
- It performs preflight destination validation first (no collisions, no overwrite), copies and verifies all candidates by SHA256, and writes a durable manifest before unlinking originals.
- It persists manifest progress during unlink so a partial failure still keeps complete archive metadata and per-file status for recovery.
- It never touches final HTML/JSON/JSONL/manifest outputs or the canonical audit PNGs that the report pipeline explicitly references.
- It never removes cache trees.

## Archive runbook

Run a dry-run inventory from the repo root:

```bash
python scripts/cleanup_sampling_artifacts.py --repo-root .
```

To archive and remove confirmed scratch PNGs, use an archive directory outside the repository:

Windows absolute-path example from the repository CWD (use a fresh archive name):

```powershell
$archive = Join-Path $env:USERPROFILE ("sampling-artifact-archive-" + [guid]::NewGuid().ToString("N"))
.\.venv-v3\Scripts\python.exe scripts\cleanup_sampling_artifacts.py --repo-root . --apply --archive-dir $archive
```

The expected operator flow is from the repository CWD with an archive path outside the repository root. The archive is not committed to the repo. It is a recovery bundle for local cleanup, not a published artifact.

Manual restoration is supported from the preserved manifest:

1. Read `cleanup_sampling_artifacts_manifest.json` from the archive root.
2. For each file entry, copy `archive_rel_path` back to `original_rel_path` under the repo root.
3. Keep the manifest in the archive as the source-of-truth recovery ledger.

## Notes

- Do not hide final report assets or reproducibility bundles with broad ignore rules.
- Keep `.gitignore` narrow and specific to clearly temporary validation screenshot directories.
- This repo intentionally does not rewrite history or promise a Git-pack size reduction from removing local scratch artifacts.
