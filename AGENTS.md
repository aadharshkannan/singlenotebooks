# Repository Agent Guidance

- Start with [README.md](README.md) and [sampling context](docs/SAMPLING_CONTEXT.md).
  Distinguish team direction, implemented behavior, and measured results.
- For sampling experiment reports, load
  [.github/skills/sampling-experiment-report/SKILL.md](.github/skills/sampling-experiment-report/SKILL.md).
  Use its template and scientific/rendering gates for HTML and PDF deliverables.
- Preserve the distinction between direct judgments, imputed values, fallbacks,
  conditional Lipschitz sensitivity bounds, and replay uncertainty. Never claim
  the weekly threshold, integrated bounds, or future judge metrics are validated
  merely because they appear in the team's preferred design.
- Keep experiments offline unless a live run is explicitly requested. Report
  generation from retained artifacts does not require embedding or judge calls.
- Use the existing package abstractions and focused pytest tests. The local
  Windows interpreter, when present, is `.venv-v3/Scripts/python.exe`.
- Follow [the layout and retention policy](docs/REPOSITORY_LAYOUT.md). Preserve
  user notebook edits, raw inputs, expensive caches, and canonical run manifests.
  Do not move notebooks/run bundles without checking their relative paths.
- Keep browser screenshots in a run's `validation_screenshots/` directory. Use
  the dry-run-first archive tool for disposable screenshots; never blanket-delete
  images or rewrite Git history as routine cleanup.