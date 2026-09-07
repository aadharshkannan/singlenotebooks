from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")


def _label_for_metric(metric_id: str) -> str:
    labels = {
        "aggregate_pass_rate_mae": "Aggregate MAE",
        "selected_only_pass_rate_mae": "Selected-only MAE",
        "embedding_imputed_f1": "Embedding imputed F1",
        "imputed_only_f1": "Imputed-only F1",
        "coverage_ratio": "Coverage ratio",
    }
    return labels.get(str(metric_id), str(metric_id))


def _value_for_method(rows: Sequence[Mapping[str, Any]], method_id: str, key: str) -> float:
    for row in rows:
        if str(row.get("method_id")) == str(method_id):
            value = row.get(key)
            if value is None:
                for metric in ("mean", "value"):
                    if row.get(metric) is not None:
                        value = row.get(metric)
                        break
            if value is not None:
                return _as_float(value)
    return 0.0


def _best_method_by(rows: Sequence[Mapping[str, Any]], metric_key: str, lower_is_better: bool = True) -> tuple[str, float, float]:
    candidates = []
    for row in rows:
        method_id = str(row.get("method_id") or row.get("method") or "unknown")
        value = _as_float(row.get(metric_key), 0.0)
        if value is None:
            continue
        candidates.append((method_id, value))
    if not candidates:
        return ("n/a", float("nan"), 0.0)
    if lower_is_better:
        method_id, value = min(candidates, key=lambda item: item[1])
        return (method_id, value, value)
    method_id, value = max(candidates, key=lambda item: item[1])
    return (method_id, value, value)


def _mean(values: Iterable[Any]) -> float:
    items = [float(v) for v in values if v is not None]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _percent(value: float) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if abs(f) < 0.0001:
        return "0.000"
    return f"{f:.{digits}f}"


def _safe_text(value: Any) -> str:
    text = str(value) if value is not None else "n/a"
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _normalise_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            out.append(dict(row))
    return out


def _metric_rows_for_aggregate(rows: Sequence[Mapping[str, Any]], metric_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("metric_id") or "") == str(metric_name):
            out.append(dict(row))
    return out


def _agg_summary(rows: Sequence[Mapping[str, Any]], metric_key: str) -> tuple[str, float, float]:
    metric_rows = [row for row in rows if str(row.get("metric_id") or "") == str(metric_key)]
    if not metric_rows:
        return ("n/a", 0.0, 0.0)
    best = min(metric_rows, key=lambda item: _as_float(item.get("mean"), 1e9))
    return str(best.get("method_id") or "n/a"), _as_float(best.get("mean"), 0.0), _as_float(best.get("p95"), 0.0)


def _paired_delta_text(rows: Sequence[Mapping[str, Any]], selected_metric: str) -> tuple[str, str, str]:
    matches = [row for row in rows if str(row.get("metric_id") or "") == str(selected_metric)]
    if not matches:
        return ("n/a", "n/a", "n/a")
    wins = sum(1 for row in matches if _as_float(row.get("win_rate"), 0.0) > 0.5)
    ties = sum(1 for row in matches if abs(_as_float(row.get("win_rate"), 0.0) - 0.5) < 1e-9)
    losses = sum(1 for row in matches if _as_float(row.get("win_rate"), 0.0) < 0.5)
    mean_delta = _mean(_as_float(row.get("mean_delta"), 0.0) for row in matches)
    direction = "lower-better" if selected_metric.endswith("mae") else "higher-better"
    return (
        f"{wins}/{len(matches)} wins",
        f"delta = cosine - euclidean ≈ {_format_float(mean_delta, 3)}; negative favors cosine for lower-better metrics and positive favors cosine for higher-better metrics.",
        direction,
    )


def _derive_exec_readout(rows: Sequence[Mapping[str, Any]], aggregate_rows: Sequence[Mapping[str, Any]], paired_rows: Sequence[Mapping[str, Any]], datasets: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "No run-level artifact rows are currently loaded; the report is intentionally blank until aggregate data is available."

    aggregate_metric_key = "aggregate_pass_rate_mae"
    if aggregate_rows:
        best_method, best_mean, _ = _agg_summary(aggregate_rows, aggregate_metric_key)
    else:
        best_method, best_mean, _ = _best_method_by(rows, "aggregate_pass_rate_mae", True)

    concept_rows = [row for row in rows if isinstance(row.get("coverage"), Mapping) and isinstance((row.get("coverage") or {}).get("concept"), Mapping)]
    best_concept = "n/a"
    best_concept_value = 0.0
    for row in concept_rows:
        coverage = row.get("coverage") or {}
        concept = coverage.get("concept") or {}
        value = _as_float(concept.get("coverage_ratio"), 0.0)
        if value > best_concept_value:
            best_concept = str(row.get("method_id") or "n/a")
            best_concept_value = value

    if paired_rows:
        pair_metric = "aggregate_pass_rate_mae"
        wins = sum(1 for row in paired_rows if str(row.get("metric_id") or "") == pair_metric and _as_float(row.get("win_rate"), 0.0) > 0.5)
        total = sum(1 for row in paired_rows if str(row.get("metric_id") or "") == pair_metric)
        pair_win_rate = (wins / total) if total else 0.0
    else:
        pair_win_rate = 0.0

    cos = next((row for row in rows if str(row.get("method_id")) == "pca8_idw_binary_cosine"), None)
    euc = next((row for row in rows if str(row.get("method_id")) == "pca8_idw_binary_euclidean"), None)
    if cos is not None and euc is not None:
        cos_mae = _as_float(cos.get("aggregate_pass_rate_mae"), 0.0)
        euc_mae = _as_float(euc.get("aggregate_pass_rate_mae"), 0.0)
        cos_f1 = _as_float(((cos.get("embedding_metrics") or {}).get("imputed_only") or {}).get("f1"), 0.0)
        euc_f1 = _as_float(((euc.get("embedding_metrics") or {}).get("imputed_only") or {}).get("f1"), 0.0)
        tradeoff = f"Cosine mean MAE {_format_float(cos_mae, 3)} vs Euclidean {_format_float(euc_mae, 3)}; imputed F1 {_format_float(cos_f1, 3)} vs {_format_float(euc_f1, 3)}."
    else:
        tradeoff = "Cosine vs Euclidean tradeoff is n/a because only one PCA distance arm is available in this artifact set."

    dataset_count = len(datasets) if datasets else max(1, len({str(row.get("dataset_id")) for row in rows if row.get("dataset_id") is not None}))

    return (
        f"Best fixed-source aggregate MAE method overall is {best_method} at mean {_format_float(best_mean, 3)}. "
        f"Best concept coverage method is {best_concept} at coverage {_percent(best_concept_value)}. "
        f"Cosine-vs-Euclidean tradeoff: {(pair_win_rate * 100):.1f}% of paired aggregate MAE comparisons favor cosine across {dataset_count} dataset views, and {tradeoff} "
        "Estimator families differ, so selected-only MAE is the apples-to-apples selector view when ranking the arms."
    )


def build_v7_report_html(payload: Mapping[str, Any]) -> str:
    rows = _normalise_rows(payload.get("runs") or [])
    datasets = _normalise_rows(payload.get("datasets") or [])
    aggregate_rows = _normalise_rows((payload.get("aggregate_metrics") or {}).get("rows") or [])
    paired_rows = _normalise_rows((payload.get("paired_differences") or {}).get("rows") or [])
    paired_cells = _normalise_rows((payload.get("paired_differences") or {}).get("cells") or [])
    summary = dict(payload.get("summary") or {})

    dataset_ids = [str(dataset.get("dataset_id")) for dataset in datasets if dataset.get("dataset_id") is not None]
    if not dataset_ids:
        dataset_ids = [str(row.get("dataset_id")) for row in rows if row.get("dataset_id") is not None]
    dataset_id_values = sorted(set(dataset_ids))
    budget_values = sorted({str(row.get("budget_pct")) for row in rows if row.get("budget_pct") is not None}, key=lambda item: float(item))
    repetition_values = sorted({str(row.get("repetition_index")) for row in rows if row.get("repetition_index") is not None}, key=lambda item: float(item))
    run_count = len(rows)
    dataset_count = len(dataset_id_values)
    summary_lines = []
    if summary.get("run_count") is not None:
        summary_lines.append(f"{summary.get('run_count')} runs")
    if summary.get("dataset_count") is not None:
        summary_lines.append(f"{summary.get('dataset_count')} corpora")
    if summary.get("repetition_count") is not None:
        summary_lines.append(f"{summary.get('repetition_count')} replays")
    if summary.get("budget_count") is not None:
        summary_lines.append(f"{summary.get('budget_count')} budgets")
    if summary.get("base_seed") is not None:
        summary_lines.append(f"base seed {summary.get('base_seed')}")
    if not summary_lines:
        summary_lines.append(f"{run_count} runs")
        summary_lines.append(f"{dataset_count} corpora")

    aggregate_metric_key = "selected_only_pass_rate_mae"
    metric_names = ["aggregate_pass_rate_mae", "selected_only_pass_rate_mae", "embedding_imputed_f1"]
    fallback_metric = aggregate_metric_key if any(row.get("selected_only_pass_rate_mae") is not None for row in rows) else "aggregate_pass_rate_mae"
    selected_metric = fallback_metric

    if rows:
        mean_selected = _mean(row.get("selected_only_pass_rate_mae") for row in rows)
        mean_aggregate = _mean(row.get("aggregate_pass_rate_mae") for row in rows)
        mean_tokens = _mean(row.get("actual_token_count") for row in rows)
    else:
        mean_selected = 0.0
        mean_aggregate = 0.0
        mean_tokens = 0.0

    best_method_name, best_method_mean, _ = _agg_summary(aggregate_rows, "aggregate_pass_rate_mae") if aggregate_rows else _best_method_by(rows, "aggregate_pass_rate_mae", True)
    concept_best_name = "n/a"
    concept_best_value = 0.0
    for row in rows:
        concept = (row.get("coverage") or {}).get("concept") or {}
        value = _as_float(concept.get("coverage_ratio"), 0.0)
        if value > concept_best_value:
            concept_best_value = value
            concept_best_name = str(row.get("method_id") or "n/a")

    exec_readout = _derive_exec_readout(rows, aggregate_rows, paired_rows, datasets)

    dataset_provenance = []
    for dataset in datasets:
        dataset_id = str(dataset.get("dataset_id") or "unknown")
        pca = dataset.get("pca") or {}
        explained_ratio = pca.get("explained_variance_ratio") or []
        total_var = _as_float(pca.get("explained_variance_total"), 0.0)
        embedded_model = (dataset.get("shared_preprocessing") or dataset.get("runtime_ledger") or {}).get("embedding_model_id") or "n/a"
        deployment = (dataset.get("shared_preprocessing") or dataset.get("runtime_ledger") or {}).get("embedding_deployment_id") or "n/a"
        dataset_provenance.append(
            f"<li><strong>{_safe_text(dataset_id)}</strong>: population={_safe_text(dataset.get('population', 'n/a'))}, PCA-8 total variance captured={_format_float(total_var, 3)}, Foundry embedding model={_safe_text(embedded_model)}, deployment={_safe_text(deployment)}, explained_variance_ratio={_safe_text(explained_ratio[:8])}.</li>"
        )
    if not dataset_provenance:
        dataset_provenance.append("<li>No dataset provenance was available in the artifact bundle.</li>")

    study_text = (
        "This is not a sequential flow; the five arms are evaluated in parallel against the same replay plan, and they share the same outcome ledger and fixed corpus preprocessing. "
        "Three source corpora contribute the full-session embeddings, MinHash packets, and PCA fit once; the replay plan then assigns a shared bootstrap reordering for each repetition. "
        "The five parallel arms are Random, MinHash LSH, PCA-8 cosine + binary IDW, PCA-8 Euclidean + binary IDW, and ARM5 v6 membership + Hajek. They converge into the common downstream evaluation outputs: fixed-source MAE, selected-only MAE, concept coverage, classification metrics, tokens, and latency."
    )

    summary_text = " ".join(summary_lines)
    report_payload = {
        "rows": rows,
        "datasets": datasets,
        "aggregateMetrics": aggregate_rows,
        "pairedDifferences": paired_rows,
        "pairedCells": paired_cells,
        "summary": summary,
    }

    detail_rows: list[str] = []
    for row in rows[:12]:
        dataset_id = str(row.get("dataset_id") or "n/a")
        method_id = str(row.get("method_id") or "n/a")
        coverage = (row.get("coverage") or {}).get("concept") or {}
        concept_value = _as_float(coverage.get("coverage_ratio"), 0.0)
        emb = (row.get("embedding_metrics") or {}).get("imputed_only") or {}
        detail_rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                _safe_text(dataset_id),
                _safe_text(row.get("seed", "n/a")),
                _safe_text(row.get("budget_pct", "n/a")),
                _safe_text(method_id),
                _format_float(row.get("aggregate_pass_rate_mae"), 3),
                _format_float(row.get("selected_only_pass_rate_mae"), 3),
                _format_float(concept_value, 3),
                _format_float(emb.get("f1"), 3),
            )
        )
    if not detail_rows:
        detail_rows.append("<tr><td colspan='8'>No rows available</td></tr>")

    template = """
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Sampling V7 Applied Science Report</title>
  <style>
    body { font-family: Segoe UI, sans-serif; margin: 0; background: #f7f5f2; color: #1d2328; }
    .page { max-width: 1200px; margin: 0 auto; padding: 24px 18px 48px; }
    .card { background: #fff; border: 1px solid #ddd6cf; border-radius: 8px; padding: 18px; margin: 14px 0; }
    h1, h2, h3 { margin: 0 0 10px; }
    h1 { font-family: Georgia, serif; font-size: 2.2rem; }
    h2 { font-size: 1.3rem; }
    .muted { color: #4f5d66; }
    .study-note { background: #f9f8f6; border-left: 3px solid #326f73; padding: 10px 12px; color: #4f5d66; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #e6e1da; padding: 8px 6px; text-align: left; }
    th { background: #f6f4f2; }
    ul { margin: 0; padding-left: 18px; }
  </style>
</head>
<body>
  <div class='page'>
    <section class='card'>
      <h1>Sampling V7 Applied-Science Report</h1>
      <p class='muted'>Experiment identity: sampling-v7 · __SUMMARY_TEXT__</p>
    </section>

    <section class='card'>
      <h2>Executive Readout</h2>
      <p class='muted'>__EXEC_READOUT__</p>
    </section>

    <section class='card'>
      <h2>Study Design</h2>
      <p class='muted'>This is not a sequential flow; the five parallel arms are evaluated in parallel against the same replay plan, and they share the same outcome ledger and fixed corpus preprocessing.</p>
      <svg id='parallel-study-graph' viewBox='0 0 320 120' aria-label='Five parallel arms' style='width:100%;height:auto;border:1px solid #ddd6cf;border-radius:8px;background:#fbfaf8;display:block;margin-top:12px;'><rect x='0' y='0' width='320' height='120' fill='#fbfaf8'/><rect x='12' y='40' width='70' height='32' rx='6' fill='#f0efeb' stroke='#d8d2ca'/><text x='47' y='60' text-anchor='middle' font-size='11'>3 corpora</text><rect x='94' y='18' width='72' height='60' rx='6' fill='#eef4f2' stroke='#c7d9d4'/><text x='130' y='43' text-anchor='middle' font-size='10'>Foundry</text><text x='130' y='58' text-anchor='middle' font-size='10'>embeddings</text><rect x='180' y='8' width='54' height='26' rx='6' fill='#f5f0e7' stroke='#e0d9c5'/><text x='207' y='24' text-anchor='middle' font-size='9'>PCA-8</text><rect x='180' y='40' width='54' height='26' rx='6' fill='#f0f2f2' stroke='#d2d8d7'/><text x='207' y='56' text-anchor='middle' font-size='9'>Random</text><rect x='180' y='72' width='54' height='26' rx='6' fill='#f0f2f2' stroke='#d2d8d7'/><text x='207' y='88' text-anchor='middle' font-size='9'>MinHash</text><rect x='244' y='20' width='58' height='72' rx='6' fill='#edf5f0' stroke='#bfd8ca'/><text x='273' y='42' text-anchor='middle' font-size='10'>ARM5</text><text x='273' y='60' text-anchor='middle' font-size='10'>shared</text><text x='273' y='76' text-anchor='middle' font-size='10'>outputs</text></svg>
      <div class='study-note'>Three source corpora contribute the full-session embeddings, MinHash packets, and PCA fit once; the replay plan then assigns a shared bootstrap reordering for each repetition. Five parallel arms are Random, MinHash LSH, PCA-8 cosine + binary IDW, PCA-8 Euclidean + binary IDW, and ARM5 v6 membership + Hajek.</div>
    </section>

    <section class='card'>
      <h2>Budget Trend</h2>
      <p class='muted'>Lower is better for MAE; higher is better for coverage and F1. Budget and repetition slices are displayed in the dataset-specific summary rows.</p>
    </section>

    <section class='card'>
      <h2>Replay Uncertainty</h2>
      <p class='muted'>Replay uncertainty is expressed across the selected metric and the bootstrap replay plan; n=0 is skipped to avoid empty groups masquerading as real zeros.</p>
    </section>

    <section class='card'>
      <h2>Classification Quality</h2>
      <p class='muted'>Primary view is imputed-only F1; secondary view is judged + imputed classification, explicitly labeled as observed-inflated.</p>
    </section>

    <section class='card'>
      <h2>Coverage &amp; Cost</h2>
      <p class='muted'>Coverage is represented as concept coverage; cost is represented as actual token count and per-method latency across the selected budget and repetition slice.</p>
    </section>

    <section class='card'>
      <h2>Paired Cosine vs Euclidean</h2>
      <p class='muted'>Delta = cosine - Euclidean; negative favors cosine for lower-better metrics, positive favors cosine for higher-better metrics.</p>
    </section>

    <section class='card'>
      <h2>PCA context</h2>
      <p class='muted'>PCA context is shown in the dataset provenance and includes explained_variance_ratio values from the shared preprocessing fit.</p>
    </section>

    <section class='card'>
      <h2>Data &amp; Cloud Provenance</h2>
      <ul>__DATASET_PROVENANCE__</ul>
    </section>

    <section class='card'>
      <h2>Analysis</h2>
      <p class='muted'>__ANALYSIS_TEXT__</p>
    </section>

    <section class='card'>
      <h2>Limitations</h2>
      <ul>
        <li>Estimator families differ: random and MinHash selection are not directly comparable to PCA binary-imputed populations without the selected-only MAE lens.</li>
        <li>Replay uncertainty reflects bootstrap with replacement; occurrences are not independent labels and population-level generalization remains separate.</li>
        <li>Budget percentages are local to each corpus and not equal absolute caps across all datasets; use population context when comparing absolute counts.</li>
      </ul>
    </section>

    <section class='card'>
      <h2>Methodology</h2>
      <p class='muted'>The study combines three source corpora, a common replay plan, and five parallel arms evaluated under a single reporting contract. The method roster includes Random, MinHash LSH, PCA-8 cosine, PCA-8 Euclidean, and ARM5.</p>
    </section>

    <section class='card'>
      <h2>Global controls</h2>
      <div class='muted'>Budget %</div>
      <div class='muted'>Metric</div>
      <div class='muted'>Options include aggregate_pass_rate_mae, selected_only_pass_rate_mae, embedding_imputed_f1</div>
    </section>

    <section class='card'>
      <h2>Detailed results</h2>
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Seed</th>
            <th>Budget</th>
            <th>Method</th>
            <th>Aggregate MAE</th>
            <th>Selected-only MAE</th>
            <th>Concept coverage</th>
            <th>Imputed F1</th>
          </tr>
        </thead>
        <tbody>__DETAIL_ROWS__</tbody>
      </table>
    </section>
  </div>

  <script>
    const reportData = {
      rows: __ROWS__,
      datasets: __DATASETS__,
      aggregateMetrics: __AGG__,
      paired_differences: __PAIRED__,
      summary: __SUMMARY__
    };
    console.info('Sampling V7 report ready', reportData.summary, reportData.paired_differences.length);
  </script>
</body>
</html>
"""

    html = (
        template
        .replace("__SUMMARY_TEXT__", summary_text)
        .replace("__EXEC_READOUT__", exec_readout)
        .replace("__DATASET_PROVENANCE__", "".join(dataset_provenance))
        .replace("__ANALYSIS_TEXT__", exec_readout)
        .replace("__DETAIL_ROWS__", "".join(detail_rows))
        .replace("__ROWS__", _safe_json(rows))
        .replace("__DATASETS__", _safe_json(datasets))
        .replace("__AGG__", _safe_json(aggregate_rows))
        .replace("__PAIRED__", _safe_json(paired_rows))
        .replace("__SUMMARY__", _safe_json(summary))
    )

    return html


# Keep direct imports of the transitional module aligned with the canonical renderer.
from sampling_comparison.v7_report_rich import build_v7_report_html as build_v7_report_html
