from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

METHOD_IDS = (
    "pca8_idw_binary_cosine",
    "pca8_idw_binary_euclidean",
)

TITLE = "Full-session embeddings -> PCA-8 -> Binary IDW: Cosine vs Euclidean"


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")


def _metric_rows(payload: Mapping[str, Any], *, method_ids: Sequence[str] = METHOD_IDS) -> list[dict[str, Any]]:
    rows = payload.get("aggregate_metrics") if isinstance(payload.get("aggregate_metrics"), Mapping) else {}
    items = rows.get("rows") if isinstance(rows, Mapping) else []
    selected = []
    for row in items:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("method_id") or "") not in method_ids:
            continue
        selected.append(dict(row))
    return selected


def _paired_rows(payload: Mapping[str, Any], *, method_ids: Sequence[str] = METHOD_IDS) -> list[dict[str, Any]]:
    paired = payload.get("paired_differences") if isinstance(payload.get("paired_differences"), Mapping) else {}
    rows = paired.get("rows") if isinstance(paired, Mapping) else []
    selected = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        method_left = str(row.get("left_method_id") or "")
        method_right = str(row.get("right_method_id") or "")
        if (method_left, method_right) == tuple(method_ids):
            selected.append(dict(row))
    return selected


def _paired_cells(payload: Mapping[str, Any], *, method_ids: Sequence[str] = METHOD_IDS) -> list[dict[str, Any]]:
    paired = payload.get("paired_differences") if isinstance(payload.get("paired_differences"), Mapping) else {}
    cells = paired.get("cells") if isinstance(paired, Mapping) else []
    valid_keys = {
        (str(row.get("dataset_id") or ""), int(row.get("budget_pct") or 0), str(row.get("metric_id") or ""))
        for row in _paired_rows(payload, method_ids=method_ids)
    }
    selected = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        key = (
            str(cell.get("dataset_id") or ""),
            int(cell.get("budget_pct") or 0),
            str(cell.get("metric_id") or ""),
        )
        if key in valid_keys:
            selected.append(dict(cell))
    return selected


def _trim_run_row(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "dataset_id",
        "seed",
        "repetition_index",
        "replay_seed",
        "replay_id",
        "replay_frequency_summary",
        "budget_pct",
        "cap",
        "sample_size",
        "method_id",
        "estimator_type",
        "aggregate_pass_rate_mae",
        "selected_only_pass_rate_mae",
        "replay_aggregate_pass_rate_mae",
        "source_corpus_census_pass_rate",
        "replay_census_pass_rate",
        "coverage",
        "actual_token_count",
        "latency_seconds",
        "embedding_metrics",
        "search_evidence",
    )
    return {field: row.get(field) for field in fields if field in row}


def _aggregate_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    dataset_rows = payload.get("datasets") or []
    dataset_count = len(dataset_rows) if isinstance(dataset_rows, list) else 0
    runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
    method_rows = [row for row in runs if isinstance(row, Mapping) and str(row.get("method_id") or "") in METHOD_IDS]
    return {
        "dataset_count": dataset_count,
        "repetition_count": int(summary.get("repetition_count") or 0),
        "budget_count": int(summary.get("budget_count") or 0),
        "run_count": len(method_rows),
        "base_seed": summary.get("base_seed"),
        "dataset_ids": _unique((row.get("dataset_id") for row in dataset_rows if isinstance(row, Mapping))),
    }


def _method_label(method_id: str) -> str:
    labels = {
    "pca8_idw_binary_cosine": "Cosine distance arm",
    "pca8_idw_binary_euclidean": "Euclidean distance arm",
    }
    return labels.get(method_id, method_id)


def _build_report_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = _aggregate_summary(payload)
    rows = [
      _trim_run_row(row)
      for row in (payload.get("runs") or [])
      if isinstance(row, Mapping) and str(row.get("method_id") or "") in METHOD_IDS
    ]
    aggregate_rows = _metric_rows(payload)
    paired_rows = _paired_rows(payload)
    paired_cells = _paired_cells(payload)
    datasets = [dict(row) for row in (payload.get("datasets") or []) if isinstance(row, Mapping)]

    return {
      "title": "Full-session embeddings -> PCA-8 -> Binary IDW: Cosine vs Euclidean",
        "summary": summary,
        "rows": rows,
        "aggregate_metrics": aggregate_rows,
        "paired_rows": paired_rows,
        "paired_cells": paired_cells,
        "datasets": datasets,
      "shared_preprocessing": dict(payload.get("shared_preprocessing") or {}) if isinstance(payload.get("shared_preprocessing"), Mapping) else {},
      "cloud_metadata": dict(payload.get("cloud_metadata") or {}) if isinstance(payload.get("cloud_metadata"), Mapping) else {},
      "transductive": bool(payload.get("transductive", False)),
      "source": dict(payload.get("source") or {}) if isinstance(payload.get("source"), Mapping) else {},
        "method_ids": list(METHOD_IDS),
        "method_labels": {method_id: _method_label(method_id) for method_id in METHOD_IDS},
    }


def build_distance_report_html(payload: Mapping[str, Any]) -> str:
    data = _build_report_data(payload)
    summary = data["summary"]
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f5f7f5; --panel: #ffffff; --ink: #17222b; --muted: #58656d; --line: #dfe7e3;
      --cosine: #3e7db8; --euclid: #cc8a2d; --alt: #edf1f0; --success: #337d67; --warn: #ad6520;
      --radius: 8px;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; width: 100%; max-width: 100%; overflow-x: hidden; }
    body { font-family: Aptos, Candara, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); line-height: 1.5; }
    h1, h2, h3 { font-family: Georgia, Cambria, serif; margin: 0 0 10px; }
    h1 { font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.1; }
    h2 { font-size: clamp(1.4rem, 2vw, 2rem); }
    h3 { font-size: 1.08rem; }
    p { margin: 0 0 10px; color: var(--muted); }
    table { width: 100%; border-collapse: collapse; font-size: 0.92rem; table-layout: fixed; }
    th, td { border: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; overflow-wrap: anywhere; word-break: break-word; }
    th { background: #f2f5f3; }
    .page { max-width: 100%; margin: 0 auto; width: 100%; min-width: 0; overflow-x: clip; }
    .hero { padding: 32px 20px 18px; background: linear-gradient(135deg, #163b3d, #1d4c4b); color: #f4f7f7; }
    .hero-inner { max-width: 1200px; margin: 0 auto; }
    .eyebrow { font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; color: #cfe3e0; font-weight: 700; }
    .hero p { color: #d9e8e6; max-width: 880px; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; min-width: 0; }
    .chip { border: 1px solid rgba(255,255,255,0.25); border-radius: 999px; padding: 6px 10px; font-size: .8rem; background: rgba(255,255,255,0.04); }
    .kpis { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 14px; margin-top: 24px; min-width: 0; }
    .kpi { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); border-radius: var(--radius); padding: 14px 16px; }
    .kpi strong { display: block; font-size: 1.4rem; margin-bottom: 4px; }
    .kpi span { color: #d9e8e6; font-size: .8rem; }
    .section { padding: 30px 20px; }
    .section-inner { max-width: 1200px; margin: 0 auto; }
    .two-up { display: grid; grid-template-columns: 1.15fr .85fr; gap: 20px; min-width: 0; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); padding: 16px; min-width: 0; overflow-wrap: anywhere; }
    .panel h3 { margin-bottom: 10px; }
    .plot, .matrix, .table-wrap { overflow: auto; max-width: 100%; }
    .plot svg { width: 100%; height: auto; min-height: 260px; display: block; background: #fff; border-radius: var(--radius); border: 1px solid var(--line); }
    .plot > .panel { margin-bottom: 12px; }
    .legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; color: var(--muted); font-size: .85rem; }
    .dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
    .matrix table td, .matrix table th { text-align: center; }
    .trace { font-family: Consolas, "Courier New", monospace; font-size: .78rem; color: #29414d; background: #edf5f8; border: 1px solid #d2e0e7; border-radius: 6px; padding: 8px; overflow-wrap: anywhere; }
    .warn { color: #874e17; font-weight: 600; }
    .ok { color: #2d6f5c; font-weight: 600; }
    .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 14px; min-width: 0; }
    .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; min-width: 0; }
    .card-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
    .caption { font-size: .82rem; color: var(--muted); margin-top: 8px; }
    .dataset-header { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; }
    .local-scroll { overflow: auto; max-width: 100%; }
    .paired-count-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 10px; }
    .paired-count-card { background: #f8faf9; border: 1px solid var(--line); border-radius: var(--radius); padding: 12px; display: grid; gap: 6px; min-width: 0; }
    .paired-count-card strong { font-size: 0.82rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
    .paired-count-card .meta { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .paired-count-card .value { font-size: 1.1rem; font-weight: 700; }
    .classification-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .classification-card { background: #fff; border: 1px solid var(--line); border-radius: var(--radius); padding: 12px; display: grid; gap: 10px; }
    .classification-card .head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .classification-card .metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .classification-card .metric { padding: 8px; border: 1px solid var(--line); border-radius: 6px; background: #f8faf9; }
    .classification-card .metric small { display: block; color: var(--muted); margin-bottom: 4px; }
    .decision-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .decision-card { border: 1px solid var(--line); border-radius: var(--radius); background: #ffffff; padding: 12px; display: grid; gap: 8px; min-width: 0; }
    .decision-card .verdict { font-weight: 700; }
    .corpus-verdicts { font-size: .86rem; color: #33444d; line-height: 1.5; }
    .evidence-details { border-top: 1px solid var(--line); padding-top: 7px; }
    .evidence-details summary { cursor: pointer; font-weight: 700; color: #33444d; }
    .evidence-details .evidence-body { margin-top: 7px; font-size: .82rem; color: var(--muted); overflow-wrap: anywhere; }
    .conclusion-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; }
    .conclusion-card { border: 1px solid var(--line); border-radius: var(--radius); background: #fff; padding: 11px; min-width: 0; }
    .conclusion-card h4 { margin: 0 0 3px; font-size: .96rem; }
    .conclusion-card .verdict { font-weight: 700; margin: 8px 0; }
    .story-block { border-left: 4px solid #d8e0dc; padding: 8px 10px; margin-top: 10px; background: #fbfcfb; }
    .story-block p { margin: 0 0 6px; color: #33444d; }
    .story-block p:last-child { margin-bottom: 0; }
    .story-label { font-weight: 700; color: #19262d; }
    @media (max-width: 640px) {
      .paired-count-grid { grid-template-columns: 1fr; }
      .paired-count-card { padding: 10px; }
      .classification-grid { grid-template-columns: 1fr; }
      .classification-card .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .decision-grid { grid-template-columns: 1fr; }
      .conclusion-grid { grid-template-columns: 1fr; }
    }
    .tool-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 10px; min-width: 0; }
    .tool-row label { display: flex; flex-direction: column; gap: 6px; min-width: 0; flex: 1 1 180px; }
    select { width: 100%; min-width: 0; padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px; background: white; color: var(--ink); }
    button { background: #1b4d4c; color: white; border: 0; border-radius: 6px; padding: 8px 12px; cursor: pointer; }
    .muted { color: var(--muted); }
    .small { font-size: .82rem; }
    .low-n { color: #7b4b12; font-weight: 600; }
    @media (max-width: 900px) {
      .two-up, .kpis, .grid-3, .grid-2 { grid-template-columns: 1fr; }
      .hero { padding-top: 24px; }
    }
    @media (max-width: 640px) {
      .hero, .section { padding-left: 12px; padding-right: 12px; }
      .panel { padding: 12px; }
      .tool-row { gap: 10px; }
      .tool-row label { flex: 1 1 100%; }
      table { font-size: 0.85rem; }
      .plot { overflow-x: auto; }
      .plot svg { width: 640px; min-width: 640px; max-width: none; }
    }
    @media print {
      body { background: white; }
      .hero { background: #ffffff !important; color: #17222b; border-bottom: 1px solid #d9e2df; }
      .hero p, .hero .eyebrow, .kpi span { color: #3a4852 !important; }
      .kpi { background: #fff; border-color: #dfe7e3; }
      button, select { display: none !important; }
      .section { break-inside: avoid; page-break-inside: avoid; }
      .panel { border-color: #cfdad5; }
      .plot svg { border-color: #d5dfdb; }
    }
  </style>
</head>
<body>
  <div class="page" id="pca8DistanceReport">
    <header class="hero">
      <div class="hero-inner">
        <div class="eyebrow">Sampling V7 / Full-session embeddings / PCA-8 binary IDW</div>
        <h1>__TITLE__</h1>
        <p>Reader-facing comparison of two parallel distance arms only: cosine vs Euclidean, with full-session embeddings, one transductive PCA-8 fit per corpus, paired replay, identical budgets, and the same binary IDW scoring contract. Canonical method IDs are retained in traceability text/tooltips, not chart headings.</p>
        <div class="chips">
          <span class="chip">__DATASET_COUNT__ corpora</span>
          <span class="chip">__REPETITION_COUNT__ paired replays</span>
          <span class="chip">__BUDGET_COUNT__ budgets</span>
          <span class="chip">__RUN_COUNT__ raw arm runs</span>
          <span class="chip">300 focused rows (3 corpora x 10 replays x 5 budgets x 2 methods)</span>
          <span class="chip">Foundry embedding provenance + Azure AI Search validation probes</span>
        </div>
        <div class="kpis">
          <div class="kpi"><strong>__DATASET_COUNT__</strong><span>corpora</span></div>
          <div class="kpi"><strong>__REPETITION_COUNT__</strong><span>paired replays</span></div>
          <div class="kpi"><strong>__BUDGET_COUNT__</strong><span>budgets</span></div>
          <div class="kpi"><strong>__RUN_COUNT__</strong><span>raw arm runs</span></div>
        </div>
      </div>
    </header>

    <section class="section" id="introduction">
      <div class="section-inner">
        <h2>Hero comparison and answer</h2>
        <div class="panel">
          <p><strong>Question.</strong> Holding embeddings, one transductive PCA-8 fit per corpus, paired replay, budget, IDW (k=8, power=2, eps=1e-6), cutoff 0.5, prior 0.5, donor scope, and scoring constant, what changes when neighborhood distance is cosine versus Euclidean?</p>
          <p id="executiveAnswer">Answer updates with filters.</p>
          <p class="caption" id="scopeCaption">Scope uses paired replay rows and reports empirical replay sensitivity, not population generalization.</p>
        </div>
        <div class="two-up" style="margin-top:14px;">
          <div class="panel">
            <h3>Method identity</h3>
            <p>Cosine distance measures angular separation: <code>d_cos(x,y) = 1 - (x dot y) / (norm(x) norm(y))</code>. Euclidean measures straight-line separation: <code>d_euc(x,y) = norm(x - y)</code>. Both are applied after the same PCA-8 projection and replay cell definition.</p>
            <div class="trace">Canonical method IDs: pca8_idw_binary_cosine, pca8_idw_binary_euclidean</div>
          </div>
          <div class="panel">
            <h3>Interactive controls</h3>
            <div class="tool-row">
              <label>Dataset<select id="datasetFilter"></select></label>
              <label>Budget %<select id="budgetFilter"></select></label>
              <label>Repetition<select id="repetitionFilter"></select></label>
              <label>Metric<select id="metricFilter"></select></label>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section" id="decisionSummary">
      <div class="section-inner">
        <h2>Decision summary / What we learned</h2>
        <div class="panel">
          <div id="decisionScorecard" class="decision-grid" aria-live="polite"></div>
        </div>
      </div>
    </section>

    <section class="section" id="pipeline">
      <div class="section-inner">
        <h2>Experiment design / What stayed constant and what changed</h2>
        <div class="grid-2">
          <div class="panel" id="pipelineGuidePanel">
            <h3>Pipeline (held constant first)</h3>
            <ol>
              <li>Canonical full-session packetization per session, bounded by max input tokens = 8191.</li>
              <li>Foundry embedding matrix per corpus; this evidence packet reused cache (cache_hit=true for all corpora).</li>
              <li>One transductive PCA-8 fit per corpus, reused by both distance arms.</li>
              <li>Paired replay perturbs frequency/order on the same corpus, then applies absolute budget cap.</li>
              <li>Parallel distance selection + binary IDW: cosine arm and Euclidean arm under the same scoring contract.</li>
            </ol>
          </div>
          <div class="panel" id="idwSemanticsPanel">
            <h3>What differs and IDW semantics</h3>
            <ul>
              <li>Only neighborhood distance changes: cosine vs Euclidean.</li>
              <li>IDW parameters are fixed: <strong>k=8</strong>, <strong>power=2</strong>, <strong>eps=1e-6</strong>.</li>
              <li>Predicted probability is binarized at <strong>cutoff >= 0.5</strong>.</li>
              <li>No-donor fallback prior is <strong>0.5</strong>, reported separately from the cutoff.</li>
              <li>Donor scope prefers within-agent donors first, then global fallback if needed.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <section class="section" id="profiles">
      <div class="section-inner">
        <h2>Corpus conditions / Why results may differ</h2>
        <div id="datasetProfiles" class="card-list"></div>
      </div>
    </section>

    <section class="section" id="token-first-results">
      <div class="section-inner">
        <h2>Token cost signal first</h2>
        <div class="panel">
          <div id="tokensAndLatency" class="plot"></div>
          <div id="tokensLatencyStory" class="story-block"></div>
        </div>
      </div>
    </section>

    <section class="section" id="results">
      <div class="section-inner">
        <h2>Accuracy and error tradeoffs</h2>
        <div class="two-up">
          <div class="panel" id="budgetTrendPanel">
            <h3 id="budgetTrendTitle">Budget trend</h3>
            <div id="budgetTrendChart" class="plot"></div>
            <p class="caption" id="budgetCapCaption"></p>
            <div id="budgetTrendStory" class="story-block"></div>
          </div>
          <div class="panel" id="deltaPanel">
            <h3 id="deltaPanelTitle">Paired delta</h3>
            <div id="pairedDeltaPlot" class="plot"></div>
            <p class="caption" id="pairedDeltaCaption">Delta = cosine - Euclidean. Interpretation is direction-aware by metric objective.</p>
            <div id="pairedDeltaStory" class="story-block"></div>
          </div>
        </div>
      </div>
    </section>

    <section class="section" id="paired-evidence">
      <div class="section-inner">
        <h2>Paired evidence and consistency</h2>
        <div class="two-up">
          <div class="panel">
            <h3 id="winLossTitle">Win / tie / loss</h3>
            <div id="winLossChart" class="plot"></div>
            <div id="winLossCounts" class="table-wrap" style="margin-top:10px;"></div>
            <div id="winLossStory" class="story-block"></div>
          </div>
          <div class="panel">
            <h3 id="classificationTitle">Classification comparison</h3>
            <p id="classificationMiniSummary" class="caption"></p>
            <div id="classificationMatrix" class="matrix"></div>
            <p class="caption">If Dataset = All corpora, results are faceted by dataset (no silent pooling).</p>
            <div id="classificationStory" class="story-block"></div>
          </div>
        </div>
      </div>
    </section>

    <section class="section" id="analysis">
      <div class="section-inner">
        <h2>Coverage, uncertainty, and diagnostics</h2>
        <div class="two-up">
          <div class="panel">
            <h3 id="coverageTitle">Coverage</h3>
            <div id="conceptCoverageChart" class="plot"></div>
            <div id="coverageStory" class="story-block"></div>
          </div>
          <div class="panel">
            <h3 id="uncertaintyTitle">Replay uncertainty and frequency context</h3>
            <div id="replayUncertaintyChart" class="plot"></div>
            <p class="caption">P05-P95 / CI95 summarize empirical replay sensitivity under paired perturbations on the same corpus, not population generalization bounds.</p>
            <div id="uncertaintyStory" class="story-block"></div>
            <div id="replayFrequencyContext" class="plot" style="margin-top:12px;"></div>
            <div id="replayFrequencyStory" class="story-block"></div>
          </div>
        </div>
      </div>
    </section>

    <section class="section" id="pca">
      <div class="section-inner">
        <h2>PCA-8 context</h2>
        <div class="panel">
          <div id="pcaScreeChart" class="plot"></div>
          <p class="caption">Cumulative retained variance axis is fixed at 0-100% for all corpora.</p>
          <div id="pcaStory" class="story-block"></div>
        </div>
      </div>
    </section>

    <section class="section" id="provenance">
      <div class="section-inner">
        <h2>Data and Cloud Provenance</h2>
        <div class="panel">
          <div id="provenanceGrid" class="table-wrap"></div>
        </div>
      </div>
    </section>

    <section class="section" id="limitations">
      <div class="section-inner">
        <h2>Limitations</h2>
        <p>These are paired replay observations, not population-level generalization claims. Paired replay perturbs frequency/order on the same corpus; uncertainty here is replay sensitivity. The corpus profile cards flag embedding-token averages near the configured packet bound and PCA-8 retention below 50%; per-session truncation incidence is not recorded in this evidence packet.</p>
      </div>
    </section>

    <section class="section" id="conclusion">
      <div class="section-inner">
        <h2>Conclusion and recommendation</h2>
        <div id="storyConclusion" class="panel" style="margin-bottom:10px;"></div>
        <p id="conclusionText">Filter-scoped paired evidence may favor either arm depending on objective and corpus. Reported deltas are direction-aware and include explicit paired n, wins/ties/losses, and mean paired delta where available. This evidence is objective-specific and does not claim a universal winner.</p>
      </div>
    </section>

    <section class="section" id="details">
      <div class="section-inner">
        <h2>Detailed results</h2>
        <div class="panel">
          <div class="tool-row"><span id="rowCount" class="small muted"></span><button id="showMore" type="button">Show More</button></div>
          <div class="table-wrap" style="margin-top:12px;">
            <table>
              <thead>
                <tr>
                  <th>Dataset</th><th>Budget %</th><th>Cap</th><th>sample_size</th><th>Repetition</th><th>Method</th><th>Aggregate MAE</th><th>Selected-only MAE</th><th>Coverage</th><th>Imputed F1</th><th>Tokens</th><th>Incremental runtime (selection+IDW s)</th><th>Shared Search reference (s)</th>
                </tr>
              </thead>
              <tbody id="detailTable"></tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const REPORT = __REPORT_JSON__;
    const METHOD_IDS = __METHOD_IDS__;
    const METHOD_LABELS = __METHOD_LABELS__;
    const COLORS = {
      pca8_idw_binary_cosine: '#3e7db8',
      pca8_idw_binary_euclidean: '#cc8a2d'
    };
    const METRIC_META = {
      aggregate_pass_rate_mae: { label: 'Fixed-source aggregate MAE', direction: 'lower_better', format: 'decimal' },
      selected_only_pass_rate_mae: { label: 'Selected-only MAE', direction: 'lower_better', format: 'decimal' },
      replay_aggregate_pass_rate_mae: { label: 'Replay-relative aggregate MAE', direction: 'lower_better', format: 'decimal' },
      coverage_concept_ratio: { label: 'Concept coverage', direction: 'higher_better', format: 'percent' },
      imputed_only_accuracy: { label: 'Imputed-only accuracy', direction: 'higher_better', format: 'percent' },
      imputed_only_precision: { label: 'Imputed-only precision', direction: 'higher_better', format: 'percent' },
      imputed_only_recall: { label: 'Imputed-only recall', direction: 'higher_better', format: 'percent' },
      imputed_only_f1: { label: 'Imputed-only F1', direction: 'higher_better', format: 'percent' },
      actual_token_count: { label: 'Actual judged tokens', direction: 'lower_better', format: 'integer' },
      latency_per_method_total_seconds: { label: 'Raw per_method_total (contains shared Search)', direction: 'lower_better', format: 'seconds' }
    };
    const PAIRED_SUPPORTED_METRICS = [
      'aggregate_pass_rate_mae',
      'selected_only_pass_rate_mae',
      'replay_aggregate_pass_rate_mae',
      'coverage_concept_ratio',
      'imputed_only_accuracy',
      'imputed_only_precision',
      'imputed_only_recall',
      'imputed_only_f1',
      'actual_token_count'
    ];
    const VERDICT_WORDS = {
      cosine: 'Cosine leads',
      euclidean: 'Euclidean leads',
      mixed: 'Mixed / no clear winner',
      tie: 'Exact tie',
      cosine_lean: 'Cosine leans',
      euclidean_lean: 'Euclidean leans'
    };
    const OBJECTIVE_CARDS = [
      { id: 'estimation', label: 'Estimation error', metricIds: ['aggregate_pass_rate_mae', 'selected_only_pass_rate_mae'] },
      { id: 'classification', label: 'Imputed classification', metricIds: ['imputed_only_accuracy', 'imputed_only_f1'] },
      { id: 'coverage', label: 'Concept coverage', metricIds: ['coverage_concept_ratio'] },
      { id: 'tokens', label: 'Judged-token cost', metricIds: ['actual_token_count'] },
      { id: 'runtime', label: 'Incremental selection+IDW runtime', metricIds: [], rawIncremental: true }
    ];

    const fmt = (value, digits = 3) => {
      const num = Number(value);
      if (!Number.isFinite(num)) return 'n/a';
      return num.toFixed(digits);
    };
    const pct = (value, digits = 1) => {
      const num = Number(value);
      if (!Number.isFinite(num)) return 'n/a';
      return `${(num * 100).toFixed(digits)}%`;
    };
    const pct1 = (value) => pct(value, 1);
    const safeText = (value) => String(value ?? 'n/a').replace(/[&<>\"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
    const getNumber = (obj, path) => {
      let cur = obj;
      for (const segment of path.split('.')) {
        if (cur == null || !(segment in cur)) return null;
        cur = cur[segment];
      }
      const value = Number(cur);
      return Number.isFinite(value) ? value : null;
    };
    const median = (values) => {
      if (!values.length) return null;
      const sorted = [...values].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };
    const mean = (values) => {
      if (!values.length) return null;
      return values.reduce((sum, value) => sum + value, 0) / values.length;
    };
    const sign = (v) => (v > 0 ? 1 : (v < 0 ? -1 : 0));
    const practicalThreshold = (metricId) => metricId === 'actual_token_count' ? 0.05 : 0.02;
    function metricRelativeMagnitude(metricId, delta, baselineMagnitude) {
      if (!Number.isFinite(delta)) return null;
      if (metricId === 'actual_token_count') {
        if (!Number.isFinite(baselineMagnitude) || baselineMagnitude <= 0) return null;
        return Math.abs(delta) / baselineMagnitude;
      }
      return Math.abs(delta);
    }
    function prefersCosine(metricId, delta) {
      const direction = METRIC_META[metricId]?.direction || 'lower_better';
      return direction === 'higher_better' ? delta > 0 : delta < 0;
    }
    function verdictFromPaired(metricId, deltas, wins, ties, losses, baselineMagnitude) {
      const clean = deltas.filter(Number.isFinite);
      const n = clean.length;
      if (!n) {
        return { verdict: VERDICT_WORDS.mixed, evidenceLevel: 'none', n: 0, wins: 0, ties: 0, losses: 0, meanDelta: null, medianDelta: null, winRate: null, favoredWinRate: null, relMagnitude: null };
      }
      const m = mean(clean);
      const med = median(clean);
      const allZero = clean.every((value) => Math.abs(value) <= 1e-12);
      const effWins = Number.isFinite(wins) ? wins : clean.filter((d) => prefersCosine(metricId, d)).length;
      const effTies = Number.isFinite(ties) ? ties : clean.filter((d) => Math.abs(d) <= 1e-12).length;
      const effLosses = Number.isFinite(losses) ? losses : Math.max(0, n - effWins - effTies);
      const favoredDir = prefersCosine(metricId, m || 0) ? 'cosine' : 'euclidean';
      const favoredWins = favoredDir === 'cosine' ? effWins : effLosses;
      const favoredWinRate = n > 0 ? favoredWins / n : null;
      const winRate = n > 0 ? effWins / n : null;
      const disagree = sign(m || 0) !== sign(med || 0) && sign(m || 0) !== 0 && sign(med || 0) !== 0;
      const relMagnitude = metricRelativeMagnitude(metricId, m, baselineMagnitude);
      const threshold = practicalThreshold(metricId);
      const practicalSmall = relMagnitude === null ? true : relMagnitude < threshold;
      if (allZero) {
        return { verdict: VERDICT_WORDS.tie, evidenceLevel: 'tie', n, wins: effWins, ties: effTies, losses: effLosses, meanDelta: m, medianDelta: med, winRate, favoredWinRate, relMagnitude };
      }
      const strongEvidence = !disagree && !practicalSmall && favoredWinRate !== null && favoredWinRate >= 0.7;
      if (strongEvidence) {
        return {
          verdict: favoredDir === 'cosine' ? VERDICT_WORDS.cosine : VERDICT_WORDS.euclidean,
          evidenceLevel: 'strong',
          n,
          wins: effWins,
          ties: effTies,
          losses: effLosses,
          meanDelta: m,
          medianDelta: med,
          winRate,
          favoredWinRate,
          relMagnitude
        };
      }
      const leanDir = favoredDir;
      const leanOnly = !disagree && favoredWinRate !== null && favoredWinRate >= 0.5 && !practicalSmall;
      if (leanOnly) {
        return {
          verdict: leanDir === 'cosine' ? VERDICT_WORDS.cosine_lean : VERDICT_WORDS.euclidean_lean,
          evidenceLevel: 'lean',
          n,
          wins: effWins,
          ties: effTies,
          losses: effLosses,
          meanDelta: m,
          medianDelta: med,
          winRate,
          favoredWinRate,
          relMagnitude
        };
      }
      return {
        verdict: practicalSmall ? VERDICT_WORDS.mixed : ((disagree || favoredWinRate === null || favoredWinRate < 0.5) ? VERDICT_WORDS.mixed : (leanDir === 'cosine' ? VERDICT_WORDS.cosine_lean : VERDICT_WORDS.euclidean_lean)),
        evidenceLevel: 'mixed',
        n,
        wins: effWins,
        ties: effTies,
        losses: effLosses,
        meanDelta: m,
        medianDelta: med,
        winRate,
        favoredWinRate,
        relMagnitude
      };
    }
    function storyBlockHtml(what, how, takeaway) {
      return `<p><span class="story-label">What this shows:</span> ${safeText(what)}</p><p><span class="story-label">How to read it:</span> ${safeText(how)}</p><p><span class="story-label">Current takeaway:</span> ${safeText(takeaway)}</p>`;
    }
    function evidenceDetailsHtml(label, body) {
      return `<details class="evidence-details"><summary>${safeText(label)}</summary><div class="evidence-body">${safeText(body)}</div></details>`;
    }
    function scopeVerdictSummary(metricId) {
      if (activeDataset() === 'all') {
        return `Per-corpus verdicts: ${selectedDatasets().map((datasetId) => {
          const verdict = objectiveVerdict(metricId, datasetId);
          return `${datasetId}: ${verdict.verdict} (n=${verdict.n}, W/T/L=${verdict.wins}/${verdict.ties}/${verdict.losses})`;
        }).join(' | ')}. No pooled winner is claimed.`;
      }
      const verdict = objectiveVerdict(metricId, activeDataset());
      return `${verdict.verdict}; paired replay n=${verdict.n}, W/T/L=${verdict.wins}/${verdict.ties}/${verdict.losses}, mean delta=${verdict.meanDelta === null ? 'n/a' : fmt(verdict.meanDelta, 4)}.`;
    }
    function rowsForFilter() {
      const dataset = document.getElementById('datasetFilter')?.value ?? 'all';
      const budget = document.getElementById('budgetFilter')?.value ?? 'all';
      const repetition = document.getElementById('repetitionFilter')?.value ?? 'all';
      return REPORT.rows.filter((row) => {
        if (dataset !== 'all' && String(row.dataset_id) !== dataset) return false;
        if (budget !== 'all' && String(row.budget_pct) !== budget) return false;
        if (repetition !== 'all' && String(row.repetition_index) !== repetition) return false;
        return true;
      });
    }
    function activeDataset() { return document.getElementById('datasetFilter')?.value ?? 'all'; }
    function activeBudget() { return document.getElementById('budgetFilter')?.value ?? 'all'; }
    function activeRepetition() { return document.getElementById('repetitionFilter')?.value ?? 'all'; }
    function activeMetric() { return document.getElementById('metricFilter')?.value ?? 'aggregate_pass_rate_mae'; }
    function metricFromRun(row, metricId) {
      if (metricId === 'coverage_concept_ratio') return getNumber(row, 'coverage.concept.coverage_ratio');
      if (metricId.startsWith('imputed_only_')) return getNumber(row, `embedding_metrics.imputed_only.${metricId.replace('imputed_only_', '')}`);
      if (metricId.startsWith('judged_plus_imputed_')) return getNumber(row, `embedding_metrics.judged_plus_imputed.${metricId.replace('judged_plus_imputed_', '')}`);
      if (metricId === 'latency_per_method_total_seconds') return getNumber(row, 'latency_seconds.per_method_total');
      return getNumber(row, metricId);
    }
    function filteredAggregate(metricId, dataset = activeDataset(), budget = activeBudget()) {
      return REPORT.aggregate_metrics.filter((row) =>
        row.metric_id === metricId && Number(row.n) > 0 &&
        (dataset === 'all' || String(row.dataset_id) === String(dataset)) &&
        (budget === 'all' || String(row.budget_pct) === String(budget))
      );
    }
    function filteredPairedRows(metricId) {
      return REPORT.paired_rows.filter((row) =>
        row.metric_id === metricId && Number(row.n) > 0 &&
        (activeDataset() === 'all' || String(row.dataset_id) === activeDataset()) &&
        (activeBudget() === 'all' || String(row.budget_pct) === activeBudget())
      );
    }
    function filteredPairedCells(metricId) {
      return REPORT.paired_cells.filter((cell) =>
        cell.metric_id === metricId &&
        (activeDataset() === 'all' || String(cell.dataset_id) === activeDataset()) &&
        (activeBudget() === 'all' || String(cell.budget_pct) === activeBudget()) &&
        (activeRepetition() === 'all' || String(cell.repetition_index) === activeRepetition())
      );
    }
    function selectedDatasets() {
      return activeDataset() === 'all'
        ? [...new Set(REPORT.rows.map((row) => String(row.dataset_id)))]
        : [activeDataset()];
    }
    function capsByDatasetBudget() {
      const map = new Map();
      REPORT.rows.forEach((row) => {
        const key = `${row.dataset_id}|${row.budget_pct}`;
        const cap = Number(row.cap);
        if (!Number.isFinite(cap)) return;
        if (!map.has(key)) map.set(key, []);
        map.get(key).push(cap);
      });
      return map;
    }
    function representativeCap(datasetId, budgetPct) {
      const values = capsByDatasetBudget().get(`${datasetId}|${budgetPct}`) || [];
      if (!values.length) return null;
      values.sort((a, b) => a - b);
      return values[Math.floor(values.length / 2)];
    }
    function lowNFromCap(cap) {
      return Number.isFinite(Number(cap)) && Number(cap) < 10;
    }
    function incrementalLatency(row) {
      const selection = getNumber(row, 'latency_seconds.selection') || 0;
      const idw = getNumber(row, 'latency_seconds.idw') || 0;
      return selection + idw;
    }
    function sharedSearchLatency(row) {
      return getNumber(row, 'search_evidence.latency_seconds') || getNumber(row, 'latency_seconds.search_evidence');
    }
    function refreshBudgetOptions() {
      const control = document.getElementById('budgetFilter');
      if (!control) return;
      const prior = control.value || 'all';
      const budgets = ['all', ...new Set(REPORT.rows.map(r => String(r.budget_pct)).sort((a, b) => Number(a) - Number(b)))];
      const datasets = [...new Set(REPORT.rows.map((r) => String(r.dataset_id)))];
      control.innerHTML = budgets.map((value) => {
        if (value === 'all') return '<option value="all">All budgets</option>';
        const capText = datasets.map((datasetId) => {
          const cap = representativeCap(datasetId, Number(value));
          return `${datasetId}:${Number.isFinite(cap) ? cap : 'n/a'}`;
        }).join(' | ');
        return `<option value="${value}">${value}% (cap ${safeText(capText)})</option>`;
      }).join('');
      control.value = budgets.includes(prior) ? prior : (budgets.includes('10') ? '10' : 'all');
    }
    function pairedMetricOptions() {
      const available = new Set(REPORT.paired_rows.map((row) => String(row.metric_id || '')));
      const supported = PAIRED_SUPPORTED_METRICS.filter((metricId) => available.has(metricId) || REPORT.aggregate_metrics.some((row) => String(row.metric_id || '') === metricId));
      return supported.length ? supported : [...available].filter((metricId) => metricId && METRIC_META[metricId]);
    }
    function initFilters() {
      const datasets = ['all', ...new Set(REPORT.rows.map(r => r.dataset_id))];
      const reps = ['all', ...new Set(REPORT.rows.map(r => String(r.repetition_index)))];
      document.getElementById('datasetFilter').innerHTML = datasets.map((value) => `<option value="${value}">${value === 'all' ? 'All corpora' : safeText(value)}</option>`).join('');
      refreshBudgetOptions();
      document.getElementById('repetitionFilter').innerHTML = reps.map((value) => `<option value="${value}">${value === 'all' ? 'All replays' : `Replay ${value}`}</option>`).join('');
      const metricOptions = pairedMetricOptions();
      document.getElementById('metricFilter').innerHTML = metricOptions.map((id) => `<option value="${id}">${safeText(METRIC_META[id]?.label || id)}</option>`).join('');
      if (!metricOptions.includes('aggregate_pass_rate_mae')) document.getElementById('metricFilter').value = metricOptions[0] || '';
      else document.getElementById('metricFilter').value = 'aggregate_pass_rate_mae';
      ['datasetFilter', 'budgetFilter', 'repetitionFilter', 'metricFilter'].forEach((id) => document.getElementById(id).addEventListener('change', () => {
        if (id === 'datasetFilter') refreshBudgetOptions();
        renderAll();
      }));
    }
    function renderDatasetProfiles() {
      const panel = document.getElementById('datasetProfiles');
      const cards = selectedDatasets().map((datasetId) => {
        const dataset = REPORT.datasets.find((row) => String(row.dataset_id) === String(datasetId));
        if (!dataset) return '';
        const ledger = dataset.runtime_ledger || {};
        const prep = dataset.shared_preprocessing || {};
        const avgTokens = Number(prep.embedding_inputs) > 0
          ? Number(prep.embedding_input_tokens || 0) / Number(prep.embedding_inputs)
          : null;
        const variance = Number(dataset.pca?.explained_variance_total);
        const weak = Number.isFinite(variance) && variance < 0.5;
        const nearCap = Number.isFinite(avgTokens) && avgTokens >= 7000;
        return `<div class="panel"><div class="dataset-header"><h3>${safeText(datasetId)}</h3><span class="small ${weak ? 'warn' : 'ok'}">PCA-8 retained variance ${pct1(variance)}</span></div>
          <div class="grid-3">
            <div><strong>Population</strong><div>${fmt(dataset.population, 0)}</div></div>
            <div><strong>Agents / concepts</strong><div>${fmt(dataset.agent_count, 0)} / ${fmt(dataset.concept_count, 0)}</div></div>
            <div><strong>Domains / tasks</strong><div>${fmt(dataset.domain_count, 0)} / ${fmt(dataset.task_count, 0)}</div></div>
            <div><strong>Base pass rate</strong><div>${pct1(dataset.label_rate)}</div></div>
            <div><strong>Embedding inputs / tokens</strong><div>${fmt(prep.embedding_inputs, 0)} / ${fmt(prep.embedding_input_tokens, 0)}</div></div>
            <div><strong>Avg tokens per session</strong><div>${avgTokens === null ? 'n/a' : fmt(avgTokens, 0)}</div></div>
            <div><strong>Cache hit</strong><div>${String(Boolean(ledger.cache_hit))}</div></div>
            <div><strong>Model / deployment</strong><div>${safeText(ledger.embedding_model_id || prep.embedding_model_id || 'n/a')} / ${safeText(ledger.embedding_deployment_id || prep.embedding_deployment_id || 'n/a')}</div></div>
            <div><strong>Packet cap</strong><div>${fmt(ledger.cache_provenance?.max_session_packet_tokens, 0)}</div></div>
          </div>
          <p class="caption ${weak ? 'warn' : ''}">${weak ? 'Caution: retained variance below 50% (historical compression summary is weak).' : 'Retained variance is above 50%.'}</p>
          <p class="caption ${nearCap ? 'warn' : ''}">${nearCap ? 'Average packet tokens are near the 8191 cap; per-session truncation incidence is not recorded in this evidence packet.' : 'Per-session truncation incidence is not recorded in this evidence packet.'}</p>
        </div>`;
      }).join('');
      panel.innerHTML = cards || '<div class="panel"><p>No dataset profile rows available.</p></div>';
    }
    function baselineMagnitude(metricId, datasetId) {
      const runs = rowsForFilter().filter((row) => row.method_id === 'pca8_idw_binary_euclidean' && (datasetId === 'all' || String(row.dataset_id) === String(datasetId)));
      const values = runs.map((row) => metricFromRun(row, metricId)).filter(Number.isFinite);
      return mean(values);
    }
    function pairedRowsFor(metricId, datasetId) {
      return REPORT.paired_rows.filter((row) => row.metric_id === metricId && Number(row.n) > 0 && (datasetId === 'all' || String(row.dataset_id) === String(datasetId)) && (activeBudget() === 'all' || String(row.budget_pct) === activeBudget()));
    }
    function pairedCellsFor(metricId, datasetId) {
      return REPORT.paired_cells.filter((cell) => cell.metric_id === metricId && (datasetId === 'all' || String(cell.dataset_id) === String(datasetId)) && (activeBudget() === 'all' || String(cell.budget_pct) === activeBudget()) && (activeRepetition() === 'all' || String(cell.repetition_index) === activeRepetition()));
    }
    function objectiveVerdict(metricId, datasetId) {
      const rows = pairedRowsFor(metricId, datasetId);
      const cells = pairedCellsFor(metricId, datasetId);
      const wins = rows.reduce((sum, row) => sum + Number(row.wins || 0), 0);
      const ties = rows.reduce((sum, row) => sum + Number(row.ties || 0), 0);
      const losses = rows.reduce((sum, row) => sum + Number(row.losses || 0), 0);
      const deltas = cells.map((cell) => Number(cell.delta_cosine_minus_euclidean)).filter(Number.isFinite);
      const baseline = baselineMagnitude(metricId, datasetId);
      return verdictFromPaired(metricId, deltas, wins, ties, losses, baseline);
    }
    function combineObjectiveVerdicts(verdicts) {
      const nonEmpty = verdicts.filter((row) => row && row.n > 0);
      if (!nonEmpty.length) return { verdict: VERDICT_WORDS.mixed, evidence: 'No paired evidence in this filter scope.' };
      if (nonEmpty.length === 1) {
        const single = nonEmpty[0];
        const singleEvidence = `${single.verdict} (n=${single.n}, W/T/L=${single.wins}/${single.ties}/${single.losses}${single.meanDelta === null ? '' : `, mean delta=${fmt(single.meanDelta, 4)}`})`;
        return { verdict: single.verdict, evidence: singleEvidence };
      }
      const strong = nonEmpty.filter((row) => row.evidenceLevel === 'strong');
      if (!strong.length) return { verdict: VERDICT_WORDS.mixed, evidence: 'Paired signals are present but consistency/margin thresholds are not met.' };
      const cosineStrong = strong.filter((row) => row.verdict === VERDICT_WORDS.cosine).length;
      const euclideanStrong = strong.filter((row) => row.verdict === VERDICT_WORDS.euclidean).length;
      if (cosineStrong && !euclideanStrong) return { verdict: VERDICT_WORDS.cosine, evidence: `Strong paired evidence in ${cosineStrong}/${strong.length} tracked metrics.` };
      if (euclideanStrong && !cosineStrong) return { verdict: VERDICT_WORDS.euclidean, evidence: `Strong paired evidence in ${euclideanStrong}/${strong.length} tracked metrics.` };
      return { verdict: VERDICT_WORDS.mixed, evidence: 'Tracked metrics disagree or only lean-level signals are available.' };
    }
    function pcaVarianceCaveat(datasetId) {
      const dataset = REPORT.datasets.find((row) => String(row.dataset_id) === String(datasetId));
      const variance = Number(dataset?.pca?.explained_variance_total);
      if (!Number.isFinite(variance) || variance >= 0.5) return null;
      return `PCA-8 retains ${pct1(variance)}; interpret arm verdicts cautiously.`;
    }
    function rawIncrementalRuntimeSummary(datasetId) {
      const rows = REPORT.rows.filter((row) => {
        if (datasetId !== 'all' && String(row.dataset_id) !== String(datasetId)) return false;
        if (activeBudget() !== 'all' && String(row.budget_pct) !== String(activeBudget())) return false;
        if (activeRepetition() !== 'all' && String(row.repetition_index) !== String(activeRepetition())) return false;
        return true;
      });
      const byMethod = {};
      for (const methodId of METHOD_IDS) {
        const vals = rows.filter((row) => row.method_id === methodId).map((row) => incrementalLatency(row)).filter(Number.isFinite);
        byMethod[methodId] = vals.length ? mean(vals) : null;
      }
      const cosine = byMethod['pca8_idw_binary_cosine'];
      const euclidean = byMethod['pca8_idw_binary_euclidean'];
      if (cosine === null || euclidean === null) {
        return { verdict: 'Mixed / no clear winner', evidence: 'No raw incremental runtime rows in this scope.', pairedN: 0, cosineMean: null, euclideanMean: null, absDelta: null, relDelta: null };
      }
      const absDelta = cosine - euclidean;
      const relDelta = euclidean > 0 ? absDelta / euclidean : null;
      const direction = Math.abs(absDelta) <= 1e-9 ? 'tie' : (absDelta < 0 ? 'Cosine' : 'Euclidean');
      const directionText = Math.abs(absDelta) <= 1e-9 ? 'Lower observed raw mean: tie' : `Lower observed raw mean: ${direction}`;
      const verdict = 'Mixed / no clear winner';
      return { verdict, evidence: `${directionText}; not a paired winner claim; cosine=${fmt(cosine, 4)}s, euclidean=${fmt(euclidean, 4)}s, abs diff=${fmt(absDelta, 4)}s, rel diff=${relDelta === null ? 'n/a' : pct(Math.abs(relDelta), 1)}, paired n=0 / no paired CI`, pairedN: 0, cosineMean: cosine, euclideanMean: euclidean, absDelta, relDelta };
    }
    function renderDecisionScorecard() {
      const container = document.getElementById('decisionScorecard');
      const datasetScope = activeDataset() === 'all' ? 'All corpora' : activeDataset();
      const budgetScope = activeBudget() === 'all' ? 'All budgets' : `${activeBudget()}% budget`;
      const datasetList = selectedDatasets();
      const cards = OBJECTIVE_CARDS.map((objective) => {
        if (objective.rawIncremental) {
          if (activeDataset() === 'all') {
            const perCorpus = datasetList.map((datasetId) => {
              const summary = rawIncrementalRuntimeSummary(datasetId);
              const caveat = pcaVarianceCaveat(datasetId) ? ` ${pcaVarianceCaveat(datasetId)}` : '';
              return `${datasetId}: cosine=${summary.cosineMean === null ? 'n/a' : fmt(summary.cosineMean, 4)}s, euclidean=${summary.euclideanMean === null ? 'n/a' : fmt(summary.euclideanMean, 4)}s, abs diff=${summary.absDelta === null ? 'n/a' : fmt(summary.absDelta, 4)}s${caveat}`;
            });
            return `<div class="decision-card"><strong>${objective.label}</strong><div class="verdict">Mixed / no clear winner</div><div class="small muted">Scope: ${safeText(datasetScope)}, ${safeText(budgetScope)}</div><div class="corpus-verdicts">Directional raw means only; no paired runtime verdict for any corpus.</div>${evidenceDetailsHtml('Observed means by corpus', perCorpus.join(' | '))}<div class="small muted">Shared Search is excluded; no paired CI or winner claim.</div></div>`;
          }
          const summary = rawIncrementalRuntimeSummary(activeDataset());
          const caveat = pcaVarianceCaveat(activeDataset());
          const runtimeEvidence = `cosine=${summary.cosineMean === null ? 'n/a' : fmt(summary.cosineMean, 4)}s, euclidean=${summary.euclideanMean === null ? 'n/a' : fmt(summary.euclideanMean, 4)}s, abs diff=${summary.absDelta === null ? 'n/a' : fmt(summary.absDelta, 4)}s, rel diff=${summary.relDelta === null ? 'n/a' : pct(Math.abs(summary.relDelta), 1)}, paired n=${summary.pairedN} / no paired CI. ${caveat || ''}`;
          return `<div class="decision-card"><strong>${objective.label}</strong><div class="verdict">${summary.verdict}</div><div class="small muted">Scope: ${safeText(datasetScope)}, ${safeText(budgetScope)}</div><div class="corpus-verdicts">Directional raw means only; no paired runtime verdict.</div>${evidenceDetailsHtml('Observed means', runtimeEvidence)}<div class="small muted">Shared Search excluded; not a paired winner claim.</div></div>`;
        }
        if (activeDataset() === 'all') {
          const perCorpus = datasetList.map((datasetId) => {
            const metricVerdicts = objective.metricIds.map((metricId) => ({ metricId, verdict: objectiveVerdict(metricId, datasetId) }));
            const merged = combineObjectiveVerdicts(metricVerdicts.map((item) => item.verdict));
            const evidence = metricVerdicts.map((item) => {
              const v = item.verdict;
              return `${METRIC_META[item.metricId]?.label || item.metricId}: ${v.verdict}, W/T/L=${v.wins}/${v.ties}/${v.losses}, n=${v.n}`;
            }).join(', ');
            const caveat = pcaVarianceCaveat(datasetId) ? ` ${pcaVarianceCaveat(datasetId)}` : '';
            return { verdict: `${datasetId}: ${merged.verdict}`, evidence: `${datasetId}: ${evidence}${caveat}` };
          });
          return `<div class="decision-card"><strong>${objective.label}</strong><div class="verdict">Varies by corpus</div><div class="small muted">Scope: ${safeText(datasetScope)}, ${safeText(budgetScope)}</div><div class="corpus-verdicts">${safeText(perCorpus.map((item) => item.verdict).join(' | '))}</div>${evidenceDetailsHtml('Evidence by corpus', perCorpus.map((item) => item.evidence).join(' | '))}<div class="small muted">No pooled winner is claimed.</div></div>`;
        }
        const metricVerdicts = objective.metricIds.map((metricId) => ({ metricId, verdict: objectiveVerdict(metricId, activeDataset()) }));
        const merged = combineObjectiveVerdicts(metricVerdicts.map((item) => item.verdict));
        const pairedN = metricVerdicts.reduce((sum, item) => sum + Number(item.verdict.n || 0), 0);
        const evidenceSummary = metricVerdicts.map((item) => {
          const v = item.verdict;
          const margin = item.metricId === 'actual_token_count'
            ? (v.relMagnitude === null ? 'n/a' : `${(v.relMagnitude * 100).toFixed(1)}%`)
            : (v.relMagnitude === null ? 'n/a' : fmt(v.relMagnitude, 4));
          return `${METRIC_META[item.metricId]?.label || item.metricId}: ${v.verdict}, W/T/L=${v.wins || 0}/${v.ties || 0}/${v.losses || 0}, mean delta=${v.meanDelta === null ? 'n/a' : fmt(v.meanDelta, 4)}, margin=${margin}, n=${v.n || 0}`;
        }).join(' | ');
        const caveat = pcaVarianceCaveat(activeDataset());
        return `<div class="decision-card"><strong>${objective.label}</strong><div class="verdict">${merged.verdict}</div><div class="small muted">Scope: ${safeText(datasetScope)}, ${safeText(budgetScope)}</div>${evidenceDetailsHtml('Paired evidence', `paired n=${pairedN}; ${evidenceSummary}${caveat ? `; ${caveat}` : ''}`)}<div class="small muted">Replay sensitivity on this corpus, not population proof.</div></div>`;
      });
      container.innerHTML = cards.join('');
    }
    function renderExecutiveAnswer() {
      const metricId = activeMetric();
      const meta = METRIC_META[metricId] || { label: metricId, direction: 'lower_better' };
      const rows = filteredPairedRows(metricId);
      const cells = filteredPairedCells(metricId);
      const wins = rows.reduce((sum, row) => sum + Number(row.wins || 0), 0);
      const ties = rows.reduce((sum, row) => sum + Number(row.ties || 0), 0);
      const losses = rows.reduce((sum, row) => sum + Number(row.losses || 0), 0);
      const n = rows.reduce((sum, row) => sum + Number(row.n || 0), 0);
      const deltas = cells.map((cell) => Number(cell.delta_cosine_minus_euclidean)).filter(Number.isFinite);
      const meanDelta = deltas.length ? deltas.reduce((s, v) => s + v, 0) / deltas.length : null;
      const direction = meanDelta === null
        ? 'no paired-difference evidence'
        : (meta.direction === 'higher_better'
          ? (meanDelta > 0 ? 'cosine-leading direction' : (meanDelta < 0 ? 'euclidean-leading direction' : 'parity direction'))
          : (meanDelta < 0 ? 'cosine-leading direction' : (meanDelta > 0 ? 'euclidean-leading direction' : 'parity direction')));
      const datasetScope = activeDataset() === 'all' ? 'all corpora' : activeDataset();
      const budgetScope = activeBudget() === 'all' ? 'all budgets' : `${activeBudget()}% budget`;
      const replayScope = activeRepetition() === 'all' ? 'all paired replays' : `replay ${activeRepetition()}`;
      if (activeDataset() === 'all') {
        const perCorpus = selectedDatasets().map((datasetId) => {
          const verdict = objectiveVerdict(metricId, datasetId);
          return `${datasetId}: ${verdict.verdict} (n=${verdict.n}, W/T/L=${verdict.wins}/${verdict.ties}/${verdict.losses})`;
        }).join(' | ');
        document.getElementById('executiveAnswer').textContent = `No pooled winner is claimed for ${meta.label || metricId}. At ${budgetScope}, ${replayScope}: ${perCorpus}. Results are objective-specific and should be read by corpus.`;
        document.getElementById('scopeCaption').textContent = `Selected metric: ${meta.label || metricId}. Delta rule is direction-aware (${meta.direction === 'higher_better' ? 'positive favors cosine' : 'negative favors cosine'}).`;
        return;
      }
      const summary = n > 0
        ? `For ${meta.label || metricId}, scope=${datasetScope}, ${budgetScope}, ${replayScope}: paired n=${n}, wins/ties/losses=${wins}/${ties}/${losses}, mean delta (cosine-euclidean)=${fmt(meanDelta, 4)} (${direction}).`
        : `No paired-difference evidence exists for ${meta.label || metricId} under scope=${datasetScope}, ${budgetScope}, ${replayScope}; this metric has run-level summaries but no paired-difference evidence in the selected scope.`;
      document.getElementById('executiveAnswer').textContent = `Winner claims remain objective-specific for this corpus. ${summary} This evidence does not claim a universal winner.`;
      document.getElementById('scopeCaption').textContent = `Selected metric: ${meta.label || metricId}. Delta rule is direction-aware (${meta.direction === 'higher_better' ? 'positive favors cosine' : 'negative favors cosine'}).`;
    }
    function renderBudgetCapCaption() {
      const text = selectedDatasets().map((datasetId) => {
        const budgets = [...new Set(REPORT.rows.filter((row) => String(row.dataset_id) === datasetId).map((row) => Number(row.budget_pct)))].sort((a, b) => a - b);
        const parts = budgets.map((budgetPct) => {
          const cap = representativeCap(datasetId, budgetPct);
          const mark = lowNFromCap(cap) ? ' low-N' : '';
          return `${budgetPct}% -> cap ${Number.isFinite(cap) ? cap : 'n/a'}${mark}`;
        });
        return `${datasetId}: ${parts.join(', ')}`;
      }).join(' | ');
      document.getElementById('budgetCapCaption').textContent = `Absolute budget caps by corpus: ${text}. Cells with cap < 10 are flagged as low-N.`;
    }
    function renderBudgetTrend() {
      const metricId = activeMetric();
      const meta = METRIC_META[metricId];
      const directionText = meta?.direction === 'higher_better' ? 'higher is better' : 'lower is better';
      document.getElementById('budgetTrendTitle').textContent = `Budget trend: ${meta?.label || metricId} (${directionText})`;
      const datasetsForPanels = selectedDatasets();
      const chart = document.getElementById('budgetTrendChart');
      const panels = datasetsForPanels.map((datasetId) => {
        let rows;
        if (activeRepetition() === 'all') {
          rows = filteredAggregate(metricId, datasetId, 'all');
        } else {
          rows = REPORT.rows
            .filter((row) => String(row.dataset_id) === datasetId && String(row.repetition_index) === activeRepetition())
            .map((row) => ({ ...row, mean: metricFromRun(row, metricId), p05: metricFromRun(row, metricId), p95: metricFromRun(row, metricId), n: 1 }))
            .filter((row) => Number.isFinite(Number(row.mean)));
        }
        if (!rows.length) return `<div class="panel"><h3>${safeText(datasetId)}</h3><p>No trend data for this slice.</p></div>`;
        const budgets = [...new Set(rows.map((r) => Number(r.budget_pct)))].sort((a, b) => a - b);
        const allValues = rows.flatMap((r) => [Number(r.p05), Number(r.p95), Number(r.mean)]).filter(Number.isFinite);
        let minVal = Math.min(...allValues, 0);
        let maxVal = Math.max(...allValues);
        if (meta?.direction === 'higher_better') { minVal = 0; maxVal = Math.max(1, maxVal); }
        if (maxVal === minVal) maxVal = minVal + 1;
        const x = (budget) => 60 + budgets.indexOf(Number(budget)) * (680 / Math.max(1, budgets.length - 1));
        const y = (value) => 20 + ((maxVal - value) / Math.max(1e-9, maxVal - minVal)) * 216;
        let svg = `<svg viewBox="0 0 760 280" aria-label="${safeText(datasetId)} budget trend" role="img">`;
        for (let i = 0; i <= 4; i += 1) {
          const value = minVal + (maxVal - minVal) * (i / 4);
          const yy = y(value);
          svg += `<line x1="60" y1="${yy}" x2="740" y2="${yy}" stroke="#e8ece9"/><text x="52" y="${yy + 4}" text-anchor="end" font-size="11" fill="#58656d">${meta?.format === 'percent' ? pct(value) : fmt(value, meta?.format === 'integer' ? 0 : 2)}</text>`;
        }
        budgets.forEach((budget) => { const xx = x(budget); const active = activeBudget() !== 'all' && String(budget) === activeBudget(); const cap = representativeCap(datasetId, budget); const lowN = lowNFromCap(cap); svg += `<line x1="${xx}" y1="20" x2="${xx}" y2="236" stroke="${active ? '#ad6520' : '#edf0ed'}" stroke-width="${active ? 2 : 1}"/><text x="${xx}" y="264" text-anchor="middle" font-size="11" fill="${active ? '#7b4b12' : '#58656d'}">${budget}% / N${Number.isFinite(cap) ? cap : '?'}</text>${lowN ? `<text x="${xx}" y="278" text-anchor="middle" font-size="10" fill="#7b4b12">low-N</text>` : ''}`; });
        METHOD_IDS.forEach((methodId) => {
          const methodRows = rows.filter((row) => row.method_id === methodId).sort((a, b) => Number(a.budget_pct) - Number(b.budget_pct));
          if (!methodRows.length) return;
          methodRows.forEach((row) => { const xx = x(row.budget_pct); svg += `<line x1="${xx}" y1="${y(Number(row.p05))}" x2="${xx}" y2="${y(Number(row.p95))}" stroke="${COLORS[methodId]}" stroke-width="4" stroke-opacity="0.45"/><circle cx="${xx}" cy="${y(Number(row.mean))}" r="5" fill="${COLORS[methodId]}" stroke="#fff" stroke-width="2"><title>${METHOD_LABELS[methodId]} ${row.budget_pct}%: ${fmt(row.mean)}, P05-P95 ${fmt(row.p05)}-${fmt(row.p95)}, n=${row.n}</title></circle>`; });
          svg += `<polyline points="${methodRows.map((row) => `${x(row.budget_pct)},${y(Number(row.mean))}`).join(' ')}" fill="none" stroke="${COLORS[methodId]}" stroke-width="2.5"/>`;
        });
        return `<div class="panel"><h3>${safeText(datasetId)}</h3>${svg}</svg><p class="small muted">${safeText(meta?.label || metricId)}; ${meta?.direction === 'higher_better' ? 'higher' : 'lower'} is better. Budget labels include cap N; cap < 10 implies low-N volatility.</p></div>`;
      });
      chart.innerHTML = panels.join('') + '<div class="legend"><span><i class="dot" style="background:#3e7db8"></i>Cosine</span><span><i class="dot" style="background:#cc8a2d"></i>Euclidean</span></div>';
      const budgetScope = activeBudget() === 'all' ? 'all budget caps' : `${activeBudget()}% cap-highlighted`;
      document.getElementById('budgetTrendStory').innerHTML = storyBlockHtml(
        `${meta?.label || metricId} by budget with point means and P05-P95 intervals per method; legend is blue=cosine and amber=Euclidean.`,
        `${directionText}; budget labels include cap N and low-N flags when cap < 10.`,
        `${scopeVerdictSummary(metricId)} Budget scope: ${budgetScope}.`
      );
    }
    function renderPairedDelta() {
      const metricId = activeMetric();
      const metricMeta = METRIC_META[metricId];
      const directionText = metricMeta?.direction === 'higher_better' ? 'higher is better' : 'lower is better';
      document.getElementById('deltaPanelTitle').textContent = `Paired delta: ${metricMeta?.label || metricId} (${directionText})`;
      const cells = filteredPairedCells(metricId);
      const plot = document.getElementById('pairedDeltaPlot');
      if (!cells.length) {
        plot.innerHTML = '<p>No paired delta cells available.</p>';
        document.getElementById('pairedDeltaStory').innerHTML = storyBlockHtml(
          `Per-replay paired deltas for ${metricMeta?.label || metricId} in the active filter scope.`,
          `${directionText}; delta would be cosine minus Euclidean around a zero reference.`,
          'No paired delta evidence is available for this filter, so no arm verdict is claimed.'
        );
        return;
      }
      const deltas = cells.map((cell) => Number(cell.delta_cosine_minus_euclidean)).filter(Number.isFinite);
      const width = 760; const height = Math.min(700, Math.max(260, 60 + cells.length * 5)); const left = 40; const right = 20; const top = 20; const bottom = 30;
      const maxAbs = Math.max(...deltas.map((v) => Math.abs(v)), 0.0001);
      const x = (delta) => left + ((delta + maxAbs) / (maxAbs * 2)) * (width - left - right);
      const y = (index) => top + index * ((height - top - bottom) / Math.max(1, cells.length - 1));
      const zeroX = x(0);
      let svg = `<svg viewBox="0 0 760 ${height}" aria-label="Paired delta plot" role="img">`;
      svg += `<line x1="${zeroX}" x2="${zeroX}" y1="20" y2="${height - bottom}" stroke="#1a1f21" stroke-dasharray="4 5" />`;
      cells.forEach((cell, index) => {
        const delta = Number(cell.delta_cosine_minus_euclidean);
        const xx = x(delta);
        const yy = y(index);
        svg += `<line x1="${xx}" y1="${yy}" x2="${zeroX}" y2="${yy}" stroke="#a7b0ae" stroke-width="1.5" />`;
        const favorsCosine = (cell.direction || metricMeta?.direction) === 'higher_better' ? delta > 0 : delta < 0;
        svg += `<circle cx="${xx}" cy="${yy}" r="4.5" fill="${delta === 0 ? '#c2c8c4' : favorsCosine ? '#3e7db8' : '#cc8a2d'}"><title>${safeText(cell.dataset_id)} ${cell.budget_pct}% replay ${cell.repetition_index}: delta ${fmt(delta)}</title></circle>`;
      });
      svg += `<text x="${left}" y="${height - 8}" text-anchor="start" fill="#58656d" font-size="11">min ${fmt(-maxAbs, 4)}</text>`;
      svg += `<text x="${zeroX}" y="${height - 8}" text-anchor="middle" fill="#1a1f21" font-size="11">0</text>`;
      svg += `<text x="${width - right}" y="${height - 8}" text-anchor="end" fill="#58656d" font-size="11">max ${fmt(maxAbs, 4)}</text>`;
      svg += '</svg>';
      const rule = metricMeta?.direction === 'higher_better' ? 'positive favors cosine; negative favors Euclidean' : 'negative favors cosine; positive favors Euclidean';
      plot.innerHTML = svg + `<div class="legend"><span><i class="dot" style="background:#3e7db8"></i>Cosine favored</span><span><i class="dot" style="background:#cc8a2d"></i>Euclidean favored</span><span>${rule}</span><span>Paired cells n=${cells.length}</span></div>`;
      document.getElementById('pairedDeltaCaption').textContent = `${metricMeta?.label || metricId}; ${directionText}. Delta = cosine - Euclidean with explicit min/zero/max ticks.`;
      document.getElementById('pairedDeltaStory').innerHTML = storyBlockHtml(
        `Per-replay paired deltas for ${metricMeta?.label || metricId} in the active filter scope.`,
        `${rule}; points left/right of zero indicate direction and magnitude.`,
        scopeVerdictSummary(metricId)
      );
    }
    function renderWinLoss() {
      const metricId = activeMetric();
      const directionText = METRIC_META[metricId]?.direction === 'higher_better' ? 'higher is better' : 'lower is better';
      let rows = filteredPairedRows(metricId);
      if (activeRepetition() !== 'all') {
        const cells = filteredPairedCells(metricId);
        const grouped = new Map();
        cells.forEach((cell) => {
          const key = `${cell.dataset_id}|${cell.budget_pct}`;
          const current = grouped.get(key) || { dataset_id: cell.dataset_id, budget_pct: cell.budget_pct, n: 0, wins: 0, ties: 0, losses: 0 };
          const delta = Number(cell.delta_cosine_minus_euclidean);
          const direction = cell.direction || METRIC_META[metricId]?.direction;
          current.n += 1;
          if (Math.abs(delta) <= 1e-12) current.ties += 1;
          else if ((direction === 'higher_better' && delta > 0) || (direction !== 'higher_better' && delta < 0)) current.wins += 1;
          else current.losses += 1;
          grouped.set(key, current);
        });
        rows = [...grouped.values()];
      }
      const plot = document.getElementById('winLossChart');
      if (!rows.length) {
        plot.innerHTML = '<p>No paired rows available.</p>';
        document.getElementById('winLossCounts').innerHTML = '';
        document.getElementById('winLossStory').innerHTML = storyBlockHtml(
          `Paired wins, ties, and losses for ${METRIC_META[metricId]?.label || metricId}.`,
          'Blue would indicate cosine wins, amber Euclidean wins, and gray ties.',
          'No paired rows are available for this filter, so no consistency verdict is claimed.'
        );
        return;
      }
      const totalN = rows.reduce((sum, row) => sum + Number(row.n || 0), 0);
      document.getElementById('winLossTitle').textContent = `Win / tie / loss: ${METRIC_META[metricId]?.label || metricId} (${directionText}; n=${totalN})`;
      const width = 760; const height = Math.max(220, 60 + rows.length * 32); const left = 10; const right = 8; const top = 26; const barHeight = 18; const plotWidth = width - left - right - 160;
      let svg = `<svg viewBox="0 0 760 ${height}" aria-label="Win loss chart" role="img">`;
      rows.forEach((row, index) => {
        const y = top + index * 36;
        const total = Math.max(1, Number(row.n));
        const winWidth = ((Number(row.wins) || 0) / total) * plotWidth;
        const tieWidth = ((Number(row.ties) || 0) / total) * plotWidth;
        const lossWidth = ((Number(row.losses) || 0) / total) * plotWidth;
        svg += `<text x="${left}" y="${y + 12}" text-anchor="start" font-size="11" fill="#58656d">${safeText(row.dataset_id)} / ${row.budget_pct}%</text>`;
        svg += `<rect x="${left + 150}" y="${y}" width="${winWidth}" height="${barHeight}" fill="#3e7db8" />`;
        svg += `<rect x="${left + 150 + winWidth}" y="${y}" width="${tieWidth}" height="${barHeight}" fill="#c2c8c4" />`;
        svg += `<rect x="${left + 150 + winWidth + tieWidth}" y="${y}" width="${lossWidth}" height="${barHeight}" fill="#cc8a2d" />`;
      });
      svg += '</svg>';
      plot.innerHTML = svg + '<div class="legend"><span><i class="dot" style="background:#3e7db8"></i>Cosine wins</span><span><i class="dot" style="background:#c2c8c4"></i>Ties</span><span><i class="dot" style="background:#cc8a2d"></i>Euclidean wins</span></div>';
      document.getElementById('winLossCounts').innerHTML = `<div class="paired-count-grid">${rows.map((row) => `
        <div class="paired-count-card">
          <strong>${safeText(row.dataset_id)}</strong>
          <div class="meta">
            <div><div class="small muted">Budget</div><div class="value">${row.budget_pct}%</div></div>
            <div><div class="small muted">n</div><div class="value">${row.n}</div></div>
            <div><div class="small muted">Wins</div><div class="value" style="color:#3e7db8">${row.wins}</div></div>
            <div><div class="small muted">Ties</div><div class="value" style="color:#6d7578">${row.ties}</div></div>
            <div><div class="small muted">Losses</div><div class="value" style="color:#cc8a2d">${row.losses}</div></div>
          </div>
        </div>
      `).join('')}</div>`;
      const wins = rows.reduce((sum, row) => sum + Number(row.wins || 0), 0);
      const ties = rows.reduce((sum, row) => sum + Number(row.ties || 0), 0);
      const losses = rows.reduce((sum, row) => sum + Number(row.losses || 0), 0);
      document.getElementById('winLossStory').innerHTML = storyBlockHtml(
        `Paired wins/ties/losses by dataset-budget cell for ${METRIC_META[metricId]?.label || metricId}, with n shown per cell.`,
        `${directionText}; blue segments are cosine wins, amber are Euclidean wins, gray are ties.`,
        activeDataset() === 'all' ? scopeVerdictSummary(metricId) : `${scopeVerdictSummary(metricId)} Visible totals W/T/L=${wins}/${ties}/${losses}.`
      );
    }
    function renderClassificationMatrix() {
      const scopeMetrics = ['imputed_only', 'judged_plus_imputed'];
      const labels = ['accuracy', 'precision', 'recall', 'f1'];
      const rows = rowsForFilter().filter((row) => row.method_id && METHOD_IDS.includes(row.method_id));
      const container = document.getElementById('classificationMatrix');
      if (!rows.length) {
        container.innerHTML = '<p>No classification data available.</p>';
        document.getElementById('classificationMiniSummary').textContent = '';
        document.getElementById('classificationStory').innerHTML = storyBlockHtml(
          'Imputed-only and judged-plus-imputed classification means by corpus and method.',
          'Higher values would be better; imputed-only is the primary comparison.',
          'No classification rows are available for this filter, so no arm verdict is claimed.'
        );
        return;
      }
      const groups = selectedDatasets();
      document.getElementById('classificationTitle').textContent = `Classification comparison: ${activeBudget() === 'all' ? 'all budgets' : `${activeBudget()}% budget`}`;
      const mini = groups.map((datasetId) => {
        const metricVerdicts = ['imputed_only_accuracy', 'imputed_only_f1'].map((metricId) => ({ metricId, verdict: objectiveVerdict(metricId, datasetId) }));
        const merged = combineObjectiveVerdicts(metricVerdicts.map((item) => item.verdict));
        const cosineRows = rows.filter((row) => String(row.dataset_id) === datasetId && row.method_id === 'pca8_idw_binary_cosine');
        const euRows = rows.filter((row) => String(row.dataset_id) === datasetId && row.method_id === 'pca8_idw_binary_euclidean');
        const meanAccCos = mean(cosineRows.map((row) => getNumber(row, 'embedding_metrics.imputed_only.accuracy')).filter(Number.isFinite));
        const meanAccEu = mean(euRows.map((row) => getNumber(row, 'embedding_metrics.imputed_only.accuracy')).filter(Number.isFinite));
        const meanF1Cos = mean(cosineRows.map((row) => getNumber(row, 'embedding_metrics.imputed_only.f1')).filter(Number.isFinite));
        const meanF1Eu = mean(euRows.map((row) => getNumber(row, 'embedding_metrics.imputed_only.f1')).filter(Number.isFinite));
        const marginAcc = (meanAccCos === null || meanAccEu === null) ? null : meanAccCos - meanAccEu;
        const marginF1 = (meanF1Cos === null || meanF1Eu === null) ? null : meanF1Cos - meanF1Eu;
        return `${datasetId}: ${merged.verdict}; imputed accuracy cosine=${pct1(meanAccCos)} vs euclidean=${pct1(meanAccEu)} (delta ${marginAcc === null ? 'n/a' : pct1(marginAcc)}), imputed F1 cosine=${pct1(meanF1Cos)} vs euclidean=${pct1(meanF1Eu)} (delta ${marginF1 === null ? 'n/a' : pct1(marginF1)}).`;
      }).join(' ');
      document.getElementById('classificationMiniSummary').textContent = `${mini} Judged+imputed cards are secondary and observed-inflated.`;
      container.innerHTML = groups.map((datasetId) => {
        const datasetRows = rows.filter((row) => String(row.dataset_id) === datasetId);
        const cards = scopeMetrics.flatMap((scope) => METHOD_IDS.map((methodId) => {
          const entries = datasetRows.filter((row) => row.method_id === methodId && row.embedding_metrics && row.embedding_metrics[scope]);
          const byMetric = {};
          labels.forEach((metric) => {
            const values = entries.map((row) => Number(row.embedding_metrics[scope][metric])).filter(Number.isFinite);
            byMetric[metric] = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
          });
          return `<div class="classification-card">
            <div class="head">
              <strong>${METHOD_LABELS[methodId]}</strong>
              <span class="small muted">${scope === 'imputed_only' ? 'Imputed-only' : 'Judged + imputed'}</span>
            </div>
            <div class="metrics">${labels.map((metric) => `<div class="metric"><small>${metric}</small><div>${byMetric[metric] === null ? 'n/a' : `${(byMetric[metric] * 100).toFixed(1)}%`}</div></div>`).join('')}</div>
          </div>`;
        }));
        return `<div class="panel"><h3>${safeText(datasetId)}</h3><div class="classification-grid">${cards.join('')}</div></div>`;
      }).join('');
      const allVerdict = activeDataset() === 'all' ? 'Mixed / no clear winner by design across corpora; review per-corpus mini-summary.' : combineObjectiveVerdicts([
        objectiveVerdict('imputed_only_accuracy', activeDataset()),
        objectiveVerdict('imputed_only_f1', activeDataset())
      ]).verdict;
      document.getElementById('classificationStory').innerHTML = storyBlockHtml(
        'Imputed-only and judged+imputed classification means by corpus and method.',
        'Higher is better; focus on imputed-only for model-selection signal while judged+imputed is context only.',
        allVerdict
      );
    }
    function renderCoverageAndCosts() {
      const rows = rowsForFilter();
      const conceptChart = document.getElementById('conceptCoverageChart');
      const tokenContainer = document.getElementById('tokensAndLatency');
      const coveragePanels = selectedDatasets().map((datasetId) => {
        const datasetRows = rows.filter((row) => String(row.dataset_id) === datasetId);
        const conceptValues = METHOD_IDS.map((methodId) => {
          const values = datasetRows.filter((r) => r.method_id === methodId && r.coverage && r.coverage.concept && Number.isFinite(Number(r.coverage.concept.coverage_ratio))).map((r) => Number(r.coverage.concept.coverage_ratio));
          return { method: methodId, value: values.length ? values.reduce((sum, v) => sum + v, 0) / values.length : null };
        }).filter((item) => item.value !== null);
        const coverageSvg = conceptValues.length ? `<svg viewBox="0 0 640 220" aria-label="Concept coverage ${safeText(datasetId)}" role="img">${conceptValues.map((item, index) => { const x = 120 + index * 200; const h = 120 * item.value; return `<rect x="${x}" y="${200 - h}" width="90" height="${h}" fill="${COLORS[item.method]}" />` + `<text x="${x + 45}" y="200" text-anchor="middle" font-size="11" fill="#58656d">${METHOD_LABELS[item.method]}</text>` + `<text x="${x + 45}" y="${190 - h}" text-anchor="middle" font-size="11" fill="#58656d">${(item.value * 100).toFixed(1)}%</text>`; }).join('')}</svg>` : '<p>No coverage rows available.</p>';
        const coveragePairedRows = pairedRowsFor('coverage_concept_ratio', datasetId);
        const coverageWins = coveragePairedRows.reduce((sum, row) => sum + Number(row.wins || 0), 0);
        const coverageTies = coveragePairedRows.reduce((sum, row) => sum + Number(row.ties || 0), 0);
        const coverageLosses = coveragePairedRows.reduce((sum, row) => sum + Number(row.losses || 0), 0);
        const coverageVerdict = objectiveVerdict('coverage_concept_ratio', datasetId);
        const coverageNarrative = `<p class="small muted">Concept coverage evidence: verdict=${coverageVerdict.verdict}; paired wins/ties/losses=${coverageWins}/${coverageTies}/${coverageLosses}, n=${coverageVerdict.n || 0}. ${pcaVarianceCaveat(datasetId) || ''}</p>`;
        return `<div class="panel coverage-panel"><h3>${safeText(datasetId)}</h3><div class="local-scroll">${coverageSvg}</div>${coverageNarrative}</div>`;
      });
      const tokenPanels = selectedDatasets().map((datasetId) => {
        const datasetRows = rows.filter((row) => String(row.dataset_id) === datasetId);
        const metricRows = METHOD_IDS.map((methodId) => {
          const methodRows = datasetRows.filter((row) => row.method_id === methodId);
          const avgTokens = methodRows.length ? methodRows.map((row) => Number(row.actual_token_count)).filter(Number.isFinite).reduce((s, v) => s + v, 0) / Math.max(1, methodRows.map((row) => Number(row.actual_token_count)).filter(Number.isFinite).length) : null;
          const incVals = methodRows.map((row) => incrementalLatency(row)).filter(Number.isFinite);
          const avgInc = incVals.length ? incVals.reduce((s, v) => s + v, 0) / incVals.length : null;
          return { methodId, avgTokens, avgInc };
        });
        const cosine = metricRows.find((row) => row.methodId === 'pca8_idw_binary_cosine');
        const euclidean = metricRows.find((row) => row.methodId === 'pca8_idw_binary_euclidean');
        const tokenDelta = (cosine?.avgTokens ?? null) !== null && (euclidean?.avgTokens ?? null) !== null ? Number(cosine.avgTokens) - Number(euclidean.avgTokens) : null;
        const tokenRel = tokenDelta !== null && Number.isFinite(euclidean.avgTokens) && Number(euclidean.avgTokens) > 0 ? tokenDelta / Number(euclidean.avgTokens) : null;
        const tokenPairedRows = pairedRowsFor('actual_token_count', datasetId);
        const tokenWins = tokenPairedRows.reduce((sum, row) => sum + Number(row.wins || 0), 0);
        const tokenTies = tokenPairedRows.reduce((sum, row) => sum + Number(row.ties || 0), 0);
        const tokenLosses = tokenPairedRows.reduce((sum, row) => sum + Number(row.losses || 0), 0);
        const tokenVerdict = objectiveVerdict('actual_token_count', datasetId);
        const runtimeDirection = (cosine?.avgInc ?? null) !== null && (euclidean?.avgInc ?? null) !== null ? ((cosine.avgInc < euclidean.avgInc) ? 'Lower observed raw mean: Cosine' : (cosine.avgInc > euclidean.avgInc ? 'Lower observed raw mean: Euclidean' : 'Lower observed raw mean: tie')) : 'Runtime directional signal unavailable';
        const costTable = `<table><thead><tr><th>Method</th><th>Avg judged tokens</th><th>Avg incremental runtime (selection+IDW s)</th></tr></thead><tbody>${metricRows.map((row) => `<tr><td>${METHOD_LABELS[row.methodId]}</td><td>${row.avgTokens === null ? 'n/a' : fmt(row.avgTokens, 0)}</td><td>${row.avgInc === null ? 'n/a' : fmt(row.avgInc, 4)}</td></tr>`).join('')}</tbody></table>`;
        const tokenNarrative = `<p class="small muted">Judged-token evidence: cosine=${cosine?.avgTokens === null || cosine?.avgTokens === undefined ? 'n/a' : fmt(cosine.avgTokens, 0)}, euclidean=${euclidean?.avgTokens === null || euclidean?.avgTokens === undefined ? 'n/a' : fmt(euclidean.avgTokens, 0)}, absolute diff (cosine-euclidean)=${tokenDelta === null ? 'n/a' : fmt(tokenDelta, 0)}, relative diff=${tokenRel === null ? 'n/a' : pct(tokenRel, 1)}. Verdict=${tokenVerdict.verdict}; paired wins/ties/losses=${tokenWins}/${tokenTies}/${tokenLosses}, n=${tokenVerdict.n || 0}.</p>`;
        const runtimeNarrative = `<p class="small muted">Incremental runtime is directional/raw-row only from selection+IDW columns; ${runtimeDirection}; not a paired winner claim; this is not interpreted as per-replay incremental runtime and remains separate from paired winner logic.</p>`;
        const costCaption = `Cost view: judged tokens and raw incremental latency only; shared Search reference is excluded from winner logic.${pcaVarianceCaveat(datasetId) ? ` ${pcaVarianceCaveat(datasetId)}` : ''}`;
        return `<div class="panel token-cost-panel"><h3>${safeText(datasetId)}</h3><div class="local-scroll">${costTable}</div>${tokenNarrative}${runtimeNarrative}<p class="caption">${safeText(costCaption)}</p></div>`;
      });
      conceptChart.innerHTML = coveragePanels.join('');
      tokenContainer.innerHTML = tokenPanels.join('');
      const tokenStoryVerdict = activeDataset() === 'all' ? 'Mixed / no clear winner; inspect corpus token deltas and paired evidence.' : objectiveVerdict('actual_token_count', activeDataset()).verdict;
      document.getElementById('tokensLatencyStory').innerHTML = storyBlockHtml(
        'Judged-token means and directional incremental selection+IDW runtime by corpus; shared Search latency is excluded from the winner logic.',
        'For token cost, lower is better; blue encodes cosine and amber encodes Euclidean. Runtime rows are raw-directional, not paired CI evidence.',
        `${tokenStoryVerdict}`
      );
      const coverageVerdict = activeDataset() === 'all' ? 'Mixed / no clear winner across corpora.' : objectiveVerdict('coverage_concept_ratio', activeDataset()).verdict;
      document.getElementById('coverageStory').innerHTML = storyBlockHtml(
        'Concept coverage means by corpus for cosine and Euclidean.',
        'Higher is better; compare bars and confirm with paired wins/ties/losses where available.',
        `${coverageVerdict}`
      );
    }
    function renderReplayUncertainty() {
      const metricId = activeMetric();
      const plot = document.getElementById('replayUncertaintyChart');
      const directionText = METRIC_META[metricId]?.direction === 'higher_better' ? 'higher is better' : 'lower is better';
      document.getElementById('uncertaintyTitle').textContent = `Replay uncertainty and frequency context: ${METRIC_META[metricId]?.label || metricId} (${directionText})`;
      if (activeRepetition() !== 'all') {
        plot.innerHTML = '<p>A single replay is selected. Replay intervals require multiple repetitions; clear the repetition filter to view P05-P95 and Student-t CI95.</p>';
        document.getElementById('uncertaintyStory').innerHTML = storyBlockHtml(
          `Replay intervals for ${METRIC_META[metricId]?.label || metricId}.`,
          'P05-P95 and CI95 require multiple paired replay observations.',
          'A single replay is selected, so interval stability and an uncertainty-based arm verdict are unavailable.'
        );
        return;
      }
      const rows = filteredAggregate(metricId);
      if (!rows.length) {
        plot.innerHTML = '<p>No uncertainty rows available.</p>';
        document.getElementById('uncertaintyStory').innerHTML = storyBlockHtml(
          `Replay intervals for ${METRIC_META[metricId]?.label || metricId}.`,
          `${directionText}; wider intervals would indicate greater replay sensitivity.`,
          'No aggregate uncertainty rows are available for this filter, so no stability verdict is claimed.'
        );
        return;
      }
      const width = 760; const height = Math.max(220, 50 + rows.length * 34); const left = 205; const right = 28; const top = 24;
      const minVal = Math.min(...rows.flatMap((r) => [r.p05, r.p95, r.mean_ci95_lower, r.mean_ci95_upper]));
      const maxVal = Math.max(...rows.flatMap((r) => [r.p05, r.p95, r.mean_ci95_lower, r.mean_ci95_upper]));
      let svg = `<svg viewBox="0 0 760 ${height}" aria-label="Replay uncertainty" role="img">`;
      rows.forEach((row, index) => {
        const y = top + index * 34;
        const methodColor = COLORS[row.method_id];
        const x = (value) => left + ((value - minVal) / Math.max(1e-9, maxVal - minVal)) * (width - left - right);
        svg += `<text x="${left - 8}" y="${y + 5}" text-anchor="end" font-size="10" fill="#58656d">${safeText(row.dataset_id)} / ${row.budget_pct}% / ${METHOD_LABELS[row.method_id]}</text>`;
        svg += `<line x1="${x(row.p05)}" y1="${y}" x2="${x(row.p95)}" y2="${y}" stroke="${methodColor}" stroke-width="4" stroke-opacity="0.5" />`;
        svg += `<line x1="${x(row.mean_ci95_lower)}" y1="${y - 6}" x2="${x(row.mean_ci95_lower)}" y2="${y + 6}" stroke="${methodColor}" />`;
        svg += `<line x1="${x(row.mean_ci95_upper)}" y1="${y - 6}" x2="${x(row.mean_ci95_upper)}" y2="${y + 6}" stroke="${methodColor}" />`;
        svg += `<circle cx="${x(row.mean)}" cy="${y}" r="5" fill="#fff" stroke="${methodColor}" stroke-width="2" />`;
      });
      svg += `<text x="${left}" y="${height - 8}" text-anchor="start" font-size="11" fill="#58656d">min ${fmt(minVal, 4)}</text>`;
      svg += `<text x="${width - right}" y="${height - 8}" text-anchor="end" font-size="11" fill="#58656d">max ${fmt(maxVal, 4)}</text>`;
      svg += '</svg>';
      plot.innerHTML = svg + '<div class="muted small">Thin line P05-P95; vertical ticks Student-t CI95; dot replay mean. This is empirical replay sensitivity under paired perturbation, not population-level generalization. n=0 groups are excluded.</div>';
      document.getElementById('uncertaintyStory').innerHTML = storyBlockHtml(
        `Replay-uncertainty intervals for ${METRIC_META[metricId]?.label || metricId} using P05-P95 and CI95 by dataset/budget/method.`,
        `${directionText}; wider intervals mean more replay sensitivity under the same corpus and design.`,
        `${scopeVerdictSummary(metricId)} These intervals measure replay sensitivity, not population generalization.`
      );
    }
    function renderReplayFrequencyContext() {
      const unique = new Map();
      rowsForFilter().forEach((row) => {
        if (!row.replay_id) return;
        const key = `${row.dataset_id}:${row.repetition_index}`;
        if (!unique.has(key)) unique.set(key, row);
      });
      const rows = [...unique.values()];
      const plot = document.getElementById('replayFrequencyContext');
      if (!rows.length) {
        plot.innerHTML = '<p>No replay-frequency context available.</p>';
        document.getElementById('replayFrequencyStory').innerHTML = storyBlockHtml(
          'Unique-source fractions for each dataset/repetition replay plan.',
          'The diagnostic would show how strongly bootstrap frequency and duplication changed each replay.',
          'Replay-frequency metadata are unavailable for this filter; this diagnostic does not select a winning arm.'
        );
        return;
      }
      const width = 760; const height = 200; const left = 40; const right = 20; const top = 20; const bottom = 30;
      let svg = '<svg viewBox="0 0 760 200" aria-label="Replay frequency context" role="img">';
      rows.forEach((row, index) => {
        const value = Number(row.replay_frequency_summary && row.replay_frequency_summary.unique_source_fraction) || 0;
        const x = left + index * ((width - left - right) / Math.max(1, rows.length - 1));
        const y = top + (1 - value) * (height - top - bottom);
        svg += `<circle cx="${x}" cy="${y}" r="5" fill="${COLORS[row.method_id] || '#7a7f8b'}" />`;
      });
      svg += '</svg>';
      plot.innerHTML = svg + '<div class="muted small">Replay frequency context is deduped by dataset/repetition and not repeated per method or budget.</div>';
      document.getElementById('replayFrequencyStory').innerHTML = storyBlockHtml(
        'Replay-frequency diagnostics summarize source-frequency structure per dataset/replay.',
        'This panel is diagnostic context and not an arm-performance objective or winner selector.',
        'Frequency context helps explain replay sensitivity patterns but does not pick cosine or Euclidean as a winner.'
      );
    }
    function renderPcaScree() {
      const plot = document.getElementById('pcaScreeChart');
      const datasets = activeDataset() === 'all' ? REPORT.datasets : REPORT.datasets.filter((dataset) => String(dataset.dataset_id) === activeDataset());
      if (!datasets.length) {
        plot.innerHTML = '<p>No PCA context available.</p>';
        document.getElementById('pcaStory').innerHTML = storyBlockHtml(
          'Per-corpus PCA-8 explained variance and cumulative retention.',
          'Higher cumulative retention would imply less representation loss from compression.',
          'No PCA profile is available for this filter; this diagnostic does not select a winning arm.'
        );
        return;
      }
      plot.innerHTML = datasets.map((dataset) => {
        const ratios = (dataset.pca?.explained_variance_ratio || []).slice(0, 8);
        if (!ratios.length) return `<div class="panel"><h3>${safeText(dataset.dataset_id)}</h3><p>No PCA variance data.</p></div>`;
        const width = 760; const height = 250; const left = 52; const right = 20; const top = 20; const bottom = 36;
        const yPct = (v) => top + (1 - v) * (height - top - bottom);
        const h = (value) => {
          const h = yPct(0) - yPct(value);
          return h;
        };
        const barWidth = (width - left - right) / Math.max(1, ratios.length);
        let cumulative = 0;
        let svg = '<svg viewBox="0 0 760 250" aria-label="PCA scree chart" role="img">';
        for (let t = 0; t <= 100; t += 20) {
          const yy = yPct(t / 100);
          svg += `<line x1="${left}" y1="${yy}" x2="${width - right}" y2="${yy}" stroke="#e8ece9"/><text x="${left - 6}" y="${yy + 4}" text-anchor="end" font-size="10" fill="#58656d">${t}%</text>`;
        }
        const cumulativePoints = [];
        ratios.forEach((rawValue, index) => {
          const value = Number(rawValue) || 0;
          cumulative += value;
          const x = left + barWidth * index + 8;
          const barHeightValue = h(value);
          const y = yPct(value);
          const barW = Math.max(14, barWidth - 18);
          svg += `<rect x="${x}" y="${y}" width="${barW}" height="${barHeightValue}" fill="#3e7db8" />`;
          svg += `<text x="${x + barW / 2}" y="${height - 10}" text-anchor="middle" font-size="10" fill="#58656d">PC${index + 1}</text>`;
          cumulativePoints.push([x + barW / 2, yPct(cumulative)]);
        });
        if (cumulativePoints.length > 1) {
          svg += `<polyline points="${cumulativePoints.map(([cx, cy]) => `${cx},${cy}`).join(' ')}" fill="none" stroke="#d79b2d" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />`;
          cumulativePoints.forEach(([cx, cy]) => svg += `<circle cx="${cx}" cy="${cy}" r="3.5" fill="#d79b2d" />`);
        }
        svg += '</svg>';
        svg += '<div class="legend"><span><i class="dot" style="background:#3e7db8"></i>blue bars = per-component explained variance</span><span><i class="dot" style="background:#d79b2d"></i>amber line/dots = cumulative explained variance</span></div>';
        const total = Number(dataset.pca?.explained_variance_total);
        const warning = Number.isFinite(total) && total < 0.5 ? '<span class="warn">Historical-style low retention caution: compression summary is weak.</span>' : '';
        return `<div class="panel"><h3>${safeText(dataset.dataset_id)}</h3>${svg}<div class="muted small">PCA-8 captures ${pct1(total)} of source-embedding variance. ${warning}</div></div>`;
      }).join('');
      document.getElementById('pcaStory').innerHTML = storyBlockHtml(
        'Per-corpus PCA-8 scree with blue bars for per-component variance and amber line for cumulative retained variance.',
        'Higher cumulative retention implies less compression loss; y-axis is fixed at 0-100% for comparability.',
        'This panel diagnoses representation quality and provenance context, not which distance arm wins performance objectives.'
      );
    }
    function renderProvenance() {
      const table = document.getElementById('provenanceGrid');
      const datasets = activeDataset() === 'all' ? REPORT.datasets : REPORT.datasets.filter((dataset) => String(dataset.dataset_id) === activeDataset());
      const rows = datasets.map((dataset) => {
        const profile = dataset.runtime_ledger || dataset.shared_preprocessing || {};
        const source = dataset.pca || {};
        const hashes = profile.cache_provenance?.source_hashes || {};
        return `<tr><td>${safeText(dataset.dataset_id)}</td><td>${safeText(profile.embedding_model_id || 'Foundry embedding model')}</td><td>${safeText(profile.embedding_deployment_id || 'Not supplied')}</td><td>${safeText(String(Boolean(profile.cache_hit)))}</td><td>${pct1(source.explained_variance_total || 0)}</td><td><span class="trace">${safeText(JSON.stringify(hashes))}</span></td></tr>`;
      }).join('');
      const indexes = [...new Set(rowsForFilter().map((row) => row.search_evidence?.index_name).filter(Boolean))];
      const scopes = [...new Set(rowsForFilter().map((row) => row.search_evidence?.source_index_scope).filter(Boolean))];
      table.innerHTML = `<table><thead><tr><th>Dataset</th><th>Embedding model</th><th>Deployment</th><th>cache_hit</th><th>PCA-8 variance</th><th>source_hashes</th></tr></thead><tbody>${rows}</tbody></table><p class="small muted"><strong>Azure AI Search indexes:</strong> ${indexes.map(safeText).join(', ') || 'n/a'}.</p><p class="small muted"><strong>Search scope:</strong> ${scopes.map(safeText).join(', ') || 'n/a'}. Search is used for PCA-8 validation probes; deterministic selection and IDW execution remain local.</p>`;
    }
    function renderConclusion() {
      const metricId = activeMetric();
      const meta = METRIC_META[metricId];
      const pairRows = filteredPairedRows(metricId);
      const wins = pairRows.reduce((sum, row) => sum + Number(row.wins || 0), 0);
      const ties = pairRows.reduce((sum, row) => sum + Number(row.ties || 0), 0);
      const losses = pairRows.reduce((sum, row) => sum + Number(row.losses || 0), 0);
      const total = wins + ties + losses;
      const allCorpora = activeDataset() === 'all';
      const conclusionGrid = document.getElementById('storyConclusion');
      const objectiveToMetric = {
        'Estimation error': ['aggregate_pass_rate_mae', 'selected_only_pass_rate_mae'],
        'Imputed classification': ['imputed_only_accuracy', 'imputed_only_f1'],
        'Concept coverage': ['coverage_concept_ratio'],
        'Judged-token cost': ['actual_token_count'],
        'Incremental selection+IDW runtime': []
      };
      const corpora = selectedDatasets();
      const objectiveRows = Object.entries(objectiveToMetric).flatMap(([objective, metricIds]) => corpora.map((datasetId) => {
        if (objective === 'Incremental selection+IDW runtime') {
          const summary = rawIncrementalRuntimeSummary(datasetId);
          return `<article class="conclusion-card"><h4>${safeText(objective)}</h4><div class="small muted">${safeText(datasetId)}</div><div class="verdict">${safeText(summary.verdict)}</div>${evidenceDetailsHtml('Observed raw means', summary.evidence)}<p class="small muted">Shared Search excluded; no paired CI winner claim.</p></article>`;
        }
        const verdicts = metricIds.map((metricId) => objectiveVerdict(metricId, datasetId));
        const merged = combineObjectiveVerdicts(verdicts);
        const pairedN = verdicts.reduce((sum, row) => sum + Number(row.n || 0), 0);
        return `<article class="conclusion-card"><h4>${safeText(objective)}</h4><div class="small muted">${safeText(datasetId)}</div><div class="verdict">${safeText(merged.verdict)}</div>${evidenceDetailsHtml('Paired evidence', `paired n=${pairedN}; ${merged.evidence}`)}<p class="small muted">Replay sensitivity on this corpus, not population generalization.</p></article>`;
      }));
      conclusionGrid.innerHTML = `<h3>Story conclusion by objective</h3><div class="conclusion-grid">${objectiveRows.join('')}</div>`;

      if (allCorpora) {
        const perCorpus = corpora.map((datasetId) => {
          const verdict = objectiveVerdict(metricId, datasetId);
          return `${datasetId}: ${verdict.verdict}`;
        }).join('; ');
        document.getElementById('conclusionText').textContent = `All-corpora view for ${meta?.label || metricId}: ${perCorpus}. No pooled winner is claimed; the evidence is objective-specific and should be read by corpus. Recommendation: keep the selected metric and direction rule, then choose per corpus and budget rather than pooling win counts across heterogeneous corpora.`;
        return;
      }
      const runs = rowsForFilter();
      const means = METHOD_IDS.map((methodId) => {
        const values = runs.filter((row) => row.method_id === methodId).map((row) => metricFromRun(row, metricId)).filter(Number.isFinite);
        return { methodId, value: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null };
      }).filter((item) => item.value !== null);
      means.sort((a, b) => meta?.direction === 'higher_better' ? b.value - a.value : a.value - b.value);
      const evidence = total ? `Across ${total} paired replay comparisons, cosine wins ${wins}, ties ${ties}, and Euclidean wins ${losses}.` : 'No aggregate paired rows match this filter.';
      const conservativeVerdict = objectiveVerdict(metricId, activeDataset());
      const leading = means.length
        ? (conservativeVerdict.verdict === VERDICT_WORDS.cosine || conservativeVerdict.verdict === VERDICT_WORDS.euclidean
          ? `${METHOD_LABELS[means[0].methodId]} leads for ${meta?.label || metricId} (${fmt(means[0].value)} vs ${fmt(means[1]?.value)}), with paired replay support n=${conservativeVerdict.n}.`
          : `${conservativeVerdict.verdict} for ${meta?.label || metricId}; means are cosine=${fmt(means.find((m) => m.methodId === 'pca8_idw_binary_cosine')?.value)} vs euclidean=${fmt(means.find((m) => m.methodId === 'pca8_idw_binary_euclidean')?.value)} with paired replay n=${conservativeVerdict.n}.`)
        : 'No run-level values match this filter.';
      document.getElementById('conclusionText').textContent = `${leading} ${evidence} Recommendation: choose cosine when judged-token efficiency is the priority; prefer or consider Euclidean when imputed classification is the priority; for MAE and concept coverage, decide per corpus/budget and do not generalize beyond this paired replay evidence.`;
    }
    function renderDetailTable() {
      const rows = rowsForFilter();
      const limit = 50;
      const visible = rows.slice(0, limit);
      document.getElementById('rowCount').textContent = `Showing ${visible.length} of ${rows.length} matching rows.`;
      document.getElementById('detailTable').innerHTML = visible.map((row) => {
        const cap = Number(row.cap);
        const sample = row.sample_size ?? row.cap;
        const lowN = lowNFromCap(cap);
        return `<tr><td>${safeText(row.dataset_id)}</td><td>${row.budget_pct ?? 'n/a'}%</td><td>${row.cap ?? 'n/a'}${lowN ? ' <span class="low-n">low-N</span>' : ''}</td><td>${sample ?? 'n/a'}</td><td>${row.repetition_index ?? 'n/a'}</td><td>${METHOD_LABELS[row.method_id] || safeText(row.method_id)}</td><td>${fmt(getNumber(row, 'aggregate_pass_rate_mae'))}</td><td>${fmt(getNumber(row, 'selected_only_pass_rate_mae'))}</td><td>${(Number(getNumber(row, 'coverage.concept.coverage_ratio')) * 100 || 0).toFixed(1)}%</td><td>${(Number(getNumber(row, 'embedding_metrics.imputed_only.f1')) * 100 || 0).toFixed(1)}%</td><td>${fmt(getNumber(row, 'actual_token_count'), 0)}</td><td>${fmt(incrementalLatency(row), 4)}</td><td>${fmt(sharedSearchLatency(row), 4)}</td></tr>`;
      }).join('');
      document.getElementById('showMore').style.display = rows.length > limit ? 'inline-block' : 'none';
    }
    function renderAll() {
      renderExecutiveAnswer();
      renderDecisionScorecard();
      renderDatasetProfiles();
      renderBudgetCapCaption();
      renderBudgetTrend();
      renderPairedDelta();
      renderWinLoss();
      renderClassificationMatrix();
      renderCoverageAndCosts();
      renderReplayUncertainty();
      renderReplayFrequencyContext();
      renderPcaScree();
      renderProvenance();
      renderConclusion();
      renderDetailTable();
    }
    initFilters();
    renderAll();
    document.getElementById('showMore').addEventListener('click', () => {
      const rows = rowsForFilter();
      const current = document.getElementById('detailTable').children.length;
      document.getElementById('detailTable').innerHTML += rows.slice(current, current + 50).map((row) => {
        const cap = Number(row.cap);
        const sample = row.sample_size ?? row.cap;
        const lowN = lowNFromCap(cap);
        return `<tr><td>${safeText(row.dataset_id)}</td><td>${row.budget_pct ?? 'n/a'}%</td><td>${row.cap ?? 'n/a'}${lowN ? ' <span class="low-n">low-N</span>' : ''}</td><td>${sample ?? 'n/a'}</td><td>${row.repetition_index ?? 'n/a'}</td><td>${METHOD_LABELS[row.method_id] || safeText(row.method_id)}</td><td>${fmt(getNumber(row, 'aggregate_pass_rate_mae'))}</td><td>${fmt(getNumber(row, 'selected_only_pass_rate_mae'))}</td><td>${(Number(getNumber(row, 'coverage.concept.coverage_ratio')) * 100 || 0).toFixed(1)}%</td><td>${(Number(getNumber(row, 'embedding_metrics.imputed_only.f1')) * 100 || 0).toFixed(1)}%</td><td>${fmt(getNumber(row, 'actual_token_count'), 0)}</td><td>${fmt(incrementalLatency(row), 4)}</td><td>${fmt(sharedSearchLatency(row), 4)}</td></tr>`;
      }).join('');
    });
  </script>
</body>
</html>
"""
    html = template
    html = html.replace("__TITLE__", TITLE)
    html = html.replace("__DATASET_COUNT__", str(summary["dataset_count"]))
    html = html.replace("__REPETITION_COUNT__", str(summary["repetition_count"]))
    html = html.replace("__BUDGET_COUNT__", str(summary["budget_count"]))
    html = html.replace("__RUN_COUNT__", str(summary["run_count"]))
    html = html.replace("__REPORT_JSON__", _safe_json(data))
    html = html.replace("__METHOD_IDS__", _safe_json(list(METHOD_IDS)))
    html = html.replace("__METHOD_LABELS__", _safe_json({method_id: _method_label(method_id) for method_id in METHOD_IDS}))
    return html


def _default_output_path(aggregate_path: Path) -> Path:
    return aggregate_path.with_name("pca8-distance-report.html")


def _update_manifest(aggregate_path: Path, out_path: Path) -> None:
    manifest_path = aggregate_path.with_name("manifest.json")
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        return
    files = manifest.setdefault("files", {})
    hashes = manifest.setdefault("hashes", {})
    files["pca8_distance_report"] = str(out_path)
    hashes["pca8_distance_report"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def validate_aggregate(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("aggregate payload must be a dictionary")
    required = ("runs", "aggregate_metrics", "paired_differences", "summary")
    for key in required:
        if key not in payload:
            raise ValueError(f"aggregate payload must include '{key}'")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("aggregate payload must include a 'runs' list")
    present_methods = {
      str(row.get("method_id") or "")
      for row in runs
      if isinstance(row, Mapping)
    }
    if not set(METHOD_IDS).issubset(present_methods):
        raise ValueError("aggregate payload must include both PCA-8 distance methods")
    aggregate = payload.get("aggregate_metrics")
    if not isinstance(aggregate, Mapping) or not isinstance((aggregate.get("rows") or []), list):
        raise ValueError("aggregate payload must include aggregate_metrics.rows")
    paired = payload.get("paired_differences")
    if not isinstance(paired, Mapping):
        raise ValueError("aggregate payload must include paired_differences")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Render the PCA-8 cosine vs Euclidean distance report")
    parser.add_argument("--input", default="outputs_sampling_v7/runs/v7-live-repeated-20260903/aggregate.json", help="Path to aggregate.json")
    parser.add_argument("--output", default=None, help="Output path for the HTML report")
    args = parser.parse_args()
    aggregate_path = Path(args.input)
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    validate_aggregate(payload)
    html = build_distance_report_html(payload)
    out_path = Path(args.output) if args.output else _default_output_path(aggregate_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    _update_manifest(aggregate_path, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
