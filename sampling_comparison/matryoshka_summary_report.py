from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from html import escape
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Mapping
from urllib.parse import quote

from sampling_comparison.matryoshka_experiment import sha256_file, write_json


LABELS = {
    "historical_300": "Historical",
    "dense_2500": "Dense",
    "cosmos_otel": "Cosmos",
    "tau2_bench": "Tau2 bench",
}
COLORS = ("#155e75", "#7c3aed", "#c2410c", "#166534")
REPORT_VERSION = "matryoshka-summary-v1"


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("report metrics must be finite numbers")
    return float(value)


def _relative(value: float, baseline: float) -> float | None:
    return 100 * (value / baseline - 1) if baseline else None


def _pct(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}%" if signed else f"{value:.2f}%"


def summarize_evidence(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    if aggregate["version"] != "matryoshka-cutoff-v1":
        raise ValueError("use a Matryoshka cutoff aggregate, not another experiment version")
    protocol = aggregate["protocol"]
    dimensions = list(protocol["dimensions"])
    if dimensions[0] != 1536 or 8 not in dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError("report requires unique dimensions, native 1536 first, and the 8d arm")
    for name in ("seeds", "rates", "schedules"):
        if not protocol[name] or len(set(protocol[name])) != len(protocol[name]):
            raise ValueError(f"invalid or duplicated {name}")
    profiles = {row["dataset_id"]: row for row in aggregate["datasets"]}
    if len(profiles) != len(aggregate["datasets"]):
        raise ValueError("duplicate dataset profiles")
    completed = [key for key in LABELS if profiles.get(key, {}).get("status") == "completed"]
    if not completed:
        raise ValueError("no completed datasets: do not generate success-shaped conclusions")
    for dataset in completed:
        provenance = profiles[dataset]["provenance"]
        if provenance["embedding_model_id"] != "text-embedding-3-small" or provenance["embedding_dimensions"] != 1536:
            raise ValueError("input is not the specified native embedding model and dimensionality")
        if provenance["live_judge_calls"] != 0:
            raise ValueError("this report's no-new-judge narrative does not match the input")

    primary = [r for r in aggregate["rows"] if r["mode"] == "end_to_end"]
    expected_keys = {
        (dataset, dimension, seed, schedule, rate)
        for dataset in completed for dimension in dimensions for seed in protocol["seeds"]
        for schedule in protocol["schedules"] for rate in protocol["rates"]
    }
    keyed = {}
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in primary:
        key = tuple(row[name] for name in ("dataset_id", "dimension", "seed", "schedule", "rate"))
        if key in keyed:
            raise ValueError("duplicate end-to-end result cell")
        keyed[key] = row
        for metric in ("mae", "accuracy", "brier"):
            if not 0 <= _number(row[metric]) <= 1:
                raise ValueError(f"{metric} is outside its valid range")
        n = profiles[row["dataset_id"]]["n"]
        selected = max(1, math.floor(n * row["rate"]))
        if row["selected_count"] != selected or row["unjudged_count"] != n - selected:
            raise ValueError("selected/unjudged counts do not match the session-budget protocol")
        if sum(row["provenance_counts"].values()) != n - selected:
            raise ValueError("unjudged provenance counts do not match the metric denominator")
        groups[(row["dataset_id"], row["dimension"])].append(row)
    if set(keyed) != expected_keys:
        raise ValueError("incomplete or unexpected end-to-end result grid")

    results = []
    agent_metrics = {
        (r["dataset_id"], r["agent_id"], r["dimension"]): r["mae"]
        for r in aggregate.get("agent_summary", [])
        if r["mode"] == "end_to_end" and r["dimension"] in (1536, 8) and r["mae"] is not None
    }
    for dataset in completed:
        native = groups[(dataset, 1536)]
        short = groups[(dataset, 8)]
        baseline = mean(r["mae"] for r in native)
        endpoint = mean(r["mae"] for r in short)
        curve = {str(d): mean(r["mae"] for r in groups[(dataset, d)]) for d in dimensions}
        budget_rows = []
        for rate in protocol["rates"]:
            before = [r for r in native if r["rate"] == rate]
            after = [r for r in short if r["rate"] == rate]
            old_mae, new_mae = mean(r["mae"] for r in before), mean(r["mae"] for r in after)
            budget_rows.append({
                "rate": rate, "selected_count": before[0]["selected_count"],
                "unjudged_count": before[0]["unjudged_count"],
                "native_mae": old_mae, "eight_mae": new_mae,
                "change_pct": _relative(new_mae, old_mae),
                "accuracy_delta_pp": 100 * (mean(r["accuracy"] for r in after) - mean(r["accuracy"] for r in before)),
            })
        fallback_shares = {}
        for dimension, rows in ((1536, native), (8, short)):
            fallback_shares[str(dimension)] = sum(
                r["provenance_counts"].get("global_mean", 0) + r["provenance_counts"].get("prior", 0) for r in rows
            ) / sum(r["unjudged_count"] for r in rows)
        agent_differences = [
            value - agent_metrics[(key[0], key[1], 1536)]
            for key, value in agent_metrics.items()
            if key[0] == dataset and key[2] == 8 and (key[0], key[1], 1536) in agent_metrics
        ]
        results.append({
            "dataset_id": dataset, "label": LABELS[dataset], "profile": profiles[dataset],
            "native_mae": baseline, "eight_mae": endpoint, "change_pct": _relative(endpoint, baseline),
            "native_accuracy": mean(r["accuracy"] for r in native), "eight_accuracy": mean(r["accuracy"] for r in short),
            "brier_change": mean(r["brier"] for r in short) - mean(r["brier"] for r in native),
            "curve": curve, "budget_rows": budget_rows, "fallback_shares": fallback_shares,
            "agents_compared": len(agent_differences),
            "agents_with_higher_mae": sum(delta > 0 for delta in agent_differences) if agent_differences else None,
        })
    no_average_drop = all(r["eight_mae"] <= r["native_mae"] for r in results)
    return {
        "version": REPORT_VERSION, "run_id": aggregate["run_id"], "protocol": protocol,
        "datasets": results, "not_tested": [key for key in LABELS if key not in completed],
        "session_count": sum(profiles[key]["n"] for key in completed),
        "primary_cells": len(primary), "retained_cells": len(aggregate["rows"]),
        "seed_count": len(protocol["seeds"]), "no_average_mae_drop": no_average_drop,
        "graph_count": 3, "method_diagram_count": 1,
    }


def _text(x: float, y: float, value: Any, *, anchor: str = "start", fill: str = "#334155", size: int = 16) -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" fill="{fill}" font-size="{size}">{escape(str(value))}</text>'


def _svg(chart_id: str, title: str, description: str, height: int, content: str) -> str:
    return (
        f'<svg viewBox="0 0 900 {height}" role="img" aria-labelledby="{chart_id}-title {chart_id}-desc">'
        f'<title id="{chart_id}-title">{escape(title)}</title>'
        f'<desc id="{chart_id}-desc">{escape(description)}</desc>'
        f'<rect width="900" height="{height}" fill="white"/>{content}</svg>'
    )


def endpoint_chart(evidence: Mapping[str, Any]) -> str:
    rows = evidence["datasets"]
    maximum = max(0.1, math.ceil(max(max(r["native_mae"], r["eight_mae"]) for r in rows) * 10) / 10)
    left, plot_width, top = 210, 580, 52
    bottom = top + len(rows) * 104
    parts = [
        '<rect x="210" y="12" width="18" height="18" rx="3" fill="#718096"/>',
        _text(237, 27, "Native: 1,536 dimensions"),
        '<rect x="510" y="12" width="18" height="18" rx="3" fill="#0f766e"/>',
        _text(537, 27, "First 8, re-normalized"),
    ]
    for tick in range(6):
        value = maximum * tick / 5
        x = left + plot_width * tick / 5
        parts.append(f'<line x1="{x}" y1="42" x2="{x}" y2="{bottom}" stroke="#e2e8f0"/>')
        parts.append(_text(x, bottom + 25, f"{value:.2f}", anchor="middle"))
    for i, row in enumerate(rows):
        y = top + i * 104
        parts.append(_text(18, y + 31, row["label"], fill="#0f172a", size=19))
        parts.append(_text(18, y + 55, f'{row["profile"]["n"]:,} sessions'))
        for offset, metric, color, name in ((8, "native_mae", "#718096", "Native"), (39, "eight_mae", "#0f766e", "8 dimensions")):
            width = plot_width * row[metric] / maximum
            parts.append(
                f'<rect x="{left}" y="{y + offset}" width="{width:.3f}" height="22" rx="3" fill="{color}">'
                f'<title>{escape(row["label"])} / {name}: MAE {row[metric]:.6f}</title></rect>'
            )
            parts.append(_text(left + width + 9, y + offset + 18, f'{row[metric]:.4f}'))
    parts.append(_text(500, bottom + 62, "Average prediction error (MAE): shorter bars are better", anchor="middle"))
    return _svg("endpoints", "Average prediction error: 1,536 versus 8 dimensions",
                "Same zero-based MAE axis for every dataset. Numbers are means over all end-to-end replay settings.",
                bottom + 82, "".join(parts))


def trend_chart(evidence: Mapping[str, Any]) -> str:
    dimensions = evidence["protocol"]["dimensions"]
    series = [
        [_relative(row["curve"][str(d)], row["native_mae"]) for d in dimensions]
        for row in evidence["datasets"]
    ]
    limit = max(1, math.ceil(max((abs(v) for values in series for v in values if v is not None), default=0)))
    left, top, width, height = 90, 72, 755, 280
    x = lambda i: left + i * width / max(1, len(dimensions) - 1)
    y = lambda value: top + (limit - value) * height / (2 * limit)
    parts = []
    for index, row in enumerate(evidence["datasets"]):
        lx = 90 + index * 190
        parts.append(f'<line x1="{lx}" y1="23" x2="{lx + 25}" y2="23" stroke="{COLORS[index]}" stroke-width="4" stroke-dasharray="{index * 3} {index * 2}"/>')
        parts.append(_text(lx + 33, 29, row["label"]))
    parts.append(_text(90, 55, "Relative change in MAE versus native: below zero is better"))
    for tick in range(5):
        value = -limit + tick * limit / 2
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy}" x2="{left + width}" y2="{yy}" stroke="#e2e8f0"/>')
        parts.append(_text(left - 12, yy + 6, f"{value:+.1f}%", anchor="end"))
    parts.append(f'<line x1="{left}" y1="{y(0)}" x2="{left + width}" y2="{y(0)}" stroke="#475569" stroke-dasharray="5 5"/>')
    for index, values in enumerate(series):
        segment = []
        for i, value in enumerate(values):
            if value is None:
                if segment:
                    parts.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{COLORS[index]}" stroke-width="3"/>')
                    segment = []
                continue
            segment.append(f"{x(i):.3f},{y(value):.3f}")
            parts.append(
                f'<circle cx="{x(i):.3f}" cy="{y(value):.3f}" r="5" fill="{COLORS[index]}">'
                f'<title>{escape(evidence["datasets"][index]["label"])} / {dimensions[i]} dimensions: {_pct(value, signed=True)} MAE change</title></circle>'
            )
        if segment:
            parts.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{COLORS[index]}" stroke-width="3" stroke-dasharray="{index * 3} {index * 2}"/>')
    for i, dimension in enumerate(dimensions):
        parts.append(_text(x(i), top + height + 30, f"{dimension:,}", anchor="middle"))
    parts.append(_text(460, 425, "Dimensions retained: ordered tested sizes, decreasing left to right", anchor="middle"))
    return _svg("trend", "What happens at the intermediate dimensions",
                "Relative MAE change uses each dataset's own native baseline. Tested sizes are equally spaced categories, not a linear dimension axis.",
                448, "".join(parts))


def budget_chart(evidence: Mapping[str, Any]) -> str:
    rates = evidence["protocol"]["rates"]
    cell_width = 660 / len(rates)
    parts = [_text(210, 26, "Share of sessions sampled")]
    for i, rate in enumerate(rates):
        parts.append(_text(210 + (i + 0.5) * cell_width, 53, f"{100 * rate:g}%", anchor="middle"))
    for i, row in enumerate(evidence["datasets"]):
        yy = 75 + i * 70
        parts.append(_text(18, yy + 32, row["label"], size=18))
        for j, budget in enumerate(row["budget_rows"]):
            value = budget["change_pct"]
            color, ink = ("#f1f5f9", "#334155") if value is None or abs(value) < 1e-12 else (
                ("#dcfce7", "#14532d") if value < 0 else ("#ffedd5", "#9a3412")
            )
            xx = 210 + j * cell_width + 4
            parts.append(
                f'<rect x="{xx}" y="{yy}" width="{cell_width - 8}" height="52" rx="6" fill="{color}"/>'
                + _text(xx + (cell_width - 8) / 2, yy + 33, _pct(value, signed=True), anchor="middle", fill=ink, size=20)
            )
    height = 75 + len(evidence["datasets"]) * 70 + 44
    parts.append(_text(210, height - 15, "Green / negative: lower MAE. Orange / positive: higher MAE."))
    return _svg("budgets", "Where sampling budgets change the result",
                "Relative MAE change at 8 dimensions versus native, computed separately for each dataset and sampling budget.",
                height, "".join(parts))


def _figure(chart_id: str, title: str, svg: str, caption: str) -> str:
    return (
        f'<figure data-graph="{chart_id}"><h3 id="{chart_id}-heading">{escape(title)}</h3>'
        f'<div class="chart-scroll" role="region" aria-labelledby="{chart_id}-heading" tabindex="0">{svg}</div>'
        f'<figcaption>{caption}</figcaption></figure>'
    )


CSS = """
*{box-sizing:border-box}body{margin:0;color:#172b3a;background:#f3f6fa;font:16px/1.58 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:auto;padding:32px 28px 64px}header{background:#103d46;color:#fff;padding:38px;border-radius:18px}
h1{font-size:clamp(2rem,4vw,3.1rem);line-height:1.14;margin:12px 0 18px;max-width:850px}
h2{font-size:1.6rem;line-height:1.3;margin:0 0 16px}h3{font-size:1.1rem;line-height:1.35;margin:0 0 12px}
p{margin:10px 0}.eyebrow{font-size:.82rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#b9e5df}
header p{max-width:870px;color:#e4f1f0}section{margin-top:30px;background:#fff;padding:30px;border:1px solid #dce5ec;border-radius:14px}
.kpis,.datasets{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:20px}
.kpi,.dataset{min-width:0;background:#fff;padding:20px;border:1px solid #dce5ec;border-radius:12px}
.kpi strong{display:block;font-size:1.75rem;line-height:1.3;color:#0f766e}.muted,figcaption{color:#526477}
.finding{font-size:1.1rem}.note{border-left:4px solid #0f766e;background:#edf7f4;padding:14px 18px;margin-top:18px}
.caution{border-left-color:#b45309;background:#fffbeb}.pipeline{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;list-style:none;padding:0;counter-reset:step}
.pipeline li{min-width:0;padding:16px;background:#eef4f8;border-radius:10px}.pipeline li:before{counter-increment:step;content:counter(step);display:block;font-weight:800;color:#0f766e}
code,pre{font-family:ui-monospace,Consolas,monospace;font-size:.88em;overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#eef4f8;padding:18px;border-radius:10px}
.table-wrap{max-width:100%;overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.94rem}th,td{text-align:left;padding:12px;border-bottom:1px solid #dde6ed;vertical-align:top}
th{background:#f3f7fa;font-weight:650}td.number{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
figure{min-width:0;margin:24px 0 0;padding:18px;border:1px solid #dce5ec;border-radius:12px}
.chart-scroll{width:100%;min-width:0;overflow-x:auto;overflow-y:hidden}svg{display:block;width:100%;min-width:720px;max-width:1000px;height:auto;font-family:system-ui,sans-serif}
figcaption{font-size:.94rem;margin-top:10px}details{margin-top:18px;border-top:1px solid #dce5ec;padding-top:16px}
summary{cursor:pointer;font-weight:650;color:#155e75}summary:focus-visible,.chart-scroll:focus-visible,a:focus-visible{outline:3px solid #b45309;outline-offset:4px}
a{color:#155e75;text-underline-offset:3px}.badge{display:inline-block;font-size:.8rem;padding:4px 9px;border-radius:20px;background:#e1f2ed;color:#14532d;font-weight:650}
.conclusion{background:#edf7f4;border-color:#bfddd2}footer{padding:20px 4px;color:#526477;font-size:.9rem;overflow-wrap:anywhere}
@media(max-width:760px){main{padding:16px 12px 40px}header,section{padding:22px 18px}.kpis,.datasets,.pipeline{grid-template-columns:minmax(0,1fr)}figure{padding:12px}th,td{padding:10px}h2{font-size:1.35rem}}
"""


def build_report(
    aggregate: Mapping[str, Any], *, aggregate_path: str, aggregate_hash: str,
    detailed_url: str, generation_command: str = "See the report builder CLI.",
) -> tuple[str, dict[str, Any]]:
    evidence = summarize_evidence(aggregate)
    rows, protocol = evidence["datasets"], evidence["protocol"]
    no_drop = evidence["no_average_mae_drop"]
    conclusion = (
        "Based on our results, we did not notice any meaningful performance drop-off when lowering "
        "the embedding dimensions from 1,536 to 8, when performance is assessed by average MAE "
        f"across the {len(rows)} datasets tested."
        if no_drop else
        "The results do not support a blanket no-drop-off conclusion: at least one tested dataset "
        "has higher average MAE at 8 dimensions than at 1,536."
    )
    cards, table_rows, audit_rows = [], [], []
    for row in rows:
        profile, dataset = row["profile"], row["dataset_id"]
        provenance = profile["provenance"]
        if dataset == "historical_300":
            origin = "Synthetic Agent365 sessions, spread thinly across many agents."
        elif dataset == "dense_2500":
            origin = "Synthetic Agent365 sessions with a much denser history per agent."
        elif dataset == "cosmos_otel":
            sources = provenance["representation_sources"]
            origin = (f'External label/span export: {sources["linked_raw_spans"]} linked-span representations '
                      f'and {sources["synthetic_label_document"]} task-description fallbacks. Not a verified production sample.')
        else:
            origin = "Recorded benchmark trajectories; see the supplied label provenance below."
        cards.append(
            f'<article class="dataset"><h3>{escape(row["label"])}</h3><span class="badge">{profile["n"]:,} sessions / {profile["agents"]} agents</span>'
            f'<p><code>{escape(dataset)}</code></p>'
            f'<p>{escape(origin)}</p><p class="muted">{escape(provenance["label_source"])}</p>'
            f'<p class="muted">Reference success rate: {100 * profile["pass_rate"]:.1f}%.</p>'
            f'<p class="muted">Input packets truncated to the token limit: {provenance.get("truncated_sessions", "unavailable")}.</p></article>'
        )
        table_rows.append(
            f'<tr><th scope="row">{escape(row["label"])}</th><td class="number">{row["native_mae"]:.6f}</td>'
            f'<td class="number">{row["eight_mae"]:.6f}</td><td class="number">{_pct(row["change_pct"], signed=True)}</td></tr>'
        )
        caps = ", ".join(str(b["selected_count"]) for b in row["budget_rows"])
        remaining = ", ".join(str(b["unjudged_count"]) for b in row["budget_rows"])
        agent_gaps = (
            f'{row["agents_with_higher_mae"]} / {row["agents_compared"]}'
            if row["agents_with_higher_mae"] is not None else "Unavailable"
        )
        audit_rows.append(
            f'<tr><th scope="row">{escape(row["label"])}</th><td>{caps}</td><td>{remaining}</td>'
            f'<td>{100 * row["fallback_shares"]["1536"]:.1f}% / {100 * row["fallback_shares"]["8"]:.1f}%</td><td>{agent_gaps}</td></tr>'
        )
    budget_regressions = [
        (row, budget) for row in rows for budget in row["budget_rows"]
        if budget["eight_mae"] > budget["native_mae"]
    ]
    if budget_regressions:
        row, budget = max(budget_regressions, key=lambda pair: pair[1]["eight_mae"] - pair[1]["native_mae"])
        caution = (
            f'For example, at a {100 * budget["rate"]:g}% sampling budget in {row["label"]}, MAE rose '
            f'from <strong>{budget["native_mae"]:.4f} to {budget["eight_mae"]:.4f}</strong> at 8 dimensions '
            f'({_pct(budget["change_pct"], signed=True)}). The dataset-wide average therefore does not guarantee the same result at every budget.'
        )
    else:
        caution = "No budget-level mean MAE regression was observed in these results; that is still not a guarantee for every agent or future dataset."
    worse_brier = [r["label"] for r in rows if r["brier_change"] > 0]
    other_metrics = (
        f'The squared-error metric (Brier score) also worsened slightly overall for {escape(" and ".join(worse_brier))}. '
        if worse_brier else ""
    )
    untested = (
        '<p class="note caution"><strong>Not included in the results:</strong> '
        + escape(", ".join(LABELS[key] for key in evidence["not_tested"]))
        + '. The retained Tau2 run has no approved expected-label mapping; saved benchmark rewards were not silently substituted.</p>'
        if evidence["not_tested"] else ""
    )
    budgets = ", ".join(f"{100 * rate:g}%" for rate in protocol["rates"])
    dimensions = " &rarr; ".join(f"{d:,}" for d in protocol["dimensions"])
    better_count = sum(r["eight_mae"] < r["native_mae"] for r in rows)
    strict_detail = (
        f'Average MAE was lower in all {len(rows)} datasets.'
        if better_count == len(rows) else f'Average MAE was lower in {better_count} of {len(rows)} datasets.'
    )
    interpretation = (
        "These results make 8 dimensions a promising candidate for this sampling-and-IDW pipeline."
        if no_drop else "These results do not support treating 8 dimensions as interchangeable with the native representation."
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>From 1,536 to 8 dimensions: the Matryoshka experiment</title><style>{CSS}</style></head><body><main>
<header><div class="eyebrow">Sampling experiment / plain-language report</div>
<h1>From 1,536 embedding dimensions to just 8</h1>
<p class="finding">{escape(conclusion)}</p>
<p>{escape(strict_detail)} This is a result about the average task-completion prediction error in these tests, not a claim that every setting or metric is unchanged.</p></header>
<div class="kpis"><div class="kpi"><strong>{len(rows)} datasets</strong>{evidence["session_count"]:,} source sessions included</div>
<div class="kpi"><strong>{evidence["seed_count"]} randomized runs</strong>per arrival pattern and sampling setting</div>
<div class="kpi"><strong>192&times; fewer values</strong>8 coordinates instead of 1,536; not a measured runtime or API-cost saving</div></div>

<section id="experiment"><h2>What did we test?</h2>
<p>Can we make full-session embeddings much smaller without making the sampling-and-prediction result worse?
We changed <strong>only the vector representation</strong> in the comparison: the embedding source, reference labels, sampling rules, scoring rules and replay settings stayed matched.</p>
<ol class="pipeline" aria-label="Experiment pipeline"><li><strong>Represent a session</strong><br>Build bounded evidence from its conversation and tool activity.</li>
<li><strong>Embed and shorten</strong><br>Generate 1,536 values, keep the first <em>d</em>, then re-normalize.</li>
<li><strong>Sample and estimate</strong><br>Select sessions; use their known reference labels to estimate the rest with IDW.</li>
<li><strong>Measure error</strong><br>Compare estimates with reference labels on unselected sessions.</li></ol>
<h3>The native model and the exact shortening method</h3>
<p>The native embedding model was <strong>OpenAI <code>text-embedding-3-small</code></strong>, used through Azure OpenAI / Foundry.
It converts the session evidence into a list of <strong>1,536 numbers</strong>: the vector's dimensions.
OpenAI documents native shortening support linked to <em>Matryoshka Representation Learning</em>.
The idea is that useful shorter representations are nested inside the larger vector; whether a particular size is sufficient still depends on the task.</p>
<p>At 8 dimensions, we kept coordinates <strong>0 through 7</strong>, in their original order. We then divided the shortened vector by its Euclidean (L2) length:</p>
<pre><code>short = full_embedding[:d]
short = short / np.linalg.norm(short)</code></pre>
<p><strong>No PCA, SVD, Gaussian random projection, coordinate sorting or retraining was used.</strong>
All shortened vectors came from the same saved full-dimensional embeddings. Tested sizes: {dimensions}.</p>
<h3>What does &ldquo;performance&rdquo; mean here?</h3>
<p>The main measure is <strong>mean absolute error (MAE)</strong> on sessions that were <strong>not selected</strong>.
For each one, we compare its predicted task-completion probability with its stored binary label, then average the absolute differences. Lower is better; zero means perfect predictions.</p>
<p>For example, predicting 0.8 when the reference label is 1 produces an error of 0.2.
This differs from binary accuracy, which first thresholds the prediction at 0.5. Known selected-session labels are excluded from the headline MAE, so they cannot make it look artificially better.</p></section>

<section id="datasets"><h2>The datasets and the comparison</h2><div class="datasets">{"".join(cards)}</div>{untested}
<p>We used existing reference labels for task completion, <strong>not a new LLM judge</strong>.
For Cosmos, the label mapping was good = 1 and bad / partial = 0; these are task-design expectations, not independently verified actual completion.</p>
<p>Each dataset used <strong>{evidence["seed_count"]} shuffled orders</strong> under {len(protocol["schedules"])} arrival patterns
(evenly spaced, random timing, bursty, front-loaded and agent-blocked), at sampling budgets of <strong>{budgets}</strong>.
The same seeded order was used across dimensions for a fair comparison. Each replay contains every source session once.</p>
<p class="muted">These reruns reuse the same sessions and labels. They test sensitivity to ordering, not new populations or independent judge reliability.
The charts below summarize {evidence["primary_cells"]:,} end-to-end result cells, giving equal weight to seeds, patterns and budgets within each dataset.
Selection is rerun at each dimension, so the unselected target set can change. This is not an independent deployment holdout.
The larger bundle also contains a separate fixed-membership diagnostic; it is not mixed into these headline results.</p></section>

<section id="results"><h2>Results at a glance</h2><p>{escape(strict_detail)} Negative percentage changes below mean less error at 8 dimensions.</p>
<div class="table-wrap" tabindex="0" role="region" aria-label="Native and eight-dimensional MAE results"><table>
<thead><tr><th>Dataset</th><th>Native MAE<br>1,536 dimensions</th><th>MAE at 8<br>dimensions</th><th>Relative change<br>in MAE</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>
{_figure("endpoints", "1. Compare the starting and ending dimensions", endpoint_chart(evidence), "What it shows: native and 8-dimensional average error, on the same axis. Read it: shorter is better. Takeaway: the 8-dimensional averages show no increase in this run." if no_drop else "What it shows: native and 8-dimensional average error. Shorter is better; inspect each dataset rather than assuming no loss.")}
{_figure("trend", "2. The full dimensionality sweep", trend_chart(evidence), "What it shows: each dataset's MAE change relative to its own 1,536-dimensional baseline. Read it: below zero is better. Takeaway: intermediate dimensions are not consistently better as dimensions increase; the curves need not be monotonic.")}
</section>

<section id="variation"><h2>What the averages do not tell us</h2>
<p>{caution}</p><p>{other_metrics}Individual agents can also regress.
The conclusion is about <strong>average MAE</strong>, not unchanged performance across every budget, agent or metric.</p>
<details id="budget-detail"><summary>Explore the sampling-budget trade-off</summary>
{_figure("budgets", "3. Budget-by-budget: 8 dimensions versus native", budget_chart(evidence), f"Each cell compares mean MAE over {evidence['seed_count']} seeds and {len(protocol['schedules'])} arrival patterns at one budget. Percentages are relative error changes, not accuracy percentage points. Positive cells are genuine regressions, not missing data.")}
</details></section>

<section class="conclusion" id="conclusion"><h2>Conclusion</h2><p class="finding"><strong>{escape(conclusion)}</strong></p>
<p>The tested change is simple: <strong>take the first 8 values of the native <code>text-embedding-3-small</code> vector and L2-normalize them</strong>.
{interpretation}</p>
<p>That does not mean all information is preserved or that every operating point is lossless. Business-use-case classification, future production data,
new judge scores and the proposed weekly sampling policy were not evaluated here. No formal acceptable-loss threshold was pre-specified for MAE.</p></section>

<section id="technical"><h2>Details, if you want to go deeper</h2><details id="method-detail"><summary>Sampling, IDW, budgets and fallbacks</summary>
<p>Selection uses the same ARM2 novelty/rarity sampler in every arm, with cosine threshold 0.55, cluster TTL 90 and a replay horizon of 10,000.
Floors are disabled and throughput is high; final membership is ranked over the full unlabelled schedule and capped by session count.
It is label-blind, but is not a fully online admission test.</p>
<p>IDW means &ldquo;inverse distance weighting&rdquo;: nearer labelled sessions receive more weight.
For a target, the estimator uses up to 8 earlier selected sessions from the same agent, angular distance arccos(cosine) / pi,
power 2 and epsilon 1e-6. Exact matches (1 &minus; cosine &le; 1e-8) are averaged.
With no earlier same-agent donor, it uses the earlier global mean; if none exists, it uses prior 0.5.</p>
<p><strong>Illustrative IDW example, not a run result:</strong> donor distances 0.1 and 0.2 with labels 1 and 0 yield weights 100 and 25
when epsilon is ignored for simplicity. Normalized weights are 0.8 and 0.2, giving an estimated probability of 0.8. It remains an estimate, not a direct judgment.</p>
<p>Canonical session evidence prioritizes goals, outcomes and tool results, and is bounded to 8,191 <code>cl100k_base</code> tokens.
&ldquo;Full-session&rdquo; therefore does not mean unlimited verbatim text. There is no fitted reduction transform or train/test reducer fit.</p>
<p>Budget order in the table is {budgets}; caps are max(1, floor(N &times; budget)).
Selected sessions receive their stored labels. The unjudged denominator includes IDW estimates and explicitly counted fallback estimates.</p>
<div class="table-wrap" tabindex="0" role="region" aria-label="Absolute budgets and fallback counts"><table><thead><tr><th>Dataset</th><th>Selected sessions</th><th>Unjudged sessions</th><th>Fallback share of unjudged<br>native / 8d</th><th>Agents with higher<br>mean MAE at 8d</th></tr></thead><tbody>{"".join(audit_rows)}</tbody></table></div>
<p>Fallback shares pool target counts across cells; headline MAE instead gives each replay cell equal weight.
Agent counts flag any increase in per-agent mean MAE, not statistical significance. More per-agent results and cohort-relative concept-coverage proxies are available in the detailed report;
coverage is not a guarantee of representing every real activity.</p>
<p>Float32 payload size is 4 &times; dimensions bytes per vector: 6,144 bytes at 1,536 dimensions versus 32 bytes at 8.
That is a modeled vector-payload reduction only. Embedding API billing is based on input tokens; no 192&times; API-cost or wall-clock speedup was measured.</p>
</details><details id="source-detail"><summary>Sources and reproducibility</summary>
<p>Run: <code>{escape(evidence["run_id"])}</code>. This is a read-only derivative of retained artifacts; generating it makes no embedding or judge calls.</p>
<p>Aggregate: <code>{escape(aggregate_path)}</code><br>SHA256: <code>{escape(aggregate_hash)}</code></p>
<p>Experiment revision: <code>{escape(str(aggregate.get("source_revision", "unavailable")))}</code>.
The source manifest also pins the exact implementation hashes, including code uncommitted at execution.
Run environment: <code>{escape(str(aggregate.get("environment", {}).get("python", "unavailable")))}</code>.</p>
<p>Build this read-only derivative from the repository root:</p><pre><code>{escape(generation_command)}</code></pre>
<p><a href="{escape(detailed_url, quote=True)}">Open the full experiment report</a> for all metrics, fixed-membership results, per-agent gaps and source provenance.</p>
<p><a href="https://developers.openai.com/api/docs/guides/embeddings">OpenAI embedding model and shortening documentation</a>;
<a href="https://arxiv.org/abs/2205.13147">Matryoshka Representation Learning paper</a>.
OpenAI's often-cited 256-dimensional benchmark comparison concerns <code>text-embedding-3-large</code> versus <code>text-embedding-ada-002</code>,
not a guarantee about 8-dimensional <code>text-embedding-3-small</code>.</p>
<p>All figures and text are embedded in this HTML. <code>summary.json</code> contains the plotted numbers;
the adjacent manifest records source and output hashes. Browser and test evidence is retained separately.</p>
</details></section><footer>{escape(evidence["run_id"])} / {len(rows)} datasets measured / 3 graphs / average-MAE conclusion only</footer>
</main></body></html>"""
    return html, evidence


def write_report(aggregate_path: Path, output: Path, *, overwrite: bool = False) -> Path:
    source = aggregate_path.resolve()
    output = output.resolve()
    if output.suffix.lower() != ".html" or output.parent == source.parent:
        raise ValueError("write a .html derivative in a separate directory, not inside the source run")
    if not overwrite and any(p.exists() for p in (output, output.parent / "summary.json", output.parent / "manifest.json")):
        raise FileExistsError("derivative bundle exists; use --overwrite only for this derivative")
    source_manifest_path = source.parent / "manifest.json"
    source_hash = sha256_file(source)
    source_manifest_hash = sha256_file(source_manifest_path)
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if manifest["files"]["aggregate"]["sha256"] != source_hash:
        raise ValueError("aggregate bytes do not match the retained run manifest")
    aggregate = json.loads(source.read_text(encoding="utf-8"))
    detailed_url = quote(os.path.relpath(source.parent / "report.html", output.parent).replace("\\", "/"), safe="/")
    command_args = [
        sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "build_matryoshka_summary_report.py"),
        "--input", str(source), "--output", str(output),
    ] + (["--overwrite"] if overwrite else [])
    command = "& " + " ".join("'" + part.replace("'", "''") + "'" for part in command_args)
    html, evidence = build_report(
        aggregate, aggregate_path=str(source), aggregate_hash=source_hash,
        detailed_url=detailed_url, generation_command=command,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    summary_path = output.parent / "summary.json"
    write_json(summary_path, evidence)
    write_json(output.parent / "manifest.json", {
        "version": REPORT_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run": aggregate["run_id"], "source_aggregate": str(source),
        "source_aggregate_sha256": source_hash, "source_manifest_sha256": source_manifest_hash,
        "generator_sha256": sha256_file(Path(__file__)), "graph_count": 3, "method_diagram_count": 1,
        "generation_command": command, "report_python": sys.version,
        "conclusion_scope": "Dataset-average end-to-end unjudged MAE; not a formal non-inferiority claim.",
        "files": {p.name: {"path": str(p), "sha256": sha256_file(p)} for p in (output, summary_path)},
    })
    if sha256_file(source) != source_hash or sha256_file(source_manifest_path) != source_manifest_hash:
        raise RuntimeError("source artifacts changed during report generation")
    return output
