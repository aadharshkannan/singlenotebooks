from __future__ import annotations

import subprocess
from html import escape
from pathlib import Path
from typing import Any

from sampling_comparison.v6_report import (
    DEFAULT_BROWSERS,
    FULL_METHOD_IDS,
    METHOD_COLOR,
    METHOD_DISPLAY,
    V6ReportInputs,
    _safe_int,
    _safe_json_script_blob,
    default_inputs,
    load_v6_artifacts,
    validate_pdf_file,
)

DEFAULT_INPUT_DIR = Path("outputs_sampling_v6") / "runs" / "full-30-7arm-20260827"
DEFAULT_OUTPUT_NAME = "sampling-v6-concise-report.html"
DEFAULT_PDF_NAME = "sampling-v6-concise-report.pdf"
CONCISE_METHOD_IDS = tuple(FULL_METHOD_IDS)


def _detect_browser() -> str:
    import os
    import shutil

    browser = os.environ.get("EDGE_BROWSER_PATH") or os.environ.get("CHROME_BROWSER_PATH")
    if browser:
        return browser
    for candidate in (shutil.which(name) for name in ("msedge", "microsoft-edge", "chrome", "google-chrome")):
        if candidate:
            return candidate
    for candidate in DEFAULT_BROWSERS:
        path = Path(candidate)
        if path.exists():
            return str(path)
    raise FileNotFoundError("No Edge or Chrome browser binary was found on this machine")


def compose_pdf_command(browser_path: str | None, html_path: Path, pdf_path: Path) -> list[str]:
    browser = browser_path or _detect_browser()
    return [
        browser,
        "--headless=new",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path.resolve()}",
        str(Path(html_path).resolve()),
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _metric_mean(row: dict[str, Any], metric: str) -> float:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    stat = metrics.get(metric) if isinstance(metrics.get(metric), dict) else {}
    return _safe_float(stat.get("mean"), default=_safe_float(row.get(metric), default=0.0))


def _metric_value_for_method(aggregate_rows: list[dict[str, Any]], method_id: str, metric: str, cap: int | None = None) -> float:
    filtered = [row for row in aggregate_rows if str(row.get("method_id") or "").strip() == str(method_id)]
    if cap is not None:
        filtered = [row for row in filtered if int(row.get("cap") or 0) == int(cap)]
    if not filtered:
        return 0.0
    return _metric_mean(filtered[0], metric)


def _caps_from_rows(rows: list[dict[str, Any]]) -> list[int]:
    caps = {int(row.get("cap") or 0) for row in rows if int(row.get("cap") or 0) > 0}
    return sorted(caps)


def _summary_row_for_method(aggregate_rows: list[dict[str, Any]], method_id: str, cap: int) -> dict[str, Any]:
    method_row = next((row for row in aggregate_rows if str(row.get("method_id") or "") == method_id and int(row.get("cap") or 0) == cap), {})
    return {
        "method_id": method_id,
        "label": METHOD_DISPLAY.get(method_id, method_id.replace("_", " ").title()),
        "color": METHOD_COLOR.get(method_id, "#34495e"),
        "cap": cap,
        "mae": _metric_mean(method_row, "absolute_aggregate_mae"),
        "concept": _metric_mean(method_row, "concept_coverage"),
    }


def _best_method_by_metric(aggregate_rows: list[dict[str, Any]], metric: str, cap: int | None = None) -> tuple[str, float, float]:
    candidates: list[tuple[str, float]] = []
    for method_id in CONCISE_METHOD_IDS:
        value = _metric_value_for_method(aggregate_rows, method_id, metric, cap)
        candidates.append((method_id, value))
    if not candidates:
        return ("arm1_global_random", 0.0, 0.0)
    if metric == "absolute_aggregate_mae":
        method_id, value = min(candidates, key=lambda item: item[1])
    else:
        method_id, value = max(candidates, key=lambda item: item[1])
    return method_id, value, _metric_value_for_method(aggregate_rows, method_id, "concept_coverage", cap)


def _canonical_bundle_context(aggregate: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    pop = aggregate.get("population_audit") if isinstance(aggregate.get("population_audit"), dict) else {}
    source_summary = aggregate.get("source_summary") if isinstance(aggregate.get("source_summary"), dict) else {}
    corpora = source_summary.get("corpora") if isinstance(source_summary.get("corpora"), list) else []
    corpus_names: list[str] = []
    for item in corpora:
        if isinstance(item, dict):
            corpus_id = str(item.get("corpus_id") or item.get("name") or "").strip()
            if corpus_id:
                corpus_names.append(corpus_id)
    if not corpus_names:
        corpus_names = ["dense_2500", "historical_300"]

    unit_count = _safe_int(pop.get("unit_count"), default=2800)
    agent_count = _safe_int(pop.get("agent_count"), default=105)
    if unit_count <= 0:
      unit_count = 2800
    if agent_count <= 0:
      agent_count = 105

    seed_values = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        try:
            seed_values.append(int(run.get("seed")))
        except (TypeError, ValueError):
            pass
    seed_count = len(set(seed_values)) if seed_values else 30
    if seed_count <= 0:
        seed_count = 30

    return {
        "unit_count": unit_count,
        "agent_count": agent_count,
        "corpora": corpus_names,
        "seed_count": seed_count,
    }


def _render_line_chart(title: str, series: list[dict[str, Any]], percent: bool = False) -> str:
    width, height = 720, 220
    left, right, top, bottom = 52, 20, 20, 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    labels = [str(item["label"]) for item in series[0]["points"]] if series and series[0].get("points") else []
    values: list[float] = []
    for item in series:
        values.extend(float(point["value"]) for point in item.get("points", []))
    if not values:
        return "<svg class='chart-svg' viewBox='0 0 720 220' aria-label='No data'></svg>"
    display_values = [value * 100.0 for value in values] if percent else values
    y_min = min(display_values)
    y_max = max(display_values)
    if y_min == y_max:
        y_min -= 0.05
        y_max += 0.05

    def _x(index: int, total: int) -> float:
        if total <= 1:
            return left + plot_w / 2.0
        return left + (index / (total - 1)) * plot_w

    def _y(value: float, *, source_value: bool = True) -> float:
        val = value * 100.0 if percent and source_value else value
        if val < y_min:
            val = float(y_min)
        if val > y_max:
            val = float(y_max)
        return top + plot_h - ((val - y_min) / (y_max - y_min + 1e-9)) * plot_h

    paths: list[str] = []
    for item in series:
        points = " ".join(f"{_x(i, len(item['points'])):.1f},{_y(float(point['value'])):.1f}" for i, point in enumerate(item["points"]))
        paths.append(f"<polyline fill='none' stroke='{item['color']}' stroke-width='3' points='{points}' />")
    y_ticks = [y_min + (y_max - y_min) * i / 4 for i in range(5)]
    tick_html = "".join(
        f"<line x1='{left}' x2='{width-right}' y1='{_y(tick, source_value=False)}' y2='{_y(tick, source_value=False)}' stroke='#e5ebf1' stroke-dasharray='4 4' />"
        f"<text x='{left-10}' y='{_y(tick, source_value=False)+4}' text-anchor='end' font-size='10' fill='#526172'>{f'{tick:.0f}%' if percent else f'{tick:.3f}'}</text>"
        for tick in y_ticks
    )
    x_labels = "".join(f"<text x='{_x(i, len(labels)):.1f}' y='{height-10}' text-anchor='middle' font-size='10' fill='#526172'>{escape(str(label))}</text>" for i, label in enumerate(labels))
    legend = "".join(
      f"<g><title>{escape(item['name'])}</title><line x1='{24 + idx * 90}' x2='{42 + idx * 90}' y1='7' y2='7' stroke='{item['color']}' stroke-width='3' />"
      f"<text x='{48 + idx * 90}' y='11' font-size='10' fill='#2a3744'>{escape(item['name'].split()[0])}</text></g>"
      for idx, item in enumerate(series)
    )
    return f"<figure class='chart-shell'><svg class='chart-svg' viewBox='0 0 {width} {height}' aria-label='{escape(title)}'><rect x='0' y='0' width='{width}' height='{height}' fill='white'/>{tick_html}<line x1='{left}' x2='{left}' y1='{top}' y2='{height-bottom}' stroke='#6c7d8f'/><line x1='{left}' x2='{width-right}' y1='{height-bottom}' y2='{height-bottom}' stroke='#6c7d8f'/>{''.join(paths)}{x_labels}<g>{legend}</g></svg><figcaption>{escape(title)}</figcaption></figure>"


def _tradeoff_svg(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<svg class='chart-svg' viewBox='0 0 720 220' aria-label='No tradeoff data'></svg>"
    rows = sorted(rows, key=lambda r: (float(r.get("concept", 0.0)), -float(r.get("mae", 0.0))))
    width, height = 720, 220
    left, right, top, bottom = 54, 20, 20, 38
    plot_w, plot_h = width - left - right, height - top - bottom
    maex = [float(r["mae"]) for r in rows]
    cover = [float(r["concept"]) for r in rows]
    xmin, xmax = min(maex) * 0.9, max(maex) * 1.1 or 1.0
    ymin, ymax = min(cover) * 0.9, max(cover) * 1.1 or 1.0
    if xmax == xmin:
        xmax = xmin + 0.05
    if ymax == ymin:
        ymax = ymin + 0.05

    def _x(value: float) -> float:
        return left + ((value - xmin) / (xmax - xmin)) * plot_w

    def _y(value: float) -> float:
        return top + plot_h - ((value - ymin) / (ymax - ymin)) * plot_h

    circles = []
    for row in rows:
        x = _x(float(row["mae"]))
        y = _y(float(row["concept"]))
        label = row["label"].split()[0]
        circles.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5' fill='{row['color']}' opacity='0.85'><title>{escape(row['label'])}: MAE {row['mae']:.4f}, concept coverage {row['concept']:.1%}</title></circle>"
            f"<text x='{x+7:.1f}' y='{y-7:.1f}' text-anchor='start' font-size='10' fill='#2a3744'>{escape(label)}</text>"
        )
    axes = f"<line x1='{left}' x2='{width-right}' y1='{height-bottom}' y2='{height-bottom}' stroke='#617284'/><line x1='{left}' x2='{left}' y1='{top}' y2='{height-bottom}' stroke='#617284'/><text x='{left}' y='{top-6}' font-size='11' fill='#2a3744'>Concept coverage</text><text x='{width-62}' y='{height-10}' font-size='11' fill='#2a3744'>MAE</text>"
    return f"<svg class='chart-svg' viewBox='0 0 {width} {height}' aria-label='MAE vs concept coverage tradeoff'>{axes}{''.join(circles)}</svg>"


def _leader_matrix_svg(aggregate_rows: list[dict[str, Any]], caps: list[int]) -> str:
    def compact_names(method_ids: list[str]) -> str:
        labels = [METHOD_DISPLAY.get(method_id, method_id).split()[0] for method_id in method_ids]
        return " / ".join(labels)

    width, height = 720, 205
    left, right = 136, 16
    cell_width = (width - left - right) / max(1, len(caps))
    rows = [("Lowest MAE", "absolute_aggregate_mae", False, 62), ("Most concepts", "concept_coverage", True, 132)]
    parts = ["<rect x='0' y='0' width='720' height='205' fill='white'/>"]
    for index, cap in enumerate(caps):
        x = left + index * cell_width
        parts.append(f"<text x='{x + cell_width / 2:.1f}' y='24' text-anchor='middle' font-size='11' font-weight='700' fill='#334e64'>{cap}</text>")
    for row_label, metric, higher_is_better, y in rows:
        parts.append(f"<text x='10' y='{y + 25}' font-size='11' font-weight='700' fill='#334e64'>{escape(row_label)}</text>")
        for index, cap in enumerate(caps):
            values = [(method_id, _metric_value_for_method(aggregate_rows, method_id, metric, cap)) for method_id in CONCISE_METHOD_IDS]
            target = max(value for _, value in values) if higher_is_better else min(values, key=lambda item: item[1])[1]
            leaders = [method_id for method_id, value in values if abs(value - target) <= 1e-9]
            x = left + index * cell_width + 4
            cell_fill = "#edf7f1" if higher_is_better else "#eef5fa"
            value_text = f"{target:.1%}" if higher_is_better else f"{target:.4f}"
            parts.append(f"<rect x='{x:.1f}' y='{y}' width='{cell_width - 8:.1f}' height='50' rx='6' fill='{cell_fill}' stroke='#d5e0ea'/>")
            parts.append(f"<text x='{x + (cell_width - 8) / 2:.1f}' y='{y + 20}' text-anchor='middle' font-size='10' font-weight='700' fill='#243849'>{escape(compact_names(leaders))}</text>")
            parts.append(f"<text x='{x + (cell_width - 8) / 2:.1f}' y='{y + 38}' text-anchor='middle' font-size='10' fill='#5f6f80'>{value_text}</text>")
    parts.append("<text x='420' y='194' text-anchor='middle' font-size='10' fill='#5f6f80'>Each column names the leader; ties are preserved.</text>")
    return "<figure class='chart-shell'><svg class='chart-svg' viewBox='0 0 720 205' aria-label='MAE and concept coverage leaders by cap'>" + "".join(parts) + "</svg><figcaption>Cap-by-cap leaders: lower MAE and higher concept coverage</figcaption></figure>"


def _render_summary_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='empty-state'>No summary rows available.</p>"
    items = rows[:7]
    header = "<tr><th>Method</th><th>Cap</th><th>MAE</th><th>Concept</th></tr>"
    body = "".join(
        f"<tr><td>{escape(item['label'])}</td><td>{item['cap']}</td><td>{item['mae']:.4f}</td><td>{item['concept']:.1%}</td></tr>"
        for item in items
    )
    return f"<div class='table-wrap'><table><thead>{header}</thead><tbody>{body}</tbody></table></div>"


def _build_recommendation(aggregate_rows: list[dict[str, Any]]) -> str:
    caps = _caps_from_rows(aggregate_rows)
    if not caps:
        return "Use the report explorer to compare the seven sampling methods on the fixed population."
    low_cap = min(caps)
    high_cap = max(caps)

    def _leader_names(metric: str, cap: int) -> str:
        values = [_metric_value_for_method(aggregate_rows, method_id, metric, cap) for method_id in CONCISE_METHOD_IDS]
        target = max(values) if metric == "concept_coverage" else min(values)
        tied = [method_id for method_id, value in zip(CONCISE_METHOD_IDS, values) if abs(value - target) <= 1e-9]
        return ", ".join(METHOD_DISPLAY.get(method_id, method_id) for method_id in tied)

    low_mae_method, low_mae, _ = _best_method_by_metric(aggregate_rows, "absolute_aggregate_mae", low_cap)
    low_cov_method, low_cov_val, _ = _best_method_by_metric(aggregate_rows, "concept_coverage", low_cap)
    high_mae_method, _, _ = _best_method_by_metric(aggregate_rows, "absolute_aggregate_mae", high_cap)
    high_cov_method, _, _ = _best_method_by_metric(aggregate_rows, "concept_coverage", high_cap)
    low_cap_label = METHOD_DISPLAY.get(low_mae_method, low_mae_method)
    low_cov_label = METHOD_DISPLAY.get(low_cov_method, low_cov_method)
    low_cov_tied = _leader_names("concept_coverage", low_cap)
    high_cap_label = METHOD_DISPLAY.get(high_mae_method, high_mae_method)
    high_cov_label = METHOD_DISPLAY.get(high_cov_method, high_cov_method)

    if "," in low_cov_tied:
        low_coverage_summary = f"{low_cov_tied} are tied for broadest concept coverage"
    else:
        low_coverage_summary = f"{low_cov_label} has the broadest concept coverage"
    if high_mae_method == high_cov_method:
        high_balance = f"{high_cap_label} remains the best combined MAE/coverage balance at {high_cap} sessions"
    else:
        high_balance = f"{high_cap_label} leads on MAE while {high_cov_label} stays strongest on concept coverage at {high_cap} sessions"

    return (
        f"At the low-cap end ({low_cap} sessions), {low_cap_label} has the lowest MAE ({low_mae:.4f}) while {low_coverage_summary} ({low_cov_val:.2%}). At the high-cap end ({high_cap} sessions), {high_balance}. Choose lowest MAE when budget is tight, and use concept coverage as the representativeness guardrail when broader source coverage matters more than a small MAE gain."
    )


def _arm_walkthroughs() -> list[tuple[str, str]]:
    return [
        (
            "ARM1",
            "<div class='arm-field'><strong>Steps</strong><span>Draw sessions uniformly across the fixed population; keep the selected set simple and budgeted.</span></div><div class='arm-field'><strong>Estimator</strong><span>Unweighted sample mean for the pass-rate estimate.</span></div><div class='arm-field'><strong>Strength</strong><span>Simple, robust low-cap baseline with minimal tuning.</span></div><div class='arm-field'><strong>Risk</strong><span>Can miss concept-rich regions when a few hot agents dominate the population.</span></div><div class='arm-field'><strong>Choose when</strong><span>Use as the clean baseline when the goal is speed and low complexity under a strict cap.</span></div>",
        ),
        (
            "ARM2",
            "<div class='arm-field'><strong>Steps</strong><span>Use embedding similarity and Azure Search donor retrieval, then same-agent IDW for unjudged units.</span></div><div class='arm-field'><strong>Estimator</strong><span>Continuous per-unit estimate averaged over the population.</span></div><div class='arm-field'><strong>Strength</strong><span>Retains calibration signal and local continuity when donor quality is strong.</span></div><div class='arm-field'><strong>Risk</strong><span>Embedding/search quality can dominate behavior and amplify donor mismatch.</span></div><div class='arm-field'><strong>Choose when</strong><span>Choose when continuous probabilities and imputation calibration are primary.</span></div>",
        ),
        (
            "ARM2.5",
            "<div class='arm-field'><strong>Steps</strong><span>Reuse ARM2 membership and ARM2 per-unit probabilities, then threshold each unit at 0.5 before averaging.</span></div><div class='arm-field'><strong>Estimator</strong><span>Binaryized per-unit population average; threshold first, then average.</span></div><div class='arm-field'><strong>Strength</strong><span>Interpretable binary pass/fail framing with the same sampled evidence as ARM2.</span></div><div class='arm-field'><strong>Risk</strong><span>Loses continuous calibration detail around the 0.5 boundary.</span></div><div class='arm-field'><strong>Choose when</strong><span>Choose when governance needs per-unit binary decisions from ARM2 membership.</span></div>",
        ),
        (
            "ARM3",
            "<div class='arm-field'><strong>Steps</strong><span>Enforce a floor to protect agent coverage, then allocate remaining budget across strata.</span></div><div class='arm-field'><strong>Estimator</strong><span>Unweighted mean across selected sessions.</span></div><div class='arm-field'><strong>Strength</strong><span>Broad representation and a minimum floor before cap is consumed.</span></div><div class='arm-field'><strong>Risk</strong><span>Some budget is spent on floor guarantees instead of pure MAE minimization.</span></div><div class='arm-field'><strong>Choose when</strong><span>Choose when broad representation is non-negotiable.</span></div>",
        ),
        (
            "ARM4",
            "<div class='arm-field'><strong>Steps</strong><span>Round-robin across agents and balance by strata without a hard floor.</span></div><div class='arm-field'><strong>Estimator</strong><span>Unweighted mean.</span></div><div class='arm-field'><strong>Strength</strong><span>Good diversity and spread under the cap.</span></div><div class='arm-field'><strong>Risk</strong><span>Rare pockets can still be missed if strata are noisy.</span></div><div class='arm-field'><strong>Choose when</strong><span>Choose when diversity matters more than explicit floor guarantees.</span></div>",
        ),
        (
            "ARM5",
            "<div class='arm-field'><strong>Steps</strong><span>Use ARM4 membership and apply Hajek weighting over agent-marginal inclusion behavior.</span></div><div class='arm-field'><strong>Estimator</strong><span>Agent-marginal Hajek ratio estimate.</span></div><div class='arm-field'><strong>Strength</strong><span>Coverage-preserving weighted correction over ARM4 membership.</span></div><div class='arm-field'><strong>Risk</strong><span>Low-cap weighting variance can increase MAE.</span></div><div class='arm-field'><strong>Choose when</strong><span>Choose when you need ARM4-style membership with a design-aware weighted estimator.</span></div>",
        ),
        (
            "ARM6",
            "<div class='arm-field'><strong>Steps</strong><span>Use ARM4 membership and apply represented (agent, use-case) joint-cell post-stratified Hajek weights.</span></div><div class='arm-field'><strong>Estimator</strong><span>Joint-cell weighted Hajek over realized represented cells only.</span></div><div class='arm-field'><strong>Strength</strong><span>Aligns estimator to represented joint structure instead of only agent marginals.</span></div><div class='arm-field'><strong>Risk</strong><span>Zero-sample joint cells are not recovered; missing cells remain missing.</span></div><div class='arm-field'><strong>Choose when</strong><span>Choose when joint agent/use-case representativeness is a key estimator objective.</span></div>",
        ),
    ]


def _agent_summary_table(agent_metrics: list[dict[str, Any]], cap: int, methods: list[str]) -> str:
    rows: list[dict[str, Any]] = []
    for method_id in methods:
        method_rows = [row for row in agent_metrics if str(row.get("method_id")) == method_id and _safe_int(row.get("cap"), default=0) == cap]
        errs = [
            _safe_float(row.get("absolute_error"), default=0.0)
            for row in method_rows
            if row.get("absolute_error") is not None
        ]
        if not errs:
            continue
        rows.append(
            {
                "label": METHOD_DISPLAY.get(method_id, method_id),
                "mae": sum(errs) / len(errs),
            }
        )
    rows = sorted(rows, key=lambda item: item["mae"])[:7]
    if not rows:
        return "<p class='empty-state'>Agent-level summary unavailable for this bundle.</p>"
    body = "".join(f"<tr><td>{escape(item['label'])}</td><td>{item['mae']:.4f}</td></tr>" for item in rows)
    return "<div class='table-wrap'><table><thead><tr><th>Method</th><th>Macro per-agent MAE</th></tr></thead><tbody>" + body + "</tbody></table></div>"


def _prepare_payload(aggregate_rows: list[dict[str, Any]], runs: list[dict[str, Any]], agent_metrics: list[dict[str, Any]], caps: list[int]) -> dict[str, Any]:
    methods = [method_id for method_id in CONCISE_METHOD_IDS if any(str(row.get("method_id")) == method_id for row in aggregate_rows + runs + agent_metrics)]
    if not methods:
        methods = list(CONCISE_METHOD_IDS)
    labels = {method_id: METHOD_DISPLAY.get(method_id, method_id) for method_id in methods}
    colors = {method_id: METHOD_COLOR.get(method_id, "#34495e") for method_id in methods}
    return {
        "caps": caps,
        "methods": methods,
        "aggregate_rows": aggregate_rows,
        "runs": runs,
        "agent_metrics": agent_metrics,
        "labels": labels,
        "colors": colors,
        "metric_keys": ["absolute_aggregate_mae", "concept_coverage", "use_case_coverage", "agent_coverage"],
    }


def _validate_concise_methods(aggregate_rows: list[dict[str, Any]], runs: list[dict[str, Any]], agent_metrics: list[dict[str, Any]]) -> None:
    available = {str(row.get("method_id")) for row in aggregate_rows + runs + agent_metrics if row.get("method_id")}
    missing = [method_id for method_id in CONCISE_METHOD_IDS if method_id not in available]
    if missing:
        raise ValueError(f"concise report requires all seven methods; missing: {', '.join(missing)}")


def render_v6_concise_html_report(inputs: V6ReportInputs) -> str:
    artifacts = load_v6_artifacts(inputs)
    aggregate_rows = artifacts.aggregate.get("aggregate_rows") if isinstance(artifacts.aggregate.get("aggregate_rows"), list) else []
    runs = artifacts.runs
    agent_metrics = artifacts.agent_metrics
    _validate_concise_methods(aggregate_rows, runs, agent_metrics)
    caps = _caps_from_rows(aggregate_rows) or _caps_from_rows(runs)
    context = _canonical_bundle_context(artifacts.aggregate, runs)
    population_units = int(context.get("unit_count") or 2800)
    agent_count = int(context.get("agent_count") or 105)
    global_seed_count = int(context.get("seed_count") or 30)
    seed_label = "paired seed" if global_seed_count == 1 else "paired seeds"
    corpus_names = context.get("corpora") or ["dense_2500", "historical_300"]
    cap_choice = min(caps) if caps else 64

    summary_rows = [_summary_row_for_method(aggregate_rows, method_id, cap_choice) for method_id in CONCISE_METHOD_IDS]
    recommendation_text = _build_recommendation(aggregate_rows)

    mae_series = []
    concept_series = []
    for method_id in CONCISE_METHOD_IDS:
        mae_points = [{"label": str(cap), "value": _metric_value_for_method(aggregate_rows, method_id, "absolute_aggregate_mae", cap)} for cap in caps]
        concept_points = [{"label": str(cap), "value": _metric_value_for_method(aggregate_rows, method_id, "concept_coverage", cap)} for cap in caps]
        mae_series.append({"name": METHOD_DISPLAY.get(method_id, method_id), "color": METHOD_COLOR.get(method_id, "#34495e"), "points": mae_points})
        concept_series.append({"name": METHOD_DISPLAY.get(method_id, method_id), "color": METHOD_COLOR.get(method_id, "#34495e"), "points": concept_points})

    tradeoff_svg = _tradeoff_svg(summary_rows)
    mae_chart = _render_line_chart("MAE by cap", mae_series)
    concept_chart = _render_line_chart("Concept coverage by cap", concept_series, percent=True)
    leader_matrix = _leader_matrix_svg(aggregate_rows, caps)
    payload = _prepare_payload(aggregate_rows, runs, agent_metrics, caps)
    payload_blob = _safe_json_script_blob(payload)
    arm_cards = "".join(f"<article class='arm-card'><h3>{escape(label)}</h3>{text}</article>" for label, text in _arm_walkthroughs())
    summary_table = _render_summary_table(summary_rows)
    methods_html = "".join(
        f"<label class='toggle'><input class='method-toggle' type='checkbox' data-method='{escape(method_id)}' checked /> <span>{escape(METHOD_DISPLAY.get(method_id, method_id))}</span></label>"
        for method_id in CONCISE_METHOD_IDS
    )
    metric_tabs = "".join(
        f"<button class='metric-tab {'active' if idx == 0 else ''}' data-metric='{metric}'>{label}</button>"
        for idx, (metric, label) in enumerate(
            [
                ("absolute_aggregate_mae", "MAE"),
                ("concept_coverage", "Concept Coverage"),
                ("use_case_coverage", "Use-Case Coverage"),
                ("agent_coverage", "Agent Coverage"),
            ]
        )
    )
    agent_summary = _agent_summary_table(agent_metrics, cap_choice, list(CONCISE_METHOD_IDS))

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8' />
<meta name='viewport' content='width=device-width, initial-scale=1' />
<title>Sampling V6 concise report</title>
<style>
:root {{ --bg:#f3f7fb; --paper:#ffffff; --ink:#1f2a37; --muted:#5f6f80; --line:#d9e3ee; --accent:#2f6f9f; --shadow:0 12px 24px rgba(18,35,49,.08); }}
* {{ box-sizing:border-box; }}
html, body {{ margin:0; font-family:'Segoe UI',Verdana,sans-serif; background:linear-gradient(180deg,#edf5fa 0%,#f9fafb 100%); color:var(--ink); }}
body {{ line-height:1.5; }}
.main {{ max-width:1200px; margin:0 auto; padding:24px 18px 48px; }}
.layout {{ display:grid; grid-template-columns:220px minmax(0,1fr); gap:18px; }}
.toc {{ position:sticky; top:18px; background:var(--paper); border:1px solid var(--line); border-radius:12px; box-shadow:var(--shadow); padding:12px 14px; }}
.toc h3 {{ margin:0 0 10px; font-size:0.92rem; letter-spacing:0.06em; text-transform:uppercase; color:var(--muted); }}
.toc a {{ display:block; padding:5px 0; font-size:0.9rem; color:#2f4659; text-decoration:none; }}
header {{ background:var(--paper); border:1px solid var(--line); border-radius:14px; box-shadow:var(--shadow); padding:22px 20px; }}
header h1 {{ margin:0 0 10px; font-size:clamp(1.8rem,2.8vw,2.7rem); letter-spacing:-0.04em; }}
.subtitle {{ margin:0 0 14px; color:var(--muted); font-size:1.02rem; }}
.key {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin:12px 0 18px; }}
.key-card {{ background:#f5f9fd; border:1px solid var(--line); border-radius:10px; padding:10px 12px; }}
.key-card .label {{ color:var(--muted); font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; }}
.key-card .value {{ font-weight:700; margin-top:4px; }}
section {{ background:var(--paper); border:1px solid var(--line); border-radius:14px; box-shadow:var(--shadow); padding:18px 20px; margin-top:20px; }}
.section-head {{ display:flex; align-items:end; justify-content:space-between; gap:12px; margin-bottom:12px; border-bottom:1px solid var(--line); padding-bottom:8px; }}
.section-head h2 {{ margin:0; font-size:1.3rem; }}
.section-head span {{ color:var(--muted); font-size:0.8rem; }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.three-col {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
.arm-card {{ background:#fbfdff; border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.arm-card h3 {{ margin:0 0 8px; }}
.arm-field {{ display:grid; grid-template-columns:82px minmax(0,1fr); gap:8px; padding:5px 0; border-top:1px solid #edf1f5; font-size:0.86rem; }}
.arm-field:first-of-type {{ border-top:0; }}
.chart-shell {{ border:1px solid var(--line); border-radius:12px; padding:10px; background:#fff; }}
.chart-svg {{ display:block; width:100%; height:auto; max-width:100%; }}
figcaption {{ color:var(--muted); font-size:0.8rem; margin-top:6px; text-align:center; }}
.table-wrap {{ overflow:auto; }}
table {{ width:100%; border-collapse:collapse; min-width:420px; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ background:#f3f7fb; color:var(--muted); font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em; }}
.meta {{ color:var(--muted); font-size:0.88rem; }}
.toggle-group {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }}
.toggle {{ display:inline-flex; align-items:center; gap:8px; background:#f7fafd; border:1px solid var(--line); border-radius:999px; padding:6px 10px; font-size:0.88rem; }}
button.metric-tab, button.mode-tab, button.agent-metric-tab, button.agent-mode-tab {{ border:1px solid var(--line); background:#f5f8fb; border-radius:8px; padding:8px 10px; cursor:pointer; font:inherit; }}
button.metric-tab.active, button.mode-tab.active, button.agent-metric-tab.active, button.agent-mode-tab.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.metrics-controls {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:12px 0; }}
select {{ padding:8px 10px; border:1px solid var(--line); border-radius:8px; background:#fff; }}
.empty-state {{ color:var(--muted); font-style:italic; }}
.screen-only {{ display:block; }}
.print-only {{ display:none; }}
@media (max-width: 900px) {{
  .layout {{ display:block; }}
  .toc {{ position:static; margin-bottom:16px; }}
  .key, .two-col, .three-col {{ grid-template-columns:1fr; }}
  .section-head {{ display:block; }}
}}
@media print {{
  @page {{ size:A4 landscape; margin:12mm; }}
  body {{ background:#fff; }}
  .toc, .screen-only {{ display:none !important; }}
  .print-only {{ display:block !important; }}
  section, header {{ break-inside:avoid; page-break-inside:avoid; }}
  .main {{ max-width:100%; padding:6mm; }}
  .chart-svg {{ max-height:150mm; }}
}}
</style>
</head>
<body>
<div class='main'>
  <div class='layout'>
    <nav class='toc screen-only' aria-label='Quick navigation'>
      <h3>Contents</h3>
      <a href='#decision'>Decision at a glance</a>
      <a href='#measured'>What is measured</a>
      <a href='#techniques'>Seven techniques</a>
      <a href='#story'>Results story</a>
      <a href='#explorer'>Interactive explorer</a>
      <a href='#agent-explorer'>Agent-by-agent explorer</a>
      <a href='#conclusion'>Conclusion</a>
    </nav>
    <div>
      <header>
        <div class='meta'>Sampling V6 concise report</div>
        <h1>Choosing a Sampling Method Under Session Caps</h1>
        <p class='subtitle'>Population context: {population_units:,} sessions across {agent_count:,} agents, spanning {', '.join(corpus_names)} corpora with {global_seed_count} {seed_label}. ARM2 uses Azure Search for donor retrieval, and use-case coverage remains provisional because the Maven labels are classification-derived and some assignments are Ambiguous.</p>
        <div class='key'>
          <div class='key-card'><div class='label'>Population</div><div class='value'>{population_units:,} sessions</div></div>
          <div class='key-card'><div class='label'>Agents</div><div class='value'>{agent_count:,} agents</div></div>
          <div class='key-card'><div class='label'>Corpora</div><div class='value'>{' / '.join(corpus_names)}</div></div>
        </div>
        <p><strong>How to read:</strong> MAE is the absolute error between the estimator and the full-population benchmark. Concept coverage is the proportion of distinct source concepts in the full population represented by at least one sampled session; higher is better for representation. The seven methods use different estimators: ARM2/2.5 impute the full population, ARM1/3/4 estimate from selected sessions, ARM5 uses agent-marginal weights, and ARM6 estimates only represented agent/use-case support.</p>
      </header>

      <section id='decision'>
        <div class='section-head'><h2>Decision at a glance</h2><span>Low-cap / high-cap pattern</span></div>
        <div class='two-col'>
          <div>
            <p>{escape(recommendation_text)}</p>
            {_tradeoff_svg(summary_rows)}
          </div>
          <div>
            <div class='meta'>Selected cap summary</div>
            {summary_table}
          </div>
        </div>
        <p class='meta'><strong>Comparability caveat:</strong> the MAE leaderboard is descriptive, not a comparison of identical estimands. ARM6 MAE includes error from agent/use-case cells with no sampled support because its represented-cell estimator cannot recover those cells.</p>
      </section>

      <section id='measured'>
        <div class='section-head'><h2>What is measured</h2><span>Use fixed-population evidence correctly</span></div>
        <div class='three-col'>
          <div class='chart-shell'><strong>MAE</strong><p class='meta'>Lower is better; absolute difference from census benchmark at a cap.</p></div>
          <div class='chart-shell'><strong>Concept coverage</strong><p class='meta'>Higher is broader source representation.</p></div>
          <div class='chart-shell'><strong>Limit</strong><p class='meta'>Descriptive fixed-population evidence only; not general inference.</p></div>
        </div>
      </section>

      <section id='techniques'>
        <div class='section-head'><h2>Seven techniques</h2><span>Selection and estimator are distinct</span></div>
        <div class='three-col'>{arm_cards}</div>
      </section>

      <section id='story'>
        <div class='section-head'><h2>Results story</h2><span>Visual-first view</span></div>
        <div class='two-col'>
          <div>{mae_chart}</div>
          <div>{concept_chart}</div>
        </div>
        <div style='margin-top:16px;'>{leader_matrix}</div>
        <div class='print-only' style='margin-top:12px;'>
          <h3>Agent-level static summary</h3>
          {agent_summary}
          <p class='meta'>Macro per-agent MAE is computed from non-null per-agent absolute errors. ARM2 and ARM2.5 can estimate an agent even when n=0 because they impute its full population.</p>
        </div>
      </section>

      <section id='explorer' class='screen-only'>
        <div class='section-head'><h2>Interactive explorer</h2><span>Metric tabs, method toggles, aggregate or trial mode</span></div>
        <div class='metrics-controls'>
            <label for='cap-select'>Cap</label>
            <select id='cap-select'>
              {''.join(f"<option value='{cap}'>{cap}</option>" for cap in caps)}
            </select>
            <div>{metric_tabs}</div>
            <div>
               <button type='button' class='mode-tab active' data-mode='aggregate'>Aggregate</button>
               <button type='button' class='mode-tab' data-mode='trial'>Trial mode</button>
            </div>
            <div id='seed-control' style='display:none;'><label for='seed-select'>Seed</label><select id='seed-select'></select></div>
        </div>
        <div class='toggle-group'>{methods_html}</div>
        <div id='interactive-chart' class='chart-shell'></div>
        <div id='interactive-table' class='chart-shell' style='margin-top:12px;'></div>
      </section>

      <section id='agent-explorer' class='screen-only'>
        <div class='section-head'><h2>Agent-by-agent explorer</h2><span>Per-agent metrics by method</span></div>
        <p class='meta'>Per-agent MAE compares each method's agent estimate against that agent census rate. A row displays N/A only when that method cannot estimate the agent; ARM2 and ARM2.5 may remain available at n=0 through full-population imputation. Within one agent, ARM5 equals ARM4 because ARM5's agent-level weight is constant and cancels in the Hajek ratio.</p>
        <div class='metrics-controls'>
          <label for='agent-select'>Agent</label><select id='agent-select'></select>
          <label for='agent-cap-select'>Cap</label><select id='agent-cap-select'></select>
          <div>
            <button type='button' class='agent-mode-tab active' data-mode='aggregate'>Aggregate</button>
            <button type='button' class='agent-mode-tab' data-mode='trial'>Trial mode</button>
          </div>
          <div id='agent-seed-control' style='display:none;'><label for='agent-seed-select'>Seed</label><select id='agent-seed-select'></select></div>
        </div>
        <div class='metrics-controls'>
          <button type='button' class='agent-metric-tab active' data-metric='absolute_error'>MAE</button>
          <button type='button' class='agent-metric-tab' data-metric='estimate'>Estimate</button>
          <button type='button' class='agent-metric-tab' data-metric='concept_coverage'>Concept coverage</button>
          <button type='button' class='agent-metric-tab' data-metric='use_case_coverage'>Use-case coverage</button>
          <button type='button' class='agent-metric-tab' data-metric='n'>Sample count n</button>
          <button type='button' class='agent-metric-tab' data-metric='represented_population_fraction'>Represented population fraction</button>
        </div>
        <div class='toggle-group'>{methods_html.replace('method-toggle', 'agent-method-toggle')}</div>
        <div id='agent-chart' class='chart-shell'></div>
        <div id='agent-table' class='chart-shell' style='margin-top:12px;'></div>
      </section>

      <section id='conclusion'>
        <div class='section-head'><h2>Conclusion</h2><span>Fixed-bundle decision guidance</span></div>
        <p><strong>Recommendation:</strong> choose by cap and objective. Overall aggregate MAE answers the global estimate question, while agent-level error answers reliability per agent. Future work should elevate agent-level evaluation so decisions can jointly optimize global MAE and per-agent reliability.</p>
      </section>
    </div>
  </div>
</div>
<script id='artifact-json' type='application/json'>{payload_blob}</script>
<script>
(function () {{
  const data = JSON.parse(document.getElementById('artifact-json').textContent || '{{}}');
  const methodOrder = (data.methods || []).slice();

  const overallState = {{ cap: String((data.caps || [64])[0] || 64), metric: 'absolute_aggregate_mae', mode: 'aggregate', seed: '', visibleMethods: new Set(methodOrder) }};
  const overall = {{
    capSelect: document.getElementById('cap-select'),
    seedSelect: document.getElementById('seed-select'),
    seedControl: document.getElementById('seed-control'),
    chart: document.getElementById('interactive-chart'),
    table: document.getElementById('interactive-table'),
    metricTabs: Array.from(document.querySelectorAll('.metric-tab')),
    modeTabs: Array.from(document.querySelectorAll('.mode-tab')),
    toggles: Array.from(document.querySelectorAll('.method-toggle')),
  }};

  function overallMetricLabel(metric) {{
    const labels = {{ absolute_aggregate_mae: 'MAE', concept_coverage: 'Concept coverage', use_case_coverage: 'Use-case coverage', agent_coverage: 'Agent coverage' }};
    return labels[metric] || metric;
  }}

  function overallRowValue(row, metric) {{
    const stats = row && row.metrics ? row.metrics[metric] : null;
    if (stats && typeof stats === 'object' && typeof stats.mean === 'number') return stats.mean;
    if (typeof stats === 'number') return stats;
    if (typeof row?.[metric] === 'number') return row[metric];
    return 0;
  }}

  function overallFmtMetric(value, metric) {{
    const numeric = Number(value || 0);
    if (metric === 'absolute_aggregate_mae') return numeric.toFixed(4);
    return (100 * numeric).toFixed(1) + '%';
  }}

  function overallSeedsForCap() {{
    const seen = new Set();
    for (const row of (data.runs || [])) {{
      if (Number(row.cap || 0) === Number(overallState.cap || 0)) seen.add(String(row.seed || ''));
    }}
    return Array.from(seen).filter(Boolean).sort((a, b) => Number(a) - Number(b));
  }}

  function overallSelectionRows() {{
    const cap = Number(overallState.cap || 0);
    const source = overallState.mode === 'aggregate' ? (data.aggregate_rows || []) : (data.runs || []);
    let filtered = source.filter(row => Number(row.cap || 0) === cap && overallState.visibleMethods.has(String(row.method_id || '')));
    if (overallState.mode === 'trial') filtered = filtered.filter(row => String(row.seed || '') === String(overallState.seed || ''));
    return filtered.sort((a, b) => methodOrder.indexOf(String(a.method_id || '')) - methodOrder.indexOf(String(b.method_id || '')));
  }}

  function renderOverallChart() {{
    const rows = overallSelectionRows();
    if (!rows.length) {{
      overall.chart.innerHTML = '<p class="empty-state">No rows available for this selection.</p>';
      return;
    }}
    const width = 720, height = 250, left = 52, right = 20, top = 20, bottom = 42;
    const plotW = width - left - right, plotH = height - top - bottom;
    const values = rows.map(row => Number(overallRowValue(row, overallState.metric) || 0));
    const minV = Math.min(0, ...values);
    const maxV = Math.max(1e-9, ...values);
    const step = plotW / Math.max(rows.length, 1);
    const bars = rows.map((row, i) => {{
      const v = Number(values[i] || 0);
      const h = ((v - minV) / (maxV - minV || 1)) * plotH;
      const x = left + i * step + 14;
      const y = top + plotH - h;
      const label = (data.labels || {{}})[String(row.method_id || '')] || String(row.method_id || '');
      const color = (data.colors || {{}})[String(row.method_id || '')] || '#2f6f9f';
      return `<g><rect x='${{x}}' y='${{y}}' width='30' height='${{h}}' fill='${{color}}'></rect><text x='${{x+15}}' y='${{height-11}}' text-anchor='middle' font-size='10' fill='#41576a'>${{label.split(' ')[0]}}</text><text x='${{x+15}}' y='${{Math.max(12, y-6)}}' text-anchor='middle' font-size='10' fill='#1f2a37'>${{overallFmtMetric(v, overallState.metric)}}</text></g>`;
    }}).join('');
    overall.chart.innerHTML = `<svg class='chart-svg' viewBox='0 0 ${{width}} ${{height}}' aria-label='${{overallMetricLabel(overallState.metric)}} by method'><line x1='${{left}}' x2='${{left}}' y1='${{top}}' y2='${{height-bottom}}' stroke='#7589a1'></line><line x1='${{left}}' x2='${{width-right}}' y1='${{height-bottom}}' y2='${{height-bottom}}' stroke='#7589a1'></line>${{bars}}</svg>`;
  }}

  function renderOverallTable() {{
    const rows = overallSelectionRows();
    if (!rows.length) {{
      overall.table.innerHTML = '<p class="empty-state">No results available.</p>';
      return;
    }}
    const body = rows.map(row => {{
      const method = (data.labels || {{}})[String(row.method_id || '')] || String(row.method_id || '');
      const value = overallRowValue(row, overallState.metric);
      const actualTokens = Number(row.actual_token_count ?? row.actual_tokens ?? 0);
      const nominal = Number(row.nominal_budget ?? 0);
      const ratio = nominal > 0 ? actualTokens / nominal : 0;
      const seedCol = overallState.mode === 'trial' ? `<td>${{row.seed || 'n/a'}}</td>` : '';
      return `<tr><td>${{method}}</td>${{seedCol}}<td>${{overallFmtMetric(value, overallState.metric)}}</td><td>${{actualTokens.toLocaleString()}}</td><td>${{(100*ratio).toFixed(2)}}%</td></tr>`;
    }}).join('');
    const header = overallState.mode === 'trial'
      ? `<tr><th>Method</th><th>Seed</th><th>${{overallMetricLabel(overallState.metric)}}</th><th>Actual tokens</th><th>Actual / nominal</th></tr>`
      : `<tr><th>Method</th><th>${{overallMetricLabel(overallState.metric)}}</th><th>Actual tokens</th><th>Actual / nominal</th></tr>`;
    overall.table.innerHTML = `<table><thead>${{header}}</thead><tbody>${{body}}</tbody></table>`;
  }}

  function renderOverall() {{
    if (!overall.capSelect) return;
    renderOverallChart();
    renderOverallTable();
  }}

  overall.capSelect.value = overallState.cap;
  overall.capSelect.addEventListener('change', () => {{
    overallState.cap = overall.capSelect.value;
    const seeds = overallSeedsForCap();
    if (!seeds.includes(overallState.seed)) overallState.seed = seeds[0] || '';
    overall.seedSelect.innerHTML = seeds.map(seed => `<option value='${{seed}}'>Seed ${{seed}}</option>`).join('');
    overall.seedSelect.value = overallState.seed;
    renderOverall();
  }});

  overall.metricTabs.forEach(button => button.addEventListener('click', () => {{
    overallState.metric = button.getAttribute('data-metric') || 'absolute_aggregate_mae';
    overall.metricTabs.forEach(btn => btn.classList.toggle('active', btn === button));
    renderOverall();
  }}));

  overall.modeTabs.forEach(button => button.addEventListener('click', () => {{
    overallState.mode = button.getAttribute('data-mode') || 'aggregate';
    overall.modeTabs.forEach(btn => btn.classList.toggle('active', btn === button));
    if (overallState.mode === 'trial') {{
      overall.seedControl.style.display = 'inline-block';
      const seeds = overallSeedsForCap();
      if (!seeds.includes(overallState.seed)) overallState.seed = seeds[0] || '';
      overall.seedSelect.innerHTML = seeds.map(seed => `<option value='${{seed}}'>Seed ${{seed}}</option>`).join('');
      overall.seedSelect.value = overallState.seed;
    }} else {{
      overall.seedControl.style.display = 'none';
    }}
    renderOverall();
  }}));

  overall.seedSelect.addEventListener('change', () => {{
    overallState.seed = overall.seedSelect.value || '';
    renderOverall();
  }});

  overall.toggles.forEach(toggle => toggle.addEventListener('change', () => {{
    const method = toggle.getAttribute('data-method') || '';
    if (!method) return;
    if (toggle.checked) overallState.visibleMethods.add(method);
    else overallState.visibleMethods.delete(method);
    renderOverall();
  }}));

  const initialOverallSeeds = overallSeedsForCap();
  overallState.seed = initialOverallSeeds[0] || '';
  overall.seedSelect.innerHTML = initialOverallSeeds.map(seed => `<option value='${{seed}}'>Seed ${{seed}}</option>`).join('');
  overall.seedSelect.value = overallState.seed;
  renderOverall();

  const agentState = {{ agent: '', cap: String((data.caps || [64])[0] || 64), metric: 'absolute_error', mode: 'aggregate', seed: '', visibleMethods: new Set(methodOrder) }};
  const agentUi = {{
    agentSelect: document.getElementById('agent-select'),
    capSelect: document.getElementById('agent-cap-select'),
    seedSelect: document.getElementById('agent-seed-select'),
    seedControl: document.getElementById('agent-seed-control'),
    metricTabs: Array.from(document.querySelectorAll('.agent-metric-tab')),
    modeTabs: Array.from(document.querySelectorAll('.agent-mode-tab')),
    toggles: Array.from(document.querySelectorAll('.agent-method-toggle')),
    chart: document.getElementById('agent-chart'),
    table: document.getElementById('agent-table'),
  }};

  function agentMetricLabel(metric) {{
    const labels = {{ absolute_error: 'MAE', estimate: 'Estimate', concept_coverage: 'Concept coverage', use_case_coverage: 'Use-case coverage', n: 'Sample count n', represented_population_fraction: 'Represented population fraction' }};
    return labels[metric] || metric;
  }}

  function agentFmtMetric(value, metric) {{
    if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A';
    const numeric = Number(value);
    if (metric === 'n') return String(Math.round(numeric));
    if (metric === 'estimate' || metric === 'concept_coverage' || metric === 'use_case_coverage' || metric === 'represented_population_fraction') return (100 * numeric).toFixed(1) + '%';
    return numeric.toFixed(4);
  }}

  function agentRowsFiltered() {{
    const metrics = (data.agent_metrics || []).filter(row =>
      String(row.agent_id || '') === String(agentState.agent || '') &&
      Number(row.cap || 0) === Number(agentState.cap || 0) &&
      agentState.visibleMethods.has(String(row.method_id || ''))
    );
    if (agentState.mode === 'trial') return metrics.filter(row => String(row.seed || '') === String(agentState.seed || ''));
    return metrics;
  }}

  function aggregateAgentRows(rows) {{
    const byMethod = new Map();
    for (const row of rows) {{
      const method = String(row.method_id || '');
      if (!byMethod.has(method)) byMethod.set(method, []);
      byMethod.get(method).push(row);
    }}
    const out = [];
    for (const [method, methodRows] of byMethod.entries()) {{
      const mean = (selector) => {{
        const vals = methodRows.map(selector).filter(v => v !== null && v !== undefined && Number.isFinite(Number(v)));
        if (!vals.length) return null;
        return vals.reduce((a, b) => a + Number(b), 0) / vals.length;
      }};
      out.push({{
        method_id: method,
        seed: 'aggregate',
        n: mean(r => r.n),
        N: mean(r => r.N),
        census_rate: mean(r => r.census_rate),
        estimate: mean(r => r.estimate),
        absolute_error: mean(r => r.absolute_error),
        concept_coverage: mean(r => r.concept_coverage),
        use_case_coverage: mean(r => r.use_case_coverage),
        represented_population_fraction: mean(r => r.represented_population_fraction),
        estimator: (methodRows[0] && methodRows[0].estimator) ? methodRows[0].estimator : 'unknown',
      }});
    }}
    return out.sort((a, b) => methodOrder.indexOf(String(a.method_id || '')) - methodOrder.indexOf(String(b.method_id || '')));
  }}

  function agentSelectionRows() {{
    const rows = agentRowsFiltered();
    if (agentState.mode === 'aggregate') return aggregateAgentRows(rows);
    const grouped = new Map();
    for (const row of rows) grouped.set(String(row.method_id || ''), row);
    return methodOrder.filter(method => grouped.has(method) && agentState.visibleMethods.has(method)).map(method => grouped.get(method)).slice(0, 7);
  }}

  function agentSeeds() {{
    const seeds = new Set();
    for (const row of (data.agent_metrics || [])) {{
      if (String(row.agent_id || '') === String(agentState.agent || '') && Number(row.cap || 0) === Number(agentState.cap || 0)) seeds.add(String(row.seed || ''));
    }}
    return Array.from(seeds).filter(Boolean).sort((a, b) => Number(a) - Number(b));
  }}

  function renderAgentChart() {{
    const rows = agentSelectionRows();
    if (!rows.length) {{
      agentUi.chart.innerHTML = '<p class="empty-state">No agent metrics available for this selection.</p>';
      return;
    }}
    const width = 720, height = 240, left = 52, right = 20, top = 20, bottom = 42;
    const plotW = width - left - right, plotH = height - top - bottom;
    const values = rows.map(row => row[agentState.metric]);
    const finite = values.filter(v => v !== null && v !== undefined && Number.isFinite(Number(v))).map(Number);
    if (!finite.length) {{
      agentUi.chart.innerHTML = '<p class="empty-state">All selected rows are N/A for this metric.</p>';
      return;
    }}
    const minV = Math.min(0, ...finite);
    const maxV = Math.max(1e-9, ...finite);
    const step = plotW / Math.max(rows.length, 1);
    const bars = rows.map((row, i) => {{
      const vRaw = values[i];
      const label = (data.labels || {{}})[String(row.method_id || '')] || String(row.method_id || '');
      const color = (data.colors || {{}})[String(row.method_id || '')] || '#2f6f9f';
      if (vRaw === null || vRaw === undefined || !Number.isFinite(Number(vRaw))) {{
        const x0 = left + i * step + 14;
        return `<g><rect x='${{x0}}' y='${{top + plotH - 2}}' width='30' height='2' fill='#c9d5e1'></rect><text x='${{x0+15}}' y='${{height-11}}' text-anchor='middle' font-size='10' fill='#41576a'>${{label.split(' ')[0]}}</text><text x='${{x0+15}}' y='${{top+plotH-8}}' text-anchor='middle' font-size='10' fill='#6f8091'>N/A</text></g>`;
      }}
      const v = Number(vRaw);
      const h = ((v - minV) / (maxV - minV || 1)) * plotH;
      const x = left + i * step + 14;
      const y = top + plotH - h;
      return `<g><rect x='${{x}}' y='${{y}}' width='30' height='${{h}}' fill='${{color}}'></rect><text x='${{x+15}}' y='${{height-11}}' text-anchor='middle' font-size='10' fill='#41576a'>${{label.split(' ')[0]}}</text><text x='${{x+15}}' y='${{Math.max(12, y-6)}}' text-anchor='middle' font-size='10' fill='#1f2a37'>${{agentFmtMetric(v, agentState.metric)}}</text></g>`;
    }}).join('');
    agentUi.chart.innerHTML = `<svg class='chart-svg' viewBox='0 0 ${{width}} ${{height}}' aria-label='Agent explorer chart'><line x1='${{left}}' x2='${{left}}' y1='${{top}}' y2='${{height-bottom}}' stroke='#7589a1'></line><line x1='${{left}}' x2='${{width-right}}' y1='${{height-bottom}}' y2='${{height-bottom}}' stroke='#7589a1'></line>${{bars}}</svg>`;
  }}

  function renderAgentTable() {{
    const rows = agentSelectionRows();
    if (!rows.length) {{
      agentUi.table.innerHTML = '<p class="empty-state">No agent table rows available.</p>';
      return;
    }}
    const body = rows.map(row => {{
      const label = (data.labels || {{}})[String(row.method_id || '')] || String(row.method_id || '');
      const n = Number(row.n || 0);
      const maeValue = (row.absolute_error === null || row.absolute_error === undefined) ? 'N/A' : Number(row.absolute_error).toFixed(4);
      const estValue = (row.estimate === null || row.estimate === undefined) ? 'N/A' : (100 * Number(row.estimate)).toFixed(1) + '%';
      return `<tr><td>${{label}}</td><td>${{agentFmtMetric(row[agentState.metric], agentState.metric)}}</td><td>${{Math.round(Number(row.n || 0))}}</td><td>${{Math.round(Number(row.N || 0))}}</td><td>${{row.census_rate == null ? 'N/A' : (100 * Number(row.census_rate)).toFixed(1) + '%'}} </td><td>${{estValue}}</td><td>${{maeValue}}</td><td>${{row.estimator || 'unknown'}}</td></tr>`;
    }}).join('');
    agentUi.table.innerHTML = `<table><thead><tr><th>Method</th><th>${{agentMetricLabel(agentState.metric)}}</th><th>n</th><th>N</th><th>Census rate</th><th>Estimate</th><th>MAE</th><th>Estimator</th></tr></thead><tbody>${{body}}</tbody></table>`;
  }}

  function renderAgent() {{
    renderAgentChart();
    renderAgentTable();
  }}

  const uniqueAgents = Array.from(new Set((data.agent_metrics || []).map(row => String(row.agent_id || '')).filter(Boolean))).sort();
  const capOptions = (data.caps || []).slice();
  agentUi.agentSelect.innerHTML = uniqueAgents.map(a => `<option value='${{a}}'>${{a}}</option>`).join('');
  agentState.agent = uniqueAgents[0] || '';
  agentUi.agentSelect.value = agentState.agent;
  agentUi.capSelect.innerHTML = capOptions.map(cap => `<option value='${{cap}}'>${{cap}}</option>`).join('');
  agentUi.capSelect.value = agentState.cap;

  agentUi.agentSelect.addEventListener('change', () => {{
    agentState.agent = agentUi.agentSelect.value || '';
    const seeds = agentSeeds();
    if (!seeds.includes(agentState.seed)) agentState.seed = seeds[0] || '';
    agentUi.seedSelect.innerHTML = seeds.map(seed => `<option value='${{seed}}'>Seed ${{seed}}</option>`).join('');
    agentUi.seedSelect.value = agentState.seed;
    renderAgent();
  }});

  agentUi.capSelect.addEventListener('change', () => {{
    agentState.cap = agentUi.capSelect.value || String(capOptions[0] || 64);
    const seeds = agentSeeds();
    if (!seeds.includes(agentState.seed)) agentState.seed = seeds[0] || '';
    agentUi.seedSelect.innerHTML = seeds.map(seed => `<option value='${{seed}}'>Seed ${{seed}}</option>`).join('');
    agentUi.seedSelect.value = agentState.seed;
    renderAgent();
  }});

  agentUi.modeTabs.forEach(button => button.addEventListener('click', () => {{
    agentState.mode = button.getAttribute('data-mode') || 'aggregate';
    agentUi.modeTabs.forEach(btn => btn.classList.toggle('active', btn === button));
    if (agentState.mode === 'trial') {{
      agentUi.seedControl.style.display = 'inline-block';
      const seeds = agentSeeds();
      if (!seeds.includes(agentState.seed)) agentState.seed = seeds[0] || '';
      agentUi.seedSelect.innerHTML = seeds.map(seed => `<option value='${{seed}}'>Seed ${{seed}}</option>`).join('');
      agentUi.seedSelect.value = agentState.seed;
    }} else {{
      agentUi.seedControl.style.display = 'none';
    }}
    renderAgent();
  }}));

  agentUi.seedSelect.addEventListener('change', () => {{
    agentState.seed = agentUi.seedSelect.value || '';
    renderAgent();
  }});

  agentUi.metricTabs.forEach(button => button.addEventListener('click', () => {{
    agentState.metric = button.getAttribute('data-metric') || 'absolute_error';
    agentUi.metricTabs.forEach(btn => btn.classList.toggle('active', btn === button));
    renderAgent();
  }}));

  agentUi.toggles.forEach(toggle => toggle.addEventListener('change', () => {{
    const method = toggle.getAttribute('data-method') || '';
    if (!method) return;
    if (toggle.checked) agentState.visibleMethods.add(method);
    else agentState.visibleMethods.delete(method);
    renderAgent();
  }}));

  const initialAgentSeeds = agentSeeds();
  agentState.seed = initialAgentSeeds[0] || '';
  agentUi.seedSelect.innerHTML = initialAgentSeeds.map(seed => `<option value='${{seed}}'>Seed ${{seed}}</option>`).join('');
  agentUi.seedSelect.value = agentState.seed;
  renderAgent();
}})();
</script>
</body>
</html>
"""
    return html


def write_v6_concise_report(output_path: Path, inputs: V6ReportInputs, *, pdf: bool = False, browser_path: str | None = None) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = render_v6_concise_html_report(inputs)
    path.write_text(html, encoding="utf-8")
    if pdf:
        pdf_path = path.with_name(DEFAULT_PDF_NAME) if path.name == DEFAULT_OUTPUT_NAME else path.with_suffix(".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        command = compose_pdf_command(browser_path, path, pdf_path)
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        validate_pdf_file(pdf_path)
    return path
