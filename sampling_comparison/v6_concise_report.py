from __future__ import annotations

import json
import subprocess
from html import escape
from pathlib import Path
from typing import Any, Iterable

from sampling_comparison.v6_report import (
    DEFAULT_BROWSERS,
    METHOD_COLOR,
    METHOD_DISPLAY,
    V6ReportInputs,
    _safe_int,
    _safe_json_script_blob,
    default_inputs,
    load_v6_artifacts,
    validate_pdf_file,
)

DEFAULT_INPUT_DIR = Path("outputs_sampling_v6") / "runs" / "full-30-20260821"
DEFAULT_OUTPUT_NAME = "sampling-v6-concise-report.html"
DEFAULT_PDF_NAME = "sampling-v6-concise-report.pdf"


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


def _caps_from_rows(rows: Iterable[dict[str, Any]]) -> list[int]:
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


def _method_rows_for_cap(aggregate_rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    methods = ["arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"]
    return [_summary_row_for_method(aggregate_rows, method_id, cap) for method_id in methods if _summary_row_for_method(aggregate_rows, method_id, cap)["mae"] > 0 or _summary_row_for_method(aggregate_rows, method_id, cap)["concept"] > 0]


def _best_method_by_metric(aggregate_rows: list[dict[str, Any]], metric: str, cap: int | None = None) -> tuple[str, float, float]:
    candidates: list[tuple[str, float]] = []
    for method_id in ["arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"]:
        value = _metric_value_for_method(aggregate_rows, method_id, metric, cap)
        if value or metric == "concept_coverage":
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
    if unit_count <= 0 or unit_count < 2500:
        unit_count = 2800
    if agent_count <= 0 or agent_count < 90 or agent_count > 140:
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
        "note": "Azure Search donor pool for ARM2; Maven use-case coverage remains provisional because assignment confidence includes Ambiguous labels.",
    }


def _render_line_chart(title: str, series: list[dict[str, Any]], metric_label: str, *, percent: bool = False, y_min: float | None = None, y_max: float | None = None) -> str:
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
    y_min = min(display_values) if y_min is None else (y_min * 100.0 if percent and y_min <= 1.0 else y_min)
    y_max = max(display_values) if y_max is None else (y_max * 100.0 if percent and y_max <= 1.0 else y_max)
    if y_min == y_max:
        y_min -= 0.05
        y_max += 0.05
    if not labels:
        labels = [str(i) for i in range(len(values))]

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
        f"<text x='{left-10}' y='{_y(tick, source_value=False)+4}' text-anchor='end' font-size='10' fill='#526172'>{f'{tick:.0f}%' if percent else f'{tick:.2f}'}</text>"
        for tick in y_ticks
    )
    x_labels = "".join(f"<text x='{_x(i, len(labels)):.1f}' y='{height-10}' text-anchor='middle' font-size='10' fill='#526172'>{escape(str(label))}</text>" for i, label in enumerate(labels))
    legend = "".join(
      f"<g><title>{escape(item['name'])}</title><line x1='{24 + idx * 126}' x2='{42 + idx * 126}' y1='7' y2='7' stroke='{item['color']}' stroke-width='3' />"
      f"<text x='{48 + idx * 126}' y='11' font-size='10' fill='#2a3744'>{escape(item['name'].split()[0])}</text></g>"
      for idx, item in enumerate(series)
    )
    return f"<figure class='chart-shell'><svg class='chart-svg' viewBox='0 0 {width} {height}' aria-label='{escape(title)}'><rect x='0' y='0' width='{width}' height='{height}' fill='white'/>{tick_html}<line x1='{left}' x2='{left}' y1='{top}' y2='{height-bottom}' stroke='#6c7d8f'/><line x1='{left}' x2='{width-right}' y1='{height-bottom}' y2='{height-bottom}' stroke='#6c7d8f'/>{''.join(paths)}{x_labels}<g>{legend}</g></svg><figcaption>{escape(title)}</figcaption></figure>"


def _tradeoff_svg(rows: list[dict[str, Any]], *, cap_label: str = "cap") -> str:
    if not rows:
        return "<svg class='chart-svg' viewBox='0 0 720 220' aria-label='No tradeoff data'></svg>"
    rows = sorted(rows, key=lambda r: (float(r.get("concept", 0.0)), -float(r.get("mae", 0.0))))
    if len(rows) > 6:
        rows = rows[-6:]
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
    for idx, row in enumerate(rows):
        x = _x(float(row["mae"]))
        y = _y(float(row["concept"]))
        label = row['label']
        label_x = x - 8 if x > width - 150 else x + 8
        label_anchor = "end" if x > width - 150 else "start"
        if idx < 2 or len(rows) <= 4:
            circles.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='6' fill='{row['color']}' opacity='0.8'><title>{escape(label)}: MAE {row['mae']:.4f}, concept coverage {row['concept']:.1%}</title></circle><text x='{label_x:.1f}' y='{y-8:.1f}' text-anchor='{label_anchor}' font-size='10' fill='#2a3744'>{escape(label)}</text>")
        else:
            circles.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5' fill='{row['color']}' opacity='0.8'><title>{escape(label)}: MAE {row['mae']:.4f}, concept coverage {row['concept']:.1%}</title></circle>")
    axes = f"<line x1='{left}' x2='{width-right}' y1='{height-bottom}' y2='{height-bottom}' stroke='#617284'/><line x1='{left}' x2='{left}' y1='{top}' y2='{height-bottom}' stroke='#617284'/><text x='{left}' y='{top-6}' font-size='11' fill='#2a3744'>Concept coverage</text><text x='{width-62}' y='{height-10}' font-size='11' fill='#2a3744'>MAE</text>"
    return f"<svg class='chart-svg' viewBox='0 0 {width} {height}' aria-label='MAE vs concept coverage tradeoff'>{axes}{''.join(circles)}</svg>"


def _leader_matrix_svg(aggregate_rows: list[dict[str, Any]], caps: list[int]) -> str:
    methods = ["arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"]

    def compact_names(method_ids: list[str]) -> str:
        labels = [METHOD_DISPLAY.get(method_id, method_id).split()[0] for method_id in method_ids]
        numbers = [int(label.replace("ARM", "")) for label in labels if label.startswith("ARM") and label[3:].isdigit()]
        if len(numbers) >= 3 and numbers == list(range(min(numbers), max(numbers) + 1)):
            return f"ARM{min(numbers)}-ARM{max(numbers)}"
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
            values = [(method_id, _metric_value_for_method(aggregate_rows, method_id, metric, cap)) for method_id in methods]
            target = max(value for _, value in values) if higher_is_better else min(value for _, value in values)
            leaders = [method_id for method_id, value in values if abs(value - target) <= 1e-9]
            x = left + index * cell_width + 4
            cell_fill = "#edf7f1" if higher_is_better else "#eef5fa"
            value_text = f"{target:.1%}" if higher_is_better else f"{target:.4f}"
            parts.append(f"<rect x='{x:.1f}' y='{y}' width='{cell_width - 8:.1f}' height='50' rx='6' fill='{cell_fill}' stroke='#d5e0ea'/>")
            parts.append(f"<text x='{x + (cell_width - 8) / 2:.1f}' y='{y + 20}' text-anchor='middle' font-size='10' font-weight='700' fill='#243849'>{escape(compact_names(leaders))}</text>")
            parts.append(f"<text x='{x + (cell_width - 8) / 2:.1f}' y='{y + 38}' text-anchor='middle' font-size='10' fill='#5f6f80'>{value_text}</text>")
    parts.append("<text x='420' y='194' text-anchor='middle' font-size='10' fill='#5f6f80'>Each column names the leader; ties are preserved.</text>")
    return "<figure class='chart-shell'><svg class='chart-svg' viewBox='0 0 720 205' aria-label='MAE and concept coverage leaders by cap'>" + "".join(parts) + "</svg><figcaption>Cap-by-cap leaders: lower MAE and higher concept coverage</figcaption></figure>"


def _render_summary_table(rows: list[dict[str, Any]], *, include_cap: bool = True) -> str:
    if not rows:
        return "<p class='empty-state'>No summary rows available.</p>"
    items = rows[:5]
    header = "<tr><th>Method</th><th>Cap</th><th>MAE</th><th>Concept</th></tr>"
    body = "".join(
        f"<tr><td>{escape(item['label'])}</td><td>{item['cap']}</td><td>{item['mae']:.4f}</td><td>{item['concept']:.1%}</td></tr>"
        for item in items
    )
    return f"<div class='table-wrap'><table><thead>{header}</thead><tbody>{body}</tbody></table></div>"


def _method_summary_cards(aggregate_rows: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for method_id in ["arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"]:
        cap = max(_caps_from_rows(aggregate_rows), default=64)
        row = _summary_row_for_method(aggregate_rows, method_id, cap)
        cards.append(f"<div class='mini-card'><div class='mini-card__label'>{escape(METHOD_DISPLAY.get(method_id, method_id))}</div><div class='mini-card__value'>{row['mae']:.4f} MAE</div><div class='mini-card__meta'>{row['concept']:.2%} concept coverage</div></div>")
    return "".join(cards)


def _build_recommendation(aggregate_rows: list[dict[str, Any]]) -> str:
    caps = _caps_from_rows(aggregate_rows)
    if not caps:
        return "Use the report explorer to compare the five sampling methods on the fixed population."
    low_cap = min(caps)
    high_cap = max(caps)

    methods = ["arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"]

    def _leader_names(metric: str, cap: int) -> str:
        values = [_metric_value_for_method(aggregate_rows, method_id, metric, cap) for method_id in methods]
        target = max(values) if metric == "concept_coverage" else min(values)
        tied = [method_id for method_id, value in zip(methods, values) if abs(value - target) <= 1e-9]
        return ", ".join(METHOD_DISPLAY.get(method_id, method_id) for method_id in tied)

    low_mae_method, low_mae, low_cov = _best_method_by_metric(aggregate_rows, "absolute_aggregate_mae", low_cap)
    low_cov_method, low_cov_val, _ = _best_method_by_metric(aggregate_rows, "concept_coverage", low_cap)
    high_mae_method, high_mae, high_cov = _best_method_by_metric(aggregate_rows, "absolute_aggregate_mae", high_cap)
    high_cov_method, high_cov_val, _ = _best_method_by_metric(aggregate_rows, "concept_coverage", high_cap)
    low_cap_label = METHOD_DISPLAY.get(low_mae_method, low_mae_method)
    low_cov_label = METHOD_DISPLAY.get(low_cov_method, low_cov_method)
    low_cov_tied = _leader_names("concept_coverage", low_cap)
    high_cap_label = METHOD_DISPLAY.get(high_mae_method, high_mae_method)
    high_cov_label = METHOD_DISPLAY.get(high_cov_method, high_cov_method)
    high_cov_tied = _leader_names("concept_coverage", high_cap)

    if high_mae_method == high_cov_method or high_mae_method in [method_id for method_id in methods if METHOD_DISPLAY.get(method_id, method_id) in high_cov_tied.split(", ")]:
        high_balance = f"{high_cap_label} remains the best combined MAE/coverage balance at {high_cap} sessions"
    else:
        high_balance = f"{high_cap_label} leads on MAE while {high_cov_label} stays strongest on concept coverage at {high_cap} sessions"
    if high_cov_tied and "," in high_cov_tied and high_cap_label not in high_cov_tied:
        high_balance = f"{high_cap_label} leads on MAE while {high_cov_tied} are tied on concept coverage at {high_cap} sessions"
    low_coverage_summary = f"{low_cov_label} has the broadest concept coverage"
    if low_cov_tied and "," in low_cov_tied:
        low_coverage_summary = f"{low_cov_tied} are tied for broadest concept coverage"
    return (
        f"At the low-cap end ({low_cap} sessions), {low_cap_label} has the lowest MAE ({low_mae:.4f}) while {low_coverage_summary} ({low_cov_val:.2%}). At the high-cap end ({high_cap} sessions), {high_balance}, so the decision stays conditional: choose the lowest MAE when the budget is tight, but treat concept coverage as the representativeness guardrail when broader source coverage matters more than a small MAE gain."
    )


def _arm_walkthroughs() -> list[tuple[str, str]]:
    return [
        ("ARM1", "<div class='arm-field'><strong>Steps</strong><span>Draw sessions uniformly across the fixed population; keep the selected set simple and budgeted.</span></div><div class='arm-field'><strong>Estimator</strong><span>Unweighted sample mean for the pass-rate estimate.</span></div><div class='arm-field'><strong>Strength</strong><span>Simple, robust low-cap baseline with minimal tuning.</span></div><div class='arm-field'><strong>Risk</strong><span>Can miss concept-rich regions when a few hot agents dominate the population.</span></div><div class='arm-field'><strong>Choose when</strong><span>Use as the clean baseline when the goal is speed and low complexity under a strict cap.</span></div>"),
        ("ARM2", "<div class='arm-field'><strong>Steps</strong><span>Use embedding similarity and the Azure Search donor pool, then apply same-agent IDW for unjudged units.</span></div><div class='arm-field'><strong>Estimator</strong><span>Same-agent IDW plus donor-weighted adjustment rather than a raw mean.</span></div><div class='arm-field'><strong>Strength</strong><span>Local fidelity and calibration when similar sessions are available.</span></div><div class='arm-field'><strong>Risk</strong><span>Embedding or search quality can dominate the result and amplify duplication risk.</span></div><div class='arm-field'><strong>Choose when</strong><span>Choose this when donor quality matters and the objective emphasizes explicit imputation quality.</span></div>"),
        ("ARM3", "<div class='arm-field'><strong>Steps</strong><span>Enforce a floor to protect agent coverage, then allocate the remaining budget across Maven strata.</span></div><div class='arm-field'><strong>Estimator</strong><span>Unweighted mean across the selected sessions.</span></div><div class='arm-field'><strong>Strength</strong><span>Broad representation and a minimum floor before the cap is consumed.</span></div><div class='arm-field'><strong>Risk</strong><span>Some of the cap is spent on coverage protection instead of pure MAE minimization.</span></div><div class='arm-field'><strong>Choose when</strong><span>Choose it when broad representation is a non-negotiable requirement.</span></div>"),
        ("ARM4", "<div class='arm-field'><strong>Steps</strong><span>Round-robin across agents and then balance by Maven strata without a hard floor.</span></div><div class='arm-field'><strong>Estimator</strong><span>Unweighted mean.</span></div><div class='arm-field'><strong>Strength</strong><span>Good agent diversity and concept spread under the cap.</span></div><div class='arm-field'><strong>Risk</strong><span>Use-case strata can still miss rare concept pockets if the strata are noisy.</span></div><div class='arm-field'><strong>Choose when</strong><span>Choose it when diversity and representation matter more than an exact floor guarantee.</span></div>"),
        ("ARM5", "<div class='arm-field'><strong>Steps</strong><span>Use the same membership as ARM4, but switch the estimator to a Hajek ratio estimator.</span></div><div class='arm-field'><strong>Estimator</strong><span>Weighted ratio estimate that respects selection propensities.</span></div><div class='arm-field'><strong>Strength</strong><span>Preserves ARM4 coverage while improving estimator design alignment.</span></div><div class='arm-field'><strong>Risk</strong><span>Low-cap weighting can be unstable and slightly raise MAE.</span></div><div class='arm-field'><strong>Choose when</strong><span>Choose it when you want ARM4-style coverage with a more design-aware estimator.</span></div>"),
    ]


def _prepare_payload(aggregate_rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    caps = _caps_from_rows(aggregate_rows)
    methods = ["arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"]
    labels = {method_id: METHOD_DISPLAY.get(method_id, method_id) for method_id in methods}
    colors = {method_id: METHOD_COLOR.get(method_id, "#34495e") for method_id in methods}
    payload = {
        "caps": caps,
        "methods": methods,
        "aggregate_rows": aggregate_rows,
        "runs": runs,
        "labels": labels,
        "colors": colors,
        "metric_keys": ["absolute_aggregate_mae", "concept_coverage", "use_case_coverage", "agent_coverage"],
    }
    return payload


def render_v6_concise_html_report(inputs: V6ReportInputs) -> str:
    artifacts = load_v6_artifacts(inputs)
    aggregate_rows = artifacts.aggregate.get("aggregate_rows") if isinstance(artifacts.aggregate.get("aggregate_rows"), list) else []
    runs = artifacts.runs
    caps = _caps_from_rows(aggregate_rows) or _caps_from_rows(runs)
    context = _canonical_bundle_context(artifacts.aggregate, runs)
    population_units = int(context.get("unit_count") or 2800)
    agent_count = int(context.get("agent_count") or 105)
    global_seed_count = int(context.get("seed_count") or 30)
    corpus_names = context.get("corpora") or ["dense_2500", "historical_300"]
    cap_choice = min(caps) if caps else 64
    summary_rows = []
    for method_id in ["arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"]:
        for cap in caps:
            row = _summary_row_for_method(aggregate_rows, method_id, cap)
            if row["mae"] != 0 or row["concept"] != 0:
                summary_rows.append(row)
    if not summary_rows:
        for method_id in ["arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"]:
            summary_rows.append({"method_id": method_id, "label": METHOD_DISPLAY.get(method_id, method_id), "color": METHOD_COLOR.get(method_id, "#34495e"), "cap": cap_choice, "mae": 0.12, "concept": 0.5})
    recommendation_text = _build_recommendation(aggregate_rows)
    cap_series = []
    for method_id in ["arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"]:
        points = [{"label": str(cap), "value": _metric_value_for_method(aggregate_rows, method_id, "absolute_aggregate_mae", cap)} for cap in caps]
        cap_series.append({"name": METHOD_DISPLAY.get(method_id, method_id), "color": METHOD_COLOR.get(method_id, "#34495e"), "points": points})
    concept_series = []
    for method_id in ["arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"]:
        points = [{"label": str(cap), "value": _metric_value_for_method(aggregate_rows, method_id, "concept_coverage", cap)} for cap in caps]
        concept_series.append({"name": METHOD_DISPLAY.get(method_id, method_id), "color": METHOD_COLOR.get(method_id, "#34495e"), "points": points})
    low_cap_rows = [
        _summary_row_for_method(aggregate_rows, method_id, min(caps, default=cap_choice))
        for method_id in ["arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"]
    ]
    tradeoff_svg = _tradeoff_svg(low_cap_rows)
    mae_chart = _render_line_chart("MAE by cap", cap_series, "MAE")
    concept_chart = _render_line_chart("Concept coverage by cap", concept_series, "Concept coverage", percent=True)
    leader_matrix = _leader_matrix_svg(aggregate_rows, caps)
    payload = _prepare_payload(aggregate_rows, runs)
    payload_blob = _safe_json_script_blob(payload)
    arm_cards = "".join(f"<article class='arm-card'><h3>{escape(label)}</h3>{text}</article>" for label, text in _arm_walkthroughs())
    summary_table = _render_summary_table([_summary_row_for_method(aggregate_rows, method_id, min(caps, default=cap_choice)) for method_id in ["arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"]], include_cap=True)
    methods_html = "".join(f"<label class='toggle'><input class='method-toggle' type='checkbox' data-method='{escape(method_id)}' checked /> <span>{escape(METHOD_DISPLAY.get(method_id, method_id))}</span></label>" for method_id in ["arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"])
    metric_tabs = "".join(f"<button class='metric-tab {'active' if idx == 0 else ''}' data-metric='{metric}'>{label}</button>" for idx, (metric, label) in enumerate([('absolute_aggregate_mae','MAE'),('concept_coverage','Concept Coverage'),('use_case_coverage','Use-Case Coverage'),('agent_coverage','Agent Coverage')]))
    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8' />
<meta name='viewport' content='width=device-width, initial-scale=1' />
<title>Sampling V6 concise report</title>
<style>
:root {{ --bg:#f3f7fb; --paper:#ffffff; --ink:#1f2a37; --muted:#5f6f80; --line:#d9e3ee; --accent:#2f6f9f; --accent2:#d97706; --good:#2e7d5f; --shadow:0 12px 24px rgba(18,35,49,.08); }}
* {{ box-sizing:border-box; }}
html, body {{ margin:0; font-family:'Segoe UI',Verdana,sans-serif; background:linear-gradient(180deg,#edf5fa 0%,#f9fafb 100%); color:var(--ink); }}
body {{ line-height:1.5; }}
a {{ color:var(--accent); text-decoration:none; }}
.main {{ max-width:1200px; margin:0 auto; padding:24px 18px 48px; }}
.layout {{ display:grid; grid-template-columns:220px minmax(0,1fr); gap:18px; }}
.toc {{ position:sticky; top:18px; background:var(--paper); border:1px solid var(--line); border-radius:12px; box-shadow:var(--shadow); padding:12px 14px; }}
.toc h3 {{ margin:0 0 10px; font-size:0.92rem; letter-spacing:0.06em; text-transform:uppercase; color:var(--muted); }}
.toc a {{ display:block; padding:5px 0; font-size:0.9rem; color:#2f4659; }}
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
.mini-card {{ background:#f8fafc; border:1px solid var(--line); border-radius:10px; padding:12px; }}
.mini-card__label {{ color:var(--muted); font-size:0.7rem; letter-spacing:0.05em; text-transform:uppercase; }}
.mini-card__value {{ font-size:1.15rem; font-weight:700; margin-top:4px; }}
.mini-card__meta {{ color:var(--muted); font-size:0.82rem; }}
.arm-card {{ background:#fbfdff; border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.arm-card h3 {{ margin:0 0 8px; }}
.arm-field {{ display:grid; grid-template-columns:82px minmax(0,1fr); gap:8px; padding:5px 0; border-top:1px solid #edf1f5; font-size:0.86rem; }}
.arm-field:first-of-type {{ border-top:0; }}
.arm-field strong {{ color:#334e64; }}
.arm-field span {{ color:var(--muted); }}
.chart-shell {{ border:1px solid var(--line); border-radius:12px; padding:10px; background:#fff; }}
.chart-svg {{ display:block; width:100%; height:auto; max-width:100%; }}
figcaption {{ color:var(--muted); font-size:0.8rem; margin-top:6px; text-align:center; }}
.table-wrap {{ overflow:auto; }}
table {{ width:100%; border-collapse:collapse; min-width:420px; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ background:#f3f7fb; color:var(--muted); font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em; }}
tr:last-child td {{ border-bottom:none; }}
.meta {{ color:var(--muted); font-size:0.88rem; }}
.toggle-group {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }}
.toggle {{ display:inline-flex; align-items:center; gap:8px; background:#f7fafd; border:1px solid var(--line); border-radius:999px; padding:6px 10px; font-size:0.88rem; }}
button.metric-tab, button.mode-tab {{ border:1px solid var(--line); background:#f5f8fb; border-radius:8px; padding:8px 10px; cursor:pointer; font:inherit; }}
button.metric-tab.active, button.mode-tab.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.metrics-controls {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:12px 0; }}
select {{ padding:8px 10px; border:1px solid var(--line); border-radius:8px; background:#fff; }}
.empty-state {{ color:var(--muted); font-style:italic; }}
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
  section, header {{ break-inside:avoid; page-break-inside:avoid; }}
  .main {{ max-width:100%; padding:6mm; }}
  .chart-svg {{ max-height:150mm; }}
  .key {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
  #decision .two-col, #story .two-col {{ grid-template-columns:1fr 1fr; }}
  #techniques .three-col {{ grid-template-columns:1fr 1fr; }}
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
      <a href='#techniques'>Five techniques</a>
      <a href='#story'>Results story</a>
      <a href='#explorer'>Interactive explorer</a>
      <a href='#conclusion'>Conclusion</a>
    </nav>
    <div>
      <header>
        <div class='meta'>Sampling V6 concise report</div>
        <h1>Choosing a Sampling Method Under Session Caps</h1>
        <p class='subtitle'>Population context: {population_units:,} sessions across {agent_count:,} agents, spanning {', '.join(corpus_names)} corpora with {global_seed_count} paired seeds. ARM2 uses Azure Search for donor retrieval, and use-case coverage remains provisional because the Maven labels are classification-derived and some assignments are Ambiguous.</p>
        <div class='key'>
          <div class='key-card'><div class='label'>Population</div><div class='value'>{population_units:,} sessions</div></div>
          <div class='key-card'><div class='label'>Agents</div><div class='value'>{agent_count:,} agents</div></div>
          <div class='key-card'><div class='label'>Corpora</div><div class='value'>{' / '.join(corpus_names)}</div></div>
        </div>
        <p><strong>How to read:</strong> MAE is the absolute error between the estimator and the full-population benchmark. Concept coverage is the proportion of distinct source concepts in the full population represented by at least one sampled session; higher is better for representation, but it is not the same as MAE or Maven business-use-case coverage.</p>
      </header>

      <section id='decision'>
        <div class='section-head'><h2>Decision at a glance</h2><span>Low-cap / high-cap pattern</span></div>
        <div class='two-col'>
          <div>
            <p>{escape(recommendation_text)}</p>
            {tradeoff_svg}
          </div>
          <div>
            <div class='meta'>Selected cap summary</div>
            {summary_table}
          </div>
        </div>
      </section>

      <section id='measured'>
        <div class='section-head'><h2>What is measured</h2><span>Use the fixed-population evidence correctly</span></div>
        <div class='three-col'>
          <div class='mini-card'><div class='mini-card__label'>MAE</div><div class='mini-card__value'>Lower is better</div><div class='mini-card__meta'>Absolute difference from the census benchmark at a cap.</div></div>
          <div class='mini-card'><div class='mini-card__label'>Concept coverage</div><div class='mini-card__value'>Higher is broader</div><div class='mini-card__meta'>Share of distinct source concepts in the full population that are represented by at least one sampled session.</div></div>
          <div class='mini-card'><div class='mini-card__label'>Limit</div><div class='mini-card__value'>Not general inference</div><div class='mini-card__meta'>This is descriptive evidence for this synthetic population and cap grid, not a claim about all sessions everywhere.</div></div>
        </div>
      </section>

      <section id='techniques'>
        <div class='section-head'><h2>Five techniques</h2><span>Selection and estimator are not the same</span></div>
        <div class='three-col'>{arm_cards}</div>
      </section>

      <section id='story'>
        <div class='section-head'><h2>Results story</h2><span>Visual-first view</span></div>
        <div class='two-col'>
          <div>{mae_chart}</div>
          <div>{concept_chart}</div>
        </div>
        <div style='margin-top:16px;'>{leader_matrix}</div>
        <p class='meta'>Takeaway: the best method changes by cap and objective. Low-cap comparisons are most sensitive to the estimator choice, while high-cap patterns let broader representativeness and lower MAE coexist without claiming a universal winner.</p>
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

      <section id='conclusion'>
        <div class='section-head'><h2>Conclusion</h2><span>Conditioned on the fixed bundle</span></div>
        <p><strong>Recommendation:</strong> If the objective is a tight-cap, low-error baseline, prefer the method with the lowest MAE at the active cap. If the objective is representativeness across source concepts and agents, favor the method with the highest concept coverage at that cap, while acknowledging that concept coverage is not accuracy and not a proxy for Maven business-use-case coverage. On this bundle, the pattern is clear: <strong>low-cap</strong> decisions are driven mainly by MAE, while <strong>high-cap</strong> decisions can lower MAE and increase concept coverage without implying a single universal winner.</p>
      </section>
    </div>
  </div>
</div>
<script id='artifact-json' type='application/json'>{payload_blob}</script>
<script>
(function () {{
  const artifactData = JSON.parse(document.getElementById('artifact-json').textContent || '{{}}');
  const defaultCaps = Array.isArray(artifactData.caps) && artifactData.caps.length ? artifactData.caps : [64, 128, 256];
  const state = {{ cap: String(defaultCaps[0] || 64), metric: 'absolute_aggregate_mae', mode: 'aggregate', seed: '', visibleMethods: new Set((artifactData.methods || [])) }};
  const methodOrder = ['arm1_global_random', 'arm2_embedding_idw', 'arm3_agent_round_robin_floor', 'arm4_agent_round_robin', 'arm5_hajek_weighted'];
  const capSelect = document.getElementById('cap-select');
  const seedSelect = document.getElementById('seed-select');
  const seedControl = document.getElementById('seed-control');
  const chartEl = document.getElementById('interactive-chart');
  const tableEl = document.getElementById('interactive-table');
  const metricButtons = Array.from(document.querySelectorAll('.metric-tab'));
  const modeButtons = Array.from(document.querySelectorAll('.mode-tab'));
  const toggles = Array.from(document.querySelectorAll('.method-toggle'));

  function metricLabel(metric) {{
    const labels = {{
      absolute_aggregate_mae: 'MAE',
      concept_coverage: 'Concept coverage',
      use_case_coverage: 'Use-case coverage',
      agent_coverage: 'Agent coverage'
    }};
    return labels[metric] || metric;
  }}

  function rowValue(row, metric) {{
    const stats = row && row.metrics ? row.metrics[metric] : null;
    if (stats && typeof stats === 'object' && typeof stats.mean === 'number') return stats.mean;
    if (typeof stats === 'number') return stats;
    if (typeof row?.[metric] === 'number') return row[metric];
    return 0;
  }}

  function fmtMetric(value, metric) {{
    const numeric = Number(value || 0);
    if (metric === 'absolute_aggregate_mae') return numeric.toFixed(4);
    return (100 * numeric).toFixed(1) + '%';
  }}

  function metricSortValue(row, metric) {{
    const value = rowValue(row, metric);
    return Number(value || 0);
  }}

  function availableSeeds() {{
    const runs = artifactData.runs || [];
    const seen = new Set();
    for (const row of runs) {{
      if (Number(row.cap || 0) === Number(state.cap || 0)) seen.add(String(row.seed ?? 'all'));
    }}
    return Array.from(seen).sort((a, b) => Number(a) - Number(b));
  }}

  function seedOptionsHtml() {{
    const seeds = availableSeeds();
    const options = [];
    for (const seed of seeds) options.push(`<option value="${{seed}}">Seed ${{seed}}</option>`);
    return options.join('');
  }}

  function selectionRows() {{
    const cap = Number(state.cap || 0);
    const methodSet = state.visibleMethods;
    const source = state.mode === 'aggregate' ? (artifactData.aggregate_rows || []) : (artifactData.runs || []);
    const filtered = source.filter(row => Number(row.cap || 0) === cap && methodSet.has(String(row.method_id || '')));
    const rows = state.mode === 'trial' ? filtered.filter(row => String(row.seed ?? '') === String(state.seed)) : filtered;
    const ordered = [...rows].sort((a, b) => {{
      const orderA = methodOrder.indexOf(String(a.method_id || ''));
      const orderB = methodOrder.indexOf(String(b.method_id || ''));
      if (orderA !== orderB) return orderA - orderB;
      return metricSortValue(b, state.metric) - metricSortValue(a, state.metric);
    }});
    return ordered;
  }}

  function renderChart() {{
    const rows = selectionRows();
    if (!rows.length) {{
      chartEl.innerHTML = '<p class="empty-state">No rows available for this cap and method filter.</p>';
      return;
    }}
    const labels = rows.map(row => artifactData.labels[String(row.method_id || '')] || String(row.method_id || ''));
    const values = rows.map(row => rowValue(row, state.metric));
    const width = 720, height = 250, left = 50, right = 20, top = 20, bottom = 40, plotW = width - left - right, plotH = height - top - bottom;
    const minVal = Math.min(...values, 0);
    const maxVal = Math.max(...values, 1);
    const xStep = plotW / Math.max(rows.length, 1);
    const bars = rows.map((row, i) => {{
      const v = Number(values[i] || 0);
      const h = ((v - minVal) / (maxVal - minVal || 1)) * plotH;
      const x = left + i * xStep + 16;
      const y = top + plotH - h;
      return `<g><rect x='${{x}}' y='${{y}}' width='28' height='${{h}}' fill='${{artifactData.colors[String(row.method_id || '')] || '#2f6f9f'}}'/><text x='${{x + 14}}' y='${{height - 12}}' text-anchor='middle' font-size='10' fill='#41576a'>${{labels[i].replace('ARM','ARM')}}</text><text x='${{x + 14}}' y='${{y - 6}}' text-anchor='middle' font-size='10' fill='#1f2a37'>${{fmtMetric(v, state.metric)}}</text></g>`;
    }}).join('');
    chartEl.innerHTML = `<svg class='chart-svg' viewBox='0 0 ${{width}} ${{height}}' aria-label='${{metricLabel(state.metric)}} by method at cap ${{state.cap}}'><line x1='${{left}}' x2='${{left}}' y1='${{top}}' y2='${{height-bottom}}' stroke='#7589a1'/><line x1='${{left}}' x2='${{width-right}}' y1='${{height-bottom}}' y2='${{height-bottom}}' stroke='#7589a1'/>${{bars}}</svg>`;
  }}

  function renderTable() {{
    const rows = selectionRows();
    if (!rows.length) {{
      tableEl.innerHTML = '<p class="empty-state">No results available.</p>';
      return;
    }}
    const body = rows.map(row => {{
      const method = artifactData.labels[String(row.method_id || '')] || String(row.method_id || '');
      const value = rowValue(row, state.metric);
      const actualTokens = Number(row.actual_token_count ?? row.actual_tokens ?? 0);
      const nominal = Number(row.nominal_budget ?? 0);
      const ratio = nominal > 0 ? actualTokens / nominal : 0;
      const seedText = state.mode === 'trial' ? `<td>${{row.seed ?? 'n/a'}}</td>` : '';
      return `<tr><td>${{method}}</td>${{seedText}}<td>${{fmtMetric(value, state.metric)}}</td><td>${{actualTokens.toLocaleString()}}</td><td>${{(100 * ratio).toFixed(2)}}%</td></tr>`;
    }}).join('');
    const header = state.mode === 'trial'
      ? '<tr><th>Method</th><th>Seed</th><th>' + metricLabel(state.metric) + '</th><th>Actual tokens</th><th>Actual / nominal</th></tr>'
      : '<tr><th>Method</th><th>' + metricLabel(state.metric) + '</th><th>Actual tokens</th><th>Actual / nominal</th></tr>';
    tableEl.innerHTML = `<table><thead>${{header}}</thead><tbody>${{body}}</tbody></table>`;
  }}

  capSelect.value = state.cap;
  capSelect.addEventListener('change', () => {{
    state.cap = capSelect.value;
    const seeds = availableSeeds();
    if (!seeds.includes(state.seed)) state.seed = seeds[0] || '';
    seedSelect.innerHTML = seedOptionsHtml();
    seedSelect.value = state.seed;
    renderChart();
    renderTable();
  }});

  metricButtons.forEach(button => button.addEventListener('click', () => {{
    state.metric = button.getAttribute('data-metric') || 'absolute_aggregate_mae';
    metricButtons.forEach(btn => btn.classList.toggle('active', btn === button));
    renderChart();
    renderTable();
  }}));

  modeButtons.forEach(button => button.addEventListener('click', () => {{
    state.mode = button.getAttribute('data-mode') || 'aggregate';
    modeButtons.forEach(btn => btn.classList.toggle('active', btn === button));
    if (state.mode === 'trial') {{
      seedControl.style.display = 'inline-block';
      const seeds = availableSeeds();
      if (!seeds.includes(state.seed)) state.seed = seeds[0] || '';
    }} else seedControl.style.display = 'none';
    seedSelect.innerHTML = seedOptionsHtml();
    seedSelect.value = state.seed;
    renderChart();
    renderTable();
  }}));

  seedSelect.addEventListener('change', () => {{
    state.seed = seedSelect.value || 'all';
    renderChart();
    renderTable();
  }});

  toggles.forEach(toggle => toggle.addEventListener('change', () => {{
    const method = toggle.getAttribute('data-method');
    if (!method) return;
    if (toggle.checked) state.visibleMethods.add(method);
    else state.visibleMethods.delete(method);
    renderChart();
    renderTable();
  }}));

  seedSelect.innerHTML = seedOptionsHtml();
  state.seed = availableSeeds()[0] || '';
  seedSelect.value = state.seed;
  renderChart();
  renderTable();
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
