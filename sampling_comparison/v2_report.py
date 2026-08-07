"""Self-contained HTML report generator for sampling v2 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from statistics import mean
from typing import Any

REPORT_VERSION = "agent365-sampling-v2-report-v1"
DEFAULT_INPUT_DIR = Path("outputs_sampling_v2") / "v2"
DEFAULT_OUTPUT_HTML = DEFAULT_INPUT_DIR / "agent365-sampling-v2-report.html"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required report input not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object at {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required report input not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object line in {path}")
        rows.append(row)
    return rows


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _label(name: str) -> str:
    labels = {
        "census": "Census baseline",
        "random_sampling_stratified": "Native stratified random",
        "adaptive_minhash_32x4": "MinHash LSH 32x4",
        "adaptive_embedding_fullsession": "Full-session embedding",
        "random_online_admission": "Online random admission",
    }
    return labels.get(name, name.replace("_", " ").title())


def _agent_display_name(scoped_agent_id: str) -> str:
    return scoped_agent_id.rsplit("|", 1)[-1]


def _color(name: str) -> str:
    colors = {
        "census": "#5b6573",
        "random_sampling_stratified": "#2878b5",
        "adaptive_minhash_32x4": "#0f7d6c",
        "adaptive_embedding_fullsession": "#c36c1f",
        "random_online_admission": "#7d4a91",
    }
    return colors.get(name, "#667085")


def _tab_button(tab_id: str, label: str, *, selected: bool = False) -> str:
    return (
        f'<button class="tab-button" id="tab-{escape(tab_id)}" role="tab" '
        f'aria-controls="panel-{escape(tab_id)}" aria-selected="{str(selected).lower()}" '
        f'tabindex="{0 if selected else -1}">{escape(label)}</button>'
    )


def _tab_panel(tab_id: str, title: str, content: str, *, selected: bool = False) -> str:
    hidden = "" if selected else " hidden"
    return (
        f'<section class="tab-panel" id="panel-{escape(tab_id)}" role="tabpanel" '
        f'aria-labelledby="tab-{escape(tab_id)}" tabindex="0"{hidden}>'
        f'<header class="section-heading"><h2>{escape(title)}</h2></header>{content}</section>'
    )


def _svg_text_wrap(label: str, *, x: float, y: float, max_chars: int = 14) -> str:
    words = label.split()
    lines: list[str] = []
    curr: list[str] = []
    for word in words:
        cand = " ".join([*curr, word])
        if curr and len(cand) > max_chars:
            lines.append(" ".join(curr))
            curr = [word]
        else:
            curr.append(word)
    if curr:
        lines.append(" ".join(curr))
    if not lines:
        lines = [label]
    start_y = y - (len(lines) - 1) * 7
    tspans = "".join(
        f'<tspan x="{x:.2f}" y="{start_y + i * 14:.2f}">{escape(line)}</tspan>'
        for i, line in enumerate(lines[:3])
    )
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="middle" class="axis">{tspans}</text>'


def _svg_grouped_bar(
    *,
    title: str,
    categories: list[str],
    series: list[dict[str, Any]],
    y_max: float,
    y_label: str,
    fmt: str = "percent",
) -> str:
    width = 1060
    height = 360
    ml = 80
    mr = 30
    mt = 30
    mb = 78
    pw = width - ml - mr
    ph = height - mt - mb
    cat_n = max(1, len(categories))
    ser_n = max(1, len(series))
    slot = pw / cat_n
    bw = max(6.0, min(24.0, slot / (ser_n + 1.4)))

    grid = []
    for t in range(6):
        yv = y_max * t / 5
        y = mt + ph - (ph * (yv / y_max if y_max else 0.0))
        lbl = _pct(yv, 0) if fmt == "percent" else _num(yv, 2)
        grid.append(
            f'<line x1="{ml}" y1="{y:.2f}" x2="{ml + pw}" y2="{y:.2f}" class="grid" />'
            f'<text x="{ml - 8}" y="{y + 4:.2f}" text-anchor="end" class="axis">{escape(lbl)}</text>'
        )

    bars = []
    x_labels = []
    for i, cat in enumerate(categories):
        base = ml + i * slot
        x_labels.append(_svg_text_wrap(cat, x=base + slot / 2, y=height - 26))
        for j, row in enumerate(series):
            val = float(row["values"][i]) if i < len(row["values"]) else 0.0
            h = ph * (val / y_max if y_max else 0.0)
            x = base + (slot - ser_n * bw) / 2 + j * bw
            y = mt + ph - h
            shown = _pct(val, 1) if fmt == "percent" else _num(val, 3)
            bars.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bw - 1:.2f}" height="{h:.2f}" fill="{row["color"]}" '
                f'aria-label="{escape(row["label"])} {escape(cat)} {escape(shown)}" />'
            )

    legend = "".join(
        f'<span><i style="background:{row["color"]}"></i>{escape(row["label"])}</span>'
        for row in series
    )
    return (
        '<figure class="chart"><figcaption>'
        + escape(title)
        + f'</figcaption><div class="chart-legend">{legend}</div><div class="chart-scroll"><svg viewBox="0 0 '
        + str(width)
        + " "
        + str(height)
        + '" role="img" aria-label="'
        + escape(title)
        + '">'
        + "".join(grid)
        + "".join(bars)
        + "".join(x_labels)
        + f'<text x="18" y="{mt + ph / 2:.2f}" transform="rotate(-90 18 {mt + ph / 2:.2f})" class="axis">{escape(y_label)}</text>'
        + "</svg></div></figure>"
    )


def _svg_line(
    *,
    title: str,
    x_values: list[int],
    series: list[dict[str, Any]],
    y_max: float,
    y_label: str,
    fmt: str = "percent",
) -> str:
    width = 980
    height = 340
    ml = 78
    mr = 26
    mt = 28
    mb = 70
    pw = width - ml - mr
    ph = height - mt - mb
    x0 = min(x_values) if x_values else 0
    x1 = max(x_values) if x_values else 1

    def xp(x: int) -> float:
        if x1 == x0:
            return ml + pw / 2
        return ml + (x - x0) / (x1 - x0) * pw

    def yp(y: float) -> float:
        return mt + ph - (ph * (y / y_max if y_max else 0.0))

    grid = []
    for t in range(6):
        yv = y_max * t / 5
        y = yp(yv)
        lbl = _pct(yv, 0) if fmt == "percent" else _num(yv, 3)
        grid.append(
            f'<line x1="{ml}" y1="{y:.2f}" x2="{ml + pw}" y2="{y:.2f}" class="grid" />'
            f'<text x="{ml - 8}" y="{y + 4:.2f}" text-anchor="end" class="axis">{escape(lbl)}</text>'
        )

    xticks = []
    for x in x_values:
        px = xp(x)
        xticks.append(f'<line x1="{px:.2f}" y1="{mt + ph}" x2="{px:.2f}" y2="{mt + ph + 6}" class="axis-stroke" />')
        xticks.append(f'<text x="{px:.2f}" y="{height - 28}" text-anchor="middle" class="axis">b{x}</text>')

    lines = []
    dots = []
    for row in series:
        pts = []
        for i, x in enumerate(x_values):
            val = float(row["values"][i]) if i < len(row["values"]) else 0.0
            pts.append((xp(x), yp(val), val))
        path = " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in pts)
        lines.append(f'<polyline fill="none" stroke="{row["color"]}" stroke-width="2.5" points="{path}" />')
        for px, py, v in pts:
            lab = _pct(v, 1) if fmt == "percent" else _num(v, 3)
            dots.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.5" fill="{row["color"]}" aria-label="{escape(row["label"])} {escape(lab)}" />')

    legend = "".join(
        f'<span><i style="background:{row["color"]}"></i>{escape(row["label"])}</span>'
        for row in series
    )
    return (
        '<figure class="chart"><figcaption>'
        + escape(title)
        + f'</figcaption><div class="chart-legend">{legend}</div><div class="chart-scroll"><svg viewBox="0 0 '
        + str(width)
        + " "
        + str(height)
        + '" role="img" aria-label="'
        + escape(title)
        + '">'
        + "".join(grid)
        + "".join(xticks)
        + "".join(lines)
        + "".join(dots)
        + f'<text x="18" y="{mt + ph / 2:.2f}" transform="rotate(-90 18 {mt + ph / 2:.2f})" class="axis">{escape(y_label)}</text>'
        + f'<text x="{ml + pw / 2:.2f}" y="{height - 8}" text-anchor="middle" class="axis">Budget</text>'
        + "</svg></div></figure>"
    )


def _svg_heatmap(
    *,
    title: str,
    x_labels: list[str],
    y_labels: list[str],
    cells: list[list[float]],
    value_min: float,
    value_max: float,
    scale_name: str,
    fmt: str = "percent",
) -> str:
    width = 980
    height = 340
    ml = 120
    mr = 24
    mt = 32
    mb = 70
    pw = width - ml - mr
    ph = height - mt - mb
    cw = pw / max(1, len(x_labels))
    ch = ph / max(1, len(y_labels))

    def color(v: float) -> str:
        if value_max <= value_min:
            t = 0.5
        else:
            t = max(0.0, min(1.0, (v - value_min) / (value_max - value_min)))
        lo = (243, 245, 247)
        hi = (33, 104, 173)
        r = int(lo[0] + (hi[0] - lo[0]) * t)
        g = int(lo[1] + (hi[1] - lo[1]) * t)
        b = int(lo[2] + (hi[2] - lo[2]) * t)
        return f"rgb({r},{g},{b})"

    rects = []
    for yi, row in enumerate(cells):
        for xi, value in enumerate(row):
            x = ml + xi * cw
            y = mt + yi * ch
            rects.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cw - 1:.2f}" height="{ch - 1:.2f}" fill="{color(value)}" />')
            rects.append(f'<text x="{x + cw / 2:.2f}" y="{y + ch / 2 + 5:.2f}" text-anchor="middle" class="heatmap-value">{escape(_pct(value, 0))}</text>')

    xs = []
    for i, label in enumerate(x_labels):
        xs.append(f'<text x="{ml + i * cw + cw / 2:.2f}" y="{height - 28}" text-anchor="middle" class="axis">{escape(label)}</text>')
    ys = []
    for i, label in enumerate(y_labels):
        ys.append(f'<text x="{ml - 8:.2f}" y="{mt + i * ch + ch / 2 + 4:.2f}" text-anchor="end" class="axis">{escape(label)}</text>')

    return (
        '<figure class="chart"><figcaption>'
        + escape(title)
        + f'</figcaption><div class="chart-scroll"><svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        + "".join(rects)
        + "".join(xs)
        + "".join(ys)
        + f'<text x="{ml + pw / 2:.2f}" y="{height - 6}" text-anchor="middle" class="axis">Evaluator throughput</text>'
        + f'<text x="22" y="{mt + ph / 2:.2f}" transform="rotate(-90 22 {mt + ph / 2:.2f})" class="axis">Arrival rate</text>'
        + f'<text x="{ml}" y="16" class="legend">{escape(scale_name)} scale from {escape(_pct(value_min,0) if fmt == "percent" else _num(value_min,3))} to {escape(_pct(value_max,0) if fmt == "percent" else _num(value_max,3))}</text>'
        + "</svg></div></figure>"
    )


def _architecture_svg(title: str, nodes: list[str], color: str) -> str:
    h = 178
    left = 24
    top = 44
    nw = 155
    nh = 70
    gap = 28
    w = left * 2 + len(nodes) * nw + max(0, len(nodes) - 1) * gap
    marker = "arr-" + "".join(ch if ch.isalnum() else "-" for ch in title.lower())
    elements = []
    for i, label in enumerate(nodes):
        x = left + i * (nw + gap)
        elements.append(f'<rect x="{x}" y="{top}" width="{nw}" height="{nh}" rx="7" ry="7" fill="#ffffff" stroke="{color}" stroke-width="1.8" />')
        parts = label.split("|")
        elements.append(f'<text x="{x + nw / 2}" y="{top + 28}" text-anchor="middle" class="flow-title">{escape(parts[0])}</text>')
        if len(parts) > 1:
            sublabel = parts[1]
            if "/" in sublabel and len(sublabel) > 24:
                first, second = sublabel.split("/", 1)
                elements.append(
                    f'<text x="{x + nw / 2}" y="{top + 44}" text-anchor="middle" class="flow-sub">'
                    f'<tspan x="{x + nw / 2}" y="{top + 44}">{escape(first)} /</tspan>'
                    f'<tspan x="{x + nw / 2}" y="{top + 59}">{escape(second)}</tspan></text>'
                )
            else:
                elements.append(f'<text x="{x + nw / 2}" y="{top + 48}" text-anchor="middle" class="flow-sub">{escape(sublabel)}</text>')
        if i < len(nodes) - 1:
            sx = x + nw
            dx = x + nw + gap
            y = top + nh / 2
            elements.append(f'<line x1="{sx}" y1="{y}" x2="{dx}" y2="{y}" stroke="{color}" stroke-width="1.5" marker-end="url(#{marker})" />')
    return (
        '<figure class="arch-figure"><figcaption>'
        + escape(title)
        + f'</figcaption><div class="chart-scroll"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}"><defs>'
        + f'<marker id="{marker}" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{color}" /></marker>'
        + "</defs>"
        + "".join(elements)
        + "</svg></div></figure>"
    )


@dataclass(frozen=True)
class V2ReportInputs:
    aggregate: Path
    corpus_audit: Path
    quadrant: Path
    throughput: Path
    selected_membership_20pct: Path
    production_storage_manifest: Path
    external_eval_manifest: Path


@dataclass(frozen=True)
class LoadedV2Artifacts:
    aggregate: dict[str, Any]
    corpus_audit: dict[str, Any]
    quadrant: dict[str, Any]
    throughput: dict[str, Any]
    selected_membership_20pct: dict[str, Any]
    production_storage_manifest: dict[str, Any]
    external_eval_manifest: dict[str, Any]
    external_eval_method_rows: dict[str, list[dict[str, Any]]]
    source_paths: dict[str, Path]


def default_inputs(base_dir: Path = DEFAULT_INPUT_DIR) -> V2ReportInputs:
    return V2ReportInputs(
        aggregate=base_dir / "aggregate.json",
        corpus_audit=base_dir / "corpus_audit.json",
        quadrant=base_dir / "quadrant.json",
        throughput=base_dir / "throughput.json",
        selected_membership_20pct=base_dir / "selected_membership_20pct.json",
        production_storage_manifest=base_dir / "production_storage_manifest.json",
        external_eval_manifest=base_dir / "external_eval_snapshots" / "manifest.json",
    )


def load_v2_artifacts(inputs: V2ReportInputs | None = None) -> LoadedV2Artifacts:
    paths = inputs or default_inputs()
    aggregate = _read_json(paths.aggregate)
    corpus_audit = _read_json(paths.corpus_audit)
    quadrant = _read_json(paths.quadrant)
    throughput = _read_json(paths.throughput)
    membership = _read_json(paths.selected_membership_20pct)
    storage = _read_json(paths.production_storage_manifest)
    snapshot_manifest = _read_json(paths.external_eval_manifest)

    methods_files = snapshot_manifest.get("methods_files")
    if not isinstance(methods_files, dict) or not methods_files:
        raise ValueError("external_eval_snapshots manifest missing methods_files")

    method_rows: dict[str, list[dict[str, Any]]] = {}
    for method, row in sorted(methods_files.items()):
        if not isinstance(row, dict):
            raise ValueError("methods_files entries must be objects")
        path = Path(str(row.get("path") or ""))
        if not path.is_file():
            raise FileNotFoundError(f"snapshot method jsonl not found: {path}")
        method_rows[method] = _read_jsonl(path)

    source_paths = {
        "aggregate": paths.aggregate,
        "corpus_audit": paths.corpus_audit,
        "quadrant": paths.quadrant,
        "throughput": paths.throughput,
        "selected_membership_20pct": paths.selected_membership_20pct,
        "production_storage_manifest": paths.production_storage_manifest,
        "external_eval_manifest": paths.external_eval_manifest,
    }
    for method, row in sorted(methods_files.items()):
        source_paths[f"external_eval_{method}"] = Path(str(row.get("path") or ""))

    return LoadedV2Artifacts(
        aggregate=aggregate,
        corpus_audit=corpus_audit,
        quadrant=quadrant,
        throughput=throughput,
        selected_membership_20pct=membership,
        production_storage_manifest=storage,
        external_eval_manifest=snapshot_manifest,
        external_eval_method_rows=method_rows,
        source_paths=source_paths,
    )


def validate_v2_artifacts(artifacts: LoadedV2Artifacts) -> None:
    if int(artifacts.aggregate.get("population_count") or 0) != 2800:
        raise ValueError("aggregate population_count must be 2800")

    source_files = artifacts.corpus_audit.get("source_files") or {}
    h_count = int((((source_files.get("historical_300") or {}).get("counts") or {}).get("units") or 0))
    d_count = int((((source_files.get("dense_2500") or {}).get("counts") or {}).get("units") or 0))
    if h_count != 300 or d_count != 2500:
        raise ValueError("corpus source counts must be historical_300=300 and dense_2500=2500")

    expected_methods = {
        "random_sampling_stratified",
        "adaptive_minhash_32x4",
        "adaptive_embedding_fullsession",
    }
    outcome_means = (artifacts.aggregate.get("outcome") or {}).get("aggregate_means") or {}
    if set(outcome_means.keys()) != expected_methods:
        raise ValueError("aggregate outcome methods do not match expected methods")

    expected_budgets = {"5", "10", "20", "30", "50"}
    for method in expected_methods:
        budgets = set(((outcome_means.get(method) or {}).keys()))
        if budgets != expected_budgets:
            raise ValueError(f"aggregate outcome budgets for {method} are incomplete")

    q_total = int(((artifacts.quadrant.get("quadrants") or {}).get("counts") or {}).get("total_units") or 0)
    if q_total != 2800:
        raise ValueError("quadrant assignment total must be 2800")

    tp_cfg = artifacts.throughput.get("config") or {}
    arrivals = tuple(tp_cfg.get("arrival_rates") or ())
    evals = tuple(tp_cfg.get("eval_throughputs") or ())
    if len(arrivals) != 4 or len(evals) != 4:
        raise ValueError("throughput config must include 4 arrival rates and 4 evaluator throughputs")

    if not artifacts.production_storage_manifest:
        raise ValueError("production storage manifest is required")

    if not artifacts.external_eval_manifest:
        raise ValueError("external eval manifest is required")


def _build_metrics_summaries(artifacts: LoadedV2Artifacts) -> dict[str, Any]:
    out_rows = (artifacts.aggregate.get("outcome") or {}).get("aggregate_means") or {}
    budgets = [5, 10, 20, 30, 50]
    methods = ["random_sampling_stratified", "adaptive_minhash_32x4", "adaptive_embedding_fullsession"]

    mae_by_method = {
        method: [float(((out_rows.get(method) or {}).get(str(b)) or {}).get("mean_absolute_error") or 0.0) for b in budgets]
        for method in methods
    }
    saved_by_method = {
        method: [float(((out_rows.get(method) or {}).get(str(b)) or {}).get("mean_fraction_saved") or 0.0) for b in budgets]
        for method in methods
    }
    cov_by_method = {
        method: [float(((out_rows.get(method) or {}).get(str(b)) or {}).get("mean_concept_coverage") or 0.0) for b in budgets]
        for method in methods
    }

    census_rate = float((((artifacts.aggregate.get("outcome") or {}).get("census_baseline") or {}).get("selected_pass_rate") or 0.0))

    diag_rows = (artifacts.aggregate.get("outcome") or {}).get("aggregate_budget_diagnostics") or {}
    source_keep: dict[str, dict[str, list[float]]] = {}
    for row in diag_rows.values():
        if not isinstance(row, dict):
            continue
        method = str(row.get("method") or "")
        per_corpus = row.get("per_corpus") or {}
        slot = source_keep.setdefault(method, {"historical_300": [], "dense_2500": []})
        for corpus_id in ("historical_300", "dense_2500"):
            slot[corpus_id].append(float(((per_corpus.get(corpus_id) or {}).get("mean_keep_rate") or 0.0)))

    return {
        "budgets": budgets,
        "methods": methods,
        "mae_by_method": mae_by_method,
        "saved_by_method": saved_by_method,
        "cov_by_method": cov_by_method,
        "census_rate": census_rate,
        "source_keep": source_keep,
    }


def _render_overview(artifacts: LoadedV2Artifacts, generated_at: str) -> str:
    summary = _build_metrics_summaries(artifacts)
    method20 = {}
    for method in summary["methods"]:
        slot = (((artifacts.aggregate.get("outcome") or {}).get("aggregate_means") or {}).get(method) or {}).get("20") or {}
        method20[method] = {
            "mae": float(slot.get("mean_absolute_error") or 0.0),
            "saved": float(slot.get("mean_fraction_saved") or 0.0),
            "cov": float(slot.get("mean_concept_coverage") or 0.0),
        }

    cards = []
    cards.append(
        "<article><h3>Question asked</h3><p>Compare census, native random, MinHash LSH 32x4, and full-session embedding on a combined 2,800-session corpus across two source blocks using expected-label-only scoring and no LLM calls.</p></article>"
    )
    cards.append(
        "<article><h3>Population</h3><p>2,800 sessions across 105 tenant|agent scoped identities from historical_300 and dense_2500 sources. Methods are compared under paired repetitions and controlled replay for mechanism analyses.</p></article>"
    )
    cards.append(
        "<article><h3>Core findings from persisted data</h3><p>At 20% nominal budget, all non-census methods save substantial evaluator load while retaining concept coverage, and source-specific keep rates can differ from nominal budget due to adaptive admission pressure and stratum allocation effects.</p></article>"
    )
    cards.append(
        "<article><h3>Caveats</h3><p>Adaptive caps may undershoot realized keep rate targets under backpressure and deterministic novelty decisions; source block keep rates differ, so budget diagnostics must be interpreted per source as well as combined.</p></article>"
    )

    rows = "".join(
        "<tr>"
        f"<td>{escape(_label(method))}</td>"
        f"<td>{escape(_num(method20[method]['mae'], 4))}</td>"
        f"<td>{escape(_pct(method20[method]['saved'], 1))}</td>"
        f"<td>{escape(_pct(method20[method]['cov'], 1))}</td>"
        "</tr>"
        for method in summary["methods"]
    )

    return (
        f"<p class=\"small\">Report generated at {escape(generated_at)} from persisted artifacts only. No experiment execution occurs during report generation.</p>"
        + f"<div class=\"grid-4\">{''.join(cards)}</div>"
        + "<h3>Concise persisted outcomes at 20% nominal budget</h3>"
        + "<table><thead><tr><th>Method</th><th>Mean MAE vs census</th><th>Mean fraction saved</th><th>Mean concept coverage</th></tr></thead>"
        + f"<tbody>{rows}</tbody></table>"
    )


def _render_input_data(artifacts: LoadedV2Artifacts) -> str:
    combined = (artifacts.corpus_audit.get("combined") or {}).get("counts") or {}
    sources = artifacts.corpus_audit.get("source_files") or {}

    sample = {
        "unit_id": "historical_300:example-unit-001",
        "tenant_id": "tenant-a",
        "agent_id": "agent-42",
        "conversation_id": "conversation-guid-redacted-shape",
        "started_at_utc": "2026-07-30T09:30:00Z",
        "ended_at_utc": "2026-07-30T09:34:12Z",
        "tool_calls": [
            {"name": "search_docs", "status": "ok"},
            {"name": "fetch_record", "status": "ok"},
        ],
        "had_error": False,
        "metadata": {
            "corpus_id": "historical_300",
            "domain": "identity",
            "task": "permissions-troubleshoot",
            "difficulty": "moderate",
        },
    }

    source_rows = []
    for corpus_id in ("historical_300", "dense_2500"):
        row = sources.get(corpus_id) or {}
        counts = row.get("counts") or {}
        source_rows.append(
            "<tr>"
            f"<td>{escape(corpus_id)}</td>"
            f"<td>{int(counts.get('units') or 0)}</td>"
            f"<td>{int(counts.get('agents') or 0)}</td>"
            f"<td>{escape(_pct(float(row.get('label_pass_rate') or 0.0), 2))}</td>"
            f"<td>{int(counts.get('concepts') or 0)}</td>"
            "</tr>"
        )

    unit_counts = [int(((sources.get("historical_300") or {}).get("counts") or {}).get("units") or 0), int(((sources.get("dense_2500") or {}).get("counts") or {}).get("units") or 0)]
    agent_counts = [int(((sources.get("historical_300") or {}).get("counts") or {}).get("agents") or 0), int(((sources.get("dense_2500") or {}).get("counts") or {}).get("agents") or 0)]

    by_agent = {}
    for row in ((artifacts.aggregate.get("outcome") or {}).get("aggregate_per_agent") or {}).values():
        if not isinstance(row, dict):
            continue
        if str(row.get("method")) != "census" or int(row.get("budget_pct") or -1) != 100:
            continue
        aid = str(row.get("agent_id") or "")
        by_agent[aid] = float(row.get("mean_sampled_count") or 0.0)
    bins = [0, 0, 0, 0, 0]
    for value in by_agent.values():
        if value <= 5:
            bins[0] += 1
        elif value <= 15:
            bins[1] += 1
        elif value <= 30:
            bins[2] += 1
        elif value <= 60:
            bins[3] += 1
        else:
            bins[4] += 1

    chart_sources = _svg_grouped_bar(
        title="Source distribution and agent counts by source",
        categories=["historical_300", "dense_2500"],
        series=[
            {"label": "Sessions", "color": "#2878b5", "values": [unit_counts[0] / 2800.0, unit_counts[1] / 2800.0]},
            {"label": "Agents", "color": "#0f7d6c", "values": [agent_counts[0] / max(combined.get("agents", 1), 1), agent_counts[1] / max(combined.get("agents", 1), 1)]},
        ],
        y_max=1.0,
        y_label="Share",
        fmt="percent",
    )

    chart_hist = _svg_grouped_bar(
        title="Sessions per agent distribution (census counts binned)",
        categories=["1-5", "6-15", "16-30", "31-60", "61+"],
        series=[{"label": "Agent count share", "color": "#c36c1f", "values": [b / max(len(by_agent), 1) for b in bins]}],
        y_max=1.0,
        y_label="Share of agents",
        fmt="percent",
    )

    return (
        "<p>This tab shows normalized input shape, not raw corpus dumps. Example JSON below is schema-aligned and sanitized for readability.</p>"
        + f"<pre>{escape(json.dumps(sample, indent=2, sort_keys=True))}</pre>"
        + "<h3>Source block summary</h3>"
        + "<table><thead><tr><th>Source</th><th>Sessions</th><th>Scoped agents</th><th>Expected-label pass rate</th><th>Distinct concepts</th></tr></thead>"
        + f"<tbody>{''.join(source_rows)}</tbody></table>"
        + chart_sources
        + chart_hist
        + "<p class=\"small\"><strong>ID rule:</strong> experiment unit IDs are source-prefixed (for example <code>historical_300:...</code>, <code>dense_2500:...</code>) to prevent collisions. External snapshot output maps back to original conversation IDs for API payloads.</p>"
    )


def _render_metrics(artifacts: LoadedV2Artifacts) -> str:
    return (
        "<h3>Exact definitions</h3>"
        "<p><strong>Unweighted selected-rate MAE vs census:</strong> for each repetition and method-budget pair, compute absolute error between selected expected-label pass rate and combined census expected-label pass rate. Report mean across paired repetitions.</p>"
        "<pre>MAE(m,b) = mean_r | selected_pass_rate(m,b,r) - census_pass_rate(r) |</pre>"
        "<p><strong>Fraction saved:</strong> <code>1 - selected_count / population_count</code> per run, mean over paired repetitions.</p>"
        "<p><strong>Concept coverage:</strong> concept key is <code>corpus|domain|task|difficulty</code> (source namespaced). Coverage per run is <code>distinct selected concepts / eligible concepts</code>, mean over paired repetitions.</p>"
        "<h3>Interpretation boundaries</h3>"
        "<p>Coverage can rise while sample size falls when methods preserve rare or diverse concept strata while discarding frequent redundant traces.</p>"
        "<p>Random inference caveat: the native random arm is scored by observed sample outcome in this persisted bundle; low-budget zero-allocation strata can prevent a fully design-valid full-population estimator without additional weighting support.</p>"
        "<p>Adaptive selections are nonprobability mechanisms; MAE and coverage are empirical diagnostics, not unbiasedness guarantees.</p>"
    )


def _render_outcomes(artifacts: LoadedV2Artifacts) -> str:
    s = _build_metrics_summaries(artifacts)
    budgets = s["budgets"]
    methods = s["methods"]

    mae_chart = _svg_line(
        title="Mean MAE vs census across budgets",
        x_values=budgets,
        series=[{"label": _label(m), "color": _color(m), "values": s["mae_by_method"][m]} for m in methods],
        y_max=max(0.05, max(max(v) for v in s["mae_by_method"].values()) * 1.2),
        y_label="MAE",
        fmt="number",
    )
    saved_chart = _svg_line(
        title="Mean fraction saved across budgets",
        x_values=budgets,
        series=[{"label": _label(m), "color": _color(m), "values": s["saved_by_method"][m]} for m in methods],
        y_max=1.0,
        y_label="Fraction saved",
        fmt="percent",
    )
    cov_chart = _svg_line(
        title="Mean concept coverage across budgets",
        x_values=budgets,
        series=[{"label": _label(m), "color": _color(m), "values": s["cov_by_method"][m]} for m in methods],
        y_max=1.0,
        y_label="Concept coverage",
        fmt="percent",
    )

    rows = []
    out_rows = (artifacts.aggregate.get("outcome") or {}).get("aggregate_means") or {}
    for b in budgets:
        for m in methods:
            slot = ((out_rows.get(m) or {}).get(str(b)) or {})
            rows.append(
                "<tr>"
                f"<td>b{b}</td>"
                f"<td>{escape(_label(m))}</td>"
                f"<td>{escape(_num(float(slot.get('mean_absolute_error') or 0.0), 4))}</td>"
                f"<td>{escape(_pct(float(slot.get('mean_fraction_saved') or 0.0), 1))}</td>"
                f"<td>{escape(_pct(float(slot.get('mean_concept_coverage') or 0.0), 1))}</td>"
                "</tr>"
            )

    diag = (artifacts.aggregate.get("outcome") or {}).get("aggregate_budget_diagnostics") or {}
    diag_rows = []
    for key in sorted(diag):
        row = diag[key]
        if not isinstance(row, dict):
            continue
        m = str(row.get("method") or "")
        b = int(row.get("budget_pct") or 0)
        nominal = float(row.get("nominal_keep_rate") or 0.0)
        realized = float(row.get("realized_keep_rate_mean") or 0.0)
        d = float(row.get("deviation_from_nominal_pp") or 0.0)
        per_c = row.get("per_corpus") or {}
        h = float(((per_c.get("historical_300") or {}).get("mean_keep_rate") or 0.0))
        den = float(((per_c.get("dense_2500") or {}).get("mean_keep_rate") or 0.0))
        diag_rows.append(
            "<tr>"
            f"<td>{escape(_label(m))}</td>"
            f"<td>b{b}</td>"
            f"<td>{escape(_pct(nominal, 1))}</td>"
            f"<td>{escape(_pct(realized, 1))}</td>"
            f"<td>{escape(_num(d, 2))}</td>"
            f"<td>{escape(_pct(h, 1))}</td>"
            f"<td>{escape(_pct(den, 1))}</td>"
            "</tr>"
        )

    return (
        "<p>Outcomes compare all required budgets 5/10/20/30/50 from persisted aggregate means and diagnostics.</p>"
        + mae_chart
        + saved_chart
        + cov_chart
        + "<h3>Compact exact table</h3>"
        + "<table><thead><tr><th>Budget</th><th>Method</th><th>Mean MAE</th><th>Mean fraction saved</th><th>Mean concept coverage</th></tr></thead>"
        + f"<tbody>{''.join(rows)}</tbody></table>"
        + f"<p><strong>Census baseline pass rate:</strong> {escape(_pct(s['census_rate'], 2))} (reference denominator only).</p>"
        + "<h3>Nominal vs realized budget and source-specific keep rates</h3>"
        + "<table><thead><tr><th>Method</th><th>Budget</th><th>Nominal keep</th><th>Realized keep mean</th><th>Deviation pp</th><th>historical_300 keep</th><th>dense_2500 keep</th></tr></thead>"
        + f"<tbody>{''.join(diag_rows)}</tbody></table>"
    )


def _render_agents(artifacts: LoadedV2Artifacts) -> str:
    per_agent = (artifacts.aggregate.get("outcome") or {}).get("aggregate_per_agent") or {}
    budget20: dict[str, dict[str, dict[str, float]]] = {}
    for row in per_agent.values():
        if not isinstance(row, dict):
            continue
        if int(row.get("budget_pct") or -1) not in {20, 100}:
            continue
        aid = str(row.get("agent_id") or "")
        method = str(row.get("method") or "")
        budget20.setdefault(aid, {})[method] = {
            "sampled": float(row.get("mean_sampled_count") or 0.0),
            "population": float(row.get("mean_sampled_count") or 0.0) if method == "census" else 0.0,
            "abs_err": float(row.get("mean_absolute_error") or 0.0),
        }

    # backfill population from census
    for aid, rows in budget20.items():
        pop = float((rows.get("census") or {}).get("sampled") or 0.0)
        for method in ("random_sampling_stratified", "adaptive_minhash_32x4", "adaptive_embedding_fullsession"):
            rows.setdefault(method, {"sampled": 0.0, "population": pop, "abs_err": None})
            rows[method]["population"] = pop

    dense5 = sorted(budget20.items(), key=lambda kv: (-(kv[1].get("census") or {}).get("sampled", 0.0), kv[0]))[:5]
    dense_categories = [_agent_display_name(k) for k, _ in dense5]
    dense_chart = _svg_grouped_bar(
        title="Top 5 agents by total sessions: sampled counts at 20% by method",
        categories=dense_categories,
        series=[
            {
                "label": "Population (census)",
                "color": "#5b6573",
                "values": [v.get("census", {}).get("sampled", 0.0) / max(v.get("census", {}).get("sampled", 1.0), 1.0) for _, v in dense5],
            },
            {
                "label": "Random sampled / population",
                "color": _color("random_sampling_stratified"),
                "values": [v.get("random_sampling_stratified", {}).get("sampled", 0.0) / max(v.get("census", {}).get("sampled", 1.0), 1.0) for _, v in dense5],
            },
            {
                "label": "MinHash sampled / population",
                "color": _color("adaptive_minhash_32x4"),
                "values": [v.get("adaptive_minhash_32x4", {}).get("sampled", 0.0) / max(v.get("census", {}).get("sampled", 1.0), 1.0) for _, v in dense5],
            },
            {
                "label": "Embedding sampled / population",
                "color": _color("adaptive_embedding_fullsession"),
                "values": [v.get("adaptive_embedding_fullsession", {}).get("sampled", 0.0) / max(v.get("census", {}).get("sampled", 1.0), 1.0) for _, v in dense5],
            },
        ],
        y_max=1.0,
        y_label="Share of agent population",
        fmt="percent",
    )

    ratios = []
    for aid, rows in budget20.items():
        pop = max(float((rows.get("census") or {}).get("sampled") or 0.0), 1.0)
        ratios.append(
            {
                "agent_id": aid,
                "population": pop,
                "random": float((rows.get("random_sampling_stratified") or {}).get("sampled") or 0.0) / pop,
                "minhash": float((rows.get("adaptive_minhash_32x4") or {}).get("sampled") or 0.0) / pop,
                "embedding": float((rows.get("adaptive_embedding_fullsession") or {}).get("sampled") or 0.0) / pop,
                "random_abs_err": (rows.get("random_sampling_stratified") or {}).get("abs_err"),
                "minhash_abs_err": (rows.get("adaptive_minhash_32x4") or {}).get("abs_err"),
                "embedding_abs_err": (rows.get("adaptive_embedding_fullsession") or {}).get("abs_err"),
            }
        )

    # sparse 100-agent distribution as bins
    bins = {"0-10": 0, "11-20": 0, "21-40": 0, "41-80": 0, "81+": 0}
    for row in ratios:
        p = int(row["population"])
        if p <= 10:
            bins["0-10"] += 1
        elif p <= 20:
            bins["11-20"] += 1
        elif p <= 40:
            bins["21-40"] += 1
        elif p <= 80:
            bins["41-80"] += 1
        else:
            bins["81+"] += 1

    sparse_hist = _svg_grouped_bar(
        title="Agent population distribution across all scoped agents",
        categories=list(bins.keys()),
        series=[{"label": "Agent share", "color": "#2878b5", "values": [v / max(len(ratios), 1) for v in bins.values()]}],
        y_max=1.0,
        y_label="Share of agents",
        fmt="percent",
    )

    sortable_rows = sorted(ratios, key=lambda row: (-row["population"], row["agent_id"]))[:35]
    table_rows = "".join(
        "<tr>"
        f"<td>{escape(row['agent_id'])}</td>"
        f"<td>{int(row['population'])}</td>"
        f"<td>{escape(_pct(row['random'], 1))}</td>"
        f"<td>{escape(_pct(row['minhash'], 1))}</td>"
        f"<td>{escape(_pct(row['embedding'], 1))}</td>"
        f"<td>{escape(_num(row['random_abs_err'], 4) if row['random_abs_err'] is not None else 'N/A')}</td>"
        f"<td>{escape(_num(row['minhash_abs_err'], 4) if row['minhash_abs_err'] is not None else 'N/A')}</td>"
        f"<td>{escape(_num(row['embedding_abs_err'], 4) if row['embedding_abs_err'] is not None else 'N/A')}</td>"
        "</tr>"
        for row in sortable_rows
    )

    return (
        "<p>Agent view at 20% budget avoids unreadable 105-label charts by combining dense top-agent bars, sparse distribution histograms, and a scrollable sortable table.</p>"
        + dense_chart
        + sparse_hist
        + "<div class=\"table-scroll\"><table class=\"sortable\" id=\"agent-table\"><thead><tr><th data-sort=\"text\">Agent</th><th data-sort=\"num\">Population</th><th data-sort=\"num\">Random keep</th><th data-sort=\"num\">MinHash keep</th><th data-sort=\"num\">Embedding keep</th><th data-sort=\"num\">Random pass-rate error</th><th data-sort=\"num\">MinHash pass-rate error</th><th data-sort=\"num\">Embedding pass-rate error</th></tr></thead>"
        + f"<tbody>{table_rows}</tbody></table></div>"
    )


def _render_quadrants(artifacts: LoadedV2Artifacts) -> str:
    quadrants = ((artifacts.quadrant.get("quadrants") or {}).get("quadrant_summary") or {})
    axis = ((artifacts.quadrant.get("quadrants") or {}).get("axis_summary_by_corpus") or {})
    groups = artifacts.quadrant.get("aggregate_groups") or {}

    q_rows = []
    for q in (
        "high_variety_high_velocity",
        "high_variety_low_velocity",
        "low_variety_high_velocity",
        "low_variety_low_velocity",
    ):
        row = quadrants.get(q) or {}
        cc = row.get("corpus_counts") or {}
        q_rows.append(
            "<tr>"
            f"<td>{escape(q)}</td>"
            f"<td>{int(row.get('unit_count') or 0)}</td>"
            f"<td>{int(row.get('agent_count') or 0)}</td>"
            f"<td>{int(cc.get('historical_300') or 0)}</td>"
            f"<td>{int(cc.get('dense_2500') or 0)}</td>"
            "</tr>"
        )

    def pick(method: str, quadrant_name: str, budget: int, field: str) -> float:
        key = f"{method}|{quadrant_name}|b{budget}"
        row = groups.get(key) or {}
        return float(row.get(field) or 0.0)

    cats = [
        "HV-HV b15", "HV-HV b30", "HV-LV b15", "HV-LV b30", "LV-HV b15", "LV-HV b30", "LV-LV b15", "LV-LV b30",
    ]
    qk = [
        "high_variety_high_velocity",
        "high_variety_high_velocity",
        "high_variety_low_velocity",
        "high_variety_low_velocity",
        "low_variety_high_velocity",
        "low_variety_high_velocity",
        "low_variety_low_velocity",
        "low_variety_low_velocity",
    ]
    bk = [15, 30, 15, 30, 15, 30, 15, 30]

    rep_chart = _svg_grouped_bar(
        title="Quadrant representation by method",
        categories=cats,
        series=[
            {"label": "Random online admission", "color": _color("random_online_admission"), "values": [pick("random_online_admission", qk[i], bk[i], "representation_mean") for i in range(len(cats))]},
            {"label": "MinHash LSH 32x4", "color": _color("adaptive_minhash_32x4"), "values": [pick("adaptive_minhash_32x4", qk[i], bk[i], "representation_mean") for i in range(len(cats))]},
            {"label": "Full-session embedding", "color": _color("adaptive_embedding_fullsession"), "values": [pick("adaptive_embedding_fullsession", qk[i], bk[i], "representation_mean") for i in range(len(cats))]},
        ],
        y_max=1.0,
        y_label="Representation",
        fmt="percent",
    )

    util_chart = _svg_grouped_bar(
        title="Quadrant budget utilization by method",
        categories=cats,
        series=[
            {"label": "Random online admission", "color": _color("random_online_admission"), "values": [pick("random_online_admission", qk[i], bk[i], "budget_utilization_mean") for i in range(len(cats))]},
            {"label": "MinHash LSH 32x4", "color": _color("adaptive_minhash_32x4"), "values": [pick("adaptive_minhash_32x4", qk[i], bk[i], "budget_utilization_mean") for i in range(len(cats))]},
            {"label": "Full-session embedding", "color": _color("adaptive_embedding_fullsession"), "values": [pick("adaptive_embedding_fullsession", qk[i], bk[i], "budget_utilization_mean") for i in range(len(cats))]},
        ],
        y_max=1.1,
        y_label="Utilization",
        fmt="percent",
    )

    starv_chart = _svg_grouped_bar(
        title="Quadrant zero-selection agent rate by method",
        categories=cats,
        series=[
            {"label": "Random online admission", "color": _color("random_online_admission"), "values": [pick("random_online_admission", qk[i], bk[i], "zero_selection_agent_rate_mean") for i in range(len(cats))]},
            {"label": "MinHash LSH 32x4", "color": _color("adaptive_minhash_32x4"), "values": [pick("adaptive_minhash_32x4", qk[i], bk[i], "zero_selection_agent_rate_mean") for i in range(len(cats))]},
            {"label": "Full-session embedding", "color": _color("adaptive_embedding_fullsession"), "values": [pick("adaptive_embedding_fullsession", qk[i], bk[i], "zero_selection_agent_rate_mean") for i in range(len(cats))]},
        ],
        y_max=1.0,
        y_label="Zero-selection agent rate",
        fmt="percent",
    )

    return (
        "<p>This section tests sampling mechanism behavior on actual 2,800-session quadrant assignments, not outcome quality. Variety is inverse concept frequency and velocity is inverse inter-arrival time, with source-stratified deterministic rank splits and stable tie handling.</p>"
        + "<h3>Quadrant/source counts</h3>"
        + "<table><thead><tr><th>Quadrant</th><th>Sessions</th><th>Agents</th><th>historical_300</th><th>dense_2500</th></tr></thead>"
        + f"<tbody>{''.join(q_rows)}</tbody></table>"
        + "<p class=\"small\">Axis split diagnostics: "
        + escape(json.dumps(axis, sort_keys=True))
        + "</p>"
        + rep_chart
        + util_chart
        + starv_chart
    )


def _render_throughput(artifacts: LoadedV2Artifacts) -> str:
    cfg = artifacts.throughput.get("config") or {}
    arrivals = [float(x) for x in (cfg.get("arrival_rates") or [])]
    evals = [float(x) for x in (cfg.get("eval_throughputs") or [])]
    budgets = [int(x) for x in (cfg.get("budgets") or [])]
    grid = artifacts.throughput.get("aggregate_grid") or {}

    def cell(method: str, arrival: float, eval_tps: float, budget: int, field: str) -> float:
        key = f"{method}|a{arrival}|e{eval_tps}|b{budget}"
        row = grid.get(key) or {}
        return float(row.get(field) or 0.0)

    def heat(method: str, budget: int, field: str, title: str) -> str:
        vals = [[cell(method, a, e, budget, field) for e in evals] for a in arrivals]
        vmax = max([max(row) if row else 0.0 for row in vals] + [0.01])
        return _svg_heatmap(
            title=title,
            x_labels=[f"{x:g}/s" for x in evals],
            y_labels=[f"{a:g}/s" for a in arrivals],
            cells=vals,
            value_min=0.0,
            value_max=max(1.0, vmax) if field in {"representation_mean", "budget_utilization_mean", "zero_selection_agent_rate_mean"} else vmax,
            scale_name=field,
        )

    sections = []
    for budget in budgets:
        sections.append(f"<h3>Budget {budget}%</h3>")
        sections.append(heat("adaptive_minhash_32x4", budget, "representation_mean", f"MinHash representation heatmap b{budget}"))
        sections.append(heat("adaptive_embedding_fullsession", budget, "representation_mean", f"Embedding representation heatmap b{budget}"))
        sections.append(heat("random_online_admission", budget, "budget_utilization_mean", f"Random utilization heatmap b{budget}"))
        sections.append(heat("adaptive_minhash_32x4", budget, "zero_selection_agent_rate_mean", f"MinHash starvation heatmap b{budget}"))
        sections.append(heat("adaptive_embedding_fullsession", budget, "decision_latency_p95_mean", f"Embedding p95 decision latency heatmap b{budget}"))

    return (
        "<p>Throughput experiment is a controlled replay over the actual high-variety subset with paired content, 4 arrival rates x 4 evaluator throughputs, budgets 15 and 30, backpressure enabled, and no anti-starvation keep-one override.</p>"
        "<p>This evaluates admission pressure mechanics, not LLM execution latency and not Azure service latency.</p>"
        + "".join(sections)
    )


def _render_methods(artifacts: LoadedV2Artifacts) -> str:
    memberships = artifacts.selected_membership_20pct.get("methods") or {}

    def selected(method: str) -> int:
        return int((memberships.get(method) or {}).get("selected_count") or 0)

    d1 = _architecture_svg(
        "Census architecture",
        ["Input stream|all sessions", "Collector|append all", "Scoring|expected labels", "Output|full population"],
        "#5b6573",
    )
    d2 = _architecture_svg(
        "Native stratified random architecture",
        ["Input stream|session metadata", "Stratum allocator|N_h to n_h", "Random draw|without replacement", "Output|sample+weights"],
        "#2878b5",
    )
    d3 = _architecture_svg(
        "MinHash LSH 32x4 architecture",
        ["Input stream|session signature", "MinHash 128|permutations", "LSH bands|32x4 buckets", "Admission|novel leaders"],
        "#0f7d6c",
    )
    d4 = _architecture_svg(
        "Full-session embedding architecture",
        ["Input stream|full packet", "Deterministic embed|offline profile", "Vector recall|leader set", "Admission|novelty threshold"],
        "#c36c1f",
    )

    return f"""
        <p>These walkthroughs describe the production-backed prototype code used by the V2 harness. Expected labels remain outside every selector and are joined only after membership is fixed.</p>
        <section class="prototype-controls"><h3>How the prototype experiment controlled the comparison</h3>
            <ol class="prototype-steps">
                <li><strong>Build one label-blind frame.</strong><br>Normalize 2,800 sessions, namespace IDs by source, and keep labels in a separate scoring map.</li>
                <li><strong>Precompute representation once.</strong><br>With seed 13, build one MinHash signature and one deterministic full-session vector per session.</li>
                <li><strong>Replay paired cells.</strong><br>Adaptive arms share order and replay clock; every cell starts with empty sampler and leader state.</li>
                <li><strong>Score only after selection.</strong><br>Join labels only after IDs are fixed, then calculate MAE, fraction saved, and concept coverage.</li>
            </ol>
            <p class="small"><strong>Comparability policy:</strong> forced anti-starvation keeps are disabled and <code>agent_floor=0</code>. Budgets are upper caps; no post-hoc fill is applied.</p>
        </section>
        <div class="method-detail-grid">
            <article class="method-detail" style="--method-color:#5b6573"><h3>Census: exact reference path</h3>
                <ol><li>Select every eligible source-prefixed session ID.</li><li>Apply no random draw, novelty index, threshold, or backpressure decision.</li><li>Read expected labels after selection and compute the exact population reference.</li><li>Group output by tenant, agent, and UTC day.</li></ol>
                <dl class="method-facts"><dt>State</dt><dd>Full membership set</dd><dt>Representative artifact</dt><dd>{selected('census'):,} selected</dd><dt>Role</dt><dd>Calibration and audit baseline</dd></dl>
            </article>
            <article class="method-detail" style="--method-color:#2878b5"><h3>Native stratified random: exact budgeted draw</h3>
                <ol><li>Convert budget to an exact target.</li><li>Allocate by tenant and agent, then form <code>turn-count band | channel</code> strata.</li><li>Use capped Hamilton allocation for integer <code>n_h</code> values.</li><li>Seed each agent RNG and draw without replacement.</li><li>Record <code>N_h</code>, <code>n_h</code>, inclusion probability, and weight before label scoring.</li></ol>
                <p><strong>Streaming variant:</strong> pressure tests use online random admission with a remaining-needed probability multiplied by backpressure.</p>
                <dl class="method-facts"><dt>State</dt><dd>Seed, allocations, strata</dd><dt>Representative artifact</dt><dd>{selected('random_sampling_stratified'):,} selected</dd><dt>Limitation</dt><dd>Low-budget strata can receive zero allocation</dd></dl>
            </article>
            <article class="method-detail" style="--method-color:#0f7d6c"><h3>MinHash LSH 32x4: lexical novelty path</h3>
                <ol><li>Canonicalize evidence and build bounded 3-token shingles.</li><li>Create a deterministic 128-value MinHash signature.</li><li>Split into 32 bands of four rows and retrieve same-agent leaders sharing buckets.</li><li>Re-rank candidates with full signature similarity; join or create an immutable leader.</li><li>Feed novelty and cadence rarity into the shared adaptive sampler.</li></ol>
                <p>Each cell receives fresh LSH state. Candidate lookup avoids exhaustive leader scans when buckets are selective.</p>
                <dl class="method-facts"><dt>Representation</dt><dd>128 values, 32 x 4</dd><dt>State</dt><dd>Signatures, buckets, timestamps, hits</dd><dt>Representative artifact</dt><dd>{selected('adaptive_minhash_32x4'):,} selected</dd></dl>
            </article>
            <article class="method-detail" style="--method-color:#c36c1f"><h3>Full-session embedding: semantic novelty path</h3>
                <ol><li>Build one bounded canonical packet from ordered messages, tools, arguments, outputs, and structural evidence.</li><li>Generate and cache one deterministic unit vector under the 8,192-token profile.</li><li>Query a fresh exact in-memory store for the nearest same-agent leader.</li><li>Join above the cosine threshold or create a new leader; pass novelty/rarity to the shared sampler.</li><li>Reuse the exact compressed packet as auditable LLM-judge evidence; vectors remain retrieval state.</li></ol>
                <p>The local run uses deterministic embeddings and memory search. Production may project vectors to Azure AI Search while Cosmos remains authoritative.</p>
                <dl class="method-facts"><dt>Representation</dt><dd>Unit vector plus compressed evidence</dd><dt>Lookup</dt><dd>Exact same-agent nearest neighbor</dd><dt>Representative artifact</dt><dd>{selected('adaptive_embedding_fullsession'):,} selected</dd></dl>
            </article>
        </div>
        <p class="small"><strong>Shared adaptive rule:</strong> MinHash and embedding differ only in their variety index. They share paired order, admission, backpressure, cap, fresh-state isolation, and post-selection label scoring.</p>
        {d1}{d2}{d3}{d4}
        """


def _render_storage(artifacts: LoadedV2Artifacts) -> str:
    manifest = artifacts.production_storage_manifest
    flow = _architecture_svg(
        "Production architecture flow",
        ["Sampler outputs|membership", "Cosmos authoritative|evaluationRuns/selectionMembership", "Outbox worker|durable fanout", "Azure Search derivative|vector recall"],
        "#1f6f8b",
    )

    data_saved_rows = []
    methods = artifacts.selected_membership_20pct.get("methods") or {}
    pop = int(artifacts.aggregate.get("population_count") or 0)
    for method in ("census", "random_sampling_stratified", "adaptive_minhash_32x4", "adaptive_embedding_fullsession"):
        row = methods.get(method) or {}
        selected = int(row.get("selected_count") or 0)
        data_saved_rows.append(
            "<tr>"
            f"<td>{escape(_label(method))}</td>"
            f"<td>{selected}</td>"
            f"<td>{escape(_pct(1.0 - (selected / max(pop, 1)), 1))}</td>"
            "</tr>"
        )

    containers = (((manifest.get("proposed_logical_model") or {}).get("containers") or []))
    cont_rows = "".join(
        "<tr>"
        f"<td>{escape(str(c.get('name') or ''))}</td>"
        f"<td>{escape(str(c.get('partitionKey') or ''))}</td>"
        f"<td>{escape(', '.join(str(x) for x in (c.get('query_paths') or [])))}</td>"
        "</tr>"
        for c in containers
        if isinstance(c, dict)
    )

    return (
        "<p>Production storage design treats ESP/Cosmos as authoritative and Azure Search as derivative retrieval state.</p>"
        + flow
        + "<h3>Data saved by method (representative 20% selection artifact)</h3>"
        + "<table><thead><tr><th>Method</th><th>Selected sessions</th><th>Fraction saved vs 2,800</th></tr></thead>"
        + f"<tbody>{''.join(data_saved_rows)}</tbody></table>"
        + "<h3>Proposed logical containers and partition paths</h3>"
        + "<table><thead><tr><th>Container</th><th>Partition key</th><th>Query paths</th></tr></thead>"
        + f"<tbody>{cont_rows}</tbody></table>"
        + "<p>A manually recorded read-only Azure MCP observation on 2026-07-30 found <code>stangoodwin-ai-search</code>/<code>maven-session-sampling-v1</code> without a vector field; this observation was not rerun by the experiment bundle. The local experiment did not use Azure resources. Production requires a separate vector index (HNSW, cosine) with tenant/agent/profile/expiry filters, prefiltering, managed identity, private endpoint, CMK as required, and deletion worker (no TTL reliance).</p>"
        + "<p>Cosmos remains authoritative state of record; Search stores derivative vector retrieval projections only.</p>"
    )


def _render_output_api(artifacts: LoadedV2Artifacts) -> str:
    manifest = artifacts.external_eval_manifest
    methods_files = manifest.get("methods_files") or {}

    # pick one sample result payload
    sample_payload = None
    sample_method = None
    for method in ("census", "random_sampling_stratified", "adaptive_minhash_32x4", "adaptive_embedding_fullsession"):
        rows = artifacts.external_eval_method_rows.get(method) or []
        if rows:
            sample_payload = rows[0]
            sample_method = method
            break
    if sample_payload is None:
        sample_payload = {"error": "no snapshot rows"}
        sample_method = "none"

    mapping_rows = [
        ("runId", "snapshot.runId", "Deterministic v2 method/day hash"),
        ("agentId", "snapshot.agentId", "Grouped per agent/day"),
        ("date", "snapshot.date", "UTC day granularity"),
        ("totalConversationsCount", "snapshot.totalConversationsCount", "Eligible group denominator"),
        ("totalSampledCount", "snapshot.totalSampledCount", "Selected group cardinality"),
        ("avgScore", "snapshot.avgScore", "Mean binary expected-label score"),
        ("results[]", "snapshot.results", "Echoed max five sampled rows"),
    ]
    map_html = "".join(
        "<tr>"
        f"<td>{escape(src)}</td><td>{escape(dst)}</td><td>{escape(note)}</td>"
        "</tr>"
        for src, dst, note in mapping_rows
    )

    file_rows = []
    for method, entry in sorted(methods_files.items()):
        if not isinstance(entry, dict):
            continue
        file_rows.append(
            "<tr>"
            f"<td>{escape(method)}</td>"
            f"<td>{escape(str(entry.get('path') or ''))}</td>"
            f"<td>{escape(str(entry.get('sha256') or '')[:16])}</td>"
            f"<td>{int(entry.get('line_count') or 0)}</td>"
            "</tr>"
        )

    return (
        "<p>Output API artifacts are generated from JSONL snapshots only. POST is not executed; manifest sets <code>not_posted=true</code>.</p>"
        + f"<p><strong>Gateway route:</strong> {escape(str(manifest.get('route_template') or 'POST /evals/service/results?api-version=1'))}. Discovery/auth must be supplied by environment service configuration and identity.</p>"
        + f"<p><strong>Sample method:</strong> {escape(_label(sample_method or ''))}</p>"
        + f"<pre>{escape(json.dumps(sample_payload, indent=2, sort_keys=True))}</pre>"
        + "<h3>Field mapping to ExternalEvalSnapshot</h3>"
        + "<table><thead><tr><th>Experiment artifact field</th><th>ExternalEvalSnapshot field</th><th>Notes</th></tr></thead>"
        + f"<tbody>{map_html}</tbody></table>"
        + "<p>Grouping is one agent/day. <code>totalSampledCount</code> includes full selected group while echoed <code>results</code> intentionally caps at five rows.</p>"
        + "<h3>Generated output files and hashes</h3>"
        + "<table><thead><tr><th>Method</th><th>JSONL path</th><th>SHA-256 prefix</th><th>Lines</th></tr></thead>"
        + f"<tbody>{''.join(file_rows)}</tbody></table>"
    )


def _render_methodology_limits(artifacts: LoadedV2Artifacts) -> str:
    return (
        "<ul>"
        "<li>Paired repetitions with fixed representation precompute and fresh adaptive state per run.</li>"
        "<li>Label permutation and expected-label scoring only; no LLM judging calls.</li>"
        "<li>Both source blocks are synthetic workloads.</li>"
        "<li>Embedding arm uses deterministic offline embeddings, not production semantic model serving.</li>"
        "<li>Adaptive selections are nonprobability mechanisms; interpret as mechanism diagnostics.</li>"
        "<li>Caps are nominal and can differ from realized keep rates under replay/backpressure.</li>"
        "<li>Throughput study boundaries: admission mechanics only, no live Azure vector/LLM latency.</li>"
        "<li>Source-block diagnostics are required because historical_300 and dense_2500 distributions differ.</li>"
        "<li>No live Azure vector index run in this artifact set.</li>"
        "<li>No PPAPI POST execution; outputs are local files with not_posted=true metadata.</li>"
        "</ul>"
    )


def _render_provenance(artifacts: LoadedV2Artifacts, generated_at: str) -> str:
    rows = []
    for key, path in sorted(artifacts.source_paths.items()):
        sha = _file_sha(path)
        display_path = str(path).replace("\\", "/")
        rows.append(
            "<tr>"
            f"<td>{escape(key)}</td>"
            f"<td>{escape(display_path)}</td>"
            f"<td>{escape(sha[:16])}</td>"
            f"<td>{path.stat().st_size:,}</td>"
            "</tr>"
        )
    return (
        f"<p class=\"small\">Artifact generation timestamp: {escape(generated_at)}.</p>"
        + "<table><thead><tr><th>Artifact</th><th>Path</th><th>SHA-256 prefix</th><th>Bytes</th></tr></thead>"
        + f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_v2_html_report(artifacts: LoadedV2Artifacts) -> str:
    validate_v2_artifacts(artifacts)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    tabs = [
        ("overview", "Overview", _render_overview(artifacts, generated_at)),
        ("input-data", "Input Data", _render_input_data(artifacts)),
        ("metrics", "Metrics", _render_metrics(artifacts)),
        ("outcomes", "Outcomes", _render_outcomes(artifacts)),
        ("agents", "Agents", _render_agents(artifacts)),
        ("quadrants", "Quadrants", _render_quadrants(artifacts)),
        ("throughput", "Throughput", _render_throughput(artifacts)),
        ("methods", "Methods", _render_methods(artifacts)),
        ("production-storage", "Production Storage", _render_storage(artifacts)),
        ("output-api", "Output/API", _render_output_api(artifacts)),
        ("methodology", "Methodology/Limits", _render_methodology_limits(artifacts)),
        ("provenance", "Provenance", _render_provenance(artifacts, generated_at)),
    ]

    tab_buttons = "".join(
        _tab_button(tab_id, label, selected=(idx == 0))
        for idx, (tab_id, label, _) in enumerate(tabs)
    )
    tab_panels = "".join(
        _tab_panel(tab_id, label, content, selected=(idx == 0))
        for idx, (tab_id, label, content) in enumerate(tabs)
    )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Agent365 Sampling V2 Report</title>
<style>
:root {{
  --ink: #1f2833;
  --muted: #46505c;
  --paper: #f4f5f7;
  --panel: #ffffff;
  --line: #d0d5dd;
  --accent: #205b83;
  --accent-2: #0f7d6c;
  --warn: #b8501d;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  background: var(--paper);
  font-family: Bahnschrift, Aptos, "Segoe UI", Tahoma, sans-serif;
  line-height: 1.45;
}}
main {{ max-width: 1260px; margin: 0 auto; padding: 14px 14px 24px; }}
header.top {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }}
h1 {{ font-size: 1.3rem; margin: 0; letter-spacing: 0.02em; }}
.small {{ color: var(--muted); font-size: 0.93rem; }}
button.print {{
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--ink);
  border-radius: 8px;
  padding: 8px 12px;
  font: inherit;
  cursor: pointer;
}}
.tab-shell {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}}
.tab-list {{
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--line);
  overflow-x: auto;
  scroll-snap-type: x mandatory;
  background: #eef1f4;
}}
.tab-button {{
  flex: 0 0 auto;
  border: none;
  border-right: 1px solid var(--line);
  background: transparent;
  color: var(--ink);
  padding: 12px 14px;
  font: inherit;
  font-size: 0.95rem;
  cursor: pointer;
  scroll-snap-align: start;
}}
.tab-button[aria-selected=\"true\"] {{
  background: var(--panel);
  color: var(--accent);
  font-weight: 700;
}}
.tab-panel {{ padding: 14px; }}
.tab-panel[hidden] {{ display: none; }}
.section-heading h2 {{ margin: 0 0 8px; font-size: 1.05rem; color: #193447; }}
h3 {{ margin: 16px 0 8px; font-size: 1rem; color: #21384a; }}
p {{ margin: 8px 0; }}
pre {{
  background: #0b1020;
  color: #f0f4ff;
  border-radius: 8px;
  padding: 10px;
  overflow-x: auto;
  border: 1px solid #21283a;
  font-size: 0.84rem;
}}
code {{ background: #f0f2f5; padding: 0 3px; border-radius: 4px; }}
ul {{ padding-left: 18px; }}
.grid-4 {{ display: grid; grid-template-columns: repeat(4, minmax(190px, 1fr)); gap: 10px; }}
.grid-4 article {{ background: #fbfcfd; border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
figure.chart, figure.arch-figure {{ margin: 12px 0; border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: #fff; }}
figure figcaption {{ font-weight: 700; margin-bottom: 6px; color: #173247; }}
.chart-legend {{ display: flex; flex-wrap: wrap; gap: 8px 12px; margin-bottom: 6px; }}
.chart-legend span {{ display: inline-flex; align-items: center; gap: 6px; font-size: 0.86rem; color: var(--muted); }}
.chart-legend i {{ width: 13px; height: 13px; display: inline-block; border-radius: 3px; }}
.chart-scroll {{ overflow-x: auto; overflow-y: hidden; -webkit-overflow-scrolling: touch; }}
svg {{ width: 100%; min-width: 860px; height: auto; }}
.axis {{ fill: #3f4a56; font-size: 12px; }}
.axis-stroke {{ stroke: #8893a0; stroke-width: 1; }}
.grid {{ stroke: #e7eaf0; stroke-width: 1; }}
.legend {{ fill: #3f4a56; font-size: 12px; }}
.heatmap-value {{ fill: #1f2833; font-size: 11px; }}
.flow-title {{ fill: #1f2833; font-size: 12px; font-weight: 700; }}
.flow-sub {{ fill: #566272; font-size: 11px; }}
.prototype-controls {{ margin: 16px 0; padding: 12px 14px; border-left: 4px solid var(--accent); background: #eef4f8; }}
.prototype-controls h3 {{ margin-top: 0; }}
.prototype-steps {{ display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 8px; list-style: none; padding: 0; counter-reset: prototype-step; }}
.prototype-steps li {{ counter-increment: prototype-step; padding: 9px 10px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
.prototype-steps li::before {{ content: counter(prototype-step); display: inline-grid; place-items: center; width: 22px; height: 22px; margin-right: 7px; border-radius: 50%; background: var(--accent); color: #fff; font-weight: 700; }}
.method-detail-grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 12px; margin: 14px 0; }}
.method-detail {{ border: 1px solid var(--line); border-top: 4px solid var(--method-color, var(--accent)); border-radius: 8px; padding: 11px 13px; background: #fff; }}
.method-detail h3 {{ margin-top: 0; }}
.method-detail ol {{ margin: 6px 0; padding-left: 20px; }}
.method-detail li {{ margin: 4px 0; }}
.method-facts {{ display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; margin: 9px 0 0; font-size: 0.88rem; }}
.method-facts dt {{ font-weight: 700; color: #31475a; }}
.method-facts dd {{ margin: 0; }}
table {{ width: 100%; border-collapse: collapse; border: 1px solid var(--line); margin: 8px 0; background: #fff; }}
.tab-panel > table {{ display: block; width: max-content; max-width: 100%; overflow-x: auto; }}
th, td {{ border: 1px solid var(--line); padding: 7px 8px; text-align: left; font-size: 0.89rem; vertical-align: top; }}
th {{ background: #edf1f6; color: #22384a; position: sticky; top: 0; z-index: 1; cursor: default; }}
.table-scroll {{ overflow: auto; max-height: 460px; border: 1px solid var(--line); border-radius: 8px; }}
.sortable th {{ cursor: pointer; }}
@media (max-width: 980px) {{
  main {{ padding: 10px; }}
  .grid-4 {{ grid-template-columns: 1fr 1fr; }}
    .prototype-steps {{ grid-template-columns: 1fr 1fr; }}
    .method-detail-grid {{ grid-template-columns: 1fr; }}
  .tab-button {{ padding: 10px 12px; font-size: 0.92rem; }}
  svg {{ min-width: 780px; }}
}}
@media (max-width: 680px) {{
  header.top {{ flex-direction: column; align-items: flex-start; }}
  .grid-4 {{ grid-template-columns: 1fr; }}
    .prototype-steps {{ grid-template-columns: 1fr; }}
  svg {{ min-width: 720px; }}
}}
@media print {{
  .tab-list, .print {{ display: none !important; }}
  .tab-panel[hidden] {{ display: block !important; }}
  .tab-shell {{ border: none; }}
  figure.chart, figure.arch-figure {{ break-inside: avoid; }}
}}
</style>
</head>
<body>
<main>
  <header class=\"top\">
    <div>
      <h1>Agent365 Sampling V2 Self-Contained Report</h1>
      <div class=\"small\">Version {escape(REPORT_VERSION)}. Inputs are persisted local artifacts, no network dependencies.</div>
    </div>
    <button class=\"print\" type=\"button\" id=\"printButton\">Print</button>
  </header>
  <div class=\"tab-shell\">
    <div class=\"tab-list\" role=\"tablist\" aria-label=\"Sampling v2 report sections\">{tab_buttons}</div>
    {tab_panels}
  </div>
</main>
<script>
(function() {{
  const tabs = Array.from(document.querySelectorAll('.tab-button'));
  const panels = Array.from(document.querySelectorAll('.tab-panel'));
  function activate(tab) {{
    const target = tab.getAttribute('aria-controls');
    tabs.forEach((t) => {{
      const on = t === tab;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
    }});
    panels.forEach((p) => {{
      const on = p.id === target;
      if (on) p.removeAttribute('hidden'); else p.setAttribute('hidden', 'hidden');
    }});
    tab.focus();
  }}
  tabs.forEach((tab, idx) => {{
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', (e) => {{
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Enter', ' '].includes(e.key)) return;
      e.preventDefault();
      if (e.key === 'Enter' || e.key === ' ') return activate(tab);
      let i = idx;
      if (e.key === 'ArrowRight') i = (idx + 1) % tabs.length;
      if (e.key === 'ArrowLeft') i = (idx - 1 + tabs.length) % tabs.length;
      if (e.key === 'Home') i = 0;
      if (e.key === 'End') i = tabs.length - 1;
      activate(tabs[i]);
    }});
  }});
  const printBtn = document.getElementById('printButton');
  if (printBtn) printBtn.addEventListener('click', () => window.print());

  // lightweight table sort for agent table
  const table = document.getElementById('agent-table');
  if (table) {{
    const tbody = table.tBodies[0];
    const headers = Array.from(table.tHead.rows[0].cells);
    headers.forEach((th, index) => {{
      th.addEventListener('click', () => {{
        const sortType = th.dataset.sort || 'text';
        const rows = Array.from(tbody.rows);
        const asc = th.dataset.asc !== 'true';
        headers.forEach((h) => h.dataset.asc = '');
        th.dataset.asc = asc ? 'true' : 'false';
        rows.sort((a, b) => {{
          const va = a.cells[index].innerText.trim();
          const vb = b.cells[index].innerText.trim();
          if (sortType === 'num') {{
            const na = parseFloat(va.replace('%', ''));
            const nb = parseFloat(vb.replace('%', ''));
            return asc ? na - nb : nb - na;
          }}
          return asc ? va.localeCompare(vb) : vb.localeCompare(va);
        }});
        rows.forEach((r) => tbody.appendChild(r));
      }});
    }});
  }}
}})();
</script>
</body>
</html>
"""
    if "https://" in html or "http://" in html or "cdn" in html.lower():
        raise ValueError("report must be self-contained and avoid external assets")
    return html


def write_v2_html_report(
    *,
    output_path: Path = DEFAULT_OUTPUT_HTML,
    inputs: V2ReportInputs | None = None,
) -> Path:
    artifacts = load_v2_artifacts(inputs)
    html = render_v2_html_report(artifacts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
