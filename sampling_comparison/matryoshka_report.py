from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_DATASETS: tuple[str, ...] = (
    "historical_300",
    "dense_2500",
    "cosmos_otel",
    "tau2_bench",
)
EXPECTED_DIMENSIONS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1536)
EXPECTED_MODES: tuple[str, ...] = ("end_to_end", "fixed_membership")
EXPECTED_RATES: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20)


@dataclass
class NativeAgentRow:
    dataset_id: str
    mode: str
    agent_id: str
    accuracy: float | None
    n: float
    selected_count: float
    unjudged_count: float
    mae: float | None
    provenance_counts: Mapping[str, float]


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _to_int(value: Any) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _fmt_num(value: float | None, *, digits: int = 3, na: str = "N/A") -> str:
    if value is None:
        return na
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, *, digits: int = 1, na: str = "N/A") -> str:
    if value is None:
        return na
    return f"{value * 100:.{digits}f}%"


def _fmt_pp(value: float | None, *, digits: int = 2, na: str = "N/A") -> str:
    if value is None:
        return na
    return f"{value:.{digits}f} pp"


def _fmt_int(value: int | None, *, na: str = "N/A") -> str:
    if value is None:
        return na
    return f"{value:,d}"


def _fmt_count(value: float | int | None, *, na: str = "N/A") -> str:
    if value is None:
        return na
    numeric = float(value)
    if not math.isfinite(numeric):
        return na
    if abs(numeric - round(numeric)) < 1e-9:
        return f"{int(round(numeric)):,d}"
    return f"{numeric:,.2f}"


def _mean(values: Iterable[float]) -> float | None:
    seq = [item for item in values if item is not None]
    if not seq:
        return None
    return float(sum(seq)) / float(len(seq))


def _pp_delta(value: float | None) -> float | None:
    if value is None:
        return None
    if abs(value) <= 1.0:
        return value * 100.0
    return value


def _slugify(value: str) -> str:
    out = []
    for char in value.lower():
        if char.isalnum():
            out.append(char)
        elif char in {"-", "_"}:
            out.append("-")
        else:
            out.append("-")
    return "".join(out).strip("-") or "chart"


def _dataset_rows(datasets: Any) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    if not isinstance(datasets, list):
        return out
    for item in datasets:
        if not isinstance(item, Mapping):
            continue
        dataset_id = item.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            continue
        out[dataset_id] = item
    return out


def _extract_embedding_call_hints(value: Any, path: str = "") -> list[tuple[str, Any]]:
    hints: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_str = str(key)
            next_path = f"{path}.{key_str}" if path else key_str
            lowered = key_str.lower()
            if "embed" in lowered and ("call" in lowered or "api" in lowered or "request" in lowered):
                hints.append((next_path, nested))
            hints.extend(_extract_embedding_call_hints(nested, next_path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            hints.extend(_extract_embedding_call_hints(nested, f"{path}[{index}]"))
    return hints


def _preview_json(value: Any, limit: int = 160) -> str:
    try:
        text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        text = repr(value)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return _esc(text)


def _candidate_path(path_text: str, base: Path) -> Path | None:
    if not path_text:
        return None
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    direct = Path(path_text)
    relative = base / path_text
    if relative.exists():
        return relative
    if direct.exists():
        return direct
    return relative


def _find_input_readiness_path(aggregate: Mapping[str, Any], aggregate_path: str) -> Path | None:
    base = Path(aggregate_path).resolve().parent
    files = aggregate.get("files") if isinstance(aggregate.get("files"), Mapping) else {}
    for key in ("input_readiness", "input_readiness.json"):
        value = files.get(key)
        path_text = ""
        if isinstance(value, str):
            path_text = value
        elif isinstance(value, Mapping) and isinstance(value.get("path"), str):
            path_text = str(value.get("path"))
        path = _candidate_path(path_text, base)
        if path is not None and path.exists():
            return path
    fallback = base / "input_readiness.json"
    if fallback.exists():
        return fallback
    return None


def _load_input_readiness(aggregate: Mapping[str, Any], aggregate_path: str) -> tuple[Mapping[str, Any] | None, str | None]:
    path = _find_input_readiness_path(aggregate, aggregate_path)
    if path is None:
        return None, "input_readiness.json not found beside aggregate and not declared in files map."
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Failed to load input readiness JSON at {path}: {exc}"
    if not isinstance(payload, Mapping):
        return None, f"Input readiness JSON at {path} is not an object."
    return payload, None


def _collect_warnings(aggregate: Mapping[str, Any], dataset_map: Mapping[str, Mapping[str, Any]], rows: list[Mapping[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if aggregate.get("version") != "matryoshka-cutoff-v1":
        warnings.append(
            "Unexpected aggregate version. Expected matryoshka-cutoff-v1; interpret outputs carefully."
        )
    missing = [dataset_id for dataset_id in REQUIRED_DATASETS if dataset_id not in dataset_map]
    if missing:
        warnings.append(
            "Required datasets missing from aggregate datasets list: "
            + ", ".join(missing)
            + ". Missing data is shown as unavailable, never treated as zero."
        )
    missing_provenance = []
    for dataset_id, entry in dataset_map.items():
        if not isinstance(entry, Mapping):
            continue
        if not isinstance(entry.get("provenance"), Mapping):
            missing_provenance.append(dataset_id)
    if missing_provenance:
        warnings.append(
            "Provenance/profile object missing for dataset(s): "
            + ", ".join(sorted(missing_provenance))
            + ". Report falls back gracefully but embedding-preparation evidence may be unavailable."
        )
    bad_rows = 0
    for row in rows:
        if not isinstance(row, Mapping):
            bad_rows += 1
            continue
        if row.get("dataset_id") is None or row.get("mode") is None or row.get("dimension") is None:
            bad_rows += 1
    if bad_rows:
        warnings.append(
            f"{bad_rows} row(s) are malformed (missing dataset_id/mode/dimension). They are excluded from charts."
        )
    return warnings


def _safe_json(value: Any) -> str:
    try:
        rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        rendered = repr(value)
    return _esc(rendered)


def _unique_id(prefix: str, index: int) -> str:
    return f"{_slugify(prefix)}-{index}"


def _render_multi_series_svg(
    *,
    chart_id: str,
    title: str,
    description: str,
    y_label: str,
    series: list[tuple[str, dict[int, float | None], str]],
    y_min: float | None = None,
    y_max: float | None = None,
    zero_line: bool = False,
    value_formatter=lambda value: f"{value:.3f}",
) -> str:
    width = 640
    legend_rows = math.ceil(len(series) / 3)
    height = 300 + legend_rows * 20
    margin_left = 56
    margin_right = 32
    margin_top = 26 + legend_rows * 20
    margin_bottom = 54
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    dims = list(EXPECTED_DIMENSIONS)
    points_available = [
        value
        for _, values, _ in series
        for value in values.values()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if not points_available:
        return (
            "<div class=\"empty-state\" role=\"status\" aria-live=\"polite\">"
            "No data available for this chart scope."
            "</div>"
        )
    inferred_min = min(points_available)
    inferred_max = max(points_available)
    lo = inferred_min if y_min is None else y_min
    hi = inferred_max if y_max is None else y_max
    if zero_line:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if abs(hi - lo) < 1e-12:
        hi = lo + 1.0
    if y_min is not None and y_max is not None and not zero_line:
        lo = float(y_min)
        hi = float(y_max) if abs(float(y_max) - float(y_min)) > 1e-12 else float(y_min) + 1.0
    else:
        pad = (hi - lo) * 0.12
        lo -= pad
        hi += pad

    def x_of(dimension: int) -> float:
        index = dims.index(dimension)
        if len(dims) == 1:
            return margin_left + plot_w / 2.0
        return margin_left + (plot_w * index / (len(dims) - 1))

    def y_of(value: float) -> float:
        return margin_top + (hi - value) * plot_h / (hi - lo)

    ticks = 5
    y_tick_values = [lo + (hi - lo) * idx / ticks for idx in range(ticks + 1)]
    grid_lines = []
    for tick_value in y_tick_values:
        y = y_of(tick_value)
        grid_lines.append(
            f"<line x1=\"{margin_left:.1f}\" y1=\"{y:.1f}\" x2=\"{margin_left + plot_w:.1f}\" y2=\"{y:.1f}\" class=\"grid\"/>"
            f"<text x=\"{margin_left - 8:.1f}\" y=\"{y + 4:.1f}\" class=\"axis-label\" text-anchor=\"end\">{_esc(value_formatter(tick_value))}</text>"
        )
    x_ticks = []
    for dim in dims:
        x = x_of(dim)
        x_ticks.append(
            f"<line x1=\"{x:.1f}\" y1=\"{margin_top + plot_h:.1f}\" x2=\"{x:.1f}\" y2=\"{margin_top + plot_h + 5:.1f}\" class=\"axis\"/>"
            f"<text x=\"{x:.1f}\" y=\"{margin_top + plot_h + 18:.1f}\" class=\"axis-label\" text-anchor=\"middle\">{dim}</text>"
        )

    rendered_series: list[str] = []
    for label, values, color in series:
        current_segment: list[tuple[float, float, int, float]] = []
        for dim in dims:
            value = values.get(dim)
            if value is None or not math.isfinite(float(value)):
                if current_segment:
                    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y, _, _ in current_segment)
                    rendered_series.append(
                        f"<polyline points=\"{coords}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2.4\"/>"
                    )
                    current_segment = []
                continue
            x = x_of(dim)
            y = y_of(float(value))
            current_segment.append((x, y, dim, float(value)))
        if current_segment:
            coords = " ".join(f"{x:.2f},{y:.2f}" for x, y, _, _ in current_segment)
            rendered_series.append(
                f"<polyline points=\"{coords}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2.4\"/>"
            )

        point_markup: list[str] = []
        for dim in dims:
            value = values.get(dim)
            if value is None or not math.isfinite(float(value)):
                continue
            point = (x_of(dim), y_of(float(value)), dim, float(value))
            point_markup.append(
                "<g>"
                f"<circle cx=\"{point[0]:.2f}\" cy=\"{point[1]:.2f}\" r=\"3.4\" fill=\"{color}\"/>"
                f"<title>{_esc(label)} at {dim}d: {value_formatter(point[3])}</title>"
                "</g>"
            )
        rendered_series.extend(point_markup)

    legend = []
    for index, (label, _, color) in enumerate(series):
        x = margin_left + (index % 3) * 180
        y = 18 + (index // 3) * 20
        legend.append(
            f"<line x1=\"{x}\" y1=\"{y - 4}\" x2=\"{x + 14}\" y2=\"{y - 4}\" stroke=\"{color}\" stroke-width=\"2.4\"/>"
            f"<text x=\"{x + 20}\" y=\"{y}\" class=\"axis-label legend-label\">{_esc(label)}</text>"
        )
    if zero_line and lo <= 0.0 <= hi:
        zero_y = y_of(0.0)
        rendered_series.append(
            f"<line x1=\"{margin_left:.1f}\" y1=\"{zero_y:.1f}\" x2=\"{margin_left + plot_w:.1f}\" y2=\"{zero_y:.1f}\" class=\"zero-line\"/>"
        )

    title_id = f"{chart_id}-title"
    desc_id = f"{chart_id}-desc"
    return (
        f"<svg class=\"chart\" role=\"img\" aria-label=\"{_esc(title)}\" aria-labelledby=\"{_esc(title_id)} {_esc(desc_id)}\" "
        f"viewBox=\"0 0 {width} {height}\" focusable=\"false\">"
        f"<title id=\"{_esc(title_id)}\">{_esc(title)}</title>"
        f"<desc id=\"{_esc(desc_id)}\">{_esc(description)}</desc>"
        f"<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"white\"/>"
        + "".join(legend)
        + "".join(grid_lines)
        + f"<line x1=\"{margin_left:.1f}\" y1=\"{margin_top + plot_h:.1f}\" x2=\"{margin_left + plot_w:.1f}\" y2=\"{margin_top + plot_h:.1f}\" class=\"axis\"/>"
        + f"<line x1=\"{margin_left:.1f}\" y1=\"{margin_top:.1f}\" x2=\"{margin_left:.1f}\" y2=\"{margin_top + plot_h:.1f}\" class=\"axis\"/>"
        + "".join(x_ticks)
        + "".join(rendered_series)
        + f"<text x=\"{margin_left + plot_w / 2:.1f}\" y=\"{height - 8:.1f}\" class=\"axis-label\" text-anchor=\"middle\">Dimensions (Matryoshka prefix, re-normalized)</text>"
        + f"<text transform=\"translate(14 {margin_top + plot_h / 2:.1f}) rotate(-90)\" class=\"axis-label\" text-anchor=\"middle\">{_esc(y_label)}</text>"
        + "</svg>"
    )


def _build_grouped_metrics(rows: list[Mapping[str, Any]]) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    dict[tuple[str, str, float, int], list[float]],
    dict[tuple[str, str, str, float], list[float]],
    list[NativeAgentRow],
]:
    grouped: dict[tuple[str, str, int], dict[str, Any]] = {}
    rate_breakdown: dict[tuple[str, str, float, int], list[float]] = defaultdict(list)
    schedule_breakdown: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    native_agents: list[NativeAgentRow] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        dataset_id = row.get("dataset_id")
        mode = row.get("mode")
        dimension = _to_int(row.get("dimension"))
        if not isinstance(dataset_id, str) or not isinstance(mode, str) or dimension is None:
            continue
        key = (dataset_id, mode, dimension)
        if key not in grouped:
            grouped[key] = {
                "accuracy": [],
                "mae": [],
                "f1": [],
                "brier": [],
                "macro_agent_accuracy": [],
                "combined_accuracy": [],
                "aggregate_rate_error": [],
                "concept_coverage": [],
                "agent_coverage": [],
                "accuracy_delta_native": [],
                "always_pass_accuracy": [],
                "selected_count_total": 0,
                "unjudged_count_total": 0,
                "row_count": 0,
                "provenance_counts": {"idw": 0, "exact_match": 0, "global_mean": 0, "prior": 0},
            }
        target = grouped[key]
        for metric in (
            "accuracy",
            "mae",
            "f1",
            "brier",
            "macro_agent_accuracy",
            "combined_accuracy",
            "aggregate_rate_error",
            "concept_coverage",
            "agent_coverage",
            "accuracy_delta_native",
        ):
            value = _to_float(row.get(metric))
            if value is not None:
                target[metric].append(value)
        selected_count = _to_int(row.get("selected_count")) or 0
        unjudged_count = _to_int(row.get("unjudged_count")) or 0
        unjudged_count_float = _to_float(row.get("unjudged_count"))
        target["selected_count_total"] += selected_count
        target["unjudged_count_total"] += unjudged_count
        target["row_count"] += 1
        tp = _to_float(row.get("tp"))
        fn = _to_float(row.get("fn"))
        if unjudged_count_float is not None and unjudged_count_float > 0 and tp is not None and fn is not None:
            always_pass_accuracy = (tp + fn) / unjudged_count_float
            if math.isfinite(always_pass_accuracy):
                target["always_pass_accuracy"].append(always_pass_accuracy)

        provenance_counts = row.get("provenance_counts")
        if isinstance(provenance_counts, Mapping):
            for key_name in ("idw", "exact_match", "global_mean", "prior"):
                target["provenance_counts"][key_name] += _to_int(provenance_counts.get(key_name)) or 0

        rate = _to_float(row.get("rate"))
        delta = _to_float(row.get("accuracy_delta_native"))
        if rate is not None and delta is not None:
            rate_breakdown[(dataset_id, mode, rate, dimension)].append(_pp_delta(delta) or 0.0)
            schedule = row.get("schedule")
            schedule_label = str(schedule) if schedule is not None else "unknown"
            schedule_breakdown[(dataset_id, mode, schedule_label, rate)].append(_pp_delta(delta) or 0.0)

        if dimension == 1536:
            per_agent = row.get("per_agent")
            if isinstance(per_agent, Mapping):
                for agent_id, metrics in per_agent.items():
                    if not isinstance(metrics, Mapping):
                        continue
                    native_agents.append(
                        NativeAgentRow(
                            dataset_id=dataset_id,
                            mode=mode,
                            agent_id=str(agent_id),
                            accuracy=_to_float(metrics.get("accuracy")),
                            n=_to_float(metrics.get("n")) or 0.0,
                            selected_count=_to_float(metrics.get("selected_count")) or 0.0,
                            unjudged_count=_to_float(metrics.get("unjudged_count")) or 0.0,
                            mae=_to_float(metrics.get("mae")),
                            provenance_counts=(
                                metrics.get("provenance_counts")
                                if isinstance(metrics.get("provenance_counts"), Mapping)
                                else {}
                            ),
                        )
                    )
    return grouped, rate_breakdown, schedule_breakdown, native_agents


def _collect_native_agents(agent_summary_rows: Any, fallback_rows: list[NativeAgentRow]) -> list[NativeAgentRow]:
    if not isinstance(agent_summary_rows, list):
        return fallback_rows
    parsed: list[NativeAgentRow] = []
    for row in agent_summary_rows:
        if not isinstance(row, Mapping):
            continue
        dataset_id = row.get("dataset_id")
        mode = row.get("mode")
        dimension = _to_int(row.get("dimension"))
        agent_id = row.get("agent_id")
        if not isinstance(dataset_id, str) or not isinstance(mode, str) or not isinstance(agent_id, str):
            continue
        if dimension != 1536:
            continue
        parsed.append(
            NativeAgentRow(
                dataset_id=dataset_id,
                mode=mode,
                agent_id=agent_id,
                accuracy=_to_float(row.get("accuracy")),
                n=_to_float(row.get("n")) or 0.0,
                selected_count=_to_float(row.get("selected_count")) or 0.0,
                unjudged_count=_to_float(row.get("unjudged_count")) or 0.0,
                mae=_to_float(row.get("mae")),
                provenance_counts=(
                    row.get("provenance_counts")
                    if isinstance(row.get("provenance_counts"), Mapping)
                    else {}
                ),
            )
        )
    return parsed or fallback_rows


def _build_summary_map(summary_rows: Any) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    summary_map: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    if not isinstance(summary_rows, list):
        return summary_map
    for row in summary_rows:
        if not isinstance(row, Mapping):
            continue
        dataset_id = row.get("dataset_id")
        mode = row.get("mode")
        dimension = _to_int(row.get("dimension"))
        if not isinstance(dataset_id, str) or not isinstance(mode, str) or dimension is None:
            continue
        summary_map[(dataset_id, mode, dimension)] = row
    return summary_map


def _metric_value(
    summary_map: Mapping[tuple[str, str, int], Mapping[str, Any]],
    grouped_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    dataset_id: str,
    mode: str,
    dimension: int,
    metric: str,
) -> float | None:
    summary_row = summary_map.get((dataset_id, mode, dimension))
    if summary_row is not None:
        direct = _to_float(summary_row.get(metric))
        if direct is not None:
            return direct
    grouped = grouped_rows.get((dataset_id, mode, dimension))
    if not grouped:
        return None
    values = grouped.get(metric)
    if isinstance(values, list):
        return _mean(values)
    return _to_float(values)


def _takeaway_line(dataset_id: str, mode: str, summary_map: Mapping[tuple[str, str, int], Mapping[str, Any]], grouped_rows: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> str:
    native = _metric_value(summary_map, grouped_rows, dataset_id, mode, 1536, "accuracy")
    small = _metric_value(summary_map, grouped_rows, dataset_id, mode, 256, "accuracy")
    if native is None or small is None:
        return "Takeaway: insufficient rows for a reliable 256d vs native comparison in this dataset/mode."
    delta_pp = (small - native) * 100.0
    return (
        "Takeaway: 256d differs from native by "
        f"{delta_pp:+.2f} pp on primary unjudged-only accuracy for {dataset_id} / {mode}."
    )


def _render_dataset_inventory(dataset_map: Mapping[str, Mapping[str, Any]], rows: list[Mapping[str, Any]]) -> str:
    rows_by_dataset: dict[str, int] = defaultdict(int)
    modes_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        dataset_id = row.get("dataset_id")
        mode = row.get("mode")
        if isinstance(dataset_id, str):
            rows_by_dataset[dataset_id] += 1
            if isinstance(mode, str):
                modes_by_dataset[dataset_id].add(mode)

    dataset_ids = list(REQUIRED_DATASETS) + [x for x in sorted(dataset_map.keys()) if x not in REQUIRED_DATASETS]
    rendered_rows: list[str] = []
    detail_blocks: list[str] = []
    for dataset_id in dataset_ids:
        entry = dataset_map.get(dataset_id)
        if entry is None:
            status = "missing"
            reason = "Missing from aggregate datasets list."
            n = "Unavailable"
            agents = "Unavailable"
            positives = "Unavailable"
            pass_rate = "Unavailable"
            provenance = {}
            hashes = {}
        else:
            status = str(entry.get("status", "unknown"))
            reason = str(entry.get("reason", "")) if entry.get("reason") else ""
            n = _fmt_int(_to_int(entry.get("n")), na="Unavailable")
            agents = _fmt_int(_to_int(entry.get("agents")), na="Unavailable")
            positives = _fmt_int(_to_int(entry.get("positive_count")), na="Unavailable")
            pass_rate = _fmt_pct(_to_float(entry.get("pass_rate")), digits=2, na="Unavailable")
            provenance = entry.get("provenance", {})
            hashes = entry.get("source_hashes", {})
        reason_cell = _esc(reason) if reason else '<span class="muted">N/A</span>'
        rendered_rows.append(
            "<tr "
            f"data-dataset=\"{_esc(dataset_id)}\">"
            f"<th scope=\"row\">{_esc(dataset_id)}</th>"
            f"<td>{_esc(status)}</td>"
            f"<td>{reason_cell}</td>"
            f"<td>{n}</td>"
            f"<td>{agents}</td>"
            f"<td>{positives}</td>"
            f"<td>{pass_rate}</td>"
            f"<td>{rows_by_dataset.get(dataset_id, 0):,d}</td>"
            f"<td>{_esc(', '.join(sorted(modes_by_dataset.get(dataset_id, set()))) or 'N/A')}</td>"
            "</tr>"
        )
        detail_blocks.append(
            "<details class=\"dataset-detail\">"
            f"<summary>{_esc(dataset_id)} provenance &amp; source hashes</summary>"
            "<div class=\"detail-grid\">"
            "<div><h4>Provenance</h4>"
            f"<pre>{_safe_json(provenance)}</pre></div>"
            "<div><h4>Source hashes</h4>"
            f"<pre>{_safe_json(hashes)}</pre></div>"
            "</div></details>"
        )
    return (
        "<section id=\"datasets\">"
        "<h2>Dataset status, counts, and provenance</h2>"
        "<p>Missing data is rendered as <strong>Unavailable</strong>; the report never inserts zero as a substitute. "
        "Dataset <code>provenance</code> is taken from input-manifest profile metadata when provided.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Dataset</th><th>Status</th><th>Reason</th><th>N</th><th>Agents</th>"
        "<th>Positive</th><th>Pass rate</th><th>Row cells</th><th>Modes seen</th>"
        "</tr></thead><tbody>"
        + "".join(rendered_rows)
        + "</tbody></table></div>"
        + "".join(detail_blocks)
        + "</section>"
    )


def _render_embedding_preparation_status(dataset_map: Mapping[str, Mapping[str, Any]]) -> str:
    dataset_ids = list(REQUIRED_DATASETS) + [x for x in sorted(dataset_map.keys()) if x not in REQUIRED_DATASETS]
    rows_html: list[str] = []
    for dataset_id in dataset_ids:
        entry = dataset_map.get(dataset_id)
        if not isinstance(entry, Mapping):
            rows_html.append(
                "<tr>"
                f"<td>{_esc(dataset_id)}</td><td>Unavailable</td>"
                "<td>No dataset profile present.</td>"
                "</tr>"
            )
            continue
        provenance = entry.get("provenance")
        if not isinstance(provenance, Mapping):
            rows_html.append(
                "<tr>"
                f"<td>{_esc(dataset_id)}</td><td>Unavailable</td>"
                "<td>Provenance profile missing or non-object.</td>"
                "</tr>"
            )
            continue
        hints = _extract_embedding_call_hints(provenance)
        if hints:
            joined = "; ".join(f"{path}={_preview_json(value, limit=60)}" for path, value in hints[:4])
            rows_html.append(
                "<tr>"
                f"<td>{_esc(dataset_id)}</td><td>Recorded</td><td>{joined}</td>"
                "</tr>"
            )
        else:
            rows_html.append(
                "<tr>"
                f"<td>{_esc(dataset_id)}</td><td>Not explicitly recorded</td>"
                "<td>No embedding-call key found in provenance profile. This is not proof of zero calls.</td>"
                "</tr>"
            )
    return (
        "<section id=\"embedding-preparation\">"
        "<h2>Embedding preparation vs offline sweep boundary</h2>"
        "<p>Runner preparation may perform fresh Azure embedding API calls and should record that evidence in dataset provenance/profile fields. "
        "The Matryoshka sweep itself runs offline on prepared 1536d vectors and does not require LLM judging.</p>"
        "<p>Do not infer zero API usage for the run as a whole from sweep-only counters.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Dataset</th><th>Embedding call evidence</th><th>Profile details</th>"
        "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div></section>"
    )


def _render_input_readiness(readiness: Mapping[str, Any] | None, error: str | None) -> str:
    if readiness is None:
        message = error or "Input readiness file unavailable."
        return (
            "<section id=\"input-readiness\">"
            "<h2>Input readiness and blockers</h2>"
            f"<p class=\"muted\">{_esc(message)}</p>"
            "<p>When unavailable, blocked datasets should still be treated as unresolved readiness work rather than silently excluded or treated as policy refusal.</p>"
            "</section>"
        )

    blocker = str(readiness.get("authentication_blocker", "")).strip()
    boundary = str(readiness.get("boundary", "")).strip()
    datasets = readiness.get("datasets")
    rows_html: list[str] = []
    if isinstance(datasets, list):
        for item in datasets:
            if not isinstance(item, Mapping):
                continue
            dataset_id = str(item.get("dataset_id", "unknown"))
            source = str(item.get("source", "N/A"))
            prep = item.get("preparation") if isinstance(item.get("preparation"), Mapping) else {}
            status = str(prep.get("status", "unknown"))
            reason = str(prep.get("reason", "")) if prep.get("reason") is not None else ""
            sessions = _fmt_count(_to_float(prep.get("sessions")))
            agents = _fmt_count(_to_float(prep.get("agents")))
            positives = _fmt_count(_to_float(prep.get("positive_count")))
            linked_spans = _fmt_count(_to_float((prep.get("representation_sources") or {}).get("linked_raw_spans")))
            label_doc_fallbacks = _fmt_count(_to_float((prep.get("representation_sources") or {}).get("synthetic_label_document")))
            truncated = _fmt_count(_to_float(prep.get("truncated_sessions")))
            label_source = str(prep.get("label_source", "")).strip()
            reason_cell = _esc(reason) if reason else '<span class="muted">N/A</span>'
            label_source_cell = _esc(label_source) if label_source else '<span class="muted">N/A</span>'
            rows_html.append(
                "<tr>"
                f"<td>{_esc(dataset_id)}</td>"
                f"<td>{_esc(status)}</td>"
                f"<td>{sessions}</td>"
                f"<td>{agents}</td>"
                f"<td>{positives}</td>"
                f"<td>{linked_spans}</td>"
                f"<td>{label_doc_fallbacks}</td>"
                f"<td>{truncated}</td>"
                f"<td>{reason_cell}</td>"
                f"<td>{label_source_cell}</td>"
                f"<td>{_esc(source)}</td>"
                "</tr>"
            )
    auth_note = ""
    lowered = blocker.lower()
    if "aadsts50020" in lowered or "tenant" in lowered and "token" in lowered:
        auth_note = (
            "<p><strong>Interpretation:</strong> this is an operational authentication blocker "
            "(tenant/token access), not a policy refusal.</p>"
        )

    table_html = (
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Dataset</th><th>Preparation status</th><th>Sessions</th><th>Agents</th><th>Positive</th>"
        "<th>Linked spans</th><th>Label-doc fallbacks</th><th>Truncated</th><th>Preparation reason</th><th>Label source note</th><th>Readiness source</th>"
        "</tr></thead><tbody>"
        + ("".join(rows_html) if rows_html else "<tr><td colspan=\"11\" class=\"muted\">No readiness dataset rows.</td></tr>")
        + "</tbody></table></div>"
    )
    return (
        "<section id=\"input-readiness\">"
        "<h2>Input readiness and blockers</h2>"
        f"<p><strong>Authentication blocker:</strong> {_esc(blocker or 'N/A')}</p>"
        f"{auth_note}"
        f"<p><strong>Boundary:</strong> {_esc(boundary or 'N/A')}</p>"
        "<p class=\"muted\">Source-only readiness counts can be populated even when sweep accuracy is unavailable for blocked datasets.</p>"
        + table_html
        + "</section>"
    )


def _render_label_boundary_callout(readiness: Mapping[str, Any] | None) -> str:
    tau_note = (
        "tau2_bench uses observed benchmark rewards (simulations[].reward_info.reward), which are not verified explicit expected task-completion labels. "
        "Because task_102 includes NL_ASSERTION in reward_basis, these rewards cannot be silently substituted under expected-label-only instructions."
    )
    if readiness is None:
        return (
            "<section id=\"label-boundary\">"
            "<h2>Expected-label boundary callout</h2>"
            f"<p>{_esc(tau_note)}</p>"
            "<p class=\"muted\">No readiness artifact loaded; blocked reasons should still preserve this boundary.</p>"
            "</section>"
        )
    blocker = str(readiness.get("authentication_blocker", "")).lower()
    auth_text = (
        "Authentication is also blocked (tenant/token acquisition), so tau2 remains blocked on both auth and expected-label mapping."
        if "aadsts50020" in blocker or ("tenant" in blocker and "token" in blocker)
        else "Authentication state is not explicit in readiness metadata."
    )
    return (
        "<section id=\"label-boundary\">"
        "<h2>Expected-label boundary callout</h2>"
        f"<p>{_esc(tau_note)}</p>"
        f"<p>{_esc(auth_text)}</p>"
        "<p>No new LLM judges are introduced in this report scope.</p>"
        "</section>"
    )


def _render_metric_matrix(
    dataset_ids: list[str],
    modes: list[str],
    summary_map: Mapping[tuple[str, str, int], Mapping[str, Any]],
    grouped_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> str:
    table_rows: list[str] = []
    for dataset_id in dataset_ids:
        for mode in modes:
            for dim in EXPECTED_DIMENSIONS:
                accuracy = _metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "accuracy")
                mae = _metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "mae")
                f1 = _metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "f1")
                brier = _metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "brier")
                macro_acc = _metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "macro_agent_accuracy")
                rate_error = _metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "aggregate_rate_error")
                combined = _metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "combined_accuracy")
                always_pass = _metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "always_pass_accuracy")
                idw_minus_always_pass_pp = (accuracy - always_pass) * 100.0 if accuracy is not None and always_pass is not None else None
                delta_pp = _pp_delta(_metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "accuracy_delta_native"))
                table_rows.append(
                    "<tr>"
                    f"<td>{_esc(dataset_id)}</td>"
                    f"<td>{_esc(mode)}</td>"
                    f"<td>{dim}</td>"
                    f"<td>{_fmt_num(accuracy, digits=4)}</td>"
                    f"<td>{_fmt_num(always_pass, digits=4)}</td>"
                    f"<td>{_fmt_pp(idw_minus_always_pass_pp, digits=3)}</td>"
                    f"<td>{_fmt_pp(delta_pp, digits=2)}</td>"
                    f"<td>{_fmt_num(mae, digits=4)}</td>"
                    f"<td>{_fmt_num(f1, digits=4)}</td>"
                    f"<td>{_fmt_num(brier, digits=4)}</td>"
                    f"<td>{_fmt_num(macro_acc, digits=4)}</td>"
                    f"<td>{_fmt_num(rate_error, digits=4)}</td>"
                    f"<td>{_fmt_num(combined, digits=4)}</td>"
                    "</tr>"
                )
    return (
        "<section id=\"metric-matrix\">"
        "<h2>Primary and diagnostic metric matrix (dimension scope)</h2>"
        "<p>Primary decision metric is <strong>unjudged-only accuracy</strong>. "
        "Combined accuracy (judged + imputed) is shown separately because direct labels inflate it. "
        "Always-pass baseline uses unjudged class prevalence from each paired cell: (tp + fn) / unjudged_count.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Dataset</th><th>Mode</th><th>Dimension</th><th>Accuracy (unjudged)</th><th>Always-pass baseline</th><th>IDW - always-pass</th><th>Δ vs native</th>"
        "<th>MAE</th><th>F1</th><th>Brier</th><th>Macro agent acc</th><th>Aggregate rate error</th>"
        "<th>Combined accuracy</th>"
        "</tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></div></section>"
    )


def _render_seed_ci_table(
    dataset_ids: list[str],
    modes: list[str],
    summary_map: Mapping[tuple[str, str, int], Mapping[str, Any]],
    grouped_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> str:
    body_rows: list[str] = []
    for dataset_id in dataset_ids:
        for mode in modes:
            for dim in EXPECTED_DIMENSIONS:
                summary_row = summary_map.get((dataset_id, mode, dim), {})
                delta = _pp_delta(_metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "accuracy_delta_native"))
                ci = summary_row.get("accuracy_delta_seed_ci95") if isinstance(summary_row, Mapping) else None
                ci_text = "N/A (single seed or unavailable)"
                passes = "N/A"
                if isinstance(ci, list) and len(ci) == 2:
                    low = _pp_delta(_to_float(ci[0]))
                    high = _pp_delta(_to_float(ci[1]))
                    if low is not None and high is not None:
                        ci_text = f"[{low:.2f}, {high:.2f}] pp"
                        passes = "yes" if low >= -1.0 else "no"
                body_rows.append(
                    "<tr>"
                    f"<td>{_esc(dataset_id)}</td><td>{_esc(mode)}</td><td>{dim}</td>"
                    f"<td>{_fmt_pp(delta, digits=2)}</td><td>{_esc(ci_text)}</td><td>{_esc(passes)}</td>"
                    "</tr>"
                )
    return (
        "<section id=\"seed-ci\">"
        "<h2>Seed-sensitivity CI view for one-pp candidate logic</h2>"
        "<p>One-pp candidate uses pointwise seed CI lower bound >= -1.00 pp for the candidate and all larger tested prefixes. "
        "If CI is null (single-seed fixture), candidate logic remains unavailable for that cell.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Dataset</th><th>Mode</th><th>Dimension</th><th>Mean Δ vs native</th><th>Δ seed CI95</th><th>Lower >= -1pp?</th>"
        "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div></section>"
    )


def _render_class_imbalance_reference(
    dataset_ids: list[str],
    modes: list[str],
    dataset_map: Mapping[str, Mapping[str, Any]],
    summary_map: Mapping[tuple[str, str, int], Mapping[str, Any]],
    grouped_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> str:
    lines: list[str] = []
    for dataset_id in dataset_ids:
        for mode in modes:
            native_accuracy = _metric_value(summary_map, grouped_rows, dataset_id, mode, 1536, "accuracy")
            always_pass = _metric_value(summary_map, grouped_rows, dataset_id, mode, 1536, "always_pass_accuracy")
            entry = dataset_map.get(dataset_id, {})
            pass_rate = _to_float(entry.get("pass_rate")) if isinstance(entry, Mapping) else None
            gap_pp = (native_accuracy - always_pass) * 100.0 if native_accuracy is not None and always_pass is not None else None
            lines.append(
                "<tr>"
                f"<td>{_esc(dataset_id)}</td>"
                f"<td>{_esc(mode)}</td>"
                f"<td>{_fmt_pct(pass_rate, digits=4)}</td>"
                f"<td>{_fmt_pct(always_pass, digits=6)}</td>"
                f"<td>{_fmt_pct(native_accuracy, digits=6)}</td>"
                f"<td>{_fmt_pp(gap_pp, digits=3)}</td>"
                "</tr>"
            )
    return (
        "<section id=\"class-imbalance-reference\">"
        "<h2>Class-imbalance reference baseline</h2>"
        "<p>Dense observed reference pass rate can be close to the always-pass baseline. "
        "Do not claim absolute IDW classification utility from compression parity alone. "
        "Small pooled shifts (including a positive 8d delta) may not indicate a stable operational elbow.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Dataset</th><th>Mode</th><th>Dataset pass rate</th><th>Always-pass accuracy (native scope)</th><th>Native IDW accuracy</th><th>IDW - always-pass</th>"
        "</tr></thead><tbody>"
        + "".join(lines)
        + "</tbody></table></div></section>"
    )


def _render_native_counts_table(dataset_ids: list[str], modes: list[str], grouped_rows: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> str:
    rows_html: list[str] = []
    for dataset_id in dataset_ids:
        for mode in modes:
            native = grouped_rows.get((dataset_id, mode, 1536), {})
            selected_total = native.get("selected_count_total")
            unjudged_total = native.get("unjudged_count_total")
            row_count = native.get("row_count")
            concept_cov = _mean(native.get("concept_coverage", []) if isinstance(native.get("concept_coverage"), list) else [])
            agent_cov = _mean(native.get("agent_coverage", []) if isinstance(native.get("agent_coverage"), list) else [])
            prov = native.get("provenance_counts") if isinstance(native, Mapping) else None
            if not isinstance(prov, Mapping):
                prov = {"idw": 0, "exact_match": 0, "global_mean": 0, "prior": 0}
            total_prov = sum(int(prov.get(key, 0)) for key in ("idw", "exact_match", "global_mean", "prior"))
            def _share(key: str) -> str:
                value = int(prov.get(key, 0))
                if total_prov <= 0:
                    return "N/A"
                return f"{value:,d} ({(100.0 * value / total_prov):.1f}%)"

            rows_html.append(
                "<tr>"
                f"<td>{_esc(dataset_id)}</td>"
                f"<td>{_esc(mode)}</td>"
                f"<td>{_fmt_int(_to_int(row_count))}</td>"
                f"<td>{_fmt_int(_to_int(selected_total))}</td>"
                f"<td>{_fmt_int(_to_int(unjudged_total))}</td>"
                f"<td>{_fmt_pct(concept_cov, digits=2)}</td>"
                f"<td>{_fmt_pct(agent_cov, digits=2)}</td>"
                f"<td>{_esc(_share('idw'))}</td>"
                f"<td>{_esc(_share('exact_match'))}</td>"
                f"<td>{_esc(_share('global_mean'))}</td>"
                f"<td>{_esc(_share('prior'))}</td>"
                "</tr>"
            )
    return (
        "<section id=\"counts-coverage\">"
        "<h2>Exact counts, proxy coverage, and fallback shares (native 1536d cells)</h2>"
        "<p>Counts are exact sums across paired cells in the native-dimension rows for each dataset/mode.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Dataset</th><th>Mode</th><th>Paired cells</th><th>Selected count</th><th>Unjudged count</th>"
        "<th>Concept coverage</th><th>Agent coverage</th><th>IDW</th><th>Exact match</th><th>Global mean</th><th>Prior</th>"
        "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div></section>"
    )


def _render_rate_table(dataset_ids: list[str], modes: list[str], rate_breakdown: Mapping[tuple[str, str, float, int], list[float]]) -> str:
    rate_keys = sorted({round(key[2], 6) for key in rate_breakdown})
    all_rates = sorted(set(rate_keys) | set(EXPECTED_RATES))
    rows_html: list[str] = []
    for dataset_id in dataset_ids:
        for mode in modes:
            for rate in all_rates:
                cells: list[str] = []
                for dim in EXPECTED_DIMENSIONS:
                    values = rate_breakdown.get((dataset_id, mode, rate, dim), [])
                    cells.append(f"<td>{_fmt_pp(_mean(values), digits=2)}</td>")
                rows_html.append(
                    "<tr>"
                    f"<td>{_esc(dataset_id)}</td><td>{_esc(mode)}</td><td>{rate:.2f}</td>"
                    + "".join(cells)
                    + "</tr>"
                )
    dim_headers = "".join(f"<th>{dim}d</th>" for dim in EXPECTED_DIMENSIONS)
    return (
        "<section id=\"rate-breakdown\">"
        "<h2>Rate-specific paired accuracy delta (pp) vs native</h2>"
        "<p>Each value is the mean paired delta within the selected dataset/mode/rate/dimension. "
        "Unavailable cells were not produced by the run and are not converted to zero.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Dataset</th><th>Mode</th><th>Rate</th>"
        + dim_headers
        + "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div></section>"
    )


def _render_schedule_table(schedule_breakdown: Mapping[tuple[str, str, str, float], list[float]]) -> str:
    rows = sorted(schedule_breakdown.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]))
    rendered = []
    for (dataset_id, mode, schedule, rate), deltas in rows:
        rendered.append(
            "<tr>"
            f"<td>{_esc(dataset_id)}</td><td>{_esc(mode)}</td><td>{_esc(schedule)}</td>"
            f"<td>{rate:.2f}</td><td>{_fmt_pp(_mean(deltas), digits=2)}</td><td>{len(deltas):,d}</td>"
            "</tr>"
        )
    return (
        "<section id=\"schedule-rate\">"
        "<h2>Schedule x rate cell summary</h2>"
        "<p>Macro means are built across schedule x rate cells first, then across seeds, per the protocol text.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Dataset</th><th>Mode</th><th>Schedule</th><th>Rate</th><th>Mean Δ vs native</th><th>Observed rows</th>"
        "</tr></thead><tbody>"
        + "".join(rendered)
        + "</tbody></table></div></section>"
    )


def _render_worst_agents(native_agents: list[NativeAgentRow]) -> str:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in native_agents:
        key = (row.dataset_id, row.mode, row.agent_id)
        if key not in grouped:
            grouped[key] = {
                "weighted_acc_numerator": 0.0,
                "weighted_acc_denominator": 0,
                "n_total": 0.0,
                "selected_count": 0.0,
                "unjudged_count": 0.0,
                "mae_values": [],
                "provenance_sums": Counter(),
            }
        bucket = grouped[key]
        weight = row.unjudged_count if row.unjudged_count > 0 else row.n
        if row.accuracy is not None and weight > 0:
            bucket["weighted_acc_numerator"] += row.accuracy * weight
            bucket["weighted_acc_denominator"] += weight
        bucket["n_total"] += row.n
        bucket["selected_count"] += row.selected_count
        bucket["unjudged_count"] += row.unjudged_count
        if row.mae is not None:
            bucket["mae_values"].append(row.mae)
        if isinstance(row.provenance_counts, Mapping):
            for name, value in row.provenance_counts.items():
                parsed = _to_float(value)
                if parsed is not None:
                    bucket["provenance_sums"][str(name)] += parsed

    by_scope: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for (dataset_id, mode, agent_id), bucket in grouped.items():
        by_scope[(dataset_id, mode)].append((agent_id, bucket))

    lines: list[str] = []
    for (dataset_id, mode), entries in sorted(by_scope.items(), key=lambda item: item[0]):
        def _score(item: tuple[str, dict[str, Any]]) -> float:
            bucket = item[1]
            denom = bucket["weighted_acc_denominator"]
            if denom <= 0:
                return 1.0
            return bucket["weighted_acc_numerator"] / denom
        agent_id, bucket = min(entries, key=_score)
        denom = bucket["weighted_acc_denominator"]
        accuracy = (bucket["weighted_acc_numerator"] / denom) if denom > 0 else None
        mae = _mean(bucket["mae_values"])
        prov = bucket.get("provenance_sums", Counter())
        prov_total = float(sum(prov.values()))
        if prov_total > 0:
            prov_text = ", ".join(
                f"{name}:{(100.0 * value / prov_total):.1f}%"
                for name, value in sorted(prov.items())
            )
        else:
            prov_text = "N/A"
        lines.append(
            "<tr>"
            f"<td>{_esc(dataset_id)}</td><td>{_esc(mode)}</td><td>{_esc(agent_id)}</td>"
            f"<td>{_fmt_num(accuracy, digits=4)}</td><td>{_fmt_count(bucket['n_total'])}</td>"
            f"<td>{_fmt_count(bucket['selected_count'])}</td><td>{_fmt_count(bucket['unjudged_count'])}</td><td>{_fmt_num(mae, digits=4)}</td>"
            f"<td>{_esc(prov_text)}</td>"
            "</tr>"
        )
    if not lines:
        lines = ["<tr><td colspan=\"9\" class=\"muted\">No per-agent rows present in native dimension payload.</td></tr>"]
    return (
        "<section id=\"worst-agent\">"
        "<h2>Worst native-dimension agent by weighted unjudged accuracy</h2>"
        "<p>Accuracy here is unjudged-only. <code>n</code> is total agent sessions; <code>unjudged</code> is the denominator for the reported accuracy metric.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Dataset</th><th>Mode</th><th>Agent</th><th>Accuracy (unjudged)</th><th>N total</th><th>Selected</th><th>Unjudged</th><th>MAE (unjudged)</th><th>Mean provenance mix</th>"
        "</tr></thead><tbody>"
        + "".join(lines)
        + "</tbody></table></div></section>"
    )


def _render_decisions(decisions: Any, dataset_ids: list[str], modes: list[str]) -> str:
    decision_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    if isinstance(decisions, list):
        for row in decisions:
            if not isinstance(row, Mapping):
                continue
            dataset_id = row.get("dataset_id")
            mode = row.get("mode")
            if isinstance(dataset_id, str) and isinstance(mode, str):
                decision_map[(dataset_id, mode)] = row
    lines: list[str] = []
    for dataset_id in dataset_ids:
        for mode in modes:
            row = decision_map.get((dataset_id, mode))
            if row is None:
                lines.append(
                    "<tr>"
                    f"<td>{_esc(dataset_id)}</td><td>{_esc(mode)}</td><td>N/A</td><td>N/A</td>"
                    "<td>No decision row provided in aggregate.</td></tr>"
                )
                continue
            lines.append(
                "<tr>"
                f"<td>{_esc(dataset_id)}</td>"
                f"<td>{_esc(mode)}</td>"
                f"<td>{_fmt_int(_to_int(row.get('smallest_non_degrading_dimension')))}</td>"
                f"<td>{_fmt_int(_to_int(row.get('one_pp_candidate_dimension')))}</td>"
                f"<td>{_esc(row.get('reason', 'N/A'))}</td>"
                "</tr>"
            )
    return (
        "<section id=\"decisions\">"
        "<h2>Decision table from aggregate artifact</h2>"
        "<p>No universal winner is claimed. Decisions stay scoped to dataset, mode, and measured labels. "
        "For one-pp candidates, interpretation is pointwise seed CI and requires the candidate plus all larger tested prefixes to pass.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Dataset</th><th>Mode</th><th>Smallest non-degrading d</th><th>One-pp candidate d</th><th>Reason</th>"
        "</tr></thead><tbody>"
        + "".join(lines)
        + "</tbody></table></div></section>"
    )


def _render_curve_findings(
    dataset_ids: list[str],
    modes: list[str],
    summary_map: Mapping[tuple[str, str, int], Mapping[str, Any]],
    grouped_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    decisions: Any,
) -> str:
    decision_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    if isinstance(decisions, list):
        for row in decisions:
            if not isinstance(row, Mapping):
                continue
            dataset_id = row.get("dataset_id")
            mode = row.get("mode")
            if isinstance(dataset_id, str) and isinstance(mode, str):
                decision_map[(dataset_id, mode)] = row

    cards: list[str] = []
    dims_desc = list(reversed(EXPECTED_DIMENSIONS))
    for dataset_id in dataset_ids:
        for mode in modes:
            points: list[tuple[int, float]] = []
            for dim in dims_desc:
                value = _metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "accuracy")
                if value is not None:
                    points.append((dim, value))
            if not points:
                cards.append(
                    "<article class=\"dataset-mode-card\">"
                    f"<h3>{_esc(dataset_id)} / {_esc(mode)}</h3>"
                    "<p class=\"muted\">No summary accuracy points available for this scope.</p>"
                    "</article>"
                )
                continue

            increases = 0
            decreases = 0
            for idx in range(len(points) - 1):
                current = points[idx][1]
                nxt = points[idx + 1][1]
                if nxt > current + 1e-12:
                    increases += 1
                elif nxt < current - 1e-12:
                    decreases += 1
            non_monotonic = increases > 0 and decreases > 0

            native = next((value for dim, value in points if dim == 1536), None)
            d8 = next((value for dim, value in points if dim == 8), None)
            if native is not None and d8 is not None:
                d8_delta = (d8 - native) * 100.0
                magnitude_note = "This is a small pooled shift." if abs(d8_delta) < 0.5 else "This is a larger pooled shift."
                d8_text = (
                    f"8d point is shown explicitly: {d8 * 100.0:.4f}% "
                    f"({d8_delta:+.3f} pp vs native). "
                    f"{magnitude_note} This alone does not establish a stable elbow or monotonic no-loss."
                )
            else:
                d8_text = "8d point unavailable for this scope."

            best_dim, best_acc = max(points, key=lambda pair: pair[1])
            decision = decision_map.get((dataset_id, mode), {})
            smallest = _fmt_int(_to_int(decision.get("smallest_non_degrading_dimension")))
            candidate = _fmt_int(_to_int(decision.get("one_pp_candidate_dimension")))
            if non_monotonic:
                shape_text = "Curve shape is non-monotonic across tested prefixes."
            elif increases > 0:
                shape_text = "Curve rises as dimensions shrink in every observed step."
            elif decreases > 0:
                shape_text = "Curve declines as dimensions shrink in every observed step."
            else:
                shape_text = "Curve is flat across observed points."

            points_text = " | ".join(f"{dim}:{value * 100.0:.4f}%" for dim, value in points)
            cards.append(
                "<article class=\"dataset-mode-card\">"
                f"<h3>{_esc(dataset_id)} / {_esc(mode)}</h3>"
                f"<p><strong>{_esc(shape_text)}</strong> Best pooled point: {best_dim}d at {best_acc * 100.0:.4f}%.</p>"
                f"<p>{_esc(d8_text)}</p>"
                f"<p>Decision rows: smallest contiguous pooled non-degrading prefix = <strong>{smallest}</strong>; "
                f"exploratory 1pp seed-interval candidate = <strong>{candidate}</strong>.</p>"
                f"<p class=\"muted\">Accuracy by dimension (1536→8): {points_text}</p>"
                "</article>"
            )
    return "<section id=\"curve-findings\"><h2>Curve shape and cutoff interpretation</h2>" + "".join(cards) + "</section>"


def _curve_points_for_scope(
    dataset_id: str,
    mode: str,
    summary_map: Mapping[tuple[str, str, int], Mapping[str, Any]],
    grouped_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for dim in reversed(EXPECTED_DIMENSIONS):
        value = _metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "accuracy")
        if value is not None:
            points.append((dim, value))
    return points


def _curve_is_non_monotonic(points: list[tuple[int, float]]) -> bool:
    increases = 0
    decreases = 0
    for idx in range(len(points) - 1):
        current = points[idx][1]
        nxt = points[idx + 1][1]
        if nxt > current + 1e-12:
            increases += 1
        elif nxt < current - 1e-12:
            decreases += 1
    return increases > 0 and decreases > 0


def _render_leading_takeaway(
    aggregate: Mapping[str, Any],
    dataset_map: Mapping[str, Mapping[str, Any]],
    summary_map: Mapping[tuple[str, str, int], Mapping[str, Any]],
    grouped_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    decisions: Any,
    readiness: Mapping[str, Any] | None,
) -> str:
    dataset_statuses = []
    for dataset_id in REQUIRED_DATASETS:
        entry = dataset_map.get(dataset_id)
        if not isinstance(entry, Mapping):
            dataset_statuses.append((dataset_id, "blocked", "missing verified input bundle"))
            continue
        dataset_statuses.append(
            (
                dataset_id,
                str(entry.get("status", "unknown")),
                str(entry.get("reason", "")) if entry.get("reason") is not None else "",
            )
        )
    completed = sum(status == "completed" for _, status, _ in dataset_statuses)
    total = len(REQUIRED_DATASETS)

    focus_dataset = "dense_2500"
    focus_mode = "end_to_end"
    points = _curve_points_for_scope(focus_dataset, focus_mode, summary_map, grouped_rows)
    if not points:
        for dataset_id, _, _ in dataset_statuses:
            for mode in EXPECTED_MODES:
                points = _curve_points_for_scope(dataset_id, mode, summary_map, grouped_rows)
                if points:
                    focus_dataset = dataset_id
                    focus_mode = mode
                    break
            if points:
                break

    decision_row: Mapping[str, Any] = {}
    if isinstance(decisions, list):
        for item in decisions:
            if not isinstance(item, Mapping):
                continue
            if item.get("dataset_id") == focus_dataset and item.get("mode") == focus_mode:
                decision_row = item
                break

    native = next((value for dim, value in points if dim == 1536), None)
    d512 = next((value for dim, value in points if dim == 512), None)
    d8 = next((value for dim, value in points if dim == 8), None)
    if native is not None and d512 is not None and d8 is not None:
        metric_line = (
            f"{focus_dataset} {focus_mode}: native {native * 100.0:.2f}%, "
            f"512d {d512 * 100.0:.2f}%, 8d {d8 * 100.0:.2f}%."
        )
    elif native is not None and d512 is not None:
        metric_line = (
            f"{focus_dataset} {focus_mode}: native {native * 100.0:.2f}%, "
            f"512d {d512 * 100.0:.2f}%."
        )
    else:
        metric_line = f"{focus_dataset} {focus_mode}: accuracy points unavailable for one or more key dimensions."

    shape_note = (
        "Curve is non-monotonic; no validated universal elbow/no-loss claim."
        if _curve_is_non_monotonic(points)
        else "Curve shape still does not validate a universal elbow/no-loss claim."
    )
    smallest = _fmt_int(_to_int(decision_row.get("smallest_non_degrading_dimension"))) if decision_row else "N/A"
    candidate = _fmt_int(_to_int(decision_row.get("one_pp_candidate_dimension"))) if decision_row else "N/A"
    decision_note = (
        f"Decision row: smallest contiguous pooled non-degrading prefix {smallest}, "
        f"exploratory 1pp seed-interval candidate {candidate}."
    )

    auth_signal = ""
    tau_signal = ""
    readiness_blocker = str(readiness.get("authentication_blocker", "")).lower() if isinstance(readiness, Mapping) else ""
    if "aadsts50020" in readiness_blocker or ("tenant" in readiness_blocker and "token" in readiness_blocker):
        auth_signal = "auth blocker remains for non-dense preparation"
    tau_reason = next((reason.lower() for dataset_id, _, reason in dataset_statuses if dataset_id == "tau2_bench"), "")
    if "expected-label" in tau_reason or "expected label" in tau_reason or "mapping" in tau_reason:
        tau_signal = "tau2 expected-label mapping remains blocked"
        if auth_signal:
            tau_signal += " (plus auth)"
    blocker_note = "; ".join(part for part in (auth_signal, tau_signal) if part)
    if blocker_note:
        blocker_note += "."
    else:
        blocker_note = "Blocked dataset readiness remains explicit in profile reasons."

    return (
        "<p class=\"headline-takeaway\"><strong>Qualified takeaway:</strong> "
        f"Run status {_esc(str(aggregate.get('status', 'unknown')))} "
        f"({completed}/{total} required datasets completed). "
        f"{_esc(metric_line)} {_esc(shape_note)} {_esc(decision_note)} {_esc(blocker_note)}"
        "</p>"
    )


def _render_charts(
    dataset_ids: list[str],
    modes: list[str],
    summary_map: Mapping[tuple[str, str, int], Mapping[str, Any]],
    grouped_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    rate_breakdown: Mapping[tuple[str, str, float, int], list[float]],
) -> str:
    cards: list[str] = []
    chart_index = 0
    palette = ["#005a9c", "#d83b01", "#107c10", "#5c2d91", "#008575", "#a4262c"]
    all_rates = sorted(set(EXPECTED_RATES) | {key[2] for key in rate_breakdown})

    for dataset_id in dataset_ids:
        for mode in modes:
            if dataset_id not in REQUIRED_DATASETS and dataset_id.strip() == "":
                continue
            accuracy_values = {
                dim: _metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "accuracy")
                for dim in EXPECTED_DIMENSIONS
            }
            delta_values = {
                dim: _pp_delta(_metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "accuracy_delta_native"))
                for dim in EXPECTED_DIMENSIONS
            }
            combined_values = {
                dim: _metric_value(summary_map, grouped_rows, dataset_id, mode, dim, "combined_accuracy")
                for dim in EXPECTED_DIMENSIONS
            }
            rate_series: list[tuple[str, dict[int, float | None], str]] = []
            for idx, rate in enumerate(all_rates):
                dim_values: dict[int, float | None] = {}
                for dim in EXPECTED_DIMENSIONS:
                    dim_values[dim] = _mean(rate_breakdown.get((dataset_id, mode, rate, dim), []))
                rate_series.append((f"rate {rate:.2f}", dim_values, palette[idx % len(palette)]))

            card_id = _slugify(f"{dataset_id}-{mode}")
            chart_index += 1
            primary_svg = _render_multi_series_svg(
                chart_id=_unique_id(card_id, chart_index),
                title=f"{dataset_id} / {mode} unjudged-only accuracy by dimension",
                description=(
                    "Primary accuracy curve across prefix dimensions. Missing points indicate unavailable cells, not zeros."
                ),
                y_label="Unjudged-only accuracy (fraction)",
                series=[("accuracy", accuracy_values, "#005a9c")],
                y_min=0.0,
                y_max=1.0,
                value_formatter=lambda v: f"{v:.3f}",
            )
            chart_index += 1
            delta_svg = _render_multi_series_svg(
                chart_id=_unique_id(card_id, chart_index),
                title=f"{dataset_id} / {mode} paired accuracy delta vs native",
                description="Paired delta in percentage points relative to native 1536d.",
                y_label="Delta vs native (pp)",
                series=[("paired Δ", delta_values, "#d83b01")],
                zero_line=True,
                value_formatter=lambda v: f"{v:.2f}",
            )
            chart_index += 1
            rate_svg = _render_multi_series_svg(
                chart_id=_unique_id(card_id, chart_index),
                title=f"{dataset_id} / {mode} rate-specific delta breakdown",
                description="Each line shows paired delta vs native for one sample rate.",
                y_label="Delta vs native (pp)",
                series=rate_series,
                zero_line=True,
                value_formatter=lambda v: f"{v:.2f}",
            )
            chart_index += 1
            combined_svg = _render_multi_series_svg(
                chart_id=_unique_id(card_id, chart_index),
                title=f"{dataset_id} / {mode} combined accuracy (judged + imputed)",
                description="Combined view includes direct labels and is shown separately from primary unjudged-only accuracy.",
                y_label="Combined accuracy (fraction)",
                series=[("combined", combined_values, "#107c10")],
                y_min=0.0,
                y_max=1.0,
                value_formatter=lambda v: f"{v:.3f}",
            )
            cards.append(
                "<article class=\"dataset-mode-card\">"
                f"<h3>{_esc(dataset_id)} / {_esc(mode)}</h3>"
                "<p><strong>What this visual shows:</strong> accuracy retention under Matryoshka truncation.</p>"
                "<p><strong>How to read it:</strong> left is 8d, right is native 1536d. Missing points are unavailable cells.</p>"
                f"<p><strong>{_esc(_takeaway_line(dataset_id, mode, summary_map, grouped_rows))}</strong></p>"
                "<div class=\"small-multiples\">"
                f"<figure><div class=\"chart-scroll\">{primary_svg}</div><figcaption>Primary unjudged-only accuracy vs dimension.</figcaption></figure>"
                f"<figure><div class=\"chart-scroll\">{delta_svg}</div><figcaption>Paired accuracy delta (percentage points) vs native.</figcaption></figure>"
                f"<figure><div class=\"chart-scroll\">{rate_svg}</div><figcaption>Rate-specific paired delta breakdown.</figcaption></figure>"
                f"<figure><div class=\"chart-scroll\">{combined_svg}</div><figcaption>Direct-label mixed combined-accuracy view (diagnostic only).</figcaption></figure>"
                "</div></article>"
            )
    return "<section id=\"charts\"><h2>Accessible small-multiple charts</h2>" + "".join(cards) + "</section>"


def _render_methodology(protocol: Mapping[str, Any]) -> str:
    dims = protocol.get("dimensions", [])
    seeds = protocol.get("seeds", [])
    rates = protocol.get("rates", [])
    schedules = protocol.get("schedules", [])
    modes = protocol.get("modes", [])
    idw = protocol.get("idw", {})
    return (
        "<section id=\"methods\">"
        "<h2>Method scope, boundaries, and scientific notes</h2>"
        "<ul>"
        "<li>Native baseline: full 1536d embedding. Truncation arm uses first-d prefix with L2 re-normalization. No fitted transform is used.</li>"
        "<li>Representation/sampling context: original ARM2 diversity sampler via existing AdaptiveSampler + AzureClusterIndex + local InMemoryVectorStore; tau=0.55, TTL=90, horizon=10000, no floors, high-throughput setup.</li>"
        "<li>Native/reject novelty-rarity rank uses hashed-cap ranking. Causal donor scoring uses same-agent angular distance arccos(cos)/pi, k=8, power=2, epsilon=1e-6, exact when 1-cos&lt;=1e-8, then mean over exact donors, else earlier global mean, else prior.</li>"
        "<li>Selection ranks the full unlabeled schedule and is <strong>not</strong> a fully causal online admission process, even though donor scoring itself is causal.</li>"
        "<li>Five schedules are expected (typically evenly_spaced, uniformly_random, bursty, front_loaded, agent_blocked; some artifacts may use even/random aliases), with 30 paired order seeds 13..42 and rates 0.01/0.02/0.05/0.10/0.20. Count rule is max(1, floor(N*rate)). No bootstrap refresh and no fresh labels.</li>"
        "<li>Boundary clarification: preparation may call live embedding APIs to build input caches (recorded in provenance/profile), while the dimensionality sweep executes offline from those prepared vectors.</li>"
        "<li>Means are macro across five schedule x rate cells and then seeds. Variability comes from re-ordering over same labels, not population generalization.</li>"
        "<li><code>end_to_end</code> is the primary mode; <code>fixed_membership</code> is diagnostic and reuses native membership for all dimensions. End-to-end target sets can differ by dimension.</li>"
        "<li>Uniform accuracy threshold is 0.5. Combined judged+imputed metrics are shown separately because observed labels inflate them.</li>"
        "<li>Illustrative IDW (not run result): donors at distance 1(label=1) and 2(label=0) with inverse-square weights yield normalized weights 0.8/0.2, estimate 0.8, binary prediction 1 at threshold 0.5.</li>"
        "<li>Business-use-case classification and the weekly Lipschitz gate are not measured in this experiment artifact.</li>"
        "</ul>"
        "<div class=\"table-wrap\"><table><thead><tr><th>Protocol field</th><th>Value from aggregate</th></tr></thead><tbody>"
        f"<tr><td>dimensions</td><td><code>{_safe_json(dims)}</code></td></tr>"
        f"<tr><td>seeds</td><td><code>{_safe_json(seeds)}</code></td></tr>"
        f"<tr><td>rates</td><td><code>{_safe_json(rates)}</code></td></tr>"
        f"<tr><td>schedules</td><td><code>{_safe_json(schedules)}</code></td></tr>"
        f"<tr><td>modes</td><td><code>{_safe_json(modes)}</code></td></tr>"
        f"<tr><td>IDW config</td><td><code>{_safe_json(idw)}</code></td></tr>"
        "</tbody></table></div>"
        "<p>References: "
        "<a href=\"https://arxiv.org/html/2205.13147v4\">Matryoshka Representation Learning paper</a>; "
        "<a href=\"https://developers.openai.com/api/docs/guides/embeddings\">OpenAI embeddings docs</a>. "
        "Important interpretation notes: model shortening requires normalization; 256d comparison is 3-large vs ada-002 (not 3-small); no 8d lossless guarantee. API billing is input tokens, while modeled storage is float32 N*d*4 bytes."
        "</p>"
        "</section>"
    )


def build_report(aggregate: Mapping[str, Any], aggregate_path: str) -> str:
    if not isinstance(aggregate, Mapping):
        raise TypeError("aggregate must be a mapping")

    datasets = _dataset_rows(aggregate.get("datasets"))
    rows_raw = aggregate.get("rows")
    rows: list[Mapping[str, Any]] = [row for row in rows_raw if isinstance(row, Mapping)] if isinstance(rows_raw, list) else []
    summary_map = _build_summary_map(aggregate.get("summary"))
    grouped_rows, rate_breakdown, schedule_breakdown, native_agents_fallback = _build_grouped_metrics(rows)
    native_agents = _collect_native_agents(aggregate.get("agent_summary"), native_agents_fallback)
    warnings = _collect_warnings(aggregate, datasets, rows)

    dataset_ids = list(REQUIRED_DATASETS) + [x for x in sorted(datasets.keys()) if x not in REQUIRED_DATASETS]
    modes = list(EXPECTED_MODES)
    for _, mode, _ in grouped_rows.keys():
        if mode not in modes:
            modes.append(mode)

    run_id = aggregate.get("run_id", "unknown-run")
    source_revision = aggregate.get("source_revision", "unavailable")
    generated_at = aggregate.get("generated_at", "unavailable")
    status = aggregate.get("status", "unknown")
    protocol = aggregate.get("protocol") if isinstance(aggregate.get("protocol"), Mapping) else {}
    files = aggregate.get("files") if isinstance(aggregate.get("files"), Mapping) else {}
    validation = aggregate.get("validation") if isinstance(aggregate.get("validation"), Mapping) else {}
    readiness, readiness_error = _load_input_readiness(aggregate, aggregate_path)
    leading_takeaway = _render_leading_takeaway(
        aggregate,
        datasets,
        summary_map,
        grouped_rows,
        aggregate.get("decisions"),
        readiness,
    )

    warning_html = (
        "<ul class=\"warnings\">"
        + "".join(f"<li>{_esc(item)}</li>" for item in warnings)
        + "</ul>"
        if warnings
        else "<p class=\"ok\">No structural warnings detected in aggregate metadata.</p>"
    )

    html_out = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>Matryoshka Prefix Truncation Report — {_esc(run_id)}</title>"
        "<style>"
        ":root{color-scheme:light;--bg:#f8f9fb;--ink:#1a1a1a;--muted:#5a5a5a;--card:#ffffff;--line:#d9dce3;--accent:#005a9c;}"
        "html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);font-family:Segoe UI,Arial,sans-serif;line-height:1.45;}"
        "main{max-width:1200px;margin:0 auto;padding:1rem;}"
        "header{background:#1f2937;color:#fff;padding:1rem;}"
        "header h1{margin:0 0 .4rem 0;font-size:1.5rem;}"
        ".status-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.75rem;margin:.75rem 0;}"
        ".status-card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.75rem;min-width:0;}"
        ".status-card strong{display:block;font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.03em;}"
        ".status-card span{display:block;overflow-wrap:anywhere;word-break:break-word;}"
        ".headline-takeaway{margin:.55rem 0 .4rem 0;background:#ecf5ff;border:1px solid #c6ddff;border-radius:8px;padding:.55rem .7rem;color:#0f2d4d;}"
        "section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1rem;margin:0 0 1rem 0;}"
        ".warnings{margin:.25rem 0 0 1.1rem;color:#8a2a0d;}"
        ".ok{color:#107c10;}"
        "h2{margin-top:0;font-size:1.25rem;}"
        "h3{margin:.6rem 0;font-size:1.05rem;}"
        "p,li{overflow-wrap:anywhere;}"
        ".table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;max-width:100%;}"
        "table{border-collapse:collapse;width:100%;min-width:720px;background:#fff;}"
        "th,td{border-bottom:1px solid var(--line);padding:.45rem .5rem;text-align:left;vertical-align:top;font-size:.92rem;}"
        "th{background:#f2f4f8;position:sticky;top:0;z-index:1;}"
        ".muted{color:var(--muted);}"
        ".small-multiples{display:grid;grid-template-columns:minmax(0,1fr);gap:.9rem;align-items:start;min-width:0;}"
        ".small-multiples>*{min-width:0;}"
        "figure{margin:0;padding:.5rem;border:1px solid var(--line);border-radius:8px;background:#fff;min-width:0;max-width:100%;}"
        "figcaption{font-size:.88rem;color:var(--muted);margin-top:.45rem;white-space:normal;overflow-wrap:anywhere;}"
        ".chart-scroll{max-width:100%;width:100%;min-width:0;overflow-x:auto;overflow-y:hidden;}"
        ".chart{width:100%;min-width:720px;max-width:960px;height:auto;display:block;}"
        ".axis{stroke:#4b5563;stroke-width:1;}"
        ".grid{stroke:#e5e7eb;stroke-width:1;}"
        ".axis-label{font-size:11px;fill:#374151;}"
        ".direct-label{font-size:11px;font-weight:600;paint-order:stroke;stroke:white;stroke-width:2px;}"
        ".zero-line{stroke:#6b7280;stroke-dasharray:4 4;stroke-width:1.4;}"
        ".dataset-mode-card{border:1px solid var(--line);border-radius:8px;padding:.75rem;margin:.6rem 0;min-width:0;}"
        ".dataset-detail summary{cursor:pointer;font-weight:600;}"
        ".detail-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.75rem;margin-top:.5rem;}"
        "pre{white-space:pre-wrap;word-break:break-word;background:#f4f6fb;padding:.5rem;border-radius:6px;border:1px solid #e5e7eb;}"
        ".skip-link{position:absolute;left:-9999px;top:auto;}"
        ".skip-link:focus{left:10px;top:10px;background:#fff;color:#000;padding:.5rem .75rem;border-radius:8px;z-index:999;}"
        "a:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:2px;}"
        ".empty-state{border:1px dashed #9ca3af;padding:1rem;border-radius:8px;color:#4b5563;background:#f8fafc;}"
        "@media (max-width:700px){table{min-width:640px;}header h1{font-size:1.25rem;}.chart{min-width:680px;}}"
        "</style></head><body>"
        "<a href=\"#main\" class=\"skip-link\">Skip to main content</a>"
        "<header><h1>Matryoshka Prefix Truncation Report</h1>"
        + leading_takeaway
        + "<p>Decision scope: prefix dimension cuts for sampling accuracy with explicit end_to_end primary mode and fixed_membership diagnostic mode.</p>"
        "</header>"
        "<main id=\"main\" tabindex=\"-1\">"
        "<section id=\"source-status\">"
        "<h2>Source and boundary status</h2>"
        "<div class=\"status-grid\">"
        f"<div class=\"status-card\"><strong>Aggregate path</strong><span>{_esc(aggregate_path)}</span></div>"
        f"<div class=\"status-card\"><strong>Version</strong><span>{_esc(aggregate.get('version', 'unknown'))}</span></div>"
        f"<div class=\"status-card\"><strong>Run ID</strong><span>{_esc(run_id)}</span></div>"
        f"<div class=\"status-card\"><strong>Generated at</strong><span>{_esc(generated_at)}</span></div>"
        f"<div class=\"status-card\"><strong>Status</strong><span>{_esc(status)}</span></div>"
        f"<div class=\"status-card\"><strong>Source revision</strong><span>{_esc(source_revision)}</span></div>"
        "</div>"
        "<p>Expected production output is all four datasets. Any blocked/missing dataset keeps conclusions conditional and explicitly scoped.</p>"
        + warning_html
        + "</section>"
        + _render_input_readiness(readiness, readiness_error)
        + _render_label_boundary_callout(readiness)
        + _render_embedding_preparation_status(datasets)
        + _render_methodology(protocol)
        + _render_dataset_inventory(datasets, rows)
        + _render_curve_findings(dataset_ids, modes, summary_map, grouped_rows, aggregate.get("decisions"))
        + _render_charts(dataset_ids, modes, summary_map, grouped_rows, rate_breakdown)
        + _render_metric_matrix(dataset_ids, modes, summary_map, grouped_rows)
        + _render_class_imbalance_reference(dataset_ids, modes, datasets, summary_map, grouped_rows)
        + _render_seed_ci_table(dataset_ids, modes, summary_map, grouped_rows)
        + _render_native_counts_table(dataset_ids, modes, grouped_rows)
        + _render_rate_table(dataset_ids, modes, rate_breakdown)
        + _render_schedule_table(schedule_breakdown)
        + _render_worst_agents(native_agents)
        + _render_decisions(aggregate.get("decisions"), dataset_ids, modes)
        + "<section id=\"repro\">"
        "<h2>Reproducibility and limitations</h2>"
        "<p>This report is static HTML with inline CSS/SVG, no external CDN, and no client-side controls. "
        "Interpretation boundaries: same-label resampling only; no new judge labels; no production-causality guarantee; no business-use-case classifier gate measurement. "
        "Preparation may include fresh embedding calls; sweep cells themselves are offline computations over prepared vectors.</p>"
        "<div class=\"detail-grid\">"
        "<div><h3>Files from aggregate</h3><pre>"
        + _safe_json(files)
        + "</pre></div>"
        "<div><h3>Validation block from aggregate</h3><pre>"
        + _safe_json(validation)
        + "</pre></div>"
        "</div>"
        "</section>"
        "</main></body></html>"
    )
    return html_out
