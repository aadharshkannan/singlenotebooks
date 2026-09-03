from __future__ import annotations

import json
import math
import os
import shutil
import subprocess

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote
from typing import Any, Sequence

REPORT_VERSION = "sampling-v6-report-v3"
DEFAULT_OUTPUT_NAME = "sampling-v6-report.html"
DEFAULT_INPUT_DIR = Path("outputs_sampling_v6") / "runs"

V6_BUNDLE_VERSION = "sampling-v6-bundle-v2"
V6_MANIFEST_VERSION = "sampling-v6-manifest-v2"
SUPPORTED_V6_BUNDLE_VERSIONS = ("sampling-v6-bundle-v1", "sampling-v6-bundle-v2")
SUPPORTED_V6_MANIFEST_VERSIONS = ("sampling-v6-manifest-v1", "sampling-v6-manifest-v2")

FULL_METHOD_IDS = (
    "arm1_global_random",
    "arm2_embedding_idw",
    "arm2_5_embedding_idw_binary",
    "arm3_agent_round_robin_floor",
    "arm4_agent_round_robin",
    "arm5_hajek_weighted",
    "arm6_agent_use_case_hajek",
)
SHORT_TO_FULL = {
    "arm1": "arm1_global_random",
    "arm2": "arm2_embedding_idw",
    "arm2.5": "arm2_5_embedding_idw_binary",
    "arm25": "arm2_5_embedding_idw_binary",
    "arm2_5": "arm2_5_embedding_idw_binary",
    "arm3": "arm3_agent_round_robin_floor",
    "arm4": "arm4_agent_round_robin",
    "arm5": "arm5_hajek_weighted",
    "arm6": "arm6_agent_use_case_hajek",
}
METHOD_DISPLAY = {
    "arm1_global_random": "ARM1 Global Random",
    "arm2_embedding_idw": "ARM2 Embedding IDW",
    "arm2_5_embedding_idw_binary": "ARM2.5 Embedding IDW Binary",
    "arm3_agent_round_robin_floor": "ARM3 Agent Round Robin Floor",
    "arm4_agent_round_robin": "ARM4 Agent Round Robin",
    "arm5_hajek_weighted": "ARM5 Hajek Weighted",
    "arm6_agent_use_case_hajek": "ARM6 Agent Use-Case Hajek",
}
METHOD_COLOR = {
    "arm1_global_random": "#8a5f3c",
    "arm2_embedding_idw": "#2c6f93",
    "arm2_5_embedding_idw_binary": "#6d4fb3",
    "arm3_agent_round_robin_floor": "#2f8f66",
    "arm4_agent_round_robin": "#c5792b",
    "arm5_hajek_weighted": "#b24c3b",
    "arm6_agent_use_case_hajek": "#b0487d",
}

METRIC_ALIASES = {
    "absolute_aggregate_mae": ("absolute_aggregate_mae", "mae"),
    "concept_coverage": ("concept_coverage",),
    "use_case_coverage": ("use_case_coverage", "maven_coverage"),
    "agent_coverage": ("agent_coverage",),
}

TOKEN_ACTUAL_ALIASES = ("actual_token_count", "actual_tokens")
TOKEN_NOMINAL_ALIASES = ("nominal_budget",)

DEFAULT_BROWSERS = (
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
)


@dataclass(frozen=True)
class V6ReportInputs:
    aggregate: Path
    runs_jsonl: Path
    memberships: Path | None = None
    classifications: Path | None = None
    agent_metrics: Path | None = None
    dataset_examples: Path | None = None
    methodology: Path | None = None
    manifest: Path | None = None


@dataclass(frozen=True)
class LoadedV6Artifacts:
    aggregate: dict[str, Any]
    runs: list[dict[str, Any]]
    memberships: list[dict[str, Any]]
    classifications: list[dict[str, Any]]
    agent_metrics: list[dict[str, Any]]
    dataset_examples: dict[str, Any]
    methodology_text: str
    manifest: dict[str, Any]


def default_inputs(base_dir: Path) -> V6ReportInputs:
    return V6ReportInputs(
        aggregate=base_dir / "aggregate.json",
        runs_jsonl=base_dir / "runs.jsonl",
        memberships=base_dir / "memberships.jsonl",
        classifications=base_dir / "classifications.jsonl",
        agent_metrics=base_dir / "agent_metrics.jsonl",
        dataset_examples=base_dir / "dataset_examples.json",
        methodology=base_dir / "methodology.md",
        manifest=base_dir / "manifest.json",
    )


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required report input not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required report input not found: {path}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected object on line {index + 1} of {path}")
        rows.append(payload)
    return rows


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"expected JSON object at {path}")


def _load_optional_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return _read_jsonl(path)


def _load_optional_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_method_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "unknown"
    if raw in FULL_METHOD_IDS:
        return raw
    if raw in SHORT_TO_FULL:
        return SHORT_TO_FULL[raw]
    low = raw.lower()
    if low in SHORT_TO_FULL:
        return SHORT_TO_FULL[low]
    return raw


def _method_label(method_id: str) -> str:
    return METHOD_DISPLAY.get(method_id, method_id.replace("_", " ").title())


def _method_color(method_id: str) -> str:
    return METHOD_COLOR.get(method_id, "#324353")


def _linear_interpolate_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    if quantile <= 0.0:
        return ordered[0]
    if quantile >= 1.0:
        return ordered[-1]
    position = quantile * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    if lower_index == upper_index:
        return lower_value
    fraction = position - lower_index
    return lower_value + (upper_value - lower_value) * fraction


def _rich_stat_block(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "sample_std": 0.0, "p05": 0.0, "p95": 0.0, "count": 0, "min": 0.0, "max": 0.0}
    values_float = [float(v) for v in values]
    count = len(values_float)
    sample_std = 0.0 if count < 2 else float(np.std(values_float, ddof=1)) if 'np' in globals() else 0.0
    return {
        "mean": float(sum(values_float) / count),
        "median": float(_linear_interpolate_quantile(values_float, 0.50)),
        "sample_std": sample_std,
        "p05": float(_linear_interpolate_quantile(values_float, 0.05)),
        "p95": float(_linear_interpolate_quantile(values_float, 0.95)),
        "count": int(count),
        "min": float(min(values_float)),
        "max": float(max(values_float)),
    }


def _normalize_metric_value(row: dict[str, Any], canonical_metric: str) -> dict[str, float]:
    aliases = METRIC_ALIASES[canonical_metric]
    raw = _first_present(row, *aliases)
    if isinstance(raw, dict):
        stats = {
            "mean": _safe_float(_first_present(raw, "mean", "value", aliases[0]), default=0.0),
            "min": _safe_float(_first_present(raw, "min", "lower"), default=0.0),
            "max": _safe_float(_first_present(raw, "max", "upper"), default=0.0),
        }
        for field in ("median", "sample_std", "p05", "p95"):
            if raw.get(field) is not None:
                stats[field] = _safe_float(raw.get(field), default=stats["mean"])
        if raw.get("count") is not None:
            stats["count"] = _safe_int(raw.get("count"), default=0)
        values = []
        for key in ("mean", "min", "max"):
            if key in raw:
                values.append(_safe_float(raw.get(key), default=0.0))
        if "median" not in stats:
            stats["median"] = float(_linear_interpolate_quantile(values, 0.50)) if values else stats["mean"]
        if "sample_std" not in stats:
            stats["sample_std"] = 0.0 if len(values) < 2 else float(np.std(values, ddof=1))
        if "p05" not in stats:
            stats["p05"] = float(_linear_interpolate_quantile(values, 0.05)) if values else stats["min"]
        if "p95" not in stats:
            stats["p95"] = float(_linear_interpolate_quantile(values, 0.95)) if values else stats["max"]
        if "count" not in stats:
            stats["count"] = 1
        return stats
    value = _safe_float(raw, default=0.0)
    return {"mean": value, "median": value, "sample_std": 0.0, "p05": value, "p95": value, "count": 1, "min": value, "max": value}


def _normalize_stat_scalar(raw: Any, *, alias: str | None = None) -> float:
    if isinstance(raw, dict):
        value = _first_present(raw, "mean", "value", alias or "")
        return _safe_float(value, default=0.0)
    return _safe_float(raw, default=0.0)


def _normalize_stat_int(raw: Any, *, alias: str | None = None) -> int:
    return int(round(_normalize_stat_scalar(raw, alias=alias)))


def _normalize_aggregate_rows(payload: dict[str, Any], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_rows: list[dict[str, Any]] = []
    for key in ("methods", "aggregate_rows", "rows"):
        candidate = payload.get(key)
        if isinstance(candidate, list) and candidate:
            source_rows = [dict(x) for x in candidate if isinstance(x, dict)]
            break

    if not source_rows:
        source_rows = _derive_aggregate_from_runs(runs)

    out: list[dict[str, Any]] = []
    for row in source_rows:
        method_id = _normalize_method_id(_first_present(row, "method_id", "method", "method_name"))
        cap = _safe_int(_first_present(row, "cap", "sample_size", "sample_cap"), default=0)
        selected_count = _normalize_stat_int(_first_present(row, "selected_count", "sample_size", "n"), alias="selected_count")
        if selected_count <= 0:
            selected_count = cap
        actual_tokens_value = _first_present(row, *TOKEN_ACTUAL_ALIASES)
        nominal_budget_value = _first_present(row, *TOKEN_NOMINAL_ALIASES)

        metrics = {metric: _normalize_metric_value(row, metric) for metric in METRIC_ALIASES}
        for metric in METRIC_ALIASES:
            metric_value = metrics[metric]
            if "median" not in metric_value:
                metric_value["median"] = metric_value.get("mean", 0.0)
            if "sample_std" not in metric_value:
                metric_value["sample_std"] = 0.0
            if "p05" not in metric_value:
                metric_value["p05"] = metric_value.get("min", metric_value.get("mean", 0.0))
            if "p95" not in metric_value:
                metric_value["p95"] = metric_value.get("max", metric_value.get("mean", 0.0))
            if "count" not in metric_value:
                metric_value["count"] = 1

        normalized = {
            "method_id": method_id,
            "method_label": _method_label(method_id),
            "color": _method_color(method_id),
            "cap": cap,
            "selected_count": selected_count,
            "metrics": metrics,
            "actual_token_count": _normalize_stat_scalar(actual_tokens_value, alias="actual_token_count"),
            "nominal_budget": _normalize_stat_scalar(nominal_budget_value, alias="nominal_budget"),
            "cap_replays": _safe_int(_first_present(row, "cap_replays", "replays"), default=0),
        }
        out.append(normalized)

    if runs:
        derived = _derive_aggregate_from_runs(runs)
        by_key = {(row["method_id"], row["cap"]): row for row in derived}
        for row in out:
            key = (row["method_id"], row["cap"])
            if key not in by_key:
                continue
            derived_row = by_key[key]
            for metric in METRIC_ALIASES:
                merged = {**row["metrics"].get(metric, {}), **derived_row["metrics"].get(metric, {})}
                row["metrics"][metric] = merged
            if row.get("selected_count") is None or row.get("selected_count") == 0:
                row["selected_count"] = int(round(float(derived_row.get("selected_count", row.get("selected_count", 0)))) )
            if row.get("actual_token_count") is None:
                row["actual_token_count"] = derived_row.get("actual_token_count", row.get("actual_token_count", 0.0))
            if row.get("nominal_budget") is None:
                row["nominal_budget"] = derived_row.get("nominal_budget", row.get("nominal_budget", 0.0))

    if not out and runs:
        return _derive_aggregate_from_runs(runs)
    return out


def _derive_aggregate_from_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        method_id = _normalize_method_id(_first_present(run, "method_id", "method", "method_name"))
        cap = _safe_int(_first_present(run, "cap", "sample_size", "sample_cap"), default=0)
        if cap <= 0:
            continue
        key = (method_id, cap)
        bucket = buckets.setdefault(
            key,
            {
                "method_id": method_id,
                "method_label": _method_label(method_id),
                "color": _method_color(method_id),
                "cap": cap,
                "selected_count": [],
                "actual_token_count": [],
                "nominal_budget": [],
                "metric_values": {m: [] for m in METRIC_ALIASES},
                "cap_replays": [],
            },
        )
        bucket["selected_count"].append(_safe_float(_first_present(run, "selected_count", "sample_size", "n"), default=float(cap)))
        bucket["actual_token_count"].append(_safe_float(_first_present(run, *TOKEN_ACTUAL_ALIASES), default=0.0))
        nominal = _first_present(run, *TOKEN_NOMINAL_ALIASES)
        if nominal is None:
            avg_tokens = _safe_float(_first_present(run, "avg_packet_tokens", "average_packet_tokens"), default=0.0)
            nominal = cap * avg_tokens if avg_tokens > 0 else 0.0
        bucket["nominal_budget"].append(_safe_float(nominal, default=0.0))
        bucket["cap_replays"].append(_safe_float(_first_present(run, "cap_replays", "replays"), default=0.0))
        for metric, aliases in METRIC_ALIASES.items():
            bucket["metric_values"][metric].append(_safe_float(_first_present(run, *aliases), default=0.0))

    out: list[dict[str, Any]] = []
    for (method_id, cap), bucket in sorted(buckets.items(), key=lambda x: (x[0][1], x[0][0])):
        def _stats(values: list[float]) -> dict[str, float | int]:
            if not values:
                return {"mean": 0.0, "median": 0.0, "sample_std": 0.0, "p05": 0.0, "p95": 0.0, "count": 0, "min": 0.0, "max": 0.0}
            values_float = [float(v) for v in values]
            count = len(values_float)
            sample_std = 0.0 if count < 2 else float(np.std(values_float, ddof=1))
            return {
                "mean": float(sum(values_float) / count),
                "median": float(_linear_interpolate_quantile(values_float, 0.50)),
                "sample_std": sample_std,
                "p05": float(_linear_interpolate_quantile(values_float, 0.05)),
                "p95": float(_linear_interpolate_quantile(values_float, 0.95)),
                "count": int(count),
                "min": float(min(values_float)),
                "max": float(max(values_float)),
            }

        out.append(
            {
                "method_id": method_id,
                "method_label": _method_label(method_id),
                "color": _method_color(method_id),
                "cap": cap,
                "selected_count": int(round(_stats(bucket["selected_count"])["mean"])),
                "metrics": {metric: _stats(vals) for metric, vals in bucket["metric_values"].items()},
                "actual_token_count": _stats(bucket["actual_token_count"])["mean"],
                "nominal_budget": _stats(bucket["nominal_budget"])["mean"],
                "cap_replays": int(round(_stats(bucket["cap_replays"])["mean"])),
            }
        )
    return out


def _normalize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in runs:
        method_id = _normalize_method_id(_first_present(row, "method_id", "method", "method_name"))
        cap = _safe_int(_first_present(row, "cap", "sample_size", "sample_cap"), default=0)
        seed = _safe_int(_first_present(row, "seed", "trial_seed", "trial"), default=0)
        selected_count = _safe_int(_first_present(row, "selected_count", "sample_size", "n"), default=cap)
        actual_tokens = _normalize_stat_scalar(_first_present(row, *TOKEN_ACTUAL_ALIASES), alias="actual_token_count")
        nominal_budget = _first_present(row, *TOKEN_NOMINAL_ALIASES)
        if nominal_budget is None:
            avg_tokens = _safe_float(_first_present(row, "avg_packet_tokens", "average_packet_tokens"), default=0.0)
            nominal_budget = cap * avg_tokens if avg_tokens > 0 else 0.0
        nominal_budget = _normalize_stat_scalar(nominal_budget, alias="nominal_budget")

        top_five: list[dict[str, Any]] = []
        for agent in row.get("top_five_agents") or []:
            if not isinstance(agent, dict):
                continue
            top_five.append(
                {
                    "agent_id": str(agent.get("agent_id") or "unknown"),
                    "N": _safe_float_or_none(_first_present(agent, "N", "population_n", "population_count")),
                    "n": _safe_float_or_none(_first_present(agent, "n", "selected_n", "count")),
                    "selected_rate": _safe_float_or_none(_first_present(agent, "selected_rate", "rate")),
                    "census_rate": _safe_float_or_none(_first_present(agent, "census_rate")),
                    "absolute_error": _safe_float_or_none(_first_present(agent, "absolute_error", "mae")),
                    "concept_coverage": _safe_float_or_none(_first_present(agent, "concept_coverage", "coverage")),
                    "use_case_coverage": _safe_float_or_none(_first_present(agent, "use_case_coverage")),
                }
            )

        normalized.append(
            {
                "method_id": method_id,
                "method_label": _method_label(method_id),
                "color": _method_color(method_id),
                "cap": cap,
                "seed": seed,
                "selected_count": selected_count,
                "metrics": {metric: _safe_float(_first_present(row, *aliases), default=0.0) for metric, aliases in METRIC_ALIASES.items()},
                "estimate": _safe_float_or_none(_first_present(row, "estimate", "estimated_rate")),
                "census_pass_rate": _safe_float_or_none(_first_present(row, "census_pass_rate")),
                "selected_rate": _safe_float_or_none(
                    _first_present(row, "selected_only_rate", "selected_label_rate", "selected_rate")
                ),
                "selected_only_absolute_error": _safe_float_or_none(_first_present(row, "selected_only_absolute_error")),
                "actual_token_count": actual_tokens,
                "nominal_budget": nominal_budget,
                "token_ratio": (actual_tokens / nominal_budget) if nominal_budget > 0 else 0.0,
                "cap_replays": _safe_int(_first_present(row, "cap_replays", "replays"), default=0),
                "idw_provenance": row.get("idw_provenance") if isinstance(row.get("idw_provenance"), dict) else {},
                "idw_validation": row.get("idw_validation") if isinstance(row.get("idw_validation"), dict) else {},
                "idw_quality": row.get("idw_quality") if isinstance(row.get("idw_quality"), dict) else {},
                "top_five_agents": top_five,
            }
        )
    return normalized


def _normalize_memberships(
    rows: list[dict[str, Any]],
    *,
    total_agent_count: int = 0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        method_id = _normalize_method_id(_first_present(row, "method_id", "method"))
        selection_records = row.get("selection_records") if isinstance(row.get("selection_records"), list) else []
        selected_by_agent: dict[str, int] = {}
        for record in selection_records:
            if not isinstance(record, dict):
                continue
            agent_id = str(record.get("agent_id") or "").strip()
            if agent_id:
                selected_by_agent[agent_id] = selected_by_agent.get(agent_id, 0) + 1
        arm3_floor = row.get("arm3_floor") if isinstance(row.get("arm3_floor"), dict) else {}
        total_floor_target_value = _first_present(arm3_floor, "total_floor_target")
        if total_floor_target_value is None:
            total_floor_target_value = _first_present(row, "total_floor_target", "arm3_total_floor_target")
        floor_prefix_count_value = _first_present(arm3_floor, "floor_prefix_count")
        if floor_prefix_count_value is None:
            floor_prefix_count_value = _first_present(row, "floor_prefix_count", "arm3_floor_prefix_count")
        floor_complete_value = _first_present(arm3_floor, "floor_complete")
        if floor_complete_value is None:
            floor_complete_value = _first_present(row, "floor_complete", "arm3_floor_complete")
        cap_value = _safe_int(_first_present(row, "cap", "sample_size"), default=0)
        selected_agent_count = _safe_int(row.get("selected_agent_count"), default=len(selected_by_agent))
        if selected_by_agent:
            selected_agent_count = len(selected_by_agent)
        selected_agents_with_at_least_3 = sum(1 for count in selected_by_agent.values() if count >= 3)
        if "eligible_agents_with_at_least_3" in row:
            eligible_agents_with_at_least_3 = _safe_int(row.get("eligible_agents_with_at_least_3"), default=0)
            if not selected_by_agent:
                selected_agents_with_at_least_3 = _safe_int(row.get("agents_with_at_least_3"), default=0)
        else:
            # Older bundles used agents_with_at_least_3 for population eligibility.
            eligible_agents_with_at_least_3 = _safe_int(row.get("agents_with_at_least_3"), default=0)

        row_total_agents = _safe_int(row.get("total_agent_count"), default=total_agent_count)
        agent_coverage = _safe_float_or_none(row.get("agent_coverage"))
        if agent_coverage is None:
            agent_coverage = (selected_agent_count / row_total_agents) if row_total_agents > 0 else 0.0

        total_floor_target = _safe_int(total_floor_target_value, default=0)
        floor_prefix_count = _safe_int(floor_prefix_count_value, default=0)
        if total_floor_target > 0:
            floor_completion_ratio = _safe_float(floor_prefix_count, default=0.0) / _safe_float(total_floor_target, default=1.0)
            floor_complete = floor_prefix_count >= total_floor_target and cap_value >= total_floor_target
        else:
            floor_completion_ratio = 1.0 if bool(floor_complete_value) else 0.0
            floor_complete = bool(floor_complete_value)

        out.append(
            {
                "method_id": method_id,
                "cap": cap_value,
                "selected_agent_count": selected_agent_count,
                "eligible_agents_with_at_least_3": eligible_agents_with_at_least_3,
                "agents_with_at_least_3": selected_agents_with_at_least_3,
                "represented_strata": (
                    len(row.get("represented_strata"))
                    if isinstance(row.get("represented_strata"), list)
                    else _safe_int(_first_present(row, "represented_strata_count", "represented_strata"), default=0)
                ),
                "agent_coverage": agent_coverage,
                "total_floor_target": total_floor_target,
                "floor_prefix_count": floor_prefix_count,
                "floor_complete": floor_complete,
                "arm3_floor_min_per_agent": _safe_int(_first_present(row, "arm3_floor_min_per_agent", "minimum_sessions"), default=0),
                "arm3_floor_completion": _safe_float(_first_present(row, "arm3_floor_completion", "floor_completion"), default=0.0),
                "floor_completion_ratio": floor_completion_ratio,
            }
        )
        current = out[-1]
        if current["total_floor_target"] > 0:
            current["arm3_floor_completion"] = current["floor_completion_ratio"]
        elif current["arm3_floor_completion"] <= 0 and current["floor_complete"]:
            current["arm3_floor_completion"] = 1.0
    return out


def _normalize_classifications(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "use_case_guid": str(row.get("use_case_guid") or "undetermined"),
                "domain": str(row.get("domain") or "undetermined"),
                "segment": str(row.get("segment") or "undetermined"),
                "category": str(row.get("category") or "undetermined"),
                "sub_category": str(row.get("sub_category") or "undetermined"),
                "sub_subcategory": str(row.get("sub_subcategory") or "undetermined"),
                "business_task": str(row.get("business_task") or "undetermined"),
                "status": str(row.get("status") or "undetermined"),
                "confidence_level": str(row.get("confidence_level") or "undetermined"),
                "combined_cosine_similarity": _safe_float_or_none(row.get("combined_cosine_similarity")),
                "agent_id": str(row.get("agent_id") or "unknown"),
                "concept_key": str(row.get("concept_key") or "unknown"),
                "corpus_id": str(row.get("corpus_id") or "unknown"),
            }
        )
    return normalized


def _normalize_agent_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        method_id = _normalize_method_id(_first_present(row, "method_id", "method", "method_name"))
        normalized.append(
            {
                "method_id": method_id,
                "method_label": _method_label(method_id),
                "color": _method_color(method_id),
                "seed": _safe_int(_first_present(row, "seed", "trial_seed", "trial"), default=0),
                "cap": _safe_int(_first_present(row, "cap", "sample_size", "sample_cap"), default=0),
                "agent_id": str(_first_present(row, "agent_id", "agent") or "unknown"),
                "N": _safe_int(_first_present(row, "N", "population_n", "population_count"), default=0),
                "n": _safe_int(_first_present(row, "n", "selected_n", "selected_count"), default=0),
                "estimate": _safe_float_or_none(_first_present(row, "estimate", "estimated_rate")),
                "census_rate": _safe_float_or_none(_first_present(row, "census_rate", "census_pass_rate")),
                "absolute_error": _safe_float_or_none(_first_present(row, "absolute_error", "mae")),
                "concept_coverage": _safe_float_or_none(_first_present(row, "concept_coverage", "coverage")),
                "use_case_coverage": _safe_float_or_none(_first_present(row, "use_case_coverage", "maven_coverage")),
                "estimator": str(_first_present(row, "estimator") or "unknown"),
                "represented_population_fraction": _safe_float_or_none(
                    _first_present(row, "represented_population_fraction", "represented_fraction")
                ),
            }
        )
    return normalized


def _normalize_dataset_examples(payload: dict[str, Any]) -> dict[str, Any]:
    examples = payload.get("examples") if isinstance(payload, dict) else []
    out_examples: list[dict[str, Any]] = []
    if isinstance(examples, list):
        for item in examples[:50]:
            if not isinstance(item, dict):
                continue
            source = item.get("source") if isinstance(item.get("source"), dict) else {}
            shape = item.get("shape") if isinstance(item.get("shape"), dict) else {}
            snippet = item.get("snippet") if isinstance(item.get("snippet"), dict) else {}
            legacy_snippet = str(item.get("snippet") or "") if not snippet else ""
            out_examples.append(
                {
                    "corpus_id": str(item.get("corpus_id") or item.get("corpus") or "unknown"),
                    "agent": str(item.get("agent") or item.get("agent_id") or "unknown"),
                    "source": source,
                    "shape": {
                        "turn_count": _safe_int(shape.get("turn_count"), default=0),
                        "tool_call_count": _safe_int(shape.get("tool_call_count"), default=0),
                        "had_error": bool(shape.get("had_error")),
                    },
                    "expected_label": str(item.get("expected_label") or item.get("label") or "unknown"),
                    "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                    "snippet": {
                        "user": str(_first_present(snippet, "user") or legacy_snippet),
                        "assistant": str(_first_present(snippet, "assistant") or ""),
                    },
                }
            )
    source_summary = payload.get("source_summary") if isinstance(payload.get("source_summary"), dict) else {}
    synthesized_fields = payload.get("synthesized_fields") if isinstance(payload.get("synthesized_fields"), dict) else {}
    schema_explanation = str(payload.get("schema_explanation") or payload.get("synthesis_explanation") or "")
    return {
        "examples": out_examples,
        "source_summary": source_summary,
        "synthesized_fields": synthesized_fields,
        "schema_explanation": schema_explanation,
    }


def load_v6_artifacts(inputs: V6ReportInputs) -> LoadedV6Artifacts:
    aggregate = _read_json(inputs.aggregate)
    runs = _read_jsonl(inputs.runs_jsonl)
    memberships = _load_optional_jsonl(inputs.memberships)
    classifications = _load_optional_jsonl(inputs.classifications)
    agent_metrics = _load_optional_jsonl(inputs.agent_metrics)
    dataset_examples = _load_optional_json(inputs.dataset_examples) or {}
    methodology_text = _load_optional_text(inputs.methodology)
    manifest = _load_optional_json(inputs.manifest) or {}

    aggregate_rows = _normalize_aggregate_rows(aggregate, runs)
    aggregate["methods"] = aggregate_rows
    aggregate["aggregate_rows"] = aggregate_rows

    return LoadedV6Artifacts(
        aggregate=aggregate,
        runs=_normalize_runs(runs),
        memberships=_normalize_memberships(
            memberships,
            total_agent_count=_safe_int(
                (aggregate.get("population_audit") or {}).get("agent_count"),
                default=0,
            ),
        ),
        classifications=_normalize_classifications(classifications),
        agent_metrics=_normalize_agent_metrics(agent_metrics),
        dataset_examples=_normalize_dataset_examples(dataset_examples),
        methodology_text=methodology_text,
        manifest=manifest,
    )


def _fmt_number(value: Any, digits: int = 3) -> str:
    return f"{_safe_float(value, default=0.0):.{digits}f}"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    return f"{100.0 * _safe_float(value, default=0.0):.{digits}f}%"


def _token_ratio(nominal: float, actual: float) -> float:
    return actual / nominal if nominal > 0 else 0.0


def _caps_from_data(aggregate_rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[int]:
    caps = {
        _safe_int(row.get("cap"), default=0)
        for row in aggregate_rows + runs
        if _safe_int(row.get("cap"), default=0) > 0
    }
    return sorted(caps)


def _agent_count_from_data(aggregate: dict[str, Any], runs: list[dict[str, Any]]) -> int:
    pop = aggregate.get("population_audit") if isinstance(aggregate.get("population_audit"), dict) else {}
    if _safe_int(pop.get("agent_count"), default=0) > 0:
        return _safe_int(pop.get("agent_count"), default=0)
    agents = {a.get("agent_id") for r in runs for a in (r.get("top_five_agents") or []) if isinstance(a, dict) and a.get("agent_id")}
    return len(agents)


def _population_count(aggregate: dict[str, Any]) -> int:
    pop = aggregate.get("population_audit") if isinstance(aggregate.get("population_audit"), dict) else {}
    return _safe_int(pop.get("unit_count"), default=0)


def _safe_json_script_blob(obj: Any) -> str:
    txt = _canonical_json(obj)
    txt = txt.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    txt = txt.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return txt


def _render_methodology_markdown(md: str) -> str:
    lines = md.splitlines()
    parts: list[str] = []
    in_ul = False
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            continue
        if line.startswith("### "):
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            parts.append(f"<h4>{escape(line[4:])}</h4>")
            continue
        if line.startswith("## "):
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            parts.append(f"<h3>{escape(line[3:])}</h3>")
            continue
        if line.startswith("# "):
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            parts.append(f"<h2>{escape(line[2:])}</h2>")
            continue
        if line.lstrip().startswith(("- ", "* ")):
            if not in_ul:
                parts.append("<ul>")
                in_ul = True
            item = line.lstrip()[2:]
            parts.append(f"<li>{escape(item)}</li>")
            continue
        if in_ul:
            parts.append("</ul>")
            in_ul = False
        parts.append(f"<p>{escape(line)}</p>")
    if in_ul:
        parts.append("</ul>")
    if not parts:
        return "<p class='empty-state'>No methodology markdown provided.</p>"
    return "".join(parts)


def _build_top_five_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket: dict[tuple[str, int, str], dict[str, Any]] = {}
    for run in runs:
        method_id = str(run.get("method_id") or "unknown")
        cap = _safe_int(run.get("cap"), default=0)
        for agent in run.get("top_five_agents") or []:
            if not isinstance(agent, dict):
                continue
            agent_id = str(agent.get("agent_id") or "unknown")
            key = (method_id, cap, agent_id)
            row = bucket.setdefault(
                key,
                {
                    "method_id": method_id,
                    "cap": cap,
                    "agent_id": agent_id,
                    "N": [],
                    "n": [],
                    "selected_rate": [],
                    "census_rate": [],
                    "absolute_error": [],
                    "concept_coverage": [],
                    "use_case_coverage": [],
                    "trials": [],
                },
            )
            for f in ("N", "n", "selected_rate", "census_rate", "absolute_error", "concept_coverage", "use_case_coverage"):
                row[f].append(_safe_float_or_none(agent.get(f)))
            row["trials"].append({"seed": run.get("seed"), **agent})

    out: list[dict[str, Any]] = []
    for _, row in bucket.items():
        def stats(values: list[float | None]) -> dict[str, float | None]:
            usable = [value for value in values if value is not None]
            if not usable:
                return {"mean": None, "min": None, "max": None}
            return {"mean": sum(usable) / len(usable), "min": min(usable), "max": max(usable)}

        out.append(
            {
                "method_id": row["method_id"],
                "cap": row["cap"],
                "agent_id": row["agent_id"],
                "N": stats(row["N"]),
                "n": stats(row["n"]),
                "selected_rate": stats(row["selected_rate"]),
                "census_rate": stats(row["census_rate"]),
                "absolute_error": stats(row["absolute_error"]),
                "concept_coverage": stats(row["concept_coverage"]),
                "use_case_coverage": stats(row["use_case_coverage"]),
                "trial_details": row["trials"],
            }
        )
    return sorted(
        out,
        key=lambda r: (
            r["cap"],
            r["method_id"],
            -_safe_float(r["selected_rate"].get("mean"), default=-1.0),
        ),
    )


def _top_five_print_table(top_five_summary: list[dict[str, Any]]) -> str:
    if not top_five_summary:
        return (
            "<div class='table-wrap print-only'><table class='compact-table print-compact'><thead><tr>"
            "<th>Method</th><th>Cap</th><th>Agent</th><th>Selected/Census rate</th><th>Abs error</th><th>n</th>"
            "</tr></thead><tbody><tr><td colspan='6'>N/A</td></tr></tbody></table></div>"
        )
    rows: list[str] = []
    for row in top_five_summary[:40]:
        selected = _safe_float_or_none((row.get("selected_rate") or {}).get("mean"))
        census = _safe_float_or_none((row.get("census_rate") or {}).get("mean"))
        rate_pair = "N/A"
        if selected is not None and census is not None:
            rate_pair = f"{_fmt_pct(selected, 2)} / {_fmt_pct(census, 2)}"
        elif selected is not None:
            rate_pair = f"{_fmt_pct(selected, 2)} / N/A"
        rows.append(
            "<tr>"
            f"<td>{escape(_method_label(str(row.get('method_id') or 'unknown')))}</td>"
            f"<td>{_safe_int(row.get('cap'), default=0)}</td>"
            f"<td>{escape(str(row.get('agent_id') or 'unknown'))}</td>"
            f"<td>{escape(rate_pair)}</td>"
            f"<td>{_fmt_number((row.get('absolute_error') or {}).get('mean'), 4)}</td>"
            f"<td>{_fmt_number((row.get('n') or {}).get('mean'), 1)}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap print-only'><table class='compact-table print-compact'><thead><tr>"
        "<th>Method</th><th>Cap</th><th>Agent</th><th>Selected/Census rate</th><th>Abs error</th><th>n</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _idw_tables_html(idw_rows: list[dict[str, Any]]) -> str:
    if not idw_rows:
        return "<p class='empty-state'>No ARM2 IDW provenance/validation rows were present.</p>"
    screen_rows: list[str] = []
    print_by_cap: dict[int, dict[str, list[float]]] = {}
    for row in sorted(idw_rows, key=lambda r: (_safe_int(r.get("cap"), default=0), _safe_int(r.get("seed"), default=0))):
        provenance_counts = row.get("provenance_counts") if isinstance(row.get("provenance_counts"), dict) else {}
        provenance_blob = ", ".join(f"{escape(str(k))}:{_safe_int(v, default=0)}" for k, v in sorted(provenance_counts.items())) or "none"
        screen_rows.append(
            "<tr>"
            f"<td>{_safe_int(row.get('cap'), default=0)}</td><td>{_safe_int(row.get('seed'), default=0)}</td>"
            f"<td>{_safe_int(row.get('observed_count'), default=0)}</td><td>{_safe_int(row.get('imputed_count'), default=0)}</td><td>{_safe_int(row.get('provenance_count'), default=0)}</td><td>{provenance_blob}</td>"
            f"<td>{_fmt_number(row.get('absolute_aggregate_rate_error'), 4)}</td><td>{_fmt_number(row.get('per_unit_mae'), 4)}</td><td>{_fmt_number(row.get('brier_score'), 4)}</td>"
            f"<td>{_fmt_number(row.get('macro_per_agent_mae'), 4)}</td><td>{_fmt_number(row.get('unjudged_only_mae'), 4)}</td><td>{_fmt_number(row.get('unjudged_only_brier'), 4)}</td><td>{_fmt_number(row.get('expected_calibration_error'), 4)}</td>"
            "</tr>"
        )
        cap = _safe_int(row.get("cap"), default=0)
        bucket = print_by_cap.setdefault(
            cap,
            {
                "observed_count": [],
                "imputed_count": [],
                "absolute_aggregate_rate_error": [],
                "brier_score": [],
                "expected_calibration_error": [],
            },
        )
        for field in bucket:
            bucket[field].append(_safe_float(row.get(field), default=0.0))

    print_rows: list[str] = []
    for cap, bucket in sorted(print_by_cap.items()):
        def avg(field: str) -> float:
            values = bucket[field]
            return sum(values) / len(values) if values else 0.0

        print_rows.append(
            "<tr>"
            f"<td>{cap}</td>"
            f"<td>{_fmt_number(avg('observed_count'), 1)}</td>"
            f"<td>{_fmt_number(avg('imputed_count'), 1)}</td>"
            f"<td>{_fmt_number(avg('absolute_aggregate_rate_error'), 4)}</td>"
            f"<td>{_fmt_number(avg('brier_score'), 4)}</td>"
            f"<td>{_fmt_number(avg('expected_calibration_error'), 4)}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap screen-only'><table class='compact-table'><thead><tr><th>Cap</th><th>Trial</th><th>Observed</th><th>Imputed</th><th>Provenance</th><th>Provenance Categories</th><th>Abs Aggregate Error</th><th>Per-Unit MAE</th><th>Brier</th><th>Macro Agent MAE</th><th>Unjudged MAE</th><th>Unjudged Brier</th><th>ECE</th></tr></thead><tbody>"
        + "".join(screen_rows)
        + "</tbody></table></div>"
        + "<div class='table-wrap print-only'><table class='compact-table print-compact'><thead><tr><th>Cap</th><th>Observed</th><th>Imputed</th><th>Aggregate Error</th><th>Brier</th><th>ECE</th></tr></thead><tbody>"
        + "".join(print_rows)
        + "</tbody></table></div>"
    )


def _taxonomy_tables_html(class_summary: dict[str, Any]) -> str:
    rows = class_summary.get("taxonomy_rows") if isinstance(class_summary.get("taxonomy_rows"), list) else []
    if not rows:
        return "<p class='empty-state'>No taxonomy rows available.</p>"
    screen_rows: list[str] = []
    print_rows: list[str] = []
    for (seg, dom, cat, sub, subsub, task), count in rows[:25]:
        screen_rows.append(
            "<tr>"
            f"<td>{escape(seg)}</td><td>{escape(dom)}</td><td>{escape(cat)}</td><td>{escape(sub)}</td><td>{escape(subsub)}</td><td>{escape(task)}</td><td>{count}</td>"
            "</tr>"
        )
    for (seg, dom, cat, sub, subsub, task), count in rows[:15]:
        print_rows.append(
            "<tr>"
            f"<td>{escape(dom)}</td><td>{escape(cat)}</td><td>{escape(subsub)}</td><td>{escape(task)}</td><td>{count}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap screen-only'><table class='compact-table'><thead><tr><th>Segment</th><th>Domain</th><th>Category</th><th>Subcategory</th><th>Sub-subcategory</th><th>Business Task</th><th>Count</th></tr></thead><tbody>"
        + "".join(screen_rows)
        + "</tbody></table></div>"
        + "<div class='table-wrap print-only'><table class='compact-table print-compact'><thead><tr><th>Domain</th><th>Category</th><th>Sub-subcategory</th><th>Business Task</th><th>Count</th></tr></thead><tbody>"
        + "".join(print_rows)
        + "</tbody></table></div>"
    )


def _membership_tables_html(membership_rows: list[dict[str, Any]]) -> str:
    if not membership_rows:
        return "<p class='empty-state'>No membership summary rows were present.</p>"
    screen: list[str] = []
    print_coverage: list[str] = []
    print_floor: list[str] = []
    for row in membership_rows:
        floor_complete_text = "Yes" if bool(row.get("floor_complete")) else "No"
        screen.append(
            "<tr>"
            f"<td>{escape(_method_label(str(row.get('method_id') or 'unknown')))}</td><td>{_safe_int(row.get('cap'), default=0)}</td>"
            f"<td>{_safe_int(row.get('selected_agent_count'), default=0)}</td><td>{_safe_int(row.get('eligible_agents_with_at_least_3'), default=0)}</td><td>{_safe_int(row.get('agents_with_at_least_3'), default=0)}</td>"
            f"<td>{_safe_int(row.get('represented_strata'), default=0)}</td><td>{_fmt_pct(row.get('agent_coverage'), 2)}</td>"
            f"<td>{_safe_int(row.get('total_floor_target'), default=0)}</td><td>{_safe_int(row.get('floor_prefix_count'), default=0)}</td><td>{_fmt_pct(row.get('arm3_floor_completion'), 2)}</td><td>{floor_complete_text}</td>"
            "</tr>"
        )
        print_coverage.append(
            "<tr>"
            f"<td>{escape(_method_label(str(row.get('method_id') or 'unknown')))}</td><td>{_safe_int(row.get('cap'), default=0)}</td><td>{_safe_int(row.get('selected_agent_count'), default=0)}</td><td>{_safe_int(row.get('represented_strata'), default=0)}</td><td>{_fmt_pct(row.get('agent_coverage'), 2)}</td>"
            "</tr>"
        )
        print_floor.append(
            "<tr>"
            f"<td>{escape(_method_label(str(row.get('method_id') or 'unknown')))}</td><td>{_safe_int(row.get('cap'), default=0)}</td><td>{_safe_int(row.get('eligible_agents_with_at_least_3'), default=0)}</td><td>{_safe_int(row.get('agents_with_at_least_3'), default=0)}</td><td>{_safe_int(row.get('total_floor_target'), default=0)}</td><td>{_safe_int(row.get('floor_prefix_count'), default=0)}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap screen-only'><table class='compact-table'><thead><tr><th>Method</th><th>Cap</th><th>Selected Agents</th><th>Eligible Agents >=3</th><th>Selected Agents >=3</th><th>Represented Strata</th><th>Agent Coverage</th><th>Arm3 Floor Target</th><th>Arm3 Floor Prefix Selected</th><th>Arm3 Floor Completion</th><th>Floor Complete</th></tr></thead><tbody>"
        + "".join(screen)
        + "</tbody></table></div>"
        + "<h3 class='print-only'>Membership Coverage (Print)</h3>"
        + "<div class='table-wrap print-only'><table class='compact-table print-compact'><thead><tr><th>Method</th><th>Cap</th><th>Selected Agents</th><th>Represented Strata</th><th>Agent Coverage</th></tr></thead><tbody>"
        + "".join(print_coverage)
        + "</tbody></table></div>"
        + "<h3 class='print-only'>Arm3 Floor Status (Print)</h3>"
        + "<div class='table-wrap print-only'><table class='compact-table print-compact'><thead><tr><th>Method</th><th>Cap</th><th>Eligible >=3</th><th>Selected >=3</th><th>Floor Target</th><th>Floor Prefix</th></tr></thead><tbody>"
        + "".join(print_floor)
        + "</tbody></table></div>"
    )


def _classification_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    use_case_counts: dict[str, int] = {}
    use_case_label_counts: dict[tuple[str, str], int] = {}
    sub_sub_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    similarities: list[float] = []
    taxonomy: dict[tuple[str, str, str, str, str, str], int] = {}
    corpus_diag: dict[str, dict[str, Any]] = {}

    for row in rows:
        use_case_counts[row["use_case_guid"]] = use_case_counts.get(row["use_case_guid"], 0) + 1
        human_label = f"{row['sub_subcategory']} / {row['business_task']}"
        key = (row["use_case_guid"], human_label)
        use_case_label_counts[key] = use_case_label_counts.get(key, 0) + 1
        sub_sub_counts[row["sub_subcategory"]] = sub_sub_counts.get(row["sub_subcategory"], 0) + 1
        confidence_counts[row["confidence_level"]] = confidence_counts.get(row["confidence_level"], 0) + 1
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        similarity = _safe_float_or_none(row.get("combined_cosine_similarity"))
        if similarity is not None:
            similarities.append(similarity)

        corpus_id = row.get("corpus_id") or "unknown"
        corpus = corpus_diag.setdefault(
            corpus_id,
            {
                "n": 0,
                "similarities": [],
                "status": {},
                "confidence": {},
            },
        )
        corpus["n"] += 1
        if similarity is not None:
            corpus["similarities"].append(similarity)
        corpus["status"][row["status"]] = corpus["status"].get(row["status"], 0) + 1
        corpus["confidence"][row["confidence_level"]] = corpus["confidence"].get(row["confidence_level"], 0) + 1

        tax_key = (
            row["segment"],
            row["domain"],
            row["category"],
            row["sub_category"],
            row["sub_subcategory"],
            row["business_task"],
        )
        taxonomy[tax_key] = taxonomy.get(tax_key, 0) + 1

    corpus_rows: list[dict[str, Any]] = []
    for corpus_id, payload in sorted(corpus_diag.items(), key=lambda kv: kv[0]):
        sims = payload["similarities"]
        corpus_rows.append(
            {
                "corpus_id": corpus_id,
                "n": payload["n"],
                "mean_similarity": (sum(sims) / len(sims)) if sims else 0.0,
                "status": sorted(payload["status"].items(), key=lambda kv: (-kv[1], kv[0])),
                "confidence": sorted(payload["confidence"].items(), key=lambda kv: (-kv[1], kv[0])),
            }
        )

    return {
        "top_use_cases": sorted(use_case_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
        "top_use_case_labels": sorted(use_case_label_counts.items(), key=lambda kv: (-kv[1], kv[0][0]))[:12],
        "top_sub_subcategory": sorted(sub_sub_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
        "confidence_levels": sorted(confidence_counts.items(), key=lambda kv: (-kv[1], kv[0])),
        "status": sorted(status_counts.items(), key=lambda kv: (-kv[1], kv[0])),
        "similarity": {
            "mean": (sum(similarities) / len(similarities)) if similarities else 0.0,
            "min": min(similarities) if similarities else 0.0,
            "max": max(similarities) if similarities else 0.0,
        },
        "taxonomy_rows": sorted(taxonomy.items(), key=lambda kv: (-kv[1], kv[0]))[:25],
        "total_unique_use_cases": len(use_case_counts),
        "total_unique_sub_subcategory": len(sub_sub_counts),
        "corpus_diagnostics": corpus_rows,
        "undetermined_count": sum(
            count for guid, count in use_case_counts.items() if str(guid).lower().startswith("undetermined")
        ),
        "low_confidence_count": sum(
            count for level, count in confidence_counts.items() if str(level).strip().lower() in {"0", "undetermined"}
        ),
    }


def _idw_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run in runs:
        if run.get("method_id") != "arm2_embedding_idw":
            continue
        prov = run.get("idw_provenance") if isinstance(run.get("idw_provenance"), dict) else {}
        val = run.get("idw_validation") if isinstance(run.get("idw_validation"), dict) else {}
        quality = run.get("idw_quality") if isinstance(run.get("idw_quality"), dict) else {}
        counts = prov.get("provenance_counts") if isinstance(prov.get("provenance_counts"), dict) else {}
        out.append(
            {
                "cap": run.get("cap"),
                "seed": run.get("seed"),
                "observed_count": _safe_int(_first_present(prov, "observed_count", "observed"), default=0),
                "imputed_count": _safe_int(_first_present(prov, "imputed_count", "imputed"), default=0),
                "provenance_count": (
                    sum(_safe_int(v, default=0) for v in counts.values())
                    if counts
                    else _safe_int(_first_present(prov, "provenance_count", "sources"), default=0)
                ),
                "provenance_counts": {str(k): _safe_int(v, default=0) for k, v in counts.items()},
                "quality": str(_first_present(val, "quality", "quality_band", "diagnostic") or "unknown"),
                "absolute_aggregate_rate_error": _safe_float(_first_present(quality, "absolute_aggregate_rate_error"), default=0.0),
                "per_unit_mae": _safe_float(_first_present(quality, "per_unit_mae", "mae", "absolute_error"), default=0.0),
                "brier_score": _safe_float(_first_present(quality, "brier_score"), default=0.0),
                "macro_per_agent_mae": _safe_float(_first_present(quality, "macro_per_agent_mae"), default=0.0),
                "unjudged_only_mae": _safe_float(_first_present(quality, "unjudged_only_mae"), default=0.0),
                "unjudged_only_brier": _safe_float(_first_present(quality, "unjudged_only_brier"), default=0.0),
                "expected_calibration_error": _safe_float(_first_present(quality, "expected_calibration_error"), default=0.0),
            }
        )
    return out


def _membership_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        method_id = str(row.get("method_id") or "unknown")
        cap = _safe_int(row.get("cap"), default=0)
        key = (method_id, cap)
        g = grouped.setdefault(
            key,
            {
                "method_id": method_id,
                "cap": cap,
                "selected_agent_count": [],
                "eligible_agents_with_at_least_3": [],
                "agents_with_at_least_3": [],
                "represented_strata": [],
                "agent_coverage": [],
                "total_floor_target": [],
                "floor_prefix_count": [],
                "floor_complete": [],
                "arm3_floor_min_per_agent": [],
                "arm3_floor_completion": [],
            },
        )
        for f in ("selected_agent_count", "eligible_agents_with_at_least_3", "agents_with_at_least_3", "represented_strata", "arm3_floor_min_per_agent"):
            g[f].append(_safe_float(row.get(f), default=0.0))
        for f in ("agent_coverage", "arm3_floor_completion", "total_floor_target", "floor_prefix_count"):
            g[f].append(_safe_float(row.get(f), default=0.0))
        g["floor_complete"].append(1.0 if bool(row.get("floor_complete")) else 0.0)

    out: list[dict[str, Any]] = []
    for _, g in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        def avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        total_floor_target = int(round(avg(g["total_floor_target"])))
        floor_prefix_count = int(round(avg(g["floor_prefix_count"])))
        cap_value = _safe_int(g["cap"], default=0)
        floor_completion_ratio = (
            (_safe_float(floor_prefix_count, default=0.0) / _safe_float(total_floor_target, default=1.0))
            if total_floor_target > 0
            else 0.0
        )
        out.append(
            {
                "method_id": g["method_id"],
                "cap": g["cap"],
                "selected_agent_count": int(round(avg(g["selected_agent_count"]))),
                "eligible_agents_with_at_least_3": int(round(avg(g["eligible_agents_with_at_least_3"]))),
                "agents_with_at_least_3": int(round(avg(g["agents_with_at_least_3"]))),
                "represented_strata": int(round(avg(g["represented_strata"]))),
                "agent_coverage": avg(g["agent_coverage"]),
                "total_floor_target": total_floor_target,
                "floor_prefix_count": floor_prefix_count,
                "floor_complete": bool(total_floor_target > 0 and floor_prefix_count >= total_floor_target and cap_value >= total_floor_target),
                "arm3_floor_min_per_agent": int(round(avg(g["arm3_floor_min_per_agent"]))),
                "arm3_floor_completion": floor_completion_ratio,
                "floor_completion_ratio": floor_completion_ratio,
            }
        )
    return out


def _dataset_table_html(dataset: dict[str, Any]) -> str:
    rows = dataset.get("examples") if isinstance(dataset.get("examples"), list) else []
    if not rows:
        return "<p class='empty-state'>No dataset examples provided.</p>"

    cards: list[str] = []
    for index, ex in enumerate(rows[:5], start=1):
        meta = ex.get("metadata") if isinstance(ex.get("metadata"), dict) else {}
        source = ex.get("source") if isinstance(ex.get("source"), dict) else {}
        shape = ex.get("shape") if isinstance(ex.get("shape"), dict) else {}
        snippet = ex.get("snippet") if isinstance(ex.get("snippet"), dict) else {}
        cards.append(
            "<article class='dataset-card'>"
            f"<h4>Example {index}: corpus {escape(str(ex.get('corpus_id') or 'unknown'))}</h4>"
            "<div class='dataset-grid'>"
            f"<div><strong>Agent</strong><br>{escape(str(ex.get('agent') or 'unknown'))}</div>"
            f"<div><strong>Expected Label</strong><br>{escape(str(ex.get('expected_label') or 'unknown'))}</div>"
            f"<div><strong>Source Metadata</strong><br>synthetic={'yes' if bool(source.get('is_synthetic')) else 'no'}; hash={escape(str(source.get('source_hash') or '')[:24])}</div>"
            f"<div><strong>Turn / Tool / Error Shape</strong><br>turns={_safe_int(shape.get('turn_count'), default=0)}; tools={_safe_int(shape.get('tool_call_count'), default=0)}; had_error={'yes' if bool(shape.get('had_error')) else 'no'}</div>"
            f"<div><strong>Metadata</strong><br>domain={escape(str(meta.get('domain') or 'n/a'))}; task={escape(str(meta.get('task') or 'n/a'))}; pass_rate={_fmt_pct(meta.get('pass_rate'), 2) if meta.get('pass_rate') is not None else 'N/A'}</div>"
            "</div>"
            "<div class='snippet-pair'>"
            f"<div><strong>User Snippet (bounded preview)</strong><p>{escape(str(snippet.get('user') or '')[:220])}</p></div>"
            f"<div><strong>Assistant Snippet (bounded preview)</strong><p>{escape(str(snippet.get('assistant') or '')[:220])}</p></div>"
            "</div>"
            "</article>"
        )
    return "<div class='dataset-stack'>" + "".join(cards) + "</div>"


def _dataset_source_summary_html(dataset: dict[str, Any]) -> str:
    summary = dataset.get("source_summary") if isinstance(dataset.get("source_summary"), dict) else {}
    schema = summary.get("schema") if isinstance(summary.get("schema"), dict) else {}
    overall = summary.get("overall") if isinstance(summary.get("overall"), dict) else {}
    corpus_rows = summary.get("by_corpus")
    if not isinstance(corpus_rows, list):
        corpus_rows = summary.get("corpora") if isinstance(summary.get("corpora"), list) else []

    synthesized = dataset.get("synthesized_fields") if isinstance(dataset.get("synthesized_fields"), dict) else {}
    source_synthetic = synthesized.get("source_synthetic") if isinstance(synthesized.get("source_synthetic"), list) else []
    report_derived = synthesized.get("report_derived") if isinstance(synthesized.get("report_derived"), list) else []

    corpus_body = []
    for row in corpus_rows[:20]:
        if not isinstance(row, dict):
            continue
        corpus_body.append(
            "<tr>"
            f"<td>{escape(str(row.get('corpus_id') or 'unknown'))}</td>"
            f"<td>{_safe_int(row.get('unit_count'), default=0)}</td>"
            f"<td>{_safe_int(row.get('pass_count'), default=0)}</td>"
            f"<td>{_fmt_pct(row.get('pass_rate'), 2)}</td>"
            f"<td>{escape(str(row.get('source_hash') or '')[:24])}</td>"
            "</tr>"
        )

    corpus_table = "<p class='empty-state'>No per-corpus source summary rows provided.</p>"
    if corpus_body:
        corpus_table = (
            "<div class='table-wrap'><table class='compact-table'><thead><tr><th>Corpus</th><th>Units</th><th>Pass Count</th><th>Pass Rate</th><th>Source Hash Prefix</th></tr></thead><tbody>"
            + "".join(corpus_body)
            + "</tbody></table></div>"
        )

    schema_items = []
    for key in ("description", "expected_label_field", "snippet_policy"):
        if key in schema:
            schema_items.append(f"<li>{escape(key)}: {escape(str(schema.get(key) or ''))}</li>")

    source_syn_items = "".join(f"<li>{escape(str(field))}</li>" for field in source_synthetic) or "<li>None</li>"
    report_derived_items = "".join(f"<li>{escape(str(field))}</li>" for field in report_derived) or "<li>None</li>"

    return (
        "<div class='two-col'>"
        "<div>"
        "<h3>Source Summary + Fidelity</h3>"
        f"<p>Overall units: {_safe_int(overall.get('unit_count'), default=0)}; pass count: {_safe_int(overall.get('pass_count'), default=0)}; pass rate: {_fmt_pct(overall.get('pass_rate'), 2)}</p>"
        "<p>Source hashes, corpus counts, and pass-rates are report-derived from the run artifact. Snippets shown below are bounded previews and are not full transcripts.</p>"
        f"{corpus_table}"
        "</div>"
        "<div>"
        "<h3>What Is Synthetic vs Report-Derived</h3>"
        f"<ul>{''.join(schema_items) or '<li>No schema block provided.</li>'}</ul>"
        "<h4>source_synthetic</h4>"
        f"<ul>{source_syn_items}</ul>"
        "<h4>report_derived</h4>"
        f"<ul>{report_derived_items}</ul>"
        "</div>"
        "</div>"
    )


def _svg_empty(chart_id: str, title: str, subtitle: str) -> str:
    return (
        f"<figure class='chart-shell keep-together' id='{escape(chart_id)}'>"
        f"<figcaption><strong>{escape(title)}</strong><br><span class='small-note'>{escape(subtitle)}</span></figcaption>"
        f"<svg class='chart static-svg' viewBox='0 0 900 320' role='img' aria-labelledby='{escape(chart_id)}-title'>"
        f"<title id='{escape(chart_id)}-title'>{escape(title)}</title>"
        "<rect x='54' y='32' width='814' height='238' fill='#fbfcfd' stroke='#d5dde5'></rect>"
        "<text x='450' y='165' text-anchor='middle' style='font-size:14px;fill:#5a6878;'>No data available for this visualization.</text>"
        "</svg></figure>"
    )


def _line_metric_svg(
    chart_id: str,
    title: str,
    y_label: str,
    aggregate_rows: list[dict[str, Any]],
    metric_key: str,
    *,
    as_percent: bool = False,
) -> str:
    grouped: dict[str, list[tuple[int, float, float, float]]] = {}
    for row in aggregate_rows:
        method_id = str(row.get("method_id") or "unknown")
        cap = _safe_int(row.get("cap"), default=0)
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        metric = metrics.get(metric_key) if isinstance(metrics.get(metric_key), dict) else {}
        mean = _safe_float(metric.get("mean"), default=0.0)
        count = _safe_int(metric.get("count"), default=1)
        low = _safe_float(metric.get("p05"), default=_safe_float(metric.get("min"), default=mean)) if count >= 10 else _safe_float(metric.get("min"), default=mean)
        high = _safe_float(metric.get("p95"), default=_safe_float(metric.get("max"), default=mean)) if count >= 10 else _safe_float(metric.get("max"), default=mean)
        if cap > 0:
            grouped.setdefault(method_id, []).append((cap, mean, low, high))

    if not grouped:
        return _svg_empty(chart_id, title, "Aggregate rows are required.")

    for method_id in grouped:
        grouped[method_id] = sorted(grouped[method_id], key=lambda item: item[0])

    caps = sorted({cap for pts in grouped.values() for cap, _, _, _ in pts})
    values = [v for pts in grouped.values() for _, _, low, high in pts for v in (low, high)]
    y_min = min(values) if values else 0.0
    y_max = max(values) if values else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    left, top, width, height = 68, 32, 800, 220

    def x_pos(cap: int) -> float:
        if len(caps) <= 1:
            return left + width / 2.0
        idx = caps.index(cap)
        return left + (idx / (len(caps) - 1)) * width

    def y_pos(value: float) -> float:
        return top + (1.0 - ((value - y_min) / (y_max - y_min))) * height

    lines: list[str] = []
    legend: list[str] = []
    y_ticks = [y_min + ((y_max - y_min) * i / 4.0) for i in range(5)]

    for method_id in sorted(grouped):
        color = _method_color(method_id)
        pts = grouped[method_id]
        points = " ".join(f"{x_pos(cap):.2f},{y_pos(mean):.2f}" for cap, mean, _, _ in pts)
        lines.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2.4'></polyline>")
        for cap, mean, low, high in pts:
            x = x_pos(cap)
            y = y_pos(mean)
            y_low = y_pos(low)
            y_high = y_pos(high)
            value_txt = _fmt_pct(mean, 2) if as_percent else _fmt_number(mean, 4)
            low_txt = _fmt_pct(low, 2) if as_percent else _fmt_number(low, 4)
            high_txt = _fmt_pct(high, 2) if as_percent else _fmt_number(high, 4)
            title_txt = (
                f"{_method_label(method_id)} cap {cap}: {value_txt}; trial spread [{low_txt}, {high_txt}]"
            )
            lines.append(
                f"<line x1='{x:.2f}' x2='{x:.2f}' y1='{y_low:.2f}' y2='{y_high:.2f}' stroke='{color}' stroke-width='1.2'>"
                f"<title>{escape(title_txt)}</title></line>"
            )
            lines.append(
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3.2' fill='{color}'><title>{escape(title_txt)}</title></circle>"
            )
        legend.append(
            f"<g><rect x='0' y='0' width='10' height='10' fill='{color}'></rect><text x='14' y='9' style='font-size:11px;fill:#334;'>"
            f"{escape(_method_label(method_id))}</text></g>"
        )

    axis = [
        f"<line x1='{left}' y1='{top+height}' x2='{left+width}' y2='{top+height}' stroke='#728291'></line>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+height}' stroke='#728291'></line>",
        f"<text x='{left+width/2:.0f}' y='{top+height+38}' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Sample Cap</text>",
        f"<text x='20' y='{top + height/2:.0f}' transform='rotate(-90 20,{top + height/2:.0f})' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>{escape(y_label)}</text>",
    ]
    for cap in caps:
        axis.append(f"<text x='{x_pos(cap):.2f}' y='{top+height+18}' text-anchor='middle' style='font-size:10px;fill:#5a6878;'>{cap}</text>")
    for tick in y_ticks:
        y = y_pos(tick)
        axis.append(f"<line x1='{left}' x2='{left+width}' y1='{y:.2f}' y2='{y:.2f}' stroke='#ecf0f4'></line>")
        label = _fmt_pct(tick, 1) if as_percent else _fmt_number(tick, 3)
        axis.append(f"<text x='{left-8}' y='{y+3:.2f}' text-anchor='end' style='font-size:10px;fill:#5a6878;'>{label}</text>")

    legend_group = "".join(
        f"<g transform='translate({left + i * 158},{top + height + 52})'>{item}</g>"
        for i, item in enumerate(legend)
    )

    return (
        f"<figure class='chart-shell keep-together' id='{escape(chart_id)}'>"
        f"<figcaption><strong>{escape(title)}</strong><br><span class='small-note'>Lines show per-method mean with empirical 5th-95th percentile descriptive spread for n>=10; smaller bundles use min/max.</span></figcaption>"
        f"<svg class='chart static-svg' viewBox='0 0 900 360' role='img' aria-labelledby='{escape(chart_id)}-title'>"
        f"<title id='{escape(chart_id)}-title'>{escape(title)}</title>"
        + "".join(axis)
        + "".join(lines)
        + legend_group
        + "</svg></figure>"
    )


def _frontier_svg(aggregate_rows: list[dict[str, Any]]) -> str:
    points: list[tuple[str, int, float, float]] = []
    for row in aggregate_rows:
        method_id = str(row.get("method_id") or "unknown")
        cap = _safe_int(row.get("cap"), default=0)
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        mae = _safe_float((metrics.get("absolute_aggregate_mae") or {}).get("mean"), default=0.0)
        coverage = _safe_float((metrics.get("use_case_coverage") or {}).get("mean"), default=0.0)
        if cap > 0:
            points.append((method_id, cap, coverage, mae))
    if not points:
        return _svg_empty("mae-vs-maven-frontier", "MAE vs Maven Use-Case Coverage Frontier", "No aggregate rows available.")

    x_min, x_max = min(p[2] for p in points), max(p[2] for p in points)
    y_min, y_max = min(p[3] for p in points), max(p[3] for p in points)
    if x_max <= x_min:
        x_max = x_min + 0.01
    if y_max <= y_min:
        y_max = y_min + 0.01
    cap_min = min(p[1] for p in points)
    cap_max = max(p[1] for p in points)

    left, top, width, height = 70, 32, 790, 220

    def x_pos(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * width

    def y_pos(value: float) -> float:
        return top + (1.0 - ((value - y_min) / (y_max - y_min))) * height

    circles: list[str] = []
    for method_id, cap, cov, mae in points:
        radius = 4.5 + (0.0 if cap_max == cap_min else ((cap - cap_min) / (cap_max - cap_min)) * 8.0)
        label = f"{_method_label(method_id)} | cap {cap} | coverage {_fmt_pct(cov,2)} | MAE {_fmt_number(mae,4)}"
        circles.append(
            f"<circle cx='{x_pos(cov):.2f}' cy='{y_pos(mae):.2f}' r='{radius:.2f}' fill='{_method_color(method_id)}' fill-opacity='0.78' stroke='#22313f' stroke-width='0.6'>"
            f"<title>{escape(label)}</title></circle>"
            f"<text x='{x_pos(cov)+radius+2:.2f}' y='{y_pos(mae)-2:.2f}' style='font-size:9px;fill:#3b4a57;'>cap {cap}</text>"
        )

    return (
        "<figure class='chart-shell keep-together' id='mae-vs-maven-frontier'>"
        "<figcaption><strong>MAE vs Maven Use-Case Coverage Frontier</strong><br><span class='small-note'>Higher coverage (x-axis) and lower MAE (y-axis) is preferable. Point size scales with cap.</span></figcaption>"
        "<svg class='chart static-svg' viewBox='0 0 900 360' role='img' aria-labelledby='mae-vs-maven-frontier-title'>"
        "<title id='mae-vs-maven-frontier-title'>MAE vs Maven Use-Case Coverage Frontier</title>"
        f"<line x1='{left}' y1='{top+height}' x2='{left+width}' y2='{top+height}' stroke='#728291'></line>"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+height}' stroke='#728291'></line>"
        f"<text x='{left+width/2:.0f}' y='{top+height+34}' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Maven Use-Case Coverage</text>"
        f"<text x='20' y='{top + height/2:.0f}' transform='rotate(-90 20,{top + height/2:.0f})' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>MAE</text>"
        + "".join(circles)
        + "</svg></figure>"
    )


def _token_ratio_svg(aggregate_rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> str:
    data: dict[tuple[str, int], list[float]] = {}
    for row in runs:
        method_id = str(row.get("method_id") or "unknown")
        cap = _safe_int(row.get("cap"), default=0)
        nominal = _safe_float(row.get("nominal_budget"), default=0.0)
        actual = _safe_float(row.get("actual_token_count"), default=0.0)
        if cap > 0 and nominal > 0:
            data.setdefault((method_id, cap), []).append(actual / nominal)
    if not data:
        for row in aggregate_rows:
            method_id = str(row.get("method_id") or "unknown")
            cap = _safe_int(row.get("cap"), default=0)
            nominal = _safe_float(row.get("nominal_budget"), default=0.0)
            actual = _safe_float(row.get("actual_token_count"), default=0.0)
            if cap > 0 and nominal > 0:
                data.setdefault((method_id, cap), []).append(actual / nominal)
    if not data:
        return _svg_empty("token-ratio-chart", "Actual vs Nominal Token Ratio by Cap", "No token diagnostics available.")

    rows = [
        (method_id, cap, sum(vals) / len(vals))
        for (method_id, cap), vals in sorted(data.items(), key=lambda item: (item[0][1], item[0][0]))
    ]
    left, top, width, height = 68, 32, 800, 220
    max_ratio = max(1.05, max(r[2] for r in rows) * 1.1)
    step = width / max(1, len(rows))
    bar_w = max(9.0, min(24.0, step * 0.7))
    bars: list[str] = []
    for idx, (method_id, cap, ratio) in enumerate(rows):
        x = left + (idx * step) + (step - bar_w) / 2.0
        h = (ratio / max_ratio) * height
        y = top + (height - h)
        label = f"{_method_label(method_id)} cap {cap}: actual/nominal {_fmt_pct(ratio,2)}"
        bars.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w:.2f}' height='{h:.2f}' fill='{_method_color(method_id)}'><title>{escape(label)}</title></rect>"
            f"<text x='{x + bar_w / 2:.2f}' y='{top+height+16:.2f}' text-anchor='middle' style='font-size:9px;fill:#566575;'>{cap}</text>"
        )
    return (
        "<figure class='chart-shell keep-together' id='token-ratio-chart'>"
        "<figcaption><strong>Actual vs Nominal Token Ratio by Method/Cap</strong><br><span class='small-note'>Nominal uses the 15k/session planning conversion.</span></figcaption>"
        "<svg class='chart static-svg' viewBox='0 0 900 360' role='img' aria-labelledby='token-ratio-chart-title'>"
        "<title id='token-ratio-chart-title'>Actual vs Nominal Token Ratio by Method and Cap</title>"
        f"<line x1='{left}' y1='{top+height}' x2='{left+width}' y2='{top+height}' stroke='#728291'></line>"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+height}' stroke='#728291'></line>"
        f"<line x1='{left}' y1='{top + (1.0/max_ratio)*height:.2f}' x2='{left+width}' y2='{top + (1.0/max_ratio)*height:.2f}' stroke='#b24c3b' stroke-dasharray='4 3'></line>"
        f"<text x='{left+width-6}' y='{top + (1.0/max_ratio)*height - 4:.2f}' text-anchor='end' style='font-size:10px;fill:#b24c3b;'>ratio 1.00</text>"
        f"<text x='{left+width/2:.0f}' y='{top+height+34}' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Cap</text>"
        f"<text x='20' y='{top + height/2:.0f}' transform='rotate(-90 20,{top + height/2:.0f})' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Actual/Nominal Ratio</text>"
        + "".join(bars)
        + "</svg></figure>"
    )


def _arm3_floor_svg(membership_rows: list[dict[str, Any]]) -> str:
    rows = [row for row in membership_rows if str(row.get("method_id")) == "arm3_agent_round_robin_floor"]
    if not rows:
        return _svg_empty("arm3-floor-chart", "ARM3 Floor Target vs Prefix Completion", "No ARM3 membership rows available.")
    rows = sorted(rows, key=lambda row: _safe_int(row.get("cap"), default=0))
    left, top, width, height = 72, 32, 790, 220
    max_value = max(max(_safe_int(r.get("total_floor_target"), default=0), _safe_int(r.get("floor_prefix_count"), default=0)) for r in rows)
    max_value = max(1, max_value)
    step = width / max(1, len(rows))
    bar_w = max(12.0, min(28.0, step * 0.34))
    bars: list[str] = []
    for idx, row in enumerate(rows):
        cap = _safe_int(row.get("cap"), default=0)
        target = _safe_int(row.get("total_floor_target"), default=0)
        prefix = _safe_int(row.get("floor_prefix_count"), default=0)
        completion = (prefix / target) if target > 0 else 0.0
        x0 = left + idx * step + (step / 2.0) - bar_w - 2
        x1 = left + idx * step + (step / 2.0) + 2
        ht = (target / max_value) * height
        hp = (prefix / max_value) * height
        yt = top + (height - ht)
        yp = top + (height - hp)
        bars.append(
            f"<rect x='{x0:.2f}' y='{yt:.2f}' width='{bar_w:.2f}' height='{ht:.2f}' fill='#c6d2dc'><title>cap {cap} target {target}</title></rect>"
            f"<rect x='{x1:.2f}' y='{yp:.2f}' width='{bar_w:.2f}' height='{hp:.2f}' fill='#2f8f66'><title>cap {cap} prefix {prefix}; completion {_fmt_pct(completion,2)}</title></rect>"
            f"<text x='{left + idx * step + step / 2.0:.2f}' y='{top+height+16:.2f}' text-anchor='middle' style='font-size:9px;fill:#566575;'>{cap}</text>"
            f"<text x='{left + idx * step + step / 2.0:.2f}' y='{max(yp - 3, top + 10):.2f}' text-anchor='middle' style='font-size:9px;fill:#2f8f66;'>{_fmt_pct(completion,1)}</text>"
        )
    return (
        "<figure class='chart-shell keep-together' id='arm3-floor-chart'>"
        "<figcaption><strong>ARM3 Floor Target vs Prefix Selected by Cap</strong><br><span class='small-note'>Green bars show realized prefix counts; labels show completion percentage.</span></figcaption>"
        "<svg class='chart static-svg' viewBox='0 0 900 360' role='img' aria-labelledby='arm3-floor-chart-title'>"
        "<title id='arm3-floor-chart-title'>ARM3 Floor Target vs Prefix Selected by Cap</title>"
        f"<line x1='{left}' y1='{top+height}' x2='{left+width}' y2='{top+height}' stroke='#728291'></line>"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+height}' stroke='#728291'></line>"
        f"<text x='{left+width/2:.0f}' y='{top+height+34}' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Cap</text>"
        f"<text x='20' y='{top + height/2:.0f}' transform='rotate(-90 20,{top + height/2:.0f})' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Session Count</text>"
        + "".join(bars)
        + "<g transform='translate(90,288)'><rect x='0' y='0' width='10' height='10' fill='#c6d2dc'></rect><text x='14' y='9' style='font-size:11px;'>Target</text><rect x='84' y='0' width='10' height='10' fill='#2f8f66'></rect><text x='98' y='9' style='font-size:11px;'>Prefix Selected</text></g>"
        + "</svg></figure>"
    )


def _idw_provenance_svg(idw_rows: list[dict[str, Any]]) -> str:
    if not idw_rows:
        return _svg_empty("idw-provenance-chart", "ARM2 IDW Provenance Composition", "No ARM2 provenance rows available.")
    by_cap: dict[int, dict[str, float]] = {}
    for row in idw_rows:
        cap = _safe_int(row.get("cap"), default=0)
        if cap <= 0:
            continue
        bucket = by_cap.setdefault(cap, {"observed": 0.0, "idw_exact": 0.0, "fallback": 0.0, "count": 0.0})
        counts = row.get("provenance_counts") if isinstance(row.get("provenance_counts"), dict) else {}
        observed = _safe_float(_first_present(counts, "observed", "exact_observed", "direct_observed"), default=float(_safe_int(row.get("observed_count"), default=0)))
        idw_exact = sum(
            _safe_float(value, default=0.0)
            for key, value in counts.items()
            if str(key).lower() in {"idw", "exact", "exact_match", "imputed", "idw_exact"}
        )
        fallback = sum(
            _safe_float(v, default=0.0)
            for k, v in counts.items()
            if str(k).lower() in {
                "agent_mean",
                "global_mean",
                "agent_fallback",
                "global_fallback",
                "prior",
                "prior_fallback",
                "global",
                "agent",
            }
        )
        if idw_exact <= 0.0 and not counts:
            idw_exact = float(_safe_int(row.get("imputed_count"), default=0))
        if fallback <= 0.0 and not counts:
            total = _safe_float(_safe_int(row.get("provenance_count"), default=0), default=0.0)
            fallback = max(0.0, total - observed - idw_exact)
        bucket["observed"] += observed
        bucket["idw_exact"] += idw_exact
        bucket["fallback"] += fallback
        bucket["count"] += 1.0

    caps = sorted(by_cap)
    if not caps:
        return _svg_empty("idw-provenance-chart", "ARM2 IDW Provenance Composition", "No valid caps in provenance rows.")
    left, top, width, height = 72, 32, 790, 220
    step = width / max(1, len(caps))
    bar_w = max(16.0, min(42.0, step * 0.58))
    max_total = 1.0
    avg_rows: list[tuple[int, float, float, float]] = []
    for cap in caps:
        b = by_cap[cap]
        count = max(1.0, b["count"])
        observed = b["observed"] / count
        idw_exact = b["idw_exact"] / count
        fallback = b["fallback"] / count
        avg_rows.append((cap, observed, idw_exact, fallback))
        max_total = max(max_total, observed + idw_exact + fallback)

    stacks: list[str] = []
    for idx, (cap, observed, idw_exact, fallback) in enumerate(avg_rows):
        x = left + idx * step + (step - bar_w) / 2.0
        h1 = (observed / max_total) * height
        h2 = (idw_exact / max_total) * height
        h3 = (fallback / max_total) * height
        y3 = top + height - h3
        y2 = y3 - h2
        y1 = y2 - h1
        stacks.append(
            f"<rect x='{x:.2f}' y='{y1:.2f}' width='{bar_w:.2f}' height='{h1:.2f}' fill='#2c6f93'><title>cap {cap}: observed avg {_fmt_number(observed,1)}</title></rect>"
            f"<rect x='{x:.2f}' y='{y2:.2f}' width='{bar_w:.2f}' height='{h2:.2f}' fill='#7fb0c8'><title>cap {cap}: IDW/exact avg {_fmt_number(idw_exact,1)}</title></rect>"
            f"<rect x='{x:.2f}' y='{y3:.2f}' width='{bar_w:.2f}' height='{h3:.2f}' fill='#c5792b'><title>cap {cap}: fallback avg {_fmt_number(fallback,1)}</title></rect>"
            f"<text x='{x + bar_w / 2:.2f}' y='{top+height+16:.2f}' text-anchor='middle' style='font-size:9px;fill:#566575;'>{cap}</text>"
        )

    return (
        "<figure class='chart-shell keep-together' id='idw-provenance-chart'>"
        "<figcaption><strong>ARM2 IDW Provenance Stacked Composition by Cap</strong><br><span class='small-note'>Values are per-cap averages across trials.</span></figcaption>"
        "<svg class='chart static-svg' viewBox='0 0 900 360' role='img' aria-labelledby='idw-provenance-chart-title'>"
        "<title id='idw-provenance-chart-title'>ARM2 IDW Provenance Stacked Composition by Cap</title>"
        f"<line x1='{left}' y1='{top+height}' x2='{left+width}' y2='{top+height}' stroke='#728291'></line>"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+height}' stroke='#728291'></line>"
        f"<text x='{left+width/2:.0f}' y='{top+height+34}' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Cap</text>"
        f"<text x='20' y='{top + height/2:.0f}' transform='rotate(-90 20,{top + height/2:.0f})' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Average Count</text>"
        + "".join(stacks)
        + "<g transform='translate(88,288)'><rect x='0' y='0' width='10' height='10' fill='#2c6f93'></rect><text x='14' y='9' style='font-size:11px;'>Observed</text><rect x='86' y='0' width='10' height='10' fill='#7fb0c8'></rect><text x='100' y='9' style='font-size:11px;'>IDW/Exact</text><rect x='170' y='0' width='10' height='10' fill='#c5792b'></rect><text x='184' y='9' style='font-size:11px;'>Fallback Groups</text></g>"
        + "</svg></figure>"
    )


def _idw_quality_svg(idw_rows: list[dict[str, Any]]) -> str:
    if not idw_rows:
        return _svg_empty("idw-quality-chart", "ARM2 IDW Quality Trends vs Cap", "No ARM2 quality rows available.")
    metrics = {
        "aggregate": "absolute_aggregate_rate_error",
        "unjudged_mae": "unjudged_only_mae",
        "brier": "brier_score",
        "ece": "expected_calibration_error",
    }
    by_cap: dict[int, dict[str, list[float]]] = {}
    for row in idw_rows:
        cap = _safe_int(row.get("cap"), default=0)
        if cap <= 0:
            continue
        cap_bucket = by_cap.setdefault(cap, {k: [] for k in metrics})
        for key, field in metrics.items():
            cap_bucket[key].append(_safe_float(row.get(field), default=0.0))
    caps = sorted(by_cap)
    if not caps:
        return _svg_empty("idw-quality-chart", "ARM2 IDW Quality Trends vs Cap", "No valid caps in IDW rows.")
    lines_data: dict[str, list[tuple[int, float]]] = {}
    for key in metrics:
        lines_data[key] = [
            (cap, (sum(by_cap[cap][key]) / len(by_cap[cap][key])) if by_cap[cap][key] else 0.0)
            for cap in caps
        ]
    y_vals = [v for pts in lines_data.values() for _, v in pts]
    y_min, y_max = min(y_vals), max(y_vals)
    if y_max <= y_min:
        y_max = y_min + 0.01
    left, top, width, height = 72, 32, 790, 220

    def x_pos(cap: int) -> float:
        if len(caps) == 1:
            return left + width / 2.0
        return left + (caps.index(cap) / (len(caps) - 1)) * width

    def y_pos(value: float) -> float:
        return top + (1.0 - ((value - y_min) / (y_max - y_min))) * height

    colors = {
        "aggregate": "#2c6f93",
        "unjudged_mae": "#b24c3b",
        "brier": "#2f8f66",
        "ece": "#c5792b",
    }
    labels = {
        "aggregate": "Aggregate MAE",
        "unjudged_mae": "Unjudged MAE",
        "brier": "Brier",
        "ece": "ECE",
    }
    lines: list[str] = []
    legends: list[str] = []
    for i, (key, pts) in enumerate(lines_data.items()):
        color = colors[key]
        points = " ".join(f"{x_pos(cap):.2f},{y_pos(val):.2f}" for cap, val in pts)
        lines.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2'></polyline>")
        for cap, val in pts:
            tip = f"{labels[key]} cap {cap}: {_fmt_number(val,4)}"
            lines.append(f"<circle cx='{x_pos(cap):.2f}' cy='{y_pos(val):.2f}' r='2.8' fill='{color}'><title>{escape(tip)}</title></circle>")
        legends.append(f"<g transform='translate({88 + i*144},288)'><rect x='0' y='0' width='10' height='10' fill='{color}'></rect><text x='14' y='9' style='font-size:11px;'>{labels[key]}</text></g>")

    return (
        "<figure class='chart-shell keep-together' id='idw-quality-chart'>"
        "<figcaption><strong>ARM2 Quality Trends vs Cap</strong><br><span class='small-note'>Trend lines are per-cap trial means.</span></figcaption>"
        "<svg class='chart static-svg' viewBox='0 0 900 360' role='img' aria-labelledby='idw-quality-chart-title'>"
        "<title id='idw-quality-chart-title'>ARM2 Quality Trends vs Cap</title>"
        f"<line x1='{left}' y1='{top+height}' x2='{left+width}' y2='{top+height}' stroke='#728291'></line>"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+height}' stroke='#728291'></line>"
        f"<text x='{left+width/2:.0f}' y='{top+height+34}' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Cap</text>"
        f"<text x='20' y='{top + height/2:.0f}' transform='rotate(-90 20,{top + height/2:.0f})' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Error / Calibration</text>"
        + "".join(lines)
        + "".join(legends)
        + "</svg></figure>"
    )


def _maven_distribution_svg(class_summary: dict[str, Any]) -> str:
    status = class_summary.get("status") if isinstance(class_summary.get("status"), list) else []
    confidence = class_summary.get("confidence_levels") if isinstance(class_summary.get("confidence_levels"), list) else []
    bars = [(f"status:{k}", int(v), "#2c6f93") for k, v in status] + [(f"confidence:{k}", int(v), "#c5792b") for k, v in confidence]
    if not bars:
        return _svg_empty("maven-status-confidence-chart", "Maven Status/Confidence Distribution", "No classification summary available.")
    max_v = max(v for _, v, _ in bars)
    left, top, width, height = 68, 32, 800, 220
    step = width / max(1, len(bars))
    bar_w = max(10.0, min(28.0, step * 0.72))
    body: list[str] = []
    for idx, (label, val, color) in enumerate(bars):
        h = (val / max(1, max_v)) * height
        x = left + idx * step + (step - bar_w) / 2.0
        y = top + (height - h)
        body.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w:.2f}' height='{h:.2f}' fill='{color}'><title>{escape(label)}={val}</title></rect>"
            f"<text x='{x + bar_w/2:.2f}' y='{top+height+16:.2f}' text-anchor='middle' style='font-size:8px;fill:#566575;'>{escape(label[:10])}</text>"
        )
    return (
        "<figure class='chart-shell keep-together' id='maven-status-confidence-chart'>"
        "<figcaption><strong>Maven Status + Confidence Distribution</strong><br><span class='small-note'>Bars summarize classifier output status and confidence buckets.</span></figcaption>"
        "<svg class='chart static-svg' viewBox='0 0 900 360' role='img' aria-labelledby='maven-status-confidence-chart-title'>"
        "<title id='maven-status-confidence-chart-title'>Maven Status and Confidence Distribution</title>"
        f"<line x1='{left}' y1='{top+height}' x2='{left+width}' y2='{top+height}' stroke='#728291'></line>"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+height}' stroke='#728291'></line>"
        f"<text x='20' y='{top + height/2:.0f}' transform='rotate(-90 20,{top + height/2:.0f})' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Count</text>"
        + "".join(body)
        + "</svg></figure>"
    )


def _similarity_histogram_svg(classifications: list[dict[str, Any]]) -> str:
    vals = [_safe_float_or_none(row.get("combined_cosine_similarity")) for row in classifications]
    values = [v for v in vals if v is not None]
    if not values:
        return _svg_empty("maven-similarity-histogram", "Maven Similarity Histogram", "No cosine similarity values available.")
    bins = [0] * 10
    for value in values:
        clamped = min(0.9999, max(0.0, value))
        idx = int(clamped * 10)
        bins[idx] += 1
    max_v = max(bins) if bins else 1
    left, top, width, height = 68, 32, 800, 220
    step = width / 10.0
    body: list[str] = []
    for idx, count in enumerate(bins):
        h = (count / max(1, max_v)) * height
        x = left + idx * step + 2
        y = top + (height - h)
        lo = idx / 10.0
        hi = (idx + 1) / 10.0
        body.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{step-4:.2f}' height='{h:.2f}' fill='#7fb0c8'><title>{lo:.1f}-{hi:.1f}: {count}</title></rect>"
            f"<text x='{x + (step-4)/2:.2f}' y='{top+height+16:.2f}' text-anchor='middle' style='font-size:8px;fill:#566575;'>{lo:.1f}</text>"
        )
    return (
        "<figure class='chart-shell keep-together' id='maven-similarity-histogram'>"
        "<figcaption><strong>Maven Similarity Histogram</strong><br><span class='small-note'>Histogram over combined cosine similarity scores.</span></figcaption>"
        "<svg class='chart static-svg' viewBox='0 0 900 360' role='img' aria-labelledby='maven-similarity-histogram-title'>"
        "<title id='maven-similarity-histogram-title'>Maven Similarity Histogram</title>"
        f"<line x1='{left}' y1='{top+height}' x2='{left+width}' y2='{top+height}' stroke='#728291'></line>"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+height}' stroke='#728291'></line>"
        f"<text x='{left+width/2:.0f}' y='{top+height+34}' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Cosine Similarity Bin</text>"
        f"<text x='20' y='{top + height/2:.0f}' transform='rotate(-90 20,{top + height/2:.0f})' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Count</text>"
        + "".join(body)
        + "</svg></figure>"
    )


def _top_five_heatmap_svg(top_five_summary: list[dict[str, Any]]) -> str:
    rows = top_five_summary
    if not rows:
        return _svg_empty("top-five-error-heatmap", "Top-Five Agent Absolute Error Heatmap", "No top-five rows available.")
    unique_methods = sorted({str(r.get("method_id") or "unknown") for r in rows})
    unique_caps = sorted({_safe_int(r.get("cap"), default=0) for r in rows})
    if not unique_methods or not unique_caps:
        return _svg_empty("top-five-error-heatmap", "Top-Five Agent Absolute Error Heatmap", "No method/cap combinations available.")
    cells: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        method_id = str(row.get("method_id") or "unknown")
        cap = _safe_int(row.get("cap"), default=0)
        error = _safe_float_or_none((row.get("absolute_error") or {}).get("mean"))
        if error is not None:
            cells.setdefault((method_id, cap), []).append(error)
    values = [sum(v) / len(v) for v in cells.values() if v]
    if not values:
        return _svg_empty("top-five-error-heatmap", "Top-Five Agent Absolute Error Heatmap", "No absolute error values available.")
    vmin, vmax = min(values), max(values)
    if vmax <= vmin:
        vmax = vmin + 0.01
    left, top = 150, 40
    cell_w = 120
    cell_h = 28

    def color_for(value: float) -> str:
        t = (value - vmin) / (vmax - vmin)
        r = int(44 + (178 * t))
        g = int(143 - (70 * t))
        b = int(147 - (64 * t))
        return f"#{r:02x}{g:02x}{b:02x}"

    body: list[str] = []
    for r_index, method_id in enumerate(unique_methods):
        y = top + r_index * cell_h
        body.append(f"<text x='{left-8}' y='{y+18}' text-anchor='end' style='font-size:10px;fill:#425363;'>{escape(_method_label(method_id)[:18])}</text>")
        for c_index, cap in enumerate(unique_caps):
            x = left + c_index * cell_w
            cell_values = cells.get((method_id, cap), [])
            if not cell_values:
                body.append(
                    f"<rect x='{x}' y='{y}' width='{cell_w-2}' height='{cell_h-2}' fill='#edf1f4' stroke='#ffffff'>"
                    f"<title>{escape(_method_label(method_id))} cap {cap}: unavailable</title></rect>"
                    f"<text x='{x + (cell_w-2)/2:.2f}' y='{y+18}' text-anchor='middle' style='font-size:9px;fill:#687783;'>N/A</text>"
                )
                continue
            avg = sum(cell_values) / len(cell_values)
            color = color_for(avg)
            body.append(
                f"<rect x='{x}' y='{y}' width='{cell_w-2}' height='{cell_h-2}' fill='{color}' stroke='#ffffff'><title>{escape(_method_label(method_id))} cap {cap}: abs error {_fmt_number(avg,4)}</title></rect>"
                f"<text x='{x + (cell_w-2)/2:.2f}' y='{y+18}' text-anchor='middle' style='font-size:9px;fill:#13202b;'>{_fmt_number(avg,3)}</text>"
            )
    for c_index, cap in enumerate(unique_caps):
        x = left + c_index * cell_w + (cell_w-2)/2
        body.append(f"<text x='{x:.2f}' y='{top-8}' text-anchor='middle' style='font-size:10px;fill:#425363;'>cap {cap}</text>")

    total_w = left + len(unique_caps) * cell_w + 30
    total_h = top + len(unique_methods) * cell_h + 60
    return (
        "<figure class='chart-shell keep-together' id='top-five-error-heatmap'>"
        "<figcaption><strong>Top-Five Agent Absolute Error Heatmap by Method/Cap</strong><br><span class='small-note'>Cell values are mean absolute error across top-five rows.</span></figcaption>"
        f"<svg class='chart static-svg' viewBox='0 0 {total_w} {total_h}' role='img' aria-labelledby='top-five-error-heatmap-title'>"
        "<title id='top-five-error-heatmap-title'>Top-Five Agent Absolute Error Heatmap</title>"
        + "".join(body)
        + "</svg></figure>"
    )


def _worked_examples_html(runs: list[dict[str, Any]], memberships: list[dict[str, Any]]) -> str:
    arm2 = next((row for row in runs if str(row.get("method_id")) == "arm2_embedding_idw"), None)
    arm3 = next((row for row in memberships if str(row.get("method_id")) == "arm3_agent_round_robin_floor"), None)

    if arm2:
        prov = arm2.get("idw_provenance") if isinstance(arm2.get("idw_provenance"), dict) else {}
        quality = arm2.get("idw_quality") if isinstance(arm2.get("idw_quality"), dict) else {}
        selected_rate = "N/A"
        selected_err = "N/A"
        if arm2.get("selected_only_absolute_error") is not None:
            selected_err = _fmt_number(arm2.get("selected_only_absolute_error"), 4)
        if arm2.get("selected_rate") is not None:
            selected_rate = _fmt_pct(arm2.get("selected_rate"), 2)
        idw_rate = "N/A"
        idw_err = "N/A"
        if arm2.get("estimate") is not None:
            idw_rate = _fmt_pct(arm2.get("estimate"), 2)
        if quality.get("absolute_aggregate_rate_error") is not None:
            idw_err = _fmt_number(quality.get("absolute_aggregate_rate_error"), 4)
        arm2_block = (
            "<article class='worked-example'>"
            "<h4>Worked Example: ARM2 IDW Row</h4>"
            f"<p>Cap={_safe_int(arm2.get('cap'), default=0)}, seed={_safe_int(arm2.get('seed'), default=0)}, observed={_safe_int(prov.get('observed_count'), default=0)}, imputed={_safe_int(prov.get('imputed_count'), default=0)}.</p>"
            f"<p>Selected-only rate/error: {escape(selected_rate)} / {escape(selected_err)}. IDW estimated rate/error: {escape(idw_rate)} / {escape(idw_err)}.</p>"
            "</article>"
        )
    else:
        arm2_block = "<article class='worked-example'><h4>Worked Example: ARM2 IDW Row</h4><p>N/A</p></article>"

    if arm3:
        completion = arm3.get("arm3_floor_completion")
        completion_text = _fmt_pct(completion, 2) if completion is not None else "N/A"
        arm3_block = (
            "<article class='worked-example'>"
            "<h4>Worked Example: ARM3 Membership Floor</h4>"
            f"<p>Cap={_safe_int(arm3.get('cap'), default=0)}, floor target={_safe_int(arm3.get('total_floor_target'), default=0)}, prefix selected={_safe_int(arm3.get('floor_prefix_count'), default=0)}, completion={completion_text}.</p>"
            "</article>"
        )
    else:
        arm3_block = "<article class='worked-example'><h4>Worked Example: ARM3 Membership Floor</h4><p>N/A</p></article>"

    return "<div class='two-col'>" + arm2_block + arm3_block + "</div>"


def _generated_methodology_html(
    aggregate_rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    memberships: list[dict[str, Any]],
    methodology_markdown: str,
) -> str:
    trial_count = len({_safe_int(r.get("seed"), default=0) for r in runs if _safe_int(r.get("seed"), default=0) > 0})
    caps = _caps_from_data(aggregate_rows, runs)
    cap_text = ", ".join(str(cap) for cap in caps) if caps else "N/A"
    has_30_trials = trial_count >= 30
    trial_note = "A 30-trial spread is available in this bundle." if has_30_trials else "A 30-trial spread is not yet available in this bundle."
    membership_rows = _membership_summary(memberships)
    floor_min = next((row.get("arm3_floor_min_per_agent") for row in membership_rows if str(row.get("method_id")) == "arm3_agent_round_robin_floor" and _safe_int(row.get("arm3_floor_min_per_agent"), default=0) > 0), 3)

    generated = (
        "<h3>Generated Methodology Narrative (Artifact-Driven)</h3>"
        "<p>This narrative is synthesized from artifact fields and does not depend on optional markdown richness.</p>"
        f"<p>Population and replay setup: caps={escape(cap_text)}; observed trial count={trial_count}. {escape(trial_note)}</p>"
        "<ol>"
        "<li>Population and labels: full-session packet units are labeled with expected outcomes and attached metadata.</li>"
        "<li>Five-arm design: ARM1 global random, ARM2 embedding-IDW, ARM3 round-robin with floor, ARM4 round-robin strata balancing, ARM5 same-membership Hajek weighted estimator.</li>"
        "<li>Cap conversion: sample budget uses a 15k tokens/session planning conversion.</li>"
        "<li>Maven classifier: confidence thresholds 0.30 and 0.70 partition ambiguous vs stronger assignments.</li>"
        "<li>Embedding and clustering: full-session packet embeddings are grouped through Azure Search neighborhood retrieval for donor selection.</li>"
        "<li>IDW donor chain: observed value, exact/donor interpolation, then agent/global/prior fallback groups when donor evidence is sparse.</li>"
        f"<li>ARM3 floor then strata: minimum floor target first (configured min-per-agent={_safe_int(floor_min, default=3)}), then remaining budget allocated by strata.</li>"
        "<li>ARM4 stratification: round-robin strata exposure without hard floor completion requirement.</li>"
        "<li>ARM5 same-membership Hajek estimator: uses membership-aligned weighted ratio estimator for aggregate rates.</li>"
        "<li>Metrics and spread: MAE, concept/use-case/agent coverage, token ratio, Brier and ECE; trial spread reported as min/max descriptive ranges.</li>"
        "<li>Trial pairing and seeds: runs are paired by cap and seed; spread is descriptive and not inferential significance testing.</li>"
        "<li>Limitations: classifier ambiguity, sparse donor contexts, and cap-limited coverage can bias observed rates.</li>"
        "</ol>"
        "<h3>Core Equations</h3>"
        "<div class='equation-block'>"
        "<p><strong>Cap from token budget:</strong> <math><mi>n</mi><mo>=</mo><mo>&lfloor;</mo><mfrac><mi>B</mi><mn>15000</mn></mfrac><mo>&rfloor;</mo></math></p>"
        "<p><strong>Absolute aggregate MAE:</strong> <math><mtext>MAE</mtext><mo>=</mo><mfrac><mn>1</mn><mi>K</mi></mfrac><munderover><mo>&sum;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></munderover><mo>|</mo><mover><mi>r</mi><mo>^</mo></mover><msub><mi></mi><mi>k</mi></msub><mo>-</mo><msub><mi>r</mi><mi>k</mi></msub><mo>|</mo></math></p>"
        "<p><strong>Concept coverage:</strong> <math><mi>C</mi><msub><mi>concept</mi><mi></mi></msub><mo>=</mo><mfrac><mrow><mo>|</mo><msub><mi>U</mi><mi>selected\u2229concept</mi></msub><mo>|</mo></mrow><mrow><mo>|</mo><msub><mi>U</mi><mi>concept</mi></msub><mo>|</mo></mrow></mfrac></math></p>"
        "<p><strong>IDW estimate:</strong> <math><mover><mi>y</mi><mo>^</mo></mover><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mfrac><munderover><mo>&sum;</mo><mi>i</mi><mi>m</mi></munderover><mrow><mi>w</mi><mi>i</mi><mo>(</mo><mi>x</mi><mo>)</mo><mi>y</mi><mi>i</mi></mrow><munderover><mo>&sum;</mo><mi>i</mi><mi>m</mi></munderover><mrow><mi>w</mi><mi>i</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mfrac>,<mspace width='0.4em'/><mi>w</mi><mi>i</mi><mo>=</mo><mfrac><mn>1</mn><msup><mi>d</mi><mi>p</mi></msup></mfrac></math></p>"
        "<p><strong>Hajek estimator:</strong> <math><mover><mi>R</mi><mo>^</mo></mover><msub><mi>H</mi><mi></mi></msub><mo>=</mo><mfrac><munderover><mo>&sum;</mo><mi>i</mi><mi>n</mi></munderover><mfrac><msub><mi>y</mi><mi>i</mi></msub><msub><mi>\u03c0</mi><mi>i</mi></msub></mfrac><munderover><mo>&sum;</mo><mi>i</mi><mi>n</mi></munderover><mfrac><mn>1</mn><msub><mi>\u03c0</mi><mi>i</mi></msub></mfrac></mfrac></math></p>"
        "</div>"
    )

    sanitized_md = _render_methodology_markdown(methodology_markdown)
    return generated + "<h3>Source Methodology (Sanitized Markdown)</h3>" + sanitized_md


def _executive_summary_html(
    aggregate_rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    class_summary: dict[str, Any],
    trial_count: int,
) -> str:
    if not aggregate_rows:
        return "<p class='empty-state'>No aggregate rows available for executive summary.</p>"

    by_cap: dict[int, list[dict[str, Any]]] = {}
    winner_counts: dict[str, int] = {}
    median_winner_counts: dict[str, int] = {}
    winner_disagreements: list[str] = []
    for row in aggregate_rows:
        cap = _safe_int(row.get("cap"), default=0)
        if cap > 0:
            by_cap.setdefault(cap, []).append(row)
    for cap, rows in by_cap.items():
        ranked_mean = sorted(rows, key=lambda item: _safe_float((item.get("metrics") or {}).get("absolute_aggregate_mae", {}).get("mean"), default=1e9))
        ranked_median = sorted(rows, key=lambda item: _safe_float((item.get("metrics") or {}).get("absolute_aggregate_mae", {}).get("median"), default=1e9))
        if ranked_mean and ranked_median:
            mean_winner = str(ranked_mean[0].get("method_id") or "unknown")
            median_winner = str(ranked_median[0].get("method_id") or "unknown")
            winner_counts[mean_winner] = winner_counts.get(mean_winner, 0) + 1
            median_winner_counts[median_winner] = median_winner_counts.get(median_winner, 0) + 1
            if mean_winner != median_winner:
                winner_disagreements.append(
                    f"cap {cap}: mean {_method_label(mean_winner)}, median {_method_label(median_winner)}"
                )

    winner_line = ", ".join(
        f"{_method_label(method_id)}: {count} cap winner(s)"
        for method_id, count in sorted(winner_counts.items(), key=lambda item: (-item[1], item[0]))
    ) or "No winner counts available"
    median_winner_line = ", ".join(
        f"{_method_label(method_id)}: {count} cap median winner(s)"
        for method_id, count in sorted(median_winner_counts.items(), key=lambda item: (-item[1], item[0]))
    ) or "No median winner counts available"
    disagreement_line = (
        "Mean/median winner disagreements: " + "; ".join(winner_disagreements)
        if winner_disagreements
        else "Mean and median MAE winners agree at every cap."
    )

    arm1_cov = []
    arm5_cov = []
    for row in aggregate_rows:
        cov = _safe_float((row.get("metrics") or {}).get("use_case_coverage", {}).get("mean"), default=0.0)
        if str(row.get("method_id")) == "arm1_global_random":
            arm1_cov.append(cov)
        if str(row.get("method_id")) == "arm5_hajek_weighted":
            arm5_cov.append(cov)
    coverage_tradeoff = "N/A"
    if arm1_cov and arm5_cov:
        status_counts = {
            str(key).strip().casefold(): value
            for key, value in class_summary.get("status") or []
        }
        ambiguous_count = _safe_int(status_counts.get("ambiguous"), default=0)
        classification_count = sum(_safe_int(value, default=0) for value in status_counts.values())
        ambiguous_rate = ambiguous_count / classification_count if classification_count else 0.0
        coverage_tradeoff = (
            f"ARM5 vs ARM1 use-case coverage delta: "
            f"{(sum(arm5_cov)/len(arm5_cov) - sum(arm1_cov)/len(arm1_cov))*100.0:+.2f} pp; "
            f"classification-derived and provisional because {_fmt_pct(ambiguous_rate,1)} of assignments are Ambiguous"
        )

    idw_pairs = [
        (row.get("metrics", {}).get("absolute_aggregate_mae"), row.get("selected_only_absolute_error"))
        for row in runs
        if str(row.get("method_id")) == "arm2_embedding_idw"
    ]
    better = sum(1 for mae, sel in idw_pairs if sel is not None and _safe_float(mae, default=0.0) < _safe_float(sel, default=0.0))
    worse = sum(1 for mae, sel in idw_pairs if sel is not None and _safe_float(mae, default=0.0) > _safe_float(sel, default=0.0))
    compared = sum(1 for _, sel in idw_pairs if sel is not None)
    ties = compared - better - worse
    idw_line = (
        f"IDW vs selected-only MAE: better in {better}/{compared}, worse in {worse}/{compared}, tied in {ties}/{compared} runs"
        if compared
        else "IDW vs selected-only MAE: N/A"
    )

    arm1_64_row = next(
        (row for row in by_cap.get(64, []) if str(row.get("method_id")) == "arm1_global_random"),
        None,
    )
    arm5_64_row = next(
        (row for row in by_cap.get(64, []) if str(row.get("method_id")) == "arm5_hajek_weighted"),
        None,
    )
    small_cap_line = "Small-cap ARM5 penalty unavailable."
    if arm1_64_row and arm5_64_row:
        arm1_64_mae = _safe_float(
            (arm1_64_row.get("metrics") or {}).get("absolute_aggregate_mae", {}).get("mean"),
            default=0.0,
        )
        arm5_64_mae = _safe_float(
            (arm5_64_row.get("metrics") or {}).get("absolute_aggregate_mae", {}).get("mean"),
            default=0.0,
        )
        small_cap_line = (
            f"At cap 64, ARM5 mean MAE {_fmt_number(arm5_64_mae,4)} vs ARM1 {_fmt_number(arm1_64_mae,4)}; "
            "ARM5 is not broadly competitive at small caps"
        )

    token_ratios = [
        _safe_float(row.get("actual_token_count"), default=0.0) / _safe_float(row.get("nominal_budget"), default=1.0)
        for row in runs
        if _safe_float(row.get("nominal_budget"), default=0.0) > 0
    ]
    token_line = (
        f"Token ratio median {_fmt_pct(sorted(token_ratios)[len(token_ratios)//2],2)}; range {_fmt_pct(min(token_ratios),2)} to {_fmt_pct(max(token_ratios),2)}"
        if token_ratios
        else "Token ratio diagnostics unavailable"
    )

    confidence = class_summary.get("confidence_levels") if isinstance(class_summary.get("confidence_levels"), list) else []
    confidence_text = ", ".join(f"{k}:{v}" for k, v in confidence) or "none"
    trial_note = "30-trial descriptive spread now available." if trial_count >= 30 else "30-trial descriptive spread pending additional artifacts."

    return (
        "<div class='executive-summary'>"
        "<p><strong>Executive Summary:</strong> Descriptive synthesis only; no inferential significance claims are made.</p>"
        f"<ul><li>Mean MAE winners by cap: {escape(winner_line)}</li>"
        f"<li>Median MAE winners by cap: {escape(median_winner_line)}</li>"
        f"<li>{escape(disagreement_line)}</li>"
        f"<li>{escape(small_cap_line)}</li>"
        f"<li>Coverage tradeoff: {escape(coverage_tradeoff)}</li>"
        f"<li>{escape(idw_line)}</li>"
        f"<li>{escape(token_line)}</li>"
        f"<li>Maven ambiguity caveat: confidence distribution {escape(confidence_text)}</li>"
        f"<li>{escape(trial_note)}</li></ul>"
        "</div>"
    )


def _build_static_overview_table(aggregate_rows: list[dict[str, Any]]) -> str:
    if not aggregate_rows:
        return "<p class='empty-state'>No aggregate rows available.</p>"
    rows = []
    for row in sorted(aggregate_rows, key=lambda r: (r.get("cap", 0), str(r.get("method_id") or ""))):
        rows.append(
            "<tr>"
            f"<td>{escape(_method_label(str(row.get('method_id') or 'unknown')))}</td>"
            f"<td>{_safe_int(row.get('cap'), default=0)}</td>"
            f"<td>{_fmt_number((row.get('metrics') or {}).get('absolute_aggregate_mae', {}).get('mean', 0.0), 4)}</td>"
            f"<td>{_fmt_pct((row.get('metrics') or {}).get('concept_coverage', {}).get('mean', 0.0), 2)}</td>"
            f"<td>{_fmt_pct((row.get('metrics') or {}).get('use_case_coverage', {}).get('mean', 0.0), 2)}</td>"
            f"<td>{_fmt_pct((row.get('metrics') or {}).get('agent_coverage', {}).get('mean', 0.0), 2)}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr><th>Method</th><th>Cap</th><th>MAE</th><th>Concept Coverage</th><th>Use-Case Coverage</th><th>Agent Coverage</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _build_token_table(aggregate_rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> str:
    combined: list[dict[str, Any]] = []
    if aggregate_rows:
        for aggregate in aggregate_rows:
            nominal = _safe_float(aggregate.get("nominal_budget"), default=0.0)
            actual = _safe_float(aggregate.get("actual_token_count"), default=0.0)
            combined.append(
                {
                    "method_id": aggregate.get("method_id"),
                    "cap": aggregate.get("cap"),
                    "nominal": nominal,
                    "actual": actual,
                    "ratio": _token_ratio(nominal, actual),
                }
            )
    else:
        grouped: dict[tuple[str, int], dict[str, list[float]]] = {}
        for run in runs:
            key = (str(run.get("method_id") or "unknown"), _safe_int(run.get("cap"), default=0))
            bucket = grouped.setdefault(key, {"nominal": [], "actual": []})
            bucket["nominal"].append(_safe_float(run.get("nominal_budget"), default=0.0))
            bucket["actual"].append(_safe_float(run.get("actual_token_count"), default=0.0))
        for (method_id, cap), bucket in grouped.items():
            nominal = sum(bucket["nominal"]) / len(bucket["nominal"]) if bucket["nominal"] else 0.0
            actual = sum(bucket["actual"]) / len(bucket["actual"]) if bucket["actual"] else 0.0
            combined.append(
                {
                    "method_id": method_id,
                    "cap": cap,
                    "nominal": nominal,
                    "actual": actual,
                    "ratio": _token_ratio(nominal, actual),
                }
            )

    if not combined:
        return "<p class='empty-state'>No token rows available.</p>"

    rows = []
    for row in sorted(combined, key=lambda r: (_safe_int(r.get("cap"), default=0), str(r.get("method_id") or ""))):
        rows.append(
            "<tr>"
            f"<td>{escape(_method_label(str(row.get('method_id') or 'unknown')))}</td>"
            f"<td>{_safe_int(row.get('cap'), default=0)}</td>"
            f"<td>{_fmt_number(row.get('nominal'), 0)}</td>"
            f"<td>{_fmt_number(row.get('actual'), 0)}</td>"
            f"<td>{_fmt_pct(row.get('ratio'), 2)}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table class='compact-table'><thead><tr><th>Method</th><th>Cap</th><th>Nominal Budget</th><th>Actual Tokens</th><th>Actual/Nominal</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _token_ratios(aggregate_rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[float]:
    ratios: list[float] = []
    for row in runs:
        nominal = _safe_float(row.get("nominal_budget"), default=0.0)
        actual = _safe_float(row.get("actual_token_count"), default=0.0)
        if nominal > 0:
            ratios.append(actual / nominal)
    if not ratios:
        for row in aggregate_rows:
            nominal = _safe_float(row.get("nominal_budget"), default=0.0)
            actual = _safe_float(row.get("actual_token_count"), default=0.0)
            if nominal > 0:
                ratios.append(actual / nominal)
    return ratios


def _token_interpretation_text(aggregate_rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> str:
    ratios = _token_ratios(aggregate_rows, runs)

    if not ratios:
        return "No token ratio diagnostics were available. The 15k-per-session value is a planning conversion and not a hard token pack."

    ratios_sorted = sorted(ratios)
    median_ratio = ratios_sorted[len(ratios_sorted) // 2]
    ratio_std = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0
    p05 = _linear_interpolate_quantile(ratios, 0.05)
    p95 = _linear_interpolate_quantile(ratios, 0.95)
    return (
        f"Observed actual/nominal ratios span {_fmt_pct(min(ratios), 2)} to {_fmt_pct(max(ratios), 2)} with median {_fmt_pct(median_ratio, 2)}, SD {_fmt_pct(ratio_std, 2)}, and empirical p05-p95 {_fmt_pct(p05, 2)} to {_fmt_pct(p95, 2)}. "
        "Nominal budget uses a 15k-per-session planning conversion, not a hard token pack."
    )


def _maven_quality_caveat(class_summary: dict[str, Any]) -> str:
    confidence_levels = class_summary.get("confidence_levels") if isinstance(class_summary.get("confidence_levels"), list) else []
    statuses = class_summary.get("status") if isinstance(class_summary.get("status"), list) else []
    conf_text = ", ".join(f"{k}:{v}" for k, v in confidence_levels) or "none"
    status_text = ", ".join(f"{k}:{v}" for k, v in statuses) or "none"

    dominant = confidence_levels[0][0] if confidence_levels else "none"
    dominant_lower = str(dominant).strip().lower().replace("_", "-")
    ambiguous_like = dominant_lower in {"ambiguous", "level-1", "level 1", "level1", "1"}

    if ambiguous_like:
        tail = "Ambiguous/level-1 confidence dominates, so use-case coverage should be treated as provisional."
    else:
        tail = "Use-case coverage remains provisional until classification confidence is externally audited."
    return f"Confidence distribution: {conf_text}. Status distribution: {status_text}. {tail}"


def _conclusion_html(aggregate_rows: list[dict[str, Any]], runs: list[dict[str, Any]], trial_count: int) -> str:
    if not aggregate_rows:
        return "<p>No aggregate rows were available for conclusion synthesis.</p>"

    by_cap: dict[int, list[dict[str, Any]]] = {}
    for row in aggregate_rows:
        cap = _safe_int(row.get("cap"), default=0)
        if cap <= 0:
            continue
        by_cap.setdefault(cap, []).append(row)

    winner_counts: dict[str, int] = {}
    median_winner_counts: dict[str, int] = {}
    cap_winner_lines: list[str] = []
    winner_disagreements: list[str] = []
    for cap in sorted(by_cap):
        rows = by_cap[cap]
        ranked = sorted(
            rows,
            key=lambda r: _safe_float((r.get("metrics") or {}).get("absolute_aggregate_mae", {}).get("mean"), default=1e9),
        )
        ranked_median = sorted(
            rows,
            key=lambda r: _safe_float((r.get("metrics") or {}).get("absolute_aggregate_mae", {}).get("median"), default=1e9),
        )
        if not ranked or not ranked_median:
            continue
        winner = str(ranked[0].get("method_id") or "unknown")
        median_winner = str(ranked_median[0].get("method_id") or "unknown")
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        median_winner_counts[median_winner] = median_winner_counts.get(median_winner, 0) + 1
        cap_winner_lines.append(
            f"Cap {cap}: mean {_method_label(winner)} ({_fmt_number((ranked[0].get('metrics') or {}).get('absolute_aggregate_mae', {}).get('mean', 0.0), 4)}); "
            f"median {_method_label(median_winner)} ({_fmt_number((ranked_median[0].get('metrics') or {}).get('absolute_aggregate_mae', {}).get('median', 0.0), 4)})"
        )
        if winner != median_winner:
            winner_disagreements.append(
                f"cap {cap}: mean {_method_label(winner)}, median {_method_label(median_winner)}"
            )

    winner_line = "No MAE winners could be computed by cap."
    if winner_counts:
        winner_line = "; ".join(
            f"{_method_label(method_id)} won {count} cap(s)"
            for method_id, count in sorted(winner_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        )
    median_winner_line = "; ".join(
        f"{_method_label(method_id)} won {count} cap median(s)"
        for method_id, count in sorted(median_winner_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ) or "No median MAE winners could be computed by cap."
    disagreement_line = (
        "Mean/median disagreement requires caution: " + "; ".join(winner_disagreements)
        if winner_disagreements
        else "Mean and median MAE winners agree at every cap."
    )

    mae_by_method_cap: dict[tuple[str, int], float] = {}
    use_case_by_method_cap: dict[tuple[str, int], float] = {}
    concept_by_method_cap: dict[tuple[str, int], float] = {}
    for row in aggregate_rows:
        method_id = str(row.get("method_id") or "unknown")
        cap = _safe_int(row.get("cap"), default=0)
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        mae_by_method_cap[(method_id, cap)] = _safe_float((metrics.get("absolute_aggregate_mae") or {}).get("mean"), default=0.0)
        use_case_by_method_cap[(method_id, cap)] = _safe_float((metrics.get("use_case_coverage") or {}).get("mean"), default=0.0)
        concept_by_method_cap[(method_id, cap)] = _safe_float((metrics.get("concept_coverage") or {}).get("mean"), default=0.0)

    coverage_lines: list[str] = []
    for method_id in ("arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"):
        diffs_use_case: list[float] = []
        diffs_concept: list[float] = []
        for cap in sorted(by_cap):
            k1 = ("arm1_global_random", cap)
            kx = (method_id, cap)
            if k1 in use_case_by_method_cap and kx in use_case_by_method_cap:
                diffs_use_case.append(use_case_by_method_cap[kx] - use_case_by_method_cap[k1])
            if k1 in concept_by_method_cap and kx in concept_by_method_cap:
                diffs_concept.append(concept_by_method_cap[kx] - concept_by_method_cap[k1])
        if diffs_use_case or diffs_concept:
            uc_avg = (sum(diffs_use_case) / len(diffs_use_case)) if diffs_use_case else 0.0
            c_avg = (sum(diffs_concept) / len(diffs_concept)) if diffs_concept else 0.0
            coverage_lines.append(
                f"Vs ARM1, {_method_label(method_id)} shifts average use-case coverage by {uc_avg * 100.0:+.2f} pp and concept coverage by {c_avg * 100.0:+.2f} pp."
            )

    caps_sorted = sorted(by_cap)
    small_caps = [cap for cap in caps_sorted if cap <= 256] or caps_sorted[: max(1, len(caps_sorted) // 2)]
    high_caps = [cap for cap in caps_sorted if cap >= 512] or caps_sorted[max(0, len(caps_sorted) // 2) :]

    def arm5_diff_for_caps(caps_subset: list[int]) -> list[float]:
        diffs: list[float] = []
        for cap in caps_subset:
            arm5_key = ("arm5_hajek_weighted", cap)
            if arm5_key not in mae_by_method_cap:
                continue
            other = [value for (mid, c), value in mae_by_method_cap.items() if c == cap and mid != "arm5_hajek_weighted"]
            if not other:
                continue
            diffs.append(mae_by_method_cap[arm5_key] - min(other))
        return diffs

    arm5_small = arm5_diff_for_caps(small_caps)
    arm5_high = arm5_diff_for_caps(high_caps)
    arm5_line = "ARM5 relative MAE trend could not be computed for the available caps."
    if arm5_small or arm5_high:
        small_avg = (sum(arm5_small) / len(arm5_small)) if arm5_small else 0.0
        high_avg = (sum(arm5_high) / len(arm5_high)) if arm5_high else 0.0
        arm5_line = (
            f"ARM5 weighted MAE gap vs best non-ARM5 is {small_avg:+.4f} on smaller caps and {high_avg:+.4f} on higher caps."
        )
        if small_avg > 0.01 and high_avg <= 0.005:
            arm5_line += " This indicates materially worse MAE at small caps, but competitive or better behavior at higher caps."

    arm2_rows = [row for row in runs if str(row.get("method_id")) == "arm2_embedding_idw"]
    arm2_compared = [
        (
            _safe_float(row.get("metrics", {}).get("absolute_aggregate_mae"), default=0.0),
            _safe_float_or_none(row.get("selected_only_absolute_error")),
        )
        for row in arm2_rows
    ]
    arm2_pairs = [(mae, selected_only) for mae, selected_only in arm2_compared if selected_only is not None]
    arm2_line = "ARM2 IDW comparison to selected-only absolute error is unavailable in these rows."
    if arm2_pairs:
        better = sum(1 for mae, selected_only in arm2_pairs if mae < float(selected_only))
        worse = sum(1 for mae, selected_only in arm2_pairs if mae > float(selected_only))
        tie = len(arm2_pairs) - better - worse
        arm2_line = (
            f"ARM2 IDW MAE vs selected-only absolute error: better in {better} run(s), worse in {worse} run(s), tied in {tie} run(s)."
        )

    descriptive_line = f"Only {trial_count} trial(s) are available; these comparisons are descriptive rather than inferential."

    coverage_html = "".join(f"<li>{escape(line)}</li>" for line in coverage_lines) or "<li>No overlapping caps were available for ARM1 coverage comparisons.</li>"
    cap_winner_html = "".join(f"<li>{escape(line)}</li>" for line in cap_winner_lines) or "<li>No per-cap MAE winner rows were available.</li>"

    return (
        "<p>Per-cap MAE winners are listed below, with winner counts summarized across available caps.</p>"
        f"<ul>{cap_winner_html}</ul>"
        f"<p>Mean winners: {escape(winner_line)}. Median winners: {escape(median_winner_line)}.</p>"
        f"<p>{escape(disagreement_line)}</p>"
        "<h3>Coverage Tradeoffs</h3>"
        f"<ul>{coverage_html}</ul>"
        f"<p>{escape(arm5_line)}</p>"
        f"<p>{escape(arm2_line)}</p>"
        f"<p>{escape(descriptive_line)}</p>"
    )


def _run_metric_value(row: dict[str, Any], metric: str) -> float:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    normalized = metrics.get(metric)
    if normalized is not None:
        return _safe_float(normalized, default=0.0)
    aliases = METRIC_ALIASES.get(metric, (metric,))
    return _safe_float(_first_present(row, *aliases), default=0.0)


def _paired_seed_comparison_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for run in runs:
        method_id = str(run.get("method_id") or "unknown")
        cap = _safe_int(run.get("cap"), default=0)
        seed = _safe_int(run.get("seed"), default=0)
        if cap <= 0 or seed <= 0:
            continue
        by_key[(method_id, cap, seed)] = run

    comparisons: list[dict[str, Any]] = []
    for method_id in ("arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"):
        for cap in sorted({cap for (mid, cap, _seed) in by_key if mid in {"arm1_global_random", method_id}}):
            rows = []
            for seed in sorted({seed for (mid, c, seed) in by_key if c == cap and mid in {"arm1_global_random", method_id}}):
                a = by_key.get(("arm1_global_random", cap, seed))
                b = by_key.get((method_id, cap, seed))
                if a is None or b is None:
                    continue
                rows.append((seed, a, b))
            if not rows:
                continue
            mae_values = []
            concept_values = []
            use_case_values = []
            agent_values = []
            for _, a, b in rows:
                a_mae = _run_metric_value(a, "absolute_aggregate_mae")
                b_mae = _run_metric_value(b, "absolute_aggregate_mae")
                a_concept = _run_metric_value(a, "concept_coverage")
                b_concept = _run_metric_value(b, "concept_coverage")
                a_use_case = _run_metric_value(a, "use_case_coverage")
                b_use_case = _run_metric_value(b, "use_case_coverage")
                a_agent = _run_metric_value(a, "agent_coverage")
                b_agent = _run_metric_value(b, "agent_coverage")
                mae_values.append(b_mae - a_mae)
                concept_values.append(b_concept - a_concept)
                use_case_values.append(b_use_case - a_use_case)
                agent_values.append(b_agent - a_agent)
            if mae_values:
                comparisons.append({
                    "method_id": method_id,
                    "cap": cap,
                    "mae_delta": _rich_stat_block(mae_values),
                    "concept_delta": _rich_stat_block(concept_values),
                    "use_case_delta": _rich_stat_block(use_case_values),
                    "agent_delta": _rich_stat_block(agent_values),
                    "mae_win_rate": float(sum(1 for value in mae_values if value < 0.0) / len(mae_values)),
                    "concept_win_rate": float(sum(1 for value in concept_values if value > 0.0) / len(concept_values)),
                    "use_case_win_rate": float(sum(1 for value in use_case_values if value > 0.0) / len(use_case_values)),
                    "agent_win_rate": float(sum(1 for value in agent_values if value > 0.0) / len(agent_values)),
                    "n_pairs": len(mae_values),
                })

    for cap in sorted({int(r.get("cap")) for r in runs if _safe_int(r.get("cap"), default=0) > 0 and str(r.get("method_id")) == "arm2_embedding_idw"}):
        pairs = []
        for run in runs:
            if str(run.get("method_id")) != "arm2_embedding_idw" or _safe_int(run.get("cap"), default=0) != cap:
                continue
            selected = _safe_float_or_none(run.get("selected_only_absolute_error"))
            if selected is None:
                continue
            mae = _run_metric_value(run, "absolute_aggregate_mae")
            pairs.append(mae - selected)
        if pairs:
            comparisons.append({
                "method_id": "arm2_selected_only",
                "cap": cap,
                "mae_delta": _rich_stat_block(pairs),
                "mae_win_rate": float(sum(1 for value in pairs if value < 0.0) / len(pairs)),
                "n_pairs": len(pairs),
                "label": "ARM2 vs selected-only",
            })

    arm4_by_key = {(int(r.get("seed")), int(r.get("cap"))): r for r in runs if str(r.get("method_id")) == "arm4_agent_round_robin"}
    arm5_by_key = {(int(r.get("seed")), int(r.get("cap"))): r for r in runs if str(r.get("method_id")) == "arm5_hajek_weighted"}
    common_keys = sorted(set(arm4_by_key) & set(arm5_by_key))
    for cap in sorted({cap for _seed, cap in common_keys}):
        mae_deltas: list[float] = []
        estimate_deltas: list[float] = []
        for seed, key_cap in common_keys:
            if key_cap != cap:
                continue
            arm4 = arm4_by_key[(seed, key_cap)]
            arm5 = arm5_by_key[(seed, key_cap)]
            arm4_mae = _run_metric_value(arm4, "absolute_aggregate_mae")
            arm5_mae = _run_metric_value(arm5, "absolute_aggregate_mae")
            mae_deltas.append(arm5_mae - arm4_mae)
            estimate_deltas.append(
                _safe_float(arm5.get("estimate"), default=0.0)
                - _safe_float(arm4.get("estimate"), default=0.0)
            )
        if mae_deltas:
            comparisons.append({
                "method_id": "arm5_vs_arm4",
                "cap": cap,
                "mae_delta": _rich_stat_block(mae_deltas),
                "estimate_delta": _rich_stat_block(estimate_deltas),
                "mae_win_rate": float(sum(1 for value in mae_deltas if value < 0.0) / len(mae_deltas)),
                "n_pairs": len(mae_deltas),
                "label": "ARM5 vs ARM4",
            })
    return comparisons


def _descriptive_stability_section(runs: list[dict[str, Any]]) -> str:
    comparisons = _paired_seed_comparison_summary(runs)
    method_caps = []
    pair_map = {}
    for item in comparisons:
        if item.get("method_id") in {"arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"}:
            pair_map.setdefault(item["method_id"], []).append(item)
    for method_id in ("arm1_global_random", "arm2_embedding_idw", "arm3_agent_round_robin_floor", "arm4_agent_round_robin", "arm5_hajek_weighted"):
        group = [row for row in runs if str(row.get("method_id")) == method_id]
        if not group:
            continue
        by_cap = {}
        for row in group:
            cap = _safe_int(row.get("cap"), default=0)
            by_cap.setdefault(cap, []).append(row)
        for cap, rows_for_cap in sorted(by_cap.items()):
            mae_values = [_run_metric_value(row, "absolute_aggregate_mae") for row in rows_for_cap]
            if not mae_values:
                continue
            delta_vs_arm1 = []
            for row in rows_for_cap:
                arm1 = next((r for r in runs if str(r.get("method_id")) == "arm1_global_random" and _safe_int(r.get("cap"), default=0) == cap and _safe_int(r.get("seed"), default=0) == _safe_int(row.get("seed"), default=0)), None)
                if arm1 is None:
                    continue
                arm1_mae = _run_metric_value(arm1, "absolute_aggregate_mae")
                delta_vs_arm1.append(_run_metric_value(row, "absolute_aggregate_mae") - arm1_mae)
            method_caps.append({
                "method_id": method_id,
                "cap": cap,
                "mae_mean": float(sum(mae_values) / len(mae_values)),
                "mae_median": float(_linear_interpolate_quantile(mae_values, 0.50)),
                "mae_sd": float(np.std(mae_values, ddof=1)) if len(mae_values) > 1 else 0.0,
                "mae_p05": float(_linear_interpolate_quantile(mae_values, 0.05)),
                "mae_p95": float(_linear_interpolate_quantile(mae_values, 0.95)),
                "mae_win_rate": (
                    None
                    if method_id == "arm1_global_random"
                    else float(sum(1 for value in delta_vs_arm1 if value < 0.0) / len(delta_vs_arm1)) if delta_vs_arm1 else 0.0
                ),
                "trial_count": len(mae_values),
            })

    if not method_caps:
        return "<p class='empty-state'>No 30-trial descriptive stability rows were available.</p>"

    rows_html = []
    for item in method_caps:
        spread_text = f"{_fmt_number(item['mae_p05'], 4)}–{_fmt_number(item['mae_p95'], 4)}" if item['trial_count'] >= 10 else f"min={_fmt_number(min(item['mae_mean'], item['mae_median']), 4)} max={_fmt_number(max(item['mae_mean'], item['mae_median']), 4)}"
        rows_html.append(
            "<tr>"
            f"<td>{escape(_method_label(item['method_id']))}</td><td>{item['cap']}</td><td>{_fmt_number(item['mae_mean'], 4)}</td><td>{_fmt_number(item['mae_median'], 4)}</td><td>{_fmt_number(item['mae_sd'], 4)}</td><td>{spread_text}</td><td>{'N/A' if item['mae_win_rate'] is None else _fmt_pct(item['mae_win_rate'], 1)}</td></tr>"
        )

    paired_rows = []
    for item in comparisons:
        if "mae_delta" not in item:
            continue
        method_id = str(item.get("method_id") or "unknown")
        if item.get("label"):
            comparison_label = str(item["label"])
        else:
            comparison_label = f"{_method_label(method_id)} vs ARM1"
        estimate_delta = item.get("estimate_delta") if isinstance(item.get("estimate_delta"), dict) else None
        estimate_text = "N/A" if estimate_delta is None else _fmt_number(estimate_delta.get("mean"), 4)
        use_case_delta = item.get("use_case_delta") if isinstance(item.get("use_case_delta"), dict) else None
        use_case_text = "N/A" if use_case_delta is None else _fmt_pct(use_case_delta.get("mean"), 2)
        paired_rows.append(
            "<tr>"
            f"<td>{escape(comparison_label)}</td><td>{item.get('cap', '')}</td>"
            f"<td>{_fmt_number(item['mae_delta'].get('mean'), 4)}</td>"
            f"<td>{_fmt_number(item['mae_delta'].get('median'), 4)}</td>"
            f"<td>{_fmt_number(item['mae_delta'].get('sample_std'), 4)}</td>"
            f"<td>{_fmt_number(item['mae_delta'].get('p05'), 4)}–{_fmt_number(item['mae_delta'].get('p95'), 4)}</td>"
            f"<td>{_fmt_pct(item.get('mae_win_rate', 0.0), 1)}</td>"
            f"<td>{escape(use_case_text)}</td><td>{escape(estimate_text)}</td></tr>"
        )

    svg_points = []
    for idx, item in enumerate(sorted(method_caps, key=lambda x: (x['cap'], x['method_id']))[:12]):
        x = 100 + idx * 68
        y = 180 - (item['mae_median'] * 140)
        svg_points.append(f"<line x1='{x}' y1='180' x2='{x}' y2='{y}' stroke='#2c6f93' stroke-width='2'></line><circle cx='{x}' cy='{y}' r='4' fill='#2c6f93'></circle>")
    svg_html = (
        "<svg class='chart static-svg' viewBox='0 0 900 260' role='img' aria-label='MAE median descriptive stability'>"
        "<rect x='12' y='20' width='876' height='220' fill='#fafcff' stroke='#d5dde5'></rect>"
        "<line x1='52' y1='180' x2='858' y2='180' stroke='#728291'></line>"
        "<line x1='52' y1='30' x2='52' y2='180' stroke='#728291'></line>"
        + "".join(svg_points)
        + "<text x='450' y='245' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Method/cap</text>"
        + "<text x='18' y='110' transform='rotate(-90 18,110)' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>MAE median</text>"
        + "</svg>"
    )

    return (
        "<p><strong>Caveat:</strong> This is fixed-population descriptive stability across the observed seeds, not a p-value test or a statement about the general population. Empirical 5th-95th percentile descriptive spread is used when at least 10 trials are present; otherwise the bundle shows limited trial support and falls back to min/max.</p>"
        "<div class='table-wrap'><table class='compact-table'><thead><tr><th>Method</th><th>Cap</th><th>MAE Mean</th><th>Median</th><th>SD</th><th>Spread</th><th>Win Rate vs ARM1</th></tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div>"
        "<h3>Paired comparisons</h3>"
        "<div class='table-wrap screen-only'><table class='compact-table'><thead><tr><th>Comparison</th><th>Cap</th><th>MAE Delta Mean</th><th>Median</th><th>SD</th><th>p05–p95</th><th>MAE Win Rate</th><th>Use-Case Delta</th><th>Estimate Delta</th></tr></thead><tbody>" + ("".join(paired_rows) if paired_rows else "<tr><td colspan='9'>No paired comparisons available.</td></tr>") + "</tbody></table></div>"
        "<div class='table-wrap print-only'><table class='compact-table print-compact'><thead><tr><th>Comparison</th><th>Cap</th><th>MAE Delta</th><th>p05–p95</th><th>Win Rate</th><th>Estimate Delta</th></tr></thead><tbody>" + ("".join(
            "<tr>"
            f"<td>{escape(str(item.get('label') or (_method_label(str(item.get('method_id') or 'unknown')) + ' vs ARM1')))}</td>"
            f"<td>{item.get('cap', '')}</td><td>{_fmt_number(item['mae_delta'].get('mean'), 4)}</td>"
            f"<td>{_fmt_number(item['mae_delta'].get('p05'), 4)}–{_fmt_number(item['mae_delta'].get('p95'), 4)}</td>"
            f"<td>{_fmt_pct(item.get('mae_win_rate', 0.0), 1)}</td>"
            f"<td>{_fmt_number((item.get('estimate_delta') or {}).get('mean'), 4) if isinstance(item.get('estimate_delta'), dict) else 'N/A'}</td></tr>"
            for item in comparisons if "mae_delta" in item
        ) if comparisons else "<tr><td colspan='6'>No paired comparisons available.</td></tr>") + "</tbody></table></div>"
        "<h3>MAE median stability (static)</h3>" + svg_html
    )


def _aggregate_row_lookup(aggregate_rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in aggregate_rows:
        out[(str(row.get("method_id") or "unknown"), _safe_int(row.get("cap"), default=0))] = row
    return out


def _metric_mean(row: dict[str, Any], metric: str) -> float:
    return _safe_float(((row.get("metrics") or {}).get(metric) or {}).get("mean"), default=0.0)


def _metric_median(row: dict[str, Any], metric: str) -> float:
    return _safe_float(((row.get("metrics") or {}).get(metric) or {}).get("median"), default=0.0)


def _recommendation_matrix_html(aggregate_rows: list[dict[str, Any]]) -> str:
    by_key = _aggregate_row_lookup(aggregate_rows)

    def fmt_arm(method_id: str, cap: int) -> str:
        row = by_key.get((method_id, cap))
        if not row:
            return "N/A"
        mae = _metric_mean(row, "absolute_aggregate_mae")
        cov = _metric_mean(row, "use_case_coverage")
        return f"MAE={_fmt_number(mae,4)}, UC={_fmt_pct(cov,1)}"

    matrix_rows = [
        ("Accuracy-first", "64/128/256", "ARM1", "Lowest mean MAE on all three small caps.", f"ARM1 {fmt_arm('arm1_global_random',64)} | ARM5 {fmt_arm('arm5_hajek_weighted',64)} (small-cap penalty)."),
        ("Balanced at mid-high cap", "512", "ARM5", "Best mean MAE at 512 while retaining higher representation than ARM1.", f"ARM5 {fmt_arm('arm5_hajek_weighted',512)} vs ARM1 {fmt_arm('arm1_global_random',512)}."),
        ("High-cap near tie", "1024", "ARM5 (mean) / ARM1 (median)", "Near-tie with mean-median disagreement.", "Mean MAE favors ARM5, median MAE favors ARM1; treat as operational tie pending shadow run."),
        ("Per-agent floor required", "Any", "ARM3", "Only choose when explicit minimum per-agent representation is a hard requirement.", "ARM3 floor introduces error tradeoff for representation guarantees."),
        ("Imputation study", "Any", "ARM2", "Use ARM2 for embedding/IDW calibration analysis, not as MAE winner.", "IDW improves some runs and worsens others; not the aggregate MAE leader."),
    ]

    body = "".join(
        "<tr>"
        f"<td>{escape(obj)}</td>"
        f"<td>{escape(cap)}</td>"
        f"<td>{escape(rec)}</td>"
        f"<td>{escape(reason)}</td>"
        f"<td>{escape(caveat)}</td>"
        "</tr>"
        for obj, cap, rec, reason, caveat in matrix_rows
    )
    return (
        "<div class='table-wrap'><table class='compact-table'><thead><tr><th>Objective</th><th>Cap</th><th>Recommendation</th><th>Why</th><th>Magnitude/Caveat</th></tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def _top_use_case_human_html(class_summary: dict[str, Any]) -> str:
    rows = class_summary.get("top_use_case_labels") if isinstance(class_summary.get("top_use_case_labels"), list) else []
    if not rows:
        return "<p class='empty-state'>No use-case rows were available.</p>"
    items: list[str] = []
    for (guid, label), count in rows[:12]:
        items.append(
            "<li>"
            f"<strong>{escape(label)}</strong> <span class='small-note'>({escape(guid)})</span>: {count}"
            "</li>"
        )
    return "<ul>" + "".join(items) + "</ul>"


def _source_composition_svg(dataset: dict[str, Any]) -> str:
    summary = dataset.get("source_summary") if isinstance(dataset.get("source_summary"), dict) else {}
    corpora = summary.get("corpora") if isinstance(summary.get("corpora"), list) else summary.get("by_corpus") if isinstance(summary.get("by_corpus"), list) else []
    if not corpora:
        return _svg_empty("source-composition-chart", "Source Composition: Counts and Pass Rates", "No corpus composition rows available.")
    valid = [row for row in corpora if isinstance(row, dict)]
    if not valid:
        return _svg_empty("source-composition-chart", "Source Composition: Counts and Pass Rates", "No valid corpus composition rows available.")

    left, top, width, height = 76, 32, 770, 220
    step = width / max(1, len(valid))
    bar_w = max(26.0, min(72.0, step * 0.48))
    max_units = max(1, max(_safe_int(row.get("unit_count"), default=0) for row in valid))
    bars: list[str] = []
    for idx, row in enumerate(valid):
        corpus_id = str(row.get("corpus_id") or "unknown")
        units = _safe_int(row.get("unit_count"), default=0)
        pass_rate = _safe_float(row.get("pass_rate"), default=0.0)
        x = left + idx * step + (step - bar_w) / 2.0
        h = (units / max_units) * height
        y = top + (height - h)
        bars.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w:.2f}' height='{h:.2f}' fill='#6c8fa8'><title>{escape(corpus_id)} units={units}</title></rect>"
            f"<line x1='{x:.2f}' y1='{top + height - (pass_rate * height):.2f}' x2='{x+bar_w:.2f}' y2='{top + height - (pass_rate * height):.2f}' stroke='#b24c3b' stroke-width='3'><title>{escape(corpus_id)} pass_rate={_fmt_pct(pass_rate,2)}</title></line>"
            f"<text x='{x + bar_w/2:.2f}' y='{top+height+16:.2f}' text-anchor='middle' style='font-size:10px;fill:#566575;'>{escape(corpus_id)}</text>"
            f"<text x='{x + bar_w/2:.2f}' y='{max(y-4, top+10):.2f}' text-anchor='middle' style='font-size:10px;fill:#324353;'>{units}</text>"
        )
    return (
        "<figure class='chart-shell keep-together' id='source-composition-chart'>"
        "<figcaption><strong>Source Composition: Counts and Pass Rates</strong><br><span class='small-note'>Bars are unit counts; red markers are source pass rates.</span></figcaption>"
        "<svg class='chart static-svg' viewBox='0 0 900 360' role='img' aria-labelledby='source-composition-chart-title'>"
        "<title id='source-composition-chart-title'>Source Composition Counts and Pass Rates</title>"
        f"<line x1='{left}' y1='{top+height}' x2='{left+width}' y2='{top+height}' stroke='#728291'></line>"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+height}' stroke='#728291'></line>"
        f"<text x='{left+width/2:.0f}' y='{top+height+34}' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Corpus</text>"
        f"<text x='20' y='{top + height/2:.0f}' transform='rotate(-90 20,{top + height/2:.0f})' text-anchor='middle' style='font-size:12px;fill:#3f4d5a;'>Unit Count / Pass Rate Marker</text>"
        + "".join(bars)
        + "</svg></figure>"
    )


def _pipeline_svg() -> str:
    return (
        "<figure class='chart-shell keep-together' id='pipeline-diagram'>"
        "<figcaption><strong>Data Pipeline Overview</strong><br><span class='small-note'>Source corpora to scoring with frozen memberships.</span></figcaption>"
        "<svg class='chart static-svg' viewBox='0 0 900 260' role='img' aria-labelledby='pipeline-diagram-title'>"
        "<title id='pipeline-diagram-title'>Pipeline from source corpora to expected-label scoring</title>"
        "<rect x='26' y='80' width='168' height='60' rx='8' fill='#e7f0f6' stroke='#8aa8bc'></rect>"
        "<text x='110' y='105' text-anchor='middle' style='font-size:11px;fill:#2a3a47;'>Source corpora</text>"
        "<text x='110' y='121' text-anchor='middle' style='font-size:10px;fill:#4f6070;'>dense_2500 + historical_300</text>"
        "<rect x='224' y='80' width='168' height='60' rx='8' fill='#edf5ef' stroke='#8aa893'></rect>"
        "<text x='308' y='105' text-anchor='middle' style='font-size:11px;fill:#2a3a47;'>Normalized sessions</text>"
        "<text x='308' y='121' text-anchor='middle' style='font-size:10px;fill:#4f6070;'>EvaluationUnit rows</text>"
        "<rect x='422' y='80' width='168' height='60' rx='8' fill='#fff2e6' stroke='#c7a27a'></rect>"
        "<text x='506' y='105' text-anchor='middle' style='font-size:11px;fill:#2a3a47;'>Maven assignment</text>"
        "<text x='506' y='121' text-anchor='middle' style='font-size:10px;fill:#4f6070;'>classification-derived labels</text>"
        "<rect x='620' y='80' width='168' height='60' rx='8' fill='#f8eced' stroke='#c08a8d'></rect>"
        "<text x='704' y='105' text-anchor='middle' style='font-size:11px;fill:#2a3a47;'>Frozen membership</text>"
        "<text x='704' y='121' text-anchor='middle' style='font-size:10px;fill:#4f6070;'>sample selection complete</text>"
        "<line x1='194' y1='110' x2='224' y2='110' stroke='#657a8c' marker-end='url(#arrow)'></line>"
        "<line x1='392' y1='110' x2='422' y2='110' stroke='#657a8c' marker-end='url(#arrow)'></line>"
        "<line x1='590' y1='110' x2='620' y2='110' stroke='#657a8c' marker-end='url(#arrow)'></line>"
        "<rect x='620' y='170' width='168' height='52' rx='8' fill='#e9eef8' stroke='#8c9dbc'></rect>"
        "<text x='704' y='195' text-anchor='middle' style='font-size:11px;fill:#2a3a47;'>Expected-label scoring</text>"
        "<line x1='704' y1='140' x2='704' y2='170' stroke='#657a8c' marker-end='url(#arrow)'></line>"
        "<defs><marker id='arrow' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='6' markerHeight='6' orient='auto-start-reverse'><path d='M 0 0 L 10 5 L 0 10 z' fill='#657a8c'></path></marker></defs>"
        "</svg></figure>"
    )


def _figure_takeaway_html(heading: str, text: str) -> str:
    return f"<p class='small-note'><strong>{escape(heading)}:</strong> {escape(text)}</p>"


def _method_matrix_html() -> str:
    rows = [
        ("ARM1", "Global deterministic random", "Unweighted mean", "Best low-cap aggregate accuracy baseline", "Can miss minority agents/use-cases at small caps"),
        ("ARM2", "Azure Search adaptive selector over full-session embeddings", "Same-agent IDW over full population", "Model-assisted estimation with donor reuse", "Donor/fallback chain can amplify calibration drift"),
        ("ARM3", "One-per-agent round-robin to min(3, N_agent), then Maven strata", "Unweighted mean", "Explicit per-agent floor representation", "Floor pressure can hurt MAE under tight caps"),
        ("ARM4", "Capacity-aware per-agent round robin + proportional use-case strata", "Unweighted mean", "Coverage-focused membership without hard floor", "No formal minimum per-agent guarantee"),
        ("ARM5", "Exact ARM4 membership", "Hajek ratio estimator with inclusion probabilities", "Same coverage as ARM4 with weighted estimator", "Low-cap weighting instability; small-cap MAE penalty"),
    ]
    body = "".join(
        "<tr>"
        f"<td>{escape(a)}</td><td>{escape(b)}</td><td>{escape(c)}</td><td>{escape(d)}</td><td>{escape(e)}</td>"
        "</tr>"
        for a, b, c, d, e in rows
    )
    return "<div class='table-wrap'><table class='compact-table'><thead><tr><th>Method</th><th>Selector</th><th>Estimator</th><th>Intended Benefit</th><th>Main Risk</th></tr></thead><tbody>" + body + "</tbody></table></div>"


def _build_html_document(inputs: V6ReportInputs, artifacts: LoadedV6Artifacts) -> str:
    aggregate_rows = artifacts.aggregate.get("aggregate_rows") if isinstance(artifacts.aggregate.get("aggregate_rows"), list) else []
    runs = artifacts.runs
    memberships = artifacts.memberships
    classifications = artifacts.classifications
    dataset = artifacts.dataset_examples

    caps = _caps_from_data(aggregate_rows, runs)
    methods = [m for m in FULL_METHOD_IDS if any(str(row.get("method_id")) == m for row in aggregate_rows + runs)]
    if not methods:
        methods = sorted({str(row.get("method_id") or "unknown") for row in aggregate_rows + runs})

    generated_at_raw = str(
        artifacts.aggregate.get("generated_at")
        or artifacts.manifest.get("generated_at")
        or datetime.now(timezone.utc).isoformat()
    )
    try:
        generated_at = datetime.fromisoformat(generated_at_raw.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        generated_at = generated_at_raw

    population_units = _population_count(artifacts.aggregate)
    agent_count = _agent_count_from_data(artifacts.aggregate, runs)
    trial_count = len({(_safe_int(r.get("seed"), default=0)) for r in runs if _safe_int(r.get("seed"), default=0) != 0})
    cap_min = min(caps) if caps else 0
    cap_max = max(caps) if caps else 0
    inferred_status = "complete" if artifacts.runs and aggregate_rows else "unknown"
    manifest_status = str(artifacts.manifest.get("status") or artifacts.aggregate.get("status") or inferred_status)
    run_status = str(artifacts.aggregate.get("run_status") or artifacts.aggregate.get("status") or inferred_status)

    top_five_summary = _build_top_five_summary(runs)
    class_summary = _classification_summary(classifications)
    idw_rows = _idw_summary(runs)
    membership_rows = _membership_summary(memberships)

    token_interpretation = _token_interpretation_text(aggregate_rows, runs)
    executive_summary_html = _executive_summary_html(aggregate_rows, runs, class_summary, trial_count)
    stability_html = _descriptive_stability_section(runs)
    methodology_html = _generated_methodology_html(aggregate_rows, runs, memberships, artifacts.methodology_text)
    taxonomy_html = _taxonomy_tables_html(class_summary)
    idw_table = _idw_tables_html(idw_rows)
    membership_table = _membership_tables_html(membership_rows)
    top_five_print_table = _top_five_print_table(top_five_summary)
    worked_examples_html = _worked_examples_html(runs, membership_rows)

    mae_chart = _line_metric_svg("mae-vs-cap-chart", "MAE vs Cap with Trial Spread", "MAE", aggregate_rows, "absolute_aggregate_mae")
    concept_cov_chart = _line_metric_svg("coverage-concept-chart", "Coverage vs Cap: Source Concept", "Coverage", aggregate_rows, "concept_coverage", as_percent=True)
    maven_cov_chart = _line_metric_svg("coverage-maven-chart", "Coverage vs Cap: Maven Use-Case", "Coverage", aggregate_rows, "use_case_coverage", as_percent=True)
    agent_cov_chart = _line_metric_svg("coverage-agent-chart", "Coverage vs Cap: Agent Coverage", "Coverage", aggregate_rows, "agent_coverage", as_percent=True)
    frontier_chart = _frontier_svg(aggregate_rows)
    token_ratio_chart = _token_ratio_svg(aggregate_rows, runs)
    arm3_floor_chart = _arm3_floor_svg(membership_rows)
    idw_provenance_chart = _idw_provenance_svg(idw_rows)
    idw_quality_chart = _idw_quality_svg(idw_rows)
    maven_status_chart = _maven_distribution_svg(class_summary)
    maven_similarity_chart = _similarity_histogram_svg(classifications)
    top_five_heatmap = _top_five_heatmap_svg(top_five_summary)
    source_composition_chart = _source_composition_svg(dataset)
    pipeline_chart = _pipeline_svg()

    class_top_use_cases = "".join(f"<li>{escape(k)}: {v}</li>" for k, v in class_summary["top_use_cases"])
    class_top_sub_sub = "".join(f"<li>{escape(k)}: {v}</li>" for k, v in class_summary["top_sub_subcategory"])
    conf_levels = "".join(f"<li>{escape(k)}: {v}</li>" for k, v in class_summary["confidence_levels"])
    statuses = "".join(f"<li>{escape(k)}: {v}</li>" for k, v in class_summary["status"])
    top_use_case_human = _top_use_case_human_html(class_summary)
    recommendation_matrix = _recommendation_matrix_html(aggregate_rows)
    method_matrix = _method_matrix_html()

    status_map = {str(k).strip().casefold(): v for k, v in class_summary.get("status", [])}
    conf_map = {str(k).strip().casefold().replace("_", "-"): v for k, v in class_summary.get("confidence_levels", [])}
    amb_n = _safe_int(status_map.get("ambiguous"), default=0)
    cor_n = _safe_int(status_map.get("corroborated"), default=0)
    agree_n = _safe_int(status_map.get("agree"), default=0)
    conf1_n = sum(_safe_int(conf_map.get(key), default=0) for key in ("1", "level-1", "level 1", "level1"))
    low0_n = _safe_int(conf_map.get("undetermined"), default=0)
    ambiguous_rate = amb_n / max(1, len(classifications))

    corpus_diag_rows = class_summary.get("corpus_diagnostics") if isinstance(class_summary.get("corpus_diagnostics"), list) else []
    corpus_diag_map = {str(r.get("corpus_id")): r for r in corpus_diag_rows if isinstance(r, dict)}
    dense_diag = corpus_diag_map.get("dense_2500", {})
    hist_diag = corpus_diag_map.get("historical_300", {})

    source_summary = dataset.get("source_summary") if isinstance(dataset.get("source_summary"), dict) else {}
    overall = source_summary.get("overall") if isinstance(source_summary.get("overall"), dict) else {}
    by_corpus = source_summary.get("corpora") if isinstance(source_summary.get("corpora"), list) else []
    corpus_lookup = {str(r.get("corpus_id")): r for r in by_corpus if isinstance(r, dict)}
    dense = corpus_lookup.get("dense_2500", {})
    hist = corpus_lookup.get("historical_300", {})

    by_key = _aggregate_row_lookup(aggregate_rows)
    arm1_64 = _metric_mean(by_key.get(("arm1_global_random", 64), {}), "absolute_aggregate_mae")
    arm5_64 = _metric_mean(by_key.get(("arm5_hajek_weighted", 64), {}), "absolute_aggregate_mae")
    arm5_1024_mean = _metric_mean(by_key.get(("arm5_hajek_weighted", 1024), {}), "absolute_aggregate_mae")
    arm1_1024_mean = _metric_mean(by_key.get(("arm1_global_random", 1024), {}), "absolute_aggregate_mae")
    arm5_1024_median = _metric_median(by_key.get(("arm5_hajek_weighted", 1024), {}), "absolute_aggregate_mae")
    arm1_1024_median = _metric_median(by_key.get(("arm1_global_random", 1024), {}), "absolute_aggregate_mae")

    token_ratios = _token_ratios(aggregate_rows, runs)
    token_ratio_range = (
        f"{_fmt_pct(min(token_ratios),2)} to {_fmt_pct(max(token_ratios),2)}"
        if token_ratios
        else "unavailable"
    )

    arm3_floor_targets = sorted(
        {
            _safe_int(row.get("total_floor_target"), default=0)
            for row in membership_rows
            if str(row.get("method_id")) == "arm3_agent_round_robin_floor"
            and _safe_int(row.get("total_floor_target"), default=0) > 0
        }
    )
    if len(arm3_floor_targets) == 1:
        arm3_floor_text = f"The artifact-derived floor target totals {arm3_floor_targets[0]} sessions."
    elif arm3_floor_targets:
        arm3_floor_text = "Artifact-derived floor targets vary by cap: " + ", ".join(str(value) for value in arm3_floor_targets) + "."
    else:
        arm3_floor_text = "No positive floor target was present in the membership artifact."

    mean_winner_parts: list[str] = []
    disagreement_parts: list[str] = []
    for cap in caps:
        cap_rows = [row for row in aggregate_rows if _safe_int(row.get("cap"), default=0) == cap]
        if not cap_rows:
            continue
        mean_winner = min(cap_rows, key=lambda row: _metric_mean(row, "absolute_aggregate_mae"))
        median_winner = min(cap_rows, key=lambda row: _metric_median(row, "absolute_aggregate_mae"))
        mean_method = str(mean_winner.get("method_id") or "unknown")
        median_method = str(median_winner.get("method_id") or "unknown")
        mean_winner_parts.append(f"{cap}: {_method_label(mean_method)}")
        if mean_method != median_method:
            disagreement_parts.append(
                f"{cap}: mean {_method_label(mean_method)}, median {_method_label(median_method)}"
            )
    accuracy_takeaway = "Lower MAE is better. Mean MAE winner by cap: " + "; ".join(mean_winner_parts) + "."
    if disagreement_parts:
        accuracy_takeaway += " Mean/median disagreement: " + "; ".join(disagreement_parts) + "."
    else:
        accuracy_takeaway += " Mean and median winners agree at every available cap."

    cap_options = "".join(f"<option value='{cap}'>{cap}</option>" for cap in caps)
    artifact_payload = {
        "aggregate_rows": aggregate_rows,
        "runs": runs,
        "memberships": membership_rows,
        "top_five": top_five_summary,
        "classifications": classifications,
        "classification_summary": class_summary,
        "idw_rows": idw_rows,
        "dataset_examples": dataset,
        "methods": methods,
        "caps": caps,
        "metric_keys": list(METRIC_ALIASES.keys()),
        "labels": {m: _method_label(m) for m in methods},
        "colors": {m: _method_color(m) for m in methods},
    }
    payload_text = _safe_json_script_blob(artifact_payload)

    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8' />
<meta name='viewport' content='width=device-width, initial-scale=1' />
<title>Sampling V6 Report</title>
<style>
:root {{
    --ink:#1b2530;
    --muted:#5a6878;
    --paper:#ffffff;
    --line:#d5dde5;
    --sky:#eaf2f7;
    --sand:#f4efe8;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin:0; padding:0; color:var(--ink); background:linear-gradient(180deg, var(--sky) 0%, #f8f8f8 35%, var(--sand) 100%); font-family:"Segoe UI", Tahoma, sans-serif; }}
.main {{ max-width:1320px; margin:0 auto; padding:28px 20px 48px; }}
.layout {{ display:grid; grid-template-columns:240px minmax(0,1fr); gap:14px; align-items:start; }}
.layout > * {{ min-width:0; }}
.toc {{ position:sticky; top:10px; background:#f9fbfd; border:1px solid var(--line); border-radius:10px; padding:10px; }}
.toc h3 {{ margin:0 0 8px; font-size:0.95rem; }}
.toc a {{ display:block; color:#30495e; text-decoration:none; font-size:0.86rem; margin:4px 0; }}
.toc a:hover {{ text-decoration:underline; }}
header {{ background:var(--paper); border:1px solid var(--line); border-radius:10px; padding:18px; }}
h1 {{ margin:0 0 12px; font-size:2.1rem; letter-spacing:-0.02em; }}
.kicker {{ color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; font-size:11px; margin-bottom:8px; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
.metric {{ min-width:0; background:#f6f8fb; border:1px solid var(--line); border-radius:8px; padding:10px; }}
.metric .l {{ font-size:10px; text-transform:uppercase; color:var(--muted); letter-spacing:0.05em; }}
.metric .v {{ font-size:1.15rem; font-weight:700; margin-top:4px; overflow-wrap:anywhere; }}
section {{ margin-top:20px; background:var(--paper); border:1px solid var(--line); border-radius:10px; padding:14px; }}
.section-title {{ display:flex; justify-content:space-between; align-items:end; border-bottom:1px solid var(--line); padding-bottom:8px; margin-bottom:12px; }}
.section-title h2 {{ margin:0; font-size:1.2rem; }}
.controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:10px; }}
label {{ font-size:0.9rem; color:var(--muted); }}
select, button {{ font:inherit; border:1px solid var(--line); background:#f8fafc; border-radius:6px; padding:6px 10px; }}
button.active {{ background:#254f6b; border-color:#254f6b; color:white; }}
.toggle-group {{ display:flex; flex-wrap:wrap; gap:8px; }}
.toggle {{ display:inline-flex; align-items:center; gap:6px; background:#f9fbfd; border:1px solid var(--line); border-radius:999px; padding:5px 10px; }}
.chart-shell {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }}
svg.chart {{ width:100%; height:auto; display:block; }}
.table-wrap {{ overflow:auto; max-height:none; }}
table {{ width:100%; border-collapse:collapse; min-width:660px; }}
th, td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
th {{ background:#f1f5f9; text-transform:uppercase; letter-spacing:0.04em; font-size:0.72rem; }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.three-col {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }}
ul {{ margin:6px 0; padding-left:18px; }}
.empty-state {{ color:var(--muted); font-style:italic; }}
.print-only {{ display:none; }}
.screen-only {{ display:block; }}
.keep-together {{ break-inside:avoid; page-break-inside:avoid; }}
.small-note {{ color:var(--muted); font-size:0.86rem; }}
.static-svg {{ width:100%; height:auto; max-width:100%; }}
.dataset-stack {{ display:grid; grid-template-columns:1fr; gap:12px; }}
.dataset-card {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fbfcfe; }}
.dataset-card h4 {{ margin:0 0 8px; }}
.dataset-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:10px; }}
.snippet-pair {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
.snippet-pair p {{ margin:6px 0 0; white-space:pre-wrap; overflow-wrap:anywhere; }}
.worked-example {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }}
.equation-block math {{ font-size:1rem; }}
.print-compact {{ min-width:0 !important; table-layout:fixed; width:100%; }}
.print-compact th, .print-compact td {{ overflow-wrap:anywhere; word-break:break-word; }}
@media (max-width: 980px) {{
    .layout {{ display:block; }}
    .toc {{ position:static; margin-bottom:12px; }}
    .main, header, section, .toc {{ width:100%; max-width:100%; }}
    .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .two-col, .three-col {{ grid-template-columns:1fr; }}
    .dataset-grid, .snippet-pair {{ grid-template-columns:1fr; }}
    h1 {{ overflow-wrap:anywhere; }}
}}
@media print {{
    @page {{ size: A4 landscape; margin: 10mm; }}
    body {{ background:white; }}
    .toc {{ position:static; page-break-after:avoid; }}
    .metrics {{ grid-template-columns:repeat(4,minmax(0,1fr)); }}
    .controls {{ display:none !important; }}
    .print-only {{ display:block !important; }}
    .screen-only {{ display:none !important; }}
    section, header {{ break-inside:auto; page-break-inside:auto; }}
    .table-wrap {{ overflow:visible !important; max-width:100%; }}
    table {{ min-width:0 !important; table-layout:fixed; width:100%; font-size:7.5pt; }}
    th, td {{ padding:2px 4px; overflow-wrap:anywhere; word-break:break-word; }}
    thead {{ display:table-header-group; }}
    tr {{ page-break-inside:auto; break-inside:auto; }}
    .page-break {{ break-before:page; page-break-before:always; }}
    .two-col, .three-col {{ display:block !important; }}
    figure.chart-shell, .keep-together {{
        display:block !important;
        break-inside:avoid-page !important;
        page-break-inside:avoid !important;
        width:100% !important;
        max-height:155mm !important;
        margin:0 0 5mm !important;
        overflow:hidden !important;
    }}
    figure.chart-shell figcaption {{ break-after:avoid-page; page-break-after:avoid; }}
    figure.chart-shell svg, svg.static-svg {{
        display:block !important;
        width:100% !important;
        height:125mm !important;
        max-height:125mm !important;
        break-inside:avoid-page !important;
        page-break-inside:avoid !important;
    }}
}}
</style>
</head>
<body>
<div class='main'>
<div class='layout'>
<nav class='toc screen-only' aria-label='Report contents'>
<h3>Contents</h3>
<a href='#purpose-and-decision'>Purpose and Decision</a>
<a href='#executive-recommendations'>Executive Recommendations</a>
<a href='#dataset-ground-truth'>Dataset and Ground Truth</a>
<a href='#maven-use-case-assignment'>Maven Business Use-Case Assignment</a>
<a href='#experiment-design-metrics'>Experiment Design and Metrics</a>
<a href='#method-guide'>Method Guide</a>
<a href='#comparative-results'>Comparative Results</a>
<a href='#five-largest-agents'>Five Largest Agents: Sentinel Drilldown</a>
<a href='#limitations-validity'>Limitations and Validity</a>
<a href='#recommendations-next-steps'>Recommendations and Next Steps</a>
<a href='#interactive-explorer'>Interactive Explorer</a>
<a href='#appendix'>Appendix</a>
</nav>
<div>
<header>
<div class='kicker'>Sampling V6 bundle report ({escape(REPORT_VERSION)})</div>
<h1>Choosing a Sampling Strategy Under Evaluation Budget Constraints</h1>
<p>Task-completion estimation over a fixed 2,800-session synthetic population under explicit evaluation budget caps.</p>
<div class='metrics'>
<div class='metric'><div class='l'>Population Units</div><div class='v'>{population_units}</div></div>
<div class='metric'><div class='l'>Agent Count</div><div class='v'>{agent_count}</div></div>
<div class='metric'><div class='l'>Methods</div><div class='v'>{len(methods)} (5 expected)</div></div>
<div class='metric'><div class='l'>Trials</div><div class='v'>{trial_count}</div></div>
<div class='metric'><div class='l'>Cap Range</div><div class='v'>{cap_min}-{cap_max}</div></div>
<div class='metric'><div class='l'>Generated At</div><div class='v'>{escape(generated_at)}</div></div>
<div class='metric'><div class='l'>Manifest Status</div><div class='v'>{escape(manifest_status)}</div></div>
<div class='metric'><div class='l'>Run Status</div><div class='v'>{escape(run_status)}</div></div>
</div>
<p><strong>Decision guide:</strong> {escape(accuracy_takeaway)} Use the recommendation matrix below to balance accuracy against representation and operating constraints.</p>
{executive_summary_html}
<p><a href='#executive-recommendations'>Executive recommendation matrix</a> | <a href='#method-guide'>Method definitions and equations</a></p>
</header>

<nav class='toc print-only' aria-label='Printed report contents'>
<h3>Contents</h3>
<a href='#purpose-and-decision'>Purpose and Decision</a>
<a href='#executive-recommendations'>Executive Recommendations</a>
<a href='#dataset-ground-truth'>Dataset and Ground Truth</a>
<a href='#maven-use-case-assignment'>Maven Business Use-Case Assignment</a>
<a href='#experiment-design-metrics'>Experiment Design and Metrics</a>
<a href='#method-guide'>Method Guide</a>
<a href='#comparative-results'>Comparative Results</a>
<a href='#five-largest-agents'>Five Largest Agents: Sentinel Drilldown</a>
<a href='#limitations-validity'>Limitations and Validity</a>
<a href='#recommendations-next-steps'>Recommendations and Next Steps</a>
<a href='#appendix'>Appendix</a>
</nav>

<section id='purpose-and-decision'>
<div class='section-title'><h2>Purpose and Decision</h2><small>Audience, decision target, and evidence ladder</small></div>
<p><strong>Audience:</strong> readers new to the project deciding which sampling method to use under explicit budget caps.</p>
<p><strong>Decision:</strong> choose the method/cap that best balances aggregate accuracy, representational coverage, and operational cost on the fixed 2,800-session synthetic population.</p>
<div class='three-col'>
<div><h3>Aggregate accuracy</h3><p>Absolute aggregate MAE (lower is better).</p></div>
<div><h3>Representational coverage</h3><p>Concept coverage, use-case coverage (classification-derived/provisional), and agent coverage (higher is better).</p></div>
<div><h3>Operational cost</h3><p>Actual/Nominal token ratio relative to a 15k planning conversion.</p></div>
</div>
<h3>Definitions</h3>
<ul>
<li>Cap = per-run selected session count target.</li>
<li>MAE = absolute difference between estimated and census pass rates.</li>
<li>Concept coverage = represented concept share in the selected sample.</li>
<li>Use-case coverage = represented Maven use-case share (classification-derived/provisional).</li>
<li>Agent coverage = represented agent share in selected sessions.</li>
</ul>
<h3>How to read this report</h3>
<ol>
<li>Use Executive Recommendations to shortlist method/cap options.</li>
<li>Check data provenance and Maven ambiguity caveats.</li>
<li>Read question-oriented comparative results and takeaways.</li>
<li>Use the Interactive Explorer and sentinel drilldown for deeper checks.</li>
</ol>
</section>

<section id='executive-recommendations'>
<div class='section-title'><h2>Executive Recommendations</h2><small>Conditional recommendations by objective and cap</small></div>
<p>All guidance is descriptive from 30 seeded deterministic replays on a fixed synthetic population; no inferential claims are made.</p>
{recommendation_matrix}
<p><strong>Small-cap penalty and crossover:</strong> ARM5 mean MAE at cap64 is {_fmt_number(arm5_64,4)} vs ARM1 {_fmt_number(arm1_64,4)}. At cap1024, ARM5 wins mean MAE ({_fmt_number(arm5_1024_mean,4)} vs {_fmt_number(arm1_1024_mean,4)}) while ARM1 wins median MAE ({_fmt_number(arm1_1024_median,4)} vs {_fmt_number(arm5_1024_median,4)}).</p>
</section>

<section id='dataset-ground-truth'>
<div class='section-title'><h2>Dataset and Ground Truth</h2><small>Two corpora, normalization, and bounded evidence examples</small></div>
<p><strong>dense_2500:</strong> 2,500 sessions, 5 named agents (500 each), seed 3652026, scenario/difficulty/variety/behavior flags with templated tool calls and pass/fail narration; outputs/manifests retained while generator source is absent from this workspace.</p>
<p><strong>historical_300:</strong> 300 sessions, 100 agents from BPS converted by synthetic_data/a365_historical_300/build_a365_otel.py; pass/fail copied from upstream expected field with deterministic IDs/timestamps where needed; original BPS authoring is outside this workspace.</p>
<p><strong>Combined census:</strong> {_safe_int(overall.get('unit_count'), default=0)} sessions, 105 agents, pass rate {_fmt_pct(overall.get('pass_rate'),4)}. Source pass rates: dense_2500 {_fmt_pct(dense.get('pass_rate'),4)} and historical_300 {_fmt_pct(hist.get('pass_rate'),4)}.</p>
<p>OTLP-style traces are normalized into EvaluationUnit rows (unit IDs, turns, tools); expected labels are joined only after sample membership is frozen. Source-synthetic fields are distinct from report-derived snippets.</p>
<div class='two-col'>
<div>{source_composition_chart}</div>
<div>{pipeline_chart}</div>
</div>
{_dataset_table_html(dataset)}
<p><strong>What to notice:</strong> four bounded case-study examples cover pass/fail behavior across both corpora and show lineage + shape metadata without exposing full transcripts.</p>
{_dataset_source_summary_html(dataset)}
</section>

<section id='maven-use-case-assignment'>
<div class='section-title'><h2>Maven Business Use-Case Assignment</h2><small>Workflow diagnostics, thresholds, and provisional coverage caveat</small></div>
<p>Workflow: cleaning, per-turn request/response embeddings with text-embedding-3-small, real v6 request/response centroids, top-5 agreement/corroboration/late-fusion, 0.30 low-confidence sentinel, 0.70 session early-stop, otherwise maximum similarity assignment.</p>
<p>This section describes classification behavior diagnostics, not an accuracy benchmark or external validation of the assigned use cases.</p>
<p><strong>Observed bundle:</strong> Ambiguous {amb_n} ({_fmt_pct(amb_n/max(1,len(classifications)),1)}), Corroborated {cor_n}, Agree {agree_n}; confidence level 1={conf1_n}; Below-0.30 fallback={low0_n}; max similarity {_fmt_number(class_summary['similarity']['max'],3)}. Use-case coverage is therefore provisional.</p>
<p><strong>Per-corpus diagnostics:</strong> dense_2500 n={_safe_int(dense_diag.get('n'), default=0)}, mean sim {_fmt_number(dense_diag.get('mean_similarity'),3)}, status {escape(str(dense_diag.get('status')))}. historical_300 n={_safe_int(hist_diag.get('n'), default=0)}, mean sim {_fmt_number(hist_diag.get('mean_similarity'),3)}, status {escape(str(hist_diag.get('status')))}, confidence {escape(str(hist_diag.get('confidence')))}.</p>
<div class='two-col'>
<div>{maven_status_chart}<p class='small-note'><strong>Takeaway:</strong> Status mix is dominated by Ambiguous, so use-case coverage remains classification-derived/provisional.</p></div>
<div>{maven_similarity_chart}<p class='small-note'><strong>What this shows:</strong> Similarity values cluster below the 0.70 early-stop threshold; 0.30 and 0.70 are operational cutoff annotations for weak fallback and strong assignment.</p></div>
</div>
<h3>Top Use Cases (Human-readable labels, GUID secondary)</h3>
{top_use_case_human}
<h3>Compact taxonomy mapping table</h3>
{taxonomy_html}
</section>

<section id='experiment-design-metrics'>
<div class='section-title'><h2>Experiment Design and Metrics</h2><small>Fixed population design, replay pairing, and metric intent</small></div>
<p>Design: fixed population, caps 64/128/256/512/1024, 30 paired deterministic seeds, and membership frozen before expected-label joins. Estimand is pass-rate estimation quality and representation behavior on this fixed population.</p>
<p>Descriptive spread is reported as mean/median/p05/p95; these are not inferential confidence intervals.</p>
<p>15k is a planning-only conversion from token budget to session cap; observed actual/nominal ratios ({escape(token_ratio_range)}) are expected and do not imply selection-budget under-run.</p>
<h3>Metric glossary</h3>
<ul>
<li>Absolute aggregate MAE (lower better): estimator accuracy.</li>
<li>Concept coverage (higher better): source concept representation.</li>
<li>Use-case coverage (higher better, provisional): classification-derived representation.</li>
<li>Agent coverage (higher better): cross-agent representation.</li>
<li>Actual/Nominal token ratio: operational cost context.</li>
</ul>
</section>

<section id='method-guide'>
<div class='section-title'><h2>Method Guide</h2><small>Selector/estimator matrix and method-specific caveats</small></div>
{method_matrix}
<div class='two-col'>
<article class='worked-example'><h3>ARM1 global deterministic random + unweighted mean</h3><p>Strong low-cap MAE baseline with simpler behavior.</p></article>
<article class='worked-example'><h3>ARM2 adaptive selector + same-agent IDW</h3><p>Full-session embeddings and Azure Search donor retrieval with fallback chain; primary use is imputation/calibration study.</p></article>
<article class='worked-example'><h3>ARM3 floor then strata + unweighted mean</h3><p>One-per-agent round robin to min(3, N_agent), then proportional Maven strata. {escape(arm3_floor_text)}</p></article>
<article class='worked-example'><h3>ARM4 capacity-aware round robin + unweighted mean</h3><p>No hard floor; emphasizes broad representation via round-robin and strata.</p></article>
<article class='worked-example'><h3>ARM5 ARM4-membership + Hajek ratio estimator</h3><p>Coverage mirrors ARM4; only estimator differs. Can be unstable at low caps.</p></article>
</div>
<p>Token-cap conversion and equations:</p>
{methodology_html}
<h3>Worked examples</h3>
{worked_examples_html}
</section>

<section id='comparative-results'>
<div class='section-title'><h2>Comparative Results</h2><small>Question-oriented evidence with explicit takeaways</small></div>
<h3>a. Accuracy by cap</h3>
{mae_chart}
<p class='small-note'><strong>Takeaway:</strong> {escape(accuracy_takeaway)}</p>
{_build_static_overview_table(aggregate_rows)}

<h3>b. Representation by cap</h3>
<div class='three-col'>
<div>{concept_cov_chart}<p class='small-note'><strong>Takeaway:</strong> Concept coverage generally rises with cap and is stronger for representation-focused methods.</p></div>
<div>{maven_cov_chart}<p class='small-note'><strong>Takeaway:</strong> Use-case coverage trends are informative but classification-derived/provisional because {_fmt_pct(ambiguous_rate,1)} of assignments are Ambiguous.</p></div>
<div>{agent_cov_chart}<p class='small-note'><strong>Takeaway:</strong> Agent coverage increases with cap; floor/round-robin designs prioritize broader agent representation.</p></div>
</div>

<h3>c. Accuracy-coverage frontier</h3>
{frontier_chart}
<p class='small-note'><strong>What this shows:</strong> Frontier points summarize lower MAE and higher use-case coverage tradeoffs; high-cap ARM5 shifts toward a better combined region.</p>

<h3>d. 30-Trial Descriptive Stability and paired deltas/win rates</h3>
{stability_html}

<h3>e. Operational cost (Token Diagnostics)</h3>
{token_ratio_chart}
{_build_token_table(aggregate_rows, runs)}
<p class='small-note'><strong>Interpretation:</strong> {escape(token_interpretation)}</p>

<h3>f. ARM2 imputation/calibration</h3>
<div class='two-col'>
<div>{idw_provenance_chart}<p class='small-note'><strong>What this shows:</strong> Provenance composition separates observed, IDW/exact, and fallback contribution by cap.</p></div>
<div>{idw_quality_chart}<p class='small-note'><strong>Takeaway:</strong> IDW has both better and worse outcomes across runs; use this arm for imputation/calibration analysis, not aggregate MAE leadership.</p></div>
</div>
{idw_table}

<h3>g. ARM3 floor behavior and ARM4/5 relationship</h3>
{arm3_floor_chart}
<p class='small-note'><strong>Takeaway:</strong> ARM3 floor mechanics explain its representation guarantees; ARM4 and ARM5 share membership while ARM5 changes only the estimator.</p>
{membership_table}
</section>

<section id='five-largest-agents'>
<div class='section-title'><h2>Five Largest Agents: Sentinel Drilldown</h2><small>Five dense 500-session agents only (not all-agent validation)</small></div>
<div class='controls'>
<label for='top-agent-select'>Agent</label>
<select id='top-agent-select'>
<option value=''>All agents</option>
{''.join(f"<option value='{escape(a['agent_id'])}'>{escape(a['agent_id'])}</option>" for a in sorted(top_five_summary, key=lambda x: x['agent_id'])[:120])}
</select>
<label for='top-cap-select'>Cap</label>
<select id='top-cap-select'>
<option value=''>All caps</option>
{cap_options}
</select>
</div>
<div id='top-five-table' class='screen-only'></div>
{top_five_print_table}
{top_five_heatmap}
<p class='small-note'><strong>Observation:</strong> At low caps this sentinel panel shows n=0/1 volatility and missing selections; at higher caps selected rates converge toward census rates.</p>
</section>

<section id='limitations-validity'>
<div class='section-title'><h2>Limitations and Validity</h2><small>Scope and caveats for interpretation</small></div>
<ul>
<li>Synthetic sessions and labels; this is not production-ground-truth evidence.</li>
<li>Fixed population and deterministic replays; no p-values/generalization inference.</li>
<li>Classification ambiguity is high; use-case coverage is provisional.</li>
<li>ARM2 includes model-assisted IDW assumptions and fallback behavior.</li>
<li>ARM5 weighting can be unstable at low caps.</li>
<li>15k planning conversion is conservative and approximate.</li>
<li>Sentinel drilldown covers only five largest dense agents.</li>
</ul>
</section>

<section id='recommendations-next-steps'>
<div class='section-title'><h2>Recommendations and Next Steps</h2><small>Actionable next work and update triggers</small></div>
<p><strong>Accuracy-first:</strong> ARM1 at caps 64/128/256.</p>
<p><strong>Representation-first:</strong> ARM4/ARM5, with explicit provisional use-case caveat.</p>
<p><strong>Balanced objective:</strong> ARM5 at 512; ARM5 vs ARM1 at 1024 is a near-tie due to mean-median disagreement.</p>
<ol>
<li>Run a production shadow evaluation with matched cap/seed structure.</li>
<li>Perform a labeled Maven classification audit.</li>
<li>Calibrate nominal token conversion empirically from production traces.</li>
<li>Increase trials only if rank ordering remains ambiguous.</li>
</ol>
<p><strong>Evidence that would change this decision:</strong> shadow-run rank reversal, improved classification confidence profile, or major token calibration shift.</p>
</section>

<section id='interactive-explorer' class='screen-only'>
<div class='section-title'><h2>Interactive Explorer</h2><small>Cap selector, metric tabs, method toggles, aggregate/trial mode</small></div>
<div class='controls'>
<label for='cap-select'>Cap</label>
<select id='cap-select'>{cap_options}</select>
<div role='tablist' aria-label='Metric tabs'>
<button type='button' class='metric-tab active' data-metric='absolute_aggregate_mae'>MAE</button>
<button type='button' class='metric-tab' data-metric='concept_coverage'>Concept Coverage</button>
<button type='button' class='metric-tab' data-metric='use_case_coverage'>Use-Case Coverage</button>
<button type='button' class='metric-tab' data-metric='agent_coverage'>Agent Coverage</button>
</div>
<div class='toggle-group' id='method-toggle-group'>
{''.join(f"<label class='toggle'><input class='method-toggle' type='checkbox' data-method='{escape(m)}' checked /><span>{escape(_method_label(m))}</span></label>" for m in methods)}
</div>
<div>
<button type='button' class='mode-tab active' data-mode='aggregate'>Aggregate</button>
<button type='button' class='mode-tab' data-mode='trial'>Trial</button>
</div>
</div>
<div class='chart-shell screen-only'>
<div id='interactive-chart' aria-live='polite'></div>
</div>
<div class='chart-shell screen-only' style='margin-top:10px;'>
<div id='interactive-table' aria-live='polite'></div>
</div>
</section>

<section id='appendix'>
<div class='section-title'><h2>Appendix</h2><small>Detailed taxonomy, trial diagnostics, and methodology references</small></div>
<div class='three-col'>
<div><h3>Top Use Cases (GUID)</h3><ul>{class_top_use_cases or "<li>None</li>"}</ul><p>Total unique: {class_summary['total_unique_use_cases']}</p><p>Explicitly unclassified: {class_summary['undetermined_count']}</p><p>Below-0.30 fallback: {low0_n}</p></div>
<div><h3>Top Sub-subcategory</h3><ul>{class_top_sub_sub or "<li>None</li>"}</ul><p>Total unique: {class_summary['total_unique_sub_subcategory']}</p></div>
<div><h3>Confidence / Status</h3><h4>Confidence Levels</h4><ul>{conf_levels or "<li>None</li>"}</ul><h4>Status</h4><ul>{statuses or "<li>None</li>"}</ul></div>
</div>
<div class='screen-only'>
{idw_table}
{membership_table}
{taxonomy_html}
{methodology_html}
</div>
<p class='print-only'>Detailed IDW, membership, taxonomy, and methodology material is printed in the corresponding sections above; it is not repeated here.</p>
</section>

<script id='artifact-json' type='application/json'>{payload_text}</script>
<script>
(function () {{
  const blobEl = document.getElementById('artifact-json');
  if (!blobEl) return;
  const artifactData = JSON.parse(blobEl.textContent || '{{}}');

  const state = {{
    cap: (artifactData.caps && artifactData.caps.length ? String(artifactData.caps[0]) : ''),
    metric: 'absolute_aggregate_mae',
    mode: 'aggregate',
    visibleMethods: new Set((artifactData.methods || [])),
    selectedAgent: '',
    topCap: ''
  }};

  const capSelect = document.getElementById('cap-select');
  const metricTabs = Array.from(document.querySelectorAll('.metric-tab'));
  const modeTabs = Array.from(document.querySelectorAll('.mode-tab'));
  const methodToggles = Array.from(document.querySelectorAll('.method-toggle'));
  const chartEl = document.getElementById('interactive-chart');
  const tableEl = document.getElementById('interactive-table');
  const topAgentSelect = document.getElementById('top-agent-select');
  const topCapSelect = document.getElementById('top-cap-select');
  const topFiveTable = document.getElementById('top-five-table');

  const labels = artifactData.labels || {{}};
  const colors = artifactData.colors || {{}};

  function metricLabel(metric) {{
    if (metric === 'absolute_aggregate_mae') return 'MAE';
    if (metric === 'concept_coverage') return 'Concept Coverage';
    if (metric === 'use_case_coverage') return 'Use-Case Coverage';
    if (metric === 'agent_coverage') return 'Agent Coverage';
    return metric;
  }}

  function valueForRow(row, metric) {{
    if (!row) return 0;
    const metrics = row.metrics || {{}};
    const m = metrics[metric];
    if (m && typeof m === 'object' && typeof m.mean === 'number') return m.mean;
    if (typeof m === 'number') return m;
    return 0;
  }}

  function trialValue(row, metric) {{
    const metrics = row.metrics || {{}};
    const v = metrics[metric];
    return (typeof v === 'number') ? v : 0;
  }}

  function fmtPct(value) {{
    return (100 * (Number(value) || 0)).toFixed(2) + '%';
  }}

  function fmtNum(value, digits) {{
    return (Number(value) || 0).toFixed(digits || 4);
  }}

    function toNullableNumber(value) {{
        if (value === null || value === undefined || value === '') return null;
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
    }}

    function fmtPctNullable(value, digits) {{
        const n = toNullableNumber(value);
        if (n === null) return 'N/A';
        return (100 * n).toFixed(digits || 2) + '%';
    }}

    function fmtNumNullable(value, digits) {{
        const n = toNullableNumber(value);
        if (n === null) return 'N/A';
        return n.toFixed(digits || 4);
    }}

    function fmtRange(stat, isPct, digits) {{
        const mean = toNullableNumber((stat || {{}}).mean);
        const min = toNullableNumber((stat || {{}}).min);
        const max = toNullableNumber((stat || {{}}).max);
        if (mean === null || min === null || max === null) return 'N/A';
        if (isPct) {{
            return `${{fmtPctNullable(mean, digits || 2)}} [${{fmtPctNullable(min, digits || 2)}}..${{fmtPctNullable(max, digits || 2)}}]`;
        }}
        return `${{fmtNumNullable(mean, digits || 4)}} [${{fmtNumNullable(min, digits || 4)}}..${{fmtNumNullable(max, digits || 4)}}]`;
    }}

  function filteredRows() {{
    const cap = Number(state.cap || 0);
    if (state.mode === 'aggregate') {{
      return (artifactData.aggregate_rows || []).filter(r => Number(r.cap || 0) === cap && state.visibleMethods.has(String(r.method_id || '')));
    }}
    return (artifactData.runs || []).filter(r => Number(r.cap || 0) === cap && state.visibleMethods.has(String(r.method_id || '')));
  }}

  function renderBars() {{
    const rows = filteredRows();
    if (!rows.length) {{
      chartEl.innerHTML = '<p class="empty-state">No rows for current selection.</p>';
      return;
    }}
    const width = 980;
    const height = 290;
    const left = 58;
    const right = 20;
    const top = 20;
    const bottom = 52;
    const plotW = width - left - right;
    const plotH = height - top - bottom;

    const items = rows.map(r => {{
      const method = String(r.method_id || 'unknown');
      const v = state.mode === 'aggregate' ? valueForRow(r, state.metric) : trialValue(r, state.metric);
      return {{ method: method, label: labels[method] || method, value: Number(v) || 0, seed: r.seed || 0, color: colors[method] || '#334455' }};
    }});

    const maxV = Math.max(1e-9, ...items.map(x => x.value));
    const step = plotW / Math.max(items.length, 1);
    const barW = Math.max(12, Math.min(72, step * 0.72));

    const bars = items.map((it, idx) => {{
      const h = (it.value / maxV) * plotH;
      const x = left + idx * step + (step - barW) / 2;
      const y = top + (plotH - h);
      const title = `${{it.label}} | cap=${{state.cap}} | mode=${{state.mode}} | ${{metricLabel(state.metric)}}=${{state.metric === 'absolute_aggregate_mae' ? fmtNum(it.value, 4) : fmtPct(it.value)}}${{state.mode === 'trial' ? ` | seed=${{it.seed}}` : ''}}`;
      return `
        <g role='listitem'>
          <title>${{title.replace(/&/g, '&amp;').replace(/</g, '&lt;')}}</title>
          <rect x='${{x.toFixed(2)}}' y='${{y.toFixed(2)}}' width='${{barW.toFixed(2)}}' height='${{h.toFixed(2)}}' fill='${{it.color}}' aria-label='${{title.replace(/'/g, '&apos;')}}'></rect>
          <text x='${{(x + barW / 2).toFixed(2)}}' y='${{(height - 30).toFixed(2)}}' text-anchor='middle' style='font-size:10px;fill:#4f5b67;'>${{it.label.replace(/ARM/g, 'A')}}</text>
          <text x='${{(x + barW / 2).toFixed(2)}}' y='${{(y - 4).toFixed(2)}}' text-anchor='middle' style='font-size:10px;fill:#2a3440;'>${{state.metric === 'absolute_aggregate_mae' ? fmtNum(it.value, 3) : fmtPct(it.value)}}</text>
        </g>`;
    }}).join('');

    chartEl.innerHTML = `
      <svg class='chart' role='img' aria-label='${{metricLabel(state.metric)}} by method at cap ${{state.cap}}' viewBox='0 0 ${{width}} ${{height}}'>
        <line x1='${{left}}' y1='${{top + plotH}}' x2='${{width - right}}' y2='${{top + plotH}}' stroke='#6f7f8f' stroke-width='1'></line>
        <line x1='${{left}}' y1='${{top}}' x2='${{left}}' y2='${{top + plotH}}' stroke='#6f7f8f' stroke-width='1'></line>
        ${{bars}}
      </svg>`;
  }}

  function renderTable() {{
    const rows = filteredRows();
    if (!rows.length) {{
      tableEl.innerHTML = '<p class="empty-state">No rows for current selection.</p>';
      return;
    }}
    const body = rows.map(r => {{
      const method = String(r.method_id || 'unknown');
      const metricVal = state.mode === 'aggregate' ? valueForRow(r, state.metric) : trialValue(r, state.metric);
      const nominal = Number(r.nominal_budget || 0);
      const actual = Number(r.actual_token_count || 0);
      const ratio = nominal > 0 ? (actual / nominal) : 0;
      return `<tr>
        <td>${{(labels[method] || method).replace(/</g, '&lt;')}}</td>
        <td>${{Number(r.cap || 0)}}</td>
        <td>${{state.mode === 'trial' ? Number(r.seed || 0) : '-'}}</td>
        <td>${{state.metric === 'absolute_aggregate_mae' ? fmtNum(metricVal, 4) : fmtPct(metricVal)}}</td>
        <td>${{fmtNum(nominal, 0)}}</td>
        <td>${{fmtNum(actual, 0)}}</td>
        <td>${{fmtPct(ratio)}}</td>
      </tr>`;
    }}).join('');
    tableEl.innerHTML = `<div class='table-wrap'><table class='compact-table'><thead><tr><th>Method</th><th>Cap</th><th>Trial</th><th>${{metricLabel(state.metric)}}</th><th>Nominal Budget</th><th>Actual Tokens</th><th>Actual/Nominal</th></tr></thead><tbody>${{body}}</tbody></table></div>`;
  }}

  function renderTopFive() {{
    const rows = (artifactData.top_five || []).filter(r => {{
      if (state.selectedAgent && String(r.agent_id || '') !== state.selectedAgent) return false;
      if (state.topCap && Number(r.cap || 0) !== Number(state.topCap)) return false;
      return true;
    }});
    if (!rows.length) {{
      topFiveTable.innerHTML = '<p class="empty-state">No top-five rows for selected filters.</p>';
      return;
    }}
    const body = rows.slice(0, 160).map(r => {{
      const method = String(r.method_id || 'unknown');
            const details = (r.trial_details || []).slice(0, 3).map(t => `seed=${{Number(t.seed || 0)}} sel=${{fmtPctNullable(t.selected_rate, 2)}} err=${{fmtNumNullable(t.absolute_error, 4)}}`).join(' | ');
      return `<tr>
        <td>${{(labels[method] || method).replace(/</g, '&lt;')}}</td>
        <td>${{Number(r.cap || 0)}}</td>
        <td>${{String(r.agent_id || '').replace(/</g, '&lt;')}}</td>
                <td>${{fmtRange((r.N || {{}}), false, 1)}}</td>
                <td>${{fmtRange((r.n || {{}}), false, 1)}}</td>
                <td>${{fmtRange((r.selected_rate || {{}}), true, 2)}}</td>
                <td>${{fmtRange((r.census_rate || {{}}), true, 2)}}</td>
                <td>${{fmtRange((r.absolute_error || {{}}), false, 4)}}</td>
                <td>${{fmtRange((r.concept_coverage || {{}}), true, 2)}}</td>
                <td>${{fmtRange((r.use_case_coverage || {{}}), true, 2)}}</td>
        <td>${{details.replace(/</g, '&lt;')}}</td>
      </tr>`;
    }}).join('');
    topFiveTable.innerHTML = `<div class='table-wrap'><table class='compact-table'><thead><tr><th>Method</th><th>Cap</th><th>Agent</th><th>N</th><th>n</th><th>Selected Rate</th><th>Census Rate</th><th>Abs Error</th><th>Concept Cov</th><th>Use-Case Cov</th><th>Trial Detail</th></tr></thead><tbody>${{body}}</tbody></table></div>`;
  }}

  function renderAll() {{
    renderBars();
    renderTable();
    renderTopFive();
  }}

  if (capSelect) {{
    capSelect.addEventListener('change', () => {{
      state.cap = capSelect.value;
      renderAll();
    }});
  }}

  metricTabs.forEach(btn => btn.addEventListener('click', () => {{
    state.metric = btn.getAttribute('data-metric') || 'absolute_aggregate_mae';
    metricTabs.forEach(b => b.classList.toggle('active', b === btn));
    renderAll();
  }}));

  modeTabs.forEach(btn => btn.addEventListener('click', () => {{
    state.mode = btn.getAttribute('data-mode') || 'aggregate';
    modeTabs.forEach(b => b.classList.toggle('active', b === btn));
    renderAll();
  }}));

  methodToggles.forEach(cb => cb.addEventListener('change', () => {{
    const m = cb.getAttribute('data-method') || '';
    if (!m) return;
    if (cb.checked) state.visibleMethods.add(m);
    else state.visibleMethods.delete(m);
    renderAll();
  }}));

  if (topAgentSelect) {{
    topAgentSelect.addEventListener('change', () => {{
      state.selectedAgent = topAgentSelect.value;
      renderTopFive();
    }});
  }}
  if (topCapSelect) {{
    topCapSelect.addEventListener('change', () => {{
      state.topCap = topCapSelect.value;
      renderTopFive();
    }});
  }}

  renderAll();
}})();
</script>
</div>
</div>
</div>
</body>
</html>
"""


def validate_v6_artifacts(artifacts: LoadedV6Artifacts) -> None:
    aggregate = artifacts.aggregate
    aggregate_version = str(aggregate.get("version") or "")
    if aggregate_version not in SUPPORTED_V6_BUNDLE_VERSIONS:
        raise ValueError(f"aggregate version must be one of {SUPPORTED_V6_BUNDLE_VERSIONS}")
    if not artifacts.runs:
        raise ValueError("runs.jsonl must contain at least one row")
    rows = aggregate.get("aggregate_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("aggregate methods/rows must be non-empty")
    manifest_version = artifacts.manifest.get("version") if isinstance(artifacts.manifest, dict) else None
    if manifest_version and str(manifest_version) not in SUPPORTED_V6_MANIFEST_VERSIONS:
        raise ValueError(f"manifest version must be one of {SUPPORTED_V6_MANIFEST_VERSIONS}")


def render_v6_html_report(inputs: V6ReportInputs) -> str:
    artifacts = load_v6_artifacts(inputs)
    validate_v6_artifacts(artifacts)
    return _build_html_document(inputs, artifacts)


def _detect_browser() -> str:
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


def _file_uri(path: Path) -> str:
    resolved = path.resolve()
    as_posix = resolved.as_posix()
    if ":" in as_posix[:3]:
        drive = as_posix[0].lower()
        rest = as_posix[2:]
        return f"file:///{drive}:{quote(rest)}"
    return f"file://{quote(as_posix)}"


def compose_pdf_command(browser_path: str | None, html_path: Path, pdf_path: Path) -> list[str]:
    browser = browser_path or _detect_browser()
    html_uri = _file_uri(Path(html_path))
    pdf_abs = Path(pdf_path).resolve()
    return [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_abs}",
        html_uri,
    ]


def validate_pdf_file(path: Path) -> bool:
    if not path.exists():
        raise FileNotFoundError(f"PDF was not generated: {path}")
    data = path.read_bytes()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"PDF does not begin with %PDF: {path}")
    if len(data) <= 1024:
        raise ValueError(f"PDF size is too small (<=1KB): {path}")
    return True


def write_v6_html_report(output_path: Path, inputs: V6ReportInputs, *, pdf: bool = False, browser_path: str | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_v6_html_report(inputs)
    output_path.write_text(html, encoding="utf-8")
    if pdf:
        pdf_path = output_path.with_suffix(".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        command = compose_pdf_command(browser_path, output_path, pdf_path)
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            fallback = [c for c in command if c != "--no-pdf-header-footer"]
            subprocess.run(fallback, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        validate_pdf_file(pdf_path)
    return output_path
