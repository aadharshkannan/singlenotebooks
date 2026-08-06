from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .v3_experiment import V3_OUTCOME_METHODS, V3_OUTCOME_VERSION
from .v4_idw import (
    IDWConfig,
    estimate_embedding_population,
    freeze_membership,
    leave_one_out_donor_diagnostics,
    validate_embedding_population,
)


V4_VERSION = "sampling-v4"
V4_OUTCOME_VERSION = "sampling-v4-outcome-v1"
V4_EMBEDDING_METHOD = "adaptive_embedding_fullsession_token"
V4_SELECTED_ONLY_MODE = "selected_only"
V4_MODEL_ASSISTED_MODE = "model_assisted_idw"


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_prob(value: Any, *, field_name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{field_name} must be finite")
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return out


def _safe_int(value: Any, *, field_name: str) -> int:
    try:
        out = int(value)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"{field_name} must be an integer") from exc
    return out


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Encountered non-finite float while building output")
        return value
    if isinstance(value, np.generic):
        cast = value.item()
        return _json_safe(cast)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _deterministic_cell_id(row: Mapping[str, Any]) -> str:
    key = {
        "method": str(row["method"]),
        "budget_tokens": int(row["budget_tokens"]),
        "legacy_tier_pct": int(row["legacy_tier_pct"]),
        "repetition": int(row["repetition"]),
        "order_hash": str(row["order_hash"]),
    }
    return f"v4-cell-{_sha256_text(_canonical_json(key))[:24]}"


def _to_expected_label_map(*, unit_ids: Sequence[str], labels_by_unit: Mapping[str, bool]) -> dict[str, float]:
    out: dict[str, float] = {}
    for uid in unit_ids:
        if uid not in labels_by_unit:
            raise ValueError(f"expected label missing for unit_id={uid}")
        out[uid] = 1.0 if bool(labels_by_unit[uid]) else 0.0
    return out


def _build_agent_and_vector_maps(
    *,
    eligible_ids: Sequence[str],
    trace_by_unit_id: Mapping[str, Any],
    embedding_vector_by_trace_id: Mapping[int, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    agent_id_by_unit: dict[str, str] = {}
    vector_by_unit: dict[str, Any] = {}
    for uid in eligible_ids:
        trace = trace_by_unit_id.get(uid)
        if trace is None:
            raise ValueError(f"trace missing for eligible unit_id={uid}")
        trace_id = int(trace.trace_id)
        vector = embedding_vector_by_trace_id.get(trace_id)
        if vector is None:
            raise ValueError(f"embedding vector missing for trace_id={trace_id} unit_id={uid}")
        agent_id_by_unit[uid] = str(trace.agent_id)
        vector_by_unit[uid] = vector
    return agent_id_by_unit, vector_by_unit


def _validate_and_normalize_v3_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    population_count: int,
    eligible_ids_set: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    allowed_methods = set(V3_OUTCOME_METHODS)

    for idx, raw_row in enumerate(rows):
        row = dict(raw_row)
        method = str(row.get("method") or "")
        if method not in allowed_methods:
            raise ValueError(f"runs[{idx}].method is not a supported V3 outcome arm: {method}")

        row["budget_tokens"] = _safe_int(row.get("budget_tokens"), field_name=f"runs[{idx}].budget_tokens")
        row["legacy_tier_pct"] = _safe_int(row.get("legacy_tier_pct"), field_name=f"runs[{idx}].legacy_tier_pct")
        row["repetition"] = _safe_int(row.get("repetition"), field_name=f"runs[{idx}].repetition")

        order_hash = str(row.get("order_hash") or "")
        if not order_hash:
            raise ValueError(f"runs[{idx}].order_hash must be non-empty")
        row["order_hash"] = order_hash

        selected_ids_raw = row.get("selected_ids")
        if not isinstance(selected_ids_raw, list):
            raise ValueError(f"runs[{idx}].selected_ids must be a list")
        selected_ids = [str(x) for x in selected_ids_raw]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(f"runs[{idx}].selected_ids contains duplicates")
        if not set(selected_ids).issubset(eligible_ids_set):
            raise ValueError(f"runs[{idx}].selected_ids must be a subset of dataset unit_ids")
        row["selected_ids"] = selected_ids

        selected_count = _safe_int(row.get("selected_count"), field_name=f"runs[{idx}].selected_count")
        if selected_count != len(selected_ids):
            raise ValueError(f"runs[{idx}].selected_count does not match selected_ids")
        if selected_count < 0 or selected_count > population_count:
            raise ValueError(f"runs[{idx}].selected_count out of bounds")
        row["selected_count"] = selected_count

        selected_pass_rate = _safe_prob(row.get("selected_pass_rate"), field_name=f"runs[{idx}].selected_pass_rate")
        census_pass_rate = _safe_prob(row.get("census_pass_rate"), field_name=f"runs[{idx}].census_pass_rate")
        absolute_error = _safe_prob(row.get("absolute_error"), field_name=f"runs[{idx}].absolute_error")

        row["selected_pass_rate"] = selected_pass_rate
        row["census_pass_rate"] = census_pass_rate
        row["absolute_error"] = absolute_error
        out.append(row)

    return out


def _aggregate_stats(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    vals = [float(r[field]) for r in rows]
    return {
        "mean": float(np.mean(vals)) if vals else 0.0,
        "empirical_low": float(min(vals)) if vals else 0.0,
        "empirical_high": float(max(vals)) if vals else 0.0,
    }


def _sum_int_field(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return int(sum(int(r[field]) for r in rows))


def _sum_dict_of_ints(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        payload = row.get(field)
        if not isinstance(payload, Mapping):
            continue
        for k, v in payload.items():
            out[str(k)] = int(out.get(str(k), 0) + int(v))
    return out


def _sum_dict_of_floats(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        payload = row.get(field)
        if not isinstance(payload, Mapping):
            continue
        for k, v in payload.items():
            out[str(k)] = float(out.get(str(k), 0.0) + float(v))
    return out


def _distance_error_bins(
    *,
    estimates: Any,
    expected_labels: Mapping[str, float],
    bin_count: int = 10,
) -> list[dict[str, Any]]:
    eligible_rows = [
        row
        for row in estimates.rows
        if row.unit_id not in estimates.judged_unit_ids and row.nearest_distance is not None
    ]
    edges = np.linspace(0.0, 1.0, int(bin_count) + 1)
    bins: list[dict[str, Any]] = []
    for index in range(int(bin_count)):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        rows = [
            row
            for row in eligible_rows
            if float(row.nearest_distance) >= lower
            and (float(row.nearest_distance) < upper or (index == bin_count - 1 and float(row.nearest_distance) <= upper))
        ]
        errors = [abs(float(row.value) - float(expected_labels[row.unit_id])) for row in rows]
        bins.append(
            {
                "bin_index": int(index),
                "lower": lower,
                "upper": upper,
                "count": int(len(rows)),
                "avg_distance": float(np.mean([float(row.nearest_distance) for row in rows])) if rows else None,
                "mae": float(np.mean(errors)) if errors else None,
            }
        )
    return bins


def _per_agent_validation_summary(
    *,
    estimates: Any,
    expected_labels: Mapping[str, float],
) -> list[dict[str, Any]]:
    by_agent: dict[str, list[Any]] = {}
    for row in estimates.rows:
        agent = str(estimates.agent_id_by_unit[row.unit_id])
        by_agent.setdefault(agent, []).append(row)

    out: list[dict[str, Any]] = []
    for agent, rows in sorted(by_agent.items()):
        predictions = [float(row.value) for row in rows]
        labels = [float(expected_labels[row.unit_id]) for row in rows]
        observed_count = sum(1 for row in rows if row.unit_id in estimates.judged_unit_ids)
        estimated_rate = float(np.mean(predictions))
        census_rate = float(np.mean(labels))
        out.append(
            {
                "agent_id": agent,
                "population_count": int(len(rows)),
                "observed_count": int(observed_count),
                "imputed_count": int(len(rows) - observed_count),
                "estimated_pass_rate": estimated_rate,
                "census_pass_rate": census_rate,
                "absolute_error": float(abs(estimated_rate - census_rate)),
            }
        )
    return out


def _build_aggregate(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in runs:
        key = (str(row["method"]), int(row["budget_tokens"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (method, budget_tokens), bucket in sorted(grouped.items(), key=lambda it: (it[0][1], it[0][0])):
        legacy_tiers = sorted({int(row["legacy_tier_pct"]) for row in bucket})
        aggregate_row: dict[str, Any] = {
            "method": method,
            "budget_tokens": int(budget_tokens),
            "legacy_tier_pct_provenance": legacy_tiers,
            "replays": int(len(bucket)),
            "selected_only_mae": _aggregate_stats(bucket, "selected_only_absolute_error"),
        }

        if method == V4_EMBEDDING_METHOD:
            aggregate_row["idw_absolute_error"] = _aggregate_stats(bucket, "idw_absolute_error")
            aggregate_row["idw_delta_vs_selected_only"] = _aggregate_stats(bucket, "idw_delta_vs_selected_only")
            aggregate_row["idw_unjudged_only_mae"] = _aggregate_stats(bucket, "idw_unjudged_only_mae")
            aggregate_row["idw_unjudged_only_brier"] = _aggregate_stats(bucket, "idw_unjudged_only_brier")
            aggregate_row["idw_expected_calibration_error"] = _aggregate_stats(bucket, "idw_expected_calibration_error")
            aggregate_row["model_assisted_counts_sum"] = {
                "population_count": _sum_int_field(bucket, "idw_population_count"),
                "observed_count": _sum_int_field(bucket, "idw_observed_count"),
                "imputed_count": _sum_int_field(bucket, "idw_imputed_count"),
                "zero_donor_agent_count": _sum_int_field(bucket, "idw_zero_donor_agent_count"),
                "prior_count": _sum_int_field(bucket, "idw_prior_count"),
                "loo_judged_count": _sum_int_field(bucket, "idw_loo_judged_count"),
            }
            aggregate_row["model_assisted_provenance_counts_sum"] = _sum_dict_of_ints(bucket, "idw_provenance_counts")
            aggregate_row["model_assisted_provenance_population_weighted_rates_sum"] = _sum_dict_of_floats(
                bucket,
                "idw_provenance_population_weighted_rates",
            )

        out.append(aggregate_row)

    return out


def augment_v3_outcome_with_idw(
    data: Any,
    runtime: Any,
    v3_outcome: Mapping[str, Any],
    idw_config: IDWConfig = IDWConfig(),
) -> dict[str, Any]:
    version = str(v3_outcome.get("version") or "")
    if version != V3_OUTCOME_VERSION:
        raise ValueError(f"v3_outcome.version must be {V3_OUTCOME_VERSION}")

    dataset_unit_ids = [str(uid) for uid in data.unit_ids]
    population_count = int(v3_outcome.get("population_count") or 0)
    if population_count != len(dataset_unit_ids):
        raise ValueError("v3_outcome.population_count must match dataset size")

    runs_raw = v3_outcome.get("runs")
    if not isinstance(runs_raw, list):
        raise ValueError("v3_outcome.runs must be a list")

    runs = _validate_and_normalize_v3_rows(
        rows=runs_raw,
        population_count=population_count,
        eligible_ids_set=set(dataset_unit_ids),
    )

    agent_id_by_unit, vector_by_unit = _build_agent_and_vector_maps(
        eligible_ids=dataset_unit_ids,
        trace_by_unit_id=data.trace_by_unit_id,
        embedding_vector_by_trace_id=runtime.embedding_vector_by_trace_id,
    )

    augmented_runs: list[dict[str, Any]] = []
    for run_row in runs:
        out_row = dict(run_row)

        selected_only_pass_rate = float(run_row["selected_pass_rate"])
        selected_only_absolute_error = float(run_row["absolute_error"])
        out_row["selected_only_pass_rate"] = selected_only_pass_rate
        out_row["selected_only_absolute_error"] = selected_only_absolute_error

        if str(run_row["method"]) != V4_EMBEDDING_METHOD:
            out_row["estimation_mode"] = V4_SELECTED_ONLY_MODE
            out_row["model_assisted"] = None
            augmented_runs.append(_json_safe(out_row))
            continue

        cell_id = _deterministic_cell_id(run_row)
        membership = freeze_membership(
            cell_id=cell_id,
            eligible_ids=dataset_unit_ids,
            selected_ids=run_row["selected_ids"],
        )

        # Materialize labels only after membership is frozen.
        expected_labels_all = _to_expected_label_map(unit_ids=membership.eligible_ids, labels_by_unit=data.labels_by_unit)
        judged_values_selected = {uid: expected_labels_all[uid] for uid in membership.selected_ids}

        estimates = estimate_embedding_population(
            membership=membership,
            agent_id_by_unit=agent_id_by_unit,
            vector_by_unit=vector_by_unit,
            judged_values_by_unit=judged_values_selected,
            config=idw_config,
        )
        validation = validate_embedding_population(estimates, expected_labels_all)
        loo = leave_one_out_donor_diagnostics(
            membership=membership,
            agent_id_by_unit=agent_id_by_unit,
            vector_by_unit=vector_by_unit,
            judged_values_by_unit=judged_values_selected,
            config=idw_config,
        )

        idw_abs_error = float(validation.absolute_aggregate_rate_error)
        idw_delta = float(idw_abs_error - selected_only_absolute_error)

        model_assisted = {
            "estimation_mode": V4_MODEL_ASSISTED_MODE,
            "idw_config": asdict(idw_config),
            "membership": {
                "cell_id": cell_id,
                "membership_hash": str(membership.membership_hash),
                "population_hash": str(membership.population_hash),
            },
            "rates": {
                "estimated_pass_rate": float(estimates.aggregate.estimated_pass_rate),
                "absolute_aggregate_rate_error": idw_abs_error,
                "delta_vs_selected_only_absolute_error": idw_delta,
                "provenance_population_weighted_rates": {
                    k: float(v) for k, v in estimates.aggregate.provenance_population_weighted_rates.items()
                },
            },
            "counts": {
                "population_count": int(estimates.aggregate.population_count),
                "observed_count": int(estimates.aggregate.observed_count),
                "imputed_count": int(estimates.aggregate.imputed_count),
                "zero_donor_agent_count": int(estimates.aggregate.zero_donor_agent_count),
                "prior_count": int(estimates.aggregate.prior_count),
                "provenance_counts": {k: int(v) for k, v in estimates.aggregate.provenance_counts.items()},
            },
            "metrics": {
                "per_unit_mae": float(validation.per_unit_mae),
                "brier_score": float(validation.brier_score),
                "expected_calibration_error": float(validation.expected_calibration_error),
                "macro_per_agent_mae": float(validation.macro_per_agent_mae),
                "unjudged_only_mae": float(validation.unjudged_only_mae) if validation.unjudged_only_mae is not None else 0.0,
                "unjudged_only_brier": (
                    float(validation.unjudged_only_brier) if validation.unjudged_only_brier is not None else 0.0
                ),
                "calibration_bins": [dict(bin_row) for bin_row in validation.calibration_bins],
                "nearest_distance_error_bins": _distance_error_bins(
                    estimates=estimates,
                    expected_labels=expected_labels_all,
                ),
                "per_agent": _per_agent_validation_summary(
                    estimates=estimates,
                    expected_labels=expected_labels_all,
                ),
                "leave_one_out": {
                    "judged_count": int(len(judged_values_selected)),
                    "mae": float(loo.mae),
                    "brier_score": float(loo.brier_score),
                },
            },
        }

        out_row["estimation_mode"] = V4_MODEL_ASSISTED_MODE
        out_row["model_assisted"] = model_assisted

        # Add flattened metrics used for aggregate rollups while keeping row compatibility.
        out_row["idw_absolute_error"] = idw_abs_error
        out_row["idw_delta_vs_selected_only"] = idw_delta
        out_row["idw_unjudged_only_mae"] = float(model_assisted["metrics"]["unjudged_only_mae"])
        out_row["idw_unjudged_only_brier"] = float(model_assisted["metrics"]["unjudged_only_brier"])
        out_row["idw_expected_calibration_error"] = float(model_assisted["metrics"]["expected_calibration_error"])
        out_row["idw_population_count"] = int(model_assisted["counts"]["population_count"])
        out_row["idw_observed_count"] = int(model_assisted["counts"]["observed_count"])
        out_row["idw_imputed_count"] = int(model_assisted["counts"]["imputed_count"])
        out_row["idw_zero_donor_agent_count"] = int(model_assisted["counts"]["zero_donor_agent_count"])
        out_row["idw_prior_count"] = int(model_assisted["counts"]["prior_count"])
        out_row["idw_provenance_counts"] = dict(model_assisted["counts"]["provenance_counts"])
        out_row["idw_provenance_population_weighted_rates"] = dict(
            model_assisted["rates"]["provenance_population_weighted_rates"]
        )
        out_row["idw_loo_judged_count"] = int(model_assisted["metrics"]["leave_one_out"]["judged_count"])

        augmented_runs.append(_json_safe(out_row))

    return {
        "version": V4_OUTCOME_VERSION,
        "derived_from_version": V3_OUTCOME_VERSION,
        "runtime_version": str(v3_outcome.get("runtime_version") or ""),
        "population_count": int(population_count),
        "eligible_token_mass": int(v3_outcome.get("eligible_token_mass") or 0),
        "pairing": _json_safe(v3_outcome.get("pairing") or {}),
        "idw_config": _json_safe(asdict(idw_config)),
        "notes": [
            "V4 augmentation reuses V3 selected memberships and does not rerun selection.",
            "For non-embedding methods, metrics remain selected-only.",
            "IDW results are model-assisted estimates and should not be interpreted as unbiased guarantees.",
        ],
        "aggregate": _json_safe(_build_aggregate(augmented_runs)),
        "runs": augmented_runs,
    }
