from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import acos, pi
from typing import Mapping, Sequence

import numpy as np


PROVENANCE_OBSERVED = "observed"
PROVENANCE_IDW = "idw"
PROVENANCE_EXACT_MATCH = "exact_match"
PROVENANCE_AGENT_MEAN = "agent_mean"
PROVENANCE_GLOBAL_MEAN = "global_mean"
PROVENANCE_PRIOR = "prior"


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_sorted_unique(ids: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    out = [str(x) for x in ids]
    if len(set(out)) != len(out):
        raise ValueError(f"{field_name} contains duplicates")
    return tuple(sorted(out))


def _population_hash(eligible_ids: Sequence[str]) -> str:
    return _sha256_text(_canonical_json({"eligible_ids": list(eligible_ids)}))


def _membership_hash(cell_id: str, eligible_ids: Sequence[str], selected_ids: Sequence[str]) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "cell_id": str(cell_id),
                "eligible_ids": list(eligible_ids),
                "selected_ids": list(selected_ids),
            }
        )
    )


def _normalize_vector(raw: object) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 1:
        return None
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return None
    return arr / norm


def _validate_probability(value: float, *, field_name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{field_name} must be finite")
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return out


@dataclass(frozen=True)
class FrozenMembership:
    cell_id: str
    membership_hash: str
    population_hash: str
    eligible_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("cell_id must be non-empty")
        if len(set(self.eligible_ids)) != len(self.eligible_ids):
            raise ValueError("eligible_ids contains duplicates")
        if len(set(self.selected_ids)) != len(self.selected_ids):
            raise ValueError("selected_ids contains duplicates")
        if not set(self.selected_ids).issubset(set(self.eligible_ids)):
            raise ValueError("selected_ids must be a subset of eligible_ids")

        expected_population_hash = _population_hash(self.eligible_ids)
        if self.population_hash != expected_population_hash:
            raise ValueError("population_hash does not match eligible_ids")

        expected_membership_hash = _membership_hash(self.cell_id, self.eligible_ids, self.selected_ids)
        if self.membership_hash != expected_membership_hash:
            raise ValueError("membership_hash does not match frozen membership")


@dataclass(frozen=True)
class IDWConfig:
    k: int = 8
    power: float = 2.0
    eps: float = 1e-6
    exact_cosine_eps: float = 1e-8
    prior: float = 0.5

    def __post_init__(self) -> None:
        if int(self.k) <= 0:
            raise ValueError("k must be > 0")
        if float(self.power) <= 0.0:
            raise ValueError("power must be > 0")
        if float(self.eps) <= 0.0:
            raise ValueError("eps must be > 0")
        if float(self.exact_cosine_eps) < 0.0:
            raise ValueError("exact_cosine_eps must be >= 0")
        _validate_probability(self.prior, field_name="prior")


@dataclass(frozen=True)
class UnitEstimate:
    unit_id: str
    value: float
    provenance: str
    donor_ids: tuple[str, ...]
    distances: tuple[float, ...]
    normalized_weights: tuple[float, ...]
    effective_donor_count: float
    nearest_distance: float | None
    exact_match: bool


@dataclass(frozen=True)
class PopulationEstimate:
    estimated_pass_rate: float
    population_count: int
    observed_count: int
    imputed_count: int
    provenance_counts: Mapping[str, int]
    provenance_population_weighted_rates: Mapping[str, float]
    zero_donor_agent_count: int
    prior_count: int
    membership_hash: str
    population_hash: str
    config: IDWConfig


@dataclass(frozen=True)
class EmbeddingPopulationEstimates:
    membership: FrozenMembership
    config: IDWConfig
    rows: tuple[UnitEstimate, ...]
    aggregate: PopulationEstimate
    agent_id_by_unit: Mapping[str, str]
    judged_unit_ids: frozenset[str]


@dataclass(frozen=True)
class EmbeddingPopulationValidation:
    census_pass_rate: float
    absolute_aggregate_rate_error: float
    per_unit_mae: float
    brier_score: float
    macro_per_agent_mae: float
    judged_only_pass_rate: float | None
    judged_only_absolute_rate_error: float | None
    unjudged_only_mae: float | None
    unjudged_only_brier: float | None
    calibration_bins: tuple[Mapping[str, float], ...]
    expected_calibration_error: float


@dataclass(frozen=True)
class LeaveOneOutDiagnostics:
    per_unit_predictions: Mapping[str, UnitEstimate]
    mae: float
    brier_score: float


def freeze_membership(
    *,
    cell_id: str,
    eligible_ids: Sequence[str],
    selected_ids: Sequence[str],
) -> FrozenMembership:
    eligible = _stable_sorted_unique(eligible_ids, field_name="eligible_ids")
    selected = _stable_sorted_unique(selected_ids, field_name="selected_ids")
    if not set(selected).issubset(set(eligible)):
        raise ValueError("selected_ids must be a subset of eligible_ids")
    pop_hash = _population_hash(eligible)
    mem_hash = _membership_hash(str(cell_id), eligible, selected)
    return FrozenMembership(
        cell_id=str(cell_id),
        membership_hash=mem_hash,
        population_hash=pop_hash,
        eligible_ids=eligible,
        selected_ids=selected,
    )


def _agent_means(judged_values_by_unit: Mapping[str, float], agent_id_by_unit: Mapping[str, str]) -> dict[str, float]:
    by_agent: dict[str, list[float]] = {}
    for uid, value in judged_values_by_unit.items():
        agent = str(agent_id_by_unit[uid])
        by_agent.setdefault(agent, []).append(float(value))
    return {agent: float(np.mean(values)) for agent, values in by_agent.items()}


def _angular_distance(cosine: np.ndarray) -> np.ndarray:
    clipped = np.clip(cosine, -1.0, 1.0)
    return np.arccos(clipped) / pi


def _estimate_for_target(
    *,
    target_id: str,
    target_vector: np.ndarray | None,
    donor_ids: np.ndarray,
    donor_vectors: np.ndarray,
    donor_values: np.ndarray,
    agent_mean: float | None,
    global_mean: float | None,
    config: IDWConfig,
    exclude_donor_id: str | None = None,
) -> UnitEstimate:
    if target_vector is None:
        if agent_mean is not None:
            return UnitEstimate(
                unit_id=target_id,
                value=float(agent_mean),
                provenance=PROVENANCE_AGENT_MEAN,
                donor_ids=(),
                distances=(),
                normalized_weights=(),
                effective_donor_count=0.0,
                nearest_distance=None,
                exact_match=False,
            )
        if global_mean is not None:
            return UnitEstimate(
                unit_id=target_id,
                value=float(global_mean),
                provenance=PROVENANCE_GLOBAL_MEAN,
                donor_ids=(),
                distances=(),
                normalized_weights=(),
                effective_donor_count=0.0,
                nearest_distance=None,
                exact_match=False,
            )
        return UnitEstimate(
            unit_id=target_id,
            value=float(config.prior),
            provenance=PROVENANCE_PRIOR,
            donor_ids=(),
            distances=(),
            normalized_weights=(),
            effective_donor_count=0.0,
            nearest_distance=None,
            exact_match=False,
        )

    if donor_ids.size == 0:
        if agent_mean is not None:
            return UnitEstimate(
                unit_id=target_id,
                value=float(agent_mean),
                provenance=PROVENANCE_AGENT_MEAN,
                donor_ids=(),
                distances=(),
                normalized_weights=(),
                effective_donor_count=0.0,
                nearest_distance=None,
                exact_match=False,
            )
        if global_mean is not None:
            return UnitEstimate(
                unit_id=target_id,
                value=float(global_mean),
                provenance=PROVENANCE_GLOBAL_MEAN,
                donor_ids=(),
                distances=(),
                normalized_weights=(),
                effective_donor_count=0.0,
                nearest_distance=None,
                exact_match=False,
            )
        return UnitEstimate(
            unit_id=target_id,
            value=float(config.prior),
            provenance=PROVENANCE_PRIOR,
            donor_ids=(),
            distances=(),
            normalized_weights=(),
            effective_donor_count=0.0,
            nearest_distance=None,
            exact_match=False,
        )

    cosine = donor_vectors @ target_vector
    if exclude_donor_id is not None:
        keep = donor_ids != str(exclude_donor_id)
        donor_ids = donor_ids[keep]
        donor_values = donor_values[keep]
        cosine = cosine[keep]
        if donor_ids.size == 0:
            return _estimate_for_target(
                target_id=target_id,
                target_vector=None,
                donor_ids=donor_ids,
                donor_vectors=np.zeros((0, 0), dtype=np.float64),
                donor_values=donor_values,
                agent_mean=agent_mean,
                global_mean=global_mean,
                config=config,
            )
    distances = _angular_distance(cosine)

    order = np.lexsort((donor_ids, distances))
    donor_ids_sorted = donor_ids[order]
    donor_values_sorted = donor_values[order]
    cosine_sorted = cosine[order]
    distances_sorted = distances[order]

    exact_mask = (1.0 - cosine_sorted) <= float(config.exact_cosine_eps)
    if np.any(exact_mask):
        exact_ids = donor_ids_sorted[exact_mask]
        exact_values = donor_values_sorted[exact_mask]
        value = float(np.mean(exact_values))
        nearest = float(np.min(distances_sorted[exact_mask]))
        return UnitEstimate(
            unit_id=target_id,
            value=float(np.clip(value, 0.0, 1.0)),
            provenance=PROVENANCE_EXACT_MATCH,
            donor_ids=tuple(exact_ids.tolist()),
            distances=tuple(float(x) for x in distances_sorted[exact_mask].tolist()),
            normalized_weights=tuple(float(1.0 / len(exact_values)) for _ in range(len(exact_values))),
            effective_donor_count=float(len(exact_values)),
            nearest_distance=nearest,
            exact_match=True,
        )

    k = min(int(config.k), donor_ids_sorted.size)
    d_use = distances_sorted[:k]
    ids_use = donor_ids_sorted[:k]
    vals_use = donor_values_sorted[:k]

    weights = 1.0 / np.power(d_use + float(config.eps), float(config.power))
    norm = float(np.sum(weights))
    if norm <= 0.0 or not np.isfinite(norm):
        if agent_mean is not None:
            return UnitEstimate(
                unit_id=target_id,
                value=float(agent_mean),
                provenance=PROVENANCE_AGENT_MEAN,
                donor_ids=(),
                distances=(),
                normalized_weights=(),
                effective_donor_count=0.0,
                nearest_distance=None,
                exact_match=False,
            )
        if global_mean is not None:
            return UnitEstimate(
                unit_id=target_id,
                value=float(global_mean),
                provenance=PROVENANCE_GLOBAL_MEAN,
                donor_ids=(),
                distances=(),
                normalized_weights=(),
                effective_donor_count=0.0,
                nearest_distance=None,
                exact_match=False,
            )
        return UnitEstimate(
            unit_id=target_id,
            value=float(config.prior),
            provenance=PROVENANCE_PRIOR,
            donor_ids=(),
            distances=(),
            normalized_weights=(),
            effective_donor_count=0.0,
            nearest_distance=None,
            exact_match=False,
        )

    weights = weights / norm
    pred = float(np.dot(weights, vals_use))
    eff = float(1.0 / np.sum(np.square(weights)))
    nearest = float(np.min(d_use))
    return UnitEstimate(
        unit_id=target_id,
        value=float(np.clip(pred, 0.0, 1.0)),
        provenance=PROVENANCE_IDW,
        donor_ids=tuple(ids_use.tolist()),
        distances=tuple(float(x) for x in d_use.tolist()),
        normalized_weights=tuple(float(x) for x in weights.tolist()),
        effective_donor_count=eff,
        nearest_distance=nearest,
        exact_match=False,
    )


def estimate_embedding_population(
    *,
    membership: FrozenMembership,
    agent_id_by_unit: Mapping[str, str],
    vector_by_unit: Mapping[str, object],
    judged_values_by_unit: Mapping[str, float],
    config: IDWConfig = IDWConfig(),
) -> EmbeddingPopulationEstimates:
    eligible_ids = membership.eligible_ids
    selected_set = set(membership.selected_ids)

    for uid in judged_values_by_unit:
        if uid not in selected_set:
            raise ValueError("judged_values_by_unit contains unit outside selected_ids")

    for uid in eligible_ids:
        if uid not in agent_id_by_unit:
            raise ValueError(f"missing agent_id for eligible unit: {uid}")

    judged_clean: dict[str, float] = {}
    for uid, raw_value in judged_values_by_unit.items():
        judged_clean[uid] = _validate_probability(raw_value, field_name=f"judged_values_by_unit[{uid}]")

    vector_norm_by_unit: dict[str, np.ndarray | None] = {}
    for uid in eligible_ids:
        vector_norm_by_unit[uid] = _normalize_vector(vector_by_unit.get(uid))

    global_mean: float | None
    if judged_clean:
        global_mean = float(np.mean(np.asarray(list(judged_clean.values()), dtype=np.float64)))
    else:
        global_mean = None

    per_agent_mean = _agent_means(judged_clean, agent_id_by_unit)

    donor_units_by_agent: dict[str, list[str]] = {}
    for uid in judged_clean:
        donor_units_by_agent.setdefault(str(agent_id_by_unit[uid]), []).append(uid)

    valid_donor_ids_by_agent: dict[str, np.ndarray] = {}
    valid_donor_values_by_agent: dict[str, np.ndarray] = {}
    valid_donor_vectors_by_agent: dict[str, np.ndarray] = {}

    for agent, donor_ids in donor_units_by_agent.items():
        valid_ids: list[str] = []
        valid_values: list[float] = []
        valid_vectors: list[np.ndarray] = []
        for uid in donor_ids:
            vec = vector_norm_by_unit.get(uid)
            if vec is None:
                continue
            valid_ids.append(uid)
            valid_values.append(judged_clean[uid])
            valid_vectors.append(vec)

        if valid_ids:
            ids_arr = np.asarray(valid_ids, dtype=object)
            vals_arr = np.asarray(valid_values, dtype=np.float64)
            vecs_arr = np.stack(valid_vectors, axis=0)
        else:
            ids_arr = np.asarray([], dtype=object)
            vals_arr = np.asarray([], dtype=np.float64)
            vecs_arr = np.zeros((0, 0), dtype=np.float64)

        valid_donor_ids_by_agent[agent] = ids_arr
        valid_donor_values_by_agent[agent] = vals_arr
        valid_donor_vectors_by_agent[agent] = vecs_arr

    rows: list[UnitEstimate] = []
    provenance_counts: dict[str, int] = {}
    provenance_weighted_rates: dict[str, float] = {}

    for uid in eligible_ids:
        if uid in judged_clean:
            row = UnitEstimate(
                unit_id=uid,
                value=float(judged_clean[uid]),
                provenance=PROVENANCE_OBSERVED,
                donor_ids=(),
                distances=(),
                normalized_weights=(),
                effective_donor_count=0.0,
                nearest_distance=None,
                exact_match=False,
            )
        else:
            agent = str(agent_id_by_unit[uid])
            donor_ids = valid_donor_ids_by_agent.get(agent)
            donor_values = valid_donor_values_by_agent.get(agent)
            donor_vectors = valid_donor_vectors_by_agent.get(agent)
            if donor_ids is None:
                donor_ids = np.asarray([], dtype=object)
                donor_values = np.asarray([], dtype=np.float64)
                donor_vectors = np.zeros((0, 0), dtype=np.float64)
            row = _estimate_for_target(
                target_id=uid,
                target_vector=vector_norm_by_unit.get(uid),
                donor_ids=donor_ids,
                donor_vectors=donor_vectors,
                donor_values=donor_values,
                agent_mean=per_agent_mean.get(agent),
                global_mean=global_mean,
                config=config,
            )

        row_value = _validate_probability(row.value, field_name=f"estimate[{uid}]")
        if not np.isfinite(row_value):
            raise ValueError(f"estimate[{uid}] is non-finite")
        rows.append(row)

        provenance_counts[row.provenance] = provenance_counts.get(row.provenance, 0) + 1
        provenance_weighted_rates[row.provenance] = provenance_weighted_rates.get(row.provenance, 0.0) + row_value

    population_count = len(rows)
    if population_count == 0:
        raise ValueError("eligible_ids is empty")

    for key in list(provenance_weighted_rates):
        provenance_weighted_rates[key] = float(provenance_weighted_rates[key] / population_count)

    values = np.asarray([row.value for row in rows], dtype=np.float64)
    estimated_pass_rate = float(np.mean(values))

    zero_donor_agents = {
        str(agent_id_by_unit[uid])
        for uid in eligible_ids
        if len(donor_units_by_agent.get(str(agent_id_by_unit[uid]), [])) == 0
    }

    observed_count = int(sum(1 for row in rows if row.provenance == PROVENANCE_OBSERVED))
    aggregate = PopulationEstimate(
        estimated_pass_rate=estimated_pass_rate,
        population_count=population_count,
        observed_count=observed_count,
        imputed_count=population_count - observed_count,
        provenance_counts=dict(provenance_counts),
        provenance_population_weighted_rates=dict(provenance_weighted_rates),
        zero_donor_agent_count=len(zero_donor_agents),
        prior_count=int(provenance_counts.get(PROVENANCE_PRIOR, 0)),
        membership_hash=membership.membership_hash,
        population_hash=membership.population_hash,
        config=config,
    )

    return EmbeddingPopulationEstimates(
        membership=membership,
        config=config,
        rows=tuple(rows),
        aggregate=aggregate,
        agent_id_by_unit={uid: str(agent_id_by_unit[uid]) for uid in eligible_ids},
        judged_unit_ids=frozenset(judged_clean.keys()),
    )


def validate_embedding_population(
    estimates: EmbeddingPopulationEstimates,
    expected_labels: Mapping[str, float],
    *,
    calibration_bin_count: int = 10,
) -> EmbeddingPopulationValidation:
    if calibration_bin_count <= 0:
        raise ValueError("calibration_bin_count must be > 0")

    rows = estimates.rows
    if not rows:
        raise ValueError("estimates.rows is empty")

    missing = [row.unit_id for row in rows if row.unit_id not in expected_labels]
    if missing:
        raise ValueError("expected_labels missing units required by estimates")

    preds = np.asarray([float(row.value) for row in rows], dtype=np.float64)
    labels = np.asarray(
        [_validate_probability(expected_labels[row.unit_id], field_name=f"expected_labels[{row.unit_id}]") for row in rows],
        dtype=np.float64,
    )

    errors = np.abs(preds - labels)
    sq_errors = np.square(preds - labels)

    census_pass_rate = float(np.mean(labels))
    aggregate_error = float(abs(estimates.aggregate.estimated_pass_rate - census_pass_rate))
    per_unit_mae = float(np.mean(errors))
    brier = float(np.mean(sq_errors))

    by_agent: dict[str, list[float]] = {}
    for row, err in zip(rows, errors, strict=False):
        agent = str(estimates.agent_id_by_unit[row.unit_id])
        by_agent.setdefault(agent, []).append(float(err))
    macro_per_agent_mae = float(np.mean([np.mean(vals) for vals in by_agent.values()]))

    judged_mask = np.asarray([row.unit_id in estimates.judged_unit_ids for row in rows], dtype=bool)
    unjudged_mask = ~judged_mask

    judged_only_pass_rate: float | None
    judged_only_abs_error: float | None
    if np.any(judged_mask):
        judged_only_pass_rate = float(np.mean(labels[judged_mask]))
        judged_only_abs_error = float(abs(judged_only_pass_rate - census_pass_rate))
    else:
        judged_only_pass_rate = None
        judged_only_abs_error = None

    unjudged_only_mae: float | None
    unjudged_only_brier: float | None
    if np.any(unjudged_mask):
        unjudged_only_mae = float(np.mean(errors[unjudged_mask]))
        unjudged_only_brier = float(np.mean(sq_errors[unjudged_mask]))
    else:
        unjudged_only_mae = None
        unjudged_only_brier = None

    bins: list[Mapping[str, float]] = []
    n = len(rows)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, calibration_bin_count + 1)
    for i in range(calibration_bin_count):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i < calibration_bin_count - 1:
            mask = (preds >= lo) & (preds < hi)
        else:
            mask = (preds >= lo) & (preds <= hi)
        count = int(np.sum(mask))
        if count == 0:
            avg_pred = 0.0
            avg_label = 0.0
            abs_gap = 0.0
        else:
            avg_pred = float(np.mean(preds[mask]))
            avg_label = float(np.mean(labels[mask]))
            abs_gap = float(abs(avg_pred - avg_label))
            ece += (count / n) * abs_gap

        bins.append(
            {
                "bin_index": float(i),
                "lower": lo,
                "upper": hi,
                "count": float(count),
                "avg_prediction": avg_pred,
                "avg_label": avg_label,
                "abs_gap": abs_gap,
            }
        )

    return EmbeddingPopulationValidation(
        census_pass_rate=census_pass_rate,
        absolute_aggregate_rate_error=aggregate_error,
        per_unit_mae=per_unit_mae,
        brier_score=brier,
        macro_per_agent_mae=macro_per_agent_mae,
        judged_only_pass_rate=judged_only_pass_rate,
        judged_only_absolute_rate_error=judged_only_abs_error,
        unjudged_only_mae=unjudged_only_mae,
        unjudged_only_brier=unjudged_only_brier,
        calibration_bins=tuple(bins),
        expected_calibration_error=float(ece),
    )


def leave_one_out_donor_diagnostics(
    *,
    membership: FrozenMembership,
    agent_id_by_unit: Mapping[str, str],
    vector_by_unit: Mapping[str, object],
    judged_values_by_unit: Mapping[str, float],
    config: IDWConfig = IDWConfig(),
) -> LeaveOneOutDiagnostics:
    eligible_set = set(membership.eligible_ids)
    selected_set = set(membership.selected_ids)
    for uid in membership.eligible_ids:
        if uid not in agent_id_by_unit:
            raise ValueError(f"missing agent_id for eligible unit: {uid}")

    clean_values: dict[str, float] = {}
    for uid, raw_value in judged_values_by_unit.items():
        if uid not in eligible_set or uid not in selected_set:
            raise ValueError("judged_values_by_unit contains unit outside selected_ids")
        clean_values[str(uid)] = _validate_probability(
            raw_value,
            field_name=f"judged_values_by_unit[{uid}]",
        )

    judged_ids = sorted(clean_values)
    if not judged_ids:
        return LeaveOneOutDiagnostics(per_unit_predictions={}, mae=0.0, brier_score=0.0)

    norm_vectors = {uid: _normalize_vector(vector_by_unit.get(uid)) for uid in membership.eligible_ids}

    judged_ids_by_agent: dict[str, list[str]] = {}
    for uid in judged_ids:
        judged_ids_by_agent.setdefault(str(agent_id_by_unit[uid]), []).append(uid)

    donor_ids_by_agent: dict[str, np.ndarray] = {}
    donor_values_by_agent: dict[str, np.ndarray] = {}
    donor_vectors_by_agent: dict[str, np.ndarray] = {}
    agent_value_sums: dict[str, float] = {}
    for agent, agent_judged_ids in judged_ids_by_agent.items():
        stable_ids = sorted(agent_judged_ids)
        agent_value_sums[agent] = float(sum(clean_values[uid] for uid in stable_ids))
        valid_ids = [uid for uid in stable_ids if norm_vectors.get(uid) is not None]
        donor_ids_by_agent[agent] = np.asarray(valid_ids, dtype=object)
        donor_values_by_agent[agent] = np.asarray(
            [clean_values[uid] for uid in valid_ids],
            dtype=np.float64,
        )
        donor_vectors_by_agent[agent] = (
            np.stack([norm_vectors[uid] for uid in valid_ids], axis=0)
            if valid_ids
            else np.zeros((0, 0), dtype=np.float64)
        )

    global_value_sum = float(sum(clean_values.values()))
    global_value_count = len(clean_values)

    rows: dict[str, UnitEstimate] = {}
    abs_errs: list[float] = []
    sq_errs: list[float] = []

    for uid in judged_ids:
        agent = str(agent_id_by_unit[uid])
        global_mean = (
            float((global_value_sum - clean_values[uid]) / (global_value_count - 1))
            if global_value_count > 1
            else None
        )
        agent_value_count = len(judged_ids_by_agent[agent])
        agent_mean = (
            float((agent_value_sums[agent] - clean_values[uid]) / (agent_value_count - 1))
            if agent_value_count > 1
            else None
        )

        pred = _estimate_for_target(
            target_id=uid,
            target_vector=norm_vectors.get(uid),
            donor_ids=donor_ids_by_agent[agent],
            donor_vectors=donor_vectors_by_agent[agent],
            donor_values=donor_values_by_agent[agent],
            agent_mean=agent_mean,
            global_mean=global_mean,
            config=config,
            exclude_donor_id=uid,
        )

        rows[uid] = pred
        err = float(abs(pred.value - clean_values[uid]))
        abs_errs.append(err)
        sq_errs.append(err * err)

    return LeaveOneOutDiagnostics(
        per_unit_predictions=rows,
        mae=float(np.mean(np.asarray(abs_errs, dtype=np.float64))),
        brier_score=float(np.mean(np.asarray(sq_errs, dtype=np.float64))),
    )
