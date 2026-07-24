from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BinaryClusterSummary:
    cluster_id: str
    centroid: tuple[float, ...]
    successes: int
    count: int

    def __post_init__(self) -> None:
        if not self.cluster_id.strip():
            raise ValueError("cluster_id must not be empty")
        vector = tuple(float(value) for value in self.centroid)
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("centroid must be a finite, non-empty vector")
        if math.sqrt(sum(value * value for value in vector)) == 0.0:
            raise ValueError("centroid must have nonzero norm")
        if isinstance(self.successes, bool) or isinstance(self.count, bool):
            raise TypeError("successes and count must be integers")
        if int(self.successes) != self.successes or int(self.count) != self.count:
            raise ValueError("successes and count must be integers")
        if self.count < 0 or self.successes < 0 or self.successes > self.count:
            raise ValueError("cluster counts must satisfy 0 <= successes <= count")
        object.__setattr__(self, "centroid", vector)


@dataclass(frozen=True)
class BoundedClusterSummary:
    cluster_id: str
    centroid: tuple[float, ...]
    value_sum: float
    count: int

    def __post_init__(self) -> None:
        if not self.cluster_id.strip():
            raise ValueError("cluster_id must not be empty")
        vector = tuple(float(value) for value in self.centroid)
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("centroid must be a finite, non-empty vector")
        if math.sqrt(sum(value * value for value in vector)) == 0.0:
            raise ValueError("centroid must have nonzero norm")
        value_sum = float(self.value_sum)
        if not math.isfinite(value_sum):
            raise ValueError("value_sum must be finite")
        if isinstance(self.count, bool) or int(self.count) != self.count:
            raise ValueError("count must be an integer")
        if self.count < 0 or value_sum < 0.0 or value_sum > self.count:
            raise ValueError("cluster values must satisfy 0 <= value_sum <= count")
        object.__setattr__(self, "centroid", vector)
        object.__setattr__(self, "value_sum", value_sum)


@dataclass(frozen=True)
class LipschitzEstimatorConfig:
    quantile: float = 0.90
    theta_floor: float = 1e-2
    beta_alpha: float = 0.5
    beta_beta: float = 0.5
    min_cluster_count: int = 1
    conservative_fallback: float = 1.0
    max_pairs: int = 200_000
    seed: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.quantile) or not 0.0 < self.quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")
        if not math.isfinite(self.theta_floor) or self.theta_floor <= 0.0:
            raise ValueError("theta_floor must be finite and > 0")
        if not math.isfinite(self.beta_alpha) or self.beta_alpha <= 0.0:
            raise ValueError("beta_alpha must be finite and > 0")
        if not math.isfinite(self.beta_beta) or self.beta_beta <= 0.0:
            raise ValueError("beta_beta must be finite and > 0")
        if self.min_cluster_count < 1:
            raise ValueError("min_cluster_count must be >= 1")
        if not math.isfinite(self.conservative_fallback) or self.conservative_fallback <= 0.0:
            raise ValueError("conservative_fallback must be finite and > 0")
        if self.max_pairs < 1:
            raise ValueError("max_pairs must be >= 1")


@dataclass(frozen=True)
class LipschitzEstimate:
    value: float
    provenance: str
    usable_clusters: int
    pair_count: int
    quantile: float
    median_slope: float
    mean_rate_variance: float
    candidate_pair_count: int = 0
    sampled_pair_count: int = 0

    def __post_init__(self) -> None:
        for field_name in ("value", "quantile", "median_slope", "mean_rate_variance"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and >= 0")
            object.__setattr__(self, field_name, value)
        if not self.provenance.strip():
            raise ValueError("provenance must not be empty")
        if (
            self.usable_clusters < 0
            or self.pair_count < 0
            or self.candidate_pair_count < 0
            or self.sampled_pair_count < 0
        ):
            raise ValueError("support counts must be >= 0")
        if self.pair_count > 0 and self.candidate_pair_count == 0 and self.sampled_pair_count == 0:
            object.__setattr__(self, "candidate_pair_count", self.pair_count)
            object.__setattr__(self, "sampled_pair_count", self.pair_count)
        if self.sampled_pair_count > self.candidate_pair_count:
            raise ValueError("sampled_pair_count must be <= candidate_pair_count")
        if self.pair_count > self.sampled_pair_count:
            raise ValueError("pair_count must be <= sampled_pair_count")


@dataclass(frozen=True)
class ProbabilityBounds:
    center: float
    allowance: float
    raw_lower: float
    raw_upper: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        for field_name in ("center", "allowance", "raw_lower", "raw_upper", "lower", "upper"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)
        if self.allowance < 0.0:
            raise ValueError("allowance must be >= 0")
        if self.raw_lower > self.raw_upper:
            raise ValueError("raw bounds must be ordered")
        if self.lower > self.upper:
            raise ValueError("display bounds must be ordered")
        if self.lower < 0.0 or self.upper > 1.0:
            raise ValueError("display bounds must be in [0, 1]")


@dataclass(frozen=True)
class ConditionalGeodesicBounds:
    weighted_geodesic_angle: float
    estimate: LipschitzEstimate
    probability: ProbabilityBounds
    is_confidence_interval: bool = False
    statement: str = (
        "Conditional geodesic bounds are deterministic Lipschitz envelopes, not confidence intervals."
    )

    def __post_init__(self) -> None:
        angle = float(self.weighted_geodesic_angle)
        if not math.isfinite(angle) or angle < 0.0 or angle > math.pi:
            raise ValueError("weighted_geodesic_angle must be finite and in [0, pi]")
        object.__setattr__(self, "weighted_geodesic_angle", angle)
        if self.is_confidence_interval:
            raise ValueError("conditional geodesic bounds are not confidence intervals")
        if not self.statement.strip():
            raise ValueError("statement must not be empty")


def estimate_binary_lipschitz(
    clusters: tuple[BinaryClusterSummary, ...],
    config: LipschitzEstimatorConfig = LipschitzEstimatorConfig(),
) -> LipschitzEstimate:
    bounded_clusters = tuple(
        BoundedClusterSummary(
            cluster_id=cluster.cluster_id,
            centroid=cluster.centroid,
            value_sum=float(cluster.successes),
            count=cluster.count,
        )
        for cluster in clusters
    )
    return _estimate_lipschitz(bounded_clusters, config, "empirical_smoothed_cluster_rates")


def estimate_bounded_lipschitz(
    clusters: tuple[BoundedClusterSummary, ...],
    config: LipschitzEstimatorConfig = LipschitzEstimatorConfig(),
) -> LipschitzEstimate:
    return _estimate_lipschitz(clusters, config, "empirical_smoothed_cluster_values")


def _estimate_lipschitz(
    clusters: tuple[BoundedClusterSummary, ...],
    config: LipschitzEstimatorConfig,
    provenance: str,
) -> LipschitzEstimate:
    ordered = tuple(sorted(clusters, key=lambda cluster: cluster.cluster_id.casefold()))
    usable = tuple(cluster for cluster in ordered if cluster.count >= config.min_cluster_count)
    if len(usable) < 2:
        return _fallback_estimate(
            value=config.conservative_fallback,
            provenance="configured_fallback_insufficient_clusters",
            usable_clusters=len(usable),
            config=config,
        )

    dimensions = {len(cluster.centroid) for cluster in usable}
    if len(dimensions) != 1:
        raise ValueError("all usable cluster centroids must have the same dimension")

    centroids = np.asarray([cluster.centroid for cluster in usable], dtype=np.float64)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    counts = np.asarray([cluster.count for cluster in usable], dtype=np.float64)
    value_sums = np.asarray([cluster.value_sum for cluster in usable], dtype=np.float64)
    prior_total = config.beta_alpha + config.beta_beta
    # Beta smoothing stabilizes sparse cluster rates and supplies the variance
    # correction used below so label noise does not inflate the estimated slope.
    rates = (value_sums + config.beta_alpha) / (counts + prior_total)
    rate_variance = counts * rates * (1.0 - rates) / np.square(counts + prior_total)

    first, second = np.triu_indices(len(usable), k=1)
    candidate_pair_count = int(first.size)
    if first.size > config.max_pairs:
        # Bound calibration cost while keeping repeated estimates reproducible.
        generator = np.random.default_rng(config.seed)
        selected = generator.choice(first.size, size=config.max_pairs, replace=False)
        first, second = first[selected], second[selected]
    sampled_pair_count = int(first.size)

    cosine = np.clip(np.sum(centroids[first] * centroids[second], axis=1), -1.0, 1.0)
    angles = np.arccos(cosine)
    separated = angles > config.theta_floor
    angles = angles[separated]
    first, second = first[separated], second[separated]
    if angles.size == 0:
        return _fallback_estimate(
            value=config.conservative_fallback,
            provenance="configured_fallback_no_usable_pairs",
            usable_clusters=len(usable),
            config=config,
            mean_rate_variance=float(np.mean(rate_variance)),
        )

    # Remove rate variance before dividing by angular separation; the remaining
    # slope estimates how quickly the underlying value surface can change.
    corrected_square = np.maximum(
        0.0,
        np.square(rates[first] - rates[second]) - rate_variance[first] - rate_variance[second],
    )
    slopes = np.sqrt(corrected_square) / angles
    return LipschitzEstimate(
        value=float(np.quantile(slopes, config.quantile)),
        provenance=provenance,
        usable_clusters=len(usable),
        pair_count=int(slopes.size),
        quantile=config.quantile,
        median_slope=float(np.median(slopes)),
        mean_rate_variance=float(np.mean(rate_variance)),
        candidate_pair_count=candidate_pair_count,
        sampled_pair_count=sampled_pair_count,
    )


def calculate_conditional_geodesic_bounds(
    probability: float,
    weighted_angle: float,
    estimate: LipschitzEstimate,
) -> ConditionalGeodesicBounds:
    probability_value = float(probability)
    if not math.isfinite(probability_value) or probability_value < 0.0 or probability_value > 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    angle = float(weighted_angle)
    if not math.isfinite(angle) or angle < 0.0 or angle > math.pi:
        raise ValueError("weighted_angle must be finite and in [0, pi]")
    lipschitz_value = float(estimate.value)
    if lipschitz_value < 0.0:
        raise ValueError("estimate.value must be >= 0")

    # Preserve raw bounds for audit, but clamp the displayed probability band.
    allowance = lipschitz_value * angle
    raw_lower = probability_value - allowance
    raw_upper = probability_value + allowance
    lower = min(1.0, max(0.0, raw_lower))
    upper = min(1.0, max(0.0, raw_upper))

    probability_bounds = ProbabilityBounds(
        center=probability_value,
        allowance=allowance,
        raw_lower=raw_lower,
        raw_upper=raw_upper,
        lower=lower,
        upper=upper,
    )

    return ConditionalGeodesicBounds(
        weighted_geodesic_angle=angle,
        estimate=estimate,
        probability=probability_bounds,
        is_confidence_interval=False,
    )


def _fallback_estimate(
    *,
    value: float,
    provenance: str,
    usable_clusters: int,
    config: LipschitzEstimatorConfig,
    mean_rate_variance: float = 0.0,
) -> LipschitzEstimate:
    return LipschitzEstimate(
        value=value,
        provenance=provenance,
        usable_clusters=usable_clusters,
        pair_count=0,
        quantile=config.quantile,
        median_slope=0.0,
        mean_rate_variance=mean_rate_variance,
        candidate_pair_count=0,
        sampled_pair_count=0,
    )
