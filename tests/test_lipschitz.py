import math

import pytest

from trace_sampling.lipschitz import (
    BinaryClusterSummary,
    LipschitzEstimate,
    LipschitzEstimatorConfig,
    calculate_conditional_geodesic_bounds,
    estimate_binary_lipschitz,
)


def test_noise_corrected_estimate_matches_reference_fixture() -> None:
    estimate = estimate_binary_lipschitz(
        (
            BinaryClusterSummary("a", (1.0, 0.0), successes=45, count=50),
            BinaryClusterSummary("b", (2**-0.5, 2**-0.5), successes=35, count=50),
            BinaryClusterSummary("c", (0.0, 1.0), successes=15, count=50),
        ),
        LipschitzEstimatorConfig(quantile=0.90),
    )

    assert estimate.value == pytest.approx(0.4629945461053807)
    assert estimate.provenance == "empirical_smoothed_cluster_rates"
    assert estimate.pair_count == 3


def test_estimator_uses_positive_fallback_when_support_is_sparse() -> None:
    estimate = estimate_binary_lipschitz(
        (BinaryClusterSummary("only", (1.0, 0.0), successes=8, count=10),),
        LipschitzEstimatorConfig(conservative_fallback=0.75),
    )

    assert estimate.value == 0.75
    assert estimate.provenance == "configured_fallback_insufficient_clusters"
    assert estimate.usable_clusters == 1


def test_estimator_uses_pair_floor_fallback_when_all_pairs_too_close() -> None:
    estimate = estimate_binary_lipschitz(
        (
            BinaryClusterSummary("a", (1.0, 0.0), successes=6, count=10),
            BinaryClusterSummary("b", (1.0, 0.0), successes=4, count=10),
        ),
        LipschitzEstimatorConfig(theta_floor=0.05, conservative_fallback=0.33),
    )

    assert estimate.value == pytest.approx(0.33)
    assert estimate.provenance == "configured_fallback_no_usable_pairs"
    assert estimate.pair_count == 0


def test_estimator_is_deterministic_under_reordering_and_subsampling() -> None:
    clusters = (
        BinaryClusterSummary("a", (1.0, 0.0), successes=42, count=60),
        BinaryClusterSummary("b", (math.cos(0.4), math.sin(0.4)), successes=28, count=50),
        BinaryClusterSummary("c", (0.0, 1.0), successes=21, count=55),
        BinaryClusterSummary("d", (-math.cos(0.4), math.sin(0.4)), successes=10, count=45),
    )
    cfg = LipschitzEstimatorConfig(max_pairs=3, seed=19, quantile=0.8)

    first = estimate_binary_lipschitz(clusters, cfg)
    second = estimate_binary_lipschitz(tuple(reversed(clusters)), cfg)

    assert first.value == pytest.approx(second.value)
    assert first.median_slope == pytest.approx(second.median_slope)
    assert first.pair_count == second.pair_count == 3


def test_conditional_geodesic_bounds_numeric_fixture() -> None:
    estimate = LipschitzEstimate(
        value=0.5,
        provenance="unit_fixture",
        usable_clusters=2,
        pair_count=1,
        quantile=0.9,
        median_slope=0.5,
        mean_rate_variance=0.0,
    )

    bounds = calculate_conditional_geodesic_bounds(0.65, 0.2, estimate)

    assert bounds.probability.allowance == pytest.approx(0.1)
    assert bounds.probability.lower == pytest.approx(0.55)
    assert bounds.probability.upper == pytest.approx(0.75)
    assert bounds.is_confidence_interval is False
    assert "not confidence intervals" in bounds.statement.lower()
