from copy import deepcopy

import pytest

from scripts.validate_matryoshka_comparison import compare_aggregates


def fixture():
    profile = {
        "dataset_id": "dense_2500", "status": "completed", "n": 20, "agents": 2,
        "positive_count": 10, "pass_rate": 0.5, "source_hashes": {"labels": "abc"},
        "provenance": {"input_manifest_sha256": "def"},
    }
    baseline = {
        "version": "matryoshka-cutoff-v1", "run_id": "baseline",
        "protocol": {"dimensions": [1536, 8], "seeds": [13], "rates": [0.2], "schedules": ["bursty"]},
        "datasets": [profile],
        "rows": [
            {"dataset_id": "dense_2500", "dimension": d, "seed": 13,
             "schedule": "bursty", "rate": 0.2, "mode": "end_to_end", "mae": 0.3}
            for d in (1536, 8)
        ],
    }
    current = deepcopy(baseline)
    current["run_id"] = "new"
    current["protocol"]["dimensions"] = [1536, 8, 6, 4, 2]
    current["rows"].extend({**baseline["rows"][0], "dimension": d} for d in (6, 4, 2))
    return current, baseline


def test_only_dimensions_change_and_overlapping_cells_are_identical():
    result = compare_aggregates(*fixture())
    assert result["only_protocol_change"] == "dimensions"
    assert result["identical_overlap_result_cells"] == 2
    assert result["comparison_tolerance"] == 0


@pytest.mark.parametrize("change", ["protocol", "environment", "profile", "source", "metric", "missing", "duplicate"])
def test_baseline_drift_is_not_silently_accepted(change):
    current, baseline = fixture()
    if change == "protocol":
        current["protocol"]["seeds"] = [14]
    elif change == "environment":
        current["environment"] = {"numpy": "different"}
    elif change == "profile":
        current["datasets"][0]["agents"] = 3
    elif change == "source":
        current["datasets"][0]["provenance"]["input_manifest_sha256"] = "changed"
    elif change == "metric":
        current["rows"][0]["mae"] += 0.000001
    elif change == "missing":
        current["rows"].pop(0)
    else:
        current["rows"].append(current["rows"][0])
    with pytest.raises(ValueError):
        compare_aggregates(current, baseline)


def refresh_fixture():
    current, baseline = fixture()
    for aggregate in (current, baseline):
        profile = deepcopy(aggregate["datasets"][0])
        profile["dataset_id"] = "cosmos_otel"
        aggregate["datasets"].append(profile)
        aggregate["rows"].extend(
            {**row, "dataset_id": "cosmos_otel"} for row in list(aggregate["rows"])
        )
    current["datasets"][1].update(n=30, positive_count=15)
    current["datasets"][1]["provenance"]["input_manifest_sha256"] = "refreshed"
    for row in current["rows"]:
        if row["dataset_id"] == "cosmos_otel":
            row["mae"] = 0.9
    return current, baseline


def test_explicit_refresh_checks_unchanged_cohort_without_comparing_changed_population():
    current, baseline = refresh_fixture()
    result = compare_aggregates(current, baseline, refreshed_datasets=("cosmos_otel",))
    assert result["input_manifests_identical"] == ["dense_2500"]
    assert result["identical_overlap_result_cells"] == 2
    assert result["refreshed_datasets"]["cosmos_otel"]["current_units"] == 30
    with pytest.raises(ValueError):
        compare_aggregates(current, baseline)
    current["rows"][0]["mae"] = 0.8
    with pytest.raises(ValueError, match="measured cell"):
        compare_aggregates(current, baseline, refreshed_datasets=("cosmos_otel",))


@pytest.mark.parametrize("names", [("missing",), ("cosmos_otel", "cosmos_otel"), ("dense_2500", "cosmos_otel")])
def test_refresh_cannot_hide_missing_duplicate_or_all_cohorts(names):
    with pytest.raises(ValueError):
        compare_aggregates(*refresh_fixture(), refreshed_datasets=names)
