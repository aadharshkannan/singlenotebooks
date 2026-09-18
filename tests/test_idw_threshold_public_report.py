from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from sampling_comparison.idw_threshold_experiment import DATASETS, DIMENSIONS, LIPSCHITZ_CONFIG, RATES, SCHEDULES, SEEDS
from sampling_comparison.idw_threshold_public_report import sanitized_aggregate, write_public_report
from sampling_comparison.idw_threshold_report import build_report
from sampling_comparison.matryoshka_experiment import canonical, sha256_file, write_json
from sampling_comparison.v4_idw import IDWConfig
from test_idw_threshold_report import fixture as report_fixture


SECRET = "SENSITIVE_UNIT_AGENT_TASK_CREDENTIAL"


def fixture():
    source = report_fixture()
    source["source_revision"] = "a" * 40
    source["run_id"] = SECRET
    source["protocol"] = {
        "dimensions": list(DIMENSIONS), "seeds": list(SEEDS), "schedules": list(SCHEDULES),
        "rates": list(RATES), "modes": ["end_to_end"], "idw": asdict(IDWConfig()),
        "lipschitz": asdict(LIPSCHITZ_CONFIG), "angular_units": "arccos(cosine)/pi",
        "budget_rule": "max(1,floor(N*rate))", "threshold_rule": "score >= threshold",
        "canonical_runtime_required": True, "network_calls": 0, "embedding_calls": 0, "judge_calls": 0,
    }
    templates = source["rows"][::5]
    source["rows"] = []
    for template in templates:
        for seed in SEEDS:
            for schedule in SCHEDULES:
                source["rows"].append({
                    **template, "seed": seed, "schedule": schedule, "cell_id": len(source["rows"]),
                    "point_auc_reason": None, "lower_auc_reason": None,
                    "all_unjudged_point_at_0_5": template["paired_point_at_0_5"],
                    "provenance_counts": {"idw": 8, "global_mean": 2},
                    "calibration_fallback_count": 1, "exact_contradiction_target_count": 0,
                    "raw_id": SECRET, "evidence_offset": [SECRET],
                })
    source["validation"].update(cache_snapshot_before={}, cache_snapshot_after={}, expected_cells=4500)
    source["score_consistency_validation"] = {"ok": True, "canonical_cells_checked": 4500, "private_path": SECRET}
    source["costs"].update(network_calls=0, embedding_calls=0, judge_calls=0, external_cost_usd=0)
    source["code_hashes"] = {
        filename: "a" * 64 for filename in (
            "idw_threshold_experiment.py", "matryoshka_experiment.py", "run_idw_threshold_experiment.py", "lipschitz.py",
        )
    }
    for row in source["agent_summary"]:
        row["agent_id"] = SECRET
    cosmos = next(p for p in source["datasets"] if p["dataset_id"] == "cosmos_otel")
    cosmos.update(
        sessions=755, truncated_sessions=725,
        representation_sources={"linked_raw_spans": 745, "synthetic_label_document": 10},
        label_source=SECRET, manifest=SECRET, snapshot_cutoff_utc="2026-09-15T13:40:42+00:00",
        embedding_preparation={"reused_vectors": 205, "requested_vectors": 550, "logical_requests": 48, "api_input_tokens": 4494400},
    )
    return source


def test_publication_keeps_metrics_and_new_cohort_without_raw_identifiers():
    source = fixture()
    public = sanitized_aggregate(source, "refreshed-report")
    assert SECRET not in canonical(public)
    assert public["agent_summary"] == []
    assert len(public["rows"]) == 4500
    assert len(public["agent_gap_summary"]) == 6
    assert all(row["agents_with_recall_loss"] == 1 for row in public["agent_gap_summary"])
    html = build_report(public, "aggregate.json")
    assert SECRET not in html
    assert "745 linked raw-span" in html
    assert "725 sessions were truncated" in html
    assert "7/15/37/75/151" in html
    assert "550 new native" in html
    assert "Largest displayed per-agent" not in html
    assert "Aggregate-only publication" in html
    assert '<option value="cosmos_otel" selected>' in html
    assert "195 linked raw-span" not in html
    assert "runtime_variants/" not in html


def test_private_files_unchanged_and_public_source_hashes_are_retained(tmp_path: Path):
    private = tmp_path / "private"
    source_path = private / "aggregate.json"
    write_json(source_path, fixture())
    write_json(private / "manifest.json", {"files": {"aggregate": {"path": str(source_path), "sha256": sha256_file(source_path)}}})
    before = (source_path.read_bytes(), (private / "manifest.json").read_bytes())
    output = tmp_path / "public"
    write_public_report(private, output)
    assert before == (source_path.read_bytes(), (private / "manifest.json").read_bytes())
    for path in output.iterdir():
        assert SECRET not in path.read_text(encoding="utf-8")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_private_aggregate_sha256"] == sha256_file(source_path)
    with pytest.raises(FileExistsError):
        write_public_report(private, output)
    source_path.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        write_public_report(private, tmp_path / "fresh")


def test_relocated_manifest_cannot_validate_a_different_aggregate(tmp_path: Path):
    original = tmp_path / "original" / "aggregate.json"
    source = fixture()
    write_json(original, source)
    copied = tmp_path / "copied"
    write_json(copied / "manifest.json", {
        "files": {"aggregate": {"path": str(original), "sha256": sha256_file(original)}},
    })
    source["rows"][0]["point_auc"] = 0.99
    write_json(copied / "aggregate.json", source)
    with pytest.raises(ValueError, match="aggregate.*manifest"):
        write_public_report(copied, tmp_path / "public")
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("change", ["protocol", "parity", "missing", "duplicate", "private-reason"])
def test_publication_rejects_inconsistent_or_unrecognized_inputs(change):
    source = deepcopy(fixture())
    if change == "protocol":
        source["protocol"]["judge_calls"] = 1
    elif change == "parity":
        source["validation"]["baseline_max_mae_absolute_error"] = 0.01
    elif change == "missing":
        source["rows"].pop()
    elif change == "duplicate":
        source["rows"][1] = source["rows"][0]
    else:
        source["rows"][0]["point_auc_reason"] = SECRET
    with pytest.raises(ValueError):
        sanitized_aggregate(source, "public")
