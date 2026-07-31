from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sampling_comparison.v2_experiment import (
    _representative_20pct_membership,
    load_combined_dataset,
    run_v2_experiment_bundle,
    slice_dataset,
)
from sampling_comparison.v2_outputs import (
    build_external_eval_snapshots,
    build_production_storage_manifest,
    validate_external_eval_snapshot,
)


def test_representative_membership_includes_census_baseline() -> None:
    data = slice_dataset(load_combined_dataset(), limit=80)
    membership = _representative_20pct_membership(data, seed=13, precomputed_runtime=None)

    assert membership["version"].endswith("membership-v2")
    assert "census" in membership["methods"]
    census = membership["methods"]["census"]
    assert census["declared_budget"] == "100%"
    assert census["selected_count"] == len(data.unit_ids)
    assert len(census["selected_ids"]) == len(data.unit_ids)


def test_snapshot_grouping_and_payload_semantics_on_slice() -> None:
    data = slice_dataset(load_combined_dataset(), limit=120)
    membership = _representative_20pct_membership(data, seed=7, precomputed_runtime=None)

    snapshots_by_method, provenance = build_external_eval_snapshots(
        data=data,
        representative_membership=membership,
        version="sampling-v2-bundle-v1",
        seed=7,
        run_time_utc=datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc),
        echo_limit=5,
    )

    assert set(snapshots_by_method) == {
        "census",
        "random_sampling_stratified",
        "adaptive_minhash_32x4",
        "adaptive_embedding_fullsession",
    }
    assert provenance["grouping"] == "method x tenant x agent x utc_day"
    assert provenance["tenant_id_omitted_in_body"] is True

    any_payload = None
    for method, payloads in snapshots_by_method.items():
        for payload in payloads:
            any_payload = payload
            assert payload["granularity"] == "day"
            assert payload["totalSampledCount"] > 0
            assert payload["totalConversationsCount"] >= payload["totalSampledCount"]
            assert payload["runId"].startswith("v2-")
            assert len(payload["runId"]) <= 128
            assert len(payload["results"]) <= 5
            for result in payload["results"]:
                assert result["conversationId"]
                assert not result["conversationId"].startswith("historical_300:")
                assert not result["conversationId"].startswith("dense_2500:")
                assert result["conversationEndTimeUtc"].endswith("Z")
                metric = result["metrics"][0]
                assert metric["name"] == "task_completion"
                assert metric["displayName"] == "Task Completion"
                assert metric["model"] == "dataset-expected-label"
                assert metric["scoreScale"] == "binary"
                assert metric["score"] in (0, 1)
                assert metric["passed"] in (True, False)
                assert metric["threshold"] == 1
                assert metric["scoreMin"] == 0
                assert metric["scoreMax"] == 1
                assert "reasoning" not in metric

        if method == "census":
            assert payloads

    assert any_payload is not None


def test_validator_rejects_missing_required_and_bad_metric() -> None:
    payload = {
        "runId": "v2-random-2026-07-30-abcd",
        "agentId": "agent-a",
        "date": "2026-07-30",
        "granularity": "day",
        "createdAtUtc": "2026-07-30T12:00:00Z",
        "completedAtUtc": "2026-07-30T12:00:00Z",
        "totalConversationsCount": 10,
        "totalSampledCount": 3,
        "avgScore": 0.6666666667,
        "results": [
            {
                "conversationId": "conv-1",
                "conversationEndTimeUtc": "2026-07-30T11:00:00Z",
                "metrics": [
                    {
                        "name": "task_completion",
                        "score": 1,
                    }
                ],
            }
        ],
    }

    validate_external_eval_snapshot(
        payload,
        expected={"expected_total_sampled_count": 3, "expected_avg_score": 0.6666666667},
    )

    bad_missing = dict(payload)
    bad_missing.pop("runId")
    with pytest.raises(ValueError):
        validate_external_eval_snapshot(bad_missing)

    bad_metric = json.loads(json.dumps(payload))
    bad_metric["results"][0]["metrics"][0].pop("name")
    with pytest.raises(ValueError):
        validate_external_eval_snapshot(bad_metric)

    bad_avg = dict(payload)
    with pytest.raises(ValueError):
        validate_external_eval_snapshot(bad_avg, expected={"expected_total_sampled_count": 3, "expected_avg_score": 0.5})


def test_bundle_writes_snapshot_and_storage_manifests_and_hashes(tmp_path: Path) -> None:
    data = slice_dataset(load_combined_dataset(), limit=180)
    out_dir = tmp_path / "v2"

    result = run_v2_experiment_bundle(
        data=data,
        enforce_integrity_counts=False,
        budget_pcts=(20,),
        outcome_repetitions=1,
        quadrant_replays=1,
        throughput_replays=1,
        seed=11,
        output_dir=out_dir,
    )

    output_paths = result["output_paths"]
    assert output_paths is not None
    assert Path(output_paths["external_eval_snapshots_manifest"]).exists()
    assert Path(output_paths["production_storage_manifest"]).exists()

    snapshots_manifest = json.loads(
        Path(output_paths["external_eval_snapshots_manifest"]).read_text(encoding="utf-8")
    )
    assert snapshots_manifest["route_template"] == "POST /evals/service/results?api-version=1"
    assert snapshots_manifest["not_posted"] is True
    assert snapshots_manifest["echo_limit"] == 5
    assert "methods_files" in snapshots_manifest

    for method in ("census", "random_sampling_stratified", "adaptive_minhash_32x4", "adaptive_embedding_fullsession"):
        entry = snapshots_manifest["methods_files"][method]
        path = Path(entry["path"])
        assert path.exists()
        assert len(entry["sha256"]) == 64
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in lines:
            row = json.loads(line)
            for result_row in row["results"]:
                metric = result_row["metrics"][0]
                assert "reasoning" not in metric
                assert metric["score"] in (0, 1)

    storage_manifest = json.loads(Path(output_paths["production_storage_manifest"]).read_text(encoding="utf-8"))
    assert storage_manifest["authoritative_state"]["source"] == "ESP/Cosmos"
    assert storage_manifest["scope"]["implemented"] is False
    assert "proposed_logical_model" in storage_manifest
    assert "azure_ai_search_assessment" in storage_manifest
    assert storage_manifest["azure_ai_search_assessment"]["inspection"]["service"] == "stangoodwin-ai-search"
    assert storage_manifest["azure_ai_search_assessment"]["inspection"]["vector_field_present"] is False


def test_storage_manifest_contains_required_sections() -> None:
    manifest = build_production_storage_manifest(generated_at_utc=datetime(2026, 7, 30, 1, 2, 3, tzinfo=timezone.utc))

    assert manifest["generatedAtUtc"] == "2026-07-30T01:02:03Z"
    assert manifest["ppapi_contract_requirements"]["route"] == "POST /evals/service/results?api-version=1"
    assert manifest["ppapi_contract_requirements"]["tenant_handling"].startswith("tenant derived")
    assert "evaluationRuns" in json.dumps(manifest["proposed_logical_model"])
    assert "selectionMembership" in json.dumps(manifest["proposed_logical_model"])
    assert "evaluationFacts" in json.dumps(manifest["proposed_logical_model"])
    assert "similarityState" in json.dumps(manifest["proposed_logical_model"])
