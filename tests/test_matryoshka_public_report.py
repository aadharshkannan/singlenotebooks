from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from sampling_comparison.matryoshka_experiment import canonical, sha256_file, write_json
from sampling_comparison.matryoshka_public_report import sanitized_aggregate, snapshot_summary, write_public_report
from sampling_comparison.matryoshka_summary_report import summarize_evidence
from sampling_comparison.v4_idw import IDWConfig
from test_matryoshka_summary_report import low_dimension_fixture


SENTINEL = "PRIVATE_SESSION_AGENT_TASK_CREDENTIAL_SENTINEL"


def fixture():
    source = low_dimension_fixture()
    source.update(source_revision="a" * 40, status="partial")
    source["run_id"] = SENTINEL
    source["files"] = {"raw": SENTINEL}
    source["environment"] = {"private_path": SENTINEL}
    source["protocol"]["endpoint"] = SENTINEL
    source["protocol"].update(
        idw=asdict(IDWConfig()), tau=0.55, modes=["end_to_end", "fixed_membership"],
        budget_unit="sessions", budget_rule="max(1,floor(N*rate))",
        causal_donors=True, causal_membership=False, bootstrap=False,
        live_embedding_calls=0, live_judge_calls=0, search_calls=0,
    )
    source["agent_summary"] = []
    for dataset in ("historical_300", "dense_2500", "cosmos_otel"):
        for mode in ("end_to_end", "fixed_membership"):
            for dimension in source["protocol"]["dimensions"]:
                for agent, delta in ((SENTINEL + "-a", 0.01), (SENTINEL + "-b", -0.01)):
                    source["agent_summary"].append({
                        "dataset_id": dataset, "mode": mode, "dimension": dimension,
                        "agent_id": agent, "mae": 0.3 + (delta if dimension != 1536 else 0),
                    })
    for profile in source["datasets"]:
        profile["reason"] = SENTINEL
        if profile["status"] == "completed":
            profile["provenance"].update(
                endpoint=SENTINEL, label_source=SENTINEL,
                input_manifest=SENTINEL, agent_counts={SENTINEL: 100},
                source_paths={"labels": SENTINEL},
            )
    for row in source["rows"]:
        row["membership_sha256"] = SENTINEL
        row["per_agent"] = {SENTINEL: {"mae": 0.5}}
    return source


def test_public_aggregate_preserves_metrics_without_identifiers():
    source = fixture()
    public = sanitized_aggregate(source)
    assert SENTINEL not in canonical(public)
    assert "agent_id" not in canonical(public)
    assert public["agent_summary"] == []
    assert len(public["rows"]) == len(source["rows"])
    before = summarize_evidence(source, focus_dimension=2)
    after = summarize_evidence(public, focus_dimension=2)
    for left, right in zip(before["datasets"], after["datasets"], strict=True):
        for key in ("curve", "native_mae", "short_mae", "budget_rows", "fallback_shares"):
            assert left[key] == right[key]
        assert right["agents_compared"] == 2
        assert right["agents_with_higher_mae"] == 1


def test_publication_is_self_contained_and_preserves_private_bytes(tmp_path: Path, monkeypatch):
    def unexpected_readiness(*args, **kwargs):
        raise AssertionError("publication must not load ancillary private files")

    monkeypatch.setattr("sampling_comparison.matryoshka_report._load_input_readiness", unexpected_readiness)
    source_root = tmp_path / "private"
    write_json(source_root / "aggregate.json", fixture())
    write_json(source_root / "manifest.json", {
        "files": {"aggregate": {"sha256": sha256_file(source_root / "aggregate.json")}},
    })
    before = [(source_root / name).read_bytes() for name in ("aggregate.json", "manifest.json")]
    output = tmp_path / "public-report"
    report = write_public_report(source_root, output)
    assert report.is_file()
    assert (output / "detail" / "report.html").is_file()
    for path in output.rglob("*"):
        if path.is_file():
            assert SENTINEL not in path.read_text(encoding="utf-8")
    html = report.read_text(encoding="utf-8")
    assert 'href="detail/report.html"' in html
    assert "Aggregate-only publication" in html
    assert "Identifiers" not in json.dumps(json.loads((output / "aggregate.json").read_text())["rows"])
    assert before == [(source_root / name).read_bytes() for name in ("aggregate.json", "manifest.json")]
    with pytest.raises(FileExistsError):
        write_public_report(source_root, output)
    with pytest.raises(ValueError, match="separate"):
        write_public_report(source_root, source_root / "public")


def test_tampered_private_source_is_rejected(tmp_path: Path):
    source_root = tmp_path / "private"
    write_json(source_root / "aggregate.json", fixture())
    write_json(source_root / "manifest.json", {"files": {"aggregate": {"sha256": "b" * 64}}})
    with pytest.raises(ValueError, match="manifest"):
        write_public_report(source_root, tmp_path / "public")
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("field", ["source_revision", "status"])
def test_free_text_cannot_escape_through_operational_metadata(field):
    source = deepcopy(fixture())
    source[field] = SENTINEL
    with pytest.raises(ValueError):
        sanitized_aggregate(source)


def test_snapshot_metadata_is_bound_to_measured_cohort_without_source_identifiers():
    source = fixture()
    cosmos = next(row for row in source["datasets"] if row["dataset_id"] == "cosmos_otel")
    cosmos["positive_count"] = 123
    cosmos["source_hashes"] = {"labels": "a" * 64, "spans": "b" * 64}
    public = sanitized_aggregate(source)
    readiness = {
        "eligible_units": 205, "agents": 7, "outcome_counts": {"good": 123},
        "containers": [{"container": "labels", "sha256": "a" * 64}, {"container": "spans", "sha256": "b" * 64}],
        "snapshot_cutoff_utc": "2026-09-15T13:40:42+00:00", "point_in_time_snapshot": False,
        "snapshot_documents": 103373, "snapshot_bytes": 869196110,
        "representation_sources": {"linked_raw_spans": 195, "synthetic_label_document": 10},
        "private_identifier": SENTINEL,
    }
    result = snapshot_summary(readiness, public)
    assert result["eligible_units"] == 205
    assert SENTINEL not in canonical(result)
    public_cosmos = next(row for row in public["datasets"] if row["dataset_id"] == "cosmos_otel")
    public_cosmos["provenance"].update(unique_packets=204, reused_vectors=100, requested_vectors=104)
    result = snapshot_summary(readiness, public)
    assert result["eligible_units"] == 205
    assert result["embedding_preparation"]["requested_vectors"] == 104
    public_cosmos["provenance"]["requested_vectors"] = 105
    with pytest.raises(ValueError, match="unique canonical packets"):
        snapshot_summary(readiness, public)
    public_cosmos["provenance"]["requested_vectors"] = 104
    public_cosmos["provenance"].pop("unique_packets")
    with pytest.raises(ValueError, match="unique-packet count"):
        snapshot_summary(readiness, public)
    readiness["eligible_units"] = 755
    with pytest.raises(ValueError, match="population"):
        snapshot_summary(readiness, public)


@pytest.mark.parametrize("field,value", [("live_judge_calls", 1), ("causal_donors", False), ("budget_unit", "tokens")])
def test_publication_cannot_replace_an_incompatible_protocol_with_the_expected_narrative(field, value):
    source = fixture()
    source["protocol"][field] = value
    with pytest.raises(ValueError, match="protocol"):
        sanitized_aggregate(source)
