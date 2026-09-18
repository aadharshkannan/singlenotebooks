from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any, Mapping

from sampling_comparison.matryoshka_experiment import DATASETS, MODES, SCHEDULES, sha256_file, write_json
from sampling_comparison.matryoshka_report import build_report as build_detailed_report
from sampling_comparison.matryoshka_summary_report import build_report, summarize_evidence
from sampling_comparison.v4_idw import IDWConfig


VERSION = "matryoshka-aggregate-publication-v1"
LABEL_SOURCES = {
    "historical_300": "Retained synthetic expected task-completion labels; no fresh judges.",
    "dense_2500": "Retained synthetic expected task-completion labels; no fresh judges.",
    "cosmos_otel": "Task-design expected_outcome from the labels container: good=1, bad/partial=0; not verified actual completion or fresh judgments.",
    "tau2_bench": "Not evaluated: no approved expected-label mapping.",
}
REPRESENTATIONS = ("combined_normalized", "linked_raw_spans", "synthetic_label_document")
SOURCE_FILES = ("historical_300", "dense_2500", "labels", "spans")
METRICS = (
    "accuracy", "mae", "f1", "brier", "macro_agent_accuracy", "combined_accuracy",
    "aggregate_rate_error", "concept_coverage", "agent_coverage", "accuracy_delta_native",
    "membership_jaccard_native", "n", "selected_count", "unjudged_count",
    "tp", "fp", "fn", "tn", "representation_fallbacks", "native_candidate_count",
)
PROVENANCE = ("observed", "idw", "exact_match", "global_mean", "prior")
PRIVACY_NOTICE = (
    '<aside style="padding:1rem;border:2px solid #155e75;margin:1rem 0">'
    "<strong>Aggregate-only publication.</strong> Raw inputs, identifiers, membership records "
    "and per-target evidence remain in protected local storage. Agent-regression counts are "
    "computed from private paired records without publishing identities. Source hashes bind "
    "this report to those records; they are not copies of the sensitive data.</aside>"
)


def _number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("publication contains a nonnumeric or nonfinite metric")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid whole-artifact SHA256")
    return value


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    if row["dataset_id"] not in DATASETS or row["mode"] not in MODES:
        raise ValueError("unexpected dataset or mode")
    dimension = row["dimension"]
    if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= 1536:
        raise ValueError("unexpected embedding dimension")
    return {"dataset_id": row["dataset_id"], "mode": row["mode"], "dimension": dimension}


def _numeric_fields(row: Mapping[str, Any], fields) -> dict[str, Any]:
    return {name: _number(row[name]) for name in fields if name in row}


def sanitized_aggregate(
    source: Mapping[str, Any], *, focus_dimension: int = 2,
    publication_id: str = "aggregate-only-dimension-study",
) -> dict[str, Any]:
    if re.fullmatch(r"[a-z0-9-]+", publication_id) is None:
        raise ValueError("publication ID must be an operational slug, not arbitrary source text")
    if source["status"] not in ("complete", "partial"):
        raise ValueError("only completed or explicitly partial runs can be published")
    original_evidence = summarize_evidence(source, focus_dimension=focus_dimension)
    protocol = source["protocol"]
    fixed_protocol = {
        "idw": asdict(IDWConfig()), "tau": 0.55, "modes": list(MODES),
        "budget_unit": "sessions", "budget_rule": "max(1,floor(N*rate))",
        "causal_donors": True, "causal_membership": False, "bootstrap": False,
        "live_embedding_calls": 0, "live_judge_calls": 0, "search_calls": 0,
    }
    if any(protocol.get(name) != value for name, value in fixed_protocol.items()):
        raise ValueError("private protocol differs from the fixed offline ARM2/IDW report contract")
    if any(schedule not in SCHEDULES for schedule in protocol["schedules"]):
        raise ValueError("unknown arrival schedule")
    public_protocol = {
        "dimensions": [_number(d) for d in protocol["dimensions"]],
        "seeds": [_number(seed) for seed in protocol["seeds"]],
        "rates": [_number(rate) for rate in protocol["rates"]],
        "schedules": list(protocol["schedules"]),
        "modes": list(MODES),
    }
    for key in ("tau", "accuracy_tolerance"):
        if key in protocol:
            public_protocol[key] = _number(protocol[key])
    if "idw" in protocol:
        public_protocol["idw"] = _numeric_fields(protocol["idw"], ("k", "power", "eps", "exact_cosine_eps", "prior"))
    public_protocol.update(
        budget_unit="sessions", budget_rule="max(1,floor(N*rate))",
        causal_donors=True, causal_membership=False, bootstrap=False,
        live_embedding_calls=0, live_judge_calls=0, search_calls=0,
    )
    profiles = []
    for profile in source["datasets"]:
        dataset = profile["dataset_id"]
        if dataset not in DATASETS or profile["status"] not in ("completed", "blocked"):
            raise ValueError("unexpected dataset profile")
        safe = {
            "dataset_id": dataset, "status": profile["status"],
            **_numeric_fields(profile, ("n", "agents", "positive_count", "pass_rate")),
        }
        provenance = profile.get("provenance", {})
        if profile["status"] == "completed":
            if provenance["embedding_model_id"] != "text-embedding-3-small" or provenance["embedding_dimensions"] != 1536:
                raise ValueError("unexpected embedding model")
            if provenance["live_judge_calls"] != 0:
                raise ValueError("fresh judge labels are outside this publication contract")
            safe["provenance"] = {
                "embedding_model_id": "text-embedding-3-small", "embedding_dimensions": 1536,
                "label_source": LABEL_SOURCES[dataset],
                "representation_policy": "V3 weighted canonical full-session evidence; cl100k_base, maximum 8191 tokens; normalized first-coordinate prefixes.",
                **_numeric_fields(provenance, (
                    "live_embedding_calls", "live_judge_calls", "truncated_sessions", "excluded_sessions",
                    "api_input_tokens", "reused_vectors", "requested_vectors", "unique_packets",
                    "logical_requests", "http_attempts", "http_successes",
                )),
                "representation_sources": {
                    name: _number(provenance["representation_sources"][name])
                    for name in REPRESENTATIONS if name in provenance.get("representation_sources", {})
                },
            }
        else:
            safe["reason"] = LABEL_SOURCES[dataset]
        safe["source_hashes"] = {
            name: _digest(profile["source_hashes"][name])
            for name in SOURCE_FILES if name in profile.get("source_hashes", {})
        }
        profiles.append(safe)
    rows = []
    for row in source["rows"]:
        if row["schedule"] not in SCHEDULES:
            raise ValueError("unknown row schedule")
        rows.append({
            **_identity(row), **_numeric_fields(row, (*METRICS, "seed", "rate")),
            "schedule": row["schedule"],
            "provenance_counts": {
                name: _number(row["provenance_counts"][name])
                for name in PROVENANCE if name in row["provenance_counts"]
            },
        })
    summaries = []
    for row in source.get("summary", []):
        safe = {**_identity(row), **_numeric_fields(row, METRICS)}
        interval = row.get("accuracy_delta_seed_ci95")
        safe["accuracy_delta_seed_ci95"] = [_number(value) for value in interval] if interval is not None else None
        summaries.append(safe)
    decisions = []
    for row in source.get("decisions", []):
        if row["dataset_id"] not in DATASETS or row["mode"] not in MODES:
            raise ValueError("unknown decision cohort")
        decisions.append({
            "dataset_id": row["dataset_id"], "mode": row["mode"],
            **_numeric_fields(row, ("smallest_non_degrading_dimension", "one_pp_candidate_dimension")),
            "reason": "Exploratory accuracy-grid decision; not an agreed MAE tolerance, production guarantee or independent cutoff validation.",
        })
    agents = {
        (row["dataset_id"], row["mode"], row["dimension"], row["agent_id"]): row["mae"]
        for row in source.get("agent_summary", []) if row["mae"] is not None
    }
    gap_counts = []
    for dataset in DATASETS:
        for mode in MODES:
            for dimension in protocol["dimensions"]:
                differences = [
                    float(value) - float(agents[(dataset, mode, 1536, agent)])
                    for (cohort, kind, size, agent), value in agents.items()
                    if (cohort, kind, size) == (dataset, mode, dimension)
                    and (dataset, mode, 1536, agent) in agents
                ]
                if differences:
                    gap_counts.append({
                        "dataset_id": dataset, "mode": mode, "dimension": dimension,
                        "agents_compared": len(differences),
                        "agents_with_higher_mae": sum(value > 0 for value in differences),
                        "median_mae_delta": median(differences), "worst_mae_delta": max(differences),
                    })
    revision = source.get("source_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("source revision must be a Git commit hash, not free text")
    public = {
        "version": "matryoshka-cutoff-v1", "run_id": publication_id,
        "source_revision": revision, "status": source["status"],
        "protocol": public_protocol, "datasets": profiles, "rows": rows,
        "summary": summaries, "decisions": decisions,
        "agent_summary": [], "agent_gap_counts": gap_counts,
        "validation": {"actual_rows": len(rows), "publication": "aggregate-only; private scientific checks retained separately"},
        "publication_boundary": "No original identifiers, raw payloads, per-unit hashes or per-target records. Private evidence is required for independent membership replay.",
    }
    safe_evidence = summarize_evidence(public, focus_dimension=focus_dimension)
    for original, exported in zip(original_evidence["datasets"], safe_evidence["datasets"], strict=True):
        for key in ("native_mae", "short_mae", "curve", "budget_rows", "fallback_shares",
                    "agents_compared", "agents_with_higher_mae"):
            if original[key] != exported[key]:
                raise AssertionError(f"publication changed the numerical {key} evidence")
    return public


def snapshot_summary(readiness: Mapping[str, Any], public: Mapping[str, Any]) -> dict[str, Any]:
    profiles = [row for row in public["datasets"] if row["dataset_id"] == "cosmos_otel" and row["status"] == "completed"]
    if len(profiles) != 1:
        raise ValueError("snapshot provenance requires exactly one completed Cosmos cohort")
    cosmos = profiles[0]
    if readiness["eligible_units"] != cosmos["n"] or readiness["agents"] != cosmos["agents"]:
        raise ValueError("snapshot population does not match the measured Cosmos cohort")
    if readiness["outcome_counts"]["good"] != cosmos["positive_count"]:
        raise ValueError("snapshot expected-label reference does not match the measured cohort")
    containers = {row["container"]: row for row in readiness["containers"]}
    for name in ("labels", "spans"):
        if containers[name]["sha256"] != cosmos["source_hashes"][name]:
            raise ValueError("snapshot checksum differs from the measured source")
    if readiness["representation_sources"] != cosmos["provenance"]["representation_sources"]:
        raise ValueError("snapshot representation counts differ from measured input")
    try:
        cutoff = datetime.fromisoformat(readiness["snapshot_cutoff_utc"])
    except (ValueError, TypeError):
        raise ValueError("invalid snapshot cutoff; source value withheld") from None
    if cutoff.tzinfo is None:
        raise ValueError("snapshot cutoff requires an explicit timezone")
    if not isinstance(readiness["point_in_time_snapshot"], bool):
        raise ValueError("snapshot atomicity flag must be explicit")
    preparation = _numeric_fields(cosmos["provenance"], (
        "reused_vectors", "requested_vectors", "logical_requests", "http_attempts",
        "http_successes", "api_input_tokens",
    ))
    if "reused_vectors" in preparation and "requested_vectors" in preparation:
        unique_packets = cosmos["provenance"].get("unique_packets")
        if isinstance(unique_packets, bool) or not isinstance(unique_packets, int) or not 1 <= unique_packets <= cosmos["n"]:
            raise ValueError("embedding preparation requires a valid unique-packet count")
        if preparation["reused_vectors"] + preparation["requested_vectors"] != unique_packets:
            raise ValueError("cached and newly requested vectors do not cover the unique canonical packets")
    return {
        "catchup_cutoff_utc": cutoff.astimezone(timezone.utc).isoformat(),
        "point_in_time_snapshot": readiness["point_in_time_snapshot"],
        "snapshot_documents": _number(readiness["snapshot_documents"]),
        "snapshot_bytes": _number(readiness["snapshot_bytes"]),
        "eligible_units": cosmos["n"], "agents": cosmos["agents"],
        "positive_count": cosmos["positive_count"],
        "representation_sources": cosmos["provenance"]["representation_sources"],
        "source_file_hashes": cosmos["source_hashes"],
        "embedding_preparation": preparation,
        "boundary": "Static local export with a catch-up cutoff, not a live feed or an atomic database snapshot. Only the original labels-plus-spans experiment adapter is used.",
    }


def write_public_report(
    private_run: Path, output: Path, *, focus_dimension: int = 2,
    snapshot_readiness: Path | None = None,
) -> Path:
    private_run, output = private_run.resolve(), output.resolve()
    if output == private_run or output.is_relative_to(private_run):
        raise ValueError("publish to a separate directory, not inside protected evidence")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh aggregate-publication directory")
    source_path, manifest_path = private_run / "aggregate.json", private_run / "manifest.json"
    source_hash, manifest_hash = sha256_file(source_path), sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["files"]["aggregate"]["sha256"] != source_hash:
        raise ValueError("private aggregate does not match its manifest")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    public = sanitized_aggregate(source, focus_dimension=focus_dimension, publication_id=output.name)
    snapshot = None
    notice = PRIVACY_NOTICE
    if snapshot_readiness is not None:
        snapshot = snapshot_summary(json.loads(snapshot_readiness.read_text(encoding="utf-8")), public)
        notice += (
            '<p style="padding:1rem;background:#edf7f4"><strong>Refreshed Cosmos source:</strong> '
            f'{snapshot["eligible_units"]:,} eligible expected-label units; catch-up cutoff '
            f'{snapshot["catchup_cutoff_utc"]}. The complete export has '
            f'{snapshot["snapshot_documents"]:,} documents, not that many evaluated sessions. '
            "Other label containers are not mixed into this comparison. Differences from earlier "
            "reports can reflect changes in the cohort and label prevalence, not only the method.</p>"
        )
        preparation = snapshot["embedding_preparation"]
        if "reused_vectors" in preparation and "requested_vectors" in preparation:
            notice += (
                '<p style="padding:1rem;background:#edf7f4"><strong>Embedding preparation:</strong> '
                f'{preparation["reused_vectors"]:,} exact cached vectors reused and '
                f'{preparation["requested_vectors"]:,} new native vectors generated. '
                "This was a one-time preparation step; all repeated dimension sweeps were offline.</p>"
            )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "aggregate.json", public)
    html, summary = build_report(
        public, aggregate_path="aggregate.json (sanitized publication)",
        aggregate_hash=sha256_file(output / "aggregate.json"), detailed_url="detail/report.html",
        focus_dimension=focus_dimension,
        generation_command="Run scripts\\build_matryoshka_public_report.py with the protected run and a fresh output directory; see manifest.json for source hashes.",
    )
    (output / "report.html").write_text(html.replace("<main>", "<main>" + notice, 1), encoding="utf-8")
    write_json(output / "summary.json", summary)
    detail = output / "detail" / "report.html"
    detail.parent.mkdir()
    detail_html = build_detailed_report(
        public, "aggregate.json (sanitized publication)", include_input_readiness=False,
    )
    detail.write_text(re.sub(r"(<main\b[^>]*>)", r"\1" + notice, detail_html, count=1), encoding="utf-8")
    files = [output / "aggregate.json", output / "summary.json", output / "report.html", detail]
    if snapshot is not None:
        write_json(output / "snapshot_provenance.json", snapshot)
        files.append(output / "snapshot_provenance.json")
    write_json(output / "manifest.json", {
        "version": VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_private_aggregate_sha256": source_hash, "source_private_manifest_sha256": manifest_hash,
        "source_revision": source["source_revision"], "focus_dimension": focus_dimension,
        "boundary": public["publication_boundary"],
        "generator_hashes": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("matryoshka_public_report.py", "matryoshka_summary_report.py", "matryoshka_report.py")
        },
        "files": {str(path.relative_to(output)): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files},
    })
    if sha256_file(source_path) != source_hash or sha256_file(manifest_path) != manifest_hash:
        raise RuntimeError("protected source artifacts changed during publication")
    return output / "report.html"
