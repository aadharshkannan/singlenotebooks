from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import median
from typing import Any, Mapping

from sampling_comparison.idw_threshold_experiment import DATASETS, DIMENSIONS, LIPSCHITZ_CONFIG, RATES, SCHEDULES, SEEDS
from sampling_comparison.idw_threshold_report import _headline, build_report
from sampling_comparison.matryoshka_experiment import canonical, sha256_file, write_json
from sampling_comparison.matryoshka_public_report import LABEL_SOURCES, _digest, _number, _numeric_fields
from sampling_comparison.v4_idw import IDWConfig


METRICS = ("n", "positive_count", "predicted_positive_count", "accuracy", "precision", "recall", "f1", "tp", "fp", "fn", "tn")
REASONS = (None, "empty_population", "no_positive_labels", "no_negative_labels")
PREPARATION = ("reused_vectors", "requested_vectors", "logical_requests", "http_attempts", "http_successes", "api_input_tokens", "live_embedding_calls")
VALIDATION_COUNTS = (
    "actual_cells", "expected_cells", "target_occurrences", "eligible_target_occurrences",
    "fallback_target_occurrences", "baseline_max_mae_absolute_error",
    "baseline_max_accuracy_absolute_error", "baseline_accuracy_mismatches",
)


def _reason(value: Any) -> str | None:
    if value not in REASONS:
        raise ValueError("unrecognized metric unavailability reason; source text withheld")
    return value


def _metrics(source: Mapping[str, Any], *, threshold: bool = False) -> dict[str, Any]:
    return {
        **_numeric_fields(source, (*METRICS, "threshold") if threshold else METRICS),
        "reason": _reason(source.get("reason")),
    }


def _scope(row: Mapping[str, Any]) -> dict[str, Any]:
    if row["dataset_id"] not in DATASETS or row["dimension"] not in DIMENSIONS or row["rate"] not in RATES:
        raise ValueError("unexpected experiment scope")
    return {key: row[key] for key in ("dataset_id", "dimension", "rate")}


def _agent_gaps(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    keyed = {
        (row["dataset_id"], row["dimension"], row["agent_id"], row["method"]): row
        for row in source["agent_summary"]
    }
    result = []
    for dataset in DATASETS:
        for dimension in DIMENSIONS:
            differences = []
            eligible = 0
            for (cohort, size, agent, method), point in keyed.items():
                if (cohort, size, method) != (dataset, dimension, "paired_point") or not point["n"]:
                    continue
                lower = keyed[(cohort, size, agent, "paired_lower")]
                if lower["n"] != point["n"]:
                    raise ValueError("paired agent denominators differ")
                eligible += 1
                if point["recall"] is not None and lower["recall"] is not None:
                    differences.append(_number(lower["recall"]) - _number(point["recall"]))
            result.append({
                "dataset_id": dataset, "dimension": dimension, "eligible_agents": eligible,
                "agents_with_defined_recall": len(differences),
                "agents_with_recall_loss": sum(value < 0 for value in differences),
                "median_recall_delta": median(differences) if differences else None,
                "worst_recall_delta": min(differences) if differences else None,
            })
    return result


def sanitized_aggregate(source: Mapping[str, Any], publication_id: str) -> dict[str, Any]:
    if re.fullmatch(r"[a-z0-9-]+", publication_id) is None:
        raise ValueError("publication ID must be an operational slug")
    protocol = source["protocol"]
    expected = {
        "dimensions": list(DIMENSIONS), "seeds": list(SEEDS), "schedules": list(SCHEDULES),
        "rates": list(RATES), "modes": ["end_to_end"], "idw": asdict(IDWConfig()),
        "lipschitz": asdict(LIPSCHITZ_CONFIG), "angular_units": "arccos(cosine)/pi",
        "budget_rule": "max(1,floor(N*rate))", "threshold_rule": "score >= threshold",
        "canonical_runtime_required": True, "network_calls": 0, "embedding_calls": 0, "judge_calls": 0,
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("unexpected private experiment protocol")
    if source["status"] != "complete" or len(source["rows"]) != 4500 or source["validation"]["actual_cells"] != 4500:
        raise ValueError("private experiment grid is incomplete")
    if source["validation"]["threshold_dominance"] != "passed_all_cells":
        raise ValueError("threshold dominance did not pass")
    if any(source["validation"][key] != 0 for key in ("baseline_max_mae_absolute_error", "baseline_max_accuracy_absolute_error", "baseline_accuracy_mismatches")):
        raise ValueError("canonical baseline parity is not exact")
    if source["validation"]["cache_snapshot_before"] != source["validation"]["cache_snapshot_after"]:
        raise ValueError("protected cache bytes changed during the run")
    rows = []
    keys = set()
    for row in source["rows"]:
        scope = _scope(row)
        if row["seed"] not in SEEDS or row["schedule"] not in SCHEDULES:
            raise ValueError("unexpected replay setting")
        key = (*scope.values(), row["seed"], row["schedule"])
        if key in keys:
            raise ValueError("duplicate replay setting")
        keys.add(key)
        safe = {
            **scope, "seed": row["seed"], "schedule": row["schedule"],
            **_numeric_fields(row, ("cell_id", "selected_count", "unjudged_count", "eligible_count",
                                   "envelope_missing_count", "calibration_fallback_count",
                                   "exact_contradiction_target_count", "point_auc", "lower_auc")),
            "point_auc_reason": _reason(row["point_auc_reason"]),
            "lower_auc_reason": _reason(row["lower_auc_reason"]),
            "provenance_counts": {
                name: _number(count) for name, count in row["provenance_counts"].items()
                if name in ("idw", "exact_match", "global_mean", "prior")
            },
        }
        for name in ("all_unjudged_point_at_0_5", "paired_point_at_0_5", "paired_lower_at_0_5"):
            safe[name] = _metrics(row[name])
        rows.append(safe)
    selectors = []
    for row in source["selectors"]:
        scope = _scope(row)
        safe = {
            **scope, "selector_id": f'{scope["dataset_id"]}|{scope["dimension"]}|{scope["rate"]:g}',
            **_numeric_fields(row, ("cell_count", "pooled_unjudged_n", "pooled_eligible_n", "point_auc", "lower_auc")),
            "point_auc_reason": _reason(row["point_auc_reason"]), "lower_auc_reason": _reason(row["lower_auc_reason"]),
            "all_unjudged_point_at_0_5": _metrics(row["all_unjudged_point_at_0_5"]),
        }
        for method in ("point", "lower"):
            safe[method + "_grid"] = [_metrics(point, threshold=True) for point in row[method + "_grid"]]
            curve = row[method + "_roc_display"]
            safe[method + "_roc_display"] = {
                **_numeric_fields(curve, ("exact_point_count", "display_point_count")),
                "fpr": [_number(value) for value in curve["fpr"]],
                "tpr": [_number(value) for value in curve["tpr"]],
            }
        selectors.append(safe)
    equal_cell = []
    for row in source["equal_cell_summary"]:
        safe = {
            **_scope(row), **_numeric_fields(row, ("cell_count", "eligible_share", "point_auc_mean",
                                                  "lower_auc_mean", "point_auc_available_cells", "lower_auc_available_cells")),
        }
        for name in ("point_auc_seed_t95", "lower_auc_seed_t95"):
            safe[name] = [_number(value) for value in row[name]] if row[name] is not None else None
        equal_cell.append(safe)
    profiles = []
    for row in source["datasets"]:
        name = row["dataset_id"]
        if name not in DATASETS:
            raise ValueError("unknown dataset profile")
        safe = {
            "dataset_id": name, **_numeric_fields(row, ("sessions", "agents", "positive_count", "pass_rate", "truncated_sessions", "excluded_sessions")),
            "label_source": LABEL_SOURCES[name],
            "representation_sources": {
                key: _number(row["representation_sources"][key])
                for key in ("combined_normalized", "linked_raw_spans", "synthetic_label_document")
                if key in row["representation_sources"]
            },
            "source_hashes": {
                key: _digest(row["source_hashes"][key])
                for key in ("labels", "spans", "historical_300", "dense_2500") if key in row.get("source_hashes", {})
            },
            "embedding_preparation": _numeric_fields(row.get("embedding_preparation", {}), PREPARATION),
        }
        if row.get("snapshot_cutoff_utc"):
            try:
                cutoff = datetime.fromisoformat(row["snapshot_cutoff_utc"])
            except (TypeError, ValueError):
                raise ValueError("invalid snapshot cutoff; content withheld") from None
            if cutoff.tzinfo is None:
                raise ValueError("snapshot cutoff must identify its timezone")
            safe["snapshot_cutoff_utc"] = cutoff.astimezone(timezone.utc).isoformat()
        profiles.append(safe)
    revision = source["source_revision"]
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("invalid source revision")
    if any(source["costs"][key] != 0 for key in ("network_calls", "embedding_calls", "judge_calls", "external_cost_usd")):
        raise ValueError("replay was not offline")
    scientific = source.get("score_consistency_validation", {})
    if not scientific.get("ok"):
        raise ValueError("source-to-score validation must pass before publication")
    result = {
        "version": "idw-threshold-envelope-v1", "status": "complete", "run_id": publication_id,
        "source_revision": revision, "protocol": expected, "rows": rows, "selectors": selectors,
        "equal_cell_summary": equal_cell, "datasets": profiles,
        "agent_summary": [], "agent_gap_summary": _agent_gaps(source),
        "publication": {"aggregate_only": True, "private_evidence": "Protected locally; not uploaded or included in this publication."},
        "validation": {
            **_numeric_fields(source["validation"], VALIDATION_COUNTS),
            "threshold_dominance": "passed_all_cells", "cache_source_preservation": "passed",
        },
        "score_consistency_validation": {
            "ok": True, **_numeric_fields(scientific, (
                "canonical_cells_checked", "stratified_source_to_score_replay_cells",
                "exact_roc_method_selector_arrays_checked", "valid_point_lower_pairs",
                "missing_envelope_fallback_targets", "empirical_l_targets", "configured_l_fallback_targets",
                "targets_with_exact_match_contradictions", "membership_records_verified",
            )),
        },
        "preregistration": {"acceptance": {
            "canonical_runtime_mae_absolute_tolerance": 0.0,
            "canonical_runtime_accuracy_absolute_tolerance": 0.0,
        }},
        "baseline": {
            "aggregate": "Protected local matched-input reference",
            "aggregate_sha256": _digest(source["baseline"]["aggregate_sha256"]),
            "memberships": "Protected local reference memberships",
            "memberships_sha256": _digest(source["baseline"]["memberships_sha256"]),
        },
        "environment": {"python": "Python 3.13.13 x64; NumPy 2.5.3; OpenBLAS 12 threads", "platform": "Windows"},
        "costs": _numeric_fields(source["costs"], ("wall_seconds", "cpu_seconds", "network_calls", "embedding_calls", "judge_calls", "external_cost_usd")),
        "code_hashes": {},
        "files": {
            "target_evidence": "Protected local target evidence; not published",
            "exact_roc_curves": "Protected local exact ROC arrays; not published",
            "preregistration": "Exact canonical acceptance recorded in the protected run",
            "score_consistency_validation": "validation.json (aggregate-only verification summary)",
        },
    }
    for filename in ("idw_threshold_experiment.py", "matryoshka_experiment.py", "run_idw_threshold_experiment.py", "lipschitz.py"):
        matches = [value for path, value in source["code_hashes"].items() if Path(path.replace("\\", "/")).name == filename]
        if len(matches) != 1:
            raise ValueError("missing or ambiguous controlling source hash")
        result["code_hashes"][filename] = _digest(matches[0])
    if _headline(result) != _headline(source):
        raise AssertionError("sanitization changed report headline values")
    return result


def write_public_report(private_run: Path, output: Path) -> Path:
    private_run, output = private_run.resolve(), output.resolve()
    if output == private_run or output.is_relative_to(private_run):
        raise ValueError("publish outside the protected run")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh public output directory")
    manifest_path, source_path = private_run / "manifest.json", private_run / "aggregate.json"
    manifest_hash, source_hash = sha256_file(manifest_path), sha256_file(source_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_hash != manifest["files"]["aggregate"]["sha256"]:
        raise ValueError("protected aggregate hash does not match its manifest")
    for record in manifest["files"].values():
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError("protected artifact hash mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    public = sanitized_aggregate(source, output.name)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "aggregate.json", public)
    write_json(output / "validation.json", public["score_consistency_validation"])
    command = (
        "Use Python 3.13.13 x64 with all four BLAS/OMP/MKL/NumExpr thread settings = 12.\n"
        "Run scripts\\run_idw_threshold_experiment.py with --baseline pointing to the matching protected\n"
        "native/8d reference and --input cosmos_otel=<refreshed-manifest>, using a fresh private --output.\n"
        "Run scripts\\validate_idw_threshold_experiment.py against that private aggregate.\n"
        "Publish using scripts\\build_idw_threshold_public_report.py --private-run <protected-run> --output <fresh-public-report>.\n"
        "See docs\\IDW_THRESHOLD_EXPERIMENT.md for the concrete protected paths and snapshot boundary."
    )
    html = build_report(public, "aggregate.json", reproduction_command=command)
    (output / "report.html").write_text(html, encoding="utf-8")
    write_json(output / "manifest.json", {
        "version": "idw-threshold-aggregate-publication-v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_private_aggregate_sha256": source_hash, "source_private_manifest_sha256": manifest_hash,
        "source_revision": source["source_revision"], "boundary": public["publication"]["private_evidence"],
        "generator_hashes": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("idw_threshold_public_report.py", "idw_threshold_report.py", "matryoshka_public_report.py")
        },
        "files": {
            name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
            for name in ("aggregate.json", "validation.json", "report.html")
        },
    })
    if sha256_file(manifest_path) != manifest_hash or sha256_file(source_path) != source_hash:
        raise RuntimeError("private evidence changed during publication")
    return output / "report.html"
