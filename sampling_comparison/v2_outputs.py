"""Production-shaped output artifacts for sampling v2.

This module emits ExternalEvalSnapshot-style payloads grouped by
tenant/agent/day/method and provides strict local contract validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_DATE_FMT = "%Y-%m-%d"
_ECHO_LIMIT = 5


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_canonical_json(dict(payload)) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + "\n")
    os.replace(tmp, path)


def _to_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso_z(value: datetime) -> str:
    return _to_utc(value).isoformat().replace("+00:00", "Z")


def _utc_day(value: datetime) -> str:
    return _to_utc(value).strftime(_DATE_FMT)


def _stable_short_hash(parts: Sequence[str], length: int = 12) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


def _validate_iso_utc_z(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field_name} must be UTC ISO-8601 string ending in Z")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} is not valid ISO-8601: {value}") from exc


def _validate_yyyy_mm_dd(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be YYYY-MM-DD string")
    try:
        datetime.strptime(value, _DATE_FMT)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD: {value}") from exc


def _ensure_metric_contract(metric: Mapping[str, Any]) -> None:
    if not isinstance(metric.get("name"), str) or not str(metric.get("name")).strip():
        raise ValueError("metric.name is required")
    has_score = "score" in metric and metric.get("score") is not None
    has_categories = "categories" in metric and metric.get("categories") is not None
    if not has_score and not has_categories:
        raise ValueError("metric must include either score or categories")
    if has_categories:
        categories = metric.get("categories")
        if not isinstance(categories, list) or not categories:
            raise ValueError("metric.categories must be non-empty when supplied")
        for category in categories:
            if not isinstance(category, Mapping):
                raise ValueError("metric.categories entries must be objects")
            if not isinstance(category.get("name"), str) or not str(category.get("name")).strip():
                raise ValueError("metric category name is required")
            options = category.get("options")
            if options is not None:
                if not isinstance(options, list) or not options:
                    raise ValueError("metric category options must be non-empty when supplied")
                for option in options:
                    if not isinstance(option, Mapping):
                        raise ValueError("metric category option entries must be objects")
                    if not isinstance(option.get("name"), str) or not str(option.get("name")).strip():
                        raise ValueError("metric category option name is required")


@dataclass(frozen=True)
class SnapshotBuildConfig:
    run_time_utc: datetime | None = None
    echo_limit: int = _ECHO_LIMIT


def build_external_eval_snapshots(
    *,
    data: Any,
    representative_membership: Mapping[str, Mapping[str, Any]],
    version: str,
    seed: int,
    run_time_utc: datetime | None = None,
    echo_limit: int = _ECHO_LIMIT,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build per-method ExternalEvalSnapshot payloads and provenance metadata.

    Args:
        data: CombinedDataset-like object from v2_experiment.
        representative_membership: membership object from _representative_20pct_membership.
        version: version marker used in deterministic runId material.
        seed: seed marker used in deterministic runId material.
        run_time_utc: optional deterministic completion time override.
        echo_limit: maximum number of echoed sampled results per snapshot payload.
    """

    if echo_limit <= 0:
        raise ValueError("echo_limit must be > 0")

    methods = representative_membership.get("methods")
    if not isinstance(methods, Mapping) or not methods:
        raise ValueError("representative_membership.methods must be a non-empty object")

    unit_by_id = {str(unit.unit_id or ""): unit for unit in data.units}

    eligible_by_key: dict[tuple[str, str, str], list[str]] = {}
    for unit_id in data.unit_ids:
        unit = unit_by_id[unit_id]
        stamp = _to_utc(unit.ended_at or unit.started_at)
        if stamp is None:
            continue
        key = (str(unit.tenant_id), str(unit.agent_id), _utc_day(stamp))
        eligible_by_key.setdefault(key, []).append(unit_id)

    snapshots_by_method: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {
        "version": "sampling-v2-external-snapshots-manifest-v1",
        "grouping": "method x tenant x agent x utc_day",
        "route_template": "POST /evals/service/results?api-version=1",
        "tenant_from_route_or_auth": True,
        "tenant_id_omitted_in_body": True,
        "echo_limit": int(echo_limit),
        "not_posted": True,
        "evaluator": {
            "metric": "task_completion",
            "model": "dataset-expected-label",
            "score_scale": "binary",
            "reasoning": "omitted",
        },
        "methods": {},
    }

    for method in sorted(methods):
        method_entry = methods[method]
        selected_ids = tuple(sorted(str(x) for x in (method_entry.get("selected_ids") or [])))
        selected_set = set(selected_ids)

        selected_by_group: dict[tuple[str, str, str], list[str]] = {}
        for unit_id in selected_ids:
            unit = unit_by_id.get(unit_id)
            if unit is None:
                continue
            stamp = _to_utc(unit.ended_at or unit.started_at)
            if stamp is None:
                continue
            key = (str(unit.tenant_id), str(unit.agent_id), _utc_day(stamp))
            selected_by_group.setdefault(key, []).append(unit_id)

        payloads: list[dict[str, Any]] = []
        method_meta: dict[str, Any] = {
            "declared_budget": "100%" if method == "census" else "20% cap",
            "group_count": 0,
            "snapshot_count": 0,
            "selected_total": len(selected_ids),
            "sha256": "",
            "groups": [],
        }

        for key in sorted(eligible_by_key):
            tenant_id, agent_id, day = key
            eligible_ids = tuple(sorted(eligible_by_key[key]))
            selected_group_ids = tuple(sorted(selected_by_group.get(key, [])))
            if not selected_group_ids:
                continue

            labels = [1 if bool(data.labels_by_unit[unit_id]) else 0 for unit_id in selected_group_ids]
            avg_score = float(sum(labels) / len(labels))

            echo_ids = sorted(
                selected_group_ids,
                key=lambda uid: (
                    _iso_z(_to_utc(unit_by_id[uid].ended_at or unit_by_id[uid].started_at)),
                    str(unit_by_id[uid].conversation_id or ""),
                    uid,
                ),
            )[:echo_limit]

            max_end = max(
                _to_utc(unit_by_id[uid].ended_at or unit_by_id[uid].started_at)
                for uid in selected_group_ids
            )
            completed_at = _to_utc(run_time_utc) if run_time_utc is not None else max_end
            created_at = completed_at

            run_id = (
                "v2"
                f"-{method}"
                f"-{day}"
                f"-{_stable_short_hash([version, str(seed), method, day, agent_id], length=16)}"
            )

            results: list[dict[str, Any]] = []
            for uid in echo_ids:
                unit = unit_by_id[uid]
                conv_id = str(unit.conversation_id or "").strip()
                if not conv_id:
                    conv_ids = tuple(str(x).strip() for x in (unit.conversation_ids or ()) if str(x).strip())
                    conv_id = conv_ids[0] if conv_ids else ""
                if not conv_id:
                    raise ValueError(f"Missing conversation id for selected unit_id={uid}")

                end_stamp = _to_utc(unit.ended_at or unit.started_at)
                if end_stamp is None:
                    raise ValueError(f"Missing ended_at/started_at for selected unit_id={uid}")

                metric_score = 1 if bool(data.labels_by_unit[uid]) else 0
                metric_row: dict[str, Any] = {
                    "name": "task_completion",
                    "displayName": "Task Completion",
                    "model": "dataset-expected-label",
                    "scoreScale": "binary",
                    "score": metric_score,
                    "passed": bool(metric_score >= 1),
                    "threshold": 1,
                    "scoreMin": 0,
                    "scoreMax": 1,
                }
                result_row: dict[str, Any] = {
                    "conversationId": conv_id,
                    "conversationEndTimeUtc": _iso_z(end_stamp),
                    "metrics": [metric_row],
                }
                result_row["interactionTimestampUtc"] = _iso_z(end_stamp)
                results.append(result_row)

            payload = {
                "runId": run_id,
                "agentId": agent_id,
                "date": day,
                "granularity": "day",
                "createdAtUtc": _iso_z(created_at),
                "completedAtUtc": _iso_z(completed_at),
                "totalConversationsCount": len(eligible_ids),
                "totalSampledCount": len(selected_group_ids),
                "avgScore": avg_score,
                "results": results,
            }

            validate_external_eval_snapshot(
                payload,
                expected={
                    "expected_total_sampled_count": len(selected_group_ids),
                    "expected_avg_score": avg_score,
                },
            )

            payloads.append(payload)
            method_meta["groups"].append(
                {
                    "tenantId": tenant_id,
                    "agentId": agent_id,
                    "date": day,
                    "eligible_count": len(eligible_ids),
                    "selected_count": len(selected_group_ids),
                    "echoed_results_count": len(results),
                    "avg_score": avg_score,
                    "selected_labels": labels,
                    "selected_unit_ids": list(selected_group_ids),
                    "echoed_unit_ids": list(echo_ids),
                }
            )

        method_meta["group_count"] = len(method_meta["groups"])
        method_meta["snapshot_count"] = len(payloads)
        snapshots_by_method[method] = payloads
        method_meta["sha256"] = _sha256_text("\n".join(_canonical_json(row) for row in payloads) + "\n")
        provenance["methods"][method] = method_meta

    return snapshots_by_method, provenance


def validate_external_eval_snapshot(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> None:
    """Strict local validator for ExternalEvalSnapshot contract constraints."""

    required_top = (
        "runId",
        "agentId",
        "date",
        "granularity",
        "createdAtUtc",
        "completedAtUtc",
        "totalConversationsCount",
        "totalSampledCount",
        "avgScore",
        "results",
    )
    for field in required_top:
        if field not in payload:
            raise ValueError(f"missing required top-level field: {field}")

    if not isinstance(payload["runId"], str) or not payload["runId"].strip():
        raise ValueError("runId is required")
    if not isinstance(payload["agentId"], str) or not payload["agentId"].strip():
        raise ValueError("agentId is required")
    _validate_yyyy_mm_dd(str(payload["date"]), "date")

    if str(payload["granularity"]) != "day":
        raise ValueError("granularity must be 'day'")

    _validate_iso_utc_z(str(payload["createdAtUtc"]), "createdAtUtc")
    _validate_iso_utc_z(str(payload["completedAtUtc"]), "completedAtUtc")

    total_conversations_count = int(payload["totalConversationsCount"])
    total_sampled_count = int(payload["totalSampledCount"])
    if total_conversations_count < 0:
        raise ValueError("totalConversationsCount must be >= 0")
    if total_sampled_count <= 0:
        raise ValueError("totalSampledCount must be > 0")

    results = payload["results"]
    if not isinstance(results, list) or not results:
        raise ValueError("results must be non-empty")

    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("each result entry must be an object")
        if not isinstance(result.get("conversationId"), str) or not str(result.get("conversationId")).strip():
            raise ValueError("result.conversationId is required")
        _validate_iso_utc_z(str(result.get("conversationEndTimeUtc")), "result.conversationEndTimeUtc")

        metrics = result.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            raise ValueError("result.metrics must be non-empty")
        for metric in metrics:
            if not isinstance(metric, Mapping):
                raise ValueError("each metric entry must be an object")
            _ensure_metric_contract(metric)

    avg_score = float(payload["avgScore"])
    if avg_score < 0.0 or avg_score > 1.0:
        raise ValueError("avgScore must be in [0, 1]")

    if expected is not None:
        if "expected_total_sampled_count" in expected:
            if int(expected["expected_total_sampled_count"]) != total_sampled_count:
                raise ValueError("totalSampledCount does not match provenance metadata")
        if "expected_avg_score" in expected:
            expected_avg_score = float(expected["expected_avg_score"])
            if abs(avg_score - expected_avg_score) > 1e-12:
                raise ValueError("avgScore does not match provenance metadata")


def write_external_eval_snapshot_artifacts(
    *,
    output_dir: Path,
    snapshots_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    provenance_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Write per-method JSONL snapshots plus manifest and return path/hash metadata."""

    snapshots_dir = output_dir / "external_eval_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    methods_meta: dict[str, Any] = {}
    for method in sorted(snapshots_by_method):
        rows = [dict(row) for row in snapshots_by_method[method]]
        path = snapshots_dir / f"{method}.jsonl"
        _write_jsonl_atomic(path, rows)
        methods_meta[method] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "line_count": len(rows),
        }

    manifest_path = snapshots_dir / "manifest.json"
    full_manifest = dict(provenance_manifest)
    full_manifest["methods_files"] = methods_meta
    _write_json_atomic(manifest_path, full_manifest)

    return {
        "dir": str(snapshots_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "methods": methods_meta,
    }


def build_production_storage_manifest(*, generated_at_utc: datetime | None = None) -> dict[str, Any]:
    """Architecture-only storage model guidance for production PPAPI-compatible ingestion."""

    stamp = _to_utc(generated_at_utc) or datetime.now(timezone.utc)
    return {
        "version": "sampling-v2-production-storage-manifest-v1",
        "generatedAtUtc": _iso_z(stamp),
        "scope": {
            "type": "architecture-artifact",
            "implemented": False,
            "resource_names": "proposed logical model only",
        },
        "authoritative_state": {
            "source": "ESP/Cosmos",
            "schema_versioning": "required",
            "concurrency": {
                "etag": "required",
                "transactional_batch": "required for same-partition consistency",
            },
            "sampler_state_privacy": "no raw session copy in sampler state",
        },
        "proposed_logical_model": {
            "containers": [
                {
                    "name": "evaluationRuns",
                    "partitionKey": "/tenantId/agentId/date",
                    "query_paths": ["tenantId", "agentId", "runId", "date"],
                },
                {
                    "name": "selectionMembership",
                    "partitionKey": "/tenantId/agentId/runId",
                    "query_paths": ["tenantId", "agentId", "runId", "method"],
                },
                {
                    "name": "evaluationFacts",
                    "partitionKey": "/tenantId/agentId/date",
                    "query_paths": ["tenantId", "agentId", "date", "conversationId"],
                    "outbox": "same logical model for reliable downstream fanout",
                },
                {
                    "name": "similarityState",
                    "partitionKey": "/tenantId/agentId/profileId",
                    "query_paths": ["tenantId", "agentId", "profileId", "expiresAt"],
                },
            ],
            "sampling_state_fields": {
                "census": ["membership_state", "population_counts"],
                "random": [
                    "seed",
                    "policy",
                    "stratum_N_h",
                    "stratum_n_h",
                    "inclusion_probability",
                    "weight",
                ],
                "lsh": ["profile", "signatures", "band_buckets", "lastSeen", "hits"],
                "embedding": [
                    "leader_metadata_authoritative_in_cosmos",
                    "derivative_vector_index_in_azure_ai_search",
                    "no_raw_text",
                    "query_vector_discarded",
                ],
            },
        },
        "ppapi_contract_requirements": {
            "route": "POST /evals/service/results?api-version=1",
            "tenant_handling": "tenant derived from route/auth, not request body",
            "body_tenant_id": "optional/ignored and omitted by producer",
            "auth": "service-to-service auth required",
            "discovery": "environment/service discovery required",
            "forbidden": ["embedded token", "tenant-specific host in artifact payload"],
        },
        "azure_ai_search_assessment": {
            "suitability": "technically suitable with vector index",
            "inspection": {
                "date": "2026-07-30",
                "service": "stangoodwin-ai-search",
                "index": "maven-session-sampling-v1",
                "vector_field_present": False,
            },
            "local_experiment": {
                "vector_store": "in-memory exact vector store",
                "embeddings": "deterministic offline",
                "live_search_experiment": False,
            },
            "production_requirements": {
                "index": "separate vector-enabled index",
                "hnsw": "required",
                "distance": "cosine",
                "filterable_fields": ["tenantId", "agentId", "profileId", "expiresAt"],
                "security": ["managed identity", "RBAC", "private endpoint", "CMK as required"],
                "lifecycle": ["prefilter before vector recall", "deletion worker required (no TTL)"],
            },
        },
    }


def write_production_storage_manifest(*, output_dir: Path) -> dict[str, Any]:
    payload = build_production_storage_manifest()
    path = output_dir / "production_storage_manifest.json"
    _write_json_atomic(path, payload)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
    }
