"""Labeled synthetic Agent 365 dataset loading for offline validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

from .agent365_otel import NormalizationResult, normalize_agent365_records
from .models import EvaluationWindow


_LABEL_INT_KEY = "evaluation.expected"
_LABEL_TEXT_KEY = "evaluation.expected.label"
_CASE_ID_KEY = "evaluation.source.case_id"
_CONVERSATION_ID_KEY = "gen_ai.conversation.id"


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _otlp_attr_map(attrs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(attrs, list):
        for row in attrs:
            if not isinstance(row, dict):
                continue
            key = row.get("key")
            if not isinstance(key, str) or not key:
                continue
            value = row.get("value")
            if isinstance(value, dict):
                for value_key in ("stringValue", "intValue", "boolValue", "doubleValue"):
                    if value_key in value:
                        out[key] = value.get(value_key)
                        break
                else:
                    out[key] = value
            else:
                out[key] = value
    elif isinstance(attrs, dict):
        out = dict(attrs)
    return out


def _coerce_label(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value) if value in (0, 1) else None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "pass", "passed", "completed"}:
            return True
        if token in {"0", "false", "fail", "failed", "not_completed", "not completed"}:
            return False
    return None


def _extract_labels_by_conversation(
    document: dict[str, Any],
) -> tuple[
    dict[tuple[str, str, str], bool],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, bool],
]:
    labels: dict[tuple[str, str, str], bool] = {}
    metadata: dict[tuple[str, str, str], dict[str, Any]] = {}
    by_conversation: dict[str, set[bool]] = {}

    resource_spans = document.get("resourceSpans")
    if not isinstance(resource_spans, list):
        raise ValueError("Expected strict OTLP top-level resourceSpans list")

    for resource_span in resource_spans:
        if not isinstance(resource_span, dict):
            continue
        resource_attrs = _otlp_attr_map(resource_span.get("resource", {}).get("attributes"))
        for scope_span in resource_span.get("scopeSpans") or []:
            if not isinstance(scope_span, dict):
                continue
            for span in scope_span.get("spans") or []:
                if not isinstance(span, dict) or "invoke_agent" not in str(span.get("name") or ""):
                    continue
                merged = {**resource_attrs, **_otlp_attr_map(span.get("attributes"))}
                conversation_id = merged.get(_CONVERSATION_ID_KEY)
                if not isinstance(conversation_id, str) or not conversation_id.strip():
                    continue
                conversation_id = conversation_id.strip()
                tenant_id = str(
                    merged.get("microsoft.tenant.id")
                    or merged.get("microsoft.agent365.tenant.id")
                    or merged.get("tenant.id")
                    or merged.get("TenantId")
                    or ""
                ).strip()
                agent_id = str(merged.get("gen_ai.agent.id") or merged.get("AgentId") or "").strip()
                if not tenant_id or not agent_id:
                    continue

                label_int = _coerce_label(merged.get(_LABEL_INT_KEY))
                label_text = _coerce_label(merged.get(_LABEL_TEXT_KEY))
                if label_int is not None and label_text is not None and label_int != label_text:
                    raise ValueError(f"Inconsistent labels for conversation {conversation_id}")
                label = label_int if label_int is not None else label_text
                if label is None:
                    continue

                scoped_key = (tenant_id, agent_id, conversation_id)
                if scoped_key in labels and labels[scoped_key] != label:
                    raise ValueError(
                        f"Conflicting labels for conversation {conversation_id} under {tenant_id}/{agent_id}"
                    )
                labels[scoped_key] = label
                metadata.setdefault(scoped_key, {}).update(
                    {
                        "case_id": merged.get(_CASE_ID_KEY),
                        "task": merged.get("evaluation.task") or merged.get("evaluation.task_name"),
                        "domain": merged.get("evaluation.domain"),
                        "difficulty": merged.get("evaluation.difficulty"),
                    }
                )
                by_conversation.setdefault(conversation_id, set()).add(label)

    unscoped: dict[str, bool] = {}
    for conversation_id, values in by_conversation.items():
        if len(values) == 1 and sum(1 for key in labels if key[2] == conversation_id) == 1:
            unscoped[conversation_id] = next(iter(values))
    return labels, metadata, unscoped


@dataclass(frozen=True)
class SyntheticDataset:
    document: dict[str, Any]
    normalization: NormalizationResult
    labels_by_unit: dict[str, bool]
    labels_by_conversation: dict[str, bool]
    labels_by_conversation_scoped: dict[tuple[str, str, str], bool]
    metadata_by_unit: dict[str, dict[str, Any]]
    evaluation_window: EvaluationWindow


def load_synthetic_a365_otel(path: str | Path) -> SyntheticDataset:
    source_path = Path(path)
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Expected top-level JSON object")
    if "resourceSpans" not in raw:
        raise ValueError("Expected strict OTLP payload with top-level resourceSpans")

    normalization = normalize_agent365_records([raw])
    scoped_labels, scoped_metadata, unscoped_labels = _extract_labels_by_conversation(raw)
    labels_by_unit: dict[str, bool] = {}
    metadata_by_unit: dict[str, dict[str, Any]] = {}
    min_started: datetime | None = None
    max_ended: datetime | None = None

    for unit in normalization.units:
        if unit.started_at is not None:
            min_started = unit.started_at if min_started is None else min(min_started, unit.started_at)
        if unit.ended_at is not None:
            max_ended = unit.ended_at if max_ended is None else max(max_ended, unit.ended_at)
        matched = {
            scoped_labels[(unit.tenant_id, unit.agent_id, conversation_id)]
            for conversation_id in unit.conversation_ids
            if (unit.tenant_id, unit.agent_id, conversation_id) in scoped_labels
        }
        if not matched:
            matched = {
                unscoped_labels[conversation_id]
                for conversation_id in unit.conversation_ids
                if conversation_id in unscoped_labels
            }
        if len(matched) != 1:
            raise ValueError(f"Unit {unit.unit_id} does not map to exactly one consistent label")
        unit_id = unit.unit_id or ""
        labels_by_unit[unit_id] = next(iter(matched))
        combined_meta: dict[str, Any] = {}
        for conversation_id in unit.conversation_ids:
            scoped_key = (unit.tenant_id, unit.agent_id, conversation_id)
            if scoped_key in scoped_metadata:
                combined_meta.update(scoped_metadata[scoped_key])
        metadata_by_unit[unit_id] = combined_meta

    if min_started is None:
        min_started = _parse_time(raw.get("startTime")) or datetime.now(timezone.utc)
    if max_ended is None:
        max_ended = _parse_time(raw.get("endTime")) or min_started
    max_ended = min_started + timedelta(microseconds=1) if max_ended <= min_started else max_ended + timedelta(microseconds=1)

    return SyntheticDataset(
        document=raw,
        normalization=normalization,
        labels_by_unit=labels_by_unit,
        labels_by_conversation=unscoped_labels,
        labels_by_conversation_scoped=scoped_labels,
        metadata_by_unit=metadata_by_unit,
        evaluation_window=EvaluationWindow(start_at=min_started, end_at=max_ended),
    )
