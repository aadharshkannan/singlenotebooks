"""Normalize Agent 365 telemetry envelopes into immutable evaluation units."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from .models import (
    EvaluationUnit,
    EvaluationWindow,
    IngestIssue,
    SessionizationPolicy,
    ToolCall,
    Turn,
)


@dataclass(frozen=True)
class NormalizationResult:
    units: tuple[EvaluationUnit, ...]
    issues: tuple[IngestIssue, ...]


def normalize_agent365_records(
    records: Iterable[Any],
    sessionization_policy: SessionizationPolicy = SessionizationPolicy(),
) -> NormalizationResult:
    spans: list[dict[str, Any]] = []
    issues: list[IngestIssue] = []

    for index, record in enumerate(records):
        record_spans, record_issues = _extract_spans(record, source_ref=f"record[{index}]")
        spans.extend(record_spans)
        issues.extend(record_issues)

    deduped_spans = _dedupe_spans(spans)
    units, unit_issues = _build_units(deduped_spans, sessionization_policy)
    issues.extend(unit_issues)
    return NormalizationResult(units=tuple(units), issues=tuple(issues))


def filter_sessions_to_window(
    result: NormalizationResult,
    window: EvaluationWindow,
) -> NormalizationResult:
    units: list[EvaluationUnit] = []
    issues = list(result.issues)
    for unit in result.units:
        if unit.ended_at is None:
            issues.append(
                IngestIssue(
                    code="missing_session_completion",
                    message="Session is missing completion timestamp and cannot be window-filtered",
                    source_kind="session",
                    source_ref=unit.unit_id,
                    details={
                        "tenant_id": unit.tenant_id,
                        "agent_id": unit.agent_id,
                        "session_id": unit.session_id,
                    },
                )
            )
            continue
        if window.contains_completion(unit.ended_at):
            units.append(unit)

    return NormalizationResult(units=tuple(units), issues=tuple(issues))


def _extract_spans(record: Any, source_ref: str) -> tuple[list[dict[str, Any]], list[IngestIssue]]:
    if isinstance(record, Mapping):
        if "traceRequest" in record:
            return _extract_from_otlp(record["traceRequest"], source_ref)
        if "resourceSpans" in record:
            return _extract_from_otlp(record, source_ref)
        if "documents" in record:
            return _extract_from_esp(record["documents"], source_ref)
        if "jsonContent" in record:
            return _extract_from_esp([record], source_ref)
        return [_normalize_flat_span(record)], []

    if isinstance(record, list):
        spans: list[dict[str, Any]] = []
        issues: list[IngestIssue] = []
        for item_index, item in enumerate(record):
            child_spans, child_issues = _extract_spans(item, source_ref=f"{source_ref}[{item_index}]")
            spans.extend(child_spans)
            issues.extend(child_issues)
        return spans, issues

    return [], [
        IngestIssue(
            code="unsupported_record",
            message="Record is neither mapping nor list",
            source_kind="input",
            source_ref=source_ref,
            details={"type": type(record).__name__},
        )
    ]


def _extract_from_otlp(trace_request: Any, source_ref: str) -> tuple[list[dict[str, Any]], list[IngestIssue]]:
    if not isinstance(trace_request, Mapping):
        return [], [
            IngestIssue(
                code="malformed_otlp",
                message="traceRequest must be a mapping",
                source_kind="otlp",
                source_ref=source_ref,
            )
        ]

    spans: list[dict[str, Any]] = []
    resource_spans = trace_request.get("resourceSpans") or []
    for r_index, resource_span in enumerate(resource_spans):
        if not isinstance(resource_span, Mapping):
            continue
        resource_attributes = _attributes_to_map(resource_span.get("resource", {}).get("attributes"))
        for s_index, scope_span in enumerate(resource_span.get("scopeSpans") or []):
            if not isinstance(scope_span, Mapping):
                continue
            for span_index, span in enumerate(scope_span.get("spans") or []):
                if not isinstance(span, Mapping):
                    continue
                span_attributes = _attributes_to_map(span.get("attributes"))
                merged = {**resource_attributes, **span_attributes}
                flat_span = _normalize_flat_span({**dict(span), **merged})
                flat_span["_source_kind"] = "otlp"
                flat_span["_source_ref"] = f"{source_ref}.resourceSpans[{r_index}].scopeSpans[{s_index}].spans[{span_index}]"
                spans.append(flat_span)
    return spans, []


def _extract_from_esp(documents: Any, source_ref: str) -> tuple[list[dict[str, Any]], list[IngestIssue]]:
    if not isinstance(documents, list):
        return [], [
            IngestIssue(
                code="malformed_esp",
                message="documents must be a list",
                source_kind="esp",
                source_ref=source_ref,
            )
        ]

    spans: list[dict[str, Any]] = []
    issues: list[IngestIssue] = []

    for doc_index, document in enumerate(documents):
        doc_ref = f"{source_ref}.documents[{doc_index}]"
        if isinstance(document, Mapping) and "jsonContent" in document:
            content = document.get("jsonContent")
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError as exc:
                    issues.append(
                        IngestIssue(
                            code="malformed_json_content",
                            message="jsonContent is not valid JSON",
                            source_kind="esp",
                            source_ref=doc_ref,
                            details={"error": str(exc)},
                        )
                    )
                    continue
            child_spans, child_issues = _extract_spans(content, source_ref=doc_ref)
            for child_span in child_spans:
                child_span.setdefault("_source_kind", "esp")
                child_span.setdefault("_source_ref", doc_ref)
            spans.extend(child_spans)
            issues.extend(child_issues)
            continue

        if isinstance(document, Mapping):
            flat = _normalize_flat_span(document)
            flat["_source_kind"] = "esp"
            flat["_source_ref"] = doc_ref
            spans.append(flat)
            continue

        issues.append(
            IngestIssue(
                code="malformed_esp_document",
                message="document is not an object",
                source_kind="esp",
                source_ref=doc_ref,
                details={"type": type(document).__name__},
            )
        )

    return spans, issues


def _attributes_to_map(attributes: Any) -> dict[str, Any]:
    if isinstance(attributes, Mapping):
        return {key: _unwrap_value(value) for key, value in attributes.items()}
    if isinstance(attributes, list):
        result: dict[str, Any] = {}
        for item in attributes:
            if not isinstance(item, Mapping):
                continue
            key = item.get("key")
            if isinstance(key, str) and key:
                result[key] = _unwrap_value(item.get("value"))
        return result
    return {}


def _unwrap_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "stringValue" in value:
            return value.get("stringValue")
        if "intValue" in value:
            return value.get("intValue")
        if "boolValue" in value:
            return value.get("boolValue")
        if "doubleValue" in value:
            return value.get("doubleValue")
        if "arrayValue" in value:
            payload = value.get("arrayValue")
            if isinstance(payload, Mapping):
                values = payload.get("values")
                if isinstance(values, list):
                    return [_unwrap_value(item) for item in values]
            return []
        # Some envelopes already flatten to {value: ...}.
        if "value" in value and len(value) == 1:
            return _unwrap_value(value.get("value"))
        return {key: _unwrap_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_unwrap_value(item) for item in value]
    return value


def _normalize_flat_span(span: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {str(key): _unwrap_value(value) for key, value in span.items()}
    # Preserve original tags if present and expose canonical aliases used downstream.
    normalized.setdefault(
        "tenant_id",
        _get_alias(
            normalized,
            "microsoft.tenant.id",
            "microsoft.agent365.tenant.id",
            "tenant.id",
            "TenantId",
        ),
    )
    normalized.setdefault("agent_id", _get_alias(normalized, "gen_ai.agent.id", "AgentId"))
    normalized.setdefault("conversation_id", _get_alias(normalized, "gen_ai.conversation.id", "ConversationId"))
    normalized.setdefault("session_id", _get_alias(normalized, "microsoft.session.id", "session_id", "SessionIdentity"))
    normalized.setdefault(
        "channel",
        _get_alias(
            normalized,
            "microsoft.channel.name",
            "microsoft.agent365.channel",
            "gen_ai.execution.source.name",
            "ChannelName",
        ),
    )
    if not _get_alias(normalized, "name", "Name", "operationName") and (
        "RequestMessages" in normalized or "ResponseMessages" in normalized
    ):
        normalized["name"] = "invoke_agent"
    return normalized


def _get_alias(values: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = values.get(key)
        text = _to_text(value)
        if text:
            return text
    return None


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def _dedupe_spans(spans: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for span in spans:
        tenant = _to_text(span.get("tenant_id")) or ""
        trace_id = _to_text(span.get("traceId") or span.get("TraceId")) or ""
        span_id = _to_text(span.get("spanId") or span.get("SpanId")) or ""
        key = (tenant, trace_id, span_id)
        if trace_id and span_id and key in seen:
            continue
        if trace_id and span_id:
            seen.add(key)
        deduped.append(span)
    return deduped


def _build_units(
    spans: list[dict[str, Any]],
    sessionization_policy: SessionizationPolicy,
) -> tuple[list[EvaluationUnit], list[IngestIssue]]:
    session_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    conversation_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    conversation_session_ids: dict[tuple[str, str, str], set[str]] = {}
    issues: list[IngestIssue] = []

    for span in spans:
        tenant_id = _to_text(span.get("tenant_id"))
        agent_id = _to_text(span.get("agent_id"))
        session_id = _to_text(span.get("session_id"))
        conversation_id = _to_text(span.get("conversation_id"))
        if tenant_id and agent_id and conversation_id and session_id:
            conversation_session_ids.setdefault(
                (tenant_id, agent_id, conversation_id), set()
            ).add(session_id)

    for span in spans:
        tenant_id = _to_text(span.get("tenant_id"))
        agent_id = _to_text(span.get("agent_id"))
        session_id = _to_text(span.get("session_id"))
        conversation_id = _to_text(span.get("conversation_id"))
        source_kind = _to_text(span.get("_source_kind")) or "record"
        source_ref = _to_text(span.get("_source_ref"))

        missing = [
            name
            for name, value in (
                ("tenant_id", tenant_id),
                ("agent_id", agent_id),
            )
            if not value
        ]
        if not session_id and not conversation_id:
            missing.append("session_id|conversation_id")
        if missing:
            issues.append(
                IngestIssue(
                    code="missing_identity",
                    message=f"Missing required identity fields: {', '.join(missing)}",
                    source_kind=source_kind,
                    source_ref=source_ref,
                )
            )
            continue

        if session_id:
            session_grouped.setdefault((tenant_id, agent_id, session_id), []).append(span)
        else:
            assert conversation_id is not None
            conversation_key = (tenant_id, agent_id, conversation_id)
            inferred_sessions = conversation_session_ids.get(conversation_key, set())
            if len(inferred_sessions) == 1:
                inferred_session_id = next(iter(inferred_sessions))
                session_grouped.setdefault(
                    (tenant_id, agent_id, inferred_session_id), []
                ).append(span)
            else:
                conversation_grouped.setdefault(conversation_key, []).append(span)

    units: list[EvaluationUnit] = []

    for key in sorted(session_grouped):
        tenant_id, agent_id, session_id = key
        ordered = sorted(session_grouped[key], key=_span_sort_key)
        unit, unit_issues = _build_unit(
            tenant_id=tenant_id,
            agent_id=agent_id,
            spans=ordered,
            session_id=session_id,
            sessionization_kind="session_id",
        )
        issues.extend(unit_issues)
        if unit is not None:
            units.append(unit)

    for key in sorted(conversation_grouped):
        tenant_id, agent_id, conversation_id = key
        segments, segmentation_issues = _split_inactivity_sessions(
            tenant_id=tenant_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            spans=conversation_grouped[key],
            inactivity_timeout=sessionization_policy.inactivity_timeout,
        )
        issues.extend(segmentation_issues)
        for segment_spans in segments:
            unit, unit_issues = _build_unit(
                tenant_id=tenant_id,
                agent_id=agent_id,
                spans=segment_spans,
                session_id=None,
                sessionization_kind="inactivity",
            )
            issues.extend(unit_issues)
            if unit is not None:
                units.append(unit)

    return units, issues


def _build_unit(
    tenant_id: str,
    agent_id: str,
    spans: list[dict[str, Any]],
    session_id: str | None,
    sessionization_kind: str,
) -> tuple[EvaluationUnit | None, list[IngestIssue]]:
    issues: list[IngestIssue] = []

    invoke_turns: list[Turn] = []
    inference_turns: list[Turn] = []
    tool_calls: list[ToolCall] = []

    conversation_ids: set[str] = set()
    channel_values: set[str] = set()
    trace_ids: list[str] = []
    started_at: datetime | None = None
    ended_at: datetime | None = None
    saw_explicit_end = False
    latest_start: datetime | None = None
    had_error = False

    for span in spans:
        conversation_id = _to_text(span.get("conversation_id"))
        if conversation_id:
            conversation_ids.add(conversation_id)

        channel = _to_text(span.get("channel"))
        if channel:
            channel_values.add(channel)

        trace_id = _to_text(span.get("traceId") or span.get("TraceId"))
        if trace_id:
            trace_ids.append(trace_id)

        span_start = _parse_timestamp(span)
        explicit_span_end = _parse_end_timestamp(span)
        span_end = explicit_span_end or span_start
        if span_start is not None:
            started_at = span_start if started_at is None else min(started_at, span_start)
            latest_start = span_start if latest_start is None else max(latest_start, span_start)
        if explicit_span_end is not None:
            saw_explicit_end = True
        if span_end is not None:
            ended_at = span_end if ended_at is None else max(ended_at, span_end)

        if _span_has_error(span):
            had_error = True

        name = (_to_text(span.get("name")) or _to_text(span.get("Name")) or "").lower()
        operation = (
            _to_text(span.get("gen_ai.operation.name"))
            or _to_text(span.get("operationName"))
            or ""
        ).lower()

        user_text, assistant_text = _extract_turn(span)
        if user_text or assistant_text:
            turn = Turn(user_text=user_text or "", assistant_text=assistant_text or "")
            if operation == "invoke_agent" or "invoke_agent" in name:
                invoke_turns.append(turn)
            elif operation in {"chat", "inference"} or "inference" in name or name.startswith("chat "):
                inference_turns.append(turn)

        tool_call = _extract_tool_call(span)
        if tool_call is not None:
            tool_calls.append(tool_call)

    if ended_at is None:
        ended_at = latest_start

    turns = tuple(invoke_turns if invoke_turns else inference_turns)
    sorted_conversation_ids = tuple(sorted(conversation_ids))
    legacy_conversation_id = sorted_conversation_ids[0] if sorted_conversation_ids else ""

    if len(channel_values) == 1:
        normalized_channel = next(iter(channel_values))
    elif len(channel_values) > 1:
        normalized_channel = "multi"
    else:
        normalized_channel = None

    unit = EvaluationUnit(
        tenant_id=tenant_id,
        agent_id=agent_id,
        conversation_id=legacy_conversation_id,
        session_id=session_id,
        channel=normalized_channel,
        source_trace_ids=tuple(sorted(set(trace_ids))),
        started_at=started_at,
        ended_at=ended_at,
        had_error=had_error,
        turns=turns,
        tool_calls=tuple(tool_calls),
        conversation_ids=sorted_conversation_ids,
        sessionization_kind=sessionization_kind,
    )

    if not saw_explicit_end:
        issues.append(
            IngestIssue(
                code="session_completion_inferred",
                message="Session missing explicit completion timestamp; latest event start was used",
                source_kind="session",
                source_ref=unit.unit_id,
                details={
                    "tenant_id": tenant_id,
                    "agent_id": agent_id,
                    "session_id": session_id,
                },
            )
        )

    if not unit.is_judgeable:
        issues.append(
            IngestIssue(
                code="incomplete_session",
                message="Session is not judgeable: requires user text and final assistant text",
                source_kind="session",
                source_ref=unit.unit_id,
            )
        )
        return None, issues

    return unit, issues


def _split_inactivity_sessions(
    tenant_id: str,
    agent_id: str,
    conversation_id: str,
    spans: list[dict[str, Any]],
    inactivity_timeout: timedelta,
) -> tuple[list[list[dict[str, Any]]], list[IngestIssue]]:
    issues: list[IngestIssue] = []
    timestamped: list[dict[str, Any]] = []
    untimestamped: list[dict[str, Any]] = []

    for span in spans:
        if _parse_timestamp(span) is None:
            untimestamped.append(span)
        else:
            timestamped.append(span)

    timestamped.sort(key=_span_sort_key)
    segments: list[list[dict[str, Any]]] = []

    current: list[dict[str, Any]] = []
    current_end: datetime | None = None
    for span in timestamped:
        start_at = _parse_timestamp(span)
        if start_at is None:
            continue
        end_at = _parse_end_timestamp(span) or start_at
        if not current:
            current = [span]
            current_end = end_at
            continue

        assert current_end is not None
        if start_at > current_end + inactivity_timeout:
            segments.append(current)
            current = [span]
            current_end = end_at
            continue

        current.append(span)
        if end_at > current_end:
            current_end = end_at

    if current:
        segments.append(current)

    if untimestamped:
        untimestamped.sort(key=_untimestamped_span_sort_key)
        segments.append(untimestamped)
        issues.append(
            IngestIssue(
                code="fallback_sessionization_uncertainty",
                message=(
                    "Fallback inactivity sessionization used deterministic segmenting "
                    "because one or more spans were missing start timestamps"
                ),
                source_kind="session",
                source_ref=f"{tenant_id}:{agent_id}:{conversation_id}",
            )
        )

    return segments, issues


def _span_sort_key(span: Mapping[str, Any]) -> tuple[datetime, datetime, str, str, str, str]:
    start_at = _parse_timestamp(span) or datetime.max.replace(tzinfo=timezone.utc)
    end_at = _parse_end_timestamp(span) or start_at
    return (
        start_at,
        end_at,
        _to_text(span.get("spanId") or span.get("SpanId")) or "",
        _to_text(span.get("traceId") or span.get("TraceId")) or "",
        _to_text(span.get("name") or span.get("Name") or span.get("operationName")) or "",
        _to_text(span.get("_source_ref")) or "",
    )


def _untimestamped_span_sort_key(span: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        _to_text(span.get("spanId") or span.get("SpanId")) or "",
        _to_text(span.get("traceId") or span.get("TraceId")) or "",
        _to_text(span.get("name") or span.get("Name") or span.get("operationName")) or "",
        _to_text(span.get("_source_ref")) or "",
    )


def _extract_turn(span: Mapping[str, Any]) -> tuple[str | None, str | None]:
    user_text = _extract_message_text(
        span.get("RequestMessages")
        or span.get("request_messages")
        or span.get("gen_ai.input.messages")
        or span.get("gen_ai.request.messages")
        or span.get("input")
        or span.get("gen_ai.input"),
        role="user",
    )
    assistant_text = _extract_message_text(
        span.get("ResponseMessages")
        or span.get("response_messages")
        or span.get("gen_ai.output.messages")
        or span.get("gen_ai.response.messages")
        or span.get("output")
        or span.get("gen_ai.output"),
        role="assistant",
    )
    return user_text, assistant_text


def _extract_message_text(payload: Any, role: str) -> str | None:
    messages = _parse_messages(payload)
    filtered = [message for message in messages if message.get("role") == role and message.get("text")]
    if not filtered:
        return None
    return "\n".join(message["text"] for message in filtered)


def _parse_messages(payload: Any) -> list[dict[str, str]]:
    data = payload
    if isinstance(data, str):
        text = data.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Preserve plain text fallback as a single message.
            return [{"role": "user", "text": text}]

    if isinstance(data, Mapping) and isinstance(data.get("messages"), list):
        data = data["messages"]

    if not isinstance(data, list):
        return []

    out: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        role = _to_text(item.get("role")) or ""
        role = role.lower()
        if role not in {"user", "assistant", "tool"}:
            continue

        text = _to_text(item.get("content"))
        if text is None and isinstance(item.get("parts"), list):
            pieces: list[str] = []
            for part in item["parts"]:
                if isinstance(part, Mapping):
                    part_type = _to_text(part.get("type")) or ""
                    if part_type in {"text", "tool_call"}:
                        part_text = _to_text(part.get("text") or part.get("content"))
                        if not part_text and part_type == "tool_call":
                            tool_name = _to_text(part.get("name"))
                            if tool_name:
                                part_text = f"[tool_call: {tool_name}]"
                        if part_text:
                            pieces.append(part_text)
            text = "\n".join(pieces) if pieces else None

        if text:
            out.append({"role": role, "text": text})

    return out


def _extract_tool_call(span: Mapping[str, Any]) -> ToolCall | None:
    name = (_to_text(span.get("name")) or _to_text(span.get("Name")) or "").lower()
    if "execute_tool" not in name and "tool" not in name:
        return None

    tool_name = _to_text(
        span.get("gen_ai.tool.name")
        or span.get("tool.name")
        or span.get("toolName")
        or span.get("ToolName")
    )
    input_text = _to_text(
        span.get("gen_ai.tool.call.arguments")
        or span.get("tool.input")
        or span.get("input")
        or span.get("arguments")
    )
    output_text = _to_text(
        span.get("gen_ai.tool.call.result")
        or span.get("tool.output")
        or span.get("output")
        or span.get("result")
    )
    status = _to_text(span.get("status") or span.get("Status") or span.get("otel.status_code"))

    details = {
        key: value
        for key, value in span.items()
        if isinstance(key, str)
        and (key.startswith("tool.") or "execute_tool" in key or key in {"operationName", "kind"})
    }
    return ToolCall(
        name=tool_name,
        input_text=input_text,
        output_text=output_text,
        status=status,
        details=details or None,
    )


def _span_has_error(span: Mapping[str, Any]) -> bool:
    raw_status = span.get("status") or span.get("Status") or span.get("otel.status_code")
    if isinstance(raw_status, Mapping):
        raw_status = raw_status.get("code") or raw_status.get("statusCode")
    if raw_status == 2:
        return True
    status = _to_text(raw_status)
    if status and status.lower() in {"error", "failed", "failure"}:
        return True
    error_message = _to_text(span.get("ErrorMessage") or span.get("error.message") or span.get("exception.message"))
    return bool(error_message)


def _parse_timestamp(span: Mapping[str, Any]) -> datetime | None:
    return _parse_datetime(
        span.get("startTimeUnixNano")
        or span.get("start_time_unix_nano")
        or span.get("startTime")
        or span.get("TimeGenerated")
    )


def _parse_end_timestamp(span: Mapping[str, Any]) -> datetime | None:
    return _parse_datetime(
        span.get("endTimeUnixNano")
        or span.get("end_time_unix_nano")
        or span.get("CompletionTime")
        or span.get("endTime")
    )


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        magnitude = abs(value)
        if magnitude >= 100_000_000_000_000_000:
            seconds = value / 1_000_000_000
        elif magnitude >= 100_000_000_000_000:
            seconds = value / 1_000_000
        elif magnitude >= 100_000_000_000:
            seconds = value / 1_000
        else:
            seconds = value
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return _parse_datetime(int(text))
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    return None
