from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from random_sampling.models import EvaluationUnit, ToolCall, Turn
from sampling_comparison.v2_experiment import (
    CombinedDataset,
    DENSE_2500_PATH,
    HISTORICAL_300_PATH,
    load_combined_dataset,
)
from trace_sampling.model import SessionEvent, Trace


COSMOS_LABELS_FILE = "genesis_observability__labels.jsonl"
COSMOS_SPANS_FILE = "genesis_observability__spans.jsonl"
HISTORICAL_DATASET_ID = "historical_300"
DENSE_DATASET_ID = "dense_2500"
COSMOS_DATASET_ID = "cosmos_otel"


@dataclass(frozen=True)
class V7Dataset:
    dataset_id: str
    ordered_unit_ids: tuple[str, ...]
    labels_by_unit: dict[str, int]
    traces_by_unit_id: dict[str, Trace]
    agent_id_by_unit: dict[str, str]
    concept_key_by_unit: dict[str, str]
    use_case_id_by_unit: dict[str, str]
    business_use_case_guid_by_unit: dict[str, str]
    metadata_by_unit: dict[str, dict[str, Any]]
    representation_source_by_unit: dict[str, str]
    representation_text_by_unit: dict[str, str]
    source_paths: dict[str, str]


def _stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    return [str(value).strip()] if str(value).strip() else []


def _label_key(value: Any) -> str:
    return str(value or "").strip().lower()


def map_expected_outcome_label(expected_outcome: Any, *, partial_label: int = 0) -> int:
    if int(partial_label) not in (0, 1):
        raise ValueError("partial_label must be 0 or 1")
    token = _label_key(expected_outcome)
    mapping = {
        "good": 1,
        "bad": 0,
        "partial": int(partial_label),
    }
    if token not in mapping:
        raise ValueError(f"unsupported expected_outcome label: {expected_outcome!r}")
    return int(mapping[token])


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.open("r", encoding="utf-8"):
        text = line.strip()
        if not text:
            continue
        obj = json.loads(text)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _iter_jsonl_stream(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if isinstance(obj, dict):
                yield obj


def _extract_span_agent_id(span: Mapping[str, Any]) -> str:
    value = span.get("agentId")
    if isinstance(value, str) and value.strip():
        return value.strip()
    attrs = span.get("attributes")
    if isinstance(attrs, Mapping):
        for key in ("gen_ai.agent.id", "agent.id", "agentId"):
            raw = attrs.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return "cosmos|unknown-agent"


def _collect_span_attribute_values(attributes: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(attributes, Mapping):
        return {}
    filtered: dict[str, Any] = {}
    skipped_prefixes = ("resource.", "service.", "telemetry.", "container.", "host.", "os.", "process.")
    for key, value in attributes.items():
        if not isinstance(key, str):
            continue
        lowered = key.lower()
        if lowered.startswith("_"):
            continue
        if any(lowered.startswith(prefix) for prefix in skipped_prefixes):
            continue
        if any(token in lowered for token in ("secret", "token", "key", "password", "authorization", "cookie")):
            continue
        filtered[key] = value
    return filtered


def _span_text(span: Mapping[str, Any]) -> str:
    name = str(span.get("name") or "")
    agent_id = _extract_span_agent_id(span)
    attrs = _collect_span_attribute_values(span.get("attributes") if isinstance(span.get("attributes"), Mapping) else {})
    events = span.get("events") if isinstance(span.get("events"), list) else []
    kept = {}
    for key in sorted(attrs.keys()):
        lowered = str(key).lower()
        if lowered in {
            "gen_ai.input_messages",
            "gen_ai.prompt",
            "gen_ai.output_messages",
            "gen_ai.completion",
            "gen_ai.response.id",
            "gen_ai.conversation.id",
            "genesis.session.id",
            "gen_ai.request.model",
            "gen_ai.response.model",
            "gen_ai.system",
            "tool.name",
            "tool.call.id",
            "tool.arguments",
            "tool.result",
            "user.message",
            "assistant.message",
            "session.id",
            "response.id",
        } or any(token in lowered for token in ("prompt", "completion", "output", "input", "tool", "session", "response")):
            kept[key] = attrs[key]
    payload = {
        "name": name,
        "agent_id": agent_id,
        "attributes": kept,
        "event_names": [str(event.get("name") or "") for event in events if isinstance(event, Mapping)],
    }
    return _canonical_json(payload)


def _span_session_ids(span: Mapping[str, Any]) -> list[str]:
    attrs = span.get("attributes") if isinstance(span.get("attributes"), Mapping) else {}
    values: list[str] = []
    for key in ("genesis.session.id", "session.id", "gen_ai.session.id"):
        raw = attrs.get(key)
        if isinstance(raw, str) and raw.strip():
            values.append(raw.strip())
    return values


def _span_response_ids(span: Mapping[str, Any]) -> list[str]:
    attrs = span.get("attributes") if isinstance(span.get("attributes"), Mapping) else {}
    values: list[str] = []
    for key in ("gen_ai.response.id", "response.id", "openai.response.id"):
        raw = attrs.get(key)
        if isinstance(raw, str) and raw.strip():
            values.append(raw.strip())
    return values


def _span_conversation_ids(span: Mapping[str, Any]) -> list[str]:
    attrs = span.get("attributes") if isinstance(span.get("attributes"), Mapping) else {}
    values: list[str] = []
    for key in ("gen_ai.conversation.id", "conversation.id", "openai.conversation.id"):
        raw = attrs.get(key)
        if isinstance(raw, str) and raw.strip():
            values.append(raw.strip())
    return values


def _span_start_sort_key(span: Mapping[str, Any]) -> tuple[float, float, str]:
    attrs = span.get("attributes") if isinstance(span.get("attributes"), Mapping) else {}

    def _parse_time(value: Any) -> float:
        text = str(value or "").strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            pass
        iso = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(iso)
        except ValueError:
            return 0.0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.timestamp())

    start_raw = span.get("startTime") or span.get("start_time") or attrs.get("startTime") or attrs.get("start_time")
    end_raw = span.get("endTime") or span.get("end_time") or attrs.get("endTime") or attrs.get("end_time")
    start_val = _parse_time(start_raw)
    end_val = _parse_time(end_raw)
    return (start_val, end_val, str(span.get("spanId") or ""))


def _sorted_spans(spans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(span) for span in spans), key=_span_start_sort_key)


def _synthetic_label_text(label: Mapping[str, Any]) -> str:
    payload = {
        "task_id": str(label.get("task_id") or ""),
        "domain": str(label.get("domain") or ""),
        "required_tools": _normalize_list(label.get("required_tools")),
        "required_docs": _normalize_list(label.get("required_docs")),
        "available_tools": _normalize_list(label.get("available_tools")),
        "available_knowledge": _normalize_list(label.get("available_knowledge")),
    }
    return _canonical_json(payload)


def _build_cosmos_trace(
    *,
    unit_id: str,
    agent_id: str,
    task_id: str,
    domain: str,
    representation_text: str,
    required_tools: Sequence[str],
) -> Trace:
    signature = tuple(required_tools) if required_tools else ("no-tool",)
    events = (
        SessionEvent(role="user", text=f"task_id={task_id} domain={domain}"),
        SessionEvent(role="assistant", text=representation_text),
    )
    return Trace(
        trace_id=_stable_int(unit_id),
        agent_id=agent_id,
        timestamp=0.0,
        signature=signature,
        span_count=max(1, len(signature)),
        duration_ms=0.0,
        status="ok",
        concept_id=_stable_int(f"{domain}|{task_id}") % 1000003,
        events=events,
    )


def build_v7_dataset_from_combined(data: CombinedDataset, *, dataset_id: str = "combined") -> V7Dataset:
    ordered_unit_ids = tuple(str(uid) for uid in data.unit_ids)
    labels_by_unit = {uid: int(1 if bool(data.labels_by_unit[uid]) else 0) for uid in ordered_unit_ids}
    agent_id_by_unit = {uid: str(data.trace_by_unit_id[uid].agent_id) for uid in ordered_unit_ids}
    concept_key_by_unit = {
        uid: "|".join(
            (
                str((data.metadata_by_unit.get(uid) or {}).get("corpus_id") or "unknown"),
                str((data.metadata_by_unit.get(uid) or {}).get("domain") or "unknown"),
                str((data.metadata_by_unit.get(uid) or {}).get("task") or "unknown"),
                str((data.metadata_by_unit.get(uid) or {}).get("difficulty") or "unknown"),
            )
        )
        for uid in ordered_unit_ids
    }
    use_case_id_by_unit = {uid: str((data.metadata_by_unit.get(uid) or {}).get("task") or "unknown-task") for uid in ordered_unit_ids}
    business_use_case_guid_by_unit = {
        uid: str((data.metadata_by_unit.get(uid) or {}).get("euw_guid") or (data.metadata_by_unit.get(uid) or {}).get("task") or "unknown-guid")
        for uid in ordered_unit_ids
    }
    return V7Dataset(
        dataset_id=str(dataset_id),
        ordered_unit_ids=ordered_unit_ids,
        labels_by_unit=labels_by_unit,
        traces_by_unit_id={uid: data.trace_by_unit_id[uid] for uid in ordered_unit_ids},
        agent_id_by_unit=agent_id_by_unit,
        concept_key_by_unit=concept_key_by_unit,
        use_case_id_by_unit=use_case_id_by_unit,
        business_use_case_guid_by_unit=business_use_case_guid_by_unit,
        metadata_by_unit={uid: dict(data.metadata_by_unit.get(uid, {})) for uid in ordered_unit_ids},
        representation_source_by_unit={uid: "combined_normalized" for uid in ordered_unit_ids},
        representation_text_by_unit={uid: "" for uid in ordered_unit_ids},
        source_paths={key: str(value) for key, value in data.source_paths.items()},
    )


def _subset_combined_by_corpus(data: CombinedDataset, *, corpus_id: str) -> CombinedDataset:
    unit_ids = [uid for uid in data.unit_ids if str((data.metadata_by_unit.get(uid) or {}).get("corpus_id") or "") == corpus_id]
    units = [unit for unit in data.units if (unit.unit_id or "") in set(unit_ids)]
    traces = [data.trace_by_unit_id[uid] for uid in unit_ids]
    return CombinedDataset(
        units=tuple(units),
        unit_ids=tuple(unit_ids),
        traces=tuple(traces),
        trace_by_unit_id={uid: data.trace_by_unit_id[uid] for uid in unit_ids},
        labels_by_unit={uid: bool(data.labels_by_unit[uid]) for uid in unit_ids},
        metadata_by_unit={uid: dict(data.metadata_by_unit[uid]) for uid in unit_ids},
        corpus_id_by_unit={uid: str(data.corpus_id_by_unit[uid]) for uid in unit_ids},
        original_unit_id_by_unit={uid: str(data.original_unit_id_by_unit[uid]) for uid in unit_ids},
        scoped_identities=tuple(sorted({f"{unit.tenant_id}|{unit.agent_id}" for unit in units})),
        source_paths={corpus_id: str(data.source_paths[corpus_id])},
    )


def to_combined_dataset(dataset: V7Dataset) -> CombinedDataset:
    units: list[EvaluationUnit] = []
    traces: list[Trace] = []
    trace_by_unit_id: dict[str, Trace] = {}
    labels_by_unit: dict[str, bool] = {}
    metadata_by_unit: dict[str, dict[str, Any]] = {}
    corpus_id_by_unit: dict[str, str] = {}
    original_unit_id_by_unit: dict[str, str] = {}

    for uid in dataset.ordered_unit_ids:
        metadata = dict(dataset.metadata_by_unit.get(uid) or {})
        trace = dataset.traces_by_unit_id[uid]
        user_texts = [event.text for event in trace.events if event.role == "user" and event.text]
        assistant_texts = [event.text for event in trace.events if event.role == "assistant" and event.text]
        turn = Turn(user_text="\n".join(user_texts), assistant_text="\n".join(assistant_texts))
        agent_id = str(dataset.agent_id_by_unit.get(uid) or trace.agent_id or "unknown-agent")
        tenant_id = str(metadata.get("tenant_id") or dataset.dataset_id)
        session_id = str((metadata.get("linked_session_ids") or [metadata.get("session_id") or uid])[0])
        source_trace_ids = tuple(str(x) for x in (metadata.get("linked_trace_ids") or metadata.get("trace_ids") or [uid]))

        unit = EvaluationUnit(
            tenant_id=tenant_id,
            agent_id=agent_id,
            conversation_id=str(metadata.get("conversation_id") or uid),
            session_id=session_id,
            channel=str(metadata.get("channel") or "v7"),
            source_trace_ids=source_trace_ids,
            started_at=datetime.fromtimestamp(0),
            ended_at=datetime.fromtimestamp(0),
            had_error=False,
            turns=(turn,),
            tool_calls=(ToolCall(name=None),),
            unit_id=uid,
        )
        units.append(unit)
        traces.append(trace)
        trace_by_unit_id[uid] = trace
        labels_by_unit[uid] = bool(dataset.labels_by_unit[uid])
        metadata_by_unit[uid] = metadata
        corpus_id_by_unit[uid] = dataset.dataset_id
        original_unit_id_by_unit[uid] = uid

    return CombinedDataset(
        units=tuple(units),
        unit_ids=tuple(dataset.ordered_unit_ids),
        traces=tuple(traces),
        trace_by_unit_id=trace_by_unit_id,
        labels_by_unit=labels_by_unit,
        metadata_by_unit=metadata_by_unit,
        corpus_id_by_unit=corpus_id_by_unit,
        original_unit_id_by_unit=original_unit_id_by_unit,
        scoped_identities=tuple(sorted({f"{unit.tenant_id}|{unit.agent_id}" for unit in units})),
        source_paths=dict(dataset.source_paths),
    )


def load_cosmos_otel_expected_dataset(
    cosmos_root: str | Path,
    *,
    partial_label: int = 0,
) -> V7Dataset:
    root = Path(cosmos_root)
    labels_path = root / COSMOS_LABELS_FILE
    spans_path = root / COSMOS_SPANS_FILE
    if not labels_path.exists():
        raise FileNotFoundError(f"Cosmos labels file not found: {labels_path}")
    if not spans_path.exists():
        raise FileNotFoundError(f"Cosmos spans file not found: {spans_path}")

    labels = _iter_jsonl(labels_path)
    spans: list[dict[str, Any]] = []
    spans_by_response_id: dict[str, list[dict[str, Any]]] = {}
    spans_by_conversation_id: dict[str, list[dict[str, Any]]] = {}
    spans_by_session_id: dict[str, list[dict[str, Any]]] = {}
    spans_by_trace_id: dict[str, list[dict[str, Any]]] = {}
    for span in _iter_jsonl_stream(spans_path):
        spans.append(span)
        trace_id = str(span.get("traceId") or "").strip()
        if trace_id:
            spans_by_trace_id.setdefault(trace_id, []).append(span)
        for response_id in _span_response_ids(span):
            spans_by_response_id.setdefault(response_id, []).append(span)
        for conversation_id in _span_conversation_ids(span):
            spans_by_conversation_id.setdefault(conversation_id, []).append(span)
        for session_id in _span_session_ids(span):
            spans_by_session_id.setdefault(session_id, []).append(span)

    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels, start=1):
        label_id = str(label.get("id") or f"label-{index}")
        unit_id = f"cosmos_otel:{label_id}"
        expected = map_expected_outcome_label(label.get("expected_outcome"), partial_label=partial_label)
        task_id = str(label.get("task_id") or f"task-{index}")
        domain = str(label.get("domain") or "unknown-domain")
        euw_guid = str(label.get("euw_guid") or "")
        required_tools = _normalize_list(label.get("required_tools"))

        label_trace_ids = [str(item).strip() for item in _normalize_list(label.get("traceIds")) if str(item).strip()]
        label_session_ids = [str(item).strip() for item in _normalize_list(label.get("sessionId")) if str(item).strip()]
        matched_spans: list[dict[str, Any]] = []
        linkage_path = "synthetic_label_document"
        linked_session_ids: list[str] = []
        linked_trace_ids: list[str] = []
        matched_response_count = 0
        linked_span_count = 0

        response_seed_spans: list[dict[str, Any]] = []
        for response_id in label_trace_ids:
            response_seed_spans.extend(spans_by_response_id.get(response_id, []))

        if response_seed_spans:
            matched_spans = response_seed_spans
            linkage_path = "response_id_to_session"
            matched_response_count = len(response_seed_spans)
        else:
            conversation_seed_spans: list[dict[str, Any]] = []
            for conversation_id in label_trace_ids:
                conversation_seed_spans.extend(spans_by_conversation_id.get(conversation_id, []))
            if conversation_seed_spans:
                matched_spans = conversation_seed_spans
                linkage_path = "conversation_id_to_session"
            elif label_session_ids:
                for session_id in label_session_ids:
                    matched_spans.extend(spans_by_session_id.get(session_id, []))
                if matched_spans:
                    linkage_path = "session_id_to_session"
            if not matched_spans:
                for trace_id in label_trace_ids:
                    matched_spans.extend(spans_by_trace_id.get(trace_id, []))
                if matched_spans:
                    linkage_path = "trace_id_to_spans"

        if matched_spans:
            derived_sessions: list[str] = []
            for span in matched_spans:
                derived_sessions.extend(_span_session_ids(span))
            if derived_sessions:
                linked_session_ids = sorted(set(session_id for session_id in derived_sessions if session_id))
                linked_spans: list[dict[str, Any]] = []
                for session_id in linked_session_ids:
                    linked_spans.extend(spans_by_session_id.get(session_id, []))
            else:
                linked_spans = list(matched_spans)
            linked_spans = _sorted_spans(linked_spans)
            linked_span_count = len(linked_spans)
            linked_trace_ids = sorted({str(span.get("traceId") or "") for span in linked_spans if str(span.get("traceId") or "")})
            representation_source = "linked_raw_spans"
            representation_text = "\n".join(_span_text(span) for span in linked_spans)
            linked_agent_ids = sorted({_extract_span_agent_id(span) for span in linked_spans})
            agent_id = linked_agent_ids[0] if linked_agent_ids else "cosmos|unknown-agent"
        else:
            representation_source = "synthetic_label_document"
            representation_text = _synthetic_label_text(label)
            linked_trace_ids = []
            linked_session_ids = label_session_ids
            matched_response_count = 0
            linked_span_count = 0
            agent_id = str(label.get("agent_id") or f"cosmos|{domain}")



        metadata = {
            "task_id": task_id,
            "domain": domain,
            "euw_guid": euw_guid,
            "gearing": str(label.get("gearing") or ""),
            "expected_outcome": str(label.get("expected_outcome") or ""),
            "required_tools": required_tools,
            "required_docs": _normalize_list(label.get("required_docs")),
            "available_tools": _normalize_list(label.get("available_tools")),
            "available_knowledge": _normalize_list(label.get("available_knowledge")),
            "representation_source": representation_source,
            "session_id": str(label.get("sessionId") or ""),
            "trace_ids": label_trace_ids,
            "linked_trace_ids": linked_trace_ids,
            "linked_session_ids": linked_session_ids,
            "linked_span_count": int(linked_span_count),
            "linkage_path": linkage_path,
            "matched_response_count": matched_response_count,
            "task_design_expected_label": expected,
        }

        rows.append(
            {
                "unit_id": unit_id,
                "label": expected,
                "task_id": task_id,
                "domain": domain,
                "agent_id": agent_id,
                "concept_key": f"{domain}|{task_id}|{str(label.get('gearing') or '')}",
                "use_case_id": task_id,
                "business_use_case_guid": euw_guid or task_id,
                "representation_source": representation_source,
                "representation_text": representation_text,
                "required_tools": required_tools,
                "metadata": metadata,
            }
        )

    rows.sort(key=lambda row: row["unit_id"])
    ordered_unit_ids = tuple(row["unit_id"] for row in rows)
    traces_by_unit_id = {
        row["unit_id"]: _build_cosmos_trace(
            unit_id=row["unit_id"],
            agent_id=row["agent_id"],
            task_id=row["task_id"],
            domain=row["domain"],
            representation_text=row["representation_text"],
            required_tools=row["required_tools"],
        )
        for row in rows
    }

    return V7Dataset(
        dataset_id=COSMOS_DATASET_ID,
        ordered_unit_ids=ordered_unit_ids,
        labels_by_unit={row["unit_id"]: int(row["label"]) for row in rows},
        traces_by_unit_id=traces_by_unit_id,
        agent_id_by_unit={row["unit_id"]: row["agent_id"] for row in rows},
        concept_key_by_unit={row["unit_id"]: row["concept_key"] for row in rows},
        use_case_id_by_unit={row["unit_id"]: row["use_case_id"] for row in rows},
        business_use_case_guid_by_unit={row["unit_id"]: row["business_use_case_guid"] for row in rows},
        metadata_by_unit={row["unit_id"]: dict(row["metadata"]) for row in rows},
        representation_source_by_unit={row["unit_id"]: row["representation_source"] for row in rows},
        representation_text_by_unit={row["unit_id"]: row["representation_text"] for row in rows},
        source_paths={
            "labels": str(labels_path),
            "spans": str(spans_path),
        },
    )


def load_v7_datasets(
    *,
    historical_path: str = HISTORICAL_300_PATH,
    dense_path: str = DENSE_2500_PATH,
    cosmos_root: str | Path = r"C:\Users\stangoodwin\synth-data\data\cosmos_otel",
    partial_label: int = 0,
) -> dict[str, V7Dataset]:
    combined = load_combined_dataset(
        historical_path=historical_path,
        dense_path=dense_path,
        enforce_integrity_counts=False,
    )
    historical = build_v7_dataset_from_combined(
        _subset_combined_by_corpus(combined, corpus_id=HISTORICAL_DATASET_ID),
        dataset_id=HISTORICAL_DATASET_ID,
    )
    dense = build_v7_dataset_from_combined(
        _subset_combined_by_corpus(combined, corpus_id=DENSE_DATASET_ID),
        dataset_id=DENSE_DATASET_ID,
    )
    cosmos = load_cosmos_otel_expected_dataset(cosmos_root, partial_label=partial_label)
    return {
        HISTORICAL_DATASET_ID: historical,
        DENSE_DATASET_ID: dense,
        COSMOS_DATASET_ID: cosmos,
    }


__all__ = [
    "COSMOS_DATASET_ID",
    "COSMOS_LABELS_FILE",
    "COSMOS_SPANS_FILE",
    "DENSE_DATASET_ID",
    "HISTORICAL_DATASET_ID",
    "V7Dataset",
    "build_v7_dataset_from_combined",
    "load_cosmos_otel_expected_dataset",
    "load_v7_datasets",
    "map_expected_outcome_label",
    "to_combined_dataset",
]