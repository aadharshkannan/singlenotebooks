"""Canonical bounded evidence packet builder for judge requests."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .models import EvaluationUnit


TRUNCATION_MARKER = " [truncated]"
TRUNCATION_POLICY = "structural-long-session-v2"


def _stable_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _utf8_prefix(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    out: list[str] = []
    used = 0
    for char in text:
        encoded = char.encode("utf-8")
        if used + len(encoded) > max_bytes:
            break
        out.append(char)
        used += len(encoded)
    return "".join(out)


def _render_truncated(text: str, budget_bytes: int) -> str:
    if budget_bytes <= 0:
        return ""
    full_bytes = len(text.encode("utf-8"))
    if full_bytes <= budget_bytes:
        return text

    marker_bytes = len(TRUNCATION_MARKER.encode("utf-8"))
    prefix_budget = budget_bytes - marker_bytes
    if prefix_budget > 0:
        prefix = _utf8_prefix(text, prefix_budget)
        if prefix:
            return prefix + TRUNCATION_MARKER

    return _utf8_prefix(text, budget_bytes)


def _min_nonempty_budget(text: str) -> int:
    return len(text[0].encode("utf-8")) if text else 0


def _mandatory_payload(unit: EvaluationUnit) -> dict[str, Any]:
    return {
        "version": "evidence-packet-v2",
        "scope": {
            "tenant_id": unit.tenant_id,
            "agent_id": unit.agent_id,
            "unit_id": unit.unit_id,
            "session_id": unit.session_id,
            "conversation_ids": list(unit.conversation_ids),
            "sessionization_kind": unit.sessionization_kind,
        },
        "trace_ids": list(unit.source_trace_ids),
        "timing": {
            "started_at": unit.started_at.isoformat() if unit.started_at else None,
            "ended_at": unit.ended_at.isoformat() if unit.ended_at else None,
            "had_error": unit.had_error,
        },
        "turns": [],
        "tool_calls": [],
        "audit": {
            "truncation_policy": TRUNCATION_POLICY,
            "version": "v2",
            "original_turn_count": 0,
            "emitted_turn_count": 0,
            "omitted_turn_count": 0,
            "original_tool_call_count": 0,
            "emitted_tool_call_count": 0,
            "omitted_tool_call_count": 0,
        },
    }


def _turn_slot(index: int, user_text: str = "", assistant_text: str = "") -> dict[str, Any]:
    return {
        "original_index": index,
        "user": user_text,
        "assistant": assistant_text,
    }


def _tool_slot(
    index: int,
    name: str | None = "",
    input_text: str | None = "",
    output_text: str | None = "",
    status: str | None = "",
) -> dict[str, Any]:
    return {
        "original_index": index,
        "name": name if name is not None else "",
        "input": input_text if input_text is not None else "",
        "output": output_text if output_text is not None else "",
        "status": status if status is not None else "",
    }


def _set_audit_counts(payload: dict[str, Any], original_turns: int, original_tools: int) -> None:
    emitted_turns = len(payload["turns"])
    emitted_tools = len(payload["tool_calls"])
    payload["audit"] = {
        "truncation_policy": TRUNCATION_POLICY,
        "version": "v2",
        "original_turn_count": original_turns,
        "emitted_turn_count": emitted_turns,
        "omitted_turn_count": original_turns - emitted_turns,
        "original_tool_call_count": original_tools,
        "emitted_tool_call_count": emitted_tools,
        "omitted_tool_call_count": original_tools - emitted_tools,
    }


def _full_payload(unit: EvaluationUnit) -> dict[str, Any]:
    payload = _mandatory_payload(unit)
    payload["turns"] = [
        _turn_slot(index=i, user_text=turn.user_text, assistant_text=turn.assistant_text)
        for i, turn in enumerate(unit.turns)
    ]
    payload["tool_calls"] = [
        _tool_slot(
            index=i,
            name=call.name,
            input_text=call.input_text,
            output_text=call.output_text,
            status=call.status,
        )
        for i, call in enumerate(unit.tool_calls)
    ]
    _set_audit_counts(payload, original_turns=len(unit.turns), original_tools=len(unit.tool_calls))
    return payload


@dataclass
class _TextField:
    field_id: tuple[str, int, str]
    slot: dict[str, Any]
    key: str
    source: str


def _set_field_to_budget(payload: dict[str, Any], field: _TextField, budget_bytes: int) -> int:
    rendered = _render_truncated(field.source, budget_bytes)
    field.slot[field.key] = rendered
    return len(rendered.encode("utf-8"))


def _max_fit_budget(
    payload: dict[str, Any],
    field: _TextField,
    max_bytes: int,
    high_budget: int,
    min_budget: int = 0,
) -> int:
    low = max(0, min_budget)
    high = max(0, high_budget)
    best = 0
    best_value = field.slot[field.key]

    while low <= high:
        mid = (low + high) // 2
        rendered = _render_truncated(field.source, mid)
        previous = field.slot[field.key]
        field.slot[field.key] = rendered
        fits = len(_stable_json_bytes(payload)) <= max_bytes
        field.slot[field.key] = previous

        if fits:
            best = mid
            best_value = rendered
            low = mid + 1
        else:
            high = mid - 1

    field.slot[field.key] = best_value
    return best


def _turn_context_order(total: int, retained: set[int]) -> list[int]:
    if total <= 0:
        return []
    out: list[int] = []
    left = 1
    right = total - 2
    while left <= right:
        if right not in retained:
            out.append(right)
        if left != right and left not in retained:
            out.append(left)
        left += 1
        right -= 1
    return out


def _tool_context_order(total: int, retained: set[int]) -> list[int]:
    return [i for i in range(total - 2, -1, -1) if i not in retained]


def _insert_turn_slot(payload: dict[str, Any], slot: dict[str, Any], max_bytes: int) -> bool:
    turns = payload["turns"]
    turns.append(slot)
    turns.sort(key=lambda row: row["original_index"])
    if len(_stable_json_bytes(payload)) <= max_bytes:
        return True
    turns.remove(slot)
    return False


def _insert_tool_slot(payload: dict[str, Any], slot: dict[str, Any], max_bytes: int) -> bool:
    calls = payload["tool_calls"]
    calls.append(slot)
    calls.sort(key=lambda row: row["original_index"])
    if len(_stable_json_bytes(payload)) <= max_bytes:
        return True
    calls.remove(slot)
    return False


def _build_truncated_payload(unit: EvaluationUnit, max_bytes: int) -> tuple[dict[str, Any], bool]:
    payload = _mandatory_payload(unit)

    turn_count = len(unit.turns)
    tool_count = len(unit.tool_calls)

    retained_turn_indexes: set[int] = set()
    if turn_count > 0:
        retained_turn_indexes.add(0)
        retained_turn_indexes.add(turn_count - 1)

    retained_tool_indexes: set[int] = set()
    if tool_count > 0:
        retained_tool_indexes.add(tool_count - 1)

    turn_slots: dict[int, dict[str, Any]] = {}
    tool_slots: dict[int, dict[str, Any]] = {}

    for index in sorted(retained_turn_indexes):
        slot = _turn_slot(index=index)
        turn_slots[index] = slot
        payload["turns"].append(slot)

    for index in sorted(retained_tool_indexes):
        call = unit.tool_calls[index]
        slot = _tool_slot(index=index, status=call.status)
        tool_slots[index] = slot
        payload["tool_calls"].append(slot)

    _set_audit_counts(payload, original_turns=turn_count, original_tools=tool_count)
    structural_size = len(_stable_json_bytes(payload))
    if structural_size > max_bytes:
        raise ValueError(
            "max_bytes too small for structural floor of evidence-packet-v2"
        )

    fields: dict[tuple[str, int, str], _TextField] = {}

    def register_field(kind: str, index: int, key: str, source: str) -> None:
        fields[(kind, index, key)] = _TextField(
            field_id=(kind, index, key),
            slot=turn_slots[index] if kind == "turn" else tool_slots[index],
            key=key,
            source=source,
        )

    for index in retained_turn_indexes:
        turn = unit.turns[index]
        register_field("turn", index, "user", turn.user_text)
        register_field("turn", index, "assistant", turn.assistant_text)

    for index in retained_tool_indexes:
        call = unit.tool_calls[index]
        register_field("tool", index, "name", call.name or "")
        register_field("tool", index, "input", call.input_text or "")
        register_field("tool", index, "output", call.output_text or "")
        register_field("tool", index, "status", call.status or "")

    mandatory_ids: list[tuple[str, int, str]] = []
    if turn_count > 0:
        first_user = unit.turns[0].user_text
        if first_user:
            mandatory_ids.append(("turn", 0, "user"))

        final_assistant = unit.turns[turn_count - 1].assistant_text
        if final_assistant:
            mandatory_ids.append(("turn", turn_count - 1, "assistant"))

    if tool_count > 0:
        latest = unit.tool_calls[tool_count - 1]
        if latest.name:
            mandatory_ids.append(("tool", tool_count - 1, "name"))
        if latest.output_text:
            mandatory_ids.append(("tool", tool_count - 1, "output"))

    for field_id in mandatory_ids:
        field = fields[field_id]
        if not field.source:
            continue
        previous = field.slot[field.key]
        field.slot[field.key] = field.source[0]
        if len(_stable_json_bytes(payload)) > max_bytes:
            field.slot[field.key] = previous
            raise ValueError(
                "max_bytes too small for mandatory evidence floor in evidence-packet-v2"
            )

    mandatory_min_budget: dict[tuple[str, int, str], int] = {}
    for field_id in mandatory_ids:
        field = fields[field_id]
        if field.source:
            mandatory_min_budget[field_id] = _min_nonempty_budget(field.source)

    mandatory_priority = [
        ("turn", turn_count - 1, "assistant"),
        ("tool", tool_count - 1, "output"),
        ("turn", 0, "user"),
        ("tool", tool_count - 1, "name"),
    ]

    for field_id in mandatory_priority:
        field = fields.get(field_id)
        if field is None or not field.source:
            continue
        _max_fit_budget(
            payload=payload,
            field=field,
            max_bytes=max_bytes,
            high_budget=min(32, len(field.source.encode("utf-8"))),
            min_budget=mandatory_min_budget.get(field_id, 0),
        )

    mandatory_target_budget: dict[tuple[str, int, str], int] = {}
    for field_id in mandatory_ids:
        field = fields[field_id]
        mandatory_target_budget[field_id] = len(field.slot[field.key].encode("utf-8"))

    for index in _turn_context_order(turn_count, retained_turn_indexes):
        slot = _turn_slot(index=index)
        if not _insert_turn_slot(payload, slot, max_bytes=max_bytes):
            continue
        turn_slots[index] = slot
        retained_turn_indexes.add(index)
        turn = unit.turns[index]
        register_field("turn", index, "user", turn.user_text)
        register_field("turn", index, "assistant", turn.assistant_text)

    for index in _tool_context_order(tool_count, retained_tool_indexes):
        call = unit.tool_calls[index]
        slot = _tool_slot(index=index, status=call.status)
        if not _insert_tool_slot(payload, slot, max_bytes=max_bytes):
            continue
        tool_slots[index] = slot
        retained_tool_indexes.add(index)
        register_field("tool", index, "name", call.name or "")
        register_field("tool", index, "input", call.input_text or "")
        register_field("tool", index, "output", call.output_text or "")
        register_field("tool", index, "status", call.status or "")

    growth_priority: list[tuple[str, int, str]] = []
    for field_id in [
        ("turn", turn_count - 1, "assistant"),
        ("tool", tool_count - 1, "output"),
        ("turn", 0, "user"),
        ("tool", tool_count - 1, "input"),
        ("tool", tool_count - 1, "name"),
    ]:
        if field_id in fields:
            growth_priority.append(field_id)

    for index in _turn_context_order(turn_count, {0, turn_count - 1} if turn_count > 0 else set()):
        for key in ("assistant", "user"):
            field_id = ("turn", index, key)
            if field_id in fields and field_id not in growth_priority:
                growth_priority.append(field_id)

    for index in _tool_context_order(tool_count, {tool_count - 1} if tool_count > 0 else set()):
        for key in ("output", "input", "name", "status"):
            field_id = ("tool", index, key)
            if field_id in fields and field_id not in growth_priority:
                growth_priority.append(field_id)

    for field_id in growth_priority:
        field = fields[field_id]
        if not field.source:
            continue
        _max_fit_budget(
            payload=payload,
            field=field,
            max_bytes=max_bytes,
            high_budget=len(field.source.encode("utf-8")),
            min_budget=mandatory_min_budget.get(field_id, 0),
        )

    _set_audit_counts(payload, original_turns=turn_count, original_tools=tool_count)

    if len(_stable_json_bytes(payload)) > max_bytes:
        for preserve_mandatory_target in (True, False):
            for field_id in reversed(growth_priority):
                if len(_stable_json_bytes(payload)) <= max_bytes:
                    break
                field = fields[field_id]
                if not field.source:
                    continue
                current_budget = len(field.slot[field.key].encode("utf-8"))
                if preserve_mandatory_target and field_id in mandatory_target_budget:
                    min_budget = mandatory_target_budget[field_id]
                else:
                    min_budget = mandatory_min_budget.get(field_id, 0)
                if current_budget <= min_budget:
                    continue
                _max_fit_budget(
                    payload=payload,
                    field=field,
                    max_bytes=max_bytes,
                    high_budget=current_budget - 1,
                    min_budget=min_budget,
                )
            if len(_stable_json_bytes(payload)) <= max_bytes:
                break

    if len(_stable_json_bytes(payload)) > max_bytes:
        if mandatory_min_budget:
            raise ValueError(
                "max_bytes too small for mandatory evidence floor in evidence-packet-v2"
            )
        raise ValueError(
            "max_bytes too small for structural floor of evidence-packet-v2"
        )

    text_truncated = False
    for field in fields.values():
        emitted = field.slot[field.key]
        if emitted != field.source:
            text_truncated = True
            break

    structure_truncated = (
        len(payload["turns"]) < turn_count or len(payload["tool_calls"]) < tool_count
    )
    return payload, (text_truncated or structure_truncated)


@dataclass(frozen=True)
class EvidencePacket:
    version: str
    canonical_json: str
    sha256: str
    truncated: bool
    original_bytes: int
    encoded_bytes: int
    omitted_turn_count: int
    omitted_tool_call_count: int


def build_evidence_packet(unit: EvaluationUnit, max_bytes: int = 32768) -> EvidencePacket:
    """Build deterministic, bounded canonical evidence for one evaluation unit."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    mandatory = _mandatory_payload(unit)
    mandatory_bytes = len(_stable_json_bytes(mandatory))
    if mandatory_bytes > max_bytes:
        raise ValueError(
            "max_bytes too small for structural floor of evidence-packet-v2"
        )

    full_payload = _full_payload(unit)
    original_bytes = len(_stable_json_bytes(full_payload))

    if original_bytes <= max_bytes:
        canonical_payload = full_payload
        truncated = False
    else:
        canonical_payload, truncated = _build_truncated_payload(unit=unit, max_bytes=max_bytes)

    canonical_bytes = _stable_json_bytes(canonical_payload)
    if len(canonical_bytes) > max_bytes:
        raise ValueError("max_bytes too small after deterministic truncation")

    canonical_json = canonical_bytes.decode("utf-8")
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    audit = canonical_payload["audit"]
    return EvidencePacket(
        version="evidence-packet-v2",
        canonical_json=canonical_json,
        sha256=digest,
        truncated=truncated,
        original_bytes=original_bytes,
        encoded_bytes=len(canonical_bytes),
        omitted_turn_count=audit["omitted_turn_count"],
        omitted_tool_call_count=audit["omitted_tool_call_count"],
    )
