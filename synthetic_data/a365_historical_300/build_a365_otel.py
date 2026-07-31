#!/usr/bin/env python3
"""Build an Agent 365-style OTLP/JSON trace copy of the BPS evaluation data."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "eval-harness" / "synthetic_observability.bps.json"
NATIVE_SOURCE = ROOT / "data" / "synthetic_observability.json"
OUTPUT = ROOT / "data" / "eval-harness" / "synthetic_observability.a365-otel.json"
METADATA_OUTPUT = OUTPUT.with_suffix(".meta.json")

SCOPE_NAME = "microsoft.agent365.synthetic.observability"
SCOPE_VERSION = "0.1.0"
SYNTHETIC_START = datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc)
TOOL_ERROR_PREFIX_RE = re.compile(
    r"^\s*(?:upstreamtimeout|timeouterror|internalservererror|serviceunavailable|"
    r"solvererror|timeout\s*:|http\s*5\d\d\b|5\d\d\b)",
    re.I,
)


def any_value(value: Any) -> dict:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def attributes(**values: Any) -> list[dict]:
    return [
        {"key": key, "value": any_value(value)}
        for key, value in values.items()
        if value is not None
    ]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "agent"


def trace_id(case_id: str) -> str:
    try:
        return uuid.UUID(case_id).hex
    except ValueError:
        return hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:32]


def span_id(seed: str) -> str:
    value = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return value if value != "0" * 16 else "0" * 15 + "1"


def unix_nano(value: datetime) -> str:
    return str(int(value.timestamp() * 1_000_000_000))


def parse_tool(raw: str) -> dict[str, str]:
    content = raw.strip()
    if content.startswith("[tool ") and content.endswith("]"):
        content = content[6:-1]
    name, separator, remainder = content.partition(" args=")
    if not separator:
        return {"name": "unknown", "arguments": "", "result": "", "raw": raw}

    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(remainder)
        arguments_text = remainder[:end]
        tail = remainder[end:]
        result_text = tail.removeprefix(" -> ") if tail.startswith(" -> ") else tail.lstrip()
    except json.JSONDecodeError:
        arguments_text, marker, result_text = remainder.partition(" -> ")
        if not marker:
            result_text = ""
    return {
        "name": name.strip() or "unknown",
        "arguments": arguments_text.strip(),
        "result": result_text.strip(),
        "raw": raw,
    }


def parse_turns(conversation: str) -> list[dict]:
    turns: list[dict] = []
    current: dict | None = None
    current_role: str | None = None

    for line in conversation.splitlines():
        if line.startswith("User:"):
            current = {"user": line.removeprefix("User:").lstrip(), "agent": "", "tools": []}
            turns.append(current)
            current_role = "user"
        elif line.startswith("Agent:"):
            if current is None:
                current = {"user": "", "agent": "", "tools": []}
                turns.append(current)
            current["agent"] = line.removeprefix("Agent:").lstrip()
            current_role = "agent"
        elif line.startswith("[tool "):
            if current is None:
                current = {"user": "", "agent": "", "tools": []}
                turns.append(current)
            current["tools"].append(parse_tool(line))
            current_role = "tool"
        elif current is not None and current_role in {"user", "agent"}:
            current[current_role] += "\n" + line
        elif current is not None and current_role == "tool" and current["tools"]:
            tool = current["tools"][-1]
            tool["raw"] += "\n" + line
            tool["result"] += "\n" + line
    return turns


def tool_result_is_error(result: str) -> bool:
    if TOOL_ERROR_PREFIX_RE.search(result):
        return True
    try:
        value = json.loads(result)
    except json.JSONDecodeError:
        return False
    if not isinstance(value, dict):
        return False
    status = str(value.get("status", "")).strip().lower()
    return status in {"error", "failed", "failure"} or bool(value.get("error"))


def base_span(trace: str, span: str, parent: str | None, name: str,
              start: datetime, end: datetime, attrs: list[dict], status: str = "STATUS_CODE_OK") -> dict:
    value = {
        "traceId": trace,
        "spanId": span,
        "traceState": "",
        "flags": 1,
        "name": name,
        "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": unix_nano(start),
        "endTimeUnixNano": unix_nano(end),
        "attributes": attrs,
        "droppedAttributesCount": 0,
        "events": [],
        "droppedEventsCount": 0,
        "links": [],
        "droppedLinksCount": 0,
        "status": {"code": status},
    }
    if parent:
        value["parentSpanId"] = parent
    return value


def native_metadata() -> tuple[dict[str, dict], dict[str, dict]]:
    records = json.loads(NATIVE_SOURCE.read_text(encoding="utf-8"))
    by_case = {record["conversation_id"]: record for record in records}
    by_agent = {
        record["agent_name"]: {
            "agent_id": record["agent_id"],
            "tenant_id": record["tenant_id"],
            "channel": record["channel"],
        }
        for record in records
    }
    return by_case, by_agent


def case_start(index: int, case: dict, native_by_case: dict[str, dict]) -> tuple[datetime, str]:
    native = native_by_case.get(case["id"])
    if native and native.get("started_at"):
        return datetime.fromisoformat(native["started_at"].replace("Z", "+00:00")), "native.started_at"
    return SYNTHETIC_START + timedelta(minutes=(index - 100) * 7), "deterministic.synthetic"


def make_case_spans(index: int, case: dict, agent_meta: dict,
                    native_by_case: dict[str, dict]) -> list[dict]:
    inputs = case["inputs"]
    turns = parse_turns(inputs["conversation"])
    trace = trace_id(case["id"])
    root_id = span_id(f"{case['id']}:root")
    started_at, timestamp_source = case_start(index, case, native_by_case)
    root_end = started_at + timedelta(seconds=max(1, len(turns) * 3))
    agent_id = agent_meta["agent_id"]

    root = base_span(
        trace,
        root_id,
        None,
        f"invoke_agent {agent_id}",
        started_at,
        root_end,
        attributes(
            **{
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.id": agent_id,
                "gen_ai.agent.name": inputs["agent"],
                "gen_ai.conversation.id": case["id"],
                "microsoft.agent365.tenant.id": agent_meta["tenant_id"],
                "microsoft.agent365.channel": agent_meta["channel"],
                "microsoft.agent365.system.message": inputs["system_message"],
                "microsoft.agent365.timestamp.source": timestamp_source,
                "evaluation.source.case_id": case["id"],
                "evaluation.source.index": index + 1,
                "evaluation.task": inputs["task"],
                "evaluation.domain": inputs["domain"],
                "evaluation.difficulty": inputs["difficulty"],
                "evaluation.expected": int(case["expected"]),
                "evaluation.expected.label": "pass" if case["expected"] == "1" else "fail",
                "evaluation.turn_count": inputs["turns"],
                "evaluation.synthetic": True,
            }
        ),
    )
    spans = [root]

    for turn_index, turn in enumerate(turns):
        turn_id = span_id(f"{case['id']}:turn:{turn_index}")
        turn_start = started_at + timedelta(seconds=turn_index * 3, milliseconds=100)
        turn_end = turn_start + timedelta(seconds=2, milliseconds=500)
        spans.append(
            base_span(
                trace,
                turn_id,
                root_id,
                f"chat {agent_id}",
                turn_start,
                turn_end,
                attributes(
                    **{
                        "gen_ai.operation.name": "chat",
                        "gen_ai.agent.id": agent_id,
                        "gen_ai.agent.name": inputs["agent"],
                        "gen_ai.conversation.id": case["id"],
                        "gen_ai.input.messages": json.dumps(
                            [{"role": "user", "content": turn["user"]}], ensure_ascii=False
                        ),
                        "gen_ai.output.messages": json.dumps(
                            [{"role": "assistant", "content": turn["agent"]}], ensure_ascii=False
                        ),
                        "microsoft.agent365.turn.index": turn_index,
                        "microsoft.agent365.turn.number": turn_index + 1,
                    }
                ),
            )
        )
        for tool_index, tool in enumerate(turn["tools"]):
            tool_id = span_id(f"{case['id']}:turn:{turn_index}:tool:{tool_index}")
            tool_start = turn_start + timedelta(milliseconds=300 + tool_index * 200)
            tool_end = tool_start + timedelta(milliseconds=150)
            is_error = tool_result_is_error(tool["result"])
            spans.append(
                base_span(
                    trace,
                    tool_id,
                    turn_id,
                    f"execute_tool {tool['name']}",
                    tool_start,
                    tool_end,
                    attributes(
                        **{
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.conversation.id": case["id"],
                            "gen_ai.tool.name": tool["name"],
                            "gen_ai.tool.type": "function",
                            "gen_ai.tool.call.id": tool_id,
                            "gen_ai.tool.call.arguments": tool["arguments"],
                            "gen_ai.tool.call.result": tool["result"],
                            "microsoft.agent365.tool.raw": tool["raw"],
                            "error.type": "tool_execution_error" if is_error else None,
                        }
                    ),
                    "STATUS_CODE_ERROR" if is_error else "STATUS_CODE_OK",
                )
            )
    return spans


def build_export(cases: list[dict]) -> dict:
    native_by_case, native_by_agent = native_metadata()
    grouped: OrderedDict[str, dict] = OrderedDict()

    for index, case in enumerate(cases):
        agent_name = case["inputs"]["agent"]
        agent_meta = native_by_agent.get(
            agent_name,
            {
                "agent_id": slugify(agent_name),
                "tenant_id": "synthetic-tenant",
                "channel": "synthetic",
            },
        )
        if agent_name not in grouped:
            grouped[agent_name] = {
                "resource": {
                    "attributes": attributes(
                        **{
                            "service.name": agent_meta["agent_id"],
                            "service.namespace": "microsoft.agent365.synthetic",
                            "service.version": "1.0.0",
                            "deployment.environment.name": "synthetic-evaluation",
                            "telemetry.sdk.name": "opentelemetry",
                            "telemetry.sdk.language": "python",
                            "telemetry.sdk.version": "1.x",
                            "gen_ai.agent.id": agent_meta["agent_id"],
                            "gen_ai.agent.name": agent_name,
                            "microsoft.agent365.tenant.id": agent_meta["tenant_id"],
                            "microsoft.agent365.channel": agent_meta["channel"],
                            "microsoft.agent365.data.classification": "synthetic",
                        }
                    ),
                    "droppedAttributesCount": 0,
                },
                "scopeSpans": [
                    {
                        "scope": {
                            "name": SCOPE_NAME,
                            "version": SCOPE_VERSION,
                            "attributes": attributes(
                                **{
                                    "microsoft.agent365.synthetic": True,
                                    "microsoft.agent365.official.semantic.conventions": False,
                                }
                            ),
                            "droppedAttributesCount": 0,
                        },
                        "spans": [],
                    }
                ],
            }
        grouped[agent_name]["scopeSpans"][0]["spans"].extend(
            make_case_spans(index, case, agent_meta, native_by_case)
        )

    return {"resourceSpans": list(grouped.values())}


def build_metadata(cases: list[dict], export: dict) -> dict:
    span_count = sum(
        len(scope["spans"])
        for resource in export["resourceSpans"]
        for scope in resource["scopeSpans"]
    )
    return {
        "format": "otlp-json",
        "profile": "unofficial-agent365-synthetic-evaluation",
        "officialMicrosoftSemanticConventions": False,
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "caseCount": len(cases),
        "spanCount": span_count,
        "generatedBy": "scripts/build_a365_otel.py",
        "notes": (
            "The trace document is a strict OTLP ExportTraceServiceRequest. Agent 365 resource and span "
            "attributes are unofficial synthetic conventions for this evaluation dataset. Trace/span IDs "
            "and non-native turn timestamps are deterministic; evaluation.expected preserves the BPS label."
        ),
    }


def main() -> None:
    cases = json.loads(SOURCE.read_text(encoding="utf-8"))
    export = build_export(cases)
    metadata = build_metadata(cases, export)
    OUTPUT.write_text(json.dumps(export, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    METADATA_OUTPUT.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} traces / {metadata['spanCount']} spans -> {OUTPUT}")
    print(f"Wrote export metadata -> {METADATA_OUTPUT}")


if __name__ == "__main__":
    main()