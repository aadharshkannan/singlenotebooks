from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Optional, Sequence, Tuple
import unicodedata

from .model import SessionEvent, Trace


CANONICAL_POLICY = "complete_session_evidence_weighted_truncate"
CANONICAL_VERSION = "2.0"
DEFAULT_MAX_UTF8_BYTES = 32768

_EVIDENCE_ALLOCATION_WEIGHTS = {
    "system_context": 6,
    "initial_user_goal": 4,
    "later_user_refinement": 3,
    "final_assistant_outcome": 8,
    "tool_result": 7,
    "tool_arguments": 2,
    "remaining_turn_content": 2,
}
_ALLOCATION_MIN_CHUNK_BYTES = 32
_MANDATORY_EVIDENCE_MIN_BYTES = 32


class RepresentationError(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalizationOptions:
    max_utf8_bytes: int = DEFAULT_MAX_UTF8_BYTES
    policy: str = CANONICAL_POLICY
    version: str = CANONICAL_VERSION

    def __post_init__(self) -> None:
        if self.max_utf8_bytes <= 0:
            raise RepresentationError("max_utf8_bytes must be > 0")
        if self.policy != CANONICAL_POLICY or self.version != CANONICAL_VERSION:
            raise RepresentationError(
                f"unsupported canonical representation {self.policy!r} version "
                f"{self.version!r}; expected {CANONICAL_POLICY!r} version "
                f"{CANONICAL_VERSION!r}"
            )


@dataclass(frozen=True)
class RepresentationAudit:
    policy: str
    version: str
    original_utf8_bytes: int
    emitted_utf8_bytes: int
    truncated: bool


@dataclass(frozen=True)
class NormalizedRepresentation:
    canonical_json: str
    audit: RepresentationAudit

    @property
    def policy(self) -> str:
        return self.audit.policy

    @property
    def version(self) -> str:
        return self.audit.version

    @property
    def original_utf8_bytes(self) -> int:
        return self.audit.original_utf8_bytes

    @property
    def emitted_utf8_bytes(self) -> int:
        return self.audit.emitted_utf8_bytes

    @property
    def truncated(self) -> bool:
        return self.audit.truncated


def canonicalize_session_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN and infinity are not supported")
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    if isinstance(value, Mapping):
        canonical = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("argument object keys must be strings")
            canonical[canonicalize_session_value(key)] = canonicalize_session_value(item)
        return canonical
    if isinstance(value, (list, tuple)):
        return [canonicalize_session_value(item) for item in value]
    raise ValueError(f"unsupported session value type: {type(value).__name__}")


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _truncate_to_utf8_boundary(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    end = max_bytes
    while end > 0 and (encoded[end] & 0b11000000) == 0b10000000:
        end -= 1
    return encoded[:end].decode("utf-8", errors="strict")


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _events_for_trace(trace: Trace) -> Tuple[SessionEvent, ...]:
    if trace.events:
        return trace.events
    return tuple(SessionEvent(role="tool", tool_name=name) for name in trace.signature)


def _arguments_json(arguments: Optional[Mapping[str, Any]]) -> Optional[str]:
    if arguments is None:
        return None
    return _canonical_json_bytes(canonicalize_session_value(arguments)).decode("utf-8")


def _trace_payload(trace: Trace, events: Sequence[SessionEvent]) -> dict:
    return {
        "policy": CANONICAL_POLICY,
        "version": CANONICAL_VERSION,
        "session": {
            "agent_id": canonicalize_session_value(trace.agent_id),
            "duration_ms": canonicalize_session_value(trace.duration_ms),
            "events": [
                {
                    "arguments_json": _arguments_json(event.arguments),
                    "event_index": event_index,
                    "output": canonicalize_session_value(event.output),
                    "role": canonicalize_session_value(event.role),
                    "text": canonicalize_session_value(event.text),
                    "tool_name": canonicalize_session_value(event.tool_name),
                }
                for event_index, event in enumerate(events)
            ],
            "signature": canonicalize_session_value(trace.signature),
            "span_count": canonicalize_session_value(trace.span_count),
            "status": canonicalize_session_value(trace.status),
            "timestamp": canonicalize_session_value(trace.timestamp),
            "trace_id": canonicalize_session_value(trace.trace_id),
        },
    }


def normalize_trace(
    trace: Trace,
    *,
    max_utf8_bytes: int = DEFAULT_MAX_UTF8_BYTES,
    policy: str = CANONICAL_POLICY,
    version: str = CANONICAL_VERSION,
) -> NormalizedRepresentation:
    options = CanonicalizationOptions(max_utf8_bytes, policy, version)
    events = _events_for_trace(trace)
    payload = _trace_payload(trace, events)
    payload["policy"] = options.policy
    payload["version"] = options.version

    original_bytes = _canonical_json_bytes(payload)
    if len(original_bytes) <= options.max_utf8_bytes:
        audit = RepresentationAudit(
            options.policy,
            options.version,
            len(original_bytes),
            len(original_bytes),
            False,
        )
        return NormalizedRepresentation(original_bytes.decode("utf-8"), audit)

    truncated_payload = _trace_payload(trace, events)
    truncated_payload["policy"] = options.policy
    truncated_payload["version"] = options.version
    truncated_events = truncated_payload["session"]["events"]
    for event in truncated_events:
        event["text"] = ""
        if event["arguments_json"] is not None:
            event["arguments_json"] = ""
        if event["output"] is not None:
            event["output"] = ""

    budget = options.max_utf8_bytes

    def current_size() -> int:
        return len(_canonical_json_bytes(truncated_payload))

    if current_size() > budget:
        raise RepresentationError(
            "max_utf8_bytes too small for non-content canonical structure"
        )

    @dataclass
    class _Segment:
        category: str
        container: dict
        key: str
        source: str
        weight: int
        order: int
        mandatory: bool
        assigned_bytes: int = 0

        @property
        def max_bytes(self) -> int:
            return _utf8_len(self.source)

        def is_full(self) -> bool:
            return self.assigned_bytes >= self.max_bytes

    first_user_index = next(
        (index for index, event in enumerate(events) if event.role == "user"),
        None,
    )
    last_assistant_index = next(
        (index for index in range(len(events) - 1, -1, -1) if events[index].role == "assistant"),
        None,
    )

    segments = []
    order = 0
    mandatory_categories = {
        "system_context",
        "initial_user_goal",
        "later_user_refinement",
        "final_assistant_outcome",
        "tool_result",
    }
    for event_index, event_payload in enumerate(truncated_events):
        source_event = events[event_index]
        if source_event.role == "system":
            category = "system_context"
        elif first_user_index is not None and event_index == first_user_index:
            category = "initial_user_goal"
        elif source_event.role == "user":
            category = "later_user_refinement"
        elif last_assistant_index is not None and event_index == last_assistant_index:
            category = "final_assistant_outcome"
        elif source_event.role == "tool":
            category = "tool_result"
        else:
            category = "remaining_turn_content"

        segments.append(
            _Segment(
                category,
                event_payload,
                "text",
                canonicalize_session_value(source_event.text),
                _EVIDENCE_ALLOCATION_WEIGHTS[category],
                order,
                category in mandatory_categories,
            )
        )
        order += 1

        arguments_json = _arguments_json(source_event.arguments)
        if arguments_json is not None:
            segments.append(
                _Segment(
                    "tool_arguments",
                    event_payload,
                    "arguments_json",
                    arguments_json,
                    _EVIDENCE_ALLOCATION_WEIGHTS["tool_arguments"],
                    order,
                    False,
                )
            )
            order += 1
        if source_event.output is not None:
            segments.append(
                _Segment(
                    "tool_result",
                    event_payload,
                    "output",
                    canonicalize_session_value(source_event.output),
                    _EVIDENCE_ALLOCATION_WEIGHTS["tool_result"],
                    order,
                    True,
                )
            )
            order += 1

    category_rank = {
        "final_assistant_outcome": 0,
        "tool_result": 1,
        "system_context": 2,
        "initial_user_goal": 3,
        "later_user_refinement": 4,
        "tool_arguments": 5,
        "remaining_turn_content": 6,
    }
    segments.sort(
        key=lambda segment: (
            category_rank[segment.category],
            -segment.order if segment.category == "tool_result" else segment.order,
        )
    )

    def assign_segment_bytes(segment: _Segment, target_bytes: int) -> int:
        text = _truncate_to_utf8_boundary(segment.source, max(0, target_bytes))
        segment.container[segment.key] = text
        segment.assigned_bytes = _utf8_len(text)
        return segment.assigned_bytes

    def grow_with_budget(segment: _Segment, max_additional_bytes: int) -> int:
        if max_additional_bytes <= 0 or segment.is_full():
            return 0
        low = segment.assigned_bytes
        high = min(segment.max_bytes, segment.assigned_bytes + max_additional_bytes)
        best = segment.assigned_bytes
        prior = segment.assigned_bytes
        while low <= high:
            middle = (low + high) // 2
            assign_segment_bytes(segment, middle)
            if current_size() <= budget:
                best = segment.assigned_bytes
                low = middle + 1
            else:
                high = middle - 1
        assign_segment_bytes(segment, best)
        return segment.assigned_bytes - prior

    for segment in segments:
        if segment.mandatory and segment.source:
            assign_segment_bytes(segment, min(_MANDATORY_EVIDENCE_MIN_BYTES, segment.max_bytes))
    if current_size() > budget:
        raise RepresentationError(
            "max_utf8_bytes too small for mandatory task-completion evidence"
        )

    while True:
        available_segments = [segment for segment in segments if not segment.is_full()]
        if not available_segments:
            break
        free_bytes = budget - current_size()
        if free_bytes <= 0:
            break

        total_weight = sum(segment.weight for segment in available_segments)
        progressed = False
        for segment in available_segments:
            free_bytes = budget - current_size()
            if free_bytes <= 0:
                break
            weighted_share = (free_bytes * segment.weight) // max(1, total_weight)
            requested = min(free_bytes, max(_ALLOCATION_MIN_CHUNK_BYTES, weighted_share))
            if grow_with_budget(segment, requested) > 0:
                progressed = True

        if progressed:
            continue
        for segment in available_segments:
            if grow_with_budget(segment, 1) > 0:
                progressed = True
                break
        if not progressed:
            break

    emitted_bytes = _canonical_json_bytes(truncated_payload)
    if len(emitted_bytes) > budget:
        raise RepresentationError(
            "failed to produce canonical output within max_utf8_bytes"
        )

    audit = RepresentationAudit(
        options.policy,
        options.version,
        len(original_bytes),
        len(emitted_bytes),
        True,
    )
    return NormalizedRepresentation(emitted_bytes.decode("utf-8"), audit)


class SessionEvidencePacketBuilder:
    """Build each trace's bounded packet once and share it across adapters."""

    def __init__(
        self,
        options: Optional[CanonicalizationOptions] = None,
        max_size: int = 4096,
    ):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.options = options or CanonicalizationOptions()
        self._max_size = max_size
        self._cache: "OrderedDict[int, Tuple[Trace, NormalizedRepresentation]]" = OrderedDict()
        self.n_builds = 0
        self.n_hits = 0

    def build(self, trace: Trace) -> NormalizedRepresentation:
        cache_key = id(trace)
        cached = self._cache.get(cache_key)
        if cached is not None and cached[0] is trace:
            self._cache.move_to_end(cache_key)
            self.n_hits += 1
            return cached[1]

        representation = normalize_trace(
            trace,
            max_utf8_bytes=self.options.max_utf8_bytes,
            policy=self.options.policy,
            version=self.options.version,
        )
        self.n_builds += 1
        self._cache[cache_key] = (trace, representation)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return representation

    def peek(self, trace: Trace) -> Optional[NormalizedRepresentation]:
        cached = self._cache.get(id(trace))
        if cached is None or cached[0] is not trace:
            return None
        return cached[1]