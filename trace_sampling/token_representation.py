from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from .model import SessionEvent, Trace
from .representation import canonicalize_session_value


CANONICAL_POLICY = "complete_session_evidence_weighted_token_truncate"
CANONICAL_VERSION = "3.0"
DEFAULT_MAX_TOKENS = 4096

_EVIDENCE_ALLOCATION_WEIGHTS = {
    "system_context": 6,
    "initial_user_goal": 4,
    "later_user_refinement": 3,
    "final_assistant_outcome": 8,
    "tool_result": 7,
    "tool_arguments": 2,
    "remaining_turn_content": 2,
}
_ALLOCATION_MIN_CHUNK_CHARS = 32
_MANDATORY_EVIDENCE_MIN_CHARS = 32


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class RepresentationError(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalizationOptions:
    tokenizer: Tokenizer
    max_tokens: int = DEFAULT_MAX_TOKENS
    policy: str = CANONICAL_POLICY
    version: str = CANONICAL_VERSION

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise RepresentationError("max_tokens must be > 0")
        if self.policy != CANONICAL_POLICY or self.version != CANONICAL_VERSION:
            raise RepresentationError(
                f"unsupported canonical representation {self.policy!r} version "
                f"{self.version!r}; expected {CANONICAL_POLICY!r} version "
                f"{CANONICAL_VERSION!r}"
            )

    @property
    def options_identity(self) -> str:
        tokenizer_name = getattr(self.tokenizer, "name", self.tokenizer.__class__.__name__)
        tokenizer_version = getattr(self.tokenizer, "version", "unknown")
        payload = json.dumps(
            {
                "max_tokens": self.max_tokens,
                "policy": self.policy,
                "tokenizer_name": tokenizer_name,
                "tokenizer_version": tokenizer_version,
                "version": self.version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RepresentationAudit:
    policy: str
    version: str
    original_tokens: int
    emitted_tokens: int
    max_tokens: int
    truncated: bool
    original_utf8_bytes: Optional[int] = None
    emitted_utf8_bytes: Optional[int] = None


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
    def original_tokens(self) -> int:
        return self.audit.original_tokens

    @property
    def emitted_tokens(self) -> int:
        return self.audit.emitted_tokens

    @property
    def max_tokens(self) -> int:
        return self.audit.max_tokens

    @property
    def truncated(self) -> bool:
        return self.audit.truncated


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _events_for_trace(trace: Trace) -> Tuple[SessionEvent, ...]:
    if trace.events:
        return trace.events
    return tuple(SessionEvent(role="tool", tool_name=name) for name in trace.signature)


def _arguments_json(arguments: Optional[Mapping[str, Any]]) -> Optional[str]:
    if arguments is None:
        return None
    return _canonical_json(canonicalize_session_value(arguments))


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
    tokenizer: Tokenizer,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    policy: str = CANONICAL_POLICY,
    version: str = CANONICAL_VERSION,
) -> NormalizedRepresentation:
    options = CanonicalizationOptions(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        policy=policy,
        version=version,
    )
    events = _events_for_trace(trace)
    payload = _trace_payload(trace, events)
    payload["policy"] = options.policy
    payload["version"] = options.version

    original_json = _canonical_json(payload)
    original_tokens = options.tokenizer.count(original_json)
    if original_tokens <= options.max_tokens:
        audit = RepresentationAudit(
            policy=options.policy,
            version=options.version,
            original_tokens=original_tokens,
            emitted_tokens=original_tokens,
            max_tokens=options.max_tokens,
            truncated=False,
            original_utf8_bytes=len(original_json.encode("utf-8")),
            emitted_utf8_bytes=len(original_json.encode("utf-8")),
        )
        return NormalizedRepresentation(original_json, audit)

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

    budget = options.max_tokens

    def current_json() -> str:
        return _canonical_json(truncated_payload)

    def current_tokens() -> int:
        return options.tokenizer.count(current_json())

    if current_tokens() > budget:
        raise RepresentationError("max_tokens too small for non-content canonical structure")

    @dataclass
    class _Segment:
        category: str
        container: dict
        key: str
        source: str
        weight: int
        order: int
        mandatory: bool
        assigned_chars: int = 0

        @property
        def max_chars(self) -> int:
            return len(self.source)

        def is_full(self) -> bool:
            return self.assigned_chars >= self.max_chars

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
                category=category,
                container=event_payload,
                key="text",
                source=canonicalize_session_value(source_event.text),
                weight=_EVIDENCE_ALLOCATION_WEIGHTS[category],
                order=order,
                mandatory=category in mandatory_categories,
            )
        )
        order += 1

        arguments_json = _arguments_json(source_event.arguments)
        if arguments_json is not None:
            segments.append(
                _Segment(
                    category="tool_arguments",
                    container=event_payload,
                    key="arguments_json",
                    source=arguments_json,
                    weight=_EVIDENCE_ALLOCATION_WEIGHTS["tool_arguments"],
                    order=order,
                    mandatory=False,
                )
            )
            order += 1

        if source_event.output is not None:
            segments.append(
                _Segment(
                    category="tool_result",
                    container=event_payload,
                    key="output",
                    source=canonicalize_session_value(source_event.output),
                    weight=_EVIDENCE_ALLOCATION_WEIGHTS["tool_result"],
                    order=order,
                    mandatory=True,
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

    def assign_segment_chars(segment: _Segment, target_chars: int) -> int:
        clamped = max(0, min(target_chars, segment.max_chars))
        segment.container[segment.key] = segment.source[:clamped]
        segment.assigned_chars = clamped
        return segment.assigned_chars

    def grow_with_budget(segment: _Segment, max_additional_chars: int) -> int:
        if max_additional_chars <= 0 or segment.is_full():
            return 0
        prior = segment.assigned_chars
        low = segment.assigned_chars
        high = min(segment.max_chars, segment.assigned_chars + max_additional_chars)
        best = segment.assigned_chars
        while low <= high:
            middle = (low + high) // 2
            assign_segment_chars(segment, middle)
            if current_tokens() <= budget:
                best = segment.assigned_chars
                low = middle + 1
            else:
                high = middle - 1
        assign_segment_chars(segment, best)
        return segment.assigned_chars - prior

    for segment in segments:
        if segment.mandatory and segment.source:
            assign_segment_chars(segment, min(_MANDATORY_EVIDENCE_MIN_CHARS, segment.max_chars))

    if current_tokens() > budget:
        raise RepresentationError(
            "max_tokens too small for mandatory task-completion evidence"
        )

    while True:
        available_segments = [segment for segment in segments if not segment.is_full()]
        if not available_segments:
            break
        free_tokens = budget - current_tokens()
        if free_tokens <= 0:
            break

        total_weight = sum(segment.weight for segment in available_segments)
        progressed = False
        for segment in available_segments:
            free_tokens = budget - current_tokens()
            if free_tokens <= 0:
                break
            weighted_share = (free_tokens * segment.weight) // max(1, total_weight)
            requested = max(_ALLOCATION_MIN_CHUNK_CHARS, weighted_share)
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

    emitted_json = current_json()
    emitted_tokens = options.tokenizer.count(emitted_json)
    if emitted_tokens > budget:
        raise RepresentationError("failed to produce canonical output within max_tokens")

    audit = RepresentationAudit(
        policy=options.policy,
        version=options.version,
        original_tokens=original_tokens,
        emitted_tokens=emitted_tokens,
        max_tokens=options.max_tokens,
        truncated=True,
        original_utf8_bytes=len(original_json.encode("utf-8")),
        emitted_utf8_bytes=len(emitted_json.encode("utf-8")),
    )
    return NormalizedRepresentation(emitted_json, audit)


class TokenSessionEvidencePacketBuilder:
    """Build and cache bounded token-based packets using content-addressed keys."""

    def __init__(
        self,
        options: CanonicalizationOptions,
        max_size: int = 4096,
    ):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.options = options
        self._max_size = max_size
        self._cache: "OrderedDict[tuple[str, str], NormalizedRepresentation]" = OrderedDict()
        self.n_builds = 0
        self.n_hits = 0

    def _cache_key(self, trace: Trace) -> tuple[str, str]:
        events = _events_for_trace(trace)
        payload = _trace_payload(trace, events)
        payload["policy"] = self.options.policy
        payload["version"] = self.options.version
        canonical = _canonical_json(payload)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return (digest, self.options.options_identity)

    def build(self, trace: Trace) -> NormalizedRepresentation:
        cache_key = self._cache_key(trace)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            self.n_hits += 1
            return cached

        representation = normalize_trace(
            trace,
            tokenizer=self.options.tokenizer,
            max_tokens=self.options.max_tokens,
            policy=self.options.policy,
            version=self.options.version,
        )
        self.n_builds += 1
        self._cache[cache_key] = representation
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return representation

    def peek(self, trace: Trace) -> Optional[NormalizedRepresentation]:
        return self._cache.get(self._cache_key(trace))