from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import unicodedata
from typing import Any, Iterable, Mapping, Optional

from trace_sampling.model import SessionEvent, Trace
from trace_sampling.representation import (
    CanonicalizationOptions,
    RepresentationError,
    SessionEvidencePacketBuilder,
    canonicalize_session_value,
)

from .config import MinHashConfig


_M61 = (1 << 61) - 1


class MinHashBuildError(ValueError):
    pass


@dataclass(frozen=True)
class MinHashRecord:
    content_sha256: str
    profile_id: str
    signature: tuple[int, ...]
    shingle_count: int
    representation_truncated: bool


def _stable_hash_u64(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _stable_hash_m61(text: str) -> int:
    return _stable_hash_u64(text) % _M61


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold().replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for ch in normalized:
        out.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return " ".join("".join(out).split())


def _iter_event_fields(event: SessionEvent) -> Iterable[str]:
    role = _normalize_text(event.role)
    if event.text:
        yield f"role:{role}|text:{_normalize_text(event.text)}"
    if event.tool_name:
        yield f"role:{role}|tool_name:{_normalize_text(event.tool_name)}"
    if event.arguments is not None:
        canonical_arguments = canonicalize_session_value(event.arguments)
        arguments_json = json.dumps(
            canonical_arguments,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        yield f"role:{role}|arguments:{_normalize_text(arguments_json)}"
    if event.output is not None:
        yield f"role:{role}|output:{_normalize_text(event.output)}"


def _fields_to_shingles(fields: Iterable[str], ngram_size: int) -> set[str]:
    shingles: set[str] = set()
    for field in fields:
        tokens = [tok for tok in field.split(" ") if tok]
        if not tokens:
            continue
        if len(tokens) < ngram_size:
            shingles.add(" ".join(tokens))
            continue
        for i in range(len(tokens) - ngram_size + 1):
            shingles.add(" ".join(tokens[i : i + ngram_size]))
    return shingles


def _jaccard_from_sets(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 1.0


def minhash_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        raise ValueError("minhash signatures must have equal length")
    agree = sum(1 for a, b in zip(left, right) if a == b)
    return agree / len(left)


class MinHashSignatureProvider:
    def __init__(
        self,
        config: Optional[MinHashConfig] = None,
        packet_builder: Optional[SessionEvidencePacketBuilder] = None,
    ) -> None:
        self.cfg = config or MinHashConfig()
        self._packet_builder = packet_builder or SessionEvidencePacketBuilder(
            options=CanonicalizationOptions(
                max_utf8_bytes=self.cfg.representation_max_utf8_bytes,
                policy=self.cfg.representation_policy,
                version=self.cfg.representation_version,
            ),
            max_size=self.cfg.cache_size,
        )
        self._cache: "OrderedDict[str, MinHashRecord]" = OrderedDict()
        self._debug_shingles: dict[str, set[str]] = {}
        self.n_builds = 0
        self.n_hits = 0
        self.n_truncations = 0

        seed_text = f"seed={self.cfg.seed}|perms={self.cfg.permutations}|n={self.cfg.ngram_size}"
        self._perm_a = tuple(((_stable_hash_u64(seed_text + f"|a|{i}") % (_M61 - 1)) + 1) for i in range(self.cfg.permutations))
        self._perm_b = tuple((_stable_hash_u64(seed_text + f"|b|{i}") % _M61) for i in range(self.cfg.permutations))

    def _cache_get(self, key: str) -> Optional[MinHashRecord]:
        item = self._cache.get(key)
        if item is None:
            return None
        self._cache.move_to_end(key)
        self.n_hits += 1
        return item

    def _cache_put(self, key: str, record: MinHashRecord) -> None:
        self._cache[key] = record
        self._cache.move_to_end(key)
        while len(self._cache) > self.cfg.cache_size:
            old_key, _ = self._cache.popitem(last=False)
            self._debug_shingles.pop(old_key, None)

    def _events_for_trace(self, trace: Trace) -> tuple[SessionEvent, ...]:
        if trace.events:
            return trace.events
        if trace.signature:
            # Fallback for traces without detailed events.
            return tuple(SessionEvent(role="tool", tool_name=name) for name in trace.signature)
        return ()

    def _bounded_event_fields(self, trace: Trace) -> tuple[list[str], str, bool]:
        packet = self._packet_builder.build(trace)
        if packet.truncated:
            self.n_truncations += 1
        payload = json.loads(packet.canonical_json)
        events = payload.get("session", {}).get("events", [])
        fields: list[str] = []
        content_events: list[dict[str, Any]] = []
        for row in events:
            role = _normalize_text(str(row.get("role") or ""))
            text = str(row.get("text") or "")
            tool_name = row.get("tool_name")
            arguments_json = row.get("arguments_json")
            output = row.get("output")
            if text:
                fields.append(f"role:{role}|text:{_normalize_text(text)}")
            if tool_name:
                fields.append(f"role:{role}|tool_name:{_normalize_text(str(tool_name))}")
            if arguments_json:
                fields.append(f"role:{role}|arguments:{_normalize_text(str(arguments_json))}")
            if output:
                fields.append(f"role:{role}|output:{_normalize_text(str(output))}")
            content_events.append(
                {
                    "role": role,
                    "text": text,
                    "tool_name": tool_name,
                    "arguments_json": arguments_json,
                    "output": output,
                }
            )
        content_json = json.dumps(
            content_events,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(content_json.encode("utf-8")).hexdigest()
        return fields, digest, packet.truncated

    def _shingles_from_fields(self, fields: Iterable[str]) -> set[str]:
        fields = list(fields)
        if not fields:
            raise MinHashBuildError("empty evidence")
        shingles = _fields_to_shingles(fields, self.cfg.ngram_size)
        if not shingles:
            raise MinHashBuildError("empty evidence")
        if len(shingles) > self.cfg.max_shingles:
            ranked = sorted((_stable_hash_u64(s), s) for s in shingles)
            shingles = {s for _, s in ranked[: self.cfg.max_shingles]}
        return shingles

    def _signature_from_shingles(self, shingles: set[str]) -> tuple[int, ...]:
        if not shingles:
            raise MinHashBuildError("empty evidence")
        hashed = tuple(_stable_hash_m61(shingle) for shingle in shingles)
        out: list[int] = []
        for a, b in zip(self._perm_a, self._perm_b):
            out.append(min((((a * x + b) % _M61) for x in hashed), default=0))
        return tuple(out)

    def build(self, trace: Trace) -> MinHashRecord:
        try:
            fields, content_sha, truncated = self._bounded_event_fields(trace)
        except RepresentationError:
            raise
        cache_key = f"{self.cfg.profile_id}|{content_sha}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        shingles = self._shingles_from_fields(fields)
        signature = self._signature_from_shingles(shingles)
        record = MinHashRecord(
            content_sha256=content_sha,
            profile_id=self.cfg.profile_id,
            signature=signature,
            shingle_count=len(shingles),
            representation_truncated=truncated,
        )
        self.n_builds += 1
        if self.cfg.retain_debug_shingles:
            self._debug_shingles[cache_key] = set(shingles)
        self._cache_put(cache_key, record)
        return record

    def shingle_jaccard(self, left: Trace, right: Trace) -> float:
        # Helper for calibration tests only.
        left_record = self.build(left)
        right_record = self.build(right)
        left_key = f"{self.cfg.profile_id}|{left_record.content_sha256}"
        right_key = f"{self.cfg.profile_id}|{right_record.content_sha256}"
        left_shingles = self._debug_shingles.get(left_key)
        right_shingles = self._debug_shingles.get(right_key)
        if left_shingles is None or right_shingles is None:
            left_shingles = self._shingles_from_fields(self._bounded_event_fields(left)[0])
            right_shingles = self._shingles_from_fields(self._bounded_event_fields(right)[0])
        return _jaccard_from_sets(left_shingles, right_shingles)
