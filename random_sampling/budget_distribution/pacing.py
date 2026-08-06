"""Deterministic virtual-time pacing for rolling TPM compliance."""
from __future__ import annotations

from dataclasses import dataclass

from .checkpoint import JsonReferenceStore


class UnpaceableReservationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reservation:
    request_id: str
    reserved_tokens: int
    scheduled_at_seconds: float


class RollingTokenPacer:
    """Virtual rolling-window governor for 20k TPM-style constraints."""

    def __init__(self, *, tpm_limit: int = 20_000, window_seconds: float = 60.0):
        if tpm_limit <= 0:
            raise ValueError("tpm_limit must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds
        self._events: list[tuple[str, float, int]] = []
        self._reservations: dict[str, Reservation] = {}
        self._actual_tokens: dict[str, int] = {}
        self._cursor = 0.0

    def _window_usage(self, at_seconds: float) -> int:
        lower = at_seconds - self.window_seconds
        return sum(tokens for _, ts, tokens in self._events if lower < ts <= at_seconds)

    def reserve(self, request_id: str, estimated_tokens: int) -> Reservation:
        if estimated_tokens <= 0:
            raise ValueError("estimated_tokens must be positive")
        if estimated_tokens > self.tpm_limit:
            raise UnpaceableReservationError(
                f"estimated_tokens {estimated_tokens} exceeds rolling tpm_limit {self.tpm_limit}"
            )
        if request_id in self._reservations:
            return self._reservations[request_id]

        t = self._cursor
        while self._window_usage(t) + estimated_tokens > self.tpm_limit:
            active_times = sorted(ts for _, ts, _ in self._events if ts > t - self.window_seconds)
            if not active_times:
                break
            t = active_times[0] + self.window_seconds

        reservation = Reservation(
            request_id=request_id,
            reserved_tokens=estimated_tokens,
            scheduled_at_seconds=t,
        )
        self._reservations[request_id] = reservation
        self._events.append((request_id, t, estimated_tokens))
        self._cursor = max(self._cursor, t)
        return reservation

    def reconcile(self, request_id: str, actual_tokens: int) -> int:
        if request_id not in self._reservations:
            raise KeyError(f"unknown reservation: {request_id}")
        if actual_tokens < 0:
            raise ValueError("actual_tokens must be non-negative")
        reservation = self._reservations[request_id]
        delta = actual_tokens - reservation.reserved_tokens
        self._actual_tokens[request_id] = actual_tokens

        updated: list[tuple[str, float, int]] = []
        for rid, ts, tokens in self._events:
            if rid == request_id:
                updated.append((rid, ts, actual_tokens))
            else:
                updated.append((rid, ts, tokens))
        self._events = updated
        return delta

    def build_schedule(self, requests: list[tuple[str, int]]) -> tuple[Reservation, ...]:
        return tuple(self.reserve(request_id, estimate) for request_id, estimate in requests)

    def is_tpm_compliant(self) -> bool:
        for _, ts, _ in self._events:
            if self._window_usage(ts) > self.tpm_limit:
                return False
        return True


def reserve_with_claim(
    *,
    pacer: RollingTokenPacer,
    reference_store: JsonReferenceStore,
    batch_id: str,
    request_id: str,
    estimated_tokens: int,
) -> Reservation:
    if not reference_store.claim_request(request_id=request_id, batch_id=batch_id):
        raise RuntimeError("request claim conflict")
    try:
        return pacer.reserve(request_id=request_id, estimated_tokens=estimated_tokens)
    except Exception:
        reference_store.release_claim(request_id=request_id, batch_id=batch_id)
        raise
