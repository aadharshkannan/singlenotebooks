"""Durable JSON checkpoint and fenced reference store."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return asdict(value)
    return value


class LeaseConflictError(RuntimeError):
    pass


class FenceRejectedError(RuntimeError):
    pass


class CheckpointTransitionError(RuntimeError):
    pass


class CheckpointCASConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class Lease:
    pipeline_id: str
    holder: str
    generation: int
    acquired_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class BatchCheckpoint:
    pipeline_id: str
    batch_id: str
    status: str
    previous_successful_watermark: datetime
    cutoff: datetime
    elapsed_minutes: float
    nominal_budget_tokens: int
    effective_budget_tokens: int
    seed: str
    frame_hash: str
    config_hash: str
    membership_hash: str
    selected_ids: tuple[str, ...]
    planned_usage_tokens: int
    actual_usage_tokens: int
    retry_count: int
    fairness_state: dict[str, Any]
    created_at: datetime
    completed_at: datetime | None = None


class JsonReferenceStore:
    """Small durable JSON lease + claim store.

    This is a single-process reference adapter for tests/examples. Writes are
    atomic at file level via temp-file + replace, but this is not a distributed
    lease/claim system.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "reference_store.json"
        if not self.path.exists():
            self._write({"leases": {}, "claims": {}})

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data, indent=2, default=_json_default)
        self._atomic_write_text(self.path, payload)

    def _atomic_write_text(self, path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def acquire_lease(
        self,
        *,
        pipeline_id: str,
        holder: str,
        ttl: timedelta,
        now: datetime | None = None,
    ) -> Lease:
        now_utc = (now or _utc_now()).astimezone(timezone.utc)
        data = self._read()
        leases = data.setdefault("leases", {})
        current = leases.get(pipeline_id)
        if current is not None:
            current_expires = _parse_dt(current["expires_at"])
            if current_expires > now_utc and current["holder"] != holder:
                raise LeaseConflictError("lease already held by another holder")
            generation = int(current["generation"]) + 1
        else:
            generation = 1

        lease = Lease(
            pipeline_id=pipeline_id,
            holder=holder,
            generation=generation,
            acquired_at=now_utc,
            expires_at=now_utc + ttl,
        )
        leases[pipeline_id] = {
            "holder": holder,
            "generation": generation,
            "acquired_at": lease.acquired_at.isoformat(),
            "expires_at": lease.expires_at.isoformat(),
        }
        self._write(data)
        return lease

    def assert_fence(self, lease: Lease) -> None:
        data = self._read()
        current = data.get("leases", {}).get(lease.pipeline_id)
        if current is None:
            raise FenceRejectedError("missing lease")
        if int(current["generation"]) != lease.generation:
            raise FenceRejectedError("stale lease generation")
        if current["holder"] != lease.holder:
            raise FenceRejectedError("stale lease holder")

    def claim_request(self, *, request_id: str, batch_id: str) -> bool:
        data = self._read()
        claims = data.setdefault("claims", {})
        if request_id in claims and claims[request_id] != batch_id:
            return False
        claims[request_id] = batch_id
        self._write(data)
        return True

    def release_claim(self, *, request_id: str, batch_id: str) -> bool:
        data = self._read()
        claims = data.setdefault("claims", {})
        if claims.get(request_id) != batch_id:
            return False
        del claims[request_id]
        self._write(data)
        return True


class JsonCheckpointStore:
    """Simple durable JSON checkpoint store with deterministic retry reuse.

    This is a single-process reference adapter. It provides atomic file writes
    but does not provide cross-process transactional guarantees.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "checkpoint_state.json"
        if not self.state_path.exists():
            self._write_state(
                {
                    "latest_successful_watermark": None,
                    "batches": {},
                    "prepared_by_fingerprint": {},
                }
            )

    def _read_state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data, indent=2, default=_json_default)
        self._atomic_write_text(self.state_path, payload)

    def _atomic_write_text(self, path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def latest_successful_watermark(self) -> datetime | None:
        state = self._read_state()
        value = state.get("latest_successful_watermark")
        return _parse_dt(value) if value else None

    def prepare(
        self,
        checkpoint: BatchCheckpoint,
        *,
        frame_hash: str,
        config_hash: str,
        seed: str,
    ) -> BatchCheckpoint:
        state = self._read_state()
        fingerprint = f"{frame_hash}|{config_hash}|{seed}"
        existing_batch_id = state["prepared_by_fingerprint"].get(fingerprint)
        if existing_batch_id:
            existing = state["batches"][existing_batch_id]
            return self._from_json(existing)

        batch_json = self._to_json(checkpoint)
        batch_json["status"] = "PREPARED"
        state["batches"][checkpoint.batch_id] = batch_json
        state["prepared_by_fingerprint"][fingerprint] = checkpoint.batch_id
        self._write_state(state)
        return self._from_json(batch_json)

    def mark_running(self, batch_id: str) -> None:
        state = self._read_state()
        batch = state["batches"][batch_id]
        status = batch["status"]
        if status != "PREPARED":
            raise CheckpointTransitionError(f"illegal transition {status} -> RUNNING")
        batch["status"] = "RUNNING"
        self._write_state(state)

    def settle(self, batch_id: str, *, actual_usage_tokens: int) -> None:
        state = self._read_state()
        batch = state["batches"][batch_id]
        status = batch["status"]
        if status != "RUNNING":
            raise CheckpointTransitionError(f"illegal transition {status} -> SETTLED")
        batch["actual_usage_tokens"] = actual_usage_tokens
        batch["status"] = "SETTLED"
        batch["completed_at"] = _utc_now().isoformat()
        self._write_state(state)

    def commit(
        self,
        batch_id: str,
        *,
        success: bool,
        new_watermark: datetime | None = None,
        expected_previous_watermark: datetime | None = None,
    ) -> None:
        state = self._read_state()
        batch = state["batches"][batch_id]
        status = batch["status"]
        if status != "SETTLED":
            raise CheckpointTransitionError(f"illegal transition {status} -> {'COMMITTED' if success else 'FAILED'}")

        latest_value = state.get("latest_successful_watermark")
        latest = _parse_dt(latest_value) if latest_value else None
        expected = expected_previous_watermark.astimezone(timezone.utc) if expected_previous_watermark else None
        if latest != expected:
            raise CheckpointCASConflictError("expected previous watermark does not match latest successful watermark")

        if success:
            if new_watermark is None:
                raise ValueError("new_watermark is required on successful commit")
            watermark_utc = new_watermark.astimezone(timezone.utc)
            if latest is not None and watermark_utc < latest:
                raise CheckpointCASConflictError("new watermark must be monotonic and not move backward")
            batch["status"] = "COMMITTED"
            state["latest_successful_watermark"] = watermark_utc.isoformat()
        else:
            batch["status"] = "FAILED"
        self._write_state(state)

    def get(self, batch_id: str) -> BatchCheckpoint:
        state = self._read_state()
        return self._from_json(state["batches"][batch_id])

    def _to_json(self, checkpoint: BatchCheckpoint) -> dict[str, Any]:
        payload = asdict(checkpoint)
        for key in ("previous_successful_watermark", "cutoff", "created_at", "completed_at"):
            value = payload[key]
            if value is not None:
                payload[key] = value.isoformat()
        payload["selected_ids"] = list(checkpoint.selected_ids)
        return payload

    def _from_json(self, payload: dict[str, Any]) -> BatchCheckpoint:
        completed_at = payload.get("completed_at")
        return BatchCheckpoint(
            pipeline_id=payload["pipeline_id"],
            batch_id=payload["batch_id"],
            status=payload["status"],
            previous_successful_watermark=_parse_dt(payload["previous_successful_watermark"]),
            cutoff=_parse_dt(payload["cutoff"]),
            elapsed_minutes=float(payload["elapsed_minutes"]),
            nominal_budget_tokens=int(payload["nominal_budget_tokens"]),
            effective_budget_tokens=int(payload["effective_budget_tokens"]),
            seed=payload["seed"],
            frame_hash=payload["frame_hash"],
            config_hash=payload["config_hash"],
            membership_hash=payload["membership_hash"],
            selected_ids=tuple(payload.get("selected_ids", [])),
            planned_usage_tokens=int(payload["planned_usage_tokens"]),
            actual_usage_tokens=int(payload.get("actual_usage_tokens", 0)),
            retry_count=int(payload.get("retry_count", 0)),
            fairness_state=dict(payload.get("fairness_state", {})),
            created_at=_parse_dt(payload["created_at"]),
            completed_at=_parse_dt(completed_at) if completed_at else None,
        )
