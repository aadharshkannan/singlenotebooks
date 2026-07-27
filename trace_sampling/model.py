from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Iterator, Optional, Tuple


class FrozenMapping(Mapping):
    """Small hashable mapping used to keep frozen session events deeply immutable."""

    def __init__(self, values: Mapping):
        self._values = {key: _freeze(value) for key, value in values.items()}

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._values.items())))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class SessionEvent:
    """One ordered message or tool event in an agent session."""
    role: str
    text: str = ""
    tool_name: Optional[str] = None
    arguments: Optional[Mapping[str, Any]] = None
    output: Optional[str] = None

    def __post_init__(self):
        if self.arguments is not None:
            object.__setattr__(self, "arguments", FrozenMapping(self.arguments))


@dataclass(frozen=True)
class Trace:
    """One OTel-style agent trace (a job-to-be-done with tool spans)."""
    trace_id: int
    agent_id: str
    timestamp: float
    signature: Tuple[str, ...]  # ordered tool/span names = the variety key
    span_count: int
    duration_ms: float
    status: str  # "ok" | "error"
    concept_id: int = -1  # ground-truth latent concept; -1 = unlabeled. Scoring only.
    events: Tuple[SessionEvent, ...] = ()


@dataclass
class AgentConfig:
    """Ground-truth dials controlling an agent's synthetic behavior."""
    agent_id: str
    velocity: float       # Poisson arrival rate (traces per sim-second)
    vocab_size: int       # number of distinct signatures the agent can emit
    zipf_s: float         # skew of the signature distribution (higher = peakier)
    start_time: float = 0.0
    error_rate: float = 0.05
    tool_pool: Tuple[str, ...] = (
        "search", "read", "edit", "run", "plan", "test", "fetch", "write",
    )
