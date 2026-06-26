from dataclasses import dataclass
from typing import Tuple


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
