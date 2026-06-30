"""Adaptive backpressure trace sampling prototype."""
from .model import Trace, AgentConfig
from .generator import generate_stream
from .stats import AgentStats
from .reservoir import WeightedReservoir
from .backpressure import BackpressureController
from .samplers import SamplerConfig, BaselineSampler, AdaptiveSampler
from .metrics import (
    signature_coverage, min_active_keep_rate, representativeness,
    kept_rate_timeseries,
)

__all__ = [
    "Trace", "AgentConfig", "generate_stream", "AgentStats",
    "WeightedReservoir", "BackpressureController", "SamplerConfig",
    "BaselineSampler", "AdaptiveSampler", "signature_coverage",
    "min_active_keep_rate", "representativeness", "kept_rate_timeseries",
]
