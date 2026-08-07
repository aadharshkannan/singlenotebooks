from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict, Tuple
import numpy as np

from .model import Trace
from .stats import AgentStats
from .reservoir import WeightedReservoir
from .backpressure import BackpressureController
from .variety import VarietyIndex, VarietyObservation, VarietyKey, ExactSignatureIndex


@dataclass
class SamplerConfig:
    llm_throughput: float = 50.0
    agent_floor: float = 0.02          # guaranteed per-active-agent budget share
    active_window: float = 30.0
    max_signatures_per_agent: int = 256
    max_reservoirs: int = 4096         # global LRU cap on (agent, sig) reservoirs
    coldstart_min_samples: int = 20
    coldstart_boost: float = 5.0
    ewma_alpha: float = 0.1
    aimd_increase: float = 0.05
    aimd_decrease: float = 0.5
    queue_high_factor: float = 2.0
    queue_low_factor: float = 0.5
    reservoir_size: int = 8
    min_multiplier: float = 0.01
    enforce_keep_one_floor: bool = True


class BaselineSampler:
    """Strategy A: fixed global keep probability, no adaptation."""

    def __init__(self, keep_prob: float, seed: int = 0):
        self.keep_prob = keep_prob
        self._rng = np.random.default_rng(seed)

    def decide(self, trace: Trace) -> bool:
        return bool(self._rng.random() < self.keep_prob)


class AdaptiveSampler:
    """Strategy B: stratified, diversity-weighted, floored, backpressure-aware.
    
    AdaptiveSampler holds two AgentStats sources per agent:
      * self._stats — velocity, coldstart, and active-agent counting (sampler-owned)
      * the injected VarietyIndex (default ExactSignatureIndex) — rarity/novelty + stratum key
    This separation lets the variety mechanism be swapped (exact-match vs embedding clusters)
    without touching the sampler's velocity/floor logic.
    """

    def __init__(self, config: SamplerConfig, seed: int = 0, 
                 variety_index: "VarietyIndex | None" = None, 
                 use_novelty: bool = False):  # If True, diversity uses max(rarity, novelty); else rarity only.
        self.cfg = config
        self._rng = np.random.default_rng(seed)
        self._stats: Dict[str, AgentStats] = {}
        # LRU-bounded map of (agent, VarietyKey) -> reservoir; keeps memory bounded.
        self._reservoirs: "OrderedDict[Tuple[str, VarietyKey], WeightedReservoir]" = OrderedDict()
        self._last_kept_ts: Dict[str, float] = {}
        self._last_seen_ts: Dict[str, float] = {}   # for active-agent counting
        self._bp = BackpressureController(
            throughput=config.llm_throughput,
            queue_high=config.llm_throughput * config.queue_high_factor,
            queue_low=config.llm_throughput * config.queue_low_factor,
            aimd_increase=config.aimd_increase,
            aimd_decrease=config.aimd_decrease,
            min_multiplier=config.min_multiplier,
        )
        self._res_seed = seed
        self._variety = variety_index or ExactSignatureIndex(max_signatures_per_agent=config.max_signatures_per_agent)
        self._use_novelty = use_novelty
        self.last_observation = None
        self.last_proposed_keep: bool | None = None
        self.last_admitted_keep: bool | None = None

    def _stats_for(self, agent_id: str) -> AgentStats:
        if agent_id not in self._stats:
            self._stats[agent_id] = AgentStats(
                coldstart_min_samples=self.cfg.coldstart_min_samples,
                max_signatures=self.cfg.max_signatures_per_agent,
                ewma_alpha=self.cfg.ewma_alpha,
            )
        return self._stats[agent_id]

    def _reservoir_for(self, key) -> WeightedReservoir:
        if key in self._reservoirs:
            self._reservoirs.move_to_end(key)
            return self._reservoirs[key]
        self._res_seed += 1
        res = WeightedReservoir(capacity=self.cfg.reservoir_size, seed=self._res_seed)
        self._reservoirs[key] = res
        if len(self._reservoirs) > self.cfg.max_reservoirs:
            self._reservoirs.popitem(last=False)  # evict least-recently-used
        return res

    def _active_agent_count(self, now: float) -> int:
        w = self.cfg.active_window
        return sum(1 for ts in self._last_seen_ts.values() if now - ts <= w) or 1

    def decide(self, trace: Trace, admit_keep: bool | None = None) -> bool:
        cfg = self.cfg
        self._bp.tick(trace.timestamp)
        stats = self._stats_for(trace.agent_id)
        stats.observe(trace.timestamp, trace.signature)   # velocity/coldstart only
        self._last_seen_ts[trace.agent_id] = trace.timestamp

        obs = self._variety.observe(trace)                # variety via index
        self.last_observation = obs                       # expose for eval harness
        key = (trace.agent_id, obs.key)                   # stratify by VarietyKey
        reservoir = self._reservoir_for(key)

        # Diversity score: rarer signatures and under-filled strata score higher.
        fill = len(reservoir) / max(cfg.reservoir_size, 1)
        base = max(obs.rarity, obs.novelty) if self._use_novelty else obs.rarity
        diversity = base * (1.0 - 0.5 * fill)

        # Velocity-based fair-share floor (spec step 4 precedence):
        # each active agent is guaranteed a share of the budget, scaled down
        # equally when too many agents are active so the total stays bounded.
        n_active = self._active_agent_count(trace.timestamp)
        guaranteed_rate = min(cfg.agent_floor * cfg.llm_throughput,
                              cfg.llm_throughput / n_active)
        velocity = max(stats.velocity(), 1e-6)
        floor_prob = min(1.0, guaranteed_rate / velocity)

        # Deterministic keep-one floor: if this agent has had no keep in the
        # trailing active_window, keep this trace (the hard anti-starvation
        # guarantee). Keep-one volume is bounded by one per agent per window,
        # negligible vs llm_throughput.
        last_kept = self._last_kept_ts.get(trace.agent_id)
        stale = last_kept is None or (trace.timestamp - last_kept) >= cfg.active_window

        if stale and cfg.enforce_keep_one_floor:
            proposed_keep = True
        else:
            boost = cfg.coldstart_boost if stats.is_coldstart() else 1.0
            prob = diversity * boost * self._bp.multiplier
            prob = max(prob, floor_prob)   # probabilistic budget-share floor
            prob = min(prob, 1.0)
            proposed_keep = bool(self._rng.random() < prob)

        admitted_keep = proposed_keep and (True if admit_keep is None else bool(admit_keep))
        self.last_proposed_keep = proposed_keep
        self.last_admitted_keep = admitted_keep

        if admitted_keep:
            reservoir.offer(item=trace.trace_id, weight=max(diversity, 1e-6))
            self._last_kept_ts[trace.agent_id] = trace.timestamp
            self._bp.on_kept()
        return admitted_keep
