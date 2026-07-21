from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .model import Trace, AgentConfig
from .concepts import ConceptSpec, SynonymMap, realize_concept


def _build_vocab(cfg: AgentConfig, rng: np.random.Generator):
    """Return (signatures, probabilities) for one agent."""
    signatures = []
    seen = set()
    # Generate distinct signatures (tuples of 1-4 tools).
    attempts = 0
    while len(signatures) < cfg.vocab_size and attempts < cfg.vocab_size * 50:
        attempts += 1
        length = int(rng.integers(1, 5))
        sig = tuple(rng.choice(cfg.tool_pool, size=length))
        if sig not in seen:
            seen.add(sig)
            signatures.append(sig)
    # Zipf weights over ranks 1..N.
    ranks = np.arange(1, len(signatures) + 1, dtype=float)
    weights = 1.0 / np.power(ranks, cfg.zipf_s)
    weights /= weights.sum()
    return signatures, weights


def generate_stream(configs: List[AgentConfig], duration: float,
                    seed: int = 0) -> List[Trace]:
    """Generate a time-sorted interleaved stream of traces across agents."""
    rng = np.random.default_rng(seed)
    events: List[Tuple[float, str, Tuple[str, ...], str]] = []
    for cfg in configs:
        signatures, weights = _build_vocab(cfg, rng)
        t = cfg.start_time
        mean_gap = 1.0 / cfg.velocity
        while True:
            t += rng.exponential(mean_gap)
            if t > duration:
                break
            idx = rng.choice(len(signatures), p=weights)
            sig = signatures[idx]
            status = "error" if rng.random() < cfg.error_rate else "ok"
            events.append((t, cfg.agent_id, sig, status))
    events.sort(key=lambda e: e[0])
    traces = []
    for i, (t, agent_id, sig, status) in enumerate(events):
        traces.append(Trace(
            trace_id=i,
            agent_id=agent_id,
            timestamp=t,
            signature=sig,
            span_count=len(sig),
            duration_ms=float(rng.uniform(5.0, 500.0)),
            status=status,
        ))
    return traces


@dataclass
class ConceptAgentConfig:
    """Agent configured to emit specific concepts with per-agent vocabulary."""
    agent_id: str
    velocity: float
    concept_ids: Tuple[int, ...]
    zipf_s: float = 1.0
    start_time: float = 0.0
    error_rate: float = 0.05
    vocab_bias: Optional[Dict[str, str]] = None


def generate_concept_stream(agents: List[ConceptAgentConfig], 
                            concepts: List[ConceptSpec], 
                            synonym_map: SynonymMap,
                            duration: float, seed: int = 0) -> List[Trace]:
    """Interleaved, time-sorted stream of concept-labeled traces across agents.
    
    Each agent emits traces labeled with concept IDs. The same concept is expressed
    differently across agents via vocab_bias, creating disjoint surface signatures.
    """
    rng = np.random.default_rng(seed)
    by_id = {c.concept_id: c for c in concepts}
    events = []
    
    for a in agents:
        ids = list(a.concept_ids)
        ranks = np.arange(1, len(ids) + 1, dtype=float)
        weights = 1.0 / np.power(ranks, a.zipf_s)
        weights /= weights.sum()
        
        t = a.start_time
        mean_gap = 1.0 / a.velocity
        while True:
            t += rng.exponential(mean_gap)
            if t > duration:
                break
            cid = ids[int(rng.choice(len(ids), p=weights))]
            sig = realize_concept(by_id[cid], synonym_map, rng, vocab_bias=a.vocab_bias)
            status = "error" if rng.random() < a.error_rate else "ok"
            events.append((t, a.agent_id, sig, status, cid))
    
    events.sort(key=lambda e: e[0])
    traces = []
    for i, (t, agent_id, sig, status, cid) in enumerate(events):
        traces.append(Trace(
            trace_id=i, agent_id=agent_id, timestamp=t, signature=sig,
            span_count=len(sig), duration_ms=float(rng.uniform(5.0, 500.0)),
            status=status, concept_id=cid,
        ))
    return traces
