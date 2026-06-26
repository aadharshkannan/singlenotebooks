import math
from collections import defaultdict
from typing import Dict, List, Tuple

from .model import Trace


def _by_agent_signatures(traces: List[Trace]):
    out = defaultdict(lambda: defaultdict(int))
    for t in traces:
        out[t.agent_id][t.signature] += 1
    return out


def signature_coverage(stream: List[Trace], kept: List[Trace]) -> Dict[str, float]:
    truth = _by_agent_signatures(stream)
    keptd = _by_agent_signatures(kept)
    cov = {}
    for agent, sigs in truth.items():
        denom = len(sigs)
        num = len(set(keptd.get(agent, {}).keys()) & set(sigs.keys()))
        cov[agent] = num / denom if denom else 0.0
    return cov


def min_active_keep_rate(stream: List[Trace], kept: List[Trace],
                         window: float) -> float:
    """Minimum, across active agents, of kept/seen within tumbling windows.

    Time is split into non-overlapping windows of `window` seconds. An agent is
    active in a window if it emitted >=1 trace there. Returns the minimum
    kept-rate over all (agent, active-window) pairs. 0.0 means some active agent
    was fully starved in some window.
    """
    if not stream:
        return 0.0
    start = min(t.timestamp for t in stream)
    # Bucket by tumbling windows.
    seen = defaultdict(lambda: defaultdict(int))
    got = defaultdict(lambda: defaultdict(int))
    for t in stream:
        w = int((t.timestamp - start) // window)
        seen[w][t.agent_id] += 1
    for t in kept:
        w = int((t.timestamp - start) // window)
        got[w][t.agent_id] += 1
    worst = 1.0
    for w, agents in seen.items():
        for agent, n in agents.items():
            rate = got[w].get(agent, 0) / n
            worst = min(worst, rate)
    return worst


def representativeness(stream: List[Trace], kept: List[Trace]) -> Dict[str, Dict[str, float]]:
    truth = _by_agent_signatures(stream)
    keptd = _by_agent_signatures(kept)
    out = {}
    for agent, sigs in truth.items():
        keys = list(sigs.keys())
        t_total = sum(sigs.values())
        k_counts = keptd.get(agent, {})
        k_total = sum(k_counts.values())
        kl = 0.0
        tv = 0.0
        for s in keys:
            p = sigs[s] / t_total
            q = (k_counts.get(s, 0) / k_total) if k_total else 0.0
            tv += abs(p - q)
            if p > 0 and q > 0:
                kl += p * math.log(p / q)
            elif p > 0 and q == 0:
                kl += p * math.log(p / 1e-9)  # penalize missing mass
        out[agent] = {"kl": kl, "tv": 0.5 * tv}
    return out


def kept_rate_timeseries(kept: List[Trace], bucket: float) -> Tuple[List[float], List[float]]:
    if not kept:
        return [], []
    start = min(t.timestamp for t in kept)
    counts = defaultdict(int)
    for t in kept:
        b = int((t.timestamp - start) // bucket)
        counts[b] += 1
    bmax = max(counts.keys())
    times = [start + b * bucket for b in range(bmax + 1)]
    rates = [counts.get(b, 0) / bucket for b in range(bmax + 1)]  # per second
    return times, rates
