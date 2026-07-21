import numpy as np
import pandas as pd
from trace_sampling.variety_metrics import (
    concept_coverage, redundancy_per_concept, cluster_agreement,
    novel_concept_latency, novel_concept_latency_traces, cross_agent_unification)

def _log():
    return pd.DataFrame([
        dict(timestamp=0.0, agent_id="a", concept_id=0, variety_key="c0", key_kind="cluster", kept=True),
        dict(timestamp=1.0, agent_id="a", concept_id=0, variety_key="c0", key_kind="cluster", kept=False),
        dict(timestamp=2.0, agent_id="a", concept_id=1, variety_key="c1", key_kind="cluster", kept=True),
        dict(timestamp=3.0, agent_id="b", concept_id=2, variety_key="c2", key_kind="cluster", kept=False),
    ])

def test_concept_coverage():
    cov = concept_coverage(_log())
    assert abs(cov - 2/3) < 1e-9

def test_redundancy_per_concept():
    r = redundancy_per_concept(_log())
    assert r[0] == 1

def test_cluster_agreement_perfect():
    ari, v = cluster_agreement(_log())
    assert ari == 1.0 and v == 1.0

def test_novel_concept_latency():
    lat = novel_concept_latency(_log())
    assert lat[0] == 0.0

def test_novel_concept_latency_traces():
    lat = novel_concept_latency_traces(_log())
    assert lat[0] == 0
    assert lat[2] == float("inf")

def test_cross_agent_unification():
    kept = pd.DataFrame([
        dict(agent_id="a", concept_id=0),
        dict(agent_id="b", concept_id=0),
        dict(agent_id="a", concept_id=1),
    ])
    emb = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=float)
    frac = cross_agent_unification(kept, emb, tau=0.9)
    assert abs(frac - 0.5) < 1e-9
