from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, v_measure_score


def concept_coverage(log: pd.DataFrame) -> float:
    present = set(log["concept_id"].unique())
    kept = set(log.loc[log["kept"], "concept_id"].unique())
    return len(kept & present) / max(1, len(present))


def redundancy_per_concept(log: pd.DataFrame) -> Dict[int, int]:
    kept = log[log["kept"]]
    return kept.groupby("concept_id").size().to_dict()


def cluster_agreement(log: pd.DataFrame) -> Tuple[float, float]:
    labels_true = log["concept_id"].to_numpy()
    labels_pred = log["variety_key"].astype("category").cat.codes.to_numpy()
    return (float(adjusted_rand_score(labels_true, labels_pred)),
            float(v_measure_score(labels_true, labels_pred)))


def novel_concept_latency(log: pd.DataFrame) -> Dict[int, float]:
    """Wall-clock latency (seconds) from a concept's first appearance to its first
    kept trace. inf if the concept is never kept."""
    out = {}
    for cid, grp in log.sort_values("timestamp").groupby("concept_id"):
        first_seen = grp["timestamp"].iloc[0]
        kept = grp[grp["kept"]]
        out[cid] = float(kept["timestamp"].iloc[0] - first_seen) if len(kept) else float("inf")
    return out


def novel_concept_latency_traces(log: pd.DataFrame) -> Dict[int, float]:
    """Number of traces of a concept observed BEFORE its first kept trace (0 if
    kept on first appearance). inf if never kept. Complements the seconds-based
    latency with a volume-based one, as the spec requires both units."""
    out = {}
    for cid, grp in log.sort_values("timestamp").reset_index(drop=True).groupby("concept_id"):
        grp = grp.reset_index(drop=True)
        kept_positions = grp.index[grp["kept"]].tolist()
        out[cid] = float(kept_positions[0]) if kept_positions else float("inf")
    return out


def cross_agent_unification(kept_log: pd.DataFrame, embeddings: np.ndarray,
                            tau: float) -> float:
    """Offline metric-only pass: globally leader-cluster KEPT traces' embeddings
    (ignoring agent scope) and report the fraction of concepts whose resulting
    global cluster spans >=2 agents."""
    if len(kept_log) != len(embeddings):
        raise ValueError("kept_log and embeddings must have the same row count")
    available = np.all(np.isfinite(embeddings), axis=1)
    kept_log = kept_log.loc[available].reset_index(drop=True)
    embeddings = embeddings[available]
    if kept_log.empty:
        return 0.0

    assign = []
    centers = []
    for i in range(len(kept_log)):
        v = embeddings[i]
        best = None
        for cvec, cidx in centers:
            s = float(v @ cvec / ((np.linalg.norm(v)*np.linalg.norm(cvec)) or 1.0))
            if best is None or s > best[1]:
                best = (cidx, s)
        if best is not None and best[1] >= tau:
            assign.append(best[0])
        else:
            assign.append(len(centers))
            centers.append((v, len(centers)))
    df = kept_log.copy()
    df["gcluster"] = assign
    ok = 0
    concepts = df["concept_id"].unique()
    for cid in concepts:
        sub = df[df["concept_id"] == cid]
        dom = sub["gcluster"].mode().iloc[0]
        spanning = sub[sub["gcluster"] == dom]["agent_id"].nunique()
        ok += 1 if spanning >= 2 else 0
    return ok / max(1, len(concepts))
