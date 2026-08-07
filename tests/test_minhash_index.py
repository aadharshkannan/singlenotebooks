from __future__ import annotations

import pytest

from minhash_sampling import MinHashClusterIndex, MinHashConfig
from minhash_sampling.signature import MinHashBuildError, MinHashSignatureProvider
from trace_sampling.model import SessionEvent, Trace
from trace_sampling.representation import RepresentationError
from trace_sampling.samplers import AdaptiveSampler, SamplerConfig


def _trace(
    trace_id: int,
    *,
    agent: str = "agent-a",
    ts: float = 0.0,
    text: str = "reset account password",
    tool: str = "search",
    concept_id: int = 0,
    events: tuple[SessionEvent, ...] | None = None,
) -> Trace:
    ev = events
    if ev is None:
        ev = (
            SessionEvent(role="user", text=text),
            SessionEvent(role="tool", tool_name=tool, arguments={"q": text}, output="ok"),
        )
    return Trace(trace_id, agent, ts, (tool,), 1, 1.0, "ok", concept_id=concept_id, events=ev)


def test_cluster_join_vs_new_threshold_behavior() -> None:
    idx = MinHashClusterIndex(MinHashConfig(similarity_threshold=0.5, permutations=128, seed=13))
    a = idx.observe(_trace(1, ts=1.0, text="reset account password"))
    b = idx.observe(_trace(2, ts=2.0, text="reset account password quickly"))
    c = idx.observe(_trace(3, ts=3.0, text="issue customer refund"))

    assert a.key.kind == "cluster" and b.key.kind == "cluster" and c.key.kind == "cluster"
    assert b.key == a.key
    assert b.novelty == 0.0
    assert c.key != a.key
    assert c.novelty == 1.0


def test_per_agent_scope_same_content_different_agent_not_merged() -> None:
    idx = MinHashClusterIndex(MinHashConfig())
    a = idx.observe(_trace(1, agent="a", ts=1.0, text="shared wording"))
    b = idx.observe(_trace(2, agent="b", ts=1.0, text="shared wording"))
    assert a.key != b.key


def test_ttl_purge_reflags_returning_behavior_as_new() -> None:
    idx = MinHashClusterIndex(MinHashConfig(ttl_s=5.0, purge_every=1, similarity_threshold=0.5))
    first = idx.observe(_trace(1, ts=0.0, text="reset account"))
    later = idx.observe(_trace(2, ts=100.0, text="reset account"))
    assert later.key != first.key
    assert later.novelty == 1.0


def test_lru_caps_enforced_per_agent_and_global() -> None:
    cfg = MinHashConfig(max_clusters_per_agent=3, max_clusters_total=4, similarity_threshold=1.0)
    idx = MinHashClusterIndex(cfg)
    # threshold=1.0 plus distinct texts generates many new clusters.
    for i in range(6):
        idx.observe(_trace(i, agent="agent-a", ts=float(i), text=f"unique-a-{i}"))
    for i in range(3):
        idx.observe(_trace(100 + i, agent="agent-b", ts=10.0 + i, text=f"unique-b-{i}"))

    telem = idx.telemetry()
    assert telem["live_clusters"] <= 4
    assert len(idx._agent_clusters.get("agent-a", {})) <= 3
    assert telem["evictions"] >= 1


def test_known_build_error_falls_back_to_exact_signature() -> None:
    idx = MinHashClusterIndex(MinHashConfig())
    # No events and no signature -> MinHashBuildError -> fallback.
    t = Trace(1, "agent-a", 0.0, (), 0, 1.0, "ok", events=())
    obs = idx.observe(t)
    assert obs.key.kind == "fallback-signature"
    assert idx.n_fallbacks >= 1


def test_representation_error_propagates() -> None:
    class _BrokenProvider(MinHashSignatureProvider):
        def _bounded_event_fields(self, trace: Trace) -> tuple[list[str], str, bool]:
            raise RepresentationError("representation failure")

    idx = MinHashClusterIndex(MinHashConfig(), signature_provider=_BrokenProvider(MinHashConfig()))
    with pytest.raises(RepresentationError):
        idx.observe(_trace(1))


def test_monotonic_late_timestamp_per_agent() -> None:
    idx = MinHashClusterIndex(MinHashConfig(similarity_threshold=0.5))
    first = idx.observe(_trace(1, ts=10.0, text="same"))
    second = idx.observe(_trace(2, ts=5.0, text="same"))  # late out-of-order ts
    assert second.key == first.key
    assert second.rarity >= 0.0


def test_telemetry_counters_move() -> None:
    idx = MinHashClusterIndex(MinHashConfig(ttl_s=1.0, purge_every=1, similarity_threshold=0.5))
    idx.observe(_trace(1, ts=0.0, text="one"))
    idx.observe(_trace(2, ts=0.2, text="one"))
    idx.observe(_trace(3, ts=10.0, text="two"))
    t = idx.telemetry()
    assert t["builds"] >= 2
    assert t["comparisons"] >= 1
    assert t["clusters"] >= 2
    assert t["purges"] >= 1


def test_adaptive_sampler_integration_with_minhash_index() -> None:
    idx = MinHashClusterIndex(MinHashConfig(similarity_threshold=0.5))
    sampler = AdaptiveSampler(
        SamplerConfig(llm_throughput=5.0),
        seed=13,
        variety_index=idx,
        use_novelty=True,
    )
    keep = sampler.decide(_trace(1, ts=1.0, text="reset account"))
    assert isinstance(keep, bool)
    assert sampler.last_observation is not None
    assert sampler.last_observation.key.kind in {"cluster", "fallback-signature"}


def test_default_sampler_mode_gives_new_cluster_nonzero_rarity() -> None:
    idx = MinHashClusterIndex(MinHashConfig(similarity_threshold=0.5))
    sampler = AdaptiveSampler(
        SamplerConfig(llm_throughput=5.0),
        seed=13,
        variety_index=idx,
    )
    sampler.decide(_trace(1, ts=1.0, text="brand new behavior"))
    assert sampler.last_observation is not None
    assert sampler.last_observation.novelty == 1.0
    assert sampler.last_observation.rarity == 0.5
