from __future__ import annotations

from minhash_sampling import BandedMinHashLSHIndex, MinHashConfig, MinHashClusterIndex
from minhash_sampling.signature import MinHashBuildError, MinHashSignatureProvider
from trace_sampling.model import SessionEvent, Trace


def _trace(
    trace_id: int,
    *,
    agent: str = "agent-a",
    ts: float = 0.0,
    text: str = "reset account password",
    tool: str = "search",
) -> Trace:
    events = (
        SessionEvent(role="user", text=text),
        SessionEvent(role="tool", tool_name=tool, arguments={"q": text}, output="ok"),
    )
    return Trace(trace_id, agent, ts, (tool,), 1, 1.0, "ok", concept_id=0, events=events)


def _cfg(**kwargs) -> MinHashConfig:
    base = dict(
        permutations=128,
        lsh_bands=32,
        lsh_rows=4,
        seed=13,
        similarity_threshold=0.5,
        ttl_s=90.0,
        purge_every=1,
    )
    base.update(kwargs)
    return MinHashConfig(**base)


def test_config_requires_bands_rows_match_permutations() -> None:
    try:
        MinHashConfig(permutations=128, lsh_bands=31, lsh_rows=4)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "lsh_bands * lsh_rows" in str(exc)


def test_lsh_candidate_join_matches_existing_leader() -> None:
    idx = BandedMinHashLSHIndex(_cfg())
    first = idx.observe(_trace(1, ts=1.0, text="reset account password"))
    second = idx.observe(_trace(2, ts=2.0, text="reset account password quickly"))

    assert first.key.kind == "cluster"
    assert second.key == first.key
    assert idx.telemetry()["candidate_lookups"] >= 1
    assert idx.telemetry()["comparisons"] >= 1


def test_lsh_tenant_agent_scope_isolation() -> None:
    idx = BandedMinHashLSHIndex(_cfg())
    first = idx.observe(_trace(1, agent="tenant-a|agent-a", ts=1.0, text="shared wording"))
    second = idx.observe(_trace(2, agent="tenant-b|agent-a", ts=2.0, text="shared wording"))
    third = idx.observe(_trace(3, agent="tenant-a|agent-a", ts=3.0, text="shared wording"))

    assert first.key != second.key
    assert third.key == first.key


def test_lsh_bounded_state_and_bucket_cleanup() -> None:
    idx = BandedMinHashLSHIndex(
        _cfg(max_clusters_per_agent=3, max_clusters_total=4, similarity_threshold=1.0)
    )
    for i in range(10):
        idx.observe(_trace(i, agent="a", ts=float(i), text=f"unique-a-{i}"))
    for i in range(10, 16):
        idx.observe(_trace(i, agent="b", ts=float(i), text=f"unique-b-{i}"))

    telemetry = idx.telemetry()
    assert telemetry["live_clusters"] <= 4
    for agent_id, buckets in idx._band_buckets.items():
        live_ids = set(idx._agent_clusters.get(agent_id, {}))
        referenced_ids = {cluster_id for members in buckets.values() for cluster_id in members}
        assert referenced_ids <= live_ids
    assert telemetry["evictions"] > 0


def test_lsh_signatures_deterministic_across_instances() -> None:
    cfg = _cfg()
    a = BandedMinHashLSHIndex(cfg)
    b = BandedMinHashLSHIndex(cfg)

    ta = _trace(1, ts=1.0, text="deterministic content")
    ra = a._provider.build(ta)
    rb = b._provider.build(ta)

    assert ra.profile_id == rb.profile_id
    assert ra.signature == rb.signature


def test_lsh_uses_fewer_comparisons_than_exhaustive_fixture() -> None:
    cfg = _cfg(similarity_threshold=0.9)
    exhaustive = MinHashClusterIndex(cfg)
    lsh = BandedMinHashLSHIndex(cfg)

    traces = []
    for i in range(80):
        traces.append(_trace(i, ts=float(i), text=f"concept-{i} unique alpha"))
    for i in range(80, 160):
        traces.append(_trace(i, ts=float(i), text=f"concept-{i - 80} unique alpha"))

    for t in traces:
        exhaustive.observe(t)
        lsh.observe(t)

    lsh_cmp = lsh.telemetry()["comparisons"]
    ex_cmp = exhaustive.telemetry()["comparisons"]
    assert lsh_cmp < ex_cmp


def test_populated_same_agent_no_bucket_hit_is_novel_without_scan() -> None:
    idx = BandedMinHashLSHIndex(_cfg(similarity_threshold=1.0))
    first = idx.observe(_trace(1, ts=1.0, text="reset account password", tool="search"))
    before = idx.telemetry()
    second = idx.observe(_trace(2, ts=2.0, text="deploy release workflow", tool="deploy"))
    after = idx.telemetry()

    assert first.key.kind == "cluster"
    assert second.key.kind == "cluster"
    assert second.key != first.key
    assert second.novelty == 1.0
    assert after["comparisons"] == before["comparisons"]
    assert after["full_scan_fallbacks"] == 0
    assert after["no_candidate_novel"] == before["no_candidate_novel"] + 1


def test_candidate_hit_still_joins_and_reranks() -> None:
    idx = BandedMinHashLSHIndex(_cfg(similarity_threshold=0.5))
    first = idx.observe(_trace(10, ts=1.0, text="reset account password", tool="search"))
    before = idx.telemetry()
    second = idx.observe(_trace(11, ts=2.0, text="reset account password quickly", tool="search"))
    after = idx.telemetry()

    assert second.key == first.key
    assert second.novelty == 0.0
    assert after["comparisons"] > before["comparisons"]
    assert after["last_candidates"] >= 1


def test_stale_candidate_ids_filtered_to_none_count_as_novel() -> None:
    idx = BandedMinHashLSHIndex(_cfg(similarity_threshold=1.0, ttl_s=90.0, purge_every=9999))
    stale = idx.observe(_trace(20, ts=1.0, text="stale target text", tool="search"))
    live = idx.observe(_trace(21, ts=2.0, text="other live leader text", tool="deploy"))
    assert stale.key.kind == "cluster"
    assert live.key.kind == "cluster"
    assert stale.key != live.key

    # Remove one cluster from live state but leave stale bucket membership behind.
    idx._agent_clusters["agent-a"].pop(stale.key.value, None)
    before = idx.telemetry()
    second = idx.observe(_trace(22, ts=3.0, text="stale target text", tool="search"))
    after = idx.telemetry()

    assert second.key.kind == "cluster"
    assert second.key != stale.key
    assert second.novelty == 1.0
    assert after["comparisons"] == before["comparisons"]
    assert after["no_candidate_novel"] == before["no_candidate_novel"] + 1


def test_build_error_uses_fallback_signature_not_no_candidate_novel() -> None:
    class _BrokenProvider(MinHashSignatureProvider):
        def build(self, trace: Trace):
            raise MinHashBuildError("forced build failure")

    idx = BandedMinHashLSHIndex(_cfg(), signature_provider=_BrokenProvider(_cfg()))
    before = idx.telemetry()
    obs = idx.observe(_trace(30, ts=1.0, text="anything"))
    after = idx.telemetry()

    assert obs.key.kind == "fallback-signature"
    assert after["fallbacks"] == before["fallbacks"] + 1
    assert after["fallback_build_errors"] == before["fallback_build_errors"] + 1
    assert after["fallback_runtime_errors"] == before["fallback_runtime_errors"]
    assert after["no_candidate_novel"] == before["no_candidate_novel"]