from trace_sampling.eval_harness import run_arm
from trace_sampling.samplers import SamplerConfig
from trace_sampling.concepts import ConceptSpec, SynonymMap
from trace_sampling.generator import generate_concept_stream, ConceptAgentConfig


def _stream():
    sm = SynonymMap([["search", "query"], ["edit", "modify"], ["run", "exec"]])
    concepts = [ConceptSpec(0, ("search", "edit")), ConceptSpec(1, ("search", "run"))]
    agents = [ConceptAgentConfig("a", 10.0, (0, 1),
                                 vocab_bias={"search": "search", "edit": "edit", "run": "run"}),
              ConceptAgentConfig("b", 10.0, (0, 1),
                                 vocab_bias={"search": "query", "edit": "modify", "run": "exec"})]
    return generate_concept_stream(agents, concepts, sm, duration=8.0, seed=1)


def _sm():
    return SynonymMap([["search", "query"], ["edit", "modify"], ["run", "exec"]])


def test_run_arm_produces_result_with_log_and_ledger():
    stream = _stream()
    result = run_arm(stream, SamplerConfig(llm_throughput=20.0), arm="adaptive_exact", seed=0)
    log = result.log
    assert set(["timestamp", "agent_id", "concept_id", "signature",
                "variety_key", "key_kind", "kept"]).issubset(log.columns)
    assert len(log) == len(stream)
    assert log["kept"].sum() > 0
    assert set(["embed_calls", "cache_hits", "cache_hit_rate", "search_queries",
                "embed_latency_p50_ms", "embed_latency_p95_ms",
                "added_latency_p50_ms", "added_latency_p95_ms", "est_cost_usd",
                "embed_chunks", "embed_tokens", "embed_failures",
                "embed_failed_chunks", "embed_failed_tokens",
                "fallbacks", "kept"]).issubset(result.ledger)
    assert result.ledger["embed_calls"] == 0
    assert result.ledger["est_cost_usd"] == 0.0


def test_run_arm_baseline_is_random_sampler_without_variety_index():
    stream = _stream()
    result = run_arm(stream, SamplerConfig(llm_throughput=20.0),
                     arm="baseline", seed=0, keep_prob=0.5)
    log = result.log
    assert len(log) == len(stream)
    assert log["kept"].sum() > 0
    # Random baseline has no variety index: every row logs the raw signature.
    assert (log["key_kind"] == "signature").all()
    assert result.ledger["embed_calls"] == 0
    assert result.ledger["est_cost_usd"] == 0.0


def test_cluster_arm_does_not_inflate_keeps_offline():
    # Calibrated keep-signal must NOT keep more than the exact arm on the same stream.
    stream = _stream()
    cfg = SamplerConfig(llm_throughput=20.0)
    exact = run_arm(stream, cfg, arm="adaptive_exact", seed=0)
    cluster = run_arm(stream, cfg, arm="adaptive_cluster_offline", seed=0, synonym_map=_sm())
    assert cluster.ledger["kept"] <= exact.ledger["kept"]
    # cluster unification still collapses signature variants into fewer distinct keys
    assert cluster.log["variety_key"].nunique() < exact.log["variety_key"].nunique()


def test_run_arm_offline_cluster_unifies_variants():
    stream = _stream()
    result = run_arm(stream, SamplerConfig(llm_throughput=20.0),
                     arm="adaptive_cluster_offline", seed=0, synonym_map=_sm())
    assert result.ledger["embed_calls"] > 0
    log = result.log
    distinct_sigs = log.groupby("agent_id")["signature"].nunique().sum()
    clustered = log[log["key_kind"] == "cluster"]
    distinct_keys = clustered.groupby("agent_id")["variety_key"].nunique().sum()
    assert distinct_keys < distinct_sigs
