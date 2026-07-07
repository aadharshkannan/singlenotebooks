from trace_sampling.model import AgentConfig
from trace_sampling.generator import generate_stream
from trace_sampling.samplers import AdaptiveSampler, BaselineSampler, SamplerConfig


def _stream():
    cfgs = [
        AgentConfig("fast_lowvar", velocity=20.0, vocab_size=2, zipf_s=2.0),
        AgentConfig("rare_highvar", velocity=0.5, vocab_size=15, zipf_s=0.6),
        AgentConfig("late", velocity=5.0, vocab_size=6, zipf_s=1.0, start_time=8.0),
    ]
    return generate_stream(cfgs, duration=30.0, seed=11)


def _run(sampler, stream):
    kept = [t for t in stream if sampler.decide(t)]
    return kept


def test_adaptive_does_not_starve_any_active_agent():
    stream = _stream()
    cfg = SamplerConfig(llm_throughput=15.0)
    kept = _run(AdaptiveSampler(cfg, seed=1), stream)
    kept_agents = {t.agent_id for t in kept}
    active_agents = {t.agent_id for t in stream}
    assert kept_agents == active_agents  # every active agent retained


def test_adaptive_respects_budget_better_than_no_cap():
    stream = _stream()
    cfg = SamplerConfig(llm_throughput=15.0)
    kept = _run(AdaptiveSampler(cfg, seed=1), stream)
    # Kept count should be far below total (sampler is selective).
    assert len(kept) < len(stream)


def test_adaptive_captures_more_rare_variety_than_baseline():
    stream = _stream()
    cfg = SamplerConfig(llm_throughput=15.0)
    adaptive_kept = _run(AdaptiveSampler(cfg, seed=1), stream)
    baseline = BaselineSampler(keep_prob=len(_run(AdaptiveSampler(cfg, seed=1),
                                                  stream)) / len(stream), seed=1)
    baseline_kept = _run(baseline, stream)

    def distinct_for(kept, agent):
        return {t.signature for t in kept if t.agent_id == agent}

    a = distinct_for(adaptive_kept, "rare_highvar")
    b = distinct_for(baseline_kept, "rare_highvar")
    assert len(a) >= len(b)


def test_baseline_keeps_roughly_keep_prob_fraction():
    stream = _stream()
    kept = _run(BaselineSampler(keep_prob=0.3, seed=2), stream)
    frac = len(kept) / len(stream)
    assert 0.2 < frac < 0.4


def test_reservoir_count_is_bounded():
    # Many distinct signatures must not grow the reservoir map without bound.
    from trace_sampling.model import Trace
    cfg = SamplerConfig(llm_throughput=50.0, max_reservoirs=16)
    sampler = AdaptiveSampler(cfg, seed=0)
    for i in range(500):
        sampler.decide(Trace(i, "a", float(i) * 0.001, (f"sig{i}",), 1, 1.0, "ok"))
    assert len(sampler._reservoirs) <= 16


def test_rare_agent_floor_keeps_almost_everything():
    # A single very-rare agent should be (near) fully retained via the floor.
    from trace_sampling.model import Trace
    cfg = SamplerConfig(llm_throughput=50.0)
    sampler = AdaptiveSampler(cfg, seed=0)
    rare = [Trace(i, "rare", float(i) * 5.0, (f"v{i}",), 1, 1.0, "ok")
            for i in range(20)]
    kept = [t for t in rare if sampler.decide(t)]
    assert len(kept) >= 18  # rare/low-velocity agent is protected


def test_adaptive_sets_last_observation():
    from trace_sampling.model import Trace
    from trace_sampling.samplers import AdaptiveSampler, SamplerConfig
    s = AdaptiveSampler(SamplerConfig(llm_throughput=50.0), seed=0)
    s.decide(Trace(0, "a", 0.0, ("search",), 1, 1.0, "ok"))
    assert s.last_observation is not None
    assert s.last_observation.key.kind == "signature"


def test_adaptive_default_variety_matches_prior_behavior():
    stream = _stream()
    cfg = SamplerConfig(llm_throughput=15.0)
    kept = _run(AdaptiveSampler(cfg, seed=1), stream)
    assert {t.agent_id for t in kept} == {t.agent_id for t in stream}
