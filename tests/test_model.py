from trace_sampling.model import AgentConfig, SessionEvent, Trace


def test_trace_is_immutable_and_hashable():
    t = Trace(trace_id=1, agent_id="a", timestamp=0.0,
              signature=("search", "read"), span_count=2,
              duration_ms=12.5, status="ok")
    assert t.signature == ("search", "read")
    # frozen dataclass -> hashable, usable as dict key
    assert {t: 1}[t] == 1


def test_trace_has_concept_id_default():
    from trace_sampling.model import Trace
    t = Trace(0, "a", 0.0, ("search",), 1, 1.0, "ok")
    assert t.concept_id == -1  # default = unknown/unlabeled


def test_trace_accepts_concept_id():
    from trace_sampling.model import Trace
    t = Trace(0, "a", 0.0, ("search",), 1, 1.0, "ok", concept_id=3)
    assert t.concept_id == 3


def test_trace_accepts_ordered_session_events_without_breaking_legacy_default():
    legacy = Trace(0, "a", 0.0, ("search",), 1, 1.0, "ok")
    events = (
        SessionEvent(role="user", text="Find the incident"),
        SessionEvent(
            role="assistant",
            text="",
            tool_name="search",
            arguments={"query": "incident"},
            output="one result",
        ),
    )
    trace = Trace(1, "a", 1.0, ("search",), 1, 2.0, "ok", events=events)

    assert legacy.events == ()
    assert trace.events == events
    assert trace.events[0].role == "user"
    assert trace.events[1].arguments == {"query": "incident"}
    assert {trace: 1}[trace] == 1


def test_agent_config_defaults():
    c = AgentConfig(agent_id="a", velocity=2.0, vocab_size=5, zipf_s=1.2)
    assert c.start_time == 0.0
    assert 0.0 <= c.error_rate <= 1.0
    assert len(c.tool_pool) > 0
