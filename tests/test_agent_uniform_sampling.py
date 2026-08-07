from agent_uniform_sampling import (
    BoundedEvidenceConfig,
    ExecutionQueue,
    ExecutionStatus,
    SessionCandidate,
    summarize_agent_scores,
    uniformly_sample_by_agent,
)
import hashlib
import json
from math import ceil
import pytest
from trace_sampling.model import SessionEvent, Trace


class CharTokenizer:
    name = "test-char"
    version = "1"

    def count(self, text: str) -> int:
        return max(1, ceil(len(text) / 4))


def _candidate(session_id: str, *, tenant: str = "contoso", agent: str = "support", cost: int = 100) -> SessionCandidate:
    return SessionCandidate(
        tenant_id=tenant,
        agent_id=agent,
        session_id=session_id,
        session_version="v1",
        estimated_tokens=cost,
    )


def _trace_for_candidate(candidate: SessionCandidate, *, text_scale: int = 1) -> Trace:
    return Trace(
        trace_id=int(hashlib.sha256(candidate.dedup_key.encode("utf-8")).hexdigest()[:12], 16),
        agent_id=candidate.agent_id,
        timestamp=1_774_761_600.0,
        signature=("search", "write"),
        span_count=2,
        duration_ms=125.0,
        status="ok",
        events=(
            SessionEvent(role="system", text="System context " + ("S " * text_scale)),
            SessionEvent(role="user", text="User goal " + ("U " * (text_scale * 4))),
            SessionEvent(role="tool", tool_name="search", arguments={"q": "x"}, output="Tool result " + ("T " * (text_scale * 3))),
            SessionEvent(role="assistant", text="Final answer " + ("A " * (text_scale * 2))),
        ),
    )


def test_sampling_is_deterministic_and_cost_neutral() -> None:
    low_cost = tuple(_candidate(f"s-{index}", cost=10 + index) for index in range(10))
    high_cost = tuple(_candidate(f"s-{index}", cost=10_000 - index) for index in range(10))

    low = uniformly_sample_by_agent(candidates=low_cost, sample_size_per_agent=4, seed="seed")
    high = uniformly_sample_by_agent(candidates=high_cost, sample_size_per_agent=4, seed="seed")

    assert [item.candidate.session_id for item in low[0].selected] == [item.candidate.session_id for item in high[0].selected]
    assert low[0].inclusion_probability == 0.4
    assert all(item.inclusion_probability == 0.4 for item in low[0].selected)


def test_sampling_is_stratified_by_tenant_and_agent() -> None:
    candidates = (
        *(_candidate(f"support-{index}", agent="support") for index in range(5)),
        *(_candidate(f"sales-{index}", agent="sales") for index in range(2)),
        *(_candidate(f"fab-{index}", tenant="fabrikam", agent="support") for index in range(3)),
    )
    samples = uniformly_sample_by_agent(candidates=candidates, sample_size_per_agent=3, seed="seed")

    by_key = {sample.stratum_key: sample for sample in samples}
    assert by_key["contoso/support"].sample_size == 3
    assert by_key["contoso/support"].inclusion_probability == 3 / 5
    assert by_key["contoso/sales"].sample_size == 2
    assert by_key["contoso/sales"].inclusion_probability == 1.0
    assert by_key["fabrikam/support"].sample_size == 3


def test_queue_paces_selected_sessions_and_marks_oversized_items(tmp_path) -> None:
    candidates = (
        _candidate("a", cost=600),
        _candidate("b", cost=600),
        _candidate("c", cost=1_500),
    )
    samples = uniformly_sample_by_agent(candidates=candidates, sample_size_per_agent=3, seed="seed")
    queue = ExecutionQueue(tmp_path / "queue.json", tpm_limit=1_000)
    queue.enqueue(samples)
    items = queue.schedule_pending()

    scheduled = [item for item in items if item.status == ExecutionStatus.SCHEDULED]
    oversized = [item for item in items if item.status == ExecutionStatus.OVERSIZED]
    assert len(scheduled) == 2
    assert scheduled[1].scheduled_at_seconds == 60.0
    assert all(item.sampling_seed == "seed" for item in items)
    assert len(oversized) == 1
    assert oversized[0].sampled.candidate.session_id == "c"


def test_default_off_preserves_oversized_behavior_and_schema_compatibility(tmp_path) -> None:
    samples = uniformly_sample_by_agent(
        candidates=(
            _candidate("a", cost=500),
            _candidate("b", cost=2_000),
        ),
        sample_size_per_agent=2,
        seed="seed",
    )
    queue = ExecutionQueue(tmp_path / "queue.json", tpm_limit=1_000)
    queue.enqueue(samples)
    items = queue.schedule_pending()

    by_session = {item.sampled.candidate.session_id: item for item in items}
    assert by_session["a"].status == ExecutionStatus.SCHEDULED
    assert by_session["b"].status == ExecutionStatus.OVERSIZED
    assert by_session["b"].status_reason == "ESTIMATED_TOKENS_EXCEEDS_TPM"


def test_membership_is_identical_regardless_of_bounded_evidence_flag_and_content_size(tmp_path) -> None:
    candidates = tuple(_candidate(f"s-{index}", cost=5_000 + index) for index in range(10))
    samples = uniformly_sample_by_agent(candidates=candidates, sample_size_per_agent=4, seed="seed")
    selected_ids = [item.candidate.session_id for item in samples[0].selected]

    queue_off = ExecutionQueue(tmp_path / "off.json", tpm_limit=1_000)
    queue_off.enqueue(samples)
    queue_off.schedule_pending()

    queue_on = ExecutionQueue(
        tmp_path / "on.json",
        tpm_limit=1_000,
        bounded_evidence=BoundedEvidenceConfig(
            enabled=True,
            evidence_max_tokens=120,
            context_window_tokens=500,
            prompt_overhead_tokens=30,
            completion_reserve_tokens=30,
        ),
    )
    queue_on.enqueue(samples)
    traces = {
        item.request_id: _trace_for_candidate(item.sampled.candidate, text_scale=1_000)
        for item in queue_on.items()
    }
    queue_on.materialize_bounded_evidence(traces, tokenizer=CharTokenizer())
    queue_on.schedule_pending()

    assert [item.candidate.session_id for item in samples[0].selected] == selected_ids
    assert [item.sampled.candidate.session_id for item in queue_off.items()] == [item.sampled.candidate.session_id for item in queue_on.items()]


def test_enabled_bounded_materialization_can_make_oversized_selected_item_schedulable(tmp_path) -> None:
    samples = uniformly_sample_by_agent(
        candidates=(_candidate("oversized", cost=2_000),),
        sample_size_per_agent=1,
        seed="seed",
    )
    queue = ExecutionQueue(
        tmp_path / "queue.json",
        tpm_limit=1_000,
        bounded_evidence=BoundedEvidenceConfig(
            enabled=True,
            evidence_max_tokens=200,
            context_window_tokens=400,
            prompt_overhead_tokens=20,
            completion_reserve_tokens=20,
        ),
    )
    queue.enqueue(samples)
    before = queue.schedule_pending()[0]
    assert before.status == ExecutionStatus.PENDING
    assert before.status_reason == "AWAITING_BOUNDED_EVIDENCE"

    trace = _trace_for_candidate(before.sampled.candidate, text_scale=500)
    queue.materialize_bounded_evidence({before.request_id: trace}, tokenizer=CharTokenizer())
    after = queue.schedule_pending()[0]

    assert after.status == ExecutionStatus.SCHEDULED
    assert after.bounded_evidence is not None
    assert after.bounded_evidence.reservation_tokens <= 1_000


def test_materialized_evidence_hash_is_deterministic_across_reload(tmp_path) -> None:
    samples = uniformly_sample_by_agent(
        candidates=(_candidate("a", cost=2_000),),
        sample_size_per_agent=1,
        seed="seed",
    )
    config = BoundedEvidenceConfig(
        enabled=True,
        evidence_max_tokens=200,
        context_window_tokens=500,
        prompt_overhead_tokens=25,
        completion_reserve_tokens=25,
    )
    queue_path = tmp_path / "queue.json"
    queue = ExecutionQueue(queue_path, tpm_limit=1_000, bounded_evidence=config)
    queue.enqueue(samples)
    item = queue.items()[0]
    trace = _trace_for_candidate(item.sampled.candidate, text_scale=700)
    queue.materialize_bounded_evidence({item.request_id: trace}, tokenizer=CharTokenizer())
    first = queue.items()[0]

    reloaded = ExecutionQueue(queue_path, tpm_limit=1_000, bounded_evidence=config)
    second = reloaded.items()[0]

    assert first.bounded_evidence is not None
    assert second.bounded_evidence is not None
    assert first.bounded_evidence.hash_sha256 == second.bounded_evidence.hash_sha256
    assert first.bounded_evidence.canonical_json == second.bounded_evidence.canonical_json


def test_reservation_accounts_for_evidence_prompt_completion_and_context_window(tmp_path) -> None:
    samples = uniformly_sample_by_agent(
        candidates=(_candidate("a", cost=9_000),),
        sample_size_per_agent=1,
        seed="seed",
    )
    queue = ExecutionQueue(
        tmp_path / "queue.json",
        tpm_limit=1_000,
        bounded_evidence=BoundedEvidenceConfig(
            enabled=True,
            evidence_max_tokens=200,
            context_window_tokens=300,
            prompt_overhead_tokens=70,
            completion_reserve_tokens=70,
        ),
    )
    queue.enqueue(samples)
    item = queue.items()[0]
    queue.materialize_bounded_evidence(
        {item.request_id: _trace_for_candidate(item.sampled.candidate, text_scale=800)},
        tokenizer=CharTokenizer(),
    )
    updated = queue.items()[0]

    assert updated.status == ExecutionStatus.UNSERVICEABLE
    assert updated.status_reason == "RESERVATION_EXCEEDS_CONTEXT_WINDOW"


def test_impossible_mandatory_floor_remains_terminal_unserviceable(tmp_path) -> None:
    samples = uniformly_sample_by_agent(
        candidates=(_candidate("a", cost=300),),
        sample_size_per_agent=1,
        seed="seed",
    )
    queue = ExecutionQueue(
        tmp_path / "queue.json",
        tpm_limit=1_000,
        bounded_evidence=BoundedEvidenceConfig(
            enabled=True,
            evidence_max_tokens=5,
            context_window_tokens=200,
            prompt_overhead_tokens=10,
            completion_reserve_tokens=10,
        ),
    )
    queue.enqueue(samples)
    item = queue.items()[0]
    queue.materialize_bounded_evidence(
        {item.request_id: _trace_for_candidate(item.sampled.candidate, text_scale=50)},
        tokenizer=CharTokenizer(),
    )
    scheduled = queue.schedule_pending()[0]

    assert scheduled.status == ExecutionStatus.UNSERVICEABLE
    assert scheduled.status_reason in {
        "MANDATORY_EVIDENCE_FLOOR_UNSERVICEABLE",
        "CANONICAL_STRUCTURE_UNSERVICEABLE",
    }


def test_materialized_evidence_is_immutable_for_request(tmp_path) -> None:
    samples = uniformly_sample_by_agent(
        candidates=(_candidate("a", cost=2_000),),
        sample_size_per_agent=1,
        seed="seed",
    )
    config = BoundedEvidenceConfig(
        enabled=True,
        evidence_max_tokens=200,
        context_window_tokens=500,
        prompt_overhead_tokens=25,
        completion_reserve_tokens=25,
    )
    queue = ExecutionQueue(tmp_path / "queue.json", tpm_limit=1_000, bounded_evidence=config)
    queue.enqueue(samples)
    item = queue.items()[0]
    first_trace = _trace_for_candidate(item.sampled.candidate, text_scale=100)
    changed_trace = _trace_for_candidate(item.sampled.candidate, text_scale=101)

    queue.materialize_bounded_evidence({item.request_id: first_trace}, tokenizer=CharTokenizer())

    with pytest.raises(ValueError, match="immutable bounded evidence conflict"):
        queue.materialize_bounded_evidence({item.request_id: changed_trace}, tokenizer=CharTokenizer())


def test_queue_drops_items_delayed_past_the_schedule_limit(tmp_path) -> None:
    candidates = (
        _candidate("a", cost=600),
        _candidate("b", cost=600),
    )
    samples = uniformly_sample_by_agent(candidates=candidates, sample_size_per_agent=2, seed="seed")
    queue = ExecutionQueue(
        tmp_path / "queue.json",
        tpm_limit=1_000,
        max_schedule_delay_seconds=59,
    )
    queue.enqueue(samples)
    items = queue.schedule_pending()

    assert sum(item.status == ExecutionStatus.SCHEDULED for item in items) == 1
    assert sum(item.status == ExecutionStatus.DROPPED for item in items) == 1


def test_queue_rejects_changed_population_for_existing_sampling_run(tmp_path) -> None:
    queue = ExecutionQueue(tmp_path / "queue.json", tpm_limit=1_000)
    first_sample = uniformly_sample_by_agent(
        candidates=(_candidate("a"), _candidate("b")),
        sample_size_per_agent=1,
        seed="seed",
    )
    changed_sample = uniformly_sample_by_agent(
        candidates=(_candidate("a"), _candidate("b"), _candidate("c")),
        sample_size_per_agent=1,
        seed="seed",
    )

    queue.enqueue(first_sample)

    try:
        queue.enqueue(changed_sample)
    except ValueError as error:
        assert str(error) == "sampling run does not match existing queue metadata"
    else:
        raise AssertionError("expected changed sampling run to be rejected")


def test_agent_summary_uses_only_completed_scores(tmp_path) -> None:
    samples = uniformly_sample_by_agent(
        candidates=tuple(_candidate(str(index)) for index in range(3)),
        sample_size_per_agent=3,
        seed="seed",
    )
    queue = ExecutionQueue(tmp_path / "queue.json", tpm_limit=1_000)
    queue.enqueue(samples)
    scheduled = [item for item in queue.schedule_pending() if item.status == ExecutionStatus.SCHEDULED]
    queue.complete(scheduled[0].request_id, score=0.5)
    queue.complete(scheduled[1].request_id, score=0.9)

    summary = summarize_agent_scores(queue.items())[0]
    assert summary.selected_count == 3
    assert summary.completed_count == 2
    assert summary.mean_score == 0.7
    assert summary.confidence_interval_95 is None


def test_agent_summary_suppresses_interval_when_selected_response_is_incomplete(tmp_path) -> None:
    samples = uniformly_sample_by_agent(
        candidates=tuple(_candidate(str(index)) for index in range(3)),
        sample_size_per_agent=3,
        seed="seed",
    )
    queue = ExecutionQueue(tmp_path / "queue.json", tpm_limit=1_000)
    queue.enqueue(samples)
    scheduled = [item for item in queue.schedule_pending() if item.status == ExecutionStatus.SCHEDULED]
    queue.complete(scheduled[0].request_id, score=0.4)
    queue.complete(scheduled[1].request_id, score=0.8)

    summary = summarize_agent_scores(queue.items())[0]

    assert summary.completed_count == 2
    assert summary.selected_count == 3
    assert summary.mean_score == pytest.approx(0.6)
    assert summary.confidence_interval_95 is None


def test_agent_summary_has_zero_width_interval_for_completed_census(tmp_path) -> None:
    samples = uniformly_sample_by_agent(
        candidates=tuple(_candidate(str(index)) for index in range(2)),
        sample_size_per_agent=2,
        seed="seed",
    )
    queue = ExecutionQueue(tmp_path / "queue.json", tpm_limit=1_000)
    queue.enqueue(samples)
    for index, item in enumerate(queue.schedule_pending()):
        queue.complete(item.request_id, score=0.2 + 0.6 * index)

    summary = summarize_agent_scores(queue.items())[0]
    assert summary.mean_score == 0.5
    assert summary.confidence_interval_95 == (0.5, 0.5)


def test_queue_persists_sampling_run_metadata_and_is_idempotent(tmp_path) -> None:
    candidates = (
        *(_candidate(f"support-{index}", tenant="contoso", agent="support", cost=100 + index) for index in range(4)),
        *(_candidate(f"sales-{index}", tenant="contoso", agent="sales", cost=200 + index) for index in range(2)),
    )
    samples = uniformly_sample_by_agent(candidates=candidates, sample_size_per_agent=2, seed="durable-seed")
    queue_path = tmp_path / "queue.json"
    queue = ExecutionQueue(queue_path, tpm_limit=1_000)

    queue.enqueue(samples)
    queue.enqueue(samples)

    data = json.loads(queue_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "agent-uniform-v2"
    assert len(data["sampling_runs"]) == 2
    assert len(data["items"]) == 4

    runs = sorted(data["sampling_runs"].values(), key=lambda run: run["stratum_key"])
    assert runs[0]["stratum_key"] == "contoso/sales"
    assert runs[0]["population_size"] == 2
    assert runs[0]["sample_size"] == 2
    assert runs[0]["inclusion_probability"] == 1.0
    assert runs[0]["seed"] == "durable-seed"
    assert len(runs[0]["selected_request_ids"]) == 2
    assert len(set(runs[0]["selected_request_ids"])) == 2

    assert runs[1]["stratum_key"] == "contoso/support"
    assert runs[1]["population_size"] == 4
    assert runs[1]["sample_size"] == 2
    assert runs[1]["inclusion_probability"] == 0.5
    assert runs[1]["seed"] == "durable-seed"
    assert len(runs[1]["selected_request_ids"]) == 2
    assert len(set(runs[1]["selected_request_ids"])) == 2