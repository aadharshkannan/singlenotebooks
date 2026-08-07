# Agent-Uniform Sampling Reference

This package is the executable Python reference for representative session sampling under an LLM tokens-per-minute constraint.

The core rule is simple: **choose sample membership without using token cost, then materialize and pace selected work afterward**. Long sessions therefore cannot lose their place in the sample merely because they are expensive.

## Start Here

1. Read [`../docs/AGENT_UNIFORM_BOUNDED_EVIDENCE_DESIGN.md`](../docs/AGENT_UNIFORM_BOUNDED_EVIDENCE_DESIGN.md) for the normative behavior and invariants.
2. Read [`../docs/BIC_EVALUATIONS_SERVICE_HANDOFF.md`](../docs/BIC_EVALUATIONS_SERVICE_HANDOFF.md) for the target-service implementation map.
3. Run [`../agent_uniform_sampling_walkthrough.ipynb`](../agent_uniform_sampling_walkthrough.ipynb) for an end-to-end example.
4. Open [`../outputs_agent_uniform_sampling/agent-uniform-sampling-overview.html`](../outputs_agent_uniform_sampling/agent-uniform-sampling-overview.html) for the standalone visual summary.
5. Run `py -3.11 -m pytest tests/test_agent_uniform_sampling.py tests/test_token_representation.py -q`.

## Reference Flow

```text
eligible completed sessions
    -> deterministic uniform sample per tenant/agent
    -> persist selected membership and inclusion probability
    -> optionally materialize deterministic bounded evidence
    -> reserve prompt + evidence + completion tokens
    -> pace immutable selected requests under rolling TPM
    -> record completion or an explicit terminal reason
    -> report response rate and per-agent results
```

## Feature Flag

Bounded evidence is disabled by default:

```python
queue = ExecutionQueue("queue.json", tpm_limit=20_000)
```

Enable it for a new queue epoch:

```python
config = BoundedEvidenceConfig(
    enabled=True,
    evidence_max_tokens=2_048,
    context_window_tokens=8_192,
    prompt_overhead_tokens=256,
    completion_reserve_tokens=512,
    tokenizer_model="gpt-5",
    tokenizer_encoding="o200k_base",
)
queue = ExecutionQueue("queue-v2.json", tpm_limit=20_000, bounded_evidence=config)
queue.enqueue(samples)
queue.materialize_bounded_evidence(traces_by_request_id)
queue.schedule_pending()
```

When enabled, scheduling waits with `AWAITING_BOUNDED_EVIDENCE` until every pending selected item has an immutable packet. Materialization reuses `trace_sampling.token_representation`, which prioritizes final assistant outcomes, tool results, system context, initial user goals, later refinements, tool arguments, and earlier assistant content.

## Status Semantics

- `PENDING`: selected and awaiting materialization or scheduling.
- `SCHEDULED`: assigned a deterministic budget-compliant dispatch time.
- `COMPLETED`: judge score recorded.
- `OVERSIZED`: legacy flag-off path; the raw estimate exceeds one TPM window.
- `UNSERVICEABLE`: bounded mode could not construct or reserve a valid request before dispatch.
- `DROPPED`: serviceable selected work exceeded the configured schedule-delay limit.
- `NONRESPONSE`: reserved for a dispatched request that produced no usable judge result in a service implementation.

No terminal item is replaced. Replacement would make inclusion depend on execution cost or outcome.

## Statistical Contract

Within tenant/agent stratum $a$, each eligible session has inclusion probability

$$
p_a = \frac{n_a}{N_a}.
$$

Token estimates, evidence contents, truncation, and scheduling status are absent from rank construction. The prototype reports means from completed selected sessions, but suppresses its finite-population interval whenever `completed_count < selected_count`; nonresponse may be informative and is not repaired by the original sampling probability.

## Prototype Boundaries

The JSON queue is a single-process reference adapter, not a distributed production queue. A service implementation needs durable concurrency control, leases or optimistic concurrency, dispatch attempts, actual token reconciliation, retention policy, and production telemetry. The canonical packet contains session content and must receive the same or stronger protection as source telemetry.
