# Random Sampling Batch Token-Budget Distribution

## Problem statement

Agent365 Task Completion judging has one global capacity limit of 20,000 tokens
per minute (TPM), shared by every tenant and agent. This component turns the
time earned since the last successful batch into a label-blind, auditable plan
of whole completed sessions, then paces execution so planning a five-minute
allowance does not create an instantaneous 100,000-token burst.

The first implementation evaluates only full-session `TASK_COMPLETION_V1`.
It is independent of MinHash, embeddings, novelty, IDW, adaptive sampling, and
judge labels or outcomes.

## Goals and non-goals

Goals are deterministic random selection, global token accounting, tenant and
agent service protection, durable retry state, and reproducible offline
simulation. Allocation and selection consume only identity, timestamps,
tenant/agent membership, estimated packet cost, and prior fairness state.

Non-goals are a provider-specific storage implementation, statistical claims
of equal inclusion probability after token-constrained packing, selection by
content quality, or replacing the existing count-based probability sampler.
The component reports its token-constrained inclusion limitation explicitly.

## Batch window and framing

All timestamps are timezone-aware UTC. The canonical event-time window is
half-open:

$$
[t_{previous}, t_{cutoff})
$$

`t_previous` is the successful checkpoint watermark, not the previous timer
tick. `t_cutoff` is frozen when the batch lease is acquired. Eligibility uses a
session's completion time; a session that started earlier but completed inside
the window is eligible. Incomplete sessions are deferred. A source ingestion
time is retained only to audit lateness.

A configurable lookback extends the source scan backward from `t_previous`.
It does not change the canonical window. Durable processed/selected session
IDs deduplicate overlap, so late arrivals can be selected once. Corrections to
an already processed session create a versioned backfill record; they do not
silently rejudge a completed result. The default first run requires an explicit
bootstrap watermark, rather than scanning unbounded history.

The elapsed allowance is actual successful-watermark elapsed time, clamped to
`max_catchup_minutes`. A missed run therefore earns its elapsed allowance up to
the cap. Unused tokens expire as rate capacity; they do not become a spendable
bank. Fairness deficits, not tokens, carry forward to offset repeated service
shortfalls.

## Token accounting

The nominal allowance is:

$$
B_{nominal}=\min(\Delta t, C_{catchup})\times20{,}000
$$

where elapsed time is measured in fractional minutes. The effective planning
budget is:

$$
B_{effective}=\max(0,B_{nominal}-B_{safety}-B_{retry}-B_{output})
$$

All deductions are explicit configuration fields and are included in the
configuration hash. A planned session cost is estimated input packet tokens
plus a bounded expected completion allowance. The production adapter must use
the actual model tokenizer and the canonical evidence packet. The existing
packet byte count is not treated as token count.

TPM is conservatively treated as input plus output plus failed/retried calls
when the provider accounts them. Requests reserve estimated total cost before
dispatch. Actual usage is reconciled after completion; overages become a debit
against the next pacing window and are telemetry, never an excuse to exceed the
rolling cap. Task Completion responses should be compact structured outputs
with a configured maximum completion token count. Estimation error, reserved,
actual, refunded, and failed-call usage are all audited.

## Checkpoint and idempotency

One durable checkpoint stores `pipeline_id`, `previous_successful_watermark`,
`cutoff`, scheduling epoch, batch run ID, status, elapsed minutes, nominal and
effective budgets, reserve breakdown, frame/configuration hashes, seed,
membership hash, selected IDs, planned and actual token usage, retry count,
completion time, and next fairness state.

Lifecycle is `PREPARED -> RUNNING -> SETTLED -> COMMITTED`. `PREPARED` persists
the frozen frame and selection plan before a judge call. Results are keyed by a
deterministic idempotency key over batch ID, session version, metric version,
packet hash, and judge configuration. `COMMITTED` atomically advances the
watermark only after required results, terminal failures/deferred retry records,
actual accounting, and fairness state are durable. A retry of the same
prepared batch reuses its saved plan and seed exactly.

## Allocation and selection

The allocator first computes token demand by tenant and agent from the frozen
frame. It grants feasible tenant floors, then proportional surplus using capped
Hamilton largest remainder. Within each tenant it applies the same process to
agents. Caps are demand caps. Ties sort by stable tenant/agent keys. Floors
that cannot be funded are proportionally scaled down; they are promises over a
rolling fairness horizon, not an impossible per-batch one-session guarantee.

Each agent receives a stable random session order using SHA-256 over the batch
seed, frame hash, tenant, agent, and session ID. The allocator packs whole
sessions in that order up to its final grant. It does not replace an expensive
selected candidate with a cheap candidate solely to maximize session count.
After a pass, unused grants are pooled and deterministically redistributed to
agents with remaining eligible demand, in stable round-robin rounds. Sessions
larger than an agent grant remain unselected in that batch and add service
deficit. Sessions larger than the entire effective batch budget are recorded as
unserviceable and require a product policy such as a larger budget or allowed
packet reduction; they are never split.

Random rank plus greedy whole-session packing is reproducible but does not
give a common closed-form inclusion probability when costs differ. The plan
therefore labels selection as `token_constrained_random`, records rank/cost,
and does not use unweighted results for population estimates. If future
reporting needs design-based estimates, it must introduce a validated
cost-aware design with auditable inclusion probabilities.

Pseudocode:

```text
acquire fenced lease(pipeline_id)
previous = checkpoint.successful_watermark or bootstrap_watermark
cutoff = utc_now()                         # immutable after lease
window = [previous, cutoff)
budget = effective_budget(clamp(cutoff - previous))
frame = freeze(sort(completed_sessions(window, lookback) - processed_ids))
if prepared checkpoint exists for frame/config/seed: resume it
costs = estimate_full_session_costs(frame)
tenant_grants = capped_hamilton(floors, tenant_demand, budget, deficits)
agent_grants = per_tenant_capped_hamilton(...)
selection = pack(random_rank(frame, seed), agent_grants)
repeat deterministic redistribution while capacity and fitting demand remain
persist PREPARED(frame, grants, selection, hashes, fairness input)
for item in paced_queue(selection): reserve tokens; submit idempotently; settle
persist results, actual usage, fairness output; checkpoint-CAS COMMITTED
release lease
```

## Pacing, concurrency, and failure recovery

Batch allowance, selection plan, and execution pacing are separate. A rolling
token bucket refills at $20,000/60$ tokens/second. Each dispatch reserves the
estimated total cost, and a request waits until enough tokens are available.
The bucket has a bounded burst no larger than the configured maximum packet
cost. The production store must share this ledger across workers. It also uses
a lease with expiry and monotonic fencing generation; stale writers are
rejected.

The JSON file store in this repository is a single-process reference adapter.
It uses atomic temp-file plus replace writes for durability on one host, but it
is not a production distributed lease or transactional checkpoint backend.

Trace loading, estimation, allocation, selection, and plan persistence failures
leave the prior checkpoint untouched. A failure after plan persistence resumes
unsettled work. Partial judge completion stores settled results and attempts;
transient calls retry under the same governor and idempotency key, while
permanently invalid sessions are terminal failures. Failed calls record actual
provider consumption where available. Result or checkpoint write failure blocks
watermark advancement. Two simultaneous triggers cannot both own the lease.

## Multi-batch fairness and telemetry

Each active tenant/agent has a fairness deficit measured in unserved effective
token share. Demand and age influence deficit priority only, never labels.
Deficit is capped and carried to the next batch, allowing a costly small agent
to eventually receive service. Telemetry includes budgets, reserves, demand,
grants, selected/actual costs, slack, rounds, coverage, zero allocations,
deficits, timing, queue duration, rolling TPM, late/deduplicated counts,
oversize counts, frame/configuration hashes, seed, and state transitions.

The allocation is $O(n\log n + a\log a)$ for $n$ sessions and $a$ active
agents; hashing and sorting dominate. The schedule adds $O(s)$ work for $s$
selected sessions.

## Worked examples

1. Five minutes earns 100,000 nominal tokens. With 5,000 safety, 10,000
   output, and 5,000 retry reserves, 80,000 are allocatable.
2. One hour earns 1,200,000 nominal tokens under the same rate; deductions are
   independently audited before allocation.
3. A 90-minute delayed batch with a 60-minute catch-up cap earns at most
   1,200,000 tokens and records the clamp.
4. A first run uses the configured UTC bootstrap watermark and persists it in
   the first prepared plan; it never silently scans all history.
5. One tenant/agent receives its demand up to the effective budget, then has
   any non-fitting whole-session slack reported.
6. Two equal tenants split feasible floor and proportional surplus equally;
   stable keys settle an indivisible final token tie.
7. A dominant tenant is demand-capped after its fair surplus; active small
   tenants first receive feasible floors and carry deficits when floors fail.
8. A tenant's dominant agent cannot consume grants reserved by its other active
   agents until deterministic redistribution observes those agents' slack.
9. More active agents than capacity produces zero grants for some agents in a
   batch, but their deficits make them earlier surplus candidates later.
10. Variable costs are packed in random stable order, not shortest-first; slack
    is a transparent consequence of indivisibility.
11. A 900-token session under an 800-token agent grant is deferred; it may fit
    after grant redistribution or receives carried deficit.
12. A 120,000-token packet under a 100,000-token effective batch is marked
    unserviceable, not truncated or split by this component.
13. When demand is below capacity, all fitting sessions are selected and unused
    capacity is recorded as expired slack.
14. A trace ingested late but completed in the lookback is selected only if its
    session ID/version has no processed ledger entry.
15. A failed batch retry reads the persisted frame, seed, allocation, and plan,
    yielding identical membership and only retrying unsettled work.
16. Two triggers contend on the fenced lease; one prepares/runs while the other
    observes the existing batch and exits or resumes safely.
17. Across epochs, agents that repeatedly miss a feasible service floor accrue
    deficits, improving their deterministic allocation priority.

## Alternatives considered

First-come-first-served and global random packing are simple but allow volume
dominance. Equal per-tenant/agent ignores demand and can strand budget. Pure
proportional allocation risks starvation. Hierarchical floor-plus-capped-
Hamilton allocation with deterministic redistribution balances demand,
isolation, and auditability. Max-min water-filling and divisor methods remain
future candidates if frequent stratum churn makes Hamilton paradoxes material.
No content-derived selector is allowed in this production path.

## Test strategy and open decisions

Unit tests cover window boundaries, fractional elapsed time, catch-up, reserves,
frame deduplication, deterministic allocation/selection, demand caps,
redistribution, oversized packets, deficit carry-over, checkpoint retries, and
rolling TPM. Simulation compares FCFS, global random, equal tenant, equal
agent, proportional, and hierarchical policies across normal, delayed, burst,
late, and failure scenarios. It measures utilization, coverage, starvation,
Jain fairness, max-min service, slack, queue duration, replay, duplicates, and
TPM compliance.

Open product decisions are the authoritative provider definition of TPM, the
tokenizer/model version, durable checkpoint/lease backend, bootstrap watermark,
maximum catch-up, reserve defaults, late-data revision policy, and whether
future reporting needs statistically calibrated cost-aware inclusion
probabilities.