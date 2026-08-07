# BIC Evaluations Service Handoff

Target repository: `C:\Users\stangoodwin\BIC-Evaluations-Service`

Source of truth for behavior: this repository's Python implementation and [`AGENT_UNIFORM_BOUNDED_EVIDENCE_DESIGN.md`](AGENT_UNIFORM_BOUNDED_EVIDENCE_DESIGN.md)

This document maps the Python reference into the existing C# service. It intentionally contains no C# implementation. The service engineer should preserve the behavior and invariants while using the target repository's established job, persistence, configuration, telemetry, and test patterns.

## Executive Summary

The service already detects completed sessions, applies percentage/count sampling, dispatches evaluator jobs through BJS, and calls CAPI through `IGenAIService`. The handoff changes the completed-session path in two ways:

1. Select a deterministic fixed sample independently inside each tenant/agent population, recording `N`, `n`, and `p=n/N`.
2. After membership is frozen, optionally construct bounded evidence and reserve the full judge request under context and TPM limits before dispatch.

Do not replace the online rate gate. It solves a different problem.

## Existing Target Seams

Paths below are relative to `BIC-Evaluations-Service`.

| Concern | Existing target location | Recommended action |
|---|---|---|
| Online rate gate | `src/BicEvalsService.BusinessLogic/Services/Sampling/SamplingService.cs` and `ISamplingService.cs` | Leave intact; it hashes conversations against a rate and is not the completed-session representative sampler. |
| Sampling configuration | `src/BicEvalsService.BusinessLogic/Services/Sampling/Contracts/SamplingConfig.cs` | Add or reference a separate completed-session uniform-sampling and bounded-evidence configuration contract. |
| Completed-session selection | `src/BicEvalsService.BusinessLogic/Services/Jobs/BackgroundJobs/SessionCompletionSelector.cs` | Replace or bypass percentage-plus-global-cap membership for this evaluation mode. Retain completion/watermark logic. |
| Durable execution | `EvalDispatcherParentJob.cs`, `EvaluationJob.cs`, `EvalJobManager.cs`, `ExternalEvaluationParentJob.cs`, `FinalizeOfflineEvaluationJob.cs` | Represent selected membership, evidence readiness, reservations, attempts, and terminal reasons using BJS metadata and durable service storage. |
| Judge boundary | `src/BicEvalsService.BusinessLogic/Services/GenAI/IGenAIService.cs` and provider factory | Insert bounded request materialization before `ExecutePromptAsync`; submit the persisted immutable artifact. |
| CAPI provider | `src/BicEvalsService.Dependencies/CAPI/ICapiClientService.cs`, `CapiClientService.cs`, `CapiGenAIProvider.cs` | Keep CAPI as the provider path. Confirm its context and usage accounting semantics. |
| Reactive resilience | `src/BicEvalsService.Dependencies/CAPI/CapiResiliencePolicies.cs` | Retain retries/circuit breaker, but do not treat them as proactive TPM pacing. |
| Session data | `EvalSpan.cs`, `EvalSpanMetadata.cs`, `EvalSpanIdentifier.cs`, `EvalSpanContent.cs` | Map the ordered span tree/messages/tool content into the canonical evidence event model. |
| Tests | `src/BicEvals.Tests/Services/Sampling/` and `src/BicEvals.Tests/Services/Jobs/` | Add parity tests mirroring the Python acceptance matrix. |
| Architecture rules | `docs/Architecture.md` | Reuse BJS, Cosmos, CAPI, DI, and logging conventions; justify any tokenizer dependency or new assembly. |

Important naming trap: the target `ITokenService` handles AAD bearer tokens, not LLM tokenization. Do not extend it for model token counting.

## Recommended Component Boundaries

### 1. Completed-Session Uniform Sampler

Add a focused service under `BicEvalsService.BusinessLogic/Services/Sampling` rather than overloading the online `SamplingService`.

Inputs:

- frozen completed-session candidates;
- tenant ID, agent ID, session ID, session version;
- sample size per agent and run seed.

Outputs:

- per-stratum population/sample metadata;
- selected identity, rank hash, and inclusion probability.

The rank algorithm and tie-breakers must match the design contract. Token estimates and contents must not enter this component.

### 2. Evidence Materializer

Add a provider-independent component upstream of evaluator dispatch:

- maps ordered `EvalSpan` content into canonical events;
- applies the weighted deterministic policy;
- builds the exact judge prompt/request envelope;
- counts with the deployed model tokenizer;
- emits immutable evidence, audit, prompt fingerprint, and reservation;
- returns typed serviceability failures.

Keep this upstream of legacy/platform evaluator duplication where possible so one packet contract feeds both paths. Confirm parity-test requirements before changing evaluator inputs.

### 3. Reservation And Dispatch Coordinator

Use BJS and existing durable storage rather than porting the Python JSON queue. The coordinator should:

- claim an evidence-ready selected request using ETag/lease semantics;
- atomically reserve rolling TPM capacity;
- schedule or delay dispatch;
- persist attempt identity and evidence hash;
- call `IGenAIService.ExecutePromptAsync` with that exact artifact;
- reconcile actual usage when available;
- record completion, judge nonresponse, or retryable provider failure.

Confirm whether CAPI/BJS already exposes an appropriate shared rate limiter. `CapiResiliencePolicies` is reactive and does not prove admission fits TPM.

### 4. Reporting

Report per tenant/agent. Preserve selected membership regardless of execution status. Include response and truncation diagnostics. Do not produce a finite-population interval when selected response is incomplete unless a reviewed nonresponse adjustment is introduced.

## Configuration

Use a versioned configuration epoch containing at least:

```text
enabled
sample_size_per_agent
sampling_seed / seed derivation policy
evidence_policy + version
evidence_max_tokens
model/deployment
resolved tokenizer encoding + version
prompt/schema fingerprint
context_window_tokens
completion_reserve_tokens
tpm_limit
max_schedule_delay
```

Do not toggle bounded mode in place for existing scheduled work. New configuration produces a new sampling/evaluation run identity.

## Target State Mapping

| Reference state | Suggested service meaning |
|---|---|
| `PENDING / AWAITING_BOUNDED_EVIDENCE` | Selected membership persisted; materialization job not complete. |
| `UNSERVICEABLE` | No provider call occurred; typed evidence/context/TPM terminal reason. |
| `SCHEDULED` | Immutable evidence and token reservation persisted; dispatch time assigned. |
| `COMPLETED` | Usable judge result persisted. |
| `NONRESPONSE` | Provider was called but no usable judge result was obtained. |
| `DROPPED` | Serviceable selected request exceeded the allowed queue delay; remains in reporting denominator. |

Map these to BJS job/operation metadata rather than changing framework states unless target owners prefer a dedicated domain enum.

## Telemetry

Follow the existing `logger.LogMetric(name, dimensions, value)` convention. Suggested low-cardinality metrics:

- eligible, selected, evidence-ready, scheduled, dispatched, completed;
- unserviceable, dropped, judge-nonresponse by bounded reason category;
- response rate and truncation rate;
- original/emitted/reserved/actual token histograms;
- reservation error and TPM utilization;
- materialization and queue latency;
- retries and immutable-artifact conflicts.

Keep run ID, policy version, prompt fingerprint, and evidence hash in structured diagnostic events where approved, not high-cardinality metric dimensions. Never log canonical evidence, tool output, or user content.

## Implementation Sequence

1. Confirm the eligible completed-session frame and agent identity with the owners of `SessionCompletionSelector`.
2. Add the uniform sampler and deterministic membership tests without changing dispatch.
3. Persist run and selected-request metadata through existing BJS/Cosmos patterns.
4. Add canonical event mapping and bounded materialization in shadow mode.
5. Select and justify a model-compatible tokenizer; pin and record its identity.
6. Count the exact serialized CAPI request envelope and implement typed serviceability results.
7. Add distributed rolling-TPM reservation or integrate an approved existing limiter.
8. Dispatch the persisted artifact through `IGenAIService` and reconcile usage.
9. Add per-agent reporting and suppress inference under incomplete response.
10. Roll out via new configuration epochs and allowlists.

## Required C# Tests

### Sampling

- Identical selected IDs when only estimated token costs change.
- Identical selected IDs when session text length changes.
- Independent tenant/agent strata and `p=min(n,N)/N`.
- Duplicate session-version rejection.
- Stable SHA-256 vectors including Unicode.

### Evidence

- Full packet unchanged when it fits.
- Deterministic weighted truncation.
- Final outcome, initial goal, and tool results retained.
- Later tool results win ties.
- Structural and mandatory-floor failures are typed.
- Same input/config produces the same canonical bytes and hash across restarts.
- Exact context and TPM boundaries accept equality and reject limit plus one.

### Lifecycle

- Flag off preserves legacy behavior.
- Flag on refuses dispatch before materialization.
- Raw oversized selected session becomes schedulable after bounding.
- Unserviceable and dropped records are never replaced.
- Retries send the same artifact hash.
- Concurrent workers cannot double-reserve or double-dispatch.
- Stale ETags/leases cannot overwrite terminal state.
- Actual usage reconciliation updates future capacity safely.

### Reporting And Security

- Partial selected response suppresses confidence intervals.
- Every terminal reason remains in the selected denominator.
- Evidence and tool output never appear in logs.
- Tenant isolation and retention/deletion propagation are tested.

## Open Decisions For Service Owners

- Which C# tokenizer library exactly matches the deployed CAPI model, and can CAPI provide authoritative request counts?
- Does BJS or another shared platform component already support distributed rolling-window token reservations?
- Where should immutable evidence live: encrypted Cosmos record, governed blob, or existing job payload storage?
- What is the authoritative session version when late spans or corrections arrive?
- Does the initial release use fixed `n` per agent or the Cochran/FPC sizing available in the separate Python `random_sampling` package?
- What response-rate threshold blocks publication, and what paired full-versus-bounded score agreement is required for rollout?

These decisions should be recorded before implementation begins; none should be silently inferred from the Python JSON adapter.

## Definition Of Done

The C# implementation is ready when it passes behavior-equivalent tests for membership, bounded evidence, reservation, lifecycle, and reporting; uses the target repository's durable/concurrent patterns; produces no content-bearing logs; and can be disabled without reinterpreting existing work. The Python notebook and tests are reference evidence, not deployable service code.
