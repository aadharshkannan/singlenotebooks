# Agent 365 Random Session Evaluation Sampling

Date: 2026-07-28

Status: implemented prototype on `stangoodwin/minhash-sampling`

Package: `random_sampling`

## Decision

Use one complete Agent 365 session as the sampling and Task Completion evaluation unit. Build a completed-session population independently for every tenant-scoped agent and UTC evaluation window, derive the selected count with Cochran sizing plus finite-population correction, then draw a deterministic proportionate stratified random sample without replacement.

This package deliberately contains no MinHash, embeddings, novelty score, or purposive diversity slice. Every selected session belongs to the probability sample and is eligible for the headline estimand.

## End-to-End Flow

```mermaid
flowchart LR
    A[Agent 365 OTLP / ESP / Kusto records] --> B[Normalize spans]
    B --> C[Sessionize by sessionId or inactivity]
    C --> D[Completed sessions in UTC window]
    D --> E[Cochran + FPC per agent]
    E --> F[Tenant or agent capacity]
    F --> G[Hamilton allocation across strata]
    G --> H[Random sample without replacement]
    H --> I[UTF-8-bounded evidence]
    I --> J[Async judge]
    J --> K[Estimate + Wilson/FPC interval]
```

## Sessionization

A source `sessionId` is authoritative and scoped by tenant and agent. Without one, spans in a conversation are sorted and split only when the next activity begins strictly more than the configured inactivity timeout after the previous activity ends. The default is 30 minutes; exactly 30 minutes remains in the same session.

A session is judgeable when it contains user text and a final assistant response. Missing identity, malformed payloads, uncertain fallback grouping, inferred completion, and incomplete sessions are represented as structured ingest issues.

## Evaluation Window

Windows are half-open: `[start_at, end_at)`. A session belongs to the window containing its completion timestamp. The default run uses the previous completed 24-hour UTC interval. One-hour or other positive durations are configurable, and explicit windows support deterministic replay and backfill.

An external scheduler invokes the library; the package does not run a daemon.

## Per-Agent Sample Size

For agent population $N$, requested margin $E$, confidence critical value $z$, and conservative expected pass rate $p=0.5$:

$$
n_0 = \left\lceil \frac{z^2p(1-p)}{E^2} \right\rceil
$$

and:

$$
n = \min\left(N,\left\lceil \frac{n_0}{1+(n_0-1)/N} \right\rceil\right).
$$

At 95% confidence and a 10 percentage-point margin, $n_0=97$. For $N=300$, FPC recommends 74 selected sessions. Low-volume populations become censuses.

Optional agent capacity caps the recommendation directly. Optional tenant capacity is allocated proportionately across per-agent recommendations with capped Hamilton allocation. Surplus is left unused and recorded.

## Stratification and Selection

Within an agent, sessions are stratified by `(turn_count_band, channel)`. Turn bands are `1`, `2-3`, `4-7`, `8-15`, and `16+`; missing channels use `unknown`, and sessions spanning several channels use `multi`.

The selected count is allocated proportionately across occupied strata with capped Hamilton rounding, then sessions are drawn uniformly without replacement. Inputs are canonically sorted and each agent's random seed is derived from policy version, configured seed, tenant, and agent. Reversing input order therefore reproduces the same selected IDs and run ID.

Every selected record persists:

- stratum key;
- $N_h$ and $n_h$ in the manifest;
- inclusion probability $\pi_i=n_h/N_h$;
- sampling weight $1/\pi_i$; and
- selection reason.

## Evidence, Judge, and Failures

Judges receive deterministic `evidence-packet-v2` JSON rather than raw traces. Full sessions are emitted unchanged when they fit. Oversized sessions preserve structural slots for first/final turns and the latest tool call, allocate bounded task/outcome evidence, omit middle objects as needed, and record exact omission counts.

The provider-neutral `AsyncJudge` contract supports a deterministic stub, synchronous adapters, Foundry GPT-5, and a hosted Maven CAPI availability probe. Requests have stable idempotency keys over run, unit, metric, evidence, and judge identity.

Timeouts and explicit transient provider failures retry. Terminal response, authentication, content-policy, and schema failures are recorded by safe code. Failed selected sessions are never replaced, and missing judge outcomes are not converted to Task Completion failures.

## Reporting

Binary Task Completion reports selected, submitted, succeeded, provider-failed, passes, outcome failures, response rate, pass rate, and Wilson interval with FPC. Likert, scalar, and categorical values remain native.

The current point estimator is an unweighted session mean under proportionate within-agent allocation. Inclusion probabilities remain available for audit. Disproportionate allocation, cross-agent aggregation, or nonresponse adjustment requires a design-weighted estimator and variance.

## Experiments

`random_sampling_experiments.ipynb` loads the labeled 300-session synthetic Agent 365 OTLP dataset and compares deterministic random seeds against one reused census judge pass. The experiment persists only scoped IDs, predictions, aggregate metrics, judge descriptor, and prompt fingerprint; it excludes credentials, prompts, evidence JSON, and judge reasoning.

The fixture contains 100 agents, most with tiny populations. Product-faithful per-agent planning therefore censuses many agents and saves fewer calls than pooling all 300 sessions. Pooling would answer a different estimand.

## Production Follow-ups

1. Connect the frame and UTC watermark to the production scheduler.
2. Query enough lookback to reconstruct sessions crossing window boundaries.
3. Add immutable window revisions and late-telemetry backfills.
4. Add durable tenant-partitioned manifests, attempts, observations, and reports.
5. Calibrate and release-gate the real Task Completion judge.
6. Define missing-response publication and adjustment policies.
7. Add design-weighted reporting before any disproportionate allocation.
