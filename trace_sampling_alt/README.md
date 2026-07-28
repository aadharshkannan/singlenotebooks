# trace_sampling_alt

Minimal, stdlib-only prototype of a scheduled Agent 365 session-sampling and
judge pipeline.

The complete implemented design is documented in
[`docs/TRACE_SAMPLING_ALT_DESIGN.md`](../docs/TRACE_SAMPLING_ALT_DESIGN.md).

The executable sampled-vs-census study is in
[`trace_sampling_alt_experiments.ipynb`](../trace_sampling_alt_experiments.ipynb).
It loads the labeled 300-session synthetic Agent 365 OTLP artifact, runs
probability-only and 20% MinHash-diversity policies against one reused census
judge pass, and writes sanitized results under `outputs_alt_sampling/`.

The notebook first probes a hosted Maven EvalHarness for the CAPI-backed
`gpt-5` model. Direct Python-to-CAPI is not supported by the verified Maven
implementation: CAPI is internal to Maven's .NET service, first-party auth, and
island routing. When Maven is unavailable, the opt-in live path uses the
configured Azure OpenAI/Foundry `gpt-5` deployment with Maven-compatible
reasoning parameters (`max_completion_tokens`, JSON response, no fixed
temperature). Live execution is disabled unless `RUN_LIVE_GPT5=1` is set.

The fixture contains 300 sessions across 100 agents. Since planning is per
agent, many small populations become censuses; the product-faithful sampled arm
therefore saves fewer calls than a pooled 300-session experiment would. Pooling
would answer a different estimand and is not used for the headline comparison.

## Architecture

1. Input normalization (`agent365_otel.py`)
- Supports typed OTLP-style envelopes, ESP-style `documents` wrappers, and flat/Kusto-like rows.
- Groups by tenant-scoped `sessionId`; missing IDs fall back to 30-minute inactivity sessions.
- Produces immutable completed session units plus `IngestIssue` records.

2. Sampling (`sampling.py`)
- Applies Cochran/FPC independently to each agent's eligible sessions in the window.
- Optional tenant-scoped conversation capacity is allocated exactly across per-agent statistical recommendations; surplus is left unused and recorded in the manifest.
- The default window is the previous completed 24-hour UTC interval; configure one hour or pass an explicit window for replay.
- Optional MinHash diversity reserves 20% inside the total budget, not on top of it.
- MinHash uses tagged n-grams from user/assistant text and tool names, inputs, and outputs.

3. Evidence boundary (`evidence.py`)
- Builds `evidence-packet-v2` canonical JSON with a hard UTF-8 byte limit.
- Full sessions remain unchanged when they fit.
- Oversized sessions retain first/final turn and latest-tool slots, omit middle objects, truncate text on code-point boundaries, and record exact omission counts.
- Excludes raw span payloads and user IDs/emails.

4. Judge boundary (`judge.py`)
- `AsyncJudge` protocol for async evaluation.
- `SyncJudge` + `SyncJudgeAdapter` for plugging sync implementations without changing runner code.
- `DeterministicJudgeStub` is the only implementation here and is intentionally non-production.

5. Evaluation runner (`evaluation.py`)
- One request per sampled unit+metric.
- Stable idempotency/request identifiers.
- Bounded concurrency, timeout, transient retries, deterministic ordering.
- Explicit failures; no replacement of selected units.

6. Reporting (`reporting.py`)
- Core binary headline uses succeeded responses only and Wilson interval with finite population correction.
- Diversity summaries are descriptive only and do not affect headline estimands.
- Likert/scalar/categorical summaries are reported without coercion.

7. Orchestrator (`pipeline.py`)
- `AlternativeSamplingPipeline.run(...)` wires normalize -> sample -> evaluate -> report.

## Statistical behavior

- Core sample is estimand-eligible and used for binary headline pass-rate/CI.
- Judge failures/missing responses are explicit and do not become failed outcomes in denominators.
- Diversity sample and non-core metrics cannot alter core headline values.
- Reserving diversity reduces the probability-core count, so headline precision is wider than when the full budget is probability sampled.

## Judge swap contract (future CAPI/Foundry)

- Implement `AsyncJudge.evaluate(JudgeRequest) -> JudgeResponse`.
- Or implement `SyncJudge` and wrap with `SyncJudgeAdapter`.
- Request contains only bounded `EvidencePacket`, metric spec, scoped identity, and idempotency key.

## Example async usage

```python
import asyncio
from datetime import timedelta

from trace_sampling_alt import (
    AlternativeSamplingPipeline,
    DeterministicJudgeStub,
    SamplePolicy,
)


async def main(records):
    pipeline = AlternativeSamplingPipeline(
        judge=DeterministicJudgeStub(),
        sample_policy=SamplePolicy(diversity_enabled=True),
        window_duration=timedelta(hours=24),
    )
    run = await pipeline.run(records, tenant_capacities={"tenant-id": 200})
    return run.report


# report = asyncio.run(main(records))
```

## Observed input support

- OTLP-like nested envelopes (`traceRequest.resourceSpans[].scopeSpans[].spans[]`)
- ESP document envelopes (`documents[].jsonContent` or object docs)
- Flat Kusto rows with PascalCase aliases and ISO or numeric epoch timestamps
- Message bodies in `role/content`, `role/parts`, and `{messages: [...]}` forms

## Limitations

- Stub judge output is deterministic scaffolding and not a quality signal.
- A CAPI or Foundry adapter is not included; either can implement the provider-neutral judge protocol.
- An external scheduler must invoke the library. The package defines aligned daily/hourly windows but does not run a daemon.
- Input should include enough lookback to reconstruct sessions crossing a window boundary; sessions are attributed by completion time.
- Concrete local captures verified `invoke_agent` and `inference` spans. Tool and guardrail extraction is tolerant, but their exact production attribute schemas were not locally verified.
- Structural truncation is deterministic extraction, not LLM summarization and not guaranteed semantically equivalent to full context.

## Privacy notes

- Evidence builder avoids raw span blobs and only emits scoped structured conversation/tool fields.
- User and assistant message content, system-relevant tool inputs, and tool outputs are sent to the configured judge because Task Completion requires that evidence. Deployments must apply Agent 365 retention, access-control, and regional data-handling requirements.
- User IDs and email fields are not copied into normalized units or judge packets.
- Reasoning text is sanitized and bounded.

## Failure semantics

- Judge transport/transient failures are retryable up to configured attempts.
- Malformed judge responses are terminal for that request.
- Failures are explicit in `EvaluationRun.failures`; selected units are never replaced.
