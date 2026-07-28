# random_sampling

Production-shaped prototype for scheduled Agent 365 session evaluation using only deterministic stratified random sampling.

The complete design is documented in [`docs/RANDOM_SAMPLING_DESIGN.md`](../docs/RANDOM_SAMPLING_DESIGN.md). The executable comparison is [`random_sampling_experiments.ipynb`](../random_sampling_experiments.ipynb), with sanitized results under `outputs_random_sampling/`.

## Flow

1. Normalize typed OTLP, ESP documents, or flat Kusto rows.
2. Group by tenant-scoped `sessionId`; use a configurable 30-minute inactivity fallback when absent.
3. Keep completed sessions in a half-open UTC evaluation window. The default is the previous completed 24-hour interval.
4. For each agent, calculate Cochran's binary-proportion recommendation and finite-population correction.
5. Apply optional tenant or agent capacity.
6. Allocate the selected count proportionately across `(turn_count_band, channel)` strata with capped Hamilton allocation.
7. Draw sessions uniformly without replacement using deterministic per-agent seeds.
8. Build bounded `evidence-packet-v2` JSON, execute the configured judge, and report estimates with explicit missingness and Wilson/FPC intervals.

Every selected session is estimand-eligible and carries its stratum population, selection count, inclusion probability, and sampling weight. There is no MinHash, embedding, novelty, or purposive diversity slice in this package.

## Example

```python
from datetime import timedelta

from random_sampling import DeterministicJudgeStub, RandomSamplingPipeline, SamplePolicy

pipeline = RandomSamplingPipeline(
    judge=DeterministicJudgeStub(),
    sample_policy=SamplePolicy(margin=0.10, confidence=0.95, seed=13),
    window_duration=timedelta(hours=24),
)
run = await pipeline.run(records, tenant_capacities={"tenant-id": 200})
```

For hourly operation, use `window_duration=timedelta(hours=1)`. For replay or backfill, pass an explicit `EvaluationWindow` or `window_end`.

## Judge boundary

The deterministic judge is scaffolding, not a quality signal. Implement `AsyncJudge.evaluate(JudgeRequest) -> JudgeResponse`, or implement `SyncJudge` and use `SyncJudgeAdapter`.

The package also includes:

- a Foundry/Azure OpenAI GPT-5 adapter using JSON output, `max_completion_tokens`, and no fixed temperature;
- a hosted Maven EvalHarness/CAPI availability probe;
- resumable sampled-vs-census experiments with strict judge-provenance validation.

Direct Python-to-CAPI is not supported by the verified Maven implementation; CAPI is internal to Maven's hosted .NET service and first-party routing.

## Long sessions and privacy

Full sessions are emitted unchanged when they fit the configured UTF-8 budget. Oversized sessions preserve first-task, final-outcome, and latest-tool evidence, omit audited middle objects, and truncate only at valid code-point boundaries.

Raw span blobs, user IDs, and email fields are not copied into judge packets. User/assistant content and relevant tool inputs/outputs are sent because Task Completion requires task and outcome evidence. Production deployments must apply Agent 365 retention, regional, tenant-isolation, and access-control requirements.

## Limitations

- An external scheduler must invoke daily or hourly runs.
- Source queries need enough lookback to reconstruct sessions crossing a window boundary.
- Concrete local captures verified invoke/inference shapes; tool and guardrail extraction remains tolerant.
- Judge nonresponse is missing data, not a failed outcome, and requires a publication policy.
- The current estimator assumes proportionate within-agent strata; disproportionate designs require design-weighted reporting.
