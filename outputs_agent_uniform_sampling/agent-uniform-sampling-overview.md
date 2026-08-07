# Agent-Uniform Sampling With Bounded Evidence

## Purpose

Provide representative per-agent evaluation samples while operating under an LLM tokens-per-minute limit. Membership is selected uniformly within each tenant/agent population before token cost or session content is considered.

## Core Contract

1. Freeze eligible completed sessions and their versions.
2. Rank deterministically within each tenant/agent using identity and a seed, never token cost.
3. Persist selected membership, population size, sample size, and inclusion probability.
4. When enabled, materialize deterministic token-bounded evidence after selection.
5. Reserve evidence, prompt envelope, and completion tokens on one tokenizer basis.
6. Pace immutable selected requests under a rolling TPM limit.
7. Keep unserviceable, dropped, and nonresponse records in the selected denominator; never replace them.
8. Report response and truncation diagnostics per agent. Suppress probability-sampling intervals when selected response is incomplete.

## Evidence Priority

| Category | Mandatory when present | Weight |
|---|---:|---:|
| Final assistant outcome | Yes | 8 |
| Tool result | Yes | 7 |
| System context | Yes | 6 |
| Initial user goal | Yes | 4 |
| Later user refinement | Yes | 3 |
| Tool arguments | No | 2 |
| Earlier assistant content | No | 2 |

## Feature Flag

Bounded evidence defaults off. Flag-off queues preserve legacy raw-estimate scheduling and `OVERSIZED`. Flag-on queues require materialization before scheduling and use explicit `UNSERVICEABLE` reasons for pre-dispatch failures.

## Production Mapping

The Python JSON queue is a single-process reference adapter. In BIC Evaluations Service:

- retain the existing online rate gate;
- add completed-session uniform sampling near `SessionCompletionSelector` and `Services/Sampling`;
- persist membership and lifecycle through BJS/Cosmos patterns;
- materialize the exact judge request before `IGenAIService.ExecutePromptAsync`;
- keep CAPI as the provider path;
- add model-compatible tokenization, distributed TPM reservations, actual-usage reconciliation, and low-cardinality telemetry.

## Handoff Files

- `agent_uniform_sampling/README.md`: operator/developer entry point.
- `docs/AGENT_UNIFORM_BOUNDED_EVIDENCE_DESIGN.md`: normative behavior contract.
- `docs/BIC_EVALUATIONS_SERVICE_HANDOFF.md`: target C# repository mapping and implementation sequence.
- `agent_uniform_sampling_walkthrough.ipynb`: executable, network-free reference.
- `tests/test_agent_uniform_sampling.py`: queue and sampling acceptance tests.
- `tests/test_token_representation.py`: weighted evidence acceptance tests.

## Validation

```powershell
py -3.11 -m pytest tests/test_agent_uniform_sampling.py tests/test_token_representation.py -q
py -3.11 scripts/build_agent_uniform_handoff.py
```
