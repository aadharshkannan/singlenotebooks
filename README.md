# singlenotebooks
Repo where I have single notebooks for one off analysis

## Adaptive Trace Sampling (`adaptive_trace_sampling.ipynb`)

**Problem:** When multiple AI agents produce OTel-style traces at high and
unequal rates, a naive random sampler either overloads the LLM backend (no
budget control), starves rare agents (they disappear from the kept stream), or
collapses variety (only the most-common signatures survive). This prototype
solves all three problems simultaneously.

**What it does:** The `trace_sampling/` package implements an online,
variety-preserving, non-starving, backpressure-bounded multi-agent trace
sampler. Key properties:

- **Anti-starvation** — every active agent keeps at least one trace per window,
  regardless of how dominant faster agents are.
- **Variety preservation** — per-agent signature coverage (distinct
  signature fraction) meets or beats a budget-matched uniform baseline.
- **Representativeness** — lower TV / KL divergence vs. the stream's true
  signature distribution, measured per agent.
- **Backpressure** — an EWMA-based rate controller caps the sustained kept-rate
  to the configured LLM throughput budget, absorbing short bursts without
  exceeding the average limit.

**Demo notebook:** `adaptive_trace_sampling.ipynb` runs a synthetic scenario
with fast/slow and low/high-variety agents plus a mid-stream burst, compares
`AdaptiveSampler` against a budget-matched `BaselineSampler`, plots four metric
families, and finishes with embedded assertions that fail the notebook if any
guarantee is violated.

**Run the tests:**
```bash
python -m pytest tests/ -q
```

**Run the notebook:**
```bash
python -m jupyter nbconvert --to notebook --execute --inplace adaptive_trace_sampling.ipynb
```
