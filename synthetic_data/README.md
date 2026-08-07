# Retained Synthetic Sampling Data

This directory contains every synthetic source needed by the Agent365 Sampling V2 experiment.

## `a365_dense_2500/`

- Five synthetic agents with 500 sessions each.
- The merged strict OTLP corpus is under `corpus/`.
- All 25 source shards and shard manifests are retained under `shards/`.
- Generation seed: `3652026`.

## `a365_historical_300/`

- 300 synthetic sessions across 100 agents.
- The normalized Agent365 OTLP source, its metadata, the upstream BPS source, and the converter script are retained together.
- These files were copied from the sibling `eval-model-comparison` workspace; the original paths and hashes are recorded in `provenance.json`.

Run `sampling_v2_runbook.ipynb` to validate hashes and load both sources. The experiment uses the existing expected labels after selection and does not call an LLM judge.
