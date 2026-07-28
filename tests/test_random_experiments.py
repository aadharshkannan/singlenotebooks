from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from random_sampling import (
    BinaryValue,
    DatasetGroundTruthJudge,
    EvaluationFailure,
    JudgeDescriptor,
    JudgeRequest,
    JudgeResponse,
    JudgeResponseError,
    RunnerConfig,
    SamplePolicy,
    TASK_COMPLETION_V1,
    build_census_batch,
    load_prior_census_observations,
    load_synthetic_a365_otel,
    resume_policies_vs_census,
    run_policies_vs_census,
    run_sampled_vs_census,
    save_experiment_result,
    save_multi_policy_result,
)


def _document(count: int = 2) -> dict:
    spans = []
    for index in range(count):
        spans.append({
            "traceId": f"{index + 1:032x}",
            "spanId": f"{index + 1:016x}",
            "name": "invoke_agent",
            "startTime": f"2026-05-04T00:00:{index * 2:02d}Z",
            "endTime": f"2026-05-04T00:00:{index * 2 + 1:02d}Z",
            "attributes": [
                {"key": "gen_ai.conversation.id", "value": {"stringValue": f"conv-{index}"}},
                {"key": "evaluation.expected", "value": {"intValue": index % 2}},
                {"key": "evaluation.expected.label", "value": {"stringValue": "pass" if index % 2 else "fail"}},
                {"key": "evaluation.source.case_id", "value": {"stringValue": f"case-{index}"}},
                {"key": "gen_ai.input.messages", "value": {"stringValue": f'[{json.dumps({"role": "user", "content": f"u{index}"})}]'}},
                {"key": "gen_ai.output.messages", "value": {"stringValue": f'[{json.dumps({"role": "assistant", "content": f"a{index}"})}]'}},
            ],
        })
    return {
        "resourceSpans": [{
            "resource": {"attributes": [
                {"key": "microsoft.tenant.id", "value": {"stringValue": "tenant-1"}},
                {"key": "gen_ai.agent.id", "value": {"stringValue": "agent-a"}},
            ]},
            "scopeSpans": [{"spans": spans}],
        }]
    }


def _dataset(tmp_path: Path, count: int = 2):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(_document(count)), encoding="utf-8")
    return load_synthetic_a365_otel(path)


class _CountingJudge:
    descriptor = JudgeDescriptor("dataset", "ground-truth", "v1")

    def __init__(self, labels: dict[str, bool]) -> None:
        self.labels = labels
        self.calls = 0

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        self.calls += 1
        return JudgeResponse(request.request_id, request.metric, BinaryValue(self.labels[request.unit_id]))


def test_load_dataset_and_build_census(tmp_path: Path):
    dataset = _dataset(tmp_path)
    assert len(dataset.normalization.units) == 2
    assert len(dataset.labels_by_unit) == 2

    batch = build_census_batch(dataset.normalization.units, SamplePolicy(), dataset.evaluation_window)
    assert batch.agents[0].plan.census is True
    assert len(batch.agents[0].units) == 2
    assert all(row.inclusion_probability == 1.0 for row in batch.agents[0].units)


def test_sampled_vs_census_judges_once(tmp_path: Path):
    async def case():
        dataset = _dataset(tmp_path)
        judge = _CountingJudge(dataset.labels_by_unit)
        result = await run_sampled_vs_census(
            dataset,
            judge,
            SamplePolicy(seed=13),
            RunnerConfig(max_concurrency=2, max_attempts=1),
            8192,
        )
        assert judge.calls == 2
        assert result.comparison["census_complete"] is True
        assert result.comparison["comparison_valid"] is True
        assert result.comparison["sampled_units_evaluated"] == 2
    asyncio.run(case())


def test_multiple_random_seeds_reuse_one_census(tmp_path: Path):
    async def case():
        dataset = _dataset(tmp_path, count=40)
        judge = _CountingJudge(dataset.labels_by_unit)
        result = await run_policies_vs_census(
            dataset,
            judge,
            {"seed_13": SamplePolicy(seed=13), "seed_29": SamplePolicy(seed=29)},
            RunnerConfig(max_concurrency=8, max_attempts=1),
            8192,
        )
        assert judge.calls == 40
        assert set(result.by_policy) == {"seed_13", "seed_29"}
        assert all(arm.comparison["comparison_valid"] for arm in result.by_policy.values())
    asyncio.run(case())


def test_ground_truth_comparison_math(tmp_path: Path):
    async def case():
        dataset = _dataset(tmp_path, count=120)
        result = await run_sampled_vs_census(
            dataset,
            DatasetGroundTruthJudge(dataset.labels_by_unit),
            SamplePolicy(seed=13),
            RunnerConfig(max_concurrency=16, max_attempts=1),
            8192,
        )
        comparison = result.comparison
        assert comparison["census_pass_rate"] == 0.5
        assert comparison["sampled_units_evaluated"] < 120
        assert comparison["sample_ci_lower"] <= comparison["sampled_units_pass_rate"] <= comparison["sample_ci_upper"]
        assert comparison["judge_accuracy_census_overall"] == 1.0
    asyncio.run(case())


class _FailOneJudge(_CountingJudge):
    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        self.calls += 1
        if self.calls == 1:
            raise JudgeResponseError("terminal test failure")
        return JudgeResponse(request.request_id, request.metric, BinaryValue(self.labels[request.unit_id]))


def test_partial_census_nulls_headlines(tmp_path: Path):
    async def case():
        dataset = _dataset(tmp_path)
        result = await run_sampled_vs_census(
            dataset,
            _FailOneJudge(dataset.labels_by_unit),
            SamplePolicy(),
            RunnerConfig(max_concurrency=1, max_attempts=1),
            8192,
        )
        comparison = result.comparison
        assert comparison["comparison_valid"] is False
        assert comparison["census_pass_rate"] is None
        assert comparison["sampled_units_pass_rate"] is None
        assert comparison["observed_census_pass_rate"] is not None
    asyncio.run(case())


def test_save_artifacts_are_random_specific_and_sanitized(tmp_path: Path):
    async def case():
        dataset = _dataset(tmp_path)
        judge = DatasetGroundTruthJudge(dataset.labels_by_unit)
        single = await run_sampled_vs_census(dataset, judge, SamplePolicy(), RunnerConfig(max_attempts=1), 8192)
        single_path = tmp_path / "single.json"
        save_experiment_result(single_path, single)
        single_payload = json.loads(single_path.read_text(encoding="utf-8"))
        assert single_payload["version"] == "random-sampled-vs-census-v1"

        multi = await run_policies_vs_census(dataset, judge, {"seed_13": SamplePolicy()}, RunnerConfig(max_attempts=1), 8192)
        multi_path = tmp_path / "multi.json"
        save_multi_policy_result(multi_path, multi)
        payload = json.loads(multi_path.read_text(encoding="utf-8"))
        blob = json.dumps(payload)
        assert payload["version"] == "random-multi-policy-sampled-vs-census-v1"
        assert "sample_kind" not in blob
        assert "canonical_json" not in blob
        assert "reasoning" not in blob
        assert "api_key" not in blob
    asyncio.run(case())


def test_resume_reuses_prior_and_rejects_other_judge(tmp_path: Path):
    async def case():
        dataset = _dataset(tmp_path, count=4)
        policy = SamplePolicy(seed=13)
        initial = await run_policies_vs_census(
            dataset,
            DatasetGroundTruthJudge(dataset.labels_by_unit),
            {"seed_13": policy},
            RunnerConfig(max_attempts=1),
            8192,
        )
        path = tmp_path / "prior.json"
        save_multi_policy_result(path, initial)

        observations, _ = load_prior_census_observations(path, dataset, DatasetGroundTruthJudge(dataset.labels_by_unit).descriptor, policy=policy)
        assert len(observations) == 4
        with pytest.raises(ValueError, match="judge_descriptor mismatch"):
            load_prior_census_observations(path, dataset, JudgeDescriptor("other", "judge", "v1"), policy=policy)

        counting = _CountingJudge(dataset.labels_by_unit)
        resumed = await resume_policies_vs_census(
            dataset,
            counting,
            {"seed_13": policy},
            RunnerConfig(max_attempts=1),
            8192,
            path,
        )
        assert counting.calls == 0
        assert resumed.census.evaluation.status == "completed"
    asyncio.run(case())


def test_external_dataset_count_if_present_or_skip():
    external = Path(r"C:\Users\stangoodwin\eval-model-comparison\data\eval-harness\synthetic_observability.a365-otel.json")
    if not external.exists():
        pytest.skip("external synthetic dataset unavailable")
    assert len(load_synthetic_a365_otel(external).normalization.units) == 300
