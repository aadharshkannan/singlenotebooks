from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from trace_sampling_alt.experiments import (
    DatasetGroundTruthJudge,
    build_census_batch,
    compare_from_observations,
    load_prior_census_observations,
    load_synthetic_a365_otel,
    resume_policies_vs_census,
    run_policies_vs_census,
    run_sampled_vs_census,
    save_experiment_result,
    save_multi_policy_result,
)
from trace_sampling_alt.judge import JudgeDescriptor, JudgeRequest, JudgeResponse
from trace_sampling_alt.metrics import BinaryValue, TASK_COMPLETION_V1
from trace_sampling_alt.models import SamplePolicy
from trace_sampling_alt.evaluation import RunnerConfig


def _make_doc() -> dict[str, object]:
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "microsoft.tenant.id", "value": {"stringValue": "tenant-1"}},
                        {"key": "gen_ai.agent.id", "value": {"stringValue": "agent-a"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "a" * 32,
                                "spanId": "1" * 16,
                                "name": "invoke_agent",
                                "startTime": "2026-05-04T00:00:00Z",
                                "endTime": "2026-05-04T00:00:01Z",
                                "attributes": [
                                    {"key": "gen_ai.conversation.id", "value": {"stringValue": "conv-1"}},
                                    {"key": "evaluation.expected", "value": {"intValue": 1}},
                                    {"key": "evaluation.expected.label", "value": {"stringValue": "pass"}},
                                    {"key": "evaluation.source.case_id", "value": {"stringValue": "case-1"}},
                                    {
                                        "key": "gen_ai.input.messages",
                                        "value": {"stringValue": '[{"role":"user","content":"u1"}]'},
                                    },
                                    {
                                        "key": "gen_ai.output.messages",
                                        "value": {"stringValue": '[{"role":"assistant","content":"a1"}]'},
                                    },
                                ],
                            },
                            {
                                "traceId": "b" * 32,
                                "spanId": "2" * 16,
                                "name": "invoke_agent",
                                "startTime": "2026-05-04T00:00:02Z",
                                "endTime": "2026-05-04T00:00:03Z",
                                "attributes": [
                                    {"key": "gen_ai.conversation.id", "value": {"stringValue": "conv-2"}},
                                    {"key": "evaluation.expected", "value": {"intValue": 0}},
                                    {"key": "evaluation.expected.label", "value": {"stringValue": "fail"}},
                                    {"key": "evaluation.source.case_id", "value": {"stringValue": "case-2"}},
                                    {
                                        "key": "gen_ai.input.messages",
                                        "value": {"stringValue": '[{"role":"user","content":"u2"}]'},
                                    },
                                    {
                                        "key": "gen_ai.output.messages",
                                        "value": {"stringValue": '[{"role":"assistant","content":"a2"}]'},
                                    },
                                ],
                            },
                        ]
                    }
                ],
            }
        ]
    }


def test_load_synthetic_dataset_labels_join_and_window(tmp_path: Path):
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(_make_doc()), encoding="utf-8")

    dataset = load_synthetic_a365_otel(path)
    assert len(dataset.normalization.units) == 2
    assert len(dataset.labels_by_unit) == 2
    assert set(dataset.labels_by_conversation) == {"conv-1", "conv-2"}
    assert len(dataset.labels_by_conversation_scoped) == 2
    assert dataset.evaluation_window.start_at < dataset.evaluation_window.end_at


def test_load_synthetic_dataset_scoped_conversation_labels_with_duplicate_ids(tmp_path: Path):
    doc = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "microsoft.tenant.id", "value": {"stringValue": "tenant-1"}},
                        {"key": "gen_ai.agent.id", "value": {"stringValue": "agent-a"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "a" * 32,
                                "spanId": "1" * 16,
                                "name": "invoke_agent",
                                "startTime": "2026-05-04T00:00:00Z",
                                "endTime": "2026-05-04T00:00:01Z",
                                "attributes": [
                                    {"key": "gen_ai.conversation.id", "value": {"stringValue": "conv-shared"}},
                                    {"key": "evaluation.expected", "value": {"intValue": 1}},
                                    {"key": "gen_ai.input.messages", "value": {"stringValue": '[{"role":"user","content":"u1"}]'}},
                                    {"key": "gen_ai.output.messages", "value": {"stringValue": '[{"role":"assistant","content":"a1"}]'}},
                                    {"key": "microsoft.session.id", "value": {"stringValue": "sess-a"}},
                                ],
                            }
                        ]
                    }
                ],
            },
            {
                "resource": {
                    "attributes": [
                        {"key": "microsoft.tenant.id", "value": {"stringValue": "tenant-1"}},
                        {"key": "gen_ai.agent.id", "value": {"stringValue": "agent-b"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "b" * 32,
                                "spanId": "2" * 16,
                                "name": "invoke_agent",
                                "startTime": "2026-05-04T00:00:02Z",
                                "endTime": "2026-05-04T00:00:03Z",
                                "attributes": [
                                    {"key": "gen_ai.conversation.id", "value": {"stringValue": "conv-shared"}},
                                    {"key": "evaluation.expected", "value": {"intValue": 0}},
                                    {"key": "gen_ai.input.messages", "value": {"stringValue": '[{"role":"user","content":"u2"}]'}},
                                    {"key": "gen_ai.output.messages", "value": {"stringValue": '[{"role":"assistant","content":"a2"}]'}},
                                    {"key": "microsoft.session.id", "value": {"stringValue": "sess-b"}},
                                ],
                            }
                        ]
                    }
                ],
            },
        ]
    }
    path = tmp_path / "dup-scoped.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    dataset = load_synthetic_a365_otel(path)

    assert len(dataset.labels_by_conversation_scoped) == 2
    assert ("tenant-1", "agent-a", "conv-shared") in dataset.labels_by_conversation_scoped
    assert ("tenant-1", "agent-b", "conv-shared") in dataset.labels_by_conversation_scoped
    # Public convenience map keeps only globally unique IDs.
    assert "conv-shared" not in dataset.labels_by_conversation


def test_build_census_batch_all_core():
    doc = _make_doc()
    tmp = Path(".tmp-mini-otel.json")
    try:
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        dataset = load_synthetic_a365_otel(tmp)
    finally:
        if tmp.exists():
            tmp.unlink()

    policy = SamplePolicy(diversity_enabled=True, diversity_fraction=0.5)
    batch = build_census_batch(dataset.normalization.units, policy, dataset.evaluation_window)
    assert len(batch.agents) == 1
    assert batch.agents[0].plan.census is True
    assert len(batch.agents[0].core) == 2
    assert len(batch.agents[0].diversity) == 0


class _CountingJudge:
    descriptor = JudgeDescriptor(provider="dataset", name="ground-truth", version="v1")

    def __init__(self, labels_by_unit: dict[str, bool]) -> None:
        self.labels_by_unit = labels_by_unit
        self.calls = 0

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        self.calls += 1
        return JudgeResponse(
            request_id=request.request_id,
            metric=request.metric,
            value=BinaryValue(passed=self.labels_by_unit[request.unit_id]),
            reasoning="reasoning not persisted",
        )


def _prior_payload_from_units(
    dataset,
    policy: SamplePolicy,
    included_unit_ids: set[str],
    metric_id: str = "task_completion",
    metric_version: str = "v1",
) -> dict[str, object]:
    batch = build_census_batch(dataset.normalization.units, policy, dataset.evaluation_window)
    rows = []
    for sampled in batch.all_units():
        uid = sampled.unit.unit_id or ""
        if uid not in included_unit_ids:
            continue
        rows.append(
            {
                "unit_id": uid,
                "tenant_id": sampled.unit.tenant_id,
                "agent_id": sampled.unit.agent_id,
                "sample_kind": "core",
                "metric_id": metric_id,
                "metric_version": metric_version,
                "prediction": int(dataset.labels_by_unit[uid]),
            }
        )

    return {
        "version": "multi-policy-sampled-vs-census-v2",
        "census": {
            "run_id": batch.run_id,
            "status": "partial",
            "selected_count": len(batch.all_units()),
            "planned_request_count": len(batch.all_units()),
            "judge_descriptor": {
                "provider": "dataset",
                "name": "ground-truth",
                "version": "v1",
            },
            "failure_count": max(0, len(batch.all_units()) - len(rows)),
            "observations": rows,
        },
        "policies": {},
    }


def test_run_sampled_vs_census_judges_once_and_reuses_subset(tmp_path: Path):
    async def _case() -> None:
        path = tmp_path / "mini.json"
        path.write_text(json.dumps(_make_doc()), encoding="utf-8")
        dataset = load_synthetic_a365_otel(path)
        judge = _CountingJudge(dataset.labels_by_unit)

        result = await run_sampled_vs_census(
            dataset=dataset,
            judge=judge,
            sample_policy=SamplePolicy(diversity_enabled=False, diversity_fraction=0.0),
            runner_config=RunnerConfig(max_concurrency=2, max_attempts=1, base_backoff_seconds=0.0),
            max_evidence_bytes=8192,
            metric=TASK_COMPLETION_V1,
        )

        assert judge.calls == len(dataset.normalization.units)
        assert result.census.evaluation.selected_count == len(dataset.normalization.units)
        assert result.sampled.evaluation.selected_count <= result.census.evaluation.selected_count
        assert result.comparison["calls_saved"] == (
            result.census.evaluation.selected_count - result.sampled.evaluation.selected_count
        )
        assert result.comparison["sampled_core_evaluated"] == len(
            [o for o in result.sampled.evaluation.observations if o.sample_kind == "core"]
        )

    asyncio.run(_case())


def test_run_policies_vs_census_judges_census_once_and_reuses_for_each_policy(tmp_path: Path):
    async def _case() -> None:
        path = tmp_path / "mini.json"
        path.write_text(json.dumps(_make_doc()), encoding="utf-8")
        dataset = load_synthetic_a365_otel(path)
        judge = _CountingJudge(dataset.labels_by_unit)

        result = await run_policies_vs_census(
            dataset=dataset,
            judge=judge,
            policies={
                "prob": SamplePolicy(diversity_enabled=False, diversity_fraction=0.0),
                "div": SamplePolicy(diversity_enabled=True, diversity_fraction=0.5),
            },
            runner_config=RunnerConfig(max_concurrency=2, max_attempts=1, base_backoff_seconds=0.0),
            max_evidence_bytes=8192,
            metric=TASK_COMPLETION_V1,
        )

        assert judge.calls == len(dataset.normalization.units)
        assert set(result.by_policy.keys()) == {"prob", "div"}
        assert result.census.evaluation.selected_count == len(dataset.normalization.units)
        assert result.by_policy["div"].sampled.evaluation.selected_count >= result.by_policy["prob"].sampled.evaluation.selected_count

    asyncio.run(_case())


def test_compare_multi_agent_non_census_weighted_interval_contains_estimate_and_truth(tmp_path: Path):
    from types import SimpleNamespace as NS

    path = tmp_path / "mini.json"
    path.write_text(json.dumps(_make_doc()), encoding="utf-8")
    dataset = load_synthetic_a365_otel(path)
    metric = TASK_COMPLETION_V1

    def obs(agent: str, uid: str, passed: bool, kind: str = "core"):
        return NS(
            metric=metric,
            sample_kind=kind,
            value=BinaryValue(passed=passed),
            tenant_id="t",
            agent_id=agent,
            unit_id=uid,
            session_id=uid,
            conversation_ids=(uid,),
            request_id=f"r-{agent}-{uid}",
            agent=NS(tenant_id="t", agent_id=agent),
            judge=None,
            evidence_sha256="x",
            reasoning=None,
            estimand_eligible=True,
        )

    sampled = tuple([obs("A", f"A{i}", True) for i in range(20)] + [obs("B", f"B{i}", False) for i in range(10)])
    census = tuple([obs("A", f"A{i}", True) for i in range(100)] + [obs("B", f"B{i}", False) for i in range(10)])

    def agent_ns(agent_id: str, pop: int, census_flag: bool):
        return NS(agent=NS(tenant_id="t", agent_id=agent_id), plan=NS(population=pop, census=census_flag))

    sampled_batch = NS(
        agents=[agent_ns("A", 100, False), agent_ns("B", 10, True)],
        all_units=lambda: [0] * 30,
        policy=SamplePolicy(confidence=0.95),
    )
    census_batch = NS(all_units=lambda: [0] * 110)
    dataset_stub = NS(normalization=NS(units=[0] * 110), labels_by_unit={})

    comparison = compare_from_observations(
        dataset=dataset_stub,
        sampled_batch=sampled_batch,
        sampled_observations=sampled,
        sampled_failures=0,
        census_batch=census_batch,
        census_observations=census,
        census_failures=0,
        metric=metric,
    )

    assert comparison["sampled_core_pass_rate"] == pytest.approx(100 / 110)
    assert comparison["observed_sampled_core_pass_rate"] == pytest.approx(100 / 110)
    assert comparison["core_ci_lower"] <= comparison["sampled_core_pass_rate"] <= comparison["core_ci_upper"]
    assert comparison["census_pass_rate"] == pytest.approx(100 / 110)
    assert comparison["observed_census_pass_rate"] == pytest.approx(100 / 110)
    assert comparison["census_truth_in_interval"] is True
    assert comparison["core_ci_method"] == "agent_weighted_normal_wilson_tilde_fpc"


def test_compare_marks_incomplete_population_coverage_invalid(tmp_path: Path):
    from types import SimpleNamespace as NS

    path = tmp_path / "mini.json"
    path.write_text(json.dumps(_make_doc()), encoding="utf-8")
    dataset = load_synthetic_a365_otel(path)
    metric = TASK_COMPLETION_V1

    def obs(agent: str, uid: str, passed: bool):
        return NS(
            metric=metric,
            sample_kind="core",
            value=BinaryValue(passed=passed),
            tenant_id="t",
            agent_id=agent,
            unit_id=uid,
            session_id=uid,
            conversation_ids=(uid,),
            request_id=f"r-{agent}-{uid}",
            agent=NS(tenant_id="t", agent_id=agent),
            judge=None,
            evidence_sha256="x",
            reasoning=None,
            estimand_eligible=True,
        )

    sampled = tuple(obs("A", f"A{i}", True) for i in range(20))
    census = tuple([obs("A", f"A{i}", True) for i in range(100)] + [obs("B", f"B{i}", False) for i in range(10)])

    def agent_ns(agent_id: str, pop: int):
        return NS(agent=NS(tenant_id="t", agent_id=agent_id), plan=NS(population=pop, census=False))

    sampled_batch = NS(
        agents=[agent_ns("A", 100), agent_ns("B", 10)],
        all_units=lambda: [0] * 20,
        policy=SamplePolicy(confidence=0.95),
    )
    census_batch = NS(all_units=lambda: [0] * 110)
    dataset_stub = NS(normalization=NS(units=[0] * 110), labels_by_unit={})

    comparison = compare_from_observations(
        dataset=dataset_stub,
        sampled_batch=sampled_batch,
        sampled_observations=sampled,
        sampled_failures=0,
        census_batch=census_batch,
        census_observations=census,
        census_failures=0,
        metric=metric,
    )

    assert comparison["complete_core_population_coverage"] is False
    assert comparison["comparison_valid"] is False
    assert comparison["census_pass_rate"] is None
    assert comparison["sampled_core_pass_rate"] is None
    assert comparison["absolute_error"] is None
    assert comparison["signed_error"] is None
    assert comparison["census_truth_in_interval"] is None
    assert comparison["judge_accuracy_census_overall"] is None
    assert comparison["observed_census_pass_rate"] == pytest.approx(100 / 110)
    assert comparison["observed_sampled_core_pass_rate"] == pytest.approx(1.0)
    assert comparison["core_population_covered"] == 100
    assert comparison["core_population_coverage"] == pytest.approx(100 / 110)
    assert comparison["comparison_note"] is not None


def test_compare_math_and_save_redaction(tmp_path: Path):
    async def _case() -> None:
        path = tmp_path / "mini.json"
        path.write_text(json.dumps(_make_doc()), encoding="utf-8")
        dataset = load_synthetic_a365_otel(path)
        judge = DatasetGroundTruthJudge(dataset.labels_by_unit)

        result = await run_sampled_vs_census(
            dataset=dataset,
            judge=judge,
            sample_policy=SamplePolicy(diversity_enabled=False, diversity_fraction=0.0),
            runner_config=RunnerConfig(max_concurrency=1, max_attempts=1, base_backoff_seconds=0.0),
            max_evidence_bytes=8192,
            metric=TASK_COMPLETION_V1,
        )

        comparison = compare_from_observations(
            dataset=dataset,
            sampled_batch=result.sampled.batch,
            sampled_observations=result.sampled.evaluation.observations,
            sampled_failures=0,
            census_batch=result.census.batch,
            census_observations=result.census.evaluation.observations,
            census_failures=0,
            metric=TASK_COMPLETION_V1,
        )
        assert comparison["census_pass_rate"] == 0.5
        assert comparison["judge_accuracy_census_overall"] == 1.0

        out = tmp_path / "result.json"
        save_experiment_result(out, result)
        payload = json.loads(out.read_text(encoding="utf-8"))
        blob = json.dumps(payload)
        assert "canonical_json" not in blob
        assert "reasoning" not in blob
        assert "api_key" not in blob
        assert payload["version"] == "sampled-vs-census-v2"
        assert payload["comparison"]["population_total"] == 2
        assert payload["sampled"]["planned_request_count"] == payload["sampled"]["selected_count"]
        assert payload["sampled"]["failure_count"] == len(result.sampled.evaluation.failures)
        assert payload["census"]["judge_descriptor"] == {
            "provider": "dataset",
            "name": "ground-truth",
            "version": "v1",
        }

    asyncio.run(_case())


def test_failure_projection_and_status_reflect_sampled_subset(tmp_path: Path):
    class _FailOneJudge:
        descriptor = JudgeDescriptor(provider="test", name="fail-one", version="v1")

        def __init__(self, labels_by_unit: dict[str, bool]) -> None:
            self.labels_by_unit = labels_by_unit
            self.failed_once = False

        async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
            if not self.failed_once:
                self.failed_once = True
                raise JudgeResponseError("synthetic failure")
            return JudgeResponse(
                request_id=request.request_id,
                metric=request.metric,
                value=BinaryValue(passed=self.labels_by_unit[request.unit_id]),
                reasoning="ok",
            )

    async def _case() -> None:
        path = tmp_path / "mini.json"
        path.write_text(json.dumps(_make_doc()), encoding="utf-8")
        dataset = load_synthetic_a365_otel(path)
        judge = _FailOneJudge(dataset.labels_by_unit)

        result = await run_sampled_vs_census(
            dataset=dataset,
            judge=judge,
            sample_policy=SamplePolicy(diversity_enabled=False, diversity_fraction=0.0),
            runner_config=RunnerConfig(max_concurrency=1, max_attempts=1, base_backoff_seconds=0.0),
            max_evidence_bytes=8192,
            metric=TASK_COMPLETION_V1,
        )

        assert len(result.census.evaluation.failures) == 1
        assert len(result.sampled.evaluation.failures) == 1
        assert result.sampled.evaluation.status == "partial"
        assert result.comparison["sampled_failure_count"] == 1
        assert result.comparison["calls_saved_kind"] == "counterfactual_if_sampled_only"

    asyncio.run(_case())


def test_external_dataset_count_if_present_or_skip():
    external = Path("C:/Users/stangoodwin/eval-model-comparison/data/eval-harness/synthetic_observability.a365-otel.json")
    if not external.exists():
        pytest.skip("External synthetic dataset is not available on this machine")

    dataset = load_synthetic_a365_otel(external)
    assert len(dataset.normalization.units) == 300


def test_invalid_comparison_nulls_per_agent_headlines(tmp_path: Path):
    async def _case() -> None:
        path = tmp_path / "mini.json"
        path.write_text(json.dumps(_make_doc()), encoding="utf-8")
        dataset = load_synthetic_a365_otel(path)

        class _FailOneJudge:
            descriptor = JudgeDescriptor(provider="test", name="fail-one", version="v1")

            def __init__(self) -> None:
                self.failed_unit = next(iter(dataset.labels_by_unit))

            async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
                if request.unit_id == self.failed_unit:
                    raise JudgeResponseError("terminal test failure")
                return JudgeResponse(
                    request_id=request.request_id,
                    metric=request.metric,
                    value=BinaryValue(dataset.labels_by_unit[request.unit_id]),
                )

        result = await run_policies_vs_census(
            dataset=dataset,
            judge=_FailOneJudge(),
            policies={"probability_only": SamplePolicy(diversity_enabled=False)},
            runner_config=RunnerConfig(max_concurrency=1, max_attempts=1),
            max_evidence_bytes=8192,
        )
        comparison = result.by_policy["probability_only"].comparison
        assert comparison["comparison_valid"] is False
        for row in comparison["per_agent"]:
            assert row["census_pass_rate"] is None
            assert row["sample_core_pass_rate"] is None
            assert row["signed_error"] is None
            assert row["ci_contains_census"] is None

    asyncio.run(_case())


def test_save_multi_policy_result_excludes_evidence_and_reasoning(tmp_path: Path):
    async def _case() -> None:
        path = tmp_path / "mini.json"
        path.write_text(json.dumps(_make_doc()), encoding="utf-8")
        dataset = load_synthetic_a365_otel(path)
        result = await __import__("trace_sampling_alt").run_policies_vs_census(
            dataset=dataset,
            judge=DatasetGroundTruthJudge(dataset.labels_by_unit),
            policies={
                "probability_only": SamplePolicy(diversity_enabled=False),
                "diversity": SamplePolicy(diversity_enabled=True),
            },
            runner_config=RunnerConfig(max_concurrency=2, max_attempts=1),
            max_evidence_bytes=8192,
        )
        output = tmp_path / "multi.json"
        save_multi_policy_result(output, result)
        payload = json.loads(output.read_text(encoding="utf-8"))
        blob = json.dumps(payload)
        assert payload["version"] == "multi-policy-sampled-vs-census-v2"
        assert set(payload["policies"]) == {"probability_only", "diversity"}
        assert "canonical_json" not in blob
        assert "reasoning" not in blob
        assert "api_key" not in blob
        assert payload["census"]["judge_descriptor"] == {
            "provider": "dataset",
            "name": "ground-truth",
            "version": "v1",
        }

    asyncio.run(_case())


def test_load_prior_census_observations_validates_and_rebuilds(tmp_path: Path):
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(_make_doc()), encoding="utf-8")
    dataset = load_synthetic_a365_otel(path)
    policy = SamplePolicy(diversity_enabled=False)
    included = {unit.unit_id or "" for unit in dataset.normalization.units[:1]}
    payload = _prior_payload_from_units(dataset, policy, included)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(payload), encoding="utf-8")

    descriptor = JudgeDescriptor(provider="dataset", name="ground-truth", version="v1")
    observations, meta = load_prior_census_observations(
        path=prior,
        dataset=dataset,
        judge_descriptor=descriptor,
        policy=policy,
    )

    assert len(observations) == 1
    assert observations[0].evidence_sha256 == "not-persisted"
    assert observations[0].reasoning is None
    assert meta["loaded_count"] == 1
    assert meta["judge_descriptor"] == {
        "provider": "dataset",
        "name": "ground-truth",
        "version": "v1",
    }


def test_load_prior_census_observations_rejects_mismatched_judge_descriptor(tmp_path: Path):
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(_make_doc()), encoding="utf-8")
    dataset = load_synthetic_a365_otel(path)
    policy = SamplePolicy(diversity_enabled=False)
    included = {unit.unit_id or "" for unit in dataset.normalization.units[:1]}
    payload = _prior_payload_from_units(dataset, policy, included)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="judge_descriptor mismatch"):
        load_prior_census_observations(
            path=prior,
            dataset=dataset,
            judge_descriptor=JudgeDescriptor(provider="azure-openai-foundry", name="gpt-5", version="2024-12-01-preview"),
            policy=policy,
        )


def test_load_prior_census_observations_rejects_gpt5_artifact_for_ground_truth_descriptor(tmp_path: Path):
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(_make_doc()), encoding="utf-8")
    dataset = load_synthetic_a365_otel(path)
    policy = SamplePolicy(diversity_enabled=False)
    included = {unit.unit_id or "" for unit in dataset.normalization.units[:1]}
    payload = _prior_payload_from_units(dataset, policy, included)
    payload["census"]["judge_descriptor"] = {
        "provider": "azure-openai-foundry",
        "name": "gpt-5",
        "version": "2024-12-01-preview",
    }
    prior = tmp_path / "prior-gpt5.json"
    prior.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="judge_descriptor mismatch"):
        load_prior_census_observations(
            path=prior,
            dataset=dataset,
            judge_descriptor=JudgeDescriptor(provider="dataset", name="ground-truth", version="v1"),
            policy=policy,
        )


def test_load_prior_census_observations_rejects_legacy_v1_artifact(tmp_path: Path):
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(_make_doc()), encoding="utf-8")
    dataset = load_synthetic_a365_otel(path)
    policy = SamplePolicy(diversity_enabled=False)
    included = {unit.unit_id or "" for unit in dataset.normalization.units[:1]}
    payload = _prior_payload_from_units(dataset, policy, included)
    payload["version"] = "multi-policy-sampled-vs-census-v1"
    prior = tmp_path / "prior-v1.json"
    prior.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported prior artifact version"):
        load_prior_census_observations(
            path=prior,
            dataset=dataset,
            judge_descriptor=JudgeDescriptor(provider="dataset", name="ground-truth", version="v1"),
            policy=policy,
        )


def test_resume_policies_vs_census_calls_only_missing_and_merges_complete(tmp_path: Path):
    async def _case() -> None:
        path = tmp_path / "mini.json"
        path.write_text(json.dumps(_make_doc()), encoding="utf-8")
        dataset = load_synthetic_a365_otel(path)
        policy = SamplePolicy(diversity_enabled=False)
        all_ids = [unit.unit_id or "" for unit in dataset.normalization.units]
        prior_ids = {all_ids[0]}
        prior_payload = _prior_payload_from_units(dataset, policy, prior_ids)
        prior_path = tmp_path / "prior.json"
        prior_path.write_text(json.dumps(prior_payload), encoding="utf-8")

        judge = _CountingJudge(dataset.labels_by_unit)
        result = await resume_policies_vs_census(
            dataset=dataset,
            judge=judge,
            policies={"prob": policy},
            runner_config=RunnerConfig(max_concurrency=2, max_attempts=1, base_backoff_seconds=0.0),
            max_evidence_bytes=8192,
            prior_path=prior_path,
        )

        assert judge.calls == len(all_ids) - len(prior_ids)
        assert len(result.census.evaluation.observations) == len(all_ids)
        assert len(result.census.evaluation.failures) == 0
        assert result.census.evaluation.status == "completed"
        assert result.census.evaluation.request_count == len(all_ids)
        assert result.census.evaluation.request_count >= (
            len(result.census.evaluation.observations) + len(result.census.evaluation.failures)
        )
        assert result.by_policy["prob"].comparison["census_complete"] is True
        assert result.by_policy["prob"].comparison["comparison_valid"] is True

    asyncio.run(_case())


def test_resume_policies_vs_census_partial_census_marks_invalid(tmp_path: Path):
    class _AlwaysFailJudge:
        descriptor = JudgeDescriptor(provider="dataset", name="ground-truth", version="v1")

        async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
            raise JudgeResponseError("synthetic failure")

    async def _case() -> None:
        path = tmp_path / "mini.json"
        path.write_text(json.dumps(_make_doc()), encoding="utf-8")
        dataset = load_synthetic_a365_otel(path)
        policy = SamplePolicy(diversity_enabled=False)
        all_ids = [unit.unit_id or "" for unit in dataset.normalization.units]
        prior_ids = {all_ids[0]}
        prior_payload = _prior_payload_from_units(dataset, policy, prior_ids)
        prior_path = tmp_path / "prior.json"
        prior_path.write_text(json.dumps(prior_payload), encoding="utf-8")

        result = await resume_policies_vs_census(
            dataset=dataset,
            judge=_AlwaysFailJudge(),
            policies={"prob": policy},
            runner_config=RunnerConfig(max_concurrency=1, max_attempts=1, base_backoff_seconds=0.0),
            max_evidence_bytes=8192,
            prior_path=prior_path,
        )

        assert result.census.evaluation.status == "partial"
        comparison = result.by_policy["prob"].comparison
        assert comparison["census_complete"] is False
        assert comparison["comparison_valid"] is False

    asyncio.run(_case())
