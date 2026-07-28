from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType

import pytest

from trace_sampling_alt.azure_openai_judge import (
    AzureOpenAIJudge,
    AzureOpenAIJudgeSettings,
    JudgeAuthenticationError,
    JudgeContentFilteredError,
    JudgeLengthExhaustedError,
    JudgeMalformedJsonError,
    JudgeProviderError,
    JudgeSkippedError,
    JudgeTerminalHttpError,
)
from trace_sampling_alt.evidence import build_evidence_packet
from trace_sampling_alt.judge import JudgeRequest, JudgeResponseError, JudgeTransientError
from trace_sampling_alt.metrics import TASK_COMPLETION_V1
from trace_sampling_alt.models import EvaluationUnit, ToolCall, Turn


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._response_payload: object | list[object] = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"score": 1, "status": "completed", "reason": "task complete"}
                        )
                    }
                }
            ]
        }
        self._raise: Exception | None = None

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise is not None:
            raise self._raise
        if isinstance(self._response_payload, list):
            if not self._response_payload:
                raise AssertionError("No fake responses left")
            return self._response_payload.pop(0)
        return self._response_payload


class _FakeClient:
    def __init__(self) -> None:
        self.chat = type("_Chat", (), {"completions": _FakeChatCompletions()})()


class _RateLimitError(Exception):
    status_code = 429


class _AuthError(Exception):
    status_code = 401


class _BadRequestWithBodyError(Exception):
    status_code = 400

    def __init__(self, code: str, inner_code: str | None = None) -> None:
        inner: dict[str, str] = {}
        if inner_code is not None:
            inner["code"] = inner_code
        self.body = {
            "code": code,
            "innererror": inner,
            # Deliberately include provider message to ensure classifier does not rely on it.
            "message": "provider-internal-details",
        }


class _NotFoundError(Exception):
    status_code = 404


class _SdkUnknownError(Exception):
    pass


def _request() -> JudgeRequest:
    unit = EvaluationUnit(
        tenant_id="tenant-a",
        agent_id="agent-a",
        conversation_id="conv-a",
        session_id="sess-a",
        channel="teams",
        source_trace_ids=("trace-a",),
        started_at=None,
        ended_at=None,
        had_error=False,
        turns=(Turn(user_text="u", assistant_text="a"),),
        tool_calls=(ToolCall(name="search", input_text="q", output_text="r"),),
    )
    evidence = build_evidence_packet(unit, max_bytes=4096)
    return JudgeRequest(
        request_id="req-1",
        idempotency_key="idem-1",
        tenant_id=unit.tenant_id,
        agent_id=unit.agent_id,
        unit_id=unit.unit_id or "",
        session_id=unit.session_id,
        conversation_ids=unit.conversation_ids,
        metric=TASK_COMPLETION_V1,
        evidence=evidence,
    )


def test_gpt5_request_uses_reasoning_kwargs_and_omits_temperature():
    async def _case() -> None:
        fake = _FakeClient()
        judge = AzureOpenAIJudge(
            AzureOpenAIJudgeSettings(endpoint="https://example.test", api_key="x", deployment="gpt-5"),
            client=fake,
        )
        response = await judge.evaluate(_request())

        assert response.value.passed is True
        payload = fake.chat.completions.calls[-1]
        assert payload["model"] == "gpt-5"
        assert "max_completion_tokens" in payload
        assert payload["max_completion_tokens"] >= 4000
        assert "max_tokens" not in payload
        assert "temperature" not in payload
        assert payload["response_format"] == {"type": "json_object"}

    asyncio.run(_case())


def test_non_reasoning_request_uses_max_tokens_and_temperature_zero():
    async def _case() -> None:
        fake = _FakeClient()
        judge = AzureOpenAIJudge(
            AzureOpenAIJudgeSettings(endpoint="https://example.test", api_key="x", deployment="gpt-4o-mini"),
            client=fake,
        )
        _ = await judge.evaluate(_request())

        payload = fake.chat.completions.calls[-1]
        assert payload["model"] == "gpt-4o-mini"
        assert "max_tokens" in payload
        assert payload["temperature"] == 0
        assert "max_completion_tokens" not in payload

    asyncio.run(_case())


def test_parses_fail_string_score():
    async def _case() -> None:
        fake = _FakeClient()
        fake.chat.completions._response_payload = {
            "choices": [{"message": {"content": json.dumps({"score": "0", "status": "completed", "reason": "no"})}}]
        }
        judge = AzureOpenAIJudge(
            AzureOpenAIJudgeSettings(endpoint="https://example.test", api_key="x"),
            client=fake,
        )
        response = await judge.evaluate(_request())
        assert response.value.passed is False

    asyncio.run(_case())


def test_malformed_json_raises_response_error():
    async def _case() -> None:
        fake = _FakeClient()
        fake.chat.completions._response_payload = {
            "choices": [{"message": {"content": "not-json"}}]
        }
        judge = AzureOpenAIJudge(
            AzureOpenAIJudgeSettings(endpoint="https://example.test", api_key="x"),
            client=fake,
        )
        with pytest.raises(JudgeMalformedJsonError, match="malformed JSON"):
            await judge.evaluate(_request())

    asyncio.run(_case())


def test_skipped_or_null_score_is_terminal_error():
    async def _case() -> None:
        fake = _FakeClient()
        judge = AzureOpenAIJudge(
            AzureOpenAIJudgeSettings(endpoint="https://example.test", api_key="x"),
            client=fake,
        )

        fake.chat.completions._response_payload = {
            "choices": [{"message": {"content": json.dumps({"score": 1, "status": "skipped", "reason": "insufficient"})}}]
        }
        with pytest.raises(JudgeSkippedError):
            await judge.evaluate(_request())

        fake.chat.completions._response_payload = {
            "choices": [{"message": {"content": json.dumps({"score": None, "status": "completed", "reason": "x"})}}]
        }
        with pytest.raises(JudgeResponseError):
            await judge.evaluate(_request())

    asyncio.run(_case())


def test_transient_and_terminal_error_classification():
    async def _case() -> None:
        fake = _FakeClient()
        judge = AzureOpenAIJudge(
            AzureOpenAIJudgeSettings(endpoint="https://example.test", api_key="x"),
            client=fake,
        )

        fake.chat.completions._raise = _RateLimitError("rate")
        with pytest.raises(JudgeTransientError):
            await judge.evaluate(_request())

        fake.chat.completions._raise = _AuthError("auth")
        with pytest.raises(JudgeAuthenticationError):
            await judge.evaluate(_request())

        fake.chat.completions._raise = _NotFoundError("missing")
        with pytest.raises(JudgeTerminalHttpError) as exc_info:
            await judge.evaluate(_request())
        assert exc_info.value.status_code == 404

        fake.chat.completions._raise = _SdkUnknownError("sdk")
        with pytest.raises(JudgeProviderError):
            await judge.evaluate(_request())

    asyncio.run(_case())


def test_structured_body_content_filter_code_maps_to_content_filtered_error():
    async def _case() -> None:
        fake = _FakeClient()
        judge = AzureOpenAIJudge(
            AzureOpenAIJudgeSettings(endpoint="https://example.test", api_key="x"),
            client=fake,
        )

        fake.chat.completions._raise = _BadRequestWithBodyError(code="content_filter")
        with pytest.raises(JudgeContentFilteredError, match="blocked by content policy"):
            await judge.evaluate(_request())

    asyncio.run(_case())


def test_structured_body_inner_responsible_ai_maps_to_content_filtered_error():
    async def _case() -> None:
        fake = _FakeClient()
        judge = AzureOpenAIJudge(
            AzureOpenAIJudgeSettings(endpoint="https://example.test", api_key="x"),
            client=fake,
        )

        fake.chat.completions._raise = _BadRequestWithBodyError(
            code="bad_request",
            inner_code="ResponsibleAIPolicyViolation",
        )
        with pytest.raises(JudgeContentFilteredError, match="blocked by content policy"):
            await judge.evaluate(_request())

    asyncio.run(_case())


def test_reasoning_model_length_retry_increases_budget_and_succeeds():
    async def _case() -> None:
        fake = _FakeClient()
        fake.chat.completions._response_payload = [
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": ""},
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps({"score": 1, "status": "completed", "reason": "ok"})
                        },
                    }
                ]
            },
        ]

        judge = AzureOpenAIJudge(
            AzureOpenAIJudgeSettings(endpoint="https://example.test", api_key="x", deployment="gpt-5", max_completion_tokens=5000),
            client=fake,
        )
        response = await judge.evaluate(_request())

        assert response.value.passed is True
        assert len(fake.chat.completions.calls) == 2
        first_budget = int(fake.chat.completions.calls[0]["max_completion_tokens"])
        second_budget = int(fake.chat.completions.calls[1]["max_completion_tokens"])
        assert second_budget > first_budget
        assert second_budget <= 12000

    asyncio.run(_case())


def test_reasoning_model_length_retry_then_empty_raises_token_exhaustion_error():
    async def _case() -> None:
        fake = _FakeClient()
        fake.chat.completions._response_payload = [
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": ""},
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": ""},
                    }
                ]
            },
        ]
        judge = AzureOpenAIJudge(
            AzureOpenAIJudgeSettings(endpoint="https://example.test", api_key="x", deployment="gpt-5"),
            client=fake,
        )

        with pytest.raises(JudgeLengthExhaustedError, match="token exhaustion"):
            await judge.evaluate(_request())

    asyncio.run(_case())


def test_reasoning_model_malformed_json_on_length_retries_once_with_12000_and_succeeds():
    async def _case() -> None:
        fake = _FakeClient()
        fake.chat.completions._response_payload = [
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "{\"score\": 1"},
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps({"score": 1, "status": "completed", "reason": "ok"})
                        },
                    }
                ]
            },
        ]

        judge = AzureOpenAIJudge(
            AzureOpenAIJudgeSettings(endpoint="https://example.test", api_key="x", deployment="gpt-5", max_completion_tokens=5000),
            client=fake,
        )
        response = await judge.evaluate(_request())
        assert response.value.passed is True
        assert len(fake.chat.completions.calls) == 2
        assert int(fake.chat.completions.calls[1]["max_completion_tokens"]) == 12000

    asyncio.run(_case())


def test_async_azure_client_constructed_with_max_retries_zero(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = ModuleType("openai")
    setattr(fake_module, "AsyncAzureOpenAI", _FakeAsyncAzureOpenAI)

    monkeypatch.setitem(sys.modules, "openai", fake_module)
    _ = AzureOpenAIJudge(AzureOpenAIJudgeSettings(endpoint="https://example.test", api_key="x"))
    assert captured.get("max_retries") == 0
