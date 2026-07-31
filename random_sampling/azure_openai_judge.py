"""Azure OpenAI-backed AsyncJudge implementation for binary task completion."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .judge import (
    AsyncJudge,
    JudgeDescriptor,
    JudgeRequest,
    JudgeResponse,
    JudgeResponseError,
    JudgeTransientError,
)
from .metrics import BinaryValue


class JudgeLengthExhaustedError(JudgeResponseError):
    """Length stop exhausted budget even after retry."""


class JudgeEmptyResponseError(JudgeResponseError):
    """Provider returned an empty content payload."""


class JudgeMalformedJsonError(JudgeResponseError):
    """Provider returned non-JSON or malformed JSON content."""


class JudgeContentFilteredError(JudgeResponseError):
    """Provider rejected the request due to Responsible AI content policy."""


class JudgeTerminalHttpError(JudgeResponseError):
    """Terminal provider HTTP response with a safe status code only."""

    def __init__(self, status_code: int) -> None:
        self.status_code = int(status_code)
        super().__init__(f"Azure OpenAI terminal HTTP status {self.status_code}")


class JudgeAuthenticationError(JudgeResponseError):
    """Terminal provider authentication or authorization failure."""


class JudgeProviderError(JudgeResponseError):
    """Terminal provider failure without a safe HTTP status classification."""


class JudgeSkippedError(JudgeResponseError):
    """Judge reported skipped status for required experiments."""


@dataclass(frozen=True)
class AzureOpenAIJudgeSettings:
    endpoint: str
    api_key: str
    api_version: str = "2024-12-01-preview"
    deployment: str = "gpt-5"
    max_completion_tokens: int = 5000

    def __post_init__(self) -> None:
        if not self.endpoint.strip():
            raise ValueError("endpoint must not be blank")
        if not self.api_key.strip():
            raise ValueError("api_key must not be blank")
        if not self.api_version.strip():
            raise ValueError("api_version must not be blank")
        if not self.deployment.strip():
            raise ValueError("deployment must not be blank")
        if self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be > 0")


def _is_reasoning_model(model_name: str) -> bool:
    lowered = model_name.strip().lower()
    return lowered.startswith("gpt-5") or lowered.startswith("o")


def _extract_first_choice_content(payload: Any) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else getattr(payload, "choices", None)
    if not choices:
        raise JudgeEmptyResponseError("Judge returned no choices")

    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
    if message is None:
        raise JudgeEmptyResponseError("Judge returned no message")

    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if content is None:
        raise JudgeEmptyResponseError("Judge returned empty content")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    raise JudgeMalformedJsonError("Judge content format is unsupported")


def _parse_binary_score(raw_score: Any) -> bool:
    if isinstance(raw_score, bool):
        return raw_score
    if isinstance(raw_score, int):
        if raw_score in (0, 1):
            return bool(raw_score)
        raise JudgeResponseError("score must be 0 or 1")
    if isinstance(raw_score, str):
        normalized = raw_score.strip().lower()
        if normalized in {"1", "true", "pass", "passed", "yes"}:
            return True
        if normalized in {"0", "false", "fail", "failed", "no"}:
            return False
    raise JudgeResponseError("score must be parseable as binary 0/1")


def _classify_openai_error(exc: Exception) -> Exception:
    name = type(exc).__name__.lower()
    status_code = getattr(exc, "status_code", None)

    body = getattr(exc, "body", None)
    body_dict = body if isinstance(body, dict) else None
    body_code = str(body_dict.get("code", "")).strip().lower() if body_dict is not None else ""
    inner = body_dict.get("innererror") if body_dict is not None else None
    inner_code = str(inner.get("code", "")).strip().lower() if isinstance(inner, dict) else ""

    if isinstance(status_code, int) and status_code == 400:
        if body_code == "content_filter":
            return JudgeContentFilteredError("Azure OpenAI request blocked by content policy")
        if inner_code in {"responsibleaipolicyviolation", "content_filter", "contentfilter"}:
            return JudgeContentFilteredError("Azure OpenAI request blocked by content policy")

    if "timeout" in name or "connection" in name:
        return JudgeTransientError(f"Azure OpenAI transient transport error: {type(exc).__name__}")
    if "ratelimit" in name:
        return JudgeTransientError("Azure OpenAI rate-limited request")

    if isinstance(status_code, int):
        if status_code in (408, 429) or 500 <= status_code <= 599:
            return JudgeTransientError(f"Azure OpenAI transient HTTP status {status_code}")
        if status_code in (401, 403):
            return JudgeAuthenticationError("Azure OpenAI authentication or authorization failed")
        if 400 <= status_code <= 499:
            return JudgeTerminalHttpError(status_code)

    if "authentication" in name or "permission" in name:
        return JudgeAuthenticationError("Azure OpenAI authentication or authorization failed")
    if "contentfilter" in name:
        return JudgeContentFilteredError("Azure OpenAI request blocked by content policy")

    return JudgeProviderError(f"Azure OpenAI terminal provider error: {type(exc).__name__}")


@dataclass
class AzureOpenAIJudge(AsyncJudge):
    settings: AzureOpenAIJudgeSettings
    _client: Any | None = None

    def __init__(self, settings: AzureOpenAIJudgeSettings, client: Any | None = None) -> None:
        self.settings = settings
        if client is not None:
            self._client = client
            return

        try:
            from openai import AsyncAzureOpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised when openai is absent.
            raise ImportError(
                "openai package is required for AzureOpenAIJudge unless a fake client is injected"
            ) from exc

        self._client = AsyncAzureOpenAI(
            azure_endpoint=self.settings.endpoint,
            api_key=self.settings.api_key,
            api_version=self.settings.api_version,
            max_retries=0,
        )

    @property
    def descriptor(self) -> JudgeDescriptor:
        return JudgeDescriptor(
            provider="azure-openai-foundry",
            name=self.settings.deployment,
            version=self.settings.api_version,
        )

    @property
    def prompt_schema_fingerprint(self) -> str:
        # Bump when prompt contract or JSON response schema requirements change.
        return "aoai-binary-task-completion-json-v1"

    async def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        if request.metric.kind != "binary":
            raise JudgeResponseError("AzureOpenAIJudge only supports binary metrics")

        system_prompt = (
            "You are an evaluator for end-to-end task completion. "
            "Judge only from the provided trajectory evidence. "
            "Return strict JSON only with fields: "
            "score (0 or 1), status ('completed' or 'skipped'), reason (short string)."
        )
        user_prompt = (
            "Evaluate whether the assistant completed the user's task in the full trajectory. "
            "Do not use external assumptions. "
            "If evidence is insufficient, set status to skipped.\n\n"
            f"Evidence JSON:\n{request.evidence.canonical_json}"
        )

        model_name = self.settings.deployment
        is_reasoning = _is_reasoning_model(model_name)
        token_budget = self.settings.max_completion_tokens

        def _payload_for_budget(budget: int) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }
            if is_reasoning:
                payload["max_completion_tokens"] = budget
            else:
                payload["max_tokens"] = budget
                payload["temperature"] = 0
            return payload

        payload = _payload_for_budget(token_budget)

        try:
            response = await self._client.chat.completions.create(**payload)
        except Exception as exc:
            mapped = _classify_openai_error(exc)
            raise mapped from exc

        def _finish_reason(resp: Any) -> str:
            choices = resp.get("choices") if isinstance(resp, dict) else getattr(resp, "choices", None)
            if not choices:
                return ""
            first = choices[0]
            reason = first.get("finish_reason") if isinstance(first, dict) else getattr(first, "finish_reason", None)
            return str(reason or "").strip().lower()

        finish_reason = _finish_reason(response)
        text = _extract_first_choice_content(response).strip()

        retried_length = False
        retry_budget = 12000
        if is_reasoning and finish_reason == "length" and retry_budget > token_budget:
            malformed_at_length = False
            if text:
                try:
                    json.loads(text)
                except json.JSONDecodeError:
                    malformed_at_length = True
            if not text or malformed_at_length:
                retried_length = True
                try:
                    response = await self._client.chat.completions.create(**_payload_for_budget(retry_budget))
                except Exception as exc:
                    mapped = _classify_openai_error(exc)
                    raise mapped from exc
                finish_reason = _finish_reason(response)
                text = _extract_first_choice_content(response).strip()

        if not text:
            if finish_reason == "length" or retried_length:
                raise JudgeLengthExhaustedError(
                    "Judge returned empty JSON payload after length termination; likely token exhaustion"
                )
            raise JudgeEmptyResponseError("Judge returned empty JSON payload")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            if finish_reason == "length" and is_reasoning and not retried_length and retry_budget > token_budget:
                try:
                    response = await self._client.chat.completions.create(**_payload_for_budget(retry_budget))
                except Exception as retry_exc:
                    mapped = _classify_openai_error(retry_exc)
                    raise mapped from retry_exc
                finish_reason = _finish_reason(response)
                text = _extract_first_choice_content(response).strip()
                if not text:
                    raise JudgeLengthExhaustedError(
                        "Judge returned empty JSON payload after length termination; likely token exhaustion"
                    )
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as retry_json_exc:
                    raise JudgeMalformedJsonError("Judge returned malformed JSON") from retry_json_exc
            else:
                raise JudgeMalformedJsonError("Judge returned malformed JSON") from exc

        if not isinstance(parsed, dict):
            raise JudgeMalformedJsonError("Judge JSON payload must be an object")

        status = parsed.get("status")
        if status is None or not isinstance(status, str):
            raise JudgeMalformedJsonError("Judge response must include string status")

        normalized_status = status.strip().lower()
        if normalized_status == "skipped":
            raise JudgeSkippedError("Judge returned skipped status for a required labeled experiment")
        if normalized_status != "completed":
            raise JudgeMalformedJsonError("Judge status must be 'completed' for labeled experiment")

        if parsed.get("score") is None:
            raise JudgeMalformedJsonError("Judge score must not be null")
        passed = _parse_binary_score(parsed.get("score"))

        reason = parsed.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise JudgeMalformedJsonError("Judge reason must be a string when present")

        return JudgeResponse(
            request_id=request.request_id,
            metric=request.metric,
            value=BinaryValue(passed=passed),
            reasoning=reason,
        )
