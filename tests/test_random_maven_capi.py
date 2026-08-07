from __future__ import annotations

import io
import json
import urllib.error

from random_sampling.maven_capi import MavenCapiSettings, probe_maven_availability


class _FakeResponse:
    def __init__(self, status: int, payload: dict[str, object]) -> None:
        self._status = status
        self._body = json.dumps(payload).encode("utf-8")

    def getcode(self) -> int:
        return self._status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_probe_success_uses_exact_expected_url_and_marks_model_available(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["auth"] = request.headers.get("Authorization")
        return _FakeResponse(
            200,
            {
                "configured": True,
                "provider": "azure-openai",
                "model": "gpt-5",
                "reachable": True,
                "authorized": True,
                "latencyMs": 12,
                "servedModel": "gpt-5",
                "detail": "ok",
            },
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    settings = MavenCapiSettings(
        base_url="http://host/maven/tenants/{tenant}/eval-harness",
        tenant_id="t-123",
        bearer_token="secret-token",
        model="gpt-5",
    )
    result = probe_maven_availability(settings)

    assert captured["url"] == "http://host/maven/tenants/t-123/eval-harness/models/check?model=gpt-5"
    assert captured["auth"] == "Bearer secret-token"
    assert result.status_code == 200
    assert result.endpoint_reachable is True
    assert result.model_available is True
    assert result.ok is True


def test_probe_unreachable_returns_distinct_endpoint_status(monkeypatch):
    def _fake_urlopen(request, timeout=0):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    settings = MavenCapiSettings(
        base_url="http://host/maven/tenants/{tenant}/eval-harness",
        tenant_id="t-123",
    )
    result = probe_maven_availability(settings)

    assert result.status_code is None
    assert result.endpoint_reachable is False
    assert result.model_available is False
    assert result.ok is False


def test_probe_401_keeps_endpoint_reachable_but_not_available(monkeypatch):
    def _fake_urlopen(request, timeout=0):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "unauthorized",
            hdrs=None,
            fp=io.BytesIO(b"{}"),
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    settings = MavenCapiSettings(
        base_url="http://host/maven/tenants/{tenant}/eval-harness",
        tenant_id="t-123",
    )
    result = probe_maven_availability(settings)

    assert result.status_code == 401
    assert result.endpoint_reachable is True
    assert result.model_available is False
    assert result.ok is False
