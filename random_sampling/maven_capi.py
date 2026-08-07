"""Optional Maven CAPI availability probe helpers.

This module does not implement the AsyncJudge contract; it only validates that a
hosted Maven endpoint is reachable and exposes expected capabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import json


@dataclass(frozen=True)
class MavenCapiSettings:
    base_url: str
    tenant_id: str
    bearer_token: str | None = None
    model: str = "gpt-5"

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must not be blank")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must not be blank")
        if not self.model.strip():
            raise ValueError("model must not be blank")


@dataclass(frozen=True)
class MavenAvailability:
    ok: bool
    status_code: int | None
    endpoint_reachable: bool
    model_available: bool
    message: str


def probe_maven_availability(settings: MavenCapiSettings, timeout_seconds: float = 5.0) -> MavenAvailability:
    """Probe hosted Maven service availability via GET /models/check.

    Returns a structured status object and never includes secrets in messages.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    base = settings.base_url.rstrip("/")
    if "{tenant}" in base:
        base = base.replace("{tenant}", urllib.parse.quote(settings.tenant_id, safe=""))
    elif not base.endswith("/eval-harness"):
        base = f"{base}/maven/tenants/{urllib.parse.quote(settings.tenant_id, safe='')}/eval-harness"

    query = urllib.parse.urlencode({"model": settings.model})
    url = base + "/models/check" + "?" + query
    headers = {"Accept": "application/json"}
    if settings.bearer_token:
        headers["Authorization"] = f"Bearer {settings.bearer_token}"

    request = urllib.request.Request(url=url, method="GET", headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = response.getcode()
            body = response.read().decode("utf-8", errors="replace")
            parsed: Any
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {}

            configured = bool(parsed.get("configured"))
            reachable = bool(parsed.get("reachable"))
            authorized = bool(parsed.get("authorized"))
            requested_model = settings.model.strip()
            served_model = str(parsed.get("servedModel") or "").strip()
            returned_model = str(parsed.get("model") or "").strip()
            model_match = (requested_model == returned_model) or bool(served_model)
            model_available = configured and reachable and authorized and model_match
            endpoint_reachable = 200 <= status < 300

            return MavenAvailability(
                ok=endpoint_reachable and model_available,
                status_code=status,
                endpoint_reachable=endpoint_reachable,
                model_available=model_available,
                message="Maven endpoint responded",
            )
    except urllib.error.HTTPError as exc:
        status = exc.code
        return MavenAvailability(
            ok=False,
            status_code=status,
            endpoint_reachable=(status != 404),
            model_available=False,
            message=f"Maven HTTP error {status}",
        )
    except urllib.error.URLError:
        return MavenAvailability(
            ok=False,
            status_code=None,
            endpoint_reachable=False,
            model_available=False,
            message="Maven endpoint unreachable",
        )
