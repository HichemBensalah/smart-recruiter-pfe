from __future__ import annotations

import os
from typing import Any
from urllib.parse import urljoin

import httpx

from src.api.config import load_api_settings


DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 10.0


class SmartRecruiterApiError(RuntimeError):
    """Raised when the Smart Recruiter FastAPI facade cannot return a clean response."""


class SmartRecruiterApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.base_url = (base_url or os.getenv("SMART_RECRUITER_API_BASE_URL") or DEFAULT_API_BASE_URL).rstrip("/")
        self.timeout = timeout

    def build_url(self, path: str) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        return urljoin(f"{self.base_url}/", normalized_path.lstrip("/"))

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("GET", path, params=params)

    def post(self, path: str, json_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", path, json_payload=json_payload)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self.build_url(path)
        headers = self._auth_headers()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, url, params=params, json=json_payload, headers=headers)
        except httpx.RequestError as exc:
            raise SmartRecruiterApiError(f"Smart Recruiter API unavailable: {exc}") from exc

        if response.status_code >= 400:
            detail = _extract_error_detail(response)
            raise SmartRecruiterApiError(f"Smart Recruiter API returned HTTP {response.status_code}: {detail}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise SmartRecruiterApiError("Smart Recruiter API returned non-JSON response") from exc
        if not isinstance(payload, dict):
            raise SmartRecruiterApiError("Smart Recruiter API response must be a JSON object")
        return payload

    def _auth_headers(self) -> dict[str, str]:
        settings = load_api_settings()
        if settings.auth_enabled and settings.api_key:
            return {settings.api_key_header: settings.api_key}
        return {}


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload)
    return str(payload)
