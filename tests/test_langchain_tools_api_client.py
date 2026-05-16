from __future__ import annotations

import httpx
import pytest

from src.core.chatbot.tools.api_client import SmartRecruiterApiClient, SmartRecruiterApiError
from src.core.chatbot.tools.graph_tools import get_neo4j_gaps, get_neo4j_transferability
from src.core.chatbot.tools.schemas import (
    CandidateProfileInput,
    DecisionCardInput,
    MatchCandidatesInput,
    Neo4jTransferabilityInput,
    TransferabilityInput,
)


def test_api_client_uses_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMART_RECRUITER_API_BASE_URL", raising=False)

    client = SmartRecruiterApiClient()

    assert client.base_url == "http://localhost:8000"
    assert client.build_url("/api/match") == "http://localhost:8000/api/match"


def test_api_client_reads_base_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMART_RECRUITER_API_BASE_URL", "http://testserver/")

    client = SmartRecruiterApiClient()

    assert client.base_url == "http://testserver"
    assert client.build_url("health") == "http://testserver/health"


def test_api_client_raises_clean_error_on_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(self, method, url, params=None, json=None):
        return httpx.Response(503, json={"detail": "service unavailable"})

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    client = SmartRecruiterApiClient(base_url="http://testserver")

    with pytest.raises(SmartRecruiterApiError) as exc:
        client.get("/api/test")

    assert "HTTP 503" in str(exc.value)
    assert "service unavailable" in str(exc.value)


def test_pydantic_tool_schemas_validate_inputs() -> None:
    assert MatchCandidatesInput(job_description="Backend Python", top_k=5).top_k == 5
    assert CandidateProfileInput(candidate_id="candidate_1").candidate_id == "candidate_1"
    assert DecisionCardInput(candidate_id="candidate_1").candidate_id == "candidate_1"
    assert TransferabilityInput(candidate_id="candidate_1").candidate_id == "candidate_1"
    assert Neo4jTransferabilityInput(candidate_id="candidate_1").target_role == "Backend Developer"


def test_neo4j_tools_return_fallback_response_when_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(self, path, params=None):
        raise SmartRecruiterApiError("Smart Recruiter API returned HTTP 503: Neo4j is not configured")

    monkeypatch.setattr(SmartRecruiterApiClient, "get", fake_get)

    transferability = get_neo4j_transferability("candidate_1")
    gaps = get_neo4j_gaps("candidate_1")

    assert transferability["available"] is False
    assert transferability["fallback_recommended"] is True
    assert gaps["available"] is False
    assert gaps["fallback_recommended"] is True
