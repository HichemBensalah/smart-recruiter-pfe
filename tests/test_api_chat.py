from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_chat_endpoint_returns_copilot_response(monkeypatch) -> None:
    def fake_run_recruiter_copilot_with_memory(message: str, session_id: str | None = None) -> dict:
        return {
            "session_id": session_id or "generated-session",
            "answer": f"Reponse pour: {message}",
            "candidates": [{"candidate_id": "candidate_1", "baseline_score_v3": 0.82}],
            "decision_cards": [{"candidate_id": "candidate_1"}],
            "transferability": {"candidate_1": {"selected_source": "yaml"}},
            "sources": ["user_message", "match_candidates"],
            "warnings": [],
            "selected_candidate_id": "candidate_1",
            "matching_metadata": {
                "matching_mode": "matching_v3_job_artifact",
                "resolved_job_id": "backend_python_django_postgresql",
                "fallback_used": False,
            },
        }

    monkeypatch.setattr(
        "src.api.routes.chat.run_recruiter_copilot_with_memory",
        fake_run_recruiter_copilot_with_memory,
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"message": "Je cherche un developpeur backend Python FastAPI MongoDB", "session_id": "session-1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "session-1"
    assert payload["answer"]
    assert payload["candidates"][0]["candidate_id"] == "candidate_1"
    assert payload["decision_cards"][0]["candidate_id"] == "candidate_1"
    assert payload["transferability"]["candidate_1"]["selected_source"] == "yaml"
    assert payload["sources"] == ["user_message", "match_candidates"]
    assert payload["warnings"] == []
    assert payload["selected_candidate_id"] == "candidate_1"
    assert payload["matching_metadata"]["matching_mode"] == "matching_v3_job_artifact"


def test_chat_endpoint_requires_api_key_when_auth_is_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("SMART_RECRUITER_API_KEY", "test-secret")
    monkeypatch.setenv("API_KEY_HEADER", "X-Smart-Recruiter-Key")
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"message": "Je cherche un developpeur backend Python FastAPI MongoDB", "session_id": "session-1"},
    )

    assert response.status_code == 401
    assert response.json()["detail"]["message"] == "API key is required for this endpoint."


def test_chat_endpoint_accepts_valid_api_key_when_auth_is_enabled(monkeypatch) -> None:
    def fake_run_recruiter_copilot_with_memory(message: str, session_id: str | None = None) -> dict:
        return {
            "session_id": session_id or "generated-session",
            "answer": "ok",
            "candidates": [],
            "decision_cards": [],
            "transferability": {},
            "sources": [],
            "warnings": [],
        }

    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("SMART_RECRUITER_API_KEY", "test-secret")
    monkeypatch.setenv("API_KEY_HEADER", "X-Smart-Recruiter-Key")
    monkeypatch.setattr(
        "src.api.routes.chat.run_recruiter_copilot_with_memory",
        fake_run_recruiter_copilot_with_memory,
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"message": "Backend Python", "session_id": "session-auth"},
        headers={"X-Smart-Recruiter-Key": "test-secret"},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "session-auth"


def test_chat_endpoint_generates_session_id_when_missing(monkeypatch) -> None:
    def fake_run_recruiter_copilot_with_memory(message: str, session_id: str | None = None) -> dict:
        return {
            "session_id": session_id or "generated-session",
            "answer": "ok",
            "candidates": [],
            "decision_cards": [],
            "transferability": {},
            "sources": [],
            "warnings": [],
        }

    monkeypatch.setattr(
        "src.api.routes.chat.run_recruiter_copilot_with_memory",
        fake_run_recruiter_copilot_with_memory,
    )
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "Backend Python"})

    assert response.status_code == 200
    assert response.json()["session_id"] == "generated-session"


def test_chat_endpoint_rejects_empty_message() -> None:
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "   "})

    assert response.status_code == 422


def test_chat_endpoint_returns_degraded_response_when_copilot_fails(monkeypatch) -> None:
    def fake_run_recruiter_copilot_with_memory(message: str, session_id: str | None = None) -> dict:
        raise RuntimeError("tool failure")

    monkeypatch.setattr(
        "src.api.routes.chat.run_recruiter_copilot_with_memory",
        fake_run_recruiter_copilot_with_memory,
    )
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "Backend Python", "session_id": "session-failure"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "session-failure"
    assert payload["candidates"] == []
    assert payload["decision_cards"] == []
    assert payload["sources"] == ["api_chat_degraded"]
    assert payload["warnings"]
    assert "tool failure" in payload["warnings"][0]
