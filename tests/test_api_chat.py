from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_chat_endpoint_returns_copilot_response(monkeypatch) -> None:
    def fake_run_recruiter_copilot(message: str) -> dict:
        return {
            "answer": f"Réponse pour: {message}",
            "candidates": [{"candidate_id": "candidate_1", "baseline_score_v3": 0.82}],
            "decision_cards": [{"candidate_id": "candidate_1"}],
            "transferability": {"candidate_1": {"selected_source": "yaml"}},
            "sources": ["user_message", "match_candidates"],
            "warnings": [],
        }

    monkeypatch.setattr("src.api.routes.chat.run_recruiter_copilot", fake_run_recruiter_copilot)
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"message": "Je cherche un développeur backend Python FastAPI MongoDB"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["candidates"][0]["candidate_id"] == "candidate_1"
    assert payload["decision_cards"][0]["candidate_id"] == "candidate_1"
    assert payload["transferability"]["candidate_1"]["selected_source"] == "yaml"
    assert payload["sources"] == ["user_message", "match_candidates"]
    assert payload["warnings"] == []


def test_chat_endpoint_rejects_empty_message() -> None:
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "   "})

    assert response.status_code == 422


def test_chat_endpoint_returns_500_when_copilot_fails(monkeypatch) -> None:
    def fake_run_recruiter_copilot(message: str) -> dict:
        raise RuntimeError("tool failure")

    monkeypatch.setattr("src.api.routes.chat.run_recruiter_copilot", fake_run_recruiter_copilot)
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "Backend Python"})

    assert response.status_code == 500
    assert "tool failure" in response.json()["detail"]
