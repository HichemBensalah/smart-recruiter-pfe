from fastapi.testclient import TestClient

from src.api.main import app


def test_decision_cards_endpoint_returns_available_cards() -> None:
    client = TestClient(app)
    response = client.get("/api/decision-cards")

    assert response.status_code == 200
    payload = response.json()
    assert "candidates" in payload
    assert payload["candidates"]


def test_decision_card_detail_endpoint_returns_one_card() -> None:
    client = TestClient(app)
    cards = client.get("/api/decision-cards").json()["candidates"]
    candidate_id = cards[0]["candidate_id"]

    response = client.get(f"/api/decision-cards/{candidate_id}")

    assert response.status_code == 200
    assert response.json()["candidate_id"] == candidate_id
