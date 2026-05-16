from fastapi.testclient import TestClient

from src.api.main import app


def test_candidates_endpoint_returns_paginated_candidates() -> None:
    client = TestClient(app)
    response = client.get("/api/candidates", params={"limit": 5, "offset": 0})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] >= len(payload["items"])
    assert len(payload["items"]) <= 5
    assert payload["items"][0]["candidate_id"]


def test_candidate_detail_endpoint_returns_card_and_optional_profile() -> None:
    client = TestClient(app)
    first_candidate = client.get("/api/candidates", params={"limit": 1}).json()["items"][0]

    response = client.get(f"/api/candidates/{first_candidate['candidate_id']}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidate"]["candidate_id"] == first_candidate["candidate_id"]
    assert "profile" in payload
