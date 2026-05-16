from fastapi.testclient import TestClient

from src.api.main import app


def test_transferability_endpoint_returns_candidate_transferability() -> None:
    client = TestClient(app)
    candidate_id = client.get("/api/candidates", params={"limit": 1}).json()["items"][0]["candidate_id"]

    response = client.get(f"/api/graph/transferability/{candidate_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidate_id"] == candidate_id
    assert isinstance(payload["transferability"]["fit_direct"], bool)
    assert 0 <= payload["transferability"]["transferability_score"] <= 1
