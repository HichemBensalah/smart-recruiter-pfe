from fastapi.testclient import TestClient

from src.api.main import app


def test_transferability_endpoint_returns_candidate_transferability() -> None:
    client = TestClient(app)
    candidate_id = client.get("/api/candidates", params={"limit": 1}).json()["items"][0]["candidate_id"]

    response = client.get(f"/api/graph/transferability/{candidate_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidate_id"] == candidate_id
    assert payload["source"] in {"neo4j", "yaml_fallback"}
    assert isinstance(payload["fallback_used"], bool)
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["transferability"]["fit_direct"], bool)
    assert 0 <= payload["transferability"]["transferability_score"] <= 1


def test_transferability_endpoint_uses_yaml_fallback_without_neo4j(monkeypatch) -> None:
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    client = TestClient(app)
    candidate_id = client.get("/api/candidates", params={"limit": 1}).json()["items"][0]["candidate_id"]

    response = client.get(f"/api/graph/transferability/{candidate_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "yaml_fallback"
    assert payload["fallback_used"] is True
    assert payload["warnings"]


def test_transferability_endpoint_returns_404_for_unknown_candidate(monkeypatch) -> None:
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    client = TestClient(app)

    response = client.get("/api/graph/transferability/unknown_candidate")

    assert response.status_code == 404
    assert "Candidate not found" in response.json()["detail"]


def test_transferability_endpoint_preserves_stable_shape_with_neo4j(monkeypatch) -> None:
    def fake_explain_transferability(candidate_id: str, target_role: str) -> dict:
        return {
            "candidate_id": candidate_id,
            "target_role": target_role,
            "coverage_score": 0.8,
            "matched_skills": ["Python"],
            "missing_skills": [],
        }

    monkeypatch.setattr("src.api.routes.graph.explain_transferability", fake_explain_transferability)
    client = TestClient(app)
    candidate_id = client.get("/api/candidates", params={"limit": 1}).json()["items"][0]["candidate_id"]

    response = client.get(f"/api/graph/transferability/{candidate_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "neo4j"
    assert payload["fallback_used"] is False
    assert payload["warnings"] == []
    assert payload["transferability"]["fit_direct"] is True
    assert payload["transferability"]["transferability_score"] == 0.8
