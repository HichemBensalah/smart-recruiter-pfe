from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_neo4j_status_endpoint_returns_200_without_configuration(monkeypatch) -> None:
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    client = TestClient(app)

    response = client.get("/api/graph/neo4j/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["neo4j_available"] is False
    assert payload["message"]


def test_neo4j_roles_endpoint_returns_503_without_configuration(monkeypatch) -> None:
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    client = TestClient(app)

    response = client.get("/api/graph/neo4j/roles")

    assert response.status_code == 503
    assert "Neo4j is not configured" in response.json()["detail"]


def test_neo4j_candidate_skills_endpoint_returns_503_without_configuration(monkeypatch) -> None:
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    client = TestClient(app)

    response = client.get("/api/graph/neo4j/candidate/candidate_1/skills")

    assert response.status_code == 503


def test_neo4j_transferability_endpoint_returns_503_without_configuration(monkeypatch) -> None:
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    client = TestClient(app)

    response = client.get("/api/graph/neo4j/transferability/candidate_1", params={"target_role": "Backend Developer"})

    assert response.status_code == 503


def test_neo4j_gaps_endpoint_returns_503_without_configuration(monkeypatch) -> None:
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    client = TestClient(app)

    response = client.get("/api/graph/neo4j/gaps/candidate_1", params={"target_role": "Backend Developer"})

    assert response.status_code == 503
