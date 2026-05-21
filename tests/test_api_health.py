from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint_returns_service_status() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "smart-recruiter"
    assert payload["version"] == "demo"
    assert "dependencies" in payload
    assert payload["dependencies"]["matching_artifacts"]["available"] is True
    assert payload["dependencies"]["matching_artifacts"]["count"] >= 1
    assert "job_ids" in payload["dependencies"]["matching_artifacts"]
    assert "neo4j" in payload["dependencies"]
    assert "mongodb_configured" in payload["dependencies"]
    assert "warnings" in payload


def test_health_endpoint_remains_public_when_auth_is_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("SMART_RECRUITER_API_KEY", "test-secret")
    monkeypatch.setenv("API_KEY_HEADER", "X-Smart-Recruiter-Key")
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
