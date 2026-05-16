from fastapi.testclient import TestClient

from src.api.main import app


def test_demo_artifact_endpoints_return_reports() -> None:
    client = TestClient(app)

    executive = client.get("/api/demo/executive-summary")
    top10 = client.get("/api/demo/top10-summary")
    manifest = client.get("/api/demo/run-summary")

    assert executive.status_code == 200
    assert top10.status_code == 200
    assert manifest.status_code == 200
    assert manifest.json()["status"] == "success"


def test_demo_run_endpoint_regenerates_manifest() -> None:
    client = TestClient(app)

    response = client.post("/api/demo/run")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["generated_outputs"]
