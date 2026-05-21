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
    assert payload["source"].endswith("scripts/run_demo_end_to_end.py")
    assert payload["fallback_used"] is False
    assert isinstance(payload["warnings"], list)
    assert payload["generated_outputs"]


def test_demo_run_endpoint_returns_404_when_script_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.api.utils.DEMO_SCRIPT", tmp_path / "missing_demo.py")
    client = TestClient(app)

    response = client.post("/api/demo/run")

    assert response.status_code == 404
    assert "Demo script not found" in response.json()["detail"]


def test_demo_run_endpoint_returns_script_failure_details(monkeypatch, tmp_path) -> None:
    script = tmp_path / "failing_demo.py"
    script.write_text("import sys\nsys.stderr.write('boom')\nsys.exit(7)\n", encoding="utf-8")
    monkeypatch.setattr("src.api.utils.DEMO_SCRIPT", script)
    client = TestClient(app)

    response = client.post("/api/demo/run")

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["message"] == "Demo end-to-end script failed"
    assert detail["returncode"] == 7
    assert "boom" in detail["stderr"]
