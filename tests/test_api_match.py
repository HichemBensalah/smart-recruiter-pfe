from fastapi.testclient import TestClient

from src.api.main import app


def test_match_endpoint_returns_matching_v3_artifact_results() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/match",
        json={"job_description": "Developpeur backend Python FastAPI MongoDB", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matching_mode"] == "demo_artifact_matching_v3_baseline"
    assert len(payload["items"]) == 3
    assert payload["items"][0]["baseline_score_v3"] is not None
    assert "Matching V3" in payload["methodological_note"]
