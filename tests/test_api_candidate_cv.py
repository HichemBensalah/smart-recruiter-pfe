from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_candidate_cv_endpoint_returns_existing_cv() -> None:
    client = TestClient(app)

    response = client.get(
        "/api/candidates/candidate_f74acce78f96/cv",
        params={"job_id": "machine_learning_python_nlp"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert "Image_73.pdf" in response.headers.get("content-disposition", "")
    assert response.content.startswith(b"%PDF")


def test_candidate_cv_endpoint_returns_404_for_unknown_candidate() -> None:
    client = TestClient(app)

    response = client.get(
        "/api/candidates/candidate_unknown_for_cv/cv",
        params={"job_id": "machine_learning_python_nlp"},
    )

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["candidate_id"] == "candidate_unknown_for_cv"
    assert "CV original not found" in detail["message"]
