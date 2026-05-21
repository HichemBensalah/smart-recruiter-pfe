from fastapi.testclient import TestClient

from src.api.main import app
from src.core.matching.live_matcher import LiveMatchResult, LiveMatchingUnavailable
from src.core.storage.repositories import RepositoryUnavailableError


def test_match_endpoint_returns_matching_v3_artifact_results() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/match",
        json={"job_description": "Developpeur backend Python FastAPI MongoDB", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resolved_job_id"] == "backend_python_django_postgresql"
    assert payload["artifact_source"] == "data/ranking/features/backend_python_django_postgresql.jsonl"
    assert payload["retrieval_source"] == "data/ranking/features/backend_python_django_postgresql.jsonl"
    assert payload["scoring_source"] == "matching_v3_artifact_features"
    assert payload["fallback_used"] is False
    assert payload["warnings"] == []
    assert payload["matching_mode"] == "matching_v3_job_artifact"
    assert len(payload["items"]) == 3
    assert payload["items"][0]["baseline_score_v3"] is not None
    assert "Matching V3" in payload["methodological_note"]


def test_match_endpoint_stays_available_when_auth_is_disabled(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("SMART_RECRUITER_API_KEY", "test-secret")
    client = TestClient(app)

    response = client.post(
        "/api/match",
        json={"job_description": "Developpeur backend Python FastAPI MongoDB", "top_k": 1},
    )

    assert response.status_code == 200
    assert response.json()["items"]


def test_match_endpoint_requires_api_key_when_auth_is_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("SMART_RECRUITER_API_KEY", "test-secret")
    monkeypatch.setenv("API_KEY_HEADER", "X-Smart-Recruiter-Key")
    client = TestClient(app)

    response = client.post(
        "/api/match",
        json={"job_description": "Developpeur backend Python FastAPI MongoDB", "top_k": 1},
    )

    assert response.status_code == 401
    detail = response.json()["detail"]
    assert detail["message"] == "API key is required for this endpoint."
    assert detail["header"] == "X-Smart-Recruiter-Key"


def test_match_endpoint_rejects_invalid_api_key_when_auth_is_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("SMART_RECRUITER_API_KEY", "test-secret")
    monkeypatch.setenv("API_KEY_HEADER", "X-Smart-Recruiter-Key")
    client = TestClient(app)

    response = client.post(
        "/api/match",
        json={"job_description": "Developpeur backend Python FastAPI MongoDB", "top_k": 1},
        headers={"X-Smart-Recruiter-Key": "wrong-secret"},
    )

    assert response.status_code == 401
    assert response.json()["detail"]["message"] == "API key is invalid."


def test_match_endpoint_accepts_valid_api_key_when_auth_is_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("SMART_RECRUITER_API_KEY", "test-secret")
    monkeypatch.setenv("API_KEY_HEADER", "X-Smart-Recruiter-Key")
    client = TestClient(app)

    response = client.post(
        "/api/match",
        json={"job_description": "Developpeur backend Python FastAPI MongoDB", "top_k": 1},
        headers={"X-Smart-Recruiter-Key": "test-secret"},
    )

    assert response.status_code == 200
    assert response.json()["items"]


def test_match_endpoint_uses_known_job_id_artifact() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/match",
        json={
            "job_description": "Developpeur backend Python FastAPI MongoDB",
            "job_id": "backend_python_fastapi_mongodb_aligned",
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "backend_python_fastapi_mongodb_aligned"
    assert payload["resolved_job_id"] == "backend_python_fastapi_mongodb_aligned"
    assert payload["artifact_source"] == "data/ranking/features/backend_python_fastapi_mongodb_aligned.jsonl"
    assert payload["fallback_used"] is False
    assert payload["warnings"] == []
    assert payload["items"][0]["candidate_id"] == "candidate_1487f3187f7b"
    assert payload["items"][0]["baseline_rank_v3"] == 1


def test_match_endpoint_falls_back_for_unknown_job_id() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/match",
        json={
            "job_description": "Role inconnu",
            "job_id": "unknown_job_id",
            "top_k": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "unknown_job_id"
    assert payload["resolved_job_id"] == "backend_python_django_postgresql"
    assert payload["artifact_source"] == "data/ranking/features/backend_python_django_postgresql.jsonl"
    assert payload["fallback_used"] is True
    assert payload["warnings"]
    assert "unknown_job_id" in payload["warnings"][0]
    assert payload["matching_mode"] == "matching_v3_job_artifact_with_fallback"
    assert payload["items"][0]["candidate_id"] == "candidate_b6f7add66ffc"


def test_match_endpoint_rejects_invalid_top_k() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/match",
        json={"job_description": "Backend Python", "top_k": 0},
    )

    assert response.status_code == 422


def test_match_endpoint_rejects_blank_job_description() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/match",
        json={"job_description": "   ", "top_k": 1},
    )

    assert response.status_code == 422


def test_match_endpoint_uses_decision_cards_when_artifact_directory_is_empty(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.api.utils.MATCHING_FEATURES_DIR", tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/match",
        json={"job_description": "Backend Python", "top_k": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["resolved_job_id"] == "backend_python_django_postgresql"
    assert payload["fallback_used"] is True
    assert payload["matching_mode"] == "decision_cards_fallback_after_missing_matching_artifact"
    assert payload["warnings"]
    assert len(payload["items"]) == 2


def test_match_endpoint_live_mode_calls_live_matcher(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRepositories:
        def close(self) -> None:
            captured["closed"] = True

    class FakeLiveMatcher:
        def __init__(self, repositories, settings):
            captured["repositories"] = repositories
            captured["settings"] = settings

        def match(self, *, job_description, job_id=None, top_k=None):
            captured["job_description"] = job_description
            captured["job_id"] = job_id
            captured["top_k"] = top_k
            return LiveMatchResult(
                job_id=job_id,
                resolved_job_id=job_id or "live_job",
                top_k=top_k or 1,
                matching_run_id="run_live_1",
                data_source="mongodb:test_db.candidate_profiles",
                retrieval_source="faiss:data/indexes/faiss/cv_index.faiss",
                items=[
                    {
                        "candidate_id": "candidate_live_1",
                        "profile_id": "profile_live_1",
                        "baseline_rank_v3": 1,
                        "baseline_score_v3": 0.88,
                        "faiss_rank": 1,
                        "faiss_score": 0.77,
                        "recommendation_status": "strong_match",
                        "matched_skills": ["Python"],
                        "missing_required_skills": [],
                        "explanation": "Live Matching V3 fake explanation.",
                    }
                ],
            )

    monkeypatch.setenv("MATCHING_MODE", "live")
    monkeypatch.setenv("MONGODB_DATABASE", "test_db")
    monkeypatch.setattr("src.api.routes.match.create_mongo_repositories", lambda uri, database: FakeRepositories())
    monkeypatch.setattr("src.api.routes.match.LiveMatcher", FakeLiveMatcher)
    client = TestClient(app)

    response = client.post(
        "/api/match",
        json={"job_description": "Backend Python", "job_id": "job_live", "top_k": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matching_mode"] == "live_mongodb_faiss_matching_v3"
    assert payload["data_backend"] == "mongodb"
    assert payload["data_source"] == "mongodb:test_db.candidate_profiles"
    assert payload["retrieval_source"] == "faiss:data/indexes/faiss/cv_index.faiss"
    assert payload["scoring_source"] == "matching_v3.score_candidate"
    assert payload["matching_run_id"] == "run_live_1"
    assert payload["fallback_used"] is False
    assert payload["items"][0]["candidate_id"] == "candidate_live_1"
    assert payload["items"][0]["faiss_rank"] == 1
    assert captured["job_id"] == "job_live"
    assert captured["closed"] is True


def test_match_endpoint_live_mode_returns_503_when_mongodb_is_unavailable(monkeypatch) -> None:
    def raise_unavailable(uri, database):
        raise RepositoryUnavailableError("test mongodb down")

    monkeypatch.setenv("MATCHING_MODE", "live")
    monkeypatch.setenv("ALLOW_ARTIFACT_FALLBACK", "true")
    monkeypatch.setattr("src.api.routes.match.create_mongo_repositories", raise_unavailable)
    client = TestClient(app)

    response = client.post(
        "/api/match",
        json={"job_description": "Backend Python", "top_k": 1},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["matching_mode"] == "live"
    assert detail["fallback_used"] is False
    assert "test mongodb down" in detail["warnings"][0]


def test_match_endpoint_hybrid_falls_back_to_artifact_when_live_fails(monkeypatch) -> None:
    def raise_live(request, *, data_settings, matching_settings):
        raise LiveMatchingUnavailable("fake FAISS unavailable")

    monkeypatch.setenv("MATCHING_MODE", "hybrid")
    monkeypatch.setenv("ALLOW_ARTIFACT_FALLBACK", "true")
    monkeypatch.setattr("src.api.routes.match._match_live", raise_live)
    client = TestClient(app)

    response = client.post(
        "/api/match",
        json={"job_description": "Backend Python", "top_k": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matching_mode"] == "hybrid_live_then_artifact"
    assert payload["fallback_used"] is True
    assert payload["items"]
    assert any("fake FAISS unavailable" in warning for warning in payload["warnings"])


def test_match_endpoint_saves_artifact_trace_when_mongodb_backend_is_configured(monkeypatch) -> None:
    saved: dict[str, object] = {}

    class FakeMatchingRuns:
        def save_matching_run(self, document):
            saved["document"] = document
            return "run_artifact_1"

    class FakeRepositories:
        matching_runs = FakeMatchingRuns()

        def close(self) -> None:
            saved["closed"] = True

    monkeypatch.setenv("MATCHING_MODE", "artifact")
    monkeypatch.setenv("DATA_BACKEND", "mongodb")
    monkeypatch.setenv("MONGODB_DATABASE", "test_db")
    monkeypatch.setattr("src.api.routes.match.create_mongo_repositories", lambda uri, database: FakeRepositories())
    client = TestClient(app)

    response = client.post(
        "/api/match",
        json={"job_description": "Backend Python", "top_k": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matching_run_id"] == "run_artifact_1"
    assert saved["document"]["matching_mode"] == "matching_v3_job_artifact"
    assert saved["document"]["candidate_ids"]
    assert saved["closed"] is True
