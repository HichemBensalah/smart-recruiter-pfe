from fastapi.testclient import TestClient

from src.api.main import app
from src.core.storage.repositories import RepositoryUnavailableError


class _FakeCandidateRepository:
    def __init__(self, candidates: list[dict]) -> None:
        self._candidates = candidates

    def list_candidates(self, limit: int = 20, offset: int = 0) -> list[dict]:
        return self._candidates[offset : offset + limit]

    def count_candidates(self) -> int:
        return len(self._candidates)

    def get_candidate(self, candidate_id: str) -> dict | None:
        for candidate in self._candidates:
            if candidate.get("candidate_id") == candidate_id:
                return candidate
        return None


class _FakeCandidateProfileRepository:
    def __init__(self, profiles: dict[str, dict]) -> None:
        self._profiles = profiles

    def get_profile(self, profile_id: str) -> dict | None:
        return self._profiles.get(profile_id)

    def get_profile_by_candidate_id(self, candidate_id: str) -> dict | None:
        for profile in self._profiles.values():
            if profile.get("candidate_id") == candidate_id:
                return profile
        return None


class _FakeMongoRepositories:
    def __init__(self, candidates: list[dict], profiles: dict[str, dict] | None = None) -> None:
        self.candidates = _FakeCandidateRepository(candidates)
        self.candidate_profiles = _FakeCandidateProfileRepository(profiles or {})
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_candidates_endpoint_returns_paginated_candidates() -> None:
    client = TestClient(app)
    response = client.get("/api/candidates", params={"limit": 5, "offset": 0})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] >= len(payload["items"])
    assert len(payload["items"]) <= 5
    assert payload["source"].endswith(".json")
    assert payload["data_backend"] == "artifacts"
    assert payload["data_source"].endswith(".json")
    assert payload["fallback_used"] is False
    assert isinstance(payload["warnings"], list)
    assert payload["items"][0]["candidate_id"]


def test_candidates_endpoint_requires_api_key_when_auth_is_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("SMART_RECRUITER_API_KEY", "test-secret")
    monkeypatch.setenv("API_KEY_HEADER", "X-Smart-Recruiter-Key")
    client = TestClient(app)

    response = client.get("/api/candidates", params={"limit": 1})

    assert response.status_code == 401
    assert response.json()["detail"]["message"] == "API key is required for this endpoint."


def test_candidate_detail_endpoint_returns_card_and_optional_profile() -> None:
    client = TestClient(app)
    first_candidate = client.get("/api/candidates", params={"limit": 1}).json()["items"][0]

    response = client.get(f"/api/candidates/{first_candidate['candidate_id']}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidate"]["candidate_id"] == first_candidate["candidate_id"]
    assert "profile" in payload
    assert payload["source"].endswith(".json")
    assert payload["data_backend"] == "artifacts"
    assert payload["data_source"].endswith(".json")
    assert payload["fallback_used"] is False
    assert isinstance(payload["warnings"], list)


def test_candidate_detail_endpoint_returns_404_for_unknown_candidate() -> None:
    client = TestClient(app)

    response = client.get("/api/candidates/unknown_candidate")

    assert response.status_code == 404
    assert "Candidate not found" in response.json()["detail"]


def test_candidates_endpoint_rejects_invalid_pagination() -> None:
    client = TestClient(app)

    response = client.get("/api/candidates", params={"limit": 0})

    assert response.status_code == 422


def test_candidates_endpoint_returns_404_when_cards_artifact_is_absent(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DATA_BACKEND", "artifacts")
    monkeypatch.setattr("src.api.utils.DECISION_CARDS_TRANSFERABILITY", tmp_path / "missing_transferability.json")
    monkeypatch.setattr("src.api.utils.DECISION_CARDS_ML", tmp_path / "missing_ml.json")
    monkeypatch.setattr("src.api.utils.DECISION_CARDS_OFFICIAL", tmp_path / "missing_official.json")
    client = TestClient(app)

    response = client.get("/api/candidates")

    assert response.status_code == 404
    assert "No decision cards artifact found" in response.json()["detail"]


def test_candidates_endpoint_reads_mongodb_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("DATA_BACKEND", "mongodb")
    monkeypatch.setenv("ALLOW_ARTIFACT_FALLBACK", "false")
    monkeypatch.setenv("MONGODB_DATABASE", "test_db")

    repositories = _FakeMongoRepositories(
        [
            {
                "candidate_id": "candidate_mongo_1",
                "profile_id": "profile_mongo_1",
                "baseline_rank_v3": 1,
                "baseline_score_v3": 0.91,
                "recommendation_status": "shortlist",
            }
        ],
        {"profile_mongo_1": {"profile_id": "profile_mongo_1", "candidate_id": "candidate_mongo_1"}},
    )
    monkeypatch.setattr("src.api.routes.candidates.create_mongo_repositories", lambda uri, database: repositories)
    client = TestClient(app)

    response = client.get("/api/candidates", params={"limit": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["data_backend"] == "mongodb"
    assert payload["data_source"] == "mongodb:test_db.candidates"
    assert payload["fallback_used"] is False
    assert payload["items"][0]["candidate_id"] == "candidate_mongo_1"
    assert repositories.closed is True


def test_candidate_detail_endpoint_reads_mongodb_profile_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("DATA_BACKEND", "mongodb")
    monkeypatch.setenv("ALLOW_ARTIFACT_FALLBACK", "false")
    monkeypatch.setenv("MONGODB_DATABASE", "test_db")

    repositories = _FakeMongoRepositories(
        [{"candidate_id": "candidate_mongo_1", "best_profile_id": "profile_mongo_1"}],
        {"profile_mongo_1": {"profile_id": "profile_mongo_1", "candidate_id": "candidate_mongo_1", "bio": {}}},
    )
    monkeypatch.setattr("src.api.routes.candidates.create_mongo_repositories", lambda uri, database: repositories)
    client = TestClient(app)

    response = client.get("/api/candidates/candidate_mongo_1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidate"]["candidate_id"] == "candidate_mongo_1"
    assert payload["profile"]["profile_id"] == "profile_mongo_1"
    assert payload["data_backend"] == "mongodb"
    assert payload["data_source"] == "mongodb:test_db.candidates"
    assert payload["fallback_used"] is False


def test_candidates_endpoint_falls_back_to_artifacts_when_mongodb_is_unavailable(monkeypatch) -> None:
    def raise_unavailable(uri, database):
        raise RepositoryUnavailableError("test mongodb down")

    monkeypatch.setenv("DATA_BACKEND", "mongodb")
    monkeypatch.setenv("ALLOW_ARTIFACT_FALLBACK", "true")
    monkeypatch.setattr("src.api.routes.candidates.create_mongo_repositories", raise_unavailable)
    client = TestClient(app)

    response = client.get("/api/candidates", params={"limit": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["data_backend"] == "mongodb"
    assert payload["source"].endswith(".json")
    assert payload["fallback_used"] is True
    assert any("MongoDB unavailable" in warning for warning in payload["warnings"])


def test_candidates_endpoint_returns_503_when_mongodb_is_unavailable_without_fallback(monkeypatch) -> None:
    def raise_unavailable(uri, database):
        raise RepositoryUnavailableError("test mongodb down")

    monkeypatch.setenv("DATA_BACKEND", "mongodb")
    monkeypatch.setenv("ALLOW_ARTIFACT_FALLBACK", "false")
    monkeypatch.setattr("src.api.routes.candidates.create_mongo_repositories", raise_unavailable)
    client = TestClient(app)

    response = client.get("/api/candidates", params={"limit": 1})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["data_backend"] == "mongodb"
    assert detail["fallback_used"] is False
