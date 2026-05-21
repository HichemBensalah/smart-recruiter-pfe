from fastapi.testclient import TestClient

from src.api.main import app
from src.core.storage.repositories import RepositoryUnavailableError


class _FakeDecisionCardRepository:
    def __init__(self, cards: list[dict]) -> None:
        self._cards = cards

    def list_decision_cards(self) -> list[dict]:
        return self._cards

    def get_decision_card(self, candidate_id: str) -> dict | None:
        for card in self._cards:
            if card.get("candidate_id") == candidate_id:
                return card
        return None


class _FakeMongoRepositories:
    def __init__(self, cards: list[dict]) -> None:
        self.decision_cards = _FakeDecisionCardRepository(cards)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_decision_cards_endpoint_returns_available_cards() -> None:
    client = TestClient(app)
    response = client.get("/api/decision-cards")

    assert response.status_code == 200
    payload = response.json()
    assert "candidates" in payload
    assert payload["candidates"]
    assert payload["source"].endswith(".json")
    assert payload["data_backend"] == "artifacts"
    assert payload["data_source"].endswith(".json")
    assert payload["fallback_used"] is False
    assert isinstance(payload["warnings"], list)


def test_decision_card_detail_endpoint_returns_one_card() -> None:
    client = TestClient(app)
    cards = client.get("/api/decision-cards").json()["candidates"]
    candidate_id = cards[0]["candidate_id"]

    response = client.get(f"/api/decision-cards/{candidate_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidate_id"] == candidate_id
    assert payload["source"].endswith(".json")
    assert payload["data_backend"] == "artifacts"
    assert payload["data_source"].endswith(".json")
    assert payload["fallback_used"] is False
    assert isinstance(payload["warnings"], list)


def test_decision_card_detail_endpoint_returns_404_for_unknown_candidate() -> None:
    client = TestClient(app)

    response = client.get("/api/decision-cards/unknown_candidate")

    assert response.status_code == 404
    assert "Candidate not found" in response.json()["detail"]


def test_decision_cards_endpoint_returns_404_when_artifact_is_absent(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("src.api.utils.DECISION_CARDS_TRANSFERABILITY", tmp_path / "missing_transferability.json")
    monkeypatch.setattr("src.api.utils.DECISION_CARDS_ML", tmp_path / "missing_ml.json")
    monkeypatch.setattr("src.api.utils.DECISION_CARDS_OFFICIAL", tmp_path / "missing_official.json")
    client = TestClient(app)

    response = client.get("/api/decision-cards")

    assert response.status_code == 404
    assert "No decision cards artifact found" in response.json()["detail"]


def test_decision_cards_endpoint_reads_mongodb_when_configured(monkeypatch) -> None:
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
            }
        ]
    )
    monkeypatch.setattr("src.api.routes.decision_cards.create_mongo_repositories", lambda uri, database: repositories)
    client = TestClient(app)

    response = client.get("/api/decision-cards")

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidate_count"] == 1
    assert payload["data_backend"] == "mongodb"
    assert payload["data_source"] == "mongodb:test_db.decision_cards"
    assert payload["fallback_used"] is False
    assert payload["candidates"][0]["candidate_id"] == "candidate_mongo_1"
    assert repositories.closed is True


def test_decision_card_detail_endpoint_reads_mongodb_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("DATA_BACKEND", "mongodb")
    monkeypatch.setenv("ALLOW_ARTIFACT_FALLBACK", "false")
    monkeypatch.setenv("MONGODB_DATABASE", "test_db")
    repositories = _FakeMongoRepositories(
        [{"candidate_id": "candidate_mongo_1", "profile_id": "profile_mongo_1", "baseline_rank_v3": 1}]
    )
    monkeypatch.setattr("src.api.routes.decision_cards.create_mongo_repositories", lambda uri, database: repositories)
    client = TestClient(app)

    response = client.get("/api/decision-cards/candidate_mongo_1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidate_id"] == "candidate_mongo_1"
    assert payload["data_backend"] == "mongodb"
    assert payload["data_source"] == "mongodb:test_db.decision_cards"
    assert payload["fallback_used"] is False


def test_decision_cards_endpoint_falls_back_to_artifacts_when_mongodb_is_unavailable(monkeypatch) -> None:
    def raise_unavailable(uri, database):
        raise RepositoryUnavailableError("test mongodb down")

    monkeypatch.setenv("DATA_BACKEND", "mongodb")
    monkeypatch.setenv("ALLOW_ARTIFACT_FALLBACK", "true")
    monkeypatch.setattr("src.api.routes.decision_cards.create_mongo_repositories", raise_unavailable)
    client = TestClient(app)

    response = client.get("/api/decision-cards")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data_backend"] == "mongodb"
    assert payload["source"].endswith(".json")
    assert payload["fallback_used"] is True
    assert any("MongoDB unavailable" in warning for warning in payload["warnings"])
