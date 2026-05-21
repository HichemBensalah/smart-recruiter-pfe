from __future__ import annotations

from typing import Any

import pytest
from pymongo.errors import OperationFailure

from src.core.storage.repositories import (
    CandidateProfileRepository,
    CandidateRepository,
    DecisionCardRepository,
    JobProfileRepository,
    MatchingRunRepository,
    MongoRepositories,
    RepositoryUnavailableError,
)


class _FakeUpdateResult:
    def __init__(self, upserted_id: str | None = None, matched_count: int = 0) -> None:
        self.upserted_id = upserted_id
        self.matched_count = matched_count


class _FakeCursor(list):
    def sort(self, sort_spec):
        specs = sort_spec if isinstance(sort_spec, list) else [sort_spec]
        for key, direction in reversed(specs):
            reverse = direction == -1
            super().sort(key=lambda item: (item.get(key) is None, item.get(key)), reverse=reverse)
        return self

    def skip(self, offset: int):
        return _FakeCursor(self[offset:])

    def limit(self, limit: int):
        return _FakeCursor(self[:limit])


class _FakeCollection:
    def __init__(
        self,
        documents: list[dict[str, Any]] | None = None,
        indexes: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.documents = [dict(document) for document in documents or []]
        self.indexes: dict[str, dict[str, Any]] = {
            "_id_": {"key": [("_id", 1)]},
            **{name: dict(index) for name, index in (indexes or {}).items()},
        }
        self.create_index_calls: list[tuple[Any, dict[str, Any]]] = []

    def find(self, query=None, projection=None):
        query = query or {}
        return _FakeCursor([self._project(document, projection) for document in self.documents if self._matches(document, query)])

    def find_one(self, query, projection=None):
        for document in self.documents:
            if self._matches(document, query):
                return self._project(document, projection)
        return None

    def count_documents(self, query):
        return len([document for document in self.documents if self._matches(document, query or {})])

    def update_one(self, query, update, upsert=False):
        for document in self.documents:
            if self._matches(document, query):
                document.update(update.get("$set", {}))
                return _FakeUpdateResult(matched_count=1)
        if upsert:
            document = dict(query)
            document.update(update.get("$set", {}))
            self.documents.append(document)
            return _FakeUpdateResult(upserted_id="fake_id")
        return _FakeUpdateResult()

    def create_index(self, keys, **kwargs):
        name = kwargs["name"]
        options = {key: value for key, value in kwargs.items() if key != "name"}
        normalized_keys = list(keys)
        for existing_name, index in self.indexes.items():
            if list(index.get("key", [])) != normalized_keys:
                continue
            if self._index_options_match(index, options):
                if existing_name == name:
                    return name
                raise OperationFailure(
                    f"Index already exists with a different name: {existing_name}",
                    code=85,
                    details={"codeName": "IndexOptionsConflict"},
                )
            raise OperationFailure(
                f"Index already exists with different options: {existing_name}",
                code=85,
                details={"codeName": "IndexOptionsConflict"},
            )
        if name in self.indexes:
            raise OperationFailure(
                f"Index name already exists with different keys: {name}",
                code=85,
                details={"codeName": "IndexOptionsConflict"},
            )
        self.indexes[name] = {"key": normalized_keys, **options}
        self.create_index_calls.append((normalized_keys, dict(kwargs)))
        return name

    def index_information(self):
        return {name: dict(index) for name, index in self.indexes.items()}

    def _project(self, document, projection):
        projected = dict(document)
        if projection and projection.get("_id") == 0:
            projected.pop("_id", None)
        return projected

    def _matches(self, document, query):
        for key, expected in query.items():
            if key == "$or":
                return any(self._matches(document, item) for item in expected)
            if isinstance(expected, dict) and "$in" in expected:
                if document.get(key) not in expected["$in"]:
                    return False
                continue
            if document.get(key) != expected:
                return False
        return True

    def _index_options_match(self, index, options):
        for option_name in ("unique", "sparse", "hidden"):
            if bool(index.get(option_name, False)) != bool(options.get(option_name, False)):
                return False
        for option_name in ("expireAfterSeconds", "partialFilterExpression", "collation", "wildcardProjection"):
            if index.get(option_name) != options.get(option_name):
                return False
        return True


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_candidate_repository_reads_from_collection_without_object_ids() -> None:
    collection = _FakeCollection(
        [
            {"_id": "hidden", "candidate_id": "candidate_2", "baseline_rank_v3": 2},
            {"_id": "hidden", "candidate_id": "candidate_1", "baseline_rank_v3": 1},
        ]
    )
    repository = CandidateRepository(collection, "candidates")

    assert repository.count_candidates() == 2
    assert repository.list_candidates(limit=1, offset=0)[0] == {"candidate_id": "candidate_1", "baseline_rank_v3": 1}
    assert repository.get_candidate("candidate_2") == {"candidate_id": "candidate_2", "baseline_rank_v3": 2}


def test_repository_upsert_document_reports_insert_update_and_skip() -> None:
    collection = _FakeCollection()
    repository = DecisionCardRepository(collection, "decision_cards")

    inserted = repository.upsert_document({"candidate_id": "candidate_1"}, {"candidate_id": "candidate_1", "score": 1})
    skipped = repository.upsert_document({"candidate_id": "candidate_1"}, {"candidate_id": "candidate_1", "score": 1})
    updated = repository.upsert_document({"candidate_id": "candidate_1"}, {"candidate_id": "candidate_1", "score": 2})

    assert inserted == "inserted"
    assert skipped == "skipped"
    assert updated == "updated"
    assert collection.find_one({"candidate_id": "candidate_1"})["score"] == 2


def test_candidate_profile_ensure_indexes_is_idempotent() -> None:
    collection = _FakeCollection()
    repository = CandidateProfileRepository(collection, "candidate_profiles")

    repository.ensure_indexes()
    first_call_count = len(collection.create_index_calls)
    repository.ensure_indexes()

    candidate_id_indexes = [
        name
        for name, index in collection.index_information().items()
        if index.get("key") == [("candidate_id", 1)]
    ]
    assert candidate_id_indexes == ["idx_profile_candidate_id"]
    assert len(collection.create_index_calls) == first_call_count


def test_candidate_profile_ensure_indexes_accepts_legacy_candidate_id_index_name() -> None:
    collection = _FakeCollection(indexes={"idx_candidate_id": {"key": [("candidate_id", 1)]}})
    repository = CandidateProfileRepository(collection, "candidate_profiles")

    repository.ensure_indexes()

    candidate_id_indexes = [
        name
        for name, index in collection.index_information().items()
        if index.get("key") == [("candidate_id", 1)]
    ]
    created_index_names = [kwargs["name"] for _, kwargs in collection.create_index_calls]
    assert candidate_id_indexes == ["idx_candidate_id"]
    assert "idx_profile_candidate_id" not in created_index_names


def test_candidate_profile_ensure_indexes_raises_on_incompatible_candidate_id_index() -> None:
    collection = _FakeCollection(indexes={"idx_candidate_id": {"key": [("candidate_id", 1)], "unique": True}})
    repository = CandidateProfileRepository(collection, "candidate_profiles")

    with pytest.raises(RepositoryUnavailableError) as exc_info:
        repository.ensure_indexes()

    assert isinstance(exc_info.value.__cause__, OperationFailure)


def test_all_repository_ensure_indexes_are_idempotent() -> None:
    candidates = _FakeCollection()
    candidate_profiles = _FakeCollection()
    job_profiles = _FakeCollection()
    matching_runs = _FakeCollection()
    decision_cards = _FakeCollection()
    repositories = MongoRepositories(
        client=_FakeClient(),
        database_name="test_db",
        candidates=CandidateRepository(candidates, "candidates"),
        candidate_profiles=CandidateProfileRepository(candidate_profiles, "candidate_profiles"),
        job_profiles=JobProfileRepository(job_profiles, "job_profiles"),
        matching_runs=MatchingRunRepository(matching_runs, "matching_runs"),
        decision_cards=DecisionCardRepository(decision_cards, "decision_cards"),
    )

    repositories.ensure_indexes()
    first_call_counts = {
        "candidates": len(candidates.create_index_calls),
        "candidate_profiles": len(candidate_profiles.create_index_calls),
        "job_profiles": len(job_profiles.create_index_calls),
        "matching_runs": len(matching_runs.create_index_calls),
        "decision_cards": len(decision_cards.create_index_calls),
    }
    repositories.ensure_indexes()

    assert {
        "candidates": len(candidates.create_index_calls),
        "candidate_profiles": len(candidate_profiles.create_index_calls),
        "job_profiles": len(job_profiles.create_index_calls),
        "matching_runs": len(matching_runs.create_index_calls),
        "decision_cards": len(decision_cards.create_index_calls),
    } == first_call_counts
