from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


DEFAULT_MONGODB_URI = "mongodb://localhost:27017"
DEFAULT_MONGODB_DATABASE = "talent_intelligence"
DEFAULT_CANDIDATES_COLLECTION = "candidates"
DEFAULT_CANDIDATE_PROFILES_COLLECTION = "candidate_profiles"
DEFAULT_JOB_PROFILES_COLLECTION = "job_profiles"
DEFAULT_MATCHING_RUNS_COLLECTION = "matching_runs"
DEFAULT_DECISION_CARDS_COLLECTION = "decision_cards"

FUTURE_COLLECTIONS = (
    "conversation_sessions",
    "audit_logs",
    "faiss_index_metadata",
)


class RepositoryUnavailableError(RuntimeError):
    """Raised when MongoDB cannot be reached or queried cleanly."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mask_mongodb_uri(uri: str) -> str:
    return re.sub(r"(mongodb(?:\+srv)?://)([^:@/]+):([^@/]+)@", r"\1***:***@", uri)


def format_mongodb_source(database_name: str, collection_name: str) -> str:
    return f"mongodb:{database_name}.{collection_name}"


def stable_id(prefix: str, parts: Iterable[Any]) -> str:
    raw = "|".join(str(part) for part in parts if part not in (None, ""))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if key != "_id"}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if value.__class__.__name__ == "ObjectId":
        return str(value)
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, default=str)


_INDEX_OPTION_KEYS = (
    "unique",
    "sparse",
    "expireAfterSeconds",
    "partialFilterExpression",
    "collation",
    "wildcardProjection",
    "hidden",
)
_BOOLEAN_INDEX_OPTION_KEYS = {"unique", "sparse", "hidden"}
_INDEX_CONFLICT_CODES = {85, 86}
_INDEX_CONFLICT_NAMES = {"IndexOptionsConflict", "IndexKeySpecsConflict"}


def _normalize_index_keys(keys: Iterable[tuple[str, Any]]) -> list[tuple[str, Any]]:
    normalized: list[tuple[str, Any]] = []
    for field, direction in keys:
        if direction in (1, -1):
            direction = int(direction)
        normalized.append((str(field), direction))
    return normalized


def _index_option_value(options: dict[str, Any], option_name: str) -> Any:
    if option_name in _BOOLEAN_INDEX_OPTION_KEYS:
        return bool(options.get(option_name, False))
    return _jsonable(options.get(option_name))


def _index_options_match(existing: dict[str, Any], requested: dict[str, Any]) -> bool:
    option_names = set(_INDEX_OPTION_KEYS)
    option_names.update(key for key in requested if key != "name")
    for option_name in option_names:
        if _canonical(_index_option_value(existing, option_name)) != _canonical(
            _index_option_value(requested, option_name)
        ):
            return False
    return True


def _index_matches_request(index_info: dict[str, Any], keys: Iterable[tuple[str, Any]], options: dict[str, Any]) -> bool:
    return _normalize_index_keys(index_info.get("key", [])) == _normalize_index_keys(keys) and _index_options_match(
        index_info,
        options,
    )


def _index_information(collection: Any) -> dict[str, dict[str, Any]]:
    index_information = getattr(collection, "index_information", None)
    if not callable(index_information):
        return {}
    return dict(index_information())


def _find_equivalent_index(
    collection: Any,
    keys: Iterable[tuple[str, Any]],
    options: dict[str, Any],
) -> str | None:
    for index_name, index_info in _index_information(collection).items():
        if _index_matches_request(index_info, keys, options):
            return str(index_name)
    return None


def _is_index_conflict_error(exc: Exception) -> bool:
    details = getattr(exc, "details", {}) or {}
    code = getattr(exc, "code", None) or details.get("code")
    code_name = details.get("codeName")
    message = str(exc).lower()
    return (
        code in _INDEX_CONFLICT_CODES
        or code_name in _INDEX_CONFLICT_NAMES
        or ("index already exists" in message and "different name" in message)
    )


def safe_create_index(collection: Any, keys: list[tuple[str, Any]], name: str, **kwargs: Any) -> str:
    """Create an index unless an equivalent index already exists under any name."""
    requested_options = dict(kwargs)
    existing_name = _find_equivalent_index(collection, keys, requested_options)
    if existing_name is not None:
        return existing_name

    try:
        return str(collection.create_index(keys, name=name, **kwargs))
    except Exception as exc:
        if exc.__class__.__name__ == "OperationFailure" and _is_index_conflict_error(exc):
            existing_name = _find_equivalent_index(collection, keys, requested_options)
            if existing_name is not None:
                return existing_name
        raise


def _without_update_only_fields(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in document.items()
        if key not in {"_id", "updated_at"}
    }


def _clean_document(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return _jsonable(document)


def _clean_documents(documents: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_jsonable(document) for document in documents]


class BaseMongoRepository:
    collection_name: str

    def __init__(self, collection: Any, collection_name: str) -> None:
        self.collection = collection
        self.collection_name = collection_name

    def _run(self, action: str, operation: Any) -> Any:
        try:
            return operation()
        except RepositoryUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - exercised through route fallback tests
            raise RepositoryUnavailableError(f"MongoDB {self.collection_name}.{action} failed: {exc}") from exc

    def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        return self._run("find_one", lambda: _clean_document(self.collection.find_one(query, {"_id": 0})))

    def count(self, query: dict[str, Any] | None = None) -> int:
        return int(self._run("count", lambda: self.collection.count_documents(query or {})))

    def upsert_document(self, query: dict[str, Any], document: dict[str, Any]) -> str:
        def operation() -> str:
            now = utc_now()
            incoming = dict(document)
            existing = self.collection.find_one(query, {"_id": 0})
            if existing:
                existing_clean = _jsonable(existing)
                incoming.setdefault("created_at", existing_clean.get("created_at") or now)
                incoming_for_compare = dict(incoming)
                incoming_for_compare["updated_at"] = existing_clean.get("updated_at")
                if _canonical(_without_update_only_fields(existing_clean)) == _canonical(
                    _without_update_only_fields(incoming_for_compare)
                ):
                    return "skipped"
                incoming["updated_at"] = now
                self.collection.update_one(query, {"$set": incoming}, upsert=True)
                return "updated"

            incoming.setdefault("created_at", now)
            incoming.setdefault("updated_at", now)
            self.collection.update_one(query, {"$set": incoming}, upsert=True)
            return "inserted"

        return str(self._run("upsert", operation))


class CandidateRepository(BaseMongoRepository):
    def list_candidates(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        def operation() -> list[dict[str, Any]]:
            cursor = self.collection.find({}, {"_id": 0})
            cursor = cursor.sort([("baseline_rank_v3", 1), ("candidate_id", 1)]).skip(offset).limit(limit)
            return _clean_documents(cursor)

        return self._run("list_candidates", operation)

    def count_candidates(self) -> int:
        return self.count()

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        return self.find_one(
            {
                "$or": [
                    {"candidate_id": candidate_id},
                    {"profile_id": candidate_id},
                    {"best_profile_id": candidate_id},
                ]
            }
        )

    def ensure_indexes(self) -> None:
        def operation() -> None:
            from pymongo import ASCENDING, DESCENDING

            safe_create_index(self.collection, [("candidate_id", ASCENDING)], name="uniq_candidate_id", unique=True)
            safe_create_index(self.collection, [("profile_id", ASCENDING)], name="idx_candidate_profile_id")
            safe_create_index(self.collection, [("best_profile_id", ASCENDING)], name="idx_best_profile_id")
            safe_create_index(self.collection, [("baseline_rank_v3", ASCENDING)], name="idx_baseline_rank_v3")
            safe_create_index(self.collection, [("updated_at", DESCENDING)], name="idx_candidates_updated_at")

        self._run("ensure_indexes", operation)


class CandidateProfileRepository(BaseMongoRepository):
    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        return self.find_one({"profile_id": profile_id})

    def get_profile_by_candidate_id(self, candidate_id: str) -> dict[str, Any] | None:
        return self.find_one({"candidate_id": candidate_id})

    def get_profiles_by_ids(self, profile_ids: list[str]) -> dict[str, dict[str, Any]]:
        clean_profile_ids = [str(profile_id) for profile_id in profile_ids if profile_id]
        if not clean_profile_ids:
            return {}

        def operation() -> dict[str, dict[str, Any]]:
            documents = _clean_documents(
                self.collection.find(
                    {"profile_id": {"$in": clean_profile_ids}},
                    {"_id": 0},
                )
            )
            return {
                str(document.get("profile_id")): document
                for document in documents
                if document.get("profile_id")
            }

        return self._run("get_profiles_by_ids", operation)

    def list_profiles(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        def operation() -> list[dict[str, Any]]:
            cursor = self.collection.find({}, {"_id": 0})
            cursor = cursor.sort([("profile_id", 1)]).skip(offset).limit(limit)
            return _clean_documents(cursor)

        return self._run("list_profiles", operation)

    def ensure_indexes(self) -> None:
        def operation() -> None:
            from pymongo import ASCENDING, DESCENDING

            safe_create_index(self.collection, [("profile_id", ASCENDING)], name="uniq_profile_id", unique=True)
            safe_create_index(self.collection, [("candidate_id", ASCENDING)], name="idx_profile_candidate_id")
            safe_create_index(self.collection, [("source_artifact", ASCENDING)], name="idx_profile_source_artifact")
            safe_create_index(self.collection, [("updated_at", DESCENDING)], name="idx_profiles_updated_at")

        self._run("ensure_indexes", operation)


class JobProfileRepository(BaseMongoRepository):
    def list_job_profiles(self) -> list[dict[str, Any]]:
        def operation() -> list[dict[str, Any]]:
            cursor = self.collection.find({}, {"_id": 0}).sort([("job_id", 1)])
            return _clean_documents(cursor)

        return self._run("list_job_profiles", operation)

    def get_job_profile(self, job_id: str) -> dict[str, Any] | None:
        return self.find_one({"job_id": job_id})

    def ensure_indexes(self) -> None:
        def operation() -> None:
            from pymongo import ASCENDING, DESCENDING

            safe_create_index(self.collection, [("job_id", ASCENDING)], name="uniq_job_id", unique=True)
            safe_create_index(self.collection, [("updated_at", DESCENDING)], name="idx_job_profiles_updated_at")

        self._run("ensure_indexes", operation)


class MatchingRunRepository(BaseMongoRepository):
    def save_matching_run(self, document: dict[str, Any]) -> str:
        now = utc_now()
        run_id = document.get("run_id") or stable_id(
            "matching_run",
            [now, document.get("job_id"), document.get("resolved_job_id"), document.get("artifact_source")],
        )
        payload = dict(document)
        payload["run_id"] = run_id
        payload.setdefault("created_at", now)

        def operation() -> str:
            self.collection.insert_one(payload)
            return run_id

        return str(self._run("save_matching_run", operation))

    def ensure_indexes(self) -> None:
        def operation() -> None:
            from pymongo import ASCENDING, DESCENDING

            safe_create_index(self.collection, [("run_id", ASCENDING)], name="uniq_run_id", unique=True)
            safe_create_index(self.collection, [("job_id", ASCENDING)], name="idx_matching_runs_job_id")
            safe_create_index(self.collection, [("created_at", DESCENDING)], name="idx_matching_runs_created_at")

        self._run("ensure_indexes", operation)


class DecisionCardRepository(BaseMongoRepository):
    def list_decision_cards(self) -> list[dict[str, Any]]:
        def operation() -> list[dict[str, Any]]:
            cursor = self.collection.find({}, {"_id": 0})
            cursor = cursor.sort([("baseline_rank_v3", 1), ("candidate_id", 1)])
            return _clean_documents(cursor)

        return self._run("list_decision_cards", operation)

    def count_decision_cards(self) -> int:
        return self.count()

    def get_decision_card(self, candidate_id: str) -> dict[str, Any] | None:
        return self.find_one(
            {
                "$or": [
                    {"candidate_id": candidate_id},
                    {"profile_id": candidate_id},
                ]
            }
        )

    def ensure_indexes(self) -> None:
        def operation() -> None:
            from pymongo import ASCENDING, DESCENDING

            safe_create_index(
                self.collection,
                [("candidate_id", ASCENDING)],
                name="uniq_decision_card_candidate_id",
                unique=True,
            )
            safe_create_index(self.collection, [("profile_id", ASCENDING)], name="idx_decision_card_profile_id")
            safe_create_index(
                self.collection,
                [("baseline_rank_v3", ASCENDING)],
                name="idx_decision_card_baseline_rank_v3",
            )
            safe_create_index(self.collection, [("updated_at", DESCENDING)], name="idx_decision_cards_updated_at")

        self._run("ensure_indexes", operation)


@dataclass
class MongoRepositories:
    client: Any
    database_name: str
    candidates: CandidateRepository
    candidate_profiles: CandidateProfileRepository
    job_profiles: JobProfileRepository
    matching_runs: MatchingRunRepository
    decision_cards: DecisionCardRepository

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def ensure_indexes(self) -> None:
        self.candidates.ensure_indexes()
        self.candidate_profiles.ensure_indexes()
        self.job_profiles.ensure_indexes()
        self.matching_runs.ensure_indexes()
        self.decision_cards.ensure_indexes()


def create_mongo_repositories(
    mongodb_uri: str | None = None,
    database_name: str | None = None,
    *,
    server_selection_timeout_ms: int = 1000,
) -> MongoRepositories:
    uri = mongodb_uri or os.getenv("MONGODB_URI", DEFAULT_MONGODB_URI)
    database = database_name or os.getenv("MONGODB_DATABASE", DEFAULT_MONGODB_DATABASE)

    try:
        from pymongo import MongoClient
    except ImportError as exc:  # pragma: no cover - pymongo is present in the project requirements
        raise RepositoryUnavailableError("pymongo is required to use DATA_BACKEND=mongodb or hybrid.") from exc

    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=server_selection_timeout_ms,
        connectTimeoutMS=server_selection_timeout_ms,
    )
    try:
        client.admin.command("ping")
    except Exception as exc:
        client.close()
        raise RepositoryUnavailableError(f"MongoDB is unavailable at {mask_mongodb_uri(uri)}: {exc}") from exc

    db = client[database]
    return MongoRepositories(
        client=client,
        database_name=database,
        candidates=CandidateRepository(
            db[os.getenv("MONGODB_CANDIDATES_COLLECTION", DEFAULT_CANDIDATES_COLLECTION)],
            os.getenv("MONGODB_CANDIDATES_COLLECTION", DEFAULT_CANDIDATES_COLLECTION),
        ),
        candidate_profiles=CandidateProfileRepository(
            db[os.getenv("MONGODB_CANDIDATE_PROFILES_COLLECTION", DEFAULT_CANDIDATE_PROFILES_COLLECTION)],
            os.getenv("MONGODB_CANDIDATE_PROFILES_COLLECTION", DEFAULT_CANDIDATE_PROFILES_COLLECTION),
        ),
        job_profiles=JobProfileRepository(
            db[os.getenv("MONGODB_JOB_PROFILES_COLLECTION", DEFAULT_JOB_PROFILES_COLLECTION)],
            os.getenv("MONGODB_JOB_PROFILES_COLLECTION", DEFAULT_JOB_PROFILES_COLLECTION),
        ),
        matching_runs=MatchingRunRepository(
            db[os.getenv("MONGODB_MATCHING_RUNS_COLLECTION", DEFAULT_MATCHING_RUNS_COLLECTION)],
            os.getenv("MONGODB_MATCHING_RUNS_COLLECTION", DEFAULT_MATCHING_RUNS_COLLECTION),
        ),
        decision_cards=DecisionCardRepository(
            db[os.getenv("MONGODB_DECISION_CARDS_COLLECTION", DEFAULT_DECISION_CARDS_COLLECTION)],
            os.getenv("MONGODB_DECISION_CARDS_COLLECTION", DEFAULT_DECISION_CARDS_COLLECTION),
        ),
    )
