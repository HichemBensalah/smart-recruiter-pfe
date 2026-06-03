"""
Live matching readiness diagnostic.

check_live_matching_readiness() returns a structured dict that tells whether
MongoDB, FAISS, id_map and SentenceTransformer are available — without
launching any matching. All infrastructure checks are best-effort: a failed
check adds a reason to blocking_reasons rather than raising.
"""
from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
from typing import Any


def check_live_matching_readiness(
    mongodb_uri: str = "mongodb://localhost:27017",
    mongodb_database: str = "talent_intelligence",
    faiss_index_path: str = "data/indexes/faiss/cv_index.faiss",
    faiss_id_map_path: str = "data/indexes/faiss/id_map.pkl",
) -> dict[str, Any]:
    """
    Probe every infrastructure dependency needed for live matching.

    Returns a dict with keys:
      live_ready                : bool — True iff all checks pass
      mongodb_available         : bool
      candidate_profiles_count  : int  (0 if unknown or MongoDB unavailable)
      faiss_index_available     : bool
      faiss_index_size          : int | None  (bytes)
      id_map_available          : bool
      id_map_size               : int | None  (number of rows)
      sentence_transformer_available : bool
      id_map_faiss_coherent     : bool | None
      live_matcher_instantiable : bool
      blocking_reasons          : list[str]
    """
    blocking_reasons: list[str] = []
    result: dict[str, Any] = {
        "live_ready": False,
        "mongodb_available": False,
        "candidate_profiles_count": 0,
        "faiss_index_available": False,
        "faiss_index_size": None,
        "id_map_available": False,
        "id_map_size": None,
        "sentence_transformer_available": False,
        "id_map_faiss_coherent": None,
        "live_matcher_instantiable": False,
        "blocking_reasons": blocking_reasons,
    }

    _check_faiss_index(faiss_index_path, result, blocking_reasons)
    _check_id_map(faiss_id_map_path, result, blocking_reasons)
    _check_sentence_transformer(result, blocking_reasons)
    _check_mongodb(mongodb_uri, mongodb_database, result, blocking_reasons)
    _check_id_map_coherence(result, blocking_reasons)

    result["live_ready"] = (
        result["mongodb_available"]
        and result["candidate_profiles_count"] > 0
        and result["faiss_index_available"]
        and result["id_map_available"]
        and result["sentence_transformer_available"]
    )

    if result["live_ready"]:
        result["live_matcher_instantiable"] = True

    return result


# ── Private helpers ───────────────────────────────────────────────────────────

def _check_faiss_index(path_str: str, result: dict, reasons: list) -> None:
    path = Path(path_str)
    if path.exists():
        result["faiss_index_available"] = True
        try:
            result["faiss_index_size"] = path.stat().st_size
        except Exception:
            pass
    else:
        reasons.append(f"FAISS index not found: {path_str}")


def _check_id_map(path_str: str, result: dict, reasons: list) -> None:
    path = Path(path_str)
    if not path.exists():
        reasons.append(f"id_map not found: {path_str}")
        return
    result["id_map_available"] = True
    try:
        with open(path, "rb") as f:
            id_map = pickle.load(f)
        result["id_map_size"] = len(id_map) if isinstance(id_map, list) else None
        if isinstance(id_map, list) and len(id_map) == 0:
            reasons.append("id_map is empty — no candidates in FAISS index")
    except Exception as exc:
        reasons.append(f"id_map could not be loaded: {exc}")


def _check_sentence_transformer(result: dict, reasons: list) -> None:
    spec = importlib.util.find_spec("sentence_transformers")
    if spec is not None:
        result["sentence_transformer_available"] = True
    else:
        reasons.append(
            "sentence_transformers package not installed — "
            "run: pip install sentence-transformers"
        )


def _check_mongodb(uri: str, database: str, result: dict, reasons: list) -> None:
    try:
        from src.core.storage.repositories import (
            RepositoryUnavailableError,
            create_mongo_repositories,
        )
        try:
            repositories = create_mongo_repositories(uri, database)
            result["mongodb_available"] = True
            try:
                collection = getattr(
                    getattr(repositories, "candidate_profiles", None),
                    "collection",
                    None,
                )
                if collection is not None:
                    count = collection.estimated_document_count()
                    result["candidate_profiles_count"] = int(count)
                    if count == 0:
                        reasons.append(
                            "candidate_profiles collection is empty — "
                            "no candidates available for live matching"
                        )
            except Exception:
                result["candidate_profiles_count"] = -1
            repositories.close()
        except RepositoryUnavailableError as exc:
            reasons.append(f"MongoDB unavailable: {exc}")
        except Exception as exc:
            reasons.append(f"MongoDB connection failed: {exc}")
    except ImportError as exc:
        reasons.append(f"Storage repositories module not importable: {exc}")


def _check_id_map_coherence(result: dict, reasons: list) -> None:
    if not result["mongodb_available"] or not result["id_map_available"]:
        return
    id_map_size = result.get("id_map_size")
    profile_count = result.get("candidate_profiles_count", 0)
    if id_map_size is not None and id_map_size > 0 and profile_count > 0:
        result["id_map_faiss_coherent"] = True
    elif id_map_size == 0 or profile_count == 0:
        result["id_map_faiss_coherent"] = False
        if id_map_size == 0:
            reasons.append("id_map is empty — FAISS index has no entries")
