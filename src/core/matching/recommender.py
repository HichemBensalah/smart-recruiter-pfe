from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .faiss_indexer import (
    DEFAULT_COLLECTION,
    DEFAULT_DATABASE,
    DEFAULT_ID_MAP_PATH,
    DEFAULT_INDEX_PATH,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_MONGODB_URI,
    DEFAULT_SENTENCE_MODEL,
)
from .job_text_builder import build_job_text
from .scoring import (
    combine_scores,
    compute_experience_score,
    compute_profile_quality_score,
    compute_skill_score,
    extract_matched_skills,
    extract_missing_required_skills,
)


def load_faiss_index(index_path: Path = DEFAULT_INDEX_PATH) -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Missing dependency: faiss-cpu") from exc
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(str(index_path))


def load_id_map(id_map_path: Path = DEFAULT_ID_MAP_PATH) -> list[dict[str, Any]]:
    if not id_map_path.exists():
        raise FileNotFoundError(f"FAISS id map not found: {id_map_path}")
    with id_map_path.open("rb") as handle:
        return pickle.load(handle)


def retrieve_candidate_profiles(
    profile_ids: list[str],
    mongodb_uri: str = DEFAULT_MONGODB_URI,
    database_name: str = DEFAULT_DATABASE,
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, dict[str, Any]]:
    """Retrieve candidate profile docs for the requested profile ids."""
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pymongo") from exc

    client = MongoClient(mongodb_uri)
    try:
        collection = client[database_name][collection_name]
        documents = list(
            collection.find(
                {"profile_id": {"$in": profile_ids}, "status": "success"},
                {
                    "_id": 0,
                    "profile_id": 1,
                    "candidate_id": 1,
                    "bio": 1,
                    "expertise": 1,
                    "experiences": 1,
                    "education": 1,
                    "profile_kind": 1,
                    "provider_route": 1,
                    "reliability_score": 1,
                    "source_path": 1,
                    "artifact_path": 1,
                    "status": 1,
                },
            )
        )
    finally:
        client.close()
    return {str(document.get("profile_id")): document for document in documents}


def group_by_candidate_id(matches: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for match in matches:
        candidate_id = str(match.get("candidate_id") or match.get("matched_profile_id") or "unknown_candidate")
        grouped.setdefault(candidate_id, []).append(match)
    return grouped


def select_best_profile_per_candidate(grouped_matches: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate_id, items in grouped_matches.items():
        best = max(items, key=lambda item: (item["final_score"], item["score_text_similarity"]))
        best["profile_count"] = len(items)
        selected.append(best)
    selected.sort(key=lambda item: item["final_score"], reverse=True)
    return selected


def recommend_candidates(job_profile: dict[str, Any], top_k: int = 10) -> list[dict[str, Any]]:
    """Retrieve nearby profiles from FAISS, rescore them, then keep unique candidates."""
    index = load_faiss_index()
    id_map = load_id_map()
    model = _load_sentence_transformer(DEFAULT_SENTENCE_MODEL)
    job_text = build_job_text(job_profile)
    job_embedding = model.encode(
        [job_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    search_k = min(max(top_k * 5, 20), len(id_map))
    distances, indices = index.search(job_embedding, search_k)

    retrieved_rows = []
    for score, row_index in zip(distances[0], indices[0]):
        if row_index < 0 or row_index >= len(id_map):
            continue
        row = dict(id_map[row_index])
        row["score_text_similarity"] = round(float(score), 4)
        retrieved_rows.append(row)

    profile_ids = [str(row["profile_id"]) for row in retrieved_rows]
    profiles_by_id = retrieve_candidate_profiles(profile_ids)

    matches: list[dict[str, Any]] = []
    for row in retrieved_rows:
        profile = profiles_by_id.get(str(row["profile_id"]))
        if not profile:
            continue

        score_text_similarity = row["score_text_similarity"]
        score_skills = compute_skill_score(job_profile, profile)
        score_experience = compute_experience_score(job_profile, profile)
        score_profile_quality = compute_profile_quality_score(profile)
        final_score = combine_scores(
            score_text_similarity=score_text_similarity,
            score_skills=score_skills,
            score_experience=score_experience,
            score_profile_quality=score_profile_quality,
        )

        bio = profile.get("bio") or {}
        matched_skills = extract_matched_skills(job_profile, profile)
        missing_required_skills = extract_missing_required_skills(job_profile, profile)
        matches.append(
            {
                "candidate_id": profile.get("candidate_id"),
                "matched_profile_id": profile.get("profile_id"),
                "full_name": bio.get("full_name"),
                "profile_kind": profile.get("profile_kind"),
                "source_path": profile.get("source_path"),
                "provider_route": profile.get("provider_route"),
                "final_score": final_score,
                "score_text_similarity": score_text_similarity,
                "score_skills": score_skills,
                "score_experience": score_experience,
                "score_profile_quality": score_profile_quality,
                "matched_skills": matched_skills,
                "missing_required_skills": missing_required_skills,
                "explanation": _build_explanation(
                    full_name=bio.get("full_name"),
                    matched_skills=matched_skills,
                    missing_required_skills=missing_required_skills,
                    score_text_similarity=score_text_similarity,
                    score_experience=score_experience,
                ),
                "profile_count": 1,
            }
        )

    grouped = group_by_candidate_id(matches)
    unique_candidates = select_best_profile_per_candidate(grouped)
    return unique_candidates[:top_k]


def _load_sentence_transformer(model_name: str) -> Any:
    import os

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Missing dependency: sentence-transformers") from exc
    DEFAULT_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(DEFAULT_MODEL_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_MODEL_CACHE_DIR))
    return SentenceTransformer(model_name, cache_folder=str(DEFAULT_MODEL_CACHE_DIR))


def _build_explanation(
    *,
    full_name: Any,
    matched_skills: list[str],
    missing_required_skills: list[str],
    score_text_similarity: float,
    score_experience: float,
) -> str:
    name = full_name if isinstance(full_name, str) and full_name.strip() else "Candidate"
    matched_text = ", ".join(matched_skills[:5]) if matched_skills else "no strong required skill match yet"
    if missing_required_skills:
        missing_text = ", ".join(missing_required_skills[:3])
        return (
            f"{name} matches on {matched_text}. "
            f"Text similarity={score_text_similarity:.2f}, experience score={score_experience:.2f}. "
            f"Missing required skills include {missing_text}."
        )
    return (
        f"{name} matches on {matched_text}. "
        f"Text similarity={score_text_similarity:.2f}, experience score={score_experience:.2f}. "
        "No required skill gap detected in the baseline scorer."
    )
