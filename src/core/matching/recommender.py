from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

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
from .matching_quality_filters import build_display_name, enrich_grounded_quality
from .profile_text_builder import build_candidate_text
from .scoring import score_candidate
from ..retrieval.cross_encoder_reranker import (
    DEFAULT_CROSS_ENCODER_MODEL,
    rerank_candidates_with_cross_encoder,
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
                    "languages": 1,
                    "certifications": 1,
                    "profile_kind": 1,
                    "provider_route": 1,
                    "reliability_score": 1,
                    "hallucination_risk": 1,
                    "quality_flags": 1,
                    "fields_nullified": 1,
                    "fields_nullified_count": 1,
                    "source_path": 1,
                    "artifact_path": 1,
                    "module2_file_path": 1,
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
    for _, items in grouped_matches.items():
        best = max(
            items,
            key=lambda item: (
                item["final_score"],
                item["score_text_similarity"],
                item["score_grounded_quality"],
            ),
        )
        best["profile_count"] = len(items)
        selected.append(best)
    selected.sort(key=lambda item: (item["final_score"], item["score_grounded_quality"]), reverse=True)
    return selected


def recommend_candidates(
    job_profile: dict[str, Any],
    top_k: int = 10,
    *,
    use_cross_encoder: bool = False,
    cross_encoder_top_n: int = 20,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> list[dict[str, Any]]:
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

    search_k = min(max(cross_encoder_top_n if use_cross_encoder else top_k * 5, top_k, 20), len(id_map))
    distances, indices = index.search(job_embedding, search_k)

    retrieved_rows = []
    for faiss_rank, (score, row_index) in enumerate(zip(distances[0], indices[0]), start=1):
        if row_index < 0 or row_index >= len(id_map):
            continue
        row = dict(id_map[row_index])
        row["faiss_rank"] = faiss_rank
        row["faiss_score"] = round(float(score), 4)
        row["score_text_similarity"] = row["faiss_score"]
        retrieved_rows.append(row)

    profile_ids = [str(row["profile_id"]) for row in retrieved_rows]
    profiles_by_id = retrieve_candidate_profiles(profile_ids)

    candidate_rows: list[dict[str, Any]] = []
    for row in retrieved_rows:
        profile = profiles_by_id.get(str(row["profile_id"]))
        if not profile:
            continue
        candidate_rows.append(
            {
                **row,
                "profile": profile,
                "candidate_text": build_candidate_text(profile),
            }
        )

    reranked_rows = candidate_rows
    if use_cross_encoder:
        reranked_rows = rerank_candidates_with_cross_encoder(
            job_text=job_text,
            candidates=candidate_rows,
            candidate_text_key="candidate_text",
            top_k=min(cross_encoder_top_n, len(candidate_rows)),
            model_name=cross_encoder_model,
        )

    matches: list[dict[str, Any]] = []
    for row in reranked_rows:
        profile = row["profile"]
        semantic_score = row["score_text_similarity"]
        if use_cross_encoder and row.get("cross_encoder_score_normalized") is not None:
            semantic_score = float(row["cross_encoder_score_normalized"])
        score_details = score_candidate(
            job_profile=job_profile,
            candidate_profile=profile,
            score_text_similarity=semantic_score,
        )
        bio = profile.get("bio") or {}
        candidate_id = profile.get("candidate_id")
        display_name, display_name_quality, name_warning = build_display_name(
            bio.get("full_name"),
            candidate_id,
        )
        grounded_quality = enrich_grounded_quality(profile)

        matches.append(
            {
                "candidate_id": candidate_id,
                "matched_profile_id": profile.get("profile_id"),
                "full_name": display_name,
                "display_name_quality": display_name_quality,
                "name_warning": name_warning,
                "profile_kind": profile.get("profile_kind"),
                "source_path": profile.get("source_path"),
                "provider_route": profile.get("provider_route"),
                "reliability_score": score_details["reliability_score"],
                "hallucination_risk": grounded_quality["hallucination_risk"],
                "faiss_rank": row.get("faiss_rank"),
                "faiss_score": row.get("faiss_score"),
                "cross_encoder_rank": row.get("cross_encoder_rank"),
                "cross_encoder_score": row.get("cross_encoder_score"),
                "cross_encoder_score_normalized": row.get("cross_encoder_score_normalized"),
                "cross_encoder_error": row.get("cross_encoder_error"),
                "quality_flags": list(profile.get("quality_flags") or []),
                "fields_nullified_count": grounded_quality["fields_nullified_count"],
                **score_details,
                "explanation": _build_explanation(
                    full_name=display_name,
                    candidate_id=candidate_id,
                    matched_skills=score_details["matched_skills"],
                    missing_required_skills=score_details["missing_required_skills"],
                    score_text_similarity=score_details["score_text_similarity"],
                    score_experience=score_details["score_experience"],
                    hallucination_risk=grounded_quality["hallucination_risk"],
                    profile_kind=profile.get("profile_kind"),
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
    candidate_id: Any,
    matched_skills: list[str],
    missing_required_skills: list[str],
    score_text_similarity: float,
    score_experience: float,
    hallucination_risk: str,
    profile_kind: Any,
) -> str:
    name, _, _ = build_display_name(full_name, candidate_id)
    matched_text = ", ".join(matched_skills[:5]) if matched_skills else "no strong required skill match yet"
    quality_text = f"profile kind={profile_kind}, hallucination risk={hallucination_risk}"
    if missing_required_skills:
        missing_text = ", ".join(missing_required_skills[:3])
        return (
            f"{name} matches on {matched_text}. "
            f"Text similarity={score_text_similarity:.2f}, experience score={score_experience:.2f}, {quality_text}. "
            f"Missing required skills include {missing_text}."
        )
    return (
        f"{name} matches on {matched_text}. "
        f"Text similarity={score_text_similarity:.2f}, experience score={score_experience:.2f}, {quality_text}. "
        "No required skill gap detected in the grounded scorer."
    )
