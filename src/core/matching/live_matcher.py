from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.core.common.seniority import compute_seniority_alignment, normalize_seniority
from src.core.jobs.job_profile_builder import build_job_profile
from src.core.matching.faiss_indexer import DEFAULT_SENTENCE_MODEL
from src.core.matching.job_text_builder import build_job_text
from src.core.matching.matching_quality_filters import build_display_name, enrich_grounded_quality
from src.core.matching.recommender import (
    _build_explanation,
    _load_sentence_transformer,
    group_by_candidate_id,
    load_faiss_index,
    load_id_map,
    select_best_profile_per_candidate,
)
from src.core.matching.scoring import score_candidate
from src.core.storage.repositories import MongoRepositories, RepositoryUnavailableError, format_mongodb_source


class LiveMatchingUnavailable(RuntimeError):
    """Raised when live MongoDB + FAISS matching cannot produce a reliable result."""


@dataclass(frozen=True)
class LiveMatcherSettings:
    mongodb_database: str
    top_n: int = 50
    default_top_k: int = 5
    faiss_index_path: Path = Path("data/indexes/faiss/cv_index.faiss")
    faiss_id_map_path: Path = Path("data/indexes/faiss/id_map.pkl")
    sentence_model: str = DEFAULT_SENTENCE_MODEL


@dataclass
class LiveMatchResult:
    job_id: str | None
    resolved_job_id: str | None
    top_k: int
    items: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    matching_run_id: str | None = None
    data_source: str | None = None
    retrieval_source: str | None = None
    scoring_source: str = "matching_v3.score_candidate"
    job_profile: dict[str, Any] = field(default_factory=dict)


class LiveMatcher:
    def __init__(
        self,
        repositories: MongoRepositories,
        settings: LiveMatcherSettings,
        *,
        index_loader: Callable[[Path], Any] = load_faiss_index,
        id_map_loader: Callable[[Path], list[dict[str, Any]]] = load_id_map,
        model_loader: Callable[[str], Any] = _load_sentence_transformer,
    ) -> None:
        self.repositories = repositories
        self.settings = settings
        self.index_loader = index_loader
        self.id_map_loader = id_map_loader
        self.model_loader = model_loader

    def match(self, *, job_description: str, job_id: str | None = None, top_k: int | None = None) -> LiveMatchResult:
        requested_top_k = top_k or self.settings.default_top_k
        warnings: list[str] = []
        job_profile = self._resolve_job_profile(job_description=job_description, job_id=job_id, warnings=warnings)
        resolved_job_id = job_profile.get("job_id") or job_id
        retrieved_rows = self._retrieve_candidates(job_profile=job_profile, top_n=max(self.settings.top_n, requested_top_k))
        profile_ids = [str(row.get("profile_id")) for row in retrieved_rows if row.get("profile_id")]
        profiles_by_id = self.repositories.candidate_profiles.get_profiles_by_ids(profile_ids)

        matches: list[dict[str, Any]] = []
        for row in retrieved_rows:
            profile_id = str(row.get("profile_id") or "")
            profile = profiles_by_id.get(profile_id)
            if not profile:
                warnings.append(f"Candidate profile not found in MongoDB for profile_id={profile_id}.")
                continue
            matches.append(self._score_retrieved_profile(job_profile=job_profile, profile=profile, retrieval_row=row))

        if not matches:
            raise LiveMatchingUnavailable("Live matching found no scoreable candidate profiles from MongoDB.")

        unique_candidates = select_best_profile_per_candidate(group_by_candidate_id(matches))
        items: list[dict[str, Any]] = []
        for rank, recommendation in enumerate(unique_candidates[:requested_top_k], start=1):
            item = dict(recommendation)
            item["rank"] = rank
            item["baseline_rank_v3"] = rank
            item["baseline_score_v3"] = item.get("final_score")
            item["profile_id"] = item.get("matched_profile_id")
            item["source"] = "live_mongodb_faiss_matching_v3"
            item["recommendation_status"] = _recommendation_status(item.get("final_score"))
            item["features"] = _features_from_match(item)
            items.append(item)

        matching_run_id = self._save_matching_run(
            job_description=job_description,
            job_profile=job_profile,
            job_id=job_id,
            resolved_job_id=resolved_job_id,
            top_k=requested_top_k,
            items=items,
            warnings=warnings,
        )
        data_source = format_mongodb_source(
            self.settings.mongodb_database,
            getattr(self.repositories.candidate_profiles, "collection_name", "candidate_profiles"),
        )
        return LiveMatchResult(
            job_id=job_id,
            resolved_job_id=resolved_job_id,
            top_k=requested_top_k,
            items=items,
            warnings=warnings,
            matching_run_id=matching_run_id,
            data_source=data_source,
            retrieval_source=f"faiss:{self.settings.faiss_index_path.as_posix()}",
            job_profile=job_profile,
        )

    def _resolve_job_profile(self, *, job_description: str, job_id: str | None, warnings: list[str]) -> dict[str, Any]:
        if job_id:
            job_profile = self.repositories.job_profiles.get_job_profile(job_id)
            if job_profile:
                payload = dict(job_profile)
                payload.setdefault("raw_job_description", job_description)
                return payload
            warnings.append(f"Job profile not found in MongoDB for job_id={job_id}; built a rule-based profile from the request.")

        built = build_job_profile(job_description).model_dump()
        if job_id:
            built["job_id"] = job_id
        return built

    def _retrieve_candidates(self, *, job_profile: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
        try:
            index = self.index_loader(self.settings.faiss_index_path)
            id_map = self.id_map_loader(self.settings.faiss_id_map_path)
            model = self.model_loader(self.settings.sentence_model)
        except Exception as exc:
            raise LiveMatchingUnavailable(f"FAISS retrieval is unavailable: {exc}") from exc

        if not id_map:
            raise LiveMatchingUnavailable("FAISS id_map is empty; live matching cannot retrieve candidates.")

        job_text = build_job_text(job_profile)
        embedding = model.encode(
            [job_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        job_embedding = np.asarray(embedding, dtype="float32")
        search_k = min(max(top_n, 1), len(id_map))
        try:
            distances, indices = index.search(job_embedding, search_k)
        except Exception as exc:
            raise LiveMatchingUnavailable(f"FAISS search failed: {exc}") from exc

        retrieved_rows: list[dict[str, Any]] = []
        for faiss_rank, (score, row_index) in enumerate(zip(distances[0], indices[0]), start=1):
            row_index = int(row_index)
            if row_index < 0 or row_index >= len(id_map):
                continue
            row = dict(id_map[row_index])
            row["faiss_rank"] = faiss_rank
            row["faiss_score"] = round(float(score), 4)
            row["score_text_similarity"] = row["faiss_score"]
            retrieved_rows.append(row)

        if not retrieved_rows:
            raise LiveMatchingUnavailable("FAISS search returned no usable candidate rows.")
        return retrieved_rows

    def _score_retrieved_profile(
        self,
        *,
        job_profile: dict[str, Any],
        profile: dict[str, Any],
        retrieval_row: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_profile = normalize_candidate_profile_for_matching(profile)
        normalized_profile.setdefault("profile_id", retrieval_row.get("profile_id"))
        normalized_profile.setdefault("candidate_id", retrieval_row.get("candidate_id"))
        semantic_score = float(retrieval_row.get("score_text_similarity") or 0.0)
        score_details = score_candidate(
            job_profile=job_profile,
            candidate_profile=normalized_profile,
            score_text_similarity=semantic_score,
        )
        bio = normalized_profile.get("bio") or {}
        candidate_id = normalized_profile.get("candidate_id") or retrieval_row.get("candidate_id")
        profile_id = normalized_profile.get("profile_id") or retrieval_row.get("profile_id")
        display_name, display_name_quality, name_warning = build_display_name(bio.get("full_name"), candidate_id)
        grounded_quality = enrich_grounded_quality(normalized_profile)
        job_seniority = normalize_seniority(job_profile.get("seniority_level"))
        candidate_seniority = normalize_seniority((normalized_profile.get("expertise") or {}).get("experience_level"))
        seniority_alignment = compute_seniority_alignment(job_seniority, candidate_seniority)

        return {
            "candidate_id": candidate_id or profile_id,
            "matched_profile_id": profile_id,
            "full_name": display_name,
            "display_name_quality": display_name_quality,
            "name_warning": name_warning,
            "profile_kind": normalized_profile.get("profile_kind"),
            "source_path": normalized_profile.get("source_path"),
            "provider_route": normalized_profile.get("provider_route"),
            "job_seniority": job_seniority,
            "candidate_seniority": candidate_seniority,
            "seniority_alignment": seniority_alignment,
            "seniority_warning": None if candidate_seniority else "missing_candidate_seniority",
            "reliability_score": score_details["reliability_score"],
            "hallucination_risk": grounded_quality["hallucination_risk"],
            "faiss_rank": retrieval_row.get("faiss_rank"),
            "faiss_score": retrieval_row.get("faiss_score"),
            "quality_flags": list(normalized_profile.get("quality_flags") or []),
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
                profile_kind=normalized_profile.get("profile_kind"),
            ),
            "profile_count": 1,
        }

    def _save_matching_run(
        self,
        *,
        job_description: str,
        job_profile: dict[str, Any],
        job_id: str | None,
        resolved_job_id: str | None,
        top_k: int,
        items: list[dict[str, Any]],
        warnings: list[str],
    ) -> str | None:
        try:
            return self.repositories.matching_runs.save_matching_run(
                {
                    "job_id": job_id,
                    "resolved_job_id": resolved_job_id,
                    "job_description": job_description,
                    "job_profile": job_profile,
                    "matching_mode": "live_mongodb_faiss_matching_v3",
                    "top_k": top_k,
                    "candidate_ids": [item.get("candidate_id") for item in items],
                    "scores": [
                        {
                            "candidate_id": item.get("candidate_id"),
                            "profile_id": item.get("profile_id") or item.get("matched_profile_id"),
                            "rank": item.get("rank"),
                            "final_score_v3": item.get("baseline_score_v3") or item.get("final_score"),
                            "faiss_score": item.get("faiss_score"),
                        }
                        for item in items
                    ],
                    "warnings": list(warnings),
                    "source_metadata": {
                        "data_source": format_mongodb_source(
                            self.settings.mongodb_database,
                            getattr(self.repositories.candidate_profiles, "collection_name", "candidate_profiles"),
                        ),
                        "retrieval_source": f"faiss:{self.settings.faiss_index_path.as_posix()}",
                        "scoring_source": "matching_v3.score_candidate",
                        "faiss_id_map_path": self.settings.faiss_id_map_path.as_posix(),
                        "sentence_model": self.settings.sentence_model,
                    },
                    "items": items,
                }
            )
        except RepositoryUnavailableError as exc:
            warnings.append(f"Matching run was not persisted because MongoDB write failed: {exc}")
            return None


def normalize_candidate_profile_for_matching(document: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(document.get("profile"), dict):
        return dict(document)

    payload = copy.deepcopy(document)
    profile = payload.get("profile") or {}
    grounding = payload.get("grounding") or {}
    normalization = payload.get("normalization") or {}
    metadata = profile.get("metadata") or payload.get("metadata") or {}
    quality_flags = _unique_strings(
        list(payload.get("quality_flags") or [])
        + list(normalization.get("quality_flags") or [])
        + list(grounding.get("quality_flags") or [])
    )
    fields_nullified = list(grounding.get("fields_nullified") or payload.get("fields_nullified") or [])

    return {
        **payload,
        "bio": copy.deepcopy(profile.get("bio") or payload.get("bio") or {}),
        "expertise": copy.deepcopy(profile.get("expertise") or payload.get("expertise") or {}),
        "experiences": copy.deepcopy(profile.get("experiences") or payload.get("experiences") or []),
        "education": copy.deepcopy(profile.get("education") or payload.get("education") or []),
        "languages": copy.deepcopy(profile.get("languages") or payload.get("languages") or []),
        "certifications": copy.deepcopy(profile.get("certifications") or payload.get("certifications") or []),
        "profile_kind": payload.get("profile_kind") or profile.get("profile_kind"),
        "provider_route": payload.get("provider_route") or payload.get("provider_used") or metadata.get("provider_route"),
        "reliability_score": float(
            payload.get("reliability_score")
            or grounding.get("reliability_score")
            or payload.get("document_confidence_score")
            or 0.0
        ),
        "hallucination_risk": payload.get("hallucination_risk") or grounding.get("hallucination_risk"),
        "quality_flags": quality_flags,
        "fields_nullified": fields_nullified,
        "fields_nullified_count": len(fields_nullified),
    }


def _features_from_match(item: dict[str, Any]) -> dict[str, Any]:
    matched_skills = item.get("matched_skills") or []
    missing_skills = item.get("missing_required_skills") or []
    return {
        "vector_similarity": item.get("score_text_similarity") or item.get("faiss_score"),
        "final_score_v3": item.get("final_score"),
        "must_have_coverage": item.get("must_have_coverage"),
        "required_skills_overlap": item.get("score_skills"),
        "experience_match_score": item.get("score_experience"),
        "seniority_alignment": item.get("seniority_alignment"),
        "profile_quality_score": item.get("score_profile_quality"),
        "reliability_score": item.get("reliability_score"),
        "missing_required_count": float(len(missing_skills)),
        "matched_required_count": float(len(matched_skills)),
    }


def _recommendation_status(score: Any) -> str:
    numeric_score = float(score or 0.0)
    if numeric_score >= 0.75:
        return "strong_match"
    if numeric_score >= 0.55:
        return "review_needed"
    return "weak_match"


def _unique_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.split()).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result
