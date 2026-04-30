from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any


WEIGHT_SKILLS = 0.40
WEIGHT_TEXT_SIMILARITY = 0.30
WEIGHT_EXPERIENCE = 0.20
WEIGHT_PROFILE_QUALITY = 0.10


def compute_skill_score(job_profile: dict[str, Any], candidate_profile: dict[str, Any]) -> float:
    required_skills = _normalize_tokens(job_profile.get("required_skills"))
    if not required_skills:
        return 0.5

    candidate_skills = _candidate_skill_tokens(candidate_profile)
    if not candidate_skills:
        return 0.0

    matched = required_skills & candidate_skills
    return round(len(matched) / max(len(required_skills), 1), 4)


def compute_experience_score(job_profile: dict[str, Any], candidate_profile: dict[str, Any]) -> float:
    required_years = job_profile.get("years_experience_required")
    candidate_years = _estimate_candidate_years(candidate_profile.get("experiences") or [])

    if not isinstance(required_years, (int, float)):
        if candidate_years > 0:
            return 1.0
        return 0.4 if candidate_profile.get("experiences") else 0.0

    if candidate_years <= 0:
        return 0.0
    return round(min(candidate_years / float(required_years), 1.0), 4)


def compute_profile_quality_score(candidate_profile: dict[str, Any]) -> float:
    reliability = float(candidate_profile.get("reliability_score") or 0.0)
    profile_kind = str(candidate_profile.get("profile_kind") or "").lower()
    provider_route = str(candidate_profile.get("provider_route") or "").lower()

    kind_bonus = 1.0 if profile_kind == "complete_profile" else 0.75 if profile_kind == "partial_profile" else 0.5
    provider_bonus = 1.0 if provider_route == "groq_secondary" else 0.85 if provider_route == "ollama_local" else 0.7
    score = (0.5 * reliability) + (0.3 * kind_bonus) + (0.2 * provider_bonus)
    return round(min(max(score, 0.0), 1.0), 4)


def combine_scores(
    score_text_similarity: float,
    score_skills: float,
    score_experience: float,
    score_profile_quality: float,
) -> float:
    final_score = (
        WEIGHT_SKILLS * score_skills
        + WEIGHT_TEXT_SIMILARITY * score_text_similarity
        + WEIGHT_EXPERIENCE * score_experience
        + WEIGHT_PROFILE_QUALITY * score_profile_quality
    )
    return round(min(max(final_score, 0.0), 1.0), 4)


def extract_matched_skills(job_profile: dict[str, Any], candidate_profile: dict[str, Any]) -> list[str]:
    required_skills = _normalize_tokens(job_profile.get("required_skills"))
    if not required_skills:
        return []
    candidate_skill_map = _candidate_skill_map(candidate_profile)
    matched = sorted(required_skills & set(candidate_skill_map.keys()))
    return [candidate_skill_map[key] for key in matched]


def extract_missing_required_skills(job_profile: dict[str, Any], candidate_profile: dict[str, Any]) -> list[str]:
    required_skill_map = _value_map(job_profile.get("required_skills"))
    candidate_skills = _candidate_skill_tokens(candidate_profile)
    missing = sorted(set(required_skill_map.keys()) - candidate_skills)
    return [required_skill_map[key] for key in missing]


def _candidate_skill_tokens(candidate_profile: dict[str, Any]) -> set[str]:
    expertise = candidate_profile.get("expertise") or {}
    hard_skills = expertise.get("hard_skills") or []
    soft_skills = expertise.get("soft_skills") or []
    return set(_value_map(list(hard_skills) + list(soft_skills)).keys())


def _candidate_skill_map(candidate_profile: dict[str, Any]) -> dict[str, str]:
    expertise = candidate_profile.get("expertise") or {}
    hard_skills = expertise.get("hard_skills") or []
    soft_skills = expertise.get("soft_skills") or []
    return _value_map(list(hard_skills) + list(soft_skills))


def _value_map(values: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(values, list):
        return result
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.split()).strip()
        if not cleaned:
            continue
        result.setdefault(_normalize_token(cleaned), cleaned)
    return result


def _normalize_tokens(values: Any) -> set[str]:
    return set(_value_map(values).keys())


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _estimate_candidate_years(experiences: list[Any]) -> float:
    if not isinstance(experiences, list) or not experiences:
        return 0.0

    total_years = 0.0
    dated_spans = 0
    for experience in experiences:
        if not isinstance(experience, dict):
            continue
        start_year = _extract_year(experience.get("start_date"))
        end_year = _extract_end_year(experience.get("end_date"))
        if start_year is not None and end_year is not None and end_year >= start_year:
            total_years += max(1.0, float(end_year - start_year))
            dated_spans += 1

    if dated_spans > 0:
        return round(min(total_years, 40.0), 2)

    return round(min(len(experiences) * 1.5, 12.0), 2)


def _extract_year(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"(19|20)\d{2}", value)
    return int(match.group(0)) if match else None


def _extract_end_year(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if cleaned in {"present", "current", "now", "ongoing"}:
        return datetime.utcnow().year
    return _extract_year(value)
