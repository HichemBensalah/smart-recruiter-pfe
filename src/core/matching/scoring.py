from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .matching_quality_filters import build_display_name, enrich_grounded_quality
from .skill_normalizer import flatten_skill_sources, normalize_skills, skills_overlap


WEIGHT_SKILLS = 0.40
WEIGHT_TEXT_SIMILARITY = 0.30
WEIGHT_EXPERIENCE = 0.20
WEIGHT_PROFILE_QUALITY = 0.10

MUST_HAVE_PENALTY_TIERS = [
    (0.8, 1.00),
    (0.6, 0.85),
    (0.4, 0.65),
    (0.0, 0.45),
]

RISK_MULTIPLIERS = {
    "low": 1.00,
    "medium": 0.90,
    "high": 0.70,
}

PROFILE_KIND_MULTIPLIERS = {
    "complete_profile": 1.00,
    "partial_profile": 0.95,
    "minimal_profile": 0.80,
    "unreadable": 0.65,
}


def compute_skill_score(job_profile: dict[str, Any], candidate_profile: dict[str, Any]) -> float:
    matched, missing = _skill_overlap(job_profile, candidate_profile)
    total = len(matched) + len(missing)
    if total == 0:
        return 0.5
    return round(len(matched) / total, 4)


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


def compute_grounded_quality_score(candidate_profile: dict[str, Any]) -> float:
    reliability = float(candidate_profile.get("reliability_score") or 0.0)
    profile_kind = str(candidate_profile.get("profile_kind") or "").lower()
    quality_flags = [str(flag) for flag in (candidate_profile.get("quality_flags") or [])]
    grounded_quality = enrich_grounded_quality(candidate_profile)

    kind_factor = {
        "complete_profile": 1.00,
        "partial_profile": 0.88,
        "minimal_profile": 0.65,
        "unreadable": 0.35,
    }.get(profile_kind, 0.75)
    risk_factor = {
        "low": 1.00,
        "medium": 0.82,
        "high": 0.55,
    }.get(grounded_quality["hallucination_risk"], 0.75)

    nullified_penalty = min(grounded_quality["fields_nullified_count"] * 0.03, 0.18)
    quality_penalty = min(len(quality_flags) * 0.015, 0.10)
    score = (
        reliability * 0.65
        + kind_factor * 0.20
        + risk_factor * 0.15
        - nullified_penalty
        - quality_penalty
    )
    return round(min(max(score, 0.0), 1.0), 4)


def compute_quality_penalty_multiplier(candidate_profile: dict[str, Any]) -> float:
    profile_kind = str(candidate_profile.get("profile_kind") or "").lower()
    grounded_quality = enrich_grounded_quality(candidate_profile)
    risk_multiplier = RISK_MULTIPLIERS.get(grounded_quality["hallucination_risk"], 0.90)
    kind_multiplier = PROFILE_KIND_MULTIPLIERS.get(profile_kind, 0.90)

    bio = candidate_profile.get("bio") or {}
    _, display_name_quality, _ = build_display_name(
        bio.get("full_name") or candidate_profile.get("full_name"),
        candidate_profile.get("candidate_id"),
    )
    risk = grounded_quality["hallucination_risk"]
    if display_name_quality == "weak":
        name_multiplier = 0.80 if risk == "medium" else 0.72 if risk == "high" else 0.85
    else:
        name_multiplier = 1.00
    multiplier = risk_multiplier * kind_multiplier * name_multiplier
    return round(min(max(multiplier, 0.55), 1.0), 4)


def combine_scores(
    score_text_similarity: float,
    score_skills: float,
    score_experience: float,
    score_profile_quality: float,
    score_grounded_quality: float,
) -> float:
    weighted = (
        WEIGHT_SKILLS * score_skills
        + WEIGHT_TEXT_SIMILARITY * score_text_similarity
        + WEIGHT_EXPERIENCE * score_experience
        + WEIGHT_PROFILE_QUALITY * score_profile_quality
    )
    final_score = (weighted * 0.85) + (score_grounded_quality * 0.15)
    return round(min(max(final_score, 0.0), 1.0), 4)


def extract_matched_skills(job_profile: dict[str, Any], candidate_profile: dict[str, Any]) -> list[str]:
    matched, _ = _skill_overlap(job_profile, candidate_profile)
    return matched


def extract_missing_required_skills(job_profile: dict[str, Any], candidate_profile: dict[str, Any]) -> list[str]:
    _, missing = _skill_overlap(job_profile, candidate_profile)
    return missing


def compute_must_have_coverage(
    job_profile: dict[str, Any],
    candidate_profile: dict[str, Any],
) -> float:
    matched, missing = _skill_overlap(job_profile, candidate_profile)
    total = len(matched) + len(missing)
    if total == 0:
        return 1.0
    return round(len(matched) / total, 4)


def apply_must_have_penalty(base_score: float, must_have_coverage: float) -> tuple[float, float, bool]:
    for threshold, multiplier in MUST_HAVE_PENALTY_TIERS:
        if must_have_coverage >= threshold:
            penalty_applied = multiplier < 1.0
            penalized = round(min(max(base_score * multiplier, 0.0), 1.0), 4)
            return penalized, multiplier, penalty_applied
    return round(base_score * 0.45, 4), 0.45, True


def score_candidate(
    job_profile: dict[str, Any],
    candidate_profile: dict[str, Any],
    score_text_similarity: float,
) -> dict[str, Any]:
    score_skills = compute_skill_score(job_profile, candidate_profile)
    score_experience = compute_experience_score(job_profile, candidate_profile)
    score_profile_quality = compute_profile_quality_score(candidate_profile)
    score_grounded_quality = compute_grounded_quality_score(candidate_profile)

    base_score = combine_scores(
        score_text_similarity=score_text_similarity,
        score_skills=score_skills,
        score_experience=score_experience,
        score_profile_quality=score_profile_quality,
        score_grounded_quality=score_grounded_quality,
    )

    must_have_coverage = compute_must_have_coverage(job_profile, candidate_profile)
    score_after_must_have, penalty_multiplier, penalty_applied = apply_must_have_penalty(base_score, must_have_coverage)
    quality_penalty_multiplier = compute_quality_penalty_multiplier(candidate_profile)
    final_score = round(min(max(score_after_must_have * quality_penalty_multiplier, 0.0), 1.0), 4)

    matched_skills = extract_matched_skills(job_profile, candidate_profile)
    missing_skills = extract_missing_required_skills(job_profile, candidate_profile)
    grounded_quality = enrich_grounded_quality(candidate_profile)
    bio = candidate_profile.get("bio") or {}
    _, display_name_quality, name_warning = build_display_name(
        bio.get("full_name") or candidate_profile.get("full_name"),
        candidate_profile.get("candidate_id"),
    )

    return {
        "final_score": final_score,
        "base_score_before_penalty": base_score,
        "score_text_similarity": round(score_text_similarity, 4),
        "score_skills": score_skills,
        "score_experience": score_experience,
        "score_profile_quality": score_profile_quality,
        "score_grounded_quality": score_grounded_quality,
        "must_have_coverage": must_have_coverage,
        "must_have_penalty_multiplier": penalty_multiplier,
        "must_have_penalty_applied": penalty_applied,
        "quality_penalty_multiplier": quality_penalty_multiplier,
        "matched_skills": matched_skills,
        "missing_required_skills": missing_skills,
        "reliability_score": round(float(candidate_profile.get("reliability_score") or 0.0), 4),
        "profile_kind": candidate_profile.get("profile_kind"),
        "hallucination_risk": grounded_quality["hallucination_risk"],
        "quality_flags": list(candidate_profile.get("quality_flags") or []),
        "fields_nullified_count": grounded_quality["fields_nullified_count"],
        "display_name_quality": display_name_quality,
        "name_warning": name_warning,
    }


def _skill_overlap(job_profile: dict[str, Any], candidate_profile: dict[str, Any]) -> tuple[list[str], list[str]]:
    job_skills = normalize_skills(list(job_profile.get("required_skills") or []))
    candidate_skills = _candidate_skills(candidate_profile)
    return skills_overlap(job_skills, candidate_skills)


def _candidate_skills(candidate_profile: dict[str, Any]) -> list[str]:
    expertise = candidate_profile.get("expertise") or {}
    hard_skills = expertise.get("hard_skills") or []
    soft_skills = expertise.get("soft_skills") or []
    return flatten_skill_sources(hard_skills, soft_skills)


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
