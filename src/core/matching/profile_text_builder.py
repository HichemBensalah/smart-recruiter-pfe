from __future__ import annotations

from typing import Any

from .matching_quality_filters import build_display_name, enrich_grounded_quality
from .skill_normalizer import normalize_skills


def normalize_skills_for_text(skills: list[Any] | None) -> list[str]:
    raw = [item for item in (skills or []) if isinstance(item, str)]
    return normalize_skills(raw)


def flatten_experiences(experiences: list[Any] | None) -> str:
    if not isinstance(experiences, list):
        return ""
    chunks: list[str] = []
    for experience in experiences:
        if not isinstance(experience, dict):
            continue
        title = _clean_text(experience.get("job_title"))
        company = _clean_text(experience.get("company"))
        city = _clean_text(experience.get("city"))
        start_date = _clean_text(experience.get("start_date"))
        end_date = _clean_text(experience.get("end_date"))
        responsibilities = experience.get("responsibilities") or []
        responsibility_text = "; ".join(
            cleaned for item in responsibilities if (cleaned := _clean_text(item))
        )
        parts = [
            title,
            company,
            f"{start_date} -> {end_date}".strip(" ->") if start_date or end_date else None,
            city,
            responsibility_text,
        ]
        line = " | ".join(part for part in parts if part)
        if line:
            chunks.append(line)
    return "\n".join(chunks)


def flatten_education(education: list[Any] | None) -> str:
    if not isinstance(education, list):
        return ""
    chunks: list[str] = []
    for item in education:
        if not isinstance(item, dict):
            continue
        degree = _clean_text(item.get("degree"))
        school = _clean_text(item.get("school") or item.get("institution"))
        field = _clean_text(item.get("field"))
        year = _clean_text(item.get("year"))
        line = " | ".join(part for part in [degree, field, school, year] if part)
        if line:
            chunks.append(line)
    return "\n".join(chunks)


def build_candidate_text_grounded(profile_doc: dict[str, Any]) -> str:
    bio = profile_doc.get("bio") or {}
    expertise = profile_doc.get("expertise") or {}
    quality_flags = {str(flag).lower() for flag in (profile_doc.get("quality_flags") or [])}
    grounded_quality = enrich_grounded_quality(profile_doc)
    display_name, _, _ = build_display_name(
        bio.get("full_name") or profile_doc.get("full_name"),
        profile_doc.get("candidate_id"),
    )
    location = _clean_text(bio.get("location")) or _clean_text(profile_doc.get("location"))
    summary = _clean_text(expertise.get("summary"))
    if "summary_not_enough_evidence" in quality_flags:
        summary = None

    hard_skills = normalize_skills_for_text(expertise.get("hard_skills"))
    soft_skills = normalize_skills_for_text(expertise.get("soft_skills"))
    experiences_text = flatten_experiences(profile_doc.get("experiences"))
    education_text = flatten_education(profile_doc.get("education"))
    certifications = normalize_skills_for_text(profile_doc.get("certifications"))
    languages = normalize_skills_for_text(profile_doc.get("languages"))
    profile_kind = _clean_text(profile_doc.get("profile_kind"))
    reliability_score = float(profile_doc.get("reliability_score") or 0.0)
    hallucination_risk = grounded_quality["hallucination_risk"]

    reliability_label = "high" if reliability_score >= 0.85 else "medium" if reliability_score >= 0.65 else "low"
    chunks = [
        f"Candidate: {display_name}",
        f"Profile Kind: {profile_kind}" if profile_kind else None,
        f"Location: {location}" if location else None,
        f"Summary: {summary}" if summary else None,
        f"Verified Hard Skills: {', '.join(hard_skills)}" if hard_skills else None,
        f"Supporting Soft Skills: {', '.join(soft_skills)}" if soft_skills else None,
        f"Experience:\n{experiences_text}" if experiences_text else None,
        f"Education:\n{education_text}" if education_text else None,
        f"Certifications: {', '.join(certifications)}" if certifications else None,
        f"Languages: {', '.join(languages)}" if languages else None,
        f"Reliability Signal: {reliability_label}; hallucination risk {hallucination_risk}",
    ]
    return "\n\n".join(chunk for chunk in chunks if chunk)


def build_candidate_text(profile_doc: dict[str, Any]) -> str:
    return build_candidate_text_grounded(profile_doc)


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None
