from __future__ import annotations

from typing import Any


def build_required_skills_text(job_profile: dict[str, Any]) -> str:
    skills = _normalize_string_list(job_profile.get("required_skills"))
    if not skills:
        return ""
    return ", ".join(skills)


def build_responsibilities_text(job_profile: dict[str, Any]) -> str:
    responsibilities = _normalize_string_list(job_profile.get("responsibilities"))
    if not responsibilities:
        return ""
    return "\n".join(responsibilities)


def build_job_text(job_profile: dict[str, Any]) -> str:
    """Build an embedding-ready job text emphasizing required skills and responsibilities."""
    job_title = _clean_text(job_profile.get("job_title"))
    seniority_level = _clean_text(job_profile.get("seniority_level"))
    years_experience_required = job_profile.get("years_experience_required")
    required_skills = build_required_skills_text(job_profile)
    nice_to_have_skills = ", ".join(_normalize_string_list(job_profile.get("nice_to_have_skills")))
    responsibilities = build_responsibilities_text(job_profile)
    domain = _clean_text(job_profile.get("domain"))
    location = _clean_text(job_profile.get("location"))
    languages = ", ".join(_normalize_string_list(job_profile.get("language_requirements")))

    chunks = [
        f"Job Title: {job_title}" if job_title else None,
        f"Seniority Level: {seniority_level}" if seniority_level else None,
        (
            f"Years Experience Required: {years_experience_required}"
            if isinstance(years_experience_required, (int, float))
            else None
        ),
        f"Required Skills: {required_skills}" if required_skills else None,
        f"Required Skills Repeated: {required_skills}" if required_skills else None,
        f"Nice To Have Skills: {nice_to_have_skills}" if nice_to_have_skills else None,
        f"Responsibilities:\n{responsibilities}" if responsibilities else None,
        f"Domain: {domain}" if domain else None,
        f"Location: {location}" if location else None,
        f"Languages: {languages}" if languages else None,
    ]
    return "\n\n".join(chunk for chunk in chunks if chunk)


def _normalize_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
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


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None
