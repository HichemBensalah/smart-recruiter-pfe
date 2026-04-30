from __future__ import annotations

from typing import Any


def normalize_skills_for_text(skills: list[Any] | None) -> list[str]:
    """Return unique, cleaned skill labels while preserving order."""
    if not isinstance(skills, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in skills:
        if not isinstance(item, str):
            continue
        cleaned = " ".join(item.split()).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def flatten_experiences(experiences: list[Any] | None) -> str:
    """Flatten experience entries into rich text for embedding."""
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
            _clean_text(item) for item in responsibilities if _clean_text(item)
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
    """Flatten education entries into short descriptive lines."""
    if not isinstance(education, list):
        return ""
    chunks: list[str] = []
    for item in education:
        if not isinstance(item, dict):
            continue
        degree = _clean_text(item.get("degree"))
        school = _clean_text(item.get("school"))
        year = _clean_text(item.get("year"))
        line = " | ".join(part for part in [degree, school, year] if part)
        if line:
            chunks.append(line)
    return "\n".join(chunks)


def build_candidate_text(profile_doc: dict[str, Any]) -> str:
    """Build a weighted candidate text emphasizing skills, summary and experience."""
    bio = profile_doc.get("bio") or {}
    expertise = profile_doc.get("expertise") or {}
    full_name = _clean_text(bio.get("full_name")) or _clean_text(profile_doc.get("full_name"))
    location = _clean_text(bio.get("location")) or _clean_text(profile_doc.get("location"))
    summary = _clean_text(expertise.get("summary"))
    hard_skills = normalize_skills_for_text(expertise.get("hard_skills"))
    soft_skills = normalize_skills_for_text(expertise.get("soft_skills"))
    experiences_text = flatten_experiences(profile_doc.get("experiences"))
    education_text = flatten_education(profile_doc.get("education"))
    profile_kind = _clean_text(profile_doc.get("profile_kind"))
    provider_route = _clean_text(profile_doc.get("provider_route"))

    chunks = [
        f"Candidate Name: {full_name}" if full_name else None,
        f"Profile Kind: {profile_kind}" if profile_kind else None,
        f"Location: {location}" if location else None,
        f"Professional Summary: {summary}" if summary else None,
        f"Core Hard Skills: {', '.join(hard_skills)}" if hard_skills else None,
        f"Core Hard Skills Repeated: {', '.join(hard_skills)}" if hard_skills else None,
        f"Soft Skills: {', '.join(soft_skills)}" if soft_skills else None,
        f"Experience Highlights:\n{experiences_text}" if experiences_text else None,
        f"Experience Highlights Repeated:\n{experiences_text}" if experiences_text else None,
        f"Education:\n{education_text}" if education_text else None,
        f"Provider Route: {provider_route}" if provider_route else None,
    ]
    return "\n\n".join(chunk for chunk in chunks if chunk)


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None
