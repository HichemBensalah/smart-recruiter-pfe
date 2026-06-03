"""
Generated job profile — creates and persists a structured job profile
produced by the Job Intake Wizard so that LiveMatcher can use it directly.

Keys align with both score_candidate (years_experience_required) and
build_job_text (seniority_level, required_skills, responsibilities, ...).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


GENERATED_PROFILES_DIR = Path("data/job_profiles/generated")


def build_generated_job_profile(
    structured_profile: dict[str, Any],
    session_id: str | None = None,
    routed_base_job_id: str | None = None,
) -> dict[str, Any]:
    """
    Convert a structured_job_profile (from job_intake.build_structured_job_profile)
    into a generated_job_profile ready for LiveMatcher.

    Key mappings:
      min_years_experience -> years_experience_required  (for score_candidate)
      seniority            -> seniority_level            (for build_job_text)
    """
    generated_job_id = f"generated_{uuid.uuid4().hex[:12]}"
    min_years = structured_profile.get("min_years_experience")
    return {
        "generated_job_id": generated_job_id,
        "job_id": generated_job_id,
        "job_title": structured_profile.get("job_title") or "",
        "target_role": structured_profile.get("target_role") or "",
        "about_role": structured_profile.get("about_role") or "",
        "responsibilities": _ensure_list(structured_profile.get("responsibilities")),
        "required_skills": _ensure_list(structured_profile.get("required_skills")),
        "nice_to_have_skills": _ensure_list(structured_profile.get("nice_to_have_skills")),
        "min_years_experience": min_years,
        "years_experience_required": min_years,
        "seniority": structured_profile.get("seniority"),
        "seniority_level": structured_profile.get("seniority"),
        "location": structured_profile.get("location"),
        "work_model": structured_profile.get("work_model"),
        "language_requirements": _ensure_list(structured_profile.get("language_requirements")),
        "routed_base_job_id": routed_base_job_id,
        "source": "job_intake_wizard",
        "created_at": datetime.utcnow().isoformat(),
        "session_id": session_id,
    }


def save_generated_job_profile(
    profile: dict[str, Any],
    base_dir: Path | None = None,
) -> Path:
    """Persist *profile* as JSON. Returns the written path."""
    directory = base_dir or GENERATED_PROFILES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    job_id = profile.get("generated_job_id") or f"generated_{uuid.uuid4().hex[:12]}"
    path = directory / f"{job_id}.json"
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_generated_job_profile(
    generated_job_id: str,
    base_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Load a previously saved generated profile. Returns None if not found."""
    directory = base_dir or GENERATED_PROFILES_DIR
    path = directory / f"{generated_job_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ensure_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [str(value)]
