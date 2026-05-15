from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SUSPECT_NAME_PHRASES = {
    "resume objective",
    "objective",
    "professional summary",
    "data scientist",
    "engineering manager",
    "multi skilled engineering manager",
    "from resume genius",
    "o ariana tunisia",
}

ROLE_HINTS = {
    "engineer",
    "scientist",
    "manager",
    "consultant",
    "developer",
    "director",
    "intern",
    "analyst",
    "objective",
    "summary",
    "resume",
}


def clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def normalize_name_key(value: Any) -> str:
    text = clean_text(value) or ""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_suspect_name(value: Any) -> bool:
    cleaned = clean_text(value)
    if not cleaned:
        return True
    key = normalize_name_key(cleaned)
    if not key:
        return True
    if key in SUSPECT_NAME_PHRASES:
        return True
    words = key.split()
    if len(words) == 1 and words[0] in ROLE_HINTS:
        return True
    if any(word in ROLE_HINTS for word in words) and len(words) <= 3 and cleaned.upper() == cleaned:
        return True
    if any(word in ROLE_HINTS for word in words) and len(words) <= 3 and cleaned.istitle():
        return True
    if "resume genius" in key or "objective" in key or "summary" in key:
        return True
    return False


def build_display_name(full_name: Any, candidate_id: Any) -> tuple[str, str, str | None]:
    cleaned = clean_text(full_name)
    if cleaned and not is_suspect_name(cleaned):
        return cleaned, "ok", None
    if candidate_id:
        return f"Candidate (ID: {candidate_id})", "weak", "missing_or_rejected_full_name"
    return "Candidate", "weak", "missing_or_rejected_full_name"


def enrich_grounded_quality(profile: dict[str, Any]) -> dict[str, Any]:
    module2_file_path = clean_text(profile.get("module2_file_path"))
    grounded_file = Path(module2_file_path) if module2_file_path else None

    hallucination_risk = profile.get("hallucination_risk")
    fields_nullified = profile.get("fields_nullified")
    fields_nullified_count = profile.get("fields_nullified_count")
    if grounded_file and grounded_file.exists():
        try:
            grounded_payload = json.loads(grounded_file.read_text(encoding="utf-8"))
            grounding = grounded_payload.get("grounding") or {}
            if hallucination_risk is None:
                hallucination_risk = grounding.get("hallucination_risk")
            if fields_nullified is None:
                fields_nullified = grounding.get("fields_nullified") or []
            if fields_nullified_count is None:
                fields_nullified_count = len(grounding.get("fields_nullified") or [])
        except Exception:
            pass

    quality_flags = list(profile.get("quality_flags") or [])
    derived_risk = derive_hallucination_risk(
        reliability_score=float(profile.get("reliability_score") or 0.0),
        quality_flags=quality_flags,
        profile_kind=str(profile.get("profile_kind") or ""),
        explicit_risk=clean_text(hallucination_risk),
    )

    return {
        "hallucination_risk": derived_risk,
        "fields_nullified": list(fields_nullified or []),
        "fields_nullified_count": int(fields_nullified_count or 0),
    }


def derive_hallucination_risk(
    *,
    reliability_score: float,
    quality_flags: list[str],
    profile_kind: str,
    explicit_risk: str | None = None,
) -> str:
    if explicit_risk in {"low", "medium", "high"}:
        return explicit_risk

    lowered_flags = {str(flag).lower() for flag in quality_flags}
    profile_kind = str(profile_kind or "").lower()
    if "unsupported_fields_nullified" in lowered_flags or "template_detected" in lowered_flags:
        return "medium"
    if "high_ocr_noise" in lowered_flags and reliability_score < 0.85:
        return "medium"
    if profile_kind == "minimal_profile":
        return "high"
    if reliability_score < 0.55:
        return "high"
    if reliability_score < 0.8:
        return "medium"
    return "low"
