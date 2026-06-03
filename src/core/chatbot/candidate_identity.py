from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DECISION_CARD_DIRS = (
    PROJECT_ROOT / "docs/reports/decision_cards",
    PROJECT_ROOT / "docs/reports/matching/v3",
)
MATCHING_REPORT_DIR = PROJECT_ROOT / "docs/reports/matching/v3"
GROUNDED_PROFILES_DIR = (
    PROJECT_ROOT / "data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles"
)

_NAME_KEYS = ("candidate_name", "full_name", "name")
_NESTED_KEYS = ("candidate", "profile", "bio", "data", "header_info", "normalization")
_INVALID_EXACT_NAMES = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "non disponible",
    "not available",
    "unavailable",
    "anonymous",
    "anonyme",
    "anonymized",
    "placeholder",
}
_INVALID_NAME_MARKERS = (
    "candidate (id:",
    "candidate_id",
    "employment history",
    "professional experience",
    "work experience",
    "technical skills",
    "education",
)
_SOURCE_PRIORITY = {"decision_card": 0, "grounded_profile": 1, "filename": 2}


def resolve_candidate_display_name(candidate_id: str) -> dict[str, Any]:
    identity = _candidate_identity_index().get(str(candidate_id))
    if identity:
        return dict(identity)
    return {
        "candidate_id": str(candidate_id),
        "candidate_name": None,
        "display_name_source": "anonymized_artifact",
        "display_name_confidence": "none",
    }


def enrich_candidate_with_display_name(candidate: dict) -> dict:
    enriched = dict(candidate)
    candidate_id = enriched.get("candidate_id")
    if not candidate_id:
        return enriched

    identity = resolve_candidate_display_name(str(candidate_id))
    if not identity["candidate_name"]:
        existing_name = _extract_valid_name(enriched)
        if existing_name:
            enriched["candidate_name"] = existing_name
            enriched["display_name_source"] = enriched.get("display_name_source") or "candidate_payload"
            enriched["display_name_confidence"] = enriched.get("display_name_confidence") or "high"
            return enriched

    enriched["candidate_name"] = identity["candidate_name"]
    enriched["display_name_source"] = identity["display_name_source"]
    enriched["display_name_confidence"] = identity["display_name_confidence"]
    return enriched


def enrich_candidates_with_display_names(candidates: list[dict]) -> list[dict]:
    return [
        enrich_candidate_with_display_name(candidate)
        for candidate in candidates
        if isinstance(candidate, dict)
    ]


@lru_cache(maxsize=1)
def _candidate_identity_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    profile_paths_by_candidate = _collect_profile_paths_by_candidate()

    for path in _decision_card_paths():
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        for item in _walk_dicts(payload):
            candidate_id = item.get("candidate_id")
            if not candidate_id:
                continue
            name = _extract_valid_name(item)
            if name:
                _put_identity(index, str(candidate_id), name, "decision_card", "high")

            transferability = item.get("transferability")
            if isinstance(transferability, dict):
                profile_path = _project_path(transferability.get("profile_path"))
                if profile_path:
                    profile_paths_by_candidate.setdefault(str(candidate_id), []).append(profile_path)

    for candidate_id, profile_paths in profile_paths_by_candidate.items():
        for profile_path in profile_paths:
            profile = _read_json(profile_path)
            if not isinstance(profile, dict):
                continue
            name = _extract_valid_name(profile)
            if name:
                _put_identity(index, candidate_id, name, "grounded_profile", "high")
                break

    for candidate_id, profile_paths in profile_paths_by_candidate.items():
        if candidate_id in index:
            continue
        for profile_path in profile_paths:
            name = _name_from_filename(profile_path)
            if name:
                _put_identity(index, candidate_id, name, "filename", "medium")
                break

    return index


def _collect_profile_paths_by_candidate() -> dict[str, list[Path]]:
    profile_paths: dict[str, list[Path]] = {}
    if not MATCHING_REPORT_DIR.exists():
        return profile_paths

    for path in sorted(MATCHING_REPORT_DIR.glob("*matching_report*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        for item in _walk_dicts(payload):
            candidate_id = item.get("candidate_id")
            source_path = item.get("source_path")
            if not candidate_id or not source_path:
                continue
            profile_path = _profile_path_from_source(str(source_path))
            if profile_path:
                profile_paths.setdefault(str(candidate_id), []).append(profile_path)
    return profile_paths


def _decision_card_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in DECISION_CARD_DIRS:
        if not directory.exists():
            continue
        pattern = "decision_cards*.json" if directory.name == "v3" else "*.json"
        paths.extend(path for path in sorted(directory.glob(pattern)) if path.is_file())
    return paths


def _put_identity(
    index: dict[str, dict[str, Any]],
    candidate_id: str,
    name: str,
    source: str,
    confidence: str,
) -> None:
    current = index.get(candidate_id)
    if current and _SOURCE_PRIORITY[current["display_name_source"]] <= _SOURCE_PRIORITY[source]:
        return
    index[candidate_id] = {
        "candidate_id": candidate_id,
        "candidate_name": name,
        "display_name_source": source,
        "display_name_confidence": confidence,
    }


def _extract_valid_name(payload: dict[str, Any]) -> str | None:
    for key in _NAME_KEYS:
        name = _clean_name(payload.get(key))
        if name:
            return name
    for key in _NESTED_KEYS:
        nested = payload.get(key)
        if isinstance(nested, dict):
            name = _extract_valid_name(nested)
            if name:
                return name
    return None


def _clean_name(value: Any) -> str | None:
    if value is None:
        return None
    name = re.sub(r"\s+", " ", str(value)).strip()
    lowered = name.lower()
    if lowered in _INVALID_EXACT_NAMES:
        return None
    if any(marker in lowered for marker in _INVALID_NAME_MARKERS):
        return None
    if re.fullmatch(r"candidate_[A-Za-z0-9_]+", name):
        return None
    if "@" in name or "/" in name or "\\" in name:
        return None
    if not re.search(r"[A-Za-zÀ-ÿ]", name):
        return None
    if "," in name and any(token in lowered for token in ("tunisia", "ariana", "palo alto")):
        return None
    return name


def _profile_path_from_source(source_path: str) -> Path | None:
    source = Path(source_path.replace("\\", "/"))
    if not source.stem or not source.parent.name:
        return None
    prefix_by_dir = {"pdf": "pdf", "images": "images", "docx": "docx"}
    prefix = prefix_by_dir.get(source.parent.name)
    if not prefix:
        return None
    return GROUNDED_PROFILES_DIR / f"{prefix}_{source.stem}.json"


def _name_from_filename(path: Path) -> str | None:
    stem = path.stem
    for prefix in ("pdf_", "images_", "docx_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    if re.fullmatch(r"(CV|Image|candidate|resume)?[_ -]?\d+", stem, flags=re.IGNORECASE):
        return None
    name = stem.replace("_", " ").replace("-", " ").strip()
    return _clean_name(name)


def _project_path(raw_path: Any) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_dicts(item)
