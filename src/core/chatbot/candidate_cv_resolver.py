from __future__ import annotations

import json
import mimetypes
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_CV_ROOT = PROJECT_ROOT / "data/raw_cv"
MATCHING_REPORT_DIR = PROJECT_ROOT / "docs/reports/matching/v3"

_SOURCE_KEYS = (
    "source_file",
    "source_path",
    "raw_cv_path",
    "original_file",
    "original_filename",
    "cv_path",
    "file_path",
    "document_path",
)
_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".txt": "text/plain",
}


def resolve_candidate_cv(candidate_id: str, job_id: str | None = None) -> dict[str, Any]:
    candidate_id = str(candidate_id)
    candidates: list[dict[str, Any]] = []

    if job_id:
        job_report = _matching_report_path(job_id)
        candidates.extend(_candidate_cv_candidates(candidate_id, job_report, "matching_report", "high"))

    for report_path in _all_matching_report_paths():
        if job_id and report_path == _matching_report_path(job_id):
            continue
        candidates.extend(
            _candidate_cv_candidates(
                candidate_id,
                report_path,
                "matching_report_alternative",
                "medium",
            )
        )

    deduped = _dedupe_candidates(candidates)
    if not deduped:
        return _not_found(candidate_id)

    selected = deduped[0]
    alternatives = [
        _public_cv_payload(candidate_id, item)
        for item in deduped[1:]
        if item["path"] != selected["path"]
    ]
    result = _public_cv_payload(candidate_id, selected)
    result["cv_alternatives"] = alternatives
    return result


def enrich_candidate_with_cv(candidate: dict, job_id: str | None = None) -> dict:
    enriched = dict(candidate)
    candidate_id = enriched.get("candidate_id")
    if not candidate_id:
        return enriched

    cv = resolve_candidate_cv(str(candidate_id), job_id)
    enriched["cv_available"] = cv["cv_available"]
    enriched["cv_filename"] = cv["cv_filename"]
    enriched["cv_mime_type"] = cv["cv_mime_type"]
    enriched["cv_source"] = cv["cv_source"]
    enriched["cv_confidence"] = cv["cv_confidence"]
    enriched["cv_download_url"] = _download_url(str(candidate_id), job_id) if cv["cv_available"] else None
    return enriched


def enrich_candidates_with_cv(candidates: list[dict], job_id: str | None = None) -> list[dict]:
    return [
        enrich_candidate_with_cv(candidate, job_id)
        for candidate in candidates
        if isinstance(candidate, dict)
    ]


def _candidate_cv_candidates(
    candidate_id: str,
    report_path: Path,
    source: str,
    confidence: str,
) -> list[dict[str, Any]]:
    payload = _read_json(report_path)
    if not isinstance(payload, dict):
        return []

    matches: list[dict[str, Any]] = []
    for item in _walk_dicts(payload):
        if item.get("candidate_id") != candidate_id:
            continue
        for key in _SOURCE_KEYS:
            raw_value = item.get(key)
            if not raw_value:
                continue
            path = _safe_raw_cv_path(raw_value)
            if path:
                matches.append(
                    {
                        "path": path,
                        "source": source,
                        "confidence": confidence,
                        "report_path": report_path,
                        "source_key": key,
                    }
                )
    return matches


def _public_cv_payload(candidate_id: str, item: dict[str, Any]) -> dict[str, Any]:
    path = item["path"]
    return {
        "candidate_id": candidate_id,
        "cv_available": True,
        "cv_filename": path.name,
        "cv_path": _relative_project_path(path),
        "cv_mime_type": _mime_type(path),
        "cv_source": item["source"],
        "cv_confidence": item["confidence"],
        "cv_alternatives": [],
    }


def _not_found(candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "cv_available": False,
        "cv_filename": None,
        "cv_path": None,
        "cv_mime_type": None,
        "cv_source": "not_found",
        "cv_confidence": "none",
        "cv_alternatives": [],
    }


def _download_url(candidate_id: str, job_id: str | None) -> str:
    url = f"/api/candidates/{quote(candidate_id)}/cv"
    if job_id:
        url += f"?job_id={quote(str(job_id))}"
    return url


def _matching_report_path(job_id: str) -> Path:
    safe_job_id = str(job_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", safe_job_id):
        return MATCHING_REPORT_DIR / "__invalid_job_id__"
    return MATCHING_REPORT_DIR / f"{safe_job_id}_matching_report_v3_normalized.json"


def _all_matching_report_paths() -> list[Path]:
    if not MATCHING_REPORT_DIR.exists():
        return []
    return sorted(path for path in MATCHING_REPORT_DIR.glob("*matching_report*.json") if path.is_file())


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate["path"]
        if path in seen:
            continue
        seen.add(path)
        deduped.append(candidate)
    return deduped


def _safe_raw_cv_path(raw_value: Any) -> Path | None:
    paths = _candidate_paths_from_value(raw_value)
    for path in paths:
        try:
            resolved = path.resolve(strict=True)
            raw_root = RAW_CV_ROOT.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not resolved.is_file() or not resolved.is_relative_to(raw_root):
            continue
        return resolved
    return None


def _candidate_paths_from_value(raw_value: Any) -> list[Path]:
    raw_text = str(raw_value).strip()
    if not raw_text:
        return []
    path = Path(raw_text.replace("\\", "/"))
    if path.is_absolute():
        return [path]
    if path.parent == Path("."):
        return sorted(RAW_CV_ROOT.rglob(path.name))
    return [PROJECT_ROOT / path]


def _relative_project_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _MIME_TYPES:
        return _MIME_TYPES[suffix]
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


@lru_cache(maxsize=64)
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
