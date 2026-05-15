from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.matching.faiss_indexer import run_faiss_indexer
from src.core.storage.import_profiles_to_mongodb import (
    clean_display_text,
    extract_schools,
    extract_urls,
    is_generic_email,
    is_real_phone,
    is_valid_name,
    normalize_email,
    normalize_phone,
    normalize_text,
    stable_id,
)
from src.core.structuring.grounding_validator import infer_experience_level, validate_and_ground
from src.core.structuring.markdown_normalizer import fix_emails_and_urls, normalize_markdown


DEFAULT_PROFILES_DIR = Path("data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles")
DEFAULT_REPORT_PATH = Path("data/patch_report_v2_fixes.json")
DEFAULT_DRY_RUN_REPORT_PATH = Path("data/patch_report_v2_fixes_dry_run.json")
DEFAULT_DRY_RUN_MARKDOWN_PATH = Path("data/patch_report_v2_fixes_dry_run.md")
DEFAULT_SAFE_URL_DRY_RUN_REPORT_PATH = Path("data/patch_report_v2_safe_url_fixes_dry_run.json")
DEFAULT_SAFE_URL_DRY_RUN_MARKDOWN_PATH = Path("data/patch_report_v2_safe_url_fixes_dry_run.md")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_profile_id(module2_root: Path, profile_path: Path, payload: dict[str, Any]) -> str:
    return stable_id(
        "profile",
        [
            payload.get("source_path"),
            payload.get("artifact_path"),
            profile_path.relative_to(module2_root),
        ],
    )


def _clean_contact_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = fix_emails_and_urls(value).strip()
    return cleaned or None


def _extract_best_email(raw_text: str) -> str | None:
    pattern = re_compile_cached(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    candidates = [normalize_email(match.group(0)) for match in pattern.finditer(fix_emails_and_urls(raw_text or ""))]
    candidates = [candidate for candidate in candidates if candidate]
    if not candidates:
        return None
    return max(candidates, key=len)


def _select_email(profile_email: Any, header_email: Any, raw_text: str) -> str | None:
    candidates = [
        _clean_contact_value(profile_email),
        _clean_contact_value(header_email),
        _extract_best_email(raw_text),
    ]
    valid_emails: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        email = normalize_email(candidate)
        if email and email not in valid_emails:
            valid_emails.append(email)
    if valid_emails:
        return max(valid_emails, key=len)
    return None


def _extract_best_url(raw_text: str, domain: str) -> str | None:
    normalized_text = fix_emails_and_urls(raw_text or "")
    compact_domain = domain.replace(".", "").lower()
    candidates: list[str] = []

    for line in (raw_text or "").splitlines():
        compact_line = line.lower().replace(".", "").replace(" ", "")
        if compact_domain not in compact_line:
            continue
        line_candidates = _extract_urls_from_line(line, domain)
        for candidate in line_candidates:
            if candidate not in candidates:
                candidates.append(candidate)

    patterns = [
        re_compile_cached(rf"\((https?://(?:www\.)?{re_escape_cached(domain)}/[^)\s]+)\)"),
        re_compile_cached(rf"\[((?:https?://)?(?:www\.)?{re_escape_cached(domain)}/[^\]\s]+)\]"),
        re_compile_cached(rf"(?:(?:https?://)?(?:www\.)?{re_escape_cached(domain)}/[^\s,;\])]+)"),
        re_compile_cached(rf"(?:(?:https?://)?(?:www\.)?{re_escape_cached(domain)}/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=% -]+)"),
        re_compile_cached(rf"[A-Za-z]?\s*{domain.split('.')[0]}\s*\.\s*{domain.split('.')[1]}\s*/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=% -]+"),
    ]
    for pattern in patterns:
        for match in pattern.finditer(normalized_text):
            value = next((group for group in match.groups() if group), match.group(0))
            cleaned = _sanitize_url_candidate(value, domain)
            if compact_domain not in cleaned.lower().replace(".", ""):
                continue
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)
    if not candidates:
        return None
    return max(candidates, key=len)


def _extract_urls_from_line(line: str, domain: str) -> list[str]:
    normalized_line = fix_emails_and_urls(line)
    if domain == "linkedin.com":
        patterns = [
            re_compile_cached(r"https?://(?:www\.)?linkedin\.com/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=% -]+"),
            re_compile_cached(r"(?:www\.)?linkedin\.com/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=% -]+"),
        ]
    else:
        patterns = [
            re_compile_cached(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9._ -]+"),
            re_compile_cached(r"[A-Za-z]?github\s*\.\s*com/[A-Za-z0-9._ -]+"),
        ]

    results: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(normalized_line):
            cleaned = _sanitize_url_candidate(match.group(0), domain)
            if cleaned and cleaned not in results:
                results.append(cleaned)
    return results


def _sanitize_url_candidate(value: str, domain: str) -> str:
    cleaned = fix_emails_and_urls(value).strip("[]()")
    nested_urls = re_compile_cached(r"(https?://[^\s)\]]+|(?:www\.)?[A-Za-z0-9-]+\.[A-Za-z]{2,}/[^\s)\]]+)").findall(cleaned)
    if nested_urls:
        domain_matches = [item for item in nested_urls if domain in item.lower()]
        if domain_matches:
            cleaned = max(domain_matches, key=len)
    cleaned = cleaned.lstrip("Qq")
    lower_cleaned = cleaned.lower()
    for other_domain in ("linkedin.com", "github.com"):
        if other_domain == domain:
            continue
        marker = lower_cleaned.find(other_domain)
        q_marker = lower_cleaned.find(f"q{other_domain}")
        if q_marker > 0 and (marker == -1 or q_marker < marker):
            marker = q_marker
        if marker > 0:
            cleaned = cleaned[:marker]
            lower_cleaned = cleaned.lower()
    cleaned = cleaned.replace(" ", "")
    if domain == "linkedin.com":
        cleaned = cleaned.rstrip("/")
        if "/in/" in cleaned and not cleaned.endswith("/"):
            cleaned += "/"
    return cleaned


_REGEX_CACHE: dict[str, Any] = {}
_ESCAPE_CACHE: dict[str, str] = {}


def re_compile_cached(pattern: str) -> Any:
    compiled = _REGEX_CACHE.get(pattern)
    if compiled is None:
        import re

        compiled = re.compile(pattern, re.I)
        _REGEX_CACHE[pattern] = compiled
    return compiled


def re_escape_cached(value: str) -> str:
    escaped = _ESCAPE_CACHE.get(value)
    if escaped is None:
        import re

        escaped = re.escape(value)
        _ESCAPE_CACHE[value] = escaped
    return escaped


def _select_url(profile_value: Any, header_value: Any, raw_text: str, domain: str) -> str | None:
    candidates = [
        _clean_contact_value(profile_value),
        _clean_contact_value(header_value),
        _extract_best_url(raw_text, domain),
    ]
    valid_urls: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        cleaned = fix_emails_and_urls(candidate)
        if domain in cleaned.lower() and cleaned not in valid_urls:
            valid_urls.append(cleaned)
    if not valid_urls:
        return None
    return max(valid_urls, key=len)


def _normalize_url_identity(value: Any, domain: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = _sanitize_url_candidate(value, domain)
    if domain == "linkedin.com":
        pattern = re_compile_cached(r"(?:https?://)?(?:www\.)?linkedin\.com/[a-z0-9._~/%-]+/?")
    else:
        pattern = re_compile_cached(r"(?:https?://)?(?:www\.)?github\.com/[a-z0-9._-]+/?")
    match = pattern.search(cleaned.lower())
    if not match:
        return None
    lowered = match.group(0)
    if lowered.startswith("https://"):
        lowered = lowered[len("https://") :]
    elif lowered.startswith("http://"):
        lowered = lowered[len("http://") :]
    if lowered.startswith("www."):
        lowered = lowered[len("www.") :]
    lowered = lowered.rstrip("/")
    return lowered or None


def _profile_kind_rank(value: Any) -> int:
    rank = {
        "complete_profile": 3,
        "partial_profile": 2,
        "skeletal_profile": 1,
    }
    return rank.get(str(value or "").strip().lower(), 0)


def _quality_regression_flags(original_payload: dict[str, Any], updated_payload: dict[str, Any]) -> dict[str, bool]:
    original_score = float((original_payload.get("grounding") or {}).get("reliability_score") or 0.0)
    updated_score = float((updated_payload.get("grounding") or {}).get("reliability_score") or 0.0)
    original_risk = str((original_payload.get("grounding") or {}).get("hallucination_risk") or "").strip().lower()
    updated_risk = str((updated_payload.get("grounding") or {}).get("hallucination_risk") or "").strip().lower()
    risk_rank = {"low": 1, "medium": 2, "high": 3}
    reliability_regressed = updated_score < original_score
    risk_regressed = risk_rank.get(updated_risk, 0) > risk_rank.get(original_risk, 0)
    kind_regressed = _profile_kind_rank(updated_payload.get("profile_kind")) < _profile_kind_rank(original_payload.get("profile_kind"))
    return {
        "reliability_regressed": reliability_regressed,
        "risk_regressed": risk_regressed,
        "profile_kind_regressed": kind_regressed,
        "has_regression": reliability_regressed or risk_regressed or kind_regressed,
    }


def patch_profile_payload(
    profile_path: Path,
    profiles_dir: Path,
    *,
    apply_changes: bool,
    only_safe_url_fixes: bool,
) -> dict[str, Any]:
    payload = read_json(profile_path)
    original_payload = copy.deepcopy(payload)
    artifact_path = ROOT_DIR / Path(str(payload["artifact_path"]))
    artifact = read_json(artifact_path)
    raw_text = artifact.get("raw_text") or ""
    profile = payload.setdefault("profile", {})
    bio = profile.setdefault("bio", {})

    previous_risk = ((payload.get("grounding") or {}).get("hallucination_risk") or "").strip()

    previous_email = (bio.get("email") or "").strip()
    previous_years = (profile.get("expertise") or {}).get("years_experience")
    previous_level = (profile.get("expertise") or {}).get("experience_level")
    url_changes: list[dict[str, Any]] = []

    if only_safe_url_fixes:
        for field_name, domain in (("linkedin", "linkedin.com"), ("github", "github.com")):
            current_value = bio.get(field_name)
            selected_value = _select_url(current_value, None, raw_text, domain)
            if not selected_value or not current_value:
                continue
            if _normalize_url_identity(current_value, domain) != _normalize_url_identity(selected_value, domain):
                continue
            if _canonical(current_value) == _canonical(selected_value):
                continue
            if _url_quality(selected_value, domain) <= _url_quality(current_value, domain):
                continue
            bio[field_name] = selected_value
            url_changes.append({"field": f"profile.bio.{field_name}", "before": current_value, "after": selected_value})
    else:
        combined_source = "\n".join(part for part in [artifact.get("markdown") or "", artifact.get("raw_text") or ""] if part)
        normalization = normalize_markdown(combined_source, artifact.get("raw_text") or "")
        grounded_input = json.loads(json.dumps(profile))
        grounded_bio = grounded_input.setdefault("bio", {})
        header_info = normalization.get("header_info") or {}

        selected_email = _select_email(grounded_bio.get("email"), header_info.get("email"), raw_text)
        if selected_email:
            grounded_bio["email"] = selected_email

        selected_linkedin = _select_url(grounded_bio.get("linkedin"), header_info.get("linkedin"), raw_text, "linkedin.com")
        if selected_linkedin:
            grounded_bio["linkedin"] = selected_linkedin

        selected_github = _select_url(grounded_bio.get("github"), header_info.get("github"), raw_text, "github.com")
        if selected_github:
            grounded_bio["github"] = selected_github

        grounded_input["status"] = payload.get("status") or grounded_input.get("status") or "success"
        grounded_result = validate_and_ground(
            grounded_input,
            normalization["cleaned_markdown"],
            normalization["detected_templates"],
            float(payload.get("document_confidence_score") or 0.0),
        )

        payload["normalization"] = normalization
        payload["grounding"] = {
            "reliability_score": grounded_result["reliability_score"],
            "hallucination_risk": grounded_result["hallucination_risk"],
            "quality_flags": grounded_result["quality_flags"],
            "fields_nullified": grounded_result["fields_nullified"],
            "fields_supported": grounded_result["fields_supported"],
            "fields_unsupported": grounded_result["fields_unsupported"],
        }
        payload["profile_kind"] = grounded_result["profile_kind"]
        payload["profile"] = grounded_result["grounded_profile"]

    quality_regression = _quality_regression_flags(original_payload, payload)
    skipped_due_to_quality_regression = False
    if quality_regression["has_regression"]:
        payload = copy.deepcopy(original_payload)
        skipped_due_to_quality_regression = True

    payload.setdefault("metadata", {})
    payload["metadata"]["patched_at"] = utc_now()
    payload["metadata"]["patch_script"] = "patch_grounded_profiles_v2.py"

    profile = payload["profile"]
    expertise = profile.get("expertise") or {}
    new_email = (profile.get("bio") or {}).get("email")
    new_years = expertise.get("years_experience")
    new_level = expertise.get("experience_level")
    new_risk = payload["grounding"]["hallucination_risk"]
    applied_url_changes = []
    if not skipped_due_to_quality_regression:
        for field in ("profile.bio.linkedin", "profile.bio.github"):
            before = _get_nested(original_payload, field)
            after = _get_nested(payload, field)
            if _canonical(before) != _canonical(after):
                applied_url_changes.append({"field": field, "before": before, "after": after})

    return {
        "original_payload": original_payload,
        "payload": payload,
        "profile_path": profile_path,
        "profile_id": build_profile_id(profiles_dir.parents[1], profile_path, payload),
        "email_fixed": bool(not only_safe_url_fixes and new_email and new_email != previous_email),
        "years_recovered": previous_years in (None, "", [], {}) and new_years not in (None, "", [], {}),
        "level_recovered": previous_level in (None, "", [], {}) and new_level not in (None, "", [], {}),
        "previous_risk": previous_risk,
        "new_risk": new_risk,
        "quality_regression": quality_regression,
        "skipped_due_to_quality_regression": skipped_due_to_quality_regression,
        "applied_url_changes": applied_url_changes,
        "proposed_url_changes": url_changes,
    }


def _url_quality(value: Any, domain: str) -> tuple[int, int]:
    if not isinstance(value, str) or not value.strip():
        return (-1000, 0)
    text = value.strip()
    score = 0
    lowered = text.lower()
    if domain in lowered:
        score += 100
    if "http" in lowered:
        score += 20
    if "](" in text or "[" in text:
        score -= 100
    if "qgithub" in lowered or "qlinkedin" in lowered:
        score -= 80
    if domain == "linkedin.com" and "github.com" in lowered:
        score -= 120
    if domain == "github.com" and "linkedin.com" in lowered:
        score -= 120
    if " " in text:
        score -= 20
    return (score, len(text))


def _email_quality(value: Any) -> tuple[int, int]:
    normalized = normalize_email(value)
    if not normalized:
        return (-1000, 0)
    return (100, len(normalized))


def _best_peer_value(values: list[Any], kind: str) -> Any:
    non_empty = [value for value in values if value not in (None, "", [], {})]
    if not non_empty:
        return None
    if kind == "email":
        return max(non_empty, key=_email_quality)
    if kind == "linkedin":
        return max(non_empty, key=lambda item: _url_quality(item, "linkedin.com"))
    if kind == "github":
        return max(non_empty, key=lambda item: _url_quality(item, "github.com"))
    if kind == "years_experience":
        positive_values = sorted([_safe_int(item) for item in non_empty if _safe_int(item) > 0])
        return positive_values[0] if positive_values else non_empty[0]
    if kind == "experience_level":
        rank = {"junior": 1, "mid": 2, "senior": 3}
        return max(non_empty, key=lambda item: rank.get(str(item).lower(), 0))
    return non_empty[0]


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def consolidate_peer_profiles(results: list[dict[str, Any]], *, apply_changes: bool) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        payload = result["payload"]
        bio = ((payload.get("profile") or {}).get("bio") or {})
        email = normalize_email(bio.get("email"))
        full_name = normalize_text(bio.get("full_name"))
        group_key = email or full_name
        if group_key:
            grouped.setdefault(group_key, []).append(result)

    for group in grouped.values():
        if len(group) <= 1:
            continue
        peer_payloads = [item["payload"] for item in group]
        best_email = _best_peer_value([((payload.get("profile") or {}).get("bio") or {}).get("email") for payload in peer_payloads], "email")
        best_linkedin = _best_peer_value([((payload.get("profile") or {}).get("bio") or {}).get("linkedin") for payload in peer_payloads], "linkedin")
        best_github = _best_peer_value([((payload.get("profile") or {}).get("bio") or {}).get("github") for payload in peer_payloads], "github")
        best_years = _best_peer_value([((payload.get("profile") or {}).get("expertise") or {}).get("years_experience") for payload in peer_payloads], "years_experience")
        best_level = infer_experience_level(best_years) if best_years is not None else _best_peer_value(
            [((payload.get("profile") or {}).get("expertise") or {}).get("experience_level") for payload in peer_payloads],
            "experience_level",
        )

        for item in group:
            payload = item["payload"]
            bio = ((payload.get("profile") or {}).setdefault("bio", {}))
            expertise = ((payload.get("profile") or {}).setdefault("expertise", {}))

            if best_email and _email_quality(best_email) > _email_quality(bio.get("email")):
                bio["email"] = best_email
            if best_linkedin and _url_quality(best_linkedin, "linkedin.com") > _url_quality(bio.get("linkedin"), "linkedin.com"):
                bio["linkedin"] = best_linkedin
            if best_github and _url_quality(best_github, "github.com") > _url_quality(bio.get("github"), "github.com"):
                bio["github"] = best_github
            current_years = _safe_int(expertise.get("years_experience"))
            if best_years is not None and (expertise.get("years_experience") in (None, "", [], {}) or current_years <= 0 or current_years > int(best_years) * 2):
                expertise["years_experience"] = best_years
            if best_level and (expertise.get("experience_level") in (None, "", [], {}) or expertise.get("years_experience") == best_years):
                expertise["experience_level"] = best_level

            if apply_changes:
                write_json(item["profile_path"], payload)


def _get_nested(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def collect_change_examples(results: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    tracked_fields = [
        "profile.bio.email",
        "profile.bio.linkedin",
        "profile.bio.github",
        "profile.expertise.years_experience",
        "profile.expertise.experience_level",
        "grounding.hallucination_risk",
        "grounding.reliability_score",
    ]
    examples: list[dict[str, Any]] = []
    for result in results:
        original = result["original_payload"]
        updated = result["payload"]
        changes: list[dict[str, Any]] = []
        for field in tracked_fields:
            before = _get_nested(original, field)
            after = _get_nested(updated, field)
            if _canonical(before) != _canonical(after):
                changes.append({"field": field, "before": before, "after": after})
        if changes:
            examples.append(
                {
                    "profile_file": str(result["profile_path"]),
                    "artifact_path": updated.get("artifact_path"),
                    "changes": changes,
                }
            )
        if len(examples) >= limit:
            break
    return examples


def has_meaningful_change(result: dict[str, Any]) -> bool:
    return bool(
        collect_change_examples([result], limit=1)
    )


def build_markdown_report(report: dict[str, Any]) -> str:
    report_title = "# Patch Dry-Run Report V2 Safe URL Fixes" if report.get("only_safe_url_fixes") else "# Patch Dry-Run Report V2 Fixes"
    lines = [
        report_title,
        "",
        f"- `total_profiles_checked`: {report['total_profiles_checked']}",
        f"- `profiles_that_would_change`: {report['profiles_that_would_change']}",
        f"- `urls_that_would_be_fixed`: {report['urls_that_would_be_fixed']}",
        f"- `skipped_due_to_quality_regression`: {report['skipped_due_to_quality_regression']}",
        f"- `reliability_regressions_count`: {report['reliability_regressions_count']}",
        f"- `safe_to_execute`: {str(report['safe_to_execute']).lower()}",
        "",
        "## Risk Before",
        "",
        "```json",
        json.dumps(report["risk_before_distribution"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Risk After Estimated",
        "",
        "```json",
        json.dumps(report["risk_after_distribution_estimated"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Examples Of URL Fixes",
        "",
        "```json",
        json.dumps(report["examples_of_url_fixes"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Recommendation",
        "",
        report["recommendation"],
        "",
    ]
    return "\n".join(lines)


def build_mongodb_update(payload: dict[str, Any], profile_id: str) -> dict[str, Any]:
    profile = payload["profile"]
    bio = profile.get("bio") or {}
    expertise = profile.get("expertise") or {}
    email_normalized = normalize_email(bio.get("email"))
    phone_normalized = normalize_phone(bio.get("phone"))
    linkedin_urls, github_urls, portfolio_urls = extract_urls(profile)

    return {
        "profile_id": profile_id,
        "source_path": payload.get("source_path"),
        "artifact_path": payload.get("artifact_path"),
        "bio": bio,
        "expertise": expertise,
        "experiences": list(profile.get("experiences") or []),
        "education": list(profile.get("education") or []),
        "languages": list(profile.get("languages") or []),
        "certifications": list(profile.get("certifications") or []),
        "profile_kind": payload.get("profile_kind"),
        "reliability_score": float((payload.get("grounding") or {}).get("reliability_score") or 0.0),
        "hallucination_risk": (payload.get("grounding") or {}).get("hallucination_risk"),
        "quality_flags": list((payload.get("grounding") or {}).get("quality_flags") or []),
        "fields_nullified": list((payload.get("grounding") or {}).get("fields_nullified") or []),
        "full_name": clean_display_text(bio.get("full_name")),
        "name_normalized": normalize_text(bio.get("full_name")),
        "has_valid_name": is_valid_name(bio.get("full_name")),
        "email_raw": bio.get("email"),
        "email_normalized": email_normalized,
        "email_class": "missing" if not email_normalized else "generic" if is_generic_email(email_normalized) else "real",
        "phone_raw": bio.get("phone"),
        "phone_normalized": phone_normalized,
        "phone_class": "missing" if not phone_normalized else "real" if is_real_phone(phone_normalized) else "weak_or_placeholder",
        "location": bio.get("location"),
        "location_normalized": normalize_text(bio.get("location")),
        "schools": extract_schools(profile),
        "school_keys": [normalize_text(school) for school in extract_schools(profile) if normalize_text(school)],
        "linkedin_urls": linkedin_urls,
        "github_urls": github_urls,
        "portfolio_urls": portfolio_urls,
        "updated_at": utc_now(),
    }


def update_mongodb_documents(
    updates: list[dict[str, Any]],
    *,
    mongodb_uri: str,
    database_name: str,
    collection_name: str,
) -> dict[str, int]:
    from pymongo import MongoClient

    client = MongoClient(mongodb_uri)
    matched_count = 0
    modified_count = 0
    try:
        collection = client[database_name][collection_name]
        for item in updates:
            profile_id = item.pop("profile_id")
            artifact_path = item.get("artifact_path")
            result = collection.update_one(
                {
                    "$or": [
                        {"profile_id": profile_id},
                        {"artifact_path": artifact_path},
                    ]
                },
                {"$set": item},
                upsert=False,
            )
            matched_count += result.matched_count
            modified_count += result.modified_count
    finally:
        client.close()
    return {"matched_count": matched_count, "modified_count": modified_count}


def run_patch(
    *,
    profiles_dir: Path,
    mongodb_uri: str,
    database_name: str,
    collection_name: str,
    dry_run: bool,
    rebuild_faiss: bool,
    report_path: Path,
    markdown_report_path: Path | None = None,
    only_safe_url_fixes: bool = False,
) -> dict[str, Any]:
    profile_paths = sorted(profiles_dir.glob("*.json"))
    updates: list[dict[str, Any]] = []
    patched_results: list[dict[str, Any]] = []
    emails_fixed = 0
    urls_fixed = 0
    years_recovered = 0
    levels_recovered = 0
    risk_reduced_to_low = 0
    risk_reduced_to_medium = 0
    risk_still_high = 0
    total_profiles_patched = 0
    risk_before = Counter()
    risk_after = Counter()
    skipped_due_to_quality_regression = 0
    reliability_regressions_count = 0
    applied_risk_regressions_count = 0
    mongodb_update_result = {"matched_count": 0, "modified_count": 0}

    for profile_path in profile_paths:
        result = patch_profile_payload(
            profile_path,
            profiles_dir,
            apply_changes=False,
            only_safe_url_fixes=only_safe_url_fixes,
        )
        patched_results.append(result)
        payload = result["payload"]
        original_payload = result["original_payload"]
        emails_fixed += int(result["email_fixed"])
        urls_fixed += len(result["applied_url_changes"])
        years_recovered += int(result["years_recovered"])
        levels_recovered += int(result["level_recovered"])
        skipped_due_to_quality_regression += int(result["skipped_due_to_quality_regression"])
        reliability_regressions_count += int(result["quality_regression"]["reliability_regressed"])
        applied_risk_regressions_count += int(
            result["quality_regression"]["risk_regressed"] and not result["skipped_due_to_quality_regression"]
        )

        previous_risk = result["previous_risk"]
        new_risk = result["new_risk"]
        risk_before[previous_risk or "missing"] += 1
        risk_after[new_risk or "missing"] += 1
        if previous_risk in {"medium", "high"} and new_risk == "low":
            risk_reduced_to_low += 1
        elif previous_risk == "high" and new_risk == "medium":
            risk_reduced_to_medium += 1
        if new_risk == "high":
            risk_still_high += 1

    if not only_safe_url_fixes:
        consolidate_peer_profiles(patched_results, apply_changes=not dry_run)
    profiles_that_would_change = sum(1 for result in patched_results if has_meaningful_change(result))
    examples_of_changes = collect_change_examples(patched_results)
    examples_of_url_fixes = [
        {
            "profile_file": str(result["profile_path"]),
            "artifact_path": result["payload"].get("artifact_path"),
            "changes": result["applied_url_changes"],
        }
        for result in patched_results
        if result["applied_url_changes"]
    ][:8]
    changed_results = [result for result in patched_results if has_meaningful_change(result)]
    total_profiles_patched = len(changed_results)
    for result in changed_results:
        updates.append(build_mongodb_update(result["payload"], result["profile_id"]))

    if not dry_run:
        for result in changed_results:
            write_json(result["profile_path"], result["payload"])
        if updates:
            mongodb_update_result = update_mongodb_documents(
                updates,
                mongodb_uri=mongodb_uri,
                database_name=database_name,
                collection_name=collection_name,
            )

    faiss_report = None
    faiss_rebuilt = False
    faiss_error: str | None = None
    if not dry_run and rebuild_faiss:
        try:
            faiss_report = run_faiss_indexer()
            faiss_rebuilt = True
        except Exception as exc:
            faiss_error = str(exc)

    if dry_run:
        report = {
            "generated_at_utc": utc_now(),
            "profiles_dir": str(profiles_dir),
            "mongodb_uri": mongodb_uri,
            "database": database_name,
            "collection": collection_name,
            "only_safe_url_fixes": only_safe_url_fixes,
            "total_profiles_checked": len(profile_paths),
            "profiles_that_would_change": profiles_that_would_change,
            "emails_that_would_be_fixed": emails_fixed,
            "urls_that_would_be_fixed": urls_fixed,
            "years_experience_that_would_be_recovered": years_recovered,
            "experience_level_that_would_be_recovered": levels_recovered,
            "skipped_due_to_quality_regression": skipped_due_to_quality_regression,
            "risk_before_distribution": dict(sorted(risk_before.items())),
            "risk_after_distribution_estimated": dict(sorted(risk_after.items())),
            "reliability_regressions_count": reliability_regressions_count,
            "high_risk_remaining": risk_after.get("high", 0),
            "examples_of_changes": examples_of_changes,
            "examples_of_url_fixes": examples_of_url_fixes,
            "safe_to_execute": faiss_error is None and reliability_regressions_count == 0 and applied_risk_regressions_count == 0,
            "recommendation": (
                "Dry-run looks safe. Apply --execute only after reviewing the examples_of_url_fixes and confirming that only URL normalizations should be persisted."
                if only_safe_url_fixes and faiss_error is None and reliability_regressions_count == 0 and applied_risk_regressions_count == 0
                else "Dry-run looks safe. Apply --execute only after reviewing the examples_of_changes and confirming that the contact and experience consolidations are acceptable."
                if faiss_error is None and reliability_regressions_count == 0 and applied_risk_regressions_count == 0
                else "Do not execute yet: quality regression was detected and skipped. Review the skipped_due_to_quality_regression count before any execution."
                if faiss_error is None
                else f"Do not execute yet: {faiss_error}"
            ),
        }
        write_json(report_path, report)
        if markdown_report_path is not None:
            markdown_report_path.write_text(build_markdown_report(report), encoding="utf-8")
        return report

    report = {
        "generated_at_utc": utc_now(),
        "profiles_dir": str(profiles_dir),
        "mongodb_uri": mongodb_uri,
        "database": database_name,
        "collection": collection_name,
        "total_profiles_scanned": len(profile_paths),
        "total_profiles_patched": total_profiles_patched,
        "emails_fixed": emails_fixed,
        "urls_fixed": urls_fixed,
        "years_experience_recovered": years_recovered,
        "experience_level_recovered": levels_recovered,
        "skipped_due_to_quality_regression": skipped_due_to_quality_regression,
        "reliability_regressions_count": reliability_regressions_count,
        "risk_before_distribution": dict(sorted(risk_before.items())),
        "risk_after_distribution": dict(sorted(risk_after.items())),
        "examples_of_url_fixes": examples_of_url_fixes,
        "mongodb_update_result": mongodb_update_result,
        "risk_reduced_to_low": risk_reduced_to_low,
        "risk_reduced_to_medium": risk_reduced_to_medium,
        "risk_still_high": risk_still_high,
        "faiss_rebuilt": faiss_rebuilt,
        "faiss_requested": rebuild_faiss,
        "faiss_error": faiss_error,
        "faiss_report": faiss_report,
    }
    write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patch grounded V2 profiles in place without rerunning Module 2 on all CVs.")
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument("--mongodb-uri", default="mongodb://localhost:27017")
    parser.add_argument("--database", default="talent_intelligence")
    parser.add_argument("--collection", default="candidate_profiles")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rebuild-faiss", action="store_true")
    parser.add_argument("--only-safe-url-fixes", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dry_run = args.dry_run or not args.execute
    if dry_run and args.report_path == DEFAULT_REPORT_PATH:
        report_path = DEFAULT_SAFE_URL_DRY_RUN_REPORT_PATH if args.only_safe_url_fixes else DEFAULT_DRY_RUN_REPORT_PATH
    else:
        report_path = args.report_path
    if dry_run:
        markdown_report_path = DEFAULT_SAFE_URL_DRY_RUN_MARKDOWN_PATH if args.only_safe_url_fixes else DEFAULT_DRY_RUN_MARKDOWN_PATH
    else:
        markdown_report_path = None
    report = run_patch(
        profiles_dir=args.profiles_dir,
        mongodb_uri=args.mongodb_uri,
        database_name=args.database,
        collection_name=args.collection,
        dry_run=dry_run,
        rebuild_faiss=args.rebuild_faiss,
        report_path=report_path,
        markdown_report_path=markdown_report_path,
        only_safe_url_fixes=args.only_safe_url_fixes,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
