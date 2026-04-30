from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_MODULE2_ROOT = Path("data/profile_builder_official_module2_rerun_ollama_fixed")
DEFAULT_ACCEPTED_PATH = Path("data/processed_official_module1/handoff/accepted.json")
DEFAULT_REPORT_PATH = Path("data/mongodb_import_report.json")
DEFAULT_MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DEFAULT_DATABASE = os.getenv("MONGODB_DATABASE", "talent_intelligence")
DEFAULT_CANDIDATES_COLLECTION = os.getenv("MONGODB_CANDIDATES_COLLECTION", "candidates")
DEFAULT_CANDIDATE_PROFILES_COLLECTION = os.getenv(
    "MONGODB_CANDIDATE_PROFILES_COLLECTION",
    "candidate_profiles",
)

REPORT_FILE_NAMES = {"run_report.json", "final_manual_summary.json"}
IGNORED_DIRECTORIES = {"_continuation_inputs"}
COMPLETE_PROFILE = "complete_profile"
PARTIAL_PROFILE = "partial_profile"
OLD_UNIQUE_CANDIDATES_COUNT = 66

GENERIC_EMAILS = {
    "email@youremail.com",
    "first.last@resumeworded.com",
    "info@qwikresume.com",
    "info@resumekraft.com",
    "resumesample@example.com",
    "example@example.com",
    "test@example.com",
    "noreply@example.com",
    "no-reply@example.com",
}
GENERIC_EMAIL_LOCAL_PREFIXES = ("contact", "info", "support", "noreply", "no-reply")
GENERIC_EMAIL_DOMAINS = {"example.com", "youremail.com"}
PLACEHOLDER_NAMES = {
    "",
    "none",
    "null",
    "firstlast",
    "first last",
    "firstname lastname",
    "john doe",
    "johndoe",
}
PLACEHOLDER_PHONE_PATTERNS = {
    "0000000000",
    "111111111",
    "1111111111",
    "1234567",
    "123456789",
    "1234567890",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_module2_json_files(module2_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in module2_root.rglob("*.json"):
        relative = path.relative_to(module2_root)
        if relative.parts[0] in IGNORED_DIRECTORIES:
            continue
        if path.name in REPORT_FILE_NAMES:
            continue
        files.append(path)
    return sorted(files)


def load_accepted_by_artifact_path(accepted_path: Path) -> dict[str, dict[str, Any]]:
    if not accepted_path.exists():
        return {}
    rows = read_json(accepted_path)
    if not isinstance(rows, list):
        raise ValueError(f"accepted file must contain a list: {accepted_path}")
    return {str(row.get("artifact_path")): row for row in rows if row.get("artifact_path")}


def stable_id(prefix: str, parts: list[Any]) -> str:
    raw = "|".join(str(part) for part in parts if part not in (None, ""))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def normalize_email(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    email = re.sub(r"\s+", "", value.strip().lower())
    if not email or "@" not in email:
        return None
    domain = email.rsplit("@", 1)[-1]
    if "." not in domain:
        return None
    return email


def is_generic_email(email: str | None) -> bool:
    if not email:
        return False
    if email in GENERIC_EMAILS:
        return True
    local, domain = email.split("@", 1)
    if domain in GENERIC_EMAIL_DOMAINS:
        return True
    return any(
        local == prefix
        or local.startswith(f"{prefix}.")
        or local.startswith(f"{prefix}-")
        for prefix in GENERIC_EMAIL_LOCAL_PREFIXES
    )


def normalize_phone(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    digits = re.sub(r"\D+", "", value)
    if len(digits) < 7:
        return None
    return digits


def is_real_phone(phone: str | None) -> bool:
    if not phone:
        return False
    if phone in PLACEHOLDER_PHONE_PATTERNS:
        return False
    if "1234567" in phone or "1111111" in phone:
        return False
    if len(set(phone)) == 1:
        return False
    if re.search(r"555\d{4}$", phone):
        return False
    return True


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def clean_display_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip()
    return text or None


def is_valid_name(value: Any) -> bool:
    normalized = normalize_text(value)
    placeholders = {normalize_text(name) for name in PLACEHOLDER_NAMES}
    return bool(normalized) and normalized not in placeholders


def name_similarity(left: Any, right: Any) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 1.0
    if left_norm == right_norm or left_norm in right_norm or right_norm in left_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def detect_name_conflicts(profile_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = [
        profile["full_name"]
        for profile in profile_docs
        if is_valid_name(profile.get("full_name"))
    ]
    conflicts: list[dict[str, Any]] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            similarity = name_similarity(left, right)
            if similarity < 0.62:
                conflicts.append(
                    {
                        "left": left,
                        "right": right,
                        "similarity": round(similarity, 4),
                    }
                )
    return conflicts


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(flatten_strings(item))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(flatten_strings(item))
        return strings
    return []


def extract_urls(profile: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    text = "\n".join(flatten_strings(profile))
    urls = [url.strip(".,);]") for url in re.findall(r"(?:https?://|www\.)\S+", text, flags=re.I)]
    linkedin = [url for url in urls if "linkedin.com" in url.lower()]
    github = [url for url in urls if "github.com" in url.lower()]
    portfolio = [url for url in urls if url not in linkedin and url not in github]
    return unique_strings(linkedin), unique_strings(github), unique_strings(portfolio)


def extract_schools(profile: dict[str, Any]) -> list[str]:
    schools: list[str] = []
    for item in profile.get("education") or []:
        if not isinstance(item, dict):
            continue
        school = clean_display_text(item.get("school"))
        if school:
            schools.append(school)
    return unique_strings(schools)


def unique_strings(values: list[Any]) -> list[str]:
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


def canonical_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def unique_items(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = canonical_dump(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def provider_rank(provider_route: str | None) -> int:
    return {
        "groq_secondary": 3,
        "ollama_local": 2,
        "primary_openai": 1,
    }.get(str(provider_route or "").lower(), 0)


def profile_kind_rank(profile_kind: str | None) -> int:
    return {
        COMPLETE_PROFILE: 2,
        PARTIAL_PROFILE: 1,
    }.get(str(profile_kind or "").lower(), 0)


def profile_quality_score(profile_doc: dict[str, Any]) -> tuple[float, int, int, str]:
    return (
        float(profile_doc.get("reliability_score") or 0.0),
        profile_kind_rank(profile_doc.get("profile_kind")),
        provider_rank(profile_doc.get("provider_route")),
        str(profile_doc.get("artifact_path") or ""),
    )


def profile_summary(profile_doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": profile_doc["profile_id"],
        "module2_file_path": profile_doc["module2_file_path"],
        "source_path": profile_doc.get("source_path"),
        "artifact_path": profile_doc.get("artifact_path"),
        "source_format": profile_doc.get("source_format"),
        "profile_kind": profile_doc.get("profile_kind"),
        "provider_route": profile_doc.get("provider_route"),
        "full_name": profile_doc.get("full_name"),
        "email_raw": profile_doc.get("email_raw"),
        "email_normalized": profile_doc.get("email_normalized"),
        "email_class": profile_doc.get("email_class"),
        "phone_raw": profile_doc.get("phone_raw"),
        "phone_normalized": profile_doc.get("phone_normalized"),
        "phone_class": profile_doc.get("phone_class"),
        "location": profile_doc.get("location"),
        "source_id": profile_doc.get("source_id"),
        "schools": profile_doc.get("schools", []),
    }


def build_profile_document(
    *,
    module2_file: Path,
    module2_root: Path,
    payload: dict[str, Any],
    accepted_entry: dict[str, Any] | None,
    imported_at: str,
) -> dict[str, Any]:
    profile = payload.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("success payload is missing profile")

    bio = profile.get("bio") or {}
    metadata = profile.get("metadata") or {}
    artifact_path = str(payload.get("artifact_path") or "")
    source_path = payload.get("source_path")
    email_raw = bio.get("email")
    email = normalize_email(email_raw)
    phone_raw = bio.get("phone")
    phone = normalize_phone(phone_raw)
    linkedin_urls, github_urls, portfolio_urls = extract_urls(profile)
    reliability_score = metadata.get("confidence_score")
    if reliability_score is None and accepted_entry:
        reliability_score = accepted_entry.get("document_confidence_score")

    source_format = payload.get("source_format")
    if accepted_entry and accepted_entry.get("source_format"):
        source_format = accepted_entry.get("source_format")

    profile_id = stable_id("profile", [source_path, artifact_path, module2_file.relative_to(module2_root)])

    return {
        "profile_id": profile_id,
        "candidate_id": None,
        "source_path": source_path,
        "artifact_path": artifact_path,
        "source_format": source_format,
        "status": payload.get("status"),
        "profile_kind": profile.get("profile_kind"),
        "provider_route": metadata.get("provider_route"),
        "bio": bio,
        "expertise": profile.get("expertise") or {},
        "experiences": list(profile.get("experiences") or []),
        "education": list(profile.get("education") or []),
        "metadata": metadata,
        "reliability_score": float(reliability_score or 0.0),
        "quality_flags": list((accepted_entry or {}).get("quality_flags") or []),
        "dedup_status": "unique_candidate",
        "dedup_confidence": 1.0,
        "dedup_evidence": {"type": "unique_source_profile", "reason": "No automatic merge evidence used."},
        "created_at": imported_at,
        "updated_at": imported_at,
        "module2_file_path": str(module2_file),
        "module2_relative_path": str(module2_file.relative_to(module2_root)),
        "source_id": profile.get("source_id"),
        "full_name": clean_display_text(bio.get("full_name")),
        "name_normalized": normalize_text(bio.get("full_name")),
        "has_valid_name": is_valid_name(bio.get("full_name")),
        "email_raw": email_raw,
        "email_normalized": email,
        "email_class": "missing" if not email else "generic" if is_generic_email(email) else "real",
        "phone_raw": phone_raw,
        "phone_normalized": phone,
        "phone_class": "missing" if not phone else "real" if is_real_phone(phone) else "weak_or_placeholder",
        "location": bio.get("location"),
        "location_normalized": normalize_text(bio.get("location")),
        "schools": extract_schools(profile),
        "school_keys": [normalize_text(school) for school in extract_schools(profile) if normalize_text(school)],
        "linkedin_urls": linkedin_urls,
        "github_urls": github_urls,
        "portfolio_urls": portfolio_urls,
    }


def assign_candidate(
    *,
    profile_doc: dict[str, Any],
    candidate_id: str,
    dedup_status: str,
    dedup_confidence: float,
    dedup_evidence: dict[str, Any],
) -> None:
    profile_doc["candidate_id"] = candidate_id
    profile_doc["dedup_status"] = dedup_status
    profile_doc["dedup_confidence"] = dedup_confidence
    profile_doc["dedup_evidence"] = dedup_evidence


def add_auto_group(
    *,
    groups: list[dict[str, Any]],
    assigned_profile_ids: set[str],
    level: str,
    evidence_type: str,
    evidence_value: str,
    members: list[dict[str, Any]],
    confidence: float,
) -> None:
    candidate_id = stable_id("candidate", [level, evidence_type, evidence_value])
    evidence = {
        "type": evidence_type,
        "value": evidence_value,
        "level": level,
        "rule": f"{level}: identical {evidence_type} and no obvious identity conflict",
    }
    for member in members:
        assign_candidate(
            profile_doc=member,
            candidate_id=candidate_id,
            dedup_status="auto_merged",
            dedup_confidence=confidence,
            dedup_evidence=evidence,
        )
        assigned_profile_ids.add(member["profile_id"])
    groups.append(
        {
            "candidate_id": candidate_id,
            "dedup_level": level,
            "dedup_status": "auto_merged",
            "dedup_confidence": confidence,
            "dedup_evidence": evidence,
            "profile_count": len(members),
            "profiles": [profile_summary(member) for member in members],
        }
    )


def add_possible_group(
    *,
    groups: list[dict[str, Any]],
    seen: set[str],
    possible_type: str,
    value: str,
    members: list[dict[str, Any]],
    reason: str,
) -> None:
    if len(members) <= 1:
        return
    assigned_ids = [member.get("candidate_id") for member in members]
    if all(assigned_ids) and len(set(assigned_ids)) == 1:
        return
    signature = canonical_dump(
        {
            "type": possible_type,
            "value": value,
            "profiles": sorted(member["profile_id"] for member in members),
        }
    )
    if signature in seen:
        return
    seen.add(signature)
    group = {
        "dedup_status": "possible_duplicate_needs_review",
        "possible_duplicate_type": possible_type,
        "evidence": {"value": value},
        "reason": reason,
        "profile_count": len(members),
        "profiles": [profile_summary(member) for member in members],
    }
    groups.append(group)
    for member in members:
        if member.get("dedup_status") == "unique_candidate":
            member["dedup_status"] = "possible_duplicate_needs_review"
            member["dedup_confidence"] = 0.4
            member["dedup_evidence"] = {
                "type": possible_type,
                "value": value,
                "reason": reason,
            }


def analyse_dedup(profile_docs: list[dict[str, Any]]) -> dict[str, Any]:
    assigned_profile_ids: set[str] = set()
    groups_auto_merged: list[dict[str, Any]] = []
    groups_not_merged: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    possible_seen: set[str] = set()

    by_real_email: dict[str, list[dict[str, Any]]] = {}
    by_real_phone: dict[str, list[dict[str, Any]]] = {}
    by_linkedin: dict[str, list[dict[str, Any]]] = {}
    by_github: dict[str, list[dict[str, Any]]] = {}
    by_generic_email: dict[str, list[dict[str, Any]]] = {}

    for doc in profile_docs:
        if doc["email_class"] == "real":
            by_real_email.setdefault(doc["email_normalized"], []).append(doc)
        elif doc["email_class"] == "generic":
            by_generic_email.setdefault(doc["email_normalized"], []).append(doc)

        if doc["phone_class"] == "real":
            by_real_phone.setdefault(doc["phone_normalized"], []).append(doc)

        for url in doc["linkedin_urls"]:
            by_linkedin.setdefault(url.lower(), []).append(doc)
        for url in doc["github_urls"]:
            by_github.setdefault(url.lower(), []).append(doc)

    for email, members in sorted(by_real_email.items()):
        if len(members) <= 1:
            continue
        name_conflicts = detect_name_conflicts(members)
        if name_conflicts:
            conflicts.append(
                {
                    "conflict_type": "real_email_name_conflict",
                    "evidence": {"email": email},
                    "reason": "Same real email but materially different names; no automatic candidate merge.",
                    "name_conflicts": name_conflicts,
                    "profiles": [profile_summary(member) for member in members],
                }
            )
            continue
        add_auto_group(
            groups=groups_auto_merged,
            assigned_profile_ids=assigned_profile_ids,
            level="A",
            evidence_type="real_email",
            evidence_value=email,
            members=members,
            confidence=0.98,
        )

    for group_map, evidence_type, confidence in (
        (by_real_phone, "real_phone", 0.95),
        (by_linkedin, "linkedin_url", 0.99),
        (by_github, "github_url", 0.99),
    ):
        for value, raw_members in sorted(group_map.items()):
            members = [member for member in raw_members if member["profile_id"] not in assigned_profile_ids]
            if len(members) <= 1:
                continue
            name_conflicts = detect_name_conflicts(members)
            if name_conflicts:
                conflicts.append(
                    {
                        "conflict_type": f"{evidence_type}_name_conflict",
                        "evidence": {evidence_type: value},
                        "reason": f"Same {evidence_type} but materially different names; no automatic candidate merge.",
                        "name_conflicts": name_conflicts,
                        "profiles": [profile_summary(member) for member in members],
                    }
                )
                continue
            add_auto_group(
                groups=groups_auto_merged,
                assigned_profile_ids=assigned_profile_ids,
                level="A",
                evidence_type=evidence_type,
                evidence_value=value,
                members=members,
                confidence=confidence,
            )

    for email, members in sorted(by_generic_email.items()):
        name_conflicts = detect_name_conflicts(members)
        unique_names = {member["name_normalized"] for member in members if member["has_valid_name"]}
        if name_conflicts or len(unique_names) > 1:
            conflict = {
                "conflict_type": "generic_email_shared_by_multiple_identities",
                "evidence": {"generic_email": email},
                "reason": "Generic/template email is shared; automatic merge prevented.",
                "name_conflicts": name_conflicts,
                "profiles": [profile_summary(member) for member in members],
            }
            conflicts.append(conflict)
            for member in members:
                if member.get("dedup_status") == "unique_candidate":
                    member["dedup_status"] = "conflict_detected_no_merge"
                    member["dedup_confidence"] = 0.0
                    member["dedup_evidence"] = {
                        "type": "generic_email_conflict",
                        "value": email,
                        "reason": conflict["reason"],
                    }

        add_possible_group(
            groups=groups_not_merged,
            seen=possible_seen,
            possible_type="generic_email_group",
            value=email,
            members=members,
            reason="Same generic/template email. Do not merge automatically; review with name, phone, school and location.",
        )

    by_name: dict[str, list[dict[str, Any]]] = {}
    by_name_school_location: dict[str, list[dict[str, Any]]] = {}
    for doc in profile_docs:
        if doc["has_valid_name"]:
            by_name.setdefault(doc["name_normalized"], []).append(doc)
        if not doc["has_valid_name"] or not doc["location_normalized"]:
            continue
        for school_key in doc["school_keys"]:
            key = f"{doc['name_normalized']}|{school_key}|{doc['location_normalized'][:32]}"
            by_name_school_location.setdefault(key, []).append(doc)

    for name, members in sorted(by_name.items()):
        add_possible_group(
            groups=groups_not_merged,
            seen=possible_seen,
            possible_type="same_normalized_name",
            value=name,
            members=members,
            reason="Same normalized name but not enough strong evidence for automatic merge.",
        )

    for key, members in sorted(by_name_school_location.items()):
        add_possible_group(
            groups=groups_not_merged,
            seen=possible_seen,
            possible_type="medium_evidence_name_school_location",
            value=key,
            members=members,
            reason="Medium evidence match. Recommended review before merge in first official import.",
        )

    for doc in profile_docs:
        if doc.get("candidate_id"):
            continue
        candidate_id = stable_id("candidate", ["single_profile", doc["profile_id"]])
        assign_candidate(
            profile_doc=doc,
            candidate_id=candidate_id,
            dedup_status=doc["dedup_status"],
            dedup_confidence=float(doc["dedup_confidence"]),
            dedup_evidence=doc["dedup_evidence"],
        )

    risky_merges_prevented = [
        {
            "old_candidate_key": f"email:{email}",
            "reason": "Old logic would merge on generic/template email; new strategy keeps profiles separate unless stronger evidence is validated.",
            "profile_count": len(members),
            "profiles": [profile_summary(member) for member in members],
        }
        for email, members in sorted(by_generic_email.items())
        if len(members) > 1
    ]

    return {
        "groups_auto_merged": groups_auto_merged,
        "groups_not_merged": groups_not_merged,
        "conflicts": conflicts,
        "risky_merges_prevented": risky_merges_prevented,
    }


def build_candidate_documents(profile_docs: list[dict[str, Any]], imported_at: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for doc in profile_docs:
        grouped.setdefault(doc["candidate_id"], []).append(doc)

    candidates: dict[str, dict[str, Any]] = {}
    for candidate_id, members in sorted(grouped.items()):
        best = max(members, key=profile_quality_score)
        names = unique_strings([member.get("full_name") for member in members if member.get("full_name")])
        canonical_name = best.get("full_name") or (names[0] if names else None)
        emails = unique_strings(
            [member["email_normalized"] for member in members if member.get("email_class") == "real"]
        )
        phones = unique_strings(
            [member["phone_normalized"] for member in members if member.get("phone_class") == "real"]
        )
        linkedin_urls = unique_strings([url for member in members for url in member.get("linkedin_urls", [])])
        github_urls = unique_strings([url for member in members for url in member.get("github_urls", [])])
        locations = unique_strings([member.get("location") for member in members if member.get("location")])
        statuses = [member.get("dedup_status") for member in members]
        if "conflict_detected_no_merge" in statuses:
            dedup_status = "conflict_detected_no_merge"
        elif "auto_merged" in statuses:
            dedup_status = "auto_merged"
        elif "possible_duplicate_needs_review" in statuses:
            dedup_status = "possible_duplicate_needs_review"
        else:
            dedup_status = "unique_candidate"

        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "canonical_name": canonical_name,
            "emails": emails,
            "phones": phones,
            "linkedin_urls": linkedin_urls,
            "github_urls": github_urls,
            "locations": locations,
            "profile_ids": sorted(member["profile_id"] for member in members),
            "profile_count": len(members),
            "best_profile_id": best["profile_id"],
            "dedup_status": dedup_status,
            "dedup_confidence": max(float(member.get("dedup_confidence") or 0.0) for member in members),
            "dedup_evidence": unique_items([member.get("dedup_evidence") or {} for member in members]),
            "created_at": imported_at,
            "updated_at": imported_at,
        }
    return candidates


def analyse_profiles(
    module2_root: Path,
    accepted_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    accepted_by_artifact = load_accepted_by_artifact_path(accepted_path)
    files = iter_module2_json_files(module2_root)
    imported_at = utc_now()

    status_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    profile_kind_counts: Counter[str] = Counter()
    failed_ignored: list[dict[str, Any]] = []
    import_errors: list[dict[str, Any]] = []
    profile_docs: list[dict[str, Any]] = []

    for path in files:
        try:
            payload = read_json(path)
        except Exception as exc:
            import_errors.append({"file_path": str(path), "stage": "read_json", "error": str(exc)})
            continue

        status = str(payload.get("status") or "unknown")
        status_counts[status] += 1
        if status != "success":
            failed_ignored.append(
                {
                    "file_path": str(path),
                    "status": status,
                    "failure_type": payload.get("failure_type"),
                    "source_path": payload.get("source_path"),
                    "artifact_path": payload.get("artifact_path"),
                    "error": payload.get("error"),
                }
            )
            continue

        artifact_path = str(payload.get("artifact_path") or "")
        accepted_entry = accepted_by_artifact.get(artifact_path)
        try:
            profile_doc = build_profile_document(
                module2_file=path,
                module2_root=module2_root,
                payload=payload,
                accepted_entry=accepted_entry,
                imported_at=imported_at,
            )
        except Exception as exc:
            import_errors.append({"file_path": str(path), "stage": "build_profile_document", "error": str(exc)})
            continue

        profile_docs.append(profile_doc)
        profile_kind_counts[str(profile_doc.get("profile_kind") or "unknown")] += 1
        provider_counts[str(profile_doc.get("provider_route") or "unknown")] += 1

    dedup = analyse_dedup(profile_docs)
    candidates = build_candidate_documents(profile_docs, imported_at)
    candidate_profiles = {doc["profile_id"]: doc for doc in profile_docs}

    analysis = {
        "total_json_read": len(files),
        "status_counts": dict(sorted(status_counts.items())),
        "success_profiles": len(profile_docs),
        "failed_ignored": len(failed_ignored),
        "failed_ignored_items": failed_ignored,
        "profile_kind_counts": dict(sorted(profile_kind_counts.items())),
        "provider_route_counts": dict(sorted(provider_counts.items())),
        "candidate_profiles_to_import": len(candidate_profiles),
        "candidates_to_create": len(candidates),
        "real_email_count": sum(1 for doc in profile_docs if doc["email_class"] == "real"),
        "generic_email_count": sum(1 for doc in profile_docs if doc["email_class"] == "generic"),
        "missing_email_count": sum(1 for doc in profile_docs if doc["email_class"] == "missing"),
        "phones_detected": sum(1 for doc in profile_docs if doc["phone_normalized"]),
        "real_phone_count": sum(1 for doc in profile_docs if doc["phone_class"] == "real"),
        "linkedin_detected_count": sum(1 for doc in profile_docs if doc["linkedin_urls"]),
        "github_detected_count": sum(1 for doc in profile_docs if doc["github_urls"]),
        "strong_merges_count": len(dedup["groups_auto_merged"]),
        "possible_duplicates_count": len(dedup["groups_not_merged"]),
        "conflicts_count": len(dedup["conflicts"]),
        "risky_merges_prevented": dedup["risky_merges_prevented"],
        "groups_auto_merged": dedup["groups_auto_merged"],
        "groups_not_merged": dedup["groups_not_merged"],
        "conflicts": dedup["conflicts"],
        "errors_import": import_errors,
    }
    return candidates, candidate_profiles, analysis


def mask_mongodb_uri(uri: str) -> str:
    return re.sub(r"(mongodb(?:\+srv)?://)([^:@/]+):([^@/]+)@", r"\1***:***@", uri)


def ensure_candidate_indexes(collection: Any) -> None:
    from pymongo import ASCENDING, DESCENDING

    collection.create_index([("candidate_id", ASCENDING)], unique=True, name="uniq_candidate_id")
    collection.create_index([("canonical_name", ASCENDING)], name="idx_canonical_name")
    collection.create_index([("emails", ASCENDING)], name="idx_emails")
    collection.create_index([("phones", ASCENDING)], name="idx_phones")
    collection.create_index([("dedup_status", ASCENDING)], name="idx_dedup_status")
    collection.create_index([("best_profile_id", ASCENDING)], name="idx_best_profile_id")
    collection.create_index([("profile_count", DESCENDING)], name="idx_profile_count")
    collection.create_index([("updated_at", DESCENDING)], name="idx_candidates_updated_at")


def ensure_candidate_profile_indexes(collection: Any) -> None:
    from pymongo import ASCENDING, DESCENDING

    collection.create_index([("profile_id", ASCENDING)], unique=True, name="uniq_profile_id")
    collection.create_index([("artifact_path", ASCENDING)], unique=True, name="uniq_artifact_path")
    collection.create_index([("candidate_id", ASCENDING)], name="idx_candidate_id")
    collection.create_index([("source_path", ASCENDING)], name="idx_source_path")
    collection.create_index([("status", ASCENDING)], name="idx_status")
    collection.create_index([("profile_kind", ASCENDING)], name="idx_profile_kind")
    collection.create_index([("provider_route", ASCENDING)], name="idx_provider_route")
    collection.create_index([("dedup_status", ASCENDING)], name="idx_profile_dedup_status")
    collection.create_index([("reliability_score", DESCENDING)], name="idx_reliability_score")
    collection.create_index([("updated_at", DESCENDING)], name="idx_profiles_updated_at")


def merge_created_at(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    if existing and existing.get("created_at"):
        incoming["created_at"] = existing["created_at"]
    return incoming


def import_documents(
    *,
    candidates: dict[str, dict[str, Any]],
    candidate_profiles: dict[str, dict[str, Any]],
    mongodb_uri: str,
    database_name: str,
    candidates_collection_name: str,
    candidate_profiles_collection_name: str,
) -> dict[str, Any]:
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise RuntimeError("pymongo is required for MongoDB import. Install it before using --execute.") from exc

    result = {
        "candidate_profiles_created": 0,
        "candidate_profiles_updated": 0,
        "candidate_profiles_upserted": 0,
        "candidates_created": 0,
        "candidates_updated": 0,
        "candidates_upserted": 0,
        "errors_import": [],
    }

    client = MongoClient(mongodb_uri)
    try:
        db = client[database_name]
        candidates_collection = db[candidates_collection_name]
        candidate_profiles_collection = db[candidate_profiles_collection_name]
        ensure_candidate_indexes(candidates_collection)
        ensure_candidate_profile_indexes(candidate_profiles_collection)

        for profile_id, incoming in sorted(candidate_profiles.items()):
            try:
                existing = candidate_profiles_collection.find_one({"profile_id": profile_id}, {"_id": 0})
                document = merge_created_at(existing, dict(incoming))
                write_result = candidate_profiles_collection.update_one(
                    {"profile_id": profile_id},
                    {"$set": document},
                    upsert=True,
                )
                if write_result.upserted_id is not None:
                    result["candidate_profiles_created"] += 1
                elif write_result.matched_count:
                    result["candidate_profiles_updated"] += 1
            except Exception as exc:
                result["errors_import"].append({"profile_id": profile_id, "stage": "upsert_candidate_profile", "error": str(exc)})

        for candidate_id, incoming in sorted(candidates.items()):
            try:
                existing = candidates_collection.find_one({"candidate_id": candidate_id}, {"_id": 0})
                document = merge_created_at(existing, dict(incoming))
                write_result = candidates_collection.update_one(
                    {"candidate_id": candidate_id},
                    {"$set": document},
                    upsert=True,
                )
                if write_result.upserted_id is not None:
                    result["candidates_created"] += 1
                elif write_result.matched_count:
                    result["candidates_updated"] += 1
            except Exception as exc:
                result["errors_import"].append({"candidate_id": candidate_id, "stage": "upsert_candidate", "error": str(exc)})
    finally:
        client.close()

    result["candidate_profiles_upserted"] = result["candidate_profiles_created"] + result["candidate_profiles_updated"]
    result["candidates_upserted"] = result["candidates_created"] + result["candidates_updated"]
    return result


def build_report(
    *,
    module2_root: Path,
    accepted_path: Path,
    report_path: Path,
    execution_mode: str,
    database_name: str,
    candidates_collection_name: str,
    candidate_profiles_collection_name: str,
    mongodb_uri: str,
    candidates: dict[str, dict[str, Any]],
    candidate_profiles: dict[str, dict[str, Any]],
    analysis: dict[str, Any],
    import_result: dict[str, Any] | None,
) -> dict[str, Any]:
    import_errors = list(analysis.get("errors_import") or [])
    if import_result:
        import_errors.extend(import_result.get("errors_import") or [])

    profile_kind_counts = analysis.get("profile_kind_counts") or {}
    execution_ran = execution_mode == "execute"

    return {
        "generated_at_utc": utc_now(),
        "execution_mode": execution_mode,
        "strategy_version": "dedup_v2_two_collections",
        "module2_root": str(module2_root),
        "accepted_path": str(accepted_path),
        "report_path": str(report_path),
        "database_used": database_name,
        "mongodb_uri": mask_mongodb_uri(mongodb_uri),
        "candidates_collection_used": candidates_collection_name,
        "candidate_profiles_collection_used": candidate_profiles_collection_name,
        "total_json_read": analysis.get("total_json_read", 0),
        "success_profiles": analysis.get("success_profiles", 0),
        "success_ready_for_import": analysis.get("success_profiles", 0),
        "failed_ignored": analysis.get("failed_ignored", 0),
        "candidate_profiles_to_import": len(candidate_profiles),
        "candidates_to_create": len(candidates),
        "real_email_count": analysis.get("real_email_count", 0),
        "generic_email_count": analysis.get("generic_email_count", 0),
        "missing_email_count": analysis.get("missing_email_count", 0),
        "phones_detected": analysis.get("phones_detected", 0),
        "real_phone_count": analysis.get("real_phone_count", 0),
        "linkedin_detected_count": analysis.get("linkedin_detected_count", 0),
        "github_detected_count": analysis.get("github_detected_count", 0),
        "strong_merges_count": analysis.get("strong_merges_count", 0),
        "possible_duplicates_count": analysis.get("possible_duplicates_count", 0),
        "conflicts_count": analysis.get("conflicts_count", 0),
        "risky_merges_prevented": analysis.get("risky_merges_prevented", []),
        "groups_auto_merged": analysis.get("groups_auto_merged", []),
        "groups_not_merged": analysis.get("groups_not_merged", []),
        "conflicts": analysis.get("conflicts", []),
        "old_unique_candidates_count": OLD_UNIQUE_CANDIDATES_COUNT,
        "new_candidates_count": len(candidates),
        "profile_documents_count": len(candidate_profiles),
        "complete_profile_ready_for_import": profile_kind_counts.get(COMPLETE_PROFILE, 0),
        "partial_profile_ready_for_import": profile_kind_counts.get(PARTIAL_PROFILE, 0),
        "complete_profile_imported": profile_kind_counts.get(COMPLETE_PROFILE, 0) if execution_ran else 0,
        "partial_profile_imported": profile_kind_counts.get(PARTIAL_PROFILE, 0) if execution_ran else 0,
        "candidate_profiles_created": (import_result or {}).get("candidate_profiles_created", 0),
        "candidate_profiles_updated": (import_result or {}).get("candidate_profiles_updated", 0),
        "candidate_profiles_upserted": (import_result or {}).get("candidate_profiles_upserted", 0),
        "candidates_created": (import_result or {}).get("candidates_created", 0),
        "candidates_updated": (import_result or {}).get("candidates_updated", 0),
        "candidates_upserted": (import_result or {}).get("candidates_upserted", 0),
        "status_counts": analysis.get("status_counts", {}),
        "profile_kind_counts": profile_kind_counts,
        "provider_route_counts": analysis.get("provider_route_counts", {}),
        "errors_import": import_errors,
        "failed_ignored_items": analysis.get("failed_ignored_items", []),
        "recommendation": {
            "can_execute_after_validation": True,
            "summary": (
                "Dry-run uses two collections: every successful Module 2 JSON is preserved in "
                "candidate_profiles, while candidates is a conservative consolidated view. "
                "Generic/template emails are not used as automatic merge keys."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import successful Module 2 candidate profiles into MongoDB with conservative deduplication."
    )
    parser.add_argument("--module2-root", type=Path, default=DEFAULT_MODULE2_ROOT)
    parser.add_argument("--accepted-path", type=Path, default=DEFAULT_ACCEPTED_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--mongodb-uri", default=DEFAULT_MONGODB_URI)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--collection", default=DEFAULT_CANDIDATES_COLLECTION, help="Candidates collection name.")
    parser.add_argument(
        "--profiles-collection",
        default=DEFAULT_CANDIDATE_PROFILES_COLLECTION,
        help="Candidate profiles collection name.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write to MongoDB. Without this flag the script only analyses inputs and writes a dry-run report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module2_root = args.module2_root
    if not module2_root.exists():
        raise FileNotFoundError(f"Module 2 root does not exist: {module2_root}")

    candidates, candidate_profiles, analysis = analyse_profiles(module2_root, args.accepted_path)
    execution_mode = "execute" if args.execute else "dry_run"
    import_result = None
    if args.execute:
        import_result = import_documents(
            candidates=candidates,
            candidate_profiles=candidate_profiles,
            mongodb_uri=args.mongodb_uri,
            database_name=args.database,
            candidates_collection_name=args.collection,
            candidate_profiles_collection_name=args.profiles_collection,
        )

    report = build_report(
        module2_root=module2_root,
        accepted_path=args.accepted_path,
        report_path=args.report_path,
        execution_mode=execution_mode,
        database_name=args.database,
        candidates_collection_name=args.collection,
        candidate_profiles_collection_name=args.profiles_collection,
        mongodb_uri=args.mongodb_uri,
        candidates=candidates,
        candidate_profiles=candidate_profiles,
        analysis=analysis,
        import_result=import_result,
    )
    write_json(args.report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
