from __future__ import annotations

import copy
import re
from typing import Any

from src.core.common.seniority import normalize_seniority

from .markdown_normalizer import detect_template_values, fix_tech_terms


GENERIC_SUMMARY_PHRASES = (
    "with experience in",
    "expertise in",
    "professional with",
    "results-driven",
    "highly motivated",
    "strong background",
)

CURRENT_YEAR = 2026
SUMMARY_SUPPORT_THRESHOLD = 0.6
RESPONSIBILITY_SUPPORT_THRESHOLD = 0.4
MEANINGFUL_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "and",
    "are",
    "been",
    "being",
    "built",
    "from",
    "have",
    "into",
    "more",
    "over",
    "that",
    "than",
    "their",
    "them",
    "then",
    "they",
    "this",
    "using",
    "with",
    "your",
}
DATE_RANGE_PATTERN = re.compile(
    r"(?P<start>(?:0?[1-9]|1[0-2])[/-]\d{4}|\d{4})"
    r"\s*(?:-|\u2013|\u2014|to)\s*"
    r"(?P<end>(?:0?[1-9]|1[0-2])[/-]\d{4}|\d{4}|present|current|ongoing)",
    re.I,
)


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9+.#/@-]+", " ", str(value or "").lower()).strip()


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _tokens(value: Any) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9][a-z0-9+.#/@-]*", str(value or "").lower()) if len(token) > 1]


def _meaningful_tokens(value: Any) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9+.#/@-]*", str(value or "").lower())
        if len(token) > 3 and token not in MEANINGFUL_STOPWORDS
    ]
    return tokens


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _template_like(value: Any, detected_templates: list[str] | None = None) -> bool:
    text = str(value or "")
    if not text:
        return False
    if detect_template_values(text):
        return True
    lowered = text.lower().strip()
    if lowered in {"null", "none", "unknown", "n/a"}:
        return True
    return any(template.lower() in lowered for template in detected_templates or [])


def check_field_presence(value: Any, source_text: str) -> str:
    if _is_missing(value):
        return "missing"
    if isinstance(value, list):
        statuses = [check_field_presence(item, source_text) for item in value]
        if not statuses:
            return "missing"
        if all(status == "supported" for status in statuses):
            return "supported"
        if any(status in {"supported", "weakly_supported"} for status in statuses):
            return "weakly_supported"
        return "unsupported"
    text = str(value).strip()
    source_norm = _normalize(source_text)
    value_norm = _normalize(text)
    source_compact = _compact(source_text)
    value_compact = _compact(text)
    if not value_norm:
        return "missing"
    if "@" in text or "linkedin.com" in text.lower() or "github.com" in text.lower():
        return "supported" if value_compact and value_compact in source_compact else "unsupported"
    if value_compact and value_compact in source_compact:
        return "supported"
    value_tokens = [token for token in _tokens(text) if token not in {"and", "the", "with", "for", "from"}]
    if not value_tokens:
        return "missing"
    present = sum(1 for token in set(value_tokens) if token in source_norm)
    coverage = present / max(1, len(set(value_tokens)))
    if coverage >= 0.8:
        return "weakly_supported"
    return "unsupported"


def extract_years_from_dates(source_text: str) -> float | None:
    experience_text = _extract_experience_source_text(source_text)
    intervals: list[tuple[int, int]] = []
    for candidate_text in [experience_text, source_text or ""]:
        for match in DATE_RANGE_PATTERN.finditer(candidate_text):
            start = _parse_date_token(match.group("start"), is_end=False)
            end = _parse_date_token(match.group("end"), is_end=True)
            if start is None or end is None or end < start:
                continue
            intervals.append((start, end))
        if intervals:
            break
    if not intervals:
        return None

    merged = _merge_month_intervals(intervals)
    total_months = sum((end - start + 1) for start, end in merged)
    if total_months <= 0:
        return None
    return round(total_months / 12.0, 1)


def parse_experience_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return None
            if text in {"present", "current", "ongoing"}:
                return None
            date_range = DATE_RANGE_PATTERN.search(text)
            if date_range:
                start_token = date_range.group("start")
                end_token = date_range.group("end")
                if re.fullmatch(r"\d{4}", start_token) and re.fullmatch(r"\d{4}", end_token):
                    return max(0.0, float(int(end_token) - int(start_token)))
                start = _parse_date_token(date_range.group("start"), is_end=False)
                end = _parse_date_token(date_range.group("end"), is_end=True)
                if start is not None and end is not None and end >= start:
                    return round((end - start + 1) / 12.0, 1)

            cleaned = text
            cleaned = cleaned.replace("ans", "")
            cleaned = cleaned.replace("years", "")
            cleaned = cleaned.replace("year", "")
            cleaned = cleaned.replace("yr", "")
            cleaned = cleaned.replace("yrs", "")
            cleaned = cleaned.replace(" ", "")

            range_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[-/]\s*(\d+(?:\.\d+)?)", cleaned)
            if range_match:
                low = float(range_match.group(1))
                high = float(range_match.group(2))
                return round((low + high) / 2.0, 1)

            cleaned = cleaned.replace("+", "")
            numeric_match = re.search(r"\d+(?:\.\d+)?", cleaned)
            if not numeric_match:
                return None
            return float(numeric_match.group(0))
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_float(value: Any) -> float | None:
    return parse_experience_value(value)


def infer_experience_level(years) -> str | None:
    if years is None:
        return None
    years_value = parse_experience_value(years)
    if years_value is None:
        return None
    if years_value < 2:
        return "junior"
    elif years_value <= 5:
        return "mid_level"
    else:
        return "senior"


def _extract_experience_source_text(source_text: str) -> str:
    lines = (source_text or "").splitlines()
    capture = False
    collected: list[str] = []
    stop_titles = {
        "academic projects",
        "languages",
        "certifications",
        "certificates",
        "soft skills",
        "interests",
    }
    for line in lines:
        stripped = line.strip()
        normalized = stripped.lstrip("#").strip()
        if re.fullmatch(r"(?:professional experience|work experience|experience)\b", normalized, re.I):
            capture = True
            continue
        if not capture:
            continue
        lowered = normalized.lower()
        if lowered in stop_titles:
            break
        if stripped.startswith("## "):
            title = stripped[3:].strip().lower()
            if title in stop_titles:
                break
        collected.append(line)
    if collected:
        return "\n".join(collected)
    return source_text or ""


def _parse_date_token(value: str, *, is_end: bool) -> int | None:
    token = str(value or "").strip().lower()
    if token in {"present", "current", "ongoing"}:
        year = CURRENT_YEAR
        month = 12 if is_end else 1
        return year * 12 + month
    if re.fullmatch(r"(?:0?[1-9]|1[0-2])[/-]\d{4}", token):
        month_str, year_str = re.split(r"[/-]", token)
        month = int(month_str)
        year = int(year_str)
        return year * 12 + month
    if re.fullmatch(r"\d{4}", token):
        year = int(token)
        month = 12 if is_end else 1
        return year * 12 + month
    return None


def _merge_month_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _compute_years_experience_value(source_text: str) -> int | None:
    total_years = extract_years_from_dates(source_text)
    if total_years is None or total_years <= 0 or total_years > 40:
        return None
    if total_years < 1:
        return 1
    return max(1, int(total_years + 0.5))


def _hydrate_experience_metadata(profile: dict[str, Any], source_text: str) -> None:
    expertise = profile.setdefault("expertise", {})
    current_years = expertise.get("years_experience")
    if current_years in (None, "", [], {}):
        expertise["years_experience"] = _compute_years_experience_value(source_text)
    if expertise.get("experience_level") in (None, "", [], {}):
        expertise["experience_level"] = infer_experience_level(expertise.get("years_experience"))
    else:
        expertise["experience_level"] = normalize_seniority(expertise.get("experience_level"))


def _normalized_skill_presence(skill: Any, source_text: str) -> str:
    if _is_missing(skill):
        return "missing"
    normalized_skill = fix_tech_terms(str(skill or ""))
    normalized_source = fix_tech_terms(source_text or "")
    return check_field_presence(normalized_skill, normalized_source)


def _summary_support_status(summary: Any, source_text: str) -> str:
    if _is_missing(summary):
        return "missing"
    summary_text = str(summary or "").strip()
    source_norm = _normalize(source_text)
    if _compact(summary_text) and _compact(summary_text) in _compact(source_text):
        return "supported"
    tokens = set(_meaningful_tokens(summary_text))
    if not tokens:
        return "unsupported"
    present = sum(1 for token in tokens if token in source_norm)
    coverage = present / max(1, len(tokens))
    if coverage >= SUMMARY_SUPPORT_THRESHOLD:
        return "supported"
    if coverage >= 0.4:
        return "weakly_supported"
    return "unsupported"


def _responsibility_support_status(responsibility: Any, source_text: str) -> str:
    if _is_missing(responsibility):
        return "missing"
    if check_field_presence(responsibility, source_text) == "supported":
        return "supported"
    responsibility_tokens = set(_meaningful_tokens(responsibility))
    if not responsibility_tokens:
        return "unsupported"
    experience_source = _normalize(_extract_experience_source_text(source_text))
    present = sum(1 for token in responsibility_tokens if token in experience_source)
    coverage = present / max(1, len(responsibility_tokens))
    if coverage >= RESPONSIBILITY_SUPPORT_THRESHOLD:
        return "weakly_supported"
    return "unsupported"


def nullify_template_fields(profile: dict[str, Any], detected_templates: list[str]) -> tuple[dict[str, Any], list[str]]:
    grounded = copy.deepcopy(profile)
    nullified: list[str] = []

    def walk(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {key: walk(item, f"{path}.{key}" if path else key) for key, item in value.items()}
        if isinstance(value, list):
            cleaned = []
            for index, item in enumerate(value):
                if isinstance(item, str) and _template_like(item, detected_templates):
                    nullified.append(f"{path}[{index}]")
                    continue
                cleaned.append(walk(item, f"{path}[{index}]"))
            return cleaned
        if isinstance(value, str) and _template_like(value, detected_templates):
            nullified.append(path)
            return None
        return value

    return walk(grounded, ""), nullified


def nullify_unsupported_fields(profile: dict[str, Any], source_text: str) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    grounded = copy.deepcopy(profile)
    nullified: list[str] = []
    supported: list[str] = []
    unsupported: list[str] = []

    def check_scalar(container: dict[str, Any], key: str, path: str) -> None:
        status = check_field_presence(container.get(key), source_text)
        if status == "supported":
            supported.append(path)
        elif status == "weakly_supported":
            supported.append(path)
        elif status == "unsupported":
            unsupported.append(path)
            nullified.append(path)
            container[key] = None

    bio = grounded.setdefault("bio", {})
    for key in ("full_name", "email", "phone", "location", "linkedin", "github"):
        check_scalar(bio, key, f"bio.{key}")

    expertise = grounded.setdefault("expertise", {})
    summary = expertise.get("summary")
    summary_status = _summary_support_status(summary, source_text)
    if summary_status == "unsupported" or (summary_status == "weakly_supported" and _summary_too_generic(summary)):
        unsupported.append("expertise.summary")
        nullified.append("expertise.summary")
        expertise["summary"] = None
    elif summary_status in {"supported", "weakly_supported"}:
        supported.append("expertise.summary")

    derived_years = _compute_years_experience_value(source_text)
    if expertise.get("years_experience") not in (None, "", [], {}):
        supported.append("expertise.years_experience")
    elif derived_years is not None:
        expertise["years_experience"] = derived_years
        supported.append("expertise.years_experience")
    else:
        expertise["years_experience"] = None

    if expertise.get("experience_level") not in (None, "", [], {}):
        expertise["experience_level"] = normalize_seniority(expertise.get("experience_level"))
        if expertise.get("years_experience") not in (None, "", [], {}):
            supported.append("expertise.experience_level")
        else:
            expertise["experience_level"] = None
    else:
        inferred_level = infer_experience_level(expertise.get("years_experience"))
        if inferred_level:
            expertise["experience_level"] = inferred_level
            supported.append("expertise.experience_level")
        else:
            expertise["experience_level"] = None

    for key in ("hard_skills", "soft_skills"):
        cleaned = []
        for index, item in enumerate(expertise.get(key) or []):
            path = f"expertise.{key}[{index}]"
            status = _normalized_skill_presence(item, source_text) if key == "hard_skills" else check_field_presence(item, source_text)
            if status in {"supported", "weakly_supported"}:
                supported.append(path)
                cleaned.append(item)
            else:
                unsupported.append(path)
                nullified.append(path)
        expertise[key] = cleaned

    cleaned_experiences = []
    for exp_index, experience in enumerate(grounded.get("experiences") or []):
        if not isinstance(experience, dict):
            continue
        item = copy.deepcopy(experience)
        for key in ("company", "job_title", "start_date", "end_date", "city"):
            check_scalar(item, key, f"experiences[{exp_index}].{key}")
        responsibilities = []
        for resp_index, responsibility in enumerate(item.get("responsibilities") or []):
            path = f"experiences[{exp_index}].responsibilities[{resp_index}]"
            status = _responsibility_support_status(responsibility, source_text)
            if status in {"supported", "weakly_supported"}:
                supported.append(path)
                responsibilities.append(responsibility)
            else:
                unsupported.append(path)
                nullified.append(path)
        item["responsibilities"] = responsibilities
        if _count_supported_item_fields(item) >= 1:
            cleaned_experiences.append(item)
        else:
            nullified.append(f"experiences[{exp_index}]")
    grounded["experiences"] = cleaned_experiences

    cleaned_education = []
    for edu_index, education in enumerate(grounded.get("education") or []):
        if not isinstance(education, dict):
            continue
        item = copy.deepcopy(education)
        for key in ("institution", "degree", "field", "year"):
            check_scalar(item, key, f"education[{edu_index}].{key}")
        if _count_supported_item_fields(item) >= 1:
            cleaned_education.append(item)
        else:
            nullified.append(f"education[{edu_index}]")
    grounded["education"] = cleaned_education

    for key in ("languages", "certifications"):
        cleaned = []
        for index, item in enumerate(grounded.get(key) or []):
            path = f"{key}[{index}]"
            status = check_field_presence(item, source_text)
            if status in {"supported", "weakly_supported"}:
                supported.append(path)
                cleaned.append(item)
            else:
                unsupported.append(path)
                nullified.append(path)
        grounded[key] = cleaned

    return grounded, sorted(set(nullified)), sorted(set(supported)), sorted(set(unsupported))


def _summary_too_generic(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return len(text.split()) < 4 or any(phrase in text for phrase in GENERIC_SUMMARY_PHRASES)


def _count_supported_item_fields(item: dict[str, Any]) -> int:
    count = 0
    for value in item.values():
        if isinstance(value, list):
            count += len([part for part in value if part])
        elif value not in (None, "", [], {}):
            count += 1
    return count


def compute_reliability_score(
    supported_count: int,
    total_checked: int,
    document_confidence_score: float,
    nullified_count: int,
    unsupported_count: int,
) -> float:
    base = supported_count / max(total_checked, 1)
    confidence_value = _safe_float(document_confidence_score)
    score = base * 0.8 + (confidence_value if confidence_value is not None else 0.0) * 0.2
    score -= 0.05 * nullified_count
    score -= 0.08 * unsupported_count
    return round(max(0.0, min(1.0, score)), 4)


def determine_profile_kind(
    reliability_score: float,
    supported_count: int,
    llm_status: str | None,
) -> str:
    if llm_status == "unreadable":
        return "unreadable"
    if reliability_score >= 0.75 and supported_count >= 6:
        return "complete_profile"
    if reliability_score >= 0.45 and supported_count >= 3:
        return "partial_profile"
    return "minimal_profile"


def determine_hallucination_risk(fields_nullified: list[str], fields_unsupported: list[str]) -> str:
    unsupported_count = len(set(fields_unsupported))
    if unsupported_count == 0:
        return "low"
    if unsupported_count <= 2:
        return "medium"
    return "high"


def validate_and_ground(
    llm_output: dict[str, Any],
    cleaned_markdown: str,
    detected_templates: list[str],
    document_confidence_score: float,
) -> dict[str, Any]:
    profile = _coerce_profile(llm_output)
    _hydrate_experience_metadata(profile, cleaned_markdown)
    quality_flags: list[str] = []
    if detected_templates:
        quality_flags.append("template_detected")

    profile, template_nullified = nullify_template_fields(profile, detected_templates)
    grounded, unsupported_nullified, supported, unsupported = nullify_unsupported_fields(profile, cleaned_markdown)
    fields_nullified = sorted(set(template_nullified + unsupported_nullified))
    total_checked = len(set(supported) | set(unsupported) | set(fields_nullified))
    reliability_score = compute_reliability_score(
        len(supported),
        total_checked,
        document_confidence_score,
        len(fields_nullified),
        len(unsupported),
    )
    if grounded.get("expertise", {}).get("summary") is None:
        quality_flags.append("summary_not_enough_evidence")
    if fields_nullified:
        quality_flags.append("unsupported_fields_nullified")
    if reliability_score < 0.45:
        quality_flags.append("low_grounded_reliability")

    llm_status = str(llm_output.get("status") or "partial")
    profile_kind = determine_profile_kind(reliability_score, len(supported), llm_status)
    hallucination_risk = determine_hallucination_risk(fields_nullified, unsupported)
    grounded["status"] = llm_status

    return {
        "grounded_profile": grounded,
        "reliability_score": reliability_score,
        "profile_kind": profile_kind,
        "quality_flags": sorted(set(quality_flags)),
        "fields_nullified": fields_nullified,
        "fields_supported": sorted(set(supported)),
        "fields_unsupported": sorted(set(unsupported)),
        "hallucination_risk": hallucination_risk,
    }


def _coerce_profile(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "bio": {
            "full_name": (output.get("bio") or {}).get("full_name"),
            "email": (output.get("bio") or {}).get("email"),
            "phone": (output.get("bio") or {}).get("phone"),
            "location": (output.get("bio") or {}).get("location"),
            "linkedin": (output.get("bio") or {}).get("linkedin"),
            "github": (output.get("bio") or {}).get("github"),
        },
        "expertise": {
            "summary": (output.get("expertise") or {}).get("summary"),
            "experience_level": (output.get("expertise") or {}).get("experience_level"),
            "years_experience": (output.get("expertise") or {}).get("years_experience"),
            "hard_skills": _string_list((output.get("expertise") or {}).get("hard_skills")),
            "soft_skills": _string_list((output.get("expertise") or {}).get("soft_skills")),
        },
        "experiences": output.get("experiences") if isinstance(output.get("experiences"), list) else [],
        "education": output.get("education") if isinstance(output.get("education"), list) else [],
        "languages": _string_list(output.get("languages")),
        "certifications": _string_list(output.get("certifications")),
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, (str, int, float)) and str(item).strip()]
