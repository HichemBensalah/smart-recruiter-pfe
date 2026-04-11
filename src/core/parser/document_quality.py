from __future__ import annotations

import re
from pathlib import Path


EXPECTED_SECTIONS = {
    "cv": [
        "HEADER",
        "PROFESSIONAL SUMMARY",
        "EDUCATION",
        "TECHNICAL SKILLS",
        "PROFESSIONAL EXPERIENCE",
        "ACADEMIC PROJECTS",
    ],
    "job_description": [
        "HEADER",
        "JOB SUMMARY",
        "RESPONSIBILITIES",
        "REQUIREMENTS",
        "TECHNICAL SKILLS",
    ],
}

ALLOWED_EXTRA_SECTIONS = {
    "cv": {
        "LANGUAGES",
        "SOFT SKILLS",
        "CERTIFICATES (IN PROGRESS)",
        "MACHINE LEARNING & DEEP LEARNING",
        "COMPUTER VISION & NATURAL LANGUAGE PROCESSING (NLP)",
        "IDES & SOFTWARE",
        "EDUCATION AND TRAINING",
        "WEBSITES, PORTFOLIOS, PROFILES",
    },
    "job_description": {
        "BENEFITS",
        "CONTEXT",
        "LOCATION",
        "COMPANY",
        "PROCESS",
    },
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")
NOISY_TOKEN_RE = re.compile(r"(?i)([bcdfghjklmnpqrstvwxyz]{6,}|[A-Z]{8,}|[^\w\s]{2,})")

OCR_VALIDATED_CONFIDENCE = 0.58
OCR_VALIDATED_TEXT_CHARS = 500
OCR_VALIDATED_SECTION_COMPLETENESS = 0.25
OCR_PARTIAL_CONFIDENCE = 0.38
OCR_PARTIAL_TEXT_CHARS = 250
OCR_PARTIAL_SECTION_COMPLETENESS = 0.15
NATIVE_VALIDATED_CONFIDENCE = 0.72
NATIVE_VALIDATED_TEXT_CHARS = 800
NATIVE_VALIDATED_SECTION_COMPLETENESS = 0.45
NATIVE_PARTIAL_CONFIDENCE = 0.42
NATIVE_PARTIAL_TEXT_CHARS = 300
NATIVE_PARTIAL_SECTION_COMPLETENESS = 0.18


def infer_document_type(path: str | Path) -> str:
    parts = [part.lower() for part in Path(path).parts]
    if any(part in {"jobs", "job", "job_descriptions", "fiches_poste", "poste"} for part in parts):
        return "job_description"
    return "cv"


def structure_to_text(structure: dict) -> str:
    chunks: list[str] = []
    for section in structure.get("sections", []):
        title = (section.get("title") or "").strip()
        if title:
            chunks.append(title)
        for line in section.get("lines", []):
            chunks.append(line["text"] if isinstance(line, dict) else str(line))
        for item in section.get("items", []):
            title = (item.get("title") or "").strip()
            if title:
                chunks.append(title)
            if item.get("date"):
                chunks.append(str(item["date"]))
            if item.get("location"):
                chunks.append(str(item["location"]))
            for detail in item.get("details", []):
                chunks.append(str(detail))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _noise_ratio(text: str) -> float:
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return 1.0
    noisy = 0
    for token in tokens:
        if NOISY_TOKEN_RE.search(token):
            noisy += 1
            continue
        alpha = "".join(ch for ch in token if ch.isalpha())
        if len(alpha) >= 8:
            vowels = sum(ch.lower() in "aeiouy" for ch in alpha)
            if vowels / len(alpha) < 0.2:
                noisy += 1
    return min(noisy / max(len(tokens), 1), 1.0)


def _ordering_score(titles: list[str], expected_titles: list[str]) -> float:
    positions = [expected_titles.index(title) for title in titles if title in expected_titles]
    if len(positions) <= 1:
        return 1.0 if positions else 0.0

    inversions = 0
    comparisons = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            comparisons += 1
            if positions[i] > positions[j]:
                inversions += 1

    if comparisons == 0:
        return 1.0
    return max(0.0, 1.0 - (inversions / comparisons))


def _section_text_length(section: dict) -> int:
    total = 0
    for line in section.get("lines", []):
        text = line["text"] if isinstance(line, dict) else str(line)
        total += len(text.strip())
    for item in section.get("items", []):
        total += len(str(item.get("title") or "").strip())
        total += len(str(item.get("date") or "").strip())
        total += len(str(item.get("location") or "").strip())
        for detail in item.get("details", []):
            total += len(str(detail).strip())
    return total


def _weird_section_ratio(section_titles: list[str], document_type: str) -> float:
    allowed = set(EXPECTED_SECTIONS.get(document_type, EXPECTED_SECTIONS["cv"]))
    allowed.update(ALLOWED_EXTRA_SECTIONS.get(document_type, set()))
    weird = 0

    for title in section_titles:
        stripped = (title or "").strip()
        if not stripped:
            weird += 1
            continue
        if stripped in allowed:
            continue
        if len(stripped) > 48 or re.search(r"\d{4,}", stripped) or re.search(r"[^\w\s,&()/.-]", stripped):
            weird += 1
            continue
        if len(stripped.split()) > 6:
            weird += 1

    return weird / max(len(section_titles), 1)


def assess_document_payload(
    payload: dict,
    *,
    source_path: str,
    source_format: str,
    document_type: str,
    route_taken: str,
) -> dict:
    structure = payload.get("structure") or {}
    sections = structure.get("sections") or []
    section_titles = [str(section.get("title") or "").strip() for section in sections if section.get("title")]
    full_text = structure_to_text(structure)

    expected_sections = EXPECTED_SECTIONS.get(document_type, EXPECTED_SECTIONS["cv"])
    expected_hits = sum(1 for title in expected_sections if title in section_titles)
    completeness = expected_hits / max(len(expected_sections), 1)

    populated_sections = sum(1 for section in sections if _section_text_length(section) >= 40)
    segmentation_stability = populated_sections / max(len(sections), 1) if sections else 0.0

    useful_text_chars = len(full_text)
    useful_text_score = min(useful_text_chars / 2500.0, 1.0)

    blocks = payload.get("blocks") or []
    if blocks:
        block_score = min(len(blocks) / 32.0, 1.0)
    elif source_format == "docx" and useful_text_chars >= 900:
        block_score = 0.55
    else:
        block_score = 0.0

    ordering = _ordering_score(section_titles, expected_sections)
    noise_ratio = _noise_ratio(full_text)
    weird_section_ratio = _weird_section_ratio(section_titles, document_type)

    noise_penalty = noise_ratio * 0.7 if source_format in {"images", "scans"} else noise_ratio * 0.2

    confidence = (
        0.28 * useful_text_score
        + 0.24 * completeness
        + 0.18 * ordering
        + 0.16 * segmentation_stability
        + 0.14 * block_score
        - 0.18 * noise_penalty
        - 0.12 * weird_section_ratio
    )
    confidence = max(0.0, min(confidence, 1.0))

    warnings: list[str] = []
    missing_sections = [title for title in expected_sections if title not in section_titles]

    if useful_text_chars < 350:
        warnings.append("very_low_text_recovery")
    elif useful_text_chars < 900:
        warnings.append("low_text_recovery")

    if completeness < 0.35:
        warnings.append("low_section_completeness")
    elif completeness < 0.55:
        warnings.append("partial_section_completeness")

    if ordering < 0.6:
        warnings.append("low_section_order_coherence")

    if segmentation_stability < 0.45:
        warnings.append("unstable_segmentation")

    noise_warning_threshold = 0.28 if source_format in {"images", "scans"} else 0.45
    if noise_ratio > noise_warning_threshold:
        warnings.append("high_ocr_noise")

    if weird_section_ratio > 0.35:
        warnings.append("abnormal_section_titles")

    document_status = classify_document_status(
        confidence=confidence,
        useful_text_chars=useful_text_chars,
        section_completeness=completeness,
        warnings=warnings,
        source_format=source_format,
    )

    return {
        "document_type": document_type,
        "route_taken": route_taken,
        "document_confidence_score": round(confidence, 4),
        "status_candidate": document_status,
        "signals": {
            "useful_text_chars": useful_text_chars,
            "section_completeness": round(completeness, 4),
            "ordering_coherence": round(ordering, 4),
            "segmentation_stability": round(segmentation_stability, 4),
            "layout_block_coverage": round(block_score, 4),
            "ocr_noise_ratio": round(noise_ratio, 4),
            "weird_section_ratio": round(weird_section_ratio, 4),
            "section_count": len(section_titles),
            "expected_section_hits": expected_hits,
        },
        "missing_sections": missing_sections,
        "warnings": warnings,
        "source_path": source_path,
        "source_format": source_format,
    }


def classify_document_status(
    *,
    confidence: float,
    useful_text_chars: int,
    section_completeness: float,
    warnings: list[str],
    source_format: str,
) -> str:
    if source_format in {"images", "scans"}:
        if (
            confidence >= OCR_VALIDATED_CONFIDENCE
            and useful_text_chars >= OCR_VALIDATED_TEXT_CHARS
            and section_completeness >= OCR_VALIDATED_SECTION_COMPLETENESS
            and "very_low_text_recovery" not in warnings
        ):
            return "validated"
        if (
            confidence >= OCR_PARTIAL_CONFIDENCE
            and useful_text_chars >= OCR_PARTIAL_TEXT_CHARS
            and section_completeness >= OCR_PARTIAL_SECTION_COMPLETENESS
        ):
            return "partial"
        return "uncertain"

    if (
        confidence >= NATIVE_VALIDATED_CONFIDENCE
        and useful_text_chars >= NATIVE_VALIDATED_TEXT_CHARS
        and section_completeness >= NATIVE_VALIDATED_SECTION_COMPLETENESS
    ):
        return "validated"
    if (
        confidence >= NATIVE_PARTIAL_CONFIDENCE
        and useful_text_chars >= NATIVE_PARTIAL_TEXT_CHARS
        and section_completeness >= NATIVE_PARTIAL_SECTION_COMPLETENESS
    ):
        return "partial"
    return "uncertain"


def should_try_fallback(
    quality: dict,
    *,
    source_format: str,
    route_taken: str,
) -> bool:
    if route_taken.startswith("secondary"):
        return False
    if quality["status_candidate"] == "validated":
        return False
    signals = quality["signals"]
    if source_format in {"images", "scans"}:
        return True
    if signals["useful_text_chars"] < 700:
        return True
    if signals["section_completeness"] < 0.4:
        return True
    return False


def choose_best_quality(primary: dict, challenger: dict) -> bool:
    primary_status = primary["status_candidate"]
    challenger_status = challenger["status_candidate"]
    rank = {"validated": 3, "partial": 2, "uncertain": 1}
    if rank[challenger_status] > rank[primary_status]:
        return True
    if rank[challenger_status] < rank[primary_status]:
        return False
    return challenger["document_confidence_score"] > primary["document_confidence_score"] + 0.05
