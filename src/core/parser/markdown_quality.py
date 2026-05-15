from __future__ import annotations

import re
from dataclasses import dataclass


USEFUL_SECTION_PATTERNS = {
    "experience": re.compile(r"\b(experience|work experience|professional experience|internship|employment)\b", re.I),
    "education": re.compile(r"\b(education|formation|degree|university|school|baccalaur|diploma)\b", re.I),
    "skills": re.compile(r"\b(skills|technical skills|competences|programming|tools|frameworks)\b", re.I),
}

MOJIBAKE_REPLACEMENTS = {
    "â€“": "-",
    "â€”": "-",
    "â€˜": "'",
    "â€™": "'",
    "â€œ": '"',
    "â€": '"',
    "Â·": " ",
    "â€¢": " ",
    "â‹„": " ",
    "Ã©": "e",
    "Ã¨": "e",
    "Ãª": "e",
    "Ã ": "a",
    "Ã¢": "a",
    "Ã´": "o",
    "Ã§": "c",
    "ï¼ˆ": "(",
    "ï¼‰": ")",
}

HEADING_REPLACEMENTS = {
    "RESUMEOBJECTIVE": "RESUME OBJECTIVE",
    "PROFESSIONALEXPERIENCE": "PROFESSIONAL EXPERIENCE",
    "PROFESSIONALSUMMARY": "PROFESSIONAL SUMMARY",
    "TECHNICALSKILLS": "TECHNICAL SKILLS",
    "ACADEMICPROJECTS": "ACADEMIC PROJECTS",
    "SOFTSKILLS": "SOFT SKILLS",
    "CORECOMPETENCIES": "CORE COMPETENCIES",
}

OCR_NOISE_LINE_RE = re.compile(
    r"^(<!--\s*image\s*-->|image|page\s+\d+|blank|copyright|usage guidelines)$",
    re.I,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")
BROKEN_LINE_END_RE = re.compile(r"\b\w{1,3}-$")


@dataclass(frozen=True)
class MarkdownQualityResult:
    markdown: str
    signals: dict[str, int | float | bool]
    warnings: list[str]


def clean_markdown_for_module2(markdown: str, *, source_format: str) -> MarkdownQualityResult:
    """Clean OCR markdown and expose quality signals before Module 2 handoff."""
    cleaned = _normalize_mojibake(markdown)
    cleaned = _fix_glued_text(cleaned)
    cleaned = _clean_lines(cleaned)
    signals = diagnose_markdown_quality(cleaned)
    warnings = _warnings_from_signals(signals, source_format=source_format)
    return MarkdownQualityResult(markdown=cleaned, signals=signals, warnings=warnings)


def diagnose_markdown_quality(markdown: str) -> dict[str, int | float | bool]:
    """Measure markdown quality with signals that are cheap and stable."""
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    text = "\n".join(lines)
    words = TOKEN_RE.findall(text)
    weird_chars = sum(1 for char in text if ord(char) > 127 or char in {"�", "§", "¤"})
    glued_lines = sum(1 for line in lines if _looks_glued(line))
    broken_lines = sum(1 for line in lines if BROKEN_LINE_END_RE.search(line))
    useful_sections = {
        name: bool(pattern.search(text))
        for name, pattern in USEFUL_SECTION_PATTERNS.items()
    }
    useful_section_count = sum(1 for found in useful_sections.values() if found)
    noise_tokens = sum(1 for token in words if _is_noisy_token(token))

    return {
        "markdown_chars": len(text),
        "markdown_words": len(words),
        "markdown_has_experience": useful_sections["experience"],
        "markdown_has_education": useful_sections["education"],
        "markdown_has_skills": useful_sections["skills"],
        "markdown_useful_section_count": useful_section_count,
        "markdown_weird_char_ratio": round(weird_chars / max(len(text), 1), 4),
        "markdown_glued_line_count": glued_lines,
        "markdown_broken_line_count": broken_lines,
        "markdown_ocr_noise_ratio": round(noise_tokens / max(len(words), 1), 4),
    }


def _normalize_mojibake(text: str) -> str:
    cleaned = text
    if "Ãƒ" in cleaned or "Ã‚" in cleaned:
        try:
            cleaned = cleaned.encode("latin1").decode("utf-8")
        except Exception:
            pass
    for src, dst in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(src, dst)
    return cleaned


def _fix_glued_text(text: str) -> str:
    cleaned = text
    for src, dst in HEADING_REPLACEMENTS.items():
        cleaned = re.sub(src, dst, cleaned, flags=re.I)
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"(?<=[a-zA-Z])(?=\d)", " ", cleaned)
    cleaned = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", cleaned)
    cleaned = re.sub(r"(?<=\w),(?=\w)", ", ", cleaned)
    cleaned = re.sub(r"(?<=\w)\.(?=\w)", ". ", cleaned)
    cleaned = re.sub(r"\s*/\s*", "/", cleaned)
    cleaned = re.sub(r"\s*&\s*", " & ", cleaned)
    return cleaned


def _clean_lines(markdown: str) -> str:
    cleaned_lines: list[str] = []
    previous_blank = False
    for raw_line in markdown.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        previous_blank = False

        stripped = line.strip("#*- ")
        if OCR_NOISE_LINE_RE.match(stripped):
            continue
        if _is_low_value_noise_line(stripped):
            continue
        if line.startswith("#"):
            line = _normalize_heading(line)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _normalize_heading(line: str) -> str:
    marker = re.match(r"^(#+)\s*", line)
    prefix = marker.group(1) if marker else "##"
    title = re.sub(r"^#+\s*", "", line).strip()
    for src, dst in HEADING_REPLACEMENTS.items():
        title = re.sub(src, dst, title, flags=re.I)
    if len(title) > 80:
        return title
    return f"{prefix} {title}".strip()


def _looks_glued(line: str) -> bool:
    if len(line) < 18:
        return False
    compact_long_words = re.findall(r"[A-Za-z]{14,}", line)
    if compact_long_words:
        return True
    return bool(re.search(r"[a-z]{5,}(with|and|from|to|of|in)[a-z]{5,}", line, re.I))


def _is_noisy_token(token: str) -> bool:
    alpha = "".join(char for char in token if char.isalpha())
    if len(alpha) >= 8:
        vowels = sum(char.lower() in "aeiouy" for char in alpha)
        return vowels / len(alpha) < 0.2
    return False


def _is_low_value_noise_line(line: str) -> bool:
    if len(line) <= 2:
        return True
    if len(line) <= 20 and re.fullmatch(r"[\W_0-9]+", line):
        return True
    return False


def _warnings_from_signals(signals: dict[str, int | float | bool], *, source_format: str) -> list[str]:
    warnings: list[str] = []
    chars = int(signals["markdown_chars"])
    words = int(signals["markdown_words"])
    useful_sections = int(signals["markdown_useful_section_count"])
    weird_ratio = float(signals["markdown_weird_char_ratio"])
    glued_lines = int(signals["markdown_glued_line_count"])
    ocr_noise_ratio = float(signals["markdown_ocr_noise_ratio"])

    if chars < 900 or words < 120 or useful_sections < 2:
        warnings.append("markdown_too_weak")
    if weird_ratio > 0.025 or ocr_noise_ratio > 0.18 or (
        glued_lines >= 10 and (words < 180 or useful_sections < 2)
    ):
        warnings.append("high_ocr_noise")
    if source_format in {"images", "scans"} and (
        "markdown_too_weak" in warnings
        or ("high_ocr_noise" in warnings and (words < 180 or useful_sections < 2))
    ):
        warnings.append("needs_ocr_repair")
    return warnings
