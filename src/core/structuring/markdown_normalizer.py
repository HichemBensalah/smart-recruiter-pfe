from __future__ import annotations

import re
from typing import Any


SECTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "contact/header": ("header", "contact", "personal information"),
    "summary/profile/objective": ("summary", "profile", "objective", "professional summary"),
    "experience/professional experience/work experience": (
        "experience",
        "professional experience",
        "work experience",
        "employment",
    ),
    "education/formation/academic": ("education", "formation", "academic"),
    "skills/technical skills/competences": ("skills", "technical skills", "competences", "competencies"),
    "projects/academic projects": ("projects", "academic projects"),
    "languages/langues": ("languages", "langues"),
    "certifications/certificates": ("certifications", "certificates"),
}

SECTION_TITLE_WORDS = {
    "header",
    "contact",
    "summary",
    "profile",
    "objective",
    "experience",
    "professional experience",
    "work experience",
    "education",
    "formation",
    "academic",
    "skills",
    "technical skills",
    "projects",
    "academic projects",
    "languages",
    "langues",
    "certifications",
    "certificates",
    "requirements",
}

ROLE_LIKE_WORDS = {
    "engineer",
    "engineering",
    "manager",
    "developer",
    "scientist",
    "consultant",
    "analyst",
    "architect",
    "director",
    "lead",
    "specialist",
    "intern",
    "officer",
    "president",
    "administrator",
    "technician",
    "coordinator",
}

NON_NAME_WORDS = {
    "academic",
    "additional",
    "air",
    "company",
    "communication",
    "conflict",
    "city",
    "clearance",
    "cmpro",
    "doors",
    "engineering",
    "evansville",
    "experience",
    "force",
    "hardware",
    "ibm",
    "information",
    "interests",
    "management",
    "microsoft",
    "navy",
    "office",
    "peer",
    "presentations",
    "process",
    "project",
    "professional",
    "qualifications",
    "quality",
    "review",
    "risk",
    "secret",
    "skills",
    "software",
    "state",
    "summary",
    "systems",
    "technical",
    "university",
    "value",
    "visio",
}

TEMPLATE_REGEXES: list[tuple[str, re.Pattern[str]]] = [
    ("email@youremail.com", re.compile(r"\bemail@youremail\.com\b", re.I)),
    ("first.last@", re.compile(r"\bfirst\.last@", re.I)),
    ("FIRSTLAST", re.compile(r"\bfirst\s*last\b", re.I)),
    ("info@qwikresume.com", re.compile(r"\binfo@qwikresume\.com\b", re.I)),
    ("info@resumeworded.com", re.compile(r"\binfo@resumeworded\.com\b", re.I)),
    ("info@resumekraft.com", re.compile(r"\binfo@resumekraft\.com\b", re.I)),
    ("example.com", re.compile(r"\bexample\.com\b", re.I)),
    ("qwikresume", re.compile(r"\bqwikresume\b", re.I)),
    ("resumeworded", re.compile(r"\bresumeworded\b", re.I)),
    ("resumekraft", re.compile(r"\bresumekraft\b", re.I)),
    ("resumesample", re.compile(r"\bresume\s*sample\b|\bresumesample\b", re.I)),
    ("null_text", re.compile(r'(?<!\w)"?null"?(?!\w)', re.I)),
    ("000-000-0000", re.compile(r"\b0{3}[- .]?0{3}[- .]?0{4}\b")),
]


def remove_ocr_artifacts(text: str) -> str:
    """Remove obvious OCR noise without destroying useful CV tokens."""
    if not text:
        return ""

    cleaned_lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if re.fullmatch(r"(?:[#=@|_\-.*]\s*){4,}", line):
            continue
        line = re.sub(r"([#=@|])\1{2,}", r"\1\1", line)
        line = re.sub(r"(?<!\d)\.{3,}(?!\d)", " ", line)
        line = re.sub(r"\s+([,;:!?])", r"\1", line)
        line = re.sub(r"([,;:!?])(?=\S)", r"\1 ", line)
        line = re.sub(r"\b((?:[A-Za-z]\s+){2,}[A-Za-z])\b", _join_spaced_letters, line)
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _join_spaced_letters(match: re.Match[str]) -> str:
    value = match.group(1)
    letters = value.split()
    if 3 <= len(letters) <= 16 and all(len(part) == 1 for part in letters):
        return "".join(letters)
    return value


def fix_emails_and_urls(text: str) -> str:
    """Repair OCR-introduced spaces inside detected emails and URLs only."""
    if not text:
        return ""

    email_pattern = re.compile(
        r"(?<!\w)"
        r"([A-Za-z0-9_%+-]+(?:\s*\.\s*[A-Za-z0-9_%+-]+)*)"
        r"\s*@\s*"
        r"([A-Za-z0-9-]+(?:\s*\.\s*[A-Za-z0-9-]+)+)"
        r"(?!\w)"
    )
    url_patterns = [
        re.compile(
            r"(?:(?:https?\s*:\s*/\s*/\s*)"
            r"(?:www\s*\.\s*)?"
            r"[A-Za-z0-9.-]+"
            r"(?:\s*/\s*[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+)*)",
            re.I,
        ),
        re.compile(
            r"(?:(?:www|linkedin|github)\s*\.\s*com"
            r"(?:\s*/\s*[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+)*)",
            re.I,
        ),
    ]

    def normalize_email_token(match: re.Match[str]) -> str:
        local = re.sub(r"\s*\.\s*", ".", match.group(1))
        local = re.sub(r"\s+", "", local)
        domain = re.sub(r"\s*\.\s*", ".", match.group(2))
        domain = re.sub(r"\s+", "", domain)
        return f"{local}@{domain}"

    def normalize_url_token(token: str) -> str:
        cleaned = token
        cleaned = re.sub(r"\s*:\s*/\s*/\s*", "://", cleaned)
        cleaned = re.sub(r"\s*([./?=&_%#])\s*", r"\1", cleaned)
        cleaned = re.sub(r"\s*-\s*", "-", cleaned)
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = re.sub(r"(?<!\w)(www|linkedin|github)\.(?=[A-Za-z])", r"\1.", cleaned, flags=re.I)
        return cleaned

    text = email_pattern.sub(normalize_email_token, text)
    for pattern in url_patterns:
        text = pattern.sub(lambda match: normalize_url_token(match.group(0)), text)
    return text


def fix_tech_terms(text: str) -> str:
    replacements = [
        (r"\bFast\s*A\s*P\s*I\b|\bFast\s+AP\s*I\b|\bFast\s+API\b", "FastAPI"),
        (r"\bGit\s+Hub\s+Actions\b", "GitHub Actions"),
        (r"\bGit\s+Hub\b", "GitHub"),
        (r"\bMongo\s+DB\b", "MongoDB"),
        (r"\bMy\s+SQL\b", "MySQL"),
        (r"\bPostgre\s+SQL\b", "PostgreSQL"),
        (r"\bNo\s+SQL\b", "NoSQL"),
        (r"\bPy\s+Torch\b", "PyTorch"),
        (r"\bTensor\s+Flow\b", "TensorFlow"),
        (r"\bNum\s+Py\b", "NumPy"),
        (r"\bPy\s+Spark\b", "PySpark"),
        (r"\bData\s+Frames\b", "DataFrames"),
        (r"\bML\s+Ops\b", "MLOps"),
        (r"\bM\s*Lflow\b", "MLflow"),
        (r"\bDev\s+Ops\b", "DevOps"),
        (r"\bPo\s+Wer\s+BI\b", "PowerBI"),
        (r"\bCNN-Bi\s+LSTM\b", "CNN-BiLSTM"),
        (r"\bBi\s+LSTM\b", "BiLSTM"),
        (r"\bspa\s+Cy\b", "spaCy"),
        (r"\bGlo\s+Ve\b", "GloVe"),
        (r"\bfast\s+Text\b", "fastText"),
        (r"\bLL\s+Ms\b", "LLMs"),
        (r"\bLL\s+M\b", "LLM"),
        (r"\bOpen\s+CV\b", "OpenCV"),
        (r"\bXG\s+Boost\b", "XGBoost"),
        (r"\bSage\s+Maker\b", "SageMaker"),
        (r"\bNeo\s*4\s*j\b", "Neo4j"),
        (r"\bJava\s+Script\b", "JavaScript"),
        (r"\bType\s+Script\b", "TypeScript"),
        (r"\bNode\s+JS\b", "NodeJS"),
        (r"\bReact\s+JS\b", "ReactJS"),
        (r"\bRest\s+API\b", "REST API"),
        (r"\bGraph\s+QL\b", "GraphQL"),
        (r"\bGroq\s+A\s*Pi\b", "Groq API"),
        (r"\bOpen\s+AI\b", "OpenAI"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def fix_common_merged_words(text: str) -> str:
    replacements = [
        (r"\bBuilta\b", "Built a"),
        (r"\bDevelopeda\b", "Developed a"),
        (r"\bdelivereda\b", "delivered a"),
        (r"\btrainingandevaluation\b", "training and evaluation"),
        (r"\blocaldeployment\b", "local deployment"),
        (r"\bwebscrapingmodule\b", "web scraping module"),
        (r"\bproduction-likeenvironment\b", "production-like environment"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def detect_template_values(text: str) -> list[str]:
    found: list[str] = []
    for label, pattern in TEMPLATE_REGEXES:
        if pattern.search(text or ""):
            found.append(label)
    return sorted(set(found))


def detect_sections(text: str) -> list[str]:
    found: list[str] = []
    lowered_lines = [line.strip().lower().strip("#:- ") for line in (text or "").splitlines()]
    lowered_text = "\n".join(lowered_lines)
    for canonical, variants in SECTION_PATTERNS.items():
        if any(re.search(rf"(^|\n)\s*{re.escape(variant)}\s*($|\n|:)", lowered_text) for variant in variants):
            found.append(canonical)
    return found


def extract_header_info(text: str) -> dict[str, str | None]:
    first_lines = [line.strip().strip("#*- ") for line in (text or "").splitlines()[:10] if line.strip()]
    header = "\n".join(first_lines)
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", header)
    phone_match = re.search(r"(?:\+?\d[\d .()/-]{7,}\d)", header)
    linkedin_match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s,;]+", header, re.I)
    github_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/[^\s,;]+", header, re.I)

    full_name: str | None = None
    for line in first_lines:
        candidate = _clean_name_candidate(line)
        if candidate:
            full_name = candidate
            break

    location: str | None = None
    for line in first_lines:
        stripped = line.strip(" -|")
        if email_match and email_match.group(0) in stripped:
            continue
        if phone_match and phone_match.group(0) in stripped:
            continue
        if re.search(r"\b[A-Z][A-Za-z .'-]+,\s*[A-Z][A-Za-z .'-]+\b", stripped):
            location = stripped
            break
        if re.search(r"\b(Tunisia|France|Germany|USA|United States|Canada|Ireland|India|UK|United Kingdom)\b", stripped, re.I):
            location = stripped.lstrip("O0 ").strip()
            break

    return {
        "full_name": full_name,
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0).strip() if phone_match else None,
        "location": location,
        "linkedin": linkedin_match.group(0) if linkedin_match else None,
        "github": github_match.group(0) if github_match else None,
    }


def _clean_name_candidate(line: str) -> str | None:
    candidate = re.sub(r"\bHEADER\b", "", line, flags=re.I).strip(" -|")
    if not candidate or len(candidate) > 80:
        return None
    lowered = candidate.lower()
    if lowered in SECTION_TITLE_WORDS:
        return None
    if detect_template_values(candidate):
        return None
    if "@" in candidate or re.search(r"\d", candidate):
        return None
    words = candidate.split()
    if not 2 <= len(words) <= 5:
        return None
    candidate_words = [re.sub(r"[^A-Za-z'-]+", "", word).lower() for word in words]
    if any(word in ROLE_LIKE_WORDS for word in candidate_words if word):
        return None
    if any(word in NON_NAME_WORDS for word in candidate_words if word):
        return None
    if not all(re.search(r"[A-Za-z]", word) for word in words):
        return None
    return candidate


def compute_quality_score(
    cleaned_markdown: str,
    detected_templates: list[str],
    sections_found: list[str],
    header_info: dict[str, Any],
) -> tuple[float, list[str]]:
    score = 1.0
    flags: list[str] = []
    if detected_templates:
        score -= 0.15 * len(detected_templates)
        flags.append("template_detected")
    if not header_info.get("full_name"):
        flags.append("missing_full_name")
    if not header_info.get("email"):
        flags.append("missing_email")
    missing_critical = {
        "experience": not any("experience" in section for section in sections_found),
        "skills": not any("skills" in section or "competences" in section for section in sections_found),
        "education": not any("education" in section or "formation" in section for section in sections_found),
    }
    for name, missing in missing_critical.items():
        if missing:
            score -= 0.10
            flags.append(f"missing_{name}_section")
    if len(cleaned_markdown) < 200:
        score -= 0.20
        flags.append("low_quality_document")
    return max(0.0, min(1.0, round(score, 4))), sorted(set(flags))


def normalize_markdown(markdown: str, raw_text: str = "") -> dict[str, Any]:
    normalization_events: list[dict[str, Any]] = []
    source = markdown or raw_text or ""
    text = source
    for name, func in (
        ("fix_emails_and_urls", fix_emails_and_urls),
        ("fix_tech_terms", fix_tech_terms),
        ("fix_common_merged_words", fix_common_merged_words),
        ("remove_ocr_artifacts", remove_ocr_artifacts),
    ):
        before = text
        text = func(text)
        if text != before:
            normalization_events.append({"event": name, "changed": True})

    detected_templates = detect_template_values(text)
    sections_found = detect_sections(text)
    header_info = extract_header_info(text)
    quality_score, quality_flags = compute_quality_score(text, detected_templates, sections_found, header_info)

    return {
        "cleaned_markdown": text,
        "header_info": header_info,
        "detected_templates": detected_templates,
        "sections_found": sections_found,
        "quality_score": quality_score,
        "quality_flags": quality_flags,
        "normalization_events": normalization_events,
    }
