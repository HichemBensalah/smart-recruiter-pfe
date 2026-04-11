import json
import re
from pathlib import Path


DATE_RE = re.compile(
    r"(19|20)\d{2}\s*[-–]\s*(19|20)\d{2}|\b(19|20)\d{2}\b|"
    r"\b(0?[1-9]|1[0-2])/(19|20)\d{2}\b|"
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b.*\b(19|20)\d{2}\b",
    re.IGNORECASE,
)
DATE_LOC_RE = re.compile(r"^(.+?\d)\s+([A-Z][^,]+,\s*[A-Z][^,]+)$")
LOCATION_RE = re.compile(r"\b[A-Z][a-zA-Z]+,\s*[A-Z][a-zA-Z]+")

SECTION_TITLES = {
    "PROFESSIONAL SUMMARY",
    "EDUCATION",
    "TECHNICAL SKILLS",
    "PROFESSIONAL EXPERIENCE",
    "ACADEMIC PROJECTS",
    "LANGUAGES",
    "SOFT SKILLS",
    "CERTIFICATES (IN PROGRESS)",
    "MACHINE LEARNING & DEEP LEARNING",
    "COMPUTER VISION & NATURAL LANGUAGE PROCESSING (NLP)",
    "IDES & SOFTWARE",
}

SECTION_KEYWORDS = {
    "PROFESSIONAL SUMMARY": ["professional summary", "summary", "profil", "profile"],
    "EDUCATION": ["education", "formation", "études", "etudes", "diplôme", "diplome"],
    "TECHNICAL SKILLS": [
        "technical skills",
        "skills",
        "compétences",
        "competences",
        "programming languages",
        "mlops",
        "deployment",
        "devops",
        "data engineering",
        "databases",
        "frameworks",
        "tools",
        "ides",
        "software",
        "cloud",
    ],
    "PROFESSIONAL EXPERIENCE": ["professional experience", "experience", "expérience", "experiences"],
    "ACADEMIC PROJECTS": ["academic projects", "projects", "projets", "project"],
    "LANGUAGES": ["languages", "langues"],
    "SOFT SKILLS": ["soft skills", "skills", "qualités", "qualites"],
    "CERTIFICATES (IN PROGRESS)": ["certificates", "certifications", "certificats"],
    "MACHINE LEARNING & DEEP LEARNING": ["machine learning", "deep learning"],
    "COMPUTER VISION & NATURAL LANGUAGE PROCESSING (NLP)": ["computer vision", "nlp", "natural language processing"],
    "IDES & SOFTWARE": ["ides", "software"],
}

ITEM_SECTIONS = {
    "PROFESSIONAL EXPERIENCE",
    "EDUCATION",
    "ACADEMIC PROJECTS",
}

TITLE_HINTS = (
    "Intern",
    "Training",
    "Degree",
    "University",
    "Project",
    "Studies",
    "Baccalaur",
)

SKILL_SECTIONS = {
    "TECHNICAL SKILLS",
    "MACHINE LEARNING & DEEP LEARNING",
    "COMPUTER VISION & NATURAL LANGUAGE PROCESSING (NLP)",
    "IDES & SOFTWARE",
}

DETAIL_VERBS = (
    "Built",
    "Implemented",
    "Delivered",
    "Developed",
    "Used",
    "Added",
    "Versioned",
    "Wrapped",
    "Designed",
    "Created",
)

DETAIL_HINTS = (
    "pcap",
    "oca",
    "prep",
    "training",
)

SKILL_PREFIXES = (
    "cloud (aws)",
    "python:",
    "databases:",
    "frameworks",
    "tools:",
    "devops",
    "mlops",
    "deployment",
    "big data",
    "etl",
    "bi & etl tools",
    "data engineering",
)

OCR_HEADING_REPLACEMENTS = {
    "PROFESSIONALEXPERIENCE": "PROFESSIONAL EXPERIENCE",
    "PROFESSIONALEXPERIENCE.": "PROFESSIONAL EXPERIENCE",
    "PROFESSIONALSUMMARY": "PROFESSIONAL SUMMARY",
    "TECHNICALSKILLS": "TECHNICAL SKILLS",
    "ACADEMICPROJECTS": "ACADEMIC PROJECTS",
    "SOFTSKILLS": "SOFT SKILLS",
    "CERTIFICATES(INPROGRESS)": "CERTIFICATES (IN PROGRESS)",
    "CERTIFICATESINPROGRESS": "CERTIFICATES (IN PROGRESS)",
    "LANGUAGES": "LANGUAGES",
    "EDUCATION": "EDUCATION",
}

def _fix_encoding(text: str) -> str:
    if "Ã" not in text and "Â" not in text:
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def _clean_ocr_text(text: str) -> str:
    cleaned = _fix_encoding(text)
    for src, dst in OCR_HEADING_REPLACEMENTS.items():
        if src in cleaned:
            cleaned = cleaned.replace(src, dst)
    cleaned = cleaned.replace("⋄", " ").replace("·", "").replace("•", "")
    cleaned = re.sub(r"\bAl\b", "AI", cleaned)
    cleaned = re.sub(r"\bCl/CD\b", "CI/CD", cleaned)
    cleaned = re.sub(r"\bDvC\b", "DVC", cleaned)
    cleaned = re.sub(r"\bFastAPl\b", "FastAPI", cleaned)
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"(?<=\w),(?=\w)", ", ", cleaned)
    cleaned = re.sub(r"(?<=\w)\.(?=\w)", ". ", cleaned)
    cleaned = re.sub(r"\s*&\s*", " & ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_section(text: str) -> bool:
    return text.strip().upper() in SECTION_TITLES or (
        text.isupper() and 2 <= len(text.split()) <= 6
    )


def _match_section_keyword(text: str) -> str | None:
    lower = text.lower().strip()
    for title, keys in SECTION_KEYWORDS.items():
        for k in keys:
            if k in lower:
                return title
    return None


def _docx_section_for_line(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    words = stripped.split()
    if len(words) <= 4:
        matched = _match_section_keyword(stripped)
        if matched:
            return matched
        if stripped.isupper():
            return stripped.upper()
    return None


def _is_date(text: str) -> bool:
    return bool(DATE_RE.search(text)) and len(text) <= 40


def _extract_date(text: str) -> str | None:
    m = DATE_RE.search(text)
    if not m:
        return None
    return m.group(0).strip()


def _is_location(text: str) -> bool:
    if ":" in text:
        return False
    tech_noise = (
        "Frameworks",
        "Tools",
        "Databases",
        "Python",
        "TensorFlow",
        "Keras",
        "PyTorch",
        "Apache",
        "Spark",
        "SQL",
    )
    if any(k in text for k in tech_noise):
        return False
    return bool(LOCATION_RE.search(text)) and len(text) <= 40


def _looks_like_item_title(text: str) -> bool:
    if text.startswith("-"):
        return False
    if "|" in text or "," in text:
        return True
    return any(h in text for h in TITLE_HINTS)


def _is_skill_line(text: str) -> bool:
    lower = text.lower().strip()
    if any(lower.startswith(p) for p in SKILL_PREFIXES):
        return True
    if ":" in text or "|" in text:
        # List-like skill lines are usually short
        return len(text.split()) <= 8
    return False


def _is_docx_skills_subsection(text: str) -> bool:
    lower = text.lower().strip()
    if DATE_RE.search(text):
        return False
    if any(
        k in lower
        for k in (
            "engineer",
            "intern",
            "scientist",
            "developer",
            "analyst",
            "manager",
            "consultant",
            "training",
            "researcher",
            "lead",
            "architect",
            "student",
        )
    ):
        return False
    return any(
        k in lower
        for k in (
            "machine learning",
            "deep learning",
            "computer vision",
            "nlp",
            "natural language processing",
            "ides",
            "software",
            "languages",
            "programming languages",
            "mlops",
            "deployment",
            "cloud",
            "data engineering",
            "databases",
            "frameworks",
            "tools",
            "big data",
            "distributed",
        )
    )


def _extract_blocks(doc: dict, source_format: str | None = None) -> list:
    blocks = []
    pages = doc.get("pages", {})

    for idx, t in enumerate(doc.get("texts", [])):
        prov = (t.get("prov") or [{}])[0]
        bbox = prov.get("bbox") or {}
        text = _fix_encoding((t.get("text") or "").strip())
        if source_format in {"image", "scan"}:
            text = _clean_ocr_text(text)
        if not text:
            continue

        page_no = str(prov.get("page_no", "1"))
        page_info = pages.get(page_no, {})
        page_width = page_info.get("size", {}).get("width", 1.0)
        page_height = page_info.get("size", {}).get("height", 1.0)

        l = bbox.get("l", 0.0)
        r = bbox.get("r", 0.0)
        top = bbox.get("t", 0.0)
        bot = bbox.get("b", 0.0)
        x_center = (l + r) / 2.0
        y_center = (top + bot) / 2.0
        width = max(r - l, 0.0)

        blocks.append(
            {
                "text": text,
                "idx": idx,
                "l": l,
                "r": r,
                "t": top,
                "b": bot,
                "x": x_center,
                "y": y_center,
                "width": width,
                "page_width": page_width,
                "page_height": page_height,
                "page_no": page_no,
            }
        )

    return blocks


def _assign_columns(blocks: list) -> list:
    by_page = {}
    for b in blocks:
        by_page.setdefault(b["page_no"], []).append(b["x"])

    medians = {
        p: sorted(xs)[len(xs) // 2] if xs else 0.0 for p, xs in by_page.items()
    }

    for b in blocks:
        if b["page_height"] and b["y"] >= 0.9 * b["page_height"]:
            b["column"] = "full"
            continue
        if b["page_width"] and b["width"] / b["page_width"] >= 0.85:
            b["column"] = "full"
        else:
            median_x = medians.get(b["page_no"], 0.0)
            b["column"] = "left" if b["x"] <= median_x else "right"
    return blocks


def _ordered_blocks(blocks: list, source_format: str | None = None) -> list:
    if source_format == "docx":
        return sorted(blocks, key=lambda x: x.get("idx", 0))
    if source_format in {"image", "scan"}:
        return sorted(blocks, key=lambda x: (x["page_no"], -x["y"], x["l"]))
    blocks = sorted(blocks, key=lambda x: (x["page_no"], -x["y"]))
    return blocks


def _is_contact_like(text: str) -> bool:
    lower = text.lower()
    return (
        "@" in text
        or "linkedin" in lower
        or "github" in lower
        or bool(re.search(r"\+?\d[\d\s]{7,}", text))
    )


def _is_summary_like(text: str) -> bool:
    lower = text.lower()
    if len(text) < 70:
        return False
    if _is_contact_like(text):
        return False
    return any(k in lower for k in ("focused", "built", "experience", "student", "pipelines", "deployment"))


def _is_education_like(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in ("degree", "studies", "baccalaur", "university", "cycle in engineering"))


def _is_ocr_skill_heading(text: str) -> bool:
    lower = text.lower().strip()
    if DATE_RE.search(text):
        return False
    if any(
        k in lower
        for k in ("engineer", "intern", "scientist", "developer", "analyst", "training", "student", "project")
    ):
        return False
    return any(
        k in lower
        for k in (
            "programming languages",
            "machine learning",
            "deep learning",
            "computer vision",
            "natural language processing",
            "nlp",
            "data engineering",
            "databases",
            "ides",
            "software",
            "ml ops",
            "mlops",
            "deployment",
            "cloud",
        )
    )


def _extract_date_ocr(text: str) -> str | None:
    month_range = re.search(r"(0?[1-9]|1[0-2])/(19|20)\d{2}\s*[-–]\s*(0?[1-9]|1[0-2])/(19|20)\d{2}", text)
    if month_range:
        return month_range.group(0)
    month_name_range = re.search(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s*(19|20)\d{2}"
        r"(?:\s*[-–]\s*(?:Present|Current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s*(19|20)\d{2}))?",
        text,
        re.IGNORECASE,
    )
    if month_name_range:
        return month_name_range.group(0).strip()
    return _extract_date(text)


def _match_ocr_heading(text: str) -> str | None:
    normalized = re.sub(r"[^a-z]", "", text.lower())
    heading_map = {
        "experience": "PROFESSIONAL EXPERIENCE",
        "professionalexperience": "PROFESSIONAL EXPERIENCE",
        "professionalsummary": "PROFESSIONAL SUMMARY",
        "summary": "PROFESSIONAL SUMMARY",
        "technicalskills": "TECHNICAL SKILLS",
        "education": "EDUCATION",
        "academicprojects": "ACADEMIC PROJECTS",
        "projects": "ACADEMIC PROJECTS",
        "languages": "LANGUAGES",
        "softskills": "SOFT SKILLS",
        "certificatesinprogress": "CERTIFICATES (IN PROGRESS)",
        "certificates": "CERTIFICATES (IN PROGRESS)",
    }
    return heading_map.get(normalized)


def _is_ocr_experience_title(text: str) -> bool:
    lower = text.lower().strip()
    if len(text.split()) > 10:
        return False
    return any(k in lower for k in ("engineer", "intern", "scientist", "developer", "analyst", "training"))


def _is_ocr_project_title(text: str) -> bool:
    lower = text.lower().strip()
    if len(text.split()) > 14:
        return False
    return any(k in lower for k in ("project", "pipeline", "prediction", "analysis", "churn", "imdb", "autism"))


def _build_structure_ocr_linear(blocks: list) -> dict:
    structure = {"sections": []}
    sections_map = {}

    def get_section(title: str):
        if title not in sections_map:
            sec = {"title": title, "items": [], "lines": []}
            sections_map[title] = sec
            structure["sections"].append(sec)
        return sections_map[title]

    current_section = get_section("HEADER")
    current_item = None
    last_exp_item = None
    last_edu_item = None
    last_proj_item = None

    for b in blocks:
        text = re.sub(r"^[\-\u2022\u00b7]+\s*", "", b["text"]).strip()
        if not text:
            continue

        matched = _match_ocr_heading(text)
        heading_like = _is_section(text) or bool(matched)
        if heading_like:
            title = text.strip().upper() if _is_section(text) else matched
            current_section = get_section(title)
            current_item = None
            continue

        if current_section["title"] == "HEADER":
            if _is_summary_like(text):
                current_section = get_section("PROFESSIONAL SUMMARY")
                current_section["lines"].append(text)
                current_section = get_section("HEADER")
                continue
            if _is_education_like(text) and _extract_date_ocr(text):
                current_section = get_section("EDUCATION")
            elif _is_education_like(text) and _extract_date_ocr(text):
                current_section = get_section("EDUCATION")
            elif _is_ocr_skill_heading(text) or (":" in text and _is_skill_line(text)):
                current_section = get_section("TECHNICAL SKILLS")
            elif _is_ocr_experience_title(text):
                current_section = get_section("PROFESSIONAL EXPERIENCE")
            elif _is_ocr_project_title(text):
                current_section = get_section("ACADEMIC PROJECTS")

        if _is_education_like(text) and (_extract_date_ocr(text) or current_section["title"] == "EDUCATION"):
            current_section = get_section("EDUCATION")
            item = {"title": text, "date": _extract_date_ocr(text), "location": None, "details": []}
            current_section["items"].append(item)
            current_item = item
            last_edu_item = item
            continue

        if _is_ocr_skill_heading(text) or (":" in text and _is_skill_line(text)):
            skill_section = get_section("TECHNICAL SKILLS")
            skill_section["lines"].append(text)
            if current_section["title"] not in {"PROFESSIONAL EXPERIENCE", "ACADEMIC PROJECTS", "EDUCATION"}:
                current_section = skill_section
            continue

        if current_section["title"] == "PROFESSIONAL EXPERIENCE":
            if _is_ocr_experience_title(text):
                item = {"title": text, "date": _extract_date_ocr(text), "location": None, "details": []}
                current_section["items"].append(item)
                current_item = item
                last_exp_item = item
                continue
            if _is_date(text) and last_exp_item:
                last_exp_item["date"] = _extract_date_ocr(text) or text
                continue
            if _is_location(text) and last_exp_item:
                last_exp_item["location"] = text
                continue
            if last_exp_item:
                last_exp_item["details"].append(text)
                continue

        if current_section["title"] == "ACADEMIC PROJECTS":
            if _is_ocr_project_title(text):
                item = {"title": text, "date": _extract_date_ocr(text), "location": None, "details": []}
                current_section["items"].append(item)
                current_item = item
                last_proj_item = item
                continue
            if _is_date(text) and last_proj_item:
                last_proj_item["date"] = _extract_date_ocr(text) or text
                continue
            if last_proj_item:
                last_proj_item["details"].append(text)
                continue

        if current_section["title"] == "EDUCATION":
            if _is_date(text) and last_edu_item:
                last_edu_item["date"] = _extract_date_ocr(text) or text
                continue
            if last_edu_item and not _match_section_keyword(text) and not _is_contact_like(text):
                last_edu_item["details"].append(text)
                continue

        if current_section["title"] == "HEADER":
            current_section["lines"].append(text)
            continue

        if current_section["title"] in {"LANGUAGES", "SOFT SKILLS", "CERTIFICATES (IN PROGRESS)", "TECHNICAL SKILLS", "PROFESSIONAL SUMMARY"}:
            current_section["lines"].append(text)
            continue

        current_section["lines"].append(text)

    for sec in structure["sections"]:
        cleaned_lines = []
        for line in sec.get("lines", []):
            if sec["title"] == "HEADER" and (_is_skill_line(line) or _is_ocr_skill_heading(line)):
                get_section("TECHNICAL SKILLS")["lines"].append(line)
                continue
            cleaned_lines.append(line)
        sec["lines"] = cleaned_lines

    return structure


def _build_structure(blocks: list, source_format: str | None = None) -> dict:
    if source_format in {"image", "scan"}:
        return _build_structure_ocr_linear(blocks)
    sections_map = {"HEADER": {"title": "HEADER", "items": [], "lines": []}}
    headers = []

    for b in blocks:
        text = b["text"]
        header_text = text
        if source_format in {"image", "scan"}:
            header_text = re.sub(r"^[\-\u2022\u00b7]+\s*", "", header_text).strip()
        if _is_section(header_text):
            title = header_text.strip().upper()
            if title not in sections_map:
                sections_map[title] = {"title": title, "items": [], "lines": []}
            headers.append(
                {"title": title, "y": b["y"], "column": b.get("column"), "page": b["page_no"]}
            )
        else:
            matched = _match_section_keyword(header_text)
            if matched:
                if matched not in sections_map:
                    sections_map[matched] = {"title": matched, "items": [], "lines": []}
                headers.append(
                    {
                        "title": matched,
                        "y": b["y"],
                        "column": b.get("column"),
                        "page": b["page_no"],
                    }
                )

    headers_sorted = sorted(headers, key=lambda h: (h["page"], -h["y"]))

    def find_section(block):
        if not headers_sorted:
            return sections_map["HEADER"]
        same_col = [
            h
            for h in headers_sorted
            if h["page"] == block["page_no"]
            and h["y"] >= block["y"]
            and h["column"] == block.get("column")
        ]
        candidates = same_col or [
            h for h in headers_sorted if h["page"] == block["page_no"] and h["y"] >= block["y"]
        ]
        if candidates:
            h = min(candidates, key=lambda h: h["y"] - block["y"])
            return sections_map[h["title"]]
        return sections_map["HEADER"]

    current_items = {}
    orphan_dates = []
    orphan_locations = []

    current_section = sections_map["HEADER"]
    for b in blocks:
        text = b["text"]
        if _is_section(text):
            continue
        # DOCX: strong section anchoring by keyword/uppercase line
        if source_format == "docx":
            matched = _docx_section_for_line(text)
            if matched:
                if matched not in sections_map:
                    sections_map[matched] = {"title": matched, "items": [], "lines": []}
                current_section = sections_map.get(matched) or sections_map["HEADER"]
                continue
        if b.get("column") == "right" and _is_date(text):
            orphan_dates.append({"text": text, "y": b["y"]})
            continue
        if b.get("column") == "right" and _is_location(text):
            orphan_locations.append({"text": text, "y": b["y"]})
            continue

        sec = find_section(b) if source_format != "docx" else current_section
        title = sec["title"]

        if source_format in {"image", "scan"}:
            norm = re.sub(r"^[\-\u2022\u00b7]+\s*", "", text).strip()
            if _is_skill_line(norm):
                if "TECHNICAL SKILLS" not in sections_map:
                    sections_map["TECHNICAL SKILLS"] = {"title": "TECHNICAL SKILLS", "items": [], "lines": []}
                sec = sections_map["TECHNICAL SKILLS"]
                title = sec["title"]

        # DOCX: force EDUCATION on degree-like lines
        if source_format == "docx":
            lower = text.lower()
            if any(k in lower for k in ("degree", "university", "baccalaur", "studies")) and (
                "20" in text or "19" in text
            ):
                if "EDUCATION" not in sections_map:
                    sections_map["EDUCATION"] = {"title": "EDUCATION", "items": [], "lines": []}
                current_section = sections_map["EDUCATION"]
                title = current_section["title"]

        if title in ITEM_SECTIONS and _looks_like_item_title(text):
            current_items[title] = {
                "title": text,
                "y": b["y"],
                "date": None,
                "location": None,
                "details": [],
            }
            sec["items"].append(current_items[title])
            continue

        if title in ITEM_SECTIONS:
            if _is_date(text):
                if current_items.get(title):
                    current_items[title]["date"] = text
                else:
                    sec["lines"].append({"text": text, "y": b["y"]})
                continue
            if _is_location(text):
                if current_items.get(title):
                    current_items[title]["location"] = text
                else:
                    sec["lines"].append({"text": text, "y": b["y"]})
                continue

            if current_items.get(title):
                if text.startswith("-"):
                    current_items[title]["details"].append(text.lstrip("- ").strip())
                else:
                    current_items[title]["details"].append(text)
            else:
                sec["lines"].append({"text": text, "y": b["y"]})
        else:
            sec["lines"].append({"text": text, "y": b["y"]})

    # Attach orphan dates/locations
    all_items = []
    for sec in sections_map.values():
        if sec["title"] in ITEM_SECTIONS:
            all_items.extend(sec["items"])
    if all_items:
        for orphan in orphan_dates:
            nearest = min(all_items, key=lambda it: abs(it["y"] - orphan["y"]))
            if not nearest.get("date"):
                nearest["date"] = orphan["text"]
        for orphan in orphan_locations:
            nearest = min(all_items, key=lambda it: abs(it["y"] - orphan["y"]))
            if not nearest.get("location"):
                nearest["location"] = orphan["text"]

    # Split date/location if fused
    for it in all_items:
        date_val = it.get("date")
        if date_val and not it.get("location"):
            m = DATE_LOC_RE.match(date_val.strip())
            if m:
                it["date"] = m.group(1).strip()
                it["location"] = m.group(2).strip()

    # Clean header/summary
    header = sections_map.get("HEADER")
    summary = sections_map.get("PROFESSIONAL SUMMARY")
    if header and summary:
        moved = []
        for line in header["lines"]:
            text = line["text"] if isinstance(line, dict) else line
            if len(text) > 80 and "@" not in text and "http" not in text:
                summary["lines"].append(line)
                moved.append(line)
        if moved:
            header["lines"] = [l for l in header["lines"] if l not in moved]

    # Reattach misplaced experience/project details from skills sections
    exp_proj_items = []
    for sec in sections_map.values():
        if sec["title"] in {"PROFESSIONAL EXPERIENCE", "ACADEMIC PROJECTS"}:
            exp_proj_items.extend(sec["items"])

    if exp_proj_items:
        for sec in sections_map.values():
            if sec["title"] not in SKILL_SECTIONS:
                continue
            moved_lines = []
            for line in sec["lines"]:
                text = line["text"] if isinstance(line, dict) else line
                lower = text.lower()
                is_skill_like = _is_skill_line(text)
                is_detail = any(v in text for v in DETAIL_VERBS) or any(
                    h in lower for h in DETAIL_HINTS
                )
                if is_detail and not is_skill_like:
                    nearest = min(exp_proj_items, key=lambda it: abs(it["y"] - line["y"]))
                    nearest["details"].append(text)
                    moved_lines.append(line)
            if moved_lines:
                sec["lines"] = [l for l in sec["lines"] if l not in moved_lines]

    # Build ordered sections list
    ordered_sections = []
    if sections_map["HEADER"]["lines"]:
        ordered_sections.append(sections_map["HEADER"])
    seen = set()
    for h in headers_sorted:
        if h["title"] in seen:
            continue
        ordered_sections.append(sections_map[h["title"]])
        seen.add(h["title"])

    return {"sections": ordered_sections}


def _render_txt(structure: dict) -> str:
    out = []
    for sec in structure["sections"]:
        out.append(sec["title"])
        out.append("")
        for line in sec.get("lines", []):
            out.append(line["text"] if isinstance(line, dict) else line)
        for item in sec.get("items", []):
            out.append(item["title"])
            if item.get("date"):
                out.append(item["date"])
            if item.get("location"):
                out.append(item["location"])
            for d in item.get("details", []):
                out.append(f"- {d}")
        out.append("")
    return "\n".join(out).strip()


def _render_md(structure: dict) -> str:
    out = []
    for sec in structure["sections"]:
        out.append(f"## {sec['title']}")
        out.append("")
        for line in sec.get("lines", []):
            out.append(line["text"] if isinstance(line, dict) else line)
        for item in sec.get("items", []):
            out.append(f"### {item['title']}")
            if item.get("date"):
                out.append(f"**Date:** {item['date']}")
            if item.get("location"):
                out.append(f"**Location:** {item['location']}")
            for d in item.get("details", []):
                out.append(f"- {d}")
        out.append("")
    return "\n".join(out).strip()


def _render_html(structure: dict) -> str:
    parts = ["<!doctype html><html><head><meta charset='utf-8'>"]
    parts.append("<title>Resume Structure</title></head><body>")
    for sec in structure["sections"]:
        parts.append(f"<h2>{sec['title']}</h2>")
        for line in sec.get("lines", []):
            text = line["text"] if isinstance(line, dict) else line
            parts.append(f"<p>{text}</p>")
        for item in sec.get("items", []):
            parts.append(f"<h3>{item['title']}</h3>")
            if item.get("date"):
                parts.append(f"<p><strong>Date:</strong> {item['date']}</p>")
            if item.get("location"):
                parts.append(f"<p><strong>Location:</strong> {item['location']}</p>")
            if item.get("details"):
                parts.append("<ul>")
                for d in item["details"]:
                    parts.append(f"<li>{d}</li>")
                parts.append("</ul>")
    parts.append("</body></html>")
    return "\n".join(parts)


def postprocess_docling(doc: dict, source_format: str | None = None) -> dict:
    blocks = _extract_blocks(doc, source_format)
    blocks = _assign_columns(blocks)
    ordered = _ordered_blocks(blocks, source_format)
    structure = _build_structure(ordered, source_format)
    return {"blocks": ordered, "structure": structure}


def postprocess_docx_markdown(md: str) -> dict:
    lines = [l.strip() for l in md.splitlines()]
    lines = [l for l in lines if l and l != "<!-- image -->"]

    structure = {"sections": []}
    sections_map = {}

    def _extract_date_docx(text: str) -> str | None:
        month_range = re.search(
            r"(0?[1-9]|1[0-2])/(19|20)\d{2}\s*[-–]\s*(0?[1-9]|1[0-2])/(19|20)\d{2}",
            text,
        )
        if month_range:
            return month_range.group(0)
        month_name_range = re.search(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s*(19|20)\d{2}"
            r"(?:\s*[-–]\s*(?:Present|Current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s*(19|20)\d{2}))?",
            text,
            re.IGNORECASE,
        )
        if month_name_range:
            return month_name_range.group(0).strip()
        return _extract_date(text)

    def get_section(title: str):
        if title not in sections_map:
            sec = {"title": title, "items": [], "lines": []}
            sections_map[title] = sec
            structure["sections"].append(sec)
        return sections_map[title]

    current_section = get_section("HEADER")
    current_item = None
    last_item = None
    skills_capture = False

    for line in lines:
        if skills_capture:
            if ":" in line or line.startswith("**") or "|" in line:
                get_section("TECHNICAL SKILLS")["lines"].append(line)
                continue
            if line.startswith("#") or _is_date(line) or _looks_like_item_title(line):
                skills_capture = False

        # Early education anchor while still in header
        if (
            current_section["title"] == "HEADER"
            and any(k in line.lower() for k in ("degree", "studies", "baccalaur", "university"))
            and _extract_date_docx(line)
        ):
            current_section = get_section("EDUCATION")
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line).strip()
            for prefix in ("#### ", "### ", "## "):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):].strip()
            current_item = {
                "title": clean,
                "date": _extract_date_docx(line),
                "location": None,
                "details": [],
            }
            current_section["items"].append(current_item)
            last_item = current_item
            continue

        # Section headers
        if line.startswith("# "):
            current_section["lines"].append(line[2:].strip())
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            title_up = title.upper()
            if title_up in SECTION_TITLES or _match_section_keyword(title):
                current_section = get_section(title_up if title_up in SECTION_TITLES else _match_section_keyword(title))
                current_item = None
                continue
        if line.startswith("### "):
            title = line[4:].strip()
            title_up = title.upper()
            if title_up in SECTION_TITLES or _match_section_keyword(title):
                current_section = get_section(title_up if title_up in SECTION_TITLES else _match_section_keyword(title))
                current_item = None
                continue
            # Force section if line looks like a known sub-section header
            forced = _match_section_keyword(title)
            if forced:
                current_section = get_section(forced)
                current_item = None
                continue
            # Skills sub-sections should stay under TECHNICAL SKILLS
            if (
                any(k in title.lower() for k in ("machine learning", "deep learning", "computer vision", "nlp", "ides", "software"))
                and current_section["title"] != "ACADEMIC PROJECTS"
            ):
                if current_section["title"] in ITEM_SECTIONS:
                    get_section("TECHNICAL SKILLS")["lines"].append(title)
                    skills_capture = True
                else:
                    current_section = get_section("TECHNICAL SKILLS")
                    current_item = None
                    current_section["lines"].append(title)
                continue
            if current_section["title"] in ITEM_SECTIONS:
                current_item = {"title": title, "date": _extract_date_docx(title), "location": None, "details": []}
                current_section["items"].append(current_item)
                last_item = current_item
                continue
        if line.startswith("#### "):
            title = line[5:].strip()
            title_clean = re.sub(r"\*\*(.+?)\*\*", r"\1", title).strip()
            if _is_docx_skills_subsection(title_clean) and current_section["title"] != "ACADEMIC PROJECTS":
                if current_section["title"] in ITEM_SECTIONS:
                    get_section("TECHNICAL SKILLS")["lines"].append(title_clean)
                    skills_capture = True
                else:
                    current_section = get_section("TECHNICAL SKILLS")
                    current_item = None
                    current_section["lines"].append(title_clean)
                continue
            if current_section["title"] in ITEM_SECTIONS:
                current_item = {"title": title_clean, "date": _extract_date_docx(title_clean), "location": None, "details": []}
                current_section["items"].append(current_item)
                last_item = current_item
                continue
        if line.startswith("**") and line.endswith("**"):
            content = line.strip("*").strip()
            # skills sub-sections in bold should force TECHNICAL SKILLS
            if any(k in content.lower() for k in ("machine learning", "deep learning", "computer vision", "nlp", "ides", "software", "languages")):
                if current_section["title"] in ITEM_SECTIONS:
                    get_section("TECHNICAL SKILLS")["lines"].append(content)
                    skills_capture = True
                else:
                    current_section = get_section("TECHNICAL SKILLS")
                    current_item = None
                    current_section["lines"].append(content)
                continue
            if content.isupper() or _match_section_keyword(content):
                title = content.upper() if content.isupper() else _match_section_keyword(content)
                current_section = get_section(title)
                current_item = None
                continue
        bold_match = re.match(r"^\*\*(.+?)\*\*\s*(.+)?$", line)
        if bold_match and bold_match.group(2):
            title = bold_match.group(1).strip()
            rest = (bold_match.group(2) or "").strip()
            if _is_docx_skills_subsection(title) and current_section["title"] != "ACADEMIC PROJECTS":
                if current_section["title"] in ITEM_SECTIONS:
                    get_section("TECHNICAL SKILLS")["lines"].append(f"{title} {rest}".strip())
                    skills_capture = True
                else:
                    current_section = get_section("TECHNICAL SKILLS")
                    current_item = None
                    current_section["lines"].append(f"{title} {rest}".strip())
                continue
            if current_section["title"] in ITEM_SECTIONS and (_extract_date_docx(line) or _looks_like_item_title(title)):
                current_item = {
                    "title": f"{title} {rest}".strip(),
                    "date": _extract_date_docx(line),
                    "location": None,
                    "details": [],
                }
                current_section["items"].append(current_item)
                last_item = current_item
                continue

        # Education lines with dates (often bold)
        if current_section["title"] == "HEADER" and any(k in line.lower() for k in ("degree", "studies", "baccalaur", "university")) and _extract_date(line):
            current_section = get_section("EDUCATION")
            current_item = None

        if current_section["title"] in ITEM_SECTIONS:
            if _is_date(line) and (current_item is not None or last_item is not None):
                target = current_item or last_item
                if target:
                    target["date"] = _extract_date_docx(line)
                continue
            if _is_location(line) and (current_item is not None or last_item is not None):
                target = current_item or last_item
                if target:
                    target["location"] = line
                continue
            if current_item is None:
                date = _extract_date_docx(line)
                current_item = {"title": line, "date": date, "location": None, "details": []}
                current_section["items"].append(current_item)
                last_item = current_item
            else:
                # attach details
                current_item["details"].append(line)
        else:
            current_section["lines"].append(line)

    return {"blocks": [], "structure": structure}


def write_outputs(
    out_dir: Path,
    base_name: str,
    payload: dict,
    source_path: str,
    metadata: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    structure = payload["structure"]

    txt = _render_txt(structure)
    md = _render_md(structure)
    html = _render_html(structure)

    (out_dir / f"{base_name}.txt").write_text(txt, encoding="utf-8")
    (out_dir / f"{base_name}.md").write_text(md, encoding="utf-8")
    (out_dir / f"{base_name}.html").write_text(html, encoding="utf-8")

    json_payload = {
        "source": source_path,
        "layout": {"columns": 2, "ordering": "page+left/right", "method": "x_median"},
        "blocks": payload["blocks"],
        "structure": payload["structure"],
    }
    if metadata:
        json_payload.update(metadata)
    (out_dir / f"{base_name}.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
