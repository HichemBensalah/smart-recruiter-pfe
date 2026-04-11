from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


SECTION_KEYWORDS = {
    "PROFESSIONAL SUMMARY": ("professional summary", "summary", "profile", "profil", "objective"),
    "EDUCATION": ("education", "formation", "studies", "degree", "training"),
    "TECHNICAL SKILLS": ("technical skills", "skills", "competences", "competencies", "tools"),
    "PROFESSIONAL EXPERIENCE": ("professional experience", "experience", "work history", "employment"),
    "ACADEMIC PROJECTS": ("projects", "academic projects", "project experience"),
    "LANGUAGES": ("languages", "langues"),
    "SOFT SKILLS": ("soft skills",),
    "CERTIFICATIONS": ("certifications", "certificates"),
    "JOB SUMMARY": ("job summary", "about the role", "overview", "mission"),
    "RESPONSIBILITIES": ("responsibilities", "what you will do", "missions"),
    "REQUIREMENTS": ("requirements", "qualifications", "what we are looking for"),
}

OCR_CONF_THRESHOLD = float(os.getenv("OCR_CONF_THRESHOLD", "28.0"))
OCR_LOW_CONF_WARNING_THRESHOLD = float(os.getenv("OCR_LOW_CONF_WARNING_THRESHOLD", "35.0"))
PDF_NATIVE_MIN_WORDS_FOR_TEXT_ONLY = int(os.getenv("PDF_NATIVE_MIN_WORDS_FOR_TEXT_ONLY", "200"))


def parse_with_secondary_parser(file_path: str, source_format: str) -> tuple[dict, dict]:
    path = Path(file_path)
    warnings: list[str] = []
    extraction_method = "secondary_text"

    text = ""
    if path.suffix.lower() == ".docx":
        text = _extract_docx_text(path)
        extraction_method = "secondary_docx_xml"
    elif path.suffix.lower() == ".pdf":
        text = _extract_pdf_text(path)
        extraction_method = "secondary_pdf_text"
        if not text.strip():
            warnings.append("secondary_pdf_text_empty")
            text, ocr_warnings = _ocr_pdf(path)
            warnings.extend(ocr_warnings)
            extraction_method = "secondary_pdf_ocr"
        elif _word_count(text) < PDF_NATIVE_MIN_WORDS_FOR_TEXT_ONLY:
            warnings.append("secondary_pdf_native_text_short")
            ocr_text, ocr_warnings = _ocr_pdf(path)
            warnings.extend(ocr_warnings)
            if _word_count(ocr_text) > _word_count(text):
                text = ocr_text
                extraction_method = "secondary_pdf_ocr_boost"
                warnings.append("secondary_pdf_ocr_selected_for_low_native_text")
    else:
        text, ocr_warnings = _ocr_image(path)
        warnings.extend(ocr_warnings)
        extraction_method = "secondary_image_ocr"

    if not text.strip():
        raise RuntimeError(f"Secondary parser produced no text for {file_path}")

    payload = _payload_from_plain_text(text, source_format)
    return payload, {"extraction_method": extraction_method, "warnings": warnings}


def _extract_pdf_text(path: Path) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF unavailable for secondary PDF parsing") from exc

    chunks: list[str] = []
    doc = fitz.open(path)
    try:
        for page in doc:
            chunks.append(page.get_text("text"))
    finally:
        doc.close()
    return "\n".join(chunks).strip()


def _ocr_pdf(path: Path) -> tuple[str, list[str]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("OCR fallback unavailable for PDF secondary parsing") from exc

    chunks: list[str] = []
    warnings: list[str] = []
    doc = fitz.open(path)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            text, page_warnings = _ocr_pil_image(
                _pil_image_from_pixmap(pix),
                source_label=f"{path.name}:page_{page.number + 1}",
            )
            chunks.append(text)
            warnings.extend(page_warnings)
    finally:
        doc.close()
    return "\n".join(chunk for chunk in chunks if chunk).strip(), warnings


def _ocr_image(path: Path) -> tuple[str, list[str]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("OCR fallback unavailable for image secondary parsing") from exc

    with Image.open(path) as image:
        return _ocr_pil_image(image, source_label=path.name)


def _extract_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
    except KeyError as exc:
        raise RuntimeError("DOCX document.xml missing") from exc

    root = ET.fromstring(xml_bytes)
    namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", namespaces):
        texts = [node.text or "" for node in para.findall(".//w:t", namespaces)]
        paragraph = "".join(texts).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n".join(paragraphs).strip()


def _payload_from_plain_text(text: str, source_format: str) -> dict:
    cleaned_lines = _normalize_lines(text)
    blocks = []
    structure = {"sections": []}
    sections_map: dict[str, dict] = {}

    def get_section(title: str) -> dict:
        if title not in sections_map:
            section = {"title": title, "items": [], "lines": []}
            sections_map[title] = section
            structure["sections"].append(section)
        return sections_map[title]

    current = get_section("HEADER")

    for idx, line in enumerate(cleaned_lines):
        blocks.append(
            {
                "text": line,
                "idx": idx,
                "l": 0.0,
                "r": 0.0,
                "t": float(idx),
                "b": float(idx),
                "x": 0.0,
                "y": float(idx),
                "width": 0.0,
                "page_width": 1.0,
                "page_height": max(float(len(cleaned_lines)), 1.0),
                "page_no": "1",
                "column": "full",
            }
        )

        matched = _match_section(line)
        if matched:
            current = get_section(matched)
            continue

        current["lines"].append(line)

    if source_format == "docx" and len(blocks) < 8:
        for idx, sentence in enumerate(_split_long_sentences(text), start=len(blocks)):
            blocks.append(
                {
                    "text": sentence,
                    "idx": idx,
                    "l": 0.0,
                    "r": 0.0,
                    "t": float(idx),
                    "b": float(idx),
                    "x": 0.0,
                    "y": float(idx),
                    "width": 0.0,
                    "page_width": 1.0,
                    "page_height": max(float(idx + 1), 1.0),
                    "page_no": "1",
                    "column": "full",
                }
            )

    return {"blocks": blocks, "structure": structure}


def _normalize_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if len(line) > 220 and "." in line:
            lines.extend(_split_long_sentences(line))
        else:
            lines.append(line)
    return lines


def _split_long_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return parts or [text.strip()]


def _match_section(line: str) -> str | None:
    stripped = line.strip()
    lower = stripped.lower()
    if stripped.isupper() and 1 <= len(stripped.split()) <= 6:
        return stripped
    for title, keywords in SECTION_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return title
    return None


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _pil_image_from_pixmap(pix) -> "Image.Image":
    from PIL import Image

    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _ocr_pil_image(image: "Image.Image", *, source_label: str) -> tuple[str, list[str]]:
    try:
        import pytesseract
        from PIL import ImageEnhance, ImageOps, ImageStat
    except ImportError as exc:
        raise RuntimeError("OCR fallback unavailable for image preprocessing") from exc

    warnings: list[str] = []
    _ = source_label

    gray = ImageOps.grayscale(image)
    boosted = ImageOps.autocontrast(gray)
    boosted = ImageEnhance.Contrast(boosted).enhance(1.8)
    mean_luma = ImageStat.Stat(boosted).mean[0] if boosted.size[0] and boosted.size[1] else 180.0
    threshold = max(120, min(int(mean_luma * 0.92), 190))
    thresholded = boosted.point(lambda px: 0 if px < threshold else 255)

    text, avg_conf = _extract_text_with_confidence(thresholded)
    if not text.strip():
        text, avg_conf = _extract_text_with_confidence(boosted)

    if avg_conf < OCR_CONF_THRESHOLD:
        warnings.append("low_ocr_confidence")
    elif avg_conf < OCR_LOW_CONF_WARNING_THRESHOLD:
        warnings.append("low_confidence_accepted")

    return text.strip(), warnings


def _extract_text_with_confidence(image: "Image.Image") -> tuple[str, float]:
    import pytesseract

    data = pytesseract.image_to_data(
        image,
        lang="eng+fra",
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6",
    )
    confidences = [
        float(conf)
        for conf in data.get("conf", [])
        if conf not in ("-1", "-1.0") and str(conf).strip()
    ]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    tokens = []
    for idx, token in enumerate(data.get("text", [])):
        if idx >= len(data.get("conf", [])):
            continue
        conf = data["conf"][idx]
        if conf in ("-1", "-1.0"):
            continue
        cleaned = token.strip()
        if cleaned:
            tokens.append(cleaned)
    return " ".join(tokens).strip(), avg_conf


def vision_extraction_check(file_path: str, artifact_path: str | None = None) -> None:
    """
    Placeholder for a future Vision API recovery path.

    Intended usage:
    - call only for documents already placed in quarantined
    - send the source file and optional Module 1 artifact to a multimodal model
    - compare the recovered text against the existing artifact before any manual revalidation
    """

    return None
