from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from src.benchmark.ocr.config import DEFAULT_PDF_DPI, IMAGE_SUFFIXES, PDF_SUFFIXES


@dataclass(slots=True)
class OCRPrediction:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def markdown_to_plain_text(markdown: str) -> str:
    lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line or line == "<!-- image -->":
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        if line.startswith("- "):
            line = line[2:].strip()
        line = line.replace("**", "").replace("*", "")
        lines.append(line)
    return "\n".join(lines).strip()


def render_document_to_images(path: str | Path, *, dpi: int = DEFAULT_PDF_DPI) -> list[Image.Image]:
    document_path = Path(path)
    suffix = document_path.suffix.lower()

    if suffix in IMAGE_SUFFIXES:
        with Image.open(document_path) as image:
            return [image.convert("RGB")]

    if suffix in PDF_SUFFIXES:
        zoom = dpi / 72.0
        images: list[Image.Image] = []
        with fitz.open(document_path) as document:
            for page in document:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
                images.append(image)
        return images

    raise ValueError(f"Unsupported OCR benchmark document type: {document_path}")

