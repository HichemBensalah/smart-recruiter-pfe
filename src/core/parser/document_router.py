from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from pathlib import Path as _Path

import fitz
from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from .document_artifact import ParserUsed, SourceFormat
except ImportError:
    artifact_path = _Path(__file__).resolve().with_name("document_artifact.py")
    spec = importlib.util.spec_from_file_location("document_artifact", artifact_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load document_artifact from {artifact_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ParserUsed = module.ParserUsed  # type: ignore[attr-defined]
    SourceFormat = module.SourceFormat  # type: ignore[attr-defined]


TEXT_PAGE_CHAR_THRESHOLD = 40
PDF_NATIVE_MIN_AVG_CHARS = 80.0
PDF_NATIVE_MIN_TEXT_PAGE_RATIO = 0.5
PDF_NATIVE_MIN_WORDS_FOR_TEXT_ONLY = 200
PDF_SCAN_IMAGE_RATIO_THRESHOLD = 0.8
PDF_SCAN_LOW_TEXT_WHEN_IMAGE_HEAVY = 140.0

PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
KNOWN_SOURCE_FOLDERS = {member.value for member in SourceFormat}


class RoutingDecision(BaseModel):
    """Pre-flight routing output emitted before any parsing strategy is executed."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    source_format: SourceFormat = Field(
        ...,
        description="Normalized input family inferred from file location and extension.",
    )
    is_scanned: bool = Field(
        ...,
        description="True when the document behaves like an image-first or scan-first source.",
    )
    ocr_required: bool = Field(
        ...,
        description="True when OCR must be enabled for the primary parsing strategy.",
    )
    primary_parser: ParserUsed = Field(
        ...,
        description="Primary parser route selected before the pipeline starts extraction.",
    )
    routing_reasons: list[str] = Field(
        default_factory=list,
        description="Ordered audit trail explaining why the route was selected.",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "RoutingDecision":
        """Keep the routing contract coherent before the orchestrator consumes it."""
        if not self.routing_reasons:
            raise ValueError("RoutingDecision must contain at least one routing reason.")
        if self.is_scanned and not self.ocr_required:
            raise ValueError("A scanned document must require OCR.")
        if self.source_format == SourceFormat.DOCX and self.ocr_required:
            raise ValueError("DOCX routing cannot require OCR.")
        if self.source_format == SourceFormat.DOCX and self.primary_parser != ParserUsed.DOCLING_MARKDOWN:
            raise ValueError("DOCX documents must route to the Docling markdown parser.")
        return self


def route_document(file_path: str | Path) -> RoutingDecision:
    """
    Analyze a raw document before parsing and choose the primary Module 1 strategy.

    Rules:
    - Native PDF -> Docling structured without OCR.
    - Scanned PDF or image -> Docling structured with OCR.
    - DOCX -> Docling markdown without OCR.
    """

    path = Path(file_path)
    source_format = infer_source_format(path)

    if source_format == SourceFormat.DOCX:
        return RoutingDecision(
            source_format=source_format,
            is_scanned=False,
            ocr_required=False,
            primary_parser=ParserUsed.DOCLING_MARKDOWN,
            routing_reasons=[
                f"source_format={source_format.value}",
                "extension=.docx detected",
                "docx documents route to the dedicated Docling markdown parser",
                "ocr_required=False because DOCX already provides machine-readable text",
            ],
        )

    if source_format == SourceFormat.IMAGES:
        return RoutingDecision(
            source_format=source_format,
            is_scanned=True,
            ocr_required=True,
            primary_parser=ParserUsed.DOCLING_STRUCTURED,
            routing_reasons=[
                f"source_format={source_format.value}",
                f"extension={path.suffix.lower()} detected",
                "raster image input has no reliable embedded text layer",
                "route=docling_structured with OCR enabled",
            ],
        )

    if path.suffix.lower() in PDF_EXTENSIONS:
        return _route_pdf(path, source_format)

    raise ValueError(f"Unsupported document for routing: {path}")


def infer_source_format(file_path: str | Path) -> SourceFormat:
    """Infer the normalized source format from folder naming first, then file extension."""

    path = Path(file_path)
    folder_hints = [part.lower() for part in path.parts]

    for part in reversed(folder_hints):
        if part in KNOWN_SOURCE_FOLDERS:
            return SourceFormat(part)

    suffix = path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return SourceFormat.PDF
    if suffix in DOCX_EXTENSIONS:
        return SourceFormat.DOCX
    if suffix in IMAGE_EXTENSIONS:
        return SourceFormat.IMAGES

    raise ValueError(f"Unsupported source format for file: {path}")


def _route_pdf(path: Path, source_format: SourceFormat) -> RoutingDecision:
    """Route a PDF after a lightweight diagnosis of its text layer."""

    is_scanned, reasons = _diagnose_pdf_scan(path)
    reasons.insert(0, f"source_format={source_format.value}")

    if is_scanned:
        reasons.append("route=docling_structured with OCR enabled because PDF behaves like a scan")
        return RoutingDecision(
            source_format=source_format,
            is_scanned=True,
            ocr_required=True,
            primary_parser=ParserUsed.DOCLING_STRUCTURED,
            routing_reasons=reasons,
        )

    reasons.append("route=docling_structured without OCR because PDF contains a usable text layer")
    return RoutingDecision(
        source_format=source_format,
        is_scanned=False,
        ocr_required=False,
        primary_parser=ParserUsed.DOCLING_STRUCTURED,
        routing_reasons=reasons,
    )


def _diagnose_pdf_scan(path: Path) -> tuple[bool, list[str]]:
    """
    Detect whether a PDF is native or scan-like using a fast text-layer probe.

    A PDF is treated as native when enough pages expose a meaningful text layer.
    Weak, sparse, or near-empty text extraction is treated as scan-like.
    """

    reasons = [f"extension={path.suffix.lower()} detected"]

    with fitz.open(path) as document:
        page_count = document.page_count
        total_text_chars = 0
        total_text_words = 0
        pages_with_text = 0
        pages_with_images = 0

        for page_index in range(page_count):
            page = document.load_page(page_index)
            text = " ".join((page.get_text("text") or "").split())
            text_chars = len(text)
            text_words = len(text.split())
            total_text_chars += text_chars
            total_text_words += text_words

            if text_chars >= TEXT_PAGE_CHAR_THRESHOLD:
                pages_with_text += 1

            if page.get_images(full=True):
                pages_with_images += 1

    avg_chars_per_page = total_text_chars / max(page_count, 1)
    text_page_ratio = pages_with_text / max(page_count, 1)
    image_page_ratio = pages_with_images / max(page_count, 1)

    reasons.append(f"pdf_page_count={page_count}")
    reasons.append(f"pdf_total_text_chars={total_text_chars}")
    reasons.append(f"pdf_total_text_words={total_text_words}")
    reasons.append(f"pdf_avg_text_chars_per_page={avg_chars_per_page:.1f}")
    reasons.append(f"pdf_text_page_ratio={text_page_ratio:.2f}")
    reasons.append(f"pdf_image_page_ratio={image_page_ratio:.2f}")

    if image_page_ratio >= PDF_SCAN_IMAGE_RATIO_THRESHOLD and avg_chars_per_page < PDF_SCAN_LOW_TEXT_WHEN_IMAGE_HEAVY:
        reasons.append(
            "classification=scanned_pdf because the PDF is image-heavy and the text layer stays weak"
        )
        reasons.append(
            f"ocr_forced_due_to_image_heavy_pdf(image_ratio>={PDF_SCAN_IMAGE_RATIO_THRESHOLD}, avg_chars<{PDF_SCAN_LOW_TEXT_WHEN_IMAGE_HEAVY})"
        )
        return True, reasons

    has_usable_text_layer = (
        total_text_chars >= TEXT_PAGE_CHAR_THRESHOLD
        and avg_chars_per_page >= PDF_NATIVE_MIN_AVG_CHARS
        and text_page_ratio >= PDF_NATIVE_MIN_TEXT_PAGE_RATIO
    )

    if has_usable_text_layer:
        if total_text_words < PDF_NATIVE_MIN_WORDS_FOR_TEXT_ONLY:
            reasons.append(
                "classification=hybrid_pdf because a text layer exists but native text volume is too short"
            )
            reasons.append(
                f"ocr_forced_due_to_low_native_text_words<{PDF_NATIVE_MIN_WORDS_FOR_TEXT_ONLY}"
            )
            return True, reasons
        reasons.append(
            "classification=native_pdf because extracted text density and text-page coverage exceed thresholds"
        )
        return False, reasons

    reasons.append(
        "classification=scanned_pdf because extracted text density or text-page coverage is below thresholds"
    )
    return True, reasons


RoutingDecision.model_rebuild()
