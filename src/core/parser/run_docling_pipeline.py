from __future__ import annotations

import html
import json
import os
import sys
import traceback
from pathlib import Path
from pathlib import Path as _Path

try:
    from .docling_parser import DoclingParser
    from .document_artifact import (
        BoundingBox,
        DocumentArtifact,
        DocumentConfidence,
        DocumentStatus,
        DocumentType,
        EvidenceSpan,
        HandoffLane,
        LogicalSection,
        LogicalSectionItem,
        ParserUsed,
        SectionType,
        SourceFormat,
    )
    from .document_quality import (
        assess_document_payload,
        choose_best_quality,
        classify_document_status,
        infer_document_type,
        should_try_fallback,
        structure_to_text,
    )
    from .handoff_policy import HandoffQueueEntry, build_handoff_registry, build_queue_entry, decide_handoff
    from .document_router import RoutingDecision, route_document
    from .postprocess_docling import postprocess_docling, postprocess_docx_markdown
    from .secondary_parser import parse_with_secondary_parser
except ImportError:
    repo_root = _Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from src.core.parser.docling_parser import DoclingParser  # type: ignore
    from src.core.parser.document_artifact import (  # type: ignore
        BoundingBox,
        DocumentArtifact,
        DocumentConfidence,
        DocumentStatus,
        DocumentType,
        EvidenceSpan,
        HandoffLane,
        LogicalSection,
        LogicalSectionItem,
        ParserUsed,
        SectionType,
        SourceFormat,
    )
    from src.core.parser.document_quality import (  # type: ignore
        assess_document_payload,
        choose_best_quality,
        classify_document_status,
        infer_document_type,
        should_try_fallback,
        structure_to_text,
    )
    from src.core.parser.handoff_policy import (  # type: ignore
        HandoffQueueEntry,
        build_handoff_registry,
        build_queue_entry,
        decide_handoff,
    )
    from src.core.parser.document_router import RoutingDecision, route_document  # type: ignore
    from src.core.parser.postprocess_docling import (  # type: ignore
        postprocess_docling,
        postprocess_docx_markdown,
    )
    from src.core.parser.secondary_parser import parse_with_secondary_parser  # type: ignore


RAW_ROOT = Path(os.getenv("MODULE1_RAW_ROOT", "data/raw_cv"))
OUT_ROOT = Path(os.getenv("MODULE1_OUT_ROOT", "data/processed"))
REPORT_PATH = Path(os.getenv("MODULE1_REPORT_PATH", str(OUT_ROOT / "module1_pipeline_report.json")))
HANDOFF_ROOT = OUT_ROOT / "handoff"

FORMAT_EXTS: dict[SourceFormat, set[str]] = {
    SourceFormat.PDF: {".pdf"},
    SourceFormat.DOCX: {".docx"},
    SourceFormat.IMAGES: {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"},
    SourceFormat.SCANS: {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"},
}

SOURCE_KIND = {
    SourceFormat.PDF: "pdf",
    SourceFormat.DOCX: "docx",
    SourceFormat.IMAGES: "image",
    SourceFormat.SCANS: "scan",
}

SECTION_TYPE_ALIASES: dict[str, SectionType] = {
    "HEADER": SectionType.HEADER,
    "PROFESSIONAL SUMMARY": SectionType.SUMMARY,
    "SUMMARY": SectionType.SUMMARY,
    "PROFILE": SectionType.SUMMARY,
    "EDUCATION": SectionType.EDUCATION,
    "EDUCATION AND TRAINING": SectionType.EDUCATION,
    "TECHNICAL SKILLS": SectionType.SKILLS,
    "SKILLS": SectionType.SKILLS,
    "MACHINE LEARNING & DEEP LEARNING": SectionType.SKILLS,
    "COMPUTER VISION & NATURAL LANGUAGE PROCESSING (NLP)": SectionType.SKILLS,
    "IDES & SOFTWARE": SectionType.SKILLS,
    "PROFESSIONAL EXPERIENCE": SectionType.EXPERIENCE,
    "WORK EXPERIENCE": SectionType.EXPERIENCE,
    "ACADEMIC PROJECTS": SectionType.PROJECTS,
    "PROJECTS": SectionType.PROJECTS,
    "LANGUAGES": SectionType.LANGUAGES,
    "SOFT SKILLS": SectionType.SOFT_SKILLS,
    "CERTIFICATIONS": SectionType.CERTIFICATIONS,
    "CERTIFICATES (IN PROGRESS)": SectionType.CERTIFICATIONS,
    "JOB SUMMARY": SectionType.JOB_SUMMARY,
    "RESPONSIBILITIES": SectionType.RESPONSIBILITIES,
    "REQUIREMENTS": SectionType.REQUIREMENTS,
}


def _iter_files() -> list[Path]:
    files: list[Path] = []
    for source_format, extensions in FORMAT_EXTS.items():
        folder = RAW_ROOT / source_format.value
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                files.append(path)
    return files


def _output_paths(source_format: SourceFormat, stem: str) -> dict[str, str]:
    out_dir = OUT_ROOT / source_format.value
    return {
        "txt": str(out_dir / f"{stem}.txt"),
        "md": str(out_dir / f"{stem}.md"),
        "json": str(out_dir / f"{stem}.json"),
        "html": str(out_dir / f"{stem}.html"),
    }


def _handoff_paths() -> dict[str, str]:
    return {
        "accepted": str(HANDOFF_ROOT / "accepted.json"),
        "repair_required": str(HANDOFF_ROOT / "repair_required.json"),
        "quarantined": str(HANDOFF_ROOT / "quarantined.json"),
        "report": str(HANDOFF_ROOT / "module1_handoff_report.json"),
    }


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _section_type_for_title(title: str) -> SectionType:
    normalized = title.strip().upper()
    if normalized in SECTION_TYPE_ALIASES:
        return SECTION_TYPE_ALIASES[normalized]
    if "EXPERIENCE" in normalized:
        return SectionType.EXPERIENCE
    if "EDUCATION" in normalized:
        return SectionType.EDUCATION
    if "SKILL" in normalized:
        return SectionType.SKILLS
    if "PROJECT" in normalized:
        return SectionType.PROJECTS
    if "LANGUAGE" in normalized:
        return SectionType.LANGUAGES
    if "CERTIF" in normalized:
        return SectionType.CERTIFICATIONS
    if "RESPONSIBIL" in normalized:
        return SectionType.RESPONSIBILITIES
    if "REQUIREMENT" in normalized or "QUALIFICATION" in normalized:
        return SectionType.REQUIREMENTS
    return SectionType.OTHER


def _block_bbox(block: dict) -> BoundingBox | None:
    if not {"l", "r", "t", "b"}.issubset(block):
        return None
    left = min(float(block["l"]), float(block["r"]))
    right = max(float(block["l"]), float(block["r"]))
    top = min(float(block["t"]), float(block["b"]))
    bottom = max(float(block["t"]), float(block["b"]))
    return BoundingBox(left=left, top=top, right=right, bottom=bottom)


def _span_from_block(block: dict, section_title: str | None = None) -> EvidenceSpan:
    return EvidenceSpan(
        text=str(block.get("text") or "").strip(),
        page_no=max(int(block.get("page_no") or 1), 1),
        block_indices=[int(block.get("idx") or 0)],
        bbox=_block_bbox(block),
        section_title=section_title,
    )


def _fallback_span(text: str, section_title: str | None = None) -> EvidenceSpan:
    return EvidenceSpan(
        text=text,
        page_no=1,
        block_indices=[],
        bbox=None,
        section_title=section_title,
    )


def _build_block_lookup(blocks: list[dict]) -> dict[str, list[dict]]:
    lookup: dict[str, list[dict]] = {}
    for block in blocks:
        normalized = _normalize_text(str(block.get("text") or ""))
        if not normalized:
            continue
        lookup.setdefault(normalized, []).append(block)
    return lookup


def _evidence_for_text(
    text: str,
    section_title: str,
    block_lookup: dict[str, list[dict]],
    usage_counters: dict[str, int],
) -> list[EvidenceSpan]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    matches = block_lookup.get(normalized, [])
    if matches:
        offset = usage_counters.get(normalized, 0)
        if offset < len(matches):
            usage_counters[normalized] = offset + 1
            return [_span_from_block(matches[offset], section_title)]
        return [_span_from_block(matches[-1], section_title)]

    return [_fallback_span(normalized, section_title)]


def _logical_sections_from_payload(payload: dict) -> tuple[list[LogicalSection], list[EvidenceSpan]]:
    structure = payload.get("structure") or {}
    sections = structure.get("sections") or []
    blocks = payload.get("blocks") or []
    block_lookup = _build_block_lookup(blocks)
    usage_counters: dict[str, int] = {}
    artifact_spans: list[EvidenceSpan] = []
    logical_sections: list[LogicalSection] = []

    for section in sections:
        title = str(section.get("title") or "").strip() or "UNTITLED"
        content: list[str] = []
        items: list[LogicalSectionItem] = []
        section_spans: list[EvidenceSpan] = []

        for line in section.get("lines", []):
            line_text = line["text"] if isinstance(line, dict) else str(line)
            normalized = _normalize_text(line_text)
            if not normalized:
                continue
            content.append(normalized)
            line_spans = _evidence_for_text(normalized, title, block_lookup, usage_counters)
            section_spans.extend(line_spans)
            artifact_spans.extend(line_spans)

        for item in section.get("items", []):
            item_title = _normalize_text(str(item.get("title") or ""))
            item_date = _normalize_text(str(item.get("date") or "")) or None
            item_location = _normalize_text(str(item.get("location") or "")) or None
            item_details = [
                normalized
                for detail in item.get("details", [])
                if (normalized := _normalize_text(str(detail)))
            ]

            item_spans: list[EvidenceSpan] = []
            for fragment in [item_title, item_date, item_location, *item_details]:
                if fragment:
                    spans = _evidence_for_text(fragment, title, block_lookup, usage_counters)
                    item_spans.extend(spans)
                    section_spans.extend(spans)
                    artifact_spans.extend(spans)

            if item_title:
                items.append(
                    LogicalSectionItem(
                        title=item_title,
                        date=item_date,
                        location=item_location,
                        details=item_details,
                        evidence_spans=item_spans,
                    )
                )

        if content or items or section_spans:
            logical_sections.append(
                LogicalSection(
                    title=title,
                    section_type=_section_type_for_title(title),
                    content=content,
                    items=items,
                    evidence_spans=section_spans,
                )
            )

    if not artifact_spans and blocks:
        artifact_spans = [_span_from_block(block) for block in blocks if str(block.get("text") or "").strip()]

    return logical_sections, artifact_spans


def _logical_sections_to_markdown(sections: list[LogicalSection]) -> str:
    chunks: list[str] = []
    for section in sections:
        chunks.append(f"## {section.title}")
        chunks.append("")
        chunks.extend(section.content)
        for item in section.items:
            chunks.append(f"### {item.title}")
            if item.date:
                chunks.append(f"**Date:** {item.date}")
            if item.location:
                chunks.append(f"**Location:** {item.location}")
            for detail in item.details:
                chunks.append(f"- {detail}")
        chunks.append("")
    return "\n".join(chunks).strip()


def _artifact_html(markdown: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Document Artifact</title></head><body><pre>"
        f"{html.escape(markdown)}"
        "</pre></body></html>"
    )


def _document_type_enum(path: Path) -> DocumentType:
    inferred = infer_document_type(path)
    return DocumentType(inferred)


def _build_parser(decision: RoutingDecision) -> DoclingParser:
    return DoclingParser(do_ocr=decision.ocr_required)


def _run_primary_parse(path: Path, decision: RoutingDecision) -> tuple[dict, str | None, ParserUsed]:
    parser = _build_parser(decision)

    if decision.primary_parser == ParserUsed.DOCLING_MARKDOWN:
        markdown = parser.parse(str(path))
        payload = postprocess_docx_markdown(markdown)
        return payload, markdown, ParserUsed.DOCLING_MARKDOWN

    doc_dict = parser.convert_to_dict(str(path))
    payload = postprocess_docling(doc_dict, SOURCE_KIND[decision.source_format])
    return payload, None, ParserUsed.DOCLING_STRUCTURED


def _run_secondary_parse(path: Path, source_format: SourceFormat) -> tuple[dict, list[str]]:
    payload, meta = parse_with_secondary_parser(str(path), source_format.value)
    return payload, list(meta.get("warnings", []))


def _score_penalty(decision: RoutingDecision, parser_used: ParserUsed, fallback_used: bool) -> float:
    penalty = 0.0
    if decision.is_scanned:
        penalty += 0.03
    if decision.ocr_required:
        penalty += 0.02
    if fallback_used:
        penalty += 0.03
    if parser_used == ParserUsed.SECONDARY_PARSER:
        penalty += 0.05
    return penalty


def _document_confidence(
    quality: dict,
    decision: RoutingDecision,
    parser_used: ParserUsed,
    *,
    fallback_used: bool,
    extra_warnings: list[str] | None = None,
) -> DocumentConfidence:
    parser_signals = dict(quality["signals"])
    parser_signals.update(
        {
            "router_is_scanned": decision.is_scanned,
            "router_ocr_required": decision.ocr_required,
            "router_primary_parser": decision.primary_parser.value,
            "router_reason_count": len(decision.routing_reasons),
            "routing_decision": json.dumps(decision.model_dump(), ensure_ascii=False),
            "parser_used": parser_used.value,
            "fallback_used": fallback_used,
        }
    )

    penalty = _score_penalty(decision, parser_used, fallback_used)
    adjusted_score = max(0.0, min(float(quality["document_confidence_score"]) - penalty, 1.0))
    warnings = list(quality["warnings"])
    if extra_warnings:
        warnings.extend(extra_warnings)

    status = classify_document_status(
        confidence=adjusted_score,
        useful_text_chars=int(parser_signals["useful_text_chars"]),
        section_completeness=float(parser_signals["section_completeness"]),
        warnings=warnings,
        source_format=decision.source_format.value,
    )

    parser_signals["confidence_penalty"] = round(penalty, 4)

    return DocumentConfidence(
        score=round(adjusted_score, 4),
        status=DocumentStatus(status),
        signals=parser_signals,
        warnings=warnings,
    )


def _build_artifact(
    path: Path,
    decision: RoutingDecision,
    payload: dict,
    *,
    parser_used: ParserUsed,
    fallback_used: bool,
    quality: dict,
    markdown_override: str | None = None,
    extra_warnings: list[str] | None = None,
) -> DocumentArtifact:
    logical_sections, evidence_spans = _logical_sections_from_payload(payload)
    raw_text = structure_to_text(payload.get("structure") or {})
    if not raw_text.strip():
        fallback_lines = [line.strip() for line in (payload.get("txt") or "").splitlines() if line.strip()]
        if fallback_lines:
            raw_text = "\n".join(fallback_lines)

    if not logical_sections and raw_text.strip():
        fallback_content = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if fallback_content:
            fallback_section_spans = evidence_spans or [_fallback_span(raw_text[:500], "RECOVERED_TEXT")]
            logical_sections = [
                LogicalSection(
                    title="RECOVERED_TEXT",
                    section_type=SectionType.OTHER,
                    content=fallback_content,
                    items=[],
                    evidence_spans=fallback_section_spans,
                )
            ]

    markdown = markdown_override or _logical_sections_to_markdown(logical_sections)
    confidence = _document_confidence(
        quality,
        decision,
        parser_used,
        fallback_used=fallback_used,
        extra_warnings=extra_warnings,
    )

    if not raw_text.strip() and markdown.strip():
        raw_text = markdown

    if not evidence_spans and raw_text.strip():
        evidence_spans = [_fallback_span(raw_text[:500], "DOCUMENT")]

    handoff_decision = decide_handoff(
        document_status=confidence.status,
        source_format=decision.source_format,
        quality_flags=list(confidence.warnings),
    )

    return DocumentArtifact(
        source_path=str(path),
        source_format=decision.source_format,
        document_type=_document_type_enum(path),
        document_status=confidence.status,
        raw_text=raw_text,
        markdown=markdown,
        logical_sections=logical_sections,
        parser_used=parser_used,
        fallback_used=fallback_used,
        document_confidence=confidence,
        handoff_decision=handoff_decision,
        evidence_spans=evidence_spans,
    )


def _write_artifact(artifact: DocumentArtifact, stem: str) -> dict[str, str]:
    out_dir = OUT_ROOT / artifact.source_format.value
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = _output_paths(artifact.source_format, stem)

    Path(outputs["txt"]).write_text(artifact.raw_text, encoding="utf-8")
    Path(outputs["md"]).write_text(artifact.markdown, encoding="utf-8")
    Path(outputs["html"]).write_text(_artifact_html(artifact.markdown), encoding="utf-8")
    Path(outputs["json"]).write_text(
        artifact.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return outputs


def _candidate_from_payload(
    path: Path,
    decision: RoutingDecision,
    payload: dict,
    parser_used: ParserUsed,
    *,
    fallback_used: bool,
    markdown_override: str | None = None,
    extra_warnings: list[str] | None = None,
) -> dict:
    quality = assess_document_payload(
        payload,
        source_path=str(path),
        source_format=decision.source_format.value,
        document_type=infer_document_type(path),
        route_taken=parser_used.value,
    )
    artifact = _build_artifact(
        path,
        decision,
        payload,
        parser_used=parser_used,
        fallback_used=fallback_used,
        quality=quality,
        markdown_override=markdown_override,
        extra_warnings=extra_warnings,
    )
    return {
        "payload": payload,
        "parser_used": parser_used,
        "fallback_used": fallback_used,
        "quality": quality,
        "artifact": artifact,
        "warnings": list(extra_warnings or []),
    }


def _process_file(path: Path) -> dict:
    decision = route_document(path)
    attempts: list[dict] = []

    try:
        primary_payload, primary_markdown, primary_parser_used = _run_primary_parse(path, decision)
        primary_candidate = _candidate_from_payload(
            path,
            decision,
            primary_payload,
            primary_parser_used,
            fallback_used=False,
            markdown_override=primary_markdown,
        )
        attempts.append(
            {
                "parser_used": primary_candidate["parser_used"].value,
                "document_confidence_score": primary_candidate["artifact"].document_confidence.score,
                "document_status": primary_candidate["artifact"].document_status.value,
                "fallback_used": False,
            }
        )
    except Exception as primary_exc:
        primary_payload = None
        primary_candidate = None
        attempts.append(
            {
                "parser_used": decision.primary_parser.value,
                "document_confidence_score": 0.0,
                "document_status": DocumentStatus.FAILED.value,
                "fallback_used": False,
                "error": f"primary_parse_failed: {type(primary_exc).__name__}: {primary_exc}",
            }
        )
        primary_error = primary_exc
    else:
        primary_error = None

    fallback_candidate = None
    fallback_warnings: list[str] = []
    should_fallback = primary_candidate is None or should_try_fallback(
        primary_candidate["quality"],
        source_format=decision.source_format.value,
        route_taken=primary_candidate["parser_used"].value,
    )

    if should_fallback:
        try:
            fallback_payload, fallback_warnings = _run_secondary_parse(path, decision.source_format)
            fallback_candidate = _candidate_from_payload(
                path,
                decision,
                fallback_payload,
                ParserUsed.SECONDARY_PARSER,
                fallback_used=True,
                extra_warnings=fallback_warnings,
            )
            attempts.append(
                {
                    "parser_used": fallback_candidate["parser_used"].value,
                    "document_confidence_score": fallback_candidate["artifact"].document_confidence.score,
                    "document_status": fallback_candidate["artifact"].document_status.value,
                    "fallback_used": True,
                }
            )
        except Exception as fallback_exc:
            attempts.append(
                {
                    "parser_used": ParserUsed.SECONDARY_PARSER.value,
                    "document_confidence_score": 0.0,
                    "document_status": DocumentStatus.FAILED.value,
                    "fallback_used": True,
                    "error": f"secondary_parse_failed: {type(fallback_exc).__name__}: {fallback_exc}",
                }
            )

    if primary_candidate is None and fallback_candidate is None:
        root_error = primary_error or RuntimeError("No candidate could be built.")
        return {
            "source_format": decision.source_format.value,
            "source_path": str(path),
            "status": "failure",
            "document_status": DocumentStatus.FAILED.value,
            "handoff_lane": HandoffLane.QUARANTINED.value,
            "eligible_for_module2": False,
            "error": f"{type(root_error).__name__}: {root_error}",
            "traceback": traceback.format_exc(limit=5),
            "routing_decision": decision.model_dump(),
            "attempts": attempts,
            "outputs": _output_paths(decision.source_format, path.stem),
        }

    final_candidate = primary_candidate
    if fallback_candidate is not None and (
        final_candidate is None
        or choose_best_quality(final_candidate["quality"], fallback_candidate["quality"])
    ):
        final_candidate = fallback_candidate

    assert final_candidate is not None
    artifact: DocumentArtifact = final_candidate["artifact"]
    outputs = _write_artifact(artifact, path.stem)
    handoff_entry = build_queue_entry(artifact, outputs["json"])

    return {
        "source_format": artifact.source_format.value,
        "source_path": artifact.source_path,
        "status": "success",
        "document_status": artifact.document_status.value,
        "error": None,
        "document_confidence_score": artifact.document_confidence.score,
        "parser_used": artifact.parser_used.value,
        "fallback_used": artifact.fallback_used,
        "document_type": artifact.document_type.value,
        "routing_decision": decision.model_dump(),
        "routing_reasons": decision.routing_reasons,
        "warnings": artifact.document_confidence.warnings,
        "handoff_lane": artifact.handoff_decision.lane.value,
        "eligible_for_module2": artifact.handoff_decision.eligible_for_module2,
        "handoff_decision": artifact.handoff_decision.model_dump(),
        "handoff_queue_entry": handoff_entry.model_dump(),
        "attempts": attempts,
        "outputs": outputs,
    }


def _build_summary(results: list[dict]) -> dict:
    summary: dict[str, object] = {
        "total_files": len(results),
        "pipeline_successes": sum(1 for result in results if result["status"] == "success"),
        "failures": sum(1 for result in results if result["status"] == "failure"),
        "document_statuses": {},
        "handoff_lanes": {},
        "by_format": {},
    }

    by_format: dict[str, dict[str, int]] = {}
    document_statuses: dict[str, int] = {}
    handoff_lanes: dict[str, int] = {}

    for result in results:
        fmt = str(result["source_format"])
        stats = by_format.setdefault(
            fmt,
            {
                "total": 0,
                "pipeline_successes": 0,
                "failures": 0,
                "validated": 0,
                "partial": 0,
                "uncertain": 0,
                "failed": 0,
            },
        )
        stats["total"] += 1
        if result["status"] == "success":
            stats["pipeline_successes"] += 1
        else:
            stats["failures"] += 1

        document_status = str(result.get("document_status", DocumentStatus.FAILED.value))
        document_statuses[document_status] = document_statuses.get(document_status, 0) + 1
        if document_status in stats:
            stats[document_status] += 1

        handoff_lane = str(result.get("handoff_lane", HandoffLane.QUARANTINED.value))
        handoff_lanes[handoff_lane] = handoff_lanes.get(handoff_lane, 0) + 1

    summary["by_format"] = by_format
    summary["document_statuses"] = document_statuses
    summary["handoff_lanes"] = handoff_lanes
    return summary


def _write_handoff_registry(results: list[dict]) -> dict:
    HANDOFF_ROOT.mkdir(parents=True, exist_ok=True)
    paths = _handoff_paths()
    entries = [
        HandoffQueueEntry(**result["handoff_queue_entry"])
        for result in results
        if result["status"] == "success" and result.get("handoff_queue_entry")
    ]
    registry = build_handoff_registry(entries)

    Path(paths["accepted"]).write_text(
        json.dumps([entry.model_dump(mode="json") for entry in registry.accepted], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(paths["repair_required"]).write_text(
        json.dumps(
            [entry.model_dump(mode="json") for entry in registry.repair_required],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    Path(paths["quarantined"]).write_text(
        json.dumps([entry.model_dump(mode="json") for entry in registry.quarantined], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    handoff_report = {
        "summary": {
            "accepted": len(registry.accepted),
            "repair_required": len(registry.repair_required),
            "quarantined": len(registry.quarantined),
            "eligible_for_module2": len(registry.accepted),
            "blocked_from_module2": len(registry.repair_required) + len(registry.quarantined),
        },
        "paths": paths,
    }
    Path(paths["report"]).write_text(
        json.dumps(handoff_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return handoff_report


def main() -> None:
    files = _iter_files()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = [_process_file(path) for path in files]
    handoff_report = _write_handoff_registry(results)
    report = {
        "raw_root": str(RAW_ROOT),
        "out_root": str(OUT_ROOT),
        "handoff_root": str(HANDOFF_ROOT),
        "results": results,
        "summary": _build_summary(results),
        "handoff": handoff_report,
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== PIPELINE REPORT ===")
    print(f"Fichiers détectés : {len(files)}")
    print(f"Succès pipeline : {report['summary']['pipeline_successes']}")
    print(f"Échecs : {report['summary']['failures']}")
    print(f"Rapport JSON : {REPORT_PATH}")
    print(f"Statuts documentaires : {report['summary']['document_statuses']}")
    print(f"Handoff lanes : {report['summary']['handoff_lanes']}")

    for fmt, stats in report["summary"]["by_format"].items():
        print(
            f"- {fmt}: total={stats['total']} "
            f"pipeline_success={stats['pipeline_successes']} "
            f"validated={stats['validated']} partial={stats['partial']} "
            f"uncertain={stats['uncertain']} failure={stats['failures']}"
        )


if __name__ == "__main__":
    main()
