from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency for local developer convenience.
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


LOGGER = logging.getLogger("profile_builder")

ACCEPTED_PATH = Path(os.getenv("MODULE1_ACCEPTED_PATH", "data/processed/handoff/accepted.json"))
DEFAULT_OPENAI_MODEL = os.getenv("PROFILE_BUILDER_MODEL", "llama-3.3-70b-specdec")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "talent_intelligence")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "profiles")
MONGODB_RUNS_COLLECTION = os.getenv("MONGODB_RUNS_COLLECTION", "profile_builder_runs")
PREVIEW_ROOT = Path(os.getenv("PROFILE_BUILDER_PREVIEW_ROOT", "data/profile_builder_preview"))
RUN_REPORT_PATH = PREVIEW_ROOT / "run_report.json"
MIN_MARKDOWN_CHARS = 120
MIN_MARKDOWN_WORDS = 25
MAX_SUMMARY_CHARS = 240
MARKETING_SUMMARY_PHRASES = (
    "results-driven",
    "highly motivated",
    "dynamic professional",
    "passionate professional",
    "proven track record",
    "world-class",
    "best-in-class",
)
GENERIC_HARD_SKILL_LABELS = {
    "areas of interest",
    "big data",
    "bi tools",
    "bi & etl tools",
    "cloud",
    "cloud (aws)",
    "computer vision",
    "data engineering",
    "data science",
    "databases",
    "deep learning",
    "deployment",
    "deployment tools",
    "devops/cicd",
    "distributed processing",
    "frameworks",
    "generative models",
    "ides & software",
    "image processing",
    "image/video processing",
    "machine learning",
    "ml ops",
    "natural language processing",
    "natural language processing (nlp)",
    "neural networks",
    "nlp",
    "programming languages",
    "software",
    "tools",
    "transfer learning",
    "video processing",
}
MARKETING_SOFT_SKILLS = {
    "dynamic",
    "innovative",
    "meticulous",
    "motivated",
    "passionate",
    "results-driven",
}
SOURCE_FORMAT_PRIORITY = {
    "docx": 0,
    "pdf": 1,
    "images": 2,
    "image": 2,
    "scans": 3,
    "scan": 3,
}


class DocumentSkipError(RuntimeError):
    """Raised when a document should be skipped before calling the LLM."""


class BusinessValidationError(RuntimeError):
    """Raised when a structured profile is valid JSON but fails factual guardrails."""


class ExperienceItem(BaseModel):
    """Professional experience item extracted from the Module 1 markdown."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    job_title: str = Field(..., description="Candidate role title exactly as supported by the source text.")
    company: str | None = Field(default=None, description="Employer or organization name if present.")
    start_date: str | None = Field(default=None, description="Best-effort normalized start date.")
    end_date: str | None = Field(default=None, description="Best-effort normalized end date or 'Present'.")
    city: str | None = Field(default=None, description="Best-effort city or location for the experience.")
    responsibilities: list[str] = Field(
        default_factory=list,
        description="Source-grounded responsibilities or missions with no hallucinated content.",
    )


class EducationItem(BaseModel):
    """Education item extracted from the Module 1 markdown."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    degree: str = Field(..., description="Degree or diploma name as supported by the source text.")
    school: str | None = Field(default=None, description="School or university name if present.")
    year: str | None = Field(default=None, description="Graduation year or best-effort completion year.")


class CandidateBio(BaseModel):
    """Identity and contact data for the candidate."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    full_name: str | None = Field(default=None, description="Candidate full name if found in the document.")
    email: str | None = Field(default=None, description="Candidate email address if found in the document.")
    phone: str | None = Field(default=None, description="Candidate phone number if found in the document.")
    location: str | None = Field(default=None, description="Candidate city, country, or best-effort location.")

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        """Reject malformed email addresses instead of polluting MongoDB."""
        if value is None or value == "":
            return None
        normalized = value.strip()
        if "@" not in normalized or "." not in normalized.rsplit("@", 1)[-1]:
            raise ValueError("Invalid email address.")
        return normalized


class CandidateExpertise(BaseModel):
    """Expertise block used by Module 2 and later matching layers."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    summary: str = Field(
        ...,
        description="Short factual summary grounded only in the provided markdown. No invented claims.",
    )
    hard_skills: list[str] = Field(
        default_factory=list,
        description="Technical or domain skills explicitly supported by the source.",
    )
    soft_skills: list[str] = Field(
        default_factory=list,
        description="Soft skills explicitly supported by the source.",
    )


class CandidateMetadata(BaseModel):
    """Traceability metadata attached to the structured profile."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    extraction_date: str = Field(..., description="UTC timestamp when the profile builder processed the document.")
    model_used: str = Field(..., description="LLM model or provider route used to build the profile.")
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score inherited from Module 1 accepted artifact.",
    )


class CandidateProfile(BaseModel):
    """Canonical structured candidate profile stored in MongoDB."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    source_id: str = Field(
        ...,
        min_length=1,
        description="Stable identifier derived from a trusted email when available, otherwise from the source file.",
    )
    bio: CandidateBio = Field(..., description="Candidate identity and contact information.")
    expertise: CandidateExpertise = Field(..., description="Candidate expertise summary and skills.")
    experiences: list[ExperienceItem] = Field(
        default_factory=list,
        description="Professional experiences grounded in the source markdown.",
    )
    education: list[EducationItem] = Field(
        default_factory=list,
        description="Education history grounded in the source markdown.",
    )
    metadata: CandidateMetadata = Field(..., description="Traceability metadata for the structured profile.")


class CandidateProfilePayload(BaseModel):
    """LLM payload before deterministic fields are injected by local code."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    bio: CandidateBio = Field(..., description="Candidate identity and contact information.")
    expertise: CandidateExpertise = Field(..., description="Candidate expertise summary and skills.")
    experiences: list[ExperienceItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)


class AcceptedArtifactRef(BaseModel):
    """Minimal accepted artifact row emitted by Module 1 handoff."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, validate_assignment=True)

    source_path: str
    artifact_path: str
    source_format: str
    document_status: str
    handoff_lane: str
    eligible_for_module2: bool
    parser_used: str
    document_confidence_score: float
    quality_flags: list[str] = Field(default_factory=list)
    next_action: str | None = None


@dataclass(slots=True)
class DocumentContext:
    """Material sent to the LLM for one accepted Module 1 artifact."""

    accepted_entry: AcceptedArtifactRef
    artifact_path: Path
    markdown_path: Path
    artifact: dict[str, Any]
    markdown: str


def configure_logging(level: int = logging.INFO) -> None:
    """Configure process-level logging once for the profile builder entrypoint."""
    if LOGGER.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_accepted_entries(accepted_path: Path = ACCEPTED_PATH) -> tuple[list[AcceptedArtifactRef], list[dict[str, Any]]]:
    """Load only trusted Module 1 handoff rows and ignore incoherent accepted entries."""
    rows = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted: list[AcceptedArtifactRef] = []
    ignored: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        try:
            entry = AcceptedArtifactRef(**row)
        except Exception as exc:
            message = f"accepted_row_invalid: {exc}"
            LOGGER.warning("Entree accepted ignoree [%s] | raison=%s", index, message)
            ignored.append(
                {
                    "row_index": index,
                    "source_path": row.get("source_path"),
                    "status": "ignored",
                    "reason": message,
                }
            )
            continue

        reasons: list[str] = []
        if entry.handoff_lane != "accepted":
            reasons.append(f"handoff_lane={entry.handoff_lane}")
        if entry.document_status != "validated":
            reasons.append(f"document_status={entry.document_status}")
        if not entry.eligible_for_module2:
            reasons.append("eligible_for_module2=false")

        if reasons:
            message = ", ".join(reasons)
            LOGGER.warning(
                "Entree accepted incoherente ignoree | source=%s | raison=%s",
                entry.source_path,
                message,
            )
            ignored.append(
                {
                    "row_index": index,
                    "source_path": entry.source_path,
                    "artifact_path": entry.artifact_path,
                    "status": "ignored",
                    "reason": message,
                }
            )
            continue

        accepted.append(entry)

    return accepted, ignored


def build_document_context(entry: AcceptedArtifactRef) -> DocumentContext:
    """Read the accepted artifact JSON and its paired markdown file from Module 1."""
    artifact_path = Path(entry.artifact_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    markdown_path = artifact_path.with_suffix(".md")
    markdown = markdown_path.read_text(encoding="utf-8").strip()

    if not markdown:
        raise DocumentSkipError("empty_markdown")

    word_count = len(re.findall(r"\w+", markdown))
    if len(markdown) < MIN_MARKDOWN_CHARS or word_count < MIN_MARKDOWN_WORDS:
        raise DocumentSkipError(f"markdown_too_short chars={len(markdown)} words={word_count}")

    return DocumentContext(
        accepted_entry=entry,
        artifact_path=artifact_path,
        markdown_path=markdown_path,
        artifact=artifact,
        markdown=markdown,
    )


def build_candidate_profile(context: DocumentContext) -> CandidateProfile:
    """Create one structured candidate profile from a trusted Module 1 document."""
    llm_payload, model_used = extract_profile_payload(context)
    llm_payload = prepare_llm_payload(llm_payload)
    payload = CandidateProfilePayload.model_validate(llm_payload)
    profile = build_final_profile(payload, context, model_used=model_used)
    profile = apply_quality_guards(profile, context)
    profile = validate_profile_business_rules(profile, context)
    profile = enrich_profile(profile, context, model_used=model_used)
    return profile


def extract_profile_payload(context: DocumentContext) -> tuple[dict[str, Any], str]:
    """
    Extract a strict JSON object from the accepted markdown using Groq.

    This Module 2 rollout is now Groq-only and intentionally ignores Gemini/Ollama.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        payload = extract_with_openai(context, api_key=openai_api_key, model=DEFAULT_OPENAI_MODEL)
        return payload, DEFAULT_OPENAI_MODEL

    raise RuntimeError(
        "No Groq/OpenAI-compatible provider configured. Set OPENAI_API_KEY."
    )

def extract_with_openai(context: DocumentContext, *, api_key: str, model: str) -> dict[str, Any]:
    """Call a Groq OpenAI-compatible endpoint and force JSON object output."""
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt(),
            },
            {
                "role": "user",
                "content": user_prompt(context),
            },
        ],
        "response_format": {"type": "json_object"},
    }

    response = _http_post_json(
        _chat_completions_url(OPENAI_BASE_URL),
        body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    text = _extract_chat_completions_text(response)
    if not text:
        raise RuntimeError(f"Groq response did not contain JSON output: {response}")
    return json.loads(text)


def build_final_profile(
    payload: CandidateProfilePayload,
    context: DocumentContext,
    *,
    model_used: str,
) -> CandidateProfile:
    """Build the final profile with deterministic fields injected locally."""
    summary = sanitize_summary(payload.expertise.summary)
    expertise = payload.expertise.model_copy(update={"summary": summary})
    metadata = CandidateMetadata(
        extraction_date=datetime.now(timezone.utc).isoformat(),
        model_used=model_used,
        confidence_score=float(context.accepted_entry.document_confidence_score),
    )
    return CandidateProfile(
        source_id="pending_source_id",
        bio=payload.bio,
        expertise=expertise,
        experiences=payload.experiences,
        education=payload.education,
        metadata=metadata,
    )


def enrich_profile(profile: CandidateProfile, context: DocumentContext, *, model_used: str) -> CandidateProfile:
    """Fill deterministic metadata and normalize the source identifier after LLM extraction."""
    email = profile.bio.email
    source_stem = context.artifact_path.stem
    source_format = context.accepted_entry.source_format
    source_id = email.strip().lower() if email else f"{source_format}:{source_stem}"
    metadata = CandidateMetadata(
        extraction_date=datetime.now(timezone.utc).isoformat(),
        model_used=model_used,
        confidence_score=float(context.accepted_entry.document_confidence_score),
    )
    return profile.model_copy(
        update={
            "source_id": source_id,
            "metadata": metadata,
        }
    )


def build_document_id(context: DocumentContext) -> str:
    """Build a stable identifier for one concrete source document."""
    source_stem = context.artifact_path.stem
    source_format = context.accepted_entry.source_format
    return f"{source_format}:{source_stem}"


def build_candidate_identity(profile: CandidateProfile, context: DocumentContext) -> tuple[str, float, bool]:
    """Build a safe candidate identity key without merging on name alone."""
    full_name = _normalize_identity_component(profile.bio.full_name)
    email = profile.bio.email.strip().lower() if profile.bio.email else ""
    phone = _digits_only(profile.bio.phone or "")
    location = _normalize_identity_component(profile.bio.location)

    if email:
        return f"email:{email}", 1.0, False
    if full_name and len(phone) >= 6:
        return f"name_phone:{full_name}|{phone}", 0.9, False
    if full_name and location:
        return f"name_location:{full_name}|{location}", 0.7, False

    return f"document:{build_document_id(context)}", 0.3, True


def _normalize_identity_component(value: str | None) -> str:
    """Normalize one identity component for stable key construction."""
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def build_persistence_document(
    profile: CandidateProfile,
    context: DocumentContext,
    *,
    run_id: str,
    mode: str,
    accepted_path: Path,
) -> dict[str, Any]:
    """Build one candidate-level MongoDB payload without mutating the preview shape."""
    source_record = build_source_record(profile, context)
    best_source_by_field = build_best_source_by_field(profile, source_record)
    candidate_key, identity_confidence, identity_ambiguous = build_candidate_identity(profile, context)
    return {
        "candidate_key": candidate_key,
        "identity_confidence": identity_confidence,
        "identity_ambiguous": identity_ambiguous,
        "bio": profile.bio.model_dump(mode="json"),
        "expertise": profile.expertise.model_dump(mode="json"),
        "experiences": _clean_experience_documents([item.model_dump(mode="json") for item in profile.experiences]),
        "education": _clean_education_documents([item.model_dump(mode="json") for item in profile.education]),
        "metadata": profile.metadata.model_dump(mode="json"),
        "quality_flags": list(dict.fromkeys(context.accepted_entry.quality_flags)),
        "reliability_score": float(context.accepted_entry.document_confidence_score),
        "sources_used": [source_record],
        "source_formats_seen": [context.accepted_entry.source_format],
        "best_source_by_field": best_source_by_field,
        "run": {
            "run_id": run_id,
            "mode": mode,
            "accepted_file_path": str(accepted_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def build_source_record(profile: CandidateProfile, context: DocumentContext) -> dict[str, Any]:
    """Capture one source document used to build or enrich a consolidated candidate."""
    return {
        "document_id": build_document_id(context),
        "source_id": profile.source_id,
        "source_path": context.accepted_entry.source_path,
        "artifact_path": context.accepted_entry.artifact_path,
        "source_format": context.accepted_entry.source_format,
        "parser_used": context.accepted_entry.parser_used,
        "document_confidence_score": float(context.accepted_entry.document_confidence_score),
        "quality_flags": list(context.accepted_entry.quality_flags),
    }


def build_best_source_by_field(profile: CandidateProfile, source_record: dict[str, Any]) -> dict[str, Any]:
    """Attach the most reliable source seen so far for each major field."""
    field_map: dict[str, Any] = {}
    if profile.bio.full_name:
        field_map["bio.full_name"] = source_record
    if profile.bio.email:
        field_map["bio.email"] = source_record
    if profile.bio.phone:
        field_map["bio.phone"] = source_record
    if profile.bio.location:
        field_map["bio.location"] = source_record
    if profile.expertise.summary:
        field_map["expertise.summary"] = source_record
    if profile.expertise.hard_skills:
        field_map["expertise.hard_skills"] = source_record
    if profile.expertise.soft_skills:
        field_map["expertise.soft_skills"] = source_record
    if profile.experiences:
        field_map["experiences"] = source_record
    if profile.education:
        field_map["education"] = source_record
    return field_map


def merge_candidate_documents(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple source documents into one consolidated candidate profile."""
    if not existing:
        return incoming

    merged = dict(existing)
    merged["candidate_key"] = incoming["candidate_key"]
    merged["bio"], bio_sources = _merge_bio(
        existing.get("bio") or {},
        incoming.get("bio") or {},
        existing.get("best_source_by_field") or {},
        incoming.get("best_source_by_field") or {},
    )
    merged_expertise, expertise_sources = _merge_expertise(
        existing.get("expertise") or {},
        incoming.get("expertise") or {},
        existing.get("best_source_by_field") or {},
        incoming.get("best_source_by_field") or {},
    )
    merged["expertise"] = merged_expertise
    merged["experiences"] = _merge_experience_documents(
        existing.get("experiences") or [],
        incoming.get("experiences") or [],
        existing.get("best_source_by_field", {}).get("experiences"),
        incoming.get("best_source_by_field", {}).get("experiences"),
    )
    merged["education"] = _merge_education_documents(
        existing.get("education") or [],
        incoming.get("education") or [],
        existing.get("best_source_by_field", {}).get("education"),
        incoming.get("best_source_by_field", {}).get("education"),
    )
    merged["metadata"] = incoming.get("metadata") or existing.get("metadata") or {}
    merged["quality_flags"] = _merge_string_lists(existing.get("quality_flags") or [], incoming.get("quality_flags") or [])
    merged["reliability_score"] = max(
        float(existing.get("reliability_score") or 0.0),
        float(incoming.get("reliability_score") or 0.0),
    )
    merged["identity_confidence"] = max(
        float(existing.get("identity_confidence") or 0.0),
        float(incoming.get("identity_confidence") or 0.0),
    )
    merged["identity_ambiguous"] = bool(existing.get("identity_ambiguous")) and bool(incoming.get("identity_ambiguous"))
    merged["sources_used"] = _merge_sources_used(existing.get("sources_used") or [], incoming.get("sources_used") or [])
    merged["source_formats_seen"] = _merge_source_formats_seen(
        existing.get("source_formats_seen") or [],
        incoming.get("source_formats_seen") or [],
    )
    merged["best_source_by_field"] = dict(existing.get("best_source_by_field") or {})
    merged["best_source_by_field"].update(bio_sources)
    merged["best_source_by_field"].update(expertise_sources)
    if merged["experiences"]:
        best_exp = _pick_better_source(
            existing.get("best_source_by_field", {}).get("experiences"),
            incoming.get("best_source_by_field", {}).get("experiences"),
        )
        if best_exp:
            merged["best_source_by_field"]["experiences"] = best_exp
    if merged["education"]:
        best_edu = _pick_better_source(
            existing.get("best_source_by_field", {}).get("education"),
            incoming.get("best_source_by_field", {}).get("education"),
        )
        if best_edu:
            merged["best_source_by_field"]["education"] = best_edu
    merged["run"] = incoming.get("run") or existing.get("run") or {}
    return merged


def prepare_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Add deterministic placeholders so Pydantic validation stays in control."""
    normalized = dict(payload)
    normalized.setdefault("bio", {})
    normalized.setdefault("expertise", {"summary": "", "hard_skills": [], "soft_skills": []})
    normalized.setdefault("experiences", [])
    normalized.setdefault("education", [])
    return normalized


def validate_profile_business_rules(profile: CandidateProfile, context: DocumentContext) -> CandidateProfile:
    """Reject profiles that are structurally valid but not sufficiently grounded in the markdown."""
    markdown = context.markdown
    normalized_markdown = _normalize_for_match(markdown)
    normalized_markdown_digits = _digits_only(markdown)

    if profile.bio.email:
        normalized_email = _normalize_for_match(profile.bio.email)
        if normalized_email not in normalized_markdown:
            raise BusinessValidationError("email_not_grounded_in_markdown")

    if profile.bio.phone:
        phone_digits = _digits_only(profile.bio.phone)
        if len(phone_digits) < 6 or phone_digits not in normalized_markdown_digits:
            raise BusinessValidationError("phone_not_grounded_in_markdown")

    if not _has_structured_anchor(profile, markdown):
        raise BusinessValidationError("no_structured_anchor_found")

    if context.accepted_entry.source_format == "scans" and _is_weak_scan_profile(profile, markdown):
        raise BusinessValidationError("scan_profile_too_weak")

    return profile


def sanitize_summary(summary: str) -> str:
    """Keep only a short, descriptive summary and blank suspicious marketing phrasing."""
    cleaned = " ".join(summary.split()).strip()
    if not cleaned:
        return ""

    lowered = cleaned.lower()
    if any(phrase in lowered for phrase in MARKETING_SUMMARY_PHRASES):
        return ""

    sentence_chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", cleaned) if chunk.strip()]
    cleaned = " ".join(sentence_chunks[:2]).strip()
    if len(cleaned) > MAX_SUMMARY_CHARS:
        cleaned = cleaned[: MAX_SUMMARY_CHARS - 3].rstrip() + "..."
    return cleaned


def apply_quality_guards(profile: CandidateProfile, context: DocumentContext) -> CandidateProfile:
    """Lightweight post-LLM cleanup before business validation."""
    cleaned_summary = sanitize_summary(profile.expertise.summary)
    cleaned_hard_skills = _clean_hard_skills(profile.expertise.hard_skills, context.markdown)
    cleaned_soft_skills = _clean_soft_skills(profile.expertise.soft_skills, context.markdown)
    expertise = profile.expertise.model_copy(
        update={
            "summary": cleaned_summary,
            "hard_skills": cleaned_hard_skills,
            "soft_skills": cleaned_soft_skills,
        }
    )
    return profile.model_copy(update={"expertise": expertise})


def _has_structured_anchor(profile: CandidateProfile, markdown: str) -> bool:
    """Require at least one meaningful experience, education, or skill anchor in the source markdown."""
    if any(_is_supported_text(item.job_title, markdown) or _is_supported_text(item.company, markdown) for item in profile.experiences):
        return True
    if any(_is_supported_text(item.degree, markdown) or _is_supported_text(item.school, markdown) for item in profile.education):
        return True
    if any(_is_supported_text(skill, markdown) for skill in profile.expertise.hard_skills):
        return True
    return False


def _is_supported_text(value: str | None, markdown: str) -> bool:
    """Check whether a candidate field is visibly supported by the source markdown."""
    if not value:
        return False
    normalized_value = _normalize_for_match(value)
    if len(normalized_value) < 4:
        return False
    return normalized_value in _normalize_for_match(markdown)


def _normalize_for_match(value: str) -> str:
    """Normalize text for loose substring matching."""
    return re.sub(r"[^a-z0-9@]+", "", value.lower())


def _digits_only(value: str) -> str:
    """Extract only digits for phone support checks."""
    return re.sub(r"\D+", "", value)


def _clean_hard_skills(skills: list[str], markdown: str) -> list[str]:
    """Keep a source-grounded hard-skill list and drop broad category labels."""
    cleaned: list[str] = []
    seen: set[str] = set()

    for raw_skill in skills:
        skill = _normalize_skill_label(raw_skill)
        if not skill:
            continue
        if len(skill) < 2:
            continue
        if _is_generic_hard_skill(skill):
            continue
        if not _is_supported_text(skill, markdown):
            continue

        key = skill.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(skill)

    return cleaned


def _clean_soft_skills(skills: list[str], markdown: str) -> list[str]:
    """Drop unsupported or marketing-flavored soft skills."""
    cleaned: list[str] = []
    seen: set[str] = set()

    for raw_skill in skills:
        skill = _normalize_skill_label(raw_skill)
        if not skill:
            continue
        if skill.lower() in MARKETING_SOFT_SKILLS:
            continue
        if not _is_supported_text(skill, markdown):
            continue

        key = skill.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(skill)

    return cleaned


def _normalize_skill_label(value: str | None) -> str:
    """Normalize one skill label without inventing new content."""
    if not value:
        return ""
    cleaned = " ".join(str(value).replace("|", " ").split()).strip(" -,:;.")
    cleaned = re.sub(r"\s+/\s+", "/", cleaned)
    return cleaned


def _is_generic_hard_skill(skill: str) -> bool:
    """Reject broad section labels and umbrella categories that inflate skill lists."""
    normalized = skill.lower().strip()
    if normalized in GENERIC_HARD_SKILL_LABELS:
        return True
    if normalized.endswith(" tools") or normalized.endswith("& etl tools"):
        return True
    if normalized.endswith(" software"):
        return True
    return False


def _is_weak_scan_profile(profile: CandidateProfile, markdown: str) -> bool:
    """Force abstention on scans that are too incomplete to trust downstream."""
    supported_experiences = sum(
        1
        for item in profile.experiences
        if _is_supported_text(item.job_title, markdown) or _is_supported_text(item.company, markdown)
    )
    supported_education = sum(
        1
        for item in profile.education
        if _is_supported_text(item.degree, markdown) or _is_supported_text(item.school, markdown)
    )
    has_grounded_name = bool(profile.bio.full_name and _is_supported_text(profile.bio.full_name, markdown))

    return (not has_grounded_name) and supported_education == 0 and supported_experiences < 2


def _source_priority(source_format: str | None) -> int:
    """Return a deterministic format priority for consolidation."""
    if not source_format:
        return 99
    return SOURCE_FORMAT_PRIORITY.get(str(source_format).lower(), 99)


def _pick_better_source(existing_source: dict[str, Any] | None, incoming_source: dict[str, Any] | None) -> dict[str, Any] | None:
    """Pick the most reliable source between two candidates."""
    if not existing_source:
        return incoming_source
    if not incoming_source:
        return existing_source

    existing_rank = _source_priority(existing_source.get("source_format"))
    incoming_rank = _source_priority(incoming_source.get("source_format"))
    if incoming_rank < existing_rank:
        return incoming_source
    if incoming_rank > existing_rank:
        return existing_source

    existing_conf = float(existing_source.get("document_confidence_score") or 0.0)
    incoming_conf = float(incoming_source.get("document_confidence_score") or 0.0)
    if incoming_conf > existing_conf:
        return incoming_source
    return existing_source


def _merge_string_lists(existing: list[str], incoming: list[str]) -> list[str]:
    """Merge two ordered string lists while removing duplicates."""
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*existing, *incoming]:
        if not value:
            continue
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(value)
    return merged


def _merge_source_formats_seen(existing: list[str], incoming: list[str]) -> list[str]:
    """Keep unique formats ordered by reliability priority."""
    values = _merge_string_lists(existing, incoming)
    return sorted(values, key=lambda item: (_source_priority(item), item))


def _merge_sources_used(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one source entry per document_id."""
    merged: dict[str, dict[str, Any]] = {}
    for source in [*existing, *incoming]:
        document_id = source.get("document_id")
        if not document_id:
            continue
        current = merged.get(document_id)
        if current is None or _pick_better_source(current, source) == source:
            merged[document_id] = source
    return sorted(merged.values(), key=lambda item: (_source_priority(item.get("source_format")), item.get("document_id", "")))


def _merge_bio(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    existing_sources: dict[str, Any],
    incoming_sources: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge bio fields field-by-field without degrading stronger values."""
    merged = dict(existing)
    chosen_sources: dict[str, Any] = {}
    for field_name in ("full_name", "email", "phone", "location"):
        field_key = f"bio.{field_name}"
        existing_value = existing.get(field_name)
        incoming_value = incoming.get(field_name)
        existing_source = existing_sources.get(field_key)
        incoming_source = incoming_sources.get(field_key)
        value, source = _choose_field_value(existing_value, incoming_value, existing_source, incoming_source)
        merged[field_name] = value
        if source:
            chosen_sources[field_key] = source
    return merged, chosen_sources


def _merge_expertise(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    existing_sources: dict[str, Any],
    incoming_sources: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge expertise while keeping the strongest summary and a controlled skill union."""
    merged = {
        "summary": existing.get("summary") or "",
        "hard_skills": list(existing.get("hard_skills") or []),
        "soft_skills": list(existing.get("soft_skills") or []),
    }
    chosen_sources: dict[str, Any] = {}

    summary, summary_source = _choose_field_value(
        existing.get("summary"),
        incoming.get("summary"),
        existing_sources.get("expertise.summary"),
        incoming_sources.get("expertise.summary"),
    )
    merged["summary"] = summary or ""
    if summary_source:
        chosen_sources["expertise.summary"] = summary_source

    hard_skills = _merge_string_lists(list(existing.get("hard_skills") or []), list(incoming.get("hard_skills") or []))
    merged["hard_skills"] = hard_skills[:15]
    hard_source = _pick_better_source(
        existing_sources.get("expertise.hard_skills"),
        incoming_sources.get("expertise.hard_skills"),
    )
    if hard_source:
        chosen_sources["expertise.hard_skills"] = hard_source

    soft_skills = _merge_string_lists(list(existing.get("soft_skills") or []), list(incoming.get("soft_skills") or []))
    merged["soft_skills"] = soft_skills
    soft_source = _pick_better_source(
        existing_sources.get("expertise.soft_skills"),
        incoming_sources.get("expertise.soft_skills"),
    )
    if soft_source:
        chosen_sources["expertise.soft_skills"] = soft_source

    return merged, chosen_sources


def _choose_field_value(
    existing_value: Any,
    incoming_value: Any,
    existing_source: dict[str, Any] | None,
    incoming_source: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any] | None]:
    """Choose the strongest non-empty field value between two candidates."""
    existing_present = _has_meaningful_value(existing_value)
    incoming_present = _has_meaningful_value(incoming_value)

    if not existing_present and incoming_present:
        return incoming_value, incoming_source
    if existing_present and not incoming_present:
        return existing_value, existing_source
    if not existing_present and not incoming_present:
        return existing_value, existing_source

    best_source = _pick_better_source(existing_source, incoming_source)
    if best_source == incoming_source:
        return incoming_value, incoming_source
    return existing_value, existing_source


def _has_meaningful_value(value: Any) -> bool:
    """Detect whether a value is present enough to keep in consolidation."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_meaningful_value(item) for item in value)
    if isinstance(value, dict):
        return any(_has_meaningful_value(item) for item in value.values())
    return True


def _clean_education_documents(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop invalid education items and de-duplicate the rest."""
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        degree = str(item.get("degree") or "").strip()
        if not degree:
            continue
        school = str(item.get("school") or "").strip() or None
        year = str(item.get("year") or "").strip() or None
        signature = "|".join(
            [
                _normalize_for_match(degree),
                _normalize_for_match(school or ""),
                _normalize_for_match(year or ""),
            ]
        )
        if signature in seen:
            continue
        seen.add(signature)
        cleaned.append({"degree": degree, "school": school, "year": year})
    return cleaned


def _merge_education_documents(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    existing_source: dict[str, Any] | None,
    incoming_source: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Merge education items without keeping invalid entries."""
    merged: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {}
    for item in _clean_education_documents(existing):
        merged[_education_signature(item)] = (item, existing_source)
    for item in _clean_education_documents(incoming):
        signature = _education_signature(item)
        current = merged.get(signature)
        if current is None or _pick_better_source(current[1], incoming_source) == incoming_source:
            merged[signature] = (item, incoming_source)
    return [value[0] for value in merged.values()]


def _education_signature(item: dict[str, Any]) -> str:
    """Build a stable dedupe signature for one education item."""
    return "|".join(
        [
            _normalize_for_match(str(item.get("degree") or "")),
            _normalize_for_match(str(item.get("school") or "")),
            _normalize_for_match(str(item.get("year") or "")),
        ]
    )


def _clean_experience_documents(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize and de-duplicate experience items."""
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        job_title = str(item.get("job_title") or "").strip()
        if not job_title:
            continue
        company = str(item.get("company") or "").strip() or None
        start_date = str(item.get("start_date") or "").strip() or None
        end_date = str(item.get("end_date") or "").strip() or None
        city = str(item.get("city") or "").strip() or None
        responsibilities = _merge_string_lists([], list(item.get("responsibilities") or []))
        normalized = {
            "job_title": job_title,
            "company": company,
            "start_date": start_date,
            "end_date": end_date,
            "city": city,
            "responsibilities": responsibilities,
        }
        signature = _experience_signature(normalized)
        if signature in seen:
            continue
        seen.add(signature)
        cleaned.append(normalized)
    return cleaned


def _merge_experience_documents(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    existing_source: dict[str, Any] | None,
    incoming_source: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Merge experience items and prefer stronger sources on duplicate entries."""
    merged: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {}
    for item in _clean_experience_documents(existing):
        merged[_experience_signature(item)] = (item, existing_source)
    for item in _clean_experience_documents(incoming):
        signature = _experience_signature(item)
        current = merged.get(signature)
        if current is None:
            merged[signature] = (item, incoming_source)
            continue
        better_source = _pick_better_source(current[1], incoming_source)
        if better_source == incoming_source:
            merged[signature] = (item, incoming_source)
        else:
            merged[signature] = (_merge_experience_item(current[0], item), current[1])
    return [value[0] for value in merged.values()]


def _experience_signature(item: dict[str, Any]) -> str:
    """Build a stable dedupe signature for one experience item."""
    return "|".join(
        [
            _normalize_for_match(str(item.get("job_title") or "")),
            _normalize_for_match(str(item.get("company") or "")),
            _normalize_for_match(str(item.get("start_date") or "")),
            _normalize_for_match(str(item.get("end_date") or "")),
        ]
    )


def _merge_experience_item(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Keep the most complete representation of a duplicated experience item."""
    merged = dict(existing)
    for field_name in ("company", "start_date", "end_date", "city"):
        if not merged.get(field_name) and incoming.get(field_name):
            merged[field_name] = incoming[field_name]
    merged["responsibilities"] = _merge_string_lists(
        list(existing.get("responsibilities") or []),
        list(incoming.get("responsibilities") or []),
    )
    return merged


def upsert_profile(profile: CandidateProfile) -> None:
    """Ensure MongoDB indexes exist before persisting consolidated candidates."""
    try:
        from pymongo import ASCENDING, DESCENDING, MongoClient
    except ImportError as exc:
        raise RuntimeError("pymongo is required to persist CandidateProfile into MongoDB.") from exc

    client = MongoClient(MONGODB_URI)
    try:
        database = client[MONGODB_DATABASE]
        try:
            database[MONGODB_COLLECTION].drop_index("uniq_document_id")
        except Exception:
            pass
        database[MONGODB_COLLECTION].create_index([("candidate_key", ASCENDING)], unique=True, name="uniq_candidate_key")
        database[MONGODB_COLLECTION].create_index([("source_formats_seen", ASCENDING)], name="idx_source_formats_seen")
        database[MONGODB_COLLECTION].create_index(
            [("reliability_score", DESCENDING)],
            name="idx_reliability_score",
        )
        database[MONGODB_RUNS_COLLECTION].create_index([("run_id", ASCENDING)], unique=True, name="uniq_run_id")
    finally:
        client.close()


def persist_profile(
    profile: CandidateProfile,
    context: DocumentContext,
    *,
    run_id: str,
    mode: str,
    accepted_path: Path,
) -> None:
    """Persist one validated profile by consolidating it into a single candidate document."""
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise RuntimeError("pymongo is required to persist CandidateProfile into MongoDB.") from exc

    upsert_profile(profile)
    incoming = build_persistence_document(profile, context, run_id=run_id, mode=mode, accepted_path=accepted_path)

    client = MongoClient(MONGODB_URI)
    try:
        collection = client[MONGODB_DATABASE][MONGODB_COLLECTION]
        existing = collection.find_one({"candidate_key": incoming["candidate_key"]}, {"_id": 0})
        document = merge_candidate_documents(existing, incoming)
        collection.update_one({"candidate_key": document["candidate_key"]}, {"$set": document}, upsert=True)
    finally:
        client.close()


def write_run_record(report: dict[str, Any]) -> None:
    """Persist one run summary for traceability."""
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise RuntimeError("pymongo is required to persist profile builder run metadata.") from exc

    client = MongoClient(MONGODB_URI)
    try:
        collection = client[MONGODB_DATABASE][MONGODB_RUNS_COLLECTION]
        collection.update_one({"run_id": report["run_id"]}, {"$set": report}, upsert=True)
    finally:
        client.close()


def run_profile_builder(
    accepted_path: Path = ACCEPTED_PATH,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    preview_root: Path = PREVIEW_ROOT,
) -> dict[str, Any]:
    """Main process: read accepted artifacts, build profiles, and upsert them one by one."""
    configure_logging()
    preview_root.mkdir(parents=True, exist_ok=True)
    entries, ignored_rows = load_accepted_entries(accepted_path)
    if limit is not None:
        entries = entries[:limit]

    total = len(entries)
    success = 0
    failed = 0
    skipped = 0
    report_items: list[dict[str, Any]] = list(ignored_rows)
    mode = "dry-run" if dry_run else "live"
    run_id = f"profile_builder_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    LOGGER.info(
        "Chargement de %s documents accepted depuis %s | ignores=%s | mode=%s",
        total,
        accepted_path,
        len(ignored_rows),
        mode,
    )

    for index, entry in enumerate(entries, start=1):
        preview_path = _preview_path_for_entry(entry, preview_root)
        try:
            context = build_document_context(entry)
            profile = build_candidate_profile(context)
            preview_payload = {
                "source_path": entry.source_path,
                "artifact_path": entry.artifact_path,
                "source_format": entry.source_format,
                "status": "success",
                "mode": mode,
                "error": None,
                "run_id": run_id,
                "profile": profile.model_dump(mode="json"),
            }
            _write_preview(preview_path, preview_payload)
            if not dry_run:
                persist_profile(profile, context, run_id=run_id, mode=mode, accepted_path=accepted_path)
            success += 1
            report_items.append(
                {
                    "source_path": entry.source_path,
                    "artifact_path": entry.artifact_path,
                    "status": "success",
                    "mode": mode,
                    "error": None,
                    "run_id": run_id,
                    "preview_path": str(preview_path),
                }
            )
            LOGGER.info("Traitement %s/%s termine | source_id=%s | mode=%s", index, total, profile.source_id, mode)
        except DocumentSkipError as exc:
            skipped += 1
            message = str(exc)
            _write_preview(
                preview_path,
                {
                    "source_path": entry.source_path,
                    "artifact_path": entry.artifact_path,
                    "source_format": entry.source_format,
                    "status": "skipped",
                    "mode": mode,
                    "error": message,
                    "run_id": run_id,
                    "profile": None,
                },
            )
            report_items.append(
                {
                    "source_path": entry.source_path,
                    "artifact_path": entry.artifact_path,
                    "status": "skipped",
                    "mode": mode,
                    "error": message,
                    "run_id": run_id,
                    "preview_path": str(preview_path),
                }
            )
            LOGGER.warning("Traitement %s/%s ignore | artifact=%s | raison=%s", index, total, entry.artifact_path, message)
        except BusinessValidationError as exc:
            failed += 1
            message = str(exc)
            _write_preview(
                preview_path,
                {
                    "source_path": entry.source_path,
                    "artifact_path": entry.artifact_path,
                    "source_format": entry.source_format,
                    "status": "rejected",
                    "mode": mode,
                    "error": message,
                    "run_id": run_id,
                    "profile": None,
                },
            )
            report_items.append(
                {
                    "source_path": entry.source_path,
                    "artifact_path": entry.artifact_path,
                    "status": "rejected",
                    "mode": mode,
                    "error": message,
                    "run_id": run_id,
                    "preview_path": str(preview_path),
                }
            )
            LOGGER.warning("Traitement %s/%s rejete | artifact=%s | raison=%s", index, total, entry.artifact_path, message)
        except Exception as exc:
            failed += 1
            message = str(exc)
            _write_preview(
                preview_path,
                {
                    "source_path": entry.source_path,
                    "artifact_path": entry.artifact_path,
                    "source_format": entry.source_format,
                    "status": "failed",
                    "mode": mode,
                    "error": message,
                    "run_id": run_id,
                    "profile": None,
                },
            )
            report_items.append(
                {
                    "source_path": entry.source_path,
                    "artifact_path": entry.artifact_path,
                    "status": "failed",
                    "mode": mode,
                    "error": message,
                    "run_id": run_id,
                    "preview_path": str(preview_path),
                }
            )
            LOGGER.exception(
                "Traitement %s/%s echoue | artifact=%s | erreur=%s",
                index,
                total,
                entry.artifact_path,
                exc,
            )

    summary = {
        "run_id": run_id,
        "accepted_file_path": str(accepted_path),
        "mode": mode,
        "total_processed": total,
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "ignored": len(ignored_rows),
        "items": report_items,
    }
    RUN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_REPORT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not dry_run:
        write_run_record(summary)
    LOGGER.info("Profile Builder termine | %s", {k: v for k, v in summary.items() if k != "items"})
    return summary


def candidate_profile_payload_schema() -> dict[str, Any]:
    """JSON schema sent to the LLM. Deterministic fields stay local and are never requested."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "bio": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "full_name": {"type": ["string", "null"]},
                    "email": {"type": ["string", "null"]},
                    "phone": {"type": ["string", "null"]},
                    "location": {"type": ["string", "null"]},
                },
                "required": ["full_name", "email", "phone", "location"],
            },
            "expertise": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string"},
                    "hard_skills": {"type": "array", "items": {"type": "string"}},
                    "soft_skills": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "hard_skills", "soft_skills"],
            },
            "experiences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "job_title": {"type": "string"},
                        "company": {"type": ["string", "null"]},
                        "start_date": {"type": ["string", "null"]},
                        "end_date": {"type": ["string", "null"]},
                        "city": {"type": ["string", "null"]},
                        "responsibilities": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "job_title",
                        "company",
                        "start_date",
                        "end_date",
                        "city",
                        "responsibilities",
                    ],
                },
            },
            "education": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "degree": {"type": "string"},
                        "school": {"type": ["string", "null"]},
                        "year": {"type": ["string", "null"]},
                    },
                    "required": ["degree", "school", "year"],
                },
            },
        },
        "required": [
            "bio",
            "expertise",
            "experiences",
            "education",
        ],
    }


def system_prompt() -> str:
    """Anti-hallucination system instructions used for Groq."""
    return (
        "You are a strict recruitment extraction engine. "
        "Return ONLY a valid JSON matching exactly the schema. No extra fields. "
        "Output one JSON object and nothing else. "
        "Never invent missing information. "
        "If a value is absent, use null or an empty list. "
        "Use only facts grounded in the provided Markdown and metadata. "
        "Do not infer companies, dates, emails, phones, locations, or names that are not explicitly supported. "
        "Use these exact top-level keys only: bio, expertise, experiences, education. "
        "Inside bio use only: full_name, email, phone, location. "
        "Inside expertise use only: summary, hard_skills, soft_skills. "
        "Inside experiences items use only: job_title, company, start_date, end_date, city, responsibilities. "
        "Inside education items use only: degree, school, year. "
        "Do not use alternative keys such as name, contact, contact_info, location at top level, summary at top level, "
        "expertise_summary, experience, professional_experience, projects, certifications, certificates, languages, candidate_id, institution, university, dates, date, role, title, or position. "
        "Write the expertise summary as a short factual synthesis strictly based on the source. "
        "Keep the summary short, descriptive, non-marketing, and specific about role, domain, or stack when the source supports it. "
        "Avoid generic summaries that could fit any candidate. "
        "For hard_skills, extract only concrete, atomic, source-visible skills such as tools, technologies, certifications, standards, protocols, platforms, or named methods. "
        "Limit hard_skills to the 15 most relevant and most specific skills supported by the source. "
        "Do not output broad category headings, section labels, umbrella domains, or generic capability buckets such as 'Tools', 'Methods', 'Software', 'Management', 'Programming Languages', 'Machine Learning', or 'Data Science' unless the source explicitly uses them as the only skill wording and they are materially informative. "
        "Prefer specific items over broad families: for example return 'Python' or 'TensorFlow' rather than a generic category. "
        "Do not promote OCR noise, repeated headings, or prose fragments into skills. "
        "Do not use promotional language, claims of excellence, or unsupported evaluation. "
        "Do not add commentary or explanations outside the JSON object."
    )


def user_prompt(context: DocumentContext) -> str:
    """Build the per-document extraction prompt from the accepted Module 1 artifact."""
    return (
        "Transform the following accepted Module 1 Markdown into the requested CandidateProfile JSON.\n\n"
        "Required JSON schema:\n"
        f"{json.dumps(candidate_profile_payload_schema(), ensure_ascii=True)}\n\n"
        f"Source path: {context.accepted_entry.source_path}\n"
        f"Artifact path: {context.accepted_entry.artifact_path}\n"
        f"Source format: {context.accepted_entry.source_format}\n"
        f"Parser used: {context.accepted_entry.parser_used}\n"
        f"Document confidence score: {context.accepted_entry.document_confidence_score}\n"
        f"Quality flags: {context.accepted_entry.quality_flags}\n\n"
        "Markdown:\n"
        "-----\n"
        f"{context.markdown}\n"
        "-----"
    )


def _http_post_json(url: str, payload: dict[str, Any], *, headers: dict[str, str]) -> dict[str, Any]:
    """POST JSON with requests for provider compatibility and clearer errors."""
    last_error: RuntimeError | None = None
    for attempt in range(1, 4):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
        except requests.RequestException as exc:
            raise RuntimeError(f"Unable to reach {url}: {exc}") from exc

        if response.ok:
            return response.json()

        body = response.text
        if response.status_code == 429 and attempt < 3:
            wait_seconds = _extract_retry_after_seconds(body)
            LOGGER.warning(
                "Rate limit provider detecte | tentative=%s/3 | attente=%.2fs",
                attempt,
                wait_seconds,
            )
            time.sleep(wait_seconds)
            continue

        last_error = RuntimeError(f"HTTP {response.status_code} calling {url}: {body}")
        break

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"HTTP call failed unexpectedly for {url}")


def _extract_chat_completions_text(response: dict[str, Any]) -> str | None:
    """Extract assistant JSON content from an OpenAI-compatible chat completions payload."""
    choices = response.get("choices") or []
    for choice in choices:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def _chat_completions_url(base_url: str) -> str:
    """Normalize a provider base URL to the chat completions endpoint."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/responses"):
        return normalized[: -len("/responses")] + "/chat/completions"
    return normalized + "/chat/completions"


def _extract_retry_after_seconds(body: str) -> float:
    """Read a Groq-style retry hint and return a small safe backoff."""
    match = re.search(r"Please try again in ([0-9]+(?:\.[0-9]+)?)s", body)
    if match:
        return min(float(match.group(1)) + 0.5, 15.0)
    return 5.0


def _preview_path_for_entry(entry: AcceptedArtifactRef, preview_root: Path) -> Path:
    """Build a collision-safe preview path for one document."""
    artifact_path = Path(entry.artifact_path)
    preview_dir = preview_root / entry.source_format
    preview_dir.mkdir(parents=True, exist_ok=True)
    return preview_dir / f"{artifact_path.stem}.json"


def _write_preview(path: Path, payload: dict[str, Any]) -> None:
    """Persist one preview payload for manual review."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Expose a minimal safe CLI for dry runs and limited test batches."""
    parser = argparse.ArgumentParser(description="Build structured candidate profiles from Module 1 accepted artifacts.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dry-run", action="store_true", help="Generate local previews without writing to MongoDB.")
    mode_group.add_argument("--live", action="store_true", help="Persist validated profiles into MongoDB.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of accepted entries to process.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_profile_builder(dry_run=not args.live, limit=args.limit)
