from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ParserUsed(str, Enum):
    """Normalized parser identifiers emitted by Module 1."""

    DOCLING_STRUCTURED = "docling_structured"
    DOCLING_MARKDOWN = "docling_markdown"
    SECONDARY_PARSER = "secondary_parser"


class DocumentStatus(str, Enum):
    """Operational quality state of a parsed document."""

    VALIDATED = "validated"
    PARTIAL = "partial"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


class SourceFormat(str, Enum):
    """Physical input family mirrored from data/raw_cv and data/processed."""

    PDF = "pdf"
    DOCX = "docx"
    IMAGES = "images"
    SCANS = "scans"


class DocumentType(str, Enum):
    """Business document kind expected by downstream modules."""

    CV = "cv"
    JOB_DESCRIPTION = "job_description"


class SectionType(str, Enum):
    """Canonical section types exposed to the Profile Builder."""

    HEADER = "header"
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    SKILLS = "skills"
    PROJECTS = "projects"
    LANGUAGES = "languages"
    SOFT_SKILLS = "soft_skills"
    CERTIFICATIONS = "certifications"
    JOB_SUMMARY = "job_summary"
    RESPONSIBILITIES = "responsibilities"
    REQUIREMENTS = "requirements"
    OTHER = "other"


class BoundingBox(BaseModel):
    """Optional geometric anchor used to trace evidence back to page layout."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    left: float = Field(
        ...,
        description="Left x-coordinate of the evidence box in the original page coordinate system.",
    )
    top: float = Field(
        ...,
        description="Top y-coordinate of the evidence box in the original page coordinate system.",
    )
    right: float = Field(
        ...,
        description="Right x-coordinate of the evidence box in the original page coordinate system.",
    )
    bottom: float = Field(
        ...,
        description="Bottom y-coordinate of the evidence box in the original page coordinate system.",
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> "BoundingBox":
        """Reject impossible boxes to keep evidence spatially defensible."""
        if self.right < self.left:
            raise ValueError("Bounding box right must be greater than or equal to left.")
        if self.bottom >= self.top:
            return self
        raise ValueError("Bounding box bottom must be greater than or equal to top.")


class EvidenceSpan(BaseModel):
    """Proof anchor that links extracted content to a real document location."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    text: str = Field(
        ...,
        min_length=1,
        description="Exact text snippet recovered from the source document and used as evidence.",
    )
    page_no: int = Field(
        ...,
        ge=1,
        description="1-based page number where the evidence was observed.",
    )
    block_indices: list[int] = Field(
        default_factory=list,
        description="Indices of low-level extraction blocks supporting this evidence span.",
    )
    bbox: BoundingBox | None = Field(
        default=None,
        description="Optional geometric box locating the evidence on the source page.",
    )
    section_title: str | None = Field(
        default=None,
        description="Human-readable section title from which this evidence span originates.",
    )

    @model_validator(mode="after")
    def validate_block_indices(self) -> "EvidenceSpan":
        """Keep evidence references safe for audit and replay."""
        if any(index < 0 for index in self.block_indices):
            raise ValueError("Evidence block indices must be non-negative.")
        return self


class LogicalSectionItem(BaseModel):
    """Structured item nested inside a logical section, typically experience or education."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    title: str = Field(
        ...,
        min_length=1,
        description="Primary label of the item, such as a role, diploma, or project title.",
    )
    date: str | None = Field(
        default=None,
        description="Best-effort date or date range associated with the item.",
    )
    location: str | None = Field(
        default=None,
        description="Best-effort geographic location associated with the item.",
    )
    details: list[str] = Field(
        default_factory=list,
        description="Ordered detail lines that enrich the item without losing source phrasing.",
    )
    evidence_spans: list[EvidenceSpan] = Field(
        default_factory=list,
        description="Evidence anchors supporting this item for downstream audit and explanation.",
    )


class LogicalSection(BaseModel):
    """Canonical section exposed by Module 1 to the Profile Builder."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    title: str = Field(
        ...,
        min_length=1,
        description="Normalized or best-effort human-readable title of the logical section.",
    )
    section_type: SectionType = Field(
        ...,
        description="Canonical section category used by Module 2 to interpret the content.",
    )
    content: list[str] = Field(
        default_factory=list,
        description="Ordered free-text lines belonging directly to this section.",
    )
    items: list[LogicalSectionItem] = Field(
        default_factory=list,
        description="Structured items contained in the section, preserved in source order.",
    )
    evidence_spans: list[EvidenceSpan] = Field(
        default_factory=list,
        description="Section-level evidence anchors supporting the section and its content.",
    )

    @model_validator(mode="after")
    def validate_non_empty_section(self) -> "LogicalSection":
        """Prevent empty logical shells that would confuse downstream extraction."""
        if self.content or self.items or self.evidence_spans:
            return self
        raise ValueError("LogicalSection must contain content, items, or evidence spans.")


SignalValue = str | int | float | bool


class DocumentConfidence(BaseModel):
    """Document-level reliability payload used to qualify parsing quality."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized reliability score between 0.0 and 1.0 for the document artifact.",
    )
    status: DocumentStatus = Field(
        ...,
        description="Quality status derived from the document reliability assessment.",
    )
    signals: dict[str, SignalValue] = Field(
        default_factory=dict,
        description="Measured signals explaining the score, including parser metrics and serialized routing context.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Human-readable warning flags that expose uncertainty or degradation.",
    )


class HandoffLane(str, Enum):
    """Trusted handoff lanes between Module 1 and Module 2."""

    ACCEPTED = "accepted"
    REPAIR_REQUIRED = "repair_required"
    QUARANTINED = "quarantined"


class RepairStatus(str, Enum):
    """Repair lifecycle state for documents that cannot safely enter Module 2 yet."""

    NOT_APPLICABLE = "not_applicable"
    PENDING_REPAIR = "pending_repair"
    REVALIDATED = "revalidated"
    QUARANTINED = "quarantined"


class HandoffDecision(BaseModel):
    """Trust-boundary decision emitted after Module 1 quality assessment."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    lane: HandoffLane = Field(
        ...,
        description="Explicit handoff lane that controls whether Module 2 may consume the artifact.",
    )
    eligible_for_module2: bool = Field(
        ...,
        description="True only when the artifact is allowed to enter the Profile Builder automatically.",
    )
    decision_reason: str = Field(
        ...,
        min_length=1,
        description="Primary human-readable rationale for the handoff decision.",
    )
    decision_rules: list[str] = Field(
        default_factory=list,
        description="Ordered list of policy rules that fired to produce the handoff lane.",
    )
    quality_flags: list[str] = Field(
        default_factory=list,
        description="Quality warnings inherited from Module 1 and used to justify repair or quarantine.",
    )
    repair_status: RepairStatus = Field(
        ...,
        description="Repair lifecycle state attached to the artifact for later replay or recovery.",
    )
    repair_reason: str | None = Field(
        default=None,
        description="Reason why the artifact must be repaired before any downstream consumption.",
    )
    quarantine_reason: str | None = Field(
        default=None,
        description="Reason why the artifact is blocked from downstream usage until manual or stronger recovery.",
    )
    next_action: str = Field(
        ...,
        min_length=1,
        description="Operational next step for the pipeline owner, such as accept, repair, or quarantine review.",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "HandoffDecision":
        """Keep the handoff policy explicit and internally coherent."""
        if not self.decision_rules:
            raise ValueError("HandoffDecision must contain at least one decision rule.")

        if self.lane == HandoffLane.ACCEPTED:
            if not self.eligible_for_module2:
                raise ValueError("Accepted artifacts must be eligible for Module 2.")
            if self.repair_status not in {RepairStatus.NOT_APPLICABLE, RepairStatus.REVALIDATED}:
                raise ValueError("Accepted artifacts must be not_applicable or revalidated.")
            if self.quarantine_reason is not None:
                raise ValueError("Accepted artifacts cannot carry a quarantine reason.")

        if self.lane == HandoffLane.REPAIR_REQUIRED:
            if self.eligible_for_module2:
                raise ValueError("Repair-required artifacts cannot enter Module 2 directly.")
            if self.repair_status != RepairStatus.PENDING_REPAIR:
                raise ValueError("Repair-required artifacts must be marked pending_repair.")
            if not self.repair_reason:
                raise ValueError("Repair-required artifacts must carry a repair reason.")
            if self.quarantine_reason is not None:
                raise ValueError("Repair-required artifacts cannot carry a quarantine reason.")

        if self.lane == HandoffLane.QUARANTINED:
            if self.eligible_for_module2:
                raise ValueError("Quarantined artifacts cannot enter Module 2 directly.")
            if self.repair_status != RepairStatus.QUARANTINED:
                raise ValueError("Quarantined artifacts must be marked quarantined.")
            if not self.quarantine_reason:
                raise ValueError("Quarantined artifacts must carry a quarantine reason.")

        return self


class DocumentArtifact(BaseModel):
    """
    Stable Module 1 contract handed off to the Canonical Profile Builder.

    This artifact is the single source of truth for downstream profiling,
    quality audit, replay, and evidence-based reasoning.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    source_path: str = Field(
        ...,
        min_length=1,
        description="Original document path consumed from data/raw_cv.",
    )
    source_format: SourceFormat = Field(
        ...,
        description="Input family used by the gateway to route parsing and storage.",
    )
    document_type: DocumentType = Field(
        ...,
        description="Business document kind expected by downstream builders.",
    )
    document_status: DocumentStatus = Field(
        ...,
        description="Final document state exposed by Module 1 after parsing and quality checks.",
    )
    raw_text: str = Field(
        ...,
        description="Plain recovered text used as the lowest-friction fallback for downstream processing.",
    )
    markdown: str = Field(
        ...,
        description="Structured markdown rendering intended to preserve layout and section readability.",
    )
    logical_sections: list[LogicalSection] = Field(
        default_factory=list,
        description="Ordered canonical sections consumed by Module 2 without ambiguity.",
    )
    parser_used: ParserUsed = Field(
        ...,
        description="Final parser route that produced the accepted artifact.",
    )
    fallback_used: bool = Field(
        ...,
        description="True when a fallback route was retained for the final artifact, not merely attempted.",
    )
    document_confidence: DocumentConfidence = Field(
        ...,
        description="Reliability payload summarizing score, status, signals, and warnings.",
    )
    handoff_decision: HandoffDecision = Field(
        ...,
        description="Trust-boundary decision controlling whether the artifact may enter Module 2.",
    )
    evidence_spans: list[EvidenceSpan] = Field(
        default_factory=list,
        description="Top-level evidence anchors that make the artifact auditable and replayable.",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "DocumentArtifact":
        """Enforce a defensive contract before Module 2 consumes the artifact."""
        if self.document_status != self.document_confidence.status:
            raise ValueError(
                "DocumentArtifact.document_status must match DocumentArtifact.document_confidence.status."
            )

        if self.document_status == DocumentStatus.FAILED:
            return self

        has_recovered_content = any(
            [
                bool(self.raw_text.strip()),
                bool(self.markdown.strip()),
                bool(self.logical_sections),
            ]
        )
        if not has_recovered_content:
            raise ValueError(
                "A non-failed DocumentArtifact must contain raw_text, markdown, or logical_sections."
            )

        if not self.logical_sections:
            raise ValueError("A non-failed DocumentArtifact must expose at least one logical section.")

        if not self.evidence_spans:
            raise ValueError("A non-failed DocumentArtifact must expose at least one evidence span.")

        if self.document_status == DocumentStatus.VALIDATED and self.handoff_decision.lane != HandoffLane.ACCEPTED:
            raise ValueError("Validated artifacts must be accepted for Module 2 handoff.")

        if self.document_status == DocumentStatus.PARTIAL and self.handoff_decision.lane != HandoffLane.REPAIR_REQUIRED:
            raise ValueError("Partial artifacts must enter the repair_required handoff lane.")

        if self.document_status == DocumentStatus.UNCERTAIN and self.handoff_decision.lane != HandoffLane.QUARANTINED:
            raise ValueError("Uncertain artifacts must enter the quarantined handoff lane.")

        return self


EvidenceSpan.model_rebuild()
LogicalSectionItem.model_rebuild()
LogicalSection.model_rebuild()
DocumentConfidence.model_rebuild()
HandoffDecision.model_rebuild()
DocumentArtifact.model_rebuild()
