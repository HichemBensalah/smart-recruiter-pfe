from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .document_artifact import (
    DocumentArtifact,
    DocumentStatus,
    HandoffDecision,
    HandoffLane,
    RepairStatus,
    SourceFormat,
)


class HandoffQueueEntry(BaseModel):
    """Serializable queue row used to drive Module 1 to Module 2 trust-boundary operations."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    source_path: str = Field(
        ...,
        min_length=1,
        description="Original raw document kept in data/raw_cv and never discarded by the handoff policy.",
    )
    artifact_path: str = Field(
        ...,
        min_length=1,
        description="Path to the raw Module 1 artifact that must be preserved for replay and audit.",
    )
    source_format: SourceFormat = Field(
        ...,
        description="Normalized source family used to apply source-specific handoff hardening rules.",
    )
    document_status: DocumentStatus = Field(
        ...,
        description="Quality status emitted by Module 1 before any Module 2 consumption.",
    )
    handoff_lane: HandoffLane = Field(
        ...,
        description="Final trust lane selected for the artifact: accepted, repair_required, or quarantined.",
    )
    eligible_for_module2: bool = Field(
        ...,
        description="Hard gate for Module 2. False means the artifact must not be consumed downstream.",
    )
    parser_used: str = Field(
        ...,
        min_length=1,
        description="Parser retained for the final artifact, kept for debugging and replay decisions.",
    )
    document_confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final confidence score emitted by Module 1 and used to justify the handoff lane.",
    )
    quality_flags: list[str] = Field(
        default_factory=list,
        description="Warnings inherited from Module 1 quality assessment that justify repair or quarantine.",
    )
    repair_status: RepairStatus = Field(
        ...,
        description="Lifecycle state used by the repair loop and quarantine workflow.",
    )
    repair_reason: str | None = Field(
        default=None,
        description="Reason why the artifact must be repaired before it may be reconsidered.",
    )
    quarantine_reason: str | None = Field(
        default=None,
        description="Reason why the artifact is blocked from downstream usage until further action.",
    )
    next_action: str = Field(
        ...,
        min_length=1,
        description="Operational next step for the artifact after the handoff decision.",
    )
    routing_decision: str | None = Field(
        default=None,
        description="Serialized routing decision preserved for traceability and later forensic analysis.",
    )


class HandoffRegistry(BaseModel):
    """Compact registry written by Module 1 to expose safe handoff queues for Module 2."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    accepted: list[HandoffQueueEntry] = Field(
        default_factory=list,
        description="Artifacts that may enter Module 2 immediately.",
    )
    repair_required: list[HandoffQueueEntry] = Field(
        default_factory=list,
        description="Artifacts blocked from Module 2 until repair and re-evaluation succeed.",
    )
    quarantined: list[HandoffQueueEntry] = Field(
        default_factory=list,
        description="Artifacts fully blocked from Module 2 until stronger recovery or manual review occurs.",
    )


def decide_handoff(
    *,
    document_status: DocumentStatus,
    source_format: SourceFormat,
    quality_flags: list[str],
    previous_decision: HandoffDecision | None = None,
) -> HandoffDecision:
    """
    Convert Module 1 quality into an explicit trust-boundary decision for Module 2.

    Rules:
    - validated -> accepted
    - partial -> repair_required
    - uncertain -> quarantined
    - failed -> quarantined
    """

    rules: list[str] = []
    status = document_status
    repaired_and_revalidated = (
        previous_decision is not None
        and previous_decision.lane == HandoffLane.REPAIR_REQUIRED
        and status == DocumentStatus.VALIDATED
    )

    if status == DocumentStatus.VALIDATED:
        rules.append("validated -> accepted")
        if repaired_and_revalidated:
            rules.append("partial repaired and revalidated -> accepted")
            return HandoffDecision(
                lane=HandoffLane.ACCEPTED,
                eligible_for_module2=True,
                decision_reason="Document repare puis revalide; consommation Module 2 autorisee.",
                decision_rules=rules,
                quality_flags=quality_flags,
                repair_status=RepairStatus.REVALIDATED,
                repair_reason=None,
                quarantine_reason=None,
                next_action="consume_in_module2",
            )

        return HandoffDecision(
            lane=HandoffLane.ACCEPTED,
            eligible_for_module2=True,
            decision_reason="Document valide; consommation Module 2 autorisee.",
            decision_rules=rules,
            quality_flags=quality_flags,
            repair_status=RepairStatus.NOT_APPLICABLE,
            repair_reason=None,
            quarantine_reason=None,
            next_action="consume_in_module2",
        )

    if status == DocumentStatus.PARTIAL:
        rules.append("partial -> repair_required")
        if source_format in {SourceFormat.IMAGES, SourceFormat.SCANS}:
            rules.append("partial from images/scans -> always repair_required")
            return HandoffDecision(
                lane=HandoffLane.REPAIR_REQUIRED,
                eligible_for_module2=False,
                decision_reason="Document OCR partiel; reparation obligatoire avant tout handoff vers Module 2.",
                decision_rules=rules,
                quality_flags=quality_flags,
                repair_status=RepairStatus.PENDING_REPAIR,
                repair_reason="ocr_partial_requires_repair",
                quarantine_reason=None,
                next_action="send_to_repair_queue",
            )

        rules.append("partial from pdf/docx -> repair_required until revalidated")
        return HandoffDecision(
            lane=HandoffLane.REPAIR_REQUIRED,
            eligible_for_module2=False,
            decision_reason="Document partiel; reparation et re-evaluation obligatoires avant Module 2.",
            decision_rules=rules,
            quality_flags=quality_flags,
            repair_status=RepairStatus.PENDING_REPAIR,
            repair_reason="partial_document_requires_repair",
            quarantine_reason=None,
            next_action="send_to_repair_queue",
        )

    if status == DocumentStatus.UNCERTAIN:
        rules.extend(
            [
                "uncertain -> quarantined",
                "uncertain from any source -> always quarantined",
            ]
        )
        return HandoffDecision(
            lane=HandoffLane.QUARANTINED,
            eligible_for_module2=False,
            decision_reason="Document incertain; quarantaine obligatoire pour proteger la factualite du systeme.",
            decision_rules=rules,
            quality_flags=quality_flags,
            repair_status=RepairStatus.QUARANTINED,
            repair_reason=None,
            quarantine_reason="uncertain_document_blocked",
            next_action="send_to_quarantine_queue",
        )

    rules.extend(
        [
            "failed -> quarantined",
            "failed artifacts are never eligible for Module 2",
        ]
    )
    return HandoffDecision(
        lane=HandoffLane.QUARANTINED,
        eligible_for_module2=False,
        decision_reason="Document en echec de parsing; quarantaine obligatoire.",
        decision_rules=rules,
        quality_flags=quality_flags,
        repair_status=RepairStatus.QUARANTINED,
        repair_reason=None,
        quarantine_reason="failed_document_blocked",
        next_action="send_to_quarantine_queue",
    )


def decide_handoff_for_artifact(
    artifact: DocumentArtifact,
    *,
    previous_decision: HandoffDecision | None = None,
) -> HandoffDecision:
    """Convenience wrapper used when a full DocumentArtifact instance already exists."""

    return decide_handoff(
        document_status=artifact.document_status,
        source_format=artifact.source_format,
        quality_flags=list(artifact.document_confidence.warnings),
        previous_decision=previous_decision,
    )


def build_queue_entry(artifact: DocumentArtifact, artifact_path: str) -> HandoffQueueEntry:
    """Project the final artifact into a lightweight queue row for downstream orchestration."""

    return HandoffQueueEntry(
        source_path=artifact.source_path,
        artifact_path=artifact_path,
        source_format=artifact.source_format,
        document_status=artifact.document_status,
        handoff_lane=artifact.handoff_decision.lane,
        eligible_for_module2=artifact.handoff_decision.eligible_for_module2,
        parser_used=artifact.parser_used.value,
        document_confidence_score=artifact.document_confidence.score,
        quality_flags=list(artifact.handoff_decision.quality_flags),
        repair_status=artifact.handoff_decision.repair_status,
        repair_reason=artifact.handoff_decision.repair_reason,
        quarantine_reason=artifact.handoff_decision.quarantine_reason,
        next_action=artifact.handoff_decision.next_action,
        routing_decision=_routing_decision_signal(artifact),
    )


def build_handoff_registry(entries: list[HandoffQueueEntry]) -> HandoffRegistry:
    """Group queue entries by lane so Module 2 can consume only the accepted lane."""

    registry = HandoffRegistry()
    for entry in entries:
        if entry.handoff_lane == HandoffLane.ACCEPTED:
            registry.accepted.append(entry)
        elif entry.handoff_lane == HandoffLane.REPAIR_REQUIRED:
            registry.repair_required.append(entry)
        else:
            registry.quarantined.append(entry)
    return registry


def _routing_decision_signal(artifact: DocumentArtifact) -> str | None:
    signal = artifact.document_confidence.signals.get("routing_decision")
    if isinstance(signal, str):
        return signal
    return None
