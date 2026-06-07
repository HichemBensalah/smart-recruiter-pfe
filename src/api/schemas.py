from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    dependencies: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class MatchRequest(BaseModel):
    job_description: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    job_id: str | None = None

    @field_validator("job_description")
    @classmethod
    def job_description_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("job_description must not be empty")
        return value.strip()


class CandidateListItem(BaseModel):
    candidate_id: str | None = None
    profile_id: str | None = None
    baseline_rank_v3: int | None = None
    baseline_score_v3: float | None = None
    rf_rank: int | None = None
    rf_score: float | None = None
    xgboost_rank: int | None = None
    xgboost_score: float | None = None
    recommendation_status: str | None = None
    transferability_score: float | None = None


class PaginatedCandidates(BaseModel):
    total: int
    limit: int
    offset: int
    source: str | None = None
    data_backend: str | None = None
    data_source: str | None = None
    fallback_used: bool = False
    warnings: list[str] = Field(default_factory=list)
    items: list[CandidateListItem]


class CandidateDetailResponse(BaseModel):
    candidate: dict[str, Any]
    profile: dict[str, Any] | None = None
    source: str | None = None
    data_backend: str | None = None
    data_source: str | None = None
    fallback_used: bool = False
    warnings: list[str] = Field(default_factory=list)


class MatchCandidate(BaseModel):
    candidate_id: str | None = None
    profile_id: str | None = None
    rank: int | None = None
    baseline_rank_v3: int | None = None
    baseline_score_v3: float | None = None
    faiss_rank: int | None = None
    faiss_score: float | None = None
    rf_rank: int | None = None
    rf_score: float | None = None
    xgboost_rank: int | None = None
    xgboost_score: float | None = None
    recommendation_status: str | None = None
    matched_skills: list[str] | None = None
    missing_required_skills: list[str] | None = None
    explanation: str | None = None
    transferability: dict[str, Any] | None = None
    cv_available: bool = False
    has_original_cv: bool = False
    cv_download_url: str | None = None
    cv_url: str | None = None
    cv_path: str | None = None
    cv_filename: str | None = None
    cv_mime_type: str | None = None
    cv_source: str | None = None
    cv_confidence: str | None = None
    score_breakdown: list[dict[str, Any]] | None = None
    base_score_before_penalty: float | None = None
    must_have_coverage: float | None = None
    must_have_penalty_multiplier: float | None = None
    must_have_penalty_applied: bool | None = None
    quality_penalty_multiplier: float | None = None


class MatchResponse(BaseModel):
    job_description: str
    job_id: str | None = None
    resolved_job_id: str | None = None
    artifact_source: str | None = None
    data_backend: str | None = None
    data_source: str | None = None
    retrieval_source: str | None = None
    scoring_source: str | None = None
    matching_run_id: str | None = None
    top_k: int
    matching_mode: str
    fallback_used: bool = False
    warnings: list[str] = Field(default_factory=list)
    methodological_note: str
    items: list[MatchCandidate]


class DecisionCardsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str | None = None
    data_backend: str | None = None
    data_source: str | None = None
    fallback_used: bool = False
    warnings: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    candidate_count: int | None = None


class DecisionCardDetailResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_id: str | None = None
    profile_id: str | None = None
    source: str | None = None
    data_backend: str | None = None
    data_source: str | None = None
    fallback_used: bool = False
    warnings: list[str] = Field(default_factory=list)


class TransferabilityResponse(BaseModel):
    candidate_id: str | None = None
    profile_id: str | None = None
    baseline_rank_v3: int | None = None
    baseline_score_v3: float | None = None
    source: str
    fallback_used: bool = False
    warnings: list[str] = Field(default_factory=list)
    transferability: dict[str, Any]


class DemoRunResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    source: str | None = None
    fallback_used: bool = False
    warnings: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be empty")
        return value.strip()


class ChatResponse(BaseModel):
    session_id: str | None = None
    answer: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    decision_cards: list[dict[str, Any]] = Field(default_factory=list)
    transferability: dict[str, Any] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    job_intake_state: dict[str, Any] | None = None
    structured_job_profile: dict[str, Any] | None = None
    routed_job_id: str | None = None
    matching_metadata: dict[str, Any] = Field(default_factory=dict)
    matching_completed: bool = False
    pending_field_edit: str | None = None
    awaiting_field_replacement: bool = False
