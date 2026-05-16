from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class MatchRequest(BaseModel):
    job_description: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


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
    items: list[CandidateListItem]


class MatchCandidate(BaseModel):
    candidate_id: str | None = None
    profile_id: str | None = None
    rank: int | None = None
    baseline_rank_v3: int | None = None
    baseline_score_v3: float | None = None
    rf_rank: int | None = None
    rf_score: float | None = None
    xgboost_rank: int | None = None
    xgboost_score: float | None = None
    recommendation_status: str | None = None
    transferability: dict[str, Any] | None = None


class MatchResponse(BaseModel):
    job_description: str
    top_k: int
    matching_mode: str
    methodological_note: str
    items: list[MatchCandidate]


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
    answer: str
    candidates: list[dict[str, Any]]
    decision_cards: list[dict[str, Any]]
    transferability: dict[str, Any]
    sources: list[str]
    warnings: list[str]
