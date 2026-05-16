from __future__ import annotations

from pydantic import BaseModel, Field


class MatchCandidatesInput(BaseModel):
    job_description: str = Field(..., min_length=1, description="Job description or hiring need.")
    top_k: int = Field(default=10, ge=1, le=50, description="Maximum number of candidates to return.")


class CandidateProfileInput(BaseModel):
    candidate_id: str = Field(..., min_length=1)


class DecisionCardInput(BaseModel):
    candidate_id: str = Field(..., min_length=1)


class TransferabilityInput(BaseModel):
    candidate_id: str = Field(..., min_length=1)


class Neo4jTransferabilityInput(BaseModel):
    candidate_id: str = Field(..., min_length=1)
    target_role: str = Field(default="Backend Developer", min_length=1)


class CompareCandidatesInput(BaseModel):
    candidate_ids: list[str] = Field(..., min_length=1)
