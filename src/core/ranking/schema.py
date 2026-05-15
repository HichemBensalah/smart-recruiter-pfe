from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RankingFeatures(BaseModel):
    """Numerical candidate-job features exported before supervised ranking."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    vector_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    final_score_v3: float = Field(default=0.0, ge=0.0, le=1.0)
    must_have_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    required_skills_overlap: float = Field(default=1.0, ge=0.0, le=1.0)
    nice_to_have_overlap: float = Field(default=0.0, ge=0.0, le=1.0)
    experience_match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    seniority_alignment: float = Field(default=0.0, ge=0.0, le=1.0)
    profile_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reliability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_risk_encoded: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_required_count: float = Field(default=0.0, ge=0.0)
    matched_required_count: float = Field(default=0.0, ge=0.0)

    @field_validator("*", mode="before")
    @classmethod
    def coerce_numeric(cls, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return 0.0
            return float(stripped)
        return float(value)


class RankingFeatureRow(BaseModel):
    """One ML-ready feature row for a candidate/job pair.

    Labels and splits are intentionally nullable until a separate annotation
    protocol creates reliable ground truth.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    job_id: str = Field(..., min_length=1)
    candidate_id: str = Field(..., min_length=1)
    profile_id: str = Field(..., min_length=1)
    rank: int = Field(..., ge=1)
    source: str = Field(default="matching_v3_normalized", min_length=1)
    features: RankingFeatures
    label: float | int | str | None = None
    split: str | None = None
