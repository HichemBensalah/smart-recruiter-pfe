from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class JobMetadata(BaseModel):
    """Traceability metadata for a structured job description."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    extraction_date: str = Field(..., description="UTC timestamp when the job profile was created.")
    parser_route: str = Field(..., description="Rule-based or model-assisted parsing route.")
    model_used: str | None = Field(default=None, description="Model name if an LLM parser was used.")
    confidence_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Best-effort confidence score for the extracted job profile.",
    )
    warnings: list[str] = Field(default_factory=list, description="Non-blocking extraction warnings.")


class CanonicalJobProfile(BaseModel):
    """Canonical job profile used as the query-side representation for matching."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)

    job_title: str = Field(..., min_length=1, description="Best-effort normalized job title.")
    seniority_level: str | None = Field(default=None, description="junior, mid, senior, lead, or principal.")
    years_experience_required: float | None = Field(
        default=None,
        ge=0.0,
        description="Minimum years of experience explicitly requested when detectable.",
    )
    required_skills: list[str] = Field(default_factory=list, description="Skills explicitly required.")
    nice_to_have_skills: list[str] = Field(default_factory=list, description="Skills marked as bonus or preferred.")
    responsibilities: list[str] = Field(default_factory=list, description="Short mission statements from the JD.")
    domain: str | None = Field(default=None, description="Best-effort business or technical domain.")
    location: str | None = Field(default=None, description="Location if stated in the JD.")
    language_requirements: list[str] = Field(default_factory=list, description="Required or preferred languages.")
    contract_type: str | None = Field(default=None, description="full_time, part_time, contract, internship, freelance.")
    remote_policy: str | None = Field(default=None, description="remote, hybrid, or on_site.")
    raw_job_description: str = Field(..., min_length=1, description="Original job description text.")
    metadata: JobMetadata = Field(..., description="Traceability metadata for this structured job profile.")
