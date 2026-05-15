from __future__ import annotations

import json
from typing import Any


def build_system_message() -> str:
    return (
        "You are a strict CV data extractor. "
        "Extract only what is explicitly written in the CV text. "
        "Never invent. Never complete missing information. "
        "Never generalize from job titles. "
        "When in doubt, return null or []."
    )


def build_grounded_extraction_prompt(
    cleaned_markdown: str,
    sections_found: list[str],
    detected_templates: list[str],
    document_confidence_score: float,
    header_info: dict[str, Any] | None = None,
) -> str:
    schema = {
        "status": "success|partial|unreadable",
        "bio": {
            "full_name": "string|null",
            "email": "string|null",
            "phone": "string|null",
            "location": "string|null",
            "linkedin": "string|null",
            "github": "string|null",
        },
        "expertise": {
            "summary": "string|null",
            "experience_level": "junior|mid|senior|null",
            "years_experience": "integer|null",
            "hard_skills": ["string"],
            "soft_skills": ["string"],
        },
        "experiences": [
            {
                "company": "string|null",
                "job_title": "string|null",
                "start_date": "string|null",
                "end_date": "string|null",
                "city": "string|null",
                "responsibilities": ["string"],
            }
        ],
        "education": [
            {
                "institution": "string|null",
                "degree": "string|null",
                "field": "string|null",
                "year": "integer|null",
            }
        ],
        "languages": ["string"],
        "certifications": ["string"],
    }
    warnings: list[str] = []
    if detected_templates:
        warnings.append(
            "Template/placeholders detected. They are not candidate data and must not be used: "
            + ", ".join(detected_templates)
        )
    if document_confidence_score < 0.6:
        warnings.append("The source confidence is below 0.6. Be extra conservative and prefer null/[] when unsure.")

    return (
        "Extract a grounded candidate profile from this CV text.\n\n"
        "Rules:\n"
        "- Extract ONLY what is explicitly written in the CV.\n"
        "- If a field is not clearly present, return null or [].\n"
        "- NEVER guess.\n"
        "- NEVER infer missing schools, dates, cities, responsibilities or skills.\n"
        "- NEVER use template values as real data.\n"
        "- NEVER create a marketing summary.\n"
        "- Summary must only compress facts already present in the CV.\n"
        "- Hard skills must be explicit.\n"
        "- Soft skills must be explicit.\n"
        "- Responsibilities must be verbatim or very close to the source.\n"
        "- Dates must not be invented or normalized if unclear.\n"
        "- Partial profiles are acceptable.\n"
        "- Hallucination is unacceptable.\n\n"
        f"Warnings: {json.dumps(warnings, ensure_ascii=False)}\n"
        f"Sections found: {json.dumps(sections_found, ensure_ascii=False)}\n"
        f"Document confidence score: {document_confidence_score}\n"
        f"Header hints from deterministic parser: {json.dumps(header_info or {}, ensure_ascii=False)}\n\n"
        "Expected JSON schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Return only valid JSON. No markdown. No explanation. No code block.\n\n"
        "CV text:\n"
        "-----\n"
        f"{cleaned_markdown}\n"
        "-----"
    )
