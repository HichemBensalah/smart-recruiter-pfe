from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.structuring.grounded_prompt import build_grounded_extraction_prompt
from src.core.structuring.grounding_validator import validate_and_ground
from src.core.structuring.markdown_normalizer import normalize_markdown


def load_entries(path: Path, limit: int) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    accepted = [
        row
        for row in rows
        if row.get("eligible_for_module2") is True
        and row.get("handoff_lane") == "accepted"
        and row.get("artifact_path")
    ]
    return accepted[:limit]


def build_fake_llm_output(normalized: dict) -> dict:
    header = normalized.get("header_info") or {}
    return {
        "status": "partial",
        "bio": {
            "full_name": header.get("full_name"),
            "email": header.get("email") or "email@youremail.com",
            "phone": header.get("phone"),
            "location": header.get("location"),
            "linkedin": header.get("linkedin"),
            "github": header.get("github"),
        },
        "expertise": {
            "summary": "Highly motivated professional with strong background in invented cloud architecture",
            "experience_level": "senior",
            "years_experience": 99,
            "hard_skills": ["Python", "ImaginarySkillXYZ"],
            "soft_skills": ["teamwork"],
        },
        "experiences": [
            {
                "company": "Unsupported Company XYZ",
                "job_title": "Invented Role",
                "start_date": None,
                "end_date": None,
                "city": "Unknown",
                "responsibilities": ["Unsupported responsibility created by fake output"],
            }
        ],
        "education": [{"institution": "Unsupported University", "degree": None, "field": None, "year": None}],
        "languages": ["English"],
        "certifications": ["Unsupported Certification"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test grounded Module 2 V2 components without provider calls.")
    parser.add_argument("--accepted-path", default="data/processed_official_module1/handoff/accepted.json")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    for index, entry in enumerate(load_entries(Path(args.accepted_path), args.limit), start=1):
        artifact = json.loads(Path(entry["artifact_path"]).read_text(encoding="utf-8"))
        normalized = normalize_markdown(artifact.get("markdown") or "", artifact.get("raw_text") or "")
        prompt = build_grounded_extraction_prompt(
            normalized["cleaned_markdown"],
            normalized["sections_found"],
            normalized["detected_templates"],
            float(entry.get("document_confidence_score") or 0.0),
            normalized["header_info"],
        )
        grounded = validate_and_ground(
            build_fake_llm_output(normalized),
            normalized["cleaned_markdown"],
            normalized["detected_templates"],
            float(entry.get("document_confidence_score") or 0.0),
        )
        print(f"\n=== SAMPLE {index}: {entry['artifact_path']} ===")
        print("cleaned_preview:", normalized["cleaned_markdown"][:260].replace("\n", " "))
        print("detected_templates:", normalized["detected_templates"])
        print("sections_found:", normalized["sections_found"])
        print("quality_score:", normalized["quality_score"])
        print("prompt_length:", len(prompt))
        print("reliability_score:", grounded["reliability_score"])
        print("profile_kind:", grounded["profile_kind"])
        print("fields_nullified:", grounded["fields_nullified"][:12])
        print("hallucination_risk:", grounded["hallucination_risk"])


if __name__ == "__main__":
    main()
