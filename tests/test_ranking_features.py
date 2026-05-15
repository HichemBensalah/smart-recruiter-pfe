from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.common.seniority import compute_seniority_alignment, normalize_seniority
from src.core.jobs.job_profile_builder import build_job_profile
from src.core.ranking.features import (
    build_feature_rows,
    build_feature_row,
    encode_hallucination_risk,
    extract_required_skills,
    write_jsonl,
)
from src.core.ranking.schema import RankingFeatures


def test_encode_hallucination_risk() -> None:
    assert encode_hallucination_risk(None) == 0.0
    assert encode_hallucination_risk("LOW") == 0.0
    assert encode_hallucination_risk("faible") == 0.0
    assert encode_hallucination_risk("medium") == 0.5
    assert encode_hallucination_risk("moyen") == 0.5
    assert encode_hallucination_risk("HIGH") == 1.0
    assert encode_hallucination_risk("élevé") == 1.0
    assert encode_hallucination_risk(1.5) == 1.0
    assert encode_hallucination_risk(-0.2) == 0.0


def test_required_skills_extraction() -> None:
    assert extract_required_skills({"required_skills": ["Python"]}) == ["Python"]
    assert extract_required_skills({"must_have_skills": ["FastAPI"]}) == ["FastAPI"]
    assert extract_required_skills({"skills": {"required": ["MongoDB"]}}) == ["MongoDB"]
    assert extract_required_skills({"requirements": {"required_skills": ["Docker"]}}) == ["Docker"]


def test_normalize_seniority_aliases() -> None:
    assert normalize_seniority("jr") == "junior"
    assert normalize_seniority("débutant") == "junior"
    assert normalize_seniority("mid") == "mid_level"
    assert normalize_seniority("intermédiaire") == "mid_level"
    assert normalize_seniority("confirmé") == "mid_level"
    assert normalize_seniority("sr") == "senior"
    assert normalize_seniority("expérimenté") == "senior"
    assert normalize_seniority("tech lead") == "lead"
    assert normalize_seniority("architecte") == "principal"
    assert normalize_seniority("unknown") is None


def test_compute_seniority_alignment_exact_match() -> None:
    assert compute_seniority_alignment("mid", "mid_level") == 1.0


def test_compute_seniority_alignment_candidate_one_level_above() -> None:
    assert compute_seniority_alignment("mid_level", "senior") == 0.85


def test_compute_seniority_alignment_candidate_one_level_below() -> None:
    assert compute_seniority_alignment("senior", "mid_level") == 0.70


def test_compute_seniority_alignment_missing_values() -> None:
    assert compute_seniority_alignment(None, "senior") == 0.0
    assert compute_seniority_alignment("senior", None) == 0.0


def test_job_profile_seniority_is_normalized() -> None:
    profile = build_job_profile("Backend Python Engineer\nThis is a mid-level role with at least 3 years of experience.")
    assert profile.seniority_level == "mid_level"


def test_build_feature_row_minimal() -> None:
    row = build_feature_row(
        {"required_skills": ["Python", "FastAPI"]},
        {
            "candidate_id": "candidate-1",
            "profile_id": "profile-1",
            "faiss_score": 0.72,
            "final_score": 0.77,
            "matched_skills": ["Python"],
            "missing_required_skills": ["FastAPI"],
            "reliability_score": 0.87,
            "hallucination_risk": "medium",
            "job_seniority": "mid_level",
            "candidate_seniority": "senior",
        },
        "job-1",
        1,
    )
    assert row.job_id == "job-1"
    assert row.rank == 1
    assert row.features.must_have_coverage == 0.5
    assert row.features.hallucination_risk_encoded == 0.5
    assert row.features.seniority_alignment == 0.85


def test_feature_builder_reads_seniority_alignment_from_matching_report() -> None:
    rows = build_feature_rows(
        {"seniority_level": "senior", "required_skills": ["Python"]},
        {
            "results": [
                {
                    "recommendations": [
                        {
                            "candidate_id": "c1",
                            "rank": 1,
                            "seniority_alignment": 0.65,
                        }
                    ]
                }
            ]
        },
        "job-1",
    )
    assert rows[0].features.seniority_alignment == 0.65


def test_feature_builder_seniority_fallback_without_values_is_zero() -> None:
    row = build_feature_row({"seniority_level": "senior"}, {"candidate_id": "c1"}, "job-1", 1)
    assert row.features.seniority_alignment == 0.0


def test_no_label_invented() -> None:
    row = build_feature_row({"required_skills": ["Python"]}, {"candidate_id": "c1"}, "job-1", 1)
    assert row.label is None
    assert row.split is None
    assert len(RankingFeatures.model_fields) == 12


def test_feature_values_are_numeric() -> None:
    row = build_feature_row({"required_skills": ["Python"]}, {"candidate_id": "c1"}, "job-1", 1)
    for value in row.features.model_dump().values():
        assert isinstance(value, float)


def test_jsonl_write(tmp_path) -> None:
    row = build_feature_row({"required_skills": ["Python"]}, {"candidate_id": "c1"}, "job-1", 1)
    output = tmp_path / "features.jsonl"
    write_jsonl([row], output)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["job_id"] == "job-1"
    assert payload["label"] is None
    assert payload["split"] is None


def test_missing_required_plus_matched_consistency_when_required_skills_available() -> None:
    row = build_feature_row(
        {"required_skills": ["Python", "FastAPI", "MongoDB"]},
        {
            "candidate_id": "c1",
            "matched_skills": ["Python", "FastAPI"],
            "missing_required_skills": ["MongoDB"],
        },
        "job-1",
        1,
    )
    features = row.features
    total = features.matched_required_count + features.missing_required_count
    assert total == 3.0


def test_generated_matching_report_contains_seniority_fields() -> None:
    report_path = ROOT / "docs/reports/matching/v3/matching_report_v3_normalized.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recommendation = report["results"][0]["recommendations"][0]
    assert "job_seniority" in recommendation
    assert "candidate_seniority" in recommendation
    assert "seniority_alignment" in recommendation
    assert 0.0 <= float(recommendation["seniority_alignment"]) <= 1.0
