"""
Integration tests for Matching V3 score_candidate — no MongoDB, no FAISS required.

Calls the REAL score_candidate function with synthetic inline data.
Proves that the scoring logic computes correctly for perfect, partial and no-match cases.

Expected scores (pre-computed from formulas in scoring.py):
  profile quality (reliability=0.90, complete_profile, groq_secondary):
    score_profile_quality  = 0.5*0.90 + 0.3*1.0 + 0.2*1.0 = 0.95
    score_grounded_quality = 0.90*0.65 + 1.0*0.20 + 1.0*0.15 = 0.935
  combine_scores:
    base = (0.40*skills + 0.30*sim + 0.20*exp + 0.10*0.95)*0.85 + 0.935*0.15
  must-have penalty tiers: >=0.8->x1.0 | >=0.6->x0.85 | >=0.4->x0.65 | >=0.0->x0.45
"""
from __future__ import annotations

from src.core.matching.scoring import score_candidate


# ── Shared job profile ────────────────────────────────────────────────────────

_JOB = {
    "job_title": "Backend Python Engineer",
    "required_skills": ["Python", "FastAPI", "MongoDB"],
    "years_experience_required": 3,
}

# ── Shared high-quality profile template ─────────────────────────────────────

def _profile(**overrides) -> dict:
    """High-quality synthetic candidate profile. Override any field via kwargs."""
    base: dict = {
        "candidate_id": "cand_test",
        "bio": {"full_name": "Alice Martin"},
        "expertise": {
            "hard_skills": ["Python", "FastAPI", "MongoDB", "Docker"],
            "soft_skills": [],
        },
        "experiences": [{"start_date": "2020", "end_date": "2024"}],
        "profile_kind": "complete_profile",
        "provider_route": "groq_secondary",
        "reliability_score": 0.90,
        "quality_flags": [],
        "hallucination_risk": "low",
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Perfect match
#
# Candidate covers ALL required skills AND has enough experience.
# Expected: final_score high (≈0.96), matched_skills full, missing empty.
#
# score_skills=1.0, score_experience=1.0 (4 yrs ≥ 3), sim=0.9
# weighted = 0.40*1 + 0.30*0.9 + 0.20*1 + 0.10*0.95 = 0.965
# base    = 0.965*0.85 + 0.935*0.15 = 0.82025+0.14025 = 0.9605
# must_have_coverage=1.0 → no penalty
# quality_multiplier = 1.0 (low risk, complete, name ok)
# final_score = 0.9605
# ─────────────────────────────────────────────────────────────────────────────

def test_perfect_match() -> None:
    result = score_candidate(
        job_profile=_JOB,
        candidate_profile=_profile(),
        score_text_similarity=0.9,
    )

    assert result["final_score"] == 0.9605, (
        f"Perfect match expected final_score=0.9605, got {result['final_score']}"
    )
    assert result["matched_skills"] == ["Python", "FastAPI", "MongoDB"], (
        f"All required skills must be matched: {result['matched_skills']}"
    )
    assert result["missing_required_skills"] == [], (
        f"No skill should be missing: {result['missing_required_skills']}"
    )
    assert result["score_skills"] == 1.0
    assert result["score_experience"] == 1.0
    assert result["must_have_coverage"] == 1.0
    assert result["must_have_penalty_applied"] is False
    assert result["must_have_penalty_multiplier"] == 1.0
    assert result["quality_penalty_multiplier"] == 1.0
    assert result["hallucination_risk"] == "low"
    assert result["profile_kind"] == "complete_profile"
    assert result["final_score"] > 0.9, "Perfect match must score above 0.9"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — No match
#
# Candidate has none of the required skills, no relevant experience.
# Expected: final_score very low (≈0.11), all required skills missing.
#
# score_skills=0.0, score_experience=0.0, sim=0.1
# weighted = 0.40*0 + 0.30*0.1 + 0.20*0 + 0.10*0.95 = 0.125
# base    = 0.125*0.85 + 0.935*0.15 = 0.10625+0.14025 = 0.2465
# must_have_coverage=0.0 → tier >=0.0 → multiplier=0.45
# score_after = 0.2465*0.45 = 0.110925 → 0.1109
# quality_multiplier = 1.0
# final_score = 0.1109
# ─────────────────────────────────────────────────────────────────────────────

def test_no_match() -> None:
    result = score_candidate(
        job_profile=_JOB,
        candidate_profile=_profile(
            expertise={"hard_skills": ["Java", "Spring", "Oracle"], "soft_skills": []},
            experiences=[],
        ),
        score_text_similarity=0.1,
    )

    assert result["final_score"] == 0.1109, (
        f"No-match expected final_score=0.1109, got {result['final_score']}"
    )
    assert set(result["missing_required_skills"]) == {"Python", "FastAPI", "MongoDB"}, (
        f"All required skills must be missing: {result['missing_required_skills']}"
    )
    assert result["matched_skills"] == [], (
        f"No skill should be matched: {result['matched_skills']}"
    )
    assert result["score_skills"] == 0.0
    assert result["score_experience"] == 0.0
    assert result["must_have_coverage"] == 0.0
    assert result["must_have_penalty_applied"] is True
    assert result["must_have_penalty_multiplier"] == 0.45
    assert result["final_score"] < 0.2, "No-match must score below 0.2"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Partial match
#
# Candidate has 2/3 required skills (Python ✓, FastAPI ✓, MongoDB ✗).
# Expected: score between no-match and perfect, skills lists coherent.
#
# score_skills=2/3=0.6667, score_experience=2/3=0.6667 (2 yrs, req=3), sim=0.6
# weighted = 0.40*0.6667 + 0.30*0.6 + 0.20*0.6667 + 0.10*0.95 = 0.67502
# base    = 0.67502*0.85 + 0.935*0.15 = 0.573767+0.14025 = 0.714017 → 0.714
# must_have_coverage=0.6667 → tier >=0.6 → multiplier=0.85
# score_after = 0.714*0.85 = 0.6069
# final_score = 0.6069
# ─────────────────────────────────────────────────────────────────────────────

def test_partial_match() -> None:
    result = score_candidate(
        job_profile=_JOB,
        candidate_profile=_profile(
            expertise={"hard_skills": ["Python", "FastAPI", "Docker"], "soft_skills": []},
            experiences=[{"start_date": "2022", "end_date": "2024"}],
        ),
        score_text_similarity=0.6,
    )

    assert result["final_score"] == 0.6069, (
        f"Partial match expected final_score=0.6069, got {result['final_score']}"
    )
    assert "Python" in result["matched_skills"]
    assert "FastAPI" in result["matched_skills"]
    assert "MongoDB" in result["missing_required_skills"], (
        "MongoDB must be missing in the partial match"
    )
    assert len(result["matched_skills"]) == 2
    assert len(result["missing_required_skills"]) == 1
    assert result["must_have_penalty_applied"] is True
    assert result["must_have_penalty_multiplier"] == 0.85

    # Ordering invariant: perfect > partial > no-match
    # partial ≈ 0.61 — must sit between 0.2 and 0.9
    assert result["final_score"] > 0.2
    assert result["final_score"] < 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Must-have penalty (same experience, same similarity; one skill gap)
#
# Candidate A: all 3 required skills → coverage=1.0 → no penalty
# Candidate B: only 2/3 required skills → coverage=0.67 → multiplier=0.85
# Both have identical quality and similarity so the gap is purely from penalty.
#
# Candidate A: score_skills=1.0, sim=0.7, exp=1/3
#   weighted_A = 0.40+0.21+0.06666+0.095 = 0.77166
#   base_A = 0.77166*0.85 + 0.935*0.15 = 0.796161 → 0.7962
#   no penalty → final_A = 0.7962
# Candidate B: score_skills=0.6667, sim=0.7, exp=1/3
#   weighted_B = 0.26668+0.21+0.06666+0.095 = 0.63834
#   base_B = 0.63834*0.85 + 0.935*0.15 = 0.682839 → 0.6828
#   penalty 0.85 → score_after = 0.6828*0.85 = 0.58038 → 0.5804
#   final_B = 0.5804
# ─────────────────────────────────────────────────────────────────────────────

def test_must_have_penalty() -> None:
    short_exp = [{"start_date": "2023", "end_date": "2024"}]  # 1 year

    result_a = score_candidate(
        job_profile=_JOB,
        candidate_profile=_profile(
            expertise={"hard_skills": ["Python", "FastAPI", "MongoDB", "Docker"], "soft_skills": []},
            experiences=short_exp,
        ),
        score_text_similarity=0.7,
    )

    result_b = score_candidate(
        job_profile=_JOB,
        candidate_profile=_profile(
            expertise={"hard_skills": ["Python", "FastAPI", "Docker"], "soft_skills": []},
            experiences=short_exp,
        ),
        score_text_similarity=0.7,
    )

    # Exact expected values
    assert result_a["final_score"] == 0.7962, (
        f"Candidate A (no penalty) expected 0.7962, got {result_a['final_score']}"
    )
    assert result_b["final_score"] == 0.5804, (
        f"Candidate B (with penalty) expected 0.5804, got {result_b['final_score']}"
    )

    # Structural assertions
    assert result_a["must_have_penalty_applied"] is False
    assert result_a["must_have_penalty_multiplier"] == 1.0

    assert result_b["must_have_penalty_applied"] is True
    assert result_b["must_have_penalty_multiplier"] == 0.85

    # The penalty must lower the score
    assert result_a["final_score"] > result_b["final_score"], (
        f"Score with penalty ({result_b['final_score']}) must be below score without ({result_a['final_score']})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Deterministic: two identical calls produce identical results
# ─────────────────────────────────────────────────────────────────────────────

def test_deterministic() -> None:
    kwargs = dict(
        job_profile=_JOB,
        candidate_profile=_profile(),
        score_text_similarity=0.75,
    )

    result_1 = score_candidate(**kwargs)
    result_2 = score_candidate(**kwargs)

    assert result_1 == result_2, (
        "score_candidate must be deterministic: two identical calls must return identical dicts"
    )
    assert result_1["final_score"] == result_2["final_score"]
    assert result_1["matched_skills"] == result_2["matched_skills"]
    assert result_1["missing_required_skills"] == result_2["missing_required_skills"]
