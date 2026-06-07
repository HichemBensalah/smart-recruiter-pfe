from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.matching.scoring import (
    WEIGHT_EXPERIENCE,
    WEIGHT_PROFILE_QUALITY,
    WEIGHT_SKILLS,
    WEIGHT_TEXT_SIMILARITY,
    _EFFECTIVE_WEIGHT_EXPERIENCE,
    _EFFECTIVE_WEIGHT_GROUNDED_QUALITY,
    _EFFECTIVE_WEIGHT_PROFILE_QUALITY,
    _EFFECTIVE_WEIGHT_SKILLS,
    _EFFECTIVE_WEIGHT_TEXT_SIMILARITY,
    build_score_breakdown,
    score_candidate,
    apply_must_have_penalty,
    compute_quality_penalty_multiplier,
)


# ── Minimal synthetic job + candidate profiles (no files on disk needed) ──────

_JOB = {
    "job_id": "test_job_001",
    "required_skills": ["Python", "SQL"],
    "must_have_skills": [],
    "seniority_level": "mid",
    "min_years_experience": 2,
}

_CANDIDATE_STRONG = {
    "candidate_id": "cand_test_001",
    "profile_id": "prof_test_001",
    "profile_kind": "structured",
    "provider_route": "api",
    "reliability_score": 0.9,
    "quality_flags": [],
    "bio": {"full_name": "Test User"},
    "skills": {"technical": ["Python", "SQL", "FastAPI"]},
    "expertise": {"experience_level": "mid", "years": 3},
    "experiences": [{"title": "Developer", "duration_years": 3}],
    "grounded_score": 0.8,
    "fields_nullified": [],
}

_CANDIDATE_WEAK = {
    "candidate_id": "cand_test_002",
    "profile_id": "prof_test_002",
    "profile_kind": "raw",
    "provider_route": None,
    "reliability_score": 0.2,
    "quality_flags": ["low_reliability"],
    "bio": {"full_name": None},
    "skills": {},
    "expertise": {},
    "experiences": [],
    "grounded_score": 0.0,
    "fields_nullified": ["name", "skills"],
}


# ── Effective weight constants ─────────────────────────────────────────────────

def test_effective_weights_sum_to_one() -> None:
    total = (
        _EFFECTIVE_WEIGHT_SKILLS
        + _EFFECTIVE_WEIGHT_TEXT_SIMILARITY
        + _EFFECTIVE_WEIGHT_EXPERIENCE
        + _EFFECTIVE_WEIGHT_PROFILE_QUALITY
        + _EFFECTIVE_WEIGHT_GROUNDED_QUALITY
    )
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"


def test_effective_weights_match_formula() -> None:
    assert _EFFECTIVE_WEIGHT_SKILLS == round(WEIGHT_SKILLS * 0.85, 4)
    assert _EFFECTIVE_WEIGHT_TEXT_SIMILARITY == round(WEIGHT_TEXT_SIMILARITY * 0.85, 4)
    assert _EFFECTIVE_WEIGHT_EXPERIENCE == round(WEIGHT_EXPERIENCE * 0.85, 4)
    assert _EFFECTIVE_WEIGHT_PROFILE_QUALITY == round(WEIGHT_PROFILE_QUALITY * 0.85, 4)
    assert _EFFECTIVE_WEIGHT_GROUNDED_QUALITY == 0.15


# ── build_score_breakdown structure ───────────────────────────────────────────

def test_build_score_breakdown_returns_five_features() -> None:
    details = score_candidate(_JOB, _CANDIDATE_STRONG, score_text_similarity=0.75)
    breakdown = build_score_breakdown(details)
    assert len(breakdown) == 5


def test_build_score_breakdown_has_required_keys() -> None:
    details = score_candidate(_JOB, _CANDIDATE_STRONG, score_text_similarity=0.75)
    breakdown = build_score_breakdown(details)
    for entry in breakdown:
        assert "feature" in entry
        assert "raw_score" in entry
        assert "weight" in entry
        assert "contribution" in entry


def test_build_score_breakdown_sorted_by_contribution_desc() -> None:
    details = score_candidate(_JOB, _CANDIDATE_STRONG, score_text_similarity=0.75)
    breakdown = build_score_breakdown(details)
    contributions = [e["contribution"] for e in breakdown]
    assert contributions == sorted(contributions, reverse=True)


def test_build_score_breakdown_contribution_equals_raw_times_weight() -> None:
    details = score_candidate(_JOB, _CANDIDATE_STRONG, score_text_similarity=0.75)
    breakdown = build_score_breakdown(details)
    for entry in breakdown:
        expected = round(entry["raw_score"] * entry["weight"], 4)
        assert abs(entry["contribution"] - expected) < 1e-6, (
            f"contribution mismatch for {entry['feature']}: {entry['contribution']} != {expected}"
        )


def test_build_score_breakdown_contributions_sum_approx_base_score() -> None:
    details = score_candidate(_JOB, _CANDIDATE_STRONG, score_text_similarity=0.75)
    breakdown = build_score_breakdown(details)
    total_contribution = sum(e["contribution"] for e in breakdown)
    base_score = details["base_score_before_penalty"]
    assert abs(total_contribution - base_score) < 0.01, (
        f"Sum of contributions {total_contribution:.4f} differs from base_score_before_penalty {base_score:.4f}"
    )


def test_build_score_breakdown_feature_names_contain_no_ml_shap() -> None:
    details = score_candidate(_JOB, _CANDIDATE_STRONG, score_text_similarity=0.75)
    breakdown = build_score_breakdown(details)
    for entry in breakdown:
        name_lower = entry["feature"].lower()
        assert "shap" not in name_lower, f"Feature name contains 'SHAP': {entry['feature']}"
        assert " ml" not in f" {name_lower}", f"Feature name contains 'ML': {entry['feature']}"


def test_build_score_breakdown_scores_in_range() -> None:
    details = score_candidate(_JOB, _CANDIDATE_STRONG, score_text_similarity=0.75)
    breakdown = build_score_breakdown(details)
    for entry in breakdown:
        assert 0.0 <= entry["raw_score"] <= 1.0, f"raw_score out of range: {entry}"
        assert 0.0 <= entry["contribution"] <= 1.0, f"contribution out of range: {entry}"


def test_build_score_breakdown_weak_candidate_lower_contributions() -> None:
    details_strong = score_candidate(_JOB, _CANDIDATE_STRONG, score_text_similarity=0.75)
    details_weak = score_candidate(_JOB, _CANDIDATE_WEAK, score_text_similarity=0.10)
    bd_strong = build_score_breakdown(details_strong)
    bd_weak = build_score_breakdown(details_weak)
    sum_strong = sum(e["contribution"] for e in bd_strong)
    sum_weak = sum(e["contribution"] for e in bd_weak)
    assert sum_strong > sum_weak, "Strong candidate should have higher total contributions"


def test_build_score_breakdown_five_distinct_feature_names() -> None:
    details = score_candidate(_JOB, _CANDIDATE_WEAK, score_text_similarity=0.0)
    breakdown = build_score_breakdown(details)
    names = [e["feature"] for e in breakdown]
    assert len(set(names)) == 5, f"Expected 5 distinct feature names, got: {names}"


def test_build_score_breakdown_contains_expected_features() -> None:
    details = score_candidate(_JOB, _CANDIDATE_STRONG, score_text_similarity=0.75)
    breakdown = build_score_breakdown(details)
    names = {e["feature"] for e in breakdown}
    assert "Compétences" in names
    assert "Expérience" in names
    assert "Similarité sémantique" in names
    assert "Qualité du profil" in names
    assert "Qualité vérifiée" in names


# ── Penalty chain coherence tests ─────────────────────────────────────────────

# Candidate with deliberately low must-have coverage (triggers penalty)
_JOB_WITH_MUST_HAVE = {
    "job_id": "test_job_must_have",
    "required_skills": ["Python", "SQL"],
    "must_have_skills": ["Python", "Java", "Kubernetes", "Terraform"],
    "seniority_level": "senior",
    "min_years_experience": 5,
}

_CANDIDATE_LOW_MUST_HAVE = {
    "candidate_id": "cand_test_low_mh",
    "profile_id": "prof_test_low_mh",
    "profile_kind": "structured",
    "provider_route": "api",
    "reliability_score": 0.85,
    "quality_flags": [],
    "bio": {"full_name": "Test Penalized"},
    "skills": {"technical": ["Python"]},  # only covers 1/4 must-haves
    "expertise": {"experience_level": "mid", "years": 2},
    "experiences": [{"title": "Dev", "duration_years": 2}],
    "grounded_score": 0.7,
    "fields_nullified": [],
}


def test_full_penalty_chain_coherence() -> None:
    """base_score_before_penalty × must_have_mult × quality_mult == final_score"""
    details = score_candidate(_JOB_WITH_MUST_HAVE, _CANDIDATE_LOW_MUST_HAVE, score_text_similarity=0.6)
    base = details["base_score_before_penalty"]
    mhm = details["must_have_penalty_multiplier"]
    qm = details["quality_penalty_multiplier"]
    final = details["final_score"]
    computed = round(base * mhm * qm, 4)
    assert abs(computed - final) < 1e-3, (
        f"Chain broken: {base} × {mhm} × {qm} = {computed} != final_score {final}"
    )


def test_penalty_applied_when_must_have_coverage_low() -> None:
    details = score_candidate(_JOB_WITH_MUST_HAVE, _CANDIDATE_LOW_MUST_HAVE, score_text_similarity=0.6)
    assert details["must_have_penalty_applied"] is True
    assert details["must_have_penalty_multiplier"] < 1.0
    assert details["must_have_coverage"] < 0.5


def test_penalty_chain_reduces_final_score() -> None:
    details = score_candidate(_JOB_WITH_MUST_HAVE, _CANDIDATE_LOW_MUST_HAVE, score_text_similarity=0.6)
    assert details["final_score"] < details["base_score_before_penalty"]


def test_no_penalty_when_required_skills_well_covered() -> None:
    # must_have_coverage is based on required_skills overlap, not must_have_skills
    # Use a job whose required_skills the candidate fully covers
    job_simple = {
        "job_id": "test_job_simple",
        "required_skills": ["Python", "FastAPI"],
        "must_have_skills": ["Python", "FastAPI"],
        "min_years_experience": 1,
    }
    candidate_full = {
        **_CANDIDATE_LOW_MUST_HAVE,
        "candidate_id": "cand_full_req",
        "expertise": {"hard_skills": ["Python", "FastAPI", "SQL"], "experience_level": "senior"},
    }
    details = score_candidate(job_simple, candidate_full, score_text_similarity=0.8)
    assert details["must_have_coverage"] >= 0.8
    assert details["must_have_penalty_multiplier"] >= 0.85


def test_penalty_fields_always_present_in_score_details() -> None:
    """score_candidate() always returns all penalty fields."""
    for candidate in (_CANDIDATE_STRONG, _CANDIDATE_WEAK, _CANDIDATE_LOW_MUST_HAVE):
        details = score_candidate(_JOB, candidate, score_text_similarity=0.5)
        assert "base_score_before_penalty" in details
        assert "must_have_coverage" in details
        assert "must_have_penalty_multiplier" in details
        assert "must_have_penalty_applied" in details
        assert "quality_penalty_multiplier" in details
        assert "final_score" in details


def test_score_breakdown_plus_penalties_form_transparent_chain() -> None:
    """sum(contributions) == base_score AND base × penalties == final."""
    details = score_candidate(_JOB_WITH_MUST_HAVE, _CANDIDATE_LOW_MUST_HAVE, score_text_similarity=0.6)
    breakdown = build_score_breakdown(details)
    sum_contrib = sum(e["contribution"] for e in breakdown)
    base = details["base_score_before_penalty"]
    final = details["final_score"]
    mhm = details["must_have_penalty_multiplier"]
    qm = details["quality_penalty_multiplier"]
    # contributions reproduce base
    assert abs(sum_contrib - base) < 0.01, f"Sum {sum_contrib:.4f} != base {base:.4f}"
    # base × penalties reproduce final
    assert abs(round(base * mhm * qm, 4) - final) < 1e-3
