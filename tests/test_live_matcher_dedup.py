"""Candidate-level deduplication in live matching.

A real person can survive under two distinct candidate_ids (e.g. a docx CV merged
by phone vs an image CV with no phone). Grouping by candidate_id alone is not
enough; deduplicate_by_identity removes same-person entries from the top list
without touching scores or MongoDB documents.
"""
from __future__ import annotations

import numpy as np

from src.core.matching.live_matcher import (
    LiveMatcher,
    LiveMatcherSettings,
    candidate_identity_key,
    deduplicate_by_identity,
)


# ── Identity key tests ────────────────────────────────────────────────────────

def test_identity_key_prefers_real_email():
    item = {
        "email_normalized": "hichem.salah1512@gmail.com",
        "email_class": "real",
        "phone_normalized": "21697118755",
        "phone_class": "real",
        "candidate_id": "candidate_1",
    }
    assert candidate_identity_key(item) == ("email", "hichem.salah1512@gmail.com")


def test_identity_key_falls_back_to_real_phone():
    item = {
        "email_normalized": None,
        "phone_normalized": "21697118755",
        "phone_class": "real",
        "candidate_id": "candidate_1",
    }
    assert candidate_identity_key(item) == ("phone", "21697118755")


def test_identity_key_uses_name_only_when_valid():
    valid = {"name_normalized": "hichembensalah", "has_valid_name": True, "candidate_id": "c1"}
    assert candidate_identity_key(valid) == ("name_normalized", "hichembensalah")

    # Garbage OCR name without has_valid_name must NOT be used as a merge key
    garbage = {"name_normalized": "oarianatunisia", "has_valid_name": False,
               "source_path": "data\\raw_cv\\images\\x.jpg", "candidate_id": "c2"}
    key = candidate_identity_key(garbage)
    assert key[0] != "name_normalized"


def test_identity_key_source_path_stem():
    item = {"source_path": "data\\raw_cv\\docx\\Hichem_resume.docx", "candidate_id": "c1"}
    assert candidate_identity_key(item) == ("source_path", "data/raw_cv/docx/hichem_resume")


def test_identity_key_falls_back_to_candidate_id():
    item = {"candidate_id": "candidate_xyz"}
    assert candidate_identity_key(item) == ("candidate_id", "candidate_xyz")


# ── deduplicate_by_identity tests ─────────────────────────────────────────────

def test_dedup_same_email_keeps_best_score():
    """Two profiles with the same real email -> one candidate, the higher score kept."""
    candidates = [
        {"candidate_id": "c_high", "email_normalized": "a@x.com", "email_class": "real", "final_score": 0.83},
        {"candidate_id": "c_low", "email_normalized": "a@x.com", "email_class": "real", "final_score": 0.72},
    ]
    deduped, info = deduplicate_by_identity(candidates)

    assert len(deduped) == 1
    assert deduped[0]["candidate_id"] == "c_high"  # higher score kept (input sorted desc)
    assert info["duplicate_candidates_filtered"] is True
    assert info["duplicates_removed_count"] == 1
    assert info["duplicate_groups"][0]["kept_candidate_id"] == "c_high"
    assert info["duplicate_groups"][0]["removed_candidate_id"] == "c_low"


def test_dedup_same_name_normalized_keeps_best():
    candidates = [
        {"candidate_id": "c1", "name_normalized": "johndoe", "has_valid_name": True, "final_score": 0.9},
        {"candidate_id": "c2", "name_normalized": "johndoe", "has_valid_name": True, "final_score": 0.5},
    ]
    deduped, info = deduplicate_by_identity(candidates)

    assert len(deduped) == 1
    assert deduped[0]["candidate_id"] == "c1"
    assert info["duplicates_removed_count"] == 1


def test_dedup_does_not_mutate_scores():
    candidates = [
        {"candidate_id": "c1", "email_normalized": "a@x.com", "email_class": "real", "final_score": 0.83},
        {"candidate_id": "c2", "email_normalized": "a@x.com", "email_class": "real", "final_score": 0.72},
    ]
    deduped, _ = deduplicate_by_identity(candidates)
    assert deduped[0]["final_score"] == 0.83  # unchanged


def test_dedup_distinct_people_without_contact_stay_separate():
    """No real email/phone and no valid name -> keyed by candidate_id, never merged."""
    candidates = [
        {"candidate_id": "c1", "has_valid_name": False, "final_score": 0.6},
        {"candidate_id": "c2", "has_valid_name": False, "final_score": 0.5},
    ]
    deduped, info = deduplicate_by_identity(candidates)
    assert len(deduped) == 2
    assert info["duplicates_removed_count"] == 0


def test_dedup_no_duplicates_reports_clean():
    candidates = [
        {"candidate_id": "c1", "email_normalized": "a@x.com", "email_class": "real", "final_score": 0.8},
        {"candidate_id": "c2", "email_normalized": "b@x.com", "email_class": "real", "final_score": 0.7},
    ]
    deduped, info = deduplicate_by_identity(candidates)
    assert len(deduped) == 2
    assert info["duplicate_candidates_filtered"] is False
    assert info["duplicates_removed_count"] == 0


# ── LiveMatcher end-to-end: top_k refilled after dedup ────────────────────────

class _FakeIndex:
    def search(self, embedding, search_k):
        return (
            np.asarray([[0.95, 0.90, 0.85]], dtype="float32"),
            np.asarray([[0, 1, 2]], dtype="int64"),
        )


class _FakeModel:
    def encode(self, texts, **kwargs):
        return np.asarray([[1.0, 0.0]], dtype="float32")


def _profile(pid, cid, name, email=None, skills=("Python", "FastAPI")):
    return {
        "profile_id": pid,
        "candidate_id": cid,
        "email_normalized": email,
        "email_class": "real" if email else "missing",
        "name_normalized": name.replace(" ", "").lower(),
        "has_valid_name": True,
        "bio": {"full_name": name},
        "expertise": {"hard_skills": list(skills), "experience_level": "mid_level"},
        "experiences": [{"start_date": "2020", "end_date": "2024"}],
        "profile_kind": "complete_profile",
        "provider_route": "groq_secondary",
        "reliability_score": 0.9,
        "quality_flags": [],
    }


class _FakeCandidateProfiles:
    collection_name = "candidate_profiles"

    def __init__(self, by_row):
        self.by_row = by_row

    def resolve_profiles_for_rows(self, rows):
        return [self.by_row.get(str(r.get("profile_id"))) for r in rows]


class _FakeJobProfiles:
    def get_job_profile(self, job_id):
        return None


class _FakeMatchingRuns:
    def save_matching_run(self, document):
        return "run_dedup"


class _FakeRepositories:
    def __init__(self, by_row):
        self.candidate_profiles = _FakeCandidateProfiles(by_row)
        self.job_profiles = _FakeJobProfiles()
        self.matching_runs = _FakeMatchingRuns()


def test_live_matcher_dedups_and_refills_top_k():
    """Two of three retrieved profiles are the same person (same email); the top_k=2
    must still return 2 DISTINCT candidates (refilled with the third)."""
    by_row = {
        "p1": _profile("p1", "cand_hichem_docx", "Hichem Bensalah", email="hichem@gmail.com"),
        "p2": _profile("p2", "cand_hichem_image", "O Ariana", email="hichem@gmail.com"),
        "p3": _profile("p3", "cand_other", "Other Person", email="other@gmail.com"),
    }
    repositories = _FakeRepositories(by_row)
    matcher = LiveMatcher(
        repositories=repositories,
        settings=LiveMatcherSettings(mongodb_database="db", top_n=3),
        index_loader=lambda path: _FakeIndex(),
        id_map_loader=lambda path: [
            {"profile_id": "p1", "candidate_id": "cand_hichem_docx"},
            {"profile_id": "p2", "candidate_id": "cand_hichem_image"},
            {"profile_id": "p3", "candidate_id": "cand_other"},
        ],
        model_loader=lambda name: _FakeModel(),
    )

    job_profile = {
        "generated_job_id": "gen_1",
        "job_title": "Backend Python Engineer",
        "seniority_level": "mid_level",
        "required_skills": ["Python", "FastAPI"],
        "responsibilities": ["Build APIs."],
        "years_experience_required": 2,
    }
    result = matcher.match(
        job_description="Backend Python FastAPI",
        job_id="gen_1",
        top_k=2,
        structured_job_profile=job_profile,
    )

    candidate_ids = [item["candidate_id"] for item in result.items]
    # Two distinct candidates
    assert len(result.items) == 2
    assert len(set(candidate_ids)) == 2
    # The two Hichem profiles do not both appear
    assert not ("cand_hichem_docx" in candidate_ids and "cand_hichem_image" in candidate_ids)
    # The third distinct candidate refilled the slot
    assert "cand_other" in candidate_ids
    # Dedup metadata exposed
    assert result.dedup_info["duplicate_candidates_filtered"] is True
    assert result.dedup_info["duplicates_removed_count"] == 1
