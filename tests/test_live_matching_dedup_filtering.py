"""Live matching must filter profiles marked as is_duplicate_of.

When a profile is marked is_duplicate_of=<other_profile_id>, it should not appear
in the top_k results. The better profile (marked as primary) is kept.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.matching.live_matcher import LiveMatcher, LiveMatcherSettings


class _FakeIndex:
    def search(self, embedding, search_k):
        # Return 3 profiles: profiles 0 and 1 are duplicates, 2 is unique
        return (
            np.asarray([[0.95, 0.90, 0.85]], dtype="float32"),
            np.asarray([[0, 1, 2]], dtype="int64"),
        )


class _FakeModel:
    def encode(self, texts, **kwargs):
        return np.asarray([[1.0, 0.0]], dtype="float32")


class _FakeCandidateProfiles:
    """Simulates MongoDB with duplicate-marked profiles."""

    collection_name = "candidate_profiles"

    def __init__(self, profiles_by_id):
        self.profiles_by_id = profiles_by_id

    def resolve_profiles_for_rows(self, rows):
        """Resolve FAISS rows to MongoDB profiles. Includes duplicates."""
        result = []
        for row in rows:
            profile_id = row.get("profile_id")
            if profile_id in self.profiles_by_id:
                result.append(self.profiles_by_id[profile_id])
            else:
                result.append(None)
        return result


class _FakeJobProfiles:
    def get_job_profile(self, job_id):
        return None


class _FakeMatchingRuns:
    def save_matching_run(self, document):
        return "run_dedup_filter"


class _FakeRepositories:
    def __init__(self, profiles_by_id):
        self.candidate_profiles = _FakeCandidateProfiles(profiles_by_id)
        self.job_profiles = _FakeJobProfiles()
        self.matching_runs = _FakeMatchingRuns()


def _profile(pid, cid, name, is_duplicate_of=None, score_basis=0.8):
    """Synthetic profile template."""
    profile = {
        "profile_id": pid,
        "candidate_id": cid,
        "bio": {"full_name": name},
        "expertise": {
            "hard_skills": ["Python", "FastAPI"],
            "experience_level": "mid_level",
        },
        "experiences": [{"start_date": "2020", "end_date": "2024"}],
        "profile_kind": "complete_profile",
        "provider_route": "groq_secondary",
        "reliability_score": score_basis,
        "quality_flags": [],
    }
    if is_duplicate_of is not None:
        profile["is_duplicate_of"] = is_duplicate_of

    return profile


def test_live_matching_filters_is_duplicate_of():
    """Profiles marked is_duplicate_of must not appear in top_k results."""
    profiles_by_id = {
        # Primary profile (best)
        "profile_primary_bayrem": _profile("profile_primary_bayrem", "cand_bayrem_1", "Bayrem Abdelli"),
        # Duplicate marked (should be filtered)
        "profile_dup_bayrem": _profile(
            "profile_dup_bayrem",
            "cand_bayrem_2",
            "Bayrem Abdelli",
            is_duplicate_of="profile_primary_bayrem",
        ),
        # Unique profile (different person)
        "profile_other": _profile("profile_other", "cand_other", "Other Person"),
    }

    repositories = _FakeRepositories(profiles_by_id)
    matcher = LiveMatcher(
        repositories=repositories,
        settings=LiveMatcherSettings(mongodb_database="db", top_n=3, default_top_k=2),
        index_loader=lambda path: _FakeIndex(),
        id_map_loader=lambda path: [
            {"profile_id": "profile_primary_bayrem", "candidate_id": "cand_bayrem_1"},
            {"profile_id": "profile_dup_bayrem", "candidate_id": "cand_bayrem_2"},
            {"profile_id": "profile_other", "candidate_id": "cand_other"},
        ],
        model_loader=lambda name: _FakeModel(),
    )

    job_profile = {
        "generated_job_id": "gen_test",
        "job_title": "Python Developer",
        "required_skills": ["Python", "FastAPI"],
        "years_experience_required": 2,
    }

    result = matcher.match(
        job_description="Python FastAPI developer",
        job_id="gen_test",
        top_k=2,
        structured_job_profile=job_profile,
    )

    # Extract candidate_ids from results
    candidate_ids = [item["candidate_id"] for item in result.items]

    # Assertions
    assert len(result.items) == 2, f"Expected 2 candidates, got {len(result.items)}"

    # The duplicate (cand_bayrem_2) must NOT appear
    assert "cand_bayrem_2" not in candidate_ids, (
        f"Duplicate profile cand_bayrem_2 must be filtered out. Got: {candidate_ids}"
    )

    # The primary Bayrem must appear
    assert "cand_bayrem_1" in candidate_ids, (
        f"Primary profile cand_bayrem_1 must be included. Got: {candidate_ids}"
    )

    # The other candidate must appear (refilled the slot)
    assert "cand_other" in candidate_ids, (
        f"Other profile cand_other must be included. Got: {candidate_ids}"
    )


def test_all_profiles_duplicates_returns_primary():
    """If all top_n are duplicates, the primary ones are returned, not the marked ones."""
    profiles_by_id = {
        "profile_main_alice": _profile("profile_main_alice", "cand_alice_1", "Alice"),
        "profile_dup_alice": _profile("profile_dup_alice", "cand_alice_2", "Alice", is_duplicate_of="profile_main_alice"),
        "profile_main_bob": _profile("profile_main_bob", "cand_bob_1", "Bob"),
        "profile_dup_bob": _profile("profile_dup_bob", "cand_bob_2", "Bob", is_duplicate_of="profile_main_bob"),
    }

    repositories = _FakeRepositories(profiles_by_id)
    matcher = LiveMatcher(
        repositories=repositories,
        settings=LiveMatcherSettings(mongodb_database="db", top_n=4, default_top_k=2),
        index_loader=lambda path: _FakeIndexAllDups(),
        id_map_loader=lambda path: [
            {"profile_id": "profile_dup_alice", "candidate_id": "cand_alice_2"},  # dup first
            {"profile_id": "profile_main_alice", "candidate_id": "cand_alice_1"},  # primary second
            {"profile_id": "profile_dup_bob", "candidate_id": "cand_bob_2"},
            {"profile_id": "profile_main_bob", "candidate_id": "cand_bob_1"},
        ],
        model_loader=lambda name: _FakeModel(),
    )

    job_profile = {
        "generated_job_id": "gen_test",
        "job_title": "Developer",
        "required_skills": ["Python"],
        "years_experience_required": 1,
    }

    result = matcher.match(
        job_description="developer",
        job_id="gen_test",
        top_k=2,
        structured_job_profile=job_profile,
    )

    candidate_ids = [item["candidate_id"] for item in result.items]

    # Only primary profiles should appear, never the duplicates
    assert "cand_alice_2" not in candidate_ids
    assert "cand_bob_2" not in candidate_ids
    assert "cand_alice_1" in candidate_ids
    assert "cand_bob_1" in candidate_ids


class _FakeIndexAllDups:
    """Index returning 4 profiles."""

    def search(self, embedding, search_k):
        return (
            np.asarray([[0.95, 0.90, 0.85, 0.80]], dtype="float32"),
            np.asarray([[0, 1, 2, 3]], dtype="int64"),
        )


def test_no_duplicates_no_filtering():
    """Profiles without is_duplicate_of are returned as-is."""
    profiles_by_id = {
        "profile_alice": _profile("profile_alice", "cand_alice", "Alice"),
        "profile_bob": _profile("profile_bob", "cand_bob", "Bob"),
    }

    repositories = _FakeRepositories(profiles_by_id)
    matcher = LiveMatcher(
        repositories=repositories,
        settings=LiveMatcherSettings(mongodb_database="db", top_n=2, default_top_k=2),
        index_loader=lambda path: _FakeIndex(),
        id_map_loader=lambda path: [
            {"profile_id": "profile_alice", "candidate_id": "cand_alice"},
            {"profile_id": "profile_bob", "candidate_id": "cand_bob"},
        ],
        model_loader=lambda name: _FakeModel(),
    )

    job_profile = {
        "generated_job_id": "gen_test",
        "job_title": "Developer",
        "required_skills": ["Python"],
        "years_experience_required": 1,
    }

    result = matcher.match(
        job_description="developer",
        job_id="gen_test",
        top_k=2,
        structured_job_profile=job_profile,
    )

    candidate_ids = [item["candidate_id"] for item in result.items]

    # Both profiles should appear
    assert "cand_alice" in candidate_ids
    assert "cand_bob" in candidate_ids
