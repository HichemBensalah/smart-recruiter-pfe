"""Test fallback behavior when MongoDB is unavailable in hybrid mode.

If MATCHING_MODE=hybrid and MongoDB is down, the system should either:
1. Gracefully fallback to artifact-based matching, or
2. Raise a clear error (not a crash).
"""
from __future__ import annotations

import pytest

from src.core.matching.live_matcher import LiveMatcher, LiveMatcherSettings, LiveMatchingUnavailable


class _FailingCandidateProfiles:
    """Simulates MongoDB that is down or unavailable."""

    collection_name = "candidate_profiles"

    def resolve_profiles_for_rows(self, rows):
        raise ConnectionError("MongoDB connection refused")

    def get_profiles_by_ids(self, profile_ids):
        raise ConnectionError("MongoDB connection refused")


class _FailingJobProfiles:
    def get_job_profile(self, job_id):
        raise ConnectionError("MongoDB connection refused")


class _FakeMatchingRuns:
    def save_matching_run(self, document):
        return "run_test"


class _FailingRepositories:
    """Simulates all MongoDB repositories being down."""

    def __init__(self):
        self.candidate_profiles = _FailingCandidateProfiles()
        self.job_profiles = _FailingJobProfiles()
        self.matching_runs = _FakeMatchingRuns()


def test_live_matching_handles_mongodb_down():
    """When MongoDB is unavailable, LiveMatcher should raise or warn, not crash."""
    repositories = _FailingRepositories()
    matcher = LiveMatcher(
        repositories=repositories,
        settings=LiveMatcherSettings(mongodb_database="talent_intelligence"),
        index_loader=lambda path: _FailingFAISSIndex(),
        id_map_loader=lambda path: [],  # Empty id_map
        model_loader=lambda name: _FakeModel(),
    )

    job_profile = {
        "generated_job_id": "gen_test",
        "job_title": "Developer",
        "required_skills": ["Python"],
        "years_experience_required": 1,
    }

    # Should raise LiveMatchingUnavailable (not crash with ConnectionError)
    with pytest.raises(LiveMatchingUnavailable):
        matcher.match(
            job_description="developer",
            job_id="gen_test",
            top_k=5,
            structured_job_profile=job_profile,
        )


def test_live_matching_no_id_map_raises_clear_error():
    """When FAISS id_map is empty, should raise with a clear message."""
    repositories = _FailingRepositories()
    matcher = LiveMatcher(
        repositories=repositories,
        settings=LiveMatcherSettings(mongodb_database="db"),
        index_loader=lambda path: _FailingFAISSIndex(),
        id_map_loader=lambda path: [],  # Empty!
        model_loader=lambda name: _FakeModel(),
    )

    job_profile = {
        "job_title": "Backend Engineer",
        "required_skills": ["Python"],
        "years_experience_required": 2,
    }

    # Should raise with clear message, not crash
    with pytest.raises(LiveMatchingUnavailable, match="id_map is empty"):
        matcher.match(
            job_description="backend engineer",
            job_id=None,
            top_k=5,
            structured_job_profile=job_profile,
        )


def test_live_matching_faiss_down_raises():
    """When FAISS index is unavailable, should raise clearly."""
    repositories = _FailingRepositories()
    matcher = LiveMatcher(
        repositories=repositories,
        settings=LiveMatcherSettings(mongodb_database="db"),
        index_loader=lambda path: None,  # None simulates load failure
        id_map_loader=lambda path: [{"profile_id": "p1"}],  # id_map exists
        model_loader=lambda name: _FakeModel(),
    )

    job_profile = {"job_title": "Engineer", "required_skills": ["Python"]}

    # Should raise with message about FAISS
    with pytest.raises(LiveMatchingUnavailable, match="FAISS"):
        matcher.match(
            job_description="engineer",
            top_k=5,
            structured_job_profile=job_profile,
        )


class _FailingFAISSIndex:
    """Simulates a FAISS index that is corrupted or down."""

    def search(self, embedding, search_k):
        raise RuntimeError("FAISS index corrupted or unavailable")


class _FakeModel:
    def encode(self, texts, **kwargs):
        import numpy as np
        return np.asarray([[1.0, 0.0]], dtype="float32")
