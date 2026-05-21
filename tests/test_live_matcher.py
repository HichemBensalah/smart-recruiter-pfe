from __future__ import annotations

import numpy as np

from src.core.matching.live_matcher import LiveMatcher, LiveMatcherSettings, normalize_candidate_profile_for_matching


class _FakeIndex:
    def search(self, embedding, search_k):
        return np.asarray([[0.92, 0.51]], dtype="float32"), np.asarray([[0, 1]], dtype="int64")


class _FakeModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        return np.asarray([[1.0, 0.0]], dtype="float32")


class _FakeCandidateProfiles:
    collection_name = "candidate_profiles"

    def __init__(self, profiles):
        self.profiles = profiles

    def get_profiles_by_ids(self, profile_ids):
        return {profile_id: self.profiles[profile_id] for profile_id in profile_ids if profile_id in self.profiles}


class _FakeJobProfiles:
    def get_job_profile(self, job_id):
        return {
            "job_id": job_id,
            "job_title": "Backend Python Engineer",
            "seniority_level": "mid_level",
            "years_experience_required": 3,
            "required_skills": ["Python", "FastAPI"],
            "nice_to_have_skills": ["Docker"],
            "responsibilities": ["Build APIs with Python and FastAPI."],
            "raw_job_description": "Backend Python Engineer with FastAPI.",
        }


class _FakeMatchingRuns:
    def __init__(self):
        self.saved = None

    def save_matching_run(self, document):
        self.saved = document
        return "run_live_1"


class _FakeRepositories:
    def __init__(self, profiles):
        self.candidate_profiles = _FakeCandidateProfiles(profiles)
        self.job_profiles = _FakeJobProfiles()
        self.matching_runs = _FakeMatchingRuns()


def test_live_matcher_uses_faiss_rows_mongodb_profiles_and_matching_v3_scoring() -> None:
    profiles = {
        "profile_1": {
            "profile_id": "profile_1",
            "candidate_id": "candidate_1",
            "bio": {"full_name": "Jane Doe"},
            "expertise": {"hard_skills": ["Python", "FastAPI"], "experience_level": "mid_level"},
            "experiences": [{"start_date": "2020", "end_date": "2024"}],
            "profile_kind": "complete_profile",
            "provider_route": "groq_secondary",
            "reliability_score": 0.95,
            "quality_flags": [],
        },
        "profile_2": {
            "profile_id": "profile_2",
            "candidate_id": "candidate_2",
            "bio": {"full_name": "John Smith"},
            "expertise": {"hard_skills": ["React"], "experience_level": "junior"},
            "experiences": [],
            "profile_kind": "complete_profile",
            "provider_route": "groq_secondary",
            "reliability_score": 0.9,
            "quality_flags": [],
        },
    }
    repositories = _FakeRepositories(profiles)
    matcher = LiveMatcher(
        repositories=repositories,
        settings=LiveMatcherSettings(mongodb_database="test_db", top_n=2),
        index_loader=lambda path: _FakeIndex(),
        id_map_loader=lambda path: [
            {"profile_id": "profile_1", "candidate_id": "candidate_1"},
            {"profile_id": "profile_2", "candidate_id": "candidate_2"},
        ],
        model_loader=lambda model_name: _FakeModel(),
    )

    result = matcher.match(job_description="Backend Python FastAPI", job_id="job_backend", top_k=1)

    assert result.matching_run_id == "run_live_1"
    assert result.data_source == "mongodb:test_db.candidate_profiles"
    assert result.retrieval_source == "faiss:data/indexes/faiss/cv_index.faiss"
    assert result.scoring_source == "matching_v3.score_candidate"
    assert result.items[0]["candidate_id"] == "candidate_1"
    assert result.items[0]["baseline_rank_v3"] == 1
    assert result.items[0]["baseline_score_v3"] is not None
    assert result.items[0]["features"]["final_score_v3"] == result.items[0]["baseline_score_v3"]
    assert repositories.matching_runs.saved["matching_mode"] == "live_mongodb_faiss_matching_v3"
    assert repositories.matching_runs.saved["candidate_ids"] == ["candidate_1"]


def test_live_matcher_normalizes_raw_seed_profile_payload_for_scoring() -> None:
    normalized = normalize_candidate_profile_for_matching(
        {
            "profile_id": "profile_raw",
            "candidate_id": "candidate_raw",
            "profile_kind": "complete_profile",
            "provider_used": "groq_secondary",
            "profile": {
                "bio": {"full_name": "Raw Candidate"},
                "expertise": {"hard_skills": ["Python"]},
                "experiences": [],
            },
            "grounding": {
                "reliability_score": 0.88,
                "hallucination_risk": "low",
                "quality_flags": ["grounded"],
                "fields_nullified": ["bio.email"],
            },
            "normalization": {"quality_flags": ["missing_email"]},
        }
    )

    assert normalized["bio"]["full_name"] == "Raw Candidate"
    assert normalized["expertise"]["hard_skills"] == ["Python"]
    assert normalized["provider_route"] == "groq_secondary"
    assert normalized["reliability_score"] == 0.88
    assert normalized["fields_nullified_count"] == 1
    assert normalized["quality_flags"] == ["missing_email", "grounded"]
