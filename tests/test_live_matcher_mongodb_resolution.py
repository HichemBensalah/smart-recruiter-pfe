"""FAISS id_map -> MongoDB candidate_profiles resolution.

Reproduces the real-world mismatch where FAISS id_map.pkl and MongoDB
candidate_profiles were built from different runs: profile_id only partially
overlaps, but artifact_path / source_path provide a perfect 1:1 join.
"""
from __future__ import annotations

import numpy as np

from src.core.matching.live_matcher import LiveMatcher, LiveMatcherSettings
from src.core.storage.repositories import CandidateProfileRepository


# ── Fake pymongo collection supporting find({"$or": [...]}) ────────────────────

class _FakeCollection:
    def __init__(self, documents: list[dict]):
        self.documents = documents

    def find(self, query, projection=None):
        or_clauses = query.get("$or", [])
        matched = []
        for doc in self.documents:
            for clause in or_clauses:
                field, cond = next(iter(clause.items()))
                values = cond.get("$in", [])
                if doc.get(field) in values:
                    matched.append(doc)
                    break
        return iter(matched)


def _make_repo(documents: list[dict]) -> CandidateProfileRepository:
    return CandidateProfileRepository(_FakeCollection(documents), "candidate_profiles")


# ── Repository-level resolution tests ─────────────────────────────────────────

def test_resolve_via_profile_id_direct():
    """Rows whose profile_id exists in MongoDB resolve directly."""
    docs = [{"profile_id": "profile_match", "candidate_id": "cand_1", "bio": {}}]
    repo = _make_repo(docs)

    rows = [{"profile_id": "profile_match", "candidate_id": "cand_1"}]
    resolved = repo.resolve_profiles_for_rows(rows)

    assert len(resolved) == 1
    assert resolved[0] is not None
    assert resolved[0]["profile_id"] == "profile_match"


def test_resolve_via_artifact_path_when_profile_id_mismatches():
    """The core fix: a row whose profile_id is absent from MongoDB still resolves
    through artifact_path (FAISS and MongoDB built in different runs)."""
    docs = [
        {
            "profile_id": "profile_MONGO_xyz",  # different from FAISS row
            "candidate_id": "cand_real",
            "artifact_path": "data\\processed_official_module1\\pdf\\CV5.json",
            "bio": {},
        }
    ]
    repo = _make_repo(docs)

    rows = [
        {
            "profile_id": "profile_FAISS_abc",  # NOT in MongoDB
            "candidate_id": "cand_faiss",  # NOT in MongoDB
            "artifact_path": "data\\processed_official_module1\\pdf\\CV5.json",  # matches
            "source_path": "data\\raw_cv\\pdf\\CV5.pdf",
        }
    ]
    resolved = repo.resolve_profiles_for_rows(rows)

    assert resolved[0] is not None
    assert resolved[0]["profile_id"] == "profile_MONGO_xyz"
    assert resolved[0]["candidate_id"] == "cand_real"


def test_resolve_via_source_path_fallback():
    """When only source_path matches, the row still resolves."""
    docs = [
        {
            "profile_id": "p_mongo",
            "candidate_id": "c_mongo",
            "source_path": "data\\raw_cv\\pdf\\CV9.pdf",
            "bio": {},
        }
    ]
    repo = _make_repo(docs)

    rows = [{"profile_id": "p_faiss", "source_path": "data\\raw_cv\\pdf\\CV9.pdf"}]
    resolved = repo.resolve_profiles_for_rows(rows)

    assert resolved[0] is not None
    assert resolved[0]["profile_id"] == "p_mongo"


def test_resolve_unresolved_row_returns_none():
    """A row with no matching key resolves to None (no crash, no spam)."""
    docs = [{"profile_id": "p_mongo", "artifact_path": "a", "source_path": "s"}]
    repo = _make_repo(docs)

    rows = [{"profile_id": "p_unknown", "artifact_path": "z", "source_path": "y"}]
    resolved = repo.resolve_profiles_for_rows(rows)

    assert resolved == [None]


def test_resolve_priority_profile_id_over_artifact_path():
    """profile_id takes priority over artifact_path when both could match."""
    docs = [
        {"profile_id": "p_target", "artifact_path": "shared.json", "candidate_id": "c_target"},
        {"profile_id": "p_other", "artifact_path": "other.json", "candidate_id": "c_other"},
    ]
    repo = _make_repo(docs)

    rows = [{"profile_id": "p_target", "artifact_path": "other.json"}]
    resolved = repo.resolve_profiles_for_rows(rows)

    # profile_id match wins over artifact_path match
    assert resolved[0]["profile_id"] == "p_target"


def test_resolve_empty_rows():
    repo = _make_repo([])
    assert repo.resolve_profiles_for_rows([]) == []


# ── LiveMatcher end-to-end with mismatched profile_id ─────────────────────────

class _FakeIndex:
    def search(self, embedding, search_k):
        return np.asarray([[0.91, 0.62]], dtype="float32"), np.asarray([[0, 1]], dtype="int64")


class _FakeModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        return np.asarray([[1.0, 0.0]], dtype="float32")


class _FakeCandidateProfilesWithResolution:
    collection_name = "candidate_profiles"

    def __init__(self, documents):
        self._repo = _make_repo(documents)

    def resolve_profiles_for_rows(self, rows):
        return self._repo.resolve_profiles_for_rows(rows)


class _FakeJobProfiles:
    def get_job_profile(self, job_id):
        return None


class _FakeMatchingRuns:
    def save_matching_run(self, document):
        return "run_resolution_1"


class _FakeRepositories:
    def __init__(self, documents):
        self.candidate_profiles = _FakeCandidateProfilesWithResolution(documents)
        self.job_profiles = _FakeJobProfiles()
        self.matching_runs = _FakeMatchingRuns()


def test_live_matcher_resolves_mismatched_profile_ids_via_artifact_path():
    """LiveMatcher returns candidates even when FAISS profile_ids differ from
    MongoDB profile_ids, by resolving through artifact_path."""
    documents = {
        # MongoDB documents keyed by their (different) profile_ids
        "doc1": {
            "profile_id": "profile_MONGO_1",
            "candidate_id": "candidate_1",
            "artifact_path": "cv1.json",
            "bio": {"full_name": "Alice Engineer"},
            "expertise": {"hard_skills": ["Python", "FastAPI"], "experience_level": "mid_level"},
            "experiences": [{"start_date": "2020", "end_date": "2024"}],
            "profile_kind": "complete_profile",
            "provider_route": "groq_secondary",
            "reliability_score": 0.95,
            "quality_flags": [],
        },
        "doc2": {
            "profile_id": "profile_MONGO_2",
            "candidate_id": "candidate_2",
            "artifact_path": "cv2.json",
            "bio": {"full_name": "Bob Coder"},
            "expertise": {"hard_skills": ["React"], "experience_level": "junior"},
            "experiences": [],
            "profile_kind": "complete_profile",
            "provider_route": "groq_secondary",
            "reliability_score": 0.9,
            "quality_flags": [],
        },
    }
    repositories = _FakeRepositories(list(documents.values()))

    matcher = LiveMatcher(
        repositories=repositories,
        settings=LiveMatcherSettings(mongodb_database="test_db", top_n=2),
        index_loader=lambda path: _FakeIndex(),
        # FAISS id_map has DIFFERENT profile_ids than MongoDB, but same artifact_path
        id_map_loader=lambda path: [
            {"profile_id": "profile_FAISS_1", "candidate_id": "cand_faiss_1", "artifact_path": "cv1.json"},
            {"profile_id": "profile_FAISS_2", "candidate_id": "cand_faiss_2", "artifact_path": "cv2.json"},
        ],
        model_loader=lambda model_name: _FakeModel(),
    )

    job_profile = {
        "generated_job_id": "gen_1",
        "job_title": "Backend Python Engineer",
        "seniority_level": "mid_level",
        "required_skills": ["Python", "FastAPI"],
        "responsibilities": ["Build APIs."],
        "years_experience_required": 3,
    }

    result = matcher.match(
        job_description="Backend Python FastAPI",
        job_id="gen_1",
        top_k=2,
        structured_job_profile=job_profile,
    )

    # Candidates resolved despite profile_id mismatch
    assert len(result.items) >= 1
    # The resolved candidate uses MongoDB's real candidate_id, not the FAISS one
    resolved_candidate_ids = {item["candidate_id"] for item in result.items}
    assert "candidate_1" in resolved_candidate_ids
    # No per-id "not found" spam warning
    assert not any("Candidate profile not found" in w for w in result.warnings)
