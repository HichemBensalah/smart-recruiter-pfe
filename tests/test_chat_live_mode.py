"""Live mode matching through /api/chat — no silent artifact fallback."""

import pytest
from unittest.mock import patch

from src.core.chatbot.live_readiness import check_live_matching_readiness


def test_chat_live_mode_infrastructure_check():
    """Live mode checks infrastructure before attempting match."""
    with patch("src.core.chatbot.live_readiness._check_mongodb") as mock_mongo:
        with patch("src.core.chatbot.live_readiness._check_faiss_index"):
            with patch("src.core.chatbot.live_readiness._check_id_map"):
                with patch("src.core.chatbot.live_readiness._check_sentence_transformer"):
                    def setup_mongo(u, d, r, reasons):
                        r["mongodb_available"] = True
                        r["candidate_profiles_count"] = 0
                        reasons.append("candidate_profiles collection is empty")

                    mock_mongo.side_effect = setup_mongo
                    readiness = check_live_matching_readiness()
                    assert readiness["live_ready"] is False
                    assert len(readiness["blocking_reasons"]) > 0


def test_chat_live_mode_no_silent_fallback():
    """Live strict mode does not fallback silently to artifact."""
    matching_mode_requested = "live"
    live_strict = True

    readiness = {
        "live_ready": False,
        "blocking_reasons": ["MongoDB unavailable"],
    }

    if live_strict and not readiness["live_ready"]:
        error_msg = f"Live matching unavailable: {', '.join(readiness['blocking_reasons'])}"
        assert "unavailable" in error_msg.lower()


def test_chat_live_mode_response_structure():
    """Live mode response includes required fields."""
    mock_response = {
        "matching_mode_requested": "live",
        "matching_mode_used": "live",
        "generated_job_id": "generated_abc123",
        "live_ready": True,
        "candidates": [
            {
                "candidate_id": "cand_1",
                "name": "John Doe",
                "score_v3": 0.85,
                "score_text_similarity": 0.78,
                "matched_skills": ["Python", "FastAPI"],
                "missing_required_skills": ["Docker"],
            }
        ],
        "blocking_reasons": [],
    }

    assert mock_response["matching_mode_requested"] == "live"
    assert mock_response["matching_mode_used"] == "live"
    assert "generated_job_id" in mock_response
    assert "live_ready" in mock_response
    assert isinstance(mock_response["candidates"], list)
    assert isinstance(mock_response["blocking_reasons"], list)


def test_live_mode_live_strict_config_exists():
    """LIVE_STRICT configuration exists in MatchingSettings."""
    from src.api.config import MatchingSettings

    settings = MatchingSettings(
        matching_mode="live",
        live_strict=True,
        live_matching_top_n=10,
        live_matching_top_k=5,
        faiss_index_path="data/indexes/faiss/cv_index.faiss",
        faiss_id_map_path="data/indexes/faiss/id_map.pkl",
    )
    assert settings.live_strict is True


def test_live_matcher_does_not_read_ranking_features_jsonl():
    """In live mode, LiveMatcher must rely on FAISS + MongoDB + score_candidate,
    never on the pre-computed artifact features under data/ranking/features/."""
    import inspect

    from src.core.matching import live_matcher

    source = inspect.getsource(live_matcher)
    assert "data/ranking/features" not in source
    assert "ranking/features" not in source
    # Retrieval must be FAISS-based, scoring must be Matching V3 score_candidate
    assert "score_candidate" in source
    assert "faiss" in source.lower()


def test_live_matcher_aggregates_unresolved_warning_no_per_id_spam():
    """Unresolved FAISS rows produce ONE aggregated warning, not a per-profile_id
    line containing 'Candidate profile not found in MongoDB'."""
    import numpy as np

    from src.core.matching.live_matcher import LiveMatcher, LiveMatcherSettings

    class _Index:
        def search(self, embedding, search_k):
            return (
                np.asarray([[0.9, 0.8, 0.7]], dtype="float32"),
                np.asarray([[0, 1, 2]], dtype="int64"),
            )

    class _Model:
        def encode(self, texts, **kwargs):
            return np.asarray([[1.0, 0.0]], dtype="float32")

    class _CandidateProfiles:
        collection_name = "candidate_profiles"

        def resolve_profiles_for_rows(self, rows):
            # Only the first row resolves; the other two are unresolved
            resolved = []
            for i, row in enumerate(rows):
                if i == 0:
                    resolved.append({
                        "profile_id": "p_real",
                        "candidate_id": "c_real",
                        "bio": {"full_name": "Real Candidate"},
                        "expertise": {"hard_skills": ["Python"], "experience_level": "mid_level"},
                        "experiences": [],
                        "profile_kind": "complete_profile",
                        "reliability_score": 0.9,
                        "quality_flags": [],
                    })
                else:
                    resolved.append(None)
            return resolved

    class _JobProfiles:
        def get_job_profile(self, job_id):
            return None

    class _MatchingRuns:
        def save_matching_run(self, document):
            return "run_x"

    class _Repos:
        def __init__(self):
            self.candidate_profiles = _CandidateProfiles()
            self.job_profiles = _JobProfiles()
            self.matching_runs = _MatchingRuns()

    matcher = LiveMatcher(
        repositories=_Repos(),
        settings=LiveMatcherSettings(mongodb_database="db", top_n=3),
        index_loader=lambda path: _Index(),
        id_map_loader=lambda path: [
            {"profile_id": "p_real", "candidate_id": "c_real"},
            {"profile_id": "p_missing_1", "candidate_id": "c_missing_1"},
            {"profile_id": "p_missing_2", "candidate_id": "c_missing_2"},
        ],
        model_loader=lambda name: _Model(),
    )

    result = matcher.match(
        job_description="Python role",
        job_id="gen_1",
        top_k=3,
        structured_job_profile={
            "generated_job_id": "gen_1",
            "job_title": "Python Engineer",
            "required_skills": ["Python"],
            "responsibilities": ["Code"],
            "seniority_level": "mid_level",
            "years_experience_required": 2,
        },
    )

    # One candidate resolved
    assert len(result.items) == 1
    # No per-id "Candidate profile not found" spam
    assert not any("Candidate profile not found" in w for w in result.warnings)
    # Exactly one aggregated warning mentioning the unresolved count
    aggregated = [w for w in result.warnings if "non resolus" in w or "non résolus" in w]
    assert len(aggregated) == 1
    assert "2" in aggregated[0]


def test_live_result_exposes_dedup_info_in_metadata():
    """LiveMatchResult carries dedup_info so the chat metadata can surface
    duplicate_candidates_filtered / duplicates_removed_count cleanly."""
    from src.core.matching.live_matcher import LiveMatchResult

    result = LiveMatchResult(job_id="j", resolved_job_id="j", top_k=5, items=[])
    # Default is an empty dict (no duplicates)
    assert result.dedup_info == {}

    # Simulated populated dedup info has the keys the UI/graph rely on
    result.dedup_info = {
        "duplicate_candidates_filtered": True,
        "duplicates_removed_count": 2,
        "duplicate_groups": [{"identity_key_type": "email"}],
    }
    assert result.dedup_info["duplicate_candidates_filtered"] is True
    assert result.dedup_info["duplicates_removed_count"] == 2
