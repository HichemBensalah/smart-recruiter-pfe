"""Tests for runtime store: current_job_profile.json and current_matching_run.json."""

import tempfile
from pathlib import Path

import pytest

from src.core.chatbot.runtime_store import (
    CURRENT_JOB_PROFILE_PATH,
    CURRENT_MATCHING_RUN_PATH,
    clear_all_runtime,
    clear_current_job_profile,
    clear_current_matching_run,
    ensure_runtime_dir,
    load_current_job_profile,
    load_current_matching_run,
    save_current_job_profile,
    save_current_matching_run,
)


@pytest.fixture(autouse=True)
def cleanup_runtime():
    """Clean up runtime files before and after each test."""
    clear_all_runtime()
    yield
    clear_all_runtime()


def test_runtime_dir_created():
    """ensure_runtime_dir() creates data/runtime/ if needed."""
    runtime_dir = ensure_runtime_dir()
    assert runtime_dir.exists()
    assert runtime_dir.is_dir()


def test_save_and_load_current_job_profile():
    """Save and load current_job_profile.json."""
    profile = {
        "job_title": "Senior Python",
        "target_role": "backend",
        "required_skills": ["Python", "FastAPI"],
    }

    saved_path = save_current_job_profile(profile)
    assert saved_path.exists()
    assert saved_path == CURRENT_JOB_PROFILE_PATH

    loaded = load_current_job_profile()
    assert loaded == profile
    assert loaded["job_title"] == "Senior Python"


def test_save_current_job_profile_overwrites():
    """Saving a new profile overwrites the old one."""
    profile1 = {"job_title": "Backend", "target_role": "backend"}
    profile2 = {"job_title": "ML Engineer", "target_role": "ml"}

    save_current_job_profile(profile1)
    loaded1 = load_current_job_profile()
    assert loaded1["job_title"] == "Backend"

    save_current_job_profile(profile2)
    loaded2 = load_current_job_profile()
    assert loaded2["job_title"] == "ML Engineer"
    assert loaded2 != loaded1


def test_clear_current_job_profile():
    """clear_current_job_profile() removes the file."""
    profile = {"job_title": "Test"}
    save_current_job_profile(profile)
    assert CURRENT_JOB_PROFILE_PATH.exists()

    clear_current_job_profile()
    assert not CURRENT_JOB_PROFILE_PATH.exists()
    assert load_current_job_profile() is None


def test_save_and_load_current_matching_run():
    """Save and load current_matching_run.json."""
    matching_result = {
        "candidates": [
            {"candidate_id": "cand_1", "name": "Alice"},
            {"candidate_id": "cand_2", "name": "Bob"},
        ],
        "decision_cards": [{"card": "data"}],
        "transferability": {"cand_1": {"score": 0.85}},
        "matching_metadata": {"matching_mode": "live"},
        "sources": ["live_mongodb_faiss"],
        "warnings": [],
        "job_description": "Find engineers",
    }

    saved_path = save_current_matching_run(matching_result, job_id="job_123")
    assert saved_path.exists()
    assert saved_path == CURRENT_MATCHING_RUN_PATH

    loaded = load_current_matching_run()
    assert loaded is not None
    assert len(loaded["candidates"]) == 2
    assert loaded["candidates"][0]["candidate_id"] == "cand_1"


def test_save_current_matching_run_overwrites():
    """Saving a new matching run overwrites the old one."""
    result1 = {
        "candidates": [{"candidate_id": "backend_1"}],
        "decision_cards": [],
        "transferability": {},
        "matching_metadata": {},
        "sources": [],
        "warnings": [],
    }
    result2 = {
        "candidates": [{"candidate_id": "ml_1"}],
        "decision_cards": [],
        "transferability": {},
        "matching_metadata": {},
        "sources": [],
        "warnings": [],
    }

    save_current_matching_run(result1)
    loaded1 = load_current_matching_run()
    assert loaded1["candidates"][0]["candidate_id"] == "backend_1"

    save_current_matching_run(result2)
    loaded2 = load_current_matching_run()
    assert loaded2["candidates"][0]["candidate_id"] == "ml_1"
    assert loaded2 != loaded1


def test_clear_current_matching_run():
    """clear_current_matching_run() removes the file."""
    matching_result = {"candidates": [{"id": "test"}], "decision_cards": [], "transferability": {}, "matching_metadata": {}, "sources": [], "warnings": []}
    save_current_matching_run(matching_result)
    assert CURRENT_MATCHING_RUN_PATH.exists()

    clear_current_matching_run()
    assert not CURRENT_MATCHING_RUN_PATH.exists()
    assert load_current_matching_run() is None


def test_clear_all_runtime():
    """clear_all_runtime() removes both files."""
    profile = {"job_title": "Test"}
    result = {"candidates": [], "decision_cards": [], "transferability": {}, "matching_metadata": {}, "sources": [], "warnings": []}

    save_current_job_profile(profile)
    save_current_matching_run(result)
    assert CURRENT_JOB_PROFILE_PATH.exists()
    assert CURRENT_MATCHING_RUN_PATH.exists()

    clear_all_runtime()
    assert not CURRENT_JOB_PROFILE_PATH.exists()
    assert not CURRENT_MATCHING_RUN_PATH.exists()
    assert load_current_job_profile() is None
    assert load_current_matching_run() is None


def test_load_nonexistent_files_return_none():
    """Loading from nonexistent files returns None."""
    clear_all_runtime()
    assert load_current_job_profile() is None
    assert load_current_matching_run() is None


def test_current_matching_run_holds_only_last_matching_candidates():
    """current_matching_run.json must reflect ONLY the latest matching's candidates,
    never a mix with a previous offer's candidates."""
    # First matching (Backend)
    backend_result = {
        "candidates": [
            {"candidate_id": "backend_1", "name": "Alice"},
            {"candidate_id": "backend_2", "name": "Bob"},
        ],
        "decision_cards": [],
        "transferability": {},
        "matching_metadata": {"matching_mode_used": "live", "generated_job_id": "gen_backend"},
        "sources": ["live_mongodb_faiss_matching_v3"],
        "warnings": [],
        "job_description": "Backend Python",
    }
    save_current_matching_run(backend_result, job_id="gen_backend")

    # Second matching (ML) overwrites
    ml_result = {
        "candidates": [
            {"candidate_id": "ml_1", "name": "Charlie"},
            {"candidate_id": "ml_2", "name": "Diana"},
        ],
        "decision_cards": [],
        "transferability": {},
        "matching_metadata": {"matching_mode_used": "live", "generated_job_id": "gen_ml"},
        "sources": ["live_mongodb_faiss_matching_v3"],
        "warnings": [],
        "job_description": "Machine Learning Engineer",
    }
    save_current_matching_run(ml_result, job_id="gen_ml")

    loaded = load_current_matching_run()
    assert loaded is not None
    assert loaded["job_id"] == "gen_ml"
    candidate_ids = {c["candidate_id"] for c in loaded["candidates"]}
    assert candidate_ids == {"ml_1", "ml_2"}
    # No bleed from the Backend matching
    assert "backend_1" not in str(loaded["candidates"])
    assert "backend_2" not in str(loaded["candidates"])
