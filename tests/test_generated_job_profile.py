"""Generated job profile creation and persistence tests."""

import json
import tempfile
from pathlib import Path

import pytest

from src.core.chatbot.generated_job_profile import (
    build_generated_job_profile,
    save_generated_job_profile,
    load_generated_job_profile,
)


@pytest.fixture
def temp_dir():
    """Temporary directory for profile storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_structured_profile():
    """Sample structured profile from job_intake."""
    return {
        "job_title": "Senior Python Engineer",
        "target_role": "backend",
        "about_role": "Build scalable APIs",
        "responsibilities": ["Design systems", "Code review"],
        "required_skills": ["Python", "FastAPI"],
        "nice_to_have_skills": ["Docker", "Kubernetes"],
        "min_years_experience": 5,
        "seniority": "senior",
        "location": "Remote",
        "work_model": "remote",
        "language_requirements": ["English"],
    }


def test_build_generated_job_profile(sample_structured_profile):
    """Build a generated profile from structured input."""
    profile = build_generated_job_profile(
        sample_structured_profile,
        session_id="session_123",
        routed_base_job_id="job_456",
    )

    assert "generated_job_id" in profile
    assert profile["generated_job_id"].startswith("generated_")
    assert profile["job_title"] == "Senior Python Engineer"
    assert profile["target_role"] == "backend"
    assert profile["required_skills"] == ["Python", "FastAPI"]
    assert profile["min_years_experience"] == 5
    assert profile["years_experience_required"] == 5
    assert profile["seniority"] == "senior"
    assert profile["seniority_level"] == "senior"
    assert profile["source"] == "job_intake_wizard"
    assert profile["session_id"] == "session_123"
    assert profile["routed_base_job_id"] == "job_456"
    assert "created_at" in profile


def test_build_generated_job_profile_missing_fields(sample_structured_profile):
    """Handle missing optional fields gracefully."""
    minimal = {"job_title": "Engineer"}
    profile = build_generated_job_profile(minimal)

    assert profile["job_title"] == "Engineer"
    assert profile["required_skills"] == []
    assert profile["responsibilities"] == []
    assert profile["nice_to_have_skills"] == []
    assert profile["min_years_experience"] is None


def test_save_and_load_generated_job_profile(sample_structured_profile, temp_dir):
    """Save a profile and reload it."""
    profile = build_generated_job_profile(
        sample_structured_profile,
        session_id="test_session",
    )
    generated_id = profile["generated_job_id"]

    saved_path = save_generated_job_profile(profile, base_dir=temp_dir)
    assert saved_path.exists()
    assert saved_path.name == f"{generated_id}.json"

    loaded = load_generated_job_profile(generated_id, base_dir=temp_dir)
    assert loaded is not None
    assert loaded["generated_job_id"] == generated_id
    assert loaded["job_title"] == "Senior Python Engineer"
    assert loaded["session_id"] == "test_session"


def test_load_nonexistent_profile(temp_dir):
    """Load a profile that does not exist."""
    result = load_generated_job_profile("nonexistent_id", base_dir=temp_dir)
    assert result is None


def test_profile_json_format(sample_structured_profile, temp_dir):
    """Saved profile is valid JSON."""
    profile = build_generated_job_profile(sample_structured_profile)
    saved_path = save_generated_job_profile(profile, base_dir=temp_dir)

    content = saved_path.read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert parsed["job_title"] == sample_structured_profile["job_title"]
