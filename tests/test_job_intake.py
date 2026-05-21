from __future__ import annotations

from src.core.chatbot.job_intake import (
    build_job_description,
    build_structured_job_profile,
    extract_min_years_experience,
    extract_seniority,
    extract_work_model,
    parse_skills,
    start_job_intake,
    update_job_intake,
)
from src.core.chatbot.job_router import infer_job_route_from_structured_profile
from src.core.chatbot.memory import ConversationMemory


def test_start_job_intake_initializes_mode() -> None:
    memory = ConversationMemory(session_id="session-1")

    intake = start_job_intake(memory)

    assert memory.mode == "job_creation"
    assert intake["current_step"] == "job_title"
    assert intake["fields"]["job_title"] == ""


def test_update_job_intake_advances_steps() -> None:
    memory = ConversationMemory(session_id="session-1")
    start_job_intake(memory)

    update_job_intake(memory, "Backend Python Engineer")
    assert memory.job_intake["fields"]["job_title"] == "Backend Python Engineer"
    assert memory.job_intake["current_step"] == "about_role"

    update_job_intake(memory, "We are looking for a backend engineer.")
    assert memory.job_intake["fields"]["about_role"] == "We are looking for a backend engineer."
    assert memory.job_intake["current_step"] == "responsibilities"


def test_parse_skills_extracts_bullets() -> None:
    skills = parse_skills("- Python\n- FastAPI\n- MongoDB\nDocker, REST API design")

    assert skills == ["Python", "FastAPI", "MongoDB", "Docker", "REST API design"]


def test_profile_extractors() -> None:
    text = "We need a mid-level engineer with at least 3 years in hybrid mode in Tunis. English required."

    assert extract_min_years_experience(text) == 3
    assert extract_seniority(text) == "mid-level"
    assert extract_work_model(text) == "hybrid"


def test_build_job_description_contains_all_sections() -> None:
    memory = _complete_memory()

    description = build_job_description(memory)

    assert "Title" in description
    assert "About the role" in description
    assert "Responsibilities" in description
    assert "Required skills" in description
    assert "Bonus skills" in description
    assert "Profile" in description


def test_build_structured_job_profile_returns_required_skills() -> None:
    memory = _complete_memory()

    profile = build_structured_job_profile(memory)

    assert profile["job_title"] == "Backend Python Engineer"
    assert profile["required_skills"] == ["Python", "FastAPI", "MongoDB", "Docker", "REST API design"]
    assert profile["nice_to_have_skills"] == ["CI/CD", "AWS"]
    assert profile["min_years_experience"] == 3
    assert profile["seniority"] == "mid-level"
    assert profile["work_model"] == "hybrid"


def test_backend_fastapi_mongodb_routes_to_backend_job_id() -> None:
    memory = _complete_memory()
    profile = build_structured_job_profile(memory)

    route = infer_job_route_from_structured_profile(profile)

    assert route["job_id"] == "backend_python_fastapi_mongodb_aligned"
    assert route["target_role"] == "Backend Developer"


def _complete_memory() -> ConversationMemory:
    memory = ConversationMemory(session_id="session-1")
    start_job_intake(memory)
    update_job_intake(memory, "Backend Python Engineer")
    update_job_intake(memory, "We are looking for a backend engineer.")
    update_job_intake(memory, "You will design APIs and services.")
    update_job_intake(memory, "- Python\n- FastAPI\n- MongoDB\n- Docker\n- REST API design")
    update_job_intake(memory, "- CI/CD\n- AWS")
    update_job_intake(memory, "Mid-level profile with at least 3 years, English, Tunis, hybrid.")
    return memory
