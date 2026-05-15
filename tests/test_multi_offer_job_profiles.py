from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB_PROFILES_DIR = ROOT / "data/job_profiles"
EXPECTED_JOB_IDS = {
    "backend_python_fastapi_mongodb",
    "frontend_react_nextjs",
    "fullstack_react_node_mongodb",
    "data_engineer_python_sql",
    "devops_docker_kubernetes",
}
VALID_SENIORITIES = {"junior", "mid_level", "senior", "lead", "principal"}


def load_profile(job_id: str) -> dict:
    return json.loads((JOB_PROFILES_DIR / f"{job_id}.json").read_text(encoding="utf-8"))


def test_five_job_profiles_exist() -> None:
    for job_id in EXPECTED_JOB_IDS:
        assert (JOB_PROFILES_DIR / f"{job_id}.json").exists()


def test_job_profiles_have_required_contract() -> None:
    for job_id in EXPECTED_JOB_IDS:
        profile = load_profile(job_id)
        assert profile["job_id"] == job_id
        assert profile["job_title"]
        assert profile["domain"]
        assert profile["seniority_level"] in VALID_SENIORITIES
        assert 4 <= len(profile["required_skills"]) <= 6
        assert profile["description"]
