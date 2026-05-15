from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB_PROFILES_DIR = ROOT / "data/job_profiles"
ALIGNED_JOB_IDS = {
    "data_analyst_python_sql_powerbi",
    "machine_learning_python_nlp",
    "backend_python_django_postgresql",
    "data_engineer_python_sql_etl_aligned",
    "backend_python_fastapi_mongodb_aligned",
}
CANONICAL_SENIORITY = {"junior", "mid_level", "senior", "lead", "principal"}


def load_job(job_id: str) -> dict:
    return json.loads((JOB_PROFILES_DIR / f"{job_id}.json").read_text(encoding="utf-8"))


def test_aligned_job_profiles_exist() -> None:
    for job_id in ALIGNED_JOB_IDS:
        assert (JOB_PROFILES_DIR / f"{job_id}.json").exists()


def test_aligned_job_profiles_contract() -> None:
    for job_id in ALIGNED_JOB_IDS:
        job = load_job(job_id)
        assert job["job_id"] == job_id
        assert job["job_title"]
        assert job["domain"]
        assert job["seniority_level"] in CANONICAL_SENIORITY
        assert job["required_skills"]
        assert "corpus_alignment_note" in job


def test_old_job_profiles_are_kept() -> None:
    assert (JOB_PROFILES_DIR / "frontend_react_nextjs.json").exists()
    assert (JOB_PROFILES_DIR / "fullstack_react_node_mongodb.json").exists()
    assert (JOB_PROFILES_DIR / "devops_docker_kubernetes.json").exists()
