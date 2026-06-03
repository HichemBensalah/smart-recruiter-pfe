from __future__ import annotations

from pathlib import Path
from typing import Any


def infer_job_route_from_structured_profile(profile: dict[str, Any]) -> dict[str, Any]:
    skills = _normalized_skills(profile.get("required_skills", [])) | _normalized_skills(
        profile.get("nice_to_have_skills", [])
    )
    title_text = " ".join(
        [
            str(profile.get("job_title") or "").lower(),
            str(profile.get("target_role") or "").lower(),
        ]
    )

    if (
        "machine learning" in title_text
        or "ml engineer" in title_text
        or "deep learning" in title_text
        or "ai engineer" in title_text
        or "data science" in title_text
    ):
        return _route(
            "machine_learning_python_nlp",
            "Machine Learning Engineer",
            0.8,
            "Intention ML detectee dans le titre.",
        )
    if "data engineer" in title_text:
        return _route(
            "data_engineer_python_sql_etl_aligned",
            "Data Engineer",
            0.85,
            "Signaux Data Engineer Python/SQL/ETL detectes.",
        )
    if "data analyst" in title_text:
        return _route(
            "data_analyst_python_sql_powerbi",
            "Data Analyst",
            0.8,
            "Signaux Data Analyst / BI detectes.",
        )

    if skills & {
        "deep learning",
        "nlp",
        "tensorflow",
        "keras",
        "pytorch",
        "mlflow",
        "computer vision",
    }:
        return _route(
            "machine_learning_python_nlp",
            "Machine Learning Engineer",
            0.8,
            "Signaux ML/NLP detectes.",
        )
    if {"sql", "etl"}.issubset(skills) and "python" in skills:
        return _route(
            "data_engineer_python_sql_etl_aligned",
            "Data Engineer",
            0.85,
            "Signaux Data Engineer Python/SQL/ETL detectes.",
        )
    if "power bi" in skills or "powerbi" in skills:
        return _route(
            "data_analyst_python_sql_powerbi",
            "Data Analyst",
            0.8,
            "Signaux Data Analyst / BI detectes.",
        )

    if {"python", "fastapi", "mongodb"}.issubset(skills):
        job_id = _first_existing_job_id(
            ["backend_python_fastapi_mongodb_aligned", "backend_python_fastapi_mongodb"]
        )
        return _route(
            job_id,
            "Backend Developer",
            0.95,
            "Python + FastAPI + MongoDB detectes.",
        )
    if "django" in skills and "postgresql" in skills:
        return _route(
            "backend_python_django_postgresql",
            "Backend Developer",
            0.9,
            "Django et PostgreSQL detectes.",
        )
    return _route(
        "backend_python_django_postgresql",
        "Backend Developer",
        0.45,
        "Fallback vers l'offre backend de demo.",
    )


def _route(job_id: str, target_role: str, confidence: float, reason: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "target_role": target_role,
        "route_confidence": confidence,
        "route_reason": reason,
    }


def _normalized_skills(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item).strip().lower() for item in value if str(item).strip()}


def _first_existing_job_id(job_ids: list[str]) -> str:
    for job_id in job_ids:
        if Path("data/job_profiles", f"{job_id}.json").exists() or Path(
            "data/job_descriptions", f"{job_id}.txt"
        ).exists():
            return job_id
    return job_ids[0]
