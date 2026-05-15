from __future__ import annotations

import re
from typing import Iterable


_ALIASES: dict[str, str] = {
    "fast api": "FastAPI",
    "fastapi": "FastAPI",
    "fast api framework": "FastAPI",
    "rest api": "REST API",
    "rest api design": "REST API",
    "restful api": "REST API",
    "restful apis": "REST API",
    "mongo db": "MongoDB",
    "mongodb": "MongoDB",
    "mongo": "MongoDB",
    "ci cd": "CI/CD",
    "cicd": "CI/CD",
    "ci/cd": "CI/CD",
    "git hub": "GitHub",
    "github": "GitHub",
    "java script": "JavaScript",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "type script": "TypeScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "reactjs": "React",
    "react js": "React",
    "react.js": "React",
    "react": "React",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "node.js": "Node.js",
    "node": "Node.js",
    "postgre sql": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "postgre": "PostgreSQL",
    "my sql": "MySQL",
    "mysql": "MySQL",
    "py torch": "PyTorch",
    "pytorch": "PyTorch",
    "tensor flow": "TensorFlow",
    "tensorflow": "TensorFlow",
    "num py": "NumPy",
    "numpy": "NumPy",
    "ml ops": "MLOps",
    "mlops": "MLOps",
    "m lflow": "MLflow",
    "mlflow": "MLflow",
    "power bi": "PowerBI",
    "powerbi": "PowerBI",
    "po wer bi": "PowerBI",
    "xg boost": "XGBoost",
    "xgboost": "XGBoost",
    "no sql": "NoSQL",
    "nosql": "NoSQL",
}


def _clean_skill_token(value: str) -> str:
    lowered = value.strip().lower()
    lowered = lowered.replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[./]+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9+#]+", " ", lowered)
    return " ".join(lowered.split())


def _title_case_fallback(cleaned: str) -> str:
    if not cleaned:
        return ""
    return " ".join(part.upper() if len(part) <= 3 and part.isalpha() else part.capitalize() for part in cleaned.split())


def normalize_skill(skill: str) -> str:
    if not isinstance(skill, str):
        return ""
    cleaned = _clean_skill_token(skill)
    if not cleaned:
        return ""
    canonical = _ALIASES.get(cleaned)
    if canonical:
        return canonical
    return _title_case_fallback(cleaned)


def normalize_skills(skills: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for skill in skills or []:
        canonical = normalize_skill(skill)
        if not canonical:
            continue
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(canonical)
    return normalized


def skills_overlap(job_skills: list[str], candidate_skills: list[str]) -> tuple[list[str], list[str]]:
    normalized_job = normalize_skills(job_skills)
    normalized_candidate = normalize_skills(candidate_skills)
    candidate_keys = {skill.lower() for skill in normalized_candidate}

    matched: list[str] = []
    missing: list[str] = []
    for skill in normalized_job:
        if skill.lower() in candidate_keys:
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing


def flatten_skill_sources(*skill_lists: Iterable[str]) -> list[str]:
    flattened: list[str] = []
    for values in skill_lists:
        for value in values or []:
            if isinstance(value, str):
                flattened.append(value)
    return normalize_skills(flattened)
