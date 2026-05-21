from __future__ import annotations

import os

from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.api.utils import DEMO_RUN_MANIFEST, MATCHING_FEATURES_DIR, list_matching_artifacts, load_best_decision_cards
from src.core.graph.neo4j_client import neo4j_status


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    warnings: list[str] = []
    matching_artifacts = _matching_artifacts_status(warnings)
    dependencies = {
        "matching_artifacts": matching_artifacts,
        "decision_cards": _decision_cards_available(warnings),
        "demo_manifest": DEMO_RUN_MANIFEST.exists(),
        "neo4j": _neo4j_dependency_status(warnings),
        "mongodb_configured": bool(os.getenv("MONGODB_URI")),
    }
    return HealthResponse(status="ok", service="smart-recruiter", version="demo", dependencies=dependencies, warnings=warnings)


def _matching_artifacts_status(warnings: list[str]) -> dict[str, object]:
    try:
        artifacts = list_matching_artifacts()
    except Exception as exc:
        warnings.append(f"Matching artifacts cannot be listed: {exc}")
        return {"available": False, "count": 0, "directory": MATCHING_FEATURES_DIR.as_posix()}
    if not artifacts:
        warnings.append("No Matching V3 feature artifact was found.")
    return {
        "available": bool(artifacts),
        "count": len(artifacts),
        "directory": MATCHING_FEATURES_DIR.as_posix(),
        "job_ids": sorted(artifacts),
    }


def _decision_cards_available(warnings: list[str]) -> bool:
    try:
        load_best_decision_cards()
        return True
    except Exception as exc:
        warnings.append(f"Decision cards unavailable: {exc}")
        return False


def _neo4j_dependency_status(warnings: list[str]) -> dict[str, object]:
    try:
        return neo4j_status()
    except Exception as exc:
        warnings.append(f"Neo4j status check failed without blocking /health: {exc}")
        return {"neo4j_available": False, "message": str(exc)}
