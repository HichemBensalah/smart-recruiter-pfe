from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.auth import require_api_key
from src.api.schemas import TransferabilityResponse
from src.api.utils import find_candidate
from src.core.graph.neo4j_client import Neo4jUnavailable
from src.core.graph.neo4j_transferability import explain_transferability


router = APIRouter(prefix="/api/graph", tags=["graph"], dependencies=[Depends(require_api_key)])


@router.get("/transferability/{candidate_id}", response_model=TransferabilityResponse)
def get_transferability(candidate_id: str, target_role: str = Query(default="Backend Developer", min_length=1)) -> dict:
    warnings: list[str] = []
    candidate = find_candidate(candidate_id)
    try:
        neo4j_payload = explain_transferability(candidate_id, target_role)
        return {
            "candidate_id": candidate_id,
            "profile_id": candidate.get("profile_id"),
            "baseline_rank_v3": candidate.get("baseline_rank_v3"),
            "baseline_score_v3": candidate.get("baseline_score_v3"),
            "source": "neo4j",
            "fallback_used": False,
            "warnings": warnings,
            "transferability": _normalize_transferability_payload(neo4j_payload),
        }
    except Neo4jUnavailable as exc:
        warnings.append(f"Neo4j unavailable, YAML fallback used: {exc}")
    except Exception as exc:
        warnings.append(f"Neo4j transferability failed, YAML fallback used: {exc}")

    transferability = candidate.get("transferability")
    if not isinstance(transferability, dict):
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Transferability not found for candidate: {candidate_id}",
                "source": "yaml_fallback",
                "fallback_used": True,
                "warnings": warnings,
            },
        )
    return {
        "candidate_id": candidate.get("candidate_id"),
        "profile_id": candidate.get("profile_id"),
        "baseline_rank_v3": candidate.get("baseline_rank_v3"),
        "baseline_score_v3": candidate.get("baseline_score_v3"),
        "source": "yaml_fallback",
        "fallback_used": True,
        "warnings": warnings,
        "transferability": _normalize_transferability_payload(transferability),
    }


def _normalize_transferability_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    score = normalized.get("transferability_score")
    if score is None:
        score = normalized.get("coverage_score")
    try:
        numeric_score = round(float(score), 4)
    except (TypeError, ValueError):
        numeric_score = 0.0
    normalized.setdefault("transferability_score", numeric_score)
    normalized.setdefault("fit_direct", numeric_score >= 0.7)
    normalized.setdefault("gaps_compensables", normalized.get("adjacent_skills", []))
    normalized.setdefault("gaps_bloquants", normalized.get("missing_skills", []))
    return normalized
