from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.utils import find_candidate


router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/transferability/{candidate_id}")
def get_transferability(candidate_id: str) -> dict:
    candidate = find_candidate(candidate_id)
    transferability = candidate.get("transferability")
    if not isinstance(transferability, dict):
        raise HTTPException(status_code=404, detail=f"Transferability not found for candidate: {candidate_id}")
    return {
        "candidate_id": candidate.get("candidate_id"),
        "profile_id": candidate.get("profile_id"),
        "baseline_rank_v3": candidate.get("baseline_rank_v3"),
        "baseline_score_v3": candidate.get("baseline_score_v3"),
        "transferability": transferability,
    }
