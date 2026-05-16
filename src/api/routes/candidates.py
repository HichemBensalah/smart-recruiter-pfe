from __future__ import annotations

from fastapi import APIRouter, Query

from src.api.schemas import CandidateListItem, PaginatedCandidates
from src.api.utils import compact_candidate, find_candidate, get_candidates, load_best_decision_cards, read_profile_from_card


router = APIRouter(prefix="/api/candidates", tags=["candidates"])


@router.get("", response_model=PaginatedCandidates)
def list_candidates(limit: int = Query(default=20, ge=1, le=100), offset: int = Query(default=0, ge=0)) -> PaginatedCandidates:
    payload = load_best_decision_cards()
    candidates = get_candidates(payload)
    page = candidates[offset : offset + limit]
    return PaginatedCandidates(
        total=len(candidates),
        limit=limit,
        offset=offset,
        items=[CandidateListItem(**compact_candidate(candidate)) for candidate in page],
    )


@router.get("/{candidate_id}")
def get_candidate(candidate_id: str) -> dict:
    candidate = find_candidate(candidate_id)
    return {
        "candidate": candidate,
        "profile": read_profile_from_card(candidate),
    }
