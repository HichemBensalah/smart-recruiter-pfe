from __future__ import annotations

from fastapi import APIRouter

from src.api.utils import find_candidate, load_best_decision_cards


router = APIRouter(prefix="/api/decision-cards", tags=["decision-cards"])


@router.get("")
def list_decision_cards() -> dict:
    return load_best_decision_cards()


@router.get("/{candidate_id}")
def get_decision_card(candidate_id: str) -> dict:
    return find_candidate(candidate_id)
