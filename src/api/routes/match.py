from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import MatchCandidate, MatchRequest, MatchResponse
from src.api.utils import compact_candidate, get_candidates, load_best_decision_cards, sort_by_v3_rank


router = APIRouter(prefix="/api/match", tags=["matching"])


METHODOLOGICAL_NOTE = (
    "Matching V3 reste la baseline officielle. Cet endpoint expose les artefacts de demo deja generes "
    "et ne relance pas FAISS, MongoDB, Matching V3 ou un entrainement ML."
)


@router.post("", response_model=MatchResponse)
def match_candidates(request: MatchRequest) -> MatchResponse:
    payload = load_best_decision_cards()
    candidates = sort_by_v3_rank(get_candidates(payload))[: request.top_k]
    items = []
    for candidate in candidates:
        compact = compact_candidate(candidate)
        items.append(
            MatchCandidate(
                candidate_id=compact["candidate_id"],
                profile_id=compact["profile_id"],
                rank=compact["baseline_rank_v3"],
                baseline_rank_v3=compact["baseline_rank_v3"],
                baseline_score_v3=compact["baseline_score_v3"],
                rf_rank=compact["rf_rank"],
                rf_score=compact["rf_score"],
                xgboost_rank=compact["xgboost_rank"],
                xgboost_score=compact["xgboost_score"],
                recommendation_status=compact["recommendation_status"],
                transferability=candidate.get("transferability") if isinstance(candidate.get("transferability"), dict) else None,
            )
        )
    return MatchResponse(
        job_description=request.job_description,
        top_k=request.top_k,
        matching_mode="demo_artifact_matching_v3_baseline",
        methodological_note=METHODOLOGICAL_NOTE,
        items=items,
    )
