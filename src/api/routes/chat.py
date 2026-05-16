from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import ChatRequest, ChatResponse
from src.core.chatbot.graph import run_recruiter_copilot


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = run_recruiter_copilot(request.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recruiter Copilot failed: {exc}") from exc

    return ChatResponse(
        answer=str(result.get("answer") or ""),
        candidates=_as_list_of_dicts(result.get("candidates")),
        decision_cards=_as_list_of_dicts(result.get("decision_cards")),
        transferability=result.get("transferability") if isinstance(result.get("transferability"), dict) else {},
        sources=[str(source) for source in result.get("sources", [])] if isinstance(result.get("sources"), list) else [],
        warnings=[str(warning) for warning in result.get("warnings", [])] if isinstance(result.get("warnings"), list) else [],
    )


def _as_list_of_dicts(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
