from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends

from src.api.auth import require_api_key
from src.api.schemas import ChatRequest, ChatResponse
from src.core.chatbot.graph import run_recruiter_copilot_with_memory
from src.core.chatbot.memory import SESSION_STORE


router = APIRouter(prefix="/api/chat", tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = run_recruiter_copilot_with_memory(request.message, request.session_id)
    except Exception as exc:
        session_id = request.session_id or str(uuid4())
        warning = f"Recruiter Copilot failed; degraded response returned: {exc}"
        result = {
            "session_id": session_id,
            "answer": (
                "Le Copilot a rencontre une erreur interne pendant le traitement. "
                "La session reste disponible, mais aucun matching n'a ete relance."
            ),
            "candidates": [],
            "decision_cards": [],
            "transferability": {},
            "sources": ["api_chat_degraded"],
            "warnings": [warning],
            "user_message": request.message,
            "matching_metadata": {},
            "matching_completed": False,
        }
        try:
            SESSION_STORE.update(session_id, result)
        except Exception as memory_exc:
            result["warnings"].append(f"Conversation memory could not be updated: {memory_exc}")

    return _chat_response_from_result(result, request.session_id)


def _chat_response_from_result(result: dict, request_session_id: str | None) -> ChatResponse:
    return ChatResponse(
        session_id=str(result.get("session_id")) if result.get("session_id") else request_session_id,
        answer=str(result.get("answer") or ""),
        candidates=_as_list_of_dicts(result.get("candidates")),
        decision_cards=_as_list_of_dicts(result.get("decision_cards")),
        transferability=result.get("transferability") if isinstance(result.get("transferability"), dict) else {},
        sources=[str(source) for source in result.get("sources", [])] if isinstance(result.get("sources"), list) else [],
        warnings=[str(warning) for warning in result.get("warnings", [])] if isinstance(result.get("warnings"), list) else [],
        selected_candidate_id=str(result.get("selected_candidate_id")) if result.get("selected_candidate_id") else None,
        job_intake_state=result.get("job_intake_state") if isinstance(result.get("job_intake_state"), dict) else None,
        structured_job_profile=result.get("structured_job_profile") if isinstance(result.get("structured_job_profile"), dict) else None,
        routed_job_id=str(result.get("routed_job_id")) if result.get("routed_job_id") else None,
        matching_metadata=result.get("matching_metadata") if isinstance(result.get("matching_metadata"), dict) else {},
        matching_completed=bool(result.get("matching_completed", False)),
        pending_field_edit=str(result.get("pending_field_edit")) if result.get("pending_field_edit") else None,
        awaiting_field_replacement=bool(result.get("awaiting_field_replacement", False)),
    )


def _as_list_of_dicts(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
