from __future__ import annotations

from typing import Any

from src.core.chatbot.state import RecruiterCopilotState
from src.core.chatbot.tools.match_tools import match_candidates_tool


FOLLOW_UP_INTENTS = {
    "explain_candidate",
    "review_needed",
    "gap_analysis",
    "compare_candidates",
    "transferability",
}


def match_candidates_node(state: RecruiterCopilotState) -> RecruiterCopilotState:
    warnings = list(state.get("warnings", []))
    sources = list(state.get("sources", []))
    if state.get("intent") in FOLLOW_UP_INTENTS and state.get("candidates"):
        if "conversation_memory" not in sources:
            sources.append("conversation_memory")
        return {"candidates": state.get("candidates", []), "sources": sources, "warnings": warnings}

    job_description = str(state.get("job_description") or state.get("user_message") or "")
    top_k = int(state.get("top_k") or 5)
    job_id = state.get("routed_job_id")
    try:
        result = match_candidates_tool.invoke({"job_description": job_description, "top_k": top_k, "job_id": job_id})
        candidates = _extract_candidates(result)
        if "match_candidates" not in sources:
            sources.append("match_candidates")
        update: RecruiterCopilotState = {"candidates": candidates, "sources": sources, "warnings": warnings}
        if isinstance(result, dict):
            update["routed_job_id"] = result.get("resolved_job_id") or result.get("job_id") or job_id
            result_warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
            warnings.extend(str(warning) for warning in result_warnings)
            update["warnings"] = warnings
            update["matching_metadata"] = {
                "job_id": result.get("job_id"),
                "resolved_job_id": result.get("resolved_job_id"),
                "artifact_source": result.get("artifact_source"),
                "matching_mode": result.get("matching_mode"),
                "fallback_used": bool(result.get("fallback_used", False)),
                "warnings": [str(warning) for warning in result_warnings],
            }
        return update
    except Exception as exc:
        warnings.append(f"match_candidates failed: {exc}")
        return {"candidates": [], "warnings": warnings, "sources": sources}


def _extract_candidates(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        items = result.get("items") or result.get("candidates") or []
        return [item for item in items if isinstance(item, dict)]
    return []
