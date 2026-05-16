from __future__ import annotations

from typing import Any

from src.core.chatbot.state import RecruiterCopilotState
from src.core.chatbot.tools.match_tools import match_candidates_tool


def match_candidates_node(state: RecruiterCopilotState) -> RecruiterCopilotState:
    warnings = list(state.get("warnings", []))
    sources = list(state.get("sources", []))
    job_description = str(state.get("job_description") or state.get("user_message") or "")
    top_k = int(state.get("top_k") or 5)
    try:
        result = match_candidates_tool.invoke({"job_description": job_description, "top_k": top_k})
        candidates = _extract_candidates(result)
        if "match_candidates" not in sources:
            sources.append("match_candidates")
        return {"candidates": candidates, "sources": sources, "warnings": warnings}
    except Exception as exc:
        warnings.append(f"match_candidates failed: {exc}")
        return {"candidates": [], "warnings": warnings, "sources": sources}


def _extract_candidates(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        items = result.get("items") or result.get("candidates") or []
        return [item for item in items if isinstance(item, dict)]
    return []
