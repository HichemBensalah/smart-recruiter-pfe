from __future__ import annotations

import re
import unicodedata

from src.core.chatbot.memory import ConversationMemory


def resolve_candidate_reference(message: str, memory: ConversationMemory) -> str | None:
    lowered = _normalize_text(message)
    explicit = re.search(r"candidate_[A-Za-z0-9_]+", message)
    if explicit:
        return explicit.group(0)
    if "premier candidat" in lowered or "1er candidat" in lowered or "meilleur candidat" in lowered:
        return _candidate_id_at(memory, 0)
    if "deuxieme candidat" in lowered or "second candidat" in lowered:
        return _candidate_id_at(memory, 1)
    if "dernier candidat" in lowered:
        return _candidate_id_at(memory, len(memory.last_candidates) - 1)
    if "ce candidat" in lowered or "candidat selectionne" in lowered:
        return memory.selected_candidate_id or _candidate_id_at(memory, 0)
    if "moins bien classe" in lowered:
        return memory.selected_candidate_id or _candidate_id_at(memory, 1) or _candidate_id_at(memory, 0)
    if "lui" in lowered or "son " in lowered or "ses " in lowered:
        return memory.selected_candidate_id or _candidate_id_at(memory, 0)
    return None


def _candidate_id_at(memory: ConversationMemory, index: int) -> str | None:
    if index < 0 or index >= len(memory.last_candidates):
        return None
    candidate_id = memory.last_candidates[index].get("candidate_id")
    return str(candidate_id) if candidate_id else None


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))
