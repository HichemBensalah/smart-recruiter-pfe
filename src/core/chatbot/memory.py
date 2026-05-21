from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any
from uuid import uuid4


MAX_CONVERSATION_TURNS = 5
MAX_CONVERSATION_MESSAGES = MAX_CONVERSATION_TURNS * 2
DEFAULT_SESSION_TTL_SECONDS = 30 * 60


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class ConversationMemory:
    session_id: str
    mode: str | None = None
    job_intake: dict[str, Any] | None = None
    pending_confirmation: str | None = None
    offer_created: bool = False
    matching_completed: bool = False
    current_job_profile: dict[str, Any] | None = None
    routed_job_id: str | None = None
    job_description: str | None = None
    last_job_query: str | None = None
    job_intake_state: dict[str, Any] | None = None
    pending_field_edit: str | None = None
    awaiting_field_replacement: bool = False
    messages: list[ConversationTurn] = field(default_factory=list)
    last_user_message: str | None = None
    last_answer: str | None = None
    last_candidates: list[dict[str, Any]] = field(default_factory=list)
    last_decision_cards: list[dict[str, Any]] = field(default_factory=list)
    last_transferability: dict[str, Any] = field(default_factory=dict)
    selected_candidate_id: str | None = None
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)

    @property
    def conversation_history(self) -> list[ConversationTurn]:
        return self.messages


class InMemorySessionStore:
    def __init__(self, ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS) -> None:
        self._sessions: dict[str, ConversationMemory] = {}
        self.ttl_seconds = ttl_seconds

    def get_or_create(self, session_id: str | None = None) -> ConversationMemory:
        resolved_session_id = session_id or str(uuid4())
        if self._is_expired(resolved_session_id):
            self.clear(resolved_session_id)
        if resolved_session_id not in self._sessions:
            self._sessions[resolved_session_id] = ConversationMemory(session_id=resolved_session_id)
        return self._sessions[resolved_session_id]

    def get(self, session_id: str) -> ConversationMemory | None:
        if self._is_expired(session_id):
            self.clear(session_id)
            return None
        return self._sessions.get(session_id)

    def update(self, session_id: str, result: dict[str, Any]) -> ConversationMemory:
        memory = self.get_or_create(session_id)
        memory.updated_at = time()
        user_message = str(result.get("user_message") or "")
        answer = str(result.get("answer") or "")
        if user_message:
            memory.last_user_message = user_message
            memory.messages.append(ConversationTurn(role="user", content=user_message))
        if answer:
            memory.last_answer = answer
            memory.messages.append(ConversationTurn(role="assistant", content=answer))
        _trim_conversation_history(memory)
        memory.last_candidates = _list_of_dicts(result.get("candidates"))
        memory.last_decision_cards = _list_of_dicts(result.get("decision_cards"))
        memory.last_transferability = result.get("transferability") if isinstance(result.get("transferability"), dict) else {}
        if isinstance(result.get("job_intake_state"), dict):
            memory.job_intake_state = result.get("job_intake_state")
        if isinstance(result.get("structured_job_profile"), dict):
            memory.current_job_profile = result.get("structured_job_profile")
            memory.offer_created = True
        if result.get("routed_job_id"):
            memory.routed_job_id = str(result.get("routed_job_id"))
        if result.get("job_description"):
            memory.job_description = str(result.get("job_description"))
            memory.last_job_query = str(result.get("job_description"))
        elif result.get("matching_completed") and user_message:
            memory.last_job_query = user_message
        if result.get("matching_completed") is not None:
            memory.matching_completed = bool(result.get("matching_completed"))
        selected_candidate_id = result.get("selected_candidate_id") or _first_candidate_id(memory.last_candidates)
        memory.selected_candidate_id = str(selected_candidate_id) if selected_candidate_id else memory.selected_candidate_id
        return memory

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _is_expired(self, session_id: str) -> bool:
        memory = self._sessions.get(session_id)
        if memory is None:
            return False
        return (time() - memory.updated_at) > self.ttl_seconds


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _first_candidate_id(candidates: list[dict[str, Any]]) -> str | None:
    if not candidates:
        return None
    candidate_id = candidates[0].get("candidate_id")
    return str(candidate_id) if candidate_id else None


def _trim_conversation_history(memory: ConversationMemory) -> None:
    if len(memory.messages) > MAX_CONVERSATION_MESSAGES:
        memory.messages = memory.messages[-MAX_CONVERSATION_MESSAGES:]


SESSION_STORE = InMemorySessionStore()
