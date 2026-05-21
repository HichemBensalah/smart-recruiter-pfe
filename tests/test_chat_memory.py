from __future__ import annotations

from src.core.chatbot.memory import InMemorySessionStore


def test_session_store_creates_session() -> None:
    store = InMemorySessionStore()

    memory = store.get_or_create()

    assert memory.session_id
    assert memory.mode is None
    assert memory.job_intake is None
    assert memory.pending_confirmation is None
    assert memory.pending_field_edit is None
    assert memory.awaiting_field_replacement is False
    assert memory.messages == []
    assert store.get(memory.session_id) is memory


def test_session_store_updates_last_artifacts() -> None:
    store = InMemorySessionStore()
    memory = store.get_or_create("session-1")

    updated = store.update(
        memory.session_id,
        {
            "user_message": "Je cherche un backend Python",
            "answer": "Voici les candidats.",
            "candidates": [{"candidate_id": "candidate_1"}, {"candidate_id": "candidate_2"}],
            "decision_cards": [{"candidate_id": "candidate_1"}],
            "transferability": {"candidate_1": {"selected_source": "yaml"}},
        },
    )

    assert updated.last_user_message == "Je cherche un backend Python"
    assert updated.last_answer == "Voici les candidats."
    assert updated.last_job_query is None
    assert updated.last_candidates[0]["candidate_id"] == "candidate_1"
    assert updated.last_decision_cards[0]["candidate_id"] == "candidate_1"
    assert "candidate_1" in updated.last_transferability
    assert updated.selected_candidate_id == "candidate_1"
    assert [turn.role for turn in updated.messages] == ["user", "assistant"]
    assert updated.conversation_history == updated.messages


def test_session_store_updates_last_job_query_after_matching() -> None:
    store = InMemorySessionStore()
    memory = store.get_or_create("session-job")

    updated = store.update(
        memory.session_id,
        {
            "user_message": "Oui lance la recherche",
            "answer": "Voici les candidats.",
            "job_description": "Backend Python Engineer\nRequired skills\nPython",
            "matching_completed": True,
            "candidates": [{"candidate_id": "candidate_1"}],
        },
    )

    assert updated.last_job_query == "Backend Python Engineer\nRequired skills\nPython"
    assert updated.job_description == updated.last_job_query
    assert updated.last_candidates[0]["candidate_id"] == "candidate_1"


def test_session_store_keeps_only_last_five_turns() -> None:
    store = InMemorySessionStore()
    memory = store.get_or_create("session-history")

    for index in range(7):
        store.update(
            memory.session_id,
            {
                "user_message": f"user-{index}",
                "answer": f"assistant-{index}",
            },
        )

    assert len(memory.conversation_history) == 10
    assert memory.conversation_history[0].content == "user-2"
    assert memory.conversation_history[-1].content == "assistant-6"


def test_session_store_expires_session_after_ttl() -> None:
    store = InMemorySessionStore(ttl_seconds=1)
    memory = store.get_or_create("session-expired")
    memory.updated_at -= 2

    assert store.get("session-expired") is None

    recreated = store.get_or_create("session-expired")
    assert recreated is not memory
    assert recreated.session_id == "session-expired"


def test_session_store_clear_removes_session() -> None:
    store = InMemorySessionStore()
    store.get_or_create("session-1")

    store.clear("session-1")

    assert store.get("session-1") is None
