from __future__ import annotations

from src.core.chatbot.memory import ConversationMemory
from src.core.chatbot.reference_resolver import resolve_candidate_reference


def _memory() -> ConversationMemory:
    return ConversationMemory(
        session_id="session-1",
        last_candidates=[
            {"candidate_id": "candidate_1"},
            {"candidate_id": "candidate_2"},
            {"candidate_id": "candidate_3"},
        ],
        selected_candidate_id="candidate_2",
    )


def test_resolves_first_candidate_reference() -> None:
    assert resolve_candidate_reference("Pourquoi le premier candidat ?", _memory()) == "candidate_1"
    assert resolve_candidate_reference("Quels sont les gaps du meilleur candidat ?", _memory()) == "candidate_1"


def test_resolves_second_candidate_reference() -> None:
    assert resolve_candidate_reference("Compare le deuxieme candidat", _memory()) == "candidate_2"
    assert resolve_candidate_reference("Compare le deuxième candidat", _memory()) == "candidate_2"


def test_resolves_last_candidate_reference() -> None:
    assert resolve_candidate_reference("Quels sont les gaps du dernier candidat ?", _memory()) == "candidate_3"


def test_resolves_pronoun_to_selected_candidate() -> None:
    assert resolve_candidate_reference("Quels sont ses gaps ?", _memory()) == "candidate_2"
    assert resolve_candidate_reference("Pourquoi ce candidat est moins bien classé ?", _memory()) == "candidate_2"


def test_resolves_explicit_candidate_id() -> None:
    assert resolve_candidate_reference("Explique candidate_abc123", _memory()) == "candidate_abc123"
