from src.core.chatbot.state import initial_state


def test_initial_state_accepts_user_message_and_defaults() -> None:
    state = initial_state("Je cherche un backend Python")

    assert state["user_message"] == "Je cherche un backend Python"
    assert state["job_description"] is None
    assert state["top_k"] == 5
    assert state["target_role"] == "Backend Developer"
    assert state["candidates"] == []
    assert state["decision_cards"] == []
    assert state["transferability"] == {}
    assert state["warnings"] == []
