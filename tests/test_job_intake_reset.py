from __future__ import annotations

from src.core.chatbot.graph import run_recruiter_copilot_with_memory
from src.core.chatbot.job_intake import detect_offer_reset_request, reset_job_intake
from src.core.chatbot.memory import ConversationMemory, SESSION_STORE


def test_detect_offer_reset_request_variants() -> None:
    assert detect_offer_reset_request("réinitialise l'offre")
    assert detect_offer_reset_request("nouvelle offre")
    assert detect_offer_reset_request("recommencer")
    assert detect_offer_reset_request("start new job")


def test_reset_job_intake_clears_previous_state() -> None:
    memory = ConversationMemory(
        session_id="reset-1",
        mode=None,
        offer_created=True,
        matching_completed=True,
        current_job_profile={"job_title": "Old"},
        routed_job_id="old_job",
        job_description="old description",
        last_candidates=[{"candidate_id": "candidate_1"}],
        last_decision_cards=[{"candidate_id": "candidate_1"}],
        last_transferability={"candidate_1": {}},
        selected_candidate_id="candidate_1",
    )

    intake = reset_job_intake(memory)

    assert memory.mode == "job_creation"
    assert intake["current_step"] == "job_title"
    assert all(value == "" for value in intake["fields"].values())
    assert memory.matching_completed is False
    assert memory.last_candidates == []
    assert memory.last_decision_cards == []
    assert memory.last_transferability == {}
    assert memory.selected_candidate_id is None


def test_reset_command_returns_new_wizard_and_does_not_match(monkeypatch) -> None:
    class FailingMatchTool:
        @staticmethod
        def invoke(payload):
            raise AssertionError("reset must not launch matching")

    SESSION_STORE.clear("reset-2")
    monkeypatch.setattr("src.core.chatbot.nodes.match_candidates.match_candidates_tool", FailingMatchTool())
    memory = SESSION_STORE.get_or_create("reset-2")
    memory.matching_completed = True
    memory.last_candidates = [{"candidate_id": "candidate_1"}]

    result = run_recruiter_copilot_with_memory("nouvelle offre", "reset-2")

    assert "nouvelle offre" in result["answer"]
    assert "titre du poste" in result["answer"]
    assert result["candidates"] == []
    assert result["decision_cards"] == []
    assert result["transferability"] == {}
    assert result["matching_completed"] is False
    assert result["job_intake_state"]["current_step"] == "job_title"


def test_follow_up_after_reset_does_not_use_old_candidates() -> None:
    SESSION_STORE.clear("reset-3")
    memory = SESSION_STORE.get_or_create("reset-3")
    memory.matching_completed = True
    memory.last_candidates = [{"candidate_id": "candidate_old"}]

    run_recruiter_copilot_with_memory("reset job", "reset-3")
    result = run_recruiter_copilot_with_memory("Pourquoi le premier candidat ?", "reset-3")

    assert "Je dois d'abord terminer l'offre" in result["answer"]
    assert "candidate_old" not in result["answer"]
    assert result["candidates"] == []
