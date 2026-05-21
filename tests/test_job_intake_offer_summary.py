from __future__ import annotations

from src.core.chatbot.graph import run_recruiter_copilot_with_memory
from src.core.chatbot.job_intake import (
    detect_offer_summary_request,
    start_job_intake,
    summarize_current_offer,
    update_job_intake,
)
from src.core.chatbot.memory import ConversationMemory, SESSION_STORE


def test_detect_offer_summary_request_variants() -> None:
    assert detect_offer_summary_request("résume l'offre")
    assert detect_offer_summary_request("resume l'offre")
    assert detect_offer_summary_request("montre-moi l'offre")
    assert detect_offer_summary_request("affiche le profil structuré")
    assert detect_offer_summary_request("qu'est-ce que j'ai rempli ?")
    assert detect_offer_summary_request("show current job")


def test_summary_shows_filled_and_missing_fields() -> None:
    memory = ConversationMemory(session_id="summary-1")
    start_job_intake(memory)
    update_job_intake(memory, "Backend Python Engineer")
    update_job_intake(memory, "We are looking for a backend engineer.")

    summary = summarize_current_offer(memory)

    assert "Backend Python Engineer" in summary
    assert "We are looking for a backend engineer." in summary
    assert "Competences obligatoires : non renseigne" in summary
    assert "Progression : 2/6" in summary
    assert "Etape actuelle : responsibilities" in summary


def test_summary_request_does_not_advance_step_or_match(monkeypatch) -> None:
    class FailingMatchTool:
        @staticmethod
        def invoke(payload):
            raise AssertionError("summary must not launch matching")

    SESSION_STORE.clear("summary-2")
    monkeypatch.setattr("src.core.chatbot.nodes.match_candidates.match_candidates_tool", FailingMatchTool())

    run_recruiter_copilot_with_memory("Backend Python Engineer", "summary-2")
    before = SESSION_STORE.get("summary-2").job_intake["current_step"]

    result = run_recruiter_copilot_with_memory("résume l'offre", "summary-2")
    after = SESSION_STORE.get("summary-2").job_intake["current_step"]

    assert before == "about_role"
    assert after == "about_role"
    assert result["matching_completed"] is False
    assert result["candidates"] == []
    assert "Progression : 1/6" in result["answer"]


def test_summary_before_confirmation_includes_routed_job_id() -> None:
    SESSION_STORE.clear("summary-3")
    _complete_conversation("summary-3")

    result = run_recruiter_copilot_with_memory("montre-moi l'offre", "summary-3")

    assert result["routed_job_id"] == "backend_python_fastapi_mongodb_aligned"
    assert "Job profile utilise : backend_python_fastapi_mongodb_aligned" in result["answer"]
    assert "Voulez-vous lancer" in result["answer"]
    assert result["matching_completed"] is False


def test_summary_after_matching_mentions_last_matching(monkeypatch) -> None:
    class FakeMatchTool:
        @staticmethod
        def invoke(payload):
            return {"items": [{"candidate_id": "candidate_1"}]}

    class FakeDecisionCardTool:
        @staticmethod
        def invoke(payload):
            return {"candidate_id": payload["candidate_id"]}

    class FakeCandidateProfileTool:
        @staticmethod
        def invoke(payload):
            return {"candidate_id": payload["candidate_id"]}

    class FakeTransferabilityTool:
        @staticmethod
        def invoke(payload):
            return {"candidate_id": payload["candidate_id"], "transferability": {}}

    class FakeNeo4jTool:
        @staticmethod
        def invoke(payload):
            return {"available": False, "fallback_recommended": True}

    SESSION_STORE.clear("summary-4")
    monkeypatch.setattr("src.core.chatbot.nodes.match_candidates.match_candidates_tool", FakeMatchTool())
    monkeypatch.setattr("src.core.chatbot.nodes.fetch_decision_cards.get_decision_card_tool", FakeDecisionCardTool())
    monkeypatch.setattr("src.core.chatbot.nodes.fetch_decision_cards.get_candidate_profile_tool", FakeCandidateProfileTool())
    monkeypatch.setattr("src.core.chatbot.nodes.analyze_transferability.get_transferability_tool", FakeTransferabilityTool())
    monkeypatch.setattr("src.core.chatbot.nodes.analyze_transferability.get_neo4j_transferability_tool", FakeNeo4jTool())

    _complete_conversation("summary-4")
    run_recruiter_copilot_with_memory("oui lance la recherche", "summary-4")
    result = run_recruiter_copilot_with_memory("show job", "summary-4")

    assert result["matching_completed"] is False
    assert "Cette offre a deja ete utilisee pour le dernier matching." in result["answer"]


def test_summary_without_offer_starts_wizard() -> None:
    SESSION_STORE.clear("summary-5")

    result = run_recruiter_copilot_with_memory("résume l'offre", "summary-5")

    assert "Aucune offre n'est encore en cours de creation" in result["answer"]
    assert SESSION_STORE.get("summary-5").job_intake["current_step"] == "job_title"


def _complete_conversation(session_id: str) -> None:
    run_recruiter_copilot_with_memory("Backend Python Engineer", session_id)
    run_recruiter_copilot_with_memory("We are looking for a backend engineer.", session_id)
    run_recruiter_copilot_with_memory("You will design APIs.", session_id)
    run_recruiter_copilot_with_memory("- Python\n- FastAPI\n- MongoDB", session_id)
    run_recruiter_copilot_with_memory("- AWS", session_id)
    run_recruiter_copilot_with_memory("Mid-level with at least 3 years, Tunis, hybrid.", session_id)
