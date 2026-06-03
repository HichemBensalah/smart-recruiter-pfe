"""Test session isolation: creating multiple offers should not mix candidates/metadata."""

from src.core.chatbot.job_intake import reset_job_intake
from src.core.chatbot.memory import ConversationMemory, InMemorySessionStore


def test_session_isolation_state_cleanup():
    """Test that reset_job_intake cleans up all necessary fields after an offer."""
    memory = ConversationMemory(session_id="test_session")

    # Set up state from a previous offer (simulating after first matching)
    memory.current_job_profile = {"job_title": "Senior Python", "target_role": "backend"}
    memory.routed_job_id = "backend_python_django_postgresql"
    memory.job_description = "Build scalable APIs"
    memory.matching_completed = True
    memory.selected_candidate_id = "cand_123"
    memory.last_candidates = [{"candidate_id": "cand_123", "name": "John Doe"}]
    memory.last_decision_cards = [{"card": "data"}]
    memory.last_transferability = {"score": 0.85}
    memory.offer_created = True
    memory.pending_field_edit = "required_skills"
    memory.job_intake_state = {"current_step": "confirmation"}

    # Simulate user saying "nouvelle offre" — should reset
    reset_job_intake(memory)

    # Verify comprehensive cleanup
    assert memory.current_job_profile is None, "current_job_profile should be None after reset"
    assert memory.routed_job_id is None, "routed_job_id should be None after reset"
    assert memory.job_description is None, "job_description should be None after reset"
    assert memory.matching_completed is False, "matching_completed should be False after reset"
    assert memory.selected_candidate_id is None, "selected_candidate_id should be None after reset"
    assert memory.last_candidates == [], "last_candidates should be empty after reset"
    assert memory.last_decision_cards == [], "last_decision_cards should be empty after reset"
    assert memory.last_transferability == {}, "last_transferability should be empty after reset"
    assert memory.offer_created is False, "offer_created should be False after reset"
    assert memory.pending_field_edit is None, "pending_field_edit should be None after reset"

    # job_intake_state is set during confirm_and_prepare_for_matching, so it's OK if None after start
    # but job_intake should be reinitialized
    assert memory.job_intake is not None, "job_intake should be reinitialized after reset"
    assert memory.mode == "job_creation", "mode should be set to job_creation after reset"

    # Verify session_id is preserved (important for session continuity)
    assert memory.session_id == "test_session", "session_id must be preserved across offers"


def test_session_isolation_no_candidate_bleed():
    """Ensure that candidates from offer A don't appear when checking offer B."""
    memory = ConversationMemory(session_id="test_session")

    # Offer A: set candidates
    memory.last_candidates = [
        {"candidate_id": "backend_1", "name": "Alice (Backend)"},
        {"candidate_id": "backend_2", "name": "Bob (Backend)"},
    ]
    memory.selected_candidate_id = "backend_1"
    memory.current_job_profile = {"target_role": "backend"}

    # User triggers "nouvelle offre"
    reset_job_intake(memory)

    # Offer B should start with clean slate
    assert memory.last_candidates == [], "No candidates should remain from previous offer"
    assert memory.selected_candidate_id is None, "No selected candidate should remain"

    # Simulate populating with new candidates (ML)
    memory.last_candidates = [
        {"candidate_id": "ml_1", "name": "Charlie (ML)"},
        {"candidate_id": "ml_2", "name": "Diana (ML)"},
    ]
    memory.current_job_profile = {"target_role": "machine_learning"}
    memory.selected_candidate_id = "ml_1"

    # Verify no bleed from offer A
    assert "backend" not in str(memory.last_candidates).lower(), "No backend candidates in offer B"
    assert memory.selected_candidate_id == "ml_1", "Selected candidate is from offer B"


def test_session_isolation_routed_job_id_isolated():
    """Verify that routed_job_id is per-offer and not reused across offers."""
    memory = ConversationMemory(session_id="test_session")

    # Offer A
    memory.routed_job_id = "backend_python_django_postgresql"
    assert memory.routed_job_id == "backend_python_django_postgresql"

    # User triggers "nouvelle offre"
    reset_job_intake(memory)

    # Offer B should have no routed_job_id initially
    assert memory.routed_job_id is None, "routed_job_id should be cleared for new offer"

    # New routed_job_id for offer B (e.g., ML-specific)
    memory.routed_job_id = "ml_pytorch_kaggle"
    assert memory.routed_job_id == "ml_pytorch_kaggle"
    assert memory.routed_job_id != "backend_python_django_postgresql", "Offers should have different routing"


def test_session_store_multiple_sessions_isolated():
    """Test that multiple sessions don't interfere with each other."""
    store = InMemorySessionStore()

    # Session 1
    mem1 = store.get_or_create("session_1")
    mem1.last_candidates = [{"candidate_id": "cand_s1_1"}]
    mem1.current_job_profile = {"target_role": "backend"}

    # Session 2
    mem2 = store.get_or_create("session_2")
    mem2.last_candidates = [{"candidate_id": "cand_s2_1"}]
    mem2.current_job_profile = {"target_role": "ml"}

    # Verify isolation
    mem1_retrieved = store.get_or_create("session_1")
    mem2_retrieved = store.get_or_create("session_2")

    assert mem1_retrieved.last_candidates[0]["candidate_id"] == "cand_s1_1"
    assert mem2_retrieved.last_candidates[0]["candidate_id"] == "cand_s2_1"
    assert mem1_retrieved.current_job_profile["target_role"] == "backend"
    assert mem2_retrieved.current_job_profile["target_role"] == "ml"
    assert mem1_retrieved.last_candidates != mem2_retrieved.last_candidates, "Sessions should have separate candidates"


def test_session_isolation_last_job_query_cleaned():
    """Verify that last_job_query is cleaned when resetting offer."""
    memory = ConversationMemory(session_id="test_session")

    # Simulate first offer with a job query
    memory.last_job_query = "Find senior backend engineers with 5+ years Python"
    memory.job_description = "Build scalable APIs with FastAPI and PostgreSQL"
    memory.current_job_profile = {"target_role": "backend"}

    # User triggers "nouvelle offre"
    reset_job_intake(memory)

    # Verify cleanup
    assert memory.last_job_query is None, "last_job_query should be cleared for new offer"
    assert memory.job_description is None, "job_description should be cleared for new offer"

    # New offer should not contain old query
    memory.last_job_query = "Find junior ML engineers"
    assert memory.last_job_query == "Find junior ML engineers", "New offer should have new query"
