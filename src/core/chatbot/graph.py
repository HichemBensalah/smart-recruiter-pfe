from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.core.chatbot.nodes.analyze_transferability import analyze_transferability_node
from src.core.chatbot.nodes.compose_answer import compose_answer_node
from src.core.chatbot.nodes.fetch_decision_cards import fetch_decision_cards_node
from src.core.chatbot.nodes.match_candidates import match_candidates_node
from src.core.chatbot.nodes.understand_query import understand_query_node
from src.core.chatbot.job_intake import (
    FIELD_LABELS,
    JOB_INTAKE_QUESTIONS,
    apply_field_edit,
    build_job_description,
    detect_field_edit_request,
    detect_offer_summary_request,
    detect_offer_reset_request,
    extract_field_edit_value,
    get_current_step,
    is_job_intake_complete,
    request_field_edit,
    reset_job_intake,
    start_job_intake,
    start_job_intake_with_title,
    summarize_current_offer,
    summarize_job_profile,
    update_job_intake,
)
from src.core.chatbot.memory import SESSION_STORE
from src.core.chatbot.reference_resolver import resolve_candidate_reference
from src.core.chatbot.state import RecruiterCopilotState, initial_state


def build_recruiter_copilot_graph():
    graph = StateGraph(RecruiterCopilotState)
    graph.add_node("understand_query", understand_query_node)
    graph.add_node("match_candidates", match_candidates_node)
    graph.add_node("fetch_decision_cards", fetch_decision_cards_node)
    graph.add_node("analyze_transferability", analyze_transferability_node)
    graph.add_node("compose_answer", compose_answer_node)

    graph.add_edge(START, "understand_query")
    graph.add_edge("understand_query", "match_candidates")
    graph.add_edge("match_candidates", "fetch_decision_cards")
    graph.add_edge("fetch_decision_cards", "analyze_transferability")
    graph.add_edge("analyze_transferability", "compose_answer")
    graph.add_edge("compose_answer", END)
    return graph.compile()


def run_recruiter_copilot(message: str) -> dict[str, Any]:
    app = build_recruiter_copilot_graph()
    final_state = app.invoke(initial_state(message))
    return _result_from_state(final_state)


def run_recruiter_copilot_with_memory(message: str, session_id: str | None = None) -> dict[str, Any]:
    memory = SESSION_STORE.get_or_create(session_id)

    if detect_offer_reset_request(message):
        reset_job_intake(memory)
        return _job_intake_response(
            memory,
            message,
            "Tres bien. Nous repartons sur une nouvelle offre. Quel est le titre du poste ?",
            ["job_intake"],
        )

    if detect_offer_summary_request(message):
        had_offer = isinstance(memory.job_intake, dict) or isinstance(memory.current_job_profile, dict)
        answer = summarize_current_offer(memory)
        if not had_offer:
            start_job_intake(memory)
        return _job_intake_response(
            memory,
            message,
            answer,
            ["job_intake"],
        )

    if memory.awaiting_field_replacement and memory.pending_field_edit:
        edited_field = memory.pending_field_edit
        apply_field_edit(memory, message)
        label = FIELD_LABELS.get(edited_field, edited_field)
        return _job_intake_response(
            memory,
            message,
            f"J'ai mis a jour {label}. Voici le resume corrige de l'offre.\n\n{summarize_job_profile(memory)}",
            ["job_intake", "job_router"],
        )

    if _must_start_single_path(message, memory):
        if _is_follow_up_intent(message):
            start_job_intake(memory)
            return _job_intake_response(
                memory,
                message,
                "Je dois d'abord terminer l'offre et lancer la recherche de candidats.\n\n"
                "Bienvenue dans Smart Recruiter. Je vais vous guider pour creer une offre d'emploi structuree. "
                + JOB_INTAKE_QUESTIONS["job_title"],
            )
        if _looks_like_job_title(message):
            start_job_intake_with_title(memory, message)
            return _job_intake_response(
                memory,
                message,
                "Tres bien, je vais vous aider a creer une offre structuree.\n\n"
                + JOB_INTAKE_QUESTIONS["about_role"],
            )
        start_job_intake(memory)
        if _looks_like_search_request(message):
            return _job_intake_response(
                memory,
                message,
                "Tres bien, je vais vous aider a creer une offre structuree. Quel est le titre exact du poste ?",
            )
        return _job_intake_response(
            memory,
            message,
            "Bienvenue dans Smart Recruiter. Je vais vous guider pour creer une offre d'emploi structuree. "
            + JOB_INTAKE_QUESTIONS["job_title"],
        )

    if memory.mode == "job_creation":
        if memory.pending_confirmation == "launch_matching":
            field_to_edit = detect_field_edit_request(message)
            if field_to_edit:
                label = FIELD_LABELS.get(field_to_edit, field_to_edit)
                inline_value = extract_field_edit_value(message, field_to_edit)
                if inline_value:
                    request_field_edit(memory, field_to_edit)
                    apply_field_edit(memory, inline_value)
                    return _job_intake_response(
                        memory,
                        message,
                        f"J'ai mis a jour {label}. Voici le resume corrige de l'offre.\n\n{summarize_job_profile(memory)}",
                        ["job_intake", "job_router"],
                    )
                request_field_edit(memory, field_to_edit)
                return _job_intake_response(
                    memory,
                    message,
                    f"D'accord. Donne-moi la nouvelle valeur pour : {label}.",
                )
            if _is_positive_confirmation(message):
                return _run_matching_for_completed_job(message, memory)
            if _is_negative_confirmation(message):
                memory.mode = None
                memory.pending_confirmation = None
                return _job_intake_response(
                    memory,
                    message,
                    "Tres bien. Quel champ souhaitez-vous modifier : titre, description, responsabilites, "
                    "competences obligatoires, competences bonus ou profil ?",
                )
            return _job_intake_response(
                memory,
                message,
                "Voulez-vous lancer la recherche de candidats ? Repondez oui pour lancer le matching.",
            )

        if _is_follow_up_intent(message):
            return _job_intake_response(
                memory,
                message,
                "Je dois d'abord terminer l'offre et lancer la recherche de candidats.",
            )

        update_job_intake(memory, message)
        if is_job_intake_complete(memory):
            return _job_intake_response(memory, message, summarize_job_profile(memory), ["job_intake", "job_router"])
        current_step = get_current_step(memory)
        return _job_intake_response(memory, message, JOB_INTAKE_QUESTIONS.get(str(current_step), "Completez l'offre."))

    if not memory.matching_completed and _is_follow_up_intent(message):
        return _job_intake_response(
            memory,
            message,
            "Je dois d'abord terminer l'offre et lancer la recherche de candidats.",
        )

    selected_candidate_id = resolve_candidate_reference(message, memory) or memory.selected_candidate_id

    state = initial_state(message)
    state.update(
        {
            "session_id": memory.session_id,
            "selected_candidate_id": selected_candidate_id,
            "candidates": list(memory.last_candidates),
            "decision_cards": list(memory.last_decision_cards),
            "transferability": dict(memory.last_transferability),
            "sources": ["conversation_memory"] if memory.last_candidates else [],
            "matching_completed": memory.matching_completed,
        }
    )

    app = build_recruiter_copilot_graph()
    final_state = app.invoke(state)
    result = _result_from_state(final_state)
    result["session_id"] = memory.session_id
    result["selected_candidate_id"] = final_state.get("selected_candidate_id") or selected_candidate_id
    result["user_message"] = message
    return _store_and_return(memory.session_id, result)


def _result_from_state(final_state: RecruiterCopilotState) -> dict[str, Any]:
    return {
        "answer": final_state.get("answer"),
        "candidates": final_state.get("candidates", []),
        "decision_cards": final_state.get("decision_cards", []),
        "transferability": final_state.get("transferability", {}),
        "sources": final_state.get("sources", []),
        "warnings": final_state.get("warnings", []),
        "selected_candidate_id": final_state.get("selected_candidate_id"),
        "matching_completed": final_state.get("matching_completed", False),
        "job_intake_state": final_state.get("job_intake_state"),
        "structured_job_profile": final_state.get("structured_job_profile"),
        "routed_job_id": final_state.get("routed_job_id"),
        "matching_metadata": final_state.get("matching_metadata", {}),
    }


def _run_matching_for_completed_job(message: str, memory) -> dict[str, Any]:
    job_description = build_job_description(memory)
    route = (memory.job_intake or {}).get("route", {}) if isinstance(memory.job_intake, dict) else {}
    routed_job_id = route.get("job_id") or "backend_python_django_postgresql"
    memory.mode = None
    memory.pending_confirmation = None

    app = build_recruiter_copilot_graph()
    state = initial_state(job_description)
    state["session_id"] = memory.session_id
    state["routed_job_id"] = routed_job_id
    final_state = app.invoke(state)
    result = _result_from_state(final_state)
    resolved_job_id = final_state.get("routed_job_id") or routed_job_id
    result["answer"] = "\n".join(
        [
            f"Recherche lancee pour le job profile route : `{resolved_job_id}`.",
            "",
            str(result.get("answer") or ""),
        ]
    )
    result["session_id"] = memory.session_id
    result["user_message"] = message
    result["sources"] = _append_unique(result.get("sources", []), ["job_intake", "job_router"])
    result["job_route"] = route
    result["structured_job_profile"] = memory.current_job_profile
    result["job_intake_state"] = memory.job_intake_state or memory.job_intake
    result["routed_job_id"] = resolved_job_id
    result["job_description"] = job_description
    result["matching_completed"] = True
    if not result.get("matching_metadata"):
        result["matching_metadata"] = {"resolved_job_id": resolved_job_id}
    return _store_and_return(memory.session_id, result)


def _job_intake_response(
    memory,
    message: str,
    answer: str,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    intake = memory.job_intake if isinstance(memory.job_intake, dict) else {}
    structured_profile = intake.get("structured_job_profile") if isinstance(intake.get("structured_job_profile"), dict) else None
    route = intake.get("route") if isinstance(intake.get("route"), dict) else {}
    job_description = build_job_description(memory) if structured_profile else None
    result = {
        "session_id": memory.session_id,
        "answer": answer,
        "candidates": [],
        "decision_cards": [],
        "transferability": {},
        "sources": sources or ["job_intake"],
        "warnings": [],
        "user_message": message,
        "job_intake_state": intake,
        "structured_job_profile": structured_profile,
        "routed_job_id": route.get("job_id"),
        "job_description": job_description,
        "matching_completed": False,
        "matching_metadata": {},
        "pending_field_edit": memory.pending_field_edit,
        "awaiting_field_replacement": memory.awaiting_field_replacement,
    }
    return _store_and_return(memory.session_id, result)


def _store_and_return(session_id: str, result: dict[str, Any]) -> dict[str, Any]:
    SESSION_STORE.update(session_id, result)
    return result


def _should_start_job_intake(message: str, memory) -> bool:
    if memory.mode == "job_creation":
        return False
    lowered = message.lower()
    triggers = [
        "je veux creer une offre",
        "je veux créer une offre",
        "creer une nouvelle offre",
        "créer une nouvelle offre",
        "nouvelle fiche de poste",
        "new job",
        "create job",
    ]
    return any(trigger in lowered for trigger in triggers)


def _must_start_single_path(message: str, memory) -> bool:
    return (
        memory.mode is None
        and not memory.offer_created
        and not memory.matching_completed
        and not memory.last_candidates
    )


def _is_follow_up_intent(message: str) -> bool:
    lowered = message.lower()
    markers = [
        "pourquoi",
        "premier candidat",
        "deux premiers",
        "compare",
        "comparer",
        "gap",
        "gaps",
        "lui",
        "ses ",
        "son ",
        "a verifier",
        "à vérifier",
        "risque",
        "desaccord",
        "désaccord",
        "transferabilite",
        "transférabilité",
        "evoluer",
        "évoluer",
    ]
    return any(marker in lowered for marker in markers)


def _looks_like_job_title(message: str) -> bool:
    lowered = message.lower().strip()
    if _should_start_job_intake(message, type("MemoryProbe", (), {"mode": None})()):
        return False
    if any(token in lowered for token in ["cherche", "trouve", "pourquoi", "compare", "gap", "gaps", "?"]):
        return False
    word_count = len(lowered.split())
    role_markers = ["engineer", "developer", "analyst", "devops", "backend", "frontend", "data", "full stack"]
    return 1 <= word_count <= 6 and any(marker in lowered for marker in role_markers)


def _looks_like_search_request(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ["je cherche", "trouve-moi", "trouve moi", "recrute", "candidat"])


def _is_positive_confirmation(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ["oui", "yes", "lance", "lancer la recherche", "confirme", "go", "recherche"])


def _is_negative_confirmation(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ["non", "no", "annule", "pas maintenant"])


def _append_unique(values: list[str], additions: list[str]) -> list[str]:
    result = list(values)
    for value in additions:
        if value not in result:
            result.append(value)
    return result
