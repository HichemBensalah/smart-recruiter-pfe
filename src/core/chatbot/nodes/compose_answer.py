from __future__ import annotations

from typing import Any

from src.core.chatbot.state import RecruiterCopilotState


def compose_answer_node(state: RecruiterCopilotState) -> RecruiterCopilotState:
    intent = str(state.get("intent") or "search_candidates")
    if intent == "explain_candidate":
        answer = compose_explain_candidate_answer(state)
    elif intent == "review_needed":
        answer = compose_review_needed_answer(state)
    elif intent == "gap_analysis":
        answer = compose_gap_analysis_answer(state)
    elif intent == "compare_candidates":
        answer = compose_compare_candidates_answer(state)
    elif intent == "transferability":
        answer = compose_transferability_answer(state)
    else:
        answer = compose_search_answer(state)
    return {"answer": answer}


def compose_search_answer(state: RecruiterCopilotState) -> str:
    candidates = _candidates(state)
    lines = _intro_lines(state)
    if not candidates:
        return _no_candidates_answer(lines)

    lines.append("Voici les meilleurs candidats :")
    for index, candidate in enumerate(candidates[:3], start=1):
        lines.append(_candidate_result_block(index, candidate, state))
    lines.extend(_methodology_lines())
    return "\n".join(lines)


def compose_explain_candidate_answer(state: RecruiterCopilotState) -> str:
    candidates = _candidates(state)
    lines = _intro_lines(state)
    if not candidates:
        return _no_candidates_answer(lines)

    candidate = _selected_candidate(state) or candidates[0]
    candidate_id = _candidate_id(candidate)
    transferability = _candidate_transferability(candidate_id, state)
    gaps_bloquants = transferability.get("gaps_bloquants") or []
    conclusion = "Il est recommandé parce que son classement et ses scores sont favorables dans les artefacts actuels."
    if _status(candidate, state) == "review_needed" or gaps_bloquants:
        conclusion = "Il est bien classé mais à vérifier parce que des signaux de revue ou des gaps sont présents."

    lines.extend(
        [
            "Explication du premier candidat / candidat selectionne :",
            f"- nom : `{_candidate_display_name(candidate, state)}`",
            f"- candidate_id : `{candidate_id}`",
            f"- rang V3 : `{_value(candidate.get('baseline_rank_v3') or candidate.get('rank'))}`",
            f"- score Matching V3 : `{_score(candidate.get('baseline_score_v3'))}`",
            f"- score RF : `{_score(candidate.get('rf_score'))}`",
            f"- score XGBoost : `{_score(candidate.get('xgboost_score'))}`",
            f"- recommendation_status : `{_status(candidate, state) or 'Information non disponible dans les artefacts actuels.'}`",
            f"- transférabilité : `{_score(transferability.get('transferability_score') or transferability.get('coverage_score'))}`",
            f"- gaps bloquants : `{gaps_bloquants or 'Information non disponible dans les artefacts actuels.'}`",
            "",
            conclusion,
        ]
    )
    lines.extend(_methodology_lines())
    return "\n".join(lines)


def compose_review_needed_answer(state: RecruiterCopilotState) -> str:
    candidates = _candidates(state)
    lines = _intro_lines(state)
    if not candidates:
        return _no_candidates_answer(lines)

    review_candidates = [candidate for candidate in candidates[:5] if _needs_review(candidate, state)]
    lines.append("Candidats à vérifier :")
    if not review_candidates:
        lines.append("Aucun candidat `review_needed` ou avec signal de risque fort dans les artefacts disponibles.")
    for candidate in review_candidates:
        lines.append(f"- {_candidate_display_name(candidate, state)} (`{_candidate_id(candidate)}`) - {_candidate_summary(candidate, state)}")
    lines.extend(_methodology_lines())
    return "\n".join(lines)


def compose_gap_analysis_answer(state: RecruiterCopilotState) -> str:
    candidates = _candidates(state)
    lines = _intro_lines(state)
    if not candidates:
        return _no_candidates_answer(lines)

    selected_candidate = _selected_candidate(state)
    analysis_candidates = [selected_candidate] if selected_candidate else candidates[:3]
    lines.append("Analyse des gaps du candidat selectionne :" if selected_candidate else "Analyse des gaps des meilleurs candidats :")
    for candidate in analysis_candidates:
        candidate_id = _candidate_id(candidate)
        transferability = _candidate_transferability(candidate_id, state)
        lines.extend(
            [
                f"- `{candidate_id}`",
                f"  - gaps compensables : `{transferability.get('gaps_compensables') or 'Information non disponible dans les artefacts actuels.'}`",
                f"  - gaps bloquants : `{transferability.get('gaps_bloquants') or 'Information non disponible dans les artefacts actuels.'}`",
            ]
        )
    lines.extend(_methodology_lines())
    return "\n".join(lines)


def compose_compare_candidates_answer(state: RecruiterCopilotState) -> str:
    candidates = _candidates(state)
    lines = _intro_lines(state)
    if len(candidates) < 2:
        lines.append("Comparaison impossible : moins de deux candidats disponibles dans les artefacts actuels.")
        return "\n".join(lines)

    lines.append("Comparaison des deux premiers candidats :")
    for candidate in candidates[:2]:
        candidate_id = _candidate_id(candidate)
        transferability = _candidate_transferability(candidate_id, state)
        lines.extend(
            [
                f"- `{candidate_id}`",
                f"  - score Matching V3 : `{_score(candidate.get('baseline_score_v3'))}`",
                f"  - score RF : `{_score(candidate.get('rf_score'))}`",
                f"  - score XGBoost : `{_score(candidate.get('xgboost_score'))}`",
                f"  - statut : `{_status(candidate, state) or 'Information non disponible dans les artefacts actuels.'}`",
                f"  - transférabilité : `{_score(transferability.get('transferability_score') or transferability.get('coverage_score'))}`",
                f"  - gaps bloquants : `{transferability.get('gaps_bloquants') or 'Information non disponible dans les artefacts actuels.'}`",
            ]
        )
    safer = _safer_candidate(candidates[:2], state)
    if safer:
        lines.append(f"Conclusion : `{safer}` semble le plus sûr selon les signaux disponibles.")
    lines.extend(_methodology_lines())
    return "\n".join(lines)


def compose_transferability_answer(state: RecruiterCopilotState) -> str:
    candidates = _candidates(state)
    lines = _intro_lines(state)
    if not candidates:
        return _no_candidates_answer(lines)

    selected_candidate = _selected_candidate(state)
    analysis_candidates = [selected_candidate] if selected_candidate else candidates[:3]
    lines.append("Analyse de transférabilité :")
    for candidate in analysis_candidates:
        candidate_id = _candidate_id(candidate)
        transferability = _candidate_transferability(candidate_id, state)
        lines.extend(
            [
                f"- `{candidate_id}`",
                f"  - fit_direct : `{transferability.get('fit_direct', 'Information non disponible dans les artefacts actuels.')}`",
                f"  - transferability_score : `{_score(transferability.get('transferability_score') or transferability.get('coverage_score'))}`",
                f"  - transitions plausibles : `{transferability.get('transitions_plausibles') or transferability.get('plausible_transitions') or 'Information non disponible dans les artefacts actuels.'}`",
                f"  - gaps compensables : `{transferability.get('gaps_compensables') or 'Information non disponible dans les artefacts actuels.'}`",
                f"  - gaps bloquants : `{transferability.get('gaps_bloquants') or 'Information non disponible dans les artefacts actuels.'}`",
            ]
        )
    lines.extend(_methodology_lines())
    return "\n".join(lines)


def _intro_lines(state: RecruiterCopilotState) -> list[str]:
    return [
        f"J'ai analysé la demande recruteur : {state.get('user_message') or ''}",
        f"Intention détectée : `{state.get('intent') or 'search_candidates'}`.",
        f"Rôle cible estimé : {state.get('target_role') or 'Backend Developer'}.",
        "Matching V3 reste la baseline officielle de scoring.",
        "",
    ]


def _no_candidates_answer(lines: list[str]) -> str:
    lines.extend(
        [
            "Aucun candidat n'a été retourné par les tools disponibles.",
            "Je ne peux donc pas recommander ou expliquer un profil sans résultat source.",
        ]
    )
    return "\n".join(lines)


def _methodology_lines() -> list[str]:
    return [
        "",
        "Notes méthodologiques :",
        "- Les scores ML/SHAP sont des couches d'analyse expérimentales si présentes.",
        "- Neo4j est utilisé uniquement si disponible ; sinon le fallback YAML de transférabilité est conservé.",
        "- Aucune décision recruteur finale n'est automatisée.",
    ]


def _candidates(state: RecruiterCopilotState) -> list[dict[str, Any]]:
    return [candidate for candidate in state.get("candidates", []) if isinstance(candidate, dict)]


def _cards_by_id(state: RecruiterCopilotState) -> dict[str, dict[str, Any]]:
    return {
        str(card.get("candidate_id")): card
        for card in state.get("decision_cards", [])
        if isinstance(card, dict) and card.get("candidate_id")
    }


def _candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_id") or "unknown_candidate")


def _candidate_display_name(candidate: dict[str, Any], state: RecruiterCopilotState) -> str:
    for key in ("candidate_name", "full_name", "name"):
        value = candidate.get(key)
        if value:
            return str(value)
    card = _cards_by_id(state).get(_candidate_id(candidate), {})
    for key in ("candidate_name", "full_name", "name"):
        value = card.get(key)
        if value:
            return str(value)
    return f"Candidat : {_candidate_id(candidate)}"


def _candidate_result_block(index: int, candidate: dict[str, Any], state: RecruiterCopilotState) -> str:
    candidate_id = _candidate_id(candidate)
    display_name = _candidate_display_name(candidate, state)
    transferability = _candidate_transferability(candidate_id, state)
    gaps = transferability.get("gaps_bloquants") or []
    lines = [
        f"{index}. {display_name}",
        f"   ID : `{candidate_id}`",
        f"   Rang Matching V3 : `{_value(candidate.get('baseline_rank_v3') or candidate.get('rank'))}`",
        f"   Score Matching V3 : `{_score(candidate.get('baseline_score_v3'))}`",
        f"   Score Random Forest : `{_score(candidate.get('rf_score'))}`",
        f"   Score XGBoost : `{_score(candidate.get('xgboost_score'))}`",
        f"   Statut : `{_status(candidate, state) or 'Information non disponible dans les artefacts actuels.'}`",
        f"   Gaps bloquants : `{gaps or 'Information non disponible dans les artefacts actuels.'}`",
    ]
    if display_name.startswith("Candidat :"):
        lines.insert(2, "   Nom : `non disponible dans les artefacts actuels`")
    return "\n".join(lines)


def _selected_candidate(state: RecruiterCopilotState) -> dict[str, Any] | None:
    selected_candidate_id = state.get("selected_candidate_id")
    if not selected_candidate_id:
        return None
    for candidate in _candidates(state):
        if _candidate_id(candidate) == selected_candidate_id:
            return candidate
    return None


def _status(candidate: dict[str, Any], state: RecruiterCopilotState) -> str:
    card = _cards_by_id(state).get(_candidate_id(candidate), {})
    return str(candidate.get("recommendation_status") or card.get("recommendation_status") or "")


def _candidate_summary(candidate: dict[str, Any], state: RecruiterCopilotState) -> str:
    candidate_id = _candidate_id(candidate)
    transferability = _candidate_transferability(candidate_id, state)
    parts = [
        f"rang V3 {_value(candidate.get('baseline_rank_v3') or candidate.get('rank'))}",
        f"score V3 {_score(candidate.get('baseline_score_v3'))}",
        f"RF {_score(candidate.get('rf_score'))}",
        f"XGBoost {_score(candidate.get('xgboost_score'))}",
        f"statut {_status(candidate, state) or 'n/a'}",
    ]
    transferability_score = transferability.get("transferability_score") or transferability.get("coverage_score")
    if transferability_score is not None:
        parts.append(f"transférabilité {_score(transferability_score)}")
    gaps = transferability.get("gaps_bloquants") or []
    if gaps:
        parts.append(f"gaps bloquants: {', '.join(str(gap) for gap in gaps[:3])}")
    return "; ".join(parts)


def _candidate_transferability(candidate_id: str, state: RecruiterCopilotState) -> dict[str, Any]:
    payload = state.get("transferability", {}).get(candidate_id, {})
    if not isinstance(payload, dict):
        return {}
    selected = payload.get("selected_source")
    raw = payload.get(selected) if selected else None
    if not isinstance(raw, dict):
        raw = payload.get("yaml") if isinstance(payload.get("yaml"), dict) else payload
    return _extract_transferability(raw)


def _extract_transferability(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("transferability")
    if isinstance(nested, dict):
        return nested
    return payload


def _needs_review(candidate: dict[str, Any], state: RecruiterCopilotState) -> bool:
    status = _status(candidate, state).lower()
    if status == "review_needed":
        return True
    rank_shifts = [
        candidate.get("rank_shift_v3_vs_rf"),
        candidate.get("rank_shift_v3_vs_xgb"),
        candidate.get("rank_shift_rf_vs_xgb"),
    ]
    if any(isinstance(value, (int, float)) and abs(value) >= 10 for value in rank_shifts):
        return True
    gaps = _candidate_transferability(_candidate_id(candidate), state).get("gaps_bloquants") or []
    return len(gaps) >= 2


def _safer_candidate(candidates: list[dict[str, Any]], state: RecruiterCopilotState) -> str:
    scored: list[tuple[float, str]] = []
    for candidate in candidates:
        score = float(candidate.get("baseline_score_v3") or 0.0)
        if _status(candidate, state).lower() == "review_needed":
            score -= 0.2
        gaps = _candidate_transferability(_candidate_id(candidate), state).get("gaps_bloquants") or []
        score -= 0.05 * len(gaps)
        scored.append((score, _candidate_id(candidate)))
    return max(scored)[1] if scored else ""


def _score(value: Any) -> str:
    if value is None:
        return "Information non disponible dans les artefacts actuels."
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _value(value: Any) -> str:
    return str(value) if value is not None else "Information non disponible dans les artefacts actuels."
