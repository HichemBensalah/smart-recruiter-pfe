from __future__ import annotations

from typing import Any

from src.core.chatbot.state import RecruiterCopilotState


def compose_answer_node(state: RecruiterCopilotState) -> RecruiterCopilotState:
    user_message = str(state.get("user_message") or "")
    target_role = str(state.get("target_role") or "Backend Developer")
    candidates = state.get("candidates", [])
    cards_by_id = {
        card.get("candidate_id"): card
        for card in state.get("decision_cards", [])
        if isinstance(card, dict) and card.get("candidate_id")
    }
    transferability = state.get("transferability", {})

    lines = [
        f"J'ai analyse la demande recruteur: {user_message}",
        f"Role cible estime: {target_role}.",
        "Matching V3 reste la baseline officielle de scoring.",
        "",
    ]

    if not candidates:
        lines.extend(
            [
                "Aucun candidat n'a ete retourne par les tools disponibles.",
                "Je ne peux donc pas recommander de profil sans resultat source.",
            ]
        )
        return {"answer": "\n".join(lines)}

    lines.append("Top candidats recommandes:")
    for index, candidate in enumerate(candidates[:5], start=1):
        candidate_id = str(candidate.get("candidate_id") or "unknown_candidate")
        card = cards_by_id.get(candidate_id, {})
        reasons = _candidate_reasons(candidate, card, transferability.get(candidate_id, {}))
        lines.append(f"{index}. `{candidate_id}` - {reasons}")

    review_needed = [
        candidate
        for candidate in candidates[:5]
        if str(candidate.get("recommendation_status") or "").lower() == "review_needed"
    ]
    if review_needed:
        lines.append("")
        lines.append("Candidats a verifier:")
        for candidate in review_needed:
            lines.append(f"- `{candidate.get('candidate_id')}`: statut review_needed dans les donnees disponibles.")

    lines.extend(
        [
            "",
            "Notes methodologiques:",
            "- Les scores ML/SHAP sont des couches d'analyse experimentales si presentes.",
            "- Neo4j est utilise uniquement si disponible ; sinon le fallback YAML de transferabilite est conserve.",
            "- Aucune decision recruteur finale n'est automatisee.",
        ]
    )
    return {"answer": "\n".join(lines)}


def _candidate_reasons(candidate: dict[str, Any], card: dict[str, Any], transferability_payload: dict[str, Any]) -> str:
    parts: list[str] = []
    baseline_rank = candidate.get("baseline_rank_v3") or candidate.get("rank")
    baseline_score = candidate.get("baseline_score_v3")
    if baseline_rank is not None:
        parts.append(f"rang V3 {baseline_rank}")
    if baseline_score is not None:
        parts.append(f"score V3 {float(baseline_score):.4f}")
    if candidate.get("rf_score") is not None:
        parts.append(f"score RF {float(candidate['rf_score']):.4f}")
    if candidate.get("xgboost_score") is not None:
        parts.append(f"score XGBoost {float(candidate['xgboost_score']):.4f}")
    status = candidate.get("recommendation_status") or card.get("recommendation_status")
    if status:
        parts.append(f"statut {status}")

    selected = transferability_payload.get("selected_source")
    raw_transferability = transferability_payload.get(selected) if selected else None
    transferability = _extract_transferability(raw_transferability)
    if transferability:
        score = transferability.get("transferability_score") or transferability.get("coverage_score")
        if score is not None:
            parts.append(f"transferabilite {float(score):.4f}")
        gaps = transferability.get("gaps_bloquants") or []
        if gaps:
            parts.append(f"gaps bloquants: {', '.join(str(gap) for gap in gaps[:3])}")

    return "; ".join(parts) if parts else "donnees sources limitees"


def _extract_transferability(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("transferability")
    if isinstance(nested, dict):
        return nested
    return payload
