from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METHODOLOGICAL_NOTE = (
    "Cette synthèse agrège plusieurs couches : Matching V3, modèles ML entraînés sur pseudo-labels métier "
    "contrôlés, SHAP et Potential Graph. Matching V3 reste la baseline officielle, Random Forest est le "
    "meilleur modèle ML actuel selon les métriques observées, XGBoost est utilisé pour l’analyse SHAP, et "
    "Potential Graph sert à analyser la transférabilité métier."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a top-k demo summary from enriched Decision Cards.")
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def build_demo_summary(cards_payload: dict[str, Any], *, top_k: int, source_path: str | Path) -> dict[str, Any]:
    cards = [card for card in cards_payload.get("candidates", []) if isinstance(card, dict)]
    ordered = sorted(cards, key=lambda item: int(item.get("baseline_rank_v3", item.get("rank", 999999))))
    top_candidates = [compact_candidate(card) for card in ordered[:top_k]]
    transfer_scores = [float(candidate["transferability_score"]) for candidate in top_candidates]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_id": cards_payload.get("job_id"),
        "generated_from": str(source_path),
        "top_k": top_k,
        "summary": {
            "total_cards": int(cards_payload.get("total_cards", cards_payload.get("candidate_count", len(cards)))),
            "top_k": top_k,
            "lookup_success_rate": cards_payload.get("lookup_success_rate"),
            "number_fit_direct": sum(1 for candidate in top_candidates if candidate["fit_direct"]),
            "number_with_transition_plausible": sum(1 for candidate in top_candidates if candidate["transitions_plausibles"]),
            "number_review_needed": sum(1 for candidate in top_candidates if candidate.get("recommendation_status") == "review_needed"),
            "average_transferability_score": round(sum(transfer_scores) / len(transfer_scores), 4) if transfer_scores else 0.0,
            "methodological_note": METHODOLOGICAL_NOTE,
        },
        "candidates": top_candidates,
    }


def compact_candidate(card: dict[str, Any]) -> dict[str, Any]:
    transfer = card.get("transferability") if isinstance(card.get("transferability"), dict) else {}
    candidate = {
        "candidate_id": card.get("candidate_id"),
        "profile_id": card.get("profile_id"),
        "baseline_rank_v3": card.get("baseline_rank_v3", card.get("rank")),
        "baseline_score_v3": card.get("baseline_score_v3", card.get("final_score")),
        "rf_rank": card.get("rf_rank"),
        "rf_score": card.get("rf_score"),
        "xgboost_rank": card.get("xgboost_rank"),
        "xgboost_score": card.get("xgboost_score"),
        "recommendation_status": card.get("recommendation_status"),
        "shap_top_features": card.get("shap_top_features", []),
        "transferability_score": float(transfer.get("transferability_score", 0.0)),
        "fit_direct": bool(transfer.get("fit_direct", False)),
        "best_source_role": transfer.get("best_source_role"),
        "target_role": transfer.get("target_role"),
        "transitions_plausibles": transfer.get("transitions_plausibles", []),
        "gaps_compensables": transfer.get("gaps_compensables", []),
        "gaps_bloquants": transfer.get("gaps_bloquants", []),
    }
    candidate["short_decision_summary"] = short_decision_summary(candidate)
    return candidate


def short_decision_summary(candidate: dict[str, Any]) -> str:
    baseline_score = _float_or_none(candidate.get("baseline_score_v3"))
    rf_score = _float_or_none(candidate.get("rf_score"))
    xgb_score = _float_or_none(candidate.get("xgboost_score"))
    has_transition = bool(candidate.get("transitions_plausibles"))
    blocking_gaps = candidate.get("gaps_bloquants") or []
    status = candidate.get("recommendation_status")

    if status == "review_needed":
        return "Candidat à vérifier : désaccord important entre XGBoost et les autres scores."
    if baseline_score is not None and rf_score is not None and xgb_score is not None:
        if baseline_score >= 0.6 and rf_score >= 0.8 and xgb_score >= 0.8:
            return "Candidat fortement recommandé : bon score V3, score RF élevé, score XGBoost élevé."
    if has_transition and not candidate.get("fit_direct"):
        return "Candidat partiellement transférable : fit direct faible mais transition plausible."
    if (baseline_score is not None and baseline_score < 0.4) and blocking_gaps:
        return "Candidat faible : scores faibles et gaps bloquants."
    return "Candidat à analyser : signaux partiels, validation humaine recommandée."


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown(summary: dict[str, Any]) -> str:
    candidates = summary["candidates"]
    review_needed = [candidate for candidate in candidates if candidate.get("recommendation_status") == "review_needed"]
    transferable = [candidate for candidate in candidates if candidate.get("transitions_plausibles")]

    lines = [
        "# Synthèse démo top 10 - Smart Recruiter",
        "",
        "## Objectif",
        "",
        "Présenter une vue unique et lisible du top 10 candidats en agrégeant Matching V3, Random Forest, XGBoost + SHAP et Potential Graph.",
        "",
        "## Architecture utilisée",
        "",
        "- Matching V3 : baseline officielle et score principal de référence.",
        "- Random Forest : meilleur modèle ML actuel selon les métriques observées.",
        "- XGBoost + SHAP : modèle avancé conservé pour l'analyse et l'explicabilité.",
        "- Potential Graph : analyse déclarative de transférabilité métier et des gaps.",
        "",
        "## Résumé",
        "",
        f"- `job_id`: `{summary['job_id']}`",
        f"- top_k: `{summary['top_k']}`",
        f"- total cartes source: `{summary['summary']['total_cards']}`",
        f"- lookup success rate: `{summary['summary'].get('lookup_success_rate')}`",
        f"- fit direct dans le top 10: `{summary['summary']['number_fit_direct']}`",
        f"- transitions plausibles dans le top 10: `{summary['summary']['number_with_transition_plausible']}`",
        f"- review_needed dans le top 10: `{summary['summary']['number_review_needed']}`",
        f"- transferability moyenne: `{summary['summary']['average_transferability_score']:.4f}`",
        "",
        "## Tableau top 10 candidats",
        "",
        "| V3 rank | candidate_id | V3 score | RF rank | RF score | XGB rank | XGB score | transferability | status | synthèse |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for candidate in candidates:
        lines.append(
            f"| {candidate['baseline_rank_v3']} | `{candidate['candidate_id']}` | {_fmt(candidate['baseline_score_v3'])} | "
            f"{_fmt_rank(candidate['rf_rank'])} | {_fmt(candidate['rf_score'])} | "
            f"{_fmt_rank(candidate['xgboost_rank'])} | {_fmt(candidate['xgboost_score'])} | "
            f"{_fmt(candidate['transferability_score'])} | {candidate.get('recommendation_status')} | "
            f"{candidate['short_decision_summary']} |"
        )

    lines.extend(["", "## Candidats à vérifier", ""])
    lines.extend(_candidate_bullets(review_needed, include_gaps=True))
    lines.extend(["", "## Transférabilité", ""])
    lines.extend(_candidate_bullets(transferable, include_gaps=True))
    lines.extend(
        [
            "",
            "## Limites méthodologiques",
            "",
            f"- {summary['summary']['methodological_note']}",
            "- Les scores ML ne sont pas des scores recruteur finaux.",
            "- Potential Graph dépend des skills structurées disponibles et d'un graphe YAML déclaratif.",
            "- Cette synthèse est une aide de démo, pas une décision automatique.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(summary), encoding="utf-8")


def _candidate_bullets(candidates: list[dict[str, Any]], *, include_gaps: bool = False) -> list[str]:
    if not candidates:
        return ["- Aucun candidat dans cette catégorie."]
    lines: list[str] = []
    for candidate in candidates:
        shap = ", ".join(item.get("feature", "") for item in candidate.get("shap_top_features", [])[:3] if isinstance(item, dict))
        gaps = ", ".join(candidate.get("gaps_bloquants", [])) or "aucun gap bloquant"
        transitions = candidate.get("transitions_plausibles") or []
        transition_text = transitions[0].get("rationale") if transitions and isinstance(transitions[0], dict) else "pas de transition explicite"
        detail = f"SHAP: {shap or 'NA'}"
        if include_gaps:
            detail += f"; gaps bloquants: {gaps}; transition: {transition_text}"
        lines.append(f"- `{candidate['candidate_id']}` : {candidate['short_decision_summary']} {detail}.")
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _fmt_rank(value: Any) -> str:
    if value is None:
        return "NA"
    return str(int(value))


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    cards_payload = read_json(args.cards)
    summary = build_demo_summary(cards_payload, top_k=args.top_k, source_path=args.cards)
    write_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(f"Demo summary JSON written: {args.output_json}")
    print(f"Demo summary Markdown written: {args.output_md}")


if __name__ == "__main__":
    main()

