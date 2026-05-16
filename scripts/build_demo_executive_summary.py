from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METHODOLOGICAL_NOTE = (
    "Cette synthèse agrège Matching V3, modèles ML entraînés sur pseudo-labels métier contrôlés, SHAP et "
    "Potential Graph. Matching V3 reste la baseline officielle, Random Forest est le meilleur modèle ML actuel "
    "selon les métriques observées, XGBoost est utilisé pour l’analyse SHAP, et Potential Graph sert à analyser "
    "la transférabilité métier."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a short executive demo summary from demo_summary_top10.json.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def build_executive_summary(payload: dict[str, Any], source_path: str | Path) -> dict[str, Any]:
    candidates = [candidate for candidate in payload.get("candidates", []) if isinstance(candidate, dict)]
    recommended = sorted(
        [candidate for candidate in candidates if candidate.get("recommendation_status") != "review_needed"],
        key=_recommended_sort_key,
    )[:3]
    needs_review = sorted(
        [candidate for candidate in candidates if candidate.get("recommendation_status") == "review_needed"],
        key=_review_sort_key,
    )[:3]

    top_recommended = [compact_candidate(candidate, mode="recommended") for candidate in recommended]
    review_items = [compact_candidate(candidate, mode="review") for candidate in needs_review]
    return {
        "job_id": payload.get("job_id"),
        "source": str(source_path),
        "executive_summary": {
            "message": "Vue courte pour démo rapide: meilleurs candidats, candidats à vérifier, signaux clés et limites.",
            "top_recommended_count": len(top_recommended),
            "needs_review_count": len(review_items),
        },
        "top_recommended": top_recommended,
        "needs_review": review_items,
        "key_metrics": {
            "source_top_k": payload.get("top_k"),
            "total_cards": (payload.get("summary") or {}).get("total_cards"),
            "number_fit_direct": (payload.get("summary") or {}).get("number_fit_direct"),
            "number_with_transition_plausible": (payload.get("summary") or {}).get("number_with_transition_plausible"),
            "number_review_needed": (payload.get("summary") or {}).get("number_review_needed"),
            "average_transferability_score": (payload.get("summary") or {}).get("average_transferability_score"),
            "lookup_success_rate": (payload.get("summary") or {}).get("lookup_success_rate"),
        },
        "methodological_note": METHODOLOGICAL_NOTE,
    }


def compact_candidate(candidate: dict[str, Any], *, mode: str) -> dict[str, Any]:
    item = {
        "candidate_id": candidate.get("candidate_id"),
        "profile_id": candidate.get("profile_id"),
        "baseline_score_v3": candidate.get("baseline_score_v3"),
        "rf_score": candidate.get("rf_score"),
        "xgboost_score": candidate.get("xgboost_score"),
        "recommendation_status": candidate.get("recommendation_status"),
        "transferability_score": candidate.get("transferability_score"),
        "transitions_plausibles": candidate.get("transitions_plausibles", []),
        "gaps_compensables": candidate.get("gaps_compensables", []),
        "gaps_bloquants": candidate.get("gaps_bloquants", []),
        "short_decision_summary": candidate.get("short_decision_summary"),
    }
    item["executive_reason"] = executive_reason(candidate, mode=mode)
    return item


def executive_reason(candidate: dict[str, Any], *, mode: str) -> str:
    if mode == "recommended":
        reasons: list[str] = []
        if _score(candidate.get("baseline_score_v3")) >= 0.6:
            reasons.append("score Matching V3 élevé")
        if _score(candidate.get("rf_score")) >= 0.8:
            reasons.append("Random Forest très favorable")
        if _score(candidate.get("xgboost_score")) >= 0.8:
            reasons.append("XGBoost favorable")
        if not candidate.get("gaps_bloquants"):
            reasons.append("pas de gap bloquant détecté")
        if candidate.get("transitions_plausibles"):
            reasons.append("transition métier plausible")
        return "Recommandé car " + ", ".join(reasons[:4]) + "."

    reasons = []
    if candidate.get("recommendation_status") == "review_needed":
        reasons.append("statut review_needed")
    if _rank_gap(candidate) >= 10:
        reasons.append("désaccord important entre les rangs")
    if candidate.get("gaps_bloquants"):
        reasons.append("gaps bloquants: " + ", ".join(candidate["gaps_bloquants"][:3]))
    if _score(candidate.get("xgboost_score")) < 0.1:
        reasons.append("score XGBoost très faible")
    return "À vérifier car " + ", ".join(reasons[:4]) + "."


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Executive summary - Smart Recruiter",
        "",
        "## Objectif",
        "",
        "Fournir une version courte de la démo: 3 candidats recommandés, 3 candidats à vérifier, signaux clés et limites.",
        "",
        "## Résumé court",
        "",
        f"- `job_id`: `{summary['job_id']}`",
        f"- candidats recommandés: `{len(summary['top_recommended'])}`",
        f"- candidats à vérifier: `{len(summary['needs_review'])}`",
        f"- transitions plausibles dans le top 10 source: `{summary['key_metrics'].get('number_with_transition_plausible')}`",
        f"- review_needed dans le top 10 source: `{summary['key_metrics'].get('number_review_needed')}`",
        "",
        "## Top 3 candidats recommandés",
        "",
    ]
    lines.extend(_candidate_section(summary["top_recommended"]))
    lines.extend(["", "## Top 3 candidats à vérifier", ""])
    lines.extend(_candidate_section(summary["needs_review"]))
    lines.extend(
        [
            "",
            "## Signaux clés utilisés",
            "",
            "- Score et rang Matching V3.",
            "- Score Random Forest.",
            "- Score XGBoost et SHAP top features dans le rapport source.",
            "- Potential Graph: transférabilité, transitions plausibles, gaps compensables et bloquants.",
            "",
            "## Limites méthodologiques",
            "",
            f"- {summary['methodological_note']}",
            "- Cette synthèse est faite pour une démo rapide et ne constitue pas une décision recruteur.",
            "",
            "## Prochaine étape",
            "",
            "Valider les candidats recommandés et les candidats à vérifier avec un recruteur ou un expert métier.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(summary), encoding="utf-8")


def _candidate_section(candidates: list[dict[str, Any]]) -> list[str]:
    if not candidates:
        return ["- Aucun candidat sélectionné."]
    lines: list[str] = []
    for candidate in candidates:
        lines.extend(
            [
                f"### `{candidate['candidate_id']}`",
                "",
                f"- Matching V3: `{_fmt(candidate.get('baseline_score_v3'))}`",
                f"- Random Forest: `{_fmt(candidate.get('rf_score'))}`",
                f"- XGBoost: `{_fmt(candidate.get('xgboost_score'))}`",
                f"- Transferability: `{_fmt(candidate.get('transferability_score'))}`",
                f"- Gaps bloquants: {', '.join(candidate.get('gaps_bloquants') or []) or 'Aucun'}",
                f"- Raison executive: {candidate['executive_reason']}",
                "",
            ]
        )
    return lines


def _recommended_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        -_score(candidate.get("baseline_score_v3")),
        -_score(candidate.get("rf_score")),
        -_score(candidate.get("xgboost_score")),
        -_score(candidate.get("transferability_score")),
        len(candidate.get("gaps_bloquants") or []),
    )


def _review_sort_key(candidate: dict[str, Any]) -> tuple[int, int, float]:
    return (
        -_rank_gap(candidate),
        -(len(candidate.get("gaps_bloquants") or [])),
        _score(candidate.get("xgboost_score")),
    )


def _rank_gap(candidate: dict[str, Any]) -> int:
    ranks = [
        candidate.get("baseline_rank_v3"),
        candidate.get("rf_rank"),
        candidate.get("xgboost_rank"),
    ]
    numeric = [int(rank) for rank in ranks if rank is not None]
    if len(numeric) < 2:
        return 0
    return max(numeric) - min(numeric)


def _score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    source = read_json(args.input)
    summary = build_executive_summary(source, args.input)
    write_json(args.output_json, summary)
    write_markdown(args.output_md, summary)
    print(f"Executive summary JSON written: {args.output_json}")
    print(f"Executive summary Markdown written: {args.output_md}")


if __name__ == "__main__":
    main()

