from __future__ import annotations

import json
from pathlib import Path
from typing import Any


INPUT_REPORT_PATH = Path("docs/reports/matching/v3/matching_report_v3_normalized.json")
OUTPUT_PATH = Path("docs/reports/matching/v3/decision_cards_v3_normalized.json")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def as_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "", [], {})]


def as_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def candidate_name_for(rec: dict[str, Any]) -> str:
    name = rec.get("full_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    candidate_id = rec.get("candidate_id")
    if isinstance(candidate_id, str) and candidate_id.strip():
        return f"Candidate (ID: {candidate_id})"
    return "Candidate"


def verdict_for(must_have_coverage: float, final_score: float) -> str:
    # "score élevé" explicité de façon déterministe pour éviter l'ambiguïté.
    if must_have_coverage >= 0.8 and final_score >= 0.65:
        return "Présélection recommandée"
    if must_have_coverage >= 0.6:
        return "À considérer avec vérification"
    return "Non prioritaire — gaps importants"


def build_strengths(rec: dict[str, Any], matched_skills: list[str], must_have_coverage: float) -> list[str]:
    strengths: list[str] = []
    if matched_skills:
        strengths.append(f"Compétences alignées: {', '.join(matched_skills)}")
    if must_have_coverage >= 0.8:
        strengths.append("Bonne couverture des must-have")
    elif must_have_coverage >= 0.6:
        strengths.append("Couverture partielle acceptable des must-have")
    explanation = rec.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        strengths.append("Explication de matching disponible dans le rapport")
    return strengths[:3]


def build_weaknesses(
    rec: dict[str, Any],
    missing_required_skills: list[str],
    reliability_score: float | None,
    hallucination_risk: str | None,
) -> list[str]:
    weaknesses: list[str] = []
    if missing_required_skills:
        weaknesses.append(f"Compétences requises manquantes: {', '.join(missing_required_skills)}")

    # Flag demandé si fiabilité faible OU hallucination medium/high
    if reliability_score is not None and reliability_score < 0.9:
        weaknesses.append("Flag qualité: reliability_score faible")
    if isinstance(hallucination_risk, str) and hallucination_risk.lower() in {"medium", "high"}:
        weaknesses.append(f"Flag qualité: hallucination_risk={hallucination_risk.lower()}")

    quality_flags = rec.get("quality_flags")
    if isinstance(quality_flags, list) and quality_flags:
        weaknesses.append(f"Quality flags: {', '.join(str(x) for x in quality_flags)}")

    return weaknesses[:4]


def build_interview_focus(
    matched_skills: list[str],
    missing_required_skills: list[str],
) -> list[str]:
    focus: list[str] = []
    for skill in missing_required_skills[:3]:
        focus.append(f"Valider la compétence manquante: {skill}")
    if "FastAPI" in matched_skills:
        focus.append("Demander un exemple concret d'API FastAPI en production")
    if "MongoDB" in matched_skills:
        focus.append("Valider la conception de schéma et les index MongoDB")
    return focus[:3]


def build_card(rec: dict[str, Any]) -> dict[str, Any]:
    rank = int(rec.get("rank") or 0)
    candidate_name = candidate_name_for(rec)
    candidate_id = rec.get("candidate_id") if rec.get("candidate_id") is not None else None
    final_score = float(rec.get("final_score") or 0.0)
    must_have_coverage = float(rec.get("must_have_coverage") or 0.0)

    matched_skills = as_list(rec.get("matched_skills"))
    missing_required_skills = as_list(rec.get("missing_required_skills"))

    reliability_score = as_float_or_none(rec.get("reliability_score"))
    hallucination_risk_raw = rec.get("hallucination_risk")
    hallucination_risk = str(hallucination_risk_raw) if hallucination_risk_raw is not None else None

    verdict = verdict_for(must_have_coverage, final_score)
    strengths = build_strengths(rec, matched_skills, must_have_coverage)
    weaknesses = build_weaknesses(rec, missing_required_skills, reliability_score, hallucination_risk)
    interview_focus = build_interview_focus(matched_skills, missing_required_skills)

    explanation = rec.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = (
            f"{candidate_name} classé #{rank} avec final_score={final_score:.4f} "
            f"et must_have_coverage={must_have_coverage:.2f}."
        )

    return {
        "rank": rank,
        "candidate_name": candidate_name,
        "candidate_id": candidate_id,
        "final_score": final_score,
        "must_have_coverage": must_have_coverage,
        "verdict": verdict,
        "matched_skills": matched_skills,
        "missing_required_skills": missing_required_skills,
        "reliability_score": reliability_score,
        "hallucination_risk": hallucination_risk,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "interview_focus": interview_focus,
        "explanation": explanation,
        "source_report": "matching_report_v3_normalized.json",
    }


def extract_recommendations(report: dict[str, Any]) -> list[dict[str, Any]]:
    results = report.get("results")
    if not isinstance(results, list) or not results:
        return []
    first = results[0] if isinstance(results[0], dict) else {}
    recs = first.get("recommendations")
    if isinstance(recs, list):
        return [r for r in recs if isinstance(r, dict)]
    return []


def main() -> None:
    report = read_json(INPUT_REPORT_PATH)
    recommendations = extract_recommendations(report)
    cards = [build_card(rec) for rec in recommendations]

    payload = {
        "source_report": "matching_report_v3_normalized.json",
        "cards_count": len(cards),
        "decision_cards": cards,
    }
    write_json(OUTPUT_PATH, payload)
    print(json.dumps({"output": str(OUTPUT_PATH), "cards_count": len(cards)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
