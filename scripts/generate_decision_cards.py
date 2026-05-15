from __future__ import annotations

import json
from pathlib import Path
from typing import Any


INPUT_REPORT_PATH = Path("docs/reports/matching/v3/matching_single_job_report_grounded_v3.json")
OUTPUT_PATH = Path("docs/reports/matching/v3/decision_cards_grounded_v3.json")


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


def display_name_for(candidate: dict[str, Any]) -> str:
    candidate_id = candidate.get("candidate_id")
    name = candidate.get("display_name") or candidate.get("full_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    if candidate_id:
        return f"Candidate (ID: {candidate_id})"
    return "Candidate"


def verdict_for(final_score: float, risk: str) -> str:
    if final_score >= 0.75 and risk == "low":
        return "Présélection recommandée"
    if final_score >= 0.50:
        return "À considérer"
    return "Non prioritaire"


def build_strengths(
    *,
    matched_skills: list[str],
    must_have_coverage: float,
    risk: str,
    profile_kind: str,
    reliability_score: float | None,
) -> list[str]:
    strengths: list[str] = []
    if matched_skills:
        strengths.append(f"Compétences correspondantes : {', '.join(matched_skills)}")
    if must_have_coverage >= 0.8:
        strengths.append("Très bonne couverture des compétences obligatoires")
    elif must_have_coverage >= 0.6:
        strengths.append("Couverture correcte des compétences obligatoires")
    if risk == "low":
        strengths.append("Risque d’hallucination faible")
    if profile_kind == "complete_profile":
        strengths.append("Profil complet")
    if reliability_score is not None and reliability_score >= 0.85:
        strengths.append("Profil très fiable selon le score de fiabilité")
    return strengths


def build_weaknesses(
    *,
    display_name: str,
    missing_required_skills: list[str],
    must_have_coverage: float,
    risk: str,
    profile_kind: str,
) -> list[str]:
    weaknesses: list[str] = []
    if missing_required_skills:
        weaknesses.append(f"Compétences obligatoires manquantes : {', '.join(missing_required_skills)}")
    if must_have_coverage < 0.5:
        weaknesses.append("Couverture faible des compétences obligatoires")
    if risk == "medium":
        weaknesses.append("Profil à vérifier : risque d’hallucination moyen")
    if risk == "high":
        weaknesses.append("Profil risqué : vérification humaine nécessaire")
    if profile_kind == "partial_profile":
        weaknesses.append("Profil partiel")
    if profile_kind == "minimal_profile":
        weaknesses.append("Profil minimal")
    if "Candidate (ID:" in display_name:
        weaknesses.append("Nom candidat absent ou non fiable")
    return weaknesses


def build_interview_focus(matched_skills: list[str], missing_required_skills: list[str], risk: str) -> list[str]:
    focus: list[str] = []
    for skill in missing_required_skills:
        focus.append(f"Vérifier la compétence manquante : {skill}")
    if "REST API" in missing_required_skills:
        focus.append("Valider l’expérience REST API en pratique")
    if "Docker" in matched_skills:
        focus.append("Confirmer l’usage de Docker sur un projet réel")
    if "FastAPI" in matched_skills:
        focus.append("Demander un exemple concret de projet FastAPI")
    if risk != "low":
        focus.append("Vérifier manuellement les informations importantes du CV")
    return focus[:3]


def build_explanation(
    *,
    display_name: str,
    rank: int,
    verdict: str,
    matched_skills: list[str],
    missing_required_skills: list[str],
    must_have_coverage: float,
    risk: str,
    profile_kind: str,
) -> str:
    if verdict == "Non prioritaire":
        return (
            f"{display_name} est moins prioritaire car la couverture des compétences obligatoires est "
            f"{'faible' if must_have_coverage < 0.5 else 'limitée'} et plusieurs compétences importantes peuvent manquer."
        )

    matched_text = ", ".join(matched_skills[:4]) if matched_skills else "des compétences pertinentes"
    quality_bits = []
    if profile_kind == "complete_profile":
        quality_bits.append("son profil est complet")
    if risk == "low":
        quality_bits.append("le risque d’hallucination est faible")
    quality_text = " et ".join(quality_bits) if quality_bits else f"le risque qualité est {risk}"

    if missing_required_skills:
        missing_text = ", ".join(missing_required_skills[:2])
        return (
            f"{display_name} est classé #{rank} car il couvre une partie importante des compétences obligatoires, "
            f"notamment {matched_text}. {quality_text.capitalize()}. "
            f"La compétence {missing_text} doit être vérifiée en entretien."
        )
    return (
        f"{display_name} est classé #{rank} car il couvre bien les compétences obligatoires, "
        f"notamment {matched_text}. {quality_text.capitalize()}."
    )


def build_decision_card(candidate: dict[str, Any]) -> dict[str, Any]:
    rank = int(candidate.get("rank") or 0)
    candidate_id = candidate.get("candidate_id")
    display_name = display_name_for(candidate)
    final_score = float(candidate.get("final_score") or 0.0)
    risk = str(candidate.get("hallucination_risk") or "unknown")
    profile_kind = str(candidate.get("profile_kind") or "unknown")
    reliability_score = as_float_or_none(candidate.get("reliability_score"))
    must_have_coverage = float(candidate.get("must_have_coverage") or 0.0)
    matched_skills = as_list(candidate.get("matched_skills"))
    missing_required_skills = as_list(candidate.get("missing_required_skills"))
    verdict = verdict_for(final_score, risk)

    strengths = build_strengths(
        matched_skills=matched_skills,
        must_have_coverage=must_have_coverage,
        risk=risk,
        profile_kind=profile_kind,
        reliability_score=reliability_score,
    )
    weaknesses = build_weaknesses(
        display_name=display_name,
        missing_required_skills=missing_required_skills,
        must_have_coverage=must_have_coverage,
        risk=risk,
        profile_kind=profile_kind,
    )
    interview_focus = build_interview_focus(matched_skills, missing_required_skills, risk)
    explanation = build_explanation(
        display_name=display_name,
        rank=rank,
        verdict=verdict,
        matched_skills=matched_skills,
        missing_required_skills=missing_required_skills,
        must_have_coverage=must_have_coverage,
        risk=risk,
        profile_kind=profile_kind,
    )

    return {
        "rank": rank,
        "candidate_id": candidate_id,
        "display_name": display_name,
        "verdict": verdict,
        "final_score": final_score,
        "risk": risk,
        "profile_kind": profile_kind,
        "reliability_score": reliability_score,
        "must_have_coverage": must_have_coverage,
        "matched_skills": matched_skills,
        "missing_required_skills": missing_required_skills,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "interview_focus": interview_focus,
        "explanation": explanation,
    }


def extract_top10_recommendations(report: dict[str, Any]) -> list[dict[str, Any]]:
    results = report.get("results") or []
    if not results:
        return []
    recommendations = results[0].get("recommendations") or []
    return list(recommendations[:10])


def main() -> None:
    report = read_json(INPUT_REPORT_PATH)
    recommendations = extract_top10_recommendations(report)
    cards = [build_decision_card(candidate) for candidate in recommendations]
    payload = {
        "source_report": str(INPUT_REPORT_PATH),
        "cards_count": len(cards),
        "decision_cards": cards,
    }
    write_json(OUTPUT_PATH, payload)
    print(json.dumps({"output": str(OUTPUT_PATH), "cards_count": len(cards)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
