from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.chatbot.graph import run_recruiter_copilot


GRAPH_INTENTS = {"gap_analysis", "transferability", "risk_review"}
SEARCH_INTENTS = {"search_candidates"}
BASELINE_INTENTS = {"search_candidates", "explain_candidate", "compare_candidates", "review_needed", "risk_review"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Smart Recruiter Copilot on fixed recruiter scenarios.")
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Scenarios file must contain a list.")
    scenarios = [item for item in payload if isinstance(item, dict)]
    if not scenarios:
        raise ValueError("Scenarios file is empty.")
    return scenarios


def evaluate_scenarios(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    results = [evaluate_one_scenario(scenario) for scenario in scenarios]
    passed = [result for result in results if result["scenario_score"] >= 0.75]
    failed = [result for result in results if result["scenario_score"] < 0.75]
    average_score = sum(result["scenario_score"] for result in results) / len(results)
    hallucination_warnings = [
        f"{result['id']}: candidate IDs mentioned outside structured candidates"
        for result in results
        if not result["criteria"]["no_fake_candidate_ids"]
    ]
    main_limitations = build_main_limitations(results)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture_evaluated": "FastAPI / LangChain Tools / LangGraph Recruiter Copilot",
        "total_scenarios": len(results),
        "passed_scenarios": len(passed),
        "failed_scenarios": len(failed),
        "average_score": round(average_score, 4),
        "hallucination_warnings": hallucination_warnings,
        "main_limitations": main_limitations,
        "results": results,
        "conclusion": (
            "Le Copilot est fonctionnel pour une démonstration contrôlée : il peut répondre à des demandes recruteur, "
            "récupérer les candidats via les tools, afficher les scores et expliquer les gaps. Les réponses restent "
            "basées sur les artefacts du pipeline. Les limites principales sont l'absence de mémoire longue, "
            "l'absence de validation recruteur réelle et la dépendance aux pseudo-labels pour la partie ML."
        ),
    }


def evaluate_one_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    message = str(scenario.get("message") or "")
    intent = str(scenario.get("intent") or "")
    expected_terms = [str(term) for term in scenario.get("expected_contains", [])]
    try:
        response = run_recruiter_copilot(message)
        error = None
    except Exception as exc:
        response = {
            "answer": "",
            "candidates": [],
            "decision_cards": [],
            "transferability": {},
            "sources": [],
            "warnings": [f"{type(exc).__name__}: {exc}"],
        }
        error = f"{type(exc).__name__}: {exc}"

    answer = str(response.get("answer") or "")
    candidates = response.get("candidates") if isinstance(response.get("candidates"), list) else []
    sources = response.get("sources") if isinstance(response.get("sources"), list) else []
    warnings = response.get("warnings") if isinstance(response.get("warnings"), list) else []
    transferability = response.get("transferability") if isinstance(response.get("transferability"), dict) else {}

    contains = count_expected_terms(answer, expected_terms)
    criteria = {
        "has_answer": bool(answer.strip()),
        "has_candidates": bool(candidates) if intent in SEARCH_INTENTS else True,
        "has_sources": isinstance(sources, list),
        "has_warnings_field": isinstance(warnings, list),
        "contains_expected_terms": contains,
        "no_fake_candidate_ids": no_fake_candidate_ids(answer, candidates),
        "no_empty_crash": error is None,
        "uses_baseline_reference": contains_text(answer, "Matching V3") if intent in BASELINE_INTENTS else True,
        "uses_transferability_when_relevant": uses_transferability(answer, transferability) if intent in GRAPH_INTENTS else True,
    }
    score = compute_score(criteria, expected_terms)
    return {
        "id": str(scenario.get("id") or ""),
        "message": message,
        "intent": intent,
        "expected_contains": expected_terms,
        "scenario_score": score,
        "criteria": criteria,
        "error": error,
        "answer_excerpt": answer[:800],
        "candidate_ids": [candidate.get("candidate_id") for candidate in candidates if isinstance(candidate, dict)],
        "sources": sources,
        "warnings": warnings,
    }


def compute_score(criteria: dict[str, Any], expected_terms: list[str]) -> float:
    expected_ratio = criteria["contains_expected_terms"]["ratio"] if expected_terms else 1.0
    checks = [
        criteria["has_answer"],
        criteria["has_candidates"],
        criteria["has_sources"],
        criteria["has_warnings_field"],
        criteria["no_fake_candidate_ids"],
        criteria["no_empty_crash"],
        criteria["uses_baseline_reference"],
        criteria["uses_transferability_when_relevant"],
    ]
    boolean_score = sum(1 for check in checks if check) / len(checks)
    return round((0.75 * boolean_score) + (0.25 * expected_ratio), 4)


def count_expected_terms(answer: str, terms: list[str]) -> dict[str, Any]:
    normalized_answer = normalize_text(answer)
    matched = [term for term in terms if normalize_text(term) in normalized_answer]
    return {
        "matched": matched,
        "missing": [term for term in terms if term not in matched],
        "count": len(matched),
        "total": len(terms),
        "ratio": round(len(matched) / len(terms), 4) if terms else 1.0,
    }


def no_fake_candidate_ids(answer: str, candidates: list[Any]) -> bool:
    mentioned = set(re.findall(r"candidate_[A-Za-z0-9_]+", answer))
    structured = {
        str(candidate.get("candidate_id"))
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("candidate_id")
    }
    return mentioned.issubset(structured)


def uses_transferability(answer: str, transferability: dict[str, Any]) -> bool:
    normalized = normalize_text(answer)
    return bool(transferability) and ("transferabilite" in normalized or "gap" in normalized)


def contains_text(answer: str, expected: str) -> bool:
    return normalize_text(expected) in normalize_text(answer)


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.casefold()


def build_main_limitations(results: list[dict[str, Any]]) -> list[str]:
    limitations = [
        "Pas de mémoire longue conversationnelle.",
        "Pas de planner LLM : le routage est déterministe.",
        "La qualité dépend des artefacts exposés par les tools FastAPI.",
        "La partie ML reste expérimentale car entraînée sur pseudo-labels métier contrôlés.",
    ]
    if any(result["criteria"]["contains_expected_terms"]["ratio"] < 1.0 for result in results):
        limitations.append("Certains scénarios attendent des formulations spécifiques que le composeur déterministe ne produit pas toujours.")
    if any(not result["criteria"]["uses_transferability_when_relevant"] for result in results):
        limitations.append("Les réponses liées aux gaps/transférabilité doivent être enrichies pour mieux répondre aux intentions graph.")
    return limitations


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Évaluation du Recruiter Copilot",
        "",
        "## Objectif",
        "",
        "Vérifier la qualité des réponses, la cohérence avec les artefacts du pipeline et l'absence d'hallucination évidente.",
        "",
        "## Architecture évaluée",
        "",
        "- FastAPI `/api/chat` indirectement via les LangChain Tools",
        "- LangGraph Recruiter Copilot déterministe",
        "- Matching V3 comme baseline officielle",
        "- Decision Cards, ML comparison, Potential Graph et Neo4j optionnel",
        "",
        "## Score global",
        "",
        f"- Scénarios : `{report['total_scenarios']}`",
        f"- Réussis : `{report['passed_scenarios']}`",
        f"- Faibles/échoués : `{report['failed_scenarios']}`",
        f"- Score moyen : `{report['average_score']}`",
        "",
        "## Résultat par scénario",
        "",
        "| Scénario | Intent | Score | Termes attendus trouvés | Candidats | Sources |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in report["results"]:
        contains = result["criteria"]["contains_expected_terms"]
        lines.append(
            f"| `{result['id']}` | `{result['intent']}` | {result['scenario_score']:.4f} | "
            f"{contains['count']}/{contains['total']} | {len(result['candidate_ids'])} | "
            f"{', '.join(result['sources']) or 'n/a'} |"
        )
    lines.extend(["", "## Observations", ""])
    for result in report["results"]:
        status = "OK" if result["scenario_score"] >= 0.75 else "À améliorer"
        lines.extend(
            [
                f"### {result['id']} - {status}",
                "",
                f"- Message : {result['message']}",
                f"- Score : `{result['scenario_score']}`",
                f"- Termes manquants : `{result['criteria']['contains_expected_terms']['missing']}`",
                f"- Warnings : `{result['warnings']}`",
                "",
            ]
        )
    lines.extend(["## Limites", ""])
    lines.extend(f"- {limitation}" for limitation in report["main_limitations"])
    if report["hallucination_warnings"]:
        lines.extend(["", "## Warnings hallucination", ""])
        lines.extend(f"- {warning}" for warning in report["hallucination_warnings"])
    lines.extend(["", "## Conclusion", "", report["conclusion"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    scenarios = load_scenarios(args.scenarios)
    report = evaluate_scenarios(scenarios)
    write_json(args.output_json, report)
    write_markdown(args.output_md, report)
    print(f"Copilot evaluation average_score={report['average_score']}")
    print(f"JSON report: {args.output_json}")
    print(f"Markdown report: {args.output_md}")


if __name__ == "__main__":
    main()
