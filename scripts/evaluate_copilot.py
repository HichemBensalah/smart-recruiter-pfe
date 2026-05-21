from __future__ import annotations

import argparse
import math
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.tools import StructuredTool


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.main import app
from src.core.chatbot.memory import SESSION_STORE
from src.core.chatbot.tools.schemas import (
    CandidateProfileInput,
    DecisionCardInput,
    MatchCandidatesInput,
    Neo4jTransferabilityInput,
    TransferabilityInput,
)


DEFAULT_SCENARIOS = ROOT / "data/evaluation/copilot_eval_scenarios.json"
DEFAULT_OUTPUT_JSON = ROOT / "docs/reports/copilot/copilot_evaluation.json"
DEFAULT_OUTPUT_MD = ROOT / "docs/reports/copilot/copilot_evaluation.md"

REQUIRED_COVERAGE = {
    "new_offer_start",
    "six_field_collection",
    "pre_confirmation_correction",
    "positive_confirmation",
    "matching_with_routed_job_id",
    "followup_explain_first_candidate",
    "compare_first_two",
    "best_candidate_gaps",
    "neo4j_yaml_fallback",
    "dependency_or_artifact_fallback",
}


@dataclass
class EvaluationContext:
    client: TestClient
    session_ids: dict[str, str]
    first_candidate_id: str | None = None
    last_payload_by_session: dict[str, dict[str, Any]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Smart Recruiter Copilot on reproducible demo scenarios.")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
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
    with without_external_services(("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")):
        client = TestClient(app)
        install_testclient_tools(client)
        context = EvaluationContext(client=client, session_ids={}, last_payload_by_session={})
        _clear_sessions(scenarios)
        results = [evaluate_one_scenario(scenario, context) for scenario in scenarios]

    return build_report(results)


def evaluate_one_scenario(scenario: dict[str, Any], context: EvaluationContext) -> dict[str, Any]:
    scenario_type = str(scenario.get("type") or "chat")
    if scenario_type == "endpoint":
        return evaluate_endpoint_scenario(scenario, context)
    return evaluate_chat_scenario(scenario, context)


def evaluate_chat_scenario(scenario: dict[str, Any], context: EvaluationContext) -> dict[str, Any]:
    session_key = str(scenario.get("session") or "default")
    session_id = context.session_ids.setdefault(session_key, f"eval-{session_key}")
    message = str(scenario.get("message") or "")

    started = time.perf_counter()
    response = context.client.post("/api/chat", json={"message": message, "session_id": session_id})
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    payload = _safe_json(response)
    if isinstance(payload, dict):
        context.last_payload_by_session[session_key] = payload
        candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
        if candidates and not context.first_candidate_id:
            first = candidates[0]
            if isinstance(first, dict) and first.get("candidate_id"):
                context.first_candidate_id = str(first["candidate_id"])

    criteria = common_chat_criteria(scenario, response.status_code, payload)
    score = score_criteria(criteria)
    return {
        "id": str(scenario.get("id") or ""),
        "type": "chat",
        "title": str(scenario.get("title") or ""),
        "message": message,
        "coverage": _as_str_list(scenario.get("coverage")),
        "latency_ms": latency_ms,
        "status_code": response.status_code,
        "scenario_score": score,
        "passed": score >= 0.8,
        "criteria": criteria,
        "answer_excerpt": str(payload.get("answer") or "")[:700] if isinstance(payload, dict) else "",
        "candidate_ids": _candidate_ids(payload),
        "sources": _as_str_list(payload.get("sources")) if isinstance(payload, dict) else [],
        "warnings": _as_str_list(payload.get("warnings")) if isinstance(payload, dict) else [],
        "matching_metadata": payload.get("matching_metadata") if isinstance(payload, dict) else {},
    }


def common_chat_criteria(scenario: dict[str, Any], status_code: int, payload: Any) -> dict[str, Any]:
    payload_dict = payload if isinstance(payload, dict) else {}
    answer = str(payload_dict.get("answer") or "")
    candidates = payload_dict.get("candidates") if isinstance(payload_dict.get("candidates"), list) else []
    sources = _as_str_list(payload_dict.get("sources"))
    expected_sources = _as_str_list(scenario.get("expected_sources"))
    expected_fields = _as_str_list(scenario.get("expected_fields"))
    expected_flags = scenario.get("expected_flags") if isinstance(scenario.get("expected_flags"), dict) else {}
    expected_metadata = _as_str_list(scenario.get("expected_matching_metadata"))
    expected_terms = _as_str_list(scenario.get("expected_answer_contains"))
    candidate_min = int(scenario.get("expected_candidates_min") or 0)

    return {
        "status_ok": status_code == int(scenario.get("expected_status", 200)),
        "has_answer": bool(answer.strip()),
        "expected_terms": expected_terms_result(answer, expected_terms),
        "expected_sources_present": subset_result(expected_sources, sources),
        "expected_fields_present": fields_result(payload_dict, expected_fields),
        "expected_flags_match": flags_result(payload_dict, expected_flags),
        "expected_matching_metadata_present": fields_result(payload_dict.get("matching_metadata") or {}, expected_metadata),
        "candidate_count_ok": len(candidates) >= candidate_min,
        "no_fake_candidate_ids": no_fake_candidate_ids(answer, candidates),
        "memory_coherent": memory_coherent(scenario, sources, payload_dict),
    }


def evaluate_endpoint_scenario(scenario: dict[str, Any], context: EvaluationContext) -> dict[str, Any]:
    method = str(scenario.get("method") or "GET").upper()
    path = replace_placeholders(str(scenario.get("path") or ""), context)
    params = scenario.get("params") if isinstance(scenario.get("params"), dict) else None
    json_payload = scenario.get("json") if isinstance(scenario.get("json"), dict) else None

    started = time.perf_counter()
    if method == "POST":
        response = context.client.post(path, json=json_payload, params=params)
    else:
        response = context.client.get(path, params=params)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    payload = _safe_json(response)
    payload_dict = payload if isinstance(payload, dict) else {}

    expected_fields = _as_str_list(scenario.get("expected_fields"))
    expected_flags = scenario.get("expected_flags") if isinstance(scenario.get("expected_flags"), dict) else {}
    expected_terms = _as_str_list(scenario.get("expected_answer_contains"))
    text_blob = json.dumps(payload, ensure_ascii=False)
    criteria = {
        "status_ok": response.status_code == int(scenario.get("expected_status", 200)),
        "has_json_payload": isinstance(payload, dict),
        "expected_terms": expected_terms_result(text_blob, expected_terms),
        "expected_fields_present": fields_result(payload_dict, expected_fields),
        "expected_flags_match": flags_result(payload_dict, expected_flags),
    }
    score = score_criteria(criteria)
    return {
        "id": str(scenario.get("id") or ""),
        "type": "endpoint",
        "title": str(scenario.get("title") or ""),
        "message": f"{method} {path}",
        "coverage": _as_str_list(scenario.get("coverage")),
        "latency_ms": latency_ms,
        "status_code": response.status_code,
        "scenario_score": score,
        "passed": score >= 0.8,
        "criteria": criteria,
        "answer_excerpt": text_blob[:700],
        "candidate_ids": _candidate_ids(payload),
        "sources": [],
        "warnings": _as_str_list(payload_dict.get("warnings")),
        "matching_metadata": payload_dict.get("matching_metadata") or {},
    }


def build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    chat_results = [result for result in results if result["type"] == "chat"]
    passed = [result for result in results if result["passed"]]
    failed = [result for result in results if not result["passed"]]
    latencies = [float(result["latency_ms"]) for result in chat_results]
    coverage = scenario_coverage(results)
    fallback_results = [result for result in results if any("fallback" in tag for tag in result["coverage"])]
    memory_results = [result for result in results if result["criteria"].get("memory_coherent") is not None]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture_evaluated": "FastAPI TestClient /api/chat + LangGraph tools bridged to local FastAPI routes",
        "ci_compatibility": {
            "uses_testclient": True,
            "requires_neo4j": False,
            "requires_mongodb": False,
            "requires_docker": False,
            "uses_matching_artifacts": True,
        },
        "total_scenarios": len(results),
        "passed_scenarios": len(passed),
        "failed_scenarios": len(failed),
        "average_score": round(sum(result["scenario_score"] for result in results) / len(results), 4),
        "tool_calling_accuracy": tool_calling_accuracy(results),
        "hallucination_free_rate": hallucination_free_rate(results),
        "average_chat_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "p95_chat_latency_ms": percentile(latencies, 95) if latencies else None,
        "memory_coherence_rate": rate(memory_results, lambda item: bool(item["criteria"].get("memory_coherent"))),
        "scenario_coverage": coverage,
        "fallback_coverage": {
            "covered": bool(fallback_results),
            "passed": sum(1 for result in fallback_results if result["passed"]),
            "total": len(fallback_results),
            "rate": rate(fallback_results, lambda item: item["passed"]),
        },
        "known_limitations": build_known_limitations(results),
        "results": results,
        "conclusion": (
            "Le Copilot est démontrable de bout en bout sur un flow recruteur contrôlé : création d'offre, "
            "correction avant confirmation, matching via routed_job_id, mémoire courte, questions de suivi, "
            "Decision Cards et fallback YAML lorsque Neo4j n'est pas disponible. L'évaluation reste volontairement "
            "rapide et locale pour ne pas rendre la CI dépendante de services externes."
        ),
    }


def tool_calling_accuracy(results: list[dict[str, Any]]) -> float:
    expected = 0
    matched = 0
    for result in results:
        subset = result["criteria"].get("expected_sources_present")
        if not isinstance(subset, dict):
            continue
        expected += int(subset.get("total") or 0)
        matched += int(subset.get("count") or 0)
    return round(matched / expected, 4) if expected else 1.0


def hallucination_free_rate(results: list[dict[str, Any]]) -> float:
    return rate(results, lambda item: bool(item["criteria"].get("no_fake_candidate_ids", True)))


def scenario_coverage(results: list[dict[str, Any]]) -> dict[str, Any]:
    covered = {tag for result in results for tag in result["coverage"] if result["passed"]}
    missing = sorted(REQUIRED_COVERAGE - covered)
    return {
        "required": sorted(REQUIRED_COVERAGE),
        "covered": sorted(covered),
        "missing": missing,
        "coverage_rate": round((len(REQUIRED_COVERAGE) - len(missing)) / len(REQUIRED_COVERAGE), 4),
    }


def build_known_limitations(results: list[dict[str, Any]]) -> list[str]:
    limitations = [
        "Mémoire courte en RAM avec TTL : ce n'est pas une mémoire longue ou multi-utilisateur persistée.",
        "Matching V3 est évalué via des artefacts pré-générés, pas via un recalcul live FAISS/MongoDB.",
        "Neo4j est volontairement optionnel dans cette évaluation ; le fallback YAML est le comportement attendu en CI.",
        "Les scores Random Forest et XGBoost restent expérimentaux car entraînés sur pseudo-labels métier.",
    ]
    if any(not result["passed"] for result in results):
        limitations.append("Certains scénarios n'ont pas atteint le seuil de réussite de 0.8.")
    if scenario_coverage(results)["missing"]:
        limitations.append("La couverture de scénarios n'est pas complète.")
    return limitations


def install_testclient_tools(client: TestClient) -> None:
    import src.core.chatbot.nodes.analyze_transferability as graph_node
    import src.core.chatbot.nodes.fetch_decision_cards as cards_node
    import src.core.chatbot.nodes.match_candidates as match_node

    def local_match(job_description: str, top_k: int = 10, job_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"job_description": job_description, "top_k": top_k}
        if job_id:
            payload["job_id"] = job_id
        return response_json_or_raise(client.post("/api/match", json=payload))

    def local_decision_card(candidate_id: str) -> dict[str, Any]:
        return response_json_or_raise(client.get(f"/api/decision-cards/{candidate_id}"))

    def local_candidate_profile(candidate_id: str) -> dict[str, Any]:
        return response_json_or_raise(client.get(f"/api/candidates/{candidate_id}"))

    def local_transferability(candidate_id: str) -> dict[str, Any]:
        return response_json_or_raise(client.get(f"/api/graph/transferability/{candidate_id}"))

    def local_neo4j_transferability(candidate_id: str, target_role: str = "Backend Developer") -> dict[str, Any]:
        response = client.get(f"/api/graph/neo4j/transferability/{candidate_id}", params={"target_role": target_role})
        if response.status_code >= 400:
            return {"available": False, "message": str(_safe_json(response)), "fallback_recommended": True}
        return response_json_or_raise(response)

    match_node.match_candidates_tool = StructuredTool.from_function(
        func=local_match,
        name="match_candidates",
        description="Local TestClient bridge for /api/match.",
        args_schema=MatchCandidatesInput,
    )
    cards_node.get_decision_card_tool = StructuredTool.from_function(
        func=local_decision_card,
        name="get_decision_card",
        description="Local TestClient bridge for /api/decision-cards/{candidate_id}.",
        args_schema=DecisionCardInput,
    )
    cards_node.get_candidate_profile_tool = StructuredTool.from_function(
        func=local_candidate_profile,
        name="get_candidate_profile",
        description="Local TestClient bridge for /api/candidates/{candidate_id}.",
        args_schema=CandidateProfileInput,
    )
    graph_node.get_transferability_tool = StructuredTool.from_function(
        func=local_transferability,
        name="get_transferability",
        description="Local TestClient bridge for /api/graph/transferability/{candidate_id}.",
        args_schema=TransferabilityInput,
    )
    graph_node.get_neo4j_transferability_tool = StructuredTool.from_function(
        func=local_neo4j_transferability,
        name="get_neo4j_transferability",
        description="Local TestClient bridge for optional Neo4j transferability.",
        args_schema=Neo4jTransferabilityInput,
    )


def response_json_or_raise(response: Any) -> dict[str, Any]:
    payload = _safe_json(response)
    if response.status_code >= 400:
        raise RuntimeError(payload)
    if not isinstance(payload, dict):
        raise RuntimeError("Expected a JSON object")
    return payload


def expected_terms_result(text: str, terms: list[str]) -> dict[str, Any]:
    normalized_text = normalize_text(text)
    matched = [term for term in terms if normalize_text(term) in normalized_text]
    return {
        "matched": matched,
        "missing": [term for term in terms if term not in matched],
        "count": len(matched),
        "total": len(terms),
        "ratio": round(len(matched) / len(terms), 4) if terms else 1.0,
        "passed": len(matched) == len(terms),
    }


def subset_result(expected: list[str], actual: list[str]) -> dict[str, Any]:
    actual_set = set(actual)
    matched = [item for item in expected if item in actual_set]
    return {
        "matched": matched,
        "missing": [item for item in expected if item not in actual_set],
        "count": len(matched),
        "total": len(expected),
        "ratio": round(len(matched) / len(expected), 4) if expected else 1.0,
        "passed": len(matched) == len(expected),
    }


def fields_result(payload: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    matched = [field for field in fields if has_path(payload, field)]
    return {
        "matched": matched,
        "missing": [field for field in fields if field not in matched],
        "count": len(matched),
        "total": len(fields),
        "ratio": round(len(matched) / len(fields), 4) if fields else 1.0,
        "passed": len(matched) == len(fields),
    }


def flags_result(payload: dict[str, Any], expected_flags: dict[str, Any]) -> dict[str, Any]:
    matched: list[str] = []
    missing_or_wrong: dict[str, Any] = {}
    for field, expected in expected_flags.items():
        actual = get_path(payload, field)
        if actual == expected:
            matched.append(field)
        else:
            missing_or_wrong[field] = {"expected": expected, "actual": actual}
    return {
        "matched": matched,
        "missing_or_wrong": missing_or_wrong,
        "count": len(matched),
        "total": len(expected_flags),
        "ratio": round(len(matched) / len(expected_flags), 4) if expected_flags else 1.0,
        "passed": not missing_or_wrong,
    }


def score_criteria(criteria: dict[str, Any]) -> float:
    checks: list[float] = []
    for value in criteria.values():
        if isinstance(value, bool):
            checks.append(1.0 if value else 0.0)
        elif isinstance(value, dict) and "ratio" in value:
            checks.append(float(value["ratio"]))
        elif value is None:
            continue
    return round(sum(checks) / len(checks), 4) if checks else 0.0


def no_fake_candidate_ids(answer: str, candidates: list[Any]) -> bool:
    mentioned = set(re.findall(r"candidate_[A-Za-z0-9]{8,}", answer))
    structured = {
        str(candidate.get("candidate_id"))
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("candidate_id")
    }
    return mentioned.issubset(structured)


def memory_coherent(scenario: dict[str, Any], sources: list[str], payload: dict[str, Any]) -> bool | None:
    if not scenario.get("requires_memory"):
        return None
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    return "conversation_memory" in sources and bool(candidates)


def replace_placeholders(path: str, context: EvaluationContext) -> str:
    if "{first_candidate_id}" in path:
        candidate_id = context.first_candidate_id or "candidate_b6f7add66ffc"
        return path.replace("{first_candidate_id}", candidate_id)
    return path


def has_path(payload: dict[str, Any], dotted_path: str) -> bool:
    value = get_path(payload, dotted_path)
    if value is None:
        return False
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _candidate_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    candidates = payload.get("candidates") or payload.get("items") or []
    if isinstance(candidates, list):
        return [
            str(candidate.get("candidate_id"))
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("candidate_id")
        ]
    candidate_id = payload.get("candidate_id")
    return [str(candidate_id)] if candidate_id else []


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _safe_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return {"raw": getattr(response, "text", "")}


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.casefold()


def rate(items: list[Any], predicate: Any) -> float:
    if not items:
        return 1.0
    return round(sum(1 for item in items if predicate(item)) / len(items), 4)


def percentile(values: list[float], percent: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((percent / 100) * len(ordered)) - 1))
    return round(ordered[index], 2)


def _clear_sessions(scenarios: list[dict[str, Any]]) -> None:
    for scenario in scenarios:
        session = scenario.get("session")
        if session:
            SESSION_STORE.clear(f"eval-{session}")


@contextmanager
def without_external_services(names: tuple[str, ...]):
    original = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
        "Vérifier le flow recruteur complet, la mémoire courte, le matching via `routed_job_id`, les questions de suivi et les fallbacks sans dépendre de Neo4j, MongoDB ou Docker réels.",
        "",
        "## Méthode",
        "",
        "- Exécution via `fastapi.testclient.TestClient`.",
        "- Tools LangGraph bridgés vers les routes FastAPI locales.",
        "- Variables Neo4j retirées pendant l'évaluation pour forcer le fallback YAML.",
        "- Matching V3 évalué via les artefacts pré-générés `data/ranking/features/*.jsonl`.",
        "",
        "## Métriques globales",
        "",
        f"- Scénarios : `{report['total_scenarios']}`",
        f"- Réussis : `{report['passed_scenarios']}`",
        f"- Échoués/faibles : `{report['failed_scenarios']}`",
        f"- Score moyen : `{report['average_score']}`",
        f"- Tool calling accuracy : `{report['tool_calling_accuracy']}`",
        f"- Taux de réponses sans hallucination : `{report['hallucination_free_rate']}`",
        f"- Latence moyenne `/api/chat` : `{report['average_chat_latency_ms']} ms`",
        f"- P95 `/api/chat` : `{report['p95_chat_latency_ms']} ms`",
        f"- Cohérence mémoire : `{report['memory_coherence_rate']}`",
        f"- Couverture des scénarios : `{report['scenario_coverage']['coverage_rate']}`",
        f"- Couverture fallback : `{report['fallback_coverage']['rate']}`",
        "",
        "## Couverture",
        "",
    ]
    lines.extend(f"- {tag}: {'OK' if tag in report['scenario_coverage']['covered'] else 'MANQUANT'}" for tag in report["scenario_coverage"]["required"])
    lines.extend(
        [
            "",
            "## Résultat par scénario",
            "",
            "| Scénario | Type | Score | Statut | Latence ms | Couverture | Sources |",
            "| --- | --- | ---: | --- | ---: | --- | --- |",
        ]
    )
    for result in report["results"]:
        status = "OK" if result["passed"] else "À améliorer"
        lines.append(
            f"| `{result['id']}` | `{result['type']}` | {result['scenario_score']:.4f} | {status} | "
            f"{result['latency_ms']} | {', '.join(result['coverage']) or 'n/a'} | {', '.join(result['sources']) or 'n/a'} |"
        )
    lines.extend(["", "## Warnings et observations", ""])
    for result in report["results"]:
        warnings = result["warnings"] or []
        missing_terms = result["criteria"].get("expected_terms", {}).get("missing", [])
        lines.extend(
            [
                f"### {result['id']}",
                "",
                f"- Message/action : {result['message']}",
                f"- Candidats structurés : `{result['candidate_ids']}`",
                f"- Termes manquants : `{missing_terms}`",
                f"- Warnings : `{warnings}`",
                "",
            ]
        )
    lines.extend(["## Limites connues", ""])
    lines.extend(f"- {limitation}" for limitation in report["known_limitations"])
    lines.extend(["", "## Conclusion", "", report["conclusion"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    scenarios = load_scenarios(args.scenarios)
    report = evaluate_scenarios(scenarios)
    write_json(args.output_json, report)
    write_markdown(args.output_md, report)
    print(f"Copilot evaluation average_score={report['average_score']}")
    print(f"Tool calling accuracy={report['tool_calling_accuracy']}")
    print(f"Average /api/chat latency={report['average_chat_latency_ms']} ms")
    print(f"JSON report: {args.output_json}")
    print(f"Markdown report: {args.output_md}")


if __name__ == "__main__":
    main()
