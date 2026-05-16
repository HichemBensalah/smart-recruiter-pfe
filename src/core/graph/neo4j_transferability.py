from __future__ import annotations

from typing import Any

from src.core.graph.neo4j_client import Neo4jClient
from src.core.graph.transferability import normalize_skill


def get_candidate_skills(candidate_id: str) -> list[str]:
    with Neo4jClient() as client:
        return client.execute_read(_query_candidate_skills, candidate_id=candidate_id)


def get_role_requirements(role_name: str) -> dict[str, list[str]]:
    with Neo4jClient() as client:
        return client.execute_read(_query_role_requirements, role_name=role_name)


def compute_candidate_role_coverage(candidate_id: str, role_name: str) -> dict[str, Any]:
    candidate_skills = get_candidate_skills(candidate_id)
    requirements = get_role_requirements(role_name)
    return build_coverage_response(candidate_id, role_name, candidate_skills, requirements)


def find_plausible_transitions(candidate_id: str, target_role: str) -> list[dict[str, Any]]:
    candidate_skills = get_candidate_skills(candidate_id)
    candidate_norms = {normalize_skill(skill) for skill in candidate_skills}
    with Neo4jClient() as client:
        transitions = client.execute_read(_query_transitions_to_role, target_role=target_role)
    plausible: list[dict[str, Any]] = []
    for transition in transitions:
        condition_skills = transition.get("condition_skills") or []
        matched = [skill for skill in condition_skills if normalize_skill(skill) in candidate_norms]
        coverage = len(matched) / len(condition_skills) if condition_skills else 0.0
        if coverage > 0:
            plausible.append(
                {
                    "from_role": transition.get("from_role"),
                    "to_role": transition.get("to_role"),
                    "condition_skills": condition_skills,
                    "matched_condition_skills": matched,
                    "condition_coverage": round(coverage, 4),
                    "rationale": transition.get("rationale") or "",
                }
            )
    return plausible


def find_missing_skills(candidate_id: str, target_role: str) -> dict[str, list[str]]:
    coverage = compute_candidate_role_coverage(candidate_id, target_role)
    return {
        "matched_skills": coverage["matched_skills"],
        "missing_skills": coverage["missing_skills"],
        "adjacent_skills": coverage["adjacent_skills"],
    }


def explain_transferability(candidate_id: str, target_role: str) -> dict[str, Any]:
    candidate_skills = get_candidate_skills(candidate_id)
    requirements = get_role_requirements(target_role)
    coverage = build_coverage_response(candidate_id, target_role, candidate_skills, requirements)
    transitions = find_plausible_transitions(candidate_id, target_role)
    return build_transferability_explanation(
        candidate_id=candidate_id,
        target_role=target_role,
        matched_skills=coverage["matched_skills"],
        missing_skills=coverage["missing_skills"],
        adjacent_skills=coverage["adjacent_skills"],
        coverage_score=coverage["coverage_score"],
        plausible_transitions=transitions,
    )


def build_coverage_response(
    candidate_id: str,
    role_name: str,
    candidate_skills: list[str],
    requirements: dict[str, list[str]],
) -> dict[str, Any]:
    candidate_norms = {normalize_skill(skill) for skill in candidate_skills}
    required_skills = requirements.get("required_skills", [])
    adjacent_skills = requirements.get("adjacent_skills", [])
    matched = [skill for skill in required_skills if normalize_skill(skill) in candidate_norms]
    missing = [skill for skill in required_skills if normalize_skill(skill) not in candidate_norms]
    coverage_score = len(matched) / len(required_skills) if required_skills else 0.0
    return {
        "candidate_id": candidate_id,
        "target_role": role_name,
        "coverage_score": round(coverage_score, 4),
        "matched_skills": matched,
        "missing_skills": missing,
        "adjacent_skills": adjacent_skills,
    }


def build_transferability_explanation(
    *,
    candidate_id: str,
    target_role: str,
    matched_skills: list[str],
    missing_skills: list[str],
    adjacent_skills: list[str],
    coverage_score: float,
    plausible_transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    adjacent_norms = {normalize_skill(skill) for skill in adjacent_skills}
    transition_condition_norms = {
        normalize_skill(skill)
        for transition in plausible_transitions
        for skill in transition.get("condition_skills", [])
    }
    gaps_compensables = [
        skill for skill in missing_skills if normalize_skill(skill) in adjacent_norms or normalize_skill(skill) in transition_condition_norms
    ]
    gaps_bloquants = [skill for skill in missing_skills if skill not in gaps_compensables]
    explanation = _build_text_explanation(target_role, coverage_score, gaps_compensables, gaps_bloquants, plausible_transitions)
    return {
        "candidate_id": candidate_id,
        "target_role": target_role,
        "coverage_score": round(float(coverage_score), 4),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "adjacent_skills": adjacent_skills,
        "plausible_transitions": plausible_transitions,
        "gaps_compensables": gaps_compensables,
        "gaps_bloquants": gaps_bloquants,
        "explanation": explanation,
    }


def _build_text_explanation(
    target_role: str,
    coverage_score: float,
    gaps_compensables: list[str],
    gaps_bloquants: list[str],
    plausible_transitions: list[dict[str, Any]],
) -> str:
    if coverage_score >= 0.7:
        return f"Le candidat couvre fortement les exigences du role {target_role} selon le graphe Neo4j."
    if plausible_transitions and gaps_compensables:
        return (
            f"Transition plausible vers {target_role}: certains gaps sont compensables "
            f"({', '.join(gaps_compensables)}), mais une validation humaine reste necessaire."
        )
    if gaps_bloquants:
        return f"Des gaps bloquants restent presents pour {target_role}: {', '.join(gaps_bloquants)}."
    return f"Analyse Neo4j disponible pour {target_role}, sans conclusion automatique de recrutement."


def _query_candidate_skills(tx: Any, candidate_id: str) -> list[str]:
    result = tx.run(
        """
        MATCH (c:Candidate {candidate_id: $candidate_id})-[:HAS_SKILL]->(s:Skill)
        RETURN s.name AS name
        ORDER BY toLower(s.name)
        """,
        candidate_id=candidate_id,
    )
    return [record["name"] for record in result]


def _query_role_requirements(tx: Any, role_name: str) -> dict[str, list[str]]:
    required = tx.run(
        """
        MATCH (:Role {name: $role_name})-[:REQUIRES_SKILL]->(s:Skill)
        RETURN s.name AS name
        ORDER BY toLower(s.name)
        """,
        role_name=role_name,
    )
    adjacent = tx.run(
        """
        MATCH (:Role {name: $role_name})-[:HAS_ADJACENT_SKILL]->(s:Skill)
        RETURN s.name AS name
        ORDER BY toLower(s.name)
        """,
        role_name=role_name,
    )
    return {
        "required_skills": [record["name"] for record in required],
        "adjacent_skills": [record["name"] for record in adjacent],
    }


def _query_transitions_to_role(tx: Any, target_role: str) -> list[dict[str, Any]]:
    result = tx.run(
        """
        MATCH (source:Role)-[rel:TRANSITIONS_TO]->(target:Role {name: $target_role})
        RETURN source.name AS from_role,
               target.name AS to_role,
               rel.condition_skills AS condition_skills,
               rel.rationale AS rationale
        ORDER BY source.name
        """,
        target_role=target_role,
    )
    return [dict(record) for record in result]
