from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml


ROLE_ALIASES = {
    "backend": "Backend Developer",
    "backend developer": "Backend Developer",
    "backend python developer": "Backend Developer",
    "frontend": "Frontend Developer",
    "frontend developer": "Frontend Developer",
    "fullstack": "Full Stack Developer",
    "full stack": "Full Stack Developer",
    "full stack developer": "Full Stack Developer",
    "data engineer": "Data Engineer",
    "data analyst": "Data Analyst",
    "machine learning engineer": "Machine Learning Engineer",
    "ml engineer": "Machine Learning Engineer",
    "devops": "DevOps Engineer",
    "devops engineer": "DevOps Engineer",
    "business intelligence analyst": "Business Intelligence Analyst",
    "bi analyst": "Business Intelligence Analyst",
}


def load_yaml_graph(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("roles"), list):
        raise ValueError("Graph YAML must contain a roles list.")
    return payload


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def role_names(graph: dict[str, Any]) -> list[str]:
    return [str(role["role"]) for role in graph.get("roles", []) if isinstance(role, dict) and role.get("role")]


def get_role(graph: dict[str, Any], role_name: str) -> dict[str, Any] | None:
    normalized = normalize_skill(role_name)
    for role in graph.get("roles", []):
        if isinstance(role, dict) and normalize_skill(role.get("role")) == normalized:
            return role
    return None


def normalize_skill(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace(".", "")
    text = re.sub(r"[^a-z0-9+#/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_skill_set(values: list[Any]) -> set[str]:
    return {normalize_skill(value) for value in values if normalize_skill(value)}


def extract_profile_skills(profile: dict[str, Any]) -> list[str]:
    expertise = ((profile.get("profile") or {}).get("expertise") or {}) if isinstance(profile.get("profile"), dict) else {}
    skills: list[str] = []
    for key in ("hard_skills", "soft_skills"):
        value = expertise.get(key)
        if isinstance(value, list):
            skills.extend(str(item) for item in value if item)
    return _dedupe_preserve_order(skills)


def extract_profile_titles(profile: dict[str, Any]) -> list[str]:
    profile_body = profile.get("profile") if isinstance(profile.get("profile"), dict) else {}
    experiences = profile_body.get("experiences") if isinstance(profile_body, dict) else []
    titles: list[str] = []
    if isinstance(experiences, list):
        for experience in experiences:
            if isinstance(experience, dict) and experience.get("job_title"):
                titles.append(str(experience["job_title"]))
    return _dedupe_preserve_order(titles)


def extract_job_required_skills(job: dict[str, Any]) -> list[str]:
    for key in ("required_skills", "must_have_skills"):
        value = job.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if item]
    nested = ((job.get("skills") or {}).get("required") or []) if isinstance(job.get("skills"), dict) else []
    return [str(item) for item in nested if item] if isinstance(nested, list) else []


def extract_job_adjacent_skills(job: dict[str, Any]) -> list[str]:
    for key in ("nice_to_have_skills", "optional_skills"):
        value = job.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if item]
    return []


def infer_target_role(job: dict[str, Any], graph: dict[str, Any]) -> str:
    candidates = [
        str(job.get("job_title") or ""),
        str(job.get("domain") or ""),
        str(job.get("job_id") or "").replace("_", " "),
    ]
    graph_roles = role_names(graph)
    for candidate in candidates:
        normalized = normalize_skill(candidate)
        for alias, role_name in ROLE_ALIASES.items():
            if alias in normalized and role_name in graph_roles:
                return role_name
    return graph_roles[0]


def infer_best_source_role(profile: dict[str, Any], graph: dict[str, Any]) -> str:
    skills = normalize_skill_set(extract_profile_skills(profile))
    titles = " ".join(extract_profile_titles(profile))
    normalized_titles = normalize_skill(titles)
    best_role = ""
    best_score = -1.0
    for role in graph.get("roles", []):
        if not isinstance(role, dict):
            continue
        role_name = str(role.get("role"))
        role_norm = normalize_skill(role_name)
        title_bonus = 1.0 if role_norm in normalized_titles else 0.0
        alias_bonus = 0.0
        for alias, alias_role in ROLE_ALIASES.items():
            if alias_role == role_name and alias in normalized_titles:
                alias_bonus = 1.0
                break
        core = normalize_skill_set(_as_list(role.get("core_skills")))
        adjacent = normalize_skill_set(_as_list(role.get("adjacent_skills")))
        overlap = len(skills & core) + 0.5 * len(skills & adjacent)
        score = overlap + title_bonus + alias_bonus
        if score > best_score:
            best_role = role_name
            best_score = score
    return best_role


def compute_transferability(
    *,
    profile: dict[str, Any],
    job: dict[str, Any],
    graph: dict[str, Any],
    candidate_id: str | None = None,
) -> dict[str, Any]:
    target_role_name = infer_target_role(job, graph)
    target_role = get_role(graph, target_role_name) or {}
    best_source_role = infer_best_source_role(profile, graph)
    profile_skills = extract_profile_skills(profile)
    profile_skill_norms = normalize_skill_set(profile_skills)
    required_skills = extract_job_required_skills(job)
    adjacent_skills = _dedupe_preserve_order(_as_list(target_role.get("adjacent_skills")) + extract_job_adjacent_skills(job))
    target_core_skills = _dedupe_preserve_order(_as_list(target_role.get("core_skills")) + required_skills)

    matched_required = _matched_skills(required_skills, profile_skill_norms)
    direct_fit_score = len(matched_required) / len(required_skills) if required_skills else 0.0
    fit_direct = direct_fit_score >= 0.7

    matched_core = _matched_skills(target_core_skills, profile_skill_norms)
    missing_core = [skill for skill in target_core_skills if normalize_skill(skill) not in profile_skill_norms]
    matched_adjacent = _matched_skills(adjacent_skills, profile_skill_norms)
    transitions = _plausible_transitions(graph, best_source_role, target_role_name, profile_skill_norms)
    transition_condition_norms = {
        normalize_skill(skill)
        for transition in transitions
        for skill in transition.get("condition_skills", [])
    }
    adjacent_norms = normalize_skill_set(adjacent_skills)

    gaps_compensables = [
        skill
        for skill in missing_core
        if normalize_skill(skill) in adjacent_norms or normalize_skill(skill) in transition_condition_norms
    ]
    gaps_bloquants = [skill for skill in missing_core if skill not in gaps_compensables]

    core_ratio = len(matched_core) / len(target_core_skills) if target_core_skills else 0.0
    adjacent_ratio = len(matched_adjacent) / len(adjacent_skills) if adjacent_skills else 0.0
    transition_ratio = max((float(t["condition_coverage"]) for t in transitions), default=0.0)
    blocking_ratio = len(gaps_bloquants) / len(target_core_skills) if target_core_skills else 0.0
    transferability_score = _clip01((0.55 * core_ratio) + (0.2 * adjacent_ratio) + (0.15 * transition_ratio) - (0.1 * blocking_ratio))

    resolved_candidate_id = candidate_id or _candidate_id_from_profile(profile)
    job_id = str(job.get("job_id") or normalize_skill(job.get("job_title") or "unknown_job").replace(" ", "_"))
    explanation = _build_explanation(
        target_role=target_role_name,
        direct_fit_score=direct_fit_score,
        transferability_score=transferability_score,
        best_source_role=best_source_role,
        fit_direct=fit_direct,
        gaps_bloquants=gaps_bloquants,
    )

    return {
        "candidate_id": resolved_candidate_id,
        "job_id": job_id,
        "target_role": target_role_name,
        "fit_direct": fit_direct,
        "direct_fit_score": round(direct_fit_score, 4),
        "best_source_role": best_source_role,
        "transferability_score": round(transferability_score, 4),
        "transitions_plausibles": transitions,
        "matched_core_skills": matched_core,
        "missing_core_skills": missing_core,
        "matched_adjacent_skills": matched_adjacent,
        "gaps_compensables": gaps_compensables,
        "gaps_bloquants": gaps_bloquants,
        "explanation": explanation,
    }


def compute_transferability_from_paths(profile_path: str | Path, job_path: str | Path, graph_path: str | Path) -> dict[str, Any]:
    profile = load_json(profile_path)
    job = load_json(job_path)
    graph = load_yaml_graph(graph_path)
    return compute_transferability(profile=profile, job=job, graph=graph, candidate_id=Path(profile_path).stem)


def _plausible_transitions(
    graph: dict[str, Any],
    source_role_name: str,
    target_role_name: str,
    profile_skill_norms: set[str],
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    source_role = get_role(graph, source_role_name)
    if not source_role:
        return transitions
    for transition in _as_list(source_role.get("transitions_to")):
        if not isinstance(transition, dict):
            continue
        condition_skills = [str(skill) for skill in _as_list(transition.get("condition_skills"))]
        matched = _matched_skills(condition_skills, profile_skill_norms)
        coverage = len(matched) / len(condition_skills) if condition_skills else 0.0
        if coverage > 0 and normalize_skill(transition.get("role")) == normalize_skill(target_role_name):
            transitions.append(
                {
                    "from_role": source_role_name,
                    "to_role": str(transition.get("role")),
                    "condition_skills": condition_skills,
                    "matched_condition_skills": matched,
                    "condition_coverage": round(coverage, 4),
                    "rationale": str(transition.get("rationale") or ""),
                }
            )
    return transitions


def _matched_skills(skills: list[str], profile_skill_norms: set[str]) -> list[str]:
    return [skill for skill in skills if normalize_skill(skill) in profile_skill_norms]


def _candidate_id_from_profile(profile: dict[str, Any]) -> str:
    for key in ("candidate_id", "profile_id"):
        if profile.get(key):
            return str(profile[key])
    source_path = str(profile.get("source_path") or "")
    if source_path:
        return Path(source_path).stem
    bio = ((profile.get("profile") or {}).get("bio") or {}) if isinstance(profile.get("profile"), dict) else {}
    if bio.get("full_name"):
        return normalize_skill(bio["full_name"]).replace(" ", "_")
    return "unknown_candidate"


def _build_explanation(
    *,
    target_role: str,
    direct_fit_score: float,
    transferability_score: float,
    best_source_role: str,
    fit_direct: bool,
    gaps_bloquants: list[str],
) -> str:
    if fit_direct:
        return (
            f"Le candidat est un fit direct pour {target_role} avec un direct_fit_score de "
            f"{direct_fit_score:.2f}. Le score de transférabilité est {transferability_score:.2f}."
        )
    if gaps_bloquants:
        return (
            f"Transition depuis {best_source_role} vers {target_role} partiellement plausible, mais des gaps "
            f"bloquants restent présents: {', '.join(gaps_bloquants)}."
        )
    return (
        f"Transition depuis {best_source_role} vers {target_role} plausible sous réserve de validation humaine. "
        f"Score de transférabilité: {transferability_score:.2f}."
    )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _dedupe_preserve_order(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        normalized = normalize_skill(text)
        if normalized and normalized not in seen:
            result.append(text)
            seen.add(normalized)
    return result

