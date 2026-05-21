from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.graph.neo4j_client import Neo4jClient
from src.core.graph.transferability import (
    extract_job_required_skills,
    extract_profile_skills,
    load_yaml_graph,
    normalize_skill,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Smart Recruiter graph data into Neo4j.")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("docs/reports/graph/neo4j_import_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = load_yaml_graph(args.graph)
    profiles = list(args.profiles_dir.glob("*.json"))
    jobs = list(args.jobs_dir.glob("*.json"))
    warnings: list[str] = []

    with Neo4jClient() as client:
        if args.reset:
            client.execute_write(_reset_graph)
        client.execute_write(_create_constraints)
        role_count = import_roles(client, graph)
        profile_count = import_profiles(client, profiles)
        job_count = import_jobs(client, jobs)
        client.execute_write(_merge_candidate_role_fits)
        counts = client.execute_read(_count_graph)

    report = {
        "status": "success",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roles_count": int(counts.get("roles_count", role_count)),
        "skills_count": int(counts.get("skills_count", 0)),
        "jobs_count": int(counts.get("jobs_count", job_count)),
        "candidates_count": int(counts.get("candidates_count", profile_count)),
        "relations_count": int(counts.get("relations_count", 0)),
        "fallback_available": args.graph.exists(),
        "cdc_nodes": ["Role", "Skill", "Job", "Candidate"],
        "cdc_relations": ["REQUIRES", "RELATED_TO", "TRANSITIONS_TO", "HAS_SKILL", "FITS_ROLE"],
        "compatibility_relations": ["REQUIRES_SKILL", "HAS_ADJACENT_SKILL"],
        "warnings": warnings,
    }
    _write_report(args.report, report)

    print("Neo4j import completed")
    print(f"roles={role_count} profiles={profile_count} jobs={job_count}")
    print(f"report={args.report}")


def import_roles(client: Neo4jClient, graph: dict[str, Any]) -> int:
    count = 0
    for role in graph.get("roles", []):
        if not isinstance(role, dict) or not role.get("role"):
            continue
        role_name = str(role["role"])
        family = str(role.get("family") or "")
        client.execute_write(_merge_role, role_name=role_name, family=family)
        for skill in _as_strings(role.get("core_skills")):
            client.execute_write(_merge_role_skill, role_name=role_name, skill=skill, rel_types=["REQUIRES", "REQUIRES_SKILL"])
        for skill in _as_strings(role.get("adjacent_skills")):
            client.execute_write(_merge_role_skill, role_name=role_name, skill=skill, rel_types=["RELATED_TO", "HAS_ADJACENT_SKILL"])
        for transition in role.get("transitions_to") or []:
            if isinstance(transition, dict) and transition.get("role"):
                client.execute_write(
                    _merge_transition,
                    from_role=role_name,
                    to_role=str(transition["role"]),
                    condition_skills=_as_strings(transition.get("condition_skills")),
                    rationale=str(transition.get("rationale") or ""),
                )
        count += 1
    return count


def import_profiles(client: Neo4jClient, profile_paths: list[Path]) -> int:
    count = 0
    for path in profile_paths:
        profile = _read_json(path)
        candidate_id = _candidate_id(profile, path)
        profile_id = str(profile.get("profile_id") or path.stem)
        name = _profile_name(profile) or candidate_id
        client.execute_write(_merge_candidate, candidate_id=candidate_id, profile_id=profile_id, name=name)
        for skill in extract_profile_skills(profile):
            client.execute_write(_merge_candidate_skill, candidate_id=candidate_id, skill=skill)
        count += 1
    return count


def import_jobs(client: Neo4jClient, job_paths: list[Path]) -> int:
    count = 0
    for path in job_paths:
        job = _read_json(path)
        job_id = str(job.get("job_id") or path.stem)
        title = str(job.get("job_title") or job_id)
        client.execute_write(_merge_job, job_id=job_id, title=title)
        for skill in extract_job_required_skills(job):
            client.execute_write(_merge_job_skill, job_id=job_id, skill=skill)
        count += 1
    return count


def _reset_graph(tx: Any) -> None:
    tx.run("MATCH (n) DETACH DELETE n")


def _create_constraints(tx: Any) -> None:
    tx.run("CREATE CONSTRAINT candidate_id IF NOT EXISTS FOR (c:Candidate) REQUIRE c.candidate_id IS UNIQUE")
    tx.run("CREATE CONSTRAINT skill_normalized_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.normalized_name IS UNIQUE")
    tx.run("CREATE CONSTRAINT role_name IF NOT EXISTS FOR (r:Role) REQUIRE r.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT job_id IF NOT EXISTS FOR (j:Job) REQUIRE j.job_id IS UNIQUE")


def _merge_candidate(tx: Any, candidate_id: str, profile_id: str, name: str) -> None:
    tx.run(
        """
        MERGE (c:Candidate {candidate_id: $candidate_id})
        SET c.profile_id = $profile_id,
            c.name = $name
        """,
        candidate_id=candidate_id,
        profile_id=profile_id,
        name=name,
    )


def _merge_candidate_skill(tx: Any, candidate_id: str, skill: str) -> None:
    tx.run(
        """
        MATCH (c:Candidate {candidate_id: $candidate_id})
        MERGE (s:Skill {normalized_name: $normalized_name})
        SET s.name = $skill
        MERGE (c)-[:HAS_SKILL]->(s)
        """,
        candidate_id=candidate_id,
        skill=skill,
        normalized_name=normalize_skill(skill),
    )


def _merge_role(tx: Any, role_name: str, family: str) -> None:
    tx.run(
        """
        MERGE (r:Role {name: $role_name})
        SET r.family = $family
        """,
        role_name=role_name,
        family=family,
    )


def _merge_role_skill(tx: Any, role_name: str, skill: str, rel_types: list[str]) -> None:
    allowed = {"REQUIRES", "REQUIRES_SKILL", "RELATED_TO", "HAS_ADJACENT_SKILL"}
    unsupported = [rel_type for rel_type in rel_types if rel_type not in allowed]
    if unsupported:
        raise ValueError(f"Unsupported role skill relation: {unsupported}")
    for rel_type in rel_types:
        tx.run(
            f"""
            MATCH (r:Role {{name: $role_name}})
            MERGE (s:Skill {{normalized_name: $normalized_name}})
            SET s.name = $skill
            MERGE (r)-[:{rel_type}]->(s)
            """,
            role_name=role_name,
            skill=skill,
            normalized_name=normalize_skill(skill),
        )


def _merge_transition(tx: Any, from_role: str, to_role: str, condition_skills: list[str], rationale: str) -> None:
    tx.run(
        """
        MATCH (source:Role {name: $from_role})
        MERGE (target:Role {name: $to_role})
        MERGE (source)-[rel:TRANSITIONS_TO]->(target)
        SET rel.condition_skills = $condition_skills,
            rel.rationale = $rationale
        """,
        from_role=from_role,
        to_role=to_role,
        condition_skills=condition_skills,
        rationale=rationale,
    )


def _merge_job(tx: Any, job_id: str, title: str) -> None:
    tx.run(
        """
        MERGE (j:Job {job_id: $job_id})
        SET j.title = $title
        """,
        job_id=job_id,
        title=title,
    )


def _merge_job_skill(tx: Any, job_id: str, skill: str) -> None:
    for rel_type in ("REQUIRES", "REQUIRES_SKILL"):
        tx.run(
            f"""
            MATCH (j:Job {{job_id: $job_id}})
            MERGE (s:Skill {{normalized_name: $normalized_name}})
            SET s.name = $skill
            MERGE (j)-[:{rel_type}]->(s)
            """,
            job_id=job_id,
            skill=skill,
            normalized_name=normalize_skill(skill),
        )


def _merge_candidate_role_fits(tx: Any) -> None:
    tx.run(
        """
        MATCH (c:Candidate)-[:HAS_SKILL]->(s:Skill)<-[:REQUIRES]-(r:Role)
        WITH c, r, count(DISTINCT s) AS matched_required_count
        WHERE matched_required_count > 0
        MERGE (c)-[rel:FITS_ROLE]->(r)
        SET rel.matched_required_count = matched_required_count
        """
    )


def _count_graph(tx: Any) -> dict[str, int]:
    result = tx.run(
        """
        MATCH (n)
        WITH
          count { MATCH (:Role) } AS roles_count,
          count { MATCH (:Skill) } AS skills_count,
          count { MATCH (:Job) } AS jobs_count,
          count { MATCH (:Candidate) } AS candidates_count,
          count { MATCH ()-->() } AS relations_count
        RETURN roles_count, skills_count, jobs_count, candidates_count, relations_count
        """
    )
    record = result.single()
    return dict(record) if record else {}


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _candidate_id(profile: dict[str, Any], path: Path) -> str:
    for key in ("candidate_id", "profile_id"):
        if profile.get(key):
            return str(profile[key])
    return f"candidate_{path.stem}"


def _profile_name(profile: dict[str, Any]) -> str:
    body = profile.get("profile") if isinstance(profile.get("profile"), dict) else {}
    bio = body.get("bio") if isinstance(body.get("bio"), dict) else {}
    return str(bio.get("full_name") or "")


def _as_strings(value: Any) -> list[str]:
    return [str(item) for item in value if item] if isinstance(value, list) else []


if __name__ == "__main__":
    main()
