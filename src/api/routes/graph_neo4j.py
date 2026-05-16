from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.core.graph.neo4j_client import Neo4jUnavailable, neo4j_status
from src.core.graph.neo4j_transferability import (
    explain_transferability,
    find_missing_skills,
    get_candidate_skills,
)


router = APIRouter(prefix="/api/graph/neo4j", tags=["graph-neo4j"])


@router.get("/status")
def get_neo4j_status() -> dict:
    return neo4j_status()


@router.get("/roles")
def list_roles() -> dict:
    return _guarded_query(_list_roles)


@router.get("/candidate/{candidate_id}/skills")
def candidate_skills(candidate_id: str) -> dict:
    return _guarded_query(lambda: {"candidate_id": candidate_id, "skills": get_candidate_skills(candidate_id)})


@router.get("/transferability/{candidate_id}")
def neo4j_transferability(candidate_id: str, target_role: str = Query(default="Backend Developer")) -> dict:
    return _guarded_query(lambda: explain_transferability(candidate_id, target_role))


@router.get("/gaps/{candidate_id}")
def neo4j_gaps(candidate_id: str, target_role: str = Query(default="Backend Developer")) -> dict:
    def query() -> dict:
        missing = find_missing_skills(candidate_id, target_role)
        transferability = explain_transferability(candidate_id, target_role)
        return {
            "candidate_id": candidate_id,
            "target_role": target_role,
            "matched_skills": missing["matched_skills"],
            "missing_skills": missing["missing_skills"],
            "gaps_compensables": transferability["gaps_compensables"],
            "gaps_bloquants": transferability["gaps_bloquants"],
        }

    return _guarded_query(query)


def _list_roles() -> dict:
    from src.core.graph.neo4j_client import Neo4jClient

    def query(tx):
        result = tx.run(
            """
            MATCH (r:Role)
            RETURN r.name AS name, r.family AS family
            ORDER BY r.name
            """
        )
        return [dict(record) for record in result]

    with Neo4jClient() as client:
        roles = client.execute_read(query)
    return {"roles": roles}


def _guarded_query(callback):
    try:
        return callback()
    except Neo4jUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
