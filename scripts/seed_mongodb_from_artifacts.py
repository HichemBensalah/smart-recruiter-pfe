from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.utils import (  # noqa: E402
    MATCHING_FEATURES_DIR,
    get_candidates,
    load_best_decision_cards_artifact,
)
from src.core.storage.repositories import (  # noqa: E402
    DEFAULT_MONGODB_DATABASE,
    DEFAULT_MONGODB_URI,
    RepositoryUnavailableError,
    create_mongo_repositories,
    mask_mongodb_uri,
    stable_id,
)


JOB_PROFILES_DIR = PROJECT_ROOT / "data/job_profiles"
GROUNDED_PROFILES_DIR = PROJECT_ROOT / "data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles"


def relative_source(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {relative_source(path)}")
    return payload


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"{relative_source(path)}:{line_number}: invalid JSON ({exc.msg})")
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            warnings.append(f"{relative_source(path)}:{line_number}: ignored non-object row")
    return rows, warnings


def resolve_artifact_path(raw_path: Any) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def profile_paths_by_card(cards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for card in cards:
        transferability = card.get("transferability")
        if not isinstance(transferability, dict):
            continue
        path = resolve_artifact_path(transferability.get("profile_path"))
        if path is None:
            continue
        by_path[relative_source(path)] = card
    return by_path


def build_candidate_documents(cards: list[dict[str, Any]], source_artifact: str) -> tuple[list[dict[str, Any]], int]:
    documents: list[dict[str, Any]] = []
    skipped = 0
    for card in cards:
        candidate_id = card.get("candidate_id")
        if not candidate_id:
            skipped += 1
            continue
        profile_id = card.get("profile_id")
        document = dict(card)
        document["candidate_id"] = candidate_id
        document["profile_id"] = profile_id
        document["best_profile_id"] = profile_id
        document["profile_ids"] = [profile_id] if profile_id else []
        document["source_artifact"] = source_artifact
        document["document_type"] = "candidate_seed_from_decision_card"
        documents.append(document)
    return documents, skipped


def build_decision_card_documents(cards: list[dict[str, Any]], source_artifact: str, artifact_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    documents: list[dict[str, Any]] = []
    skipped = 0
    for card in cards:
        candidate_id = card.get("candidate_id")
        if not candidate_id:
            skipped += 1
            continue
        document = dict(card)
        document["candidate_id"] = candidate_id
        document["source_artifact"] = source_artifact
        document["card_type"] = artifact_payload.get("card_type") or "decision_card"
        document["job_id"] = document.get("job_id") or artifact_payload.get("job_id")
        document["document_type"] = "decision_card_seed"
        documents.append(document)
    return documents, skipped


def build_candidate_profile_documents(cards: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    card_by_profile_path = profile_paths_by_card(cards)
    documents: list[dict[str, Any]] = []
    skipped = 0
    if not GROUNDED_PROFILES_DIR.exists():
        return documents, skipped

    for path in sorted(GROUNDED_PROFILES_DIR.glob("*.json")):
        try:
            payload = read_json(path)
        except Exception:
            skipped += 1
            continue
        source_artifact = relative_source(path)
        card = card_by_profile_path.get(source_artifact, {})
        profile_id = card.get("profile_id") or payload.get("profile_id") or stable_id("profile", [source_artifact])
        document = dict(payload)
        document["profile_id"] = profile_id
        document["candidate_id"] = card.get("candidate_id") or payload.get("candidate_id")
        document["source_artifact"] = source_artifact
        document["document_type"] = "candidate_profile_artifact"
        documents.append(document)
    return documents, skipped


def build_job_profile_documents() -> tuple[list[dict[str, Any]], int]:
    documents: list[dict[str, Any]] = []
    skipped = 0
    if not JOB_PROFILES_DIR.exists():
        return documents, skipped

    for path in sorted(JOB_PROFILES_DIR.glob("*.json")):
        try:
            payload = read_json(path)
        except Exception:
            skipped += 1
            continue
        job_id = payload.get("job_id") or path.stem
        document = dict(payload)
        document["job_id"] = job_id
        document["source_artifact"] = relative_source(path)
        document["document_type"] = "job_profile_artifact"
        documents.append(document)
    return documents, skipped


def build_matching_run_seed_documents() -> tuple[list[dict[str, Any]], int, list[str]]:
    documents: list[dict[str, Any]] = []
    skipped = 0
    warnings: list[str] = []
    if not MATCHING_FEATURES_DIR.exists():
        return documents, skipped, [f"Matching features directory not found: {relative_source(MATCHING_FEATURES_DIR)}"]

    for path in sorted(MATCHING_FEATURES_DIR.glob("*.jsonl")):
        rows, row_warnings = read_jsonl(path)
        warnings.extend(row_warnings)
        if not rows:
            skipped += 1
            continue
        job_id = path.stem
        documents.append(
            {
                "run_id": f"artifact_{job_id}",
                "job_id": job_id,
                "resolved_job_id": job_id,
                "source_artifact": relative_source(path),
                "artifact_source": relative_source(path),
                "matching_mode": "matching_v3_artifact_seed",
                "candidate_count": len(rows),
                "items": rows,
                "document_type": "matching_v3_artifact_seed",
            }
        )
    return documents, skipped, warnings


def new_counter(extra_skipped: int = 0) -> dict[str, int]:
    return {"inserted": 0, "updated": 0, "skipped": extra_skipped}


def increment(counter: dict[str, int], status: str) -> None:
    if status not in counter:
        counter[status] = 0
    counter[status] += 1


def upsert_query(
    key_field: str,
    key: Any,
    document: dict[str, Any],
    alternate_key_fields: list[str] | None = None,
) -> dict[str, Any]:
    queries = [{key_field: key}]
    seen = {(key_field, str(key))}
    for field in alternate_key_fields or []:
        value = document.get(field)
        if value in (None, ""):
            continue
        marker = (field, str(value))
        if marker in seen:
            continue
        queries.append({field: value})
        seen.add(marker)
    if len(queries) == 1:
        return queries[0]
    return {"$or": queries}


def upsert_documents(
    repository: Any,
    key_field: str,
    documents: list[dict[str, Any]],
    counter: dict[str, int],
    *,
    alternate_key_fields: list[str] | None = None,
) -> None:
    for document in documents:
        key = document.get(key_field)
        if not key:
            counter["skipped"] += 1
            continue
        status = repository.upsert_document(upsert_query(key_field, key, document, alternate_key_fields), document)
        increment(counter, status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed MongoDB business collections from Smart Recruiter artifacts.")
    parser.add_argument("--mongodb-uri", default=os.getenv("MONGODB_URI", DEFAULT_MONGODB_URI))
    parser.add_argument("--database", default=os.getenv("MONGODB_DATABASE", DEFAULT_MONGODB_DATABASE))
    parser.add_argument("--dry-run", action="store_true", help="Prepare documents and print counts without writing to MongoDB.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = load_best_decision_cards_artifact()
    cards = get_candidates(artifact.payload)

    candidates, skipped_candidates = build_candidate_documents(cards, artifact.source)
    decision_cards, skipped_decision_cards = build_decision_card_documents(cards, artifact.source, artifact.payload)
    candidate_profiles, skipped_profiles = build_candidate_profile_documents(cards)
    job_profiles, skipped_jobs = build_job_profile_documents()
    matching_runs, skipped_matching, matching_warnings = build_matching_run_seed_documents()

    summary: dict[str, Any] = {
        "mongodb_uri": mask_mongodb_uri(args.mongodb_uri),
        "database": args.database,
        "dry_run": args.dry_run,
        "collections": {
            "candidates": new_counter(skipped_candidates),
            "candidate_profiles": new_counter(skipped_profiles),
            "job_profiles": new_counter(skipped_jobs),
            "decision_cards": new_counter(skipped_decision_cards),
            "matching_runs": new_counter(skipped_matching),
        },
        "prepared": {
            "candidates": len(candidates),
            "candidate_profiles": len(candidate_profiles),
            "job_profiles": len(job_profiles),
            "decision_cards": len(decision_cards),
            "matching_runs": len(matching_runs),
        },
        "warnings": matching_warnings,
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    try:
        repositories = create_mongo_repositories(args.mongodb_uri, args.database)
    except RepositoryUnavailableError as exc:
        summary["error"] = str(exc)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1

    try:
        repositories.ensure_indexes()
        upsert_documents(repositories.candidates, "candidate_id", candidates, summary["collections"]["candidates"])
        upsert_documents(
            repositories.candidate_profiles,
            "profile_id",
            candidate_profiles,
            summary["collections"]["candidate_profiles"],
            alternate_key_fields=["source_artifact", "artifact_path", "source_path"],
        )
        upsert_documents(repositories.job_profiles, "job_id", job_profiles, summary["collections"]["job_profiles"])
        upsert_documents(repositories.decision_cards, "candidate_id", decision_cards, summary["collections"]["decision_cards"])
        upsert_documents(repositories.matching_runs, "run_id", matching_runs, summary["collections"]["matching_runs"])
    finally:
        repositories.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
