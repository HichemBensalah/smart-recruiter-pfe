from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.storage.repositories import (
    DEFAULT_MONGODB_DATABASE,
    DEFAULT_MONGODB_URI,
    RepositoryUnavailableError,
    create_mongo_repositories,
    mask_mongodb_uri,
    stable_id,
)


GROUNDED_PROFILES_DIR = PROJECT_ROOT / "data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles"
FAISS_ID_MAP_PATH = PROJECT_ROOT / "data/indexes/faiss/id_map.pkl"


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


def load_faiss_id_map() -> dict[str, dict[str, Any]]:
    """Load FAISS id_map.pkl and index by artifact_path for lookup."""
    if not FAISS_ID_MAP_PATH.exists():
        return {}
    try:
        with FAISS_ID_MAP_PATH.open("rb") as f:
            id_map = pickle.load(f)
        # Index by artifact_path for fast lookup
        by_path = {}
        for entry in id_map:
            artifact_path = entry.get("artifact_path")
            if artifact_path:
                by_path[artifact_path] = entry
        return by_path
    except Exception:
        return {}


def build_candidate_profile_documents(id_map_by_path: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Build candidate profile documents from grounded profiles.

    CRITICAL: Uses profile_id and candidate_id from FAISS id_map.pkl to ensure
    correspondence with FAISS index. This guarantees LiveMatcher can resolve
    profiles via FAISS retrieval.

    Fallback: If a profile is not in id_map (shouldn't happen), generates a
    stable ID to avoid breaking the import.
    """
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
        artifact_path = payload.get("artifact_path")

        # CRITICAL: Look up profile_id and candidate_id from FAISS id_map
        id_map_entry = id_map_by_path.get(artifact_path, {})
        profile_id = id_map_entry.get("profile_id") or stable_id("profile", [source_artifact])
        candidate_id = id_map_entry.get("candidate_id") or stable_id(
            "candidate",
            [payload.get("source_path"), payload.get("source_format", "unknown")],
        )

        # Build document from grounded profile
        document = dict(payload)
        document["profile_id"] = profile_id
        document["candidate_id"] = candidate_id
        document["source_artifact"] = source_artifact
        document["document_type"] = "candidate_profile_grounded"

        documents.append(document)

    return documents, skipped


def deduplicate_documents(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Detect and mark duplicate profiles.

    Ultra-prudent deduplication strategy:
    - Duplicates detected ONLY if: email identical AND full_name identical (after normalization).
    - NO person_id cross-linking (email generics like 9@gmail.com create false positives).
    - ALL profiles kept in MongoDB with FAISS profile_id intact.
    - Duplicates marked with is_duplicate_of field (links to best profile's profile_id).
    """
    # Normalize and group by (email, full_name) key
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for doc in documents:
        bio = doc.get("profile", {}).get("bio", {}) or {}
        email = (bio.get("email") or "").lower().strip()
        full_name = (bio.get("full_name") or "").lower().strip()

        # Skip if no email OR no name (can't deduplicate)
        if not email or not full_name:
            # Assign to unique key if missing
            unique_key = (email or f"no_email_{id(doc)}", full_name or f"no_name_{id(doc)}")
            groups[unique_key].append(doc)
        else:
            key = (email, full_name)
            groups[key].append(doc)

    # Identify duplicates and mark them
    duplicates_detected = 0
    profiles_marked = 0
    duplicate_groups = []

    for key, group in groups.items():
        if len(group) > 1:
            # Multiple profiles with same (email, full_name)
            duplicates_detected += 1

            # Find best profile (highest reliability_score)
            best_profile = max(
                group,
                key=lambda p: p.get("grounding", {}).get("reliability_score", 0),
            )
            best_profile_id = best_profile.get("profile_id")

            # Mark others as duplicates
            for profile in group:
                if profile is not best_profile:
                    profile["is_duplicate_of"] = best_profile_id
                    profiles_marked += 1

            # Record for reporting
            duplicate_groups.append({
                "email": key[0],
                "full_name": key[1],
                "count": len(group),
                "best_profile_id": best_profile_id,
                "best_score": best_profile.get("grounding", {}).get("reliability_score", 0),
            })

    return documents, {
        "duplicate_groups_detected": duplicates_detected,
        "profiles_marked_as_duplicate": profiles_marked,
        "duplicate_groups_sample": duplicate_groups[:5],  # First 5 for reporting
    }


def upsert_documents(
    repository: Any,
    key_field: str,
    documents: list[dict[str, Any]],
) -> int:
    """Upsert documents into a MongoDB collection."""
    upserted = 0
    for doc in documents:
        key_value = doc.get(key_field)
        if not key_value:
            continue
        try:
            repository.collection.update_one(
                {key_field: key_value},
                {"$set": doc},
                upsert=True,
            )
            upserted += 1
        except Exception as e:
            print(f"Warning: Failed to upsert {key_field}={key_value}: {e}")
    return upserted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed MongoDB candidate_profiles collection from grounded CV profiles."
    )
    parser.add_argument(
        "--mongodb-uri",
        default=os.getenv("MONGODB_URI", DEFAULT_MONGODB_URI),
        help=f"MongoDB connection URI (default: {DEFAULT_MONGODB_URI})",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("MONGODB_DATABASE", DEFAULT_MONGODB_DATABASE),
        help=f"MongoDB database name (default: {DEFAULT_MONGODB_DATABASE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare documents and print counts without writing to MongoDB.",
    )

    args = parser.parse_args()

    # Load FAISS id_map for ID correspondence
    id_map_by_path = load_faiss_id_map()

    # Build documents
    candidate_profiles, skipped_profiles = build_candidate_profile_documents(id_map_by_path)

    # Deduplicate
    candidate_profiles, dedup_stats = deduplicate_documents(candidate_profiles)

    summary: dict[str, Any] = {
        "mongodb_uri": mask_mongodb_uri(args.mongodb_uri),
        "database": args.database,
        "dry_run": args.dry_run,
        "source_directory": relative_source(GROUNDED_PROFILES_DIR),
        "faiss_id_map_used": FAISS_ID_MAP_PATH.exists(),
        "id_map_entries": len(id_map_by_path),
        "deduplication": dedup_stats,
        "prepared": {
            "candidate_profiles": len(candidate_profiles),
        },
        "skipped": {
            "candidate_profiles": skipped_profiles,
        },
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    # Connect and upsert
    try:
        repositories = create_mongo_repositories(args.mongodb_uri, args.database)
    except RepositoryUnavailableError as exc:
        summary["error"] = str(exc)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1

    try:
        repositories.ensure_indexes()
        upserted = upsert_documents(
            repositories.candidate_profiles,
            "profile_id",
            candidate_profiles,
        )
        summary["upserted"] = {
            "candidate_profiles": upserted,
        }
    except Exception as e:
        summary["error"] = str(e)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1
    finally:
        repositories.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
