from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .profile_text_builder import build_candidate_text

DEFAULT_MONGODB_URI = "mongodb://localhost:27017"
DEFAULT_DATABASE = "talent_intelligence"
DEFAULT_COLLECTION = "candidate_profiles"
DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = Path("data/indexes/faiss")
DEFAULT_INDEX_PATH = DEFAULT_INDEX_DIR / "cv_index.faiss"
DEFAULT_ID_MAP_PATH = DEFAULT_INDEX_DIR / "id_map.pkl"
DEFAULT_REPORT_PATH = DEFAULT_INDEX_DIR / "index_report.json"
DEFAULT_MODEL_CACHE_DIR = Path("data/indexes/faiss/hf_cache")


def load_candidate_profiles(
    mongodb_uri: str = DEFAULT_MONGODB_URI,
    database_name: str = DEFAULT_DATABASE,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict[str, Any]]:
    """Load successful candidate profile documents from MongoDB."""
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pymongo") from exc

    client = MongoClient(mongodb_uri)
    try:
        collection = client[database_name][collection_name]
        documents = list(
            collection.find(
                {"status": "success"},
                {
                    "_id": 0,
                    "profile_id": 1,
                    "candidate_id": 1,
                    "bio": 1,
                    "expertise": 1,
                    "experiences": 1,
                    "education": 1,
                    "profile_kind": 1,
                    "provider_route": 1,
                    "reliability_score": 1,
                    "source_path": 1,
                    "artifact_path": 1,
                    "status": 1,
                },
            )
        )
    finally:
        client.close()
    return documents


def encode_profiles(
    profiles: list[dict[str, Any]],
    model_name: str = DEFAULT_SENTENCE_MODEL,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Encode profile texts into normalized dense vectors."""
    model = _load_sentence_transformer(model_name)
    texts = [build_candidate_text(profile) for profile in profiles]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    id_map = [
        {
            "row_id": index,
            "profile_id": profile.get("profile_id"),
            "candidate_id": profile.get("candidate_id"),
            "full_name": ((profile.get("bio") or {}).get("full_name")),
            "profile_kind": profile.get("profile_kind"),
            "provider_route": profile.get("provider_route"),
            "source_path": profile.get("source_path"),
            "artifact_path": profile.get("artifact_path"),
            "reliability_score": profile.get("reliability_score"),
            "text_preview": text[:280],
        }
        for index, (profile, text) in enumerate(zip(profiles, texts))
    ]
    return embeddings, id_map


def build_faiss_index(embeddings: np.ndarray) -> Any:
    """Build a cosine-like FAISS index using normalized vectors and inner product."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Missing dependency: faiss-cpu") from exc

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")
    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_index(
    index: Any,
    id_map: list[dict[str, Any]],
    index_path: Path = DEFAULT_INDEX_PATH,
    id_map_path: Path = DEFAULT_ID_MAP_PATH,
) -> None:
    """Persist the FAISS index and the row-to-profile mapping."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Missing dependency: faiss-cpu") from exc

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with id_map_path.open("wb") as handle:
        pickle.dump(id_map, handle)


def write_index_report(report: dict[str, Any], report_path: Path = DEFAULT_REPORT_PATH) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def run_faiss_indexer() -> dict[str, Any]:
    start_time = time.perf_counter()
    profiles = load_candidate_profiles()
    embeddings, id_map = encode_profiles(profiles)
    index = build_faiss_index(embeddings)
    save_index(index, id_map)

    duration_seconds = round(time.perf_counter() - start_time, 4)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mongodb_uri": DEFAULT_MONGODB_URI,
        "database": DEFAULT_DATABASE,
        "collection": DEFAULT_COLLECTION,
        "sentence_transformer_model": DEFAULT_SENTENCE_MODEL,
        "profiles_read_from_mongodb": len(profiles),
        "profiles_indexed": len(id_map),
        "embedding_dimension": int(embeddings.shape[1]),
        "faiss_metric": "inner_product_on_normalized_vectors",
        "index_path": str(DEFAULT_INDEX_PATH),
        "id_map_path": str(DEFAULT_ID_MAP_PATH),
        "execution_time_seconds": duration_seconds,
    }
    write_index_report(report)
    return report


def _load_sentence_transformer(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Missing dependency: sentence-transformers") from exc
    DEFAULT_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(DEFAULT_MODEL_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_MODEL_CACHE_DIR))
    return SentenceTransformer(model_name, cache_folder=str(DEFAULT_MODEL_CACHE_DIR))


def main() -> None:
    report = run_faiss_indexer()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
