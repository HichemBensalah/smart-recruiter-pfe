from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


DEFAULT_API_KEY_HEADER = "X-Smart-Recruiter-Key"
DEFAULT_DATA_BACKEND = "artifacts"
VALID_DATA_BACKENDS = {"artifacts", "mongodb", "hybrid"}
DEFAULT_MATCHING_MODE = "hybrid"
VALID_MATCHING_MODES = {"artifact", "live", "hybrid"}


@dataclass(frozen=True)
class ApiSettings:
    auth_enabled: bool
    api_key: str
    api_key_header: str


@dataclass(frozen=True)
class DataSettings:
    data_backend: str
    allow_artifact_fallback: bool
    mongodb_uri: str
    mongodb_database: str


@dataclass(frozen=True)
class MatchingSettings:
    matching_mode: str
    live_strict: bool
    live_matching_top_n: int
    live_matching_top_k: int
    faiss_index_path: str
    faiss_id_map_path: str


def load_api_settings() -> ApiSettings:
    return ApiSettings(
        auth_enabled=_env_bool(os.getenv("AUTH_ENABLED", "false")),
        api_key=os.getenv("SMART_RECRUITER_API_KEY", ""),
        api_key_header=os.getenv("API_KEY_HEADER", DEFAULT_API_KEY_HEADER).strip() or DEFAULT_API_KEY_HEADER,
    )


def load_data_settings() -> DataSettings:
    data_backend = (os.getenv("DATA_BACKEND", DEFAULT_DATA_BACKEND).strip().lower() or DEFAULT_DATA_BACKEND)
    if data_backend not in VALID_DATA_BACKENDS:
        valid_values = ", ".join(sorted(VALID_DATA_BACKENDS))
        raise ValueError(f"DATA_BACKEND must be one of: {valid_values}")

    return DataSettings(
        data_backend=data_backend,
        allow_artifact_fallback=_env_bool(os.getenv("ALLOW_ARTIFACT_FALLBACK", "true")),
        mongodb_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017").strip() or "mongodb://localhost:27017",
        mongodb_database=os.getenv("MONGODB_DATABASE", "talent_intelligence").strip() or "talent_intelligence",
    )


def load_matching_settings() -> MatchingSettings:
    matching_mode = (os.getenv("MATCHING_MODE", DEFAULT_MATCHING_MODE).strip().lower() or DEFAULT_MATCHING_MODE)
    if matching_mode not in VALID_MATCHING_MODES:
        valid_values = ", ".join(sorted(VALID_MATCHING_MODES))
        raise ValueError(f"MATCHING_MODE must be one of: {valid_values}")

    return MatchingSettings(
        matching_mode=matching_mode,
        live_strict=_env_bool(os.getenv("LIVE_STRICT", "false")),
        live_matching_top_n=_env_int("LIVE_MATCHING_TOP_N", 50, minimum=1),
        live_matching_top_k=_env_int("LIVE_MATCHING_TOP_K", 5, minimum=1),
        faiss_index_path=os.getenv("FAISS_INDEX_PATH", "data/indexes/faiss/cv_index.faiss").strip()
        or "data/indexes/faiss/cv_index.faiss",
        faiss_id_map_path=os.getenv("FAISS_ID_MAP_PATH", "data/indexes/faiss/id_map.pkl").strip()
        or "data/indexes/faiss/id_map.pkl",
    )


def _env_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value
