"""Runtime store for current job profile and matching results.

Manages current_job_profile.json and current_matching_run.json to avoid
accumulation of temporary files during interactive sessions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


RUNTIME_DIR = Path("data/runtime")
CURRENT_JOB_PROFILE_PATH = RUNTIME_DIR / "current_job_profile.json"
CURRENT_MATCHING_RUN_PATH = RUNTIME_DIR / "current_matching_run.json"


def ensure_runtime_dir() -> Path:
    """Create data/runtime/ if it doesn't exist."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DIR


def save_current_job_profile(profile: dict[str, Any]) -> Path:
    """Save or overwrite current_job_profile.json."""
    ensure_runtime_dir()
    path = CURRENT_JOB_PROFILE_PATH
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_current_job_profile() -> dict[str, Any] | None:
    """Load current_job_profile.json if it exists."""
    if not CURRENT_JOB_PROFILE_PATH.exists():
        return None
    try:
        return json.loads(CURRENT_JOB_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def clear_current_job_profile() -> None:
    """Delete current_job_profile.json."""
    if CURRENT_JOB_PROFILE_PATH.exists():
        CURRENT_JOB_PROFILE_PATH.unlink()


def save_current_matching_run(
    matching_result: dict[str, Any],
    job_id: str | None = None,
) -> Path:
    """Save or overwrite current_matching_run.json."""
    ensure_runtime_dir()
    path = CURRENT_MATCHING_RUN_PATH
    run_data = {
        "job_id": job_id,
        "candidates": matching_result.get("candidates", []),
        "decision_cards": matching_result.get("decision_cards", []),
        "transferability": matching_result.get("transferability", {}),
        "matching_metadata": matching_result.get("matching_metadata", {}),
        "sources": matching_result.get("sources", []),
        "warnings": matching_result.get("warnings", []),
        "job_description": matching_result.get("job_description"),
    }
    path.write_text(json.dumps(run_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_current_matching_run() -> dict[str, Any] | None:
    """Load current_matching_run.json if it exists."""
    if not CURRENT_MATCHING_RUN_PATH.exists():
        return None
    try:
        return json.loads(CURRENT_MATCHING_RUN_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def clear_current_matching_run() -> None:
    """Delete current_matching_run.json."""
    if CURRENT_MATCHING_RUN_PATH.exists():
        CURRENT_MATCHING_RUN_PATH.unlink()


def clear_all_runtime() -> None:
    """Clear both current_job_profile.json and current_matching_run.json."""
    clear_current_job_profile()
    clear_current_matching_run()
