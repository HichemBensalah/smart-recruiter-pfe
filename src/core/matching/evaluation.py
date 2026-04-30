from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .faiss_indexer import DEFAULT_INDEX_PATH, DEFAULT_REPORT_PATH, DEFAULT_SENTENCE_MODEL
from .recommender import recommend_candidates

DEFAULT_JOB_PROFILES_DIR = Path("data/job_profiles")
DEFAULT_MATCHING_REPORT_PATH = Path("data/matching_test_report.json")
DEFAULT_TOP_K = 10


def evaluate_single_job_profile(job_profile_path: Path, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    job_profile = json.loads(job_profile_path.read_text(encoding="utf-8"))
    recommendations = recommend_candidates(job_profile, top_k=top_k)
    return {
        "job_profile_file": str(job_profile_path),
        "job_title": job_profile.get("job_title"),
        "top_k": top_k,
        "recommendations_count": len(recommendations),
        "recommendations": recommendations,
    }


def write_matching_report(report: dict[str, Any], output_path: Path = DEFAULT_MATCHING_REPORT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def run_matching_evaluation(top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    start_time = time.perf_counter()
    job_profile_paths = sorted(
        path
        for path in DEFAULT_JOB_PROFILES_DIR.glob("*.json")
        if path.name != "job_profile_builder_report.json"
    )
    evaluations = [evaluate_single_job_profile(path, top_k=top_k) for path in job_profile_paths]
    duration_seconds = round(time.perf_counter() - start_time, 4)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_profiles_tested": len(job_profile_paths),
        "top_k": top_k,
        "sentence_transformer_model": DEFAULT_SENTENCE_MODEL,
        "faiss_index_path": str(DEFAULT_INDEX_PATH),
        "faiss_index_report_path": str(DEFAULT_REPORT_PATH),
        "execution_time_seconds": duration_seconds,
        "results": evaluations,
    }
    write_matching_report(report)
    return report


def main() -> None:
    report = run_matching_evaluation()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
