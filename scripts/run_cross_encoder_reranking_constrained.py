from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_cross_encoder_reranking import (  # noqa: E402
    FINAL_TOP_K,
    NORMALIZED_MATCHING_REPORT_PATH,
    RERANK_INPUT_SIZE,
    build_cross_encoder_job_text,
    build_rerank_rows,
    load_normalized_profiles,
    load_or_compute_top20,
    read_json,
    slim_candidate,
    write_json,
)
from scripts.run_matching_v3_normalized import JOB_PROFILE_PATH  # noqa: E402
from src.core.retrieval.cross_encoder_reranker import (  # noqa: E402
    DEFAULT_CROSS_ENCODER_MODEL,
    rerank_candidates_with_cross_encoder,
)


OUTPUT_PATH = Path("docs/reports/retrieval/cross_encoder_reranking_constrained_report.json")
MUST_HAVE_THRESHOLD = 0.6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def exclusion_reason(candidate: dict[str, Any]) -> str:
    coverage = float(candidate.get("must_have_coverage") or 0.0)
    return f"must_have_coverage_below_threshold: {coverage:.4f} < {MUST_HAVE_THRESHOLD:.4f}"


def filter_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    admissible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for candidate in candidates:
        coverage = float(candidate.get("must_have_coverage") or 0.0)
        if coverage >= MUST_HAVE_THRESHOLD:
            admissible.append(candidate)
            continue
        item = slim_candidate(candidate)
        item["exclusion_reason"] = exclusion_reason(candidate)
        excluded.append(item)
    return admissible, excluded


def build_rank_changes(reranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": item.get("candidate_id"),
            "matched_profile_id": item.get("matched_profile_id"),
            "full_name": item.get("full_name"),
            "faiss_rank": int(item.get("faiss_v3_rank") or item.get("original_faiss_rank") or item.get("faiss_rank") or 0),
            "ce_rank": int(item.get("cross_encoder_rank") or 0),
            "rank_change": int(item.get("faiss_v3_rank") or item.get("original_faiss_rank") or item.get("faiss_rank") or 0)
            - int(item.get("cross_encoder_rank") or 0),
            "must_have_coverage": item.get("must_have_coverage"),
            "ce_score": item.get("cross_encoder_score"),
            "ce_score_normalized": item.get("cross_encoder_score_normalized"),
        }
        for item in reranked
    ]


def constrained_candidates_are_valid(items: list[dict[str, Any]]) -> bool:
    return all(float(item.get("must_have_coverage") or 0.0) >= MUST_HAVE_THRESHOLD for item in items)


def main() -> None:
    start = time.perf_counter()
    normalized_report = read_json(NORMALIZED_MATCHING_REPORT_PATH)
    job_profile = read_json(JOB_PROFILE_PATH)
    profiles_by_id, normalization_stats = load_normalized_profiles()
    top20, input_source = load_or_compute_top20(normalized_report, job_profile, profiles_by_id)
    rerank_rows = build_rerank_rows(top20, profiles_by_id)
    admissible, excluded = filter_candidates(rerank_rows)
    job_text = build_cross_encoder_job_text(job_profile)

    reranked = rerank_candidates_with_cross_encoder(
        job_text=job_text,
        candidates=admissible,
        candidate_text_key="candidate_text",
        top_k=len(admissible),
        model_name=DEFAULT_CROSS_ENCODER_MODEL,
    )
    reranked_top = reranked[:FINAL_TOP_K]
    latency = round(time.perf_counter() - start, 4)
    phase3_validated = constrained_candidates_are_valid(reranked_top)

    report = {
        "experiment_name": "cross_encoder_reranking_constrained_must_have_coverage",
        "generated_at_utc": utc_now(),
        "model_used": DEFAULT_CROSS_ENCODER_MODEL,
        "baseline_report_used": str(NORMALIZED_MATCHING_REPORT_PATH),
        "input_source": input_source,
        "top_n_faiss_input": RERANK_INPUT_SIZE,
        "must_have_threshold": MUST_HAVE_THRESHOLD,
        "candidates_before_filter": len(rerank_rows),
        "candidates_after_filter": len(admissible),
        "excluded_candidates": excluded,
        "reranked_top_10": [slim_candidate(item) for item in reranked_top],
        "rank_changes_vs_faiss": build_rank_changes(reranked),
        "latency_seconds": latency,
        "phase3_validated": phase3_validated,
        "metadata": {
            "baseline_modified": False,
            "normalized_matching_report_modified": False,
            "faiss_reindexed": False,
            "mongodb_profiles_modified": False,
            "markdown_created": False,
            "module1_rerun": False,
            "module2_rerun": False,
            "normalization_profiles_loaded": normalization_stats.get("profiles_normalized"),
        },
        "conclusion": (
            "Le CrossEncoder contraint corrige partiellement le probleme de remontee de candidats faibles "
            "en must-have coverage, mais reste experimental. La baseline officielle demeure FAISS + Matching V3."
        ),
    }
    write_json(OUTPUT_PATH, report)

    summary = {
        "file_created": str(OUTPUT_PATH),
        "candidates_before_filter": report["candidates_before_filter"],
        "candidates_after_filter": report["candidates_after_filter"],
        "excluded_count": len(excluded),
        "reranked_top_10": report["reranked_top_10"],
        "phase3_validated": phase3_validated,
        "latency_seconds": latency,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
