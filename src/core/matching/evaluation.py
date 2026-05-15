from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .faiss_indexer import DEFAULT_INDEX_PATH, DEFAULT_REPORT_PATH, DEFAULT_SENTENCE_MODEL
from .matching_quality_filters import build_display_name
from .recommender import recommend_candidates
from ..retrieval.cross_encoder_reranker import DEFAULT_CROSS_ENCODER_MODEL

DEFAULT_JOB_PROFILES_DIR = Path("data/job_profiles")
DEFAULT_MATCHING_REPORT_PATH = Path("data/matching_test_report.json")
DEFAULT_TOP_K = 10


def evaluate_single_job_profile(
    job_profile_path: Path,
    top_k: int = DEFAULT_TOP_K,
    *,
    use_cross_encoder: bool = False,
    cross_encoder_top_n: int = 20,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> dict[str, Any]:
    job_profile = json.loads(job_profile_path.read_text(encoding="utf-8"))
    recommendations = _prepare_recommendations(
        recommend_candidates(
            job_profile,
            top_k=top_k,
            use_cross_encoder=use_cross_encoder,
            cross_encoder_top_n=cross_encoder_top_n,
            cross_encoder_model=cross_encoder_model,
        )
    )
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


def run_matching_evaluation(
    top_k: int = DEFAULT_TOP_K,
    *,
    use_cross_encoder: bool = False,
    cross_encoder_top_n: int = 20,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    job_profile_paths = sorted(
        path
        for path in DEFAULT_JOB_PROFILES_DIR.glob("*.json")
        if path.name != "job_profile_builder_report.json"
    )
    evaluations = [
        evaluate_single_job_profile(
            path,
            top_k=top_k,
            use_cross_encoder=use_cross_encoder,
            cross_encoder_top_n=cross_encoder_top_n,
            cross_encoder_model=cross_encoder_model,
        )
        for path in job_profile_paths
    ]
    duration_seconds = round(time.perf_counter() - start_time, 4)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_profiles_tested": len(job_profile_paths),
        "top_k": top_k,
        "sentence_transformer_model": DEFAULT_SENTENCE_MODEL,
        "faiss_index_path": str(DEFAULT_INDEX_PATH),
        "faiss_index_report_path": str(DEFAULT_REPORT_PATH),
        "metadata": _build_metadata(use_cross_encoder, cross_encoder_top_n, cross_encoder_model),
        "execution_time_seconds": duration_seconds,
        "results": evaluations,
    }
    write_matching_report(report)
    return report


def build_single_job_report(
    job_profile_path: Path,
    top_k: int = DEFAULT_TOP_K,
    *,
    use_cross_encoder: bool = False,
    cross_encoder_top_n: int = 20,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    evaluation = evaluate_single_job_profile(
        job_profile_path,
        top_k=top_k,
        use_cross_encoder=use_cross_encoder,
        cross_encoder_top_n=cross_encoder_top_n,
        cross_encoder_model=cross_encoder_model,
    )
    duration_seconds = round(time.perf_counter() - start_time, 4)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_profiles_tested": 1,
        "top_k": top_k,
        "sentence_transformer_model": DEFAULT_SENTENCE_MODEL,
        "faiss_index_path": str(DEFAULT_INDEX_PATH),
        "faiss_index_report_path": str(DEFAULT_REPORT_PATH),
        "metadata": _build_metadata(use_cross_encoder, cross_encoder_top_n, cross_encoder_model),
        "execution_time_seconds": duration_seconds,
        "results": [evaluation],
    }


def _build_metadata(use_cross_encoder: bool, cross_encoder_top_n: int, cross_encoder_model: str) -> dict[str, Any]:
    return {
        "retrieval_engine": "faiss",
        "reranker": "cross_encoder" if use_cross_encoder else "none",
        "cross_encoder_model": cross_encoder_model if use_cross_encoder else None,
        "cross_encoder_top_n": cross_encoder_top_n if use_cross_encoder else None,
        "baseline_compared_to": "baseline1_faiss_matching_v3" if use_cross_encoder else None,
    }


def _prepare_recommendations(recommendations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for rank, recommendation in enumerate(recommendations, start=1):
        item = dict(recommendation)
        display_name, display_name_quality, name_warning = build_display_name(
            full_name=item.get("full_name"),
            candidate_id=item.get("candidate_id"),
        )
        item["rank"] = rank
        item["full_name"] = display_name
        item["display_name_quality"] = item.get("display_name_quality") or display_name_quality
        item["name_warning"] = item.get("name_warning") or name_warning
        item["quality_flags"] = list(item.get("quality_flags") or [])
        item["fields_nullified_count"] = int(item.get("fields_nullified_count") or 0)
        prepared.append(item)
    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FAISS matching on one or more job profiles.")
    parser.add_argument("--job-profile", type=Path, help="Path to a single job profile JSON to evaluate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MATCHING_REPORT_PATH,
        help="Where to write the matching report JSON.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of candidates to return per job profile.")
    parser.add_argument("--use-cross-encoder", action="store_true", help="Rerank FAISS top-N candidates with a CrossEncoder.")
    parser.add_argument("--cross-encoder-top-n", type=int, default=20, help="Number of FAISS candidates to rerank.")
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL, help="CrossEncoder model name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.job_profile:
        report = build_single_job_report(
            args.job_profile,
            top_k=args.top_k,
            use_cross_encoder=args.use_cross_encoder,
            cross_encoder_top_n=args.cross_encoder_top_n,
            cross_encoder_model=args.cross_encoder_model,
        )
        write_matching_report(report, output_path=args.output)
    else:
        report = run_matching_evaluation(
            top_k=args.top_k,
            use_cross_encoder=args.use_cross_encoder,
            cross_encoder_top_n=args.cross_encoder_top_n,
            cross_encoder_model=args.cross_encoder_model,
        )
        if args.output != DEFAULT_MATCHING_REPORT_PATH:
            write_matching_report(report, output_path=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
