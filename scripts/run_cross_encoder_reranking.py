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

from scripts.run_matching_v3_normalized import (  # noqa: E402
    JOB_PROFILE_PATH,
    PROFILES_DIR,
    TAXONOMY_PATH,
    build_variant_map,
    load_profile_payloads,
    load_profiles_by_id,
    load_taxonomy,
    recommend_candidates_with_normalized_profiles,
)
from src.core.matching.job_text_builder import _normalize_string_list  # noqa: E402
from src.core.matching.matching_quality_filters import build_display_name  # noqa: E402
from src.core.matching.profile_text_builder import flatten_experiences  # noqa: E402
from src.core.matching.recommender import load_id_map  # noqa: E402
from src.core.matching.skill_normalizer import normalize_skills  # noqa: E402
from src.core.retrieval.cross_encoder_reranker import (  # noqa: E402
    DEFAULT_CROSS_ENCODER_MODEL,
    rerank_candidates_with_cross_encoder,
)


NORMALIZED_MATCHING_REPORT_PATH = Path("docs/reports/matching/v3/matching_report_v3_normalized.json")
CROSS_ENCODER_REPORT_PATH = Path("docs/reports/retrieval/cross_encoder_reranking_report.json")
RERANK_INPUT_SIZE = 20
FINAL_TOP_K = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_cross_encoder_job_text(job_profile: dict[str, Any]) -> str:
    job_title = _clean_text(job_profile.get("job_title"))
    required_skills = ", ".join(normalize_skills(_normalize_string_list(job_profile.get("required_skills"))))
    nice_to_have = ", ".join(normalize_skills(_normalize_string_list(job_profile.get("nice_to_have_skills"))))
    responsibilities = "\n".join(_normalize_string_list(job_profile.get("responsibilities")))
    chunks = [
        f"Job Title: {job_title}" if job_title else None,
        f"Must Have Skills: {required_skills}" if required_skills else None,
        f"Nice To Have Skills: {nice_to_have}" if nice_to_have else None,
        f"Responsibilities:\n{responsibilities}" if responsibilities else None,
    ]
    return "\n\n".join(chunk for chunk in chunks if chunk)


def build_cross_encoder_candidate_text(profile: dict[str, Any]) -> str:
    bio = profile.get("bio") or {}
    expertise = profile.get("expertise") or {}
    full_name, _, _ = build_display_name(bio.get("full_name"), profile.get("candidate_id"))
    hard_skills = [skill for skill in (expertise.get("hard_skills") or []) if isinstance(skill, str)]
    summary = _clean_text(expertise.get("summary"))
    experiences = flatten_experiences(profile.get("experiences"))
    chunks = [
        f"Candidate: {full_name}" if full_name else None,
        f"Hard Skills: {', '.join(hard_skills)}" if hard_skills else None,
        f"Summary: {summary}" if summary else None,
        f"Experiences:\n{experiences}" if experiences else None,
    ]
    return "\n\n".join(chunk for chunk in chunks if chunk)


def load_normalized_profiles() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    variant_map = build_variant_map(taxonomy)
    payloads = load_profile_payloads(PROFILES_DIR)
    return load_profiles_by_id(payloads, load_id_map(), variant_map)


def load_or_compute_top20(
    normalized_report: dict[str, Any],
    job_profile: dict[str, Any],
    profiles_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    existing = list(((normalized_report.get("results") or [{}])[0]).get("recommendations") or [])
    if len(existing) >= RERANK_INPUT_SIZE:
        return existing[:RERANK_INPUT_SIZE], "matching_report_v3_normalized_top20"
    computed = recommend_candidates_with_normalized_profiles(
        job_profile,
        profiles_by_id,
        top_k=RERANK_INPUT_SIZE,
    )
    return computed, (
        "computed_top20_from_existing_faiss_index_in_memory; "
        f"input_report_only_had_{len(existing)}_recommendations"
    )


def build_rerank_rows(
    top20: list[dict[str, Any]],
    profiles_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(top20, start=1):
        profile_id = str(item.get("matched_profile_id") or "")
        profile = profiles_by_id.get(profile_id)
        if not profile:
            continue
        faiss_v3_rank = int(item.get("rank") or index)
        rows.append(
            {
                **item,
                "faiss_rank": faiss_v3_rank,
                "faiss_v3_rank": faiss_v3_rank,
                "raw_faiss_retrieval_rank": item.get("faiss_rank"),
                "original_faiss_rank": faiss_v3_rank,
                "original_faiss_score": item.get("faiss_score"),
                "candidate_text": build_cross_encoder_candidate_text(profile),
            }
        )
    return rows


def slim_candidate(item: dict[str, Any]) -> dict[str, Any]:
    faiss_rank = int(item.get("faiss_v3_rank") or item.get("original_faiss_rank") or item.get("faiss_rank") or 0)
    ce_rank = int(item.get("cross_encoder_rank") or 0)
    return {
        "candidate_id": item.get("candidate_id"),
        "matched_profile_id": item.get("matched_profile_id"),
        "full_name": item.get("full_name"),
        "faiss_rank": faiss_rank,
        "raw_faiss_retrieval_rank": item.get("raw_faiss_retrieval_rank"),
        "faiss_score": item.get("faiss_score"),
        "ce_rank": ce_rank,
        "ce_score": item.get("cross_encoder_score"),
        "ce_score_normalized": item.get("cross_encoder_score_normalized"),
        "rank_change": faiss_rank - ce_rank,
        "must_have_coverage": item.get("must_have_coverage"),
        "matched_skills": item.get("matched_skills") or [],
        "missing_required_skills": item.get("missing_required_skills") or [],
    }


def compare_top10(faiss_top10: list[dict[str, Any]], ce_top10: list[dict[str, Any]]) -> dict[str, Any]:
    faiss_keys = {candidate_key(item): item for item in faiss_top10}
    ce_keys = {candidate_key(item): item for item in ce_top10}
    entered = [slim_candidate(item) for key, item in ce_keys.items() if key not in faiss_keys]
    dropped = [slim_candidate(item) for key, item in faiss_keys.items() if key not in ce_keys]
    stayed = [slim_candidate(item) for key, item in ce_keys.items() if key in faiss_keys]
    moved_up = [item for item in stayed if item["rank_change"] > 0]
    moved_down = [item for item in stayed if item["rank_change"] < 0]
    unchanged = [item for item in stayed if item["rank_change"] == 0]
    interesting = [
        item
        for item in [slim_candidate(item) for item in ce_top10]
        if abs(int(item.get("rank_change") or 0)) > 2
    ]
    return {
        "entered_top10": entered,
        "dropped_out_top10": dropped,
        "stayed_top10": stayed,
        "moved_up": moved_up,
        "moved_down": moved_down,
        "unchanged": unchanged,
        "interesting_rank_disagreements": interesting,
    }


def candidate_key(item: dict[str, Any]) -> str:
    return str(item.get("candidate_id") or item.get("matched_profile_id") or item.get("full_name") or "")


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def main() -> None:
    start = time.perf_counter()
    normalized_report = read_json(NORMALIZED_MATCHING_REPORT_PATH)
    job_profile = read_json(JOB_PROFILE_PATH)
    profiles_by_id, normalization_stats = load_normalized_profiles()
    top20, input_source = load_or_compute_top20(normalized_report, job_profile, profiles_by_id)
    rerank_rows = build_rerank_rows(top20, profiles_by_id)
    job_text = build_cross_encoder_job_text(job_profile)

    reranked = rerank_candidates_with_cross_encoder(
        job_text=job_text,
        candidates=rerank_rows,
        candidate_text_key="candidate_text",
        top_k=RERANK_INPUT_SIZE,
        model_name=DEFAULT_CROSS_ENCODER_MODEL,
    )
    ce_top10 = reranked[:FINAL_TOP_K]
    faiss_top10 = rerank_rows[:FINAL_TOP_K]
    comparison = compare_top10(faiss_top10, ce_top10)
    latency = round(time.perf_counter() - start, 4)

    report = {
        "generated_at_utc": utc_now(),
        "status": "success" if all(item.get("cross_encoder_score") is not None for item in ce_top10) else "fallback_or_partial",
        "model_used": DEFAULT_CROSS_ENCODER_MODEL,
        "input_matching_report": str(NORMALIZED_MATCHING_REPORT_PATH),
        "input_source": input_source,
        "job_profile_file": str(JOB_PROFILE_PATH),
        "profiles_dir": str(PROFILES_DIR),
        "skills_taxonomy_file": str(TAXONOMY_PATH),
        "rerank_input_size": len(rerank_rows),
        "final_top_k": FINAL_TOP_K,
        "latency_total_seconds": latency,
        "metadata": {
            "baseline_status": "experimental_cross_encoder_reranking",
            "faiss_index_rebuilt": False,
            "module1_rerun": False,
            "module2_rerun": False,
            "original_profiles_overwritten": False,
            "normalized_matching_report_overwritten": False,
            "rank_change_definition": "positive means candidate moved up after CrossEncoder: faiss_rank - ce_rank",
            "normalization_profiles_loaded": normalization_stats.get("profiles_normalized"),
        },
        "job_text_preview": job_text[:600],
        "top_20_faiss_v3_input": [slim_candidate(item) for item in rerank_rows],
        "top_10_reranked_candidates": [slim_candidate(item) for item in ce_top10],
        "entered_top10": comparison["entered_top10"],
        "dropped_out_top10": comparison["dropped_out_top10"],
        "stayed_top10": comparison["stayed_top10"],
        "moved_up": comparison["moved_up"],
        "moved_down": comparison["moved_down"],
        "unchanged": comparison["unchanged"],
        "interesting_rank_disagreements": comparison["interesting_rank_disagreements"],
    }
    write_json(CROSS_ENCODER_REPORT_PATH, report)

    summary = {
        "file_created": str(CROSS_ENCODER_REPORT_PATH),
        "top_10_final": report["top_10_reranked_candidates"],
        "entered_top10": report["entered_top10"],
        "dropped_out_top10": report["dropped_out_top10"],
        "moved_up": report["moved_up"],
        "moved_down": report["moved_down"],
        "latency_total_seconds": latency,
        "interesting_rank_disagreements": report["interesting_rank_disagreements"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
