from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def first_result(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results") or []
    return results[0] if results else {}


def top10(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list((first_result(report).get("recommendations") or [])[:10])


def candidate_key(item: dict[str, Any]) -> str:
    return str(item.get("candidate_id") or item.get("matched_profile_id") or item.get("full_name") or "")


def slim(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": item.get("rank"),
        "candidate_id": item.get("candidate_id"),
        "full_name": item.get("full_name"),
        "final_score": item.get("final_score"),
        "faiss_rank": item.get("faiss_rank"),
        "faiss_score": item.get("faiss_score"),
        "cross_encoder_rank": item.get("cross_encoder_rank"),
        "cross_encoder_score": item.get("cross_encoder_score"),
        "profile_kind": item.get("profile_kind"),
        "hallucination_risk": item.get("hallucination_risk"),
        "must_have_coverage": item.get("must_have_coverage"),
        "matched_skills": item.get("matched_skills") or [],
        "missing_required_skills": item.get("missing_required_skills") or [],
    }


def compare_reports(baseline: dict[str, Any], cross_encoder: dict[str, Any]) -> dict[str, Any]:
    baseline_top = top10(baseline)
    cross_top = top10(cross_encoder)
    baseline_ranks = {candidate_key(item): int(item.get("rank") or index + 1) for index, item in enumerate(baseline_top)}
    cross_ranks = {candidate_key(item): int(item.get("rank") or index + 1) for index, item in enumerate(cross_top)}

    moved_up = []
    moved_down = []
    for key, after_rank in cross_ranks.items():
        before_rank = baseline_ranks.get(key)
        if before_rank is None:
            continue
        delta = before_rank - after_rank
        item = next(item for item in cross_top if candidate_key(item) == key)
        movement = {"candidate_id": key, "full_name": item.get("full_name"), "before_rank": before_rank, "after_rank": after_rank, "delta": delta}
        if delta > 0:
            moved_up.append(movement)
        elif delta < 0:
            moved_down.append(movement)

    exited = [slim(item) for item in baseline_top if candidate_key(item) not in cross_ranks]
    entered = [slim(item) for item in cross_top if candidate_key(item) not in baseline_ranks]
    baseline_top1 = baseline_top[0] if baseline_top else {}
    cross_top1 = cross_top[0] if cross_top else {}

    medium_baseline = {candidate_key(item): item.get("rank") for item in baseline_top if item.get("hallucination_risk") == "medium"}
    medium_cross = {candidate_key(item): item.get("rank") for item in cross_top if item.get("hallucination_risk") == "medium"}

    better_skill_candidates = [
        slim(item)
        for item in cross_top
        if float(item.get("must_have_coverage") or 0.0) >= 0.8
    ]
    qualitative = _qualitative_assessment(baseline_top, cross_top)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_compared_to": "baseline1_faiss_matching_v3",
        "top_10_baseline_faiss": [slim(item) for item in baseline_top],
        "top_10_faiss_cross_encoder": [slim(item) for item in cross_top],
        "candidates_moved_up": moved_up,
        "candidates_moved_down": moved_down,
        "candidates_exited_top_10": exited,
        "new_candidates_in_top_10": entered,
        "top1_before": slim(baseline_top1) if baseline_top1 else None,
        "top1_after": slim(cross_top1) if cross_top1 else None,
        "top1_score_before": baseline_top1.get("final_score"),
        "top1_score_after": cross_top1.get("final_score"),
        "hichem_remains_top1": cross_top1.get("full_name") == "Hichem Bensalah",
        "medium_risk_ranks_before": medium_baseline,
        "medium_risk_ranks_after": medium_cross,
        "high_skill_coverage_candidates_after": better_skill_candidates,
        "qualitative_analysis": qualitative,
    }


def _qualitative_assessment(baseline_top: list[dict[str, Any]], cross_top: list[dict[str, Any]]) -> str:
    if not baseline_top or not cross_top:
        return "Comparison is inconclusive because one report has no recommendations."
    baseline_top1 = baseline_top[0].get("full_name")
    cross_top1 = cross_top[0].get("full_name")
    medium_after = sum(1 for item in cross_top if item.get("hallucination_risk") == "medium")
    strong_coverage_after = sum(1 for item in cross_top if float(item.get("must_have_coverage") or 0.0) >= 0.8)
    if baseline_top1 == cross_top1 and strong_coverage_after >= 2:
        return (
            "CrossEncoder preserves the strongest top candidate while giving the final scorer a finer semantic signal. "
            f"The new top 10 contains {strong_coverage_after} candidates with >=0.8 must-have coverage and {medium_after} medium-risk candidates."
        )
    return (
        "CrossEncoder changes the ranking materially. Review moved candidates and medium-risk positions before adopting it as default."
    )


def build_markdown(report: dict[str, Any]) -> str:
    def lines_for(items: list[dict[str, Any]]) -> list[str]:
        return [
            f"- #{item.get('rank')} {item.get('full_name')} | final={item.get('final_score')} | "
            f"faiss_rank={item.get('faiss_rank')} | ce_rank={item.get('cross_encoder_rank')} | risk={item.get('hallucination_risk')}"
            for item in items
        ]

    lines = [
        "# CrossEncoder Comparison Report",
        "",
        f"- `generated_at_utc`: {report['generated_at_utc']}",
        f"- `baseline_compared_to`: {report['baseline_compared_to']}",
        f"- `hichem_remains_top1`: {str(report['hichem_remains_top1']).lower()}",
        f"- `top1_score_before`: {report['top1_score_before']}",
        f"- `top1_score_after`: {report['top1_score_after']}",
        "",
        "## Top 10 Baseline FAISS",
        "",
        *lines_for(report["top_10_baseline_faiss"]),
        "",
        "## Top 10 FAISS + CrossEncoder",
        "",
        *lines_for(report["top_10_faiss_cross_encoder"]),
        "",
        "## Candidates Moved Up",
        "",
        "```json",
        json.dumps(report["candidates_moved_up"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Candidates Moved Down",
        "",
        "```json",
        json.dumps(report["candidates_moved_down"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## New / Exited Top 10",
        "",
        "```json",
        json.dumps(
            {
                "new_candidates_in_top_10": report["new_candidates_in_top_10"],
                "candidates_exited_top_10": report["candidates_exited_top_10"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Risk And Skill Coverage",
        "",
        "```json",
        json.dumps(
            {
                "medium_risk_ranks_before": report["medium_risk_ranks_before"],
                "medium_risk_ranks_after": report["medium_risk_ranks_after"],
                "high_skill_coverage_candidates_after": report["high_skill_coverage_candidates_after"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Qualitative Analysis",
        "",
        report["qualitative_analysis"],
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline FAISS matching with FAISS + CrossEncoder reranking.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cross-encoder", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_reports(read_json(args.baseline), read_json(args.cross_encoder))
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
