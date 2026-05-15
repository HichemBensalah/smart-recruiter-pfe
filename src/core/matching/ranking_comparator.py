from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .matching_quality_filters import is_suspect_name


def load_top10(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results") or []
    if not results:
        return []
    recommendations = results[0].get("recommendations") or []
    top10: list[dict[str, Any]] = []
    for index, recommendation in enumerate(recommendations[:10], start=1):
        item = dict(recommendation)
        item.setdefault("rank", index)
        top10.append(item)
    return top10


def summarize_top10(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": item.get("rank"),
            "candidate_id": item.get("candidate_id"),
            "full_name": item.get("full_name"),
            "final_score": item.get("final_score"),
            "profile_kind": item.get("profile_kind"),
            "hallucination_risk": item.get("hallucination_risk"),
            "display_name_quality": item.get("display_name_quality"),
        }
        for item in items
    ]


def compare_rankings(old_items: list[dict[str, Any]], v2_items: list[dict[str, Any]], v3_items: list[dict[str, Any]]) -> dict[str, Any]:
    old_map = {str(item.get("candidate_id")): item for item in old_items}
    v2_map = {str(item.get("candidate_id")): item for item in v2_items}
    v3_map = {str(item.get("candidate_id")): item for item in v3_items}
    all_ids = list(dict.fromkeys(list(old_map.keys()) + list(v2_map.keys()) + list(v3_map.keys())))

    movements: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    removed_from_top10: list[dict[str, Any]] = []
    new_in_v3: list[dict[str, Any]] = []

    for candidate_id in all_ids:
        old_item = old_map.get(candidate_id)
        v2_item = v2_map.get(candidate_id)
        v3_item = v3_map.get(candidate_id)
        label = (
            (v3_item or v2_item or old_item or {}).get("full_name")
            or f"Candidate (ID: {candidate_id})"
        )

        movement = {
            "candidate_id": candidate_id,
            "full_name": label,
            "old_rank": old_item.get("rank") if old_item else None,
            "grounded_v2_rank": v2_item.get("rank") if v2_item else None,
            "grounded_v3_rank": v3_item.get("rank") if v3_item else None,
            "old_score": old_item.get("final_score") if old_item else None,
            "grounded_v2_score": v2_item.get("final_score") if v2_item else None,
            "grounded_v3_score": v3_item.get("final_score") if v3_item else None,
            "grounded_v3_hallucination_risk": v3_item.get("hallucination_risk") if v3_item else None,
            "grounded_v3_display_name_quality": v3_item.get("display_name_quality") if v3_item else None,
        }
        movements.append(movement)

        if old_item and v3_item and v3_item.get("rank", 999) < old_item.get("rank", 999):
            promoted.append(movement)
        if old_item and v3_item and v3_item.get("rank", 999) > old_item.get("rank", 999):
            dropped.append(movement)
        if old_item and not v3_item:
            removed_from_top10.append(movement)
        if v3_item and not old_item:
            new_in_v3.append(movement)

    suspect_names_v2 = [
        item.get("full_name")
        for item in v2_items
        if is_suspect_name(item.get("full_name"))
    ]
    suspect_names_v3 = [
        item.get("full_name")
        for item in v3_items
        if item.get("display_name_quality") == "weak"
        or item.get("name_warning") == "missing_or_rejected_full_name"
    ]

    return {
        "top10_v1": summarize_top10(old_items),
        "top10_grounded_v2": summarize_top10(v2_items),
        "top10_grounded_v3": summarize_top10(v3_items),
        "movements": movements,
        "candidates_that_rise": promoted,
        "candidates_that_drop": dropped,
        "candidates_removed_from_top10": removed_from_top10,
        "new_candidates_in_top10": new_in_v3,
        "suspect_display_names_v2": suspect_names_v2,
        "suspect_display_names_v3": suspect_names_v3,
        "analysis": {
            "skill_normalizer_effect": "Canonical aliases reduce fragmentation such as Fast API -> FastAPI and REST API design -> REST API.",
            "grounded_quality_effect": "V3 applies reliability, profile_kind, hallucination_risk and nullified-field penalties on top of FAISS similarity.",
            "display_name_fix_effect": "Suspicious names are replaced with Candidate (ID: ...) instead of surfacing OCR titles or template text.",
            "high_risk_penalty_effect": "Medium and high hallucination-risk profiles receive softer score multipliers rather than exclusion.",
        },
    }


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Matching Ranking Comparison Grounded V3",
        "",
        "## Top 10 V1",
        "",
    ]
    lines.extend(_format_top10(report["top10_v1"]))
    lines.extend(["", "## Top 10 Grounded V2", ""])
    lines.extend(_format_top10(report["top10_grounded_v2"]))
    lines.extend(["", "## Top 10 Grounded V3", ""])
    lines.extend(_format_top10(report["top10_grounded_v3"]))
    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            f"- Suspect display names in V2: {len(report['suspect_display_names_v2'])}",
            f"- Suspect display names in V3: {len(report['suspect_display_names_v3'])}",
            f"- New candidates in V3 top 10: {len(report['new_candidates_in_top10'])}",
            f"- Removed from top 10 in V3: {len(report['candidates_removed_from_top10'])}",
            f"- Skill normalizer effect: {report['analysis']['skill_normalizer_effect']}",
            f"- Grounded quality effect: {report['analysis']['grounded_quality_effect']}",
            f"- Display-name fix: {report['analysis']['display_name_fix_effect']}",
            f"- High-risk penalty effect: {report['analysis']['high_risk_penalty_effect']}",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _format_top10(items: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for item in items:
        rows.append(
            f"- #{item.get('rank')} {item.get('full_name')} | score={item.get('final_score')} | "
            f"profile_kind={item.get('profile_kind')} | risk={item.get('hallucination_risk')} | "
            f"display_name_quality={item.get('display_name_quality')}"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare V1, grounded V2 and grounded V3 matching reports.")
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--grounded-v2", type=Path, required=True)
    parser.add_argument("--grounded-v3", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old_items = load_top10(args.old)
    v2_items = load_top10(args.grounded_v2)
    v3_items = load_top10(args.grounded_v3)
    comparison = compare_rankings(old_items, v2_items, v3_items)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(comparison, args.output_md)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
