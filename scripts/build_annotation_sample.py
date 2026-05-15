from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "job_id",
    "query_group",
    "candidate_id",
    "profile_id",
    "rank",
    "final_score_v3",
    "vector_similarity",
    "must_have_coverage",
    "required_skills_overlap",
    "experience_match_score",
    "seniority_alignment",
    "reliability_score",
    "hallucination_risk_encoded",
    "missing_required_count",
    "matched_required_count",
    "label",
    "comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced annotation sample from an unlabeled ranking dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--middle-n", type=int, default=5)
    parser.add_argument("--low-n", type=int, default=5)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_rows(rows: list[dict[str, Any]], top_n: int, middle_n: int, low_n: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: int(row["rank"]))
    if len(sorted_rows) <= top_n + middle_n + low_n:
        return sorted_rows

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(candidates: list[dict[str, Any]], limit: int) -> None:
        for row in candidates:
            key = (str(row["job_id"]), str(row["candidate_id"]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len([item for item in selected if item in candidates]) >= limit:
                break

    add(sorted_rows[:top_n], top_n)
    middle_start = max(top_n, (len(sorted_rows) // 2) - (middle_n // 2))
    add(sorted_rows[middle_start : middle_start + middle_n], middle_n)
    add(list(reversed(sorted_rows[-low_n:])), low_n)
    return sorted(selected, key=lambda row: int(row["rank"]))


def to_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    features = row["features"]
    return {
        "job_id": row["job_id"],
        "query_group": row["query_group"],
        "candidate_id": row["candidate_id"],
        "profile_id": row["profile_id"],
        "rank": row["rank"],
        "final_score_v3": features["final_score_v3"],
        "vector_similarity": features["vector_similarity"],
        "must_have_coverage": features["must_have_coverage"],
        "required_skills_overlap": features["required_skills_overlap"],
        "experience_match_score": features["experience_match_score"],
        "seniority_alignment": features["seniority_alignment"],
        "reliability_score": features["reliability_score"],
        "hallucination_risk_encoded": features["hallucination_risk_encoded"],
        "missing_required_count": features["missing_required_count"],
        "matched_required_count": features["matched_required_count"],
        "label": "",
        "comment": "",
    }


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.dataset):
        grouped[str(row["job_id"])].append(row)

    output_rows: list[dict[str, Any]] = []
    for _, rows in sorted(grouped.items()):
        output_rows.extend(to_csv_row(row) for row in select_rows(rows, args.top_n, args.middle_n, args.low_n))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)

    print(json.dumps({"rows": len(output_rows), "jobs": sorted(grouped)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
