from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABELING_STRATEGY = "multi_criteria_v2"
METHODOLOGY_WARNING = "Ces labels sont des pseudo-labels métier contrôlés et non des labels recruteur."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build controlled business pseudo-labels for ranking experiments.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--annotation-sample", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _feature_float(features: dict[str, Any], key: str) -> float:
    return float(features[key])


def pseudo_label_v1(features: dict[str, Any]) -> int:
    must_have_coverage = _feature_float(features, "must_have_coverage")
    missing_required_count = _feature_float(features, "missing_required_count")
    experience_match_score = _feature_float(features, "experience_match_score")
    seniority_alignment = _feature_float(features, "seniority_alignment")
    reliability_score = _feature_float(features, "reliability_score")
    hallucination_risk_encoded = _feature_float(features, "hallucination_risk_encoded")
    required_skills_overlap = _feature_float(features, "required_skills_overlap")
    vector_similarity = _feature_float(features, "vector_similarity")

    if (
        must_have_coverage >= 0.8
        and missing_required_count <= 1
        and experience_match_score >= 0.7
        and seniority_alignment >= 0.7
        and reliability_score >= 0.7
        and hallucination_risk_encoded <= 0.5
    ):
        return 3
    if (
        must_have_coverage >= 0.6
        and missing_required_count <= 2
        and experience_match_score >= 0.5
        and reliability_score >= 0.6
    ):
        return 2
    if must_have_coverage >= 0.4 or required_skills_overlap >= 0.4 or vector_similarity >= 0.4:
        return 1
    return 0


def pseudo_label_v2(features: dict[str, Any]) -> int:
    must_have_coverage = _feature_float(features, "must_have_coverage")
    missing_required_count = _feature_float(features, "missing_required_count")
    experience_match_score = _feature_float(features, "experience_match_score")
    reliability_score = _feature_float(features, "reliability_score")
    hallucination_risk_encoded = _feature_float(features, "hallucination_risk_encoded")
    required_skills_overlap = _feature_float(features, "required_skills_overlap")
    vector_similarity = _feature_float(features, "vector_similarity")

    if (
        must_have_coverage >= 0.8
        and missing_required_count <= 1
        and experience_match_score >= 0.6
        and reliability_score >= 0.6
        and hallucination_risk_encoded <= 0.5
    ):
        return 3
    if (
        must_have_coverage >= 0.6
        and missing_required_count <= 3
        and experience_match_score >= 0.4
        and reliability_score >= 0.5
    ):
        return 2
    if must_have_coverage >= 0.3 and (required_skills_overlap >= 0.3 or vector_similarity >= 0.35):
        return 1
    return 0


def pseudo_label(features: dict[str, Any]) -> int:
    return pseudo_label_v2(features)


def label_distribution(labels: list[int]) -> Counter[int]:
    return Counter(labels)


def label_percentages(label_counts: Counter[int], total_rows: int) -> dict[str, float]:
    if total_rows == 0:
        return {str(label): 0.0 for label in range(4)}
    return {str(label): round((label_counts[label] / total_rows) * 100, 2) for label in range(4)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset)
    labeled_rows: list[dict[str, Any]] = []
    label_counts: Counter[int] = Counter()
    rows_per_job: Counter[str] = Counter()
    distribution_per_job: dict[str, Counter[int]] = {}

    for row in rows:
        item = dict(row)
        label = pseudo_label(item["features"])
        item["label"] = label
        item["label_type"] = "pseudo"
        item["labeling_strategy"] = LABELING_STRATEGY
        labeled_rows.append(item)
        label_counts[label] += 1
        job_id = str(item["job_id"])
        rows_per_job[job_id] += 1
        distribution_per_job.setdefault(job_id, Counter())[label] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in labeled_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    positive_count = label_counts[2] + label_counts[3]
    positive_rate = round((positive_count / len(labeled_rows)) * 100, 2) if labeled_rows else 0.0
    summary = {
        "generated_at_utc": utc_now(),
        "total_rows": len(labeled_rows),
        "label_distribution": {str(label): label_counts[label] for label in range(4)},
        "label_percentages": label_percentages(label_counts, len(labeled_rows)),
        "positive_count": positive_count,
        "positive_rate": positive_rate,
        "rows_per_job": dict(sorted(rows_per_job.items())),
        "distribution_per_job": {
            job_id: {str(label): counts[label] for label in range(4)}
            for job_id, counts in sorted(distribution_per_job.items())
        },
        "jobs": sorted(rows_per_job),
        "labeling_strategy": LABELING_STRATEGY,
        "methodology_warning": METHODOLOGY_WARNING,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
