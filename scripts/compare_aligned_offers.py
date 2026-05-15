from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare original pseudo-label dataset with aligned-offer pseudo-label dataset.")
    parser.add_argument("--old-summary", type=Path, required=True)
    parser.add_argument("--aligned-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def positive_count(summary: dict[str, Any]) -> int:
    distribution = summary.get("label_distribution", {})
    return int(distribution.get("2", 0)) + int(distribution.get("3", 0))


def positive_rate(summary: dict[str, Any]) -> float:
    total_rows = int(summary.get("total_rows", 0))
    return round((positive_count(summary) / total_rows) * 100, 2) if total_rows else 0.0


def recommendation(old_summary: dict[str, Any], aligned_summary: dict[str, Any]) -> str:
    old_positive = positive_count(old_summary)
    aligned_positive = positive_count(aligned_summary)
    if aligned_positive < 25:
        return "not_ready_for_training_adjust_rule_or_jobs"
    if aligned_positive <= old_positive:
        return "not_improved_enough"
    return "aligned_dataset_can_be_considered_for_next_evaluation_step"


def build_report(old_summary: dict[str, Any], aligned_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "old_dataset": {
            "total_rows": old_summary.get("total_rows"),
            "label_distribution": old_summary.get("label_distribution"),
            "positive_count": positive_count(old_summary),
            "positive_rate": positive_rate(old_summary),
            "distribution_per_job": old_summary.get("distribution_per_job"),
        },
        "aligned_dataset": {
            "total_rows": aligned_summary.get("total_rows"),
            "label_distribution": aligned_summary.get("label_distribution"),
            "positive_count": positive_count(aligned_summary),
            "positive_rate": positive_rate(aligned_summary),
            "distribution_per_job": aligned_summary.get("distribution_per_job"),
        },
        "delta": {
            "positive_count_delta": positive_count(aligned_summary) - positive_count(old_summary),
            "positive_rate_delta": round(positive_rate(aligned_summary) - positive_rate(old_summary), 2),
        },
        "recommendation": recommendation(old_summary, aligned_summary),
        "methodology_warning": "Comparaison basée sur des pseudo-labels métier contrôlés, pas sur des labels recruteur.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    old = report["old_dataset"]
    aligned = report["aligned_dataset"]
    lines = [
        "# Aligned Offers Comparison Report",
        "",
        "## Summary",
        f"- Old dataset rows: {old['total_rows']}",
        f"- Aligned dataset rows: {aligned['total_rows']}",
        f"- Old positive count: {old['positive_count']} ({old['positive_rate']}%)",
        f"- Aligned positive count: {aligned['positive_count']} ({aligned['positive_rate']}%)",
        f"- Positive count delta: {report['delta']['positive_count_delta']}",
        f"- Positive rate delta: {report['delta']['positive_rate_delta']} points",
        f"- Recommendation: {report['recommendation']}",
        "",
        "## Old Label Distribution",
    ]
    for label, count in old["label_distribution"].items():
        lines.append(f"- label {label}: {count}")
    lines.append("")
    lines.append("## Aligned Label Distribution")
    for label, count in aligned["label_distribution"].items():
        lines.append(f"- label {label}: {count}")
    lines.append("")
    lines.append("## Aligned Distribution Per Job")
    for job_id, distribution in aligned["distribution_per_job"].items():
        lines.append(f"- {job_id}: {distribution}")
    lines.extend(["", "## Methodological Warning", report["methodology_warning"]])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    old_summary = load_json(args.old_summary)
    aligned_summary = load_json(args.aligned_summary)
    report = build_report(old_summary, aligned_summary)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
