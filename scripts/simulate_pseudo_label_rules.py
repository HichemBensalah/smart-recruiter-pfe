from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pseudo_labels import pseudo_label_v1, pseudo_label_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate pseudo-labeling rule distributions without writing labels.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def distribution_for(rows: list[dict[str, Any]], rule: Callable[[dict[str, Any]], int]) -> Counter[int]:
    return Counter(rule(row["features"]) for row in rows)


def percentages(counts: Counter[int], total_rows: int) -> dict[str, float]:
    if total_rows == 0:
        return {str(label): 0.0 for label in range(4)}
    return {str(label): round((counts[label] / total_rows) * 100, 2) for label in range(4)}


def distribution_per_job(rows: list[dict[str, Any]], rule: Callable[[dict[str, Any]], int]) -> dict[str, dict[str, int]]:
    per_job: dict[str, Counter[int]] = {}
    for row in rows:
        job_id = str(row["job_id"])
        per_job.setdefault(job_id, Counter())[rule(row["features"])] += 1
    return {
        job_id: {str(label): counts[label] for label in range(4)}
        for job_id, counts in sorted(per_job.items())
    }


def recommendation(total_rows: int, counts_v2: Counter[int]) -> str:
    positive_count = counts_v2[2] + counts_v2[3]
    if total_rows and positive_count > total_rows * 0.6:
        return "v2_too_loose"
    if positive_count < 25:
        return "v2_still_too_strict"
    if counts_v2[2] >= 10 and counts_v2[3] >= 10:
        return "v2_acceptable"
    return "v2_acceptable"


def build_report(dataset: Path) -> dict[str, Any]:
    before_hash = sha256_file(dataset)
    rows = read_jsonl(dataset)
    counts_v1 = distribution_for(rows, pseudo_label_v1)
    counts_v2 = distribution_for(rows, pseudo_label_v2)
    after_hash = sha256_file(dataset)
    total_rows = len(rows)
    positive_count_v1 = counts_v1[2] + counts_v1[3]
    positive_count_v2 = counts_v2[2] + counts_v2[3]

    return {
        "generated_at_utc": utc_now(),
        "dataset_path": str(dataset),
        "dataset_sha256_before": before_hash,
        "dataset_sha256_after": after_hash,
        "dataset_unchanged": before_hash == after_hash,
        "total_rows": total_rows,
        "distribution_v1": {str(label): counts_v1[label] for label in range(4)},
        "distribution_v2": {str(label): counts_v2[label] for label in range(4)},
        "percentage_v1": percentages(counts_v1, total_rows),
        "percentage_v2": percentages(counts_v2, total_rows),
        "positive_count_v1": positive_count_v1,
        "positive_count_v2": positive_count_v2,
        "positive_rate_v1": round((positive_count_v1 / total_rows) * 100, 2) if total_rows else 0.0,
        "positive_rate_v2": round((positive_count_v2 / total_rows) * 100, 2) if total_rows else 0.0,
        "distribution_per_job_v1": distribution_per_job(rows, pseudo_label_v1),
        "distribution_per_job_v2": distribution_per_job(rows, pseudo_label_v2),
        "recommendation": recommendation(total_rows, counts_v2),
    }


def main() -> None:
    args = parse_args()
    report = build_report(args.dataset)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
