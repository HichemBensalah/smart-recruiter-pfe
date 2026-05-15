from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.ranking.dataset import build_unlabeled_dataset, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a global unlabeled ranking dataset from per-job feature JSONL files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--include-job-ids", default=None, help="Comma-separated job_id allow-list.")
    return parser.parse_args()


def parse_job_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    args = parse_args()
    rows, summary = build_unlabeled_dataset(args.input_dir, args.dataset_id, parse_job_ids(args.include_job_ids))
    write_jsonl(rows, args.output)
    write_json(summary, args.summary_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
