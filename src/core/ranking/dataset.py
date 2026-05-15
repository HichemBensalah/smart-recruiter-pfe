from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_FEATURE_COUNT = 12


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_feature_files(input_dir: str | Path) -> list[Path]:
    return sorted(Path(input_dir).glob("*.jsonl"))


def build_unlabeled_dataset(
    input_dir: str | Path,
    dataset_id: str,
    include_job_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_rows: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    rows_per_job: Counter[str] = Counter()
    warnings: list[str] = []

    for feature_path in discover_feature_files(input_dir):
        for row in read_jsonl(feature_path):
            job_id = str(row["job_id"])
            if include_job_ids is not None and job_id not in include_job_ids:
                continue
            features = row.get("features") or {}
            names = list(features.keys())
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                warnings.append(f"Feature name mismatch in {feature_path}")
            if len(features) != EXPECTED_FEATURE_COUNT:
                warnings.append(f"Expected {EXPECTED_FEATURE_COUNT} features in {feature_path}, got {len(features)}")

            rows_per_job[job_id] += 1
            dataset_rows.append(
                {
                    "dataset_id": dataset_id,
                    "job_id": job_id,
                    "query_group": job_id,
                    "candidate_id": row["candidate_id"],
                    "profile_id": row["profile_id"],
                    "rank": row["rank"],
                    "source": row["source"],
                    "features": features,
                    "label": None,
                    "split": None,
                }
            )

    summary = {
        "generated_at_utc": utc_now(),
        "dataset_id": dataset_id,
        "input_dir": str(input_dir),
        "include_job_ids": sorted(include_job_ids) if include_job_ids else None,
        "total_rows": len(dataset_rows),
        "jobs": sorted(rows_per_job),
        "rows_per_job": dict(sorted(rows_per_job.items())),
        "feature_count": len(feature_names or []),
        "feature_names": feature_names or [],
        "label_status": "unlabeled",
        "warnings": warnings,
    }
    return dataset_rows, summary
