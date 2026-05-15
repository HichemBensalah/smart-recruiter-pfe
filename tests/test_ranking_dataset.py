from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/ranking/datasets/ranking_dataset_unlabeled.jsonl"
SUMMARY_PATH = ROOT / "data/ranking/datasets/ranking_dataset_summary.json"


def read_rows() -> list[dict]:
    return [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_global_unlabeled_dataset_exists() -> None:
    assert DATASET_PATH.exists()
    assert SUMMARY_PATH.exists()
    assert read_rows()


def test_unlabeled_dataset_contract() -> None:
    for row in read_rows():
        assert row["query_group"] == row["job_id"]
        assert row["label"] is None
        assert row["split"] is None
        assert len(row["features"]) == 12
        assert all(isinstance(value, (int, float)) for value in row["features"].values())


def test_dataset_summary_is_consistent() -> None:
    rows = read_rows()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["total_rows"] == len(rows)
    assert summary["feature_count"] == 12
    assert summary["label_status"] == "unlabeled"
