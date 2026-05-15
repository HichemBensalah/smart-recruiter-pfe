from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_PATH = ROOT / "data/ranking/annotations/annotation_sample.csv"


def read_rows() -> list[dict[str, str]]:
    with ANNOTATION_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_annotation_sample_exists_and_has_no_auto_labels() -> None:
    assert ANNOTATION_PATH.exists()
    rows = read_rows()
    assert rows
    assert all(row["label"] == "" for row in rows)
    assert all(row["comment"] == "" for row in rows)


def test_annotation_sample_has_no_duplicate_job_candidate_pairs() -> None:
    rows = read_rows()
    keys = [(row["job_id"], row["candidate_id"]) for row in rows]
    assert len(keys) == len(set(keys))


def test_annotation_sample_covers_top_middle_low_when_available() -> None:
    rows = read_rows()
    ranks_by_job: dict[str, list[int]] = {}
    for row in rows:
        ranks_by_job.setdefault(row["job_id"], []).append(int(row["rank"]))
    for ranks in ranks_by_job.values():
        assert min(ranks) == 1
        assert max(ranks) >= 15 or len(ranks) < 20
    assert all(count <= 20 for count in Counter(row["job_id"] for row in rows).values())
