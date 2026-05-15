from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALIGNED_JOB_IDS = {
    "data_analyst_python_sql_powerbi",
    "machine_learning_python_nlp",
    "backend_python_django_postgresql",
    "data_engineer_python_sql_etl_aligned",
    "backend_python_fastapi_mongodb_aligned",
}
UNLABELED_OLD = ROOT / "data/ranking/datasets/ranking_dataset_unlabeled.jsonl"
ALIGNED_UNLABELED = ROOT / "data/ranking/datasets/ranking_dataset_aligned_unlabeled.jsonl"
ALIGNED_PSEUDO = ROOT / "data/ranking/datasets/ranking_dataset_aligned_pseudo_labeled.jsonl"
ALIGNED_SUMMARY = ROOT / "data/ranking/datasets/pseudo_label_aligned_summary.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_ranking_dataset_supports_include_job_ids_without_modifying_old_dataset(tmp_path: Path) -> None:
    before_hash = sha256_file(UNLABELED_OLD)
    output = tmp_path / "aligned.jsonl"
    summary = tmp_path / "aligned_summary.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_ranking_dataset.py"),
            "--input-dir",
            str(ROOT / "data/ranking/features"),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--dataset-id",
            "aligned_test",
            "--include-job-ids",
            ",".join(sorted(ALIGNED_JOB_IDS)),
        ],
        cwd=ROOT,
        check=True,
    )
    assert sha256_file(UNLABELED_OLD) == before_hash
    rows = read_jsonl(output)
    assert rows
    assert {row["job_id"] for row in rows} == ALIGNED_JOB_IDS


def test_aligned_dataset_artifacts_exist_and_are_filtered() -> None:
    assert ALIGNED_UNLABELED.exists()
    rows = read_jsonl(ALIGNED_UNLABELED)
    assert rows
    assert {row["job_id"] for row in rows} == ALIGNED_JOB_IDS


def test_aligned_pseudo_labels_exist_without_training_fields() -> None:
    assert ALIGNED_PSEUDO.exists()
    rows = read_jsonl(ALIGNED_PSEUDO)
    assert rows
    for row in rows:
        assert row["label"] in {0, 1, 2, 3}
        assert row["label_type"] == "pseudo"
        assert row["labeling_strategy"] == "multi_criteria_v2"
        assert "smote" not in row
        assert "model_prediction" not in row
    summary = json.loads(ALIGNED_SUMMARY.read_text(encoding="utf-8"))
    assert summary["positive_count"] >= 0
    assert summary["labeling_strategy"] == "multi_criteria_v2"
