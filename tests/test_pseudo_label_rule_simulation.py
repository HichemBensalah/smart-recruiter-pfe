from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data/ranking/datasets/ranking_dataset_unlabeled.jsonl"
REPORT_PATH = ROOT / "docs/reports/ml/pseudo_label_rule_simulation.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pseudo_label_rule_simulation_report_exists() -> None:
    assert REPORT_PATH.exists()
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["total_rows"] > 0
    assert "distribution_v1" in report
    assert "distribution_v2" in report
    assert "positive_count_v1" in report
    assert "positive_count_v2" in report
    assert report["recommendation"] in {"v2_acceptable", "v2_still_too_strict", "v2_too_loose"}


def test_simulation_does_not_modify_official_dataset(tmp_path: Path) -> None:
    before_hash = sha256_file(DATASET_PATH)
    report_path = tmp_path / "pseudo_label_rule_simulation.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/simulate_pseudo_label_rules.py"),
            "--dataset",
            str(DATASET_PATH),
            "--report",
            str(report_path),
        ],
        check=True,
        cwd=ROOT,
    )
    after_hash = sha256_file(DATASET_PATH)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert before_hash == after_hash
    assert report["dataset_unchanged"] is True
