from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import pytest
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_ranking_models import (
    build_models,
    build_report,
    ensure_label_binary,
    parse_args,
)
from src.core.ranking.evaluation import extract_feature_matrix, load_ranking_dataset


DATASET = ROOT / "data/ranking/datasets/ranking_dataset_aligned_pseudo_labeled.jsonl"


def _copy_dataset(tmp_path: Path) -> Path:
    target = tmp_path / "dataset.jsonl"
    target.write_text(DATASET.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_label_binary_is_added_and_target_is_used(tmp_path: Path) -> None:
    dataset = _copy_dataset(tmp_path)
    rows = load_ranking_dataset(dataset)
    for row in rows:
        row.pop("label_binary", None)
    assert ensure_label_binary(rows, dataset, "label_binary") is True
    updated = load_ranking_dataset(dataset)
    assert all(row["label_binary"] == (1 if int(row["label"]) >= 2 else 0) for row in updated)
    _x, y, _feature_names = extract_feature_matrix(updated, target="label_binary")
    assert set(y.tolist()) <= {0, 1}


def test_leave_one_group_out_uses_job_id_groups() -> None:
    rows = load_ranking_dataset(DATASET)
    for row in rows:
        row["label_binary"] = 1 if int(row["label"]) >= 2 else 0
    x, y, _feature_names = extract_feature_matrix(rows, target="label_binary")
    groups = [row["job_id"] for row in rows]
    splits = list(LeaveOneGroupOut().split(x, y, groups))
    assert len(splits) == 5
    for _train_index, test_index in splits:
        assert len({groups[index] for index in test_index}) == 1


def test_xgboost_absence_does_not_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "xgboost":
            raise ImportError("simulated missing xgboost")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    models, warnings = build_models()
    assert {"logistic_regression", "random_forest"} <= set(models)
    assert "xgboost" not in models
    assert warnings


def test_training_report_is_generated_and_final_score_is_baseline_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _copy_dataset(tmp_path)
    output_dir = tmp_path / "models"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_ranking_models.py",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--target",
            "label_binary",
        ],
    )
    args = parse_args()
    report = build_report(args)
    report_path = output_dir / "training_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    assert report_path.exists()
    assert report["target"] == "label_binary"
    assert "final_score_v3" in report["features"]
    assert report["matching_v3_baseline"]["mean_metrics"]
    assert report["dataset"] == str(dataset)
    assert report["split_strategy"].startswith("LeaveOneGroupOut par job_id")
    assert report["target"] != "final_score_v3"
