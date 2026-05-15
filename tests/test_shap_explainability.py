from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/explain_xgboost_shap.py"
MODEL = ROOT / "data/ranking/models/xgboost.joblib"
DATASET = ROOT / "data/ranking/datasets/ranking_dataset_aligned_pseudo_labeled.jsonl"
FEATURE_NAMES = ROOT / "data/ranking/models/feature_names.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shap_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except Exception:
        return False


def run_script(output_dir: Path, model: Path = MODEL) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            str(model),
            "--dataset",
            str(DATASET),
            "--feature-names",
            str(FEATURE_NAMES),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_script_exists() -> None:
    assert SCRIPT.exists()


def test_shap_reports_are_generated_when_available(tmp_path: Path) -> None:
    if not shap_available():
        return
    output_dir = tmp_path / "shap"
    run_script(output_dir)
    assert (output_dir / "shap_global_summary.json").exists()
    assert (output_dir / "shap_global_summary.md").exists()
    assert (output_dir / "shap_local_examples.json").exists()
    assert (output_dir / "shap_methodology_note.md").exists()

    payload = json.loads((output_dir / "shap_global_summary.json").read_text(encoding="utf-8"))
    assert payload["status"] == "available"
    assert payload["top_features"]


def test_script_exits_cleanly_with_warning_when_model_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "missing"
    result = run_script(output_dir, model=tmp_path / "missing_xgboost.joblib")
    assert "WARNING" in result.stdout
    payload = json.loads((output_dir / "shap_global_summary.json").read_text(encoding="utf-8"))
    assert payload["status"] == "unavailable"


def test_methodology_note_mentions_pseudo_labels(tmp_path: Path) -> None:
    output_dir = tmp_path / "shap"
    run_script(output_dir)
    note = (output_dir / "shap_methodology_note.md").read_text(encoding="utf-8").lower()
    assert "pseudo-labels" in note
    assert "labels recruteur" in note


def test_script_does_not_modify_dataset_or_model(tmp_path: Path) -> None:
    dataset_before = sha256(DATASET)
    model_before = sha256(MODEL)
    run_script(tmp_path / "shap")
    assert sha256(DATASET) == dataset_before
    assert sha256(MODEL) == model_before
