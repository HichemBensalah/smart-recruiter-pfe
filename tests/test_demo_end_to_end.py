from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_demo_end_to_end.py"
DATASET = ROOT / "data/ranking/datasets/ranking_dataset_aligned_pseudo_labeled.jsonl"
RF_MODEL = ROOT / "data/ranking/models/random_forest.joblib"
XGB_MODEL = ROOT / "data/ranking/models/xgboost.joblib"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_demo_end_to_end_generates_manifest_and_summaries_without_mutating_inputs(tmp_path: Path) -> None:
    dataset_before = _sha256(DATASET)
    rf_before = _sha256(RF_MODEL)
    xgb_before = _sha256(XGB_MODEL)
    output_dir = tmp_path / "demo"

    assert SCRIPT.exists()
    subprocess.run(
        [
            sys.executable,
            "scripts/run_demo_end_to_end.py",
            "--features",
            "data/ranking/features/backend_python_django_postgresql.jsonl",
            "--job",
            "data/job_profiles/backend_python_django_postgresql.json",
            "--profiles-dir",
            "data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles",
            "--graph",
            "data/graph/skills_roles_graph.yaml",
            "--rf-model",
            "data/ranking/models/random_forest.joblib",
            "--xgb-ranking",
            "docs/reports/ml/xgboost_primary_ranking.json",
            "--feature-names",
            "data/ranking/models/feature_names.json",
            "--cards-ml",
            "docs/reports/decision_cards/decision_cards_ml_comparison.json",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )

    manifest_path = output_dir / "demo_run_manifest.json"
    run_summary_path = output_dir / "demo_run_summary.md"
    executive_path = output_dir / "demo_executive_summary.json"
    top10_path = output_dir / "demo_summary_top10.json"

    assert manifest_path.exists()
    assert run_summary_path.exists()
    assert executive_path.exists()
    assert top10_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["checked_inputs"]
    assert manifest["generated_outputs"]
    assert _sha256(DATASET) == dataset_before
    assert _sha256(RF_MODEL) == rf_before
    assert _sha256(XGB_MODEL) == xgb_before

