from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "docs/reports/decision_cards/decision_cards_with_transferability.json"
DATASET = ROOT / "data/ranking/datasets/ranking_dataset_aligned_pseudo_labeled.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_demo_summary_is_generated_without_modifying_dataset(tmp_path: Path) -> None:
    dataset_before = _sha256(DATASET)
    output_json = tmp_path / "demo_summary_top10.json"
    output_md = tmp_path / "demo_summary_top10.md"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_demo_summary.py",
            "--cards",
            str(CARDS),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--top-k",
            "10",
        ],
        cwd=ROOT,
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert output_json.exists()
    assert output_md.exists()
    assert len(payload["candidates"]) <= payload["top_k"]
    assert "pseudo-labels" in payload["summary"]["methodological_note"]

    for candidate in payload["candidates"]:
        assert "baseline_score_v3" in candidate
        assert "rf_score" in candidate
        assert "xgboost_score" in candidate
        assert "transferability_score" in candidate
        assert "short_decision_summary" in candidate

    assert _sha256(DATASET) == dataset_before

