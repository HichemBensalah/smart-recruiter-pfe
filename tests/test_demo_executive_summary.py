from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "docs/reports/demo/demo_summary_top10.json"
DATASET = ROOT / "data/ranking/datasets/ranking_dataset_aligned_pseudo_labeled.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_demo_executive_summary_generation(tmp_path: Path) -> None:
    dataset_before = _sha256(DATASET)
    output_json = tmp_path / "demo_executive_summary.json"
    output_md = tmp_path / "demo_executive_summary.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_demo_executive_summary.py",
            "--input",
            str(INPUT),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=ROOT,
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert output_json.exists()
    assert output_md.exists()
    assert len(payload["top_recommended"]) <= 3
    assert len(payload["needs_review"]) <= 3
    assert "pseudo-labels" in payload["methodological_note"]
    for candidate in payload["top_recommended"] + payload["needs_review"]:
        assert "executive_reason" in candidate
    assert _sha256(DATASET) == dataset_before

