from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.graph.decision_cards_transferability_enricher import LOOKUP_STATUSES  # noqa: E402

CARDS = ROOT / "docs/reports/decision_cards/decision_cards_ml_comparison.json"
PROFILES_DIR = ROOT / "data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles"
JOB = ROOT / "data/job_profiles/backend_python_django_postgresql.json"
GRAPH = ROOT / "data/graph/skills_roles_graph.yaml"
OFFICIAL_DECISION_CARDS = ROOT / "docs/reports/matching/v3/decision_cards_v3_normalized.json"
DATASET = ROOT / "data/ranking/datasets/ranking_dataset_aligned_pseudo_labeled.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_decision_cards_with_transferability_generated_without_modifying_inputs(tmp_path: Path) -> None:
    official_before = _sha256(OFFICIAL_DECISION_CARDS)
    dataset_before = _sha256(DATASET)
    output_json = tmp_path / "decision_cards_with_transferability.json"
    output_md = tmp_path / "decision_cards_with_transferability.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_decision_cards_with_transferability.py",
            "--cards",
            str(CARDS),
            "--profiles-dir",
            str(PROFILES_DIR),
            "--job",
            str(JOB),
            "--graph",
            str(GRAPH),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=ROOT,
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")
    assert output_json.exists()
    assert output_md.exists()
    assert payload["card_type"] == "decision_cards_with_transferability"
    assert "Potential Graph" in payload["methodological_note"]
    assert "Potential Graph" in markdown
    assert payload["candidates"]
    assert "lookup_success_rate" in payload
    assert "profiles_found" in payload
    assert "profiles_not_found" in payload
    assert payload["lookup_success_rate"] >= 0.8

    candidate = payload["candidates"][0]
    assert candidate["profile_lookup_status"] in LOOKUP_STATUSES
    assert "transferability" in candidate
    transfer = candidate["transferability"]
    assert isinstance(transfer["fit_direct"], bool)
    assert 0.0 <= float(transfer["direct_fit_score"]) <= 1.0
    assert 0.0 <= float(transfer["transferability_score"]) <= 1.0
    assert isinstance(transfer["gaps_compensables"], list)
    assert isinstance(transfer["gaps_bloquants"], list)

    assert all(card["profile_lookup_status"] in LOOKUP_STATUSES for card in payload["candidates"])

    assert _sha256(OFFICIAL_DECISION_CARDS) == official_before
    assert _sha256(DATASET) == dataset_before
