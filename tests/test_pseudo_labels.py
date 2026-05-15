from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pseudo_labels import LABELING_STRATEGY, pseudo_label


PSEUDO_PATH = ROOT / "data/ranking/datasets/ranking_dataset_pseudo_labeled.jsonl"
SUMMARY_PATH = ROOT / "data/ranking/datasets/pseudo_label_summary.json"


def read_rows() -> list[dict]:
    return [json.loads(line) for line in PSEUDO_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_pseudo_labels_contract() -> None:
    rows = read_rows()
    assert rows
    for row in rows:
        assert row["label"] in {0, 1, 2, 3}
        assert row["label_type"] == "pseudo"
        assert row["labeling_strategy"] == LABELING_STRATEGY


def test_pseudo_label_summary_distribution_non_empty() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["labeling_strategy"] == LABELING_STRATEGY
    assert summary["label_distribution"]
    assert "pseudo-labels métier contrôlés" in summary["methodology_warning"]
    assert "positive_count" in summary
    assert "positive_rate" in summary


def test_pseudo_label_rule_does_not_depend_on_final_score_or_rank() -> None:
    features = {
        "vector_similarity": 0.1,
        "final_score_v3": 0.99,
        "must_have_coverage": 0.0,
        "required_skills_overlap": 0.0,
        "nice_to_have_overlap": 0.0,
        "experience_match_score": 0.0,
        "seniority_alignment": 0.0,
        "profile_quality_score": 1.0,
        "reliability_score": 1.0,
        "hallucination_risk_encoded": 0.0,
        "missing_required_count": 5.0,
        "matched_required_count": 0.0,
    }
    assert pseudo_label(features) == 0


def test_seniority_alignment_is_not_blocking_for_label_3() -> None:
    features = {
        "vector_similarity": 0.1,
        "final_score_v3": 0.1,
        "must_have_coverage": 0.8,
        "required_skills_overlap": 0.8,
        "nice_to_have_overlap": 0.0,
        "experience_match_score": 0.6,
        "seniority_alignment": 0.0,
        "profile_quality_score": 0.6,
        "reliability_score": 0.6,
        "hallucination_risk_encoded": 0.5,
        "missing_required_count": 1.0,
        "matched_required_count": 4.0,
    }
    assert pseudo_label(features) == 3


def test_label_1_requires_must_have_and_secondary_signal() -> None:
    no_must_have = {
        "vector_similarity": 0.9,
        "final_score_v3": 0.1,
        "must_have_coverage": 0.2,
        "required_skills_overlap": 0.9,
        "nice_to_have_overlap": 0.0,
        "experience_match_score": 0.0,
        "seniority_alignment": 0.0,
        "profile_quality_score": 1.0,
        "reliability_score": 1.0,
        "hallucination_risk_encoded": 0.0,
        "missing_required_count": 5.0,
        "matched_required_count": 0.0,
    }
    assert pseudo_label(no_must_have) == 0
