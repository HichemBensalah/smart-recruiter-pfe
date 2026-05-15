from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.ranking.evaluation import (
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    extract_feature_matrix,
    group_rows_by_job_id,
    load_ranking_dataset,
)


DATASET = ROOT / "data/ranking/datasets/ranking_dataset_aligned_pseudo_labeled.jsonl"


def test_dataset_is_loaded_correctly() -> None:
    rows = load_ranking_dataset(DATASET)
    assert len(rows) == 250
    assert {"label", "features", "job_id", "query_group"} <= rows[0].keys()


def test_extracts_12_features_and_uses_label_binary_target() -> None:
    rows = load_ranking_dataset(DATASET)
    for row in rows:
        row["label_binary"] = 1 if int(row["label"]) >= 2 else 0
    x, y, feature_names = extract_feature_matrix(rows, target="label_binary")
    assert x.shape == (250, 12)
    assert len(feature_names) == 12
    assert set(y.tolist()) <= {0, 1}


def test_identifiers_are_not_used_as_features() -> None:
    rows = load_ranking_dataset(DATASET)
    for row in rows:
        row["label_binary"] = 1 if int(row["label"]) >= 2 else 0
    _x, _y, feature_names = extract_feature_matrix(rows, target="label_binary")
    forbidden = {"job_id", "candidate_id", "profile_id", "rank", "label", "label_binary", "split"}
    assert not forbidden & set(feature_names)


def test_group_rows_by_job_id_and_ranking_metrics_are_valid() -> None:
    rows = load_ranking_dataset(DATASET)[:20]
    for row in rows:
        row["label_binary"] = 1 if int(row["label"]) >= 2 else 0
    grouped = group_rows_by_job_id(rows)
    assert grouped
    scores = [float(row["features"]["final_score_v3"]) for row in rows]
    assert 0.0 <= compute_precision_at_k(rows, scores, 5) <= 1.0
    assert 0.0 <= compute_precision_at_k(rows, scores, 10) <= 1.0
    assert 0.0 <= compute_ndcg_at_k(rows, scores, 10) <= 1.0
    assert 0.0 <= compute_mrr(rows, scores) <= 1.0
