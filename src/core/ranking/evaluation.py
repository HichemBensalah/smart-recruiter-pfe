from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


EXCLUDED_FEATURE_KEYS = {
    "job_id",
    "query_group",
    "candidate_id",
    "profile_id",
    "rank",
    "label",
    "label_binary",
    "label_type",
    "labeling_strategy",
    "split",
}


def load_ranking_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    return [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    feature_names = list((rows[0].get("features") or {}).keys())
    for row in rows:
        names = list((row.get("features") or {}).keys())
        if names != feature_names:
            raise ValueError("Inconsistent feature names across ranking rows.")
    return feature_names


def extract_feature_matrix(rows: list[dict[str, Any]], target: str = "label_binary") -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_names = get_feature_names(rows)
    forbidden = sorted(set(feature_names) & EXCLUDED_FEATURE_KEYS)
    if forbidden:
        raise ValueError(f"Forbidden keys present in features: {forbidden}")
    if target in feature_names:
        raise ValueError(f"Target {target!r} must not be used as a feature.")

    matrix = np.asarray(
        [[float((row.get("features") or {}).get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=float,
    )
    labels = np.asarray([int(row[target]) for row in rows], dtype=int)
    return matrix, labels, feature_names


def compute_binary_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float],
) -> dict[str, float | None]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    y_score_arr = np.asarray(list(y_score), dtype=float)

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "roc_auc": None,
    }
    if len(set(y_true_arr.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
    return metrics


def group_rows_by_job_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["job_id"])].append(row)
    return dict(grouped)


def _row_relevance(row: dict[str, Any]) -> int:
    if "label_binary" in row:
        return int(row["label_binary"])
    return 1 if int(row["label"]) >= 2 else 0


def _ordered_group(rows: list[dict[str, Any]], scores: Iterable[float]) -> list[tuple[dict[str, Any], float]]:
    paired = list(zip(rows, [float(score) for score in scores]))
    return sorted(paired, key=lambda item: item[1], reverse=True)


def compute_precision_at_k(rows: list[dict[str, Any]], scores: Iterable[float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    ordered = _ordered_group(rows, scores)
    if not ordered:
        return 0.0
    top_k = ordered[:k]
    return float(sum(_row_relevance(row) for row, _score in top_k) / min(k, len(ordered)))


def compute_ndcg_at_k(rows: list[dict[str, Any]], scores: Iterable[float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    ordered = _ordered_group(rows, scores)[:k]
    gains = [_row_relevance(row) for row, _score in ordered]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_gains = sorted((_row_relevance(row) for row in rows), reverse=True)[:k]
    ideal_dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    if ideal_dcg == 0:
        return 0.0
    return float(dcg / ideal_dcg)


def compute_mrr(rows: list[dict[str, Any]], scores: Iterable[float]) -> float:
    ordered = _ordered_group(rows, scores)
    for index, (row, _score) in enumerate(ordered, start=1):
        if _row_relevance(row) > 0:
            return float(1 / index)
    return 0.0
