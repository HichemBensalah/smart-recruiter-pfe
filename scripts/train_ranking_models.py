from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.ranking.evaluation import (  # noqa: E402
    compute_binary_classification_metrics,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    extract_feature_matrix,
    get_feature_names,
    group_rows_by_job_id,
    load_ranking_dataset,
)


METHODOLOGY_WARNINGS = [
    "Les labels sont des pseudo-labels metier controles.",
    "Ce ne sont pas des labels recruteur.",
    "Le dataset reste petit.",
    "L'evaluation mesure une experimentation ML, pas un modele final valide.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train controlled ML ranking baselines on the aligned dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("data/ranking/datasets/ranking_dataset_aligned_pseudo_labeled.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ranking/models"))
    parser.add_argument("--target", default="label_binary")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def ensure_label_binary(rows: list[dict[str, Any]], dataset_path: Path, target: str) -> bool:
    changed = False
    for row in rows:
        expected = 1 if int(row["label"]) >= 2 else 0
        if row.get(target) != expected:
            row[target] = expected
            changed = True
        row.setdefault("label_type", "pseudo")
        row.setdefault("labeling_strategy", "multi_criteria_v2")
        row.setdefault("query_group", row["job_id"])
    if changed:
        write_jsonl(rows, dataset_path)
    return changed


def build_models(random_state: int = 42) -> tuple[dict[str, Any], list[str]]:
    models: dict[str, Any] = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    warnings_list: list[str] = []
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
    except Exception as exc:  # pragma: no cover - exercised in tests via monkeypatch
        warnings_list.append(f"xgboost indisponible: {exc}. Entrainement poursuivi sans XGBoost.")
    return models, warnings_list


def positive_scores(model: Any, x_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        if probabilities.shape[1] == 1:
            return np.repeat(float(model.classes_[0]), len(x_test))
        positive_index = list(model.classes_).index(1)
        return probabilities[:, positive_index]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x_test)
        return 1 / (1 + np.exp(-raw))
    return model.predict(x_test)


def ranking_metrics_for_rows(rows: list[dict[str, Any]], scores: list[float] | np.ndarray) -> dict[str, float]:
    grouped_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped_indices.setdefault(str(row["job_id"]), []).append(index)

    per_job: list[dict[str, float]] = []
    score_arr = np.asarray(scores, dtype=float)
    for indices in grouped_indices.values():
        group_rows = [rows[index] for index in indices]
        group_scores = score_arr[indices]
        per_job.append(
            {
                "precision@5": compute_precision_at_k(group_rows, group_scores, 5),
                "precision@10": compute_precision_at_k(group_rows, group_scores, 10),
                "ndcg@10": compute_ndcg_at_k(group_rows, group_scores, 10),
                "mrr": compute_mrr(group_rows, group_scores),
            }
        )
    return average_metric_dicts(per_job)


def average_metric_dicts(items: list[dict[str, float | None]]) -> dict[str, float | None]:
    if not items:
        return {}
    averaged: dict[str, float | None] = {}
    for key in items[0]:
        values = [item[key] for item in items if item.get(key) is not None]
        averaged[key] = float(np.mean(values)) if values else None
    return averaged


def evaluate_matching_v3_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_job: dict[str, dict[str, float]] = {}
    for job_id, group_rows in group_rows_by_job_id(rows).items():
        scores = [float(row["features"]["final_score_v3"]) for row in group_rows]
        per_job[job_id] = {
            "precision@5": compute_precision_at_k(group_rows, scores, 5),
            "precision@10": compute_precision_at_k(group_rows, scores, 10),
            "ndcg@10": compute_ndcg_at_k(group_rows, scores, 10),
            "mrr": compute_mrr(group_rows, scores),
        }
    return {"per_job": per_job, "mean_metrics": average_metric_dicts(list(per_job.values()))}


def train_and_evaluate(rows: list[dict[str, Any]], target: str) -> tuple[dict[str, Any], list[str], list[str]]:
    x, y, feature_names = extract_feature_matrix(rows, target=target)
    groups = np.asarray([row["job_id"] for row in rows])
    models, warning_list = build_models()
    logo = LeaveOneGroupOut()
    results: dict[str, Any] = {}

    for model_name, model in models.items():
        fold_results: list[dict[str, Any]] = []
        for fold_index, (train_index, test_index) in enumerate(logo.split(x, y, groups), start=1):
            train_groups = sorted(set(groups[train_index].tolist()))
            test_group = str(groups[test_index][0])
            model.fit(x[train_index], y[train_index])
            predictions = model.predict(x[test_index])
            scores = positive_scores(model, x[test_index])
            classification = compute_binary_classification_metrics(y[test_index], predictions, scores)
            ranking = ranking_metrics_for_rows([rows[index] for index in test_index], scores)
            fold_results.append(
                {
                    "fold": fold_index,
                    "train_job_ids": train_groups,
                    "test_job_id": test_group,
                    "classification_metrics": classification,
                    "ranking_metrics": ranking,
                    "test_label_binary_distribution": dict(Counter(y[test_index].tolist())),
                }
            )

        results[model_name] = {
            "folds": fold_results,
            "mean_classification_metrics": average_metric_dicts(
                [fold["classification_metrics"] for fold in fold_results]
            ),
            "mean_ranking_metrics": average_metric_dicts([fold["ranking_metrics"] for fold in fold_results]),
        }

    return results, feature_names, warning_list


def fit_and_save_models(rows: list[dict[str, Any]], target: str, output_dir: Path) -> list[str]:
    x, y, _feature_names = extract_feature_matrix(rows, target=target)
    models, _warnings = build_models()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for model_name, model in models.items():
        model.fit(x, y)
        path = output_dir / f"{model_name}.joblib"
        joblib.dump(model, path)
        saved.append(str(path))
    return saved


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# ML Experiment Report - Aligned Ranking Dataset",
        "",
        "## Scope",
        "Baseline experimentale controlee sur le dataset aligne. Les labels utilises sont des pseudo-labels metier controles, pas des labels recruteur.",
        "",
        "## Dataset",
        f"- Dataset: `{report['dataset']}`",
        f"- Target: `{report['target']}`",
        f"- Rows: {report['row_count']}",
        f"- Label distribution: {report['label_distribution']}",
        f"- Label binary distribution: {report['label_binary_distribution']}",
        "",
        "## Split",
        f"- {report['split_strategy']}",
        "",
        "## Features",
        *[f"- `{name}`" for name in report["features"]],
        "",
        "## Mean Metrics",
    ]
    for model_name, result in report["models"].items():
        lines.extend(
            [
                f"### {model_name}",
                f"- Classification: {result['mean_classification_metrics']}",
                f"- Ranking: {result['mean_ranking_metrics']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Matching V3 Baseline",
            f"- Score used for comparison only: `final_score_v3`",
            f"- Metrics: {report['matching_v3_baseline']['mean_metrics']}",
            "",
            "## Methodological Warnings",
            *[f"- {warning}" for warning in report["methodology_warnings"]],
            "",
            "## Decision",
            report["decision"],
            "",
        ]
    )
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_ranking_dataset(args.dataset)
    label_binary_added = ensure_label_binary(rows, args.dataset, args.target)
    model_results, feature_names, model_warnings = train_and_evaluate(rows, args.target)
    saved_models = fit_and_save_models(rows, args.target, args.output_dir)
    matching_v3_baseline = evaluate_matching_v3_baseline(rows)
    best_model = max(
        model_results,
        key=lambda name: model_results[name]["mean_ranking_metrics"].get("ndcg@10") or 0.0,
    )

    report = {
        "generated_at_utc": utc_now(),
        "dataset": str(args.dataset),
        "target": args.target,
        "row_count": len(rows),
        "label_distribution": {str(label): Counter(int(row["label"]) for row in rows)[label] for label in range(4)},
        "label_binary_distribution": {
            str(label): Counter(int(row[args.target]) for row in rows)[label] for label in range(2)
        },
        "features": feature_names,
        "trained_models": sorted(model_results),
        "saved_models": saved_models,
        "split_strategy": "LeaveOneGroupOut par job_id: entrainement sur 4 offres, test sur 1 offre.",
        "fold_count": len(set(row["job_id"] for row in rows)),
        "models": model_results,
        "matching_v3_baseline": matching_v3_baseline,
        "best_model_by_ndcg@10": best_model,
        "label_binary_added_or_updated": label_binary_added,
        "warnings": model_warnings,
        "methodology_warnings": METHODOLOGY_WARNINGS,
        "decision": (
            "Modele experimental acceptable comme baseline de recherche, non acceptable comme modele final production "
            "sans labels recruteur et validation supplementaire."
        ),
    }
    return report


def main() -> None:
    args = parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        report = build_report(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json({"feature_names": report["features"]}, args.output_dir / "feature_names.json")
    write_json(report, args.output_dir / "training_report.json")
    report_dir = ROOT / "docs/reports/ml"
    write_json(report, report_dir / "ml_experiment_report.json")
    (report_dir / "ml_experiment_report.md").write_text(render_markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
