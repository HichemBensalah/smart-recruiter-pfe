from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np


METHODOLOGY_SENTENCE = (
    "Ces explications SHAP concernent un modèle entraîné sur pseudo-labels métier contrôlés, "
    "pas sur labels recruteur."
)

METHODOLOGY_NOTE = """# Note méthodologique SHAP

SHAP explique le comportement du modèle XGBoost expérimental sauvegardé.

Ce modèle est entraîné sur des pseudo-labels métier contrôlés, pas sur des labels recruteur. Ces pseudo-labels sont dérivés de règles métier construites à partir de features comme `must_have_coverage`, `experience_match_score`, `reliability_score` ou d'autres signaux de matching.

Il existe donc une circularité partielle : certaines variables qui ont contribué à produire la cible pseudo-labellisée sont aussi utilisées comme entrées du modèle. Les explications SHAP montrent principalement quelles features permettent au modèle de reproduire la règle métier de pseudo-labeling.

Elles ne doivent pas être interprétées comme une vérité recruteur, ni comme une explication de décisions humaines réelles.

Une validation humaine/recruteur reste nécessaire pour conclure sur la qualité métier finale du modèle.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP explainability reports for the experimental XGBoost model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_feature_names(path: Path) -> list[str]:
    payload = read_json(path)
    if isinstance(payload, dict):
        return list(payload["feature_names"])
    return list(payload)


def feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [[float((row.get("features") or {}).get(feature, 0.0)) for feature in feature_names] for row in rows],
        dtype=float,
    )


def import_shap() -> Any | None:
    try:
        import shap

        return shap
    except Exception:
        return None


def unavailable_outputs(args: argparse.Namespace, warning: str) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_report = {
        "generated_at_utc": utc_now(),
        "status": "unavailable",
        "warning": warning,
        "top_features": [],
        "methodology_warning": METHODOLOGY_SENTENCE,
    }
    write_json(global_report, args.output_dir / "shap_global_summary.json")
    (args.output_dir / "shap_global_summary.md").write_text(
        "# SHAP Global Summary\n\n"
        f"Status: unavailable\n\nWarning: {warning}\n\n{METHODOLOGY_SENTENCE}\n",
        encoding="utf-8",
    )
    write_json(
        {
            "generated_at_utc": utc_now(),
            "status": "unavailable",
            "warning": warning,
            "examples": [],
            "methodology_warning": METHODOLOGY_SENTENCE,
        },
        args.output_dir / "shap_local_examples.json",
    )
    (args.output_dir / "shap_methodology_note.md").write_text(METHODOLOGY_NOTE, encoding="utf-8")


def normalize_shap_values(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, -1]
    return arr


def predicted_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        if probabilities.shape[1] == 1:
            return np.asarray([float(probabilities[0][0])] * len(x), dtype=float)
        class_list = list(model.classes_)
        positive_index = class_list.index(1) if 1 in class_list else 1
        return probabilities[:, positive_index]
    return np.asarray(model.predict(x), dtype=float)


def load_xgboost_importance() -> list[dict[str, Any]]:
    path = Path("docs/reports/ml/xgboost_feature_importance.json")
    if not path.exists():
        return []
    return read_json(path).get("top_features", [])


def select_example_indices(rows: list[dict[str, Any]]) -> list[int]:
    selected: list[int] = []
    seen_jobs: set[str] = set()
    for label_binary in (1, 0):
        count = 0
        for index, row in enumerate(rows):
            if int(row.get("label_binary", 1 if int(row["label"]) >= 2 else 0)) != label_binary:
                continue
            job_id = str(row["job_id"])
            if count < 2 and (job_id not in seen_jobs or len(seen_jobs) >= 2):
                selected.append(index)
                seen_jobs.add(job_id)
                count += 1
            if count == 2:
                break
    return selected


def contribution_rows(feature_names: list[str], values: np.ndarray, feature_values: np.ndarray) -> list[dict[str, Any]]:
    rows = [
        {"feature": name, "shap_value": float(value), "feature_value": float(feature_value)}
        for name, value, feature_value in zip(feature_names, values, feature_values)
    ]
    return rows


def build_local_examples(
    rows: list[dict[str, Any]],
    x: np.ndarray,
    shap_values: np.ndarray,
    probabilities: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for index in select_example_indices(rows):
        row = rows[index]
        contributions = contribution_rows(feature_names, shap_values[index], x[index])
        positive = sorted([item for item in contributions if item["shap_value"] > 0], key=lambda item: item["shap_value"], reverse=True)[:5]
        negative = sorted([item for item in contributions if item["shap_value"] < 0], key=lambda item: item["shap_value"])[:5]
        label_binary = int(row.get("label_binary", 1 if int(row["label"]) >= 2 else 0))
        top_pos = positive[0]["feature"] if positive else "aucune contribution positive nette"
        top_neg = negative[0]["feature"] if negative else "aucune contribution négative nette"
        examples.append(
            {
                "job_id": row["job_id"],
                "candidate_id": row["candidate_id"],
                "label": int(row["label"]),
                "label_binary": label_binary,
                "predicted_probability": float(probabilities[index]),
                "top_positive_shap_contributions": positive,
                "top_negative_shap_contributions": negative,
                "interpretation": (
                    f"Pour cet exemple pseudo-labellisé {label_binary}, les signaux les plus favorables au modèle sont dominés par {top_pos}; "
                    f"les signaux qui réduisent le score sont dominés par {top_neg}. "
                    "Cette lecture explique le modèle expérimental, pas une décision recruteur."
                ),
            }
        )
    return examples


def render_global_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SHAP Global Summary - XGBoost expérimental",
        "",
        METHODOLOGY_SENTENCE,
        "",
        "## Top features SHAP",
        "",
        "| Rank | Feature | Mean absolute SHAP | XGBoost importance |",
        "|---:|---|---:|---:|",
    ]
    xgb_by_feature = {
        item["feature"]: item.get("importance") for item in report.get("xgboost_feature_importance_top_features", [])
    }
    for index, row in enumerate(report["top_features"], start=1):
        xgb_importance = xgb_by_feature.get(row["feature"])
        xgb_text = "" if xgb_importance is None else f"{float(xgb_importance):.6f}"
        lines.append(f"| {index} | `{row['feature']}` | {row['mean_abs_shap']:.6f} | {xgb_text} |")
    lines.extend(
        [
            "",
            "## Comparaison rapide avec xgboost_feature_importance",
            report["comparison_with_xgboost_feature_importance"],
            "",
            "## Avertissement méthodologique",
            report["methodology_warning"],
            "",
        ]
    )
    return "\n".join(lines)


def generate_reports(args: argparse.Namespace) -> int:
    shap = import_shap()
    if shap is None:
        unavailable_outputs(args, "shap n'est pas installé ou n'est pas importable.")
        print("WARNING: shap n'est pas installé ou n'est pas importable.")
        return 0
    if not args.model.exists():
        unavailable_outputs(args, f"Modèle XGBoost introuvable: {args.model}")
        print(f"WARNING: Modèle XGBoost introuvable: {args.model}")
        return 0

    rows = read_jsonl(args.dataset)
    feature_names = load_feature_names(args.feature_names)
    x = feature_matrix(rows, feature_names)
    model = joblib.load(args.model)
    explainer = shap.TreeExplainer(model)
    shap_values = normalize_shap_values(explainer.shap_values(x))
    probabilities = predicted_probabilities(model, x)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_features = sorted(
        [{"feature": name, "mean_abs_shap": float(value)} for name, value in zip(feature_names, mean_abs)],
        key=lambda item: item["mean_abs_shap"],
        reverse=True,
    )
    xgb_importance = load_xgboost_importance()
    xgb_top_names = [item["feature"] for item in xgb_importance[:5]]
    shap_top_names = [item["feature"] for item in top_features[:5]]
    overlap = [name for name in shap_top_names if name in xgb_top_names]

    global_report = {
        "generated_at_utc": utc_now(),
        "status": "available",
        "model": str(args.model),
        "dataset": str(args.dataset),
        "feature_names": str(args.feature_names),
        "row_count": len(rows),
        "top_features": top_features,
        "xgboost_feature_importance_top_features": xgb_importance,
        "comparison_with_xgboost_feature_importance": (
            "Top SHAP and XGBoost built-in importance overlap on: "
            + (", ".join(overlap) if overlap else "no top-5 feature")
            + ". Both explain a model trained to reproduce controlled business pseudo-labels."
        ),
        "methodology_warning": METHODOLOGY_SENTENCE,
    }
    local_report = {
        "generated_at_utc": utc_now(),
        "status": "available",
        "examples": build_local_examples(rows, x, shap_values, probabilities, feature_names),
        "methodology_warning": METHODOLOGY_SENTENCE,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(global_report, args.output_dir / "shap_global_summary.json")
    (args.output_dir / "shap_global_summary.md").write_text(render_global_markdown(global_report), encoding="utf-8")
    write_json(local_report, args.output_dir / "shap_local_examples.json")
    (args.output_dir / "shap_methodology_note.md").write_text(METHODOLOGY_NOTE, encoding="utf-8")
    return 0


def main() -> None:
    raise SystemExit(generate_reports(parse_args()))


if __name__ == "__main__":
    main()
