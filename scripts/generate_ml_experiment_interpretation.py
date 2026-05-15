from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the methodological interpretation report for the ML experiment.")
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_line(metrics: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in metrics.items())


def build_interpretation(training_report: dict[str, Any]) -> dict[str, Any]:
    models = training_report.get("models", {})
    xgboost_trained = "xgboost" in training_report.get("trained_models", [])
    model_comparison = {
        model_name: {
            "classification": model_result.get("mean_classification_metrics", {}),
            "ranking": model_result.get("mean_ranking_metrics", {}),
        }
        for model_name, model_result in models.items()
    }
    model_comparison["matching_v3_baseline"] = {
        "classification": None,
        "ranking": training_report.get("matching_v3_baseline", {}).get("mean_metrics", {}),
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Documenter une première expérimentation ML contrôlée sur le dataset aligné pseudo-labellisé.",
        "dataset": training_report.get("dataset"),
        "target": training_report.get("target"),
        "row_count": training_report.get("row_count"),
        "label_distribution": training_report.get("label_distribution"),
        "label_binary_distribution": training_report.get("label_binary_distribution"),
        "trained_models": training_report.get("trained_models", []),
        "split_strategy": training_report.get("split_strategy"),
        "model_comparison": model_comparison,
        "why_metrics_are_high": [
            "Les pseudo-labels sont dérivés de règles métier contrôlées.",
            "Plusieurs features utilisées pour créer les pseudo-labels sont aussi utilisées pour entraîner les modèles.",
            "Cette réutilisation crée une circularité partielle entre la cible et les variables explicatives.",
            "Le modèle apprend principalement à reproduire la logique de pseudo-labeling.",
            "Les résultats valident le pipeline ML end-to-end, mais ne prouvent pas une supériorité réelle sur des labels recruteur.",
        ],
        "methodological_limits": [
            "Les labels sont des pseudo-labels métier contrôlés, pas des labels recruteur.",
            "Le dataset reste petit avec 250 lignes et 43 exemples positifs binaires.",
            "La cible label_binary est construite à partir des labels 0/1/2/3 pseudo-labellisés.",
            "final_score_v3 est utilisé seulement comme score de baseline Matching V3, jamais comme label.",
            "Les métriques très élevées doivent être interprétées comme un contrôle technique du pipeline.",
        ],
        "validated": [
            "construction du dataset",
            "génération de pseudo-labels",
            "entraînement de modèles",
            "évaluation LeaveOneGroupOut par job_id",
            "comparaison avec Matching V3",
            "sauvegarde des modèles",
        ],
        "not_validated": [
            "un modèle final",
            "un modèle prêt pour production",
            "une supériorité réelle sur des décisions recruteur",
        ],
        "decision": {
            "matching_v3": "baseline officielle",
            "random_forest": "baseline ML expérimentale",
            "logistic_regression": "baseline ML simple",
            "xgboost": "baseline ML expérimentale testée" if xgboost_trained else "à tester si disponible",
            "human_validation": "nécessaire pour conclure",
        },
        "next_step": "Construire un petit jeu de labels humains/recruteur et réévaluer les modèles sans conclure uniquement à partir des pseudo-labels.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    comparison = report["model_comparison"]
    lines = [
        "# Interprétation méthodologique de l'expérimentation ML",
        "",
        "## Objectif de l'expérience",
        report["objective"],
        "",
        "## Dataset et cible",
        f"- Dataset: `{report['dataset']}`",
        f"- Target: `{report['target']}`",
        f"- Nombre de lignes: {report['row_count']}",
        f"- Distribution label 0/1/2/3: {report['label_distribution']}",
        f"- Distribution label_binary: {report['label_binary_distribution']}",
        "",
        "## Modèles et stratégie d'évaluation",
        f"- Modèles entraînés: {', '.join(report['trained_models'])}",
        f"- Split: {report['split_strategy']}",
        "",
        "## Comparaison Matching V3 vs modèles ML",
    ]
    for name, metrics in comparison.items():
        lines.append(f"- {name}: ranking({metric_line(metrics.get('ranking') or {})})")
        if metrics.get("classification"):
            lines.append(f"  classification({metric_line(metrics['classification'])})")

    lines.extend(
        [
            "",
            "## Pourquoi les métriques sont très élevées ?",
            "Les métriques sont presque parfaites parce que l'expérience contient une circularité partielle. Les pseudo-labels sont dérivés de règles métier, puis certaines features ayant servi à produire ces pseudo-labels, comme `must_have_coverage`, `experience_match_score` ou `reliability_score`, sont réutilisées comme entrées des modèles. Le modèle apprend donc principalement à reproduire la logique de pseudo-labeling, pas encore à imiter une décision recruteur indépendante.",
            "",
            "Cette observation ne rend pas l'expérience inutile. Elle valide que le pipeline ML fonctionne de bout en bout, depuis le dataset jusqu'aux modèles sauvegardés et aux métriques LeaveOneGroupOut. En revanche, elle ne démontre pas encore une supériorité réelle sur des labels recruteur.",
            "",
            "## Limites méthodologiques",
        ]
    )
    lines.extend(f"- {item}" for item in report["methodological_limits"])
    lines.extend(
        [
            "",
            "## Conclusion",
            "L'expérimentation valide techniquement la construction du dataset, la génération de pseudo-labels, l'entraînement de modèles, l'évaluation LeaveOneGroupOut, la comparaison avec Matching V3 et la sauvegarde des modèles.",
            "",
            "Elle ne valide pas encore un modèle final, un modèle prêt pour production, ni une supériorité réelle sur des décisions recruteur.",
            "",
            "## Décision",
            "- Matching V3 reste la baseline officielle.",
            "- Random Forest est une baseline ML expérimentale.",
            "- Logistic Regression est une baseline ML simple.",
            f"- XGBoost: {report['decision']['xgboost']}.",
            "- La validation humaine/recruteur reste nécessaire pour conclure.",
            "",
            "## Prochaine étape",
            report["next_step"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    training_report = read_json(args.training_report)
    report = build_interpretation(training_report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    write_json(report, args.output_json)


if __name__ == "__main__":
    main()
