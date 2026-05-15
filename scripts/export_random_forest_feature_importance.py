from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


WARNING = (
    "Ces importances sont calculées sur un modèle entraîné avec des pseudo-labels métier contrôlés. "
    "Elles indiquent quelles features aident à reproduire la règle de pseudo-labeling, "
    "pas nécessairement les critères réels d’un recruteur."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Random Forest feature importances.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_feature_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload["feature_names"])
    return list(payload)


def export_importance(model_path: Path, feature_names_path: Path) -> dict[str, Any]:
    model = joblib.load(model_path)
    if not hasattr(model, "feature_importances_"):
        raise ValueError("The loaded model does not expose feature_importances_.")
    feature_names = load_feature_names(feature_names_path)
    importances = [float(value) for value in model.feature_importances_]
    rows = sorted(
        [{"feature": name, "importance": score} for name, score in zip(feature_names, importances)],
        key=lambda item: item["importance"],
        reverse=True,
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "feature_names": str(feature_names_path),
        "method": "random_forest_feature_importances_",
        "top_features": rows,
        "methodology_warning": WARNING,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Random Forest Feature Importance",
        "",
        "## Top features",
        "",
        "| Rank | Feature | Importance |",
        "|---:|---|---:|",
    ]
    for index, row in enumerate(report["top_features"], start=1):
        lines.append(f"| {index} | `{row['feature']}` | {row['importance']:.6f} |")
    lines.extend(["", "## Avertissement méthodologique", report["methodology_warning"], ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = export_importance(args.model, args.feature_names)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    write_json(report, args.output_json)


if __name__ == "__main__":
    main()
