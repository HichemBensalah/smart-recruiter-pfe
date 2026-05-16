from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.ranking.decision_cards_ml_enricher import (  # noqa: E402
    build_ml_comparison_report,
    write_json_report,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build separate Decision Cards enriched with ML model comparison.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--rf-model", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, required=True)
    parser.add_argument("--xgb-ranking", type=Path, required=True)
    parser.add_argument("--shap-global", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_ml_comparison_report(
        features_path=args.features,
        rf_model_path=args.rf_model,
        feature_names_path=args.feature_names,
        xgb_ranking_path=args.xgb_ranking,
        shap_global_path=args.shap_global,
    )
    write_json_report(report, args.output_json)
    write_markdown_report(report, args.output_md)
    print(f"Decision Cards ML comparison JSON written: {args.output_json}")
    print(f"Decision Cards ML comparison Markdown written: {args.output_md}")


if __name__ == "__main__":
    main()

