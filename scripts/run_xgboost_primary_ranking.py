from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.ranking.ml_primary_ranker import (  # noqa: E402
    run_primary_ranking,
    write_json_report,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XGBoost as primary ranking engine over Matching V3 features.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_primary_ranking(
        features_path=args.features,
        model_path=args.model,
        feature_names_path=args.feature_names,
    )
    write_json_report(report, args.output_json)
    write_markdown_report(report, args.output_md)
    print(f"XGBoost primary ranking JSON written: {args.output_json}")
    print(f"XGBoost primary ranking Markdown written: {args.output_md}")


if __name__ == "__main__":
    main()

