from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.ranking.ml_reranker import (  # noqa: E402
    build_reranking_report,
    load_feature_names,
    load_jsonl,
    load_model,
    write_json_report,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone experimental XGBoost re-ranking on Matching V3 features.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.features)
    model = load_model(args.model)
    feature_names = load_feature_names(args.feature_names)
    report = build_reranking_report(rows=rows, model=model, feature_names=feature_names)
    write_json_report(report, args.output_json)
    write_markdown_report(report, args.output_md)
    print(f"ML reranking report written: {args.output_json}")
    print(f"Markdown report written: {args.output_md}")


if __name__ == "__main__":
    main()

