from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.graph.decision_cards_transferability_enricher import (  # noqa: E402
    build_transferability_cards_report,
    write_json_report,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build separate Decision Cards enriched with Potential Graph transferability.")
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_transferability_cards_report(
        cards_path=args.cards,
        profiles_dir=args.profiles_dir,
        job_path=args.job,
        graph_path=args.graph,
    )
    write_json_report(report, args.output_json)
    write_markdown_report(report, args.output_md)
    print(f"Decision Cards with transferability JSON written: {args.output_json}")
    print(f"Decision Cards with transferability Markdown written: {args.output_md}")


if __name__ == "__main__":
    main()

