from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.benchmark.ocr.run_ocr_benchmark import summarize, write_csv


def main() -> None:
    args = parse_args()
    records = read_csv(Path(args.metrics_per_sample))
    summary = summarize(records)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    write_csv(output_csv, summary)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary written to: {output_csv} and {output_json}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild OCR benchmark summary files from metrics_per_sample.csv.")
    parser.add_argument("--metrics-per-sample", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
