from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from src.benchmark.ocr.config import DEFAULT_ENGINES, DEFAULT_OUTPUT_ROOT
from src.benchmark.ocr.dataset import BenchmarkSample, iter_limited, load_ground_truth, load_manifest
from src.benchmark.ocr.metrics import compute_metrics
from src.benchmark.ocr.runners.docling_runner import DoclingRunner
from src.benchmark.ocr.runners.easyocr_runner import EasyOCRRunner
from src.benchmark.ocr.runners.mistral_runner import MistralOCRRunner
from src.benchmark.ocr.runners.pytesseract_runner import PyTesseractRunner
from src.benchmark.ocr.runners.trocr_runner import TrOCRRunner


RUNNER_FACTORIES = {
    "docling": DoclingRunner,
    "mistralocr": MistralOCRRunner,
    "trocr": TrOCRRunner,
    "easyocr": EasyOCRRunner,
    "pytesseract": PyTesseractRunner,
}


def main() -> None:
    args = parse_args()
    samples = iter_limited(load_manifest(args.manifest), args.limit)
    run_dir = build_run_dir(args.output_root)
    prediction_root = run_dir / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)

    runners: dict[str, Any] = {}
    init_errors: dict[str, str] = {}
    for engine in args.engines:
        try:
            runners[engine] = RUNNER_FACTORIES[engine]()
        except Exception as exc:
            init_errors[engine] = f"{type(exc).__name__}: {exc}"
    records: list[dict[str, Any]] = []

    for sample in samples:
        ground_truth = load_ground_truth(sample)
        for engine in args.engines:
            engine_dir = prediction_root / engine
            engine_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = engine_dir / f"{sample.sample_id}.txt"
            metadata_path = engine_dir / f"{sample.sample_id}.json"

            if engine in init_errors:
                records.append(
                    {
                        "sample_id": sample.sample_id,
                        "engine": engine,
                        "status": "failed",
                        "source_path": str(sample.source_path),
                        "ground_truth_path": str(sample.ground_truth_path),
                        "prediction_path": str(prediction_path),
                        "error": f"runner_init_failed: {init_errors[engine]}",
                        "raw_wer": "",
                        "raw_cer": "",
                        "normalized_wer": "",
                        "normalized_cer": "",
                    }
                )
                continue

            runner = runners[engine]
            try:
                prediction = runner.extract(sample)
                prediction_path.write_text(prediction.text, encoding="utf-8")
                metadata_path.write_text(json.dumps(prediction.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                metrics = compute_metrics(ground_truth, prediction.text)
                records.append(
                    {
                        "sample_id": sample.sample_id,
                        "engine": engine,
                        "status": "success",
                        "source_path": str(sample.source_path),
                        "ground_truth_path": str(sample.ground_truth_path),
                        "prediction_path": str(prediction_path),
                        "error": "",
                        **metrics,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "sample_id": sample.sample_id,
                        "engine": engine,
                        "status": "failed",
                        "source_path": str(sample.source_path),
                        "ground_truth_path": str(sample.ground_truth_path),
                        "prediction_path": str(prediction_path),
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_wer": "",
                        "raw_cer": "",
                        "normalized_wer": "",
                        "normalized_cer": "",
                    }
                )

    metrics_path = run_dir / "metrics_per_sample.csv"
    write_csv(metrics_path, records)

    summary = summarize(records)
    summary_path_csv = run_dir / "metrics_summary.csv"
    summary_path_json = run_dir / "metrics_summary.json"
    write_csv(summary_path_csv, summary)
    summary_path_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "manifest": str(Path(args.manifest).resolve()),
                "engines": args.engines,
                "limit": args.limit,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "output_root": str(run_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Benchmark completed. Results written to: {run_dir}")


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["engine"], []).append(record)

    rows: list[dict[str, Any]] = []
    for engine, engine_records in grouped.items():
        successes = [row for row in engine_records if row["status"] == "success"]
        normalized_cer_values = [float(row["normalized_cer"]) for row in successes]
        normalized_wer_values = [float(row["normalized_wer"]) for row in successes]

        row = {
            "engine": engine,
            "samples": len(engine_records),
            "successes": len(successes),
            "failures": len(engine_records) - len(successes),
            "mean_wer": _safe_stat(mean, normalized_wer_values),
            "mean_cer": _safe_stat(mean, normalized_cer_values),
            "median_wer": _safe_stat(median, normalized_wer_values),
            "median_cer": _safe_stat(median, normalized_cer_values),
            "std_wer": _safe_stat(pstdev, normalized_wer_values),
            "std_cer": _safe_stat(pstdev, normalized_cer_values),
            "rank": 0,
        }
        rows.append(row)

    ranked = sorted(rows, key=lambda item: (item["mean_cer"] if item["mean_cer"] != "" else 999, item["mean_wer"] if item["mean_wer"] != "" else 999))
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def build_run_dir(output_root: str | Path) -> Path:
    base = Path(output_root)
    timestamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = base / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_stat(fn, values: list[float]) -> float | str:
    if not values:
        return ""
    if len(values) == 1 and fn is pstdev:
        return 0.0
    return round(float(fn(values)), 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an isolated OCR benchmark on a manifest of documents.")
    parser.add_argument("--manifest", required=True, help="CSV manifest describing benchmark samples.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory where benchmark outputs are written.")
    parser.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES, choices=sorted(RUNNER_FACTORIES))
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
