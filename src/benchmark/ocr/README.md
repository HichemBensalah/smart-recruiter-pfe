# OCR Benchmark

This module is intentionally isolated from `src/core/parser` and `src/core/structuring`.

## Purpose

Run a repeatable OCR benchmark across:

- Docling
- Mistral OCR
- TrOCR
- EasyOCR
- PyTesseract

while computing:

- WER
- CER

## Important

The benchmark scaffold can be used in two modes:

1. `scientific benchmark`
   Requires manually curated ground truth text for each sample.
2. `starter proxy benchmark`
   Uses text extracted from matching DOCX files already present in the repo as proxy ground truth.

The starter proxy benchmark is useful for validating the tooling and producing first comparative runs,
but it is not equivalent to a manually transcribed gold standard.

## Main commands

Bootstrap a starter dataset from repo assets:

```powershell
python -m src.benchmark.ocr.bootstrap_existing_dataset
```

Run the benchmark:

```powershell
python -m src.benchmark.ocr.run_ocr_benchmark `
  --manifest data\benchmarks\ocr\dataset\manifests\benchmark_manifest.csv `
  --output-root data\benchmarks\ocr\outputs `
  --engines docling pytesseract trocr
```

Rebuild summary files:

```powershell
python -m src.benchmark.ocr.summarize_results `
  --metrics-per-sample data\benchmarks\ocr\outputs\run_YYYYMMDD_HHMMSS\metrics_per_sample.csv `
  --output-csv data\benchmarks\ocr\outputs\run_YYYYMMDD_HHMMSS\metrics_summary.csv `
  --output-json data\benchmarks\ocr\outputs\run_YYYYMMDD_HHMMSS\metrics_summary.json
```
