# OCR Benchmark Starter Run

Run directory: `data/benchmarks/ocr/outputs/run_20260412_231312`

## Dataset

- Samples: `9`
- Source: bootstrap from existing repository assets
- Ground truth type: `proxy_ground_truth_from_matching_docx_text`

This run is a starter benchmark, not yet a final scientific benchmark with manually curated gold annotations.

## Engines executed

- Docling
- PyTesseract
- EasyOCR
- TrOCR

`MistralOCR` was not executed in this run because no `MISTRAL_API_KEY` was available locally.

## Summary ranking

| Rank | Engine | Successes | Mean WER | Mean CER |
|---|---:|---:|---:|---:|
| 1 | pytesseract | 9/9 | 0.181838 | 0.094410 |
| 2 | easyocr | 9/9 | 0.299233 | 0.105863 |
| 3 | docling | 9/9 | 0.417059 | 0.206549 |
| 4 | trocr | 9/9 | 1.000000 | 0.995365 |

## Interpretation

- On this starter proxy dataset, `PyTesseract` achieved the best average CER and WER.
- `EasyOCR` ranked second and remained fully executable on the local environment.
- `Docling` remained robust and fully executable, but its plain-text output was less aligned with this proxy ground-truth convention.
- `TrOCR` executed successfully but performed very poorly on full-page CV inputs in its current configuration.

## Files to inspect

- `metrics_per_sample.csv`
- `metrics_summary.csv`
- `metrics_summary.json`
- `predictions/`
