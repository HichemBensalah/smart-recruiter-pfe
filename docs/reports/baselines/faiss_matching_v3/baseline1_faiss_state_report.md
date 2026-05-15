# Baseline 1 FAISS + Matching State Report

Generated: `2026-05-07T12:04:15.943673+00:00`

## Decision

- `baseline_name`: baseline1_faiss_matching_v3
- `baseline_frozen`: true
- `ready_for_demo`: true
- `faiss_rebuilt_now`: false
- `matching_rerun_now`: false

## Module State

- Module 1 accepted outputs: `90`
- Module 2 V2 grounded profiles: `90`
- MongoDB `candidate_profiles`: `90`
- MongoDB `candidates`: `75`
- FAISS profiles indexed: `90`
- Matching V3 report exists: `True`

## Current Quality Distribution

```json
{
  "low": 89,
  "medium": 1
}
```

## FAISS

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: `384`
- Metric: `inner_product_on_normalized_vectors`
- Index path: `data\indexes\faiss\cv_index.faiss`
- ID map path: `data\indexes\faiss\id_map.pkl`
- Index file exists: `True`
- ID map file exists: `True`

## Matching V3 Top 1

- Candidate: `Hichem Bensalah`
- Score: `0.8172`
- Risk: `low`
- Explanation: Hichem Bensalah matches on Python, FastAPI, MongoDB, Docker. Text similarity=0.60, experience score=1.00, profile kind=complete_profile, hallucination risk=low. Missing required skills include REST API.

## Patch Status

- URL fixes applied: `1`
- Profile patched: `docx_Aziz_resumer.json`
- MongoDB update: `{'matched_count': 1, 'modified_count': 1}`
- Reliability regressions: `0`
- Risk after patch: `{'low': 89, 'medium': 1}`

## Baseline Files

- `docs/reports/baselines/faiss_matching_v3/baseline1_faiss_matching_report.json`
- `docs/reports/baselines/faiss_matching_v3/baseline1_faiss_ranking_comparison.md`
- `docs/reports/baselines/faiss_matching_v3/baseline1_faiss_state_report.md`
