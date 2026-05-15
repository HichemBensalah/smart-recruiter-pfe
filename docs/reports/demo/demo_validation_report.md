# Demo Validation Report

Generated: `2026-05-07T12:04:15.943673+00:00`

## Decision

`ready_for_demo`: **true**

The current FAISS + Matching V3 baseline is frozen for the technical demo. No Module 1 rerun, Module 2 rerun, LLM call, MongoDB write, FAISS rebuild, matching rerun, CrossEncoder, XGBoost, Qdrant, or LangGraph execution was performed during this validation step.

## Confirmed Numbers

- Module 1 accepted outputs: `90`
- Module 2 V2 grounded outputs: `90`
- MongoDB `candidate_profiles`: `90`
- MongoDB `candidates`: `75`
- FAISS indexed profiles: `90`
- Matching V3 exists: `True`
- Current hallucination risk distribution: `{'low': 89, 'medium': 1}`
- Patch LinkedIn applied without regression: `True`

## Current Top 10

- #1 Hichem Bensalah | score=0.8172 | risk=low | kind=complete_profile | skills=Python, FastAPI, MongoDB, Docker
- #2 MOHAMED AZIZ BELAWEID | score=0.5528 | risk=low | kind=complete_profile | skills=Python, FastAPI, Docker
- #3 Candidate (ID: candidate_8eea1b635447) | score=0.5466 | risk=medium | kind=complete_profile | skills=Python, FastAPI, MongoDB, Docker
- #4 JEFFERYGORCZANY | score=0.4332 | risk=low | kind=complete_profile | skills=Python, Docker
- #5 Candidate (ID: candidate_71e03ea99985) | score=0.3502 | risk=low | kind=complete_profile | skills=Python, MongoDB
- #6 Candidate (ID: candidate_1d475044c93c) | score=0.3464 | risk=low | kind=complete_profile | skills=Python, MongoDB
- #7 Markus Rohan | score=0.3181 | risk=medium | kind=partial_profile | skills=Python, Docker
- #8 Candidate (ID: candidate_c564b8eceb3d) | score=0.2731 | risk=medium | kind=complete_profile | skills=Python, MongoDB
- #9 MILDREDZEMLAK | score=0.2628 | risk=low | kind=complete_profile | skills=Python
- #10 Justine Hendrickson | score=0.2573 | risk=low | kind=complete_profile | skills=Python

## Top 1 Explanation

Hichem Bensalah matches on Python, FastAPI, MongoDB, Docker. Text similarity=0.60, experience score=1.00, profile kind=complete_profile, hallucination risk=low. Missing required skills include REST API.

## Module 1 State

Module 1 is treated as stable existing input for the demo. The accepted handoff contains `90` CVs at `data/processed_official_module1/handoff/accepted.json`.

## Module 2 V2 Grounded State

The grounded profile directory contains `90` JSON profiles. Current profile kind distribution is `{'complete_profile': 90}` and current hallucination risk distribution is `{'low': 89, 'medium': 1}`.

## MongoDB State

MongoDB was validated read-only after the selective LinkedIn patch. `candidate_profiles` contains `90` documents and `candidates` contains `75` deduplicated candidates.

## FAISS State

FAISS currently indexes `90` profiles with `sentence-transformers/all-MiniLM-L6-v2` and dimension `384`. Rebuild is not required now because the executed patch only normalized a LinkedIn contact URL.

## Matching V3 State

The current Matching V3 report is present at `docs/reports/matching/v3/matching_single_job_report_grounded_v3.json`. Hichem Bensalah remains Top 1 with score `0.8172`.

## Limits To Say Honestly

- Le rapport qualite Module 2 historique a ete genere avant les corrections finales; pour la demo, la distribution actuelle est prise depuis les profils grounded et MongoDB.
- Un profil medium-risk reste dans les donnees et peut apparaitre dans le Top 10 avec penalite qualite, ce qui est volontaire mais a expliquer.
- Certains noms faibles sont masques sous forme Candidate (ID: ...); c'est plus sur que d'afficher un nom hallucine, mais moins lisible en demo.

## Files To Show In Demo

- `docs/reports/demo/demo_validation_report.md`
- `docs/reports/baselines/faiss_matching_v3/baseline1_faiss_state_report.md`
- `docs/reports/baselines/faiss_matching_v3/baseline1_faiss_matching_report.json`
- `docs/reports/baselines/faiss_matching_v3/baseline1_faiss_ranking_comparison.md`
- `data/indexes/faiss/index_report.json`
- `docs/reports/patches/module2_v2_url_fixes/patch_report_v2_fixes.json`

## Cleanup Note

No files were deleted in this step. The workspace contains older runs and reports that may be archive candidates, but deletion should be done only after naming exact paths to remove.
