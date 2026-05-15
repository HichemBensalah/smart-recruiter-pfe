# CrossEncoder Integration Audit

## Current FAISS Retrieval

- `src/core/matching/recommender.py` loads the existing FAISS index from `data/indexes/faiss/cv_index.faiss`.
- The ID map is loaded from `data/indexes/faiss/id_map.pkl`.
- `build_job_text()` from `src/core/matching/job_text_builder.py` creates the job query text.
- The sentence-transformer model embeds the job text, then FAISS retrieves the nearest profile rows.
- MongoDB `candidate_profiles` is read only after retrieval to hydrate full profile documents.
- `score_candidate()` from `src/core/matching/scoring.py` computes the business score.
- Profiles are grouped by `candidate_id`, the best profile per candidate is selected, and the final top-k is returned.

## CrossEncoder Insertion Point

The reranker should run after FAISS returns top-N profile rows and after MongoDB hydrates the profile documents, but before final business scoring and candidate grouping.

Flow:

1. FAISS retrieves top-N profile rows.
2. MongoDB hydrates only those profile documents.
3. Candidate text is built with `build_candidate_text()`.
4. CrossEncoder scores `(job_text, candidate_text)` pairs.
5. The normalized CrossEncoder score is passed into existing Matching V3 scoring as the semantic similarity signal.
6. Existing business scoring, quality penalties, skill coverage, risk handling, and explanations remain active.

## Texts Used

- Job text: `build_job_text(job_profile)` from `src/core/matching/job_text_builder.py`.
- Candidate text: `build_candidate_text(profile_doc)` from `src/core/matching/profile_text_builder.py`.

These are already the canonical text builders used by the current matching stack, so no Module 1 or Module 2 output changes are required.

## Report Fields To Add

Each recommendation should keep existing fields and add:

- `faiss_rank`
- `faiss_score`
- `cross_encoder_rank`
- `cross_encoder_score`
- `cross_encoder_score_normalized`
- `cross_encoder_error` when fallback occurs

The top-level report should add metadata:

- `retrieval_engine`
- `reranker`
- `cross_encoder_model`
- `cross_encoder_top_n`
- `baseline_compared_to`

## Files To Create Or Modify

Created:

- `src/core/retrieval/__init__.py`
- `src/core/retrieval/cross_encoder_reranker.py`
- `src/core/retrieval/cross_encoder_comparison.py`
- `docs/reports/cross_encoder/cross_encoder_integration_audit.md`
- `docs/reports/cross_encoder/cross_encoder_integration_audit.json`

Modified:

- `src/core/matching/recommender.py`
- `src/core/matching/evaluation.py`

## Why FAISS Baseline Will Not Be Overwritten

- CrossEncoder is opt-in through `use_cross_encoder=False` by default.
- Baseline reports stay under `docs/reports/baselines/faiss_matching_v3/`.
- New reports are written only under `docs/reports/cross_encoder/`.
- FAISS index files are read, not rebuilt or written.
- MongoDB is read, not updated.
- Module 1 and Module 2 outputs are not touched.
