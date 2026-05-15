# Matching Grounded V2 Audit

## How the current matcher reads V2 profiles

The matcher reads candidate_profiles from MongoDB, uses FAISS id_map.pkl to retrieve profile ids, then loads full profile documents by profile_id.

## Fields used in the V2 candidate text

- io.full_name
- io.location
- expertise.summary
- expertise.hard_skills
- expertise.soft_skills
- experiences
- education

## Fields that were ignored but should matter

- 
eliability_score
- profile_kind
- hallucination_risk
- quality_flags
- ields_nullified
- ull_name that is null, template-like, or title-like

## Why suspicious names were still visible

Entries like O Ariana, Tunisia, Data Scientist, RESUME OBJECTIVE, and rom Resume Genius were not a FAISS bug. They came from a combination of:

- noisy io.full_name values still present in grounded MongoDB documents
- candidate text generation that injected those values into embeddings
- display logic that showed any non-empty ull_name without validation
- scoring that did not penalize weak names or hallucination risk strongly enough

## Root cause assessment

- MongoDB content: partly responsible
- profile_text_builder: strongly responsible
- scoring: strongly responsible
- display-name handling: strongly responsible
- FAISS: not the root cause

## Takeaway

The matching problem came from a mix of noisy grounded content and insufficient downstream quality controls. The grounded data reduced hallucinations, but the matching layer still needed to become quality-aware and name-safe.
