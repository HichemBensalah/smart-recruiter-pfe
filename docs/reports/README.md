# Reports Organization

This folder groups generated reports and demo artifacts so `data/` stays focused on datasets, indexes, and pipeline outputs.

## Structure

- `baselines/faiss_matching_v3/`: frozen FAISS + Matching V3 baseline for the technical demo.
- `demo/`: demo validation, current-state recap, and presentation guide.
- `matching/v3/`: Matching V3 reports and ranking comparisons.
- `module1/`: Module 1 parser documentation reports.
- `module2/audits/`: Module 2 hallucination and grounding audit reports.
- `mongodb/`: MongoDB import dry-run and execute reports.
- `patches/module2_v2_url_fixes/`: selective Module 2 V2 URL patch reports.

Prompts for Codex handoff live in `docs/prompts/`.

## Notes

- No source CVs were moved.
- No Module 1 or Module 2 outputs were regenerated.
- No FAISS rebuild or matching rerun was performed for this organization pass.
- No report was deleted; files were moved into thematic folders.
