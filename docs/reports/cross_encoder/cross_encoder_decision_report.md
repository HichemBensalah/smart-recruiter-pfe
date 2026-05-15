# CrossEncoder Decision Report

## 1. Experiment Summary

The CrossEncoder experiment was run as a controlled reranking layer on top of the official FAISS + Matching V3 baseline.

- Model used: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- FAISS retrieval size: `top_n=20`
- Approximate latency: `15.9004s`
- Pipeline used: `FAISS top-20 -> CrossEncoder reranking -> Matching V3 business scoring`
- Official baseline compared to: `baseline1_faiss_matching_v3`

Generated files:

- `docs/reports/cross_encoder/matching_single_job_report_faiss_cross_encoder.json`
- `docs/reports/cross_encoder/cross_encoder_comparison_report.json`
- `docs/reports/cross_encoder/cross_encoder_comparison_report.md`
- `docs/reports/cross_encoder/cross_encoder_integration_audit.md`
- `docs/reports/cross_encoder/cross_encoder_integration_audit.json`

## 2. Positive Results

- Hichem Bensalah remains top 1 after CrossEncoder reranking.
- The CrossEncoder top 10 contains `0` medium-risk candidates.
- Candidates with strong must-have coverage remain high in the final ranking.
- CrossEncoder adds a finer semantic signal than FAISS similarity alone.
- The existing Matching V3 scoring still acts as a useful control layer after reranking.

## 3. Problematic Results

- Some candidates receive strong CrossEncoder ranks despite weak must-have coverage.
- `MILDREDZEMLAK` receives CrossEncoder rank `1` with must-have coverage `0.2`.
- `Karina Blick` receives CrossEncoder rank `3` with must-have coverage `0.2`.
- The default MS MARCO reranker is not specialized for HR, CV parsing, or job matching.
- Latency around `15.9s` is significantly higher than the FAISS baseline.
- Hichem's final score decreases from `0.8172` to `0.7078`.
- CrossEncoder scores are not calibrated with the current business weights.

## 4. Decision

- Official demo baseline: `FAISS + Matching V3`
- Experimental baseline: `FAISS + CrossEncoder + Matching V3`
- CrossEncoder is not activated by default.
- CrossEncoder can be used for advanced demos, ranking comparison, and future calibration work.
- The current business scoring and must-have penalties remain the final decision layer.

## 5. Future Recommendation

- Calibrate the CrossEncoder contribution before considering default activation.
- Test reranker models that are more appropriate for CV/job matching than MS MARCO.
- Keep CrossEncoder after a top-N retriever, whether FAISS now or Qdrant later.
- Do not replace the business scoring layer with CrossEncoder.
- Keep the must-have coverage penalty as a guardrail.
- Measure `NDCG@10`, `MRR`, and ranking stability after annotating relevance labels.

## 6. Soutenance Phrase

Nous avons testé un CrossEncoder comme couche de reranking au-dessus de notre baseline FAISS + Matching V3. L'objectif était d'ajouter un signal sémantique plus fin sans modifier les profils, MongoDB, ni l'index FAISS. Le résultat est intéressant: le meilleur candidat reste top 1 et les profils à risque moyen disparaissent du top 10. Cependant, le modèle testé, entraîné sur MS MARCO, favorise aussi certains profils dont la couverture des compétences obligatoires est faible. Nous avons donc décidé de conserver CrossEncoder comme baseline expérimentale et de garder FAISS + Matching V3 comme baseline officielle de démonstration, avec le scoring métier et les pénalités must-have comme garde-fous.
