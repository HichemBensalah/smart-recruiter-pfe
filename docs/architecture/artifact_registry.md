# Registre des artefacts

Ce registre classe les fichiers importants pour eviter de confondre baseline officielle, experimentation et demo.

## OFFICIEL

- Parsing Module 1 : `src/core/parser/`, `data/processed_official_module1/`
- Structuring Grounded Module 2 : `src/core/structuring/`, `data/profile_builder_module2_v2_grounded_all/`
- MongoDB import : `src/core/storage/`, `docs/reports/mongodb/mongodb_import_report_v2_grounded_execute.json`
- FAISS index : `data/indexes/faiss/index_report.json`, `cv_index.faiss`, `id_map.pkl`
- Matching V3 : `src/core/matching/`, `docs/reports/matching/v3/*_matching_report_v3_normalized.json`
- Decision Cards officielles : `docs/reports/matching/v3/decision_cards_v3_normalized.json`
- API FastAPI : `src/api/`

## EXPERIMENTAL

- CrossEncoder : `scripts/run_cross_encoder_reranking.py`, `scripts/run_cross_encoder_reranking_constrained.py`, `docs/reports/retrieval/`
- Pseudo-labels : `scripts/build_pseudo_labels.py`, `data/ranking/datasets/*pseudo_labeled*.jsonl`
- ML LR/RF/XGBoost : `scripts/train_ranking_models.py`, `data/ranking/models/`
- SHAP : `scripts/explain_xgboost_shap.py`, `docs/reports/ml/shap/`
- ML comparison cards : `docs/reports/decision_cards/decision_cards_ml_comparison.*`
- XGBoost primary ranking : `docs/reports/ml/xgboost_primary_ranking.*`

## DEMO

- Script end-to-end : `scripts/run_demo_end_to_end.py`
- Summary top 10 : `docs/reports/demo/demo_summary_top10.*`
- Executive summary : `docs/reports/demo/demo_executive_summary.*`
- Manifest : `docs/reports/demo/demo_run_manifest.json`
- Run summary : `docs/reports/demo/demo_run_summary.md`
- Decision Cards with Transferability : `docs/reports/decision_cards/decision_cards_with_transferability.*`

## OPTIONNEL / AVANCE

- Potential Graph YAML : `data/graph/skills_roles_graph.yaml`
- Neo4j Graph-RAG : `src/core/graph/neo4j_client.py`, `src/core/graph/neo4j_transferability.py`, `scripts/import_graph_to_neo4j.py`
- Futur Copilot LangGraph : a developper.

## ARCHIVE / A NE PAS UTILISER COMME BASELINE

- `data/archive_old_runs/`
- Anciens rapports de matching hors `docs/reports/matching/v3/`
- Rapports CrossEncoder non contraints
- Fichiers temporaires `.tmp/`, `.tmp_pytest/`, caches pytest/ruff/mypy

Ces fichiers peuvent rester utiles pour l'historique, mais ne doivent pas etre presentes comme baseline officielle.
