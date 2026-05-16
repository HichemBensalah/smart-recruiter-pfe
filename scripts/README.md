# `scripts/`

Scripts d'orchestration, generation de rapports et experimentation.

Les scripts restent pour l'instant a la racine de `scripts/` afin de ne pas casser les commandes existantes, les tests et la documentation. Une reorganisation en sous-dossiers pourra etre faite progressivement.

## Parsing / Structuring

- `test_grounded_module2_v2.py`
- `patch_grounded_profiles_v2.py`
- `test_experience_parsing.py`

## Matching / Decision Cards

- `run_matching_v3_normalized.py`
- `generate_decision_cards.py`
- `generate_decision_cards_v3_normalized.py`
- `generate_decision_cards_with_ml_experimental.py`
- `build_decision_cards_ml_comparison.py`
- `build_decision_cards_with_transferability.py`

## Ranking ML / XAI

- `build_ranking_features.py`
- `build_ranking_dataset.py`
- `build_pseudo_labels.py`
- `train_ranking_models.py`
- `run_ml_reranking.py`
- `run_xgboost_primary_ranking.py`
- `explain_xgboost_shap.py`
- `export_random_forest_feature_importance.py`
- `export_xgboost_feature_importance.py`

## Graph

- `compute_transferability_score.py`
- `import_graph_to_neo4j.py`

## Demo

- `check_demo_readiness.py`
- `build_demo_summary.py`
- `build_demo_executive_summary.py`
- `run_demo_end_to_end.py`

## Maintenance / Analyse

- `analyze_candidate_corpus.py`
- `build_annotation_sample.py`
- `compare_aligned_offers.py`
- `cleanup_project_artifacts.py`
- `simulate_pseudo_label_rules.py`

## Regle

Ne pas deplacer un script tant que ses imports, tests et commandes documentees n'ont pas ete adaptes. Pour LangGraph, utiliser d'abord l'API FastAPI comme contrat stable.
