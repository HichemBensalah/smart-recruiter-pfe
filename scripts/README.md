# `scripts/`

Scripts d'orchestration, de génération de rapports et d'expérimentation.

Les scripts restent pour l'instant à la racine de `scripts/` afin de ne pas casser les commandes existantes, les tests et la documentation. Une réorganisation en sous-dossiers pourra être faite progressivement après la branche `stable/mvp-demo`.

## Scripts officiels / critiques

- `run_fast_tests.py` : suite rapide officielle du MVP.
- `seed_mongodb_from_artifacts.py` : seed idempotent E1 des collections metier MongoDB depuis les artefacts MVP.
- `evaluate_copilot.py` : évaluation Copilot Phase 9.
- `run_demo_end_to_end.py` : régénération du manifeste et des résumés de démo.
- `check_demo_readiness.py` : vérification de readiness démo.
- `import_graph_to_neo4j.py` : import optionnel du graphe YAML vers Neo4j.
- `run_matching_v3_normalized.py` : pipeline Matching V3 normalisé.
- `build_ranking_features.py` : construction des features ranking.
- `build_decision_cards_with_transferability.py` : Decision Cards enrichies par transferability.
- `build_demo_summary.py` et `build_demo_executive_summary.py` : synthèses de démonstration.

Ces scripts sont critiques pour reproduire le MVP actuel ou le défendre devant un jury.

## Parsing / Structuring

- `test_grounded_module2_v2.py`
- `patch_grounded_profiles_v2.py`
- `test_experience_parsing.py`

Ces scripts sont historiques ou liés à la qualité Module 1 / Module 2. Ne pas les supprimer sans archive.

## Matching / Decision Cards

- `run_matching_v3_normalized.py`
- `generate_decision_cards.py`
- `generate_decision_cards_v3_normalized.py`
- `generate_decision_cards_with_ml_experimental.py`
- `build_decision_cards_ml_comparison.py`
- `build_decision_cards_with_transferability.py`

Matching V3 reste la baseline officielle. Les scripts ML/experimental cards enrichissent l'analyse mais ne remplacent pas Matching V3.

En E2, `/api/match` peut fonctionner avec `MATCHING_MODE=artifact|live|hybrid`. Les scripts Matching V3 continuent de produire les artefacts seed/fallback ; le mode live reutilise MongoDB + FAISS + `score_candidate()` sans reconstruire l'index dynamiquement.

## MongoDB / seed E1

- `seed_mongodb_from_artifacts.py`

Ce script importe les artefacts existants vers MongoDB sans supprimer les fichiers sources :

- candidats depuis les Decision Cards ;
- profils candidats depuis `data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/` ;
- job profiles depuis `data/job_profiles/` ;
- Decision Cards dans `decision_cards` ;
- artefacts Matching V3 dans `matching_runs` comme seeds techniques.

Il upsert les documents par cle stable et affiche un resume `inserted`, `updated`, `skipped`.

```bash
python scripts/seed_mongodb_from_artifacts.py --dry-run
python scripts/seed_mongodb_from_artifacts.py
```

## Ranking ML / XAI expérimental

- `build_ranking_features.py`
- `build_ranking_dataset.py`
- `build_pseudo_labels.py`
- `train_ranking_models.py`
- `run_ml_reranking.py`
- `run_xgboost_primary_ranking.py`
- `explain_xgboost_shap.py`
- `export_random_forest_feature_importance.py`
- `export_xgboost_feature_importance.py`
- `generate_ml_experiment_interpretation.py`
- `simulate_pseudo_label_rules.py`

Ces scripts sont utiles pour la recherche ML/XAI, mais ils doivent rester séparés du chemin critique entreprise tant que des labels recruteur réels ne sont pas disponibles.

## Graph

- `compute_transferability_score.py`
- `import_graph_to_neo4j.py`

Neo4j est optionnel dans le MVP actuel. Le fallback YAML doit rester disponible tant que la branche entreprise n'a pas rendu Neo4j obligatoire avec seed, healthcheck et tests d'intégration.

## Demo / évaluation

- `evaluate_copilot.py`
- `check_demo_readiness.py`
- `build_demo_summary.py`
- `build_demo_executive_summary.py`
- `run_demo_end_to_end.py`

Ces scripts soutiennent la démonstration et les preuves de fonctionnement.

## Maintenance / analyse

- `analyze_candidate_corpus.py`
- `build_annotation_sample.py`
- `compare_aligned_offers.py`
- `cleanup_project_artifacts.py`

`cleanup_project_artifacts.py` est sensible : ne jamais l'utiliser sans dry-run clair, revue manuelle et sauvegarde.

## Règles

- Ne pas déplacer un script tant que ses imports, tests et commandes documentées n'ont pas été adaptés.
- Pour LangGraph, utiliser d'abord l'API FastAPI comme contrat stable.
- Garder `run_fast_tests.py` comme point d'entrée de smoke test.
- Créer plus tard des sous-dossiers `demo/`, `evaluation/`, `graph/`, `ml/`, `maintenance/`, `legacy/` seulement avec wrappers de compatibilité.
