# Demo run summary

## Objectif de la démo

Préparer en une commande les rapports principaux de démonstration Smart Recruiter sans réentraîner de modèle ni modifier les briques métier.

## Artefacts vérifiés

- `features`: `data\ranking\features\backend_python_django_postgresql.jsonl`
- `job`: `data\job_profiles\backend_python_django_postgresql.json`
- `profiles_dir`: `data\profile_builder_module2_v2_grounded_all\profiles\grounded_profiles`
- `graph`: `data\graph\skills_roles_graph.yaml`
- `rf_model`: `data\ranking\models\random_forest.joblib`
- `xgb_ranking`: `docs\reports\ml\xgboost_primary_ranking.json`
- `feature_names`: `data\ranking\models\feature_names.json`
- `cards_ml`: `docs\reports\decision_cards\decision_cards_ml_comparison.json`

## Artefacts générés

- `decision_cards_with_transferability_json`: `docs\reports\decision_cards\decision_cards_with_transferability.json`
- `decision_cards_with_transferability_md`: `docs\reports\decision_cards\decision_cards_with_transferability.md`
- `demo_summary_top10_json`: `docs\reports\demo\demo_summary_top10.json`
- `demo_summary_top10_md`: `docs\reports\demo\demo_summary_top10.md`
- `demo_executive_summary_json`: `docs\reports\demo\demo_executive_summary.json`
- `demo_executive_summary_md`: `docs\reports\demo\demo_executive_summary.md`
- `demo_run_manifest_json`: `docs\reports\demo\demo_run_manifest.json`
- `demo_run_summary_md`: `docs\reports\demo\demo_run_summary.md`

## Top recommended

- `candidate_9b0508063f03`: Recommandé car score Matching V3 élevé, Random Forest très favorable, XGBoost favorable, pas de gap bloquant détecté.
- `candidate_1487f3187f7b`: Recommandé car score Matching V3 élevé, Random Forest très favorable, XGBoost favorable, transition métier plausible.
- `candidate_8eea1b635447`: Recommandé car score Matching V3 élevé, Random Forest très favorable, XGBoost favorable, transition métier plausible.

## Needs review

- `candidate_664415ab2fe1`: À vérifier car statut review_needed, désaccord important entre les rangs, gaps bloquants: Git, score XGBoost très faible.
- `candidate_073b7a3d39ba`: À vérifier car statut review_needed, désaccord important entre les rangs, gaps bloquants: Git, score XGBoost très faible.
- `candidate_4eef95d3a3fa`: À vérifier car statut review_needed, désaccord important entre les rangs, gaps bloquants: Git, score XGBoost très faible.

## Rapports principaux

- Executive summary: `docs\reports\demo\demo_executive_summary.md`
- Summary top 10: `docs\reports\demo\demo_summary_top10.md`

## Limites méthodologiques courtes

- Matching V3 reste la baseline officielle.
- Les modèles ML sont entraînés sur pseudo-labels métier contrôlés.
- Potential Graph est déclaratif et ne remplace pas une décision recruteur.
