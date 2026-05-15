# SHAP Global Summary - XGBoost expérimental

Ces explications SHAP concernent un modèle entraîné sur pseudo-labels métier contrôlés, pas sur labels recruteur.

## Top features SHAP

| Rank | Feature | Mean absolute SHAP | XGBoost importance |
|---:|---|---:|---:|
| 1 | `final_score_v3` | 3.624729 | 0.382791 |
| 2 | `must_have_coverage` | 0.586121 | 0.596345 |
| 3 | `vector_similarity` | 0.097566 | 0.004178 |
| 4 | `profile_quality_score` | 0.051121 | 0.006199 |
| 5 | `experience_match_score` | 0.012372 | 0.007428 |
| 6 | `reliability_score` | 0.009846 | 0.003059 |
| 7 | `required_skills_overlap` | 0.000000 | 0.000000 |
| 8 | `nice_to_have_overlap` | 0.000000 | 0.000000 |
| 9 | `seniority_alignment` | 0.000000 | 0.000000 |
| 10 | `hallucination_risk_encoded` | 0.000000 | 0.000000 |
| 11 | `missing_required_count` | 0.000000 | 0.000000 |
| 12 | `matched_required_count` | 0.000000 | 0.000000 |

## Comparaison rapide avec xgboost_feature_importance
Top SHAP and XGBoost built-in importance overlap on: final_score_v3, must_have_coverage, vector_similarity, profile_quality_score, experience_match_score. Both explain a model trained to reproduce controlled business pseudo-labels.

## Avertissement méthodologique
Ces explications SHAP concernent un modèle entraîné sur pseudo-labels métier contrôlés, pas sur labels recruteur.
