# XGBoost Feature Importance

## Top features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `must_have_coverage` | 0.596345 |
| 2 | `final_score_v3` | 0.382791 |
| 3 | `experience_match_score` | 0.007428 |
| 4 | `profile_quality_score` | 0.006199 |
| 5 | `vector_similarity` | 0.004178 |
| 6 | `reliability_score` | 0.003059 |
| 7 | `required_skills_overlap` | 0.000000 |
| 8 | `nice_to_have_overlap` | 0.000000 |
| 9 | `seniority_alignment` | 0.000000 |
| 10 | `hallucination_risk_encoded` | 0.000000 |
| 11 | `missing_required_count` | 0.000000 |
| 12 | `matched_required_count` | 0.000000 |

## Comparaison courte avec Random Forest
Top Random Forest features: final_score_v3, missing_required_count, required_skills_overlap, matched_required_count, must_have_coverage. Compare with XGBoost top features below; both remain tied to pseudo-label reproduction.

## Avertissement méthodologique
Ces importances sont calculées sur un modèle entraîné avec des pseudo-labels métier contrôlés et dans un contexte de circularité partielle. Elles indiquent quelles features aident à reproduire la règle de pseudo-labeling, pas nécessairement les critères réels d’un recruteur.
