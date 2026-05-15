# Random Forest Feature Importance

## Top features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `final_score_v3` | 0.228767 |
| 2 | `missing_required_count` | 0.214371 |
| 3 | `required_skills_overlap` | 0.189578 |
| 4 | `matched_required_count` | 0.173939 |
| 5 | `must_have_coverage` | 0.170015 |
| 6 | `reliability_score` | 0.005580 |
| 7 | `vector_similarity` | 0.005580 |
| 8 | `experience_match_score` | 0.005361 |
| 9 | `seniority_alignment` | 0.003451 |
| 10 | `profile_quality_score` | 0.003076 |
| 11 | `hallucination_risk_encoded` | 0.000281 |
| 12 | `nice_to_have_overlap` | 0.000000 |

## Avertissement méthodologique
Ces importances sont calculées sur un modèle entraîné avec des pseudo-labels métier contrôlés. Elles indiquent quelles features aident à reproduire la règle de pseudo-labeling, pas nécessairement les critères réels d’un recruteur.
