# ML re-ranking expérimental XGBoost

## Objectif

Ajouter une couche de re-ranking ML standalone au-dessus des résultats Matching V3, sans remplacer la baseline officielle.

## Job

- `job_id`: `backend_python_django_postgresql`
- modèle: `xgboost`
- mode: `experimental_ml_reranking`
- candidats rerankés: `50`

## Top 10 Matching V3

| rank_v3 | candidate_id | final_score_v3 | experimental_ml_score | ml_rank |
| --- | --- | ---: | ---: | ---: |
| 1 | `candidate_b6f7add66ffc` | 0.7754 | 0.9826 | 1 |
| 2 | `candidate_9b0508063f03` | 0.7682 | 0.9826 | 2 |
| 3 | `candidate_1487f3187f7b` | 0.7183 | 0.9718 | 4 |
| 4 | `candidate_8eea1b635447` | 0.6688 | 0.9676 | 5 |
| 5 | `candidate_56424ea73690` | 0.6084 | 0.9826 | 3 |
| 6 | `candidate_206d746034ef` | 0.5382 | 0.9209 | 6 |
| 7 | `candidate_e1fda1d70be1` | 0.4349 | 0.0042 | 24 |
| 8 | `candidate_073b7a3d39ba` | 0.4297 | 0.0038 | 42 |
| 9 | `candidate_4eef95d3a3fa` | 0.4284 | 0.0040 | 38 |
| 10 | `candidate_664415ab2fe1` | 0.4256 | 0.0037 | 48 |

## Top 10 XGBoost experimental reranking

| ml_rank | candidate_id | experimental_ml_score | final_score_v3 | rank_v3 |
| --- | --- | ---: | ---: | ---: |
| 1 | `candidate_b6f7add66ffc` | 0.9826 | 0.7754 | 1 |
| 2 | `candidate_9b0508063f03` | 0.9826 | 0.7682 | 2 |
| 3 | `candidate_56424ea73690` | 0.9826 | 0.6084 | 5 |
| 4 | `candidate_1487f3187f7b` | 0.9718 | 0.7183 | 3 |
| 5 | `candidate_8eea1b635447` | 0.9676 | 0.6688 | 4 |
| 6 | `candidate_206d746034ef` | 0.9209 | 0.5382 | 6 |
| 7 | `candidate_851fca236e70` | 0.0043 | 0.4073 | 13 |
| 8 | `candidate_9f5823fd4c74` | 0.0043 | 0.4025 | 15 |
| 9 | `candidate_c5201502b687` | 0.0043 | 0.4023 | 16 |
| 10 | `candidate_418b74b9d404` | 0.0043 | 0.4023 | 17 |

## Changements de rang les plus importants

| candidate_id | rank_v3 | ml_rank | delta_rank | score_delta |
| --- | ---: | ---: | ---: | ---: |
| `candidate_664415ab2fe1` | 10 | 48 | 38 | -0.4219 |
| `candidate_36923c8bf20e` | 11 | 46 | 35 | -0.4167 |
| `candidate_073b7a3d39ba` | 8 | 42 | 34 | -0.4259 |
| `candidate_4eef95d3a3fa` | 9 | 38 | 29 | -0.4244 |
| `candidate_ecc1fcef13a6` | 48 | 23 | -25 | -0.2188 |
| `candidate_7a19fd397cfc` | 44 | 22 | -22 | -0.2275 |
| `candidate_1df043636ea4` | 28 | 49 | 21 | -0.2711 |
| `candidate_7a599a8c3c9d` | 42 | 21 | -21 | -0.2300 |
| `candidate_7e5aa7ad70d7` | 40 | 20 | -20 | -0.2390 |
| `candidate_e8755c28e52c` | 38 | 19 | -19 | -0.2433 |

## Limites méthodologiques

- Le score ML est expérimental et entraîné sur pseudo-labels métier contrôlés. Matching V3 reste la baseline officielle.
- `experimental_ml_score` n'est pas un score recruteur final.
- Les pseudo-labels utilisés pour entraîner le modèle sont dérivés de règles métier contrôlées.
- Cette étape valide une intégration technique de re-ranking, pas un modèle supervisé final.
- Les Decision Cards ne sont pas modifiées par cette expérimentation.
