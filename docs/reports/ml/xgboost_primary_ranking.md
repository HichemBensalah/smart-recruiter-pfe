# XGBoost primary ranking

## Objectif

Produire un ranking principal basé sur XGBoost à partir des features métier issues de Matching V3, tout en conservant Matching V3 comme baseline de comparaison.

## Architecture

- FAISS reste utilisé pour le retrieval initial.
- Matching V3 reste présent et sert de feature engine métier.
- XGBoost calcule `xgboost_score` et définit `final_rank_ml`.
- Matching V3 conserve `baseline_score_v3` et `baseline_rank_v3`.
- SHAP peut expliquer le score XGBoost dans les rapports d'explicabilité existants.

## Job

- `job_id`: `backend_python_django_postgresql`
- modèle: `xgboost`
- mode: `xgboost_primary_with_matching_v3_features`
- candidats rankés: `50`

## Top 10 XGBoost ranking principal

| final_rank_ml | candidate_id | xgboost_score | baseline_rank_v3 | baseline_score_v3 | rank_shift |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `candidate_b6f7add66ffc` | 0.9826 | 1 | 0.7754 | 0 |
| 2 | `candidate_9b0508063f03` | 0.9826 | 2 | 0.7682 | 0 |
| 3 | `candidate_56424ea73690` | 0.9826 | 5 | 0.6084 | 2 |
| 4 | `candidate_1487f3187f7b` | 0.9718 | 3 | 0.7183 | -1 |
| 5 | `candidate_8eea1b635447` | 0.9676 | 4 | 0.6688 | -1 |
| 6 | `candidate_206d746034ef` | 0.9209 | 6 | 0.5382 | 0 |
| 7 | `candidate_851fca236e70` | 0.0043 | 13 | 0.4073 | 6 |
| 8 | `candidate_9f5823fd4c74` | 0.0043 | 15 | 0.4025 | 7 |
| 9 | `candidate_c5201502b687` | 0.0043 | 16 | 0.4023 | 7 |
| 10 | `candidate_418b74b9d404` | 0.0043 | 17 | 0.4023 | 7 |

## Top 10 Matching V3 baseline

| baseline_rank_v3 | candidate_id | baseline_score_v3 | final_rank_ml | xgboost_score | rank_shift |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `candidate_b6f7add66ffc` | 0.7754 | 1 | 0.9826 | 0 |
| 2 | `candidate_9b0508063f03` | 0.7682 | 2 | 0.9826 | 0 |
| 3 | `candidate_1487f3187f7b` | 0.7183 | 4 | 0.9718 | -1 |
| 4 | `candidate_8eea1b635447` | 0.6688 | 5 | 0.9676 | -1 |
| 5 | `candidate_56424ea73690` | 0.6084 | 3 | 0.9826 | 2 |
| 6 | `candidate_206d746034ef` | 0.5382 | 6 | 0.9209 | 0 |
| 7 | `candidate_e1fda1d70be1` | 0.4349 | 24 | 0.0042 | -17 |
| 8 | `candidate_073b7a3d39ba` | 0.4297 | 42 | 0.0038 | -34 |
| 9 | `candidate_4eef95d3a3fa` | 0.4284 | 38 | 0.0040 | -29 |
| 10 | `candidate_664415ab2fe1` | 0.4256 | 48 | 0.0037 | -38 |

## Candidats remontés par XGBoost

| candidate_id | baseline_rank_v3 | final_rank_ml | rank_shift | xgboost_score |
| --- | ---: | ---: | ---: | ---: |
| `candidate_ecc1fcef13a6` | 48 | 23 | 25 | 0.0043 |
| `candidate_7a19fd397cfc` | 44 | 22 | 22 | 0.0043 |
| `candidate_7a599a8c3c9d` | 42 | 21 | 21 | 0.0043 |
| `candidate_7e5aa7ad70d7` | 40 | 20 | 20 | 0.0043 |
| `candidate_e8755c28e52c` | 38 | 19 | 19 | 0.0043 |
| `candidate_789ce8311eb4` | 36 | 18 | 18 | 0.0043 |
| `candidate_f74acce78f96` | 32 | 17 | 15 | 0.0043 |
| `candidate_e7454f1e925e` | 50 | 37 | 13 | 0.0040 |
| `candidate_8c87c4fde8c1` | 25 | 14 | 11 | 0.0043 |
| `candidate_7fcd1b7a9f35` | 26 | 15 | 11 | 0.0043 |

## Candidats descendus par XGBoost

| candidate_id | baseline_rank_v3 | final_rank_ml | rank_shift | xgboost_score |
| --- | ---: | ---: | ---: | ---: |
| `candidate_664415ab2fe1` | 10 | 48 | -38 | 0.0037 |
| `candidate_36923c8bf20e` | 11 | 46 | -35 | 0.0038 |
| `candidate_073b7a3d39ba` | 8 | 42 | -34 | 0.0038 |
| `candidate_4eef95d3a3fa` | 9 | 38 | -29 | 0.0040 |
| `candidate_1df043636ea4` | 28 | 49 | -21 | 0.0037 |
| `candidate_e1fda1d70be1` | 7 | 24 | -17 | 0.0042 |
| `candidate_351493983fdf` | 12 | 28 | -16 | 0.0040 |
| `candidate_7eb2ad2aeb2f` | 23 | 39 | -16 | 0.0038 |
| `candidate_d813a4aedd03` | 29 | 43 | -14 | 0.0038 |
| `candidate_b5ec7c5f3f96` | 14 | 25 | -11 | 0.0042 |

## Limites méthodologiques

- XGBoost est utilisé ici comme moteur principal de ranking ML à partir des features Matching V3. Le modèle reste entraîné sur pseudo-labels métier contrôlés, pas sur labels recruteur. Matching V3 est conservé comme baseline de comparaison.
- `xgboost_score` n'est pas un score recruteur final.
- Le modèle n'a pas été réentraîné dans cette étape.
- Les datasets, FAISS, MongoDB et Decision Cards officielles ne sont pas modifiés.
- La validation humaine reste nécessaire avant toute décision produit.
