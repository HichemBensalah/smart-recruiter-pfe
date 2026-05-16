# Decision Cards ML comparison

## Objectif

Créer une version séparée des Decision Cards qui compare Matching V3, Random Forest et XGBoost sans modifier les cartes officielles.

## Architecture

- Matching V3 reste la baseline officielle et le score de référence.
- Random Forest est le meilleur modèle ML actuel selon les métriques observées.
- XGBoost est conservé pour SHAP et l'analyse avancée.
- Les trois scores sont affichés pour transparence, sans décision automatique finale.

## Synthèse

- `job_id`: `backend_python_django_postgresql`
- cartes générées: `50`
- cartes officielles rattachées: `9`

## Top 10 Matching V3

| baseline_rank_v3 | candidate_id | baseline_score_v3 | rf_rank | xgboost_rank | status |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `candidate_b6f7add66ffc` | 0.7754 | 5 | 1 | review_needed |
| 2 | `candidate_9b0508063f03` | 0.7682 | 1 | 2 | agreement_high |
| 3 | `candidate_1487f3187f7b` | 0.7183 | 2 | 4 | agreement_high |
| 4 | `candidate_8eea1b635447` | 0.6688 | 3 | 5 | agreement_high |
| 5 | `candidate_56424ea73690` | 0.6084 | 4 | 3 | agreement_high |
| 6 | `candidate_206d746034ef` | 0.5382 | 6 | 6 | agreement_high |
| 7 | `candidate_e1fda1d70be1` | 0.4349 | 7 | 24 | review_needed |
| 8 | `candidate_073b7a3d39ba` | 0.4297 | 8 | 42 | review_needed |
| 9 | `candidate_4eef95d3a3fa` | 0.4284 | 9 | 38 | review_needed |
| 10 | `candidate_664415ab2fe1` | 0.4256 | 10 | 48 | review_needed |

## Top 10 Random Forest

| rf_rank | candidate_id | rf_score | baseline_rank_v3 | xgboost_rank | status |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `candidate_9b0508063f03` | 1.0000 | 2 | 2 | agreement_high |
| 2 | `candidate_1487f3187f7b` | 1.0000 | 3 | 4 | agreement_high |
| 3 | `candidate_8eea1b635447` | 1.0000 | 4 | 5 | agreement_high |
| 4 | `candidate_56424ea73690` | 1.0000 | 5 | 3 | agreement_high |
| 5 | `candidate_b6f7add66ffc` | 0.9950 | 1 | 1 | review_needed |
| 6 | `candidate_206d746034ef` | 0.9950 | 6 | 6 | agreement_high |
| 7 | `candidate_e1fda1d70be1` | 0.0000 | 7 | 24 | review_needed |
| 8 | `candidate_073b7a3d39ba` | 0.0000 | 8 | 42 | review_needed |
| 9 | `candidate_4eef95d3a3fa` | 0.0000 | 9 | 38 | review_needed |
| 10 | `candidate_664415ab2fe1` | 0.0000 | 10 | 48 | review_needed |

## Top 10 XGBoost

| xgboost_rank | candidate_id | xgboost_score | baseline_rank_v3 | rf_rank | status |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `candidate_b6f7add66ffc` | 0.9826 | 1 | 5 | review_needed |
| 2 | `candidate_9b0508063f03` | 0.9826 | 2 | 1 | agreement_high |
| 3 | `candidate_56424ea73690` | 0.9826 | 5 | 4 | agreement_high |
| 4 | `candidate_1487f3187f7b` | 0.9718 | 3 | 2 | agreement_high |
| 5 | `candidate_8eea1b635447` | 0.9676 | 4 | 3 | agreement_high |
| 6 | `candidate_206d746034ef` | 0.9209 | 6 | 6 | agreement_high |
| 7 | `candidate_851fca236e70` | 0.0043 | 13 | 13 | review_needed |
| 8 | `candidate_9f5823fd4c74` | 0.0043 | 15 | 15 | review_needed |
| 9 | `candidate_c5201502b687` | 0.0043 | 16 | 16 | review_needed |
| 10 | `candidate_418b74b9d404` | 0.0043 | 17 | 17 | review_needed |

## Candidats avec fort désaccord

| candidate_id | baseline_rank_v3 | rf_rank | xgboost_rank | v3_vs_rf | v3_vs_xgb | rf_vs_xgb | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `candidate_664415ab2fe1` | 10 | 10 | 48 | 0 | -38 | -38 | review_needed |
| `candidate_36923c8bf20e` | 11 | 11 | 46 | 0 | -35 | -35 | review_needed |
| `candidate_073b7a3d39ba` | 8 | 8 | 42 | 0 | -34 | -34 | review_needed |
| `candidate_4eef95d3a3fa` | 9 | 9 | 38 | 0 | -29 | -29 | review_needed |
| `candidate_ecc1fcef13a6` | 48 | 48 | 23 | 0 | 25 | 25 | review_needed |
| `candidate_7a19fd397cfc` | 44 | 44 | 22 | 0 | 22 | 22 | review_needed |
| `candidate_1df043636ea4` | 28 | 28 | 49 | 0 | -21 | -21 | review_needed |
| `candidate_7a599a8c3c9d` | 42 | 42 | 21 | 0 | 21 | 21 | review_needed |
| `candidate_7e5aa7ad70d7` | 40 | 40 | 20 | 0 | 20 | 20 | review_needed |
| `candidate_e8755c28e52c` | 38 | 38 | 19 | 0 | 19 | 19 | review_needed |

## Exemples de cartes candidats

### Hichem Bensalah

- `candidate_id`: `candidate_1487f3187f7b`
- verdict officiel: Présélection recommandée
- Matching V3: rank `3`, score `0.7183`
- Random Forest: rank `2`, score `1.0000`
- XGBoost: rank `4`, score `0.9718`
- status: `agreement_high`
- SHAP top features: final_score_v3, must_have_coverage, vector_similarity

### Candidate (ID: candidate_8eea1b635447)

- `candidate_id`: `candidate_8eea1b635447`
- verdict officiel: Présélection recommandée
- Matching V3: rank `4`, score `0.6688`
- Random Forest: rank `3`, score `1.0000`
- XGBoost: rank `5`, score `0.9676`
- status: `agreement_high`
- SHAP top features: final_score_v3, must_have_coverage, vector_similarity

### MOHAMED AZIZ BELAWEID

- `candidate_id`: `candidate_206d746034ef`
- verdict officiel: À considérer avec vérification
- Matching V3: rank `6`, score `0.5382`
- Random Forest: rank `6`, score `0.9950`
- XGBoost: rank `6`, score `0.9209`
- status: `agreement_high`
- SHAP top features: final_score_v3, must_have_coverage, vector_similarity

### LEATRICEFRIESEN

- `candidate_id`: `candidate_e1fda1d70be1`
- verdict officiel: Non prioritaire — gaps importants
- Matching V3: rank `7`, score `0.4349`
- Random Forest: rank `7`, score `0.0000`
- XGBoost: rank `24`, score `0.0042`
- status: `review_needed`
- SHAP top features: final_score_v3, must_have_coverage, vector_similarity

### Markus Rohan

- `candidate_id`: `candidate_073b7a3d39ba`
- verdict officiel: Non prioritaire — gaps importants
- Matching V3: rank `8`, score `0.4297`
- Random Forest: rank `8`, score `0.0000`
- XGBoost: rank `42`, score `0.0038`
- status: `review_needed`
- SHAP top features: final_score_v3, must_have_coverage, vector_similarity

## Note méthodologique

- Matching V3 reste la baseline officielle. Random Forest est le meilleur modèle ML actuel selon les métriques observées. XGBoost est conservé pour SHAP et l’analyse avancée. Les scores ML sont issus de modèles entraînés sur pseudo-labels métier contrôlés.
- Les Decision Cards officielles ne sont pas modifiées.
- Les datasets, modèles, FAISS et MongoDB ne sont pas modifiés.
