# Decision Cards enrichies avec ML expérimental

## Objectif

Ajouter les annotations du re-ranking XGBoost expérimental aux Decision Cards, sans modifier les cartes officielles et sans remplacer Matching V3.

## Statut

- cartes enrichies: `10`
- cartes trouvées dans le rapport ML: `9`
- cartes non trouvées dans le rapport ML: `1`
- score principal conservé: `final_score`
- score ML expérimental: `experimental_ml_score`

## Top 10 Matching V3 avec annotations ML

| rank_v3 | candidate_id | final_score | experimental_ml_score | ml_rank | rank_shift | verdict |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `candidate_1487f3187f7b` | 0.7702 | 0.9718 | 4 | -3 | Présélection recommandée |
| 2 | `candidate_8eea1b635447` | 0.6709 | 0.9676 | 5 | -3 | Présélection recommandée |
| 3 | `candidate_206d746034ef` | 0.5546 | 0.9209 | 6 | -3 | À considérer avec vérification |
| 4 | `candidate_1df043636ea4` | 0.4347 | 0.0037 | 49 | -45 | Non prioritaire — gaps importants |
| 5 | `candidate_073b7a3d39ba` | 0.4192 | 0.0038 | 42 | -37 | Non prioritaire — gaps importants |
| 6 | `candidate_71e03ea99985` | 0.3527 | 0.0038 | 45 | -39 | Non prioritaire — gaps importants |
| 7 | `candidate_c564b8eceb3d` | 0.3505 | 0.0038 | 44 | -37 | Non prioritaire — gaps importants |
| 8 | `candidate_1d475044c93c` | 0.3476 | NA | NA | NA | Non prioritaire — gaps importants |
| 9 | `candidate_e1fda1d70be1` | 0.2645 | 0.0042 | 24 | -15 | Non prioritaire — gaps importants |
| 10 | `candidate_4eef95d3a3fa` | 0.2628 | 0.0040 | 38 | -28 | Non prioritaire — gaps importants |

## Top XGBoost expérimental parmi les Decision Cards

| ml_rank | candidate_id | experimental_ml_score | rank_v3 | final_score | rank_shift | verdict |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 4 | `candidate_1487f3187f7b` | 0.9718 | 1 | 0.7702 | -3 | Présélection recommandée |
| 5 | `candidate_8eea1b635447` | 0.9676 | 2 | 0.6709 | -3 | Présélection recommandée |
| 6 | `candidate_206d746034ef` | 0.9209 | 3 | 0.5546 | -3 | À considérer avec vérification |
| 24 | `candidate_e1fda1d70be1` | 0.0042 | 9 | 0.2645 | -15 | Non prioritaire — gaps importants |
| 38 | `candidate_4eef95d3a3fa` | 0.0040 | 10 | 0.2628 | -28 | Non prioritaire — gaps importants |
| 42 | `candidate_073b7a3d39ba` | 0.0038 | 5 | 0.4192 | -37 | Non prioritaire — gaps importants |
| 44 | `candidate_c564b8eceb3d` | 0.0038 | 7 | 0.3505 | -37 | Non prioritaire — gaps importants |
| 45 | `candidate_71e03ea99985` | 0.0038 | 6 | 0.3527 | -39 | Non prioritaire — gaps importants |
| 49 | `candidate_1df043636ea4` | 0.0037 | 4 | 0.4347 | -45 | Non prioritaire — gaps importants |

## Changements de rang les plus visibles

| candidate_id | rank_v3 | ml_rank | rank_shift | score_delta |
| --- | ---: | ---: | ---: | ---: |
| `candidate_1df043636ea4` | 4 | 49 | -45 | -0.2711 |
| `candidate_71e03ea99985` | 6 | 45 | -39 | -0.2193 |
| `candidate_073b7a3d39ba` | 5 | 42 | -37 | -0.4259 |
| `candidate_c564b8eceb3d` | 7 | 44 | -37 | -0.2197 |
| `candidate_4eef95d3a3fa` | 10 | 38 | -28 | -0.4244 |
| `candidate_e1fda1d70be1` | 9 | 24 | -15 | -0.4307 |
| `candidate_1487f3187f7b` | 1 | 4 | -3 | 0.2535 |
| `candidate_8eea1b635447` | 2 | 5 | -3 | 0.2988 |
| `candidate_206d746034ef` | 3 | 6 | -3 | 0.3827 |

## Cartes sans annotation ML

- `candidate_1d475044c93c`: absent du rapport ML re-ranking fourni.

## Limites méthodologiques

- Le score ML est expérimental et entraîné sur pseudo-labels métier contrôlés. Matching V3 reste la baseline officielle.
- Matching V3 reste le ranking principal et la baseline officielle.
- `experimental_ml_score` n'est pas un score recruteur final.
- Cette sortie est une version expérimentale séparée, pas une modification des Decision Cards officielles.
