# Decision Cards avec Potential Graph

## Objectif

Créer une version séparée des Decision Cards enrichie avec une analyse de transférabilité métier basée sur le Potential Graph YAML.

## Architecture

- Matching V3 reste la baseline officielle et le score de référence.
- Les scores ML sont conservés si les cartes d'entrée les contiennent.
- Potential Graph ajoute une lecture explicable: fit direct, transition plausible, gaps compensables et gaps bloquants.
- Aucun modèle n'est réentraîné et aucune carte officielle n'est modifiée.

## Synthèse

- `job_id`: `backend_python_django_postgresql`
- total cartes: `50`
- profils retrouvés: `50`
- profils non retrouvés: `0`
- taux de succès lookup: `100.00%`
- distribution lookup: `{'found_by_profile_id': 50}`

## Top cartes avec fit direct

| candidate_id | fit_direct | direct_fit_score | transferability_score | best_source_role | target_role | lookup |
| --- | --- | ---: | ---: | --- | --- | --- |
| Aucun | NA | NA | NA | NA | NA | NA |

## Top cartes avec transition plausible

| candidate_id | fit_direct | direct_fit_score | transferability_score | best_source_role | target_role | lookup |
| --- | --- | ---: | ---: | --- | --- | --- |
| `candidate_1487f3187f7b` | False | 0.6000 | 0.5333 | Machine Learning Engineer | Backend Developer | found_by_profile_id |
| `candidate_8eea1b635447` | False | 0.6000 | 0.5333 | Machine Learning Engineer | Backend Developer | found_by_profile_id |
| `candidate_56424ea73690` | False | 0.6000 | 0.3250 | Machine Learning Engineer | Backend Developer | found_by_profile_id |
| `candidate_206d746034ef` | False | 0.6000 | 0.5000 | Machine Learning Engineer | Backend Developer | found_by_profile_id |
| `candidate_073b7a3d39ba` | False | 0.4000 | 0.2917 | DevOps Engineer | Backend Developer | found_by_profile_id |
| `candidate_36923c8bf20e` | False | 0.4000 | 0.2167 | Machine Learning Engineer | Backend Developer | found_by_profile_id |
| `candidate_351493983fdf` | False | 0.4000 | 0.2167 | Machine Learning Engineer | Backend Developer | found_by_profile_id |
| `candidate_418b74b9d404` | False | 0.4000 | 0.2167 | Machine Learning Engineer | Backend Developer | found_by_profile_id |
| `candidate_d46244805ac2` | False | 0.4000 | 0.2167 | Machine Learning Engineer | Backend Developer | found_by_profile_id |
| `candidate_66466a398907` | False | 0.4000 | 0.2167 | Machine Learning Engineer | Backend Developer | found_by_profile_id |

## Exemples de gaps compensables

| candidate_id | gaps | explanation |
| --- | --- | --- |
| `candidate_b6f7add66ffc` | REST API, PostgreSQL | Transition depuis Data Analyst vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: SQL, Git. |
| `candidate_9b0508063f03` | REST API, Django, PostgreSQL | Transition depuis Data Analyst vers Backend Developer plausible sous réserve de validation humaine. Score de transférabilité: 0.30. |
| `candidate_1487f3187f7b` | REST API, Django | Transition depuis Machine Learning Engineer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: SQL. |
| `candidate_8eea1b635447` | REST API, Django | Transition depuis Machine Learning Engineer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: SQL. |
| `candidate_56424ea73690` | REST API, Django, PostgreSQL | Transition depuis Machine Learning Engineer vers Backend Developer plausible sous réserve de validation humaine. Score de transférabilité: 0.32. |
| `candidate_206d746034ef` | REST API, Django, PostgreSQL | Transition depuis Machine Learning Engineer vers Backend Developer plausible sous réserve de validation humaine. Score de transférabilité: 0.50. |
| `candidate_e1fda1d70be1` | REST API, Django, PostgreSQL | Transition depuis Data Engineer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: Python, Git. |
| `candidate_073b7a3d39ba` | REST API, Django, PostgreSQL | Transition depuis DevOps Engineer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: Git. |
| `candidate_4eef95d3a3fa` | REST API, Django, PostgreSQL | Transition depuis Backend Developer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: Git. |
| `candidate_664415ab2fe1` | REST API, Django, PostgreSQL | Transition depuis Backend Developer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: Git. |

## Exemples de gaps bloquants

| candidate_id | gaps | explanation |
| --- | --- | --- |
| `candidate_b6f7add66ffc` | SQL, Git | Transition depuis Data Analyst vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: SQL, Git. |
| `candidate_1487f3187f7b` | SQL | Transition depuis Machine Learning Engineer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: SQL. |
| `candidate_8eea1b635447` | SQL | Transition depuis Machine Learning Engineer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: SQL. |
| `candidate_e1fda1d70be1` | Python, Git | Transition depuis Data Engineer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: Python, Git. |
| `candidate_073b7a3d39ba` | Git | Transition depuis DevOps Engineer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: Git. |
| `candidate_4eef95d3a3fa` | Git | Transition depuis Backend Developer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: Git. |
| `candidate_664415ab2fe1` | Git | Transition depuis Backend Developer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: Git. |
| `candidate_36923c8bf20e` | Git | Transition depuis Machine Learning Engineer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: Git. |
| `candidate_351493983fdf` | Git | Transition depuis Machine Learning Engineer vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: Git. |
| `candidate_851fca236e70` | Git | Transition depuis Data Analyst vers Backend Developer partiellement plausible, mais des gaps bloquants restent présents: Git. |

## Exemples de profils non retrouvés

| candidate_id | profile_id | baseline_rank_v3 |
| --- | --- | ---: |
| Aucun | NA | NA |

## Limites méthodologiques

- Potential Graph est une couche explicative déclarative basée sur les skills structurées et un graphe YAML de rôles. Il ne remplace pas Matching V3, ne remplace pas les modèles ML et ne constitue pas une décision recruteur. Il sert à analyser la transférabilité métier et les gaps.
- Les résultats dépendent de la qualité des skills structurées par le Module 2.
- Les transitions sont déclaratives et doivent être validées par un recruteur ou expert métier.
