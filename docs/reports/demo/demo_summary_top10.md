# Synthèse démo top 10 - Smart Recruiter

## Objectif

Présenter une vue unique et lisible du top 10 candidats en agrégeant Matching V3, Random Forest, XGBoost + SHAP et Potential Graph.

## Architecture utilisée

- Matching V3 : baseline officielle et score principal de référence.
- Random Forest : meilleur modèle ML actuel selon les métriques observées.
- XGBoost + SHAP : modèle avancé conservé pour l'analyse et l'explicabilité.
- Potential Graph : analyse déclarative de transférabilité métier et des gaps.

## Résumé

- `job_id`: `backend_python_django_postgresql`
- top_k: `10`
- total cartes source: `50`
- lookup success rate: `1.0`
- fit direct dans le top 10: `0`
- transitions plausibles dans le top 10: `5`
- review_needed dans le top 10: `5`
- transferability moyenne: `0.3075`

## Tableau top 10 candidats

| V3 rank | candidate_id | V3 score | RF rank | RF score | XGB rank | XGB score | transferability | status | synthèse |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `candidate_b6f7add66ffc` | 0.7754 | 5 | 0.9950 | 1 | 0.9826 | 0.2000 | review_needed | Candidat à vérifier : désaccord important entre XGBoost et les autres scores. |
| 2 | `candidate_9b0508063f03` | 0.7682 | 1 | 1.0000 | 2 | 0.9826 | 0.3000 | agreement_high | Candidat fortement recommandé : bon score V3, score RF élevé, score XGBoost élevé. |
| 3 | `candidate_1487f3187f7b` | 0.7183 | 2 | 1.0000 | 4 | 0.9718 | 0.5333 | agreement_high | Candidat fortement recommandé : bon score V3, score RF élevé, score XGBoost élevé. |
| 4 | `candidate_8eea1b635447` | 0.6688 | 3 | 1.0000 | 5 | 0.9676 | 0.5333 | agreement_high | Candidat fortement recommandé : bon score V3, score RF élevé, score XGBoost élevé. |
| 5 | `candidate_56424ea73690` | 0.6084 | 4 | 1.0000 | 3 | 0.9826 | 0.3250 | agreement_high | Candidat fortement recommandé : bon score V3, score RF élevé, score XGBoost élevé. |
| 6 | `candidate_206d746034ef` | 0.5382 | 6 | 0.9950 | 6 | 0.9209 | 0.5000 | agreement_high | Candidat partiellement transférable : fit direct faible mais transition plausible. |
| 7 | `candidate_e1fda1d70be1` | 0.4349 | 7 | 0.0000 | 24 | 0.0042 | 0.0583 | review_needed | Candidat à vérifier : désaccord important entre XGBoost et les autres scores. |
| 8 | `candidate_073b7a3d39ba` | 0.4297 | 8 | 0.0000 | 42 | 0.0038 | 0.2917 | review_needed | Candidat à vérifier : désaccord important entre XGBoost et les autres scores. |
| 9 | `candidate_4eef95d3a3fa` | 0.4284 | 9 | 0.0000 | 38 | 0.0040 | 0.1667 | review_needed | Candidat à vérifier : désaccord important entre XGBoost et les autres scores. |
| 10 | `candidate_664415ab2fe1` | 0.4256 | 10 | 0.0000 | 48 | 0.0037 | 0.1667 | review_needed | Candidat à vérifier : désaccord important entre XGBoost et les autres scores. |

## Candidats à vérifier

- `candidate_b6f7add66ffc` : Candidat à vérifier : désaccord important entre XGBoost et les autres scores. SHAP: final_score_v3, must_have_coverage, vector_similarity; gaps bloquants: SQL, Git; transition: pas de transition explicite.
- `candidate_e1fda1d70be1` : Candidat à vérifier : désaccord important entre XGBoost et les autres scores. SHAP: final_score_v3, must_have_coverage, vector_similarity; gaps bloquants: Python, Git; transition: pas de transition explicite.
- `candidate_073b7a3d39ba` : Candidat à vérifier : désaccord important entre XGBoost et les autres scores. SHAP: final_score_v3, must_have_coverage, vector_similarity; gaps bloquants: Git; transition: Un profil DevOps peut évoluer vers backend s'il combine automatisation, API et Python..
- `candidate_4eef95d3a3fa` : Candidat à vérifier : désaccord important entre XGBoost et les autres scores. SHAP: final_score_v3, must_have_coverage, vector_similarity; gaps bloquants: Git; transition: pas de transition explicite.
- `candidate_664415ab2fe1` : Candidat à vérifier : désaccord important entre XGBoost et les autres scores. SHAP: final_score_v3, must_have_coverage, vector_similarity; gaps bloquants: Git; transition: pas de transition explicite.

## Transférabilité

- `candidate_1487f3187f7b` : Candidat fortement recommandé : bon score V3, score RF élevé, score XGBoost élevé. SHAP: final_score_v3, must_have_coverage, vector_similarity; gaps bloquants: SQL; transition: Un profil ML peut évoluer vers backend Python s'il sait exposer des modèles via API..
- `candidate_8eea1b635447` : Candidat fortement recommandé : bon score V3, score RF élevé, score XGBoost élevé. SHAP: final_score_v3, must_have_coverage, vector_similarity; gaps bloquants: SQL; transition: Un profil ML peut évoluer vers backend Python s'il sait exposer des modèles via API..
- `candidate_56424ea73690` : Candidat fortement recommandé : bon score V3, score RF élevé, score XGBoost élevé. SHAP: final_score_v3, must_have_coverage, vector_similarity; gaps bloquants: aucun gap bloquant; transition: Un profil ML peut évoluer vers backend Python s'il sait exposer des modèles via API..
- `candidate_206d746034ef` : Candidat partiellement transférable : fit direct faible mais transition plausible. SHAP: final_score_v3, must_have_coverage, vector_similarity; gaps bloquants: aucun gap bloquant; transition: Un profil ML peut évoluer vers backend Python s'il sait exposer des modèles via API..
- `candidate_073b7a3d39ba` : Candidat à vérifier : désaccord important entre XGBoost et les autres scores. SHAP: final_score_v3, must_have_coverage, vector_similarity; gaps bloquants: Git; transition: Un profil DevOps peut évoluer vers backend s'il combine automatisation, API et Python..

## Limites méthodologiques

- Cette synthèse agrège plusieurs couches : Matching V3, modèles ML entraînés sur pseudo-labels métier contrôlés, SHAP et Potential Graph. Matching V3 reste la baseline officielle, Random Forest est le meilleur modèle ML actuel selon les métriques observées, XGBoost est utilisé pour l’analyse SHAP, et Potential Graph sert à analyser la transférabilité métier.
- Les scores ML ne sont pas des scores recruteur finaux.
- Potential Graph dépend des skills structurées disponibles et d'un graphe YAML déclaratif.
- Cette synthèse est une aide de démo, pas une décision automatique.
