# Évaluation du Recruiter Copilot

## Objectif

Vérifier le flow recruteur complet, la mémoire courte, le matching via `routed_job_id`, les questions de suivi et les fallbacks sans dépendre de Neo4j, MongoDB ou Docker réels.

## Méthode

- Exécution via `fastapi.testclient.TestClient`.
- Tools LangGraph bridgés vers les routes FastAPI locales.
- Variables Neo4j retirées pendant l'évaluation pour forcer le fallback YAML.
- Matching V3 évalué via les artefacts pré-générés `data/ranking/features/*.jsonl`.

## Métriques globales

- Scénarios : `14`
- Réussis : `14`
- Échoués/faibles : `0`
- Score moyen : `1.0`
- Tool calling accuracy : `1.0`
- Taux de réponses sans hallucination : `1.0`
- Latence moyenne `/api/chat` : `24.04 ms`
- P95 `/api/chat` : `196.24 ms`
- Cohérence mémoire : `1.0`
- Couverture des scénarios : `1.0`
- Couverture fallback : `1.0`

## Couverture

- best_candidate_gaps: OK
- compare_first_two: OK
- dependency_or_artifact_fallback: OK
- followup_explain_first_candidate: OK
- matching_with_routed_job_id: OK
- neo4j_yaml_fallback: OK
- new_offer_start: OK
- positive_confirmation: OK
- pre_confirmation_correction: OK
- six_field_collection: OK

## Résultat par scénario

| Scénario | Type | Score | Statut | Latence ms | Couverture | Sources |
| --- | --- | ---: | --- | ---: | --- | --- |
| `01_start_new_offer` | `chat` | 1.0000 | OK | 10.94 | new_offer_start | job_intake |
| `02_collect_job_title` | `chat` | 1.0000 | OK | 3.96 | six_field_collection | job_intake |
| `03_collect_about_role` | `chat` | 1.0000 | OK | 3.82 | six_field_collection | job_intake |
| `04_collect_responsibilities` | `chat` | 1.0000 | OK | 3.88 | six_field_collection | job_intake |
| `05_collect_required_skills` | `chat` | 1.0000 | OK | 3.95 | six_field_collection | job_intake |
| `06_collect_bonus_skills` | `chat` | 1.0000 | OK | 3.66 | six_field_collection | job_intake |
| `07_collect_profile_and_route` | `chat` | 1.0000 | OK | 4.35 | six_field_collection, matching_with_routed_job_id | job_intake, job_router |
| `08_edit_before_confirmation` | `chat` | 1.0000 | OK | 4.22 | pre_confirmation_correction, matching_with_routed_job_id | job_intake, job_router |
| `09_confirm_matching` | `chat` | 1.0000 | OK | 196.24 | positive_confirmation, matching_with_routed_job_id | user_message, match_candidates, get_decision_card, get_transferability, job_intake, job_router |
| `10_followup_explain_first` | `chat` | 1.0000 | OK | 17.48 | followup_explain_first_candidate, memory_coherence | conversation_memory, user_message |
| `11_followup_compare_first_two` | `chat` | 1.0000 | OK | 17.25 | compare_first_two, memory_coherence | conversation_memory, user_message |
| `12_followup_best_candidate_gaps` | `chat` | 1.0000 | OK | 18.75 | best_candidate_gaps, memory_coherence | conversation_memory, user_message |
| `13_graph_yaml_fallback_without_neo4j` | `endpoint` | 1.0000 | OK | 5.86 | neo4j_yaml_fallback, dependency_or_artifact_fallback | n/a |
| `14_match_unknown_job_id_fallback` | `endpoint` | 1.0000 | OK | 7.06 | dependency_or_artifact_fallback | n/a |

## Warnings et observations

### 01_start_new_offer

- Message/action : nouvelle offre
- Candidats structurés : `[]`
- Termes manquants : `[]`
- Warnings : `[]`

### 02_collect_job_title

- Message/action : Développeur Backend Python FastAPI MongoDB
- Candidats structurés : `[]`
- Termes manquants : `[]`
- Warnings : `[]`

### 03_collect_about_role

- Message/action : Construire et maintenir des APIs backend pour une plateforme RH.
- Candidats structurés : `[]`
- Termes manquants : `[]`
- Warnings : `[]`

### 04_collect_responsibilities

- Message/action : Développer des endpoints FastAPI, intégrer MongoDB, écrire des tests et documenter les APIs.
- Candidats structurés : `[]`
- Termes manquants : `[]`
- Warnings : `[]`

### 05_collect_required_skills

- Message/action : Python, FastAPI, MongoDB, Docker, Git
- Candidats structurés : `[]`
- Termes manquants : `[]`
- Warnings : `[]`

### 06_collect_bonus_skills

- Message/action : Neo4j, LangChain, Streamlit, CI GitHub Actions
- Candidats structurés : `[]`
- Termes manquants : `[]`
- Warnings : `[]`

### 07_collect_profile_and_route

- Message/action : Profil confirmé senior 3 ans minimum, autonome, à Tunis ou remote hybride.
- Candidats structurés : `[]`
- Termes manquants : `[]`
- Warnings : `[]`

### 08_edit_before_confirmation

- Message/action : change les competences obligatoires en Python, FastAPI, MongoDB, Docker
- Candidats structurés : `[]`
- Termes manquants : `[]`
- Warnings : `[]`

### 09_confirm_matching

- Message/action : oui lance la recherche
- Candidats structurés : `['candidate_1487f3187f7b', 'candidate_8eea1b635447', 'candidate_56424ea73690', 'candidate_9b0508063f03', 'candidate_206d746034ef']`
- Termes manquants : `[]`
- Warnings : `[]`

### 10_followup_explain_first

- Message/action : Pourquoi le premier candidat ?
- Candidats structurés : `['candidate_1487f3187f7b', 'candidate_8eea1b635447', 'candidate_56424ea73690', 'candidate_9b0508063f03', 'candidate_206d746034ef']`
- Termes manquants : `[]`
- Warnings : `[]`

### 11_followup_compare_first_two

- Message/action : Compare le premier et le deuxième candidat
- Candidats structurés : `['candidate_1487f3187f7b', 'candidate_8eea1b635447', 'candidate_56424ea73690', 'candidate_9b0508063f03', 'candidate_206d746034ef']`
- Termes manquants : `[]`
- Warnings : `[]`

### 12_followup_best_candidate_gaps

- Message/action : Quels sont les gaps du meilleur candidat ?
- Candidats structurés : `['candidate_1487f3187f7b', 'candidate_8eea1b635447', 'candidate_56424ea73690', 'candidate_9b0508063f03', 'candidate_206d746034ef']`
- Termes manquants : `[]`
- Warnings : `[]`

### 13_graph_yaml_fallback_without_neo4j

- Message/action : GET /api/graph/transferability/candidate_1487f3187f7b
- Candidats structurés : `[]`
- Termes manquants : `[]`
- Warnings : `['Neo4j unavailable, YAML fallback used: Neo4j is not configured. Missing environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD']`

### 14_match_unknown_job_id_fallback

- Message/action : POST /api/match
- Candidats structurés : `['candidate_b6f7add66ffc']`
- Termes manquants : `[]`
- Warnings : `["Matching artifact not found for job_id 'unknown_job_for_eval'. Fallback used: 'backend_python_django_postgresql'."]`

## Limites connues

- Mémoire courte en RAM avec TTL : ce n'est pas une mémoire longue ou multi-utilisateur persistée.
- Matching V3 est évalué via des artefacts pré-générés, pas via un recalcul live FAISS/MongoDB.
- Neo4j est volontairement optionnel dans cette évaluation ; le fallback YAML est le comportement attendu en CI.
- Les scores Random Forest et XGBoost restent expérimentaux car entraînés sur pseudo-labels métier.

## Conclusion

Le Copilot est démontrable de bout en bout sur un flow recruteur contrôlé : création d'offre, correction avant confirmation, matching via routed_job_id, mémoire courte, questions de suivi, Decision Cards et fallback YAML lorsque Neo4j n'est pas disponible. L'évaluation reste volontairement rapide et locale pour ne pas rendre la CI dépendante de services externes.
