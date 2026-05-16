# Évaluation du Recruiter Copilot

## Objectif

Vérifier la qualité des réponses, la cohérence avec les artefacts du pipeline et l'absence d'hallucination évidente.

## Architecture évaluée

- FastAPI `/api/chat` indirectement via les LangChain Tools
- LangGraph Recruiter Copilot déterministe
- Matching V3 comme baseline officielle
- Decision Cards, ML comparison, Potential Graph et Neo4j optionnel

## Score global

- Scénarios : `8`
- Réussis : `8`
- Faibles/échoués : `0`
- Score moyen : `1.0`

## Résultat par scénario

| Scénario | Intent | Score | Termes attendus trouvés | Candidats | Sources |
| --- | --- | ---: | ---: | ---: | --- |
| `scenario_01_backend_search` | `search_candidates` | 1.0000 | 3/3 | 5 | user_message, match_candidates, get_decision_card, get_transferability |
| `scenario_02_data_engineer_search` | `search_candidates` | 1.0000 | 2/2 | 5 | user_message, match_candidates, get_decision_card, get_transferability |
| `scenario_03_review_needed` | `review_needed` | 1.0000 | 2/2 | 5 | user_message, match_candidates, get_decision_card, get_transferability |
| `scenario_04_explain_first_candidate` | `explain_candidate` | 1.0000 | 2/2 | 5 | user_message, match_candidates, get_decision_card, get_transferability |
| `scenario_05_best_candidate_gaps` | `gap_analysis` | 1.0000 | 1/1 | 5 | user_message, match_candidates, get_decision_card, get_transferability |
| `scenario_06_backend_transferability` | `transferability` | 1.0000 | 2/2 | 5 | user_message, match_candidates, get_decision_card, get_transferability |
| `scenario_07_compare_first_two` | `compare_candidates` | 1.0000 | 1/1 | 5 | user_message, match_candidates, get_decision_card, get_transferability |
| `scenario_08_risk_or_disagreement` | `risk_review` | 1.0000 | 2/2 | 5 | user_message, match_candidates, get_decision_card, get_transferability |

## Observations

### scenario_01_backend_search - OK

- Message : Je cherche un développeur backend Python FastAPI MongoDB
- Score : `1.0`
- Termes manquants : `[]`
- Warnings : `[]`

### scenario_02_data_engineer_search - OK

- Message : Trouve-moi un profil Data Engineer Python SQL
- Score : `1.0`
- Termes manquants : `[]`
- Warnings : `[]`

### scenario_03_review_needed - OK

- Message : Quels candidats sont à vérifier ?
- Score : `1.0`
- Termes manquants : `[]`
- Warnings : `[]`

### scenario_04_explain_first_candidate - OK

- Message : Pourquoi le premier candidat est recommandé ?
- Score : `1.0`
- Termes manquants : `[]`
- Warnings : `[]`

### scenario_05_best_candidate_gaps - OK

- Message : Quels sont les gaps du meilleur candidat ?
- Score : `1.0`
- Termes manquants : `[]`
- Warnings : `[]`

### scenario_06_backend_transferability - OK

- Message : Est-ce que le meilleur candidat peut évoluer vers Backend Developer ?
- Score : `1.0`
- Termes manquants : `[]`
- Warnings : `[]`

### scenario_07_compare_first_two - OK

- Message : Compare les deux premiers candidats
- Score : `1.0`
- Termes manquants : `[]`
- Warnings : `[]`

### scenario_08_risk_or_disagreement - OK

- Message : Donne-moi les candidats avec risque ou désaccord
- Score : `1.0`
- Termes manquants : `[]`
- Warnings : `[]`

## Limites

- Pas de mémoire longue conversationnelle.
- Pas de planner LLM : le routage est déterministe.
- La qualité dépend des artefacts exposés par les tools FastAPI.
- La partie ML reste expérimentale car entraînée sur pseudo-labels métier contrôlés.

## Conclusion

Le Copilot est fonctionnel pour une démonstration contrôlée : il peut répondre à des demandes recruteur, récupérer les candidats via les tools, afficher les scores et expliquer les gaps. Les réponses restent basées sur les artefacts du pipeline. Les limites principales sont l'absence de mémoire longue, l'absence de validation recruteur réelle et la dépendance aux pseudo-labels pour la partie ML.
