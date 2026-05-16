# Chat API - Recruiter Copilot

## Objectif

`POST /api/chat` expose le workflow LangGraph Recruiter Copilot via FastAPI. Il permet a une interface future, par exemple Streamlit ou web, d'envoyer une demande recruteur et de recevoir une reponse conversationnelle structuree.

L'endpoint utilise :

```python
from src.core.chatbot.graph import run_recruiter_copilot
```

## Lien avec LangGraph

Le workflow LangGraph execute les nodes :

```text
understand_query
-> match_candidates
-> fetch_decision_cards
-> analyze_transferability
-> compose_answer
```

Cette version est deterministe : pas de LLM planner, pas de memoire longue, pas d'appel externe non controle.

## Schema d'entree

```json
{
  "message": "Je cherche un développeur backend Python FastAPI MongoDB",
  "session_id": "optional-session-id"
}
```

`session_id` est accepte pour preparer la suite, mais aucune memoire conversationnelle n'est encore activee.

## Schema de sortie

```json
{
  "answer": "...",
  "candidates": [],
  "decision_cards": [],
  "transferability": {},
  "sources": [],
  "warnings": []
}
```

## Exemple de requete

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"Je cherche un développeur backend Python FastAPI MongoDB\"}"
```

## Exemple de reponse

```json
{
  "answer": "J'ai analyse la demande recruteur...",
  "candidates": [
    {
      "candidate_id": "candidate_1",
      "baseline_score_v3": 0.82
    }
  ],
  "decision_cards": [],
  "transferability": {},
  "sources": ["user_message", "match_candidates"],
  "warnings": []
}
```

## Limites actuelles

- Pas encore de memoire longue.
- Pas encore de LLM planner.
- La reponse est basee sur les tools et artefacts existants.
- Neo4j reste optionnel avec fallback YAML.
- Matching V3 reste la baseline officielle.
