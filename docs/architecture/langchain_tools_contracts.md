# LangChain Tools Contracts

## Objectif

Cette couche expose les capacites Smart Recruiter sous forme de LangChain `StructuredTool`. Elle prepare le futur Recruiter Copilot LangGraph sans creer encore de workflow agentique.

Les tools appellent uniquement l'API FastAPI existante. Ils ne modifient pas Matching V3, FAISS, MongoDB, les datasets ou les modeles ML.

## Configuration

Variable optionnelle :

```bash
SMART_RECRUITER_API_BASE_URL=http://localhost:8000
```

Valeur par defaut : `http://localhost:8000`.

## Tools disponibles

| Tool | Endpoint FastAPI | Entree | Sortie |
|---|---|---|---|
| `match_candidates` | `POST /api/match` | `job_description`, `top_k` | Top candidats avec scores fournis par l'API |
| `get_candidate_profile` | `GET /api/candidates/{candidate_id}` | `candidate_id` | Profil candidat et donnees disponibles |
| `get_decision_card` | `GET /api/decision-cards/{candidate_id}` | `candidate_id` | Decision Card disponible |
| `get_transferability` | `GET /api/graph/transferability/{candidate_id}` | `candidate_id` | Transferability YAML/fallback |
| `get_neo4j_transferability` | `GET /api/graph/neo4j/transferability/{candidate_id}` | `candidate_id`, `target_role` | Explication Neo4j si disponible |
| `get_neo4j_gaps` | `GET /api/graph/neo4j/gaps/{candidate_id}` | `candidate_id`, `target_role` | Gaps compensables et bloquants Neo4j |
| `get_demo_executive_summary` | `GET /api/demo/executive-summary` | aucun | Synthese executive demo |
| `get_demo_top10_summary` | `GET /api/demo/top10-summary` | aucun | Synthese detaillee top 10 |
| `run_demo` | `POST /api/demo/run` | aucun | Manifest d'execution demo |

## Registry

Les tools sont centralises dans :

```python
from src.core.chatbot.tools.registry import get_smart_recruiter_tools

tools = get_smart_recruiter_tools()
```

## Usage futur LangGraph

Ces tools peuvent devenir des nodes ou tools d'un workflow LangGraph :

- `match_candidates` pour identifier les candidats pertinents.
- `get_decision_card` pour expliquer un candidat.
- `get_transferability` pour analyser les gaps metier via YAML.
- `get_neo4j_transferability` et `get_neo4j_gaps` pour enrichir avec Graph-RAG si Neo4j est disponible.
- `get_demo_*` pour repondre aux questions de demonstration.

## Comportement si API indisponible

Le client leve une erreur controlee `SmartRecruiterApiError`. Dans un futur agent, cette erreur devra etre transformee en message utilisateur clair.

## Comportement si Neo4j est indisponible

Les tools Neo4j retournent une reponse explicite :

```json
{
  "available": false,
  "message": "...",
  "fallback_recommended": true
}
```

Le fallback recommande est `get_transferability`, base sur le Potential Graph YAML.

## Limites

- Les tools ne doivent pas inventer de candidats, scores ou explications.
- Matching V3 reste la baseline officielle.
- Les scores ML restent experimentaux car entraines sur pseudo-labels metier controles.
- Neo4j est optionnel et ne remplace pas le YAML.
- LangGraph n'est pas encore cree a cette etape.
