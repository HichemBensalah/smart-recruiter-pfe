# LangGraph Recruiter Copilot Workflow

## Objectif

Ce workflow cree une premiere version deterministe du Recruiter Copilot Smart Recruiter. Il orchestre les LangChain Tools existants sans appeler de LLM et sans modifier Matching V3, FAISS, MongoDB, ML, Neo4j ou les datasets.

## Pourquoi LangGraph

LangGraph permet de rendre explicite le chemin de decision du copilot :

```text
START
  -> understand_query
  -> match_candidates
  -> fetch_decision_cards
  -> analyze_transferability
  -> compose_answer
END
```

Chaque node a une responsabilite unique et peut etre teste sans serveur FastAPI reel grace au mocking des tools.

## Etat du workflow

L'etat `RecruiterCopilotState` contient :

- `user_message`
- `job_description`
- `top_k`
- `target_role`
- `candidates`
- `decision_cards`
- `transferability`
- `neo4j_available`
- `answer`
- `sources`
- `warnings`

## Nodes

### understand_query

Analyse deterministe du message recruteur :

- `backend` -> `Backend Developer`
- `data engineer` -> `Data Engineer`
- `data analyst` -> `Data Analyst`
- sinon `Backend Developer`

Le node fixe `job_description = user_message` et `top_k = 5` par defaut.

### match_candidates

Appelle le tool `match_candidates`, qui utilise `POST /api/match`.

### fetch_decision_cards

Appelle `get_decision_card` sur les meilleurs candidats.

### analyze_transferability

Appelle :

- `get_transferability` pour le fallback YAML ;
- `get_neo4j_transferability` si disponible.

Si Neo4j est indisponible, le workflow conserve le fallback YAML.

### compose_answer

Construit une reponse naturelle en francais uniquement avec les donnees presentes dans l'etat :

- candidats retournes ;
- scores fournis ;
- Decision Cards disponibles ;
- transferabilite/gaps disponibles.

Le node ne doit pas inventer de candidat, score, skill ou gap.

## Tools utilises

- `match_candidates`
- `get_decision_card`
- `get_transferability`
- `get_neo4j_transferability`

## Limites

- Pas de LLM planner dans cette version.
- Pas de memoire conversationnelle.
- Pas de Streamlit.
- Pas d'appel direct a MongoDB, FAISS ou Matching V3.
- Les reponses dependent des tools et artefacts exposes par l'API.

## Evolution suivante

1. Ajouter un endpoint FastAPI `POST /api/chat`.
2. Ajouter un planner LLM contraint.
3. Ajouter une memoire courte de conversation.
4. Ajouter une interface Streamlit ou web.
