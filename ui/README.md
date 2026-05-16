# Interface Streamlit - Smart Recruiter Copilot RH

Cette interface est une demo simple du Recruiter Copilot. Elle appelle uniquement l'endpoint FastAPI `POST /api/chat`.

## Lancer l'API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Swagger :

```text
http://localhost:8000/docs
```

## Lancer Streamlit

```bash
streamlit run ui/streamlit_app.py
```

## Tester une requete

Exemple :

```text
Je cherche un développeur backend Python FastAPI MongoDB
```

L'interface affiche :

- la reponse conversationnelle du Copilot ;
- les candidats retournes ;
- les scores Matching V3, Random Forest et XGBoost si disponibles ;
- les Decision Cards si presentes ;
- la transferabilite et les gaps si presents ;
- les warnings eventuels.

## Limites actuelles

- Pas encore de memoire longue.
- Pas encore de planner LLM.
- Neo4j est optionnel ; le workflow garde le fallback YAML.
- Les reponses sont basees sur le workflow LangGraph et les tools exposes par l'API.
- Streamlit ne lance aucun script et n'appelle pas Matching V3 directement.
