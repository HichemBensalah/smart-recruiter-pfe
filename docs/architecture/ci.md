# CI rapide GitHub Actions

## Objectif

La CI verifie les briques rapides et stables du projet Smart Recruiter sans lancer les pipelines lourds, sans reentrainer de modele et sans dependre de services externes.

Elle sert de garde-fou pour :

- API FastAPI ;
- endpoint `/api/chat` ;
- LangChain Tools ;
- LangGraph Recruiter Copilot ;
- Streamlit en verification statique ;
- Neo4j fallback sans serveur reel ;
- configuration Docker.

## Pourquoi tests rapides seulement

Le projet contient des briques lourdes : OCR/Docling, MongoDB, FAISS, entrainement ML, SHAP, generation de datasets et imports Neo4j. Ces briques sont importantes, mais elles ne doivent pas bloquer chaque push.

La CI rapide valide les contrats applicatifs sans relancer les traitements couteux.

## Tests inclus

- `tests/test_api_health.py`
- `tests/test_api_chat.py`
- `tests/test_api_candidates.py`
- `tests/test_api_match.py`
- `tests/test_api_decision_cards.py`
- `tests/test_api_graph.py`
- `tests/test_api_demo.py`
- `tests/test_langchain_tools_api_client.py`
- `tests/test_langchain_tools_registry.py`
- `tests/test_langchain_tools_contracts.py`
- `tests/test_langgraph_copilot_state.py`
- `tests/test_langgraph_copilot_nodes.py`
- `tests/test_langgraph_copilot_graph.py`
- `tests/test_streamlit_app_static.py`
- `tests/test_neo4j_transferability.py`
- `tests/test_api_graph_neo4j.py`
- `tests/test_docker_configuration.py`

## Tests exclus

Sont exclus de cette premiere CI :

- tests OCR / Docling ;
- tests necessitant MongoDB reel ;
- tests necessitant Neo4j reel ;
- tests de reentrainement ML ;
- tests SHAP lourds ;
- tests FAISS avec indexation complete ;
- tests Streamlit serveur ;
- tests dependants d'une API LLM externe.

## Services externes

La CI ne lance pas :

- MongoDB ;
- Neo4j ;
- Streamlit ;
- serveur FastAPI reel.

Les tests API utilisent `TestClient` ou des mocks. Les tests Neo4j valident le comportement degrade quand Neo4j n'est pas configure.

## Lancer localement la meme suite

```bash
python scripts/run_fast_tests.py
```

Commande equivalente :

```bash
pytest tests/test_api_health.py tests/test_api_chat.py tests/test_api_candidates.py tests/test_api_match.py tests/test_api_decision_cards.py tests/test_api_graph.py tests/test_api_demo.py tests/test_langchain_tools_api_client.py tests/test_langchain_tools_registry.py tests/test_langchain_tools_contracts.py tests/test_langgraph_copilot_state.py tests/test_langgraph_copilot_nodes.py tests/test_langgraph_copilot_graph.py tests/test_streamlit_app_static.py tests/test_neo4j_transferability.py tests/test_api_graph_neo4j.py tests/test_docker_configuration.py -q -p no:cacheprovider --basetemp=.tmp/pytest_ci_fast
```

## Limites

- Cette CI ne prouve pas que les pipelines lourds fonctionnent.
- Elle ne valide pas un serveur MongoDB, Neo4j ou FAISS reconstruit.
- Elle ne valide pas la qualite recruteur reelle.
- Elle ne remplace pas une future CI integration plus complete.
