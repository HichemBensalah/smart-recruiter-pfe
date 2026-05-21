# `tests/`

Tests de non-régression du projet Smart Recruiter.

Les tests restent pour l'instant dans un dossier plat afin de ne pas casser la résolution pytest, les imports existants ni les commandes documentées. Une réorganisation par domaine pourra être faite après stabilisation Git du MVP.

## Tests rapides officiels

La suite rapide officielle est :

```bash
python scripts/run_fast_tests.py
```

Elle exécute les tests critiques pour le MVP démo :

- API FastAPI : `test_api_*.py`
- Chat et mémoire : `test_api_chat.py`, `test_chat_memory.py`
- LangChain Tools : `test_langchain_tools_*.py`
- LangGraph Copilot : `test_langgraph_copilot_*.py`
- Wizard offre : `test_job_intake*.py`
- Références candidat : `test_reference_resolver.py`
- Streamlit statique : `test_streamlit_app_static.py`
- Neo4j optionnel : `test_neo4j_transferability.py`, `test_api_graph_neo4j.py`
- Docker/config : `test_docker_configuration.py`

Cette suite doit rester rapide, locale et sans dépendance à Neo4j, MongoDB ou Docker réels.

## Tests complets

Collecte complète :

```bash
pytest --collect-only -q
```

Exécution complète :

```bash
pytest -q
```

La suite complète collecte plus de tests que le runner rapide. Elle couvre aussi les datasets, rapports, ranking ML, SHAP, XGBoost, génération de Decision Cards et démo end-to-end.

## Tests expérimentaux ou lourds

Les familles suivantes sont utiles, mais hors fast runner par défaut :

- Ranking ML : `test_ranking_*.py`, `test_train_ranking_models.py`, `test_ml_*`
- XGBoost / SHAP : `test_xgboost_primary_ranking.py`, `test_shap_explainability.py`
- Génération de rapports : `test_demo_*`, `test_decision_cards_*`, `test_candidate_corpus_analysis.py`
- Données et pseudo-labels : `test_pseudo_*`, `test_aligned_*`, `test_annotation_sample.py`
- Matching/report contracts : `test_matching_v3_report_contract.py`

Ces tests doivent être lancés avant une release importante ou dans une suite nightly, mais pas forcément à chaque itération rapide.

## Règles

- Ne pas supprimer un test actif sans identifier la fonctionnalité protégée.
- Ne pas déplacer les tests tant que les imports et scripts CI n'ont pas été adaptés.
- Garder `scripts/run_fast_tests.py` comme gate minimal avant toute branche entreprise.
- Marquer explicitement les futurs tests lourds avec `slow`, `integration`, `experimental` ou `requires_external` si nécessaire.
