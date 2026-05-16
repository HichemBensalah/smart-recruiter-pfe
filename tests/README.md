# `tests/`

Tests de non-regression du projet.

Les tests restent pour l'instant dans un dossier plat afin de ne pas casser la resolution pytest ni les commandes existantes. Une reorganisation par domaine pourra etre faite apres stabilisation des LangChain Tools.

## Categories

- API : `test_api_*.py`
- Demo : `test_demo_*.py`
- Graph : `test_transferability_score.py`, `test_neo4j_transferability.py`, `test_api_graph_neo4j.py`
- Ranking ML : `test_ranking_*.py`, `test_train_ranking_models.py`, `test_ml_*`
- Decision Cards : `test_decision_cards_*.py`
- Dataset / pseudo-labels : `test_pseudo_*`, `test_aligned_*`

## Commandes utiles

```bash
pytest tests/test_api_health.py tests/test_api_candidates.py tests/test_api_match.py tests/test_api_decision_cards.py tests/test_api_graph.py tests/test_api_demo.py -q
pytest tests/test_neo4j_transferability.py tests/test_api_graph_neo4j.py -q
pytest tests/test_demo_end_to_end.py tests/test_demo_summary.py tests/test_demo_executive_summary.py tests/test_transferability_score.py tests/test_decision_cards_with_transferability.py -q
```

Si Windows pose un probleme de cache/temporaire :

```bash
pytest -p no:cacheprovider --basetemp=.tmp/pytest_run_architecture
```
