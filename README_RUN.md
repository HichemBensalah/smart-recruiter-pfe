# Lancer Smart Recruiter

Ce guide resume les commandes de lancement du MVP Smart Recruiter pour une demonstration locale ou Docker.

## Option A - Lancement local

Installer les dependances :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Lancer FastAPI :

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8010
```

Lancer Streamlit dans un deuxieme terminal :

```bash
streamlit run ui/streamlit_app.py
```

Dans la sidebar Streamlit, utiliser :

```text
http://127.0.0.1:8010
```

## Option B - Docker Compose

```bash
docker compose up --build
```

URLs :

- API Swagger : http://localhost:8000/docs
- API health : http://localhost:8000/health
- Streamlit : http://localhost:8501
- Neo4j Browser : http://localhost:7474
- MongoDB : `mongodb://localhost:27017`

## Backend donnees E1

Le backend de donnees est pilote par :

```bash
DATA_BACKEND=artifacts|mongodb|hybrid
ALLOW_ARTIFACT_FALLBACK=true|false
MATCHING_MODE=artifact|live|hybrid
LIVE_MATCHING_TOP_N=50
LIVE_MATCHING_TOP_K=5
FAISS_INDEX_PATH=data/indexes/faiss/cv_index.faiss
FAISS_ID_MAP_PATH=data/indexes/faiss/id_map.pkl
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=talent_intelligence
```

Par defaut, `DATA_BACKEND=artifacts` conserve le comportement MVP. Avec `mongodb`, `/api/candidates` et `/api/decision-cards` lisent MongoDB. Si MongoDB est indisponible et `ALLOW_ARTIFACT_FALLBACK=true`, les routes retombent sur les artefacts avec `fallback_used=true` et un warning. Avec `hybrid`, MongoDB est prioritaire et les artefacts restent fallback de demo.

`MATCHING_MODE` pilote uniquement `/api/match` :

- `artifact` : mode MVP stable, lecture des artefacts `data/ranking/features/{job_id}.jsonl`.
- `live` : retrieval FAISS depuis `FAISS_INDEX_PATH` / `FAISS_ID_MAP_PATH`, profils depuis MongoDB, scoring officiel Matching V3 via `score_candidate()`.
- `hybrid` : essaye le live, puis fallback artefacts si `ALLOW_ARTIFACT_FALLBACK=true`.

Matching V3 reste le scoring officiel. RF / XGBoost / SHAP restent experimentaux et ne sont pas relances dans le matching live E2.

Seed initial depuis les artefacts :

```bash
python scripts/seed_mongodb_from_artifacts.py --dry-run
python scripts/seed_mongodb_from_artifacts.py
```

Le seed upsert les collections `candidates`, `candidate_profiles`, `job_profiles`, `decision_cards` et des traces d'artefacts Matching V3 dans `matching_runs`.

Arreter :

```bash
docker compose down
```

## Import Neo4j optionnel

Neo4j est optionnel. Le fallback YAML fonctionne sans import.

Pour importer le graphe :

```bash
docker compose exec api python scripts/import_graph_to_neo4j.py \
  --graph data/graph/skills_roles_graph.yaml \
  --profiles-dir data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles \
  --jobs-dir data/job_profiles \
  --reset
```

## Tester rapidement

```bash
python scripts/run_fast_tests.py
```

## Évaluation Copilot

```bash
python scripts/evaluate_copilot.py
```

Rapports générés :

- `docs/reports/copilot/copilot_evaluation.json`
- `docs/reports/copilot/copilot_evaluation.md`

## Scenario de demo court

Dans Streamlit :

1. Envoyer `nouvelle offre`.
2. Renseigner les 6 champs du wizard.
3. Envoyer `oui lance la recherche`.
4. Poser une question de suivi : `Pourquoi le premier candidat ?`.

Le script complet de soutenance est disponible ici :

- `docs/demo/demo_script.md`

## Notes

- Matching V3 reste la baseline officielle.
- `/api/match` utilise les artefacts `data/ranking/features/*.jsonl`.
- `/api/match` peut utiliser un matching live MongoDB + FAISS + Matching V3 avec `MATCHING_MODE=live` ou `hybrid`.
- Neo4j est optionnel avec fallback YAML.
- MongoDB est disponible dans Docker. Les artefacts restent seed/fallback pour ne pas casser la demo.
- E2 ne reconstruit pas dynamiquement l'index FAISS et ne rend pas MongoDB/Neo4j obligatoires pour les tests.
