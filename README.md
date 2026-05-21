# Smart Recruiter - Talent Intelligence Copilot RH

Smart Recruiter est un MVP RH pour analyser un corpus de CV, structurer les profils candidats, faire un matching explicable et exposer un Copilot recruteur via FastAPI, LangGraph et Streamlit.

Le systeme aide a accelerer la preselection. Il ne remplace pas la decision humaine finale.

## Etat actuel

- Projet stabilise sur la branche `main`.
- FastAPI, Streamlit, MongoDB repositories, live matcher, LangGraph Copilot, Docker et CI sont couverts.
- Suite rapide validee avec `python scripts/run_fast_tests.py`.
- Resultat attendu actuel : `141 passed`.
- Matching V3 reste la baseline officielle.
- RF, XGBoost, SHAP et Graph-RAG restent des couches experimentales ou optionnelles.

## Fonctionnalites principales

- Parsing documentaire de CV.
- Structuration grounded des profils candidats.
- Stockage MongoDB optionnel avec fallback artefacts.
- Indexation et retrieval FAISS.
- Matching V3 explicable.
- Live matcher MongoDB + FAISS + Matching V3.
- Decision Cards pour expliquer les recommandations.
- Potential Graph YAML pour la transferabilite metier.
- Neo4j Graph-RAG optionnel.
- API FastAPI metier.
- LangChain Tools autour de l'API.
- LangGraph Recruiter Copilot.
- Interface Streamlit chatbot.
- Evaluation automatique du Copilot.
- Docker Compose pour API, UI, MongoDB et Neo4j.

## Architecture

```text
CV bruts
  -> Parsing
  -> Structuration grounded
  -> Artefacts / MongoDB
  -> FAISS retrieval
  -> Matching V3
  -> Decision Cards
  -> Potential Graph / Neo4j Graph-RAG
  -> FastAPI
  -> LangChain Tools
  -> LangGraph Copilot
  -> Streamlit UI
```

## Installation locale

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copier l'exemple d'environnement si necessaire :

```bash
copy .env.example .env
```

Par defaut, le projet reste en mode artefacts pour ne pas rendre MongoDB, Neo4j ou FAISS obligatoires pendant la demo.

## Tests rapides

```bash
python scripts/run_fast_tests.py
```

Resultat attendu :

```text
141 passed
```

## Lancer FastAPI

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8010
```

Swagger :

```text
http://127.0.0.1:8010/docs
```

Health check :

```text
http://127.0.0.1:8010/health
```

## Lancer Streamlit

Dans un deuxieme terminal :

```bash
streamlit run ui/streamlit_app.py
```

URL :

```text
http://localhost:8501
```

Dans la sidebar Streamlit, utiliser l'URL API locale :

```text
http://127.0.0.1:8010
```

## Lancer avec Docker Compose

```bash
docker compose up --build
```

URLs :

- API Swagger : http://localhost:8000/docs
- API health : http://localhost:8000/health
- Streamlit Copilot : http://localhost:8501
- Neo4j Browser : http://localhost:7474
- MongoDB : `mongodb://localhost:27017`

Arreter :

```bash
docker compose down
```

## Modes de donnees

Le backend de donnees se configure via `.env` :

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

- `DATA_BACKEND=artifacts` : mode MVP stable, lecture des fichiers versionnes.
- `DATA_BACKEND=mongodb` : routes candidats et decision cards depuis MongoDB.
- `DATA_BACKEND=hybrid` : MongoDB prioritaire, artefacts en fallback.
- `MATCHING_MODE=artifact` : matching depuis `data/ranking/features/*.jsonl`.
- `MATCHING_MODE=live` : retrieval FAISS, profils MongoDB, scoring Matching V3.
- `MATCHING_MODE=hybrid` : tente le live puis retombe sur les artefacts si autorise.

Seed MongoDB depuis les artefacts :

```bash
python scripts/seed_mongodb_from_artifacts.py --dry-run
python scripts/seed_mongodb_from_artifacts.py
```

## Exemple de demo Copilot

Dans Streamlit :

```text
nouvelle offre
```

Le Copilot collecte les champs de l'offre, permet une correction, puis lance le matching avec :

```text
oui lance la recherche
```

Questions de suivi utiles :

```text
Pourquoi le premier candidat ?
Compare le premier et le deuxieme candidat
Quels sont les gaps du meilleur candidat ?
```

Script de soutenance :

- `docs/demo/demo_script.md`

## Evaluation Copilot

```bash
python scripts/evaluate_copilot.py
```

Rapports :

- `docs/reports/copilot/copilot_evaluation.json`
- `docs/reports/copilot/copilot_evaluation.md`

Etat actuel :

- 14 scenarios evalues.
- Flow offre -> correction -> confirmation -> matching -> suivis couvert.
- Tool calling accuracy, hallucination-free rate, latence `/api/chat` et memoire conversationnelle courte suivis.
- Fallback Neo4j/YAML et fallback artefacts Matching couverts.
- Validation recruteur humaine encore a faire.

## Structure du projet

```text
src/
  api/                 # API FastAPI
  core/                # Parsing, matching, ranking, graph, chatbot, storage
  benchmark/           # Benchmarks OCR
  models/              # Schemas applicatifs existants

scripts/               # Orchestration, tests rapides, seed MongoDB, graph, evaluation
data/                  # Artefacts, jobs, ranking, graph, donnees de demo
docs/                  # Architecture, rapports, demo
tests/                 # Tests API, graph, copilot, matching, storage, UI
ui/                    # Interface Streamlit
```

## Documentation utile

- `README_RUN.md` : guide court de lancement local et Docker.
- `docs/architecture/repository_audit.md` : audit de stabilisation.
- `docs/architecture/git_stabilization_plan.md` : plan de sauvegarde Git.
- `docs/architecture/docker_demo.md` : details Docker.
- `docs/architecture/ci.md` : CI GitHub Actions.
- `docs/architecture/pipeline_contracts.md` : contrats de pipeline.

## Limites connues

- Les modeles ML sont entraines sur pseudo-labels metier controles.
- Les pseudo-labels ne sont pas des labels recruteur reels.
- Matching V3 reste la baseline officielle.
- Neo4j est optionnel ; le fallback YAML reste disponible.
- MongoDB et FAISS ne sont pas obligatoires en mode artefacts.
- Pas encore d'integration ATS.
- Les donnees CV sensibles ne doivent pas etre ajoutees au repository.

## Auteur

Hichem Bensalah  
Projet PFE - Smart Recruiter
