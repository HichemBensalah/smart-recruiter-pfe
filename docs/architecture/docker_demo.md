# Docker demo

## Objectif

Docker Compose lance un environnement de demonstration complet pour Smart Recruiter sans reentrainer de modele et sans relancer les pipelines lourds.

Services :

| Service | Role | Ports |
| --- | --- | --- |
| `api` | API FastAPI Smart Recruiter | `8000` |
| `streamlit` | Interface chatbot Recruiter Copilot | `8501` |
| `neo4j` | Neo4j Graph-RAG optionnel | `7474`, `7687` |
| `mongodb` | MongoDB pour stockage applicatif | `27017` |

## Lancement local sans Docker

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8010
streamlit run ui/streamlit_app.py
```

Dans Streamlit, configurer l'URL API selon le port FastAPI :

```text
http://127.0.0.1:8010
```

## Lancement Docker Compose

```bash
docker compose up --build
```

Arret :

```bash
docker compose down
```

Arret avec suppression des volumes Neo4j/MongoDB :

```bash
docker compose down -v
```

## URLs

- Swagger FastAPI : http://localhost:8000/docs
- Health API : http://localhost:8000/health
- Streamlit Copilot : http://localhost:8501
- Neo4j Browser : http://localhost:7474
- MongoDB : `mongodb://localhost:27017`

## Variables d'environnement

En local :

```text
SMART_RECRUITER_API_BASE_URL=http://localhost:8000
MONGODB_URI=mongodb://localhost:27017
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

Dans Docker Compose :

```text
SMART_RECRUITER_API_BASE_URL=http://api:8000
MONGODB_URI=mongodb://mongodb:27017
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## Import Neo4j

L'import Neo4j n'est pas automatique pour ne pas fragiliser le demarrage Docker. Une fois les services lances, executer :

```bash
docker compose exec api python scripts/import_graph_to_neo4j.py \
  --graph data/graph/skills_roles_graph.yaml \
  --profiles-dir data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles \
  --jobs-dir data/job_profiles \
  --reset
```

Le rapport est genere ici :

```text
docs/reports/graph/neo4j_import_report.json
```

## Verification MongoDB

MongoDB est disponible dans Docker pour les modules de stockage et les evolutions runtime. La demo API actuelle peut encore lire certains artefacts locaux pre-generes pour rester rapide et reproductible.

Verifier MongoDB :

```bash
docker compose exec mongodb mongosh --quiet --eval "db.runCommand({ ping: 1 })"
```

## Fallbacks

- Si Neo4j est absent ou non importe, `/api/graph/transferability/{candidate_id}` utilise le fallback YAML.
- Si un `job_id` de matching n'a pas d'artefact, `/api/match` utilise le fallback officiel `backend_python_django_postgresql`.
- MongoDB est lance mais certains endpoints de demo restent bases sur artefacts locaux.

## Tests

Les tests Docker sont statiques et ne lancent pas Docker en CI :

```bash
pytest tests/test_docker_configuration.py -q
python scripts/run_fast_tests.py
```

## Limites connues

- Docker Compose est une configuration de demonstration locale, pas encore une configuration production.
- L'import Neo4j reste manuel.
- Les pipelines OCR, FAISS et ML lourds ne sont pas executes au demarrage Docker.
