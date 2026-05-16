# `src/`

Code applicatif du projet Smart Recruiter.

## Sous-dossiers principaux

- `api/` : facade FastAPI exposee pour la demo et les futurs tools LangChain.
- `core/parser/` : Module 1 Parsing documentaire.
- `core/structuring/` : Module 2 Structuration Grounded.
- `core/storage/` : import MongoDB.
- `core/matching/` : FAISS, scoring metier et Matching V3.
- `core/retrieval/` : experimentations retrieval/reranking, notamment CrossEncoder.
- `core/ranking/` : feature builder ML, evaluation, reranking experimental.
- `core/graph/` : Potential Graph YAML, transferability et Neo4j optionnel.
- `core/jobs/` : job profile builder.

## Regle de maintenance

Matching V3, Module 1, Module 2, FAISS et MongoDB sont des briques officielles. Les modules ML, SHAP, CrossEncoder et Neo4j sont utiles mais doivent rester clairement identifies comme experimentaux ou optionnels tant qu'ils ne remplacent pas la baseline.
