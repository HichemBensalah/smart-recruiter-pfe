# Smart Recruiter - Repo Map

Ce document explique les dossiers principaux du repository. Il sert de guide pour une personne qui ouvre le projet dans VS Code.

## Vue rapide

| Dossier | Role | Statut | Nettoyage |
| --- | --- | --- | --- |
| `src/` | Code applicatif | critique | non |
| `src/api/` | FastAPI | critique | non |
| `src/core/` | Logique metier IA/RH | critique | non |
| `scripts/` | Orchestration et generation | critique | plus tard, avec prudence |
| `data/` | Artefacts et donnees | critique | seulement apres audit |
| `docs/` | Architecture et preuves | critique | archiver, pas supprimer |
| `tests/` | Validation | critique | non |
| `ui/` | Interface Streamlit | important | nettoyage leger |
| `.github/` | CI | important | non |

## `src/`

### Role
Contient le code applicatif principal.

### Contenu important
API, modules core, schemas et logique metier.

### Fichiers critiques
- `src/api/main.py`
- `src/core/`

### Statut
critique

### Peut etre nettoye ou non
Non, sauf caches Python.

### Risques
Un deplacement dans `src/` peut casser les imports et les tests.

## `src/api/`

### Role
Expose le backend via FastAPI.

### Contenu important
Routes health, candidates, match, decision cards, graph, chat, demo.

### Fichiers critiques
- `src/api/main.py`
- `src/api/schemas.py`
- `src/api/config.py`
- `src/api/auth.py`
- `src/api/routes/`

### Statut
stable

### Peut etre nettoye ou non
Non.

### Risques
Changer les schemas casse LangChain tools, Streamlit et tests.

## `src/core/`

### Role
Regroupe la logique metier et IA.

### Contenu important
Parser, structuring, storage, jobs, matching, ranking, graph, chatbot.

### Fichiers critiques
Tous les sous-dossiers listes ci-dessous.

### Statut
critique

### Peut etre nettoye ou non
Non, hors `__pycache__`.

### Risques
Les modules sont relies par artefacts et contrats ; un refactor global est risque avant soutenance.

## `src/core/parser/`

### Role
Parsing documentaire des CV.

### Contenu important
Docling wrapper, router, quality, handoff.

### Fichiers critiques
- `document_artifact.py`
- `document_router.py`
- `document_quality.py`
- `handoff_policy.py`
- `run_docling_pipeline.py`

### Statut
ferme

### Peut etre nettoye ou non
Non.

### Risques
Relancer ou modifier ce module peut changer les artefacts officiels.

## `src/core/structuring/`

### Role
Structuration grounded des CV.

### Contenu important
Prompt, normalisation Markdown, validation grounded, reporting.

### Fichiers critiques
- `profile_builder_grounded.py`
- `grounding_validator.py`
- `markdown_normalizer.py`
- `grounded_reporting.py`

### Statut
ferme

### Peut etre nettoye ou non
Non.

### Risques
Changer les regles peut modifier reliability, risk et champs candidats.

## `src/core/storage/`

### Role
MongoDB repositories et import des profils.

### Contenu important
Repositories, indexes Mongo, import/dedup.

### Fichiers critiques
- `repositories.py`
- `import_profiles_to_mongodb.py`

### Statut
stable

### Peut etre nettoye ou non
Non.

### Risques
Les routes API hybrid/live dependent des noms de collections et cles stables.

## `src/core/jobs/`

### Role
Construire et normaliser les job profiles.

### Contenu important
Builder, schema, parsing d'offre.

### Fichiers critiques
- `job_profile_builder.py`
- `job_profile_schema.py`

### Statut
stable

### Peut etre nettoye ou non
Non.

### Risques
Renommer ou changer les champs casse matching et routing.

## `src/core/matching/`

### Role
FAISS, scoring Matching V3 et live matcher.

### Contenu important
Scoring, normalisation skills, quality filters, FAISS indexer, recommender.

### Fichiers critiques
- `scoring.py`
- `live_matcher.py`
- `faiss_indexer.py`
- `recommender.py`
- `skill_normalizer.py`
- `matching_quality_filters.py`

### Statut
ferme pour Matching V3, stable pour live matcher.

### Peut etre nettoye ou non
Non.

### Risques
Modifier les poids ou penalites change la baseline officielle.

## `src/core/retrieval/`

### Role
Reranking CrossEncoder experimental.

### Contenu important
CrossEncoder local/offline et fallback.

### Fichiers critiques
- `cross_encoder_reranker.py`

### Statut
experimental

### Peut etre nettoye ou non
Non, garder pour traçabilite experimentale.

### Risques
Le remettre dans le chemin officiel rendrait la demo moins defendable.

## `src/core/ranking/`

### Role
Features ML, dataset, evaluation, ML reranking.

### Contenu important
Feature builder, dataset, evaluation, ML primary/reranking, Decision Cards ML.

### Fichiers critiques
- `features.py`
- `dataset.py`
- `evaluation.py`
- `ml_reranker.py`
- `ml_primary_ranker.py`
- `decision_cards_ml_enricher.py`

### Statut
experimental

### Peut etre nettoye ou non
Non, mais documenter comme experimental.

### Risques
Confondre ML experimental et baseline officielle.

## `src/core/graph/`

### Role
Transferability YAML et Neo4j Graph-RAG.

### Contenu important
YAML graph loader, score transferability, client Neo4j, queries Cypher.

### Fichiers critiques
- `transferability.py`
- `neo4j_client.py`
- `neo4j_transferability.py`

### Statut
YAML stable, Neo4j partiel.

### Peut etre nettoye ou non
Non.

### Risques
Neo4j ne doit pas devenir obligatoire tant que l'import reel n'est pas valide.

## `src/core/chatbot/`

### Role
LangGraph Copilot, memoire, wizard offre et reference resolver.

### Contenu important
Graph, state, memory, job intake, router, nodes.

### Fichiers critiques
- `graph.py`
- `state.py`
- `memory.py`
- `job_intake.py`
- `job_router.py`
- `reference_resolver.py`
- `nodes/`

### Statut
stable MVP

### Peut etre nettoye ou non
Non.

### Risques
Complexifier le flow avant soutenance peut creer des regressions conversationnelles.

## `src/core/chatbot/tools/`

### Role
LangChain tools autour de l'API.

### Contenu important
Client API, schemas, registry, wrappers de tools.

### Fichiers critiques
- `registry.py`
- `api_client.py`
- `schemas.py`
- fichiers tools individuels.

### Statut
stable

### Peut etre nettoye ou non
Non.

### Risques
Les tools dependent des routes FastAPI et de leurs schemas.

## `scripts/`

### Role
Scripts de generation, evaluation, seed, import, demo et maintenance.

### Contenu important
Tests rapides, seed MongoDB, matching V3, ML, SHAP, graph, demo, cleanup audit.

### Fichiers critiques
- `run_fast_tests.py`
- `seed_mongodb_from_artifacts.py`
- `evaluate_copilot.py`
- `run_demo_end_to_end.py`
- `run_matching_v3_normalized.py`
- `import_graph_to_neo4j.py`

### Statut
stable mais encombre.

### Peut etre nettoye ou non
Pas maintenant. Reorganisation possible plus tard avec wrappers.

### Risques
Deplacer les scripts casse README, docs, tests et chemins d'artefacts.

## `data/`

### Role
Stocker les donnees, artefacts et sorties techniques.

### Contenu important
CV, outputs parser/structuring, job profiles, FAISS, ranking, graph, evaluation.

### Fichiers critiques
- `data/processed_official_module1/`
- `data/profile_builder_module2_v2_grounded_all/`
- `data/indexes/faiss/cv_index.faiss`
- `data/indexes/faiss/id_map.pkl`
- `data/job_profiles/`
- `data/ranking/features/`
- `data/graph/skills_roles_graph.yaml`

### Statut
critique

### Peut etre nettoye ou non
Seulement apres audit et validation.

### Risques
Un nettoyage agressif casse la demo sans modifier le code.

## `docs/`

### Role
Documentation, architecture, rapports et preuves.

### Contenu important
Architecture, demo, rapports Module 1/2, matching, ML, graph, copilot.

### Fichiers critiques
- `docs/architecture/`
- `docs/reports/`
- `docs/demo/demo_script.md`

### Statut
important

### Peut etre nettoye ou non
Archiver plutot que supprimer.

### Risques
Supprimer un rapport peut retirer une preuve utile pour le jury.

## `tests/`

### Role
Valider le comportement du MVP.

### Contenu important
Tests API, matching, storage, graph, chatbot, tools, Streamlit, Docker.

### Fichiers critiques
- Tous les tests references par `scripts/run_fast_tests.py`.

### Statut
critique

### Peut etre nettoye ou non
Non.

### Risques
Retirer des tests masque les regressions.

## `ui/`

### Role
Interface Streamlit du Copilot.

### Contenu important
Chat, sidebar d'etat, rendu candidats/cards/transferability.

### Fichiers critiques
- `ui/streamlit_app.py`
- `ui/README.md`

### Statut
fonctionnel mais fragile en finition.

### Peut etre nettoye ou non
Oui pour caches et fichiers vides, pas pour l'app.

### Risques
Des problemes d'encodage ou d'UX peuvent nuire a la demo.

## `.github/`

### Role
CI GitHub Actions.

### Contenu important
Workflow de tests rapides.

### Fichiers critiques
- `.github/workflows/tests.yml`

### Statut
stable

### Peut etre nettoye ou non
Non.

### Risques
Changer la CI sans verifier localement peut bloquer les pushes/PR.
