# Smart Recruiter - Official vs Experimental

Ce document separe les briques officielles, experimentales, optionnelles et a ne plus toucher.

## Officiel / stable

### Parsing
Le parsing Docling/OCR est la premiere brique officielle. Il transforme les CV bruts en artefacts exploitables et a produit 90 CV accepted.

### Structuration
La structuration grounded est officielle. Elle produit des profils candidats controles contre le texte source.

### MongoDB / FAISS
MongoDB et FAISS forment le data layer et retrieval layer. MongoDB reste optionnel grace au mode artefacts, mais les repositories et l'index FAISS sont stables.

### Matching V3
Matching V3 est la baseline officielle. Il doit rester la reference de ranking pour le rapport et la soutenance.

### Decision Cards
Les Decision Cards rendent les resultats exploitables par un recruteur. Les cards basees sur Matching V3 sont officielles ; les enrichissements ML doivent rester qualifies.

### FastAPI
FastAPI est la couche d'exposition officielle : health, candidates, match, decision cards, graph, demo et chat.

### LangGraph Copilot
Le Copilot LangGraph est officiel dans le MVP. Il orchestre un workflow controle, base sur tools et sorties JSON.

### Job Intake Wizard
Le wizard de collecte d'offre est officiel. Il donne au Copilot une entree structuree avant matching.

### Streamlit
Streamlit est l'interface officielle de demonstration. Elle reste une UI simple, pas une plateforme RH complete.

### CI rapide
La suite `python scripts/run_fast_tests.py` est la validation rapide officielle.

## Experimental

### CrossEncoder
Le CrossEncoder est une experience de reranking. Il n'est pas retenu comme baseline officielle.

### ML LR/RF/XGBoost
Les modeles ML servent a comparer une approche supervisee avec Matching V3. Ils sont entraines sur pseudo-labels, donc ne doivent pas etre presentes comme verite recruteur.

### SHAP
SHAP explique XGBoost, pas Matching V3. Il est utile pour l'analyse XAI mais reste experimental.

### XGBoost primary ranking
XGBoost en ranker principal est une piste de recherche. Il ne remplace pas Matching V3 dans le MVP.

### Certains rapports ML
Les rapports de feature importance, pseudo-labels, ML reranking et primary ranking doivent etre classes comme preuves experimentales.

## Optionnel / partiel

### Neo4j reel
Le code Neo4j, les endpoints et le script d'import existent. L'import reel courant est marque `not_run`, donc Neo4j doit etre presente comme optionnel/partiel. Le fallback YAML est le comportement stable.

### Docker reel
Dockerfile et Docker Compose existent avec tests statiques. Une validation manuelle de `docker compose up --build` reste a faire pour affirmer une demo Docker complete.

### Matching dynamique sur job profile runtime
Le mode live existe via MongoDB + FAISS + `score_candidate()`. Le mode artefacts reste le plus stable pour la soutenance. Le matching dynamique doit etre presente comme disponible sous conditions de services et dependances.

## Ne plus toucher

- Parsing Module 1.
- Structuration grounded Module 2.
- Profils grounded finaux.
- Index FAISS et `id_map.pkl`.
- Matching V3.
- Decision Cards officielles.
- Potential Graph YAML.
- Schemas FastAPI stables.
- LangGraph flow actuel.
- Job Intake Wizard actuel.
- Suite rapide de tests.

## Peut etre ameliore plus tard

- Validation avec labels recruteur reels.
- Calibration des poids Matching V3.
- Neo4j importe et valide en live.
- Docker Compose teste de bout en bout.
- UI plus professionnelle.
- Persistance de memoire conversationnelle.
- RBAC, audit logs et securite production.
- Integration ATS.
- Observabilite LangSmith ou equivalente.
- Reorganisation progressive de `scripts/` et `data/`.

## Message a retenir pour le jury

La baseline defendable est Matching V3 + Decision Cards + Copilot outille. Les briques ML, SHAP, CrossEncoder et Neo4j montrent l'exploration avancee, mais elles sont separees de la decision officielle pour rester honnetes et defendables.
