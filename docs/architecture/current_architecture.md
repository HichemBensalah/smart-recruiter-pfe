# Architecture actuelle - Smart Recruiter

Ce document donne une vue lisible de l'architecture actuelle avant l'ajout de LangChain Tools et LangGraph. Il ne remplace pas les rapports d'execution ; il sert de carte technique du projet.

## Pipeline officiel

CV bruts -> Module 1 Parsing -> Handoff -> Module 2 Structuration Grounded -> MongoDB -> FAISS -> Matching V3 normalized -> Decision Cards v3 normalized.

Les briques ML, SHAP, Potential Graph, Neo4j et demo enrichissent l'analyse mais ne remplacent pas Matching V3.

## Module 1 - Parsing documentaire

- Dossier principal : `src/core/parser/`
- Objectif : convertir les CV bruts en artefacts texte/markdown/json exploitables.
- Fichiers principaux : `docling_parser.py`, `secondary_parser.py`, `run_docling_pipeline.py`, `document_router.py`, `handoff_policy.py`.
- Entrees : `data/raw_cv/`
- Sorties : `data/processed_official_module1/`, `data/processed_official_module1/handoff/accepted.json`
- Resultats actuels : 90 CV traites et acceptes.
- Statut : officiel.

## Module 2 - Structuration Grounded

- Dossier principal : `src/core/structuring/`
- Objectif : transformer les sorties Module 1 en profils candidats structures, valides et grounded.
- Fichiers principaux : `profile_builder_grounded.py`, `grounding_validator.py`, `markdown_normalizer.py`, `grounded_reporting.py`.
- Scripts associes : `scripts/test_grounded_module2_v2.py`, `scripts/patch_grounded_profiles_v2.py`.
- Entrees : `accepted.json` et markdown Module 1.
- Sorties : `data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/*.json`
- Resultats actuels : 90 profils grounded complets ; reliability_score moyen annonce autour de 0.9286.
- Statut : officiel.

## Module 3 - MongoDB / FAISS

- Dossiers principaux : `src/core/storage/`, `src/core/matching/faiss_indexer.py`
- Objectif : stocker les profils puis construire l'index vectoriel de retrieval.
- Entrees : profils grounded.
- Sorties MongoDB : base `talent_intelligence`, collections `candidate_profiles` et `candidates`.
- Sorties FAISS : `data/indexes/faiss/cv_index.faiss`, `data/indexes/faiss/id_map.pkl`, `data/indexes/faiss/index_report.json`.
- Resultats actuels : 90 `candidate_profiles`, 75 `candidates`, 90 profils indexes.
- Statut : officiel.

## Module 4 - Matching V3

- Dossier principal : `src/core/matching/`
- Scripts associes : `scripts/run_matching_v3_normalized.py`
- Objectif : calculer le score metier officiel a partir du retrieval FAISS et des regles metier.
- Entrees : job profile, index FAISS, profils candidats.
- Sorties : `docs/reports/matching/v3/*_matching_report_v3_normalized.json`
- Resultats actuels : Matching V3 valide comme baseline officielle.
- Statut : officiel.

## Module 5 - Decision Cards

- Fichiers principaux : `scripts/generate_decision_cards_v3_normalized.py`, `docs/reports/matching/v3/decision_cards_v3_normalized.json`
- Objectif : produire une explication RH lisible des recommandations.
- Entrees : rapports Matching V3.
- Sorties : Decision Cards officielles.
- Statut : officiel.

## Module 6 - ML / SHAP

- Dossier principal : `src/core/ranking/`
- Scripts associes : `build_ranking_features.py`, `build_ranking_dataset.py`, `build_pseudo_labels.py`, `train_ranking_models.py`, `explain_xgboost_shap.py`.
- Entrees : features JSONL dans `data/ranking/features/`.
- Sorties : datasets, modeles `joblib`, rapports ML et SHAP.
- Resultats actuels : Logistic Regression, Random Forest et XGBoost entraines ; Random Forest meilleur modele ML actuel selon les metriques observees ; XGBoost conserve pour SHAP.
- Statut : experimental controle. Les labels sont des pseudo-labels metier, pas des labels recruteur.

## Module 7 - Potential Graph YAML

- Dossier principal : `src/core/graph/`
- Fichier source : `data/graph/skills_roles_graph.yaml`
- Objectif : analyser la transferabilite metier, les transitions plausibles et les gaps.
- Scripts associes : `scripts/compute_transferability_score.py`, `scripts/build_decision_cards_with_transferability.py`.
- Sorties : `docs/reports/graph/transferability_examples.*`, `docs/reports/decision_cards/decision_cards_with_transferability.*`
- Resultats actuels : 50 cartes enrichies, 100% de profils retrouves.
- Statut : officiel explicatif / demo.

## Module 8 - Neo4j Graph-RAG

- Fichiers principaux : `src/core/graph/neo4j_client.py`, `src/core/graph/neo4j_transferability.py`
- Script associe : `scripts/import_graph_to_neo4j.py`
- Objectif : proposer une couche Graph-RAG read-only optionnelle.
- Entrees : YAML, profils grounded, job profiles.
- Sorties : graphe Neo4j avec noeuds `Candidate`, `Skill`, `Role`, `Job`.
- Statut : optionnel avance. Si Neo4j est absent, l'API reste fonctionnelle.

## Module 9 - API FastAPI

- Dossier principal : `src/api/`
- Objectif : exposer les capacites du projet sous forme d'endpoints JSON reutilisables plus tard comme tools.
- Routes : health, candidates, match, decision cards, graph, Neo4j, demo.
- Commande : `uvicorn src.api.main:app --reload --port 8000`
- Statut : officiel comme facade API de demo.

## Module 10 - Demo end-to-end

- Script principal : `scripts/run_demo_end_to_end.py`
- Objectif : verifier/regenerer les artefacts principaux de demonstration en une commande.
- Sorties : `docs/reports/demo/demo_run_manifest.json`, `demo_run_summary.md`, `demo_summary_top10.*`, `demo_executive_summary.*`
- Statut : demo officielle.

## Module 11 - Futur LangChain / LangGraph

- Dossier cible recommande : `src/core/chatbot/` ou `src/services/copilot/`
- Objectif : orchestrer les endpoints existants en tools, puis construire un workflow LangGraph.
- Statut : a developper.
- Pre-requis : contrats API stables, distinction claire entre officiel, experimental et demo.
