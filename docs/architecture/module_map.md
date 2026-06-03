# Smart Recruiter - Module Map

Ce document cartographie les modules du projet Smart Recruiter. Il sert de reference rapide pour comprendre le role de chaque brique avant tout nettoyage ou refactoring.

Regle de lecture :

- `stable` : utilisable dans le MVP.
- `ferme` : ne plus modifier sauf bug critique.
- `experimental` : utile pour l'analyse, mais pas baseline officielle.
- `partiel` : present mais pas completement valide en execution reelle.
- `fragile` : fonctionne, mais demande une attention particuliere.
- `a developper` : prevu mais pas encore mature.

## Module 1 - Parsing CV

### Role
Convertir les CV PDF, DOCX, images et scans en texte, Markdown et artefacts exploitables.

### Pourquoi ce module existe
Le matching ne peut pas etre fiable si le texte source des CV est mal extrait.

### Entrees
CV bruts dans `data/raw_cv/`.

### Sorties
Artefacts Markdown, JSON, TXT, HTML et fichiers de handoff dans `data/processed_official_module1/`.

### Fonctionnement simple
Le routeur detecte le type de document, utilise Docling en principal, puis applique un fallback OCR si necessaire. Les documents sont classes en `accepted`, `repair_required` ou `quarantined`.

### Dossiers et fichiers principaux
- `src/core/parser/`
- `src/core/parser/document_artifact.py`
- `src/core/parser/document_router.py`
- `src/core/parser/document_quality.py`
- `src/core/parser/handoff_policy.py`
- `src/core/parser/run_docling_pipeline.py`

### Scripts associes
- `src/core/parser/run_docling_pipeline.py`

### Artefacts produits
- `data/processed_official_module1/handoff/accepted.json`
- `data/processed_official_module1/handoff/module1_handoff_report.json`
- Sorties `.md`, `.json`, `.txt`, `.html`

### Tests associes
Tests de parsing, qualite documentaire et handoff selon les fichiers presents dans `tests/`.

### Metriques de validation
90 CV traites, 90 accepted, 0 repair_required, 0 quarantined.

### Statut
ferme

### Ne plus toucher
Ne pas modifier le choix Docling principal + OCR fallback avant la soutenance.

### Amelioration future
Ajouter un benchmark OCR plus large sur des CV tres bruités.

### Risques
Les scans tres pauvres, les mises en page atypiques et les images basse resolution peuvent degrader l'extraction.

## Module 2 - Structuration grounded

### Role
Transformer les Markdown de CV en profils candidats structures et controles.

### Pourquoi ce module existe
Il reduit les hallucinations en verifiant les champs extraits contre le texte source.

### Entrees
Markdown et artefacts acceptes du Module 1.

### Sorties
Profils JSON grounded dans `data/profile_builder_module2_v2_grounded_all/`.

### Fonctionnement simple
Le module normalise le Markdown, appelle un provider LLM, valide les champs, nullifie les informations non supportees et calcule un score de fiabilite.

### Dossiers et fichiers principaux
- `src/core/structuring/profile_builder_grounded.py`
- `src/core/structuring/grounding_validator.py`
- `src/core/structuring/markdown_normalizer.py`
- `src/core/structuring/grounded_prompt.py`
- `src/core/structuring/grounded_reporting.py`

### Scripts associes
- `scripts/test_grounded_module2_v2.py`
- `scripts/patch_grounded_profiles_v2.py`

### Artefacts produits
- `data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/`
- `data/profile_builder_module2_v2_grounded_all/reports/grounded_quality_report.json`

### Tests associes
Tests de structuration, validation et scripts historiques.

### Metriques de validation
Etat courant documente : 90 profils complets/exploitables, reliability moyenne 0.9286, 89 low risk, 1 medium risk. Un rapport historique plus ancien indique 0.8441 avant corrections finales.

### Statut
ferme

### Ne plus toucher
Ne pas modifier les profils finaux ni les regles de grounding sans besoin critique.

### Amelioration future
Clarifier dans le rapport la difference entre les metriques historiques et l'etat courant.

### Risques
La qualite depend du texte extrait et du provider LLM utilise lors de la generation.

## Module 3 - MongoDB / Storage

### Role
Stocker les candidats, profils, job profiles, matching runs et Decision Cards.

### Pourquoi ce module existe
Il donne au projet une couche data applicative proche d'un produit reel.

### Entrees
Profils grounded, job profiles, artefacts Matching V3 et Decision Cards.

### Sorties
Collections MongoDB : `candidate_profiles`, `candidates`, `job_profiles`, `matching_runs`, `decision_cards`.

### Fonctionnement simple
Les artefacts locaux peuvent etre importes dans MongoDB. Les routes API peuvent fonctionner en mode `artifacts`, `mongodb` ou `hybrid`.

### Dossiers et fichiers principaux
- `src/core/storage/repositories.py`
- `src/core/storage/import_profiles_to_mongodb.py`
- `src/api/config.py`

### Scripts associes
- `scripts/seed_mongodb_from_artifacts.py`

### Artefacts produits
- `docs/reports/mongodb/mongodb_import_report_v2_grounded_execute.json`

### Tests associes
- `tests/test_mongodb_repositories.py`
- Tests API candidats et Decision Cards en mode fallback.

### Metriques de validation
90 `candidate_profiles`, 75 `candidates` consolides apres deduplication.

### Statut
stable

### Ne plus toucher
Ne pas changer les cles stables et la logique de fallback avant soutenance.

### Amelioration future
Valider une demonstration complete en mode MongoDB live.

### Risques
MongoDB reste optionnel ; si le service est absent, il faut assumer le mode artefacts.

## Module 4 - FAISS / Retrieval

### Role
Indexer les profils candidats et retrouver les profils les plus proches d'une offre.

### Pourquoi ce module existe
Il permet de prefiltrer les candidats avant le scoring metier.

### Entrees
Profils candidats issus de MongoDB ou des artefacts.

### Sorties
Index FAISS, mapping d'identifiants et rapport d'indexation.

### Fonctionnement simple
Les profils sont transformes en textes, encodes avec `sentence-transformers/all-MiniLM-L6-v2`, normalises, puis indexes dans FAISS.

### Dossiers et fichiers principaux
- `src/core/matching/faiss_indexer.py`
- `src/core/matching/recommender.py`
- `src/core/matching/profile_text_builder.py`
- `src/core/matching/job_text_builder.py`

### Scripts associes
Generation historique de l'index via le module FAISS.

### Artefacts produits
- `data/indexes/faiss/cv_index.faiss`
- `data/indexes/faiss/id_map.pkl`
- `data/indexes/faiss/index_report.json`

### Tests associes
- `tests/test_live_matcher.py`
- Tests API matching en mode artefacts/hybrid.

### Metriques de validation
90 profils indexes, embeddings 384 dimensions.

### Statut
stable

### Ne plus toucher
Ne pas reconstruire l'index sauf ajout controle de nouveaux CV.

### Amelioration future
Documenter une procedure de rebuild reproductible.

### Risques
Dependances FAISS et sentence-transformers a surveiller ; le cache Hugging Face est volumineux.

## Module 5 - Job Profiles

### Role
Transformer une offre en profil de poste structure.

### Pourquoi ce module existe
Le matching a besoin d'une representation stable des competences, responsabilites, seniorite et contraintes de poste.

### Entrees
Offres texte, champs du wizard ou fichiers JSON.

### Sorties
Job profiles JSON.

### Fonctionnement simple
Le builder extrait titre, role cible, competences requises, competences bonus, experience, seniorite, localisation et mode de travail.

### Dossiers et fichiers principaux
- `src/core/jobs/job_profile_builder.py`
- `src/core/jobs/job_profile_schema.py`
- `src/core/chatbot/job_router.py`

### Scripts associes
Scripts de matching et ranking qui consomment les job profiles.

### Artefacts produits
- `data/job_profiles/*.json`
- `data/job_profiles/job_profile_builder_report.json`

### Tests associes
- `tests/test_job_intake.py`
- `tests/test_job_intake_single_path.py`
- Tests de routing via le Copilot.

### Metriques de validation
Plusieurs offres de demonstration sont disponibles et routables par `job_id`.

### Statut
stable

### Ne plus toucher
Ne pas renommer les `job_id` de demo sans mettre a jour matching, Decision Cards et tests.

### Amelioration future
Aligner un scenario canonique unique pour toute la demonstration.

### Risques
Une offre hors domaine peut tomber sur un fallback generique.

## Module 6 - Matching V3

### Role
Classer les candidats pour une offre avec un score explicable.

### Pourquoi ce module existe
Il est la baseline officielle du projet, plus defendable qu'un modele ML entraine sur pseudo-labels.

### Entrees
Job profile, profils candidats, similarite vectorielle et qualite grounded.

### Sorties
Ranking, scores, competences matchees/manquantes, penalites et explications.

### Fonctionnement simple
Matching V3 combine couverture des competences, similarite texte, experience, qualite du profil, reliability, hallucination risk et penalites must-have.

### Dossiers et fichiers principaux
- `src/core/matching/scoring.py`
- `src/core/matching/live_matcher.py`
- `src/core/matching/skill_normalizer.py`
- `src/core/matching/matching_quality_filters.py`

### Scripts associes
- `scripts/run_matching_v3_normalized.py`

### Artefacts produits
- `docs/reports/matching/v3/*_matching_report_v3_normalized.json`
- `docs/reports/matching/v3/matching_report_v3_normalized.json`
- `data/ranking/features/*.jsonl`

### Tests associes
- `tests/test_api_match.py`
- `tests/test_live_matcher.py`

### Metriques de validation
Scores V3, must-have coverage, missing skills, reliability, hallucination risk, Precision@K/nDCG via rapports ML.

### Statut
ferme

### Ne plus toucher
Ne pas modifier les poids, penalites et normalisations avant soutenance.

### Amelioration future
Calibrer les poids avec labels recruteur reels.

### Risques
Les poids restent des regles metier, pas des parametres appris sur un grand dataset labelise.

## Module 7 - Decision Cards

### Role
Produire des fiches lisibles par recruteur pour expliquer les recommandations.

### Pourquoi ce module existe
Un score seul n'est pas exploitable par un recruteur ; il faut un verdict et des raisons.

### Entrees
Resultats Matching V3, ML experimental, SHAP et transferability selon la carte.

### Sorties
Decision Cards JSON et Markdown.

### Fonctionnement simple
Les scripts agregent score, rang, forces, gaps, risques et explications dans un format consommable par API, Copilot et UI.

### Dossiers et fichiers principaux
- `src/api/routes/decision_cards.py`
- `src/api/utils.py`
- Scripts de generation dans `scripts/`

### Scripts associes
- `scripts/generate_decision_cards.py`
- `scripts/generate_decision_cards_v3_normalized.py`
- `scripts/build_decision_cards_ml_comparison.py`
- `scripts/build_decision_cards_with_transferability.py`

### Artefacts produits
- `docs/reports/matching/v3/decision_cards_v3_normalized.json`
- `docs/reports/decision_cards/decision_cards_ml_comparison.json`
- `docs/reports/decision_cards/decision_cards_with_transferability.json`

### Tests associes
- `tests/test_api_decision_cards.py`

### Metriques de validation
Cards disponibles pour les top candidats ; lookup success transferability 100% dans le rapport enrichi.

### Statut
stable

### Ne plus toucher
Ne pas casser le schema consomme par API et Streamlit.

### Amelioration future
Harmoniser les sources de cards autour d'un scenario de demo unique.

### Risques
Les cartes enrichies ML peuvent brouiller le message si elles sont presentees comme baseline.

## Module 8 - CrossEncoder

### Role
Tester un reranking neural plus fin entre offre et candidats.

### Pourquoi ce module existe
Evaluer si un modele cross-encoder ameliore le ranking apres FAISS.

### Entrees
Texte de job et textes candidats.

### Sorties
Scores cross-encoder et rapports de comparaison.

### Fonctionnement simple
Le module charge un CrossEncoder local si disponible, score les paires job-candidat et renvoie un reranking ou un fallback.

### Dossiers et fichiers principaux
- `src/core/retrieval/cross_encoder_reranker.py`
- `src/core/retrieval/`

### Scripts associes
- `scripts/run_cross_encoder_reranking.py`
- `scripts/run_cross_encoder_reranking_constrained.py`

### Artefacts produits
- `docs/reports/cross_encoder/`
- `docs/reports/retrieval/`

### Tests associes
Tests indirects de contrats et fallbacks.

### Metriques de validation
Comparaisons ablation et rapports CrossEncoder.

### Statut
experimental

### Ne plus toucher
Ne pas le reintegrer comme baseline officielle.

### Amelioration future
Reprendre avec calibration metier et labels humains.

### Risques
Peut favoriser la proximite textuelle au detriment du fit metier.

## Module 9 - Feature Builder ML

### Role
Construire les features offre-candidat pour les experimentations ML.

### Pourquoi ce module existe
Comparer Matching V3 avec des modeles supervises et produire une analyse quantitative.

### Entrees
Rapports Matching V3 multi-offres.

### Sorties
Fichiers JSONL de features.

### Fonctionnement simple
Le builder extrait 12 features : similarite, final_score_v3, coverage, overlaps, experience, seniority, qualite, reliability, risque encode, missing/matched counts.

### Dossiers et fichiers principaux
- `src/core/ranking/features.py`
- `src/core/ranking/dataset.py`

### Scripts associes
- `scripts/build_ranking_features.py`
- `scripts/build_ranking_dataset.py`

### Artefacts produits
- `data/ranking/features/*.jsonl`
- `data/ranking/datasets/ranking_dataset_aligned_summary.json`

### Tests associes
Tests ranking/dataset selon couverture existante.

### Metriques de validation
250 lignes, 5 offres alignees, 12 features.

### Statut
stable

### Ne plus toucher
Ne pas regenerer les features avant soutenance.

### Amelioration future
Ajouter features basees sur retours recruteur reels.

### Risques
Les features dependent fortement de Matching V3, ce qui cree une circularite dans les experiences ML.

## Module 10 - Pseudo-labels

### Role
Creer des labels simulés pour entrainer et comparer les modeles ML.

### Pourquoi ce module existe
Le projet ne dispose pas encore de labels recruteur reels.

### Entrees
Dataset de features ranking.

### Sorties
Labels binaires et multi-classes simules.

### Fonctionnement simple
Des regles metier transforment les signaux de matching en labels approximatifs.

### Dossiers et fichiers principaux
- `src/core/ranking/`
- Scripts pseudo-labels.

### Scripts associes
- `scripts/build_pseudo_labels.py`
- `scripts/simulate_pseudo_label_rules.py`

### Artefacts produits
- `data/ranking/datasets/pseudo_label_aligned_summary.json`
- `docs/reports/ml/pseudo_label_rule_simulation*.json`

### Tests associes
Tests de generation et coherences dataset selon fichiers disponibles.

### Metriques de validation
250 lignes, 43 positifs, taux positif 17.2%.

### Statut
experimental

### Ne plus toucher
Ne pas presenter comme labels recruteur reels.

### Amelioration future
Collecter annotations humaines.

### Risques
Biais fort : les labels derivent des regles qui evaluent ensuite les modeles.

## Module 11 - ML Pipeline

### Role
Entrainer et comparer Logistic Regression, Random Forest et XGBoost.

### Pourquoi ce module existe
Fournir une couche d'analyse experimentale et montrer une competence ML/XAI.

### Entrees
Dataset ranking avec pseudo-labels.

### Sorties
Modeles, metriques et rapports.

### Fonctionnement simple
Le pipeline entraine plusieurs modeles, evalue par LeaveOneGroupOut sur `job_id` et compare des metriques ranking.

### Dossiers et fichiers principaux
- `src/core/ranking/evaluation.py`
- `src/core/ranking/ml_reranker.py`
- `src/core/ranking/ml_primary_ranker.py`

### Scripts associes
- `scripts/train_ranking_models.py`
- `scripts/run_ml_reranking.py`
- `scripts/run_xgboost_primary_ranking.py`
- `scripts/generate_ml_experiment_interpretation.py`

### Artefacts produits
- `data/ranking/models/*.joblib`
- `data/ranking/models/training_report.json`
- `docs/reports/ml/ml_experiment_report.*`

### Tests associes
Tests de rapports/contrats et scripts selon couverture existante.

### Metriques de validation
Random Forest meilleur modele ML courant ; Matching V3 reste la baseline officielle.

### Statut
experimental

### Ne plus toucher
Ne pas reentrainer les modeles avant soutenance.

### Amelioration future
Entrainer avec labels humains et plus d'offres.

### Risques
Scores tres eleves mais interpretes avec prudence car pseudo-labels et petit dataset.

## Module 12 - SHAP

### Role
Expliquer le modele XGBoost experimental.

### Pourquoi ce module existe
Donner une preuve XAI et identifier les features qui influencent le modele.

### Entrees
Modele XGBoost, dataset de features.

### Sorties
SHAP global summary et exemples locaux.

### Fonctionnement simple
SHAP calcule l'importance moyenne des features et les contributions locales.

### Dossiers et fichiers principaux
- `src/core/ranking/`
- Scripts SHAP.

### Scripts associes
- `scripts/explain_xgboost_shap.py`

### Artefacts produits
- `docs/reports/ml/shap/shap_global_summary.json`
- `docs/reports/ml/shap/shap_local_examples.json`
- `docs/reports/ml/shap/shap_methodology_note.md`

### Tests associes
Validation par presence/forme des rapports.

### Metriques de validation
Features importantes : `final_score_v3`, `must_have_coverage`, `vector_similarity`, `profile_quality_score`, `experience_match_score`.

### Statut
experimental

### Ne plus toucher
Ne pas utiliser SHAP comme explication officielle de Matching V3.

### Amelioration future
SHAP sur un modele entraine avec labels recruteur reels.

### Risques
SHAP explique XGBoost, pas la decision finale du systeme.

## Module 13 - Potential Graph YAML

### Role
Representer roles, skills, transitions et gaps dans un graphe YAML.

### Pourquoi ce module existe
Ajouter une explication metier de transferabilite entre roles.

### Entrees
Profil candidat, job profile, graphe YAML.

### Sorties
Score de transferabilite, fit direct, transitions plausibles, gaps compensables et bloquants.

### Fonctionnement simple
Le module compare les skills du candidat aux roles cibles et cherche des transitions plausibles.

### Dossiers et fichiers principaux
- `src/core/graph/transferability.py`
- `data/graph/skills_roles_graph.yaml`

### Scripts associes
- `scripts/compute_transferability_score.py`
- `scripts/build_decision_cards_with_transferability.py`

### Artefacts produits
- `docs/reports/graph/transferability_examples.json`
- `docs/reports/decision_cards/decision_cards_with_transferability.json`

### Tests associes
- `tests/test_api_graph.py`
- `tests/test_neo4j_transferability.py`

### Metriques de validation
8 roles dans le YAML, lookup success 100% dans les cards enrichies.

### Statut
stable

### Ne plus toucher
Garder le YAML comme fallback officiel.

### Amelioration future
Enrichir la taxonomie skills/roles avec donnees RH.

### Risques
Le graphe reste construit manuellement et peut manquer de granularite.

## Module 14 - Neo4j Graph-RAG

### Role
Importer le graphe et les profils dans Neo4j pour interroger roles, skills et gaps en Cypher.

### Pourquoi ce module existe
Fournir une brique Graph-RAG plus demonstrable et visualisable que le YAML.

### Entrees
YAML roles/skills, profils grounded, job profiles.

### Sorties
Noeuds Neo4j, relations et reponses API graph.

### Fonctionnement simple
Le script d'import cree les contraintes, importe roles, skills, jobs, candidats et relations. Les endpoints interrogent Neo4j si configure, sinon retournent une erreur explicite ou fallback YAML selon route.

### Dossiers et fichiers principaux
- `src/core/graph/neo4j_client.py`
- `src/core/graph/neo4j_transferability.py`
- `src/api/routes/graph_neo4j.py`

### Scripts associes
- `scripts/import_graph_to_neo4j.py`

### Artefacts produits
- `docs/reports/graph/neo4j_import_report.json`
- `docs/reports/graph/neo4j_graph_rag.md`

### Tests associes
- `tests/test_neo4j_transferability.py`
- `tests/test_api_graph_neo4j.py`

### Metriques de validation
Endpoints et fallback testes ; rapport d'import courant marque `not_run`.

### Statut
partiel

### Ne plus toucher
Ne pas rendre Neo4j obligatoire tant que l'import reel n'est pas valide.

### Amelioration future
Executer l'import Neo4j reel et capturer un rapport avec counts non nuls.

### Risques
Sur-vendre Neo4j comme production-ready alors qu'il est optionnel dans le MVP.

## Module 15 - FastAPI

### Role
Exposer les capacites Smart Recruiter via REST et Swagger.

### Pourquoi ce module existe
Centraliser l'acces aux candidats, matching, Decision Cards, graph, demo et chat.

### Entrees
Requetes HTTP JSON.

### Sorties
Reponses JSON structurees.

### Fonctionnement simple
FastAPI charge les settings, choisit artefacts/Mongo/hybrid, expose les routes et applique une API key optionnelle.

### Dossiers et fichiers principaux
- `src/api/main.py`
- `src/api/schemas.py`
- `src/api/config.py`
- `src/api/routes/`
- `src/api/utils.py`

### Scripts associes
Lancement via `uvicorn src.api.main:app`.

### Artefacts produits
Pas d'artefact principal ; expose les artefacts existants.

### Tests associes
- `tests/test_api_health.py`
- `tests/test_api_candidates.py`
- `tests/test_api_match.py`
- `tests/test_api_decision_cards.py`
- `tests/test_api_graph.py`
- `tests/test_api_demo.py`
- `tests/test_api_chat.py`

### Metriques de validation
Endpoints couverts par la suite rapide et Swagger disponible au lancement.

### Statut
stable

### Ne plus toucher
Ne pas changer les schemas de sortie sans mettre a jour tools, Streamlit et tests.

### Amelioration future
Ajouter pagination avancee, RBAC et audit logs.

### Risques
Les modes `live` et `hybrid` dependent des services locaux.

## Module 16 - LangChain Tools

### Role
Transformer les endpoints et capacites backend en tools appelables par le Copilot.

### Pourquoi ce module existe
Separer orchestration conversationnelle et logique metier.

### Entrees
Schemas Pydantic de tools.

### Sorties
JSON normalise depuis l'API.

### Fonctionnement simple
Les tools appellent FastAPI via un client HTTP et propagent les erreurs sous forme controlee.

### Dossiers et fichiers principaux
- `src/core/chatbot/tools/`
- `src/core/chatbot/tools/registry.py`
- `src/core/chatbot/tools/api_client.py`
- `src/core/chatbot/tools/schemas.py`

### Scripts associes
Pas de script lourd ; utilises par tests et LangGraph.

### Artefacts produits
Aucun artefact persistant.

### Tests associes
- `tests/test_langchain_tools_api_client.py`
- `tests/test_langchain_tools_registry.py`
- `tests/test_langchain_tools_contracts.py`

### Metriques de validation
9 tools declares dans le registry.

### Statut
stable

### Ne plus toucher
Garder les descriptions, schemas et noms de tools stables pour LangGraph.

### Amelioration future
Ajouter tracing/observabilite des tool calls.

### Risques
Si les endpoints API changent, les tools cassent.

## Module 17 - LangGraph Copilot

### Role
Orchestrer la conversation recruteur et les tools metier.

### Pourquoi ce module existe
Donner une experience Copilot RH au lieu d'une simple API.

### Entrees
Message utilisateur, session_id et etat conversationnel.

### Sorties
Reponse naturelle, candidats, Decision Cards, transferability, sources et warnings.

### Fonctionnement simple
Le graphe passe par understanding, matching, cards, transferability et composition de reponse. Le flow job intake gere les offres avant matching.

### Dossiers et fichiers principaux
- `src/core/chatbot/graph.py`
- `src/core/chatbot/nodes/`
- `src/api/routes/chat.py`

### Scripts associes
- `scripts/evaluate_copilot.py`

### Artefacts produits
- `docs/reports/copilot/copilot_evaluation.json`
- `docs/reports/copilot/copilot_evaluation.md`

### Tests associes
- `tests/test_langgraph_copilot_state.py`
- `tests/test_langgraph_copilot_nodes.py`
- `tests/test_langgraph_copilot_graph.py`
- `tests/test_api_chat.py`

### Metriques de validation
14 scenarios Copilot passes, tool calling accuracy 1.0, hallucination-free rate 1.0 dans le rapport courant.

### Statut
stable

### Ne plus toucher
Ne pas complexifier le workflow avant soutenance.

### Amelioration future
Ajouter un LLM controle pour reformulation, avec garde-fous et sources.

### Risques
Le Copilot est un workflow controle, pas un agent autonome generaliste.

## Module 18 - Memoire courte

### Role
Conserver le contexte court d'une session de chat.

### Pourquoi ce module existe
Permettre les questions de suivi comme "le premier candidat" ou "compare les deux".

### Entrees
session_id, messages et resultats precedents.

### Sorties
Etat de conversation en memoire.

### Fonctionnement simple
Un store en RAM garde les derniers tours, les candidats, les cards, le selected_candidate_id et l'etat du wizard.

### Dossiers et fichiers principaux
- `src/core/chatbot/memory.py`
- `src/core/chatbot/reference_resolver.py`

### Scripts associes
Pas de script dedie.

### Artefacts produits
Aucun artefact persistant.

### Tests associes
- `tests/test_chat_memory.py`
- `tests/test_reference_resolver.py`

### Metriques de validation
TTL 30 minutes, historique court limite, coherence memoire couverte par l'evaluation Copilot.

### Statut
stable

### Ne plus toucher
Garder la memoire simple et previsible pour la demo.

### Amelioration future
Persistance en base, multi-utilisateur et expiration configurable.

### Risques
La memoire est volatile et non adaptee a une production multi-tenant.

## Module 19 - Job Intake Wizard

### Role
Collecter une offre recruteur en plusieurs etapes.

### Pourquoi ce module existe
Eviter les offres incompletes et produire un job profile exploitable.

### Entrees
Messages utilisateur pendant le flow `nouvelle offre`.

### Sorties
Champs collectes, profil structure et job_id route.

### Fonctionnement simple
Le wizard demande titre, contexte, responsabilites, skills requis, skills bonus et profil recherche, puis propose confirmation.

### Dossiers et fichiers principaux
- `src/core/chatbot/job_intake.py`
- `src/core/chatbot/job_router.py`

### Scripts associes
Evaluation Copilot.

### Artefacts produits
Aucun artefact permanent.

### Tests associes
- `tests/test_job_intake.py`
- `tests/test_job_intake_offer_summary.py`
- `tests/test_job_intake_single_path.py`
- `tests/test_job_intake_reset.py`

### Metriques de validation
Flow six champs couvert dans l'evaluation Copilot.

### Statut
stable

### Ne plus toucher
Ne pas changer l'ordre des champs sans adapter tests et Streamlit.

### Amelioration future
Ameliorer la detection de champs dans des offres longues.

### Risques
Un texte utilisateur ambigu peut etre classe comme correction ou reponse de champ.

## Module 20 - Modification de champ avant confirmation

### Role
Permettre au recruteur de corriger une offre avant lancement du matching.

### Pourquoi ce module existe
Un vrai utilisateur corrige souvent son besoin avant de valider.

### Entrees
Message de correction de champ.

### Sorties
Champ modifie et resume d'offre mis a jour.

### Fonctionnement simple
Le module detecte les intentions de correction, identifie le champ cible et applique le remplacement avant confirmation.

### Dossiers et fichiers principaux
- `src/core/chatbot/job_intake.py`
- `src/core/chatbot/graph.py`

### Scripts associes
Evaluation Copilot.

### Artefacts produits
Aucun artefact permanent.

### Tests associes
- `tests/test_job_intake_field_edit.py`
- Scenarios Copilot de correction pre-confirmation.

### Metriques de validation
Scenario `pre_confirmation_correction` couvert dans `copilot_evaluation.json`.

### Statut
stable

### Ne plus toucher
Garder le comportement actuel jusqu'a la demo.

### Amelioration future
Ajouter une UI explicite pour editer les champs.

### Risques
La detection par mots-cles peut rater certaines formulations.

## Module 21 - Streamlit UI

### Role
Fournir une interface visuelle de demonstration du Copilot.

### Pourquoi ce module existe
Permettre une soutenance sans terminal et montrer les candidats/cards/gaps.

### Entrees
Messages utilisateur et URL FastAPI.

### Sorties
Chat, sidebar d'etat, candidats, Decision Cards et transferability.

### Fonctionnement simple
Streamlit appelle `/health` et `/api/chat`, conserve le session_id et rend les blocs de reponse.

### Dossiers et fichiers principaux
- `ui/streamlit_app.py`
- `ui/README.md`

### Scripts associes
Lancement via `streamlit run ui/streamlit_app.py`.

### Artefacts produits
Aucun artefact.

### Tests associes
- `tests/test_streamlit_app_static.py`

### Metriques de validation
Test statique verifie usage API, session_id, rendered candidates, cards et transferability.

### Statut
fragile

### Ne plus toucher
Ne pas embarquer de logique metier dans Streamlit.

### Amelioration future
Corriger les finitions d'encodage et ameliorer l'UX.

### Risques
Des libelles avec encodage incorrect peuvent nuire a la qualite percue en soutenance.

## Module 22 - CI rapide

### Role
Executer une suite rapide de tests en local et GitHub Actions.

### Pourquoi ce module existe
Proteger le MVP contre les regressions.

### Entrees
Code source, tests rapides, artefacts legers.

### Sorties
Resultat pytest.

### Fonctionnement simple
`scripts/run_fast_tests.py` force les modes artefacts et lance une selection de tests sans services externes.

### Dossiers et fichiers principaux
- `.github/workflows/tests.yml`
- `scripts/run_fast_tests.py`
- `pytest.ini`

### Scripts associes
- `scripts/run_fast_tests.py`

### Artefacts produits
Pas d'artefact fonctionnel ; logs CI uniquement.

### Tests associes
Suite rapide configuree dans le script.

### Metriques de validation
README indique un resultat attendu de 141 tests passes.

### Statut
stable

### Ne plus toucher
Ne pas retirer des tests de la suite rapide sans justification.

### Amelioration future
Ajouter une suite integration separee pour MongoDB/Neo4j/Docker.

### Risques
Les tests rapides ne prouvent pas les services externes reels.

## Module 23 - Docker

### Role
Lancer API, Streamlit, MongoDB et Neo4j en environnement local compose.

### Pourquoi ce module existe
Rendre la demonstration plus reproductible.

### Entrees
Dockerfile, docker-compose et variables `.env`.

### Sorties
Services API/UI/DB accessibles localement.

### Fonctionnement simple
Compose build l'image projet, lance API et Streamlit, puis demarre MongoDB et Neo4j avec healthchecks.

### Dossiers et fichiers principaux
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `.env.example`

### Scripts associes
- Import Neo4j manuel via `scripts/import_graph_to_neo4j.py`
- Seed MongoDB via `scripts/seed_mongodb_from_artifacts.py`

### Artefacts produits
Volumes Docker locaux, non versionnes.

### Tests associes
- `tests/test_docker_configuration.py`

### Metriques de validation
Tests statiques verifient services, ports, variables et documentation.

### Statut
partiel

### Ne plus toucher
Ne pas rendre Docker obligatoire pour lancer le MVP local.

### Amelioration future
Tester manuellement `docker compose up --build` et documenter le resultat.

### Risques
L'import Neo4j n'est pas automatique et Docker n'est pas valide par un test d'execution complet.

## Module 24 - Tests

### Role
Verifier API, matching, chatbot, graph, storage, UI et configuration.

### Pourquoi ce module existe
Fournir des preuves de non-regression et de qualite logicielle.

### Entrees
Code source, artefacts de demo et fixtures.

### Sorties
Resultats pytest.

### Fonctionnement simple
Les tests rapides s'executent sans MongoDB, Neo4j, Docker ou Streamlit.

### Dossiers et fichiers principaux
- `tests/`
- `pytest.ini`
- `scripts/run_fast_tests.py`

### Scripts associes
- `scripts/run_fast_tests.py`

### Artefacts produits
Caches pytest eventuels.

### Tests associes
Tous les tests du dossier `tests/`.

### Metriques de validation
README annonce 141 tests rapides passes ; audit historique mentionne aussi une collecte plus large.

### Statut
stable

### Ne plus toucher
Ne pas supprimer les tests qui documentent des contrats API ou Copilot.

### Amelioration future
Ajouter tests integration reels pour Docker, MongoDB et Neo4j.

### Risques
Les tests peuvent passer en mode artefacts tout en masquant une panne live service.

## Module 25 - Documentation

### Role
Documenter architecture, pipeline, demo, preuves et limites.

### Pourquoi ce module existe
Le projet PFE doit etre defendable et comprehensible.

### Entrees
Etat du code, rapports, decisions techniques.

### Sorties
Markdown d'architecture, rapports JSON/MD et guides de lancement.

### Fonctionnement simple
Les docs separent l'architecture courante, les contrats, les rapports et les guides de demo.

### Dossiers et fichiers principaux
- `docs/architecture/`
- `docs/reports/`
- `docs/demo/`
- `README.md`
- `README_RUN.md`

### Scripts associes
Scripts de generation de rapports dans `scripts/`.

### Artefacts produits
Rapports Markdown et JSON.

### Tests associes
Tests statiques lisant README, docs Docker et UI README.

### Metriques de validation
Presence de rapports pour parsing, matching, ML, graph, copilot et demo.

### Statut
stable

### Ne plus toucher
Ne pas supprimer les preuves historiques sans archivage.

### Amelioration future
Creer une matrice CDC -> implementation -> preuve.

### Risques
Certaines docs sont plus anciennes que le code et peuvent etre partiellement obsoletes.

## Module 26 - Scripts

### Role
Orchestrer les generations d'artefacts, evaluations, imports et maintenance.

### Pourquoi ce module existe
Reproduire les etapes techniques sans tout melanger dans l'application.

### Entrees
Artefacts, profils, jobs, datasets et configs.

### Sorties
Rapports, datasets, models, cards, imports.

### Fonctionnement simple
Chaque script execute une etape precise : seed, matching, ML, SHAP, graph, demo, evaluation ou cleanup dry-run.

### Dossiers et fichiers principaux
- `scripts/`
- `scripts/README.md`

### Scripts associes
Tous les scripts du dossier.

### Artefacts produits
Principalement dans `data/` et `docs/reports/`.

### Tests associes
Certains scripts sont couverts directement ou indirectement par les tests rapides.

### Metriques de validation
Scripts critiques identifies dans `scripts/README.md`.

### Statut
stable

### Ne plus toucher
Ne pas deplacer les scripts avant d'avoir adapte imports, docs et tests.

### Amelioration future
Reorganiser progressivement en sous-dossiers avec wrappers de compatibilite.

### Risques
Un nettoyage agressif de scripts peut casser la reproductibilite des preuves.

## Module 27 - Data / artefacts

### Role
Conserver les donnees, artefacts et sorties techniques du MVP.

### Pourquoi ce module existe
Le projet en mode demo depend de fichiers pre-generes pour rester rapide et reproductible.

### Entrees
CV, outputs parser/structuring, reports, datasets, index, models.

### Sorties
Artefacts consommes par API, matching, copilot et rapports.

### Fonctionnement simple
Les modules lisent des fichiers versionnes en mode artefacts et evitent de relancer les pipelines lourds.

### Dossiers et fichiers principaux
- `data/raw_cv/`
- `data/processed_official_module1/`
- `data/profile_builder_module2_v2_grounded_all/`
- `data/indexes/faiss/`
- `data/job_profiles/`
- `data/ranking/`
- `data/graph/`

### Scripts associes
Tous les scripts de generation, seed, matching, ranking et graph.

### Artefacts produits
Index FAISS, id_map, features, datasets, models, job profiles, grounded profiles, graph YAML.

### Tests associes
Tests API, matching, graph, copilot et UI consomment plusieurs artefacts.

### Metriques de validation
90 CV/profils, 75 candidats consolides, 90 profils indexes, 250 lignes ML, 8 roles graph.

### Statut
stable

### Ne plus toucher
Ne pas supprimer les artefacts critiques : FAISS, profiles, features, job profiles, graph YAML, Decision Cards.

### Amelioration future
Separer `official`, `experimental`, `archive` et `cache`.

### Risques
Un nettoyage non controle peut casser la demo et les tests sans toucher au code.
