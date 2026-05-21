# Audit H0 du repository Smart Recruiter

Date d'audit : 2026-05-20  
Objectif : reprendre le contrôle du repository avant les branches entreprise, sans suppression, déplacement, renommage ni modification runtime.

## Résumé global

Le repository est un MVP avancé et fonctionnel, mais il mélange aujourd'hui plusieurs natures de fichiers :

- runtime FastAPI / Streamlit / LangGraph ;
- modules métier historiques et actuels ;
- tests rapides de non-régression ;
- tests plus larges de génération de rapports et d'expérimentation ML ;
- artefacts nécessaires à la démo ;
- artefacts générés lourds ;
- archives et sorties historiques ;
- caches locaux.

Commandes exécutées pendant l'audit :

```bash
python scripts/run_fast_tests.py
pytest --collect-only -q
```

Résultats :

- `python scripts/run_fast_tests.py` : `113 passed in 43.83s`
- `pytest --collect-only -q` : `216 tests collected in 5.21s`

Constats de volume :

- fichiers locaux hors `.git` et `.venv` : `1948`
- taille locale hors `.git` et `.venv` : `407.81 MB`
- fichiers suivis Git : `1641`
- fichiers non suivis mais non ignorés : `24`
- répertoire dominant en taille : `data/indexes/faiss/hf_cache`, environ `349.8 MB`

Le projet est testable et démontrable, mais une branche entreprise doit commencer par stabiliser un socle `stable/mvp-demo`, puis isoler les évolutions lourdes.

## Nombre de fichiers par catégorie

Les catégories ci-dessous sont volontairement fonctionnelles. Certains fichiers peuvent être utiles à plusieurs catégories ; les nombres sont donc des ordres de grandeur de cartographie, pas un plan de suppression.

| Catégorie | Nombre estimé | Commentaire |
| --- | ---: | --- |
| A. Runtime essentiel | ~100 | `src/api`, `src/core`, `ui/streamlit_app.py`, configs racine |
| B. Tests actifs rapides | 24 fichiers / 113 tests | Fichiers inclus dans `scripts/run_fast_tests.py` |
| C. Tests suspects ou redondants | 26 fichiers / 103 tests | Collectés par pytest mais hors fast runner ; surtout ML, rapports et pipelines |
| D. Scripts critiques | ~10 | Démo, évaluation, import, matching, tests rapides |
| E. Scripts expérimentaux | ~25 | ML, SHAP, CrossEncoder, anciens patchs, analyses |
| F. Artefacts nécessaires | ~70 | Matching V3, job profiles, Decision Cards, FAISS core, graph YAML, rapports Copilot |
| G. Artefacts lourds/générés | ~1300 | raw/processed CV, profiles générés, archives, caches Hugging Face |
| H. Documentation active | ~40 | `README`, `README_RUN`, `docs/architecture`, `docs/demo`, rapports actuels |
| I. Legacy / archive possible | ~640 | `data/archive_old_runs`, anciens runs Module 1/2, rapports de patchs |
| J. Suppression possible plus tard | ~300 | caches Python/pytest/temp, `ui/app.py` vide, doublons évidents après validation |

## A. Runtime essentiel

À garder comme socle de fonctionnement du MVP :

- `src/api/main.py`
- `src/api/schemas.py`
- `src/api/utils.py`
- `src/api/routes/*.py`
- `src/core/chatbot/`
- `src/core/chatbot/nodes/`
- `src/core/chatbot/tools/`
- `src/core/chatbot/job_intake.py`
- `src/core/chatbot/job_router.py`
- `src/core/chatbot/memory.py`
- `src/core/chatbot/reference_resolver.py`
- `src/core/graph/`
- `src/core/matching/`
- `src/core/ranking/`
- `src/core/retrieval/`
- `src/core/storage/`
- `src/core/structuring/`
- `src/core/parser/`
- `src/core/jobs/`
- `ui/streamlit_app.py`
- `ui/README.md`
- `requirements.txt`
- `.env.example`
- `Dockerfile`
- `docker-compose.yml`
- `pytest.ini`
- `README.md`
- `README_RUN.md`

Ces fichiers sont essentiels à ne jamais supprimer sans remplacement explicite et tests de non-régression.

## B. Tests actifs

`scripts/run_fast_tests.py` lance 24 fichiers de tests :

```text
tests/test_api_health.py
tests/test_api_chat.py
tests/test_chat_memory.py
tests/test_api_candidates.py
tests/test_api_match.py
tests/test_api_decision_cards.py
tests/test_api_graph.py
tests/test_api_demo.py
tests/test_langchain_tools_api_client.py
tests/test_langchain_tools_registry.py
tests/test_langchain_tools_contracts.py
tests/test_langgraph_copilot_state.py
tests/test_langgraph_copilot_nodes.py
tests/test_langgraph_copilot_graph.py
tests/test_job_intake.py
tests/test_job_intake_field_edit.py
tests/test_job_intake_offer_summary.py
tests/test_job_intake_reset.py
tests/test_job_intake_single_path.py
tests/test_reference_resolver.py
tests/test_streamlit_app_static.py
tests/test_neo4j_transferability.py
tests/test_api_graph_neo4j.py
tests/test_docker_configuration.py
```

Ces fichiers représentent `113` tests collectés et exécutés par le runner rapide. Ils couvrent :

- API FastAPI ;
- mémoire conversationnelle ;
- LangChain Tools ;
- LangGraph Copilot ;
- workflow de création d'offre ;
- resolver de références candidat ;
- Streamlit statique ;
- Neo4j optionnel ;
- Docker configuration.

## C. Tests suspects ou redondants

`pytest --collect-only -q` collecte 50 fichiers de tests et 216 tests. Les tests non inclus dans le runner rapide sont :

| Fichier | Tests | Observation |
| --- | ---: | --- |
| `tests/test_aligned_dataset_generation.py` | 3 | Données/ranking ; utile mais hors smoke suite |
| `tests/test_aligned_job_profiles.py` | 3 | Contrat job profiles |
| `tests/test_annotation_sample.py` | 3 | Données annotation |
| `tests/test_candidate_corpus_analysis.py` | 5 | Génère/valide rapports d'analyse |
| `tests/test_copilot_evaluation.py` | 6 | Important Phase 9 ; candidat à intégrer au fast runner ou CI nightly |
| `tests/test_decision_cards_ml_comparison.py` | 1 | Génération artefact |
| `tests/test_decision_cards_with_transferability.py` | 1 | Génération artefact |
| `tests/test_demo_end_to_end.py` | 1 | Démo end-to-end, plus intégration que smoke |
| `tests/test_demo_executive_summary.py` | 1 | Génération rapport |
| `tests/test_demo_summary.py` | 1 | Génération rapport |
| `tests/test_feature_importance_exports.py` | 3 | ML/XAI, dépend du modèle si présent |
| `tests/test_langgraph_copilot_intents.py` | 11 | Core Copilot ; candidat fort à ajouter au fast runner |
| `tests/test_matching_v3_report_contract.py` | 6 | Contrat artefact Matching V3 |
| `tests/test_ml_experiment_reports.py` | 2 | Rapports ML |
| `tests/test_ml_reranker.py` | 4 | Reranking ML expérimental |
| `tests/test_multi_offer_job_profiles.py` | 2 | Job profiles multi-offres |
| `tests/test_multi_offer_pipeline_config.py` | 4 | Pipeline multi-offres |
| `tests/test_pseudo_label_rule_simulation.py` | 2 | Simulation pseudo-labels |
| `tests/test_pseudo_labels.py` | 5 | Contrat pseudo-labels |
| `tests/test_ranking_dataset.py` | 3 | Dataset ranking |
| `tests/test_ranking_evaluation.py` | 4 | Évaluation ranking |
| `tests/test_ranking_features.py` | 16 | Features ranking ; volumineux mais utile |
| `tests/test_shap_explainability.py` | 5 | SHAP/XAI expérimental |
| `tests/test_train_ranking_models.py` | 4 | Entraînement ML |
| `tests/test_transferability_score.py` | 4 | Graph YAML ; candidat possible fast |
| `tests/test_xgboost_primary_ranking.py` | 3 | XGBoost expérimental |

Interprétation :

- Ces tests ne sont pas mauvais.
- Ils sont suspects au sens H0 car ils ne protègent pas le chemin rapide de CI actuel.
- Plusieurs tests de génération de rapports peuvent modifier des artefacts si mal isolés.
- `test_copilot_evaluation.py`, `test_langgraph_copilot_intents.py`, `test_transferability_score.py` et une partie de `test_ranking_features.py` méritent une décision explicite : fast runner, suite nightly, ou suite ML.

## Comparaison tests collectés vs tests rapides

| Mesure | Valeur |
| --- | ---: |
| Tests collectés par pytest | 216 |
| Fichiers collectés par pytest | 50 |
| Tests réellement lancés par `run_fast_tests.py` | 113 |
| Fichiers lancés par `run_fast_tests.py` | 24 |
| Tests non inclus dans les tests rapides | 103 |
| Fichiers non inclus dans les tests rapides | 26 |
| Fichiers du fast runner non collectés | 0 |

Conclusion : le fast runner est cohérent, mais il ne couvre qu'environ 52% des tests collectés.

## D. Scripts critiques

Scripts ou fichiers d'orchestration à garder et documenter :

- `scripts/run_fast_tests.py` : smoke suite locale/CI rapide.
- `scripts/evaluate_copilot.py` : preuve Phase 9, évaluation Copilot.
- `scripts/run_demo_end_to_end.py` : régénération manifeste et résumés de démo.
- `scripts/check_demo_readiness.py` : vérification de readiness démo.
- `scripts/import_graph_to_neo4j.py` : import Neo4j depuis YAML/profils/jobs.
- `src/core/storage/import_profiles_to_mongodb.py` : import MongoDB actuel, équivalent seed/source initiale.
- `scripts/run_matching_v3_normalized.py` : génération Matching V3 multi-offres.
- `scripts/build_ranking_features.py` : features ranking.
- `scripts/build_decision_cards_with_transferability.py` : Decision Cards enrichies.
- `scripts/build_demo_summary.py` et `scripts/build_demo_executive_summary.py` : synthèses démo.

Point à noter : il n'existe pas encore de script `seed_mongodb.py` explicite. Pour la branche entreprise, il serait préférable de créer un script de seed/import nommé clairement, plutôt que de faire porter ce rôle uniquement à `src/core/storage/import_profiles_to_mongodb.py`.

## E. Scripts expérimentaux ou historiques

Scripts à conserver pour l'instant, mais à isoler plus tard dans `scripts/experiments/`, `scripts/ml/` ou `scripts/legacy/` :

- `scripts/run_cross_encoder_reranking.py`
- `scripts/run_cross_encoder_reranking_constrained.py`
- `scripts/run_ml_reranking.py`
- `scripts/run_xgboost_primary_ranking.py`
- `scripts/explain_xgboost_shap.py`
- `scripts/export_random_forest_feature_importance.py`
- `scripts/export_xgboost_feature_importance.py`
- `scripts/generate_decision_cards_with_ml_experimental.py`
- `scripts/generate_ml_experiment_interpretation.py`
- `scripts/train_ranking_models.py`
- `scripts/simulate_pseudo_label_rules.py`
- `scripts/analyze_candidate_corpus.py`
- `scripts/compare_aligned_offers.py`
- `scripts/patch_grounded_profiles_v2.py`
- `scripts/test_grounded_module2_v2.py`
- `scripts/test_experience_parsing.py`
- `scripts/cleanup_project_artifacts.py`

`scripts/cleanup_project_artifacts.py` est particulièrement sensible : il peut être utile, mais ne doit jamais être utilisé dans une phase entreprise sans dry-run clair, revue manuelle et sauvegarde.

## F. Artefacts nécessaires

Artefacts à garder pour que le MVP et la démo restent reproductibles :

- Matching V3 JSONL : `data/ranking/features/*.jsonl` (`10` fichiers).
- Job profiles : `data/job_profiles/*.json` (`11` fichiers).
- Decision Cards actives :
  - `docs/reports/decision_cards/decision_cards_with_transferability.json`
  - `docs/reports/decision_cards/decision_cards_with_transferability.md`
  - `docs/reports/decision_cards/decision_cards_ml_comparison.json`
  - `docs/reports/decision_cards/decision_cards_ml_comparison.md`
- FAISS core :
  - `data/indexes/faiss/cv_index.faiss`
  - `data/indexes/faiss/id_map.pkl`
  - `data/indexes/faiss/index_report.json`
- Graphe YAML : `data/graph/skills_roles_graph.yaml`
- Évaluation Copilot :
  - `data/evaluation/copilot_eval_scenarios.json`
  - `docs/reports/copilot/copilot_evaluation.json`
  - `docs/reports/copilot/copilot_evaluation.md`
- Rapports de démo :
  - `docs/reports/demo/demo_run_manifest.json`
  - `docs/reports/demo/demo_summary_top10.json`
  - `docs/reports/demo/demo_executive_summary.json`
- Rapports Graph :
  - `docs/reports/graph/neo4j_graph_rag.md`
  - `docs/reports/graph/neo4j_import_report.json`
- Rapports Matching V3 : `docs/reports/matching/v3/*.json`
- Datasets ranking et modèles ML si la partie ML/XGBoost reste démontrée :
  - `data/ranking/datasets/*`
  - `data/ranking/models/*`

## G. Artefacts lourds ou générés

Ces fichiers ne doivent pas être supprimés maintenant, mais ils doivent être explicitement gouvernés :

| Dossier | Fichiers | Taille approx. | Statut |
| --- | ---: | ---: | --- |
| `data/indexes/faiss/hf_cache` | 43 | 349.8 MB | Cache Hugging Face très lourd |
| `data/raw_cv` | 90 | 29.33 MB | Données brutes sensibles |
| `data/processed_official_module1` | 365 | 5.67 MB | Sorties parser générées |
| `data/profile_builder_module2_v2_grounded_all` | 114 | 2.05 MB | Profils utilisés par certaines cards |
| `data/profile_builder_official_module2_rerun_ollama_fixed` | 106 | 0.30 MB | Ancien run officiel |
| `data/archive_old_runs` | 527 | 6.13 MB | Archive historique |
| `data/benchmarks` | 75+ | 6.42 MB | OCR/benchmark |
| `docs/reports` | 89 | 2.66 MB | Rapports générés, utiles mais à classer |

Le cache `data/indexes/faiss/hf_cache` est la cible principale de réduction future si le projet accepte de retélécharger les modèles en local/CI. Pour une démo offline, il peut rester utile.

## H. Documentation active

Documentation à garder et maintenir :

- `README.md`
- `README_RUN.md`
- `ui/README.md`
- `docs/demo/demo_script.md`
- `docs/architecture/current_architecture.md`
- `docs/architecture/pipeline_contracts.md`
- `docs/architecture/artifact_registry.md`
- `docs/architecture/langgraph_copilot_workflow.md`
- `docs/architecture/langchain_tools_contracts.md`
- `docs/architecture/chat_api.md`
- `docs/architecture/docker_demo.md`
- `docs/architecture/ci.md`
- `docs/reports/copilot/copilot_evaluation.md`
- `docs/reports/graph/neo4j_graph_rag.md`
- `docs/reports/demo/demo_guide_current_project.md`

Ce rapport H0 devient la référence de cartographie :

- `docs/architecture/repository_audit.md`

## I. Legacy / archive possible

À ne pas supprimer maintenant, mais candidats à déplacer plus tard vers `archive/` ou stockage externe :

- `data/archive_old_runs/`
- `data/profile_builder_official_module2_rerun_ollama_fixed/`
- `docs/reports/patches/`
- `docs/reports/cross_encoder/`
- `docs/reports/retrieval/`
- anciens rapports ML non utilisés par la démo courante ;
- `data/yyyy.md` et `data/xxxx.md`, qui semblent être des notes manuelles avec mojibake, non référencées par le code ;
- `data/benchmarks/ocr/outputs/` si les benchmarks OCR ne sont plus au centre de la soutenance.

## J. Suppression possible plus tard

À supprimer uniquement dans une phase dédiée de nettoyage, jamais dans H0 :

- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`
- `.tmp/`
- `.tmp_pytest/`
- `ui/app.py` : fichier vide, non référencé.
- caches Hugging Face dans `data/indexes/faiss/hf_cache/`, si la démo offline n'en dépend plus.
- doublons d'anciens runs après validation qu'ils sont archivés ailleurs.

Ces éléments sont déjà en grande partie couverts par `.gitignore`, mais plusieurs existent encore localement et certains artefacts historiques sont suivis Git.

## Fichiers essentiels à ne jamais supprimer

Liste courte de garde-fous :

```text
src/api/main.py
src/api/schemas.py
src/api/utils.py
src/api/routes/
src/core/chatbot/
src/core/graph/
src/core/matching/
src/core/ranking/
src/core/retrieval/
src/core/storage/
src/core/structuring/
ui/streamlit_app.py
scripts/run_fast_tests.py
scripts/evaluate_copilot.py
scripts/run_demo_end_to_end.py
scripts/import_graph_to_neo4j.py
data/ranking/features/
data/job_profiles/
data/indexes/faiss/cv_index.faiss
data/indexes/faiss/id_map.pkl
data/graph/skills_roles_graph.yaml
docs/reports/decision_cards/
docs/reports/copilot/
docs/demo/demo_script.md
Dockerfile
docker-compose.yml
requirements.txt
pytest.ini
.env.example
README.md
README_RUN.md
```

## Fichiers non suivis importants

`git status` montre 24 fichiers non suivis non ignorés. Plusieurs sont essentiels aux phases récentes :

- `.dockerignore`
- `.github/workflows/tests.yml`
- `Dockerfile`
- `README_RUN.md`
- `docker-compose.yml`
- `docs/architecture/ci.md`
- `docs/architecture/docker_demo.md`
- `docs/demo/demo_script.md`
- `docs/reports/graph/neo4j_import_report.json`
- `pytest.ini`
- `scripts/run_fast_tests.py`
- `src/core/chatbot/job_intake.py`
- `src/core/chatbot/job_router.py`
- `src/core/chatbot/memory.py`
- `src/core/chatbot/reference_resolver.py`
- tests Phase 2/3/6 associés.

Recommandation : avant de créer `stable/mvp-demo`, faire une revue Git et inclure ces fichiers si leur contenu correspond bien aux phases validées.

## Structure cible recommandée

Structure cible proposée, à atteindre progressivement :

```text
src/
  api/
  core/
    copilot/
    matching/
    ranking/
    graph/
    storage/
    parsing/
  infrastructure/
    auth/
    mongodb/
    neo4j/

scripts/
  demo/
  evaluation/
  data/
  graph/
  ml/
  maintenance/
  legacy/

tests/
  fast/
  api/
  copilot/
  graph/
  matching/
  ranking/
  data/
  integration/
  experimental/

data/
  runtime/
    ranking/features/
    job_profiles/
    graph/
    indexes/faiss/
  fixtures/
  generated/
  archive/

docs/
  architecture/
  demo/
  reports/
```

Principe : ne pas bouger tout de suite. D'abord stabiliser, puis déplacer en petites PRs avec imports et tests mis à jour.

## Plan de nettoyage en 3 niveaux

### Niveau 1 : safe cleanup

Sans impact fonctionnel attendu :

- supprimer caches locaux `__pycache__`, `*.pyc`, `.pytest_cache`, `.tmp`, `.tmp_pytest` ;
- vérifier que `.env` reste ignoré ;
- documenter clairement les 24 fichiers non suivis à intégrer ou ignorer ;
- confirmer que `ui/app.py` vide n'est référencé nulle part avant suppression future ;
- ajouter une section dans `README_RUN.md` listant les suites de tests : fast, full collect, expérimental.

### Niveau 2 : organisation / archive

Après tag/branche stable :

- déplacer `data/archive_old_runs/` vers une archive externe ou `data/archive/` ;
- déplacer les anciens rapports de patchs vers `docs/reports/archive/` ;
- isoler `data/indexes/faiss/hf_cache/` hors Git si la démo accepte le téléchargement modèle ;
- classer les rapports `cross_encoder`, `retrieval`, `ml` en `experimental`;
- regrouper les scripts en sous-dossiers, avec wrappers temporaires pour compatibilité.

### Niveau 3 : refactor léger tests/scripts

Après nettoyage de niveau 1 et 2 :

- scinder `scripts/run_fast_tests.py` en suites : `fast`, `data`, `ml`, `demo`, `nightly` ;
- intégrer `tests/test_copilot_evaluation.py`, `tests/test_langgraph_copilot_intents.py` et peut-être `tests/test_transferability_score.py` dans une suite rapide ou semi-rapide ;
- marquer explicitement les tests `slow`, `integration`, `experimental`, `requires_external` ;
- créer un vrai `scripts/seed_mongodb.py` ;
- ajouter un script `scripts/enterprise_check.py` qui vérifie auth/config/Mongo/Neo4j avant branches entreprise.

## Risques

- Beaucoup d'artefacts générés sont mélangés avec les artefacts nécessaires à la démo.
- Plusieurs fichiers importants des phases récentes sont encore non suivis Git.
- `data/raw_cv` contient potentiellement des données sensibles.
- `data/indexes/faiss/hf_cache` pèse très lourd et peut gonfler le repo.
- Certains tests hors fast runner peuvent régénérer des rapports ou dépendre de modèles optionnels.
- Les branches entreprise risquent de casser le MVP si elles modifient en même temps auth, MongoDB live, matching live, mémoire et Neo4j.
- Le runner rapide ne couvre pas toute l'évaluation Copilot ni tous les contrats ranking.

## Branches Git recommandées

Séquence recommandée :

1. `stable/mvp-demo`
   - figer l'état actuel validé ;
   - inclure Docker, Phase 1 à 11, tests rapides verts, rapport H0.

2. `enterprise/auth`
   - première branche entreprise ;
   - ajouter authentification sans toucher au matching live.

3. `enterprise/mongodb-live`
   - promouvoir MongoDB en source of truth ;
   - ajouter seed/import clair et migration contrôlée.

4. `enterprise/matching-live`
   - remplacer progressivement les artefacts par matching live ;
   - garder fallback artefact tant que la parité n'est pas prouvée.

5. `enterprise/graph-required` ou sous-branche de `enterprise/matching-live`
   - rendre Neo4j requis seulement après seed, healthcheck et tests d'intégration robustes.

`enterprise/auth` doit partir de `stable/mvp-demo`, pas de branches expérimentales ML.

## Prochaine action recommandée

Action immédiate conseillée :

1. Créer ou préparer la branche `stable/mvp-demo`.
2. Revoir les fichiers non suivis listés par `git status`.
3. Ajouter ce rapport H0.
4. Garder `scripts/run_fast_tests.py` comme gate minimal.
5. Ouvrir ensuite `enterprise/auth` en gardant les fallbacks et artefacts actuels intacts.

Ne pas commencer MongoDB live, matching live ou Neo4j required avant d'avoir figé `stable/mvp-demo`.
