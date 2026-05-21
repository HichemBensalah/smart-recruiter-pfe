# Plan H1 - Stabilisation Git et Safe Cleanup

Date : 2026-05-20  
Objectif : stabiliser le MVP actuel avant les branches entreprise, sans changement runtime ni suppression.

## Résumé

La phase H0 a confirmé que le projet est fonctionnel mais que plusieurs fichiers essentiels des phases récentes sont encore non suivis Git. H1 ne supprime rien et ne déplace rien. Le but est de clarifier ce qui doit être ajouté, ignoré, vérifié ou seulement supprimé plus tard.

Commandes de référence :

```bash
git status --short
python scripts/run_fast_tests.py
```

Dernier résultat attendu de stabilisation :

- `python scripts/run_fast_tests.py` doit rester vert.
- Les fichiers essentiels non suivis doivent être ajoutés dans un commit `stable/mvp-demo`.
- Les caches et temporaires doivent rester ignorés.

## État Git observé

Le dépôt contient beaucoup de fichiers déjà modifiés par les phases précédentes. H1 ne les requalifie pas tous ; ce plan se concentre sur les fichiers non suivis et sur la stabilisation sûre.

Fichiers non suivis observés :

```text
.dockerignore
.github/workflows/tests.yml
Dockerfile
README_RUN.md
docker-compose.yml
docs/architecture/ci.md
docs/architecture/docker_demo.md
docs/architecture/repository_audit.md
docs/demo/demo_script.md
docs/reports/graph/neo4j_import_report.json
pytest.ini
scripts/run_fast_tests.py
src/core/chatbot/job_intake.py
src/core/chatbot/job_router.py
src/core/chatbot/memory.py
src/core/chatbot/reference_resolver.py
tests/test_chat_memory.py
tests/test_docker_configuration.py
tests/test_job_intake.py
tests/test_job_intake_field_edit.py
tests/test_job_intake_offer_summary.py
tests/test_job_intake_reset.py
tests/test_job_intake_single_path.py
tests/test_langgraph_copilot_intents.py
tests/test_reference_resolver.py
```

Tous les fichiers récents explicitement demandés pour vérification existent.

## À ajouter Git immédiatement

Ces fichiers sont essentiels au MVP actuel ou à sa reproductibilité :

### Docker / CI / lancement

- `.dockerignore`
- `.github/workflows/tests.yml`
- `Dockerfile`
- `docker-compose.yml`
- `pytest.ini`
- `README_RUN.md`

Justification : Docker Compose, CI rapide et runner pytest font partie des phases validées.

### Documentation active

- `docs/architecture/ci.md`
- `docs/architecture/docker_demo.md`
- `docs/architecture/repository_audit.md`
- `docs/architecture/git_stabilization_plan.md`
- `docs/demo/demo_script.md`

Justification : documentation active H0/H1, Docker, CI et script soutenance.

### Rapports actifs

- `docs/reports/graph/neo4j_import_report.json`

Justification : preuve Phase 5, import Neo4j documenté et vérifié par tests.

### Scripts critiques

- `scripts/run_fast_tests.py`

Justification : gate rapide officiel du MVP.

### Modules Copilot récents

- `src/core/chatbot/job_intake.py`
- `src/core/chatbot/job_router.py`
- `src/core/chatbot/memory.py`
- `src/core/chatbot/reference_resolver.py`

Justification : phases 2 et 3. Ces fichiers sont runtime, mais ils existent déjà et doivent être suivis pour figer le MVP ; H1 ne les modifie pas.

### Tests récents actifs

- `tests/test_chat_memory.py`
- `tests/test_docker_configuration.py`
- `tests/test_job_intake.py`
- `tests/test_job_intake_field_edit.py`
- `tests/test_job_intake_offer_summary.py`
- `tests/test_job_intake_reset.py`
- `tests/test_job_intake_single_path.py`
- `tests/test_langgraph_copilot_intents.py`
- `tests/test_reference_resolver.py`

Justification : tests des phases mémoire, intake, Docker et resolver. Plusieurs sont inclus dans `scripts/run_fast_tests.py`; `test_langgraph_copilot_intents.py` est hors fast runner mais utile pour le contrat Copilot.

## À ignorer via `.gitignore`

La règle actuelle couvre déjà :

- `.venv/`
- `__pycache__/`
- `*.py[cod]`
- `*.pyc`
- `.env`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- `.tmp/`
- `.tmp_pytest/`
- logs

H1 ajoute ou confirme l'ignorance de :

- `.cache/`
- `.coverage`
- `coverage.xml`
- `htmlcov/`
- `build/`
- `dist/`
- `*.egg-info/`
- `*.tmp`
- `*.bak`
- `*.swp`
- `~$*`
- `data/indexes/faiss/hf_cache/`

Note importante : `data/indexes/faiss/hf_cache/` contient des fichiers déjà suivis Git. La règle `.gitignore` empêche surtout d'ajouter de nouveaux fichiers de cache par accident ; elle ne supprime ni ne désindexe les fichiers existants.

## À vérifier manuellement

Avant commit, revoir les fichiers déjà modifiés mais suivis Git. Ils appartiennent aux phases 7, 8, 9, 11 ou à la stabilisation H1 :

- API FastAPI durcie : `src/api/**`
- Copilot/LangGraph : `src/core/chatbot/**`
- Graph Neo4j : `src/core/graph/neo4j_transferability.py`
- Tests API/Copilot/Streamlit modifiés
- Rapports Copilot et demo régénérés
- `README.md`, `ui/README.md`, `README_RUN.md`
- `requirements.txt`, `.env.example`

Vérification recommandée :

```bash
git diff --stat
git diff -- README.md README_RUN.md ui/README.md
git diff -- src/api src/core/chatbot src/core/graph
git diff -- tests
```

Ne pas faire de commit avant d'avoir confirmé que ces changements correspondent bien aux phases validées.

## À supprimer plus tard seulement après validation

Ne rien supprimer dans H1. Candidats futurs, après `stable/mvp-demo` :

- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`
- `.tmp/`
- `.tmp_pytest/`
- `ui/app.py` vide, après confirmation qu'il n'est pas référencé.
- caches Hugging Face `data/indexes/faiss/hf_cache/`, uniquement si externalisation validée.
- anciens runs dans `data/archive_old_runs/`, uniquement après sauvegarde/archive externe.
- notes non référencées `data/xxxx.md` et `data/yyyy.md`, après revue manuelle.

## Fichiers récents essentiels vérifiés

Tous présents :

```text
Dockerfile
docker-compose.yml
.dockerignore
.github/workflows/tests.yml
pytest.ini
README_RUN.md
docs/architecture/ci.md
docs/architecture/docker_demo.md
docs/demo/demo_script.md
docs/reports/graph/neo4j_import_report.json
scripts/run_fast_tests.py
src/core/chatbot/job_intake.py
src/core/chatbot/job_router.py
src/core/chatbot/memory.py
src/core/chatbot/reference_resolver.py
tests/test_chat_memory.py
tests/test_docker_configuration.py
tests/test_job_intake.py
tests/test_job_intake_field_edit.py
tests/test_job_intake_offer_summary.py
tests/test_job_intake_reset.py
tests/test_job_intake_single_path.py
tests/test_langgraph_copilot_intents.py
tests/test_reference_resolver.py
```

## Commandes Git recommandées

Ne pas exécuter automatiquement sans revue humaine.

Prévisualisation :

```bash
git status --short
git diff --stat
```

Ajout recommandé pour stabiliser le MVP :

```bash
git add .dockerignore .github/workflows/tests.yml Dockerfile docker-compose.yml pytest.ini README_RUN.md
git add docs/architecture/ci.md docs/architecture/docker_demo.md docs/architecture/repository_audit.md docs/architecture/git_stabilization_plan.md docs/demo/demo_script.md docs/reports/graph/neo4j_import_report.json
git add scripts/run_fast_tests.py
git add src/core/chatbot/job_intake.py src/core/chatbot/job_router.py src/core/chatbot/memory.py src/core/chatbot/reference_resolver.py
git add tests/test_chat_memory.py tests/test_docker_configuration.py tests/test_job_intake.py tests/test_job_intake_field_edit.py tests/test_job_intake_offer_summary.py tests/test_job_intake_reset.py tests/test_job_intake_single_path.py tests/test_langgraph_copilot_intents.py tests/test_reference_resolver.py
git add .gitignore tests/README.md scripts/README.md data/README.md
```

Puis, après revue des modifications déjà suivies :

```bash
python scripts/run_fast_tests.py
git commit -m "chore: stabilize MVP demo repository"
```

## Branche recommandée

Créer ensuite :

```bash
git checkout -b stable/mvp-demo
```

ou, si le commit est fait sur la branche courante, tagger/figer cette branche puis créer :

```bash
git checkout -b enterprise/auth
```

Ordre recommandé :

1. Stabiliser et committer le MVP.
2. Créer `stable/mvp-demo`.
3. Démarrer `enterprise/auth`.
4. Reporter `enterprise/mongodb-live`, `enterprise/matching-live` et `enterprise/neo4j-required` après auth.

## Risques H1

- Plusieurs fichiers essentiels sont encore non suivis ; oublier de les ajouter casserait une reprise propre.
- Le cache Hugging Face est lourd et déjà suivi ; ne pas le traiter dans H1.
- Les modifications suivies existantes sont nombreuses ; il faut relire le diff avant commit.
- Ne pas mélanger stabilisation Git avec refactor ou fonctionnalités entreprise.

## Conclusion

H1 doit produire un commit de stabilisation, pas un nettoyage agressif. La priorité est de figer le MVP actuel avec ses tests rapides verts, sa documentation de démo, son Docker Compose, son runner CI et les fichiers Copilot récents.
