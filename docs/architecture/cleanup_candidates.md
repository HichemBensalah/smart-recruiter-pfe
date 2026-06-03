# Smart Recruiter - Cleanup Candidates

Ce document propose des candidats au nettoyage. Il ne demande aucune suppression immediate.

Regle : avant toute action, faire un dry-run, verifier les references, sauvegarder si necessaire, puis lancer `python scripts/run_fast_tests.py`.

| Fichier/dossier | Type | Pourquoi suspect | Risque si supprime | Recommandation |
| --- | --- | --- | --- | --- |
| `__pycache__/` | cache supprimable | Cache Python genere automatiquement | Faible | Supprimable apres tests |
| `scripts/__pycache__/` | cache supprimable | Cache Python genere automatiquement | Faible | Supprimable |
| `ui/__pycache__/` | cache supprimable | Cache Python genere automatiquement | Faible | Supprimable |
| `.pytest_cache/` | cache supprimable | Cache local pytest | Faible | Supprimable |
| `.tmp/` | cache supprimable | Dossier temporaire de tests | Faible a moyen selon contenu | Verifier puis supprimer si uniquement temporaire |
| `.tmp_pytest/` | cache supprimable | Dossier temporaire pytest | Faible | Supprimable si non reference |
| `ui/app.py` | fichier vide | Fichier de longueur 0, non utilise par l'app principale | Faible | Supprimer ou documenter apres verification |
| `ui/__init__.py` | fichier vide | Peut etre inutile, mais peut aider les imports | Faible | Garder sauf raison claire |
| `data/archive_old_runs/` | ancien rapport / archive | Anciennes sorties volumineuses hors flux MVP | Moyen | Archiver hors flux actif, ne pas supprimer brutalement |
| `data/profile_builder_official_module2_rerun_ollama_fixed/` | ancien run a archiver | Ancien run de structuration | Moyen | Archiver apres verification des references |
| `data/benchmarks/` | dossier a documenter | Sorties benchmark OCR probablement historiques | Moyen | Garder ou archiver selon rapport OCR |
| `data/indexes/faiss/hf_cache/` | cache volumineux | Cache Hugging Face, environ 350 MB dans audit precedent | Moyen | Ne pas supprimer si demo offline necessaire |
| `docs/reports/patches/` | ancien rapport a archiver | Rapports de correction Module 2 | Moyen | Garder comme trace ou archiver |
| `docs/reports/cross_encoder/` | script/rapport experimental a garder | Rapports d'experimentation CrossEncoder | Moyen | Garder, classer experimental |
| `docs/reports/retrieval/` | rapport experimental a garder | Ablations retrieval/CrossEncoder | Moyen | Garder, classer experimental |
| `docs/reports/ml/` | rapport experimental a garder | Preuves ML, SHAP, pseudo-labels | Eleve pour le rapport | Garder, ne pas supprimer |
| `data/ranking/models/` | artefact critique experimental | Modeles LR/RF/XGB utilises pour rapports ML | Moyen | Garder avant soutenance |
| `data/ranking/datasets/` | artefact critique experimental | Dataset ML et pseudo-labels | Moyen | Garder |
| `data/ranking/features/` | artefact critique | Source Matching V3 API artifact mode | Eleve | Ne pas supprimer |
| `data/job_profiles/` | artefact critique | Job profiles routables et matching | Eleve | Ne pas supprimer |
| `data/graph/skills_roles_graph.yaml` | artefact critique | Fallback stable transferability | Eleve | Ne pas supprimer |
| `data/indexes/faiss/cv_index.faiss` | artefact critique | Index retrieval | Eleve | Ne pas supprimer |
| `data/indexes/faiss/id_map.pkl` | artefact critique | Mapping index vers profils | Eleve | Ne pas supprimer |
| `data/profile_builder_module2_v2_grounded_all/` | artefact critique | Profils grounded utilises partout | Eleve | Ne pas supprimer |
| `data/processed_official_module1/` | artefact critique | Sorties officielles parsing | Eleve | Ne pas supprimer |
| `docs/reports/matching/v3/` | artefact critique | Preuves Matching V3 | Eleve | Ne pas supprimer |
| `docs/reports/decision_cards/` | artefact critique | Decision Cards consommees par API/Copilot | Eleve | Ne pas supprimer |
| `scripts/run_cross_encoder_reranking.py` | script experimental a garder | Reproductibilite CrossEncoder | Faible a moyen | Garder, documenter experimental |
| `scripts/train_ranking_models.py` | script experimental a garder | Reproductibilite ML | Moyen | Garder, ne pas lancer avant soutenance |
| `scripts/explain_xgboost_shap.py` | script experimental a garder | Reproductibilite SHAP | Moyen | Garder |
| `scripts/cleanup_project_artifacts.py` | script sensible | Peut nettoyer des artefacts | Eleve | Garder mais utiliser seulement en dry-run controle |
| `.env` | fichier sensible | Variables locales potentiellement secretes | Eleve | Ne pas partager ; ne pas supprimer sans backup |
| `.env.example` | documentation config | Exemple necessaire au lancement | Eleve | Ne pas supprimer |
| `Dockerfile` | configuration demo | Necessaire Docker | Moyen | Garder |
| `docker-compose.yml` | configuration demo | Services API/UI/MongoDB/Neo4j | Moyen | Garder |
| `.dockerignore` | configuration build | Evite d'inclure caches/data sensibles | Moyen | Garder |

## Recommandation de nettoyage en phases

### Phase 1 - Sans risque

- Supprimer uniquement caches Python et pytest.
- Verifier que `python scripts/run_fast_tests.py` passe.

### Phase 2 - Archive, pas suppression

- Archiver anciens runs et anciens rapports hors flux actif.
- Garder un index de ce qui a ete archive.
- Relancer les tests rapides.

### Phase 3 - Clarification documentaire

- Marquer les dossiers experimentaux dans README/docs.
- Ne pas modifier les artefacts critiques.

### Phase 4 - Nettoyage lourd eventuel

- A faire seulement apres soutenance ou sur branche dediee.
- Inclure sauvegarde, verification des chemins et tests avant/apres.
