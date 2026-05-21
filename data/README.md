# `data/`

Données, artefacts intermédiaires et sorties techniques du pipeline Smart Recruiter.

Ce dossier mélange des artefacts critiques pour le MVP, des sorties générées, des archives et des données potentiellement sensibles. Ne pas nettoyer ce dossier sans rapport d'audit et sans sauvegarde.

## Artefacts critiques à garder

- `ranking/features/*.jsonl` : artefacts Matching V3 utilisés par `/api/match`.
- `job_profiles/*.json` : offres structurées et job profiles routables.
- `graph/skills_roles_graph.yaml` : fallback YAML pour la transferability.
- `indexes/faiss/cv_index.faiss` : index FAISS de démonstration.
- `indexes/faiss/id_map.pkl` : mapping FAISS vers profils.
- `indexes/faiss/index_report.json` : rapport de l'index.
- `evaluation/copilot_eval_scenarios.json` : scénarios d'évaluation Copilot.
- `profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/` : profils utilisés par plusieurs artefacts et Decision Cards.

## Données et sorties importantes

- `processed_official_module1/` : sorties officielles du parsing Module 1.
- `profile_builder_module2_v2_grounded_all/` : profils grounded Module 2.
- `ranking/datasets/` : datasets ranking et pseudo-labels.
- `ranking/models/` : modèles et feature names ML expérimentaux.
- `skills_taxonomy.yaml` : taxonomie de compétences.
- `skills_audit/` : audit des variantes de compétences.

## Données sensibles ou lourdes

- `raw_cv/` : CV bruts locaux. À ne pas pousser si données sensibles.
- `archive_old_runs/` : anciens runs conservés pour historique.
- `benchmarks/` : sorties de benchmarks OCR.
- `profile_builder_official_module2_rerun_ollama_fixed/` : ancien run officiel.

## Cache Hugging Face / FAISS

`data/indexes/faiss/hf_cache/` est lourd, environ 350 MB lors de l'audit H0.

Ne pas le supprimer dans H1. Il peut être externalisé plus tard si :

- les modèles peuvent être retéléchargés ;
- la démo n'exige pas de mode offline ;
- les tests ne dépendent pas de ce cache local ;
- un plan de restauration est documenté.

La règle `.gitignore` empêche de nouveaux fichiers de cache d'être ajoutés par accident, mais elle ne retire pas les fichiers déjà suivis.

## Règles

- Ne pas modifier manuellement les datasets, modèles ou index.
- Ne pas supprimer d'artefact nécessaire à Matching V3, Decision Cards, FAISS ou Copilot.
- Les changements doivent passer par les scripts dédiés et être documentés.
- Les nettoyages doivent être faits dans une phase séparée, avec `python scripts/run_fast_tests.py` avant et après.
