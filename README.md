# Smart Recruiter PFE

Base de travail alignée sur le cahier des charges de référence :
`Système de Recommandation de CV Intelligent & Assistant RH Conversationnel`.

## État actuel

Le projet n'implémente pas encore tout le périmètre cible du cahier des charges.
La base réellement utile aujourd'hui est :

1. `Module 1` : parsing documentaire, fallback, évaluation qualité, handoff policy.
2. `Module 2` : structuration contrôlée en `dry-run`, validation Pydantic, previews JSON, rapport de run.

Les briques `API FastAPI`, `matching hybride`, `chatbot RH` et `Neo4j / LangGraph` ne sont pas encore construites.
Elles doivent être développées sur une base propre, et non simulées par des fichiers vides.

## Structure utile du repo

```text
src/
  core/
    parser/        # Module 1
    structuring/   # Module 2
data/
  raw_cv/          # CV bruts
  processed/       # Artefacts Module 1
```

## Fichiers clés

- `src/core/parser/run_docling_pipeline.py`
- `src/core/parser/handoff_policy.py`
- `src/core/structuring/profile_builder.py`
- `data/processed/module1_pipeline_report.json`
- `data/processed/handoff/accepted.json`

## Ce qui est considéré comme référence

- Le cahier des charges PDF fourni par l'utilisateur
- Le handoff Module 1 vers Module 2 via `data/processed/handoff/accepted.json`
- Le fonctionnement de `profile_builder.py` en `--dry-run`

## Ce qui a été volontairement nettoyé

- scripts de test anciens ou redondants
- fichiers vides donnant une fausse impression d'implémentation
- sorties de tests obsolètes ou ratées
- dépendances non utilisées dans la base actuelle

## Priorité de développement

1. Stabiliser la qualité de Module 2.
2. Valider un lot de test plus large en `dry-run`.
3. Construire ensuite l'API.
4. Ajouter ensuite le matching hybride.
5. Ajouter enfin la couche conversationnelle outillée.
