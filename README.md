# Smart Recruiter PFE — Pipeline intelligent de parsing, structuration et matching de CV

Projet de fin d'études autour d'un système d'aide à la présélection RH. Le projet transforme des CV bruts en profils structurés, les stocke, les indexe, puis recommande les meilleurs candidats pour une offre donnée avec un score et une explication.

Le système aide le recruteur à analyser plus vite un corpus de CV. Il ne remplace pas la décision humaine finale.

## Pipeline officiel

```text
CV bruts
  -> Module 1 Parsing
  -> Handoff
  -> Module 2 V2 Grounded
  -> MongoDB
  -> FAISS
  -> Matching V3 normalized
  -> Decision Cards v3 normalized
```

## État actuel

| Brique | Statut | Résultat actuel |
| --- | --- | --- |
| Module 1 Parsing | Opérationnel | 90 CV traités |
| Handoff Module 1 | Opérationnel | 90 accepted, 0 repair_required, 0 quarantined |
| Module 2 V2 Grounded | Opérationnel | 90 profils grounded générés |
| MongoDB | Opérationnel | `candidate_profiles = 90`, `candidates = 75` |
| FAISS | Opérationnel | 90 profils indexés |
| Matching V3 normalized | Opérationnel | 10 recommandations générées |
| Decision Cards v3 normalized | Opérationnel | 10 cartes générées |
| CrossEncoder | Expérimental | testé en reranking, non retenu comme baseline officielle |
| ML / XGBoost / SHAP | Expérimental | pipeline ML fonctionnel, non production-ready |
| API / UI / Chatbot | Non finalisés | hors baseline officielle actuelle |

## Architecture utile du repo

```text
src/
  core/parser/          # Module 1 Parsing documentaire
  core/structuring/     # Module 2 V2 Grounded
  core/storage/         # Import et stockage MongoDB
  core/matching/        # FAISS, scoring métier, Matching V3
  core/retrieval/       # Expériences CrossEncoder
  benchmark/            # Benchmark OCR isolé

scripts/                # Scripts d'exécution, rapports, ML, SHAP
docs/reports/           # Rapports de démo, matching, retrieval, ML
data/job_descriptions/  # Offres au format texte
data/job_profiles/      # Offres structurées en JSON
data/indexes/faiss/     # Index FAISS, id_map et rapport d'indexation
```

## Modules

### Module 1 — Parsing documentaire

Le Module 1 lit les CV bruts et produit des artefacts documentaires exploitables par la suite du pipeline.

- Parseur principal : `Docling`
- Fallback OCR : `PyTesseract`
- Formats d'artefacts : Markdown, texte, JSON, HTML
- Évaluation qualité documentaire
- Handoff vers trois files : `accepted`, `repair_required`, `quarantined`

Résultat actuel : les 90 CV du corpus de démo passent en `accepted`.

### Module 2 — Structuration Grounded

Le Module 2 lit le Markdown produit par le Module 1 et génère des profils candidats structurés.

- Entrée : Markdown Module 1
- LLM via Groq ou Ollama local
- Validation par schémas Pydantic
- Validation grounded pour limiter les hallucinations
- Calcul de `reliability_score`
- Signalement de `hallucination_risk`

Résultat actuel : 90 profils grounded générés.

### MongoDB

MongoDB sert au stockage applicatif des profils structurés.

- Base : `talent_intelligence`
- Collections principales :
  - `candidate_profiles`
  - `candidates`

État actuel observé : `candidate_profiles = 90`, `candidates = 75`.

### FAISS

FAISS est le moteur officiel de retrieval vectoriel du projet.

- Modèle d'embedding : `sentence-transformers/all-MiniLM-L6-v2`
- Index : `data/indexes/faiss/cv_index.faiss`
- Mapping : `data/indexes/faiss/id_map.pkl`
- Rapport : `data/indexes/faiss/index_report.json`

État actuel : 90 profils indexés.

### Matching V3 normalized

Matching V3 normalized combine retrieval FAISS et scoring métier.

Il prend en compte notamment :

- `must_have_coverage`
- `matched_skills`
- `missing_required_skills`
- `reliability_score`
- `hallucination_risk`
- pénalités métier
- score final normalisé

La baseline officielle actuelle est :

```text
FAISS -> Matching V3 normalized -> Decision Cards v3 normalized
```

### Decision Cards v3 normalized

Les Decision Cards transforment les résultats de matching en cartes lisibles pour un recruteur.

Chaque carte contient notamment :

- verdict
- score
- forces
- faiblesses
- points à vérifier en entretien (`interview_focus`)

Ces cartes ne sont pas une décision automatique finale. Elles servent d'aide à l'analyse.

### CrossEncoder

Le CrossEncoder a été testé comme reranking expérimental.

- Version non contrainte : dangereuse car elle peut remonter des candidats non adaptés.
- Version contrainte : plus sûre, mais trop restrictive dans les résultats observés.
- Décision actuelle : le CrossEncoder reste expérimental et n'est pas la baseline officielle.

### ML expérimental

Une première couche ML expérimentale existe pour tester la faisabilité d'un apprentissage supervisé contrôlé.

Elle inclut :

- Feature Builder
- pseudo-labels métier contrôlés
- Logistic Regression
- Random Forest
- XGBoost
- rapports SHAP sur XGBoost

Important : ces modèles ne sont pas production-ready. Les pseudo-labels ne sont pas des labels recruteur. Les résultats ML valident surtout le pipeline expérimental, pas une supériorité réelle sur des décisions humaines.

## Démo actuelle

Fichiers utiles pour vérifier rapidement l'état de la démo :

- `docs/reports/demo/demo_readiness_check.json`
- `data/processed_official_module1/handoff/accepted.json`
- `data/indexes/faiss/index_report.json`
- `data/job_profiles/backend_python_fastapi_mongodb.json`
- `docs/reports/matching/v3/matching_report_v3_normalized.json`
- `docs/reports/matching/v3/decision_cards_v3_normalized.json`
- `docs/reports/retrieval/faiss_cross_encoder_ablation_summary.json`

## Commandes principales

Module 1 Parsing :

```bash
python src/core/parser/run_docling_pipeline.py
```

Le script utilise les chemins configurables par variables d'environnement, notamment `MODULE1_RAW_ROOT`, `MODULE1_OUT_ROOT` et `MODULE1_REPORT_PATH`.

Module 2 V2 Grounded :

```bash
python -m src.core.structuring.profile_builder_grounded --run-all --resume
```

Matching V3 normalized :

```bash
python scripts/run_matching_v3_normalized.py \
  --job-profile data/job_profiles/backend_python_fastapi_mongodb.json \
  --output-report docs/reports/matching/v3/matching_report_v3_normalized.json \
  --top-k 10
```

Decision Cards v3 normalized :

```bash
python scripts/generate_decision_cards_v3_normalized.py
```

Demo readiness check :

```bash
python scripts/check_demo_readiness.py
```

Pour les autres expérimentations, voir `scripts/`.

## Ce qui n'est pas finalisé

- API complète
- UI complète
- Chatbot RH
- labels recruteur
- ML supervisé final
- Docker complet
- CI complet

## Décisions importantes

- FAISS est le retrieval officiel.
- Qdrant n'est pas utilisé dans la baseline actuelle.
- Matching V3 normalized est la baseline de matching officielle.
- Decision Cards v3 normalized est la sortie explicative officielle actuelle.
- CrossEncoder reste expérimental.
- XGBoost et SHAP restent expérimentaux sans labels humains.
- Le système assiste le recruteur, mais ne le remplace pas.

## Note GitHub

Certaines données lourdes ou sensibles peuvent être ignorées selon `.gitignore`. Le dépôt sert à présenter le code, les rapports, les résultats et l'état actuel du pipeline.

