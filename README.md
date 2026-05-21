# Smart Recruiter — Talent Intelligence Copilot RH

## 1. Présentation

Smart Recruiter est un système intelligent d'aide à la présélection RH. Il transforme des CV bruts en profils structurés, indexe les candidats, applique un matching explicable, enrichit les résultats avec ML/SHAP et Graph-RAG, puis expose un Copilot RH conversationnel via FastAPI, LangChain Tools, LangGraph et Streamlit.

Le système aide le recruteur à analyser plus rapidement un corpus de CV. Il ne remplace pas la décision humaine finale.

## 2. Fonctionnalités principales

- Parsing documentaire de CV.
- Structuration Grounded des profils candidats.
- Stockage MongoDB.
- Indexation et retrieval FAISS.
- Matching V3 explicable, baseline officielle du projet.
- Decision Cards pour expliquer les recommandations.
- Ranking ML expérimental avec Logistic Regression, Random Forest et XGBoost.
- Explicabilité SHAP sur XGBoost.
- Potential Graph YAML pour la transférabilité métier.
- Neo4j Graph-RAG optionnel.
- API FastAPI métier.
- LangChain Tools autour des endpoints API.
- LangGraph Recruiter Copilot.
- Interface Streamlit chatbot.
- Évaluation automatique du Copilot.

## 3. Architecture globale

```text
CV bruts
  -> Parsing
  -> Structuration Grounded
  -> MongoDB / FAISS
  -> Matching V3
  -> Decision Cards
  -> ML / SHAP
  -> Potential Graph / Neo4j Graph-RAG
  -> FastAPI
  -> LangChain Tools
  -> LangGraph Copilot
  -> Streamlit UI
```

## 4. Modules du projet

| Module | Rôle | Statut |
| --- | --- | --- |
| Module 1 Parsing | Convertir les CV bruts en artefacts texte/markdown/json | Terminé |
| Module 2 Structuring | Générer des profils candidats structurés et grounded | Terminé |
| Storage / MongoDB | Stocker `candidate_profiles` et `candidates` | Terminé |
| Retrieval / FAISS | Indexer les profils et récupérer les candidats proches | Terminé |
| Matching V3 | Scoring métier explicable | Baseline officielle |
| Decision Cards | Explication RH des recommandations | Terminé |
| ML / SHAP | Ranking expérimental, RF/XGBoost, SHAP | Expérimental contrôlé |
| Graph / Neo4j | Transférabilité métier et Graph-RAG optionnel | Avancé optionnel |
| API FastAPI | Exposer les capacités métier en JSON | Terminé |
| LangChain Tools | Wrappers tool-ready autour de l'API | Terminé |
| LangGraph Copilot | Workflow conversationnel déterministe | Terminé version démo |
| Streamlit UI | Interface chatbot de démonstration | Terminé |
| Docker | Lancement API + UI + Neo4j optionnel | Ajouté |

## 5. Installation locale

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 6. Lancer l'API FastAPI

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8010
```

Swagger :

```text
http://127.0.0.1:8010/docs
```

## 7. Lancer l'interface Streamlit

```bash
streamlit run ui/streamlit_app.py
```

URL :

```text
http://localhost:8501
```

Dans la sidebar Streamlit, mettre l'URL API :

```text
http://127.0.0.1:8010
```

## 8. Lancer avec Docker

```bash
docker compose up --build
```

URLs :

- API Swagger : http://localhost:8000/docs
- Streamlit Copilot : http://localhost:8501
- Neo4j Browser : http://localhost:7474

Pour arrêter :

```bash
docker compose down
```

## 8.1 Modes de fonctionnement

### Matching V3 baseline

Matching V3 reste la baseline officielle du projet. Les scores `final_score_v3` et les rangs V3 proviennent des rapports et artefacts produits par le pipeline de matching metier.

### Mode demo avec artefacts Matching V3

L'endpoint `POST /api/match` utilise un registre d'artefacts pre-generes dans :

```text
data/ranking/features/*.jsonl
```

Quand le Copilot cree une offre, le `job_router` choisit un `routed_job_id`. Ce `job_id` est transmis a `/api/match`, qui resout l'artefact correspondant si disponible.

La reponse API expose :

- `job_id` : job demande ;
- `resolved_job_id` : job reellement utilise ;
- `artifact_source` : fichier artefact utilise ;
- `matching_mode` : mode de matching ;
- `fallback_used` : indique si un fallback a ete applique ;
- `warnings` : details si fallback.

### Fallbacks

Fallback officiel :

```text
backend_python_django_postgresql
```

Si aucun `job_id` n'est fourni, l'API conserve le comportement historique et utilise ce fallback comme job par defaut, sans considerer cela comme une erreur.

Si un `job_id` inconnu est fourni, l'API retourne `fallback_used=true` et ajoute un warning.

### Modules experimentaux

Les couches Random Forest, XGBoost, SHAP, ML comparison cards et Graph-RAG enrichissent l'analyse, mais ne remplacent pas Matching V3. Les modeles ML restent entraines sur pseudo-labels metier controles, pas sur labels recruteur reels.

## 9. Exemple d'utilisation du Copilot

Flow recommandé pour la démonstration :

```text
nouvelle offre
```

Le Copilot collecte ensuite les 6 champs de l'offre, permet une correction avant confirmation, puis lance le matching avec :

```text
oui lance la recherche
```

Questions de suivi utiles :

```text
Pourquoi le premier candidat ?
Compare le premier et le deuxième candidat
Quels sont les gaps du meilleur candidat ?
```

Le script complet de soutenance se trouve dans :

- `docs/demo/demo_script.md`

## 10. Évaluation

Le protocole d'évaluation du Copilot se trouve dans :

- `data/evaluation/copilot_eval_scenarios.json`
- `scripts/evaluate_copilot.py`
- `docs/reports/copilot/copilot_evaluation.json`
- `docs/reports/copilot/copilot_evaluation.md`

État actuel :

- 14 scénarios évalués ;
- flow offre -> correction -> confirmation -> matching -> suivis couvert ;
- métriques : tool calling accuracy, hallucination-free rate, latence `/api/chat`, cohérence mémoire ;
- fallback Neo4j/YAML et fallback artefact Matching couverts ;
- pas encore de validation recruteur humaine.

## 11. Limites connues

- Les modèles ML sont entraînés sur pseudo-labels métier contrôlés.
- Les pseudo-labels ne sont pas des labels recruteur réels.
- Matching V3 reste la baseline officielle.
- Neo4j est optionnel ; le fallback YAML reste disponible.
- Pas encore de mémoire longue conversationnelle.
- Pas encore de planner LLM avancé.
- Pas encore d'intégration ATS.
- Les données CV sensibles ne doivent pas être versionnées.

## 12. Roadmap future

- Collecter des labels recruteur réels.
- Ajouter une mémoire conversationnelle courte puis longue.
- Ajouter un planner LLM contrôlé.
- Enrichir Neo4j Graph-RAG.
- Créer une interface Next.js.
- Ajouter une intégration ATS.
- Préparer un déploiement cloud.

## 13. Structure du projet

```text
src/
  api/                 # API FastAPI
  core/                # Modules métier : parsing, matching, ranking, graph, chatbot
  benchmark/           # Benchmarks OCR
  models/              # Modèles/schémas applicatifs existants

scripts/               # Scripts d'orchestration, ML, graph, démo, maintenance
data/                  # Données, artefacts, job profiles, ranking, graph
docs/                  # Architecture et rapports
tests/                 # Tests API, graph, ML, demo, copilot
ui/                    # Interface Streamlit
```

## 14. Auteur

Hichem Bensalah  
Projet PFE — Smart Recruiter
