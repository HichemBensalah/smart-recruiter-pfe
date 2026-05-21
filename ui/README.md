# Interface Streamlit - Smart Recruiter Copilot RH

Cette interface est une démonstration simple du Recruiter Copilot. Elle appelle uniquement l'endpoint FastAPI `POST /api/chat`.

## Option A - FastAPI sur 8000

Lancer FastAPI :

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

URL API dans Streamlit :

```text
http://localhost:8000
```

## Option B - FastAPI sur 8010

Lancer FastAPI :

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8010
```

URL API dans Streamlit :

```text
http://127.0.0.1:8010
```

## Lancer Streamlit

```bash
streamlit run ui/streamlit_app.py
```

## Flow recommandé pour la démo

Premier message exact :

```text
nouvelle offre
```

Répondre ensuite aux 6 étapes du wizard :

```text
Développeur Backend Python FastAPI MongoDB
```

```text
Construire et maintenir des APIs backend pour une plateforme RH.
```

```text
Développer des endpoints FastAPI, intégrer MongoDB, écrire des tests et documenter les APIs.
```

```text
Python, FastAPI, MongoDB, Docker, Git
```

```text
Neo4j, LangChain, Streamlit, CI GitHub Actions
```

```text
Profil confirmé senior 3 ans minimum, autonome, à Tunis ou remote hybride.
```

Correction avant confirmation :

```text
change les competences obligatoires en Python, FastAPI, MongoDB, Docker
```

Confirmation :

```text
oui lance la recherche
```

Questions de suivi :

```text
Pourquoi le premier candidat ?
```

```text
Compare le premier et le deuxième candidat
```

```text
Quels sont les gaps du meilleur candidat ?
```

Le script complet de soutenance est disponible dans `docs/demo/demo_script.md`.

## Ce que l'interface affiche

- la réponse conversationnelle du Copilot ;
- l'état de session et la progression du wizard ;
- le résumé de l'offre et le `routed_job_id` ;
- les candidats recommandés ;
- les scores Matching V3, Random Forest et XGBoost si disponibles ;
- les métadonnées `matching_mode`, `resolved_job_id`, `fallback_used` et `warnings` ;
- les Decision Cards si présentes ;
- la transferability et les gaps si présents.

## Problème fréquent : API non joignable

Si Streamlit affiche une erreur du type `Failed to establish a new connection`, l'API FastAPI n'est probablement pas lancée sur l'URL configurée dans la sidebar.

Solution :

1. Lancer FastAPI dans un premier terminal.
2. Lancer Streamlit dans un deuxième terminal.
3. Vérifier dans la sidebar Streamlit que l'URL API correspond au port FastAPI.

Le champ URL API reste éditable : utilisez `http://localhost:8000` ou `http://127.0.0.1:8010` selon votre commande FastAPI.

## Limites actuelles

- Pas encore de mémoire longue.
- Pas encore de planner LLM.
- Neo4j est optionnel ; le workflow garde le fallback YAML.
- Matching V3 utilise des artefacts pré-générés pour garder la démo rapide et reproductible.
- Les réponses sont basées sur le workflow LangGraph et les tools exposés par l'API.
- Streamlit ne lance aucun script et n'appelle pas Matching V3 directement.
