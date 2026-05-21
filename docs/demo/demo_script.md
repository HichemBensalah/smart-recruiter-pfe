# Script de démonstration soutenance - Smart Recruiter

## Objectif

Montrer un MVP RH complet et défendable : le recruteur crée une offre, le Copilot route l'offre vers un `routed_job_id`, lance le matching sur les artefacts Matching V3, affiche les candidats, explique les Decision Cards, répond aux questions de suivi avec mémoire courte et garde un fallback YAML si Neo4j est absent.

## Prérequis

- Python 3.11 recommandé.
- Dépendances installées depuis `requirements.txt`.
- Artefacts présents dans `data/ranking/features/*.jsonl`.
- Rapports Decision Cards présents dans `docs/reports/decision_cards/`.
- Neo4j et MongoDB sont optionnels pour la démo locale.

## Lancement local sans Docker

Terminal 1 :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8010
```

Terminal 2 :

```bash
.venv\Scripts\activate
streamlit run ui/streamlit_app.py
```

Dans la sidebar Streamlit, utiliser :

```text
http://127.0.0.1:8010
```

## Lancement avec Docker Compose

```bash
docker compose up --build
```

URLs à ouvrir :

- Streamlit : http://localhost:8501
- API health : http://localhost:8000/health
- Swagger API : http://localhost:8000/docs
- Neo4j Browser optionnel : http://localhost:7474

Arrêt :

```bash
docker compose down
```

## Vérification avant démo

Ouvrir :

```text
http://localhost:8000/health
```

ou en local sans Docker :

```text
http://127.0.0.1:8010/health
```

Résultat attendu :

- `status = ok`
- dépendances listées sans faire échouer l'endpoint ;
- `matching_artifacts.available = true` ;
- Neo4j peut être indisponible, ce n'est pas bloquant.

## Flow exact dans Streamlit

Premier message exact :

```text
nouvelle offre
```

Répondre ensuite au wizard avec ces messages :

1. Titre du poste

```text
Développeur Backend Python FastAPI MongoDB
```

2. About role

```text
Construire et maintenir des APIs backend pour une plateforme RH.
```

3. Responsabilités

```text
Développer des endpoints FastAPI, intégrer MongoDB, écrire des tests et documenter les APIs.
```

4. Compétences obligatoires

```text
Python, FastAPI, MongoDB, Docker, Git
```

5. Compétences bonus

```text
Neo4j, LangChain, Streamlit, CI GitHub Actions
```

6. Profil recherché

```text
Profil confirmé senior 3 ans minimum, autonome, à Tunis ou remote hybride.
```

Le Copilot doit afficher un résumé de l'offre, un `routed_job_id` et demander confirmation avant matching.

## Correction avant confirmation

Envoyer :

```text
change les competences obligatoires en Python, FastAPI, MongoDB, Docker
```

Résultat attendu :

- le champ compétences obligatoires est remplacé ;
- `structured_job_profile` est reconstruit ;
- `routed_job_id` est recalculé ;
- le matching n'est pas lancé ;
- le Copilot redemande confirmation.

## Confirmation et matching

Message exact :

```text
oui lance la recherche
```

Résultat attendu :

- candidats affichés dans Streamlit ;
- scores Matching V3, Random Forest et XGBoost visibles si disponibles ;
- Decision Cards consultables ;
- transferability et gaps visibles ;
- metadata affichée : `matching_mode`, `resolved_job_id`, `fallback_used`, `warnings`.

## Questions de suivi

Envoyer :

```text
Pourquoi le premier candidat ?
```

Puis :

```text
Compare le premier et le deuxième candidat
```

Puis :

```text
Quels sont les gaps du meilleur candidat ?
```

Résultat attendu :

- le Copilot réutilise la mémoire courte de la session ;
- il résout `premier`, `deuxième` et `meilleur candidat` ;
- il ne relance pas toute la création d'offre ;
- il garde les candidats et Decision Cards du dernier matching.

## Expliquer les métadonnées au jury

- `matching_mode` : indique le mode de matching utilisé. En démo, la valeur normale est `matching_v3_job_artifact`.
- `resolved_job_id` : job réellement utilisé pour choisir l'artefact `data/ranking/features/{job_id}.jsonl`.
- `fallback_used` : `false` si l'artefact demandé existe ; `true` si l'API a dû basculer vers l'artefact officiel de fallback.
- `warnings` : signaux utiles pour expliquer les dégradations contrôlées, par exemple un fallback YAML ou un artefact absent.

Formulation orale simple :

```text
Le système ne cache pas les fallbacks : il expose le mode de matching, l'artefact réellement utilisé et les warnings. C'est important pour défendre un MVP entreprise.
```

## Plan B si Neo4j est absent

Neo4j est optionnel. Si Neo4j n'est pas configuré ou inaccessible :

- `/api/graph/transferability/{candidate_id}` retourne `source = yaml_fallback` ;
- `fallback_used = true` ;
- un warning explique pourquoi Neo4j n'a pas été utilisé ;
- la démo reste fonctionnelle avec le graphe YAML.

Phrase à dire :

```text
Neo4j enrichit l'explication Graph-RAG, mais le MVP reste démontrable sans service externe grâce au fallback YAML.
```

## Plan B si Docker échoue

Utiliser le lancement local sans Docker :

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8010
streamlit run ui/streamlit_app.py
```

Puis configurer Streamlit avec :

```text
http://127.0.0.1:8010
```

Le matching reste basé sur les artefacts locaux, donc MongoDB et Neo4j ne sont pas nécessaires pour terminer la démo.

## Preuve d'évaluation

Avant la soutenance, régénérer le rapport :

```bash
python scripts/evaluate_copilot.py
```

Rapports :

- `docs/reports/copilot/copilot_evaluation.json`
- `docs/reports/copilot/copilot_evaluation.md`

## Message final à dire au jury

```text
Smart Recruiter n'est pas seulement un score de matching. C'est un Copilot RH qui structure l'offre, route vers le bon artefact Matching V3, explique les candidats avec Decision Cards, garde une mémoire courte pour les questions de suivi et expose clairement ses fallbacks. Le MVP est testable, dockerisé, démontrable et honnête sur ses limites.
```
