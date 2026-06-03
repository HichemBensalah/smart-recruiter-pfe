# Demo Contract — Smart Recruiter

## 1. Objectif du contrat

Ce document fixe le scénario produit officiel de Smart Recruiter, tel que validé par les tests end-to-end automatisés. Il sert de référence unique pour les démonstrations, les validations QA et les décisions d'évolution du produit. Toute régression sur ce scénario doit être détectée et bloquée avant mise en production.

---

## 2. Scénario principal validé

Le parcours officiel de démonstration suit les étapes suivantes :

1. Le recruteur envoie une demande initiale (recherche ou création d'offre).
2. Le Copilot détecte l'intention et lance le **Job Intake Wizard**.
3. Le recruteur remplit les **6 champs obligatoires**, un par message :
   - `job_title`
   - `about_role`
   - `responsibilities`
   - `required_skills`
   - `bonus_skills`
   - `profile`
4. Le système construit automatiquement le **`structured_job_profile`**.
5. Le système route vers un **`job_id` existant** via `job_router.py`.
6. Le système affiche un résumé de l'offre et **demande confirmation** avant tout matching.
7. Après confirmation explicite du recruteur, le **matching est lancé**.
8. Les **candidats sont affichés** avec leurs scores et statuts.
9. Les **questions de suivi** (« Pourquoi le premier candidat ? », « Quels sont ses gaps ? ») utilisent la mémoire de session — aucun matching supplémentaire n'est déclenché.

---

## 3. Offre de référence principale

L'offre ci-dessous est l'offre de référence utilisée dans le scénario principal E2E.

**Job title :**
Backend Python Engineer

**About the role :**
We are looking for a Backend Python Engineer to join our product engineering team working on a recruitment platform used by HR teams and hiring managers.

**Responsibilities :**
- Design, build and maintain REST APIs.
- Develop backend services with Python and FastAPI.
- Model and query application data in MongoDB.
- Collaborate with frontend developers, product managers and QA.
- Improve performance, monitoring and deployment quality.

**Required skills :**
- Python
- FastAPI
- MongoDB
- Docker
- REST API design

**Bonus skills :**
- CI/CD
- AWS

**Profile :**
At least 3 years of experience, mid-level, English, Tunis, hybrid.

---

## 4. `structured_job_profile` attendu

Après remplissage des 6 champs, le système construit automatiquement le profil structuré suivant :

```json
{
  "job_title": "Backend Python Engineer",
  "target_role": "Backend Developer",
  "required_skills": ["Python", "FastAPI", "MongoDB", "Docker", "REST API design"],
  "nice_to_have_skills": ["CI/CD", "AWS"],
  "min_years_experience": 3,
  "seniority": "mid-level",
  "location": "Tunis",
  "work_model": "hybrid",
  "language_requirements": ["English"]
}
```

Ce profil est construit par `job_intake.build_structured_job_profile()` à partir des champs du wizard. Il est exposé dans la réponse `ChatResponse.structured_job_profile`.

---

## 5. `job_id` attendu pour le scénario principal

Le `job_id` attendu pour l'offre de référence est :

```
backend_python_fastapi_mongodb_aligned
```

**Justification :**
- Le `job_router.infer_job_route_from_structured_profile()` détecte la combinaison `python + fastapi + mongodb` dans les compétences (required ∪ bonus).
- Il utilise `_first_existing_job_id(["backend_python_fastapi_mongodb_aligned", "backend_python_fastapi_mongodb"])` et retourne le premier fichier trouvé dans `data/job_profiles/`.
- Le fichier `data/job_profiles/backend_python_fastapi_mongodb_aligned.json` existe → job_id retenu.
- Un **fallback** (`backend_python_django_postgresql`, confiance 0.45) existe pour les offres ne correspondant à aucune règle connue.

---

## 6. Routing validé

La table suivante documente les règles de routing réelles déduites de `job_router.py` et les résultats validés par les tests E2E :

| Type d'offre | Signaux détectés | `job_id` attendu | Confiance | Statut |
|---|---|---|---|---|
| Backend FastAPI MongoDB | `python` + `fastapi` + `mongodb` ⊆ skills | `backend_python_fastapi_mongodb_aligned` | 0.95 | ✅ validé |
| Backend Django PostgreSQL | `django` + `postgresql` ∈ skills | `backend_python_django_postgresql` | 0.90 | ✅ validé |
| Data Engineer | `"data engineer"` dans le titre **ou** `sql` + `etl` + `python` ⊆ skills | `data_engineer_python_sql_etl_aligned` | 0.85 | ✅ validé |
| Data Analyst / BI | `"data analyst"` dans le titre **ou** `power bi` / `powerbi` / `bi` ∈ skills | `data_analyst_python_sql_powerbi` | 0.80 | ✅ validé |
| Machine Learning / NLP | `"machine learning"` dans le titre **ou** `nlp` ∈ skills | `machine_learning_python_nlp` | 0.80 | ✅ validé |
| Offre inconnue (fallback) | aucun signal connu | `backend_python_django_postgresql` | 0.45 | ✅ validé |

**Correction appliquée :** un bug affectait le routing Data Analyst. La condition `{"sql","etl"} & skills` était truthy dès que `sql` seul était présent dans les compétences (sans `etl`), causant le routage incorrect d'un profil Data Analyst avec Python + SQL vers `data_engineer_python_sql_etl_aligned`. Corrigé en remplaçant par `{"sql","etl"}.issubset(skills)` dans `job_router.py` ligne 19.

---

## 7. Règles obligatoires du scénario

Les règles suivantes sont contractuelles et vérifiées par les tests E2E :

- **Le matching ne démarre jamais avant confirmation explicite** du recruteur.
- **Les 6 champs du wizard doivent être collectés** avant de proposer le matching.
- **`structured_job_profile` doit être présent** dans la réponse après le 6e champ.
- **`routed_job_id` doit être présent** et non vide après le 6e champ.
- **`candidates` doit être non vide** après confirmation et matching.
- **Les noms de candidats ne doivent jamais être inventés** : seuls les noms présents dans les artefacts (`candidate_name`, `full_name`, `name`) peuvent être affichés.
- **`candidate_id` est utilisé comme fallback** si aucun nom n'est disponible dans les artefacts.
- **Les questions de suivi doivent utiliser la mémoire de session** : pas de nouveau matching, pas de redémarrage du wizard.
- **Une question de suivi avant matching doit être bloquée proprement** avec un message clair guidant le recruteur à finaliser l'offre d'abord.
- **Une offre inconnue doit déclencher un fallback propre**, pas un crash : `routed_job_id` non vide, `matching_completed` false avant confirmation.

---

## 8. Modules impliqués

| Étape | Module | Fichier principal | Rôle |
|---|---|---|---|
| Interface utilisateur | Streamlit UI | `streamlit_app.py` | Point d'entrée recruteur, saisie des messages |
| Entrée HTTP | FastAPI `/api/chat` | `src/api/routes/chat.py` | Reçoit le message et retourne `ChatResponse` |
| Mémoire de session | Session Store | `src/core/chatbot/memory.py` | Maintient l'état de session entre les messages |
| Wizard collecte | Job Intake | `src/core/chatbot/job_intake.py` | Collecte les 6 champs, construit `structured_job_profile` |
| Routing | Job Router | `src/core/chatbot/job_router.py` | Détermine le `job_id` à partir du profil structuré |
| Orchestration LangGraph | Graph | `src/core/chatbot/graph.py` | Orchestre le flux : intake → confirmation → matching → réponse |
| Matching | Node match_candidates | `src/core/chatbot/nodes/match_candidates.py` | Appelle Matching V3 via l'API interne |
| Decision Cards | Node fetch_decision_cards | `src/core/chatbot/nodes/fetch_decision_cards.py` | Récupère les fiches candidats enrichies |
| Transférabilité | Node analyze_transferability | `src/core/chatbot/nodes/analyze_transferability.py` | Enrichit avec YAML ou Neo4j selon disponibilité |
| Composition réponse | Node compose_answer | `src/core/chatbot/nodes/compose_answer.py` | Génère la réponse textuelle selon l'intention détectée |
| Matching V3 | Scoring pipeline | `src/core/matching/scoring.py` | Baseline officielle — FAISS + RF + XGBoost |
| Decision Cards data | Artefacts YAML | `data/` | Source des scores et statuts candidats |
| Potential Graph | Transférabilité YAML | `data/` / Neo4j optionnel | Gaps et transférabilité candidat → rôle cible |

---

## 9. Tests associés

### `tests/test_e2e_main_scenario.py`

| Test | Ce qu'il valide |
|---|---|
| `test_e2e_main_scenario` | Scénario complet A→E : wizard, 6 champs, structured_job_profile, routed_job_id, confirmation, matching, follow-up |
| `test_e2e_followup_before_matching` | Blocage propre d'un follow-up avant matching (nouvelle session) |
| `test_e2e_fallback_routing` | Offre exotic → fallback routing → matching après confirmation |

### `tests/test_e2e_routing.py`

| Test | Type d'offre | `job_id` attendu |
|---|---|---|
| `test_route_backend_fastapi_mongodb` | Backend FastAPI MongoDB | `backend_python_fastapi_mongodb_aligned` |
| `test_route_backend_django_postgresql` | Backend Django PostgreSQL | `backend_python_django_postgresql` |
| `test_route_data_engineer` | Data Engineer Python SQL ETL | `data_engineer_python_sql_etl_aligned` |
| `test_route_data_analyst` | Data Analyst SQL Power BI | `data_analyst_python_sql_powerbi` |
| `test_route_machine_learning` | Machine Learning NLP | `machine_learning_python_nlp` |
| `test_route_fallback_unknown_role` | Offre inconnue | `backend_python_django_postgresql` (fallback) |
| `test_route_data_analyst_with_python_routes_to_data_analyst` | Data Analyst + Python + SQL | `data_analyst_python_sql_powerbi` (bug corrigé) |

### `scripts/run_fast_tests.py`

Suite de régression rapide incluant tous les tests ci-dessus et l'ensemble des tests unitaires et d'intégration du projet.

**Résultat de référence :** `151 passed, 0 xfailed, 0 errors`

---

## 10. Critères d'acceptation

Le scénario de démonstration est considéré **accepté** si et seulement si :

- [ ] Le wizard démarre proprement sur un message de recherche ou de création d'offre.
- [ ] Les 6 champs sont collectés en 6 messages distincts dans la même session.
- [ ] `structured_job_profile.required_skills` contient `Python`, `FastAPI`, `MongoDB`.
- [ ] `structured_job_profile.min_years_experience` vaut `3`.
- [ ] `structured_job_profile.seniority` vaut `"mid-level"`.
- [ ] `routed_job_id` vaut `backend_python_fastapi_mongodb_aligned`.
- [ ] La confirmation est obligatoire : `matching_completed` reste `false` avant confirmation.
- [ ] `candidates` est non vide après confirmation.
- [ ] La question « Pourquoi le premier candidat ? » retourne une réponse référençant le premier candidat.
- [ ] La question « Quels sont ses gaps ? » retourne des gaps ou un message d'indisponibilité clair.
- [ ] Aucun `candidate_id` inventé n'apparaît dans les réponses.
- [ ] Le routing multi-offres est correct pour les 6 types d'offres validés.
- [ ] La suite rapide est verte (`151 passed, 0 xfailed, 0 errors`).

---

## 11. Limites connues

- **Tests E2E avec tools mockés :** les tests utilisent des stubs pour `match_candidates_tool`, `get_decision_card_tool`, `get_candidate_profile_tool`, `get_transferability_tool` et `get_neo4j_transferability_tool` afin d'éviter les appels HTTP externes. Les résultats réels dépendent des artefacts YAML disponibles.
- **MongoDB / FAISS / Neo4j réels :** non lancés dans les tests E2E. Les données sont servies depuis les artefacts `data/`. Neo4j reste optionnel avec fallback YAML.
- **ML expérimental :** les couches Random Forest et XGBoost sont des analyses complémentaires ; Matching V3 reste la baseline officielle.
- **Docker réel :** non testé dans la suite rapide. La configuration est validée statiquement.
- **Streamlit :** l'interface visuelle n'est pas couverte par les tests automatisés et doit encore être polie pour refléter clairement le scénario validé (étape du wizard, offre en cours, `structured_job_profile`, `routed_job_id`, candidats avec nom ou `candidate_id` fallback).
- **Profiles sans règle de routing dédiée :** les job profiles `frontend_react_nextjs`, `fullstack_react_node_mongodb` et `devops_docker_kubernetes` existent dans `data/job_profiles/` mais n'ont pas encore de règle dans `job_router.py`. Ces offres tombent en fallback `backend_python_django_postgresql` (confiance 0.45).

---

## 12. Commandes de validation

```bash
# Scénario principal E2E
pytest tests/test_e2e_main_scenario.py -q

# Routing multi-offres
pytest tests/test_e2e_routing.py -q

# Suite rapide complète
python scripts/run_fast_tests.py
```

**Résultat attendu :**

```
tests/test_e2e_main_scenario.py  →  3 passed
tests/test_e2e_routing.py        →  7 passed
scripts/run_fast_tests.py        →  151 passed, 0 xfailed, 0 errors
```

---

## 13. Prochaine étape après ce contrat

**Polish minimal de Streamlit** pour que l'interface reflète clairement le scénario validé par ce contrat :

- Affichage de l'étape courante du wizard (ex. : « Étape 3/6 — Responsabilités »).
- Affichage de l'offre en cours avec ses champs remplis.
- Affichage du `structured_job_profile` structuré (compétences, séniorité, localisation).
- Affichage du `routed_job_id` et de la raison du routing.
- Affichage des candidats avec `candidate_name` si disponible, sinon `candidate_id` comme fallback visible.
- Séparation claire entre la phase wizard (collecte) et la phase résultat (matching + suivi).

**Règle :** aucun champ ne doit être inventé côté interface — uniquement ce qui est retourné par `ChatResponse`.
