# Contrats entre modules

Ce document decrit les entrees et sorties attendues entre les briques du pipeline. Il sert de reference avant la creation des LangChain Tools.

## Parsing

- Input : `data/raw_cv/`
- Output : artefacts markdown, txt, json, html dans `data/processed_official_module1/`
- Output handoff : `data/processed_official_module1/handoff/accepted.json`
- Contrat : chaque CV accepte doit pointer vers un artefact exploitable par Module 2.

## Structuring

- Input : `accepted.json` + markdown/texte Module 1
- Output : `data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/*.json`
- Contrat : chaque profil doit contenir une structure candidat, des skills, un `reliability_score` et un statut grounded.

## Storage

- Input : profils grounded
- Output : MongoDB `candidates`, `candidate_profiles`, `job_profiles`, `matching_runs`, `decision_cards`
- Collections reservees pour phases suivantes : `conversation_sessions`, `audit_logs`, `faiss_index_metadata`
- Contrat : `candidate_profiles` conserve les profils importes ; `candidates` consolide les candidats uniques ; `job_profiles` expose les offres structurees ; `decision_cards` conserve les cartes explicatives ; `matching_runs` historise les resultats ou seeds Matching V3.
- Configuration runtime : `DATA_BACKEND=artifacts|mongodb|hybrid`, `ALLOW_ARTIFACT_FALLBACK=true|false`, `MONGODB_URI`, `MONGODB_DATABASE`.
- Contrat de migration E1 : `artifacts` conserve le comportement MVP ; `mongodb` lit MongoDB pour les endpoints migres ; `hybrid` donne priorite a MongoDB et garde les artefacts comme fallback de demo.

## Retrieval

- Input : profils candidats depuis MongoDB
- Output : `data/indexes/faiss/cv_index.faiss`, `id_map.pkl`, `index_report.json`
- Contrat : chaque vecteur FAISS doit avoir un mapping vers un profil candidat.

## Matching V3

- Input : job profile JSON + FAISS + profils candidats + scoring metier
- Output : `docs/reports/matching/v3/*_matching_report_v3_normalized.json`
- Contrat : conserver `final_score_v3`, `rank`, skills matched/missing, reliability et penalites metier.

### Matching via API demo

- Endpoint : `POST /api/match`
- Input API : `job_description`, `top_k`, `job_id` optionnel.
- Configuration : `MATCHING_MODE=artifact|live|hybrid`, `LIVE_MATCHING_TOP_N`, `LIVE_MATCHING_TOP_K`, `FAISS_INDEX_PATH`, `FAISS_ID_MAP_PATH`.
- Registre runtime : `data/ranking/features/*.jsonl`
- Fallback officiel : `backend_python_django_postgresql`
- Contrat : l'API choisit l'artefact `data/ranking/features/{job_id}.jsonl` quand il existe. Si le `job_id` est absent, elle conserve le comportement historique avec le fallback officiel sans signaler d'erreur. Si le `job_id` est inconnu, elle retourne `fallback_used=true`, `resolved_job_id=backend_python_django_postgresql` et un warning.
- Mode `artifact` : mode MVP stable ; la route expose les artefacts Matching V3 pre-generes pour rendre la demo stable et rapide.
- Mode `live` : la route lit MongoDB (`job_profiles`, `candidate_profiles`, `matching_runs`), encode le job avec le modele documente dans l'index FAISS, recupere les profils proches via `FAISS_INDEX_PATH` / `FAISS_ID_MAP_PATH`, puis recalcule le score officiel avec `src/core/matching/scoring.py::score_candidate`.
- Mode `hybrid` : la route essaye le live, puis fallback vers les artefacts si erreur ou donnees insuffisantes et `ALLOW_ARTIFACT_FALLBACK=true`.
- E1 : si MongoDB est disponible et `DATA_BACKEND=mongodb|hybrid`, la route peut enregistrer le resultat retourne dans `matching_runs` sans changer le contrat de matching.
- E2 : quand le live reussit, `matching_runs` conserve `run_id`, `job_id`, `job_description`, `matching_mode`, `top_k`, `candidate_ids`, scores, warnings et metadata de source. Quand le fallback artefact est utilise, l'erreur live reste visible dans `warnings`.
- Limites E2 : l'index FAISS existant est reutilise ; il n'est pas reconstruit dynamiquement. Les tests restent mockes/fakes et ne dependent pas d'un vrai MongoDB, FAISS, Neo4j ou Docker.
- Evolution future : remplacer ou completer ce mode par un matching live FAISS/MongoDB capable de scorer un job profile cree dynamiquement.

## Decision Cards

- Input : matching report
- Output : `docs/reports/matching/v3/decision_cards_v3_normalized.json`
- Contrat : expliquer les recommandations sans presenter le systeme comme decision automatique finale.
- E1 API : `/api/decision-cards` lit les artefacts en mode `artifacts`, lit `decision_cards` en mode `mongodb`, et peut fallback artefacts si MongoDB est indisponible et `ALLOW_ARTIFACT_FALLBACK=true`.

## ML

- Input : `data/ranking/features/*.jsonl`
- Output : datasets, modeles `data/ranking/models/*.joblib`, rapports ML et SHAP
- Contrat : utiliser `label_binary` comme target experimentale ; ne pas utiliser `final_score_v3` comme label.
- Statut : Random Forest / XGBoost / SHAP enrichissent l'analyse, mais ne remplacent pas Matching V3.

## Graph

- Input : `data/graph/skills_roles_graph.yaml` + profils + job profile
- Output : transferability score, gaps compensables, gaps bloquants, transitions plausibles
- Contrat : le graph explique la transferabilite ; il ne remplace pas Matching V3.
- Fallback : le Potential Graph YAML reste disponible meme si Neo4j est indisponible.

## API

- Input : artefacts existants, rapports demo, cards, graph
- Output : endpoints JSON FastAPI
- Contrat : ne pas relancer les pipelines lourds sauf endpoint demo explicite ; retourner 404/503 propres quand un artefact ou service optionnel manque.
- E1 : `/api/candidates` et `/api/decision-cards` retournent aussi `data_backend`, `data_source`, `fallback_used` et `warnings` quand pertinent. Les tests utilisent des fakes/mocks MongoDB ; la CI ne depend pas d'une instance MongoDB reelle.
