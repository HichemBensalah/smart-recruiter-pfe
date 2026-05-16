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
- Output : MongoDB `candidate_profiles` et `candidates`
- Contrat : `candidate_profiles` conserve les profils importes ; `candidates` consolide les candidats uniques.

## Retrieval

- Input : profils candidats depuis MongoDB
- Output : `data/indexes/faiss/cv_index.faiss`, `id_map.pkl`, `index_report.json`
- Contrat : chaque vecteur FAISS doit avoir un mapping vers un profil candidat.

## Matching V3

- Input : job profile JSON + FAISS + profils candidats + scoring metier
- Output : `docs/reports/matching/v3/*_matching_report_v3_normalized.json`
- Contrat : conserver `final_score_v3`, `rank`, skills matched/missing, reliability et penalites metier.

## Decision Cards

- Input : matching report
- Output : `docs/reports/matching/v3/decision_cards_v3_normalized.json`
- Contrat : expliquer les recommandations sans presenter le systeme comme decision automatique finale.

## ML

- Input : `data/ranking/features/*.jsonl`
- Output : datasets, modeles `data/ranking/models/*.joblib`, rapports ML et SHAP
- Contrat : utiliser `label_binary` comme target experimentale ; ne pas utiliser `final_score_v3` comme label.

## Graph

- Input : `data/graph/skills_roles_graph.yaml` + profils + job profile
- Output : transferability score, gaps compensables, gaps bloquants, transitions plausibles
- Contrat : le graph explique la transferabilite ; il ne remplace pas Matching V3.

## API

- Input : artefacts existants, rapports demo, cards, graph
- Output : endpoints JSON FastAPI
- Contrat : ne pas relancer les pipelines lourds sauf endpoint demo explicite ; retourner 404/503 propres quand un artefact ou service optionnel manque.
