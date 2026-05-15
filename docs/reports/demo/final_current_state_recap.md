# Récapitulatif actuel du projet smart-recruiter-pfe

Date de synthèse : 2026-05-04

## 1. État global actuel du projet

### Ce qui est fait

- Module 1 a déjà servi de filtre d'entrée et a retenu 90 CV acceptés pour la suite.
- Un audit de l'ancien Module 2 a confirmé que des profils JSON pouvaient être techniquement valides mais factuellement peu fiables.
- Un nouveau pipeline `Module 2 V2 Grounded` a été construit et exécuté sur les 90 CV acceptés.
- Les 90 sorties V2 ont ensuite été préparées et importées dans MongoDB.
- L'index FAISS a été reconstruit sur la base V2 grounded.
- Le matching a été relancé avec les profils grounded, puis amélioré en `Matching Grounded V3`.
- Un comparatif de ranking V1 / grounded V2 / grounded V3 a été produit.

### Ce qui marche

- Les 90 CV acceptés ont maintenant une sortie V2 grounded.
- Le pipeline grounded a produit 90 profils avec `0 failed`.
- L'import MongoDB V2 grounded a abouti avec 90 `candidate_profiles` et 75 `candidates`.
- FAISS a été reconstruit correctement sur 90 profils avec embeddings 384 dimensions.
- Le matching grounded V3 produit un top 10 cohérent, avec pénalités qualité et affichage protégé des noms suspects.

### Ce qui a été corrigé

- Le projet ne repose plus uniquement sur une validation de structure JSON.
- Les champs non supportés par le markdown source sont maintenant nullifiés.
- Les noms suspects, titres OCR et valeurs template sont mieux détectés.
- Le matching V3 tient compte de la fiabilité grounded, du type de profil, du risque d'hallucination et des champs nullifiés.
- Les faux noms visibles en V2 comme `O Ariana, Tunisia`, `Data Scientist`, `RESUME OBJECTIVE`, `from Resume Genius` sont maintenant remplacés par `Candidate (ID: ...)` en V3 quand le nom est jugé faible.

### Ce qui reste à faire

- Revue manuelle prioritaire des profils `medium` et `high` risk.
- Nettoyage ou flag explicite des profils les plus fragiles avant démo.
- Stabilisation finale du rapport de matching V3 pour support de démonstration.
- Évolutions futures non encore intégrées : Qdrant, CrossEncoder, XGBoost, SHAP, Neo4j, LangGraph.

### Différence entre ancien pipeline et nouveau pipeline

- Ancien pipeline : Module 1 -> `profile_builder.py` -> JSON valide -> import / matching possibles, mais sans vraie garantie de vérité terrain.
- Nouveau pipeline : Module 1 markdown -> normalisation -> prompt grounded -> LLM -> validation Pydantic -> validation factuelle grounded -> score de fiabilité + risque + nullification -> rapports qualité -> import MongoDB -> FAISS -> matching V3.

### Résumé par couche

- Module 1 : extraction et filtrage des CV acceptés.
- Ancien Module 2 : `src/core/structuring/profile_builder.py`, orienté structure JSON et garde-fous métier, mais encore vulnérable aux hallucinations.
- Nouveau Module 2 V2 Grounded : pipeline plus conservateur, centré sur l'évidence présente dans le markdown source.
- MongoDB V2 : import en deux collections, `candidate_profiles` et `candidates`.
- FAISS V2 : index sémantique reconstruit sur les profils grounded.
- Matching Grounded V3 : ranking enrichi par des signaux de qualité grounded.

## 2. Pourquoi on a créé Module 2 V2 Grounded

### Problème de l'ancien Module 2

L'audit ancien Module 2 a montré un problème central : un JSON peut être techniquement valide sans être factuellement fiable.

- Pydantic validait la structure.
- Mais Pydantic ne valide pas la vérité du contenu.
- Des hallucinations ont été observées dans :
  - `summary`
  - expériences
  - éducation
  - responsabilités
  - `full_name`
  - champs contaminés par des templates

L'audit `docs/reports/module2/audits/module2_hallucination_audit_report.json` montre :

- 86 profils succès analysés
- 14 profils avec hallucinations critiques
- 38 profils avec hallucinations majeures
- 34 profils seulement jugés clean ou acceptables
- 27 profils à exclure du matching
- 25 profils à revoir
- taux de profils fiables : `39.53%`

Les erreurs les plus fréquentes étaient :

- `summary_unsupported` : 58
- `experience_responsibility_unsupported` : 49
- `experience_unsupported` : 40
- `education_unsupported` : 25
- `identity_template_value` : 15
- `generic_template_field` : 15

### Principe du pipeline V2 Grounded

Chaîne logique :

`Module 1 markdown`
-> `markdown_normalizer.py`
-> `grounded_prompt.py`
-> LLM `Groq/Ollama`
-> validation de structure
-> `grounding_validator.py`
-> JSON grounded
-> rapports qualité

### Rôle des fichiers

- `src/core/structuring/markdown_normalizer.py`
  - nettoie les artefacts OCR
  - corrige emails/URLs/termes techniques fusionnés
  - détecte sections, templates et indices d'identité
  - calcule un score qualité de document en amont

- `src/core/structuring/grounded_prompt.py`
  - construit un prompt strict
  - impose `null` ou `[]` quand la preuve est absente
  - interdit l'invention de dates, écoles, responsabilités, skills ou résumés marketing

- `src/core/structuring/grounding_validator.py`
  - compare les champs du LLM au markdown source
  - nullifie les champs template et non supportés
  - calcule `reliability_score`
  - détermine `profile_kind`
  - détermine `hallucination_risk`
  - produit `quality_flags` et `fields_nullified`

- `src/core/structuring/profile_builder_grounded.py`
  - orchestre le pipeline grounded
  - charge les CV acceptés
  - appelle Ollama ou Groq selon disponibilité
  - applique normalisation + prompt + validation grounded
  - écrit les profils et déclenche la génération des rapports

- `src/core/structuring/grounded_reporting.py`
  - agrège les résultats de run
  - produit les rapports qualité, risques, providers, champs nullifiés et profils partiels/minimaux

### Différence conceptuelle

- Ancien Module 2 : "est-ce que le JSON est valide ?"
- Module 2 V2 Grounded : "est-ce que le champ est réellement supporté par le CV ?"

## 3. Résultat Module 2 V2 Grounded sur les 90 CV

Source principale : `data/profile_builder_module2_v2_grounded_all/reports/grounded_quality_report.json`

### Chiffres exacts

- `accepted_count` : 90
- total sorties V2 grounded : 90
- `complete_profile` : 73
- `partial_profile` : 16
- `minimal_profile` : 1
- `unreadable` : 0
- `failed` : 0
- `average_reliability_score` : `0.8441`
- distribution `hallucination_risk` :
  - `low` : 47
  - `medium` : 42
  - `high` : 1
- `full_name_rejected_count` : 23
- `fields_nullified_count` : 73
- providers utilisés :
  - `groq_secondary` : 61
  - `ollama_local` : 29

### Répartition détaillée utile

- profils complets par provider :
  - Groq : 54
  - Ollama : 19
- profils partiels par provider :
  - Groq : 7
  - Ollama : 9
- profil minimal :
  - Ollama : 1

### Signification des catégories

- `complete_profile`
  - profil grounded avec assez de champs supportés et une fiabilité élevée

- `partial_profile`
  - profil utile mais incomplet, souvent parce que certaines informations n'étaient pas assez supportées et ont été volontairement retirées

- `minimal_profile`
  - profil très faible, conservé pour ne pas perdre complètement la trace du CV mais à manier avec prudence

- `unreadable`
  - document trop faible ou trop bruité pour produire un profil exploitable

- `hallucination_risk`
  - estimation du risque que des champs aient été fragiles, reconstruits ou trop incertains

- `reliability_score`
  - score synthétique de fiabilité grounded, calculé à partir du support réel des champs, du score de confiance du document et des pénalités de champs nullifiés/non supportés

- `fields_nullified`
  - liste des champs explicitement mis à `null` ou retirés car non prouvés par le texte source

- `full_name rejected`
  - cas où le nom trouvé a été jugé suspect, template-like, trop faible ou non fiable

### Point à dire clairement

Les 90 CV acceptés ont maintenant une sortie V2 grounded.

### Indices complémentaires

- `skipped_existing` : 78
  - cela signifie que le run a repris avec checkpoint/résumé et n'a pas retraité inutilement tous les fichiers déjà produits
  - mais le total final reste bien 90 profils disponibles

- templates détectés dans le corpus :
  - `FIRSTLAST` : 4
  - `resumeworded` : 4
  - `example.com` : 4
  - `info@qwikresume.com` : 4
  - `qwikresume` : 4
  - `resumesample` : 3
  - `email@youremail.com` : 2
  - `info@resumekraft.com` : 2
  - `resumekraft` : 2

## 4. MongoDB après V2 Grounded

Sources :

- `docs/reports/mongodb/mongodb_import_report_v2_grounded_execute.json`
- `docs/reports/mongodb/mongodb_import_report_v2_grounded_dry_run.json`

### Résultat import

- base MongoDB utilisée : `talent_intelligence`
- URI : `mongodb://localhost:27017`
- collections utilisées :
  - `candidate_profiles`
  - `candidates`

### Chiffres exacts

- `candidate_profiles_to_import` : 90
- `candidate_profiles_created` : 90
- `candidate_profiles_upserted` : 90
- `candidates_to_create` : 75
- `candidates_created` : 75
- `candidates_upserted` : 75
- `strong_merges_count` : 12
- `possible_duplicates_count` : 7
- `conflicts_count` : 1

### Différence entre les deux collections

- `candidate_profiles`
  - contient les profils structurés individuels issus du Module 2 grounded
  - 1 document par profil/source traité

- `candidates`
  - contient la vue consolidée/dédupliquée d'un candidat
  - plusieurs profils peuvent appartenir au même candidat

### Point clé métier

`candidate_profiles = profils individuels structurés`

`candidates = candidats consolidés/dédupliqués`

Un candidat peut avoir plusieurs profils.

### Pourquoi MongoDB est utile ici

- stocker proprement les profils structurés
- séparer le niveau "profil" du niveau "candidat"
- préparer la déduplication
- fournir la base source pour FAISS
- faciliter les requêtes de matching, d'audit et d'explication

## 5. FAISS après V2 Grounded

Source : `data/indexes/faiss/index_report.json`

### Chiffres exacts

- profils lus depuis MongoDB : 90
- profils indexés : 90
- modèle SentenceTransformers : `sentence-transformers/all-MiniLM-L6-v2`
- dimension des embeddings : `384`
- métrique FAISS : `inner_product_on_normalized_vectors`

### Rôle de FAISS

FAISS sert à retrouver rapidement les profils proches sémantiquement d'une offre.

### Pourquoi reconstruire FAISS après MongoDB V2

- parce que le contenu textuel des profils a changé après grounding
- parce que des champs ont été retirés/nullifiés
- parce que les profils V2 grounded sont la nouvelle vérité exploitable pour la recherche sémantique
- parce qu'un index construit sur les anciens profils transporterait les anciens biais/hallucinations

## 6. Matching Grounded V3

### Pourquoi on a refait le matching après V2

Le matching devait être recalculé car la base de profils a changé :

- profils plus grounded
- noms suspects mieux filtrés
- skills plus propres
- qualité des profils explicitement mesurée

### Champs qualité pris en compte en V3

Le scoring V3 utilise désormais :

- `reliability_score`
- `profile_kind`
- `hallucination_risk`
- `quality_flags`
- `fields_nullified`

### Améliorations de code observées

- `src/core/matching/skill_normalizer.py`
  - normalise des alias comme `Fast API -> FastAPI`, `REST API design -> REST API`, `mongo db -> MongoDB`

- `src/core/matching/matching_quality_filters.py`
  - reconstruit un `hallucination_risk` fiable si besoin
  - détecte les noms suspects
  - remplace les noms faibles par `Candidate (ID: ...)`

- `src/core/matching/scoring.py`
  - ajoute `compute_grounded_quality_score`
  - ajoute `compute_quality_penalty_multiplier`
  - applique des multiplicateurs selon :
    - risque `low/medium/high`
    - type `complete/partial/minimal`
    - qualité du nom affiché
    - nombre de champs nullifiés et flags qualité

### Différence entre les versions de matching

- matching avant V2
  - basé surtout sur similarité FAISS + heuristiques skills/expérience
  - pas de signaux grounded explicites
  - faux noms ou profils douteux pouvaient apparaître sans protection

- matching grounded V2
  - recalculé sur la nouvelle base grounded
  - améliore déjà le contenu sémantique
  - mais ne pénalise pas encore assez les noms suspects et profils moyens

- matching grounded V3
  - ajoute normalisation skills
  - ajoute score qualité grounded
  - ajoute pénalité qualité
  - protège l'affichage des `full_name` suspects
  - rend le ranking plus défendable métier

## 7. Comparaison des rankings

Sources :

- `data/matching_single_job_report_v2.json`
- `data/matching_single_job_report_grounded_v2.json`
- `docs/reports/matching/v3/matching_single_job_report_grounded_v3.json`
- `docs/reports/matching/v3/matching_ranking_comparison_grounded_v3.json`
- `docs/reports/matching/v3/matching_ranking_comparison_grounded_v3.md`

### Ancien top 10

| Rang | Candidat | Score |
|---|---|---:|
| 1 | Hichem Bensalah | 0.5749 |
| 2 | MOHAMED AZIZ BELAWEID | 0.3537 |
| 3 | JefferyGorczany | 0.2342 |
| 4 | Candidate (ID: candidate_f9f052160651) | 0.2290 |
| 5 | Wenzhe(Evelyn)Xu | 0.2280 |
| 6 | JESSICACLAIRE | 0.2267 |
| 7 | Candidate (ID: candidate_c6a411e09ae7) | 0.2262 |
| 8 | MILDREDZEMLAK | 0.2259 |
| 9 | Candidate (ID: candidate_ef0e8855357d) | 0.2258 |
| 10 | Candidate (ID: candidate_e25119c3eefc) | 0.2253 |

### Nouveau top 10 grounded V3

| Rang | Candidat | Score | Risk | Name Quality |
|---|---|---:|---|---|
| 1 | Hichem Bensalah | 0.8172 | low | ok |
| 2 | MOHAMED AZIZ BELAWEID | 0.5528 | low | ok |
| 3 | Candidate (ID: candidate_8eea1b635447) | 0.5466 | medium | weak |
| 4 | JEFFERYGORCZANY | 0.4332 | low | ok |
| 5 | Candidate (ID: candidate_71e03ea99985) | 0.3502 | low | weak |
| 6 | Candidate (ID: candidate_1d475044c93c) | 0.3464 | low | weak |
| 7 | Markus Rohan | 0.3181 | medium | ok |
| 8 | Candidate (ID: candidate_c564b8eceb3d) | 0.2731 | medium | weak |
| 9 | MILDREDZEMLAK | 0.2628 | low | ok |
| 10 | Justine Hendrickson | 0.2573 | low | ok |

### Évolution des rangs et scores

- Hichem Bensalah reste top 1.
- Son score évolue de `0.5749` -> `0.7671` -> `0.8172`.
- MOHAMED AZIZ BELAWEID passe de rang 2 -> 3 -> 2, avec score `0.3537` -> `0.5362` -> `0.5528`.
- Le candidat affiché comme `O Ariana, Tunisia` en grounded V2 tombe de rang 2 à rang 3 en V3 et son affichage est sécurisé en `Candidate (ID: ...)`.
- `JEFFERYGORCZANY` reste bien placé et passe à `0.4332` en V3.
- `Markus Rohan` reste dans le top 10 mais subit une pénalité de qualité : `0.3712` -> `0.3181`.
- `MILDREDZEMLAK` améliore légèrement son score : `0.2259` -> `0.2316` -> `0.2628`.
- `Justine Hendrickson` progresse légèrement : `0.2355` -> `0.2573`.

### Qui monte

- Hichem consolide sa première place.
- Mohamed Aziz remonte de 3e à 2e entre grounded V2 et V3.
- MILDREDZEMLAK passe de 10e à 9e entre grounded V2 et V3.

### Qui descend

- le faux affichage `O Ariana, Tunisia` est pénalisé et sécurisé
- le faux affichage `Data Scientist` passe de 5e grounded V2 à 8e grounded V3 sous identifiant candidat protégé
- plusieurs profils du vieux top 10 disparaissent complètement après grounding

### Candidats sortis du top 10 historique

Sortis du top 10 après grounding :

- JefferyGorczany
- Candidate `candidate_f9f052160651`
- Wenzhe(Evelyn)Xu
- JESSICACLAIRE
- Candidate `candidate_c6a411e09ae7`
- Candidate `candidate_ef0e8855357d`
- Candidate `candidate_e25119c3eefc`

Le rapport de comparaison liste aussi `MILDREDZEMLAK` comme retiré de l'ancien top 10 par `candidate_id`, mais ce candidat réapparaît bien dans le top 10 grounded sous un autre identifiant consolidé. Cela indique que la comparaison par identifiant reflète aussi la reconstruction/déduplication MongoDB, pas seulement un changement de score.

### Faux noms et titres

En grounded V2, 4 noms suspects apparaissent encore visiblement :

- `O Ariana, Tunisia`
- `Data Scientist`
- `RESUME OBJECTIVE`
- `from Resume Genius`

En grounded V3, ces cas restent présents fonctionnellement dans le ranking mais ne sont plus exposés comme noms réels ; ils sont remplacés par des labels neutres `Candidate (ID: ...)` lorsque le nom est jugé faible.

### Conclusion métier

Oui, le ranking grounded V3 est plus défendable métier :

- les meilleurs profils gardent ou renforcent leur place
- les profils à nom douteux sont pénalisés ou neutralisés visuellement
- les signaux qualité deviennent visibles dans le score final
- le top 10 paraît moins arbitraire que l'ancien ranking

## 8. Problèmes restants / limites

- 42 profils restent `hallucination_risk = medium`
- 1 profil reste `hallucination_risk = high`
- 23 noms ont été rejetés, donc certains profils peuvent rester sans nom fiable affichable
- 16 profils restent `partial_profile`
- 1 profil reste `minimal_profile`
- le matching reste une combinaison `FAISS + heuristiques`
- pas encore de ranking supervisé ML
- pas encore de Qdrant
- pas encore de CrossEncoder
- pas encore de XGBoost
- pas encore de SHAP
- pas encore de Neo4j
- pas encore de LangGraph

Ces limites ne cassent pas le pipeline actuel, mais elles indiquent les prochaines améliorations.

## 9. Prochaine étape recommandée

### Étape 1

Audit final qualité V2 + validation manuelle des profils `high` / `medium risk`.

### Étape 2

Nettoyer ou flagger les profils `high risk` avant démo.

### Étape 3

Stabiliser le matching V3 et préparer un rapport lisible.

### Étape 4

Préparer une démonstration avec :

- CV source
- profil V2 grounded
- MongoDB
- FAISS
- matching top candidats
- explication du score

### Étape 5

Plus tard :

- Qdrant
- CrossEncoder
- ML Ranker / XGBoost
- SHAP
- Decision Card

## 10. Résumé oral pour soutenance

Au départ, notre pipeline de structuration produisait des profils JSON valides, mais l'audit a montré qu'une partie des informations n'était pas toujours réellement supportée par les CV. En particulier, nous avons détecté des hallucinations dans les résumés, certaines expériences, l'éducation, les responsabilités et parfois même les noms complets, souvent à cause d'OCR bruité ou de templates de CV.

Pour corriger cela, nous avons conçu un nouveau Module 2 V2 Grounded. L'idée est de partir du markdown produit par le Module 1, de le normaliser, d'envoyer un prompt beaucoup plus strict au LLM, puis de vérifier après extraction que chaque champ est bien appuyé par le texte source. Si un champ n'est pas suffisamment prouvé, il est volontairement nullifié au lieu d'être inventé. Le pipeline calcule aussi un score de fiabilité, un type de profil et un niveau de risque d'hallucination.

Ce nouveau module a été exécuté sur les 90 CV acceptés, et les 90 ont maintenant une sortie V2 grounded. Ensuite, nous avons importé ces profils dans MongoDB avec deux collections : une collection `candidate_profiles` pour les profils individuels, et une collection `candidates` pour les candidats consolidés et dédupliqués. À partir de cette base V2, nous avons reconstruit FAISS pour la recherche sémantique.

Nous avons ensuite relancé le matching et amélioré le module en version Grounded V3. Cette version ne se contente plus de la similarité sémantique : elle prend aussi en compte la fiabilité du profil, le type de profil, le risque d'hallucination, les champs nullifiés et la qualité du nom affiché. Le résultat est un ranking plus propre et plus défendable métier, tout en gardant Hichem en top 1.

Il reste encore quelques limites : certains profils ont un risque moyen ou élevé, certains noms sont volontairement masqués, et le matching reste basé sur FAISS plus heuristiques, sans modèle supervisé avancé. La prochaine étape recommandée est donc un audit final qualité, puis une stabilisation du matching V3 et la préparation d'une démonstration technique claire.

## 11. Tableau final des livrables

| Composant | Chemin fichier/dossier | Rôle | Statut |
|---|---|---|---|
| Module 2 V2 grounded profiles | `data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/` | Profils structurés grounded des 90 CV acceptés | Produit |
| Legacy projection | `data/profile_builder_module2_v2_grounded_all/profiles/legacy_projection/` | Projection/compatibilité de certaines sorties vers un format legacy | Présent |
| Run report V2 | `data/profile_builder_module2_v2_grounded_all/reports/run_report.json` | Rapport d'exécution complet du run grounded | Produit |
| Rapport qualité grounded | `data/profile_builder_module2_v2_grounded_all/reports/grounded_quality_report.json` | Statistiques de qualité et de fiabilité grounded | Produit |
| Rapport réduction hallucinations | `data/profile_builder_module2_v2_grounded_all/reports/hallucination_reduction_report.json` | Justification métier du nouveau pipeline | Produit |
| Résumé confiance champs | `data/profile_builder_module2_v2_grounded_all/reports/field_confidence_summary.csv` | Vue détaillée champ par champ supporté/nullifié | Produit |
| Profils failed/partial | `data/profile_builder_module2_v2_grounded_all/reports/failed_or_partial_profiles.csv` | Liste des profils non complets | Produit |
| Rapport providers | `data/profile_builder_module2_v2_grounded_all/reports/provider_comparison_report.json` | Répartition Ollama/Groq | Produit |
| Checkpoint reprise | `data/profile_builder_module2_v2_grounded_all/reports/resume_checkpoint.json` | État de reprise / traitement | Produit |
| Audit ancien Module 2 | `docs/reports/module2/audits/module2_hallucination_audit_report.json` | Diagnostic des hallucinations de l'ancien pipeline | Produit |
| Synthèse audit ancien Module 2 | `docs/reports/module2/audits/module2_hallucination_audit_summary.md` | Résumé lisible de l'audit | Produit |
| Import MongoDB V2 dry run | `docs/reports/mongodb/mongodb_import_report_v2_grounded_dry_run.json` | Vérification avant import | Produit |
| Import MongoDB V2 execute | `docs/reports/mongodb/mongodb_import_report_v2_grounded_execute.json` | Rapport d'import effectif MongoDB | Produit |
| Index FAISS | `data/indexes/faiss/index_report.json` | Rapport de reconstruction de l'index | Produit |
| Matching V1 | `data/matching_single_job_report_v2.json` | Référence du ranking avant grounded | Produit |
| Matching grounded V2 | `data/matching_single_job_report_grounded_v2.json` | Premier matching après reconstruction grounded | Produit |
| Matching grounded V3 | `docs/reports/matching/v3/matching_single_job_report_grounded_v3.json` | Matching enrichi par qualité grounded | Produit |
| Comparaison ranking V2 | `data/matching_ranking_comparison_grounded_v2.json` | Différences ancien vs grounded V2 | Produit |
| Comparaison ranking V3 JSON | `docs/reports/matching/v3/matching_ranking_comparison_grounded_v3.json` | Différences V1 vs grounded V2 vs grounded V3 | Produit |
| Comparaison ranking V3 Markdown | `docs/reports/matching/v3/matching_ranking_comparison_grounded_v3.md` | Synthèse lisible des rankings | Produit |

## Conclusion synthétique

Le projet est dans un état techniquement cohérent pour une démo technique interne :

- les 90 CV acceptés ont été restructurés en V2 grounded
- MongoDB a été rechargé
- FAISS a été reconstruit
- le matching a été recalculé puis amélioré en V3
- le ranking final est plus défendable qu'avant

La principale réserve n'est plus la chaîne technique, mais la validation qualitative finale des profils `medium` / `high risk` avant démonstration.
