# Guide  du projet smart-recruiter-pfe


## 1. Résumé global

`smart-recruiter-pfe` est un projet de recrutement intelligent qui transforme des CV bruts en profils structurés, puis compare ces profils à une offre d'emploi pour proposer les meilleurs candidats.

Le problème principal du projet était le suivant : on arrivait à produire du JSON valide à partir des CV, mais ce JSON n'était pas toujours fiable. En particulier, l'ancien Module 2 pouvait produire des hallucinations dans les résumés, les expériences, l'éducation ou même les noms complets.

La chaîne actuelle est :

CV bruts
-> Module 1 extraction et acceptation
-> ancien Module 2 `profile_builder.py`
-> audit des hallucinations
-> nouveau Module 2 V2 Grounded
-> import MongoDB
-> reconstruction FAISS
-> Matching Grounded V3
-> rapports de comparaison

Aujourd'hui, ce qui marche :

- 90 CV ont été acceptés par Module 1
- ces 90 CV ont une sortie Module 2 V2 Grounded
- les profils grounded ont été importés dans MongoDB
- FAISS a été reconstruit sur cette nouvelle base
- le matching a été relancé et amélioré en V3
- les rapports finaux existent et sont exploitables pour une démo

Ce qui a été amélioré récemment :

- audit de l'ancien Module 2
- création du pipeline grounded anti-hallucination
- ajout d'un score de fiabilité et d'un risque d'hallucination
- ajout de pénalités qualité dans le matching
- masquage des faux noms ou noms suspects dans les résultats

## 2. Module 1

### Rôle

Le Module 1 sert à lire les CV bruts, à extraire leur contenu exploitable et à décider quels CV sont assez corrects pour passer à la structuration.

Autrement dit, Module 1 joue le rôle de filtre d'entrée du pipeline.

### Où sont les résultats

- dossier source : `data/raw_cv/`
- sorties Module 1 : `data/processed_official_module1/`
- liste de handoff acceptée : `data/processed_official_module1/handoff/accepted.json`

### Chiffres clés

- nombre total de lignes dans `accepted.json` : 90
- nombre de CV acceptés pour Module 2 : 90

### Pourquoi Module 1 est important

- il transforme le CV brut en artefact structuré intermédiaire
- il produit le markdown qui sert ensuite de source à Module 2
- il évite d'envoyer au Module 2 des documents hors périmètre ou trop mauvais
- il sert de point de traçabilité entre le fichier source et le profil final

###

À montrer :

- un CV brut dans `data/raw_cv/`
- le dossier correspondant dans `data/processed_official_module1/`
- `data/processed_official_module1/handoff/accepted.json`

Ce qu'il faut dire :

"Ici, on part des CV bruts. Module 1 les lit, extrait leur contenu et décide s'ils sont assez exploitables pour continuer. Les 90 CV visibles ici ont été acceptés et transmis au Module 2."

## 3. Ancien Module 2

### Rôle de l'ancien `profile_builder.py`

L'ancien Module 2 avait pour rôle de transformer le contenu issu du Module 1 en profil candidat JSON structuré.

Le fichier principal est :

- `src/core/structuring/profile_builder.py`

Les sorties historiques sont dans :

- `data/profile_builder_official_module2_rerun_ollama_fixed/`

### Pourquoi il posait problème

Le problème n'était pas la forme du JSON, mais sa vérité.

Le système arrivait à produire un JSON techniquement correct, mais certains champs n'étaient pas vraiment supportés par le CV source. Par exemple :

- résumé trop reconstruit
- responsabilités inventées ou extrapolées
- dates ou villes peu fiables
- noms template ou placeholders

### JSON valide vs JSON fiable

- JSON valide :
  - la structure respecte le schéma attendu
  - les types sont corrects

- JSON fiable :
  - les valeurs sont réellement supportées par le CV
  - il n'y a pas d'invention ou d'extrapolation abusive

### Rôle de Pydantic

Pydantic vérifie surtout :

- la structure
- les types
- les champs obligatoires
- quelques règles métier simples

### Pourquoi Pydantic ne suffit pas

Pydantic ne lit pas le CV source pour vérifier si une information est vraie.

Exemple simple :

- `{"summary": "expert en cloud distribué"}` peut être un JSON parfaitement valide
- mais si cette phrase n'est pas vraiment écrite dans le CV, alors le champ n'est pas fiable

### Audit de l'ancien Module 2

Chemins importants :

- `docs/reports/module2/audits/module2_hallucination_audit_report.json`
- `docs/reports/module2/audits/module2_hallucination_audit_summary.md`
- `data/module2_hallucination_audit_table.csv`

### Chiffres clés

- profils succès analysés : 86
- profils fiables / keep : 34
- profils à review : 25
- profils à exclure : 27
- taux de profils fiables : 39.53%

### Principaux types d'erreurs

- `summary_unsupported` : 58
- `experience_responsibility_unsupported` : 49
- `experience_unsupported` : 40
- `education_unsupported` : 25
- `identity_template_value` : 15
- `generic_template_field` : 15
- `identity_unsupported` : 4

### 

"L'ancien Module 2 savait produire des profils bien formés, mais pas assez contrôlés factuellement. L'audit a montré qu'une partie importante des profils ne devait pas être utilisée telle quelle pour le matching."

## 4. Nouveau Module 2 V2 Grounded

### Pourquoi on l'a créé

On l'a créé pour réduire les hallucinations de l'ancien Module 2.

Le but n'était plus seulement de générer un JSON propre, mais de produire un JSON fondé sur les preuves présentes dans le markdown source issu de Module 1.

### Ce qu'il ajoute

Par rapport à l'ancien Module 2, il ajoute :

- une normalisation du markdown
- un prompt beaucoup plus strict
- une validation factuelle après le LLM
- la nullification des champs non prouvés
- un `reliability_score`
- un `hallucination_risk`
- des `quality_flags`
- une distinction entre `complete_profile`, `partial_profile` et `minimal_profile`

### Pourquoi il réduit les hallucinations

Parce que le pipeline grounded préfère retirer un champ plutôt que l'inventer.

Principe :

- si un champ est clairement supporté, on le garde
- s'il est douteux ou template-like, on le nullifie
- si la preuve est insuffisante, on accepte un profil partiel au lieu d'un faux profil complet

### Pourquoi il produit des profils complets, partiels ou minimaux

Parce que tous les CV n'ont pas la même qualité :

- un CV clair et propre donne souvent un `complete_profile`
- un CV partiellement lisible donne un `partial_profile`
- un CV très faible ou très bruité donne un `minimal_profile`

### Rôle des fichiers

- `src/core/structuring/markdown_normalizer.py`
  - nettoie les artefacts OCR
  - détecte sections, templates et indices d'identité
  - produit un markdown plus propre pour le LLM

- `src/core/structuring/grounded_prompt.py`
  - construit les instructions strictes envoyées au LLM
  - impose la règle : "si ce n'est pas clairement présent, mettre `null` ou `[]`"

- `src/core/structuring/grounding_validator.py`
  - compare la sortie LLM au markdown source
  - calcule `reliability_score`
  - détermine `profile_kind`
  - détermine `hallucination_risk`
  - nullifie les champs non supportés

- `src/core/structuring/profile_builder_grounded.py`
  - orchestre le pipeline complet grounded
  - appelle Groq ou Ollama
  - écrit les profils grounded
  - lance la génération des rapports

- `src/core/structuring/grounded_reporting.py`
  - agrège les résultats
  - produit les rapports de qualité, de providers et de profils partiels

### Chaîne réelle

Module 1 markdown
-> `markdown_normalizer.py`
-> `grounded_prompt.py`
-> LLM Groq/Ollama
-> validation de structure
-> `grounding_validator.py`
-> JSON grounded
-> rapports qualité

###  les résultats

- profils grounded : `data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/`
- projection legacy : `data/profile_builder_module2_v2_grounded_all/profiles/legacy_projection/`
- rapports : `data/profile_builder_module2_v2_grounded_all/reports/`

### Chiffres clés V2 Grounded

- CV acceptés : 90
- sorties V2 : 90
- `complete_profile` : 73
- `partial_profile` : 16
- `minimal_profile` : 1
- `unreadable` : 0
- `failed` : 0
- `average_reliability_score` : 0.8441
- `hallucination_risk` :
  - low : 47
  - medium : 42
  - high : 1
- `full_name_rejected_count` : 23
- `fields_nullified_count` : 73
- providers utilisés :
  - Groq : 61
  - Ollama : 29

### Quoi;;;

À montrer en priorité :

- un profil grounded propre, par exemple `data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/docx_Hichem_resume.json`
- `data/profile_builder_module2_v2_grounded_all/reports/grounded_quality_report.json`
- `data/profile_builder_module2_v2_grounded_all/reports/failed_or_partial_profiles.csv`

### Ce qu'il faut 

"Le nouveau Module 2 grounded ne se contente plus de demander un JSON au LLM. Il vérifie ensuite si les champs sont réellement supportés par le texte source. Quand la preuve manque, on préfère nullifier ou produire un profil partiel."

## 5. MongoDB

### Pourquoi on a importé les profils V2 dans MongoDB

MongoDB sert de couche de stockage structurée pour les profils candidat après le Module 2 grounded.

Cela permet :

- de stocker les profils de façon requêtable
- de séparer les profils individuels des candidats consolidés
- de préparer le matching et l'indexation FAISS

### Base et collections

D'après `docs/reports/mongodb/mongodb_import_report_v2_grounded_execute.json` :

- base utilisée : `talent_intelligence`
- collection 1 : `candidate_profiles`
- collection 2 : `candidates`

### Explication simple

- `candidate_profiles = profils individuels`
- `candidates = candidats consolidés / dédupliqués`

### Pourquoi 90 `candidate_profiles` mais 75 `candidates`

Parce qu'un même candidat peut apparaître plusieurs fois dans plusieurs formats ou plusieurs variantes de CV.

Donc :

- 90 profils individuels ont été importés
- après consolidation/déduplication, cela correspond à 75 candidats uniques

### Chiffres clés MongoDB

- `candidate_profiles_to_import` : 90
- `candidate_profiles_created` : 90
- `candidates_created` : 75
- `strong_merges_count` : 12
- `possible_duplicates_count` : 7
- `conflicts_count` : 1

 ouvrir MongoDB Compass

Dans MongoDB Compass :

1. se connecter à `mongodb://localhost:27017`
2. ouvrir la base `talent_intelligence`
3. montrer `candidate_profiles`
4. montrer `candidates`

### Quoi montrer exactement

Dans `candidate_profiles` :

- un document de profil grounded
- les champs `profile_kind`, `reliability_score`, `quality_flags`, `source_path`

Dans `candidates` :

- un candidat consolidé
- montrer qu'un candidat peut agréger plusieurs profils

###

"MongoDB est la base de travail intermédiaire entre la structuration et la recherche sémantique. On y stocke les profils individuels et les candidats consolidés."

## 6. FAISS

### Rôle de FAISS

FAISS sert à rechercher rapidement les profils les plus proches sémantiquement d'une offre d'emploi.

### Pourquoi on l'a reconstruit après MongoDB V2

Parce que les profils ont changé après le grounding :

- certains champs ont été retirés
- certains noms ont été corrigés ou neutralisés
- la qualité globale du texte utilisé pour le matching a changé

Donc il fallait reconstruire l'index sur la nouvelle base grounded.

### Chiffres clés

D'après `data/indexes/faiss/index_report.json` :

- profils lus depuis MongoDB : 90
- profils indexés : 90
- modèle d'embeddings : `sentence-transformers/all-MiniLM-L6-v2`
- dimension : 384

### Explication simple

- un embedding = une représentation numérique d'un texte
- indexer un profil = transformer le texte du profil en vecteur pour pouvoir le comparer rapidement à une offre

### Quoi montrer 

À montrer :

- `data/indexes/faiss/index_report.json`

Ce qu'il faut dire :

"FAISS n'est pas la décision finale. C'est le moteur de recherche sémantique qui nous permet de retrouver rapidement les profils potentiellement pertinents avant de les scorer plus finement."

## 7. Matching Grounded V3

### Rôle du matching

Le matching sert à comparer une offre d'emploi à la base de profils candidats pour produire un classement des meilleurs profils.

### Entrée du matching

- une offre d'emploi textuelle dans `data/job_descriptions/`
- un job profile structuré dans `data/job_profiles/`
- les profils grounded stockés dans MongoDB et indexés dans FAISS

### Sortie du matching

- un top candidats classés
- un score final
- des explications sur les skills manquants, la qualité du profil et le risque d'hallucination

### Différence entre matching et scoring

- matching :
  - processus complet de récupération et de classement

- scoring :
  - calcul du score d'un candidat donné pour une offre donnée

### Pourquoi on a refait le matching après Module 2 V2

Parce que la base de profils a changé. Si les profils deviennent plus grounded, les résultats du matching doivent être recalculés sur cette nouvelle vérité.

### Ce que V3 ajoute

- `skill_normalizer`
- `reliability_score`
- `profile_kind`
- `hallucination_risk`
- `quality_flags`
- `fields_nullified`
- pénalité qualité
- affichage sécurisé des noms suspects

### Ce que fait le code

- `src/core/matching/skill_normalizer.py`
  - normalise des variantes comme `Fast API -> FastAPI`, `REST API design -> REST API`

- `src/core/matching/matching_quality_filters.py`
  - détecte les noms suspects
  - remplace les noms faibles par `Candidate (ID: ...)`

- `src/core/matching/scoring.py`
  - combine similarité texte, skills, expérience et qualité grounded
  - applique des multiplicateurs selon le risque et le type de profil

- `src/core/matching/recommender.py`
  - récupère les profils via FAISS
  - appelle le scoring
  - conserve le meilleur profil par candidat

- `src/core/matching/ranking_comparator.py`
  - compare V1, grounded V2 et grounded V3

### Top 10 actuel

1. Hichem Bensalah — 0.8172
2. MOHAMED AZIZ BELAWEID — 0.5528
3. Candidate `(candidate_8eea1b635447)` — 0.5466
4. JEFFERYGORCZANY — 0.4332
5. Candidate `(candidate_71e03ea99985)` — 0.3502
6. Candidate `(candidate_1d475044c93c)` — 0.3464
7. Markus Rohan — 0.3181
8. Candidate `(candidate_c564b8eceb3d)` — 0.2731
9. MILDREDZEMLAK — 0.2628
10. Justine Hendrickson — 0.2573

### Pourquoi Hichem reste top 1

Parce qu'il combine :

- un profil `complete_profile`
- un `reliability_score` très élevé : 0.9776
- un risque `low`
- une bonne couverture de skills requis
- aucune pénalité qualité supplémentaire

Il lui manque seulement `REST API`, ce qui reste compatible avec une première place vu la qualité globale du profil.

### Quels candidats sont pénalisés

Exemples visibles dans V3 :

- `candidate_8eea1b635447`
  - risque `medium`
  - `display_name_quality = weak`
  - `quality_penalty_multiplier = 0.72`

- `candidate_c564b8eceb3d`
  - risque `medium`
  - `display_name_quality = weak`
  - `quality_penalty_multiplier = 0.72`

- Markus Rohan
  - `partial_profile`
  - risque `medium`
  - `quality_penalty_multiplier = 0.855`

### Comment les faux noms sont masqués

Au lieu d'afficher des titres OCR ou des noms suspects comme :

- `O Ariana, Tunisia`
- `Data Scientist`
- `RESUME OBJECTIVE`
- `from Resume Genius`

le système affiche :

- `Candidate (ID: ...)`

quand le nom est jugé trop faible ou suspect.

### Pourquoi le ranking V3 est plus défendable métier

- il ne se base pas seulement sur la similarité sémantique
- il tient compte de la qualité réelle du profil
- il évite de survaloriser des profils bruités
- il réduit l'exposition de faux noms
- il explique mieux les scores et les pénalités

## 8. Fichiers importants à montrer en démo

### 1. CV source

- chemin : `data/raw_cv/`
- 

### 2. accepted.json Module 1

- chemin : `data/processed_official_module1/handoff/accepted.json`
- la liste des CV que Module 1 a validés pour la suite."
- montre le passage du brut vers le pipeline structuré

### 3. Un profil V2 grounded

- chemin : `data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/docx_Hichem_resume.json`
- ce qu'il faut dire : "Voici un profil grounded avec source, score de confiance et statut de grounding."
- pourquoi c'est important : c'est la pièce centrale de la démo

### 4. Rapport qualité grounded

- chemin : `data/profile_builder_module2_v2_grounded_all/reports/grounded_quality_report.json`
- ce qu'il faut dire : "Ce rapport résume la qualité globale des 90 profils grounded."
- pourquoi c'est important : il donne les chiffres-clés

### 5. MongoDB Compass

- chemin logique : base `talent_intelligence`, collections `candidate_profiles` et `candidates`
- ce qu'il faut dire : "On stocke les profils individuels d'un côté et les candidats consolidés de l'autre."
- pourquoi c'est important : montre la structuration exploitable

### 6. Rapport FAISS

- chemin : `data/indexes/faiss/index_report.json`
- ce qu'il faut dire : "L'index FAISS a été reconstruit sur 90 profils grounded."
- pourquoi c'est important : montre la recherche sémantique réelle

### 7. Job description

- chemin : `data/job_descriptions/backend_python_fastapi_mongodb.txt`
- ce qu'il faut dire : "Voici l'offre d'emploi brute utilisée pour la démonstration."
- pourquoi c'est important : montre l'entrée côté entreprise

### 8. Job profile JSON

- chemin : `data/job_profiles/backend_python_fastapi_mongodb.json`
- ce qu'il faut dire : "Voici la version structurée de l'offre, utilisée par le matching."
- pourquoi c'est important : lien entre texte métier et scoring

### 9. Rapport matching V3

- chemin : `docs/reports/matching/v3/matching_single_job_report_grounded_v3.json`
- ce qu'il faut dire : "Voici le top candidats avec score, pénalités et signaux qualité."
- pourquoi c'est important : c'est le résultat final du pipeline

### 10. Comparaison ranking V3

- chemin : `docs/reports/matching/v3/matching_ranking_comparison_grounded_v3.md`
- ce qu'il faut dire : "Ce rapport compare l'ancien ranking, le grounded V2 et le grounded V3."
- pourquoi c'est important : montre la progression du projet

### 11. Récapitulatif final

- chemin : `docs/reports/demo/final_current_state_recap.md`
- ce qu'il faut dire : "C'est la synthèse globale de l'état actuel du projet."
- pourquoi c'est important : support de conclusion

## 9. Scénario de démo

### Étape 1 : montrer les CV sources

Quoi ouvrir :

- `data/raw_cv/`

Quoi dire :

"On part ici de CV bruts, en PDF, DOCX ou image. Le projet doit être capable d'en extraire de l'information exploitable pour le recrutement."

Message important :

Le système commence avec des documents réels et hétérogènes.

### Étape 2 : montrer Module 1 accepted

Quoi ouvrir :

- `data/processed_official_module1/handoff/accepted.json`

Quoi dire :

"Module 1 sert de filtre. Il prépare les artefacts et décide quels CV sont assez corrects pour être structurés. Ici, 90 CV ont été acceptés."

Message important :

Le pipeline ne travaille pas directement sur du bruit non filtré.

### Étape 3 : montrer un profil V2 grounded

Quoi ouvrir :

- `data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/docx_Hichem_resume.json`

Quoi dire :

"Voici un profil grounded. On voit la source, le type de profil, la partie `profile`, et la partie `grounding` avec le score de fiabilité et le risque."

Message important :

Le profil final n'est pas seulement généré, il est aussi contrôlé.

### Étape 4 : expliquer l'anti-hallucination

Quoi ouvrir :

- `docs/reports/module2/audits/module2_hallucination_audit_summary.md`
- `data/profile_builder_module2_v2_grounded_all/reports/grounded_quality_report.json`

Quoi dire :

"L'ancien Module 2 produisait parfois des champs valides mais non fiables. On a donc créé une version grounded qui retire les champs non prouvés au lieu de les inventer."

Message important :

Le projet a progressé en fiabilité, pas seulement en formatage.

### Étape 5 : montrer MongoDB

Quoi ouvrir :

- MongoDB Compass, base `talent_intelligence`

Quoi dire :

"Les profils grounded sont stockés dans `candidate_profiles`, puis consolidés dans `candidates`. C'est pour cela qu'on a 90 profils mais 75 candidats."

Message important :

Le système sait distinguer profil individuel et candidat unique.

### Étape 6 : montrer FAISS

Quoi ouvrir :

- `data/indexes/faiss/index_report.json`



"FAISS indexe les profils sous forme de vecteurs pour retrouver rapidement les profils proches d'une offre. L'index actuel contient 90 profils grounded."

Message important :

La recherche sémantique repose sur la nouvelle base propre.

### Étape 7 : montrer la job description



- `data/job_descriptions/backend_python_fastapi_mongodb.txt`
- `data/job_profiles/backend_python_fastapi_mongodb.json`



"Ici on part d'une offre texte, puis on la structure en job profile avec skills requis, expérience et contexte."

Message important :

Le matching compare des structures cohérentes, pas seulement deux blobs de texte.

### Étape 8 : montrer le matching top candidats



- `docs/reports/matching/v3/matching_single_job_report_grounded_v3.json`



 Hichem reste premier avec 0.8172, devant Mohamed Aziz, car son profil est plus complet, plus fiable et moins pénalisé."

 important :

Le ranking final est justifiable.

### Étape 9 : expliquer le score



- `docs/reports/matching/v3/matching_ranking_comparison_grounded_v3.md`



"Le score final ne dépend pas seulement de la similarité. Il tient aussi compte des skills, de l'expérience et de la qualité grounded du profil."

 important :

Le système ne récompense pas aveuglément des profils bruités.

### Étape 10 : expliquer les limites et prochaines étapes

 :

- `docs/reports/demo/final_current_state_recap.md`

:

"Le pipeline est cohérent, mais il reste des profils medium/high risk. La prochaine étape logique est une validation finale qualité, puis éventuellement un ranking plus avancé avec Qdrant ou un modèle supervisé."

important :

Le projet est solide pour une démo, mais encore perfectible.



Au départ, l'objectif du projet était de construire une chaîne complète capable de partir de CV bruts et d'arriver à un matching de candidats pour une offre d'emploi. Le problème qu'on a rencontré, c'est que l'ancien Module 2 produisait des profils JSON bien structurés, mais pas toujours fiables sur le fond.

On a donc lancé un audit des hallucinations sur l'ancien pipeline. Cet audit a montré qu'une partie importante des profils contenait des résumés non supportés, des responsabilités extrapolées, des données d'éducation fragiles, et parfois même des noms ou des emails template. En clair, le JSON était valide techniquement, mais pas suffisamment défendable métier.

Pour répondre à ce problème, on a construit un nouveau Module 2 V2 Grounded. L'idée est simple : on part du markdown produit par Module 1, on le nettoie, on envoie un prompt strict au LLM, puis on vérifie après extraction si les champs sont réellement supportés par le texte source. Si la preuve manque, on nullifie le champ au lieu de l'inventer.

Ce nouveau pipeline a été exécuté sur les 90 CV acceptés par Module 1. Résultat : les 90 ont maintenant une sortie grounded, avec 73 profils complets, 16 profils partiels et 1 profil minimal. On a aussi un score moyen de fiabilité de 0.8441, ce qui donne une vision plus fine de la qualité réelle des profils.

Ensuite, on a importé ces profils dans MongoDB, avec deux niveaux : `candidate_profiles` pour les profils individuels et `candidates` pour les candidats consolidés. Cela nous donne 90 profils mais 75 candidats uniques après déduplication. À partir de cette base, on a reconstruit l'index FAISS pour la recherche sémantique.

Enfin, on a relancé le matching et on l'a amélioré en version Grounded V3. Cette version ajoute des signaux de qualité comme le `reliability_score`, le `profile_kind`, le `hallucination_risk`, les `quality_flags` et une pénalité qualité. Elle masque aussi les faux noms suspects en `Candidate (ID: ...)`. Le résultat est un ranking plus défendable, avec Hichem toujours top 1.

Aujourd'hui, le pipeline est prêt pour une démo technique, mais il reste des limites : certains profils sont encore medium ou high risk, et le matching reste basé sur FAISS plus heuristiques. La prochaine étape logique est donc de faire une validation qualité finale, puis d'envisager des améliorations comme Qdrant, CrossEncoder ou un ranker supervisé.

## 11. Questions possibles de l'encadrante

### Pourquoi MongoDB ?

Parce qu'on a besoin d'un stockage structuré et requêtable entre la structuration des profils et le matching.

### Pourquoi 90 profils mais 75 candidats ?

Parce qu'un candidat peut apparaître dans plusieurs profils ou formats, puis être consolidé après déduplication.

### Pourquoi Pydantic ne suffit pas ?

Parce que Pydantic valide surtout la structure et les types, pas la vérité factuelle du contenu.

### C'est quoi "grounded" ?

Grounded veut dire que le profil doit être appuyé par le texte source. Si la preuve manque, on retire le champ.

### Pourquoi certains noms sont `null` ou masqués ?

Parce que certains noms détectés ressemblent à des titres, à des placeholders ou à du bruit OCR. On préfère masquer plutôt que montrer un faux nom.

### Pourquoi FAISS ?

Pour retrouver rapidement les profils sémantiquement proches d'une offre d'emploi.

### C'est quoi un embedding ?

C'est une représentation numérique d'un texte qui permet de comparer des similarités de sens.

### Quelle différence entre matching et scoring ?

Le matching est le processus global de recherche et de classement. Le scoring est le calcul du score d'un candidat donné.

### Est-ce que le système utilise du machine learning ?

Oui, partiellement. Le système utilise déjà des embeddings et un LLM, donc il y a bien une composante ML.

### Où est le ML exactement ?

Dans l'extraction assistée par LLM et dans la représentation sémantique via `sentence-transformers`.

### Pourquoi pas encore XGBoost ?

Parce que le projet a d'abord priorisé une base de données fiable et un matching explicable avant d'ajouter un ranker supervisé.

### Quelles sont les limites actuelles ?

Il reste des profils `medium/high risk`, et le matching repose encore sur FAISS plus heuristiques plutôt que sur un modèle supervisé complet.

### Quelle est la prochaine étape ?

Faire une validation finale des profils à risque, stabiliser le matching V3, puis envisager un ranker plus avancé.

## 12. Prochaine étape recommandée

La prochaine étape logique est :

1. revoir manuellement les profils `medium` et `high risk`
2. décider lesquels doivent être flaggés 
3. figer le jeu de démonstration
4. ensuite seulement, envisager des améliorations plus avancées de ranking

## 13. Résultats clés à retenir

- 90 CV acceptés par Module 1
- 90 sorties grounded
- 73 profils complets
- 16 profils partiels
- 1 profil minimal
- score moyen de fiabilité : 0.8441
- 90 `candidate_profiles` MongoDB
- 75 `candidates` MongoDB
- 90 profils indexés dans FAISS
- top 1 matching V3 : Hichem Bensalah avec 0.8172
--------------------------------------------------------------------------------------------------------------------------
1. Pénalité must-have skills
2. Pénalité qualité du profil grounded
3. Pénalité profile_kind
4. Pénalité hallucination_risk
5. Pénalité nom suspect / full_name faible

-----------------------------------------------------
Pénalité skills :
quand le candidat ne couvre pas assez les compétences obligatoires.

Pénalité qualité :
quand le profil est incomplet, risqué ou peu fiable.

Pénalité nom :
quand le nom affiché est suspect ou rejeté.

But :
ne pas laisser un candidat monter juste parce qu’il est proche sémantiquement avec FAISS.
--------------------------------------------------------------------------------------------------
Module 1 donner  markdown
→ markdown_normalizer.py
→ grounded_prompt.py
→ LLM Groq/Ollama
→ grounding_validator.py
→ profil V2 grounded
--------------------------------------------------------------------------------------------------

Ce module ajoute :

reliability_score
hallucination_risk
quality_flags
fields_nullified
profile_kind
--------------------------------------------------------------------------------------
Ce qui est encore faible

Je ne vais pas te mentir :

42 profils medium risk
1 profil high risk
16 profils partiels
1 profil minimal
matching encore heuristique
pas de labels recruteur
pas de ML supervisé réel
pas de CrossEncoder
pas de Qdrant
pas de Neo4j réel
pas de LangGraph réel
-------------------------------------------------------------------------------------------------