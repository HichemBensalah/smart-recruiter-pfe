# Résumé Module 1 / Parser du projet smart-recruiter-pfe

Date de synthèse : 2026-05-06

## 1. Rôle global de Module 1

Le Module 1 est le parser du projet.

Son rôle n'est pas de construire directement un profil candidat complet.

Son rôle est de :

- lire les CV bruts
- choisir la bonne stratégie de parsing selon le format
- extraire du texte et une structure exploitable
- mesurer la qualité du document
- produire des artefacts intermédiaires stables
- décider si le document peut entrer dans Module 2

### Logique générale

Chaîne :

CV brut
-> Module 1 / parser
-> artefacts intermédiaires
-> handoff `accepted.json`
-> Module 2 V2 Grounded

### Entrée de Module 1

- fichiers bruts dans `data/raw_cv/`
- formats principaux : PDF, DOCX, images

### Sortie de Module 1

Pour chaque document, Module 1 produit :

- un `.md`
- un `.json`
- un `.txt`
- un `.html`

Et globalement :

- `data/processed_official_module1/handoff/accepted.json`
- `data/processed_official_module1/handoff/repair_required.json`
- `data/processed_official_module1/handoff/quarantined.json`
- `data/processed_official_module1/handoff/module1_handoff_report.json`

### Pourquoi Module 1 existe avant Module 2

Parce que Module 2 grounded a besoin d'une matière première déjà nettoyée et traçable :

- texte récupéré
- markdown plus lisible
- score de confiance
- métadonnées
- décision de handoff

Module 1 sert donc de frontière de confiance avant le LLM.

### Pourquoi Module 1 ne fait pas toute la structuration candidat

Parce que son but est surtout :

- l'extraction documentaire
- la qualité
- la traçabilité

La structuration métier du candidat appartient au Module 2.

## 2. Schéma du pipeline Module 1

1. Le pipeline lit les fichiers bruts dans `data/raw_cv/`
2. `document_router.py` choisit la stratégie de parsing
3. `docling_parser.py` essaie le parsing principal
4. si besoin, `secondary_parser.py` sert de fallback
5. `postprocess_docling.py` nettoie et restructure la sortie
6. `document_quality.py` mesure la qualité du document
7. `markdown_quality.py` évalue si le markdown est assez exploitable pour Module 2
8. `handoff_policy.py` décide accepted / repair_required / quarantined
9. `run_docling_pipeline.py` écrit les artefacts et les fichiers de handoff

## 3. Explication de chaque fichier dans src/core/parser/

### `__init__.py`

- rôle : expose une entrée simple `parse_cv()`
- entrée : chemin du fichier
- sortie : résultat du parsing Docling
- utilité : simplifie l'import comme package Python
- criticité démo : faible

### `docling_parser.py`

- rôle : wrapper autour de Docling
- entrée : chemin d'un document
- sortie :
  - markdown via `parse()`
  - dictionnaire structuré via `convert_to_dict()`
- utilité :
  - parsing principal du projet
  - conversion PDF / image avec OCR Docling si nécessaire
  - conversion DOCX
- pourquoi utile :
  - Docling fournit une extraction plus riche qu'un simple texte brut
- criticité démo : élevée

### `document_artifact.py`

- rôle : définit le contrat de données de Module 1
- entrée : valeurs calculées par le pipeline
- sortie : objets Pydantic comme :
  - `DocumentArtifact`
  - `DocumentConfidence`
  - `HandoffDecision`
  - `LogicalSection`
  - `EvidenceSpan`
- utilité :
  - structure standard du JSON Module 1
  - traçabilité
  - cohérence entre statut, confiance et handoff
- pourquoi utile :
  - c'est le modèle central des artefacts Module 1
- criticité démo : élevée

### `document_quality.py`

- rôle : évalue la qualité du document parsé
- entrée :
  - payload structuré
  - format source
  - type de document
- sortie :
  - score `document_confidence_score`
  - statut candidat : `validated`, `partial`, `uncertain`
  - signaux et warnings
- utilité :
  - mesure si le document est assez bon pour la suite
- signaux possibles :
  - volume de texte utile
  - complétude des sections
  - ordre des sections
  - stabilité de segmentation
  - bruit OCR
  - titres de sections bizarres
- criticité démo : élevée

### `document_router.py`

- rôle : décide quelle stratégie appliquer selon le fichier
- entrée : chemin du fichier
- sortie : `RoutingDecision`
- logique :
  - DOCX -> Docling markdown, sans OCR
  - image -> Docling structuré avec OCR
  - PDF natif -> Docling structuré sans OCR
  - PDF scanné -> Docling structuré avec OCR
- utilité :
  - évite d'utiliser la même stratégie pour tous les formats
- criticité démo : moyenne à élevée

### `handoff_policy.py`

- rôle : convertit la qualité du document en décision de handoff vers Module 2
- entrée :
  - `document_status`
  - `source_format`
  - `quality_flags`
- sortie :
  - `accepted`
  - `repair_required`
  - `quarantined`
  - booléen `eligible_for_module2`
- règles principales :
  - `validated` -> `accepted`
  - `partial` -> `repair_required`
  - `uncertain` -> `quarantined`
  - `failed` -> `quarantined`
- utilité :
  - protège Module 2 contre des entrées trop faibles
- criticité démo : très élevée

### `markdown_quality.py`

- rôle : évalue et nettoie le markdown destiné à Module 2
- entrée : markdown brut
- sortie :
  - markdown nettoyé
  - signaux
  - warnings
- utilité :
  - détecte si le markdown est trop faible, trop bruité ou trop cassé
- pourquoi utile :
  - le LLM du Module 2 dépend fortement de la qualité du markdown
- criticité démo : élevée

### `postprocess_docling.py`

- rôle : post-traitement de la sortie Docling
- entrée : markdown ou structure Docling
- sortie : payload plus propre et mieux segmenté
- utilité :
  - corrige / réorganise la sortie extraction avant l'évaluation qualité
- différence avec `markdown_normalizer.py` du Module 2 :
  - `postprocess_docling.py` agit côté Module 1 juste après l'extraction
  - `markdown_normalizer.py` agit côté Module 2 grounded juste avant le prompt LLM
- criticité démo : moyenne

### `run_docling_pipeline.py`

- rôle : script principal de Module 1
- entrée :
  - documents bruts dans `data/raw_cv/`
- sortie :
  - `.txt`
  - `.md`
  - `.json`
  - `.html`
  - fichiers de handoff
  - rapport global pipeline
- utilité :
  - orchestre tout Module 1
- criticité démo : très élevée

### `secondary_parser.py`

- rôle : parser secondaire de secours
- entrée : fichier et format source
- sortie :
  - payload simplifié
  - métadonnées de fallback
- utilité :
  - fallback si Docling est trop faible ou si la qualité est insuffisante
- cas typiques :
  - OCR image
  - OCR PDF
  - extraction texte DOCX via XML
- criticité démo : moyenne

## 4. Pourquoi Module 1 génère 4 sorties

### A. Markdown `.md`

- rôle :
  - version textuelle structurée et lisible
- pourquoi important :
  - c'est la source principale pour Module 2 V2 Grounded
- pourquoi le LLM aime le markdown :
  - sections plus claires
  - meilleure lisibilité que le texte brut
  - moins de confusion que du HTML

### B. JSON `.json`

- rôle :
  - artefact de contrôle complet Module 1
- ce qu'il contient réellement dans ce projet :
  - `source_path`
  - `source_format`
  - `document_type`
  - `document_status`
  - `raw_text`
  - `markdown`
  - `logical_sections`
  - `parser_used`
  - `fallback_used`
  - `document_confidence`
  - `handoff_decision`
  - `evidence_spans`
- remarque :
  - il ne stocke pas directement `markdown_path`, `html_path` ou `txt_path`
  - ces chemins sont gérés au niveau du pipeline et du fichier lui-même

### C. TXT `.txt`

- rôle :
  - texte brut récupéré
- utilité :
  - fallback
  - debug
  - inspection rapide sans structure
- différence avec markdown :
  - moins structuré
  - moins adapté au LLM

### D. HTML `.html`

- rôle :
  - visualisation simple du markdown dans un navigateur
- utilité :
  - inspection humaine rapide
- Module 2 l'utilise ?
  - non, pas au coeur du pipeline

### Tableau synthétique

| Format | Rôle | Utilisé par Module 2 ? | À montrer en démo ? |
|---|---|---|---|
| `.md` | Texte structuré principal | Oui | Oui |
| `.json` | Artefact complet de contrôle | Oui | Oui |
| `.txt` | Texte brut fallback/debug | Non directement | Plutôt non |
| `.html` | Visualisation humaine | Non | Optionnel |

## 5. Ce qui est utilisé par Module 2 V2

Le fichier `src/core/structuring/profile_builder_grounded.py` consomme vraiment :

### `accepted.json`

- utilisé : oui
- rôle : sélectionner les documents autorisés à entrer dans Module 2
- important pour démo : oui

### JSON artifact Module 1

- utilisé : oui
- rôle :
  - ouvrir l'artefact
  - récupérer `markdown`
  - récupérer `raw_text`
  - fallback sur `document_confidence.score`
  - garder la traçabilité
- important pour démo : oui

### Markdown

- utilisé : oui
- rôle :
  - texte principal envoyé au `normalize_markdown()`
  - base du prompt grounded
- important pour démo : oui

### `raw_text`

- utilisé : oui
- rôle :
  - fourni au normaliseur en complément du markdown
- important pour démo : moyen

### TXT

- utilisé : non directement par `profile_builder_grounded.py`
- rôle : surtout debug/fallback côté Module 1
- important pour démo : non

### HTML

- utilisé : non
- rôle : visualisation seulement
- important pour démo : optionnel

### `document_confidence_score`

- utilisé : oui
- rôle :
  - récupéré depuis `accepted.json`
  - sinon depuis `artifact.document_confidence.score`
  - sert à rendre le prompt et le validator plus prudents
- important pour démo : oui

### `logical_sections`

- utilisé : non directement par `profile_builder_grounded.py`
- rôle : surtout traçabilité et richesse de l'artefact Module 1
- important pour démo : non indispensable

### `evidence_spans`

- utilisé : non directement par `profile_builder_grounded.py`
- rôle : audit / traçabilité Module 1
- important pour démo : optionnel

### `source_path`

- utilisé : oui
- rôle : traçabilité dans les sorties grounded
- important pour démo : oui

## 6. Explication du handoff summary

Bloc observé :

```json
{
  "accepted": 90,
  "repair_required": 0,
  "quarantined": 0,
  "eligible_for_module2": 90,
  "blocked_from_module2": 0
}
```

### `accepted = 90`

- signifie :
  - 90 artefacts ont été jugés assez bons par Module 1
- accepté par quoi ?
  - par la politique de handoff de Module 1
- accepté pour quoi ?
  - pour entrer directement dans Module 2
- est-ce que cela veut dire exploitables ?
  - oui, exploitables pour le pipeline aval

### `repair_required = 0`

- signifie :
  - aucun document n'a été classé "partiel à réparer" dans ce run
- quand cela pourrait arriver ?
  - extraction partielle
  - markdown trop faible
  - OCR insuffisant mais pas totalement perdu

### `quarantined = 0`

- signifie :
  - aucun document n'a été bloqué totalement
- quand cela pourrait arriver ?
  - fichier corrompu
  - extraction trop faible
  - document trop incertain
  - parsing en échec

### `eligible_for_module2 = 90`

- signifie :
  - 90 documents sont effectivement autorisés à passer dans Module 2
- pourquoi important ?
  - c'est la vraie porte d'entrée opérationnelle vers Module 2

### `blocked_from_module2 = 0`

- signifie :
  - aucun document n'a été bloqué
- quand cela pourrait être > 0 ?
  - si des documents sont `repair_required` ou `quarantined`

## 7. Différence entre `accepted` et `eligible_for_module2`

### Idée simple

- `accepted` = lane de handoff décidée par Module 1
- `eligible_for_module2` = autorisation effective de consommation par Module 2

### Pourquoi ici ils sont égaux

Parce que :

- 90 documents sont dans la lane `accepted`
- tous ces documents sont donc `eligible_for_module2 = true`

### Cas où ils pourraient différer

Exemple 1 :

- `accepted = 90`
- `repair_required = 5`
- `eligible_for_module2 = 85`

Interprétation :

- 90 documents ont été traités avec succès pipeline
- mais 5 doivent être réparés avant Module 2

Exemple 2 :

- `accepted = 90`
- `blocked_from_module2 = 3`
- `eligible_for_module2 = 87`

Interprétation :

- certains documents ont peut-être produit un artefact, mais ont été bloqués par la politique de qualité

Dans le code actuel, le rapport de handoff calcule simplement :

- `eligible_for_module2 = len(accepted)`
- `blocked_from_module2 = len(repair_required) + len(quarantined)`

Donc dans ce projet, `accepted` et `eligible_for_module2` sont très proches conceptuellement, mais `eligible_for_module2` exprime explicitement la permission aval.

## 8. Ce qu'il faut montrer en démo

### À montrer absolument

1. `data/raw_cv/`
   - quoi dire :
     - "Voici les CV bruts d'entrée."
   - pourquoi important :
     - montre le point de départ réel

2. `data/processed_official_module1/handoff/accepted.json`
   - quoi dire :
     - "Voici la liste des documents autorisés à entrer dans Module 2."
   - pourquoi important :
     - montre la frontière qualité

3. un artifact JSON Module 1
   - exemple : `data/processed_official_module1/pdf/0_anonyme.json`
   - quoi dire :
     - "Cet artefact contient le texte, le markdown, les sections, le score de confiance et la décision de handoff."
   - pourquoi important :
     - c'est la sortie de référence de Module 1

4. un markdown Module 1
   - exemple : `data/processed_official_module1/docx/0_anonyme.md`
   - quoi dire :
     - "C'est cette représentation structurée que Module 2 grounded va surtout exploiter."
   - pourquoi important :
     - lien direct avec le LLM

5. éventuellement un HTML
   - si besoin seulement
   - quoi dire :
     - "Le HTML sert surtout à visualiser rapidement le markdown."
   - pourquoi important :
     - optionnel, pas coeur du pipeline

### À éviter ou montrer rapidement

- `.txt`
- HTML si le temps est court
- logs secondaires
- fallback parser en détail

## 9. Discours oral de 1 minute pour expliquer Module 1

Le Module 1 est la couche de parsing et de contrôle qualité du projet. Son rôle est de partir des CV bruts, de choisir la bonne stratégie de lecture selon le format, puis de produire des artefacts intermédiaires fiables : du texte brut, du markdown, un artefact JSON complet et une décision de handoff.

Il ne construit pas encore le profil candidat final. Il prépare surtout une matière première propre pour le Module 2 grounded. C'est lui qui décide si un document est accepté, à réparer ou à mettre en quarantaine. Dans notre état actuel, les 90 documents traités ont été acceptés et sont tous éligibles pour Module 2.

Le point important à retenir, c'est que Module 1 protège le reste du pipeline : il transforme des documents hétérogènes en artefacts traçables et assez propres pour que le LLM travaille dans de meilleures conditions.

