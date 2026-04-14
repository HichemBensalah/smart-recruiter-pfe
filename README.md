# Smart Recruiter PFE

Projet de fin d'etudes centre sur l'extraction, l'evaluation et la structuration de CV pour un futur systeme RH intelligent.

## Etat actuel

Le depot contient aujourd'hui deux modules principaux deja exploitables:

1. `Module 1 - Parsing documentaire`
   - lecture de CV PDF, images et DOCX
   - extraction avec `Docling`
   - fallback OCR avec `PyTesseract` sur les cas faibles
   - evaluation qualitative du document
   - decision de handoff vers la suite du pipeline
2. `Module 2 - Structuration du profil`
   - lecture des artefacts valides du Module 1
   - extraction structuree assistee par LLM
   - validation Pydantic
   - generation de previews JSON avant insertion finale

Une campagne de benchmark OCR se trouve egalement dans un sous-module separe afin de comparer plusieurs moteurs d'extraction sans perturber le pipeline principal.

## Architecture utile du repo

```text
src/
  core/
    parser/              # Module 1
    structuring/         # Module 2
  benchmark/
    ocr/                 # Benchmark OCR isole
data/
  raw_cv/                # CV bruts utilises par Module 1 (ignore de Git)
  processed/             # Artefacts Module 1 (ignore de Git)
  benchmarks/ocr/        # Dataset et sorties utiles du benchmark OCR
```

## Fichiers cles

- [src/core/parser/run_docling_pipeline.py](src/core/parser/run_docling_pipeline.py)
- [src/core/parser/document_router.py](src/core/parser/document_router.py)
- [src/core/parser/document_quality.py](src/core/parser/document_quality.py)
- [src/core/parser/secondary_parser.py](src/core/parser/secondary_parser.py)
- [src/core/structuring/profile_builder.py](src/core/structuring/profile_builder.py)
- [src/benchmark/ocr/README.md](src/benchmark/ocr/README.md)

## Avancement reel

### Module 1

Le pipeline de parsing est operationnel. Il orchestre:

- le routage des documents
- le parsing principal avec `Docling`
- le post-traitement documentaire
- le calcul d'un score de confiance
- une logique de fallback OCR via `PyTesseract`
- le classement final en `accepted`, `repair_required` ou `quarantined`

Un tuning minimal et prudent a ete applique a partir des resultats du benchmark OCR:

- `Docling` reste le parseur principal
- `PyTesseract` est renforce comme fallback
- les cas `image`, `scan` et `faible sortie Docling` sont mieux recuperes

### Module 2

Le module de structuration est deja en place et fonctionne sur les artefacts acceptes du Module 1:

- lecture des fichiers `accepted`
- appel LLM pour extraire un profil candidat
- validation par schemas Pydantic
- generation de profils JSON exploitables

### Benchmark OCR

Le benchmark OCR est volontairement separe du pipeline principal.

Il permet de:

- comparer plusieurs moteurs OCR sur un meme dataset
- mesurer `WER` et `CER`
- conserver les predictions et les rapports de run
- appuyer les choix techniques de Module 1

Le benchmark retenu dans ce depot repose sur un nouveau ground truth prepare manuellement et sur un run final consolide.

## Resultat utile a retenir

Les premiers resultats exploitables ont montre que:

- `PyTesseract` est meilleur que `Docling` pour l'OCR brut sur le benchmark retenu
- `Docling` reste plus pertinent dans le pipeline principal car il apporte une structure documentaire exploitable
- la bonne decision d'architecture n'est donc pas de remplacer `Docling`, mais de renforcer le fallback OCR dans les cas difficiles

## Ce qui n'est pas encore finalise

Les briques suivantes ne sont pas encore completes dans ce depot:

- API applicative complete
- moteur de matching hybride final
- assistant RH conversationnel final
- insertion base de donnees industrialisee end-to-end

Le depot montre donc un avancement reel sur les modules coeur, pas une application complete.

## Notes GitHub

Le depot a ete nettoye pour ne garder que:

- le code utile
- le benchmark OCR final utile
- les sorties demonstratives pertinentes

Les elements locaux sensibles ou volumineux restent exclus via `.gitignore`, notamment:

- `.env`
- `data/raw_cv/`
- `data/processed/`
- `data/profile_builder_preview*/`
- `models/`

## Documentation complementaire

Pour le benchmark OCR, voir:

- [src/benchmark/ocr/README.md](src/benchmark/ocr/README.md)
