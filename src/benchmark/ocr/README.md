# OCR Benchmark

Ce sous-module est volontairement isole de `src/core/parser` et de `src/core/structuring`.

Son but est d'evaluer des moteurs OCR de maniere reproductible, sans modifier le pipeline principal.

## Objectif

Comparer plusieurs moteurs OCR sur un meme jeu de CV et mesurer la qualite de l'extraction texte a l'aide de:

- `WER` : Word Error Rate
- `CER` : Character Error Rate

## Moteurs prevus dans le benchmark

- `Docling`
- `Mistral OCR`
- `TrOCR`
- `EasyOCR`
- `PyTesseract`

Dans l'etat actuel du depot, le run final conserve et exploitable repose sur les moteurs effectivement stables dans l'environnement:

- `Docling`
- `PyTesseract`

## Structure utile

```text
data/benchmarks/ocr/
  dataset/
    raw/                 # documents benchmark
    ground_truth/        # verite de reference
    manifests/
      benchmark_manifest.csv
  outputs/
    run_20260413_final_new_gt/
      predictions/
      metrics_per_sample.csv
      metrics_summary.csv
      metrics_summary.json
      run_config.json
```

## Definitions utiles

### Ground truth

Le `ground truth` est le texte de reference considere comme correct pour chaque document benchmark.

Chaque fichier dans `raw/` est associe a un fichier `.txt` dans `ground_truth/`.

### Manifest

Le fichier [benchmark_manifest.csv](../../../data/benchmarks/ocr/dataset/manifests/benchmark_manifest.csv) est la table de correspondance du benchmark.

Chaque ligne indique:

- l'identifiant du sample
- le chemin du document source
- le chemin du ground truth
- le type de source (`pdf` ou `image`)
- des metadonnees legeres

Le benchmark ne parcourt pas le dossier `raw/` automatiquement: il suit uniquement le manifest.

## Flow du benchmark

Le flux est le suivant:

`raw CV`
-> lecture via le manifest
-> extraction OCR par moteur
-> sauvegarde de la prediction texte
-> comparaison avec le ground truth
-> calcul de `WER` / `CER`
-> production des rapports de synthese

## Metriques produites

Le benchmark calcule:

- `raw_wer`
- `raw_cer`
- `normalized_wer`
- `normalized_cer`

Les scores `normalized_*` appliquent un nettoyage leger du texte avant comparaison:

- normalisation Unicode
- harmonisation des retours ligne
- reduction des espaces multiples
- suppression des espaces parasites en debut et fin

Plus `WER` et `CER` sont faibles, meilleur est l'OCR.

## Etat retenu dans ce depot

Le depot contient uniquement:

- le nouveau ground truth retenu
- le manifest aligne sur ce ground truth
- le run final utile

Les anciens runs lies a l'ancien ground truth ont ete supprimes pour garder un chantier lisible et coherent.

## Resultat utile

Sur le benchmark final retenu:

- `PyTesseract` obtient les meilleurs scores d'OCR brut
- `Docling` reste interessant pour la structuration documentaire

Ce resultat a servi a guider un tuning minimal de `Module 1`:

- `Docling` reste le parseur principal
- `PyTesseract` est renforce comme fallback sur les cas faibles

## Scripts utiles

- [run_ocr_benchmark.py](run_ocr_benchmark.py)
- [summarize_results.py](summarize_results.py)
- [dataset.py](dataset.py)
- [metrics.py](metrics.py)

## Sorties finales utiles

Le run de reference actuellement conserve est:

- [run_20260413_final_new_gt](../../../data/benchmarks/ocr/outputs/run_20260413_final_new_gt)

Fichiers principaux:

- [metrics_per_sample.csv](../../../data/benchmarks/ocr/outputs/run_20260413_final_new_gt/metrics_per_sample.csv)
- [metrics_summary.csv](../../../data/benchmarks/ocr/outputs/run_20260413_final_new_gt/metrics_summary.csv)
- [metrics_summary.json](../../../data/benchmarks/ocr/outputs/run_20260413_final_new_gt/metrics_summary.json)
- [run_config.json](../../../data/benchmarks/ocr/outputs/run_20260413_final_new_gt/run_config.json)
