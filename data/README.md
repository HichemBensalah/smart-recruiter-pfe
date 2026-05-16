# `data/`

Donnees, artefacts intermediaires et sorties techniques du pipeline.

## Dossiers importants

- `raw_cv/` : CV bruts locaux. A ne pas pousser si donnees sensibles.
- `processed_official_module1/` : sorties officielles du parsing Module 1.
- `profile_builder_module2_v2_grounded_all/` : profils grounded Module 2.
- `job_profiles/` : offres structurees.
- `indexes/faiss/` : index FAISS et mapping.
- `ranking/` : features, datasets pseudo-labelises et modeles ML experimentaux.
- `graph/` : Potential Graph YAML.
- `archive_old_runs/` : anciens essais conserves pour historique.

## Regle

Ne pas modifier manuellement les datasets, modeles ou index. Les changements doivent passer par les scripts dedies et etre documentes.
