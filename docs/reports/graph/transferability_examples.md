# Potential Graph YAML - exemples de transférabilité

## Objectif

Tester une couche YAML simple et explicable pour analyser le fit direct, les transitions de rôle et les gaps métier.

| candidate_id | job_id | target_role | fit_direct | direct_fit_score | transferability_score | gaps compensables | gaps bloquants |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| `docx_Aziz_resumer` | `data_engineer_python_sql` | Data Engineer | False | 0.2000 | 0.2367 | Python, Pandas | ETL, Data Pipeline |
| `docx_Aziz_resumer` | `backend_python_django_postgresql` | Backend Developer | False | 0.2000 | 0.1500 | Python, REST API, Django, PostgreSQL | Git |
| `docx_0_anonyme` | `devops_docker_kubernetes` | DevOps Engineer | False | 0.0000 | 0.0000 | Git | Docker, Kubernetes, Linux, CI/CD |

## Limites méthodologiques

- Le graphe est déclaratif et YAML-based, pas appris automatiquement.
- Les scores dépendent des skills structurées dans les profils candidats.
- Les transitions plausibles sont des règles explicables, pas des décisions recruteur.
- Cette brique ne modifie pas Matching V3, FAISS, MongoDB, les datasets ou les modèles ML.
