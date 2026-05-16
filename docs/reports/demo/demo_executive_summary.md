# Executive summary - Smart Recruiter

## Objectif

Fournir une version courte de la démo: 3 candidats recommandés, 3 candidats à vérifier, signaux clés et limites.

## Résumé court

- `job_id`: `backend_python_django_postgresql`
- candidats recommandés: `3`
- candidats à vérifier: `3`
- transitions plausibles dans le top 10 source: `5`
- review_needed dans le top 10 source: `5`

## Top 3 candidats recommandés

### `candidate_9b0508063f03`

- Matching V3: `0.7682`
- Random Forest: `1.0000`
- XGBoost: `0.9826`
- Transferability: `0.3000`
- Gaps bloquants: Aucun
- Raison executive: Recommandé car score Matching V3 élevé, Random Forest très favorable, XGBoost favorable, pas de gap bloquant détecté.

### `candidate_1487f3187f7b`

- Matching V3: `0.7183`
- Random Forest: `1.0000`
- XGBoost: `0.9718`
- Transferability: `0.5333`
- Gaps bloquants: SQL
- Raison executive: Recommandé car score Matching V3 élevé, Random Forest très favorable, XGBoost favorable, transition métier plausible.

### `candidate_8eea1b635447`

- Matching V3: `0.6688`
- Random Forest: `1.0000`
- XGBoost: `0.9676`
- Transferability: `0.5333`
- Gaps bloquants: SQL
- Raison executive: Recommandé car score Matching V3 élevé, Random Forest très favorable, XGBoost favorable, transition métier plausible.


## Top 3 candidats à vérifier

### `candidate_664415ab2fe1`

- Matching V3: `0.4256`
- Random Forest: `0.0000`
- XGBoost: `0.0037`
- Transferability: `0.1667`
- Gaps bloquants: Git
- Raison executive: À vérifier car statut review_needed, désaccord important entre les rangs, gaps bloquants: Git, score XGBoost très faible.

### `candidate_073b7a3d39ba`

- Matching V3: `0.4297`
- Random Forest: `0.0000`
- XGBoost: `0.0038`
- Transferability: `0.2917`
- Gaps bloquants: Git
- Raison executive: À vérifier car statut review_needed, désaccord important entre les rangs, gaps bloquants: Git, score XGBoost très faible.

### `candidate_4eef95d3a3fa`

- Matching V3: `0.4284`
- Random Forest: `0.0000`
- XGBoost: `0.0040`
- Transferability: `0.1667`
- Gaps bloquants: Git
- Raison executive: À vérifier car statut review_needed, désaccord important entre les rangs, gaps bloquants: Git, score XGBoost très faible.


## Signaux clés utilisés

- Score et rang Matching V3.
- Score Random Forest.
- Score XGBoost et SHAP top features dans le rapport source.
- Potential Graph: transférabilité, transitions plausibles, gaps compensables et bloquants.

## Limites méthodologiques

- Cette synthèse agrège Matching V3, modèles ML entraînés sur pseudo-labels métier contrôlés, SHAP et Potential Graph. Matching V3 reste la baseline officielle, Random Forest est le meilleur modèle ML actuel selon les métriques observées, XGBoost est utilisé pour l’analyse SHAP, et Potential Graph sert à analyser la transférabilité métier.
- Cette synthèse est faite pour une démo rapide et ne constitue pas une décision recruteur.

## Prochaine étape

Valider les candidats recommandés et les candidats à vérifier avec un recruteur ou un expert métier.
