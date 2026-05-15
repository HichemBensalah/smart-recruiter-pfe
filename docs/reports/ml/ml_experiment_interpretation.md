# Interprétation méthodologique de l'expérimentation ML

## Objectif de l'expérience
Documenter une première expérimentation ML contrôlée sur le dataset aligné pseudo-labellisé.

## Dataset et cible
- Dataset: `data\ranking\datasets\ranking_dataset_aligned_pseudo_labeled.jsonl`
- Target: `label_binary`
- Nombre de lignes: 250
- Distribution label 0/1/2/3: {'0': 104, '1': 103, '2': 32, '3': 11}
- Distribution label_binary: {'0': 207, '1': 43}

## Modèles et stratégie d'évaluation
- Modèles entraînés: logistic_regression, random_forest, xgboost
- Split: LeaveOneGroupOut par job_id: entrainement sur 4 offres, test sur 1 offre.

## Comparaison Matching V3 vs modèles ML
- logistic_regression: ranking(precision@5=1.0, precision@10=0.74, ndcg@10=0.9861137556126446, mrr=1.0)
  classification(accuracy=0.9960000000000001, macro_f1=0.9946666666666667, precision=0.9846153846153847, recall=1.0, roc_auc=0.9978070175438596)
- random_forest: ranking(precision@5=1.0, precision@10=0.76, ndcg@10=1.0, mrr=1.0)
  classification(accuracy=0.9960000000000001, macro_f1=0.9946666666666667, precision=0.9846153846153847, recall=1.0, roc_auc=0.9986842105263157)
- xgboost: ranking(precision@5=1.0, precision@10=0.74, ndcg@10=0.9872758423602097, mrr=1.0)
  classification(accuracy=0.9879999999999999, macro_f1=0.983701754385965, precision=0.9846153846153847, recall=0.9692307692307693, roc_auc=0.9969298245614034)
- matching_v3_baseline: ranking(precision@5=1.0, precision@10=0.76, ndcg@10=1.0, mrr=1.0)

## Pourquoi les métriques sont très élevées ?
Les métriques sont presque parfaites parce que l'expérience contient une circularité partielle. Les pseudo-labels sont dérivés de règles métier, puis certaines features ayant servi à produire ces pseudo-labels, comme `must_have_coverage`, `experience_match_score` ou `reliability_score`, sont réutilisées comme entrées des modèles. Le modèle apprend donc principalement à reproduire la logique de pseudo-labeling, pas encore à imiter une décision recruteur indépendante.

Cette observation ne rend pas l'expérience inutile. Elle valide que le pipeline ML fonctionne de bout en bout, depuis le dataset jusqu'aux modèles sauvegardés et aux métriques LeaveOneGroupOut. En revanche, elle ne démontre pas encore une supériorité réelle sur des labels recruteur.

## Limites méthodologiques
- Les labels sont des pseudo-labels métier contrôlés, pas des labels recruteur.
- Le dataset reste petit avec 250 lignes et 43 exemples positifs binaires.
- La cible label_binary est construite à partir des labels 0/1/2/3 pseudo-labellisés.
- final_score_v3 est utilisé seulement comme score de baseline Matching V3, jamais comme label.
- Les métriques très élevées doivent être interprétées comme un contrôle technique du pipeline.

## Conclusion
L'expérimentation valide techniquement la construction du dataset, la génération de pseudo-labels, l'entraînement de modèles, l'évaluation LeaveOneGroupOut, la comparaison avec Matching V3 et la sauvegarde des modèles.

Elle ne valide pas encore un modèle final, un modèle prêt pour production, ni une supériorité réelle sur des décisions recruteur.

## Décision
- Matching V3 reste la baseline officielle.
- Random Forest est une baseline ML expérimentale.
- Logistic Regression est une baseline ML simple.
- XGBoost: baseline ML expérimentale testée.
- La validation humaine/recruteur reste nécessaire pour conclure.

## Prochaine étape
Construire un petit jeu de labels humains/recruteur et réévaluer les modèles sans conclure uniquement à partir des pseudo-labels.
