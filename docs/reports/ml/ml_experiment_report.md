# ML Experiment Report - Aligned Ranking Dataset

## Scope
Baseline experimentale controlee sur le dataset aligne. Les labels utilises sont des pseudo-labels metier controles, pas des labels recruteur.

## Dataset
- Dataset: `data\ranking\datasets\ranking_dataset_aligned_pseudo_labeled.jsonl`
- Target: `label_binary`
- Rows: 250
- Label distribution: {'0': 104, '1': 103, '2': 32, '3': 11}
- Label binary distribution: {'0': 207, '1': 43}

## Split
- LeaveOneGroupOut par job_id: entrainement sur 4 offres, test sur 1 offre.

## Features
- `vector_similarity`
- `final_score_v3`
- `must_have_coverage`
- `required_skills_overlap`
- `nice_to_have_overlap`
- `experience_match_score`
- `seniority_alignment`
- `profile_quality_score`
- `reliability_score`
- `hallucination_risk_encoded`
- `missing_required_count`
- `matched_required_count`

## Mean Metrics
### logistic_regression
- Classification: {'accuracy': 0.9960000000000001, 'macro_f1': 0.9946666666666667, 'precision': 0.9846153846153847, 'recall': 1.0, 'roc_auc': 0.9978070175438596}
- Ranking: {'precision@5': 1.0, 'precision@10': 0.74, 'ndcg@10': 0.9861137556126446, 'mrr': 1.0}

### random_forest
- Classification: {'accuracy': 0.9960000000000001, 'macro_f1': 0.9946666666666667, 'precision': 0.9846153846153847, 'recall': 1.0, 'roc_auc': 0.9986842105263157}
- Ranking: {'precision@5': 1.0, 'precision@10': 0.76, 'ndcg@10': 1.0, 'mrr': 1.0}

### xgboost
- Classification: {'accuracy': 0.9879999999999999, 'macro_f1': 0.983701754385965, 'precision': 0.9846153846153847, 'recall': 0.9692307692307693, 'roc_auc': 0.9969298245614034}
- Ranking: {'precision@5': 1.0, 'precision@10': 0.74, 'ndcg@10': 0.9872758423602097, 'mrr': 1.0}

## Matching V3 Baseline
- Score used for comparison only: `final_score_v3`
- Metrics: {'precision@5': 1.0, 'precision@10': 0.76, 'ndcg@10': 1.0, 'mrr': 1.0}

## Methodological Warnings
- Les labels sont des pseudo-labels metier controles.
- Ce ne sont pas des labels recruteur.
- Le dataset reste petit.
- L'evaluation mesure une experimentation ML, pas un modele final valide.

## Decision
Modele experimental acceptable comme baseline de recherche, non acceptable comme modele final production sans labels recruteur et validation supplementaire.
