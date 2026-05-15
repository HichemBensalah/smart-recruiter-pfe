FAISS = retrieval rapide
Matching V3 = scoring métier officiel
CrossEncoder = expérience de reranking secondaire
XGBoost = futur modèle supervisé si labels
------------------------------------------------------------------
Architecture propre actuelle
✅ BASELINE ACTUELLE

Job Profile
→ FAISS retrieval
→ Matching V3 scoring
→ Decision Cards
Architecture expérimentale
🧪 EXPÉRIMENT

Job Profile
→ FAISS top-N
→ CrossEncoder reranking
→ comparaison avec Matching V3

ou version contrôlée :

Job Profile
→ FAISS top-N
→ filtre must_have_coverage >= 0.6
→ CrossEncoder
→ rapport d’ablation
Mais ce n’est pas la baseline.
---------------------------------------------------------
Architecture future si tu as des labels
🔮 FUTURE WORK

Job Profile
→ FAISS retrieval
→ features candidat-job
→ XGBoost Ranker
→ SHAP explanations
→ Decision Cards avancées

Là, XGBoost pourrait remplacer une partie du scoring heuristique Matching V3. Mais seulement si tu as des labels.