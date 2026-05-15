# Note méthodologique SHAP

SHAP explique le comportement du modèle XGBoost expérimental sauvegardé.

Ce modèle est entraîné sur des pseudo-labels métier contrôlés, pas sur des labels recruteur. Ces pseudo-labels sont dérivés de règles métier construites à partir de features comme `must_have_coverage`, `experience_match_score`, `reliability_score` ou d'autres signaux de matching.

Il existe donc une circularité partielle : certaines variables qui ont contribué à produire la cible pseudo-labellisée sont aussi utilisées comme entrées du modèle. Les explications SHAP montrent principalement quelles features permettent au modèle de reproduire la règle métier de pseudo-labeling.

Elles ne doivent pas être interprétées comme une vérité recruteur, ni comme une explication de décisions humaines réelles.

Une validation humaine/recruteur reste nécessaire pour conclure sur la qualité métier finale du modèle.
