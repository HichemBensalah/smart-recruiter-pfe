Le projet Smart Recruiter PFE est un pipeline d’aide à la présélection RH. L’objectif n’est pas de remplacer le recruteur, mais de l’aider à prioriser les CV les plus pertinents pour une offre donnée.

Le pipeline officiel est :
CV bruts → Module 1 Parser → Module 2 V2 Grounded → MongoDB → FAISS → Matching V3 normalized → Decision Cards v3 normalized.

Aujourd’hui, la démo est techniquement prête : Module 1 a accepté 90 CV, Module 2 a généré 90 profils grounded, FAISS indexe les profils, Matching V3 produit 10 recommandations, et les Decision Cards expliquent ces recommandations de façon lisible.

FAISS est utilisé pour récupérer rapidement les profils proches de l’offre. Matching V3 ajoute ensuite la logique métier : couverture des compétences obligatoires, compétences matchées, compétences manquantes, qualité du profil, fiabilité et pénalités de risque. C’est pour ça que FAISS + Matching V3 reste la baseline officielle.

J’ai aussi testé CrossEncoder comme expérimentation. Le CrossEncoder non contraint donne un signal sémantique intéressant, mais il peut remonter des candidats avec faible couverture des must-have skills. Avec un filtre `must_have_coverage >= 0.6`, il devient plus sûr, mais trop restrictif. Donc il reste expérimental et ne remplace pas la baseline.

Les limites actuelles sont claires : la validation est faite surtout sur un job profile de démo, la qualité dépend du parsing et de la structuration en amont, et le système reste une aide à la décision humaine.

La suite logique serait de tester plusieurs offres, d’étudier un reranking hybride contrôlé, puis d’envisager XGBoost + SHAP uniquement si on collecte des labels recruteur fiables. L’API, l’UI et le chatbot restent des extensions, pas le cœur validé aujourd’hui.