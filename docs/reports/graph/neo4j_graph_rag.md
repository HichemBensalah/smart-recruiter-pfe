# Neo4j Graph-RAG read-only

## Objectif

Neo4j ajoute une couche Graph-RAG optionnelle au projet Smart Recruiter pour analyser les relations entre candidats, compétences, rôles et offres. Cette brique sert à expliquer la transférabilité métier, les gaps compensables et les gaps bloquants.

Neo4j ne remplace pas Matching V3. Matching V3 reste la baseline officielle de matching et de scoring. Le graphe sert à enrichir l'analyse et à préparer de futurs tools LangChain/LangGraph.

## Schéma du graphe

Noeuds :

- `Candidate {candidate_id, profile_id, name}`
- `Skill {name, normalized_name}`
- `Role {name, family}`
- `Job {job_id, title}`

Relations :

- `(Candidate)-[:HAS_SKILL]->(Skill)`
- `(Role)-[:REQUIRES_SKILL]->(Skill)`
- `(Role)-[:HAS_ADJACENT_SKILL]->(Skill)`
- `(Role)-[:TRANSITIONS_TO {condition_skills, rationale}]->(Role)`
- `(Job)-[:REQUIRES_SKILL]->(Skill)`

## Import

```bash
python scripts/import_graph_to_neo4j.py \
  --graph data/graph/skills_roles_graph.yaml \
  --profiles-dir data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles \
  --jobs-dir data/job_profiles \
  --reset
```

Variables d'environnement :

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## Exemples Cypher

Lister les rôles :

```cypher
MATCH (r:Role)
RETURN r.name, r.family
ORDER BY r.name;
```

Lister les compétences d'un candidat :

```cypher
MATCH (c:Candidate {candidate_id: $candidate_id})-[:HAS_SKILL]->(s:Skill)
RETURN s.name
ORDER BY toLower(s.name);
```

Comparer un candidat à un rôle :

```cypher
MATCH (r:Role {name: $role_name})-[:REQUIRES_SKILL]->(s:Skill)
OPTIONAL MATCH (c:Candidate {candidate_id: $candidate_id})-[:HAS_SKILL]->(s)
RETURN s.name AS skill, c IS NOT NULL AS matched;
```

Trouver les transitions vers un rôle cible :

```cypher
MATCH (source:Role)-[rel:TRANSITIONS_TO]->(target:Role {name: $target_role})
RETURN source.name, target.name, rel.condition_skills, rel.rationale;
```

## API

- `GET /api/graph/neo4j/status`
- `GET /api/graph/neo4j/roles`
- `GET /api/graph/neo4j/candidate/{candidate_id}/skills`
- `GET /api/graph/neo4j/transferability/{candidate_id}?target_role=Backend Developer`
- `GET /api/graph/neo4j/gaps/{candidate_id}?target_role=Backend Developer`

## Fallback YAML

Le fichier `data/graph/skills_roles_graph.yaml` reste le fallback stable. Si Neo4j n'est pas configuré ou lancé, l'API principale continue de fonctionner et les endpoints Neo4j retournent une erreur contrôlée.

## Limites

- Le graphe dépend des compétences structurées par Module 2.
- Les transitions de rôles sont déclaratives et doivent être validées métier.
- Neo4j n'est pas utilisé pour remplacer Matching V3.
- Cette couche ne prend pas de décision recruteur finale.
