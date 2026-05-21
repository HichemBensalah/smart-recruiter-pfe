# Neo4j Graph-RAG read-only

## Objectif

Neo4j ajoute une couche Graph-RAG optionnelle au projet Smart Recruiter pour analyser les relations entre candidats, competences, roles et offres. Cette brique sert a expliquer la transferabilite metier, les gaps compensables et les gaps bloquants.

Neo4j ne remplace pas Matching V3. Matching V3 reste la baseline officielle de matching et de scoring. Le graphe sert a enrichir l'analyse et a preparer de futurs tools LangChain/LangGraph.

## Schema du graphe

Noeuds CDC :

- `Candidate {candidate_id, profile_id, name}`
- `Skill {name, normalized_name}`
- `Role {name, family}`
- `Job {job_id, title}`

Relations CDC :

- `(Candidate)-[:HAS_SKILL]->(Skill)`
- `(Role)-[:REQUIRES]->(Skill)`
- `(Role)-[:RELATED_TO]->(Skill)`
- `(Role)-[:TRANSITIONS_TO {condition_skills, rationale}]->(Role)`
- `(Job)-[:REQUIRES]->(Skill)`
- `(Candidate)-[:FITS_ROLE {matched_required_count}]->(Role)`

Relations conservees pour compatibilite :

- `(Role)-[:REQUIRES_SKILL]->(Skill)`
- `(Role)-[:HAS_ADJACENT_SKILL]->(Skill)`
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

Rapport genere :

```text
docs/reports/graph/neo4j_import_report.json
```

Le rapport contient `status`, `roles_count`, `skills_count`, `jobs_count`, `candidates_count`, `relations_count`, `fallback_available`, les noeuds CDC et les relations CDC.

## Exemples Cypher

Lister les roles :

```cypher
MATCH (r:Role)
RETURN r.name, r.family
ORDER BY r.name;
```

Lister les competences d'un candidat :

```cypher
MATCH (c:Candidate {candidate_id: $candidate_id})-[:HAS_SKILL]->(s:Skill)
RETURN s.name
ORDER BY toLower(s.name);
```

Comparer un candidat a un role :

```cypher
MATCH (r:Role {name: $role_name})-[:REQUIRES]->(s:Skill)
OPTIONAL MATCH (c:Candidate {candidate_id: $candidate_id})-[:HAS_SKILL]->(s)
RETURN s.name AS skill, c IS NOT NULL AS matched;
```

Lister les roles plausibles d'un candidat :

```cypher
MATCH (c:Candidate {candidate_id: $candidate_id})-[rel:FITS_ROLE]->(r:Role)
RETURN r.name, rel.matched_required_count
ORDER BY rel.matched_required_count DESC;
```

Trouver les transitions vers un role cible :

```cypher
MATCH (source:Role)-[rel:TRANSITIONS_TO]->(target:Role {name: $target_role})
RETURN source.name, target.name, rel.condition_skills, rel.rationale;
```

## API

- `GET /api/graph/transferability/{candidate_id}?target_role=Backend Developer`
- `GET /api/graph/neo4j/status`
- `GET /api/graph/neo4j/roles`
- `GET /api/graph/neo4j/candidate/{candidate_id}/skills`
- `GET /api/graph/neo4j/transferability/{candidate_id}?target_role=Backend Developer`
- `GET /api/graph/neo4j/gaps/{candidate_id}?target_role=Backend Developer`

L'endpoint principal `/api/graph/transferability/{candidate_id}` tente Neo4j en premier. Si Neo4j n'est pas configure ou indisponible, il retourne le fallback YAML avec :

- `source = yaml_fallback`
- `fallback_used = true`
- `warnings` indiquant la cause du fallback

Quand Neo4j repond, il retourne :

- `source = neo4j`
- `fallback_used = false`

## Fallback YAML

Le fichier `data/graph/skills_roles_graph.yaml` reste le fallback stable. Si Neo4j n'est pas configure ou lance, l'API principale continue de fonctionner et les endpoints Neo4j retournent une erreur controlee.

## Limites

- Le graphe depend des competences structurees par Module 2.
- Les transitions de roles sont declaratives et doivent etre validees metier.
- Neo4j n'est pas utilise pour remplacer Matching V3.
- Cette couche ne prend pas de decision recruteur finale.
- En CI, les tests n'exigent pas une instance Neo4j reelle ; ils verifient le fallback et la forme des reponses.

## Statut MVP

Neo4j est exploitable comme extension read-only si une instance est configuree et si l'import est lance. Pour une demo sans service externe, le fallback YAML reste le chemin stable et teste.
