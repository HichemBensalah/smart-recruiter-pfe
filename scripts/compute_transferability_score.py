from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.graph.transferability import compute_transferability_from_paths  # noqa: E402


DEFAULT_EXAMPLES = [
    (
        Path("data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/docx_Aziz_resumer.json"),
        Path("data/job_profiles/data_engineer_python_sql.json"),
    ),
    (
        Path("data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/docx_Aziz_resumer.json"),
        Path("data/job_profiles/backend_python_django_postgresql.json"),
    ),
    (
        Path("data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/docx_0_anonyme.json"),
        Path("data/job_profiles/devops_docker_kubernetes.json"),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute YAML-based role transferability score.")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=Path("data/graph/skills_roles_graph.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Potential Graph YAML - exemples de transférabilité",
        "",
        "## Objectif",
        "",
        "Tester une couche YAML simple et explicable pour analyser le fit direct, les transitions de rôle et les gaps métier.",
        "",
        "| candidate_id | job_id | target_role | fit_direct | direct_fit_score | transferability_score | gaps compensables | gaps bloquants |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for item in examples:
        lines.append(
            f"| `{item['candidate_id']}` | `{item['job_id']}` | {item['target_role']} | "
            f"{item['fit_direct']} | {item['direct_fit_score']:.4f} | {item['transferability_score']:.4f} | "
            f"{', '.join(item['gaps_compensables']) or 'Aucun'} | {', '.join(item['gaps_bloquants']) or 'Aucun'} |"
        )
    lines.extend(
        [
            "",
            "## Limites méthodologiques",
            "",
            "- Le graphe est déclaratif et YAML-based, pas appris automatiquement.",
            "- Les scores dépendent des skills structurées dans les profils candidats.",
            "- Les transitions plausibles sont des règles explicables, pas des décisions recruteur.",
            "- Cette brique ne modifie pas Matching V3, FAISS, MongoDB, les datasets ou les modèles ML.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_examples(graph_path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for profile_path, job_path in DEFAULT_EXAMPLES:
        if profile_path.exists() and job_path.exists():
            examples.append(compute_transferability_from_paths(profile_path, job_path, graph_path))
    return examples


def main() -> None:
    args = parse_args()
    result = compute_transferability_from_paths(args.profile, args.job, args.graph)
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    examples = generate_examples(args.graph)
    if examples:
        write_json(Path("docs/reports/graph/transferability_examples.json"), {"examples": examples})
        write_markdown(Path("docs/reports/graph/transferability_examples.md"), examples)


if __name__ == "__main__":
    main()

