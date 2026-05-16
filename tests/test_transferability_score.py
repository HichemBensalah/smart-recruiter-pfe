from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.graph.transferability import compute_transferability_from_paths, load_yaml_graph, role_names  # noqa: E402


GRAPH = ROOT / "data/graph/skills_roles_graph.yaml"
PROFILE = ROOT / "data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles/docx_Aziz_resumer.json"
JOB = ROOT / "data/job_profiles/backend_python_django_postgresql.json"
MATCHING_V3_REPORT = ROOT / "docs/reports/matching/v3/matching_report_v3_normalized.json"

REQUIRED_ROLES = {
    "Backend Developer",
    "Frontend Developer",
    "Full Stack Developer",
    "Data Engineer",
    "Data Analyst",
    "Machine Learning Engineer",
    "DevOps Engineer",
    "Business Intelligence Analyst",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_yaml_loads_and_contains_required_roles() -> None:
    graph = load_yaml_graph(GRAPH)
    assert REQUIRED_ROLES <= set(role_names(graph))


def test_each_role_has_core_and_adjacent_skills() -> None:
    graph = load_yaml_graph(GRAPH)
    for role in graph["roles"]:
        assert role["core_skills"]
        assert role["adjacent_skills"]
        assert "transitions_to" in role


def test_compute_transferability_score_shape() -> None:
    result = compute_transferability_from_paths(PROFILE, JOB, GRAPH)
    assert 0.0 <= result["transferability_score"] <= 1.0
    assert 0.0 <= result["direct_fit_score"] <= 1.0
    assert isinstance(result["fit_direct"], bool)
    assert isinstance(result["gaps_compensables"], list)
    assert isinstance(result["gaps_bloquants"], list)


def test_script_generates_report_without_modifying_inputs(tmp_path: Path) -> None:
    profile_before = _sha256(PROFILE)
    matching_before = _sha256(MATCHING_V3_REPORT)
    output = tmp_path / "transferability_example_single.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/compute_transferability_score.py",
            "--profile",
            str(PROFILE),
            "--job",
            str(JOB),
            "--graph",
            str(GRAPH),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    examples = ROOT / "docs/reports/graph/transferability_examples.json"
    assert output.exists()
    assert examples.exists()
    assert payload["candidate_id"]
    assert _sha256(PROFILE) == profile_before
    assert _sha256(MATCHING_V3_REPORT) == matching_before

