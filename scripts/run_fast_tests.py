from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


FAST_TESTS = [
    "tests/test_api_health.py",
    "tests/test_api_chat.py",
    "tests/test_chat_memory.py",
    "tests/test_api_candidates.py",
    "tests/test_api_match.py",
    "tests/test_live_matcher.py",
    "tests/test_live_matcher_mongodb_resolution.py",
    "tests/test_live_matcher_dedup.py",
    "tests/test_api_decision_cards.py",
    "tests/test_mongodb_repositories.py",
    "tests/test_api_graph.py",
    "tests/test_api_demo.py",
    "tests/test_langchain_tools_api_client.py",
    "tests/test_langchain_tools_registry.py",
    "tests/test_langchain_tools_contracts.py",
    "tests/test_langgraph_copilot_state.py",
    "tests/test_langgraph_copilot_nodes.py",
    "tests/test_langgraph_copilot_graph.py",
    "tests/test_job_intake.py",
    "tests/test_job_intake_field_edit.py",
    "tests/test_job_intake_offer_summary.py",
    "tests/test_job_intake_reset.py",
    "tests/test_job_intake_single_path.py",
    "tests/test_runtime_store.py",
    "tests/test_session_isolation.py",
    "tests/test_reference_resolver.py",
    "tests/test_streamlit_app_static.py",
    "tests/test_neo4j_transferability.py",
    "tests/test_api_graph_neo4j.py",
    "tests/test_docker_configuration.py",
    "tests/test_e2e_main_scenario.py",
    "tests/test_e2e_routing.py",
]


def main() -> int:
    Path(".tmp/pytest_ci_fast").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "DATA_BACKEND": "artifacts",
            "MATCHING_MODE": "artifact",
            "ALLOW_ARTIFACT_FALLBACK": "true",
            "AUTH_ENABLED": "false",
        }
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        *FAST_TESTS,
        "-q",
        "-p",
        "no:cacheprovider",
        "--basetemp=.tmp/pytest_ci_fast",
    ]
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
