from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Artifact availability checks
# ---------------------------------------------------------------------------

def _artifact_mode_available() -> bool:
    """True when matching artifact files (.jsonl) or decision cards exist."""
    features_dir = ROOT / "data/ranking/features"
    if features_dir.is_dir() and any(features_dir.glob("*.jsonl")):
        return True
    dc_files = [
        ROOT / "docs/reports/matching/v3/decision_cards_v3_normalized.json",
        ROOT / "docs/reports/decision_cards/decision_cards_with_transferability.json",
        ROOT / "docs/reports/decision_cards/decision_cards_ml_comparison.json",
    ]
    return any(p.exists() for p in dc_files)


def _demo_reports_available() -> bool:
    return (ROOT / "docs/reports/demo/demo_executive_summary.json").exists()


def _neo4j_report_available() -> bool:
    return (ROOT / "docs/reports/graph/neo4j_import_report.json").exists()


# ---------------------------------------------------------------------------
# Skip reasons
# ---------------------------------------------------------------------------

_ARTIFACT_SKIP_REASON = (
    "Artifact mode is legacy; live matching is the default. "
    "Run Module G/H pipeline to regenerate artifacts to re-enable these tests."
)

_REPORT_SKIP_REASON = (
    "Requires generated demo/neo4j reports not present in live-only setup."
)

# ---------------------------------------------------------------------------
# Test nodeids that depend on artifact files being present
# (auto-reactivate if artifacts are regenerated)
# ---------------------------------------------------------------------------

_ARTIFACT_DEPENDENT_NODEIDS = frozenset({
    # Health — checks matching_artifacts.available count
    "tests/test_api_health.py::test_health_endpoint_returns_service_status",
    # Chat — enriches candidate names from decision cards
    "tests/test_api_chat.py::test_chat_endpoint_enriches_candidate_names_and_preserves_order",
    # Candidates — reads from decision_cards artifact
    "tests/test_api_candidates.py::test_candidates_endpoint_returns_paginated_candidates",
    "tests/test_api_candidates.py::test_candidate_detail_endpoint_returns_card_and_optional_profile",
    "tests/test_api_candidates.py::test_candidate_detail_endpoint_returns_404_for_unknown_candidate",
    "tests/test_api_candidates.py::test_candidates_endpoint_falls_back_to_artifacts_when_mongodb_is_unavailable",
    # Match — artifact-specific assertions (resolved_job_id, artifact_source, etc.)
    # Note: _stays_available and _accepts_valid_api_key verify auth config but rely
    # on matching returning 200; they fail in artifact runner with no artifacts.
    "tests/test_api_match.py::test_match_endpoint_stays_available_when_auth_is_disabled",
    "tests/test_api_match.py::test_match_endpoint_accepts_valid_api_key_when_auth_is_enabled",
    "tests/test_api_match.py::test_match_endpoint_returns_matching_v3_artifact_results",
    "tests/test_api_match.py::test_match_endpoint_uses_known_job_id_artifact",
    "tests/test_api_match.py::test_match_endpoint_falls_back_for_unknown_job_id",
    "tests/test_api_match.py::test_match_endpoint_uses_decision_cards_when_artifact_directory_is_empty",
    "tests/test_api_match.py::test_match_endpoint_hybrid_falls_back_to_artifact_when_live_fails",
    "tests/test_api_match.py::test_match_endpoint_saves_artifact_trace_when_mongodb_backend_is_configured",
    # Decision cards — entire endpoint depends on artifact
    "tests/test_api_decision_cards.py::test_decision_cards_endpoint_returns_available_cards",
    "tests/test_api_decision_cards.py::test_decision_card_detail_endpoint_returns_one_card",
    "tests/test_api_decision_cards.py::test_decision_card_detail_endpoint_returns_404_for_unknown_candidate",
    "tests/test_api_decision_cards.py::test_decision_cards_endpoint_falls_back_to_artifacts_when_mongodb_is_unavailable",
    # Graph — depends on /api/candidates which reads decision cards
    "tests/test_api_graph.py::test_transferability_endpoint_returns_candidate_transferability",
    "tests/test_api_graph.py::test_transferability_endpoint_uses_yaml_fallback_without_neo4j",
    "tests/test_api_graph.py::test_transferability_endpoint_returns_404_for_unknown_candidate",
    "tests/test_api_graph.py::test_transferability_endpoint_preserves_stable_shape_with_neo4j",
})

# Tests that depend on generated pipeline reports (demo run, neo4j import)
_DEMO_REPORT_NODEIDS = frozenset({
    "tests/test_api_demo.py::test_demo_artifact_endpoints_return_reports",
    "tests/test_api_demo.py::test_demo_run_endpoint_regenerates_manifest",
})

_NEO4J_REPORT_NODEIDS = frozenset({
    "tests/test_neo4j_transferability.py::test_neo4j_import_report_has_expected_shape",
})


# ---------------------------------------------------------------------------
# Auto-skip hook
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    artifact_available = _artifact_mode_available()
    demo_available = _demo_reports_available()
    neo4j_available = _neo4j_report_available()

    artifact_skip = pytest.mark.skip(reason=_ARTIFACT_SKIP_REASON)
    report_skip = pytest.mark.skip(reason=_REPORT_SKIP_REASON)

    for item in items:
        nid = item.nodeid.replace("\\", "/")

        if not artifact_available and nid in _ARTIFACT_DEPENDENT_NODEIDS:
            item.add_marker(artifact_skip)
        elif not demo_available and nid in _DEMO_REPORT_NODEIDS:
            item.add_marker(report_skip)
        elif not neo4j_available and nid in _NEO4J_REPORT_NODEIDS:
            item.add_marker(report_skip)
