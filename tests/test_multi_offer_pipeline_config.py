from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_multi_offer_ranking_pipeline import DEFAULT_TOP_K, discover_job_profiles


def test_multi_offer_default_top_k_is_50() -> None:
    assert DEFAULT_TOP_K == 50


def test_discover_job_profiles_excludes_report_file() -> None:
    profiles = discover_job_profiles(ROOT / "data/job_profiles")
    assert all(path.name != "job_profile_builder_report.json" for path in profiles)
    assert len(profiles) >= 5


def test_discover_job_profiles_supports_only_job_id() -> None:
    profiles = discover_job_profiles(ROOT / "data/job_profiles", only_job_id="frontend_react_nextjs")
    assert len(profiles) == 1
    payload = json.loads(profiles[0].read_text(encoding="utf-8"))
    assert payload["job_id"] == "frontend_react_nextjs"


def test_expected_output_path_shapes() -> None:
    job_id = "frontend_react_nextjs"
    assert Path("docs/reports/matching/v3") / f"{job_id}_matching_report_v3_normalized.json"
    assert Path("data/ranking/features") / f"{job_id}.jsonl"
