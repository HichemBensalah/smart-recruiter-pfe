from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_candidate_corpus.py"
PROFILES_DIR = ROOT / "data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles"
JOB_PROFILES_DIR = ROOT / "data/job_profiles"
OUTPUT_JSON = ROOT / "docs/reports/ml/candidate_corpus_analysis.json"
OUTPUT_MD = ROOT / "docs/reports/ml/candidate_corpus_analysis.md"
UNLABELED_DATASET = ROOT / "data/ranking/datasets/ranking_dataset_unlabeled.jsonl"
PSEUDO_DATASET = ROOT / "data/ranking/datasets/ranking_dataset_pseudo_labeled.jsonl"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_analysis(output_json: Path = OUTPUT_JSON, output_md: Path = OUTPUT_MD) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--profiles-dir",
            str(PROFILES_DIR),
            "--job-profiles-dir",
            str(JOB_PROFILES_DIR),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=ROOT,
        check=True,
    )


def test_script_generates_candidate_corpus_report() -> None:
    run_analysis()
    assert OUTPUT_JSON.exists()
    assert OUTPUT_MD.exists()


def test_report_contains_top_skills_and_seniority_distribution() -> None:
    report = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert report["total_profiles"] > 0
    assert report["top_skills"]
    assert "seniority_distribution" in report
    assert "unknown" in report["seniority_distribution"]


def test_report_contains_job_alignment_with_decisions() -> None:
    report = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert report["job_alignment"]
    for job in report["job_alignment"]:
        assert job["decision"] in {"keep", "adjust", "replace"}
        assert "required_skill_frequencies" in job


def test_recommended_offers_have_required_skills() -> None:
    report = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert report["recommended_job_profiles"]
    for job in report["recommended_job_profiles"]:
        assert job["required_skills"]
        assert job["expected_candidate_pool_size"] > 0


def test_analysis_does_not_modify_existing_datasets(tmp_path: Path) -> None:
    before_unlabeled = sha256_file(UNLABELED_DATASET)
    before_pseudo = sha256_file(PSEUDO_DATASET)
    run_analysis(tmp_path / "candidate_corpus_analysis.json", tmp_path / "candidate_corpus_analysis.md")
    assert sha256_file(UNLABELED_DATASET) == before_unlabeled
    assert sha256_file(PSEUDO_DATASET) == before_pseudo
