from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_TOP_K = 50
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Matching V3 and ranking feature export for multiple job profiles.")
    parser.add_argument("--job-profiles-dir", type=Path, default=Path("data/job_profiles"))
    parser.add_argument("--matching-output-dir", type=Path, default=Path("docs/reports/matching/v3"))
    parser.add_argument("--features-output-dir", type=Path, default=Path("data/ranking/features"))
    parser.add_argument("--profiles-dir", type=Path, default=Path("data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles"))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--only-job-id", default=None)
    parser.add_argument("--include-job-ids", default=None, help="Comma-separated job_id allow-list.")
    return parser.parse_args()


def parse_job_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def discover_job_profiles(
    job_profiles_dir: Path,
    only_job_id: str | None = None,
    include_job_ids: set[str] | None = None,
) -> list[Path]:
    profiles: list[Path] = []
    for path in sorted(job_profiles_dir.glob("*.json")):
        if path.name == "job_profile_builder_report.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        job_id = payload.get("job_id") or path.stem
        if only_job_id and job_id != only_job_id:
            continue
        if include_job_ids is not None and job_id not in include_job_ids:
            continue
        profiles.append(path)
    return profiles


def count_jsonl_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def run_command(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    args.matching_output_dir.mkdir(parents=True, exist_ok=True)
    args.features_output_dir.mkdir(parents=True, exist_ok=True)

    processed: list[dict[str, object]] = []
    warnings: list[str] = []
    include_job_ids = parse_job_ids(args.include_job_ids)
    for job_profile_path in discover_job_profiles(args.job_profiles_dir, args.only_job_id, include_job_ids):
        job_profile = json.loads(job_profile_path.read_text(encoding="utf-8"))
        job_id = job_profile.get("job_id") or job_profile_path.stem
        matching_report_path = args.matching_output_dir / f"{job_id}_matching_report_v3_normalized.json"
        features_path = args.features_output_dir / f"{job_id}.jsonl"

        run_command(
            [
                sys.executable,
                "scripts/run_matching_v3_normalized.py",
                "--job-profile",
                str(job_profile_path),
                "--output-report",
                str(matching_report_path),
                "--job-id",
                str(job_id),
                "--top-k",
                str(args.top_k),
            ]
        )
        run_command(
            [
                sys.executable,
                "scripts/build_ranking_features.py",
                "--job-profile",
                str(job_profile_path),
                "--matching-report",
                str(matching_report_path),
                "--profiles-dir",
                str(args.profiles_dir),
                "--output",
                str(features_path),
                "--job-id",
                str(job_id),
            ]
        )

        processed.append(
            {
                "job_id": job_id,
                "job_profile": str(job_profile_path),
                "matching_report": str(matching_report_path),
                "features": str(features_path),
                "feature_rows": count_jsonl_rows(features_path),
            }
        )

    if args.only_job_id and not processed:
        warnings.append(f"No job profile found for only_job_id={args.only_job_id}")
    if include_job_ids:
        processed_job_ids = {str(item["job_id"]) for item in processed}
        missing_job_ids = sorted(include_job_ids - processed_job_ids)
        for job_id in missing_job_ids:
            warnings.append(f"No job profile found for include_job_id={job_id}")

    summary = {
        "offers_processed": len(processed),
        "matching_reports": [item["matching_report"] for item in processed],
        "feature_files": [item["features"] for item in processed],
        "feature_rows_by_job": {str(item["job_id"]): item["feature_rows"] for item in processed},
        "total_feature_rows": sum(int(item["feature_rows"]) for item in processed),
        "warnings": warnings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
