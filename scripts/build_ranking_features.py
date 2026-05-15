from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.ranking.features import build_feature_rows, load_candidate_profiles_by_source_path, load_json, write_jsonl
from src.core.ranking.schema import RankingFeatures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML ranking feature rows from Matching V3 normalized output.")
    parser.add_argument("--job-profile", required=True, type=Path)
    parser.add_argument("--matching-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--profiles-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_profile = load_json(args.job_profile)
    matching_report = load_json(args.matching_report)
    job_id = args.job_id or args.job_profile.stem

    profile_lookup = load_candidate_profiles_by_source_path(args.profiles_dir)
    rows = build_feature_rows(job_profile, matching_report, job_id, profile_lookup)
    write_jsonl(rows, args.output)

    feature_count = len(RankingFeatures.model_fields)
    warnings: list[str] = []
    if not rows:
        warnings.append("No candidate rows were found in the matching report.")
    if args.profiles_dir and not profile_lookup:
        warnings.append(f"No candidate profiles loaded from {args.profiles_dir}.")
    zero_seniority = sum(1 for row in rows if row.features.seniority_alignment == 0.0)
    if zero_seniority:
        warnings.append(f"{zero_seniority} rows have seniority_alignment=0.0.")

    print(f"rows_generated: {len(rows)}")
    print(f"job_id: {job_id}")
    print(f"output_path: {args.output}")
    print(f"features_per_line: {feature_count}")
    print(f"warnings: {warnings if warnings else 'none'}")


if __name__ == "__main__":
    main()
