from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASELINE_NAME = (
    "CV bruts -> Module 1 Parser -> Module 2 V2 Grounded -> MongoDB -> "
    "FAISS -> Matching V3 normalized -> Decision Cards v3 normalized"
)

OUTPUT_PATH = Path("docs/reports/demo/demo_readiness_check.json")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_exists_and_readable(path: Path, *, binary: bool = False) -> tuple[bool, bool, str | None]:
    if not path.exists():
        return False, False, "missing"
    try:
        if binary:
            _ = path.read_bytes()
        else:
            _ = path.read_text(encoding="utf-8")
        return True, True, None
    except Exception as exc:  # pragma: no cover - defensive
        return True, False, f"unreadable: {type(exc).__name__}: {exc}"


def build_check(
    *,
    check_id: str,
    description: str,
    path: str,
    expected: Any,
    actual: Any,
    status: str,
    message: str = "",
) -> dict[str, Any]:
    return {
        "id": check_id,
        "description": description,
        "path": path,
        "expected": expected,
        "actual": actual,
        "status": status,
        "message": message,
    }


def main() -> None:
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    # 1) accepted.json exists and contains 90 accepted
    accepted_path = Path("data/processed_official_module1/handoff/accepted.json")
    ex, rd, err = check_exists_and_readable(accepted_path)
    if not ex:
        checks.append(
            build_check(
                check_id="check_01",
                description="accepted.json exists and has 90 accepted",
                path=str(accepted_path),
                expected=90,
                actual=None,
                status="failed",
                message="File missing",
            )
        )
    elif not rd:
        checks.append(
            build_check(
                check_id="check_01",
                description="accepted.json exists and has 90 accepted",
                path=str(accepted_path),
                expected=90,
                actual=None,
                status="failed",
                message=err or "File unreadable",
            )
        )
    else:
        accepted_payload = read_json(accepted_path)
        accepted_count = len(accepted_payload) if isinstance(accepted_payload, list) else None
        checks.append(
            build_check(
                check_id="check_01",
                description="accepted.json exists and has 90 accepted",
                path=str(accepted_path),
                expected=90,
                actual=accepted_count,
                status="passed" if accepted_count == 90 else "failed",
                message="" if accepted_count == 90 else "accepted count differs from expected",
            )
        )

    # 2) grounded module2 folder exists and contains 90 JSON profiles
    grounded_dir = Path("data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles")
    if not grounded_dir.exists() or not grounded_dir.is_dir():
        checks.append(
            build_check(
                check_id="check_02",
                description="Grounded Module 2 profiles folder exists and contains 90 JSON files",
                path=str(grounded_dir),
                expected=90,
                actual=None,
                status="failed",
                message="Directory missing",
            )
        )
    else:
        profile_files = list(grounded_dir.glob("*.json"))
        checks.append(
            build_check(
                check_id="check_02",
                description="Grounded Module 2 profiles folder exists and contains 90 JSON files",
                path=str(grounded_dir),
                expected=90,
                actual=len(profile_files),
                status="passed" if len(profile_files) == 90 else "failed",
                message="" if len(profile_files) == 90 else "Profile JSON count differs from expected",
            )
        )

    # 3) faiss index exists
    faiss_index_path = Path("data/indexes/faiss/cv_index.faiss")
    ex, rd, err = check_exists_and_readable(faiss_index_path, binary=True)
    checks.append(
        build_check(
            check_id="check_03",
            description="FAISS index file exists and is readable",
            path=str(faiss_index_path),
            expected=True,
            actual=ex and rd,
            status="passed" if (ex and rd) else "failed",
            message="" if (ex and rd) else (err or "Missing/unreadable FAISS index"),
        )
    )

    # 4) id_map exists
    id_map_path = Path("data/indexes/faiss/id_map.pkl")
    ex, rd, err = check_exists_and_readable(id_map_path, binary=True)
    checks.append(
        build_check(
            check_id="check_04",
            description="FAISS id_map file exists and is readable",
            path=str(id_map_path),
            expected=True,
            actual=ex and rd,
            status="passed" if (ex and rd) else "failed",
            message="" if (ex and rd) else (err or "Missing/unreadable id_map"),
        )
    )

    # 5) index_report exists and indicates 90 profiles indexed if key available
    index_report_path = Path("data/indexes/faiss/index_report.json")
    ex, rd, err = check_exists_and_readable(index_report_path)
    if not ex or not rd:
        checks.append(
            build_check(
                check_id="check_05",
                description="FAISS index_report exists and profiles_indexed is 90 (if key available)",
                path=str(index_report_path),
                expected=90,
                actual=None,
                status="failed",
                message=err or "Missing/unreadable index_report",
            )
        )
    else:
        report = read_json(index_report_path)
        profiles_indexed = report.get("profiles_indexed") if isinstance(report, dict) else None
        if profiles_indexed is None:
            warnings.append("index_report.json does not contain 'profiles_indexed'; existence/readability verified only.")
            checks.append(
                build_check(
                    check_id="check_05",
                    description="FAISS index_report exists and profiles_indexed is 90 (if key available)",
                    path=str(index_report_path),
                    expected=90,
                    actual=None,
                    status="passed",
                    message="profiles_indexed key unavailable; check passed per conditional rule.",
                )
            )
        else:
            checks.append(
                build_check(
                    check_id="check_05",
                    description="FAISS index_report exists and profiles_indexed is 90 (if key available)",
                    path=str(index_report_path),
                    expected=90,
                    actual=profiles_indexed,
                    status="passed" if profiles_indexed == 90 else "failed",
                    message="" if profiles_indexed == 90 else "profiles_indexed differs from expected",
                )
            )

    # 6) job profile exists
    job_profile_path = Path("data/job_profiles/backend_python_fastapi_mongodb.json")
    ex, rd, err = check_exists_and_readable(job_profile_path)
    checks.append(
        build_check(
            check_id="check_06",
            description="Backend job profile exists and is readable",
            path=str(job_profile_path),
            expected=True,
            actual=ex and rd,
            status="passed" if (ex and rd) else "failed",
            message="" if (ex and rd) else (err or "Missing/unreadable job profile"),
        )
    )

    # 7) matching v3 normalized report exists and contains 10 recommendations
    matching_report_path = Path("docs/reports/matching/v3/matching_report_v3_normalized.json")
    ex, rd, err = check_exists_and_readable(matching_report_path)
    if not ex or not rd:
        checks.append(
            build_check(
                check_id="check_07",
                description="Matching report v3 normalized exists and contains 10 recommendations",
                path=str(matching_report_path),
                expected=10,
                actual=None,
                status="failed",
                message=err or "Missing/unreadable matching report",
            )
        )
    else:
        report = read_json(matching_report_path)
        recommendations_count = None
        if isinstance(report, dict):
            results = report.get("results")
            if isinstance(results, list) and results and isinstance(results[0], dict):
                recs = results[0].get("recommendations")
                if isinstance(recs, list):
                    recommendations_count = len(recs)
        checks.append(
            build_check(
                check_id="check_07",
                description="Matching report v3 normalized exists and contains 10 recommendations",
                path=str(matching_report_path),
                expected=10,
                actual=recommendations_count,
                status="passed" if recommendations_count == 10 else "failed",
                message="" if recommendations_count == 10 else "Recommendations count differs from expected",
            )
        )

    # 8) decision cards v3 normalized exists and contains 10 cards
    decision_cards_path = Path("docs/reports/matching/v3/decision_cards_v3_normalized.json")
    ex, rd, err = check_exists_and_readable(decision_cards_path)
    if not ex or not rd:
        checks.append(
            build_check(
                check_id="check_08",
                description="Decision cards v3 normalized exists and contains 10 cards",
                path=str(decision_cards_path),
                expected=10,
                actual=None,
                status="failed",
                message=err or "Missing/unreadable decision cards report",
            )
        )
    else:
        cards_report = read_json(decision_cards_path)
        cards_count = None
        if isinstance(cards_report, dict):
            if isinstance(cards_report.get("cards_count"), int):
                cards_count = cards_report.get("cards_count")
            else:
                cards = cards_report.get("decision_cards")
                if isinstance(cards, list):
                    cards_count = len(cards)
        checks.append(
            build_check(
                check_id="check_08",
                description="Decision cards v3 normalized exists and contains 10 cards",
                path=str(decision_cards_path),
                expected=10,
                actual=cards_count,
                status="passed" if cards_count == 10 else "failed",
                message="" if cards_count == 10 else "Decision cards count differs from expected",
            )
        )

    # 9) CrossEncoder ablation summary exists
    ablation_path = Path("docs/reports/retrieval/faiss_cross_encoder_ablation_summary.json")
    ex, rd, err = check_exists_and_readable(ablation_path)
    checks.append(
        build_check(
            check_id="check_09",
            description="FAISS CrossEncoder ablation summary exists and is readable",
            path=str(ablation_path),
            expected=True,
            actual=ex and rd,
            status="passed" if (ex and rd) else "failed",
            message="" if (ex and rd) else (err or "Missing/unreadable ablation summary"),
        )
    )

    passed_count = sum(1 for check in checks if check["status"] == "passed")
    failed_count = sum(1 for check in checks if check["status"] == "failed")

    # Baseline check stays read-only; no MongoDB write action is performed.
    if failed_count:
        next_actions = [
            "Restore or regenerate missing artifacts listed in failed checks.",
            "If FAISS files are missing, recover cv_index.faiss and id_map.pkl from official artifacts backup.",
            "Re-run readiness check script after artifact recovery.",
        ]
    else:
        next_actions = [
            "Proceed with demo dry-run using existing official baseline artifacts.",
            "Keep this report as evidence of technical readiness.",
        ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_name": BASELINE_NAME,
        "checks": checks,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "warnings": warnings,
        "demo_ready": failed_count == 0,
        "next_actions": next_actions,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH),
                "passed_count": passed_count,
                "failed_count": failed_count,
                "demo_ready": failed_count == 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
