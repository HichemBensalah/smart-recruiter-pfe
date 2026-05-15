from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARCHIVE_ROOT = DATA_DIR / "archive_old_runs"
REPORT_MD = DATA_DIR / "cleanup_plan_report.md"
REPORT_JSON = DATA_DIR / "cleanup_plan_report.json"


@dataclass
class PlanItem:
    path: str
    type: str
    category: str
    reason: str
    recommended_action: str
    risk_level: str
    size_bytes: int = 0
    archive_destination: str | None = None
    exists: bool = True


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("/", "\\")
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def item_type(path: Path) -> str:
    if path.exists() and path.is_dir():
        return "directory"
    return "file"


def add_item(
    items: list[PlanItem],
    path: Path,
    *,
    category: str,
    reason: str,
    recommended_action: str,
    risk_level: str,
    archive_destination: Path | None = None,
    exists: bool | None = None,
) -> None:
    actual_exists = path.exists() if exists is None else exists
    items.append(
        PlanItem(
            path=relpath(path),
            type=item_type(path),
            category=category,
            reason=reason,
            recommended_action=recommended_action,
            risk_level=risk_level,
            size_bytes=path_size(path) if actual_exists else 0,
            archive_destination=relpath(archive_destination) if archive_destination else None,
            exists=actual_exists,
        )
    )


def discover_project_pycache() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("__pycache__")
        if path.is_dir()
        and ".git" not in path.parts
        and ".venv" not in path.parts
        and ("src" in path.parts or "scripts" in path.parts)
    )


def discover_venv_pycache() -> list[Path]:
    return sorted(path for path in ROOT.rglob("__pycache__") if path.is_dir() and ".venv" in path.parts)


def discover_temp_files() -> list[Path]:
    patterns = ("*.tmp", "*.bak", "*.old", "*.log")
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(
            path
            for path in ROOT.rglob(pattern)
            if ".git" not in path.parts and ".venv" not in path.parts
        )
    unique = sorted(set(matches))
    return [path for path in unique if path.is_file()]


def discover_empty_dirs() -> list[Path]:
    empty_dirs: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_dir():
            continue
        if ".git" in path.parts or ".venv" in path.parts:
            continue
        try:
            next(path.iterdir())
        except StopIteration:
            empty_dirs.append(path)
    return sorted(empty_dirs)


def archive_target(subdir: str, source: Path) -> Path:
    return ARCHIVE_ROOT / subdir / source.name


def build_plan() -> tuple[list[PlanItem], list[str]]:
    items: list[PlanItem] = []
    notes: list[str] = []

    protected_keep = [
        DATA_DIR / "profile_builder_module2_v2_grounded_all",
        DATA_DIR / "profile_builder_module2_v2_grounded_all" / "profiles" / "grounded_profiles",
        DATA_DIR / "profile_builder_module2_v2_grounded_all" / "profiles" / "legacy_projection",
        DATA_DIR / "profile_builder_module2_v2_grounded_all" / "reports",
        DATA_DIR / "processed_official_module1",
        DATA_DIR / "raw_cv",
        DATA_DIR / "job_descriptions",
        DATA_DIR / "job_profiles",
        DATA_DIR / "indexes" / "faiss",
        DATA_DIR / "mongodb_import_report_v2_grounded_execute.json",
        DATA_DIR / "mongodb_import_report_v2_grounded_dry_run.json",
        DATA_DIR / "matching_single_job_report_grounded_v3.json",
        DATA_DIR / "matching_ranking_comparison_grounded_v3.json",
        DATA_DIR / "matching_ranking_comparison_grounded_v3.md",
        DATA_DIR / "final_current_state_recap.md",
        DATA_DIR / "profile_builder_official_module2_rerun_ollama_fixed",
        DATA_DIR / "module2_hallucination_audit_report.json",
        DATA_DIR / "module2_hallucination_audit_summary.md",
        DATA_DIR / "module2_hallucination_audit_table.csv",
        ROOT / "src" / "core" / "structuring" / "profile_builder_grounded.py",
        ROOT / "src" / "core" / "structuring" / "markdown_normalizer.py",
        ROOT / "src" / "core" / "structuring" / "grounded_prompt.py",
        ROOT / "src" / "core" / "structuring" / "grounding_validator.py",
        ROOT / "src" / "core" / "structuring" / "grounded_reporting.py",
        ROOT / "src" / "core" / "storage" / "import_profiles_to_mongodb.py",
        ROOT / "src" / "core" / "matching",
        ROOT / "scripts" / "test_grounded_module2_v2.py",
        DATA_DIR / "benchmarks",
        ARCHIVE_ROOT,
    ]
    for path in protected_keep:
        add_item(
            items,
            path,
            category="keep",
            reason="Protected current artifact or historical baseline required for the current demo and/or comparisons.",
            recommended_action="keep",
            risk_level="high",
        )

    missing_test_run = DATA_DIR / "profile_builder_module2_v2_grounded_test1"
    if missing_test_run.exists():
        add_item(
            items,
            missing_test_run,
            category="temporary_test_output",
            reason="Single-run Module 2 grounded test output; not part of the current protected production artifacts.",
            recommended_action="archive",
            risk_level="low",
            archive_destination=archive_target("module2_test_runs", missing_test_run),
        )
    else:
        notes.append("Targeted folder not found: data\\profile_builder_module2_v2_grounded_test1")

    grounded_logs = DATA_DIR / "profile_builder_module2_v2_grounded_all" / "logs"
    if grounded_logs.exists():
        add_item(
            items,
            grounded_logs,
            category="safe_to_archive",
            reason="Current grounded run debug logs contain provider errors and nullification traces; useful for debugging, not required by the pipeline outputs.",
            recommended_action="archive",
            risk_level="medium",
            archive_destination=archive_target("logs", grounded_logs),
        )

    archive_candidates = [
        DATA_DIR / "matching_single_job_report.json",
        DATA_DIR / "matching_single_job_report_v2.json",
        DATA_DIR / "matching_single_job_report_after_fix.json",
        DATA_DIR / "matching_single_job_report_grounded_v2.json",
        DATA_DIR / "matching_ranking_comparison_grounded_v2.json",
        DATA_DIR / "matching_grounded_v2_audit.json",
        DATA_DIR / "matching_grounded_v2_audit.md",
        DATA_DIR / "matching_test_report.json",
        DATA_DIR / "mongodb_import_report.json",
        DATA_DIR / "mongodb_import_report_v2_dry_run.json",
        DATA_DIR / "mongodb_dedup_strategy_report.json",
    ]
    for path in archive_candidates:
        if not path.exists():
            continue
        reason = "Historical report superseded by grounded V3 outputs, but still useful as a comparison trace."
        risk = "low"
        category = "obsolete_report"
        subdir = "matching_old_reports"
        if "mongodb" in path.name:
            subdir = "mongodb_old_reports"
            reason = "Historical MongoDB import report from pre-grounded or intermediate import flow; keep for traceability, archive for readability."
            risk = "medium"
        elif "matching_grounded_v2_audit" in path.name:
            category = "duplicate_report"
            reason = "Intermediate grounded V2 audit report, useful historically but not part of the current final demo path."
            subdir = "module2_debug_reports"
        add_item(
            items,
            path,
            category=category,
            reason=reason,
            recommended_action="archive",
            risk_level=risk,
            archive_destination=archive_target(subdir, path),
        )

    for path in discover_project_pycache():
        add_item(
            items,
            path,
            category="cache",
            reason="Interpreter cache generated from project source files; safe to regenerate.",
            recommended_action="delete",
            risk_level="low",
        )

    venv_pycache = discover_venv_pycache()
    if venv_pycache:
        total_size = sum(path_size(path) for path in venv_pycache)
        items.append(
            PlanItem(
                path=".venv\\**\\__pycache__",
                type="directory",
                category="cache",
                reason="Virtualenv bytecode caches are safe to regenerate, but they are numerous and should be deleted only deliberately.",
                recommended_action="delete",
                risk_level="low",
                size_bytes=total_size,
                exists=True,
            )
        )

    for path in discover_temp_files():
        add_item(
            items,
            path,
            category="safe_to_delete",
            reason="Temporary or backup artifact with no protected role detected.",
            recommended_action="delete",
            risk_level="low",
        )

    for path in discover_empty_dirs():
        if path == DATA_DIR / "indexes" / "faiss" / "hf_cache" / ".locks" / "models--sentence-transformers--all-MiniLM-L6-v2":
            add_item(
                items,
                path,
                category="keep",
                reason="Empty lock directory under the protected FAISS/HuggingFace cache area; not worth touching during a safe cleanup pass.",
                recommended_action="keep",
                risk_level="medium",
            )
            continue
        add_item(
            items,
            path,
            category="safe_to_delete",
            reason="Empty directory not protected by the current pipeline rules.",
            recommended_action="delete",
            risk_level="low",
        )

    items.sort(key=lambda item: (item.recommended_action, item.path.lower()))
    return items, notes


def summarize(items: Iterable[PlanItem]) -> dict[str, object]:
    items = list(items)
    keep = [item for item in items if item.recommended_action == "keep"]
    archive = [item for item in items if item.recommended_action == "archive"]
    delete = [item for item in items if item.recommended_action == "delete"]
    return {
        "generated_at_utc": utc_now(),
        "counts": {
            "keep": len(keep),
            "archive": len(archive),
            "delete": len(delete),
            "total": len(items),
        },
        "size_mb": {
            "keep": round(sum(item.size_bytes for item in keep) / 1024 / 1024, 2),
            "archive": round(sum(item.size_bytes for item in archive) / 1024 / 1024, 2),
            "delete": round(sum(item.size_bytes for item in delete) / 1024 / 1024, 2),
        },
        "delete_risk_counts": {
            "low": sum(1 for item in delete if item.risk_level == "low"),
            "medium": sum(1 for item in delete if item.risk_level == "medium"),
            "high": sum(1 for item in delete if item.risk_level == "high"),
        },
    }


def write_reports(items: list[PlanItem], notes: list[str]) -> dict[str, object]:
    summary = summarize(items)
    payload = {
        "summary": summary,
        "notes": notes,
        "items": [asdict(item) for item in items],
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Cleanup Plan Report",
        "",
        f"Generated at: {summary['generated_at_utc']}",
        "",
        "## Summary",
        "",
        f"- Total classified items: {summary['counts']['total']}",
        f"- Keep: {summary['counts']['keep']}",
        f"- Archive: {summary['counts']['archive']}",
        f"- Delete: {summary['counts']['delete']}",
        f"- Approx size in archive bucket: {summary['size_mb']['archive']} MB",
        f"- Approx size in delete bucket: {summary['size_mb']['delete']} MB",
        "",
        "## Notes",
        "",
    ]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- No additional notes.")

    lines.extend(
        [
            "",
            "## Classified Items",
            "",
            "| path | type | category | reason | recommended_action | risk_level | size_mb | archive_destination |",
            "|---|---|---|---|---|---|---:|---|",
        ]
    )
    for item in items:
        size_mb = round(item.size_bytes / 1024 / 1024, 3)
        archive_dest = item.archive_destination or ""
        reason = item.reason.replace("|", "\\|")
        lines.append(
            f"| `{item.path}` | {item.type} | {item.category} | {reason} | {item.recommended_action} | {item.risk_level} | {size_mb} | `{archive_dest}` |"
        )

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    return payload


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def archive_item(source: Path, destination: Path) -> None:
    ensure_parent(destination)
    if destination.exists():
        raise FileExistsError(f"Archive destination already exists: {destination}")
    shutil.move(str(source), str(destination))


def delete_item(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def execute_plan(items: list[PlanItem]) -> dict[str, int]:
    archived = 0
    deleted = 0
    skipped = 0
    for item in items:
        if item.recommended_action == "keep":
            skipped += 1
            continue
        if item.path == ".venv\\**\\__pycache__":
            for path in discover_venv_pycache():
                delete_item(path)
                deleted += 1
            continue
        source = ROOT / Path(item.path)
        if not source.exists():
            skipped += 1
            continue
        if item.recommended_action == "archive":
            if not item.archive_destination:
                raise RuntimeError(f"Missing archive destination for {item.path}")
            archive_item(source, ROOT / Path(item.archive_destination))
            archived += 1
        elif item.recommended_action == "delete":
            delete_item(source)
            deleted += 1
    return {"archived": archived, "deleted": deleted, "skipped": skipped}


def print_dry_run(payload: dict[str, object]) -> None:
    summary = payload["summary"]
    counts = summary["counts"]
    sizes = summary["size_mb"]
    items = payload["items"]
    print("CLEANUP DRY-RUN")
    print(f"Keep: {counts['keep']} | Archive: {counts['archive']} | Delete: {counts['delete']}")
    print(f"Approx archive bucket: {sizes['archive']} MB")
    print(f"Approx delete bucket: {sizes['delete']} MB")
    print("")

    for action in ("keep", "archive", "delete"):
        print(f"[{action.upper()}]")
        bucket = [item for item in items if item["recommended_action"] == action]
        for item in bucket[:20]:
            extra = f" -> {item['archive_destination']}" if item.get("archive_destination") else ""
            print(f"- {item['path']} ({item['category']}, risk={item['risk_level']}, size_mb={item['size_bytes']/1024/1024:.3f}){extra}")
        if len(bucket) > 20:
            print(f"- ... {len(bucket) - 20} more")
        print("")

    if payload["notes"]:
        print("[NOTES]")
        for note in payload["notes"]:
            print(f"- {note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan and optionally execute a controlled cleanup of project artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Generate reports and print the proposed actions without changing files.")
    parser.add_argument("--execute", action="store_true", help="Apply the cleanup plan by archiving or deleting planned items.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run == args.execute:
        raise SystemExit("Choose exactly one mode: --dry-run or --execute")

    items, notes = build_plan()
    payload = write_reports(items, notes)

    if args.dry_run:
        print_dry_run(payload)
        return

    result = execute_plan(items)
    print(json.dumps({"mode": "execute", **result}, indent=2))


if __name__ == "__main__":
    main()
