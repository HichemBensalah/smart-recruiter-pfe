from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_reports(
    *,
    output_dir: Path,
    accepted_count: int,
    processed_items: list[dict[str, Any]],
    failed_items: list[dict[str, Any]],
    provider: str,
    fallback_provider: str,
    compare_with: Path | None,
    skipped_existing: int = 0,
) -> dict[str, Any]:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    profile_kind_counts = Counter(item.get("profile_kind") for item in processed_items)
    risk_counts = Counter(item.get("hallucination_risk") for item in processed_items)
    provider_counts = Counter(item.get("provider_used") for item in processed_items)
    template_counts = Counter()
    unsupported_counts = Counter()
    total_nullified = 0
    reliability_scores: list[float] = []
    full_name_rejected_count = 0

    for item in processed_items:
        reliability_scores.append(float(item.get("reliability_score") or 0.0))
        total_nullified += len(item.get("fields_nullified") or [])
        template_counts.update(item.get("detected_templates") or [])
        unsupported_counts.update(_field_roots(item.get("fields_unsupported") or []))
        quality_flags = item.get("quality_flags") or []
        normalization_quality_flags = item.get("normalization_quality_flags") or []
        if "missing_full_name" in quality_flags or "missing_full_name" in normalization_quality_flags:
            full_name_rejected_count += 1

    avg_reliability = round(sum(reliability_scores) / len(reliability_scores), 4) if reliability_scores else 0.0
    comparison = build_old_module2_comparison(processed_items, compare_with)

    run_report = {
        "generated_at": utc_now(),
        "accepted_count": accepted_count,
        "processed": len(processed_items),
        "success": len([item for item in processed_items if item.get("status") == "success"]),
        "failed": len(failed_items),
        "skipped_existing": skipped_existing,
        "provider": provider,
        "fallback_provider": fallback_provider,
        "provider_used_counts": dict(provider_counts),
        "items": processed_items,
        "failures": failed_items,
    }
    import_readiness = "not_ready_provider_integration_needed" if failed_items or not processed_items else (
        "not_ready_without_review" if risk_counts.get("high", 0) else "candidate_for_review"
    )
    quality_report = {
        "generated_at": utc_now(),
        "total_accepted_cvs_found": accepted_count,
        "total_processed": len(processed_items),
        "complete_profile": profile_kind_counts.get("complete_profile", 0),
        "partial_profile": profile_kind_counts.get("partial_profile", 0),
        "minimal_profile": profile_kind_counts.get("minimal_profile", 0),
        "unreadable": profile_kind_counts.get("unreadable", 0),
        "failed": len(failed_items),
        "skipped_existing": skipped_existing,
        "average_reliability_score": avg_reliability,
        "hallucination_risk_distribution": dict(risk_counts),
        "templates_detected": dict(template_counts),
        "full_name_rejected_count": full_name_rejected_count,
        "fields_nullified_count": total_nullified,
        "unsupported_field_types": dict(unsupported_counts),
        "mongodb_import_readiness": import_readiness,
        "faiss_rebuild_readiness": import_readiness,
    }
    reduction_report = {
        "generated_at": utc_now(),
        "comparison_with_old_module2": comparison,
        "statement": (
            "The new V2 grounded pipeline reduces hallucination by nullifying unsupported fields "
            "and producing partial profiles when evidence is insufficient."
        ),
    }
    provider_report = {
        "generated_at": utc_now(),
        "provider_requested": provider,
        "fallback_provider_requested": fallback_provider,
        "provider_used_counts": dict(provider_counts),
        "failed_provider_errors": [item for item in failed_items if item.get("failure_type") == "provider_error"],
    }

    write_json(reports_dir / "run_report.json", run_report)
    write_json(reports_dir / "grounded_quality_report.json", quality_report)
    write_json(reports_dir / "hallucination_reduction_report.json", reduction_report)
    write_json(reports_dir / "provider_comparison_report.json", provider_report)
    write_field_confidence_summary(reports_dir / "field_confidence_summary.csv", processed_items)
    write_failed_or_partial_profiles(reports_dir / "failed_or_partial_profiles.csv", processed_items, failed_items)
    return {
        "run_report": run_report,
        "grounded_quality_report": quality_report,
        "hallucination_reduction_report": reduction_report,
        "provider_comparison_report": provider_report,
    }


def _field_roots(fields: list[str]) -> list[str]:
    roots = []
    for field in fields:
        root = field.split("[", 1)[0].split(".", 1)[0]
        if root:
            roots.append(root)
    return roots


def write_field_confidence_summary(path: Path, items: list[dict[str, Any]]) -> None:
    rows = []
    for item in items:
        for field in item.get("fields_supported") or []:
            rows.append({**_base_row(item), "field_name": field, "field_status": "supported"})
        for field in item.get("fields_unsupported") or []:
            rows.append({**_base_row(item), "field_name": field, "field_status": "unsupported"})
        for field in item.get("fields_nullified") or []:
            rows.append({**_base_row(item), "field_name": field, "field_status": "nullified"})
    _write_csv(path, rows, [
        "profile_file",
        "source_path",
        "artifact_path",
        "profile_kind",
        "provider_used",
        "reliability_score",
        "hallucination_risk",
        "field_name",
        "field_status",
    ])


def write_failed_or_partial_profiles(
    path: Path,
    processed_items: list[dict[str, Any]],
    failed_items: list[dict[str, Any]],
) -> None:
    rows = []
    for item in processed_items:
        if item.get("profile_kind") != "complete_profile":
            rows.append({
                **_base_row(item),
                "status": item.get("status"),
                "reason": ",".join(item.get("quality_flags") or []),
            })
    for item in failed_items:
        rows.append({
            "profile_file": "",
            "source_path": item.get("source_path"),
            "artifact_path": item.get("artifact_path"),
            "profile_kind": "failed",
            "provider_used": item.get("provider_used"),
            "reliability_score": "",
            "hallucination_risk": "",
            "status": "failed",
            "reason": item.get("error"),
        })
    _write_csv(path, rows, [
        "profile_file",
        "source_path",
        "artifact_path",
        "profile_kind",
        "provider_used",
        "reliability_score",
        "hallucination_risk",
        "status",
        "reason",
    ])


def build_old_module2_comparison(items: list[dict[str, Any]], compare_with: Path | None) -> dict[str, Any]:
    if compare_with is None or not compare_with.exists():
        return {"available": False, "reason": "old_module2_folder_missing_or_not_requested"}
    old_files = {path.stem: path for path in compare_with.rglob("*.json") if path.name not in {"run_report.json", "final_manual_summary.json"}}
    matched = 0
    examples = []
    for item in items:
        key = Path(str(item.get("artifact_path") or "")).stem
        old_path = old_files.get(key)
        if not old_path:
            continue
        matched += 1
        if len(examples) < 10:
            examples.append({
                "source_path": item.get("source_path"),
                "old_profile_file": str(old_path),
                "new_profile_file": item.get("profile_file"),
                "new_profile_kind": item.get("profile_kind"),
                "new_reliability_score": item.get("reliability_score"),
                "fields_nullified": item.get("fields_nullified", [])[:8],
            })
    return {
        "available": True,
        "old_folder": str(compare_with),
        "matched_profiles": matched,
        "new_profiles_compared": len(items),
        "examples_before_after": examples,
    }


def _base_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_file": item.get("profile_file"),
        "source_path": item.get("source_path"),
        "artifact_path": item.get("artifact_path"),
        "profile_kind": item.get("profile_kind"),
        "provider_used": item.get("provider_used"),
        "reliability_score": item.get("reliability_score"),
        "hallucination_risk": item.get("hallucination_risk"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
