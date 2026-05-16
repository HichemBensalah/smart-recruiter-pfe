from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_demo_executive_summary import build_executive_summary, write_json as write_json_file
from scripts.build_demo_executive_summary import write_markdown as write_executive_markdown
from scripts.build_demo_summary import build_demo_summary, write_markdown as write_demo_markdown
from src.core.graph.decision_cards_transferability_enricher import (
    build_transferability_cards_report,
    write_json_report,
    write_markdown_report,
)


TRANSFERABILITY_JSON = Path("docs/reports/decision_cards/decision_cards_with_transferability.json")
TRANSFERABILITY_MD = Path("docs/reports/decision_cards/decision_cards_with_transferability.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Smart Recruiter demo report generation end-to-end.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--rf-model", type=Path, required=True)
    parser.add_argument("--xgb-ranking", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, required=True)
    parser.add_argument("--cards-ml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    warnings: list[str] = []
    checked_inputs = check_inputs(
        {
            "features": args.features,
            "job": args.job,
            "profiles_dir": args.profiles_dir,
            "graph": args.graph,
            "rf_model": args.rf_model,
            "xgb_ranking": args.xgb_ranking,
            "feature_names": args.feature_names,
            "cards_ml": args.cards_ml,
        }
    )

    transferability_report = build_transferability_cards_report(
        cards_path=args.cards_ml,
        profiles_dir=args.profiles_dir,
        job_path=args.job,
        graph_path=args.graph,
    )
    write_json_report(transferability_report, TRANSFERABILITY_JSON)
    write_markdown_report(transferability_report, TRANSFERABILITY_MD)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_summary_json = output_dir / "demo_summary_top10.json"
    demo_summary_md = output_dir / "demo_summary_top10.md"
    executive_json = output_dir / "demo_executive_summary.json"
    executive_md = output_dir / "demo_executive_summary.md"
    manifest_path = output_dir / "demo_run_manifest.json"
    run_summary_md = output_dir / "demo_run_summary.md"

    demo_summary = build_demo_summary(transferability_report, top_k=args.top_k, source_path=TRANSFERABILITY_JSON)
    write_json_file(demo_summary_json, demo_summary)
    write_demo_markdown(demo_summary_md, demo_summary)

    executive_summary = build_executive_summary(demo_summary, demo_summary_json)
    write_json_file(executive_json, executive_summary)
    write_executive_markdown(executive_md, executive_summary)

    generated_outputs = {
        "decision_cards_with_transferability_json": str(TRANSFERABILITY_JSON),
        "decision_cards_with_transferability_md": str(TRANSFERABILITY_MD),
        "demo_summary_top10_json": str(demo_summary_json),
        "demo_summary_top10_md": str(demo_summary_md),
        "demo_executive_summary_json": str(executive_json),
        "demo_executive_summary_md": str(executive_md),
        "demo_run_manifest_json": str(manifest_path),
        "demo_run_summary_md": str(run_summary_md),
    }

    manifest = {
        "status": "success",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_id": demo_summary.get("job_id"),
        "checked_inputs": checked_inputs,
        "generated_outputs": generated_outputs,
        "total_candidates": int(transferability_report.get("candidate_count", 0)),
        "top_k": args.top_k,
        "executive_top_recommended": [candidate["candidate_id"] for candidate in executive_summary["top_recommended"]],
        "executive_needs_review": [candidate["candidate_id"] for candidate in executive_summary["needs_review"]],
        "tests_hint": "pytest tests/test_demo_end_to_end.py -q",
        "warnings": warnings,
    }
    write_json(manifest_path, manifest)
    write_run_summary(run_summary_md, manifest, executive_summary)
    return manifest


def check_inputs(inputs: dict[str, Path]) -> dict[str, dict[str, Any]]:
    checked: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name, path in inputs.items():
        exists = path.exists()
        checked[name] = {
            "path": str(path),
            "exists": exists,
            "is_dir": path.is_dir() if exists else False,
            "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        }
        if not exists:
            missing.append(f"{name}: {path}")
    if missing:
        raise FileNotFoundError("Missing required demo inputs: " + "; ".join(missing))
    return checked


def write_run_summary(path: str | Path, manifest: dict[str, Any], executive_summary: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Demo run summary",
        "",
        "## Objectif de la démo",
        "",
        "Préparer en une commande les rapports principaux de démonstration Smart Recruiter sans réentraîner de modèle ni modifier les briques métier.",
        "",
        "## Artefacts vérifiés",
        "",
    ]
    for name, item in manifest["checked_inputs"].items():
        lines.append(f"- `{name}`: `{item['path']}`")
    lines.extend(["", "## Artefacts générés", ""])
    for name, path_value in manifest["generated_outputs"].items():
        lines.append(f"- `{name}`: `{path_value}`")
    lines.extend(["", "## Top recommended", ""])
    lines.extend(_candidate_lines(executive_summary["top_recommended"]))
    lines.extend(["", "## Needs review", ""])
    lines.extend(_candidate_lines(executive_summary["needs_review"]))
    lines.extend(
        [
            "",
            "## Rapports principaux",
            "",
            f"- Executive summary: `{manifest['generated_outputs']['demo_executive_summary_md']}`",
            f"- Summary top 10: `{manifest['generated_outputs']['demo_summary_top10_md']}`",
            "",
            "## Limites méthodologiques courtes",
            "",
            "- Matching V3 reste la baseline officielle.",
            "- Les modèles ML sont entraînés sur pseudo-labels métier contrôlés.",
            "- Potential Graph est déclaratif et ne remplace pas une décision recruteur.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def _candidate_lines(candidates: list[dict[str, Any]]) -> list[str]:
    if not candidates:
        return ["- Aucun candidat."]
    return [f"- `{candidate['candidate_id']}`: {candidate['executive_reason']}" for candidate in candidates]


def main() -> None:
    args = parse_args()
    try:
        manifest = run_demo(args)
    except Exception as exc:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        failed = {
            "status": "failed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "job_id": None,
            "checked_inputs": {},
            "generated_outputs": {},
            "total_candidates": 0,
            "top_k": args.top_k,
            "executive_top_recommended": [],
            "executive_needs_review": [],
            "tests_hint": "pytest tests/test_demo_end_to_end.py -q",
            "warnings": [f"{type(exc).__name__}: {exc}"],
        }
        write_json(output_dir / "demo_run_manifest.json", failed)
        raise
    print(f"Demo end-to-end status: {manifest['status']}")
    print(f"Manifest written: {args.output_dir / 'demo_run_manifest.json'}")
    print(f"Run summary written: {args.output_dir / 'demo_run_summary.md'}")


if __name__ == "__main__":
    main()

