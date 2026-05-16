from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.core.ranking.ml_reranker import (
    extract_feature_matrix,
    load_feature_names,
    load_jsonl,
    load_model,
    predict_positive_scores,
)


METHODOLOGICAL_NOTE = (
    "XGBoost est utilisé ici comme moteur principal de ranking ML à partir des features Matching V3. "
    "Le modèle reste entraîné sur pseudo-labels métier contrôlés, pas sur labels recruteur. "
    "Matching V3 est conservé comme baseline de comparaison."
)


def build_primary_ranked_candidates(
    rows: list[dict[str, Any]],
    scores: list[float],
    feature_names: list[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row, score in zip(rows, scores, strict=True):
        features = row.get("features") or {}
        xgboost_score = float(np.clip(score, 0.0, 1.0))
        candidates.append(
            {
                "candidate_id": row.get("candidate_id"),
                "profile_id": row.get("profile_id"),
                "baseline_rank_v3": int(row.get("rank", 0)),
                "baseline_score_v3": float(features.get("final_score_v3", 0.0)),
                "xgboost_score": xgboost_score,
                "final_rank_ml": None,
                "rank_shift": None,
                "features": {name: float(features.get(name, 0.0)) for name in feature_names},
            }
        )

    ordered = sorted(candidates, key=lambda item: item["xgboost_score"], reverse=True)
    for index, candidate in enumerate(ordered, start=1):
        candidate["final_rank_ml"] = index
        candidate["rank_shift"] = int(candidate["baseline_rank_v3"]) - index
    return ordered


def build_primary_ranking_report(
    *,
    rows: list[dict[str, Any]],
    model: Any,
    feature_names: list[str],
    model_name: str = "xgboost",
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot rank an empty feature file.")
    matrix = extract_feature_matrix(rows, feature_names)
    scores = predict_positive_scores(model, matrix).tolist()
    job_ids = sorted({str(row.get("job_id")) for row in rows})
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": job_ids[0] if len(job_ids) == 1 else job_ids,
        "ranking_mode": "xgboost_primary_with_matching_v3_features",
        "model_name": model_name,
        "methodological_note": METHODOLOGICAL_NOTE,
        "feature_names": feature_names,
        "candidate_count": len(rows),
        "candidates": build_primary_ranked_candidates(rows, scores, feature_names),
    }


def run_primary_ranking(
    *,
    features_path: str | Path,
    model_path: str | Path,
    feature_names_path: str | Path,
) -> dict[str, Any]:
    rows = load_jsonl(features_path)
    model = load_model(model_path)
    feature_names = load_feature_names(feature_names_path)
    return build_primary_ranking_report(rows=rows, model=model, feature_names=feature_names)


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown_report(report: dict[str, Any]) -> str:
    candidates = list(report["candidates"])
    top_ml = sorted(candidates, key=lambda item: int(item["final_rank_ml"]))[:10]
    top_v3 = sorted(candidates, key=lambda item: int(item["baseline_rank_v3"]))[:10]
    promoted = sorted(candidates, key=lambda item: int(item["rank_shift"]), reverse=True)[:10]
    demoted = sorted(candidates, key=lambda item: int(item["rank_shift"]))[:10]

    lines = [
        "# XGBoost primary ranking",
        "",
        "## Objectif",
        "",
        "Produire un ranking principal basé sur XGBoost à partir des features métier issues de Matching V3, tout en conservant Matching V3 comme baseline de comparaison.",
        "",
        "## Architecture",
        "",
        "- FAISS reste utilisé pour le retrieval initial.",
        "- Matching V3 reste présent et sert de feature engine métier.",
        "- XGBoost calcule `xgboost_score` et définit `final_rank_ml`.",
        "- Matching V3 conserve `baseline_score_v3` et `baseline_rank_v3`.",
        "- SHAP peut expliquer le score XGBoost dans les rapports d'explicabilité existants.",
        "",
        "## Job",
        "",
        f"- `job_id`: `{report['job_id']}`",
        f"- modèle: `{report['model_name']}`",
        f"- mode: `{report['ranking_mode']}`",
        f"- candidats rankés: `{report['candidate_count']}`",
        "",
        "## Top 10 XGBoost ranking principal",
        "",
        "| final_rank_ml | candidate_id | xgboost_score | baseline_rank_v3 | baseline_score_v3 | rank_shift |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(_candidate_table_lines(top_ml, mode="ml"))
    lines.extend(
        [
            "",
            "## Top 10 Matching V3 baseline",
            "",
            "| baseline_rank_v3 | candidate_id | baseline_score_v3 | final_rank_ml | xgboost_score | rank_shift |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(_candidate_table_lines(top_v3, mode="v3"))
    lines.extend(
        [
            "",
            "## Candidats remontés par XGBoost",
            "",
            "| candidate_id | baseline_rank_v3 | final_rank_ml | rank_shift | xgboost_score |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(_movement_lines([candidate for candidate in promoted if int(candidate["rank_shift"]) > 0]))
    lines.extend(
        [
            "",
            "## Candidats descendus par XGBoost",
            "",
            "| candidate_id | baseline_rank_v3 | final_rank_ml | rank_shift | xgboost_score |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(_movement_lines([candidate for candidate in demoted if int(candidate["rank_shift"]) < 0]))
    lines.extend(
        [
            "",
            "## Limites méthodologiques",
            "",
            f"- {report['methodological_note']}",
            "- `xgboost_score` n'est pas un score recruteur final.",
            "- Le modèle n'a pas été réentraîné dans cette étape.",
            "- Les datasets, FAISS, MongoDB et Decision Cards officielles ne sont pas modifiés.",
            "- La validation humaine reste nécessaire avant toute décision produit.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown_report(report), encoding="utf-8")


def _candidate_table_lines(candidates: list[dict[str, Any]], *, mode: str) -> list[str]:
    lines: list[str] = []
    for candidate in candidates:
        if mode == "ml":
            lines.append(
                f"| {candidate['final_rank_ml']} | `{candidate['candidate_id']}` | "
                f"{candidate['xgboost_score']:.4f} | {candidate['baseline_rank_v3']} | "
                f"{candidate['baseline_score_v3']:.4f} | {candidate['rank_shift']} |"
            )
        else:
            lines.append(
                f"| {candidate['baseline_rank_v3']} | `{candidate['candidate_id']}` | "
                f"{candidate['baseline_score_v3']:.4f} | {candidate['final_rank_ml']} | "
                f"{candidate['xgboost_score']:.4f} | {candidate['rank_shift']} |"
            )
    return lines


def _movement_lines(candidates: list[dict[str, Any]]) -> list[str]:
    if not candidates:
        return ["| Aucun | NA | NA | NA | NA |"]
    return [
        f"| `{candidate['candidate_id']}` | {candidate['baseline_rank_v3']} | "
        f"{candidate['final_rank_ml']} | {candidate['rank_shift']} | {candidate['xgboost_score']:.4f} |"
        for candidate in candidates
    ]

