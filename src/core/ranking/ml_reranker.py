from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np


METHODOLOGY_WARNING = (
    "Le score ML est expérimental et entraîné sur pseudo-labels métier contrôlés. "
    "Matching V3 reste la baseline officielle."
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_feature_names(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        names = payload.get("feature_names")
    else:
        names = payload
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("feature_names must be a JSON list or an object containing a feature_names list.")
    return names


def load_model(path: str | Path) -> Any:
    return joblib.load(path)


def extract_feature_vector(row: dict[str, Any], feature_names: list[str]) -> list[float]:
    features = row.get("features")
    if not isinstance(features, dict):
        raise ValueError("Each row must contain a features object.")
    values: list[float] = []
    missing: list[str] = []
    for name in feature_names:
        if name not in features:
            missing.append(name)
            values.append(0.0)
        else:
            values.append(float(features[name]))
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return values


def extract_feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray([extract_feature_vector(row, feature_names) for row in rows], dtype=float)


def predict_positive_scores(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(matrix)
        if probabilities.ndim == 1:
            return np.asarray(probabilities, dtype=float)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
        return np.asarray(probabilities[:, positive_index], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(matrix), dtype=float)
        return 1 / (1 + np.exp(-raw))
    return np.asarray(model.predict(matrix), dtype=float)


def build_reranked_candidates(rows: list[dict[str, Any]], scores: list[float], feature_names: list[str]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row, score in zip(rows, scores, strict=True):
        features = row.get("features") or {}
        final_score_v3 = float(features.get("final_score_v3", 0.0))
        experimental_ml_score = float(np.clip(score, 0.0, 1.0))
        enriched.append(
            {
                "candidate_id": row.get("candidate_id"),
                "profile_id": row.get("profile_id"),
                "rank_v3": int(row.get("rank", 0)),
                "final_score_v3": final_score_v3,
                "experimental_ml_score": experimental_ml_score,
                "ml_rank": None,
                "score_delta": experimental_ml_score - final_score_v3,
                "features": {name: float(features.get(name, 0.0)) for name in feature_names},
            }
        )

    ordered = sorted(enriched, key=lambda item: item["experimental_ml_score"], reverse=True)
    for index, candidate in enumerate(ordered, start=1):
        candidate["ml_rank"] = index
    return sorted(enriched, key=lambda item: int(item["rank_v3"]))


def build_reranking_report(
    *,
    rows: list[dict[str, Any]],
    model: Any,
    feature_names: list[str],
    model_name: str = "xgboost",
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot rerank an empty feature file.")
    matrix = extract_feature_matrix(rows, feature_names)
    scores = predict_positive_scores(model, matrix).tolist()
    job_ids = sorted({str(row.get("job_id")) for row in rows})
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": job_ids[0] if len(job_ids) == 1 else job_ids,
        "model_name": model_name,
        "ranking_mode": "experimental_ml_reranking",
        "methodology_warning": METHODOLOGY_WARNING,
        "feature_names": feature_names,
        "candidate_count": len(rows),
        "candidates": build_reranked_candidates(rows, scores, feature_names),
    }


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown_report(report: dict[str, Any]) -> str:
    candidates = list(report["candidates"])
    top_v3 = sorted(candidates, key=lambda item: int(item["rank_v3"]))[:10]
    top_ml = sorted(candidates, key=lambda item: int(item["ml_rank"]))[:10]
    biggest_changes = sorted(
        candidates,
        key=lambda item: abs(int(item["rank_v3"]) - int(item["ml_rank"])),
        reverse=True,
    )[:10]

    lines = [
        "# ML re-ranking expérimental XGBoost",
        "",
        "## Objectif",
        "",
        "Ajouter une couche de re-ranking ML standalone au-dessus des résultats Matching V3, sans remplacer la baseline officielle.",
        "",
        "## Job",
        "",
        f"- `job_id`: `{report['job_id']}`",
        f"- modèle: `{report['model_name']}`",
        f"- mode: `{report['ranking_mode']}`",
        f"- candidats rerankés: `{report['candidate_count']}`",
        "",
        "## Top 10 Matching V3",
        "",
        "| rank_v3 | candidate_id | final_score_v3 | experimental_ml_score | ml_rank |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(_candidate_table_lines(top_v3, order_key="rank_v3"))
    lines.extend(
        [
            "",
            "## Top 10 XGBoost experimental reranking",
            "",
            "| ml_rank | candidate_id | experimental_ml_score | final_score_v3 | rank_v3 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    lines.extend(_candidate_table_lines(top_ml, order_key="ml_rank"))
    lines.extend(
        [
            "",
            "## Changements de rang les plus importants",
            "",
            "| candidate_id | rank_v3 | ml_rank | delta_rank | score_delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in biggest_changes:
        rank_v3 = int(candidate["rank_v3"])
        ml_rank = int(candidate["ml_rank"])
        lines.append(
            f"| `{candidate['candidate_id']}` | {rank_v3} | {ml_rank} | {ml_rank - rank_v3} | "
            f"{candidate['score_delta']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Limites méthodologiques",
            "",
            f"- {report['methodology_warning']}",
            "- `experimental_ml_score` n'est pas un score recruteur final.",
            "- Les pseudo-labels utilisés pour entraîner le modèle sont dérivés de règles métier contrôlées.",
            "- Cette étape valide une intégration technique de re-ranking, pas un modèle supervisé final.",
            "- Les Decision Cards ne sont pas modifiées par cette expérimentation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown_report(report), encoding="utf-8")


def _candidate_table_lines(candidates: list[dict[str, Any]], *, order_key: str) -> list[str]:
    lines: list[str] = []
    for candidate in sorted(candidates, key=lambda item: int(item[order_key])):
        if order_key == "rank_v3":
            lines.append(
                f"| {candidate['rank_v3']} | `{candidate['candidate_id']}` | "
                f"{candidate['final_score_v3']:.4f} | {candidate['experimental_ml_score']:.4f} | "
                f"{candidate['ml_rank']} |"
            )
        else:
            lines.append(
                f"| {candidate['ml_rank']} | `{candidate['candidate_id']}` | "
                f"{candidate['experimental_ml_score']:.4f} | {candidate['final_score_v3']:.4f} | "
                f"{candidate['rank_v3']} |"
            )
    return lines

