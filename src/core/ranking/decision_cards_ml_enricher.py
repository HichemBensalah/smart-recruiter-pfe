from __future__ import annotations

import copy
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
    "Matching V3 reste la baseline officielle. Random Forest est le meilleur modèle ML actuel selon les "
    "métriques observées. XGBoost est conservé pour SHAP et l’analyse avancée. Les scores ML sont issus "
    "de modèles entraînés sur pseudo-labels métier contrôlés."
)


DEFAULT_DECISION_CARDS_PATH = Path("docs/reports/matching/v3/decision_cards_v3_normalized.json")


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_decision_cards(path: str | Path = DEFAULT_DECISION_CARDS_PATH) -> dict[str, dict[str, Any]]:
    card_path = Path(path)
    if not card_path.exists():
        return {}
    payload = read_json(card_path)
    cards = payload.get("decision_cards", [])
    if not isinstance(cards, list):
        return {}
    return {
        str(card.get("candidate_id")): copy.deepcopy(card)
        for card in cards
        if isinstance(card, dict) and card.get("candidate_id")
    }


def load_shap_top_features(path: str | Path | None, limit: int = 5) -> list[dict[str, Any]]:
    if path is None:
        return []
    shap_path = Path(path)
    if not shap_path.exists():
        return []
    payload = read_json(shap_path)
    top_features = payload.get("top_features", [])
    if not isinstance(top_features, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in top_features[:limit]:
        if isinstance(item, dict) and item.get("feature") is not None:
            normalized.append(
                {
                    "feature": str(item["feature"]),
                    "mean_abs_shap": float(item.get("mean_abs_shap", item.get("importance", 0.0))),
                }
            )
    return normalized


def score_random_forest(rows: list[dict[str, Any]], rf_model: Any, feature_names: list[str]) -> list[float]:
    matrix = extract_feature_matrix(rows, feature_names)
    return [float(np.clip(score, 0.0, 1.0)) for score in predict_positive_scores(rf_model, matrix).tolist()]


def build_rf_rank_lookup(rows: list[dict[str, Any]], rf_scores: list[float]) -> dict[str, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row, score in zip(rows, rf_scores, strict=True):
        candidates.append(
            {
                "candidate_id": str(row.get("candidate_id")),
                "rf_score": float(score),
            }
        )
    ordered = sorted(candidates, key=lambda item: item["rf_score"], reverse=True)
    for index, candidate in enumerate(ordered, start=1):
        candidate["rf_rank"] = index
    return {candidate["candidate_id"]: candidate for candidate in ordered}


def build_xgb_lookup(xgb_ranking: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = xgb_ranking.get("candidates", [])
    if not isinstance(candidates, list):
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate.get("candidate_id"):
            continue
        lookup[str(candidate["candidate_id"])] = candidate
    return lookup


def recommendation_status(
    *,
    baseline_rank_v3: int,
    rf_rank: int,
    xgboost_rank: int,
) -> str:
    v3_vs_rf = baseline_rank_v3 - rf_rank
    v3_vs_xgb = baseline_rank_v3 - xgboost_rank
    rf_vs_xgb = rf_rank - xgboost_rank
    max_gap = max(abs(v3_vs_rf), abs(v3_vs_xgb), abs(rf_vs_xgb))
    if max_gap <= 3:
        return "agreement_high"
    if v3_vs_rf >= 10 and v3_vs_xgb >= 10:
        return "ml_promoted"
    if v3_vs_rf <= -10 and v3_vs_xgb <= -10:
        return "ml_demoted"
    if max_gap >= 10:
        return "review_needed"
    return "review_needed"


def build_ml_comparison_cards(
    *,
    rows: list[dict[str, Any]],
    rf_model: Any,
    feature_names: list[str],
    xgb_ranking: dict[str, Any],
    shap_top_features: list[dict[str, Any]],
    official_cards_by_candidate: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot build ML comparison cards from an empty feature file.")

    rf_lookup = build_rf_rank_lookup(rows, score_random_forest(rows, rf_model, feature_names))
    xgb_lookup = build_xgb_lookup(xgb_ranking)
    official_lookup = official_cards_by_candidate or {}
    job_ids = sorted({str(row.get("job_id")) for row in rows})

    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id"))
        profile_id = str(row.get("profile_id"))
        features = row.get("features") or {}
        rf_candidate = rf_lookup[candidate_id]
        xgb_candidate = xgb_lookup.get(candidate_id)
        if xgb_candidate is None:
            raise ValueError(f"Candidate {candidate_id} is missing from XGBoost ranking.")

        baseline_rank_v3 = int(row.get("rank", xgb_candidate.get("baseline_rank_v3", 0)))
        baseline_score_v3 = float(features.get("final_score_v3", xgb_candidate.get("baseline_score_v3", 0.0)))
        rf_rank = int(rf_candidate["rf_rank"])
        xgboost_rank = int(xgb_candidate["final_rank_ml"])
        rank_shift_v3_vs_rf = baseline_rank_v3 - rf_rank
        rank_shift_v3_vs_xgb = baseline_rank_v3 - xgboost_rank
        rank_shift_rf_vs_xgb = rf_rank - xgboost_rank
        official_card = official_lookup.get(candidate_id)

        candidate: dict[str, Any] = {
            "candidate_id": candidate_id,
            "profile_id": profile_id,
            "baseline_score_v3": baseline_score_v3,
            "baseline_rank_v3": baseline_rank_v3,
            "rf_score": float(rf_candidate["rf_score"]),
            "rf_rank": rf_rank,
            "xgboost_score": float(xgb_candidate["xgboost_score"]),
            "xgboost_rank": xgboost_rank,
            "rank_shift_v3_vs_rf": rank_shift_v3_vs_rf,
            "rank_shift_v3_vs_xgb": rank_shift_v3_vs_xgb,
            "rank_shift_rf_vs_xgb": rank_shift_rf_vs_xgb,
            "shap_top_features": shap_top_features,
            "recommendation_status": recommendation_status(
                baseline_rank_v3=baseline_rank_v3,
                rf_rank=rf_rank,
                xgboost_rank=xgboost_rank,
            ),
            "features": {name: float(features.get(name, 0.0)) for name in feature_names},
        }
        if official_card:
            candidate["official_decision_card"] = {
                "rank": official_card.get("rank"),
                "candidate_name": official_card.get("candidate_name"),
                "verdict": official_card.get("verdict"),
                "strengths": official_card.get("strengths", []),
                "weaknesses": official_card.get("weaknesses", []),
                "interview_focus": official_card.get("interview_focus", []),
            }
        candidates.append(candidate)

    candidates.sort(key=lambda item: int(item["baseline_rank_v3"]))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": job_ids[0] if len(job_ids) == 1 else job_ids,
        "card_type": "ml_comparison",
        "methodological_note": METHODOLOGICAL_NOTE,
        "candidate_count": len(candidates),
        "official_cards_matched_count": sum(1 for candidate in candidates if "official_decision_card" in candidate),
        "candidates": candidates,
    }


def build_ml_comparison_report(
    *,
    features_path: str | Path,
    rf_model_path: str | Path,
    feature_names_path: str | Path,
    xgb_ranking_path: str | Path,
    shap_global_path: str | Path | None = None,
    decision_cards_path: str | Path = DEFAULT_DECISION_CARDS_PATH,
) -> dict[str, Any]:
    rows = load_jsonl(features_path)
    rf_model = load_model(rf_model_path)
    feature_names = load_feature_names(feature_names_path)
    xgb_ranking = read_json(xgb_ranking_path)
    shap_top_features = load_shap_top_features(shap_global_path)
    official_cards = load_decision_cards(decision_cards_path)
    report = build_ml_comparison_cards(
        rows=rows,
        rf_model=rf_model,
        feature_names=feature_names,
        xgb_ranking=xgb_ranking,
        shap_top_features=shap_top_features,
        official_cards_by_candidate=official_cards,
    )
    report["input_paths"] = {
        "features": str(features_path),
        "rf_model": str(rf_model_path),
        "feature_names": str(feature_names_path),
        "xgb_ranking": str(xgb_ranking_path),
        "shap_global": str(shap_global_path) if shap_global_path else None,
        "decision_cards": str(decision_cards_path),
    }
    return report


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown_report(report: dict[str, Any]) -> str:
    candidates = list(report["candidates"])
    top_v3 = sorted(candidates, key=lambda item: int(item["baseline_rank_v3"]))[:10]
    top_rf = sorted(candidates, key=lambda item: int(item["rf_rank"]))[:10]
    top_xgb = sorted(candidates, key=lambda item: int(item["xgboost_rank"]))[:10]
    disagreements = sorted(
        candidates,
        key=lambda item: max(
            abs(int(item["rank_shift_v3_vs_rf"])),
            abs(int(item["rank_shift_v3_vs_xgb"])),
            abs(int(item["rank_shift_rf_vs_xgb"])),
        ),
        reverse=True,
    )[:10]

    examples = [candidate for candidate in candidates if "official_decision_card" in candidate][:5]

    lines = [
        "# Decision Cards ML comparison",
        "",
        "## Objectif",
        "",
        "Créer une version séparée des Decision Cards qui compare Matching V3, Random Forest et XGBoost sans modifier les cartes officielles.",
        "",
        "## Architecture",
        "",
        "- Matching V3 reste la baseline officielle et le score de référence.",
        "- Random Forest est le meilleur modèle ML actuel selon les métriques observées.",
        "- XGBoost est conservé pour SHAP et l'analyse avancée.",
        "- Les trois scores sont affichés pour transparence, sans décision automatique finale.",
        "",
        "## Synthèse",
        "",
        f"- `job_id`: `{report['job_id']}`",
        f"- cartes générées: `{report['candidate_count']}`",
        f"- cartes officielles rattachées: `{report['official_cards_matched_count']}`",
        "",
        "## Top 10 Matching V3",
        "",
        "| baseline_rank_v3 | candidate_id | baseline_score_v3 | rf_rank | xgboost_rank | status |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    lines.extend(_comparison_lines(top_v3, mode="v3"))
    lines.extend(
        [
            "",
            "## Top 10 Random Forest",
            "",
            "| rf_rank | candidate_id | rf_score | baseline_rank_v3 | xgboost_rank | status |",
            "| ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    lines.extend(_comparison_lines(top_rf, mode="rf"))
    lines.extend(
        [
            "",
            "## Top 10 XGBoost",
            "",
            "| xgboost_rank | candidate_id | xgboost_score | baseline_rank_v3 | rf_rank | status |",
            "| ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    lines.extend(_comparison_lines(top_xgb, mode="xgb"))
    lines.extend(
        [
            "",
            "## Candidats avec fort désaccord",
            "",
            "| candidate_id | baseline_rank_v3 | rf_rank | xgboost_rank | v3_vs_rf | v3_vs_xgb | rf_vs_xgb | status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for candidate in disagreements:
        lines.append(
            f"| `{candidate['candidate_id']}` | {candidate['baseline_rank_v3']} | {candidate['rf_rank']} | "
            f"{candidate['xgboost_rank']} | {candidate['rank_shift_v3_vs_rf']} | "
            f"{candidate['rank_shift_v3_vs_xgb']} | {candidate['rank_shift_rf_vs_xgb']} | "
            f"{candidate['recommendation_status']} |"
        )

    lines.extend(["", "## Exemples de cartes candidats", ""])
    for candidate in examples:
        card = candidate["official_decision_card"]
        shap_names = ", ".join(item["feature"] for item in candidate.get("shap_top_features", [])[:3]) or "NA"
        lines.extend(
            [
                f"### {card.get('candidate_name') or candidate['candidate_id']}",
                "",
                f"- `candidate_id`: `{candidate['candidate_id']}`",
                f"- verdict officiel: {card.get('verdict')}",
                f"- Matching V3: rank `{candidate['baseline_rank_v3']}`, score `{candidate['baseline_score_v3']:.4f}`",
                f"- Random Forest: rank `{candidate['rf_rank']}`, score `{candidate['rf_score']:.4f}`",
                f"- XGBoost: rank `{candidate['xgboost_rank']}`, score `{candidate['xgboost_score']:.4f}`",
                f"- status: `{candidate['recommendation_status']}`",
                f"- SHAP top features: {shap_names}",
                "",
            ]
        )

    lines.extend(
        [
            "## Note méthodologique",
            "",
            f"- {report['methodological_note']}",
            "- Les Decision Cards officielles ne sont pas modifiées.",
            "- Les datasets, modèles, FAISS et MongoDB ne sont pas modifiés.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown_report(report), encoding="utf-8")


def _comparison_lines(candidates: list[dict[str, Any]], *, mode: str) -> list[str]:
    lines: list[str] = []
    for candidate in candidates:
        if mode == "v3":
            lines.append(
                f"| {candidate['baseline_rank_v3']} | `{candidate['candidate_id']}` | "
                f"{candidate['baseline_score_v3']:.4f} | {candidate['rf_rank']} | "
                f"{candidate['xgboost_rank']} | {candidate['recommendation_status']} |"
            )
        elif mode == "rf":
            lines.append(
                f"| {candidate['rf_rank']} | `{candidate['candidate_id']}` | {candidate['rf_score']:.4f} | "
                f"{candidate['baseline_rank_v3']} | {candidate['xgboost_rank']} | "
                f"{candidate['recommendation_status']} |"
            )
        else:
            lines.append(
                f"| {candidate['xgboost_rank']} | `{candidate['candidate_id']}` | {candidate['xgboost_score']:.4f} | "
                f"{candidate['baseline_rank_v3']} | {candidate['rf_rank']} | {candidate['recommendation_status']} |"
            )
    return lines

