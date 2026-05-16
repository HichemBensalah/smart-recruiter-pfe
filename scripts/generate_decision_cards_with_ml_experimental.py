from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_WARNING = (
    "Le score ML est expérimental et entraîné sur pseudo-labels métier contrôlés. "
    "Matching V3 reste la baseline officielle."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate experimental Decision Cards enriched with standalone ML reranking scores."
    )
    parser.add_argument(
        "--decision-cards",
        type=Path,
        default=Path("docs/reports/matching/v3/decision_cards_v3_normalized.json"),
    )
    parser.add_argument(
        "--ml-reranking",
        type=Path,
        default=Path("docs/reports/ml/ml_reranking_example.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/reports/ml/decision_cards_with_ml_experimental.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/reports/ml/decision_cards_with_ml_experimental.md"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def enrich_cards(decision_cards_payload: dict[str, Any], ml_payload: dict[str, Any]) -> dict[str, Any]:
    warning = str(ml_payload.get("methodology_warning") or DEFAULT_WARNING)
    ml_candidates = {
        str(candidate.get("candidate_id")): candidate
        for candidate in ml_payload.get("candidates", [])
        if isinstance(candidate, dict) and candidate.get("candidate_id")
    }

    source_cards = decision_cards_payload.get("decision_cards", [])
    if not isinstance(source_cards, list):
        raise ValueError("decision_cards payload must contain a decision_cards list.")

    enriched_cards: list[dict[str, Any]] = []
    matched_count = 0
    for card in source_cards:
        if not isinstance(card, dict):
            continue
        enriched = copy.deepcopy(card)
        candidate_id = str(enriched.get("candidate_id"))
        ml_candidate = ml_candidates.get(candidate_id)
        rank_v3 = int(enriched.get("rank", 0))
        enriched["rank_v3"] = rank_v3
        if ml_candidate:
            matched_count += 1
            ml_rank = int(ml_candidate["ml_rank"])
            enriched["experimental_ml_score"] = float(ml_candidate["experimental_ml_score"])
            enriched["ml_rank"] = ml_rank
            enriched["score_delta"] = float(ml_candidate["score_delta"])
            enriched["rank_shift"] = rank_v3 - ml_rank
        else:
            enriched["experimental_ml_score"] = None
            enriched["ml_rank"] = None
            enriched["score_delta"] = None
            enriched["rank_shift"] = None
        enriched["ml_methodology_warning"] = warning
        enriched_cards.append(enriched)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_decision_cards": str(decision_cards_payload.get("source_report", "decision_cards_v3_normalized.json")),
        "source_ml_reranking": "ml_reranking_example.json",
        "ranking_mode": "matching_v3_with_experimental_ml_annotations",
        "primary_score": "final_score",
        "experimental_score": "experimental_ml_score",
        "methodology_warning": warning,
        "cards_count": len(enriched_cards),
        "ml_matched_cards_count": matched_count,
        "ml_unmatched_cards_count": len(enriched_cards) - matched_count,
        "decision_cards": enriched_cards,
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    cards = report["decision_cards"]
    top_matching = sorted(cards, key=lambda item: int(item.get("rank", 0)))[:10]
    top_ml = sorted(
        [card for card in cards if card.get("ml_rank") is not None],
        key=lambda item: int(item["ml_rank"]),
    )[:10]
    rank_changes = sorted(
        [card for card in cards if card.get("rank_shift") is not None],
        key=lambda item: abs(int(item["rank_shift"])),
        reverse=True,
    )[:10]

    lines = [
        "# Decision Cards enrichies avec ML expérimental",
        "",
        "## Objectif",
        "",
        "Ajouter les annotations du re-ranking XGBoost expérimental aux Decision Cards, sans modifier les cartes officielles et sans remplacer Matching V3.",
        "",
        "## Statut",
        "",
        f"- cartes enrichies: `{report['cards_count']}`",
        f"- cartes trouvées dans le rapport ML: `{report['ml_matched_cards_count']}`",
        f"- cartes non trouvées dans le rapport ML: `{report['ml_unmatched_cards_count']}`",
        f"- score principal conservé: `{report['primary_score']}`",
        f"- score ML expérimental: `{report['experimental_score']}`",
        "",
        "## Top 10 Matching V3 avec annotations ML",
        "",
        "| rank_v3 | candidate_id | final_score | experimental_ml_score | ml_rank | rank_shift | verdict |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(_card_table_lines(top_matching))
    lines.extend(
        [
            "",
            "## Top XGBoost expérimental parmi les Decision Cards",
            "",
        "| ml_rank | candidate_id | experimental_ml_score | rank_v3 | final_score | rank_shift | verdict |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for card in top_ml:
        lines.append(
            f"| {card['ml_rank']} | `{card['candidate_id']}` | {_fmt(card['experimental_ml_score'])} | "
            f"{card['rank_v3']} | {_fmt(card.get('final_score'))} | {_fmt_int(card.get('rank_shift'))} | "
            f"{card.get('verdict', '')} |"
        )
    lines.extend(
        [
            "",
            "## Changements de rang les plus visibles",
            "",
            "| candidate_id | rank_v3 | ml_rank | rank_shift | score_delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for card in rank_changes:
        lines.append(
            f"| `{card['candidate_id']}` | {card['rank_v3']} | {card['ml_rank']} | "
            f"{card['rank_shift']} | {_fmt(card.get('score_delta'))} |"
        )

    if report["ml_unmatched_cards_count"]:
        lines.extend(
            [
                "",
                "## Cartes sans annotation ML",
                "",
            ]
        )
        for card in cards:
            if card.get("ml_rank") is None:
                lines.append(f"- `{card.get('candidate_id')}`: absent du rapport ML re-ranking fourni.")

    lines.extend(
        [
            "",
            "## Limites méthodologiques",
            "",
            f"- {report['methodology_warning']}",
            "- Matching V3 reste le ranking principal et la baseline officielle.",
            "- `experimental_ml_score` n'est pas un score recruteur final.",
            "- Cette sortie est une version expérimentale séparée, pas une modification des Decision Cards officielles.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown_report(report), encoding="utf-8")


def _card_table_lines(cards: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for card in cards:
        lines.append(
            f"| {card.get('rank_v3')} | `{card.get('candidate_id')}` | {_fmt(card.get('final_score'))} | "
            f"{_fmt(card.get('experimental_ml_score'))} | {_fmt_int(card.get('ml_rank'))} | "
            f"{_fmt_int(card.get('rank_shift'))} | {card.get('verdict', '')} |"
        )
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "NA"
    return str(int(value))


def main() -> None:
    args = parse_args()
    decision_cards_payload = read_json(args.decision_cards)
    ml_payload = read_json(args.ml_reranking)
    report = enrich_cards(decision_cards_payload, ml_payload)
    report["input_paths"] = {
        "decision_cards": str(args.decision_cards),
        "ml_reranking": str(args.ml_reranking),
    }
    write_json(args.output_json, report)
    write_markdown(args.output_md, report)
    print(f"Experimental ML Decision Cards JSON written: {args.output_json}")
    print(f"Experimental ML Decision Cards Markdown written: {args.output_md}")


if __name__ == "__main__":
    main()
