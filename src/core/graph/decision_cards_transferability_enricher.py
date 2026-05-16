from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.graph.transferability import compute_transferability, load_json, load_yaml_graph


METHODOLOGICAL_NOTE = (
    "Potential Graph est une couche explicative déclarative basée sur les skills structurées et un graphe YAML "
    "de rôles. Il ne remplace pas Matching V3, ne remplace pas les modèles ML et ne constitue pas une décision "
    "recruteur. Il sert à analyser la transférabilité métier et les gaps."
)

DEFAULT_MONGODB_REPORT_PATH = Path("docs/reports/mongodb/mongodb_import_report_v2_grounded_execute.json")
DEFAULT_MATCHING_REPORT_PATH = Path("docs/reports/matching/v3/backend_python_django_postgresql_matching_report_v3_normalized.json")
LOOKUP_STATUSES = {
    "found_by_profile_id",
    "found_by_candidate_id",
    "found_by_source_filename",
    "found_by_original_filename",
    "found_by_fallback",
    "profile_not_found",
}


def load_cards(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return [copy.deepcopy(card) for card in candidates if isinstance(card, dict)]
    decision_cards = payload.get("decision_cards")
    if isinstance(decision_cards, list):
        return [copy.deepcopy(card) for card in decision_cards if isinstance(card, dict)]
    raise ValueError("Cards input must contain either candidates or decision_cards.")


def build_profile_index(
    profiles_dir: str | Path,
    metadata_report_path: str | Path = DEFAULT_MONGODB_REPORT_PATH,
    matching_report_path: str | Path = DEFAULT_MATCHING_REPORT_PATH,
) -> dict[str, dict[str, Path]]:
    root = Path(profiles_dir)
    index: dict[str, dict[str, Path]] = {
        "by_profile_id": {},
        "by_candidate_id": {},
        "by_source_filename": {},
        "by_original_filename": {},
        "by_normalized_name": {},
        "by_fallback": {},
    }

    if Path(metadata_report_path).exists():
        metadata = load_json(metadata_report_path)
        for profile_meta in _iter_profile_metadata(metadata):
            _add_metadata_to_index(index, profile_meta, root)

    if Path(matching_report_path).exists():
        matching_report = load_json(matching_report_path)
        for profile_meta in _iter_matching_profile_metadata(matching_report):
            _add_metadata_to_index(index, profile_meta, root)

    if root.exists():
        for path in sorted(root.glob("*.json")):
            _add_path_keys(index, path)
            try:
                profile = load_json(path)
            except Exception:
                continue
            for key in ("source_path", "artifact_path", "original_filename", "source_filename"):
                _add_filename_keys(index, profile.get(key), path)
            full_name = None
            if isinstance(profile.get("profile"), dict):
                full_name = ((profile.get("profile") or {}).get("bio") or {}).get("full_name")
            if full_name:
                index["by_normalized_name"].setdefault(_normalize_key(full_name), path)
    return index


def enrich_cards_with_transferability(
    *,
    cards: list[dict[str, Any]],
    profiles_dir: str | Path,
    job_profile: dict[str, Any],
    graph: dict[str, Any],
) -> dict[str, Any]:
    profile_index = build_profile_index(profiles_dir)
    enriched: list[dict[str, Any]] = []
    profiles_found = 0
    status_counter: Counter[str] = Counter()

    for card in cards:
        candidate = copy.deepcopy(card)
        profile_path, lookup_status = _resolve_profile_path(candidate, profile_index)
        candidate["profile_lookup_status"] = lookup_status
        status_counter[lookup_status] += 1
        if profile_path is None:
            candidate["transferability"] = _missing_profile_transferability(candidate, job_profile)
        else:
            profile = load_json(profile_path)
            transferability = compute_transferability(
                profile=profile,
                job=job_profile,
                graph=graph,
                candidate_id=str(candidate.get("candidate_id") or profile_path.stem),
            )
            transferability["profile_path"] = str(profile_path)
            candidate["transferability"] = transferability
            profiles_found += 1
        enriched.append(candidate)

    total_cards = len(enriched)
    profiles_not_found = total_cards - profiles_found
    lookup_success_rate = profiles_found / total_cards if total_cards else 0.0
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": str(job_profile.get("job_id") or "unknown_job"),
        "card_type": "decision_cards_with_transferability",
        "methodological_note": METHODOLOGICAL_NOTE,
        "candidate_count": total_cards,
        "total_cards": total_cards,
        "profiles_matched_count": profiles_found,
        "profiles_missing_count": profiles_not_found,
        "profiles_found": profiles_found,
        "profiles_not_found": profiles_not_found,
        "lookup_success_rate": round(lookup_success_rate, 4),
        "lookup_status_distribution": dict(sorted(status_counter.items())),
        "candidates": enriched,
    }


def build_transferability_cards_report(
    *,
    cards_path: str | Path,
    profiles_dir: str | Path,
    job_path: str | Path,
    graph_path: str | Path,
) -> dict[str, Any]:
    cards = load_cards(cards_path)
    job_profile = load_json(job_path)
    graph = load_yaml_graph(graph_path)
    report = enrich_cards_with_transferability(
        cards=cards,
        profiles_dir=profiles_dir,
        job_profile=job_profile,
        graph=graph,
    )
    report["input_paths"] = {
        "cards": str(cards_path),
        "profiles_dir": str(profiles_dir),
        "job": str(job_path),
        "graph": str(graph_path),
    }
    return report


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown_report(report: dict[str, Any]) -> str:
    candidates = list(report["candidates"])
    direct_fits = [candidate for candidate in candidates if candidate.get("transferability", {}).get("fit_direct") is True][:10]
    plausible_transitions = [
        candidate for candidate in candidates if candidate.get("transferability", {}).get("transitions_plausibles")
    ][:10]
    compensable_examples = [
        candidate for candidate in candidates if candidate.get("transferability", {}).get("gaps_compensables")
    ][:10]
    blocking_examples = [
        candidate for candidate in candidates if candidate.get("transferability", {}).get("gaps_bloquants")
    ][:10]
    not_found_examples = [candidate for candidate in candidates if candidate.get("profile_lookup_status") == "profile_not_found"][:10]

    lines = [
        "# Decision Cards avec Potential Graph",
        "",
        "## Objectif",
        "",
        "Créer une version séparée des Decision Cards enrichie avec une analyse de transférabilité métier basée sur le Potential Graph YAML.",
        "",
        "## Architecture",
        "",
        "- Matching V3 reste la baseline officielle et le score de référence.",
        "- Les scores ML sont conservés si les cartes d'entrée les contiennent.",
        "- Potential Graph ajoute une lecture explicable: fit direct, transition plausible, gaps compensables et gaps bloquants.",
        "- Aucun modèle n'est réentraîné et aucune carte officielle n'est modifiée.",
        "",
        "## Synthèse",
        "",
        f"- `job_id`: `{report['job_id']}`",
        f"- total cartes: `{report['total_cards']}`",
        f"- profils retrouvés: `{report['profiles_found']}`",
        f"- profils non retrouvés: `{report['profiles_not_found']}`",
        f"- taux de succès lookup: `{float(report['lookup_success_rate']):.2%}`",
        f"- distribution lookup: `{report['lookup_status_distribution']}`",
        "",
        "## Top cartes avec fit direct",
        "",
    ]
    lines.extend(_transferability_table(direct_fits))
    lines.extend(["", "## Top cartes avec transition plausible", ""])
    lines.extend(_transferability_table(plausible_transitions))
    lines.extend(["", "## Exemples de gaps compensables", ""])
    lines.extend(_gap_lines(compensable_examples, "gaps_compensables"))
    lines.extend(["", "## Exemples de gaps bloquants", ""])
    lines.extend(_gap_lines(blocking_examples, "gaps_bloquants"))
    lines.extend(["", "## Exemples de profils non retrouvés", ""])
    lines.extend(_not_found_lines(not_found_examples))
    lines.extend(
        [
            "",
            "## Limites méthodologiques",
            "",
            f"- {report['methodological_note']}",
            "- Les résultats dépendent de la qualité des skills structurées par le Module 2.",
            "- Les transitions sont déclaratives et doivent être validées par un recruteur ou expert métier.",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown_report(report), encoding="utf-8")


def _iter_profile_metadata(payload: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        if payload.get("profile_id") and payload.get("module2_file_path"):
            found.append(payload)
        for value in payload.values():
            found.extend(_iter_profile_metadata(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_iter_profile_metadata(item))
    return found


def _iter_matching_profile_metadata(payload: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        profile_id = payload.get("profile_id") or payload.get("matched_profile_id")
        candidate_id = payload.get("candidate_id")
        source_path = payload.get("source_path")
        if profile_id and (source_path or candidate_id):
            found.append(
                {
                    "profile_id": profile_id,
                    "candidate_id": candidate_id,
                    "source_path": source_path,
                    "module2_file_path": _module2_path_from_source_path(source_path),
                }
            )
        for value in payload.values():
            found.extend(_iter_matching_profile_metadata(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_iter_matching_profile_metadata(item))
    return found


def _add_metadata_to_index(index: dict[str, dict[str, Path]], profile_meta: dict[str, Any], profiles_root: Path) -> None:
    path = _resolve_metadata_path(profile_meta, profiles_root)
    if path is None:
        return
    profile_id = profile_meta.get("profile_id") or profile_meta.get("matched_profile_id")
    candidate_id = profile_meta.get("candidate_id")
    full_name = profile_meta.get("full_name")
    if profile_id:
        index["by_profile_id"].setdefault(str(profile_id), path)
    if candidate_id:
        index["by_candidate_id"].setdefault(str(candidate_id), path)
    for key in ("source_path", "original_filename", "module2_file_path", "artifact_path"):
        _add_filename_keys(index, profile_meta.get(key), path)
    if full_name:
        index["by_normalized_name"].setdefault(_normalize_key(full_name), path)
    _add_path_keys(index, path)


def _resolve_metadata_path(profile_meta: dict[str, Any], profiles_root: Path) -> Path | None:
    for key in ("module2_file_path", "profile_path", "grounded_profile_path"):
        value = profile_meta.get(key)
        if value:
            path = Path(str(value))
            if path.exists():
                return path
            rooted = profiles_root / path.name
            if rooted.exists():
                return rooted
    source_path = profile_meta.get("source_path") or profile_meta.get("original_filename")
    guessed = _module2_path_from_source_path(source_path, profiles_root)
    if guessed and guessed.exists():
        return guessed
    return None


def _resolve_profile_path(card: dict[str, Any], profile_index: dict[str, dict[str, Path]]) -> tuple[Path | None, str]:
    for key in ("profile_id", "matched_profile_id"):
        value = card.get(key)
        if value and str(value) in profile_index["by_profile_id"]:
            return profile_index["by_profile_id"][str(value)], "found_by_profile_id"

    value = card.get("candidate_id")
    if value and str(value) in profile_index["by_candidate_id"]:
        return profile_index["by_candidate_id"][str(value)], "found_by_candidate_id"

    for key in ("source_filename", "source_path", "module2_file_path"):
        for filename_key in _filename_keys(card.get(key)):
            if filename_key in profile_index["by_source_filename"]:
                return profile_index["by_source_filename"][filename_key], "found_by_source_filename"

    for key in ("original_filename", "artifact_path"):
        for filename_key in _filename_keys(card.get(key)):
            if filename_key in profile_index["by_original_filename"]:
                return profile_index["by_original_filename"][filename_key], "found_by_original_filename"

    full_name = card.get("candidate_name") or card.get("full_name")
    if full_name:
        normalized_name = _normalize_key(full_name)
        if normalized_name in profile_index["by_normalized_name"]:
            return profile_index["by_normalized_name"][normalized_name], "found_by_fallback"

    for key in ("profile_id", "candidate_id", "matched_profile_id"):
        value = card.get(key)
        if value and str(value) in profile_index["by_fallback"]:
            return profile_index["by_fallback"][str(value)], "found_by_fallback"
    return None, "profile_not_found"


def _missing_profile_transferability(card: dict[str, Any], job_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(card.get("candidate_id") or "unknown_candidate"),
        "job_id": str(job_profile.get("job_id") or "unknown_job"),
        "target_role": str(job_profile.get("job_title") or "unknown_role"),
        "fit_direct": False,
        "direct_fit_score": 0.0,
        "best_source_role": "unknown",
        "transferability_score": 0.0,
        "transitions_plausibles": [],
        "matched_core_skills": [],
        "missing_core_skills": [],
        "matched_adjacent_skills": [],
        "gaps_compensables": [],
        "gaps_bloquants": [],
        "explanation": "Profil candidat introuvable dans le dossier grounded profiles; transférabilité non calculée.",
        "status": "profile_not_found",
    }


def _module2_path_from_source_path(source_path: Any, profiles_root: Path | None = None) -> Path | None:
    if not source_path:
        return None
    source = Path(str(source_path))
    stem = source.stem
    parent_name = source.parent.name.lower()
    if parent_name in {"pdf", "docx", "images", "image"}:
        prefix = "images" if parent_name == "image" else parent_name
        filename = f"{prefix}_{stem}.json"
    else:
        filename = f"{stem}.json"
    root = profiles_root or Path("data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles")
    return root / filename


def _add_path_keys(index: dict[str, dict[str, Path]], path: Path) -> None:
    for key in (path.stem, path.name):
        index["by_source_filename"].setdefault(key, path)
        index["by_original_filename"].setdefault(key, path)
        index["by_fallback"].setdefault(key, path)


def _add_filename_keys(index: dict[str, dict[str, Path]], value: Any, path: Path) -> None:
    for key in _filename_keys(value):
        index["by_source_filename"].setdefault(key, path)
        index["by_original_filename"].setdefault(key, path)
        index["by_fallback"].setdefault(key, path)


def _filename_keys(value: Any) -> list[str]:
    if not value:
        return []
    path = Path(str(value))
    keys = {path.name, path.stem}
    guessed = _module2_path_from_source_path(value)
    if guessed:
        keys.add(guessed.name)
        keys.add(guessed.stem)
    return [key for key in keys if key]


def _normalize_key(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _transferability_table(candidates: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| candidate_id | fit_direct | direct_fit_score | transferability_score | best_source_role | target_role | lookup |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    if not candidates:
        lines.append("| Aucun | NA | NA | NA | NA | NA | NA |")
        return lines
    for candidate in candidates:
        transfer = candidate["transferability"]
        lines.append(
            f"| `{candidate.get('candidate_id')}` | {transfer['fit_direct']} | "
            f"{float(transfer['direct_fit_score']):.4f} | {float(transfer['transferability_score']):.4f} | "
            f"{transfer['best_source_role']} | {transfer['target_role']} | {candidate.get('profile_lookup_status')} |"
        )
    return lines


def _gap_lines(candidates: list[dict[str, Any]], key: str) -> list[str]:
    lines = ["| candidate_id | gaps | explanation |", "| --- | --- | --- |"]
    if not candidates:
        lines.append("| Aucun | NA | NA |")
        return lines
    for candidate in candidates:
        transfer = candidate["transferability"]
        gaps = ", ".join(transfer.get(key, [])) or "Aucun"
        explanation = str(transfer.get("explanation", "")).replace("|", "/")
        lines.append(f"| `{candidate.get('candidate_id')}` | {gaps} | {explanation} |")
    return lines


def _not_found_lines(candidates: list[dict[str, Any]]) -> list[str]:
    lines = ["| candidate_id | profile_id | baseline_rank_v3 |", "| --- | --- | ---: |"]
    if not candidates:
        lines.append("| Aucun | NA | NA |")
        return lines
    for candidate in candidates:
        lines.append(
            f"| `{candidate.get('candidate_id')}` | `{candidate.get('profile_id')}` | "
            f"{candidate.get('baseline_rank_v3', 'NA')} |"
        )
    return lines

