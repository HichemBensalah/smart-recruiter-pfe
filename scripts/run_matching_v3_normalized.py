from __future__ import annotations

import copy
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.matching.faiss_indexer import DEFAULT_INDEX_PATH, DEFAULT_REPORT_PATH, DEFAULT_SENTENCE_MODEL
from src.core.matching.job_text_builder import build_job_text
from src.core.matching.matching_quality_filters import build_display_name, enrich_grounded_quality
from src.core.matching.recommender import (
    _build_explanation,
    _load_sentence_transformer,
    group_by_candidate_id,
    load_faiss_index,
    load_id_map,
    select_best_profile_per_candidate,
)
from src.core.matching.scoring import score_candidate
from src.core.matching.skill_normalizer import normalize_skill
from src.core.common.seniority import compute_seniority_alignment, normalize_seniority


PROFILES_DIR = Path("data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles")
JOB_PROFILE_PATH = Path("data/job_profiles/backend_python_fastapi_mongodb.json")
FROZEN_REPORT_PATH = Path("docs/reports/matching/v3/matching_single_job_report_grounded_v3.json")
AUDIT_PATH = Path("data/skills_audit/raw_skills_variants.json")
TAXONOMY_PATH = Path("data/skills_taxonomy.yaml")
NORMALIZED_REPORT_PATH = Path("docs/reports/matching/v3/matching_report_v3_normalized.json")
TOP_K = 10


BASE_TAXONOMY: dict[str, list[str]] = {
    "FastAPI": ["FastAPI", "Fast API", "fastapi", "fast-api", "fast api", "FastAPI framework"],
    "React": ["React", "React.js", "ReactJS", "React JS", "react.js", "reactjs", "react"],
    "Node.js": ["Node.js", "NodeJS", "Node", "Node JS", "node.js", "nodejs", "node"],
    "MongoDB": ["MongoDB", "Mongo DB", "Mongo", "mongo", "mongo db", "MySQL/SQ Lite 3/MongoDB/JSON"],
    "PostgreSQL": ["PostgreSQL", "Postgres", "Postgre SQL", "postgre sql", "postgresql", "postgres"],
    "JavaScript": ["JavaScript", "Javascript", "Java Script", "JS", "java script", "javascript", "js"],
    "TypeScript": ["TypeScript", "Typescript", "Type Script", "TS", "typescript", "type script", "ts"],
    "Python": [".Python", "Python", "python", "Python/R", "PHP/Ruby on Rails/Java/C++/Python"],
    "Docker": ["Docker", "docker"],
    "Kubernetes": ["Kubernetes", "kubernetes", "K8s", "k8s"],
    "scikit-learn": ["scikit-learn", "Scikit-Learn", "Scikit-learn", "Scikit Learn", "Sci Kit-learn", "sklearn"],
    "TensorFlow": ["TensorFlow", "Tensorflow", "tensorflow", "Tensor Flow"],
    "PyTorch": ["PyTorch", "Pytorch", "pytorch", "Py Torch", "Pytorch Lightning"],
    "REST API": ["REST API", "REST API design", "REST APIs", "Rest API", "RESTful API", "RESTful APIs"],
    "CI/CD": ["CI/CD", "CI CD", "CICD", "DevOps/CI/CD"],
    "AWS": ["AWS", "Aws", "Amazon Web Services", "Cloud (AWS)", "AWStechnologies"],
    "AWS S3": ["AWS S3", "S3"],
    "Apache Airflow": ["Apache Airflow", "Airflow"],
    "Apache Cassandra": ["Apache Cassandra", "Cassandra"],
    "Apache Kafka": ["Apache Kafka", "Kafka"],
    "Apache Spark": ["Apache Spark", "Spark", "Databricks Spark"],
    "Spark MLlib": ["Spark ML", "Spark ML/MLlib", "Spark-MLlib", "mllib"],
    "Hadoop": ["Hadoop", "HADOOP", "Apache Hadoop", "Hadoop/Hive", "NoSQL Hadoop"],
    "Hive": ["Hive", "HIVE", "Hadoop/Hive", "Hive/HQL"],
    "MySQL": ["MySQL", "MySQL/SQL", "MySQL/SQ Lite 3/MongoDB/JSON"],
    "SQLite": ["SQLite", "SQLite 3", "SQ Lite 3", "SQL/SQLite 3", "MySQL/SQ Lite 3/MongoDB/JSON"],
    "SQL": ["SQL", "MySQL/SQL", "SQL/SQLite 3", "Microsoft SQL Server", "SQL Server"],
    "Microsoft SQL Server": ["Microsoft SQL Server", "MSSQL Server", "MSSQLServer", "SQL Server"],
    "NoSQL": ["NoSQL", "NoSQL Hadoop"],
    "PHP": ["PHP", "php", "PHP/Ruby on Rails/Java/C++/Python"],
    "Ruby on Rails": ["Ruby on Rails", "Rubyon", "Ralls", "PHP/Ruby on Rails/Java/C++/Python"],
    "Java": ["Java", "JAVA", "PHP/Ruby on Rails/Java/C++/Python"],
    "C++": ["C++", "C/C++", "PHP/Ruby on Rails/Java/C++/Python"],
    "C": ["C", "C programming", "C/C++"],
    "Angular": ["Angular", "Angular.js"],
    "D3.js": ["D3.js", "D3", "D 3"],
    "Express.js": ["Express Js", "Express.js"],
    "HTML": ["HTML", "HTML 5", "HTML/CSS"],
    "CSS": ["CSS", "HTML/CSS"],
    "Power BI": ["Power BI", "PowerBI", "Power Bl", "Microsoft Power Bl"],
    "MATLAB": ["MATLAB", "Matlab", "Mat Lab"],
    "NumPy": ["NumPy", "Numpy", "numpy"],
    "Pandas": ["Pandas", "pandas"],
    "Matplotlib": ["Matplotlib", "matplotlib"],
    "OpenCV": ["OpenCV", "Open CV"],
    "Keras": ["Keras", "kera"],
    "R": ["R", "Python/R", "R Programming", "R-Programming"],
    "RStudio": ["RStudio", "R Studio"],
    "GitHub": ["GitHub", "Git Hub"],
    "GitHub Actions": ["GitHub Actions"],
    "Machine Learning": ["Machine Learning", "Machine learning", "machine learning", "ML", "ML/Al"],
    "Deep Learning": ["Deep Learning", "deep learning"],
    "Data Science": ["Data Science", "data science"],
    "Neural Networks": ["Neural Networks", "Neural networks", "neural networks", "ANN"],
    "K-Means": ["k-means clustering", "K-Means method", "K-Means method of clustering"],
    "PowerPoint": ["Powerpoint"],
    "Microsoft Office": ["Microsoft Office", "Ms Office"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def skill_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    lowered = value.strip().lower()
    lowered = lowered.replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[./]+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9+#]+", " ", lowered)
    return " ".join(lowered.split())


def clean_skill(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def unique_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_skill(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def load_profile_payloads(profiles_dir: Path = PROFILES_DIR) -> list[tuple[Path, dict[str, Any]]]:
    files = sorted(profiles_dir.glob("*.json"))
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        payloads.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return payloads


def collect_skill_counters(payloads: list[tuple[Path, dict[str, Any]]]) -> tuple[Counter[str], Counter[str], dict[str, set[str]]]:
    hard_counter: Counter[str] = Counter()
    soft_counter: Counter[str] = Counter()
    skill_files: dict[str, set[str]] = defaultdict(set)
    for path, payload in payloads:
        expertise = ((payload.get("profile") or {}).get("expertise") or {})
        for skill in expertise.get("hard_skills") or []:
            cleaned = clean_skill(skill)
            if cleaned:
                hard_counter[cleaned] += 1
                skill_files[cleaned].add(str(path))
        for skill in expertise.get("soft_skills") or []:
            cleaned = clean_skill(skill)
            if cleaned:
                soft_counter[cleaned] += 1
    return hard_counter, soft_counter, skill_files


def build_taxonomy_from_audit(hard_counter: Counter[str]) -> dict[str, dict[str, list[str]]]:
    taxonomy_sets: dict[str, set[str]] = {canonical: set(variants) for canonical, variants in BASE_TAXONOMY.items()}
    manual_variant_keys = {skill_key(variant) for variants in BASE_TAXONOMY.values() for variant in variants}
    manual_groups_by_key: dict[str, list[str]] = defaultdict(list)
    for canonical, variants in BASE_TAXONOMY.items():
        for variant in variants:
            manual_groups_by_key[skill_key(variant)].append(canonical)

    auto_groups: dict[str, set[str]] = defaultdict(set)
    for raw_skill in hard_counter:
        raw_key = skill_key(raw_skill)
        if raw_key in manual_variant_keys:
            for canonical in manual_groups_by_key[raw_key]:
                taxonomy_sets.setdefault(canonical, set()).add(raw_skill)
            continue
        auto_groups[raw_key].add(raw_skill)

    for _, variants in auto_groups.items():
        canonical = choose_auto_canonical(variants, hard_counter)
        taxonomy_sets.setdefault(canonical, set()).update(variants)

    return {
        canonical: {"variants": sorted(variants, key=lambda item: item.lower())}
        for canonical, variants in sorted(taxonomy_sets.items(), key=lambda item: item[0].lower())
    }


def choose_auto_canonical(variants: set[str], hard_counter: Counter[str]) -> str:
    if len(variants) == 1:
        return next(iter(variants))
    normalized = normalize_skill(next(iter(variants)))
    if normalized:
        return normalized
    return sorted(variants, key=lambda item: (-hard_counter[item], item.lower()))[0]


def write_taxonomy(path: Path, taxonomy: dict[str, dict[str, list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(taxonomy, sort_keys=False, allow_unicode=False), encoding="utf-8")


def load_taxonomy(path: Path) -> dict[str, dict[str, list[str]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"taxonomy must be a mapping: {path}")
    taxonomy: dict[str, dict[str, list[str]]] = {}
    for canonical, spec in payload.items():
        variants = (spec or {}).get("variants") if isinstance(spec, dict) else None
        taxonomy[str(canonical)] = {"variants": [str(item) for item in (variants or [])]}
    return taxonomy


def build_variant_map(taxonomy: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    variant_map: dict[str, list[str]] = defaultdict(list)
    for canonical, spec in taxonomy.items():
        for variant in [canonical, *(spec.get("variants") or [])]:
            key = skill_key(variant)
            if key and canonical not in variant_map[key]:
                variant_map[key].append(canonical)
    return dict(variant_map)


def normalize_skill_values(skills: list[Any], variant_map: dict[str, list[str]]) -> tuple[list[str], dict[str, int]]:
    normalized: list[str] = []
    seen: set[str] = set()
    stats = {
        "input_values": 0,
        "values_changed": 0,
        "values_expanded": 0,
        "duplicates_removed": 0,
    }
    for raw in skills or []:
        cleaned = clean_skill(raw)
        if not cleaned:
            continue
        stats["input_values"] += 1
        mapped = variant_map.get(skill_key(cleaned), [cleaned])
        if len(mapped) > 1:
            stats["values_expanded"] += 1
        if mapped != [cleaned]:
            stats["values_changed"] += 1
        for canonical in mapped:
            canonical_key = canonical.lower()
            if canonical_key in seen:
                stats["duplicates_removed"] += 1
                continue
            seen.add(canonical_key)
            normalized.append(canonical)
    return normalized, stats


def normalize_profile_skills(profile_doc: dict[str, Any], variant_map: dict[str, list[str]]) -> tuple[dict[str, Any], dict[str, int], list[dict[str, Any]]]:
    normalized_doc = copy.deepcopy(profile_doc)
    expertise = normalized_doc.setdefault("expertise", {})
    stats = {
        "hard_input_values": 0,
        "soft_input_values": 0,
        "hard_values_changed": 0,
        "soft_values_changed": 0,
        "hard_values_expanded": 0,
        "soft_values_expanded": 0,
        "duplicates_removed": 0,
    }
    changes: list[dict[str, Any]] = []

    for field in ("hard_skills", "soft_skills"):
        before = list(expertise.get(field) or [])
        after, field_stats = normalize_skill_values(before, variant_map)
        expertise[field] = after
        prefix = "hard" if field == "hard_skills" else "soft"
        stats[f"{prefix}_input_values"] += field_stats["input_values"]
        stats[f"{prefix}_values_changed"] += field_stats["values_changed"]
        stats[f"{prefix}_values_expanded"] += field_stats["values_expanded"]
        stats["duplicates_removed"] += field_stats["duplicates_removed"]
        if before != after:
            changes.append({"field": field, "before": before, "after": after})

    return normalized_doc, stats, changes


def build_local_profile_doc(path: Path, payload: dict[str, Any], id_row: dict[str, Any]) -> dict[str, Any]:
    profile = payload.get("profile") or {}
    grounding = payload.get("grounding") or {}
    normalization = payload.get("normalization") or {}
    quality_flags = unique_strings(list(normalization.get("quality_flags") or []) + list(grounding.get("quality_flags") or []))
    expertise = copy.deepcopy(profile.get("expertise") or {})
    expertise["experience_level"] = normalize_seniority(expertise.get("experience_level"))

    return {
        "profile_id": id_row.get("profile_id"),
        "candidate_id": id_row.get("candidate_id"),
        "source_path": payload.get("source_path"),
        "artifact_path": payload.get("artifact_path"),
        "source_format": payload.get("source_format"),
        "status": payload.get("status"),
        "profile_kind": payload.get("profile_kind"),
        "provider_route": payload.get("provider_used"),
        "bio": copy.deepcopy(profile.get("bio") or {}),
        "expertise": expertise,
        "experiences": copy.deepcopy(profile.get("experiences") or []),
        "education": copy.deepcopy(profile.get("education") or []),
        "languages": copy.deepcopy(profile.get("languages") or []),
        "certifications": copy.deepcopy(profile.get("certifications") or []),
        "reliability_score": float(grounding.get("reliability_score") or payload.get("document_confidence_score") or 0.0),
        "hallucination_risk": grounding.get("hallucination_risk"),
        "quality_flags": quality_flags,
        "fields_nullified": list(grounding.get("fields_nullified") or []),
        "fields_nullified_count": len(grounding.get("fields_nullified") or []),
        "module2_file_path": str(path),
    }


def load_profiles_by_id(
    payloads: list[tuple[Path, dict[str, Any]]],
    id_map: list[dict[str, Any]],
    variant_map: dict[str, list[str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows_by_artifact = {row.get("artifact_path"): row for row in id_map if row.get("artifact_path")}
    rows_by_source = {row.get("source_path"): row for row in id_map if row.get("source_path")}
    profiles_by_id: dict[str, dict[str, Any]] = {}
    profile_changes: list[dict[str, Any]] = []
    aggregate_stats: Counter[str] = Counter()

    for path, payload in payloads:
        id_row = rows_by_artifact.get(payload.get("artifact_path")) or rows_by_source.get(payload.get("source_path"))
        if not id_row:
            raise ValueError(f"Profile cannot be matched to FAISS id_map: {path}")
        profile_doc = build_local_profile_doc(path, payload, id_row)
        normalized_doc, stats, changes = normalize_profile_skills(profile_doc, variant_map)
        aggregate_stats.update(stats)
        if changes:
            profile_changes.append(
                {
                    "profile_id": normalized_doc.get("profile_id"),
                    "candidate_id": normalized_doc.get("candidate_id"),
                    "module2_file_path": str(path),
                    "changes": changes,
                }
            )
        profiles_by_id[str(normalized_doc["profile_id"])] = normalized_doc

    return profiles_by_id, {
        "profiles_normalized": len(profiles_by_id),
        "profiles_with_skill_changes": len(profile_changes),
        "skill_occurrences_seen": aggregate_stats["hard_input_values"] + aggregate_stats["soft_input_values"],
        "hard_skill_occurrences_seen": aggregate_stats["hard_input_values"],
        "soft_skill_occurrences_seen": aggregate_stats["soft_input_values"],
        "skill_occurrences_normalized": aggregate_stats["hard_values_changed"] + aggregate_stats["soft_values_changed"],
        "hard_skill_occurrences_normalized": aggregate_stats["hard_values_changed"],
        "soft_skill_occurrences_normalized": aggregate_stats["soft_values_changed"],
        "skill_occurrences_expanded": aggregate_stats["hard_values_expanded"] + aggregate_stats["soft_values_expanded"],
        "duplicates_removed_after_normalization": aggregate_stats["duplicates_removed"],
        "profile_change_examples": profile_changes[:12],
    }


def write_audit(
    path: Path,
    payloads: list[tuple[Path, dict[str, Any]]],
    hard_counter: Counter[str],
    skill_files: dict[str, set[str]],
    taxonomy: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    variant_groups = {
        canonical: [variant for variant in spec["variants"] if variant in hard_counter]
        for canonical, spec in taxonomy.items()
    }
    variant_groups = {canonical: variants for canonical, variants in variant_groups.items() if len(variants) > 1}
    audit = {
        "generated_at_utc": utc_now(),
        "profiles_dir": str(PROFILES_DIR),
        "profiles_scanned": len(payloads),
        "unique_hard_skills_count": len(hard_counter),
        "total_hard_skill_occurrences": sum(hard_counter.values()),
        "hard_skills": [
            {
                "value": skill,
                "count": hard_counter[skill],
                "files": sorted(skill_files.get(skill, [])),
            }
            for skill in sorted(hard_counter, key=lambda item: item.lower())
        ],
        "variant_groups": variant_groups,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def recommend_candidates_with_normalized_profiles(
    job_profile: dict[str, Any],
    profiles_by_id: dict[str, dict[str, Any]],
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    job_seniority = normalize_seniority(job_profile.get("seniority_level"))
    index = load_faiss_index(DEFAULT_INDEX_PATH)
    id_map = load_id_map()
    model = _load_sentence_transformer(DEFAULT_SENTENCE_MODEL)
    job_text = build_job_text(job_profile)
    job_embedding = model.encode(
        [job_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    search_k = min(max(top_k * 5, top_k, 20), len(id_map))
    distances, indices = index.search(job_embedding, search_k)

    matches: list[dict[str, Any]] = []
    for faiss_rank, (score, row_index) in enumerate(zip(distances[0], indices[0]), start=1):
        if row_index < 0 or row_index >= len(id_map):
            continue
        row = dict(id_map[row_index])
        profile = profiles_by_id.get(str(row.get("profile_id")))
        if not profile:
            continue

        semantic_score = round(float(score), 4)
        score_details = score_candidate(
            job_profile=job_profile,
            candidate_profile=profile,
            score_text_similarity=semantic_score,
        )
        bio = profile.get("bio") or {}
        candidate_id = profile.get("candidate_id")
        display_name, display_name_quality, name_warning = build_display_name(
            bio.get("full_name"),
            candidate_id,
        )
        grounded_quality = enrich_grounded_quality(profile)
        candidate_seniority = normalize_seniority((profile.get("expertise") or {}).get("experience_level"))
        seniority_alignment = compute_seniority_alignment(job_seniority, candidate_seniority)
        seniority_warning = None if candidate_seniority else "missing_candidate_seniority"
        matches.append(
            {
                "candidate_id": candidate_id,
                "matched_profile_id": profile.get("profile_id"),
                "full_name": display_name,
                "display_name_quality": display_name_quality,
                "name_warning": name_warning,
                "profile_kind": profile.get("profile_kind"),
                "source_path": profile.get("source_path"),
                "provider_route": profile.get("provider_route"),
                "job_seniority": job_seniority,
                "candidate_seniority": candidate_seniority,
                "seniority_alignment": seniority_alignment,
                "seniority_warning": seniority_warning,
                "reliability_score": score_details["reliability_score"],
                "hallucination_risk": grounded_quality["hallucination_risk"],
                "faiss_rank": faiss_rank,
                "faiss_score": semantic_score,
                "cross_encoder_rank": None,
                "cross_encoder_score": None,
                "cross_encoder_score_normalized": None,
                "cross_encoder_error": None,
                "quality_flags": list(profile.get("quality_flags") or []),
                "fields_nullified_count": grounded_quality["fields_nullified_count"],
                **score_details,
                "explanation": _build_explanation(
                    full_name=display_name,
                    candidate_id=candidate_id,
                    matched_skills=score_details["matched_skills"],
                    missing_required_skills=score_details["missing_required_skills"],
                    score_text_similarity=score_details["score_text_similarity"],
                    score_experience=score_details["score_experience"],
                    hallucination_risk=grounded_quality["hallucination_risk"],
                    profile_kind=profile.get("profile_kind"),
                ),
                "profile_count": 1,
            }
        )

    unique_candidates = select_best_profile_per_candidate(group_by_candidate_id(matches))
    recommendations: list[dict[str, Any]] = []
    for rank, recommendation in enumerate(unique_candidates[:top_k], start=1):
        item = dict(recommendation)
        item["rank"] = rank
        recommendations.append(item)
    return recommendations


def load_frozen_recommendations(path: Path = FROZEN_REPORT_PATH) -> list[dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return list(((report.get("results") or [{}])[0]).get("recommendations") or [])


def compare_to_frozen(frozen: list[dict[str, Any]], normalized: list[dict[str, Any]]) -> dict[str, Any]:
    frozen_by_candidate = {str(item.get("candidate_id")): item for item in frozen}
    normalized_by_candidate = {str(item.get("candidate_id")): item for item in normalized}
    candidate_ids = sorted(
        set(frozen_by_candidate) | set(normalized_by_candidate),
        key=lambda candidate_id: (
            normalized_by_candidate.get(candidate_id, {}).get("rank", 999),
            frozen_by_candidate.get(candidate_id, {}).get("rank", 999),
            candidate_id,
        ),
    )

    rank_changes: list[dict[str, Any]] = []
    coverage_changes: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        before = frozen_by_candidate.get(candidate_id)
        after = normalized_by_candidate.get(candidate_id)
        before_rank = before.get("rank") if before else None
        after_rank = after.get("rank") if after else None
        before_coverage = before.get("must_have_coverage") if before else None
        after_coverage = after.get("must_have_coverage") if after else None
        coverage_delta = None
        if isinstance(before_coverage, (int, float)) and isinstance(after_coverage, (int, float)):
            coverage_delta = round(float(after_coverage) - float(before_coverage), 4)
        if before_rank != after_rank:
            rank_changes.append(
                {
                    "candidate_id": candidate_id,
                    "full_name_before": before.get("full_name") if before else None,
                    "full_name_after": after.get("full_name") if after else None,
                    "rank_before": before_rank,
                    "rank_after": after_rank,
                    "rank_delta": None if before_rank is None or after_rank is None else int(after_rank) - int(before_rank),
                }
            )
        if coverage_delta not in (None, 0.0):
            coverage_changes.append(
                {
                    "candidate_id": candidate_id,
                    "full_name_before": before.get("full_name") if before else None,
                    "full_name_after": after.get("full_name") if after else None,
                    "must_have_coverage_before": before_coverage,
                    "must_have_coverage_after": after_coverage,
                    "delta": coverage_delta,
                    "matched_skills_before": before.get("matched_skills") if before else None,
                    "matched_skills_after": after.get("matched_skills") if after else None,
                    "missing_required_skills_before": before.get("missing_required_skills") if before else None,
                    "missing_required_skills_after": after.get("missing_required_skills") if after else None,
                }
            )

    return {
        "frozen_report_path": str(FROZEN_REPORT_PATH),
        "rank_changes": rank_changes,
        "must_have_coverage_changes": coverage_changes,
        "top10_before": summarize_top10(frozen),
        "top10_after": summarize_top10(normalized),
    }


def summarize_top10(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": item.get("rank"),
            "candidate_id": item.get("candidate_id"),
            "matched_profile_id": item.get("matched_profile_id"),
            "full_name": item.get("full_name"),
            "final_score": item.get("final_score"),
            "job_seniority": item.get("job_seniority"),
            "candidate_seniority": item.get("candidate_seniority"),
            "seniority_alignment": item.get("seniority_alignment"),
            "must_have_coverage": item.get("must_have_coverage"),
            "matched_skills": item.get("matched_skills"),
            "missing_required_skills": item.get("missing_required_skills"),
            "scores": {
                "final_score": item.get("final_score"),
                "base_score_before_penalty": item.get("base_score_before_penalty"),
                "score_text_similarity": item.get("score_text_similarity"),
                "score_skills": item.get("score_skills"),
                "score_experience": item.get("score_experience"),
                "score_profile_quality": item.get("score_profile_quality"),
                "score_grounded_quality": item.get("score_grounded_quality"),
            },
        }
        for item in items
    ]


def write_report(report: dict[str, Any], path: Path = NORMALIZED_REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_matching_report(
    *,
    job_profile_path: Path = JOB_PROFILE_PATH,
    output_report_path: Path = NORMALIZED_REPORT_PATH,
    job_id: str | None = None,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    start = time.perf_counter()
    payloads = load_profile_payloads()
    hard_counter, soft_counter, skill_files = collect_skill_counters(payloads)
    taxonomy = build_taxonomy_from_audit(hard_counter)
    write_taxonomy(TAXONOMY_PATH, taxonomy)
    audit = write_audit(AUDIT_PATH, payloads, hard_counter, skill_files, taxonomy)

    variant_map = build_variant_map(load_taxonomy(TAXONOMY_PATH))
    id_map = load_id_map()
    profiles_by_id, normalization_stats = load_profiles_by_id(payloads, id_map, variant_map)
    job_profile = json.loads(job_profile_path.read_text(encoding="utf-8"))
    resolved_job_id = job_id or job_profile.get("job_id") or job_profile_path.stem
    recommendations = recommend_candidates_with_normalized_profiles(job_profile, profiles_by_id, top_k=top_k)
    if job_profile_path == JOB_PROFILE_PATH and output_report_path == NORMALIZED_REPORT_PATH and top_k == TOP_K:
        frozen_recommendations = load_frozen_recommendations()
        comparison = compare_to_frozen(frozen_recommendations, recommendations)
    else:
        comparison = {
            "skipped": True,
            "reason": "Frozen baseline comparison is only computed for the default historical backend run.",
        }
    duration_seconds = round(time.perf_counter() - start, 4)

    all_normalized_unique = set()
    for _, payload in payloads:
        expertise = ((payload.get("profile") or {}).get("expertise") or {})
        for field in ("hard_skills", "soft_skills"):
            for raw_skill in expertise.get(field) or []:
                cleaned = clean_skill(raw_skill)
                if cleaned and variant_map.get(skill_key(cleaned), [cleaned]) != [cleaned]:
                    all_normalized_unique.add(cleaned)

    report = {
        "generated_at_utc": utc_now(),
        "job_id": resolved_job_id,
        "job_profile_file": str(job_profile_path),
        "job_title": job_profile.get("job_title"),
        "top_k": top_k,
        "sentence_transformer_model": DEFAULT_SENTENCE_MODEL,
        "faiss_index_path": str(DEFAULT_INDEX_PATH),
        "faiss_index_report_path": str(DEFAULT_REPORT_PATH),
        "metadata": {
            "retrieval_engine": "faiss",
            "reranker": "none",
            "normalization_mode": "skills_taxonomy_in_memory",
            "original_profiles_overwritten": False,
            "frozen_report_modified": False,
        },
        "execution_time_seconds": duration_seconds,
        "normalization": {
            "audit_path": str(AUDIT_PATH),
            "taxonomy_path": str(TAXONOMY_PATH),
            "profiles_scanned": len(payloads),
            "unique_hard_skills_count": audit["unique_hard_skills_count"],
            "total_hard_skill_occurrences": audit["total_hard_skill_occurrences"],
            "unique_soft_skills_count": len(soft_counter),
            "total_soft_skill_occurrences": sum(soft_counter.values()),
            "unique_skill_variants_normalized": len(all_normalized_unique),
            "unique_skill_variants_normalized_examples": sorted(all_normalized_unique, key=lambda item: item.lower())[:60],
            **normalization_stats,
        },
        "results": [
            {
                "job_id": resolved_job_id,
                "job_profile_file": str(job_profile_path),
                "job_title": job_profile.get("job_title"),
                "top_k": top_k,
                "recommendations_count": len(recommendations),
                "recommendations": recommendations,
            }
        ],
        "top_10_candidates": summarize_top10(recommendations),
        "comparison_to_frozen": comparison,
    }
    write_report(report, output_report_path)

    summary = {
        "files_created": [str(AUDIT_PATH), str(TAXONOMY_PATH), str(output_report_path)],
        "normalization_stats": report["normalization"],
        "rank_changes": comparison.get("rank_changes", []),
        "must_have_coverage_changes": comparison.get("must_have_coverage_changes", []),
        "top10_after": comparison.get("top10_after", summarize_top10(recommendations)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Matching V3 normalized for one job profile.")
    parser.add_argument("--job-profile", type=Path, default=JOB_PROFILE_PATH)
    parser.add_argument("--output-report", type=Path, default=NORMALIZED_REPORT_PATH)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_matching_report(
        job_profile_path=args.job_profile,
        output_report_path=args.output_report,
        job_id=args.job_id,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
