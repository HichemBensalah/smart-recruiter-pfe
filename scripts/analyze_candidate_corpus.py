from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.common.seniority import normalize_seniority


IMPORTANT_KEYWORDS = [
    "Python",
    "FastAPI",
    "Django",
    "Flask",
    "JavaScript",
    "TypeScript",
    "React",
    "Next.js",
    "Node.js",
    "Express.js",
    "MongoDB",
    "PostgreSQL",
    "SQL",
    "Pandas",
    "Spark",
    "Airflow",
    "Docker",
    "Kubernetes",
    "Linux",
    "Git",
    "AWS",
    "Machine Learning",
    "NLP",
    "Power BI",
    "Excel",
]

DOMAIN_KEYWORDS = {
    "backend": ["python", "fastapi", "django", "flask", "java", "spring", "api", "rest", "backend"],
    "frontend": ["react", "nextjs", "next.js", "typescript", "javascript", "html", "css", "frontend"],
    "fullstack": ["fullstack", "full stack", "react", "nodejs", "node.js", "mongodb", "expressjs", "express.js"],
    "data": ["python", "sql", "etl", "pipeline", "pandas", "spark", "airflow", "data"],
    "devops": ["docker", "kubernetes", "linux", "git", "aws", "ci/cd", "devops", "terraform"],
    "machine_learning": ["machine learning", "ml", "scikit", "sklearn", "nlp", "tensorflow", "pytorch"],
    "mobile": ["android", "ios", "flutter", "react native", "kotlin", "swift"],
    "business_intelligence": ["power bi", "tableau", "excel", "business intelligence", "dashboard", "reporting"],
}

RECOMMENDED_JOB_TEMPLATES = [
    {
        "job_id": "backend_python_django_postgresql",
        "job_title": "Backend Python Django Developer",
        "domain": "backend",
        "seniority_level": "mid_level",
        "years_experience_required": 3,
        "required_skills": ["Python", "Django", "SQL", "PostgreSQL"],
        "nice_to_have_skills": ["REST API", "Docker", "Git", "Linux"],
    },
    {
        "job_id": "data_analyst_python_sql_powerbi",
        "job_title": "Data Analyst",
        "domain": "business_intelligence",
        "seniority_level": "mid_level",
        "years_experience_required": 2,
        "required_skills": ["SQL", "Python", "Excel", "Power BI"],
        "nice_to_have_skills": ["Pandas", "Dashboard", "Reporting", "Machine Learning"],
    },
    {
        "job_id": "machine_learning_python_nlp",
        "job_title": "Machine Learning Engineer",
        "domain": "machine_learning",
        "seniority_level": "mid_level",
        "years_experience_required": 3,
        "required_skills": ["Python", "Machine Learning", "NLP", "SQL"],
        "nice_to_have_skills": ["Pandas", "Scikit-learn", "TensorFlow", "Docker"],
    },
    {
        "job_id": "backend_python_fastapi_mongodb_aligned",
        "job_title": "Backend Python FastAPI Developer",
        "domain": "backend",
        "seniority_level": "mid_level",
        "years_experience_required": 3,
        "required_skills": ["Python", "FastAPI", "MongoDB", "Git"],
        "nice_to_have_skills": ["Docker", "SQL", "Linux", "REST API"],
    },
    {
        "job_id": "devops_linux_docker_git",
        "job_title": "DevOps Engineer",
        "domain": "devops",
        "seniority_level": "mid_level",
        "years_experience_required": 3,
        "required_skills": ["Linux", "Docker", "Git", "AWS"],
        "nice_to_have_skills": ["Kubernetes", "CI/CD", "Terraform", "Monitoring"],
    },
    {
        "job_id": "data_engineer_python_sql_etl",
        "job_title": "Data Engineer",
        "domain": "data",
        "seniority_level": "mid_level",
        "years_experience_required": 3,
        "required_skills": ["Python", "SQL", "ETL", "Data Pipeline"],
        "nice_to_have_skills": ["Pandas", "Spark", "Airflow", "Docker"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze candidate corpus alignment before pseudo-labeling/training.")
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--job-profiles-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_token(value: str) -> str:
    text = value.lower().strip()
    replacements = {
        "next.js": "nextjs",
        "next js": "nextjs",
        "node.js": "nodejs",
        "node js": "nodejs",
        "express.js": "expressjs",
        "express js": "expressjs",
        "postgresql": "postgresql",
        "postgre sql": "postgresql",
        "powerbi": "power bi",
        "power-bi": "power bi",
        "ci cd": "ci/cd",
    }
    text = text.replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    return replacements.get(text, text)


def canonical_skill(value: str) -> str:
    normalized = normalize_token(value)
    aliases = {
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "mongo": "mongodb",
        "postgres": "postgresql",
        "sklearn": "scikit-learn",
        "scikit learn": "scikit-learn",
        "ms excel": "excel",
        "microsoft excel": "excel",
        "microsoft office excel": "excel",
    }
    return aliases.get(normalized, normalized)


def display_skill(skill: str) -> str:
    mapping = {
        "nextjs": "Next.js",
        "nodejs": "Node.js",
        "expressjs": "Express.js",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "mongodb": "MongoDB",
        "postgresql": "PostgreSQL",
        "sql": "SQL",
        "python": "Python",
        "fastapi": "FastAPI",
        "django": "Django",
        "flask": "Flask",
        "pandas": "Pandas",
        "spark": "Spark",
        "airflow": "Airflow",
        "docker": "Docker",
        "kubernetes": "Kubernetes",
        "linux": "Linux",
        "git": "Git",
        "aws": "AWS",
        "machine learning": "Machine Learning",
        "nlp": "NLP",
        "power bi": "Power BI",
        "excel": "Excel",
    }
    return mapping.get(skill, skill.title())


def profile_text(profile: dict[str, Any]) -> str:
    parts = [
        profile.get("normalization", {}).get("cleaned_markdown"),
        profile.get("profile", {}).get("expertise", {}).get("summary"),
    ]
    return " ".join(str(part) for part in parts if part).lower()


def extract_candidate(profile_path: Path) -> dict[str, Any]:
    raw = load_json(profile_path)
    profile = raw.get("profile", {})
    expertise = profile.get("expertise", {})
    hard_skills = [str(skill) for skill in expertise.get("hard_skills", []) if skill]
    soft_skills = [str(skill) for skill in expertise.get("soft_skills", []) if skill]
    skill_set = {canonical_skill(skill) for skill in hard_skills + soft_skills if str(skill).strip()}
    text = profile_text(raw)

    keyword_hits = set()
    for keyword in IMPORTANT_KEYWORDS:
        normalized = canonical_skill(keyword)
        variants = {keyword.lower(), normalized}
        if any(variant in text for variant in variants) or normalized in skill_set:
            keyword_hits.add(normalized)

    all_signals = skill_set | keyword_hits
    domains = infer_domains(all_signals, text)
    seniority = normalize_seniority(expertise.get("experience_level")) or "unknown"
    years_experience = expertise.get("years_experience")
    try:
        years = float(years_experience) if years_experience is not None else None
    except (TypeError, ValueError):
        years = None

    return {
        "profile_path": str(profile_path),
        "profile_id": raw.get("profile_id") or profile_path.stem,
        "skills": sorted(skill_set),
        "keyword_hits": sorted(keyword_hits),
        "skill_signals": sorted(all_signals),
        "domains": domains,
        "seniority": seniority,
        "years_experience": years,
        "reliability_score": raw.get("grounding", {}).get("reliability_score"),
        "hallucination_risk": raw.get("grounding", {}).get("hallucination_risk"),
        "skills_empty": not bool(skill_set),
        "seniority_missing": seniority == "unknown",
        "years_experience_missing": years is None,
    }


def infer_domains(skill_signals: set[str], text: str) -> list[str]:
    domains = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            normalized = canonical_skill(keyword)
            if normalized in skill_signals or normalize_token(keyword) in text:
                score += 1
        if score:
            domains.append(domain)
    return sorted(domains) or ["unknown"]


def years_bucket(years: float | None) -> str:
    if years is None:
        return "unknown"
    if years < 1:
        return "0-1"
    if years < 3:
        return "1-3"
    if years < 5:
        return "3-5"
    return "5+"


def count_candidates_with_skill(candidates: list[dict[str, Any]], skill: str) -> int:
    canonical = canonical_skill(skill)
    return sum(1 for candidate in candidates if canonical in set(candidate["skill_signals"]))


def required_coverage(candidate: dict[str, Any], required_skills: list[str]) -> float:
    if not required_skills:
        return 1.0
    signals = set(candidate["skill_signals"])
    matched = sum(1 for skill in required_skills if canonical_skill(skill) in signals)
    return matched / len(required_skills)


def decision_from_alignment(score: float, candidates_80: int) -> str:
    if score >= 0.35 and candidates_80 >= 3:
        return "keep"
    if score >= 0.15:
        return "adjust"
    return "replace"


def analyze_job_alignment(candidates: list[dict[str, Any]], job_profile: dict[str, Any]) -> dict[str, Any]:
    required_skills = [str(skill) for skill in job_profile.get("required_skills", [])]
    coverages = [required_coverage(candidate, required_skills) for candidate in candidates]
    any_count = sum(1 for value in coverages if value > 0)
    two_count = sum(1 for candidate in candidates if sum(1 for skill in required_skills if canonical_skill(skill) in set(candidate["skill_signals"])) >= 2)
    fifty_count = sum(1 for value in coverages if value >= 0.5)
    eighty_count = sum(1 for value in coverages if value >= 0.8)
    total = len(candidates) or 1
    alignment_score = round(((any_count / total) * 0.2) + ((fifty_count / total) * 0.4) + ((eighty_count / total) * 0.4), 4)
    return {
        "job_id": job_profile.get("job_id") or job_profile.get("job_title"),
        "job_title": job_profile.get("job_title"),
        "domain": job_profile.get("domain"),
        "seniority_level": job_profile.get("seniority_level"),
        "required_skills": required_skills,
        "required_skill_frequencies": {
            skill: {
                "count": count_candidates_with_skill(candidates, skill),
                "percentage": round((count_candidates_with_skill(candidates, skill) / total) * 100, 2),
            }
            for skill in required_skills
        },
        "candidates_with_any_required_skill": any_count,
        "candidates_with_at_least_2_required_skills": two_count,
        "candidates_with_50_percent_required_skills": fifty_count,
        "candidates_with_80_percent_required_skills": eighty_count,
        "corpus_alignment_score": alignment_score,
        "decision": decision_from_alignment(alignment_score, eighty_count),
    }


def load_job_profiles(path: Path) -> list[dict[str, Any]]:
    jobs = []
    for file_path in sorted(path.glob("*.json")):
        if file_path.name == "job_profile_builder_report.json":
            continue
        job = load_json(file_path)
        if "required_skills" in job:
            jobs.append(job)
    return jobs


def build_recommendations(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recommendations = []
    for template in RECOMMENDED_JOB_TEMPLATES:
        alignment = analyze_job_alignment(candidates, template)
        if alignment["candidates_with_50_percent_required_skills"] == 0:
            continue
        item = dict(template)
        item["justification"] = (
            f"{alignment['candidates_with_any_required_skill']} candidats ont au moins une compétence requise, "
            f"{alignment['candidates_with_50_percent_required_skills']} couvrent au moins 50% des compétences, "
            f"{alignment['candidates_with_80_percent_required_skills']} couvrent au moins 80%."
        )
        item["expected_candidate_pool_size"] = alignment["candidates_with_50_percent_required_skills"]
        item["corpus_alignment_score"] = alignment["corpus_alignment_score"]
        recommendations.append(item)
    return sorted(recommendations, key=lambda row: row["expected_candidate_pool_size"], reverse=True)


def summarize_corpus(candidates: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(candidates)
    skill_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    seniority_counter: Counter[str] = Counter()
    years_values: list[float] = []
    hallucination_counter: Counter[str] = Counter()
    reliability_values: list[float] = []

    for candidate in candidates:
        skill_counter.update(candidate["skills"])
        keyword_counter.update(candidate["keyword_hits"])
        domain_counter.update(candidate["domains"])
        seniority_counter.update([candidate["seniority"]])
        if candidate["years_experience"] is not None:
            years_values.append(candidate["years_experience"])
        if candidate["reliability_score"] is not None:
            reliability_values.append(float(candidate["reliability_score"]))
        if candidate["hallucination_risk"] is not None:
            hallucination_counter.update([str(candidate["hallucination_risk"]).lower()])

    job_alignment = [analyze_job_alignment(candidates, job) for job in jobs]
    frontend_diagnosis = {
        keyword: {
            "count": count_candidates_with_skill(candidates, keyword),
            "percentage": round((count_candidates_with_skill(candidates, keyword) / (total or 1)) * 100, 2),
        }
        for keyword in ["React", "Next.js", "TypeScript", "Node.js", "JavaScript", "MongoDB"]
    }

    return {
        "generated_at_utc": utc_now(),
        "total_profiles": total,
        "top_skills": [
            {"skill": display_skill(skill), "count": count, "percentage": round((count / (total or 1)) * 100, 2)}
            for skill, count in skill_counter.most_common(50)
        ],
        "important_keyword_counts": {
            keyword: {
                "count": keyword_counter[canonical_skill(keyword)],
                "percentage": round((keyword_counter[canonical_skill(keyword)] / (total or 1)) * 100, 2),
            }
            for keyword in IMPORTANT_KEYWORDS
        },
        "domain_distribution": dict(sorted(domain_counter.items())),
        "seniority_distribution": {level: seniority_counter[level] for level in ["junior", "mid_level", "senior", "lead", "principal", "unknown"]},
        "experience_statistics": {
            "min_years_experience": min(years_values) if years_values else None,
            "max_years_experience": max(years_values) if years_values else None,
            "mean_years_experience": round(statistics.mean(years_values), 2) if years_values else None,
            "median_years_experience": round(statistics.median(years_values), 2) if years_values else None,
            "bucket_distribution": dict(sorted(Counter(years_bucket(candidate["years_experience"]) for candidate in candidates).items())),
        },
        "profile_quality": {
            "mean_reliability_score": round(statistics.mean(reliability_values), 4) if reliability_values else None,
            "hallucination_risk_distribution": dict(sorted(hallucination_counter.items())),
            "profiles_with_empty_skills": sum(1 for candidate in candidates if candidate["skills_empty"]),
            "profiles_with_missing_seniority": sum(1 for candidate in candidates if candidate["seniority_missing"]),
            "profiles_with_missing_years_experience": sum(1 for candidate in candidates if candidate["years_experience_missing"]),
        },
        "job_alignment": job_alignment,
        "frontend_fullstack_diagnosis": {
            "technology_counts": frontend_diagnosis,
            "frontend_react_nextjs_represented": frontend_diagnosis["React"]["count"] > 0 and frontend_diagnosis["Next.js"]["count"] > 0,
            "fullstack_react_node_mongodb_represented": (
                frontend_diagnosis["React"]["count"] > 0
                and frontend_diagnosis["Node.js"]["count"] > 0
                and frontend_diagnosis["MongoDB"]["count"] > 0
            ),
        },
        "recommended_job_profiles": build_recommendations(candidates),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Candidate Corpus Analysis",
        "",
        f"- Total profiles: {report['total_profiles']}",
        "",
        "## Top 20 Skills",
    ]
    for item in report["top_skills"][:20]:
        lines.append(f"- {item['skill']}: {item['count']} ({item['percentage']}%)")
    lines.extend(["", "## Domain Distribution"])
    for domain, count in report["domain_distribution"].items():
        lines.append(f"- {domain}: {count}")
    lines.extend(["", "## Seniority Distribution"])
    for level, count in report["seniority_distribution"].items():
        lines.append(f"- {level}: {count}")
    lines.extend(["", "## Current Job Alignment"])
    for job in report["job_alignment"]:
        lines.append(
            f"- {job['job_id']}: decision={job['decision']}, score={job['corpus_alignment_score']}, "
            f"any={job['candidates_with_any_required_skill']}, 50%={job['candidates_with_50_percent_required_skills']}, "
            f"80%={job['candidates_with_80_percent_required_skills']}"
        )
    lines.extend(["", "## Frontend / Fullstack Diagnosis"])
    for tech, stats in report["frontend_fullstack_diagnosis"]["technology_counts"].items():
        lines.append(f"- {tech}: {stats['count']} ({stats['percentage']}%)")
    lines.extend(["", "## Recommended Job Profiles"])
    for job in report["recommended_job_profiles"]:
        lines.append(
            f"- {job['job_id']} ({job['domain']}): expected_pool={job['expected_candidate_pool_size']}, "
            f"required={', '.join(job['required_skills'])}"
        )
    lines.extend([
        "",
        "## Methodological Note",
        "This report does not train a model, does not use SMOTE, and does not modify datasets or job profiles.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    profile_paths = sorted(args.profiles_dir.glob("*.json"))
    candidates = [extract_candidate(path) for path in profile_paths]
    jobs = load_job_profiles(args.job_profiles_dir)
    report = summarize_corpus(candidates, jobs)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "total_profiles": report["total_profiles"],
        "top_skill": report["top_skills"][0] if report["top_skills"] else None,
        "jobs_analyzed": len(report["job_alignment"]),
        "recommended_jobs": len(report["recommended_job_profiles"]),
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
