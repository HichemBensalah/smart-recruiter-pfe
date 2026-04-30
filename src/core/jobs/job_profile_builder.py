from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .job_profile_schema import CanonicalJobProfile, JobMetadata

OUTPUT_DIR = Path("data/job_profiles")
REPORT_PATH = OUTPUT_DIR / "job_profile_builder_report.json"

KNOWN_SKILLS = [
    "Python",
    "FastAPI",
    "MongoDB",
    "PostgreSQL",
    "SQL",
    "Machine Learning",
    "Deep Learning",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "React",
    "TypeScript",
    "JavaScript",
    "HTML",
    "CSS",
    "Docker",
    "Kubernetes",
    "CI/CD",
    "GitHub Actions",
    "Terraform",
    "AWS",
    "Azure",
    "GCP",
    "Spark",
    "Hadoop",
    "Airflow",
    "ETL",
    "Data Modeling",
    "REST API",
    "Linux",
]
SKILL_PATTERNS = {skill: re.compile(rf"(?<!\w){re.escape(skill)}(?!\w)", re.I) for skill in KNOWN_SKILLS}
NICE_TO_HAVE_CUES = ("nice to have", "plus", "bonus", "preferred", "would be a plus", "good to have")
RESPONSIBILITY_STARTERS = (
    "you will",
    "you'll",
    "responsibilities",
    "mission",
    "your role",
    "in this role",
    "the role involves",
    "you are responsible for",
)
SENIORITY_PATTERNS = {
    "junior": re.compile(r"\b(junior|entry level|graduate|intern)\b", re.I),
    "mid": re.compile(r"\b(mid|intermediate)\b", re.I),
    "senior": re.compile(r"\b(senior|sr\.?)\b", re.I),
    "lead": re.compile(r"\b(lead|staff)\b", re.I),
    "principal": re.compile(r"\b(principal|head of)\b", re.I),
}
REMOTE_PATTERNS = {
    "remote": re.compile(r"\b(remote|work from home|fully remote)\b", re.I),
    "hybrid": re.compile(r"\b(hybrid)\b", re.I),
    "on_site": re.compile(r"\b(on[- ]site|onsite)\b", re.I),
}
CONTRACT_PATTERNS = {
    "full_time": re.compile(r"\b(full[- ]time|permanent)\b", re.I),
    "part_time": re.compile(r"\b(part[- ]time)\b", re.I),
    "contract": re.compile(r"\b(contract|contractor|fixed[- ]term)\b", re.I),
    "internship": re.compile(r"\b(internship|intern)\b", re.I),
    "freelance": re.compile(r"\b(freelance)\b", re.I),
}
LANGUAGE_PATTERNS = {
    "English": re.compile(r"\benglish\b", re.I),
    "French": re.compile(r"\bfrench\b", re.I),
    "Arabic": re.compile(r"\barabic\b", re.I),
    "German": re.compile(r"\bgerman\b", re.I),
    "Spanish": re.compile(r"\bspanish\b", re.I),
}
DOMAIN_PATTERNS = {
    "backend_engineering": re.compile(r"\b(backend|api|microservices)\b", re.I),
    "data_science": re.compile(r"\b(data science|machine learning|ml model)\b", re.I),
    "frontend_engineering": re.compile(r"\b(frontend|react|ui|ux)\b", re.I),
    "devops": re.compile(r"\b(devops|ci/cd|infrastructure|platform engineering)\b", re.I),
    "data_engineering": re.compile(r"\b(data engineer|etl|spark|hadoop|data pipeline)\b", re.I),
}
JOB_TITLE_LINE_HINTS = re.compile(
    r"\b(engineer|developer|scientist|analyst|manager|architect|specialist|intern)\b",
    re.I,
)
YEARS_PATTERNS = [
    re.compile(r"(\d+)\+?\s+years? of experience", re.I),
    re.compile(r"minimum of (\d+)\s+years?", re.I),
    re.compile(r"at least (\d+)\s+years?", re.I),
]
LOCATION_PATTERNS = [
    re.compile(r"\bbased in ([A-Z][A-Za-z .-]+(?:,\s*[A-Z][A-Za-z .-]+)?)"),
    re.compile(r"\blocation[:\s]+([A-Z][A-Za-z .-]+(?:,\s*[A-Z][A-Za-z .-]+)?)", re.I),
    re.compile(r"\bin ([A-Z][A-Za-z .-]+(?:,\s*[A-Z][A-Za-z .-]+)?)"),
]

TEST_JOB_DESCRIPTIONS = [
    {
        "slug": "backend_python_fastapi_mongodb",
        "raw_job_description": (
            "Backend Python Engineer\n"
            "We are hiring a backend engineer to design and maintain REST APIs for our recruiting platform. "
            "You will build services with Python, FastAPI and MongoDB, collaborate with product and ensure API performance. "
            "Requirements: 3+ years of experience, Python, FastAPI, MongoDB, REST API, Docker. "
            "Nice to have: AWS and CI/CD. Full-time hybrid role based in Tunis. English required."
        ),
    },
    {
        "slug": "data_science_machine_learning_python",
        "raw_job_description": (
            "Senior Data Scientist\n"
            "Our AI team is looking for a senior data scientist to develop machine learning models and analyze large datasets. "
            "Responsibilities include experimentation, feature engineering, model evaluation and stakeholder communication. "
            "Must have Python, Pandas, Scikit-learn, SQL and Machine Learning. "
            "Bonus: Deep Learning and PyTorch. Remote position. 5 years of experience preferred. English is required."
        ),
    },
    {
        "slug": "frontend_react",
        "raw_job_description": (
            "Frontend React Developer wanted for a web product team. "
            "You will implement user interfaces, improve usability, collaborate with designers and maintain component quality. "
            "Required skills: React, TypeScript, JavaScript, HTML, CSS. "
            "Nice to have: FastAPI exposure. On-site in Sfax. Mid-level preferred."
        ),
    },
    {
        "slug": "devops_docker_cicd",
        "raw_job_description": (
            "DevOps Engineer\n"
            "Mission: automate deployments, improve infrastructure reliability and maintain CI/CD pipelines. "
            "The role involves Docker, Kubernetes, Terraform, Linux and GitHub Actions. "
            "At least 4 years of experience. Contract position, hybrid in Paris. French and English are a plus."
        ),
    },
    {
        "slug": "data_engineer_sql_hadoop_spark",
        "raw_job_description": (
            "Data Engineer\n"
            "Join our analytics platform team to build ETL workflows and scalable data pipelines. "
            "You are responsible for Spark jobs, Hadoop ecosystem integration, SQL optimization and Airflow orchestration. "
            "Required: SQL, Spark, Hadoop, Airflow, Data Modeling. "
            "Preferred: Python and AWS. Senior profile. Full-time remote."
        ),
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_space(text: str) -> str:
    return " ".join(text.split()).strip()


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = normalize_space(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [normalize_space(chunk) for chunk in chunks if normalize_space(chunk)]


def extract_job_title(text: str) -> str | None:
    lines = [normalize_space(line) for line in text.splitlines() if normalize_space(line)]
    for line in lines[:3]:
        if JOB_TITLE_LINE_HINTS.search(line) and len(line.split()) <= 8:
            return line

    prefix_patterns = [
        re.compile(r"\b([A-Z][A-Za-z/+ -]{0,40}(?:Engineer|Developer|Scientist|Analyst|Architect))\b"),
        re.compile(r"\b([A-Z][A-Za-z/+ -]{0,40}React Developer)\b"),
    ]
    for pattern in prefix_patterns:
        match = pattern.search(text)
        if match:
            return normalize_space(match.group(1))

    title_patterns = [
        re.compile(r"\b((?:Senior|Junior|Lead|Principal|Staff|Mid[- ]Level)?\s*(?:Backend|Frontend|Full[- ]Stack|Data|DevOps)?\s*(?:Engineer|Developer|Scientist|Analyst|Architect))\b", re.I),
        re.compile(r"\b(Data Engineer|Data Scientist|Frontend React Developer|Backend Python Engineer|DevOps Engineer)\b", re.I),
    ]
    for pattern in title_patterns:
        match = pattern.search(text)
        if match:
            return normalize_space(match.group(1))
    return None


def detect_seniority(text: str, job_title: str | None) -> str | None:
    search_space = f"{job_title or ''}\n{text}"
    for label in ("principal", "lead", "senior", "mid", "junior"):
        if SENIORITY_PATTERNS[label].search(search_space):
            return label
    return None


def detect_years_experience(text: str) -> float | None:
    for pattern in YEARS_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def extract_skills(text: str) -> tuple[list[str], list[str]]:
    required: list[str] = []
    nice_to_have: list[str] = []
    sentences = split_sentences(text)

    for sentence in sentences:
        lowered = sentence.lower()
        target = nice_to_have if any(cue in lowered for cue in NICE_TO_HAVE_CUES) else required
        for skill, pattern in SKILL_PATTERNS.items():
            if pattern.search(sentence):
                target.append(skill)

    return unique_strings(required), unique_strings(nice_to_have)


def extract_responsibilities(text: str) -> list[str]:
    responsibilities: list[str] = []
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if any(starter in lowered for starter in RESPONSIBILITY_STARTERS):
            responsibilities.append(sentence)
            continue
        if sentence.startswith(("Build ", "Design ", "Develop ", "Maintain ", "Collaborate ", "Implement ", "Automate ")):
            responsibilities.append(sentence)

    return unique_strings(responsibilities[:6])


def detect_domain(text: str) -> str | None:
    counts: Counter[str] = Counter()
    for label, pattern in DOMAIN_PATTERNS.items():
        counts[label] = len(pattern.findall(text))
    if not counts:
        return None
    best, score = counts.most_common(1)[0]
    return best if score > 0 else None


def detect_location(text: str) -> str | None:
    for pattern in LOCATION_PATTERNS:
        match = pattern.search(text)
        if match:
            return normalize_space(match.group(1).rstrip("."))
    return None


def detect_remote_policy(text: str) -> str | None:
    for label in ("hybrid", "remote", "on_site"):
        if REMOTE_PATTERNS[label].search(text):
            return label
    return None


def detect_contract_type(text: str) -> str | None:
    for label in ("internship", "contract", "freelance", "part_time", "full_time"):
        if CONTRACT_PATTERNS[label].search(text):
            return label
    return None


def detect_languages(text: str) -> list[str]:
    found = [label for label, pattern in LANGUAGE_PATTERNS.items() if pattern.search(text)]
    return unique_strings(found)


def estimate_confidence(job_profile_data: dict[str, Any], warnings: list[str]) -> float:
    score = 0.35
    if job_profile_data.get("job_title"):
        score += 0.15
    if job_profile_data.get("required_skills"):
        score += min(0.2, len(job_profile_data["required_skills"]) * 0.03)
    if job_profile_data.get("responsibilities"):
        score += 0.1
    if job_profile_data.get("seniority_level"):
        score += 0.05
    if job_profile_data.get("years_experience_required") is not None:
        score += 0.05
    if job_profile_data.get("location"):
        score += 0.03
    if job_profile_data.get("remote_policy"):
        score += 0.03
    score -= min(0.2, len(warnings) * 0.04)
    return round(max(0.0, min(score, 0.99)), 4)


def build_job_profile(raw_job_description: str) -> CanonicalJobProfile:
    raw_text = normalize_space(raw_job_description)
    if not raw_text:
        raise ValueError("raw_job_description must not be empty")

    warnings: list[str] = []
    job_title = extract_job_title(raw_job_description)
    if not job_title:
        job_title = "Unknown Role"
        warnings.append("missing_job_title_detected")

    required_skills, nice_to_have_skills = extract_skills(raw_job_description)
    responsibilities = extract_responsibilities(raw_job_description)
    seniority_level = detect_seniority(raw_job_description, job_title)
    years_experience_required = detect_years_experience(raw_job_description)
    domain = detect_domain(raw_job_description)
    location = detect_location(raw_job_description)
    remote_policy = detect_remote_policy(raw_job_description)
    contract_type = detect_contract_type(raw_job_description)
    language_requirements = detect_languages(raw_job_description)

    if not required_skills:
        warnings.append("missing_required_skills_signal")
    if not responsibilities:
        warnings.append("missing_responsibilities_signal")
    if seniority_level is None:
        warnings.append("missing_seniority_signal")
    if years_experience_required is None:
        warnings.append("missing_years_experience_signal")
    if location is None and remote_policy is None:
        warnings.append("missing_location_or_remote_signal")
    if len(raw_text.split()) < 25:
        warnings.append("low_information_job_description")

    payload = {
        "job_title": job_title,
        "seniority_level": seniority_level,
        "years_experience_required": years_experience_required,
        "required_skills": required_skills,
        "nice_to_have_skills": nice_to_have_skills,
        "responsibilities": responsibilities,
        "domain": domain,
        "location": location,
        "language_requirements": language_requirements,
        "contract_type": contract_type,
        "remote_policy": remote_policy,
        "raw_job_description": raw_job_description,
    }
    metadata = JobMetadata(
        extraction_date=utc_now(),
        parser_route="rule_based_v1",
        model_used=None,
        confidence_score=estimate_confidence(payload, warnings),
        warnings=unique_strings(warnings),
    )
    return CanonicalJobProfile(**payload, metadata=metadata)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "job_profile"


def run_test_jobs() -> dict[str, Any]:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_files: list[str] = []
    extracted_skill_examples: list[dict[str, Any]] = []
    warnings_total = 0
    success = 0
    failed = 0

    for job in TEST_JOB_DESCRIPTIONS:
        slug = job["slug"]
        output_path = OUTPUT_DIR / f"{slug}.json"
        try:
            profile = build_job_profile(job["raw_job_description"])
            write_json(output_path, profile.model_dump())
            generated_files.append(str(output_path))
            extracted_skill_examples.append(
                {
                    "job_slug": slug,
                    "job_title": profile.job_title,
                    "required_skills": profile.required_skills[:8],
                    "nice_to_have_skills": profile.nice_to_have_skills[:8],
                }
            )
            warnings_total += len(profile.metadata.warnings)
            success += 1
        except Exception as exc:
            failed += 1
            extracted_skill_examples.append(
                {
                    "job_slug": slug,
                    "error": str(exc),
                }
            )

    duration_seconds = round(time.perf_counter() - started, 4)
    report = {
        "generated_at": utc_now(),
        "parser_route": "rule_based_v1",
        "total_jobs": len(TEST_JOB_DESCRIPTIONS),
        "success": success,
        "failed": failed,
        "warnings": warnings_total,
        "generated_files": generated_files,
        "required_skills_examples": extracted_skill_examples,
        "execution_time_seconds": duration_seconds,
    }
    write_json(REPORT_PATH, report)
    return report


def main() -> None:
    report = run_test_jobs()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
