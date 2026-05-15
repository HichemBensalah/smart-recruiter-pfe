from __future__ import annotations

import re
import unicodedata


SENIORITY_RANKS = {
    "junior": 0,
    "mid_level": 1,
    "senior": 2,
    "lead": 3,
    "principal": 4,
}

SENIORITY_ALIASES = {
    "junior": "junior",
    "jr": "junior",
    "debutant": "junior",
    "entry": "junior",
    "entry level": "junior",
    "entry_level": "junior",
    "mid": "mid_level",
    "middle": "mid_level",
    "mid level": "mid_level",
    "mid_level": "mid_level",
    "intermediate": "mid_level",
    "intermediaire": "mid_level",
    "confirme": "mid_level",
    "senior": "senior",
    "sr": "senior",
    "experienced": "senior",
    "experimente": "senior",
    "lead": "lead",
    "tech lead": "lead",
    "team lead": "lead",
    "staff": "lead",
    "principal": "principal",
    "head": "principal",
    "architect": "principal",
    "architecte": "principal",
}


def normalize_seniority(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_token(value)
    if not normalized:
        return None
    return SENIORITY_ALIASES.get(normalized)


def get_seniority_rank(value: str | None) -> int | None:
    seniority = normalize_seniority(value)
    if seniority is None:
        return None
    return SENIORITY_RANKS[seniority]


def compute_seniority_alignment(job_seniority: str | None, candidate_seniority: str | None) -> float:
    job_rank = get_seniority_rank(job_seniority)
    candidate_rank = get_seniority_rank(candidate_seniority)
    if job_rank is None or candidate_rank is None:
        return 0.0

    delta = candidate_rank - job_rank
    if delta == 0:
        return 1.0
    if delta == 1:
        return 0.85
    if delta == -1:
        return 0.70
    if delta == 2:
        return 0.65
    if delta == -2:
        return 0.40
    return 0.25


def _normalize_token(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.strip().lower()
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())
