from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.core.chatbot.job_router import infer_job_route_from_structured_profile
from src.core.chatbot.memory import ConversationMemory


JOB_INTAKE_STEPS = [
    "job_title",
    "about_role",
    "responsibilities",
    "required_skills",
    "bonus_skills",
    "profile",
]

JOB_INTAKE_QUESTIONS = {
    "job_title": "Quel est le titre du poste ?",
    "about_role": "Decris la section About the role.",
    "responsibilities": "Quelles sont les responsabilites principales ?",
    "required_skills": "Quelles sont les competences obligatoires ?",
    "bonus_skills": "Quelles sont les competences bonus ?",
    "profile": "Decris le profil recherche : experience, seniorite, langue, localisation et mode de travail.",
}

FIELD_LABELS = {
    "job_title": "titre du poste",
    "about_role": "description / About the role",
    "responsibilities": "responsabilites",
    "required_skills": "competences obligatoires",
    "bonus_skills": "competences bonus",
    "profile": "profil recherche",
}


def start_job_intake(memory: ConversationMemory) -> dict[str, Any]:
    memory.mode = "job_creation"
    memory.pending_confirmation = None
    memory.job_intake = {
        "mode": "job_creation",
        "current_step": JOB_INTAKE_STEPS[0],
        "fields": {step: "" for step in JOB_INTAKE_STEPS},
        "structured_job_profile": {},
        "route": {},
    }
    return memory.job_intake


def start_job_intake_with_title(memory: ConversationMemory, job_title: str) -> dict[str, Any]:
    intake = start_job_intake(memory)
    intake["fields"]["job_title"] = job_title.strip()
    intake["current_step"] = "about_role"
    return intake


def get_current_step(memory: ConversationMemory) -> str | None:
    intake = memory.job_intake if isinstance(memory.job_intake, dict) else {}
    step = intake.get("current_step")
    return str(step) if step else None


def update_job_intake(memory: ConversationMemory, user_message: str) -> dict[str, Any]:
    if not isinstance(memory.job_intake, dict):
        start_job_intake(memory)
    intake = memory.job_intake or {}
    current_step = str(intake.get("current_step") or JOB_INTAKE_STEPS[0])
    fields = intake.setdefault("fields", {step: "" for step in JOB_INTAKE_STEPS})
    fields[current_step] = user_message.strip()

    current_index = JOB_INTAKE_STEPS.index(current_step)
    if current_index + 1 < len(JOB_INTAKE_STEPS):
        intake["current_step"] = JOB_INTAKE_STEPS[current_index + 1]
    else:
        intake["current_step"] = None
        structured_profile = build_structured_job_profile(memory)
        intake["structured_job_profile"] = structured_profile
        intake["route"] = infer_job_route_from_structured_profile(structured_profile)
        memory.pending_confirmation = "launch_matching"
    return intake


def detect_field_edit_request(message: str) -> str | None:
    lowered = _normalize(message)
    if not any(verb in lowered for verb in ["modifie", "change", "corrige", "remplace"]):
        return None
    if "titre" in lowered or "title" in lowered:
        return "job_title"
    if "about" in lowered or "description" in lowered:
        return "about_role"
    if "responsabilite" in lowered or "responsabilites" in lowered or "responsibilities" in lowered:
        return "responsibilities"
    if (
        "competences obligatoires" in lowered
        or "competence obligatoire" in lowered
        or "required skills" in lowered
        or "skills obligatoires" in lowered
    ):
        return "required_skills"
    if "competences bonus" in lowered or "competence bonus" in lowered or "bonus skills" in lowered:
        return "bonus_skills"
    if "profil" in lowered or "profile" in lowered or "experience" in lowered or "localisation" in lowered or "location" in lowered:
        return "profile"
    return None


def extract_field_edit_value(message: str, field_name: str | None = None) -> str | None:
    normalized = _normalize(message)
    markers = [" en ", " : ", ": ", " avec ", " par "]
    best_index = -1
    best_marker = ""
    for marker in markers:
        index = normalized.find(marker)
        if index > best_index:
            best_index = index
            best_marker = marker
    if best_index < 0:
        return None
    value = message[best_index + len(best_marker) :].strip()
    return value or None


def detect_offer_summary_request(message: str) -> bool:
    lowered = _normalize(message)
    triggers = [
        "resume l'offre",
        "resume loffre",
        "montre-moi l'offre",
        "montre moi l'offre",
        "montre-moi loffre",
        "montre moi loffre",
        "affiche l'offre",
        "affiche loffre",
        "affiche le profil structure",
        "qu'est-ce que j'ai rempli",
        "quest-ce que jai rempli",
        "show job",
        "show current job",
    ]
    return any(trigger in lowered for trigger in triggers)


def detect_offer_reset_request(message: str) -> bool:
    lowered = _normalize(message)
    triggers = [
        "reinitialise l'offre",
        "reinitialise loffre",
        "reset l'offre",
        "reset loffre",
        "recommencer",
        "nouvelle offre",
        "creer une autre offre",
        "reset job",
        "start new job",
        "new job",
    ]
    return any(trigger in lowered for trigger in triggers)


def reset_job_intake(memory: ConversationMemory) -> dict[str, Any]:
    from src.core.chatbot.runtime_store import clear_all_runtime

    memory.mode = None
    memory.job_intake = None
    memory.pending_confirmation = None
    memory.pending_field_edit = None
    memory.awaiting_field_replacement = False
    memory.offer_created = False
    memory.matching_completed = False
    memory.current_job_profile = None
    memory.routed_job_id = None
    memory.job_description = None
    memory.last_job_query = None
    memory.job_intake_state = None
    memory.last_candidates = []
    memory.last_decision_cards = []
    memory.last_transferability = {}
    memory.selected_candidate_id = None

    # Clean up runtime files when starting a new offer
    clear_all_runtime()

    return start_job_intake(memory)


def summarize_current_offer(memory: ConversationMemory) -> str:
    if not isinstance(memory.job_intake, dict) and not memory.current_job_profile:
        return (
            "Aucune offre n'est encore en cours de creation. "
            "Je vais vous guider pour en creer une. Quel est le titre du poste ?"
        )

    fields = _fields(memory)
    structured_profile = _current_structured_profile(memory)
    route = _current_route(memory)
    filled_count = sum(1 for step in JOB_INTAKE_STEPS if str(fields.get(step) or "").strip())
    current_step = get_current_step(memory)
    if not current_step and filled_count == len(JOB_INTAKE_STEPS):
        current_step = "confirmation"
    required_skills = structured_profile.get("required_skills") or parse_skills(fields.get("required_skills", ""))
    bonus_skills = structured_profile.get("nice_to_have_skills") or parse_skills(fields.get("bonus_skills", ""))

    lines = [
        "Offre en cours de creation :",
        "",
        f"Titre : {_field_or_missing(fields.get('job_title'))}",
        f"About the role : {_field_or_missing(fields.get('about_role'))}",
        f"Responsibilities : {_summarize_section(fields.get('responsibilities'))}",
        f"Competences obligatoires : {_list_or_missing(required_skills)}",
        f"Competences bonus : {_list_or_missing(bonus_skills)}",
        f"Profil : {_field_or_missing(fields.get('profile'))}",
        "",
        f"Progression : {filled_count}/6 champs remplis",
        f"Etape actuelle : {current_step or 'non renseignee'}",
    ]
    if structured_profile:
        lines.extend(
            [
                "",
                "Profil structure :",
                f"- target_role : {structured_profile.get('target_role') or 'non renseigne'}",
                f"- min_years_experience : {structured_profile.get('min_years_experience') or 'non renseigne'}",
                f"- seniority : {structured_profile.get('seniority') or 'non renseigne'}",
                f"- location : {structured_profile.get('location') or 'non renseigne'}",
                f"- work_model : {structured_profile.get('work_model') or 'non renseigne'}",
                f"- language_requirements : {structured_profile.get('language_requirements') or 'non renseigne'}",
            ]
        )
    if route.get("job_id") or memory.routed_job_id:
        lines.extend(
            [
                "",
                f"Job profile utilise : {route.get('job_id') or memory.routed_job_id}",
                f"Raison du routing : {route.get('route_reason') or 'non renseignee'}",
            ]
        )
    if memory.pending_confirmation == "launch_matching" and not memory.matching_completed:
        lines.append("\nVoulez-vous lancer la recherche de candidats ?")
    if memory.matching_completed:
        lines.append("\nCette offre a deja ete utilisee pour le dernier matching.")
    return "\n".join(lines)


def request_field_edit(memory: ConversationMemory, field_name: str) -> None:
    memory.pending_field_edit = field_name
    memory.awaiting_field_replacement = True


def apply_field_edit(memory: ConversationMemory, user_message: str) -> dict[str, Any]:
    if not memory.pending_field_edit:
        raise ValueError("No pending field edit to apply")
    if not isinstance(memory.job_intake, dict):
        start_job_intake(memory)
    fields = memory.job_intake.setdefault("fields", {step: "" for step in JOB_INTAKE_STEPS})
    fields[memory.pending_field_edit] = user_message.strip()
    updated_field = memory.pending_field_edit
    memory.pending_field_edit = None
    memory.awaiting_field_replacement = False
    rebuild_after_edit(memory)
    memory.job_intake["last_edited_field"] = updated_field
    return memory.job_intake


def rebuild_after_edit(memory: ConversationMemory) -> dict[str, Any]:
    if not isinstance(memory.job_intake, dict):
        start_job_intake(memory)
    structured_profile = build_structured_job_profile(memory)
    route = infer_job_route_from_structured_profile(structured_profile)
    memory.job_intake["current_step"] = None
    memory.job_intake["structured_job_profile"] = structured_profile
    memory.job_intake["route"] = route
    memory.pending_confirmation = "launch_matching"
    memory.current_job_profile = structured_profile
    memory.routed_job_id = str(route.get("job_id")) if route.get("job_id") else None
    memory.job_description = build_job_description(memory)
    memory.offer_created = True
    memory.job_intake_state = memory.job_intake
    return memory.job_intake


def _current_structured_profile(memory: ConversationMemory) -> dict[str, Any]:
    intake = memory.job_intake if isinstance(memory.job_intake, dict) else {}
    profile = intake.get("structured_job_profile")
    if isinstance(profile, dict) and profile:
        return profile
    if isinstance(memory.current_job_profile, dict):
        return memory.current_job_profile
    return build_structured_job_profile(memory) if any(_fields(memory).values()) else {}


def _current_route(memory: ConversationMemory) -> dict[str, Any]:
    intake = memory.job_intake if isinstance(memory.job_intake, dict) else {}
    route = intake.get("route")
    return route if isinstance(route, dict) else {}


def _field_or_missing(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "non renseigne"


def _summarize_section(value: Any) -> str:
    items = parse_skills(value)
    if items:
        return f"{len(items)} element(s) renseignes"
    return _field_or_missing(value)


def _list_or_missing(values: Any) -> str:
    if isinstance(values, list) and values:
        return ", ".join(str(value) for value in values)
    return "non renseigne"


def is_job_intake_complete(memory: ConversationMemory) -> bool:
    fields = _fields(memory)
    return all(str(fields.get(step) or "").strip() for step in JOB_INTAKE_STEPS)


def build_job_description(memory: ConversationMemory) -> str:
    fields = _fields(memory)
    return "\n\n".join(
        [
            f"Title\n{fields.get('job_title', '')}",
            f"About the role\n{fields.get('about_role', '')}",
            f"Responsibilities\n{fields.get('responsibilities', '')}",
            f"Required skills\n{fields.get('required_skills', '')}",
            f"Bonus skills\n{fields.get('bonus_skills', '')}",
            f"Profile\n{fields.get('profile', '')}",
        ]
    ).strip()


def build_structured_job_profile(memory: ConversationMemory) -> dict[str, Any]:
    fields = _fields(memory)
    profile_text = str(fields.get("profile") or "")
    required_skills = parse_skills(fields.get("required_skills", ""))
    bonus_skills = parse_skills(fields.get("bonus_skills", ""))
    return {
        "job_title": str(fields.get("job_title") or ""),
        "target_role": infer_target_role(str(fields.get("job_title") or ""), required_skills),
        "about_role": str(fields.get("about_role") or ""),
        "responsibilities": parse_skills(fields.get("responsibilities", "")),
        "required_skills": required_skills,
        "nice_to_have_skills": bonus_skills,
        "min_years_experience": extract_min_years_experience(profile_text),
        "seniority": extract_seniority(profile_text),
        "location": extract_location(profile_text),
        "work_model": extract_work_model(profile_text),
        "language_requirements": extract_languages(profile_text),
        "raw_sections": dict(fields),
    }


def summarize_job_profile(memory: ConversationMemory) -> str:
    intake = memory.job_intake if isinstance(memory.job_intake, dict) else {}
    structured_profile = intake.get("structured_job_profile") or build_structured_job_profile(memory)
    route = intake.get("route") or infer_job_route_from_structured_profile(structured_profile)
    required_skills = ", ".join(structured_profile.get("required_skills", [])) or "non renseigne"
    bonus_skills = ", ".join(structured_profile.get("nice_to_have_skills", [])) or "non renseigne"
    return "\n".join(
        [
            "J'ai construit l'offre suivante :",
            "",
            build_job_description(memory),
            "",
            "Job profile structure :",
            f"- job_title : {structured_profile.get('job_title')}",
            f"- target_role : {structured_profile.get('target_role')}",
            f"- required_skills : {required_skills}",
            f"- nice_to_have_skills : {bonus_skills}",
            f"- min_years_experience : {structured_profile.get('min_years_experience')}",
            f"- seniority : {structured_profile.get('seniority')}",
            f"- location : {structured_profile.get('location')}",
            f"- work_model : {structured_profile.get('work_model')}",
            f"- language_requirements : {structured_profile.get('language_requirements')}",
            "",
            f"Job profile recommande : {route.get('job_id')}.",
            f"Raison du routing : {route.get('route_reason')}",
            "",
            "Voulez-vous lancer la recherche de candidats ?",
        ]
    )


def parse_skills(value: Any) -> list[str]:
    text = str(value or "")
    parts: list[str] = []
    for line in text.replace(";", "\n").splitlines():
        cleaned = re.sub(r"^\s*[-*•]\s*", "", line).strip()
        if not cleaned:
            continue
        parts.extend(piece.strip() for piece in cleaned.split(",") if piece.strip())
    return _dedupe(parts)


def extract_min_years_experience(text: str) -> int | None:
    match = re.search(r"(?:at least\s*)?(\d+)\s*(?:years|year|ans|an)\b", text.lower())
    return int(match.group(1)) if match else None


def extract_seniority(text: str) -> str | None:
    lowered = text.lower()
    if "senior" in lowered:
        return "senior"
    if "mid-level" in lowered or "mid level" in lowered or "intermediate" in lowered:
        return "mid-level"
    if "junior" in lowered:
        return "junior"
    return None


def extract_location(text: str) -> str | None:
    lowered = text.lower()
    if "tunis" in lowered:
        return "Tunis"
    if "tunisia" in lowered or "tunisie" in lowered:
        return "Tunisia"
    return None


def extract_work_model(text: str) -> str | None:
    lowered = text.lower()
    if "hybrid" in lowered or "hybride" in lowered:
        return "hybrid"
    if "remote" in lowered or "a distance" in lowered:
        return "remote"
    if "onsite" in lowered or "on-site" in lowered or "presentiel" in lowered:
        return "onsite"
    return None


def extract_languages(text: str) -> list[str]:
    lowered = text.lower()
    languages = []
    for language in ["English", "French", "Arabic"]:
        if language.lower() in lowered:
            languages.append(language)
    return languages


def infer_target_role(job_title: str, required_skills: list[str]) -> str:
    text = " ".join([job_title.lower(), " ".join(skill.lower() for skill in required_skills)])
    if "data engineer" in text:
        return "Data Engineer"
    if "data analyst" in text or "powerbi" in text or "power bi" in text:
        return "Data Analyst"
    if "machine learning" in text or "nlp" in text:
        return "Machine Learning Engineer"
    if "frontend" in text:
        return "Frontend Developer"
    if "full stack" in text or "fullstack" in text:
        return "Full Stack Developer"
    return "Backend Developer"


def _fields(memory: ConversationMemory) -> dict[str, Any]:
    intake = memory.job_intake if isinstance(memory.job_intake, dict) else {}
    fields = intake.get("fields", {})
    return fields if isinstance(fields, dict) else {}


def _dedupe(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _normalize(value: str) -> str:
    lowered = value.lower()
    mojibake_replacements = {
        "Ã©": "é",
        "Ã¨": "è",
        "Ãª": "ê",
        "Ã ": "à",
        "Ã¢": "â",
        "Ã¹": "ù",
        "Ã§": "ç",
        "Ã´": "ô",
        "Ã®": "î",
        "â€™": "'",
        "’": "'",
    }
    for source, target in mojibake_replacements.items():
        lowered = lowered.replace(source, target)
    lowered = unicodedata.normalize("NFKD", lowered)
    lowered = "".join(char for char in lowered if not unicodedata.combining(char))
    replacements = {
        "l'offre": "loffre",
        "l offre": "loffre",
    }
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return lowered
