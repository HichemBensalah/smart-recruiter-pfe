# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import requests
import streamlit as st


def _env_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_API_BASE_URL = os.getenv("SMART_RECRUITER_API_BASE_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("SMART_RECRUITER_API_KEY", "") if _env_bool(os.getenv("AUTH_ENABLED")) else ""
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-Smart-Recruiter-Key")
API_UNAVAILABLE_MESSAGE = "API indisponible. Vérifiez que FastAPI est lancé et que l'URL API correspond au bon port."
API_TIMEOUT_CHAT_ARTIFACT = int(os.getenv("API_TIMEOUT_CHAT_ARTIFACT", "60"))
API_TIMEOUT_CHAT_LIVE = int(os.getenv("API_TIMEOUT_CHAT_LIVE", "180"))
API_TIMEOUT_HEALTH = int(os.getenv("API_TIMEOUT_HEALTH", "10"))

EXAMPLE_PROMPTS = [
    "Pourquoi le premier candidat ?",
    "Quels sont ses gaps ?",
    "Compare le premier et le deuxième candidat",
    "Quels candidats sont à vérifier ?",
]

MOJIBAKE_REPLACEMENTS: dict[str, str] = {
    "Ã©": "é",    # Ã© -> é
    "Ã¨": "è",    # Ã¨ -> è
    "Ãª": "ê",    # Ãª -> ê
    "Ã«": "ë",    # Ã« -> ë
    "Ã ": "à",    # Ã  -> à
    "Ã¢": "â",    # Ã¢ -> â
    "Ã¹": "ù",    # Ã¹ -> ù
    "Ã»": "û",    # Ã» -> û
    "Ã´": "ô",    # Ã´ -> ô
    "Ã¶": "ö",    # Ã¶ -> ö
    "Ã®": "î",    # Ã® -> î
    "Ã¯": "ï",    # Ã¯ -> ï
    "Ã§": "ç",    # Ã§ -> ç
    "Ã‰": "É",    # Ã‰ -> É
    "Â": "",                 # Â -> (removed)
    "â€™": "'",  # â€™ -> '
    "â€œ": '"',  # â€œ -> "
    "â€�": '"',  # â€? -> "
    "â€“": "-",  # â€" -> -
    "â€”": "-",  # â€" -> -
}

_STATUS_LABELS: dict[str, str] = {
    "agreement_high": "✅ Accord élevé",
    "review_needed": "⚠️ À vérifier",
    "agreement_low": "⬇️ Accord faible",
    "no_data": "— Données insuffisantes",
}

_PHASE_LABELS: dict[str, str] = {
    "intake": "Création d'offre",
    "confirmation": "Offre prête pour confirmation",
    "matching": "Matching terminé — Résultats candidats",
    "followup": "Analyse candidat / Question de suivi",
    "start": "Démarrage",
}


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Smart Recruiter", page_icon="SR", layout="wide")
    st.title("Smart Recruiter")
    st.caption("Créez une offre structurée, trouvez les meilleurs candidats et analysez leurs forces en quelques étapes.")

    init_chat_history()
    api_base_url, api_key = render_sidebar()

    if not st.session_state.messages:
        render_welcome_card(api_base_url, api_key)
    else:
        render_chat_history()

    prompt = st.chat_input("Décrivez votre besoin recruteur...")
    if prompt:
        submit_message(api_base_url, prompt, api_key)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str]:
    st.sidebar.title("Smart Recruiter")

    # ── Paramètres techniques (hidden by default) ─────────────────────────────
    api_base_url = DEFAULT_API_BASE_URL
    api_key = DEFAULT_API_KEY
    with st.sidebar.expander("Paramètres techniques", expanded=False):
        api_base_url = st.text_input("URL de l'API", value=DEFAULT_API_BASE_URL).rstrip("/")
        api_key = st.text_input("API key API", value=DEFAULT_API_KEY, type="password")
        if st.button("Vérifier API"):
            health = call_health_api(api_base_url, api_key)
            if health.get("error"):
                st.error(health["error"])
            else:
                deps = health.get("dependencies") if isinstance(health.get("dependencies"), dict) else {}
                st.success(f"API disponible : {health.get('service')} / {health.get('version')}")
                if deps:
                    st.caption(f"Dépendances : {', '.join(deps.keys())}")
                warnings = health.get("warnings") if isinstance(health.get("warnings"), list) else []
                if warnings:
                    st.warning("\n".join(clean_display_text(str(w)) for w in warnings))

    # ── Nouvelle offre ────────────────────────────────────────────────────────
    if st.sidebar.button("Nouvelle offre", type="primary"):
        _reset_session(api_base_url, api_key)

    st.sidebar.divider()

    # ── Parcours recommandé ───────────────────────────────────────────────────
    st.sidebar.subheader("Parcours recommandé")
    st.sidebar.markdown(
        "1. Créer l'offre\n"
        "2. Confirmer le profil\n"
        "3. Lancer la recherche\n"
        "4. Analyser les candidats"
    )

    st.sidebar.divider()
    render_session_state_panel()

    st.sidebar.divider()
    st.sidebar.subheader("Actions rapides")
    for example in EXAMPLE_PROMPTS:
        if st.sidebar.button(example, key=f"quick_{example}"):
            submit_message(api_base_url, example, api_key)
            st.rerun()

    st.session_state.api_base_url = api_base_url
    return api_base_url, api_key


def _reset_session(api_base_url: str, api_key: str) -> None:
    st.session_state.messages = []
    st.session_state.last_payload = {}
    st.session_state.session_id = str(uuid4())
    payload = call_chat_api(api_base_url, "nouvelle offre", api_key)
    add_message("user", "Nouvelle offre")
    add_message("assistant", payload.get("answer", ""), payload=payload)
    st.rerun()


def render_welcome_card(api_base_url: str, api_key: str) -> None:
    with st.container(border=True):
        st.markdown("## Bienvenue dans Smart Recruiter")
        st.write("Votre assistant RH vous aide à :")
        st.markdown(
            "- créer une offre en 6 étapes ;\n"
            "- confirmer le profil recherché ;\n"
            "- lancer un matching explicable ;\n"
            "- analyser les scores, gaps et Decision Cards."
        )
        if st.button("Commencer une nouvelle offre", type="primary", key="welcome_start"):
            _reset_session(api_base_url, api_key)


# ── Session init ──────────────────────────────────────────────────────────────

def init_chat_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "last_payload" not in st.session_state:
        st.session_state.last_payload = {}


def submit_message(api_base_url: str, message: str, api_key: str = "") -> None:
    if not message.strip():
        st.warning("Message vide ignoré.")
        return
    add_message("user", message)
    with st.chat_message("user"):
        st.write(message)
    with st.chat_message("assistant"):
        last_payload = st.session_state.get("last_payload", {})
        is_live_mode = "live" in str(last_payload.get("matching_metadata", {}).get("matching_mode_used", "")).lower()
        spinner_msg = (
            "Matching live en cours : MongoDB + FAISS + Matching V3…"
            if is_live_mode
            else "Analyse du besoin recruteur..."
        )
        with st.spinner(spinner_msg):
            payload = call_chat_api(api_base_url, message, api_key)
        if payload.get("error"):
            st.error(payload["error"])
            add_message("assistant", payload["error"], payload=payload)
        else:
            render_copilot_response(payload)
            add_message("assistant", payload.get("answer", ""), payload=payload)


def add_message(role: str, content: str, payload: dict[str, Any] | None = None) -> None:
    st.session_state.messages.append({"role": role, "content": content, "payload": payload or {}})


# ── Sidebar — état de session ─────────────────────────────────────────────────

def render_session_state_panel() -> None:
    st.sidebar.divider()
    st.sidebar.subheader("État de session")
    payload = st.session_state.get("last_payload", {})
    session_id = str(st.session_state.get("session_id", ""))
    st.sidebar.caption(f"Session: `{_short_session_id(session_id)}`")

    intake = payload.get("job_intake_state") if isinstance(payload.get("job_intake_state"), dict) else {}
    structured_profile = payload.get("structured_job_profile") if isinstance(payload.get("structured_job_profile"), dict) else {}
    fields = intake.get("fields") if isinstance(intake.get("fields"), dict) else {}
    steps = ["job_title", "about_role", "responsibilities", "required_skills", "bonus_skills", "profile"]
    filled = sum(1 for s in steps if fields.get(s))
    current_step = intake.get("current_step") or ("confirmation" if filled == len(steps) and fields else "non démarré")
    matching_completed = bool(payload.get("matching_completed", False))
    routed_job_id = payload.get("routed_job_id") or "non encore défini"
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    selected_candidate_id = payload.get("selected_candidate_id") or "non sélectionné"
    offer_active = bool(intake or structured_profile)

    st.sidebar.write(f"Offre en cours : {'oui' if offer_active and not matching_completed else 'terminée' if matching_completed else 'non'}")
    st.sidebar.write(f"Étape : `{current_step}`")
    st.sidebar.write(f"Progression : {filled}/6")
    st.sidebar.progress(filled / len(steps) if steps else 0)
    st.sidebar.write(f"Matching lancé : {'oui' if matching_completed else 'non'}")
    st.sidebar.write(f"Job ID : `{routed_job_id}`")
    st.sidebar.write(f"Candidats retournés : {len(candidates)}")
    st.sidebar.write(f"Candidat sélectionné : `{selected_candidate_id}`")
    render_matching_metadata_sidebar(payload)
    render_offer_summary_sidebar(payload)


def render_matching_metadata_sidebar(payload: dict[str, Any]) -> None:
    metadata = payload.get("matching_metadata") if isinstance(payload.get("matching_metadata"), dict) else {}
    if not metadata:
        return
    st.sidebar.divider()
    st.sidebar.subheader("Matching")
    mode = metadata.get("matching_mode") or "non disponible"
    st.sidebar.write(f"Mode : `{mode}`")
    st.sidebar.write(f"Job résolu : `{metadata.get('resolved_job_id') or 'non disponible'}`")
    st.sidebar.write(f"Artefact : `{metadata.get('artifact_source') or 'non disponible'}`")
    st.sidebar.write(f"Fallback : `{'oui' if metadata.get('fallback_used') else 'non'}`")
    if mode == "matching_v3_job_artifact" or mode == "artifact":
        st.sidebar.caption("Résultats pré-calculés pour une démonstration stable.")
    metadata_warnings = metadata.get("warnings") if isinstance(metadata.get("warnings"), list) else []
    if metadata_warnings:
        st.sidebar.warning("\n".join(clean_display_text(str(w)) for w in metadata_warnings))


def render_offer_summary_sidebar(payload: dict[str, Any]) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("Résumé de l'offre")
    profile = payload.get("structured_job_profile") if isinstance(payload.get("structured_job_profile"), dict) else {}
    intake = payload.get("job_intake_state") if isinstance(payload.get("job_intake_state"), dict) else {}
    fields = intake.get("fields") if isinstance(intake.get("fields"), dict) else {}
    routed_job_id = payload.get("routed_job_id") or "non encore défini"

    if not profile and not fields:
        st.sidebar.caption("Aucune offre complète pour le moment.")
        return

    required_skills = profile.get("required_skills") or _split_field_values(fields.get("required_skills"))
    nice_to_have_skills = profile.get("nice_to_have_skills") or _split_field_values(fields.get("bonus_skills"))

    st.sidebar.write(f"Titre : `{profile.get('job_title') or fields.get('job_title') or 'non renseigné'}`")
    st.sidebar.write(f"Rôle cible : `{profile.get('target_role') or 'non renseigné'}`")
    st.sidebar.write(f"Compétences obligatoires : `{_format_list(required_skills)}`")
    st.sidebar.write(f"Bonus : `{_format_list(nice_to_have_skills)}`")
    st.sidebar.write(f"Expérience : `{_format_years(profile.get('min_years_experience'))}`")
    st.sidebar.write(f"Séniorité : `{profile.get('seniority') or 'non renseignée'}`")
    st.sidebar.write(f"Localisation : `{profile.get('location') or 'non renseignée'}`")
    st.sidebar.write(f"Mode : `{profile.get('work_model') or 'non renseigné'}`")
    st.sidebar.write(f"Job ID : `{routed_job_id}`")

    if profile:
        with st.sidebar.expander("Voir JSON technique", expanded=False):
            st.json(profile)


# ── Chat history ──────────────────────────────────────────────────────────────

def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("payload") and not message["payload"].get("error"):
                render_copilot_response(message["payload"])
            else:
                st.write(message.get("content", ""))


# ── API helpers ───────────────────────────────────────────────────────────────

def call_health_api(api_base_url: str, api_key: str = "") -> dict[str, Any]:
    try:
        response = requests.get(f"{api_base_url}/health", headers=build_api_headers(api_key), timeout=API_TIMEOUT_HEALTH)
        if response.status_code >= 400:
            return {"error": f"{API_UNAVAILABLE_MESSAGE} HTTP {response.status_code}: {extract_error_message(response)}"}
        payload = response.json()
    except requests.RequestException as exc:
        return {"error": f"{API_UNAVAILABLE_MESSAGE} Détail technique : {exc}"}
    except ValueError:
        return {"error": "La réponse /health n'est pas un JSON valide."}
    if not isinstance(payload, dict):
        return {"error": "La réponse /health doit être un objet JSON."}
    if payload.get("status") != "ok":
        return {"error": f"API joignable mais statut inattendu : {payload}"}
    return payload


def call_chat_api(api_base_url: str, message: str, api_key: str = "") -> dict[str, Any]:
    last_payload = st.session_state.get("last_payload", {})
    is_live_mode = "live" in str(last_payload.get("matching_metadata", {}).get("matching_mode_used", "")).lower()
    timeout = API_TIMEOUT_CHAT_LIVE if is_live_mode else API_TIMEOUT_CHAT_ARTIFACT

    try:
        response = requests.post(
            f"{api_base_url}/api/chat",
            json={"message": message, "session_id": st.session_state.session_id},
            headers=build_api_headers(api_key),
            timeout=timeout,
        )
        if response.status_code >= 400:
            return {"error": f"/api/chat a retourne HTTP {response.status_code}: {extract_error_message(response)}"}
        payload = response.json()
    except requests.Timeout as exc:
        return {
            "error": (
                f"Le matching live prend trop de temps (>{timeout}s). "
                "Vérifiez les logs FastAPI et l'état de MongoDB/FAISS. "
                f"Détail : {exc}"
            )
        }
    except requests.ConnectionError as exc:
        return {"error": f"{API_UNAVAILABLE_MESSAGE} Détail technique : {exc}"}
    except requests.RequestException as exc:
        return {"error": f"{API_UNAVAILABLE_MESSAGE} Détail technique : {exc}"}
    except ValueError:
        return {"error": "La réponse /api/chat n'est pas un JSON valide."}
    if not isinstance(payload, dict):
        return {"error": "La réponse /api/chat doit être un objet JSON."}
    if payload.get("session_id"):
        st.session_state.session_id = str(payload["session_id"])
    st.session_state.last_payload = payload
    return payload


def build_api_headers(api_key: str = "") -> dict[str, str]:
    cleaned_api_key = str(api_key or "").strip()
    if not cleaned_api_key:
        return {}
    return {API_KEY_HEADER: cleaned_api_key}


def extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, dict):
        return clean_display_text(str(payload.get("detail") or payload))
    return clean_display_text(str(payload))


# ── Phase detection ───────────────────────────────────────────────────────────

def _detect_phase(payload: dict[str, Any]) -> str:
    sources = list(payload.get("sources") or [])
    matching_completed = bool(payload.get("matching_completed"))
    intake = payload.get("job_intake_state") if isinstance(payload.get("job_intake_state"), dict) else {}
    fields = intake.get("fields") if isinstance(intake.get("fields"), dict) else {}
    steps = ["job_title", "about_role", "responsibilities", "required_skills", "bonus_skills", "profile"]
    filled = sum(1 for s in steps if fields.get(s))

    if matching_completed:
        # A fresh matching result is marked either by the artifact graph node
        # ("match_candidates") or by the live pipeline ("live_*" sources).
        is_matching_result = "match_candidates" in sources or any(
            str(source).startswith("live_") for source in sources
        )
        if is_matching_result:
            return "matching"
        return "followup"
    if filled == 6:
        return "confirmation"
    if filled > 0 or intake:
        return "intake"
    return "start"


def _render_phase_badge(phase: str) -> None:
    label = _PHASE_LABELS.get(phase, phase)
    if phase == "matching":
        st.success(f"**{label}**")
    elif phase == "followup":
        st.info(f"**{label}**")
    elif phase == "confirmation":
        st.warning(f"**{label}**")
    elif phase == "intake":
        st.info(f"**{label}**")


# ── Main response renderer ────────────────────────────────────────────────────

def render_copilot_response(payload: dict[str, Any]) -> None:
    phase = _detect_phase(payload)
    _render_phase_badge(phase)

    st.markdown(clean_display_text(payload.get("answer") or "Aucune réponse textuelle retournée."))

    render_job_intake(payload, phase=phase)
    render_matching_metadata(payload)

    # Only show candidates/transferability/decision_cards in matching phase, not follow-up
    is_follow_up = phase == "followup"

    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    if candidates and not is_follow_up:
        st.subheader("Candidats recommandés")
        for i, candidate in enumerate(candidates, start=1):
            render_candidate(candidate, rank=i)

    transferability = payload.get("transferability") if isinstance(payload.get("transferability"), dict) else {}
    if transferability and not is_follow_up:
        st.subheader("Transférabilité et gaps")
        render_transferability(transferability)

    decision_cards = payload.get("decision_cards") if isinstance(payload.get("decision_cards"), list) else []
    if decision_cards and not is_follow_up:
        with st.expander("Decision Cards", expanded=False):
            for card in decision_cards:
                render_decision_card(card)

    # Show warnings in a technical details expander to avoid clutter
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    if warnings:
        with st.expander("Détails techniques", expanded=False):
            st.warning("\n".join(clean_display_text(str(w)) for w in warnings))

    sources = payload.get("sources") if isinstance(payload.get("sources"), list) else []
    if sources:
        st.caption("Sources : " + ", ".join(str(s) for s in sources))


def render_payload_warnings(payload: dict[str, Any]) -> None:
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    if warnings:
        st.warning("\n".join(clean_display_text(str(w)) for w in warnings))


# ── Matching metadata (inline) ────────────────────────────────────────────────

def render_matching_metadata(payload: dict[str, Any]) -> None:
    metadata = payload.get("matching_metadata") if isinstance(payload.get("matching_metadata"), dict) else {}
    routed_job_id = payload.get("routed_job_id")
    matching_completed = bool(payload.get("matching_completed"))

    # Only show this panel after matching is done or when metadata has actual content
    has_content = bool(metadata.get("matching_mode") or metadata.get("resolved_job_id"))
    if not matching_completed and not has_content:
        return

    with st.expander("Mode matching et routing", expanded=False):
        if routed_job_id:
            st.write(f"**Job ID routé :** `{routed_job_id}`")

        if has_content:
            mode_requested = metadata.get("matching_mode_requested") or ""
            mode_used = metadata.get("matching_mode_used") or metadata.get("matching_mode") or "non disponible"

            if mode_used and "live" in str(mode_used).lower():
                st.write(f"**Mode matching :** :green[`live` (recalculé en temps réel)]")
                generated_job_id = metadata.get("generated_job_id")
                if generated_job_id:
                    st.write(f"**Job généré :** `{generated_job_id}`")
                generated_path = metadata.get("generated_job_profile_path")
                if generated_path:
                    st.caption(f"Profile: `{generated_path}`")
                live_ready = metadata.get("live_ready")
                if live_ready is False:
                    st.error("**Live matching unavailable**")
                    blocking = metadata.get("live_blocking_reasons", [])
                    if isinstance(blocking, list) and blocking:
                        for reason in blocking:
                            st.caption(f"- {reason}")
                else:
                    st.caption("MongoDB + FAISS + SentenceTransformer (Matching V3 runtime)")
                removed = metadata.get("duplicates_removed_count", 0)
                if isinstance(removed, int) and removed > 0:
                    st.caption(f"Doublons candidats filtrés : {removed}")
            else:
                mode = mode_used
                st.write(f"**Mode matching :** `{mode}`")
                if "artifact" in str(mode):
                    st.caption("Résultats pré-calculés pour une démonstration stable.")

                resolved = metadata.get("resolved_job_id")
                if resolved:
                    st.write(f"**Job résolu :** `{resolved}`")

                artifact = metadata.get("artifact_source")
                if artifact:
                    st.write(f"**Artefact :** `{artifact}`")

                fallback_used = metadata.get("fallback_used")
                if fallback_used is not None:
                    st.write(f"**Fallback utilisé :** `{'oui' if fallback_used else 'non'}`")
        elif matching_completed and routed_job_id:
            st.caption("Détails du mode de matching non disponibles pour cette session.")

        metadata_warnings = metadata.get("warnings") if isinstance(metadata.get("warnings"), list) else []
        if metadata_warnings:
            st.warning("\n".join(clean_display_text(str(w)) for w in metadata_warnings))


# ── Candidate card ────────────────────────────────────────────────────────────

def render_candidate(candidate: dict[str, Any], rank: int = 0) -> None:
    candidate_id = str(candidate.get("candidate_id") or "candidat_inconnu")
    raw_name = candidate.get("candidate_name") or candidate.get("full_name") or candidate.get("name")
    display_name = clean_display_text(str(raw_name)) if raw_name else None

    rank_prefix = f"{rank}." if rank > 0 else ""
    if display_name:
        title = f"{rank_prefix} {display_name}".strip()
    else:
        rank_label = rank if rank > 0 else "?"
        title = f"{rank_prefix} Candidat {rank_label}".strip()

    status_raw = str(candidate.get("recommendation_status") or "")
    if status_raw == "agreement_high":
        status_badge = ":green[✅ Accord élevé]"
    elif status_raw == "review_needed":
        status_badge = ":orange[⚠️ À vérifier]"
    elif status_raw:
        status_badge = f":gray[{status_raw}]"
    else:
        status_badge = ":gray[Statut non disponible]"

    v3_score = candidate.get("baseline_score_v3")
    v3_rank = candidate.get("baseline_rank_v3") or candidate.get("rank")
    rf_score = candidate.get("rf_score")
    xgb_score = candidate.get("xgboost_score")

    with st.container(border=True):
        header_cols = st.columns([5, 1])
        with header_cols[0]:
            st.markdown(f"### {title}")
            st.caption(f"ID candidat : `{candidate_id}`")
            if not display_name:
                st.caption("Candidat anonymisé dans les artefacts de démonstration.")
        with header_cols[1]:
            st.markdown(status_badge)

        score_cols = st.columns([2, 1])
        with score_cols[0]:
            score_display = f"{float(v3_score):.4f}" if v3_score is not None else "n/a"
            st.metric(label="Score Matching V3 — baseline officielle", value=score_display)
        with score_cols[1]:
            rank_display = str(v3_rank) if v3_rank is not None else "n/a"
            st.metric(label="Rang V3", value=rank_display)

        score_breakdown = candidate.get("score_breakdown")
        base_score = candidate.get("base_score_before_penalty")
        mh_coverage = candidate.get("must_have_coverage")
        mh_mult = candidate.get("must_have_penalty_multiplier")
        mh_applied = candidate.get("must_have_penalty_applied")
        q_mult = candidate.get("quality_penalty_multiplier")

        if score_breakdown and isinstance(score_breakdown, list):
            with st.expander("Décomposition du score Matching V3", expanded=False):
                st.caption("Contributions des critères au score de base (avant pénalités).")
                for entry in score_breakdown:
                    feat = entry.get("feature", "")
                    raw = entry.get("raw_score", 0.0)
                    weight = entry.get("weight", 0.0)
                    contrib = entry.get("contribution", 0.0)
                    bar_pct = min(int(contrib * 100 / 0.35 * 100), 100)
                    st.markdown(
                        f"**{feat}** — score brut `{raw:.2f}` × poids `{weight:.3f}` = contribution **`{contrib:.3f}`**"
                    )
                    st.progress(bar_pct)

                if base_score is not None:
                    st.divider()
                    st.markdown(f"**Score avant pénalités (base) :** `{base_score:.4f}`")
                    if mh_mult is not None:
                        coverage_pct = f"{mh_coverage:.0%}" if mh_coverage is not None else "n/a"
                        applied_label = " — pénalité appliquée" if mh_applied else " — aucune pénalité"
                        st.markdown(
                            f"**Pénalité must-have :** ×`{mh_mult:.2f}`"
                            f" (couverture {coverage_pct}){applied_label}"
                        )
                    if q_mult is not None and q_mult < 1.0:
                        st.markdown(f"**Pénalité qualité :** ×`{q_mult:.3f}`")
                    if v3_score is not None:
                        st.markdown(f"**Score final :** `{float(v3_score):.4f}`")

        if rf_score is not None or xgb_score is not None:
            with st.expander("Comparaison ML expérimentale", expanded=False):
                st.caption("Ces scores viennent de modèles expérimentaux sur pseudo-labels.")
                ml_cols = st.columns(2)
                ml_cols[0].metric("Random Forest", format_score(rf_score))
                ml_cols[1].metric("XGBoost", format_score(xgb_score))

        render_candidate_cv_link(candidate)

        summary = candidate.get("short_decision_summary")
        if summary:
            st.write(clean_display_text(str(summary)))


def render_candidate_cv_link(candidate: dict[str, Any]) -> None:
    if candidate.get("cv_available") and candidate.get("cv_download_url"):
        cv_url = _absolute_api_url(str(candidate["cv_download_url"]))
        mime_type = str(candidate.get("cv_mime_type") or "")
        label = "Voir CV" if mime_type in {"application/pdf", "image/jpeg", "image/png", "text/plain"} else "Télécharger CV original"
        st.link_button(label, cv_url)
        if candidate.get("cv_filename"):
            st.caption(f"CV original : `{candidate['cv_filename']}`")
    else:
        st.caption("CV original non disponible")


# ── Job intake wizard UI ──────────────────────────────────────────────────────

def render_job_intake(payload: dict[str, Any], phase: str = "") -> None:
    intake = payload.get("job_intake_state") if isinstance(payload.get("job_intake_state"), dict) else {}
    structured_profile = payload.get("structured_job_profile") if isinstance(payload.get("structured_job_profile"), dict) else {}
    routed_job_id = payload.get("routed_job_id")

    if not intake and not structured_profile and not routed_job_id:
        return

    # After matching: show compact summary only, not the full wizard
    if phase in ("matching", "followup"):
        if routed_job_id:
            st.caption(f"Offre utilisée pour ce matching : `{routed_job_id}`")
        return

    steps = ["job_title", "about_role", "responsibilities", "required_skills", "bonus_skills", "profile"]
    labels = {
        "job_title": "Titre du poste",
        "about_role": "About the role",
        "responsibilities": "Responsabilités",
        "required_skills": "Compétences obligatoires",
        "bonus_skills": "Compétences bonus",
        "profile": "Profil recherché",
    }
    fields = intake.get("fields") if isinstance(intake.get("fields"), dict) else {}
    filled = sum(1 for s in steps if fields.get(s))
    current_step = intake.get("current_step")

    if phase == "confirmation":
        st.subheader("Offre prête pour confirmation")
    else:
        # "Offre en cours de création" — shown during intake phase
        st.subheader("Offre en cours de création")

    st.progress(filled / len(steps))
    if current_step:
        step_index = steps.index(current_step) + 1 if current_step in steps else filled + 1
        st.caption(f"Étape {step_index}/6 — {labels.get(str(current_step), str(current_step))}")
    else:
        st.caption("Progression 6/6 — offre structurée prête pour confirmation")

    with st.expander("Champs déjà remplis", expanded=False):
        for step in steps:
            st.write(f"**{labels[step]}**")
            st.write(fields.get(step) or "_à compléter_")

    if structured_profile:
        _render_structured_profile(structured_profile, routed_job_id)
    elif routed_job_id:
        st.info(f"Job profile routé : `{routed_job_id}`")


def _render_structured_profile(profile: dict[str, Any], routed_job_id: str | None) -> None:
    st.subheader("Profil structuré de l'offre")

    required_skills = profile.get("required_skills") or []
    nice_to_have = profile.get("nice_to_have_skills") or []
    languages = profile.get("language_requirements") or []

    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Titre :** {profile.get('job_title') or 'non renseigné'}")
        st.write(f"**Rôle cible :** {profile.get('target_role') or 'non renseigné'}")
        st.write(f"**Séniorité :** {profile.get('seniority') or 'non renseignée'}")
        st.write(f"**Expérience min. :** {_format_years(profile.get('min_years_experience'))}")
    with col_b:
        st.write(f"**Localisation :** {profile.get('location') or 'non renseignée'}")
        st.write(f"**Mode de travail :** {profile.get('work_model') or 'non renseigné'}")
        st.write(f"**Langues :** {_format_list(languages) if languages else 'non renseignées'}")
        if routed_job_id:
            st.write(f"**Job ID routé :** `{routed_job_id}`")

    if required_skills:
        st.write("**Compétences obligatoires :**")
        st.write(" · ".join(str(s) for s in required_skills))
    if nice_to_have:
        st.write("**Compétences bonus :**")
        st.write(" · ".join(str(s) for s in nice_to_have))

    with st.expander("Voir JSON technique", expanded=False):
        st.json(profile)


# ── Transferability and gaps ──────────────────────────────────────────────────

def render_transferability(transferability: dict[str, Any]) -> None:
    for candidate_id, payload in transferability.items():
        selected = payload.get("selected_source") if isinstance(payload, dict) else None
        raw = payload.get(selected) if isinstance(payload, dict) and selected else payload
        details = extract_transferability_details(raw)

        score = details.get("transferability_score") or details.get("coverage_score")
        fit = details.get("fit_direct")
        gaps_bloquants = details.get("gaps_bloquants") or []
        gaps_compensables = details.get("gaps_compensables") or []
        transitions = details.get("transitions_plausibles") or details.get("plausible_transitions") or []

        with st.expander(f"Candidat : {candidate_id}", expanded=False):
            meta_cols = st.columns(2)
            with meta_cols[0]:
                st.metric("Score de transférabilité", format_score(score) if score is not None else "n/a")
            with meta_cols[1]:
                st.write(f"**Fit direct :** {fit if fit is not None else 'Information non disponible'}")

            if gaps_bloquants:
                st.write("**Gaps bloquants :**")
                for gap in gaps_bloquants:
                    st.markdown(f"- :red[{clean_display_text(str(gap))}]")
            else:
                st.write("**Gaps bloquants :** Information non disponible")

            if gaps_compensables:
                st.write("**Gaps compensables :**")
                for gap in gaps_compensables:
                    st.markdown(f"- :orange[{clean_display_text(str(gap))}]")
            else:
                st.write("**Gaps compensables :** Information non disponible")

            if transitions:
                st.write("**Transitions plausibles :**")
                for t in transitions:
                    _render_transition(t)


def extract_transferability_details(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("transferability")
    if isinstance(nested, dict):
        return nested
    return payload


def _render_transition(t: Any) -> None:
    if isinstance(t, dict):
        fr = t.get("from_role") or "?"
        to = t.get("to_role") or "?"
        st.markdown(f"**{clean_display_text(str(fr))} → {clean_display_text(str(to))}**")
        cond = t.get("condition_skills") or []
        if cond:
            st.write(f"Compétences attendues : {', '.join(clean_display_text(str(s)) for s in cond)}")
        matched = t.get("matched_condition_skills") or []
        if matched:
            st.write(f"Compétences validées : {', '.join(clean_display_text(str(s)) for s in matched)}")
        coverage = t.get("condition_coverage")
        if coverage is not None:
            try:
                st.write(f"Couverture : {int(float(coverage) * 100)}%")
            except (TypeError, ValueError):
                st.write(f"Couverture : {coverage}")
        rationale = t.get("rationale")
        if rationale:
            st.write(f"Raison : {clean_display_text(str(rationale))}")
        else:
            st.write("Raison : Information non disponible")
    else:
        st.write(f"- {clean_display_text(str(t))}")


# ── Decision card ─────────────────────────────────────────────────────────────

def render_decision_card(card: dict[str, Any]) -> None:
    cleaned = _clean_for_display(card)
    if not isinstance(cleaned, dict):
        return

    candidate_id = cleaned.get("candidate_id") or "candidat inconnu"
    raw_name = cleaned.get("candidate_name") or cleaned.get("full_name") or cleaned.get("name")
    title = clean_display_text(str(raw_name)) if raw_name else f"Candidat : {candidate_id}"

    status_raw = str(cleaned.get("recommendation_status") or "")
    status_label = _STATUS_LABELS.get(status_raw, status_raw or "n/a")

    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(f"ID candidat : `{candidate_id}`")

        score_v3 = cleaned.get("baseline_score_v3")
        if score_v3 is not None:
            st.metric("Score Matching V3 — baseline officielle", format_score(score_v3))
        st.write(f"**Statut :** {status_label}")

        summary = cleaned.get("short_decision_summary") or cleaned.get("decision_summary")
        if summary:
            st.write(clean_display_text(str(summary)))

        rf = cleaned.get("rf_score")
        xgb = cleaned.get("xgboost_score")
        if rf is not None or xgb is not None:
            with st.expander("Comparaison ML expérimentale", expanded=False):
                st.caption("Ces scores viennent de modèles expérimentaux sur pseudo-labels.")
                ml_cols = st.columns(2)
                ml_cols[0].metric("RF", format_score(rf))
                ml_cols[1].metric("XGBoost", format_score(xgb))

        transferability = cleaned.get("transferability") if isinstance(cleaned.get("transferability"), dict) else {}
        if transferability:
            gb = transferability.get("gaps_bloquants") or []
            gc = transferability.get("gaps_compensables") or []
            if gb:
                st.write("**Gaps bloquants :**")
                for g in gb:
                    st.markdown(f"- :red[{clean_display_text(str(g))}]")
            if gc:
                st.write("**Gaps compensables :**")
                for g in gc:
                    st.markdown(f"- :orange[{clean_display_text(str(g))}]")

        with st.expander("Détails JSON", expanded=False):
            st.json(cleaned)


# ── Utility helpers ───────────────────────────────────────────────────────────

def format_score(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _absolute_api_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    base_url = str(st.session_state.get("api_base_url") or DEFAULT_API_BASE_URL).rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{base_url}{normalized_path}"


def _split_field_values(value: Any) -> list[str]:
    text = str(value or "")
    items: list[str] = []
    for line in text.replace(";", "\n").splitlines():
        cleaned = line.strip().lstrip("-*").strip()
        if not cleaned:
            continue
        items.extend(piece.strip() for piece in cleaned.split(",") if piece.strip())
    return items


def _format_list(values: Any) -> str:
    if isinstance(values, list) and values:
        return clean_display_text(", ".join(str(v) for v in values))
    return "non renseigné"


def _format_years(value: Any) -> str:
    if value is None:
        return "non renseignée"
    try:
        return f"{int(value)} ans"
    except (TypeError, ValueError):
        return str(value)


def _short_session_id(session_id: str) -> str:
    if not session_id:
        return "n/a"
    return session_id[:8] + "..."


def clean_display_text(value: Any) -> str:
    text = str(value)
    for bad, replacement in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, replacement)
    return text


def _clean_for_display(value: Any) -> Any:
    if isinstance(value, dict):
        return {clean_display_text(k): _clean_for_display(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_for_display(item) for item in value]
    if isinstance(value, str):
        return clean_display_text(value)
    return value


if __name__ == "__main__":
    main()
