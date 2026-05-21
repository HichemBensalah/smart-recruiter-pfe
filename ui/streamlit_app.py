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
EXAMPLE_PROMPTS = [
    "Nouvelle offre",
    "Pourquoi le premier candidat ?",
    "Compare le premier et le deuxième candidat",
    "Quels sont les gaps du meilleur candidat ?",
]
MOJIBAKE_REPLACEMENTS = {
    "\u00c3\u00a9": "é",
    "\u00c3\u00a8": "è",
    "\u00c3\u00aa": "ê",
    "\u00c3\u00ab": "ë",
    "\u00c3\u00a0": "à",
    "\u00c3\u00a2": "â",
    "\u00c3\u00b9": "ù",
    "\u00c3\u00bb": "û",
    "\u00c3\u00b4": "ô",
    "\u00c3\u00b6": "ö",
    "\u00c3\u00ae": "î",
    "\u00c3\u00af": "ï",
    "\u00c3\u00a7": "ç",
    "\u00c3\u2030": "É",
    "\u00c2": "",
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\ufffd": '"',
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u201d": "-",
}


def main() -> None:
    st.set_page_config(page_title="Smart Recruiter Copilot RH", page_icon="SR", layout="wide")
    st.title("Smart Recruiter - Talent Intelligence Copilot RH")
    st.caption("Interface de démonstration RH connectée à FastAPI, LangGraph et Matching V3.")
    st.info("Créez une offre, confirmez-la, puis explorez candidats, Decision Cards et transferability.")

    init_chat_history()
    api_base_url, api_key = render_sidebar()
    render_chat_history()

    prompt = st.chat_input("Décrivez votre besoin recruteur...")
    if prompt:
        submit_message(api_base_url, prompt, api_key)


def render_sidebar() -> tuple[str, str]:
    st.sidebar.header("Configuration")
    api_base_url = st.sidebar.text_input("URL de l'API", value=DEFAULT_API_BASE_URL).rstrip("/")
    api_key = st.sidebar.text_input("API key API", value=DEFAULT_API_KEY, type="password")

    if st.sidebar.button("Vérifier API"):
        health = call_health_api(api_base_url, api_key)
        if health.get("error"):
            st.sidebar.error(health["error"])
        else:
            deps = health.get("dependencies") if isinstance(health.get("dependencies"), dict) else {}
            st.sidebar.success(f"API disponible : {health.get('service')} / {health.get('version')}")
            if deps:
                st.sidebar.caption(f"Dépendances suivies : {', '.join(deps.keys())}")
            warnings = health.get("warnings") if isinstance(health.get("warnings"), list) else []
            if warnings:
                st.sidebar.warning("\n".join(clean_display_text(str(warning)) for warning in warnings))

    render_session_state_panel()

    if st.sidebar.button("Nouvelle offre", type="primary"):
        st.session_state.messages = []
        st.session_state.last_payload = {}
        st.session_state.session_id = str(uuid4())
        payload = call_chat_api(api_base_url, "nouvelle offre", api_key)
        add_message("user", "Nouvelle offre")
        add_message("assistant", payload.get("answer", ""), payload=payload)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Actions rapides")
    for example in EXAMPLE_PROMPTS:
        if st.sidebar.button(example, key=f"quick_{example}"):
            submit_message(api_base_url, example, api_key)
            st.rerun()

    return api_base_url, api_key


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
        with st.spinner("Analyse du besoin recruteur..."):
            payload = call_chat_api(api_base_url, message, api_key)
        if payload.get("error"):
            st.error(payload["error"])
            add_message("assistant", payload["error"], payload=payload)
        else:
            render_copilot_response(payload)
            add_message("assistant", payload.get("answer", ""), payload=payload)


def add_message(role: str, content: str, payload: dict[str, Any] | None = None) -> None:
    st.session_state.messages.append({"role": role, "content": content, "payload": payload or {}})


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
    filled = sum(1 for step in steps if fields.get(step))
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
    st.sidebar.write(f"Mode : `{metadata.get('matching_mode') or 'non disponible'}`")
    st.sidebar.write(f"Job resolu : `{metadata.get('resolved_job_id') or 'non disponible'}`")
    st.sidebar.write(f"Artefact : `{metadata.get('artifact_source') or 'non disponible'}`")
    st.sidebar.write(f"Fallback : `{'oui' if metadata.get('fallback_used') else 'non'}`")
    metadata_warnings = metadata.get("warnings") if isinstance(metadata.get("warnings"), list) else []
    if metadata_warnings:
        st.sidebar.warning("\n".join(clean_display_text(str(warning)) for warning in metadata_warnings))


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


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("payload") and not message["payload"].get("error"):
                render_copilot_response(message["payload"])
            else:
                st.write(message.get("content", ""))


def call_health_api(api_base_url: str, api_key: str = "") -> dict[str, Any]:
    try:
        response = requests.get(f"{api_base_url}/health", headers=build_api_headers(api_key), timeout=10)
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
    try:
        response = requests.post(
            f"{api_base_url}/api/chat",
            json={"message": message, "session_id": st.session_state.session_id},
            headers=build_api_headers(api_key),
            timeout=60,
        )
        if response.status_code >= 400:
            return {"error": f"/api/chat a retourne HTTP {response.status_code}: {extract_error_message(response)}"}
        payload = response.json()
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


def render_copilot_response(payload: dict[str, Any]) -> None:
    st.markdown(clean_display_text(payload.get("answer") or "Aucune réponse textuelle retournée."))
    render_job_intake(payload)
    render_matching_metadata(payload)
    render_payload_warnings(payload)

    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    if candidates:
        st.subheader("Candidats recommandés")
        for candidate in candidates:
            render_candidate(candidate)

    transferability = payload.get("transferability") if isinstance(payload.get("transferability"), dict) else {}
    if transferability:
        st.subheader("Transferability et gaps")
        render_transferability(transferability)

    decision_cards = payload.get("decision_cards") if isinstance(payload.get("decision_cards"), list) else []
    if decision_cards:
        with st.expander("Decision Cards", expanded=False):
            for card in decision_cards:
                render_decision_card(card)

    sources = payload.get("sources") if isinstance(payload.get("sources"), list) else []
    if sources:
        st.caption("Sources/tools: " + ", ".join(str(source) for source in sources))


def render_payload_warnings(payload: dict[str, Any]) -> None:
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    if warnings:
        st.warning("\n".join(clean_display_text(str(warning)) for warning in warnings))


def render_matching_metadata(payload: dict[str, Any]) -> None:
    metadata = payload.get("matching_metadata") if isinstance(payload.get("matching_metadata"), dict) else {}
    if not metadata:
        return
    with st.expander("Metadata matching / fallback", expanded=False):
        st.write(f"matching_mode: `{metadata.get('matching_mode') or 'non disponible'}`")
        st.write(f"resolved_job_id: `{metadata.get('resolved_job_id') or 'non disponible'}`")
        st.write(f"artifact_source: `{metadata.get('artifact_source') or 'non disponible'}`")
        st.write(f"fallback_used: `{bool(metadata.get('fallback_used', False))}`")
        metadata_warnings = metadata.get("warnings") if isinstance(metadata.get("warnings"), list) else []
        if metadata_warnings:
            st.warning("\n".join(clean_display_text(str(warning)) for warning in metadata_warnings))


def render_candidate(candidate: dict[str, Any]) -> None:
    candidate_id = candidate.get("candidate_id") or "candidate inconnu"
    display_name = clean_display_text(candidate.get("candidate_name") or candidate.get("full_name") or candidate.get("name") or candidate_id)
    with st.container(border=True):
        st.markdown(f"**{display_name}**")
        if display_name != candidate_id:
            st.caption(f"ID candidat: {candidate_id}")
        else:
            st.caption("Nom: non disponible dans les artefacts actuels")
        cols = st.columns(4)
        cols[0].metric("Matching V3", format_score(candidate.get("baseline_score_v3")))
        cols[1].metric("Random Forest", format_score(candidate.get("rf_score")))
        cols[2].metric("XGBoost", format_score(candidate.get("xgboost_score")))
        cols[3].metric("Statut", str(candidate.get("recommendation_status") or "n/a"))

        summary = candidate.get("short_decision_summary")
        if summary:
            st.write(clean_display_text(str(summary)))


def render_job_intake(payload: dict[str, Any]) -> None:
    intake = payload.get("job_intake_state") if isinstance(payload.get("job_intake_state"), dict) else {}
    structured_profile = payload.get("structured_job_profile") if isinstance(payload.get("structured_job_profile"), dict) else {}
    routed_job_id = payload.get("routed_job_id")
    if not intake and not structured_profile and not routed_job_id:
        return

    st.subheader("Offre en cours de création")
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
    filled = sum(1 for step in steps if fields.get(step))
    current_step = intake.get("current_step")
    st.progress(filled / len(steps))
    if current_step:
        step_index = steps.index(current_step) + 1 if current_step in steps else filled + 1
        st.caption(f"Étape {step_index}/6 - {labels.get(str(current_step), str(current_step))}")
    else:
        st.caption("Progression 6/6 - offre structurée prête pour confirmation")

    with st.expander("Champs déjà remplis", expanded=False):
        for step in steps:
            st.write(f"**{labels[step]}**")
            st.write(fields.get(step) or "_à compléter_")

    if structured_profile:
        with st.expander("structured_job_profile", expanded=True):
            st.json(structured_profile)
    if routed_job_id:
        st.info(f"Job profile routé : `{routed_job_id}`")


def render_decision_card(card: dict[str, Any]) -> None:
    cleaned = _clean_for_display(card)
    if not isinstance(cleaned, dict):
        return
    candidate_id = cleaned.get("candidate_id") or "candidate inconnu"
    title = cleaned.get("candidate_name") or cleaned.get("full_name") or candidate_id
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(f"ID candidat: `{candidate_id}`")
        cols = st.columns(4)
        cols[0].metric("Matching V3", format_score(cleaned.get("baseline_score_v3")))
        cols[1].metric("RF", format_score(cleaned.get("rf_score")))
        cols[2].metric("XGBoost", format_score(cleaned.get("xgboost_score")))
        cols[3].metric("Décision", str(cleaned.get("recommendation_status") or "n/a"))

        summary = cleaned.get("short_decision_summary") or cleaned.get("decision_summary")
        if summary:
            st.write(summary)

        transferability = cleaned.get("transferability") if isinstance(cleaned.get("transferability"), dict) else {}
        if transferability:
            gap_cols = st.columns(2)
            gap_cols[0].write("Gaps compensables")
            gap_cols[0].write(transferability.get("gaps_compensables", []))
            gap_cols[1].write("Gaps bloquants")
            gap_cols[1].write(transferability.get("gaps_bloquants", []))

        with st.expander("Détails JSON", expanded=False):
            st.json(cleaned)


def render_transferability(transferability: dict[str, Any]) -> None:
    for candidate_id, payload in transferability.items():
        selected = payload.get("selected_source") if isinstance(payload, dict) else None
        raw = payload.get(selected) if isinstance(payload, dict) and selected else payload
        details = extract_transferability_details(raw)
        with st.expander(str(candidate_id)):
            cols = st.columns(2)
            cols[0].write(f"fit_direct: `{details.get('fit_direct', 'n/a')}`")
            score = details.get("transferability_score") or details.get("coverage_score")
            cols[1].write(f"transferability_score: `{format_score(score)}`")
            st.write("gaps_compensables:", _clean_for_display(details.get("gaps_compensables", [])))
            st.write("gaps_bloquants:", _clean_for_display(details.get("gaps_bloquants", [])))
            st.write("transitions_plausibles:", _clean_for_display(details.get("transitions_plausibles") or details.get("plausible_transitions", [])))


def extract_transferability_details(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("transferability")
    if isinstance(nested, dict):
        return nested
    return payload


def format_score(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


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
        return clean_display_text(", ".join(str(value) for value in values))
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
        return {clean_display_text(key): _clean_for_display(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_for_display(item) for item in value]
    if isinstance(value, str):
        return clean_display_text(value)
    return value


if __name__ == "__main__":
    main()
