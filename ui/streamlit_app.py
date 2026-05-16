from __future__ import annotations

from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE_URL = "http://localhost:8000"
EXAMPLE_PROMPTS = [
    "Je cherche un développeur backend Python FastAPI MongoDB",
    "Trouve-moi un profil Data Engineer Python SQL",
    "Quels candidats sont à vérifier ?",
]


def main() -> None:
    st.set_page_config(page_title="Smart Recruiter Copilot RH", page_icon="SR", layout="wide")
    st.title("Smart Recruiter — Talent Intelligence Copilot RH")
    st.caption("Interface de démonstration. Toute l'intelligence reste côté API FastAPI / LangGraph.")

    api_base_url = render_sidebar()
    init_chat_history()
    render_chat_history()

    prompt = st.chat_input("Décrivez votre besoin recruteur...")
    if prompt:
        add_message("user", prompt)
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyse du besoin recruteur..."):
                payload = call_chat_api(api_base_url, prompt)
            if payload.get("error"):
                st.error(payload["error"])
                add_message("assistant", payload["error"], payload=payload)
            else:
                render_copilot_response(payload)
                add_message("assistant", payload.get("answer", ""), payload=payload)


def render_sidebar() -> str:
    with st.sidebar:
        st.header("Configuration")
        api_base_url = st.text_input("URL de l'API", value=DEFAULT_API_BASE_URL).rstrip("/")
        if st.button("Vérifier API"):
            health = call_health_api(api_base_url)
            if health.get("error"):
                st.error(health["error"])
            else:
                st.success(f"API disponible: {health.get('service')} / {health.get('version')}")

        st.divider()
        st.subheader("Exemples de requêtes")
        for example in EXAMPLE_PROMPTS:
            st.code(example, language=None)

        st.divider()
        if st.button("Réinitialiser la conversation"):
            st.session_state.messages = []
            st.rerun()
    return api_base_url


def init_chat_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def add_message(role: str, content: str, payload: dict[str, Any] | None = None) -> None:
    st.session_state.messages.append({"role": role, "content": content, "payload": payload or {}})


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("payload") and not message["payload"].get("error"):
                render_copilot_response(message["payload"])
            else:
                st.write(message.get("content", ""))


def call_health_api(api_base_url: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{api_base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": f"API indisponible: {exc}"}
    except ValueError:
        return {"error": "La réponse /health n'est pas un JSON valide."}


def call_chat_api(api_base_url: str, message: str) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{api_base_url}/api/chat",
            json={"message": message},
            timeout=60,
        )
        if response.status_code >= 400:
            return {"error": f"/api/chat a retourné HTTP {response.status_code}: {extract_error_message(response)}"}
        payload = response.json()
    except requests.RequestException as exc:
        return {"error": f"Impossible d'appeler /api/chat: {exc}"}
    except ValueError:
        return {"error": "La réponse /api/chat n'est pas un JSON valide."}
    if not isinstance(payload, dict):
        return {"error": "La réponse /api/chat doit être un objet JSON."}
    return payload


def extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload)
    return str(payload)


def render_copilot_response(payload: dict[str, Any]) -> None:
    answer = payload.get("answer") or "Aucune réponse textuelle retournée."
    st.markdown(answer)

    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    if candidates:
        st.subheader("Candidats")
        for candidate in candidates:
            render_candidate(candidate)

    decision_cards = payload.get("decision_cards") if isinstance(payload.get("decision_cards"), list) else []
    if decision_cards:
        with st.expander("Decision Cards"):
            for card in decision_cards:
                st.json(card)

    transferability = payload.get("transferability") if isinstance(payload.get("transferability"), dict) else {}
    if transferability:
        st.subheader("Transférabilité")
        render_transferability(transferability)

    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    if warnings:
        st.warning("\n".join(str(warning) for warning in warnings))

    sources = payload.get("sources") if isinstance(payload.get("sources"), list) else []
    if sources:
        st.caption("Sources/tools: " + ", ".join(str(source) for source in sources))


def render_candidate(candidate: dict[str, Any]) -> None:
    candidate_id = candidate.get("candidate_id") or "candidate inconnu"
    with st.container(border=True):
        st.markdown(f"**{candidate_id}**")
        cols = st.columns(4)
        cols[0].metric("Matching V3", format_score(candidate.get("baseline_score_v3")))
        cols[1].metric("Random Forest", format_score(candidate.get("rf_score")))
        cols[2].metric("XGBoost", format_score(candidate.get("xgboost_score")))
        cols[3].metric("Statut", str(candidate.get("recommendation_status") or "n/a"))
        summary = candidate.get("short_decision_summary")
        if summary:
            st.write(summary)


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
            st.write("gaps_compensables:", details.get("gaps_compensables", []))
            st.write("gaps_bloquants:", details.get("gaps_bloquants", []))
            st.write("transitions_plausibles:", details.get("transitions_plausibles") or details.get("plausible_transitions", []))


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


if __name__ == "__main__":
    main()
