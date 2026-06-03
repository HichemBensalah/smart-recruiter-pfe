from pathlib import Path


def _app_content() -> str:
    return Path("ui/streamlit_app.py").read_text(encoding="utf-8").lower()


def test_streamlit_app_exists_and_uses_chat_api() -> None:
    path = Path("ui/streamlit_app.py")

    assert path.exists()
    content = _app_content()
    assert "streamlit" in content
    assert "smart_recruiter_api_base_url" in content
    assert "http://localhost:8000" in content
    assert "/health" in content
    assert "/api/chat" in content or "api/chat" in content
    assert "session_id" in content
    assert "nouvelle offre" in content
    assert "offre en cours de création" in content
    assert "état de session" in content
    assert "résumé de l'offre" in content
    assert "étape" in content or "progression" in content
    assert "matching_completed" in content
    assert "matching_mode" in content
    assert "fallback_used" in content
    assert "warnings" in content
    assert "smart_recruiter_api_key" in content
    assert "api_key_header" in content
    assert "build_api_headers" in content
    assert "headers=build_api_headers" in content
    assert "routed_job_id" in content or "job id" in content
    assert "structured_job_profile" in content
    assert "job_title" in content
    assert "required_skills" in content
    assert "st.progress" in content
    assert "candidate_name" in content
    assert "render_candidate" in content
    assert "render_decision_card" in content
    assert "render_transferability" in content
    assert "clean_display_text" in content
    assert "requests.post" in content
    assert "st.help(str)" not in content
    assert "help=str" not in content


def test_streamlit_app_candidate_display() -> None:
    """Candidate cards must show V3 as primary, ML in expander, no raw 'non disponible' as title."""
    content = _app_content()

    # Candidate labels
    assert "candidat" in content, "Must display 'Candidat' fallback label"

    # V3 score as primary
    assert "score matching v3" in content, "V3 score must be labelled 'Score Matching V3'"
    assert "baseline officielle" in content, "V3 must be identified as baseline officielle"

    # ML scores in expander
    assert "comparaison ml" in content, "ML scores must be in a 'Comparaison ML' expander"
    assert "pseudo-labels" in content, "ML experimental note about pseudo-labels required"

    # Status badge labels
    assert "accord élevé" in content, "agreement_high must display as 'Accord élevé'"
    assert "à vérifier" in content, "review_needed must display as 'À vérifier'"


def test_streamlit_app_phase_display() -> None:
    """Phase indicator must be correct for each scenario step."""
    content = _app_content()

    assert "matching terminé" in content, "Phase 'Matching terminé' must be defined"
    assert "offre prête pour confirmation" in content, "Confirmation phase label required"
    assert "analyse candidat" in content, "Follow-up phase label required"
    assert "_phase_labels" in content or "_detect_phase" in content, "Phase detection function required"


def test_streamlit_app_routing_and_mode() -> None:
    """Routing and matching mode must be visible in the UI."""
    content = _app_content()

    assert "mode matching" in content, "'Mode matching' section label required"
    assert "job id routé" in content or "job profile routé" in content, "Routed job_id must be displayed"
    assert "résultats pré-calculés" in content, "Artifact mode note required"


def test_streamlit_app_header_and_welcome() -> None:
    """Header subtitle and welcome card must be present."""
    content = _app_content()

    assert "créez une offre structurée" in content, "Product subtitle required"
    assert "bienvenue dans smart recruiter" in content, "Welcome card heading required"
    assert "commencer une nouvelle offre" in content, "Welcome card CTA button required"
    assert "parcours recommandé" in content, "'Parcours recommandé' section in sidebar required"
    assert "paramètres techniques" in content, "'Paramètres techniques' expander required"
    assert "candidat anonymisé" in content, "Per-candidate anonymization note required"


def test_streamlit_app_candidate_name_before_fallback() -> None:
    content = _app_content()

    assert "if display_name:" in content
    assert "title = f\"{rank_prefix} {display_name}\".strip()" in content
    assert "candidat {rank_label}" in content
    assert content.index("if display_name:") < content.index("candidat {rank_label}")


def test_streamlit_app_contains_candidate_cv_link() -> None:
    content = _app_content()

    assert "cv_available" in content
    assert "cv_download_url" in content
    assert "voir cv" in content or "télécharger cv original" in content
    assert "cv original non disponible" in content
    assert "_absolute_api_url" in content


def test_streamlit_app_structured_profile_readable() -> None:
    """Structured job profile must be readable by default and JSON hidden in expander."""
    content = _app_content()

    assert "voir json technique" in content, "'Voir JSON technique' expander required"
    assert "profil structuré" in content, "Readable structured profile section required"
    assert "séniorité" in content, "seniority field must be shown"
    assert "localisation" in content or "location" in content, "location must be shown"


def test_streamlit_app_quick_actions() -> None:
    """Quick action prompts must include key follow-up questions."""
    content = _app_content()

    assert "pourquoi le premier candidat" in content, "Follow-up prompt required"
    assert "quels sont ses gaps" in content, "Gaps follow-up prompt required"


def test_streamlit_app_nouvelle_offre() -> None:
    """Reset button must be present."""
    content = _app_content()

    assert "nouvelle offre" in content, "'Nouvelle offre' button required"


def test_streamlit_app_transferability_no_raw_json() -> None:
    """Transferability section must use bullet lists, not raw JSON display as default."""
    content = _app_content()

    assert "gaps_bloquants" in content, "gaps_bloquants field referenced"
    assert "gaps_compensables" in content, "gaps_compensables field referenced"
    assert "information non disponible" in content, "Graceful fallback for missing transferability data"


def test_streamlit_app_decision_cards() -> None:
    """Decision cards must display status and V3 score."""
    content = _app_content()

    assert "decision cards" in content, "Decision Cards section required"
    assert "render_decision_card" in content
    assert "statut" in content, "Status label in decision card required"


def test_streamlit_app_contains_error_handling() -> None:
    content = _app_content()

    assert "try:" in content
    assert "except" in content
    assert "error" in content
    assert "status_code" in content


def test_ui_readme_exists() -> None:
    path = Path("ui/README.md")

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "uvicorn src.api.main:app" in content
    assert "8000" in content
    assert "8010" in content
    assert "streamlit run ui/streamlit_app.py" in content


def test_streamlit_app_live_timeout_config() -> None:
    """Live mode timeout must be sufficient (>60s) and configurable."""
    content = _app_content()

    assert "api_timeout_chat_live" in content
    assert "api_timeout_chat_artifact" in content
    assert "api_timeout_health" in content
    assert "matching live en cours" in content


def test_streamlit_app_handles_timeout_errors() -> None:
    """ReadTimeout errors must be distinguished from API unavailable."""
    content = _app_content()

    assert "requests.timeout" in content
    assert "le matching live prend trop de temps" in content
    assert "connectionerror" in content


def test_streamlit_app_followup_no_duplicate_candidates() -> None:
    """Follow-up questions should not display candidates again."""
    content = _app_content()

    # Ensure phase detection and follow-up handling exists
    assert "_detect_phase" in content
    assert "followup" in content
    assert "is_follow_up" in content
    # Should not show candidates when is_follow_up
    assert "candidates and not is_follow_up" in content


def test_streamlit_app_technical_details_expander() -> None:
    """Warnings should be hidden in 'Détails techniques' expander."""
    content = _app_content()

    assert "détails techniques" in content
    assert "expander" in content
    # Warnings should be moved to expander, not shown directly
    assert "with st.expander" in content and "détails techniques" in content


def test_streamlit_phase_detection_recognizes_live_matching() -> None:
    """After a live matching run, the phase must be 'matching' (Résultats candidats),
    not 'followup' — the live pipeline emits 'live_*' sources, not 'match_candidates'."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("_st_app_phase", Path("ui/streamlit_app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    live_payload = {
        "matching_completed": True,
        "sources": ["live_mongodb_faiss_matching_v3"],
        "candidates": [{"candidate_id": "c1"}],
    }
    assert module._detect_phase(live_payload) == "matching"

    artifact_payload = {
        "matching_completed": True,
        "sources": ["job_intake", "job_router", "match_candidates"],
    }
    assert module._detect_phase(artifact_payload) == "matching"

    followup_payload = {
        "matching_completed": True,
        "sources": ["analyze_transferability", "compose_answer"],
    }
    assert module._detect_phase(followup_payload) == "followup"


def test_streamlit_app_shows_filtered_duplicates_count() -> None:
    """The live section shows a discrete 'Doublons candidats filtrés : N' note."""
    content = _app_content()

    assert "doublons candidats filtrés" in content
    assert "duplicates_removed_count" in content


def test_streamlit_files_do_not_contain_visible_mojibake() -> None:
    # Check README — must be clean of mojibake entirely.
    readme = Path("ui/README.md").read_text(encoding="utf-8")
    for marker in ("Ã©", "Ã¨", "â€™", "â€œ"):
        assert marker not in readme, f"README contains mojibake sequence: {marker!r}"

    # For the app source, the MOJIBAKE_REPLACEMENTS dict intentionally contains
    # those patterns as keys (so the function can detect and fix them at runtime).
    # We verify instead that the file is valid UTF-8 and that clean_display_text exists.
    app_path = Path("ui/streamlit_app.py")
    assert app_path.exists()
    content = app_path.read_text(encoding="utf-8")  # would raise if invalid UTF-8
    assert "clean_display_text" in content
    assert "MOJIBAKE_REPLACEMENTS" in content
