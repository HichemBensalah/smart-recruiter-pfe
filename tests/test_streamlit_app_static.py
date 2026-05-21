from pathlib import Path


def test_streamlit_app_exists_and_uses_chat_api() -> None:
    path = Path("ui/streamlit_app.py")

    assert path.exists()
    content = path.read_text(encoding="utf-8").lower()
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


def test_streamlit_app_contains_error_handling() -> None:
    content = Path("ui/streamlit_app.py").read_text(encoding="utf-8").lower()

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


def test_streamlit_files_do_not_contain_visible_mojibake() -> None:
    markers = ["Ã", "Â", "�", "â€™", "â€œ", "â€"]
    for path in (Path("ui/streamlit_app.py"), Path("ui/README.md")):
        content = path.read_text(encoding="utf-8")
        for marker in markers:
            assert marker not in content
