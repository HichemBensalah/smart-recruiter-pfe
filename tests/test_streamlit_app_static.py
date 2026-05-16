from pathlib import Path


def test_streamlit_app_exists_and_uses_chat_api() -> None:
    path = Path("ui/streamlit_app.py")

    assert path.exists()
    content = path.read_text(encoding="utf-8").lower()
    assert "streamlit" in content
    assert "/api/chat" in content or "api/chat" in content
    assert "requests.post" in content


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
    assert "streamlit run ui/streamlit_app.py" in content
