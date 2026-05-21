from pathlib import Path
import re


def test_docker_files_exist() -> None:
    assert Path("Dockerfile").exists()
    assert Path("docker-compose.yml").exists()
    assert Path(".dockerignore").exists()


def test_readme_mentions_core_project_elements() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "Smart Recruiter" in content
    assert "FastAPI" in content
    assert "LangGraph" in content
    assert "Streamlit" in content
    assert "Docker" in content
    assert "Matching V3" in content


def test_docker_documentation_exists() -> None:
    path = Path("docs/architecture/docker_demo.md")

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "docker compose up --build" in content
    assert "http://localhost:8000/docs" in content
    assert "http://localhost:8501" in content


def test_docker_compose_declares_required_services() -> None:
    content = Path("docker-compose.yml").read_text(encoding="utf-8")

    for service in ["api:", "streamlit:", "neo4j:", "mongodb:"]:
        assert re.search(rf"^\s{{2}}{re.escape(service)}", content, flags=re.MULTILINE)


def test_docker_compose_exposes_core_ports_and_healthchecks() -> None:
    content = Path("docker-compose.yml").read_text(encoding="utf-8")

    for port in ['"8000:8000"', '"8501:8501"', '"7474:7474"', '"7687:7687"', '"27017:27017"']:
        assert port in content
    assert content.count("healthcheck:") >= 3


def test_docker_compose_declares_environment_for_services() -> None:
    content = Path("docker-compose.yml").read_text(encoding="utf-8")

    for variable in [
        "SMART_RECRUITER_API_BASE_URL",
        "AUTH_ENABLED",
        "SMART_RECRUITER_API_KEY",
        "API_KEY_HEADER",
        "MONGODB_URI",
        "MONGODB_DATABASE",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
    ]:
        assert variable in content
    assert "http://api:8000" in content
    assert "mongodb://mongodb:27017" in content
    assert "bolt://neo4j:7687" in content


def test_env_example_documents_docker_variables() -> None:
    content = Path(".env.example").read_text(encoding="utf-8")

    for variable in [
        "SMART_RECRUITER_API_BASE_URL",
        "AUTH_ENABLED",
        "SMART_RECRUITER_API_KEY",
        "API_KEY_HEADER",
        "MONGODB_URI",
        "MONGODB_CANDIDATES_COLLECTION",
        "MONGODB_CANDIDATE_PROFILES_COLLECTION",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
    ]:
        assert variable in content


def test_readme_run_exists() -> None:
    content = Path("README_RUN.md").read_text(encoding="utf-8")

    assert "docker compose up --build" in content
    assert "MongoDB" in content
    assert "Neo4j" in content
