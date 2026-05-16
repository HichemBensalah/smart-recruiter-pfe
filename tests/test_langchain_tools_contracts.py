from pathlib import Path


def test_langchain_tools_contract_documentation_exists() -> None:
    path = Path("docs/architecture/langchain_tools_contracts.md")

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "match_candidates" in content
    assert "get_decision_card" in content
    assert "Neo4j" in content
    assert "fallback_recommended" in content
    assert "LangGraph" in content
