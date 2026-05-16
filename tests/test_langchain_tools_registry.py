from __future__ import annotations

from src.core.chatbot.tools.match_tools import match_candidates_tool
from src.core.chatbot.tools.registry import SMART_RECRUITER_TOOLS, get_smart_recruiter_tools


def test_registry_returns_non_empty_tool_list() -> None:
    tools = get_smart_recruiter_tools()

    assert tools
    assert tools is not SMART_RECRUITER_TOOLS
    assert len(tools) >= 8


def test_each_tool_has_name_description_and_is_callable() -> None:
    for tool in get_smart_recruiter_tools():
        assert tool.name
        assert tool.description
        assert callable(tool.invoke)


def test_expected_tool_names_are_registered() -> None:
    names = {tool.name for tool in get_smart_recruiter_tools()}

    assert "match_candidates" in names
    assert "get_candidate_profile" in names
    assert "get_decision_card" in names
    assert "get_transferability" in names
    assert "get_neo4j_transferability" in names
    assert "get_neo4j_gaps" in names
    assert "get_demo_executive_summary" in names
    assert "get_demo_top10_summary" in names
    assert "run_demo" in names


def test_match_tool_can_be_mocked(monkeypatch) -> None:
    def fake_match_candidates(job_description: str, top_k: int = 10):
        return {"items": [{"candidate_id": "candidate_1"}], "top_k": top_k, "job_description": job_description}

    monkeypatch.setattr("src.core.chatbot.tools.match_tools.match_candidates", fake_match_candidates)
    mocked_tool = match_candidates_tool
    mocked_tool.func = fake_match_candidates

    payload = mocked_tool.invoke({"job_description": "Backend Python", "top_k": 1})

    assert payload["items"][0]["candidate_id"] == "candidate_1"
    assert payload["top_k"] == 1
