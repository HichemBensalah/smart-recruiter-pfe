from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from src.core.chatbot.tools.api_client import SmartRecruiterApiClient


class NoInput(BaseModel):
    pass


def get_demo_executive_summary() -> dict[str, Any]:
    """Return the executive demo summary."""
    client = SmartRecruiterApiClient()
    return client.get("/api/demo/executive-summary")


def get_demo_top10_summary() -> dict[str, Any]:
    """Return the detailed top 10 demo summary."""
    client = SmartRecruiterApiClient()
    return client.get("/api/demo/top10-summary")


def run_demo() -> dict[str, Any]:
    """Run the demo end-to-end script through the FastAPI facade and return the manifest."""
    client = SmartRecruiterApiClient()
    return client.post("/api/demo/run", {})


get_demo_executive_summary_tool = StructuredTool.from_function(
    func=get_demo_executive_summary,
    name="get_demo_executive_summary",
    description="Fetch the executive demo summary with top recommended and needs-review candidates.",
    args_schema=NoInput,
)

get_demo_top10_summary_tool = StructuredTool.from_function(
    func=get_demo_top10_summary,
    name="get_demo_top10_summary",
    description="Fetch the detailed top 10 demo summary with V3, ML, SHAP and transferability signals.",
    args_schema=NoInput,
)

run_demo_tool = StructuredTool.from_function(
    func=run_demo,
    name="run_demo",
    description="Run the Smart Recruiter demo report generation through the API and return the run manifest.",
    args_schema=NoInput,
)
