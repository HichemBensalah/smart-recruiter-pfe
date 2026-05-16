from __future__ import annotations

from fastapi import APIRouter

from src.api.utils import DEMO_EXECUTIVE_SUMMARY, DEMO_RUN_MANIFEST, DEMO_TOP10_SUMMARY, read_json_file, run_demo_script


router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.get("/executive-summary")
def get_executive_summary() -> dict:
    return read_json_file(DEMO_EXECUTIVE_SUMMARY, "demo executive summary")


@router.get("/top10-summary")
def get_top10_summary() -> dict:
    return read_json_file(DEMO_TOP10_SUMMARY, "demo top10 summary")


@router.get("/run-summary")
def get_run_summary() -> dict:
    return read_json_file(DEMO_RUN_MANIFEST, "demo run manifest")


@router.post("/run")
def run_demo() -> dict:
    return run_demo_script()
