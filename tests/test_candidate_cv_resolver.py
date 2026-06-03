from __future__ import annotations

from pathlib import Path

from src.core.chatbot import candidate_cv_resolver as resolver
from src.core.chatbot.candidate_cv_resolver import resolve_candidate_cv


ML_JOB_ID = "machine_learning_python_nlp"


def test_resolve_candidate_cv_returns_image_73_for_ml_job() -> None:
    cv = resolve_candidate_cv("candidate_f74acce78f96", ML_JOB_ID)

    assert cv["cv_available"] is True
    assert cv["cv_filename"] == "Image_73.pdf"
    assert cv["cv_path"] == "data/raw_cv/pdf/Image_73.pdf"
    assert cv["cv_mime_type"] == "application/pdf"
    assert cv["cv_source"] == "matching_report"
    assert cv["cv_confidence"] == "high"


def test_resolve_candidate_cv_returns_hichem_image_for_ml_job() -> None:
    cv = resolve_candidate_cv("candidate_8eea1b635447", ML_JOB_ID)

    assert cv["cv_available"] is True
    assert cv["cv_filename"] == "Hichem_image.jpg"
    assert cv["cv_path"] == "data/raw_cv/images/Hichem_image.jpg"
    assert cv["cv_mime_type"] == "image/jpeg"


def test_resolve_candidate_cv_returns_jessica_pdf_for_ml_job() -> None:
    cv = resolve_candidate_cv("candidate_418b74b9d404", ML_JOB_ID)

    assert cv["cv_available"] is True
    assert cv["cv_filename"] == "2_Jessica.pdf"
    assert cv["cv_path"] == "data/raw_cv/pdf/2_Jessica.pdf"


def test_resolve_candidate_cv_prefers_ml_pdf_and_keeps_docx_alternative() -> None:
    cv = resolve_candidate_cv("candidate_1487f3187f7b", ML_JOB_ID)

    assert cv["cv_available"] is True
    assert cv["cv_filename"] == "Hichem_resume.pdf"
    assert cv["cv_path"] == "data/raw_cv/pdf/Hichem_resume.pdf"
    assert any(
        alternative["cv_path"] == "data/raw_cv/docx/Hichem_resume.docx"
        for alternative in cv["cv_alternatives"]
    )


def test_resolve_candidate_cv_returns_not_found_for_unknown_candidate() -> None:
    cv = resolve_candidate_cv("candidate_unknown_for_cv", ML_JOB_ID)

    assert cv == {
        "candidate_id": "candidate_unknown_for_cv",
        "cv_available": False,
        "cv_filename": None,
        "cv_path": None,
        "cv_mime_type": None,
        "cv_source": "not_found",
        "cv_confidence": "none",
        "cv_alternatives": [],
    }


def test_resolve_candidate_cv_refuses_path_outside_raw_cv(monkeypatch) -> None:
    outside_file = Path("README.md").resolve()

    def fake_read_json(path):
        return {
            "results": [
                {
                    "recommendations": [
                        {
                            "candidate_id": "candidate_escape",
                            "source_path": str(outside_file),
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(resolver, "_read_json", fake_read_json)
    monkeypatch.setattr(resolver, "_all_matching_report_paths", lambda: [])

    cv = resolve_candidate_cv("candidate_escape", "unsafe")

    assert cv["cv_available"] is False
