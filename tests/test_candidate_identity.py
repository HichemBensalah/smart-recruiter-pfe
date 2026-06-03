from __future__ import annotations

from src.core.chatbot.candidate_identity import (
    enrich_candidate_with_display_name,
    enrich_candidates_with_display_names,
    resolve_candidate_display_name,
)


def test_resolve_candidate_display_name_from_grounded_profile() -> None:
    identity = resolve_candidate_display_name("candidate_f74acce78f96")

    assert identity["candidate_name"] == "SONA MAYERT"
    assert identity["display_name_source"] == "grounded_profile"
    assert identity["display_name_confidence"] == "high"


def test_resolve_candidate_display_name_from_decision_card() -> None:
    identity = resolve_candidate_display_name("candidate_1487f3187f7b")

    assert identity["candidate_name"] == "Hichem Bensalah"
    assert identity["display_name_source"] == "decision_card"
    assert identity["display_name_confidence"] == "high"


def test_resolve_candidate_display_name_for_aziz() -> None:
    identity = resolve_candidate_display_name("candidate_206d746034ef")

    assert identity["candidate_name"] == "MOHAMED AZIZ BELAWEID"
    assert identity["display_name_source"] == "decision_card"
    assert identity["display_name_confidence"] == "high"


def test_resolve_candidate_display_name_returns_null_for_unknown_candidate() -> None:
    identity = resolve_candidate_display_name("candidate_without_known_name")

    assert identity == {
        "candidate_id": "candidate_without_known_name",
        "candidate_name": None,
        "display_name_source": "anonymized_artifact",
        "display_name_confidence": "none",
    }


def test_enrich_candidate_with_display_name_does_not_change_scores() -> None:
    candidate = {
        "candidate_id": "candidate_f74acce78f96",
        "baseline_rank_v3": 1,
        "baseline_score_v3": 0.8017,
    }

    enriched = enrich_candidate_with_display_name(candidate)

    assert enriched["candidate_name"] == "SONA MAYERT"
    assert enriched["baseline_rank_v3"] == 1
    assert enriched["baseline_score_v3"] == 0.8017
    assert "candidate_name" not in candidate


def test_enrich_candidate_preserves_existing_valid_name_when_resolver_has_no_name() -> None:
    candidate = {
        "candidate_id": "candidate_without_known_name",
        "candidate_name": "Existing Real Name",
    }

    enriched = enrich_candidate_with_display_name(candidate)

    assert enriched["candidate_name"] == "Existing Real Name"
    assert enriched["display_name_source"] == "candidate_payload"
    assert enriched["display_name_confidence"] == "high"


def test_enrich_candidates_with_display_names_preserves_order() -> None:
    candidates = [
        {"candidate_id": "candidate_206d746034ef"},
        {"candidate_id": "candidate_without_known_name"},
        {"candidate_id": "candidate_1487f3187f7b"},
    ]

    enriched = enrich_candidates_with_display_names(candidates)

    assert [candidate["candidate_id"] for candidate in enriched] == [
        "candidate_206d746034ef",
        "candidate_without_known_name",
        "candidate_1487f3187f7b",
    ]
    assert [candidate["candidate_name"] for candidate in enriched] == [
        "MOHAMED AZIZ BELAWEID",
        None,
        "Hichem Bensalah",
    ]
