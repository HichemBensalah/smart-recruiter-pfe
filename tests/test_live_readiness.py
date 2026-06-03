"""Live matching readiness diagnostic tests."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.chatbot.live_readiness import check_live_matching_readiness


def test_live_readiness_all_ok():
    """All infrastructure available."""
    with patch("src.core.chatbot.live_readiness._check_mongodb") as mock_mongo:
        with patch("src.core.chatbot.live_readiness._check_faiss_index") as mock_faiss:
            with patch("src.core.chatbot.live_readiness._check_id_map") as mock_idmap:
                with patch("src.core.chatbot.live_readiness._check_sentence_transformer") as mock_st:
                    def setup_mongo(u, d, r, reasons):
                        r["mongodb_available"] = True
                        r["candidate_profiles_count"] = 42
                    def setup_faiss(p, r, reasons):
                        r["faiss_index_available"] = True
                        r["faiss_index_size"] = 512000
                    def setup_idmap(p, r, reasons):
                        r["id_map_available"] = True
                        r["id_map_size"] = 42
                    def setup_st(r, reasons):
                        r["sentence_transformer_available"] = True

                    mock_mongo.side_effect = setup_mongo
                    mock_faiss.side_effect = setup_faiss
                    mock_idmap.side_effect = setup_idmap
                    mock_st.side_effect = setup_st

                    result = check_live_matching_readiness()
                    assert result["live_ready"] is True
                    assert result["mongodb_available"] is True
                    assert result["candidate_profiles_count"] == 42
                    assert result["faiss_index_available"] is True
                    assert result["id_map_available"] is True
                    assert result["sentence_transformer_available"] is True
                    assert len(result["blocking_reasons"]) == 0


def test_live_readiness_mongodb_unavailable():
    """MongoDB unavailable."""
    with patch("src.core.chatbot.live_readiness._check_mongodb") as mock_mongo:
        with patch("src.core.chatbot.live_readiness._check_faiss_index"):
            with patch("src.core.chatbot.live_readiness._check_id_map"):
                with patch("src.core.chatbot.live_readiness._check_sentence_transformer"):
                    def setup_mongo(u, d, r, reasons):
                        reasons.append("MongoDB unavailable: connection refused")

                    mock_mongo.side_effect = setup_mongo
                    result = check_live_matching_readiness()
                    assert result["live_ready"] is False
                    assert result["mongodb_available"] is False
                    assert any("MongoDB unavailable" in r for r in result["blocking_reasons"])


def test_live_readiness_faiss_missing():
    """FAISS index missing."""
    with patch("src.core.chatbot.live_readiness._check_mongodb") as mock_mongo:
        with patch("src.core.chatbot.live_readiness._check_faiss_index") as mock_faiss:
            with patch("src.core.chatbot.live_readiness._check_id_map"):
                with patch("src.core.chatbot.live_readiness._check_sentence_transformer"):
                    def setup_mongo(u, d, r, reasons):
                        r["mongodb_available"] = True
                        r["candidate_profiles_count"] = 42
                    def setup_faiss(p, r, reasons):
                        reasons.append(f"FAISS index not found: {p}")

                    mock_mongo.side_effect = setup_mongo
                    mock_faiss.side_effect = setup_faiss
                    result = check_live_matching_readiness()
                    assert result["live_ready"] is False
                    assert result["faiss_index_available"] is False
                    assert any("FAISS index not found" in r for r in result["blocking_reasons"])


def test_live_readiness_candidate_profiles_empty():
    """candidate_profiles collection empty."""
    with patch("src.core.chatbot.live_readiness._check_mongodb") as mock_mongo:
        with patch("src.core.chatbot.live_readiness._check_faiss_index"):
            with patch("src.core.chatbot.live_readiness._check_id_map"):
                with patch("src.core.chatbot.live_readiness._check_sentence_transformer"):
                    def setup_mongo(u, d, r, reasons):
                        r["mongodb_available"] = True
                        r["candidate_profiles_count"] = 0
                        reasons.append("candidate_profiles collection is empty")

                    mock_mongo.side_effect = setup_mongo
                    result = check_live_matching_readiness()
                    assert result["live_ready"] is False
                    assert result["candidate_profiles_count"] == 0
                    assert any("empty" in r for r in result["blocking_reasons"])
