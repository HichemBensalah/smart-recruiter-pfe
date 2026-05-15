from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_MODEL_CACHE_DIR = Path("data/indexes/faiss/hf_cache")


def load_cross_encoder_model(model_name: str | None = None) -> Any:
    resolved_model = model_name or DEFAULT_CROSS_ENCODER_MODEL
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: sentence-transformers is required for CrossEncoder reranking."
        ) from exc

    DEFAULT_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(DEFAULT_MODEL_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_MODEL_CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        local_model_path = _resolve_local_model_path(resolved_model)
        return CrossEncoder(str(local_model_path), cache_folder=str(DEFAULT_MODEL_CACHE_DIR))
    except Exception as exc:
        raise RuntimeError(_model_load_error(resolved_model, exc)) from exc


def rerank_candidates_with_cross_encoder(
    job_text: str,
    candidates: list[dict[str, Any]],
    candidate_text_key: str = "candidate_text",
    top_k: int = 10,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    if not job_text or not job_text.strip():
        return _with_fallback_ranks(candidates, "empty_job_text")

    scoped_candidates = list(candidates[: max(top_k, 0)])
    if not scoped_candidates:
        return []

    valid_pairs: list[tuple[str, str]] = []
    valid_indices: list[int] = []
    output = [dict(candidate) for candidate in scoped_candidates]

    for index, candidate in enumerate(output):
        candidate.setdefault("original_faiss_rank", candidate.get("faiss_rank"))
        candidate.setdefault("original_faiss_score", candidate.get("faiss_score"))
        candidate_text = candidate.get(candidate_text_key)
        if not isinstance(candidate_text, str) or not candidate_text.strip():
            candidate["cross_encoder_score"] = None
            candidate["cross_encoder_score_normalized"] = 0.0
            candidate["cross_encoder_error"] = "empty_candidate_text"
            continue
        valid_pairs.append((job_text, candidate_text))
        valid_indices.append(index)

    if not valid_pairs:
        return _with_fallback_ranks(output, "no_valid_candidate_text")

    try:
        model = load_cross_encoder_model(model_name)
        raw_scores = model.predict(valid_pairs)
    except Exception as exc:
        return _with_fallback_ranks(output, str(exc))

    scores = [float(score) for score in raw_scores]
    normalized_scores = _min_max_normalize(scores)
    for candidate_index, raw_score, normalized_score in zip(valid_indices, scores, normalized_scores):
        output[candidate_index]["cross_encoder_score"] = round(raw_score, 6)
        output[candidate_index]["cross_encoder_score_normalized"] = round(normalized_score, 4)
        output[candidate_index].pop("cross_encoder_error", None)

    output.sort(
        key=lambda item: (
            item.get("cross_encoder_score") is not None,
            float(item.get("cross_encoder_score") or -1_000_000.0),
            float(item.get("faiss_score") or item.get("original_faiss_score") or 0.0),
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(output, start=1):
        candidate["cross_encoder_rank"] = rank
    return output


def _with_fallback_ranks(candidates: list[dict[str, Any]], error: str) -> list[dict[str, Any]]:
    output = []
    for rank, candidate in enumerate(candidates, start=1):
        item = dict(candidate)
        item.setdefault("original_faiss_rank", item.get("faiss_rank") or rank)
        item.setdefault("original_faiss_score", item.get("faiss_score"))
        item["cross_encoder_rank"] = None
        item["cross_encoder_score"] = None
        item["cross_encoder_score_normalized"] = None
        item["cross_encoder_error"] = error
        output.append(item)
    return output


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [0.5 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def _model_load_error(model_name: str, exc: Exception) -> str:
    return (
        f"CrossEncoder model '{model_name}' is not available locally or could not be loaded. "
        "Install/download it before running reranking, or run without --use-cross-encoder. "
        f"Original error: {exc}"
    )


def _resolve_local_model_path(model_name: str) -> Path | str:
    model_path = Path(model_name)
    if model_path.exists():
        return model_path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Missing dependency: huggingface_hub is required to locate cached CrossEncoder models.") from exc
    return Path(
        snapshot_download(
            repo_id=model_name,
            cache_dir=str(DEFAULT_MODEL_CACHE_DIR),
            local_files_only=True,
        )
    )
