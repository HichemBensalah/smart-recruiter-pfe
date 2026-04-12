from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def compute_metrics(ground_truth: str, prediction: str) -> dict[str, float]:
    raw_gt = ground_truth or ""
    raw_pred = prediction or ""
    norm_gt = normalize_text(raw_gt)
    norm_pred = normalize_text(raw_pred)

    return {
        "raw_wer": word_error_rate(raw_gt, raw_pred),
        "raw_cer": char_error_rate(raw_gt, raw_pred),
        "normalized_wer": word_error_rate(norm_gt, norm_pred),
        "normalized_cer": char_error_rate(norm_gt, norm_pred),
    }


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    return _rate(ref_tokens, hyp_tokens)


def char_error_rate(reference: str, hypothesis: str) -> float:
    return _rate(list(reference), list(hypothesis))


def _rate(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(reference, hypothesis) / len(reference)


def _levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]

