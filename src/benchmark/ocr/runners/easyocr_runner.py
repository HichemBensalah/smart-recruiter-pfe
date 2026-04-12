from __future__ import annotations

import numpy as np

from src.benchmark.ocr.config import (
    DEFAULT_EASYOCR_LANGS,
    LOCAL_EASYOCR_MODEL_DIR,
    LOCAL_EASYOCR_USER_NETWORK_DIR,
)
from src.benchmark.ocr.dataset import BenchmarkSample
from src.benchmark.ocr.runners import OCRPrediction, render_document_to_images


class EasyOCRRunner:
    engine_name = "easyocr"

    def __init__(self, langs: list[str] | None = None) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("easyocr is not installed. Install benchmark dependencies first.") from exc

        LOCAL_EASYOCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        LOCAL_EASYOCR_USER_NETWORK_DIR.mkdir(parents=True, exist_ok=True)
        self._reader = easyocr.Reader(
            langs or DEFAULT_EASYOCR_LANGS,
            gpu=False,
            model_storage_directory=str(LOCAL_EASYOCR_MODEL_DIR),
            user_network_directory=str(LOCAL_EASYOCR_USER_NETWORK_DIR),
            verbose=False,
        )

    def extract(self, sample: BenchmarkSample) -> OCRPrediction:
        pages = render_document_to_images(sample.source_path)
        page_chunks: list[str] = []
        for image in pages:
            results = self._reader.readtext(np.array(image), detail=0, paragraph=True)
            page_chunks.append("\n".join(str(item).strip() for item in results if str(item).strip()))
        return OCRPrediction(
            text="\n\n".join(chunk for chunk in page_chunks if chunk).strip(),
            metadata={"engine": self.engine_name, "page_count": len(pages)},
        )
