from __future__ import annotations

import os

from src.benchmark.ocr.config import DEFAULT_MISTRAL_MODEL
from src.benchmark.ocr.dataset import BenchmarkSample
from src.benchmark.ocr.runners import OCRPrediction, markdown_to_plain_text


class MistralOCRRunner:
    engine_name = "mistralocr"

    def __init__(self, model_name: str = DEFAULT_MISTRAL_MODEL) -> None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set.")

        try:
            try:
                from mistralai.client import Mistral
            except ImportError:
                from mistralai import Mistral  # type: ignore
        except ImportError as exc:
            raise RuntimeError("mistralai SDK is not installed. Install benchmark dependencies first.") from exc

        self._client = Mistral(api_key=api_key)
        self._model_name = model_name

    def extract(self, sample: BenchmarkSample) -> OCRPrediction:
        if not sample.public_url:
            raise RuntimeError(
                "Mistral OCR runner currently requires a public_url column in the benchmark manifest."
            )

        document = (
            {"type": "document_url", "document_url": sample.public_url}
            if sample.source_path.suffix.lower() == ".pdf"
            else {"type": "image_url", "image_url": sample.public_url}
        )

        response = self._client.ocr.process(
            model=self._model_name,
            document=document,
            include_image_base64=False,
        )
        pages = getattr(response, "pages", None) or response.get("pages", [])
        markdown = "\n\n".join(getattr(page, "markdown", None) or page.get("markdown", "") for page in pages).strip()

        return OCRPrediction(
            text=markdown_to_plain_text(markdown),
            metadata={"engine": self.engine_name, "page_count": len(pages), "model_name": self._model_name},
        )

