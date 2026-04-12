from __future__ import annotations

from src.benchmark.ocr.config import DEFAULT_TESSERACT_LANG
from src.benchmark.ocr.dataset import BenchmarkSample
from src.benchmark.ocr.runners import OCRPrediction, render_document_to_images


class PyTesseractRunner:
    engine_name = "pytesseract"

    def __init__(self, lang: str = DEFAULT_TESSERACT_LANG) -> None:
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError("pytesseract is not installed. Install benchmark dependencies first.") from exc

        self._pytesseract = pytesseract
        self._lang = lang

    def extract(self, sample: BenchmarkSample) -> OCRPrediction:
        pages = render_document_to_images(sample.source_path)
        chunks = [
            self._pytesseract.image_to_string(image, lang=self._lang).strip()
            for image in pages
        ]
        return OCRPrediction(
            text="\n\n".join(chunk for chunk in chunks if chunk).strip(),
            metadata={"engine": self.engine_name, "page_count": len(pages), "lang": self._lang},
        )

