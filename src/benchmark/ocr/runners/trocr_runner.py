from __future__ import annotations

from src.benchmark.ocr.config import (
    DEFAULT_TROCR_MAX_NEW_TOKENS,
    DEFAULT_TROCR_MODEL,
    LOCAL_HF_CACHE_DIR,
)
from src.benchmark.ocr.dataset import BenchmarkSample
from src.benchmark.ocr.runners import OCRPrediction, render_document_to_images


class TrOCRRunner:
    engine_name = "trocr"

    def __init__(
        self,
        model_name: str = DEFAULT_TROCR_MODEL,
        *,
        max_new_tokens: int = DEFAULT_TROCR_MAX_NEW_TOKENS,
    ) -> None:
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RuntimeError("transformers/torch are not installed. Install benchmark dependencies first.") from exc

        self._torch = torch
        LOCAL_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=str(LOCAL_HF_CACHE_DIR))
        self._model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=str(LOCAL_HF_CACHE_DIR))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._max_new_tokens = max_new_tokens
        self._model_name = model_name

    def extract(self, sample: BenchmarkSample) -> OCRPrediction:
        pages = render_document_to_images(sample.source_path)
        chunks: list[str] = []
        for image in pages:
            pixel_values = self._processor(images=image, return_tensors="pt").pixel_values.to(self._device)
            generated_ids = self._model.generate(pixel_values, max_new_tokens=self._max_new_tokens)
            text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if text:
                chunks.append(text)

        return OCRPrediction(
            text="\n\n".join(chunks).strip(),
            metadata={
                "engine": self.engine_name,
                "page_count": len(pages),
                "model_name": self._model_name,
                "device": self._device,
            },
        )
