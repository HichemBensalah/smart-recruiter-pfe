from __future__ import annotations

from src.benchmark.ocr.dataset import BenchmarkSample
from src.benchmark.ocr.runners import OCRPrediction, markdown_to_plain_text
from src.core.parser.docling_parser import DoclingParser


class DoclingRunner:
    engine_name = "docling"

    def __init__(self) -> None:
        self._parser = DoclingParser(do_ocr=True)

    def extract(self, sample: BenchmarkSample) -> OCRPrediction:
        markdown = self._parser.parse(str(sample.source_path))
        return OCRPrediction(
            text=markdown_to_plain_text(markdown),
            metadata={"engine": self.engine_name, "source_format": sample.source_kind},
        )

