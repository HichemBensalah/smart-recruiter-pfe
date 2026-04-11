from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

import rapidocr
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.pipeline.standard_pdf_pipeline import ThreadedPdfPipelineOptions


class DoclingParser:
    """
    Parser Docling minimal avec configuration OCR locale optionnelle.
    """

    def __init__(self, converter: Optional[DocumentConverter] = None, *, do_ocr: bool = True):
        self.ocr_model_dir = Path(
            os.getenv("RAPIDOCR_MODEL_DIR", Path(rapidocr.__file__).parent / "models")
        )
        self.ocr_model_paths = {
            "det": self.ocr_model_dir / "ch_PP-OCRv4_det_infer.onnx",
            "rec": self.ocr_model_dir / "ch_PP-OCRv4_rec_infer.onnx",
            "cls": self.ocr_model_dir / "ch_ppocr_mobile_v2.0_cls_infer.onnx",
            "keys": self.ocr_model_dir / "ppocr_keys_v1.txt",
        }
        self.ocr_models_ready = all(p.exists() for p in self.ocr_model_paths.values())

        if converter is not None:
            self._converter = converter
            return

        ocr_options = None
        if do_ocr and self.ocr_models_ready:
            ocr_options = RapidOcrOptions(
                backend="onnxruntime",
                det_model_path=str(self.ocr_model_paths["det"]),
                rec_model_path=str(self.ocr_model_paths["rec"]),
                cls_model_path=str(self.ocr_model_paths["cls"]),
                rec_keys_path=str(self.ocr_model_paths["keys"]),
            )

        if do_ocr and ocr_options is not None:
            pipeline_options = ThreadedPdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
        else:
            pipeline_options = ThreadedPdfPipelineOptions(do_ocr=do_ocr)

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }

        self._converter = DocumentConverter(format_options=format_options)

    def parse(self, file_path: str) -> str:
        result = self._converter.convert(file_path, raises_on_error=False)
        if result is None or result.document is None:
            raise RuntimeError(f"Docling: conversion échouée pour {file_path}")
        return result.document.export_to_markdown()

    def convert_to_dict(self, file_path: str) -> dict:
        result = self._converter.convert(file_path, raises_on_error=False)
        if result is None or result.document is None:
            raise RuntimeError(f"Docling: conversion échouée pour {file_path}")
        doc = result.document
        if hasattr(doc, "export_to_dict"):
            return doc.export_to_dict()
        if hasattr(doc, "model_dump"):
            return doc.model_dump()
        if hasattr(doc, "to_dict"):
            return doc.to_dict()
        raise RuntimeError("Docling: impossible d'exporter le document en dict")
