from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "benchmarks" / "ocr" / "outputs"
LOCAL_MODEL_ROOT = PROJECT_ROOT / "models" / "ocr_benchmark"
LOCAL_EASYOCR_MODEL_DIR = LOCAL_MODEL_ROOT / "easyocr"
LOCAL_EASYOCR_USER_NETWORK_DIR = LOCAL_MODEL_ROOT / "easyocr_user_network"
LOCAL_HF_CACHE_DIR = LOCAL_MODEL_ROOT / "huggingface"

DEFAULT_ENGINES = ["docling", "mistralocr", "trocr", "easyocr", "pytesseract"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_SUFFIXES = {".pdf"}

DEFAULT_TESSERACT_LANG = os.getenv("OCR_BENCHMARK_TESSERACT_LANG", "eng+fra")
DEFAULT_EASYOCR_LANGS = [
    lang.strip()
    for lang in os.getenv("OCR_BENCHMARK_EASYOCR_LANGS", "en,fr").split(",")
    if lang.strip()
]
DEFAULT_TROCR_MODEL = os.getenv("OCR_BENCHMARK_TROCR_MODEL", "microsoft/trocr-base-printed")
DEFAULT_TROCR_MAX_NEW_TOKENS = int(os.getenv("OCR_BENCHMARK_TROCR_MAX_NEW_TOKENS", "512"))
DEFAULT_PDF_DPI = int(os.getenv("OCR_BENCHMARK_PDF_DPI", "200"))
DEFAULT_MISTRAL_MODEL = os.getenv("OCR_BENCHMARK_MISTRAL_MODEL", "mistral-ocr-latest")
