"""OCR wrappers: EasyOCR (primary) and Tesseract (fallback)."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import numpy as np


@dataclass
class OCRConfig:
    languages: list[str] | None = None
    allowlist: str = ""


class OCRClient:
    """Unified OCR interface."""

    def text(self, image: np.ndarray, config: OCRConfig | None = None) -> str:
        raise NotImplementedError


class EasyOCRClient(OCRClient):
    """EasyOCR-based OCR client. Models are loaded lazily on first call."""

    def __init__(self, gpu: bool = False):
        self._gpu = gpu
        self._readers: dict[str, object] = {}

    def _get_reader(self, languages: list[str]):
        import easyocr

        key = ",".join(sorted(languages))
        if key not in self._readers:
            self._readers[key] = easyocr.Reader(languages, gpu=self._gpu)
        return self._readers[key]

    def text(self, image: np.ndarray, config: OCRConfig | None = None) -> str:
        config = config or OCRConfig()
        languages = config.languages or ["pt", "en"]
        reader = self._get_reader(languages)

        kwargs: dict = {"detail": 0, "paragraph": True}
        if config.allowlist:
            kwargs["allowlist"] = config.allowlist

        results = reader.readtext(image, **kwargs)
        if not results:
            return ""
        if isinstance(results[0], str):
            return "\n".join(results)
        return "\n".join(str(r) for r in results)


class TesseractClient(OCRClient):
    """Tesseract via pytesseract (fallback)."""

    def text(self, image: np.ndarray, config: OCRConfig | None = None) -> str:
        if not shutil.which("tesseract"):
            raise RuntimeError("tesseract nao encontrado no PATH")
        import pytesseract

        config = config or OCRConfig()
        lang_map = {"pt": "por", "en": "eng"}
        languages = config.languages or ["pt", "en"]
        tess_langs = "+".join(lang_map.get(la, la) for la in languages)

        tess_config = "--psm 7"
        if config.allowlist:
            tess_config += f" -c tessedit_char_whitelist={config.allowlist}"

        return pytesseract.image_to_string(image, lang=tess_langs, config=tess_config)


def create_ocr_client(backend: str | None = None, gpu: bool = False) -> OCRClient:
    """Create OCR client based on backend preference or env var OCR_BACKEND."""
    backend = backend or os.environ.get("OCR_BACKEND", "easyocr")
    if backend == "tesseract":
        return TesseractClient()
    return EasyOCRClient(gpu=gpu)


# ---- Per-process global for worker reuse ----

_worker_client: OCRClient | None = None


def init_worker_client(backend: str | None, gpu: bool) -> None:
    """Initialise the per-process OCR client (called once by ProcessPoolExecutor)."""
    global _worker_client
    _worker_client = create_ocr_client(backend=backend, gpu=gpu)


def get_worker_client() -> OCRClient:
    """Return the OCR client pre-loaded in this worker process."""
    if _worker_client is None:
        raise RuntimeError("worker OCR client not initialised — call init_worker_client first")
    return _worker_client
