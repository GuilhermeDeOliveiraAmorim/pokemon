"""Extraction pipeline: orchestrates OCR on image regions and parses results."""

from __future__ import annotations

import gc
import re
import time
from dataclasses import dataclass, field
from itertools import chain
from typing import Iterable

import numpy as np

from . import log as _log
from .catalog import Record
from .imageutil import _make_ocr_variants, prepare_regions
from .ocr import OCRClient, OCRConfig
from .parsing import (
    CARD_NO_PATTERN,
    best_set_code_candidate,
    better_name,
    parse_bottom_text,
    sanitize_name,
)

_logger = _log.get("extract")


def _chain_first(first: list, rest: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
    """Yield items from *first* list then lazily from *rest* generator."""
    return chain(first, rest)


@dataclass
class Details:
    raw_text: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class ExtractOptions:
    default_quality: str = "M"
    default_language: str = "PT"
    default_quantity: int = 1


class Extractor:
    def __init__(self, ocr_client: OCRClient, options: ExtractOptions | None = None):
        self.ocr = ocr_client
        self.options = options or ExtractOptions()

    def extract(self, image_path: str) -> tuple[Record, Details]:
        t0 = time.perf_counter()
        _logger.info("[IMAGE] inicio: %s", image_path)
        record = Record(
            quantity=self.options.default_quantity,
            quality=self.options.default_quality.strip(),
            language=self.options.default_language.strip().upper(),
        )
        details = Details()

        t_reg = time.perf_counter()
        regions = prepare_regions(image_path)
        _logger.debug(
            "[REGIONS] %.3fs — card_found=%s, resolucao=%dx%d (original %dx%d): %s",
            time.perf_counter() - t_reg,
            regions.card_found,
            regions.width,
            regions.height,
            regions.source_width,
            regions.source_height,
            image_path,
        )

        # 1. Name
        if regions.name is not None:
            try:
                text = self._timed_ocr("nome", regions.name, OCRConfig(languages=["pt", "en"]))
                details.raw_text["nome"] = text
                record.card_pt = sanitize_name(text)
            except Exception as e:
                details.notes.append(f"falha no OCR do nome: {e}")

        # 2. Name alt
        if regions.name_alt is not None:
            try:
                text = self._timed_ocr("nome_alternativo", regions.name_alt, OCRConfig(languages=["en", "pt"]))
                details.raw_text["nome_alternativo"] = text
                record.card_pt = better_name(record.card_pt, sanitize_name(text))
            except Exception as e:
                details.notes.append(f"falha no OCR do nome alternativo: {e}")

        # 3. Bottom
        if regions.bottom is not None:
            try:
                text = self._timed_ocr("faixa_inferior", regions.bottom, OCRConfig(languages=["pt", "en"]))
                details.raw_text["faixa_inferior"] = text
                parse_bottom_text(text, image_path, record)
            except Exception as e:
                details.notes.append(f"falha no OCR da faixa inferior: {e}")

        # 4. Footer (try primary then variants) — skip if bottom already found number+sigla
        if regions.footer is not None and (not record.card_number.strip() or not record.edition_sigla.strip()):
            footer_config = OCRConfig(languages=["en"], allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/")
            footer_variants = _make_ocr_variants(regions.footer_raw, scale=6) if regions.footer_raw is not None else iter([])
            text = self._best_ocr_for_number(
                _chain_first([regions.footer], footer_variants),
                footer_config,
                details,
                "rodape",
            )
            if text:
                parse_bottom_text(text, image_path, record)

        # 5. Code (if sigla or number still missing)
        if not record.edition_sigla.strip() or not record.card_number.strip():
            if regions.code is not None:
                code_config = OCRConfig(languages=["en"], allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/")
                code_variants = _make_ocr_variants(regions.code_raw, scale=6) if regions.code_raw is not None else iter([])
                text = self._best_ocr_for_number(
                    _chain_first([regions.code], code_variants),
                    code_config,
                    details,
                    "codigo",
                )
                if text:
                    candidate = best_set_code_candidate(text)
                    if candidate:
                        record.edition_sigla = candidate
                    parse_bottom_text(text, image_path, record)

        # 6. Number (if still missing — try variants)
        if not record.card_number.strip() or not record.edition_card_count.strip():
            if regions.number is not None:
                number_config = OCRConfig(languages=["en"], allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/")
                number_variants = _make_ocr_variants(regions.number_raw, scale=4) if regions.number_raw is not None else iter([])
                text = self._best_ocr_for_number(
                    _chain_first([regions.number], number_variants),
                    number_config,
                    details,
                    "numeracao",
                )
                if text:
                    parse_bottom_text(text, image_path, record)

        # 7. Notes for missing fields
        if not record.card_pt.strip():
            details.notes.append("nome da carta nao foi detectado")
        if not record.edition_sigla.strip():
            details.notes.append("sigla da edicao nao foi detectada")
        if not record.card_number.strip() or not record.edition_card_count.strip():
            details.notes.append("numero da carta nao foi detectado com confianca")

        if record.quantity == 0:
            record.quantity = 1

        # Release all image arrays eagerly — numpy arrays are not freed by
        # reference counting alone when held in cycles; explicit deletion plus
        # a collection pass keeps peak RSS low during batch processing.
        del regions
        gc.collect()

        elapsed = time.perf_counter() - t0
        _logger.info(
            "[IMAGE] fim: %s -> '%s' #%s/%s (%s) — %.3fs total",
            image_path,
            record.card_pt,
            record.card_number,
            record.edition_card_count,
            record.edition_sigla,
            elapsed,
        )

        return record, details

    def _timed_ocr(self, region: str, image: np.ndarray, config: OCRConfig) -> str:
        """Run OCR on *image* and log elapsed time for the region."""
        t = time.perf_counter()
        text = self.ocr.text(image, config)
        _logger.debug("[OCR] %-20s %.3fs -> %r", region, time.perf_counter() - t, text[:60] if text else "")
        return text

    def _best_ocr_for_number(
        self,
        images,
        config: OCRConfig,
        details: Details,
        region_name: str,
    ) -> str:
        """Try OCR on image variants, return the first one containing a card number pattern.

        Accepts any iterable (list or generator) of images. Stops as soon as a
        match is found (early exit) so unneeded variants are never computed when
        a generator is passed.
        """
        fallback_text = ""
        for i, img in enumerate(images):
            try:
                label = region_name if i == 0 else f"{region_name}_v{i}"
                text = self._timed_ocr(label, img, config)
                if not text.strip():
                    continue
                if not fallback_text:
                    fallback_text = text
                # Return immediately on first card number match
                if re.search(CARD_NO_PATTERN, text.upper()):
                    details.raw_text[label] = text
                    details.raw_text[region_name] = text
                    return text
            except Exception:
                continue
        if fallback_text:
            details.raw_text[region_name] = fallback_text
        return fallback_text
