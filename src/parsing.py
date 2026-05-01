"""Text parsing helpers: regex patterns, name sanitization, bottom-text extraction."""

from __future__ import annotations

import os
import re

CARD_NO_PATTERN = re.compile(r"(\d{1,3})\s*/\s*(\d{1,3})")
SET_CODE_PATTERN = re.compile(r"\b[A-Z]{2,4}\b")
SET_CODE_SUFFIX_PATTERN = re.compile(r"^([A-Z]{2,4})(EN|PT)$")
NAME_CLEANER = re.compile(r"[^\w\s\-\.']", re.UNICODE)
MULTI_SPACE = re.compile(r"\s+")


def left_pad_number(value: str) -> str:
    trimmed = value.strip()
    if len(trimmed) == 1:
        return "00" + trimmed
    if len(trimmed) == 2:
        return "0" + trimmed
    return trimmed


def sanitize_name(raw: str) -> str:
    for line in raw.split("\n"):
        cleaned = NAME_CLEANER.sub(" ", line)
        cleaned = MULTI_SPACE.sub(" ", cleaned).strip()
        if _looks_like_name(line, cleaned):
            return cleaned
    return ""


def _looks_like_name(raw_line: str, cleaned: str) -> bool:
    if len(cleaned) < 3 or len(cleaned) > 40:
        return False
    has_upper = any(c.isupper() for c in raw_line if c.isascii())
    has_letter = any(c.isalpha() for c in cleaned)
    has_digit = any(c.isdigit() for c in cleaned)
    return has_upper and has_letter and not has_digit


def better_name(current: str, candidate: str) -> str:
    c = current.strip()
    n = candidate.strip()
    if not n:
        return c
    if not c:
        return n
    if len(n) >= len(c) + 3:
        return n
    return c


def normalize_set_code_candidate(raw: str) -> str:
    candidate = raw.upper().strip(" \t\r\n.,:;!?()[]{}<>\"'`-_")
    if not candidate or candidate in ("HP", "PT", "EN"):
        return ""
    m = SET_CODE_SUFFIX_PATTERN.match(candidate)
    if m:
        candidate = m.group(1)
    if len(candidate) < 2 or len(candidate) > 4:
        return ""
    if not all("A" <= c <= "Z" for c in candidate):
        return ""
    if candidate in ("HP", "PT", "EN"):
        return ""
    return candidate


def best_set_code_candidate(raw: str) -> str:
    upper = raw.upper()
    for match in SET_CODE_PATTERN.findall(upper):
        candidate = normalize_set_code_candidate(match)
        if candidate:
            return candidate
    return ""


def set_code_before_card_number(raw: str) -> str:
    tokens = raw.split()
    for token in reversed(tokens):
        candidate = normalize_set_code_candidate(token)
        if candidate:
            return candidate
    return ""


def infer_set_code_from_filename(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0].upper()
    match = SET_CODE_PATTERN.search(base)
    if match:
        return normalize_set_code_candidate(match.group().strip())
    return ""


def parse_bottom_text(raw: str, image_path: str, record) -> None:
    """Parse bottom text to extract card_number, edition_card_count, edition_sigla."""
    upper = raw.upper()
    m = CARD_NO_PATTERN.search(upper)
    if m:
        record.card_number = left_pad_number(m.group(1))
        record.edition_card_count = m.group(2).strip()
        record.edition_sigla = set_code_before_card_number(upper[: m.start()])

    if not record.edition_sigla:
        record.edition_sigla = best_set_code_candidate(upper)
    if not record.edition_sigla:
        record.edition_sigla = infer_set_code_from_filename(image_path)
