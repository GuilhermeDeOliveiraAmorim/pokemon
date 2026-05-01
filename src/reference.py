"""Reference data loading and enrichment logic."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .catalog import Record

_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
_reference_records: list[Record] | None = None
_idx_set_number: dict[tuple[str, str], Record] | None = None
_idx_set_name: dict[tuple[str, str], Record] | None = None
_idx_number: dict[str, list[Record]] | None = None
_idx_name: dict[str, list[Record]] | None = None


def _load_reference() -> list[Record]:
    global _reference_records, _idx_set_number, _idx_set_name, _idx_number, _idx_name
    if _reference_records is not None:
        return _reference_records
    path = _REFERENCE_DIR / "reference_cards.json"
    if not path.exists():
        _reference_records = []
        _idx_set_number = {}
        _idx_set_name = {}
        _idx_number = {}
        _idx_name = {}
        return _reference_records
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    _reference_records = [_json_to_record(item) for item in data]
    _build_indexes(_reference_records)
    return _reference_records


def _build_indexes(records: list[Record]) -> None:
    global _idx_set_number, _idx_set_name, _idx_number, _idx_name
    idx_sn: dict[tuple[str, str], Record] = {}
    idx_sname: dict[tuple[str, str], Record] = {}
    idx_num: dict[str, list[Record]] = {}
    idx_name_: dict[str, list[Record]] = {}

    for r in records:
        sigla = _norm(r.edition_sigla)
        number = _norm(r.card_number)
        pt = _norm(r.card_pt)
        en = _norm(r.card_en)

        if sigla and number:
            idx_sn.setdefault((sigla, number), r)
        if sigla and pt:
            idx_sname.setdefault((sigla, pt), r)
        if sigla and en:
            idx_sname.setdefault((sigla, en), r)
        if number:
            idx_num.setdefault(number, []).append(r)
        if pt:
            idx_name_.setdefault(pt, []).append(r)
        if en and en != pt:
            idx_name_.setdefault(en, []).append(r)

    _idx_set_number = idx_sn
    _idx_set_name = idx_sname
    _idx_number = idx_num
    _idx_name = idx_name_


def _json_to_record(d: dict) -> Record:
    return Record(
        edition_ptbr=d.get("EditionPTBR", ""),
        edition_en=d.get("EditionEN", ""),
        edition_sigla=d.get("EditionSigla", ""),
        card_pt=d.get("CardPT", ""),
        card_en=d.get("CardEN", ""),
        quantity=d.get("Quantity", 1),
        quality=d.get("Quality", ""),
        language=d.get("Language", ""),
        rarity=d.get("Rarity", ""),
        color=d.get("Color", ""),
        extras=d.get("Extras", ""),
        card_number=d.get("CardNumber", ""),
        comment=d.get("Comment", ""),
        edition_card_count=d.get("EditionCardCount", ""),
    )


def _norm(v: str) -> str:
    return v.upper().strip()


def find_reference_by_set_and_number(sigla: str, number: str) -> Record | None:
    _load_reference()
    return _idx_set_number.get((_norm(sigla), _norm(number)))


def find_reference_by_set_and_name(sigla: str, name: str) -> Record | None:
    ns, nn = _norm(sigla), _norm(name)
    if not ns or not nn:
        return None
    _load_reference()
    return _idx_set_name.get((ns, nn))


def find_unique_reference_by_number(number: str) -> Record | None:
    nn = _norm(number)
    if not nn:
        return None
    _load_reference()
    matches = _idx_number.get(nn)
    if matches is None or len(matches) != 1:
        return None
    return matches[0]


def find_reference_by_number_and_count(number: str, edition_card_count: str) -> Record | None:
    nn = _norm(number)
    nc = _norm(edition_card_count)
    _load_reference()
    candidates = _idx_number.get(nn)
    if not candidates:
        return None
    match = None
    for r in candidates:
        if _norm(r.edition_card_count) != nc:
            continue
        if match is not None:
            return None
        match = r
    return match


def enrich_record(catalog, record: Record, ocr_name: str = "") -> None:
    """Enrich a record using catalog then reference data (5-level cascade).
    
    If ocr_name is provided and enrichment by number yields a different name,
    try name-based lookup as a cross-check correction.
    """
    ocr_name_norm = ocr_name.upper().strip()

    found = catalog.find_by_set_and_number(record.edition_sigla, record.card_number)
    if found is not None:
        record.fill_missing_from(found)
        if ocr_name_norm and _names_conflict(ocr_name_norm, found):
            _try_name_correction(record, ocr_name_norm)
        return

    ref = find_reference_by_set_and_number(record.edition_sigla, record.card_number)
    if ref is not None:
        if ocr_name_norm and _names_conflict(ocr_name_norm, ref):
            # OCR name doesn't match number-based lookup — trust the name
            name_ref = find_reference_by_set_and_name(record.edition_sigla, ocr_name_norm)
            if name_ref is not None:
                record.apply_reference(name_ref)
                return
        record.apply_reference(ref)
        return

    name = record.card_pt.strip() or record.card_en.strip()
    ref = find_reference_by_set_and_name(record.edition_sigla, name)
    if ref is not None:
        record.apply_reference(ref)
        return

    # For fallback lookups (unique by number, number+count), cross-check with OCR name
    # because the card number from OCR may be wrong
    ref = find_unique_reference_by_number(record.card_number)
    if ref is not None:
        if ocr_name_norm and _names_conflict(ocr_name_norm, ref):
            # Number lookup gives wrong name — try name-based across all sets
            name_ref = _find_reference_by_name_any_set(ocr_name_norm)
            if name_ref is not None:
                record.apply_reference(name_ref)
                return
        record.apply_reference(ref)
        return

    ref = find_reference_by_number_and_count(record.card_number, record.edition_card_count)
    if ref is not None:
        if ocr_name_norm and _names_conflict(ocr_name_norm, ref):
            name_ref = _find_reference_by_name_any_set(ocr_name_norm)
            if name_ref is not None:
                record.apply_reference(name_ref)
                return
        record.apply_reference(ref)
        return

    # Last resort: if we have a name but no number match, try name in any set
    if ocr_name_norm:
        name_ref = _find_reference_by_name_any_set(ocr_name_norm)
        if name_ref is not None:
            record.apply_reference(name_ref)


def _names_conflict(ocr_name: str, ref: Record) -> bool:
    """Check if the OCR-detected name conflicts with the reference record name."""
    ref_pt = ref.card_pt.upper().strip()
    ref_en = ref.card_en.upper().strip()
    if not ocr_name:
        return False
    # No conflict if OCR name matches either PT or EN name
    if ocr_name == ref_pt or ocr_name == ref_en:
        return False
    # No conflict if one contains the other (partial match)
    if ocr_name in ref_pt or ref_pt in ocr_name:
        return False
    if ocr_name in ref_en or ref_en in ocr_name:
        return False
    return True


def _try_name_correction(record: Record, ocr_name: str) -> None:
    """Attempt to correct a record by looking up the OCR-detected name in reference."""
    ref = find_reference_by_set_and_name(record.edition_sigla, ocr_name)
    if ref is not None:
        record.apply_reference(ref)


def _find_reference_by_name_any_set(name: str) -> Record | None:
    """Find a reference record by name across all sets.
    
    Returns the best match (preferring records with non-empty sigla).
    Returns None only if no match found or matches have conflicting card numbers.
    """
    nn = _norm(name)
    if not nn:
        return None
    _load_reference()
    matches = _idx_name.get(nn)
    if not matches:
        return None
    # Check if all matches refer to the same card number
    numbers = {_norm(m.card_number) for m in matches if m.card_number.strip()}
    if len(numbers) > 1:
        return None  # ambiguous — different card numbers
    # Prefer match with non-empty sigla
    for m in matches:
        if m.edition_sigla.strip():
            return m
    return matches[0]
