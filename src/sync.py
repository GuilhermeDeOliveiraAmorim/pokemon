"""Sync reference data from PokémonTCG API and TCGDex."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import quote as url_quote

import requests

from .catalog import Record

POKEMONTCG_API_BASE = "https://api.pokemontcg.io/v2"
TCGDEX_GITHUB_API_BASE = "https://api.github.com/repos/tcgdex/cards-database/contents"
TCGDEX_RAW_BASE = "https://raw.githubusercontent.com/tcgdex/cards-database/master"

RARITY_MAP = {
    "common": "C",
    "uncommon": "U",
    "rare": "R",
    "rare holo": "H",
    "rare holo ex": "H",
    "rare holo gx": "H",
    "rare holo lv.x": "H",
    "rare holo star": "H",
    "rare holo vmax": "H",
    "rare holo v": "H",
    "rare holo vstar": "H",
    "double rare": "R",
    "illustration rare": "I",
    "ace spec rare": "A",
    "special illustration rare": "S",
    "ultra rare": "S",
    "hyper rare": "E",
    "promo": "P",
}

TYPE_COLOR_MAP = {
    "darkness": "D",
    "dragon": "Y",
    "colorless": "C",
    "lightning": "L",
    "fighting": "R",
    "fire": "F",
    "grass": "G",
    "metal": "M",
    "psychic": "P",
    "water": "W",
}


def sync_reference_file(
    *,
    source: str = "pokemontcg",
    query: str = "",
    edition_ptbr: str = "",
    edition_sigla: str = "",
    language: str = "PT",
    quality: str = "M",
    quantity: int = 1,
    output_path: str = "",
    overrides_path: str = "",
    api_key: str = "",
    tcgdex_set_path: str = "",
    tcgdex_locale: str = "pt",
    page_size: int = 250,
) -> None:
    if not output_path:
        raise ValueError("output path obrigatorio")

    api_key = api_key or os.environ.get("POKEMONTCG_API_KEY", "")

    if source == "pokemontcg":
        if not query:
            raise ValueError("query obrigatoria")
        records = _fetch_pokemontcg(query, edition_ptbr, edition_sigla, language, quality, quantity, api_key, page_size)
    elif source == "tcgdex":
        if not tcgdex_set_path:
            raise ValueError("tcgdex set path obrigatorio")
        records = _fetch_tcgdex(tcgdex_set_path, tcgdex_locale, edition_ptbr, edition_sigla, language, quality, quantity)
    else:
        raise ValueError(f"source desconhecida: {source}")

    overrides = _load_overrides(overrides_path)
    records = _apply_overrides(records, overrides)
    existing = _load_reference_file(output_path)
    merged = _merge_records(existing, records)
    _write_reference_file(output_path, merged)


# ---------------------------------------------------------------------------
# PokémonTCG API
# ---------------------------------------------------------------------------


def _fetch_pokemontcg(
    query: str, edition_ptbr: str, edition_sigla: str,
    language: str, quality: str, quantity: int, api_key: str, page_size: int,
) -> list[Record]:
    records: list[Record] = []
    page = 1
    while True:
        params = {
            "q": query,
            "page": str(page),
            "pageSize": str(page_size),
            "orderBy": "number",
            "select": "name,number,rarity,types,set.name,set.ptcgoCode,set.total,set.printedTotal",
        }
        headers = {}
        if api_key:
            headers["X-Api-Key"] = api_key
        resp = requests.get(f"{POKEMONTCG_API_BASE}/cards", params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        for card in data:
            r = _api_card_to_record(card, edition_ptbr, edition_sigla, language, quality, quantity)
            if r:
                records.append(r)
        if len(data) < page_size:
            break
        page += 1
    return records


def _api_card_to_record(
    card: dict, edition_ptbr: str, edition_sigla: str,
    language: str, quality: str, quantity: int,
) -> Record | None:
    number = _normalize_card_number(card.get("number", ""))
    if not number:
        return None
    card_set = card.get("set", {})
    sigla = edition_sigla.strip() or card_set.get("ptcgoCode", "").strip()
    ptbr = edition_ptbr.strip() or card_set.get("name", "").strip()
    en = card_set.get("name", "").strip()
    total = card_set.get("total", 0) or card_set.get("printedTotal", 0)
    return Record(
        edition_ptbr=ptbr,
        edition_en=en,
        edition_sigla=sigla,
        card_pt=card.get("name", "").strip(),
        card_en=card.get("name", "").strip(),
        quantity=quantity,
        quality=quality.upper().strip(),
        language=language.upper().strip(),
        rarity=_map_rarity(card.get("rarity", "")),
        color=_map_type_color(card.get("types", [])),
        card_number=number,
        edition_card_count=str(total),
    )


# ---------------------------------------------------------------------------
# TCGDex
# ---------------------------------------------------------------------------


def _fetch_tcgdex(
    set_path: str, locale: str, edition_ptbr: str, edition_sigla: str,
    language: str, quality: str, quantity: int,
) -> list[Record]:
    set_content = _fetch_text(_raw_url(set_path))
    set_meta = _parse_set_meta(set_content, edition_ptbr, edition_sigla)

    cards_dir = set_path.removesuffix(".ts")
    items = _fetch_github_dir(cards_dir)
    records: list[Record] = []
    for item in items:
        if item.get("type") != "file" or not item.get("name", "").endswith(".ts"):
            continue
        url = item.get("download_url") or _raw_url(item["path"])
        content = _fetch_text(url)
        r = _parse_card_record(item["name"], content, set_meta, locale, language, quality, quantity)
        if r:
            records.append(r)
    records.sort(key=lambda r: r.card_number)
    return records


def _raw_url(file_path: str) -> str:
    segments = file_path.lstrip("/").split("/")
    return TCGDEX_RAW_BASE + "/" + "/".join(url_quote(s) for s in segments)


def _fetch_text(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def _fetch_github_dir(dir_path: str) -> list[dict]:
    url = TCGDEX_GITHUB_API_BASE.rstrip("/") + "/" + dir_path.lstrip("/")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_set_meta(content: str, edition_ptbr: str, edition_sigla: str) -> dict:
    names = _parse_ts_object_entries(content, "name")
    en = names.get("en", edition_ptbr.strip())
    pt = edition_ptbr.strip() or names.get("pt", en)
    sigla = edition_sigla.strip()
    if not sigla:
        sigla = _parse_ts_nested_string(content, "abbreviations", "official")
    if not sigla:
        sigla = _parse_ts_string(content, "id")
    card_count = _parse_ts_nested_int(content, "cardCount", "official")
    return {"edition_ptbr": pt, "edition_en": en, "edition_sigla": sigla, "edition_card_count": str(card_count)}


def _parse_card_record(
    filename: str, content: str, set_meta: dict, locale: str,
    language: str, quality: str, quantity: int,
) -> Record | None:
    number = _normalize_card_number(os.path.splitext(os.path.basename(filename))[0])
    if not number:
        return None
    names = _parse_ts_object_entries(content, "name")
    card_pt = names.get(locale) or names.get("pt") or names.get("en", "")
    card_en = names.get("en") or card_pt
    return Record(
        edition_ptbr=set_meta["edition_ptbr"],
        edition_en=set_meta["edition_en"],
        edition_sigla=set_meta["edition_sigla"],
        card_pt=card_pt,
        card_en=card_en,
        quantity=quantity,
        quality=quality.upper().strip(),
        language=language.upper().strip(),
        rarity=_map_rarity(_parse_ts_string(content, "rarity")),
        color=_map_type_color(_parse_ts_array(content, "types")),
        card_number=number,
        edition_card_count=set_meta["edition_card_count"],
    )


# ---------------------------------------------------------------------------
# TS parsing helpers
# ---------------------------------------------------------------------------

_TS_OBJECT_RE = re.compile(r"(?s)%s\s*:\s*\{(.*?)\}")
_TS_ENTRY_RE = re.compile(r'(?m)(?:"([^"]+)"|([A-Za-z][A-Za-z0-9-]*))\s*:\s*"([^"]*)"')
_TS_STRING_RE = re.compile(r'(?m)%s\s*:\s*"([^"]+)"')
_TS_INT_RE = re.compile(r"(?m)%s\s*:\s*(\d+)")
_TS_ARRAY_RE = re.compile(r"(?s)%s\s*:\s*\[(.*?)\]")
_QUOTED_ITEM_RE = re.compile(r'"([^"]+)"')


def _parse_ts_object_entries(content: str, field: str) -> dict[str, str]:
    pattern = re.compile(_TS_OBJECT_RE.pattern % re.escape(field))
    m = pattern.search(content)
    if not m:
        return {}
    result = {}
    for entry in _TS_ENTRY_RE.finditer(m.group(1)):
        key = entry.group(1) or entry.group(2)
        result[key] = entry.group(3)
    return result


def _parse_ts_string(content: str, field: str) -> str:
    pattern = re.compile(_TS_STRING_RE.pattern % re.escape(field))
    m = pattern.search(content)
    return m.group(1).strip() if m else ""


def _parse_ts_nested_string(content: str, field: str, nested: str) -> str:
    return _parse_ts_object_entries(content, field).get(nested, "")


def _parse_ts_nested_int(content: str, field: str, nested: str) -> int:
    obj_pattern = re.compile(_TS_OBJECT_RE.pattern % re.escape(field))
    m = obj_pattern.search(content)
    if not m:
        return 0
    int_pattern = re.compile(_TS_INT_RE.pattern % re.escape(nested))
    nm = int_pattern.search(m.group(1))
    return int(nm.group(1)) if nm else 0


def _parse_ts_array(content: str, field: str) -> list[str]:
    pattern = re.compile(_TS_ARRAY_RE.pattern % re.escape(field))
    m = pattern.search(content)
    if not m:
        return []
    return [item.group(1) for item in _QUOTED_ITEM_RE.finditer(m.group(1))]


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------


def _map_rarity(rarity: str) -> str:
    return RARITY_MAP.get(rarity.lower().strip(), "")


def _map_type_color(types: list[str]) -> str:
    if not types:
        return ""
    return TYPE_COLOR_MAP.get(types[0].lower().strip(), "")


def _normalize_card_number(number: str) -> str:
    trimmed = number.strip()
    if not trimmed or not trimmed.isdigit():
        return ""
    return trimmed.zfill(3)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _load_reference_file(path: str) -> list[Record]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [_json_to_record(d) for d in data]


def _load_overrides(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _apply_overrides(records: list[Record], overrides: list[dict]) -> list[Record]:
    if not overrides:
        return records
    index = {}
    for ov in overrides:
        key = (ov.get("EditionSigla", "").upper().strip(), ov.get("CardNumber", "").upper().strip())
        if key != ("", ""):
            index[key] = ov
    result = []
    for r in records:
        key = (r.edition_sigla.upper().strip(), r.card_number.upper().strip())
        if key in index:
            r = _apply_override(r, index[key])
        result.append(r)
    return result


def _apply_override(record: Record, ov: dict) -> Record:
    for json_key, attr in [
        ("EditionPTBR", "edition_ptbr"), ("EditionEN", "edition_en"),
        ("CardPT", "card_pt"), ("CardEN", "card_en"),
        ("Quality", "quality"), ("Language", "language"),
        ("Rarity", "rarity"), ("Color", "color"),
        ("Extras", "extras"), ("Comment", "comment"),
        ("EditionCardCount", "edition_card_count"),
    ]:
        val = ov.get(json_key, "").strip()
        if val:
            setattr(record, attr, val)
    qty = ov.get("Quantity", 0)
    if qty > 0:
        record.quantity = qty
    return record


def _merge_records(existing: list[Record], incoming: list[Record]) -> list[Record]:
    merged: list[Record] = []
    index: dict[tuple[str, str], int] = {}
    for r in existing:
        key = (r.edition_sigla.upper().strip(), r.card_number.upper().strip())
        index[key] = len(merged)
        merged.append(r)
    for r in incoming:
        key = (r.edition_sigla.upper().strip(), r.card_number.upper().strip())
        if key in index:
            merged[index[key]] = r
        else:
            index[key] = len(merged)
            merged.append(r)
    return merged


def _write_reference_file(path: str, records: list[Record]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = [_record_to_json(r) for r in records]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


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


def _record_to_json(r: Record) -> dict:
    return {
        "EditionPTBR": r.edition_ptbr,
        "EditionEN": r.edition_en,
        "EditionSigla": r.edition_sigla,
        "CardPT": r.card_pt,
        "CardEN": r.card_en,
        "Quantity": r.quantity,
        "Quality": r.quality,
        "Language": r.language,
        "Rarity": r.rarity,
        "Color": r.color,
        "Extras": r.extras,
        "CardNumber": r.card_number,
        "Comment": r.comment,
        "EditionCardCount": r.edition_card_count,
    }
