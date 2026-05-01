"""Card catalog: Record dataclass, CSV I/O and validation."""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field, fields

HEADER = [
    "Edicao (PTBR)",
    "Edicao (EN)",
    "Edicao (Sigla)",
    "Card (PT)",
    "Card (EN)",
    "Quantidade",
    "Qualidade (M NM SP MP HP D)",
    "Idioma (BR EN DE ES FR IT JP KO RU TW)",
    "Raridade (C I U R H E X U P A L S)",
    "Cor (C D O E Y F R G L M P W)",
    "Extras",
    "Card #",
    "Comentario",
    "# Cards na Edicao",
]

VALID_QUALITY = {"M", "NM", "SP", "MP", "HP", "D"}
VALID_LANGUAGE = {"PT", "BR", "EN", "DE", "ES", "FR", "IT", "JP", "KO", "RU", "TW"}
VALID_RARITY = {"C", "I", "U", "R", "H", "E", "X", "P", "A", "L", "S"}
VALID_COLOR = {"C", "D", "O", "E", "Y", "F", "R", "G", "L", "M", "P", "W"}

_SET_CODE_RE = re.compile(r"^[A-Z0-9]{2,5}$")
_NUMBER_RE = re.compile(r"^\d{1,3}$")
_TEXT_VALUE_RE = re.compile(r"[\w]", re.UNICODE)


@dataclass
class Record:
    edition_ptbr: str = ""
    edition_en: str = ""
    edition_sigla: str = ""
    card_pt: str = ""
    card_en: str = ""
    quantity: int = 1
    quality: str = ""
    language: str = ""
    rarity: str = ""
    color: str = ""
    extras: str = ""
    card_number: str = ""
    comment: str = ""
    edition_card_count: str = ""

    # ---- CSV conversion ----

    def to_csv_row(self) -> list[str]:
        return [
            self.edition_ptbr,
            self.edition_en,
            self.edition_sigla,
            self.card_pt,
            self.card_en,
            str(self.quantity),
            self.quality,
            self.language,
            self.rarity,
            self.color,
            self.extras,
            self.card_number,
            self.comment,
            self.edition_card_count,
        ]

    def to_csv_line(self) -> str:
        escaped = ['"' + v.replace('"', '""') + '"' for v in self.to_csv_row()]
        return ",".join(escaped) + "\n"

    @classmethod
    def from_csv_row(cls, row: list[str]) -> Record:
        if len(row) != len(HEADER):
            raise ValueError(f"esperava {len(HEADER)} colunas, recebi {len(row)}")
        quantity = 0
        raw_qty = row[5].strip()
        if raw_qty:
            quantity = int(raw_qty)
        return cls(
            edition_ptbr=row[0],
            edition_en=row[1],
            edition_sigla=row[2],
            card_pt=row[3],
            card_en=row[4],
            quantity=quantity,
            quality=row[6],
            language=row[7],
            rarity=row[8],
            color=row[9],
            extras=row[10],
            card_number=row[11],
            comment=row[12],
            edition_card_count=row[13],
        )

    # ---- Enrichment ----

    def fill_missing_from(self, other: Record) -> None:
        for f in fields(self):
            current = getattr(self, f.name)
            if f.name == "quantity":
                if current == 0:
                    setattr(self, f.name, getattr(other, f.name))
            elif isinstance(current, str) and not current.strip():
                setattr(self, f.name, getattr(other, f.name))

    def apply_reference(self, other: Record) -> None:
        self.edition_ptbr = other.edition_ptbr
        self.edition_en = other.edition_en
        self.edition_sigla = other.edition_sigla
        self.card_pt = other.card_pt
        self.card_en = other.card_en
        self.rarity = other.rarity
        self.color = other.color
        self.card_number = other.card_number
        self.edition_card_count = other.edition_card_count
        if not self.extras.strip():
            self.extras = other.extras
        if self.quantity == 0:
            self.quantity = other.quantity
        if not self.quality.strip():
            self.quality = other.quality
        if not self.language.strip():
            self.language = other.language

    # ---- Validation ----

    def validate(self) -> list[str]:
        missing = []
        if not self.edition_sigla.strip():
            missing.append("Edicao (Sigla)")
        if not self.card_pt.strip():
            missing.append("Card (PT)")
        if not self.card_number.strip():
            missing.append("Card #")
        if not self.edition_card_count.strip():
            missing.append("# Cards na Edicao")
        return missing

    def validate_for_write(self) -> list[str]:
        errors: list[str] = []
        missing = self.validate()
        if missing:
            errors.append(f"campos essenciais faltando: {', '.join(missing)}")
            return errors

        if self.quantity <= 0:
            errors.append("quantidade deve ser maior que zero")

        q = self.quality.upper().strip()
        if not q:
            errors.append("qualidade obrigatoria")
        elif q not in VALID_QUALITY:
            errors.append(f"qualidade invalida: {self.quality}")

        lang = self.language.upper().strip()
        if not lang:
            errors.append("idioma obrigatorio")
        elif lang not in VALID_LANGUAGE:
            errors.append(f"idioma invalido: {self.language}")

        sigla = self.edition_sigla.strip()
        if not _SET_CODE_RE.match(sigla):
            errors.append(f"edicao (sigla) invalida: {sigla}")

        if not _NUMBER_RE.match(self.card_number.strip()):
            errors.append(f"card # invalido: {self.card_number}")

        if not _NUMBER_RE.match(self.edition_card_count.strip()):
            errors.append(f"# cards na edicao invalido: {self.edition_card_count}")

        if not _TEXT_VALUE_RE.search(self.edition_ptbr.strip()):
            errors.append("edicao (ptbr) obrigatoria")
        if not _TEXT_VALUE_RE.search(self.edition_en.strip()):
            errors.append("edicao (en) obrigatoria")
        if not _TEXT_VALUE_RE.search(self.card_pt.strip()):
            errors.append("card (pt) obrigatorio")
        if not _TEXT_VALUE_RE.search(self.card_en.strip()):
            errors.append("card (en) obrigatorio")

        r = self.rarity.upper().strip()
        if not r:
            errors.append("raridade obrigatoria")
        elif r not in VALID_RARITY:
            errors.append(f"raridade invalida: {self.rarity}")

        c = self.color.upper().strip()
        if not c:
            errors.append("cor obrigatoria")
        elif c not in VALID_COLOR:
            errors.append(f"cor invalida: {self.color}")

        return errors


class Catalog:
    def __init__(self, path: str, records: list[Record] | None = None):
        self.path = path
        self.records: list[Record] = records or []
        self._index: dict[tuple[str, str], Record] = {}
        self._buffer: list[Record] = []
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = {}
        for r in self.records:
            key = (r.edition_sigla.upper().strip(), r.card_number.upper().strip())
            if key[0] and key[1]:
                self._index.setdefault(key, r)

    def exists(self, sigla: str, number: str) -> bool:
        return self.find_by_set_and_number(sigla, number) is not None

    def find_by_set_and_number(self, sigla: str, number: str) -> Record | None:
        ns = sigla.upper().strip()
        nn = number.upper().strip()
        return self._index.get((ns, nn))

    def append(self, record: Record, flush: bool = False) -> None:
        self.records.append(record)
        key = (record.edition_sigla.upper().strip(), record.card_number.upper().strip())
        if key[0] and key[1]:
            self._index.setdefault(key, record)
        self._buffer.append(record)
        if flush:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        with open(self.path, "r+b") as f:
            f.seek(0, 2)
            pos = f.tell()
            if pos > 0:
                f.seek(pos - 1)
                if f.read(1) != b"\n":
                    f.write(b"\n")
            for r in self._buffer:
                f.write(r.to_csv_line().encode("utf-8"))
        self._buffer.clear()


def load_csv(path: str) -> Catalog:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV vazio")

    records: list[Record] = []
    for i, row in enumerate(rows[1:], start=2):
        try:
            records.append(Record.from_csv_row(row))
        except Exception as e:
            raise ValueError(f"linha {i}: {e}") from e

    return Catalog(path, records)
