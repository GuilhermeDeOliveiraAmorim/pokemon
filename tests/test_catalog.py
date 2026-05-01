"""Tests for catalog module."""

import os
import tempfile

import pytest

from src.catalog import Catalog, Record, load_csv, HEADER


def _write_csv(path: str, rows: list[list[str]] | None = None):
    with open(path, "w", encoding="utf-8") as f:
        escaped = ['"' + h.replace('"', '""') + '"' for h in HEADER]
        f.write(",".join(escaped) + "\n")
        if rows:
            for row in rows:
                escaped = ['"' + v.replace('"', '""') + '"' for v in row]
                f.write(",".join(escaped) + "\n")


class TestRecord:
    def test_to_csv_line_and_back(self):
        r = Record(
            edition_ptbr="Herois Excelsos",
            edition_en="Ascended Heroes",
            edition_sigla="ASC",
            card_pt="Pikachu",
            card_en="Pikachu",
            quantity=1,
            quality="M",
            language="PT",
            rarity="C",
            color="L",
            card_number="055",
            edition_card_count="217",
        )
        line = r.to_csv_line()
        assert '"Pikachu"' in line
        assert '"ASC"' in line
        assert line.endswith("\n")

    def test_to_csv_line_escapes_quotes(self):
        r = Record(card_pt='Gloom da Érica', card_en='Erika\'s Gloom')
        line = r.to_csv_line()
        assert "Erika's Gloom" in line

    def test_from_csv_row_wrong_columns(self):
        with pytest.raises(ValueError, match="esperava 14"):
            Record.from_csv_row(["a", "b"])

    def test_validate_for_write_complete(self):
        r = Record(
            edition_ptbr="Herois Excelsos",
            edition_en="Ascended Heroes",
            edition_sigla="ASC",
            card_pt="Pikachu",
            card_en="Pikachu",
            quantity=1,
            quality="M",
            language="PT",
            rarity="C",
            color="L",
            card_number="055",
            edition_card_count="217",
        )
        assert r.validate_for_write() == []

    def test_validate_for_write_missing_fields(self):
        r = Record()
        errors = r.validate_for_write()
        assert len(errors) > 0

    def test_fill_missing_from(self):
        r = Record(card_pt="Pikachu")
        other = Record(card_pt="Other", edition_sigla="ASC", rarity="C")
        r.fill_missing_from(other)
        assert r.card_pt == "Pikachu"  # not overwritten
        assert r.edition_sigla == "ASC"  # filled
        assert r.rarity == "C"

    def test_apply_reference(self):
        r = Record(card_pt="Pikachu", quality="NM", language="EN")
        ref = Record(
            edition_ptbr="Herois", edition_en="Heroes", edition_sigla="ASC",
            card_pt="Pikachu", card_en="Pikachu", rarity="C", color="L",
            card_number="055", edition_card_count="217",
        )
        r.apply_reference(ref)
        assert r.edition_sigla == "ASC"
        assert r.quality == "NM"  # preserved
        assert r.language == "EN"  # preserved


class TestCatalog:
    def test_load_and_find(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            path = f.name
            _write_csv(path, [
                ["Herois Excelsos", "Ascended Heroes", "ASC", "Pikachu", "Pikachu",
                 "1", "M", "PT", "C", "L", "", "055", "", "217"],
            ])
        try:
            cat = load_csv(path)
            assert len(cat.records) == 1
            found = cat.find_by_set_and_number("ASC", "055")
            assert found is not None
            assert found.card_pt == "Pikachu"
            assert cat.exists("ASC", "055")
            assert not cat.exists("ASC", "999")
        finally:
            os.unlink(path)

    def test_append_and_exists(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            path = f.name
            _write_csv(path)
        try:
            cat = load_csv(path)
            assert len(cat.records) == 0
            r = Record(
                edition_ptbr="Test", edition_en="Test", edition_sigla="TST",
                card_pt="Test Card", card_en="Test Card", quantity=1,
                quality="M", language="PT", rarity="C", color="L",
                card_number="001", edition_card_count="100",
            )
            cat.append(r)
            assert cat.exists("TST", "001")
            cat.flush()
            # Reload to verify persistence
            cat2 = load_csv(path)
            assert len(cat2.records) == 1
            assert cat2.records[0].card_pt == "Test Card"
        finally:
            os.unlink(path)

    def test_load_empty_csv_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="CSV vazio"):
                load_csv(path)
        finally:
            os.unlink(path)
