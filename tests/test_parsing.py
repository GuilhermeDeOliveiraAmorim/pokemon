"""Tests for parsing module."""

from src.parsing import (
    left_pad_number,
    sanitize_name,
    better_name,
    normalize_set_code_candidate,
    best_set_code_candidate,
    set_code_before_card_number,
    infer_set_code_from_filename,
    parse_bottom_text,
)
from src.catalog import Record


class TestLeftPadNumber:
    def test_single_digit(self):
        assert left_pad_number("5") == "005"

    def test_two_digits(self):
        assert left_pad_number("55") == "055"

    def test_three_digits(self):
        assert left_pad_number("217") == "217"

    def test_whitespace(self):
        assert left_pad_number("  2  ") == "002"


class TestSanitizeName:
    def test_basic(self):
        assert sanitize_name("Pikachu") == "Pikachu"

    def test_removes_special_chars(self):
        assert sanitize_name("Pikachu!@#$") == "Pikachu"

    def test_skips_lines_with_digits(self):
        assert sanitize_name("123\nPikachu") == "Pikachu"

    def test_empty_for_short(self):
        assert sanitize_name("AB") == ""


class TestBetterName:
    def test_candidate_empty(self):
        assert better_name("Pikachu", "") == "Pikachu"

    def test_current_empty(self):
        assert better_name("", "Pikachu") == "Pikachu"

    def test_candidate_much_longer(self):
        assert better_name("Pik", "Pikachu") == "Pikachu"

    def test_current_kept_when_similar(self):
        assert better_name("Pikachu", "Pikach") == "Pikachu"


class TestSetCodeCandidate:
    def test_valid(self):
        assert normalize_set_code_candidate("ASC") == "ASC"

    def test_rejects_hp(self):
        assert normalize_set_code_candidate("HP") == ""

    def test_rejects_pt(self):
        assert normalize_set_code_candidate("PT") == ""

    def test_strips_suffix(self):
        assert normalize_set_code_candidate("ASCPT") == "ASC"

    def test_too_long(self):
        assert normalize_set_code_candidate("ABCDE") == ""

    def test_too_short(self):
        assert normalize_set_code_candidate("A") == ""


class TestBestSetCodeCandidate:
    def test_finds_code(self):
        assert best_set_code_candidate("ASC 055/217") == "ASC"

    def test_skips_hp(self):
        assert best_set_code_candidate("HP 70") == ""


class TestSetCodeBeforeCardNumber:
    def test_finds_code_before_number(self):
        assert set_code_before_card_number("ASC 055") == "ASC"

    def test_empty(self):
        assert set_code_before_card_number("") == ""


class TestInferSetCodeFromFilename:
    def test_no_code_in_timestamp(self):
        # IMG is surrounded by _ which are word chars, so \b doesn't match
        assert infer_set_code_from_filename("IMG_20260430_103425.jpg") == ""

    def test_with_code_space_separated(self):
        assert infer_set_code_from_filename("ASC 001.jpg") == "ASC"


class TestParseBottomText:
    def test_basic(self):
        r = Record()
        parse_bottom_text("ASC 055/217", "test.jpg", r)
        assert r.card_number == "055"
        assert r.edition_card_count == "217"
        assert r.edition_sigla == "ASC"

    def test_no_match(self):
        r = Record()
        parse_bottom_text("nothing here", "test.jpg", r)
        assert r.card_number == ""
