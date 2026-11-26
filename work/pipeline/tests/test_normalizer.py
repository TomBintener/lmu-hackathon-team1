import csv
from pathlib import Path

import pytest

from app.models import LookupRequest
from app.normalizer import (
    barcode_variants,
    extract_label_catno_hint,
    normalize_artist,
    normalize_barcode,
    normalize_input,
    strip_discogs_disambiguator,
)


def _find_dev_csv() -> Path:
    in_container = Path("/app/data/development.csv")
    if in_container.exists():
        return in_container
    return Path(__file__).resolve().parents[3] / "data" / "development.csv"


def test_all_development_csv_rows_normalize_and_validate() -> None:
    csv_path = _find_dev_csv()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 25
    normed = []
    for row in rows:
        req = LookupRequest(
            barcode=row["barcode"],
            album_title=row["album_title"],
            album_artist=row["album_artist"],
        )
        norm = normalize_input(req.barcode, req.album_title, req.album_artist)
        assert norm.title_norm
        normed.append(norm)

    assert sum(1 for n in normed if n.barcode_norm) == 15
    assert sum(1 for n in normed if not n.barcode_norm) == 10
    assert sum(1 for n in normed if n.is_compilation) == 7


def test_barcode_sanitization_and_upc_ean_variants() -> None:
    assert normalize_barcode("") == ""
    assert normalize_barcode("0") == ""
    assert normalize_barcode("00000") == ""
    assert normalize_barcode("0 7777-46440-2 0") == "077774644020"

    assert barcode_variants("") == []
    assert barcode_variants("198028875721") == ["198028875721", "0198028875721"]
    assert barcode_variants("0198028875721") == ["0198028875721", "198028875721"]
    assert barcode_variants("4065629733131") == ["4065629733131"]


def test_artist_normalization_inversions_compilations_and_disambiguators() -> None:
    assert strip_discogs_disambiguator("Casper (9)") == "Casper"
    assert strip_discogs_disambiguator("Epica (2)") == "Epica"

    assert normalize_artist("Fantastischen Vier, Die") == (
        False,
        "Die Fantastischen Vier",
        "die fantastischen vier",
    )
    assert normalize_artist("Beatles, The") == (False, "The Beatles", "the beatles")
    assert normalize_artist("Negresses Vertes, Les") == (
        False,
        "Les Negresses Vertes",
        "les negresses vertes",
    )

    for alias in ("Various Artists", "VARIOUS", "VA", "Various", "Various (Sampler)"):
        is_comp, clean, norm = normalize_artist(alias)
        assert is_comp is True
        assert clean == "Various"
        assert norm == "various"


def test_format_extraction_title_cleaning_and_input_hash_dedup() -> None:
    jethro = normalize_input("4029759202219", "JETHRO TULL;BALOISE SESSION(2CD)", "JETHRO TULL")
    assert jethro.title_clean == "BALOISE SESSION"
    assert jethro.format_hint.medium == "CD"
    assert jethro.format_hint.qty == 2
    assert jethro.format_hint.tags == ["2CD"]

    foreigner = normalize_input("4029759205296", "LIVE IN CHICAGO (2LP)", "FOREIGNER")
    assert foreigner.title_clean == "LIVE IN CHICAGO"
    assert foreigner.format_hint.medium == "Vinyl"
    assert foreigner.format_hint.qty == 2

    # Case-insensitive tag deduplication when leading CD and parenthetical (CD) both appear
    dup_tag = normalize_input("", "CD - Example (cd)", "Artist")
    assert dup_tag.format_hint.tags == ["CD"]
    assert dup_tag.title_clean == "Example"

    # Inverted vs normal artist + RSD hint produce identical input_hash
    fanta_inv = normalize_input(
        "4262459550239", "Long Player (RSD Exklusive)", "Fantastischen Vier, Die"
    )
    fanta_norm = normalize_input(
        "4262459550239", "Long Player (RSD Exklusive)", "Die Fantastischen Vier"
    )
    assert fanta_inv.input_hash == fanta_norm.input_hash

    smart = normalize_input("", "It‘s raining Cats & AutocRats", "Various (Sampler)")
    assert smart.title_clean == "It's raining Cats & AutocRats"
    assert smart.is_compilation is True


def test_lookup_request_rejects_empty_or_zero_only_payloads() -> None:
    with pytest.raises(ValueError):
        LookupRequest(barcode="", album_title="", album_artist="EPICA")
    with pytest.raises(ValueError):
        LookupRequest(barcode="0", album_title="   ", album_artist="Tindersticks")


def test_extract_label_catno_hint() -> None:
    assert extract_label_catno_hint("432HERTZ 002") == ("432HERTZ", "002", 2)
    assert extract_label_catno_hint("Jackfruit Traxx 1") == ("Jackfruit Traxx", "1", 1)
    assert extract_label_catno_hint("SCHLAGER ARENA 2025") is None
    assert extract_label_catno_hint("Traxx Salad Vol. 1") is None
    assert extract_label_catno_hint("Aspiral") is None
