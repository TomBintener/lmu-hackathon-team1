import hashlib
import json
import re
import unicodedata

from app.models import FormatHint, NormalizedInput

COMPILATION_ALIASES = {
    "various",
    "various artists",
    "va",
    "v.a.",
    "various (sampler)",
    "various sampler",
    "sampler",
    "verschiedene interpret:innen",
    "verschiedene interpreten",
}

TRAILING_ARTICLE_RE = re.compile(
    r"^(.+?),\s*(Die|Der|Das|The|Les|La|Le|Los|Las|El|Het|De)$",
    re.IGNORECASE,
)
DISAMBIG_RE = re.compile(r"\s*\(\d+\)\s*$")
LEADING_FORMAT_RE = re.compile(
    r"^\s*(2CD|2LP|3CD|3LP|CD|LP|EP|VINYL)\s*[-:]\s+",
    re.IGNORECASE,
)
PAREN_FORMAT_RE = re.compile(
    r"\(\s*(2CD|3CD|CD|2LP|3LP|LP|VINYL|RSD[^)]*|DELUXE[^)]*|LIMITED[^)]*|REISSUE[^)]*|LIVE[^)]*)\s*\)",
    re.IGNORECASE,
)
QTY_MEDIUM_RE = re.compile(r"^(\d+)\s*(CD|LP)$", re.IGNORECASE)
EP_SUFFIX_RE = re.compile(r"\bEP\b", re.IGNORECASE)
LABEL_CATNO_TITLE_RE = re.compile(r"^(.{2,}?)\s+#?(0*\d{1,4})$")
NON_LABEL_SEQ_WORDS = {
    "vol",
    "vol.",
    "volume",
    "part",
    "pt",
    "pt.",
    "chapter",
    "episode",
    "disc",
    "cd",
    "lp",
    "edition",
    "no",
    "no.",
}


def normalize_quotes(text: str) -> str:
    return (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201a", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


def fold_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", normalize_quotes(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def strip_discogs_disambiguator(name: str) -> str:
    return DISAMBIG_RE.sub("", (name or "").strip()).strip()


def normalize_barcode(raw: str) -> str:
    digits = "".join(ch for ch in (raw or "").strip() if ch.isdigit())
    if not digits or set(digits) == {"0"}:
        return ""
    return digits


def barcode_variants(barcode_norm: str) -> list[str]:
    if not barcode_norm:
        return []
    variants = [barcode_norm]
    if len(barcode_norm) == 12:
        variants.append("0" + barcode_norm)
    elif len(barcode_norm) == 13 and barcode_norm.startswith("0"):
        variants.append(barcode_norm[1:])
    return list(dict.fromkeys(variants))


def normalize_artist(raw_artist: str) -> tuple[bool, str, str]:
    raw_clean = unicodedata.normalize("NFKC", normalize_quotes(raw_artist or "")).strip()
    raw_clean = re.sub(r"\s+", " ", raw_clean)
    lowered = raw_clean.lower()
    if lowered in COMPILATION_ALIASES:
        return True, "Various", "various"

    match = TRAILING_ARTICLE_RE.match(raw_clean)
    if match:
        rest, article = match.group(1).strip(), match.group(2).strip()
        # Preserve original capitalization style of the article/rest while un-inverting
        uninverted = f"{article} {rest}"
        return False, uninverted, normalize_text(uninverted)

    clean_no_disambig = strip_discogs_disambiguator(raw_clean)
    return False, clean_no_disambig, normalize_text(clean_no_disambig)


def _apply_format_token(token: str, hint: FormatHint) -> None:
    upper = token.strip().upper()
    if upper not in {t.upper() for t in hint.tags}:
        hint.tags.append(token.strip())

    qty_match = QTY_MEDIUM_RE.match(upper)
    if qty_match:
        qty = int(qty_match.group(1))
        med = qty_match.group(2).upper()
        hint.qty = qty
        hint.medium = "Vinyl" if med == "LP" else "CD"
        return

    if upper in {"LP", "VINYL"}:
        hint.medium = "Vinyl"
        hint.qty = hint.qty or 1
    elif upper == "CD":
        hint.medium = "CD"
        hint.qty = hint.qty or 1
    elif upper == "EP":
        if "EP" not in hint.tags:
            hint.tags.append("EP")


def extract_format_and_clean_title(raw_title: str, artist_norm: str) -> tuple[str, str, FormatHint]:
    cleaned = unicodedata.normalize("NFKC", normalize_quotes(raw_title or "")).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    hint = FormatHint()

    # Strip duplicated leading artist prefix separated by ';' (e.g. "JETHRO TULL;BALOISE SESSION(2CD)")
    if ";" in cleaned:
        prefix, rest = cleaned.split(";", 1)
        prefix_norm = normalize_text(prefix)
        if artist_norm and prefix_norm and (
            prefix_norm == artist_norm
            or prefix_norm in artist_norm
            or artist_norm in prefix_norm
        ):
            cleaned = rest.strip()

    # Extract leading medium prefix like "CD - schönhauser EP"
    lead_match = LEADING_FORMAT_RE.match(cleaned)
    if lead_match:
        _apply_format_token(lead_match.group(1), hint)
        cleaned = cleaned[lead_match.end():].strip()

    # Extract parenthetical format/edition tags without eating core non-format subtitles
    def _replace_paren(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        upper = inner.upper()
        if upper.startswith("LIVE"):
            # Keep Live in title unless purely format; still record as hint tag
            if inner not in hint.tags:
                hint.tags.append(inner)
            return match.group(0)
        _apply_format_token(inner, hint)
        return " "

    cleaned = PAREN_FORMAT_RE.sub(_replace_paren, cleaned)
    if EP_SUFFIX_RE.search(cleaned) and "EP" not in [t.upper() for t in hint.tags]:
        hint.tags.append("EP")

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -;:,")
    title_norm = normalize_text(cleaned)
    return cleaned, title_norm, hint


def extract_label_catno_hint(title_clean: str) -> tuple[str, str, int] | None:
    """Detect '<Label> <Number>' inputs where the title encodes label + catalog number."""
    text = re.sub(r"\s+", " ", (title_clean or "")).strip()
    match = LABEL_CATNO_TITLE_RE.match(text)
    if not match:
        return None
    prefix = match.group(1).strip(" -–—:;/")
    raw_num = match.group(2)
    num_int = int(raw_num)
    if len(prefix) < 3 or num_int <= 0 or (1900 <= num_int <= 2099):
        return None
    last_word = prefix.split()[-1].lower()
    if last_word in NON_LABEL_SEQ_WORDS:
        return None
    return prefix, raw_num, num_int


def normalize_input(
    barcode: str = "",
    album_title: str = "",
    album_artist: str = "",
) -> NormalizedInput:
    barcode_raw = (barcode or "").strip()
    album_title_raw = (album_title or "").strip()
    album_artist_raw = (album_artist or "").strip()

    barcode_norm = normalize_barcode(barcode_raw)
    variants = barcode_variants(barcode_norm)
    is_compilation, artist_clean, artist_norm = normalize_artist(album_artist_raw)
    title_clean, title_norm, format_hint = extract_format_and_clean_title(
        album_title_raw, artist_norm
    )

    canonical_payload = json.dumps(
        [
            barcode_norm,
            title_norm,
            artist_norm,
            format_hint.medium,
            format_hint.qty,
            sorted(t.lower() for t in format_hint.tags),
        ],
        ensure_ascii=False,
    )
    input_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()

    return NormalizedInput(
        barcode_raw=barcode_raw,
        album_title_raw=album_title_raw,
        album_artist_raw=album_artist_raw,
        barcode_norm=barcode_norm,
        barcode_variants=variants,
        title_clean=title_clean,
        title_norm=title_norm,
        artist_clean=artist_clean,
        artist_norm=artist_norm,
        format_hint=format_hint,
        is_compilation=is_compilation,
        input_hash=input_hash,
    )
