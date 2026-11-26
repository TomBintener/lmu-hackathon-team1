import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.models import TrackArtistOut, TrackCreditOut, TrackOut
from app.normalizer import (
    barcode_variants,
    normalize_barcode,
    normalize_text,
    strip_discogs_disambiguator,
)

HEADING_DISC_RE = re.compile(
    r"\b(?:CD|DISC|LP|DVD|RECORD|VOL(?:UME)?)\s*#?\s*(\d+)\b", re.IGNORECASE
)
HEADING_SIDE_RE = re.compile(r"\bSIDE\s+([A-Z])\b", re.IGNORECASE)
POS_DISC_PREFIX_RE = re.compile(
    r"^(?:(?:CD|DISC|LP)\s*)?(\d+)\s*[-/]\s*(.+)$", re.IGNORECASE
)
POS_SIDE_RE = re.compile(r"^([A-L])(?:\d+[a-z]?)?$", re.IGNORECASE)
POS_NUMERIC_RE = re.compile(r"^(\d+)$")
DURATION_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})$")
BRACKET_QUALIFIER_RE = re.compile(r"\[[^\]]*\]")

CREDIT_BUCKETS = {
    "composer": "composers",
    "lyricist": "lyricists",
    "producer": "producers",
    "featured": "featured",
    "other": "other",
}

UNRESOLVED_ARTIST_NAMES = {
    "various",
    "various artists",
    "va",
    "v.a.",
    "verschiedene interpret:innen",
    "verschiedene interpreten",
    "unknown",
    "unknown artist",
}


def is_resolved_artist_name(name: str) -> bool:
    cleaned = (name or "").strip().lower()
    return bool(cleaned) and cleaned not in UNRESOLVED_ARTIST_NAMES


@dataclass
class ExtractedCredit:
    discogs_artist_id: int | None
    artist_name: str
    anv: str
    role_raw: str
    role_category: str
    credit_source: str
    source_url: str = ""


@dataclass
class ExtractedTrack:
    track_order: int
    disc_number: int
    disc_label: str
    position: str
    entry_type: str
    parent_position: str
    title: str
    duration_raw: str
    duration_sec: int | None
    performing_artists: list[TrackArtistOut]
    performing_artist_display: str
    has_valid_duration: bool
    has_resolved_performer: bool
    credits: list[ExtractedCredit] = field(default_factory=list)
    isrc: str = ""
    isrc_source: str = ""
    isrc_source_url: str = ""
    duration_source: str = ""
    duration_source_url: str = ""
    performer_source: str = ""
    performer_source_url: str = ""
    enrichment_notes: list[str] = field(default_factory=list)


@dataclass
class ExtractedRelease:
    discogs_release_id: int
    master_id: int | None
    url: str
    resource_url: str
    title: str
    title_norm: str
    artist_display: str
    artist_norm: str
    year: int | None
    country: str
    formats: list[dict[str, Any]]
    identifiers: list[dict[str, Any]]
    labels: list[dict[str, Any]]
    data_quality: str
    community_status: str
    community_have: int
    community_want: int
    notes: str
    track_count: int
    tracklist_hash: str
    unscoped_credits: list[dict[str, Any]]
    raw_json: dict[str, Any]
    barcode_entries: list[tuple[str, str]]
    tracks: list[ExtractedTrack]
    source_provider: str = "discogs"


def parse_duration_seconds(duration_raw: str) -> int | None:
    text = (duration_raw or "").strip()
    if not text:
        return None
    match = DURATION_RE.match(text)
    if not match:
        return None
    hours = int(match.group(1)) if match.group(1) is not None else 0
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    if seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def format_duration_raw(duration_sec: int | None) -> str:
    if duration_sec is None or duration_sec < 0:
        return ""
    if duration_sec >= 3600:
        hours, rem = divmod(duration_sec, 3600)
        mins, secs = divmod(rem, 60)
        return f"{hours}:{mins:02d}:{secs:02d}"
    mins, secs = divmod(duration_sec, 60)
    return f"{mins}:{secs:02d}"


def split_roles(role_raw: str) -> list[str]:
    """Split Discogs role strings on commas outside [...] qualifiers."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in role_raw or "":
        if ch == "[":
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                parts.append(item)
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts or ([(role_raw or "").strip()] if (role_raw or "").strip() else [])


def classify_role(role_part: str) -> str:
    base = BRACKET_QUALIFIER_RE.sub("", role_part).strip().lower()
    if not base:
        return "other"
    if any(kw in base for kw in ("lyric", "words by", "text by", "libretto")):
        return "lyricist"
    if any(
        kw in base
        for kw in ("composed", "composer", "written-by", "written by", "music by", "songwriter")
    ) or base == "music":
        return "composer"
    if "producer" in base or "produced by" in base:
        return "producer"
    if "featuring" in base or base.startswith("feat"):
        return "featured"
    return "other"


def format_artists(
    raw_artists: list[dict[str, Any]], inherited: bool = False
) -> tuple[list[TrackArtistOut], str, bool]:
    parsed: list[TrackArtistOut] = []
    chunks: list[str] = []
    resolved = False

    for idx, item in enumerate(raw_artists or []):
        name_clean = strip_discogs_disambiguator(str(item.get("name") or ""))
        anv = str(item.get("anv") or "").strip()
        join = str(item.get("join") or "").strip()
        artist_id = item.get("id")
        if name_clean or anv:
            parsed.append(
                TrackArtistOut(
                    discogs_artist_id=int(artist_id) if artist_id is not None else None,
                    name=name_clean or anv,
                    anv=anv,
                    join=join,
                    inherited=inherited,
                )
            )
            display_name = anv or name_clean
            if is_resolved_artist_name(display_name):
                resolved = True
            chunks.append(display_name)
            if idx < len(raw_artists) - 1:
                if join == ",":
                    chunks.append(", ")
                elif join:
                    chunks.append(f" {join} ")
                else:
                    chunks.append(", ")

    return parsed, "".join(chunks).strip(), resolved


def _infer_disc_from_position(
    position: str,
    heading_disc: int,
    prev_numeric_pos: int | None,
) -> tuple[int, int | None]:
    pos = (position or "").strip()
    if not pos:
        return heading_disc, prev_numeric_pos

    pref_match = POS_DISC_PREFIX_RE.match(pos)
    if pref_match:
        return int(pref_match.group(1)), None

    side_match = POS_SIDE_RE.match(pos)
    if side_match:
        letter = side_match.group(1).upper()
        disc = (ord(letter) - ord("A")) // 2 + 1
        return max(1, disc), None

    num_match = POS_NUMERIC_RE.match(pos)
    if num_match:
        num = int(num_match.group(1))
        if prev_numeric_pos is not None and num == 1 and prev_numeric_pos > 1:
            return heading_disc + 1, num
        return heading_disc, num

    return heading_disc, prev_numeric_pos


def _parse_extraartist_credits(
    entries: list[dict[str, Any]], credit_source: str, source_url: str = ""
) -> list[ExtractedCredit]:
    credits: list[ExtractedCredit] = []
    for entry in entries or []:
        name_clean = strip_discogs_disambiguator(str(entry.get("name") or ""))
        anv = str(entry.get("anv") or "").strip()
        role_raw_full = str(entry.get("role") or "").strip()
        artist_id = entry.get("id")
        if not (name_clean or anv) or not role_raw_full:
            continue
        for role_part in split_roles(role_raw_full):
            credits.append(
                ExtractedCredit(
                    discogs_artist_id=int(artist_id) if artist_id is not None else None,
                    artist_name=name_clean or anv,
                    anv=anv,
                    role_raw=role_part,
                    role_category=classify_role(role_part),
                    credit_source=credit_source,
                    source_url=source_url,
                )
            )
    return credits


def _resolve_track_indices(selector: str, ordered_positions: list[str]) -> list[int] | None:
    """Resolve Discogs release-level extraartist `tracks` strings into track indices.

    Returns None when any token cannot be matched cleanly so we never guess attribution.
    """
    pos_upper = [p.strip().upper() for p in ordered_positions]
    matched_indices: list[int] = []
    selector_clean = selector.replace("\u2013", " to ").replace("\u2014", " to ")
    tokens = [tok.strip() for tok in re.split(r"[,;]+", selector_clean) if tok.strip()]
    if not tokens:
        return None

    def _find_positions(tok_up: str) -> list[int]:
        exact = [i for i, p in enumerate(pos_upper) if p and p == tok_up]
        if exact:
            return exact
        if tok_up.isdigit():
            target_int = int(tok_up)
            return [i for i, p in enumerate(pos_upper) if p.isdigit() and int(p) == target_int]
        return []

    for token in tokens:
        tok_up = token.upper()
        exact = _find_positions(tok_up)
        if exact:
            matched_indices.extend(exact)
            continue

        # Range match priority: " TO " -> " - " (spaced) -> "-"
        start_tok = end_tok = ""
        if " TO " in tok_up:
            parts = tok_up.split(" TO ", 1)
            start_tok, end_tok = parts[0].strip(), parts[1].strip()
        elif " - " in tok_up:
            parts = tok_up.split(" - ", 1)
            start_tok, end_tok = parts[0].strip(), parts[1].strip()
        elif "-" in tok_up:
            parts = tok_up.split("-", 1)
            start_tok, end_tok = parts[0].strip(), parts[1].strip()

        if start_tok and end_tok:
            start_indices = _find_positions(start_tok)
            if start_indices:
                start_idx = start_indices[0]
                end_indices = [i for i in _find_positions(end_tok) if i >= start_idx]
                if end_indices:
                    end_idx = end_indices[0]
                    matched_indices.extend(range(start_idx, end_idx + 1))
                    continue

        return None

    return list(dict.fromkeys(matched_indices))


def extract_release(raw_json: dict[str, Any]) -> ExtractedRelease:
    release_id = int(raw_json["id"])
    master_id_raw = raw_json.get("master_id")
    master_id = int(master_id_raw) if master_id_raw is not None else None
    url = str(raw_json.get("uri") or f"https://www.discogs.com/release/{release_id}")
    if url.startswith("/"):
        url = f"https://www.discogs.com{url}"

    title = str(raw_json.get("title") or "").strip()
    title_norm = normalize_text(title)
    release_artists_raw = raw_json.get("artists") or []
    release_artists, artist_display, _ = format_artists(release_artists_raw, inherited=False)
    if (
        len(release_artists) == 1
        and strip_discogs_disambiguator(release_artists[0].name).lower()
        in {
            "various",
            "various artists",
            "va",
            "v.a.",
            "verschiedene interpret:innen",
            "verschiedene interpreten",
        }
    ):
        artist_norm = "various"
    else:
        artist_norm = normalize_text(artist_display)

    year_raw = raw_json.get("year")
    year = int(year_raw) if isinstance(year_raw, int) and year_raw > 0 else None
    country = str(raw_json.get("country") or "").strip()
    formats = list(raw_json.get("formats") or [])
    identifiers = list(raw_json.get("identifiers") or [])
    labels = list(raw_json.get("labels") or [])
    data_quality = str(raw_json.get("data_quality") or "").strip()
    community = raw_json.get("community") or {}
    community_status = str(community.get("status") or "").strip()
    community_have = int(community.get("have") or 0)
    community_want = int(community.get("want") or 0)
    notes = str(raw_json.get("notes") or "").strip()

    barcode_entries_map: dict[str, str] = {}
    for ident in identifiers:
        ident_type = str(ident.get("type") or "").strip().lower()
        if "barcode" in ident_type:
            raw_val = str(ident.get("value") or "").strip()
            norm_val = normalize_barcode(raw_val)
            for variant in barcode_variants(norm_val):
                barcode_entries_map.setdefault(variant, raw_val)
    barcode_entries = list(barcode_entries_map.items())

    source_provider = str(raw_json.get("_source_provider") or "discogs")

    tracks: list[ExtractedTrack] = []
    heading_disc = 1
    heading_label = ""
    prev_numeric_pos: int | None = None
    track_order = 0

    def _append_track(
        item: dict[str, Any],
        entry_type: str,
        parent_position: str = "",
        parent_artists_raw: list[dict[str, Any]] | None = None,
        parent_extra_raw: list[dict[str, Any]] | None = None,
    ) -> None:
        nonlocal heading_disc, prev_numeric_pos, track_order
        track_title = str(item.get("title") or "").strip()
        if not track_title:
            return
        track_order += 1
        position = str(item.get("position") or "").strip()
        disc_pos = parent_position if (entry_type == "sub_track" and parent_position) else position
        disc_num, prev_numeric_pos = _infer_disc_from_position(
            disc_pos, heading_disc, prev_numeric_pos
        )
        # Persist implicit or explicit disc advances for subsequent unprefixed/bonus tracks
        if disc_num > heading_disc:
            heading_disc = disc_num

        raw_track_artists = item.get("artists") or parent_artists_raw or []
        if raw_track_artists:
            artists, artist_disp, resolved = format_artists(raw_track_artists, inherited=False)
        else:
            artists, artist_disp, resolved = format_artists(release_artists_raw, inherited=True)

        duration_raw = str(item.get("duration") or "").strip()
        duration_sec = parse_duration_seconds(duration_raw)
        has_valid_dur = duration_sec is not None and duration_sec > 0
        merged_extra_raw = list(parent_extra_raw or []) + list(item.get("extraartists") or [])
        track_extra = _parse_extraartist_credits(
            merged_extra_raw, credit_source="track", source_url=url
        )

        tracks.append(
            ExtractedTrack(
                track_order=track_order,
                disc_number=disc_num,
                disc_label=heading_label,
                position=position,
                entry_type=entry_type,
                parent_position=parent_position,
                title=track_title,
                duration_raw=duration_raw,
                duration_sec=duration_sec,
                performing_artists=artists,
                performing_artist_display=artist_disp,
                has_valid_duration=has_valid_dur,
                has_resolved_performer=resolved,
                credits=track_extra,
                duration_source=source_provider if has_valid_dur else "",
                duration_source_url=url if has_valid_dur else "",
                performer_source=source_provider if resolved else "",
                performer_source_url=url if resolved else "",
            )
        )

    for item in raw_json.get("tracklist") or []:
        item_type = str(item.get("type_") or item.get("type") or "track").strip().lower()
        if item_type == "heading":
            heading_title = str(item.get("title") or "").strip()
            heading_label = heading_title
            disc_match = HEADING_DISC_RE.search(heading_title)
            if disc_match:
                heading_disc = int(disc_match.group(1))
                prev_numeric_pos = None
            else:
                side_match = HEADING_SIDE_RE.search(heading_title)
                if side_match:
                    letter = side_match.group(1).upper()
                    heading_disc = max(1, (ord(letter) - ord("A")) // 2 + 1)
                    prev_numeric_pos = None
            continue

        if item_type == "index":
            sub_tracks = item.get("sub_tracks") or []
            parent_pos = str(item.get("position") or "").strip()
            if sub_tracks:
                for sub in sub_tracks:
                    _append_track(
                        sub,
                        entry_type="sub_track",
                        parent_position=parent_pos,
                        parent_artists_raw=item.get("artists") or [],
                        parent_extra_raw=item.get("extraartists") or [],
                    )
            else:
                _append_track(item, entry_type="index")
            continue

        _append_track(item, entry_type="track")

    # Merge release-level extraartists using `tracks` scope selector
    ordered_positions = [t.position for t in tracks]
    unscoped_credits: list[dict[str, Any]] = []
    for rel_extra in raw_json.get("extraartists") or []:
        scope = str(rel_extra.get("tracks") or "").strip()
        if not scope:
            parsed_credits = _parse_extraartist_credits(
                [rel_extra], credit_source="release_all", source_url=url
            )
            for track in tracks:
                track.credits.extend(parsed_credits)
            continue

        target_indices = _resolve_track_indices(scope, ordered_positions)
        if target_indices is None:
            unscoped_credits.append(
                {
                    "discogs_artist_id": rel_extra.get("id"),
                    "artist_name": strip_discogs_disambiguator(str(rel_extra.get("name") or "")),
                    "anv": str(rel_extra.get("anv") or "").strip(),
                    "role_raw": str(rel_extra.get("role") or "").strip(),
                    "tracks_selector": scope,
                }
            )
            continue

        parsed_credits = _parse_extraartist_credits(
            [rel_extra], credit_source="release_range", source_url=url
        )
        for idx in target_indices:
            tracks[idx].credits.extend(parsed_credits)

    # Deduplicate credits per track
    for track in tracks:
        seen_credit_keys: set[tuple[str, str, str]] = set()
        deduped: list[ExtractedCredit] = []
        for c in track.credits:
            key = (c.artist_name.lower(), c.role_raw.lower(), c.credit_source)
            if key not in seen_credit_keys:
                seen_credit_keys.add(key)
                deduped.append(c)
        track.credits = deduped

    signature_items = [
        [t.disc_number, normalize_text(t.title), t.duration_sec]
        for t in tracks
    ]
    tracklist_hash = hashlib.sha256(
        json.dumps(signature_items, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    return ExtractedRelease(
        discogs_release_id=release_id,
        master_id=master_id,
        url=url,
        resource_url=str(raw_json.get("resource_url") or ""),
        title=title,
        title_norm=title_norm,
        artist_display=artist_display,
        artist_norm=artist_norm,
        year=year,
        country=country,
        formats=formats,
        identifiers=identifiers,
        labels=labels,
        data_quality=data_quality,
        community_status=community_status,
        community_have=community_have,
        community_want=community_want,
        notes=notes,
        track_count=len(tracks),
        tracklist_hash=tracklist_hash,
        unscoped_credits=unscoped_credits,
        raw_json=raw_json,
        barcode_entries=barcode_entries,
        tracks=tracks,
        source_provider=source_provider,
    )


def extracted_track_to_out(track: ExtractedTrack) -> TrackOut:
    buckets: dict[str, list[TrackCreditOut]] = {
        "composers": [],
        "lyricists": [],
        "producers": [],
        "featured": [],
        "other": [],
    }
    for credit in track.credits:
        bucket_key = CREDIT_BUCKETS.get(credit.role_category, "other")
        buckets[bucket_key].append(
            TrackCreditOut(
                discogs_artist_id=credit.discogs_artist_id,
                artist_name=credit.artist_name,
                anv=credit.anv,
                role_raw=credit.role_raw,
                role_category=credit.role_category,
                credit_source=credit.credit_source,
                source_url=credit.source_url,
            )
        )
    return TrackOut(
        track_order=track.track_order,
        disc_number=track.disc_number,
        disc_label=track.disc_label,
        position=track.position,
        entry_type=track.entry_type,
        parent_position=track.parent_position,
        title=track.title,
        duration_raw=track.duration_raw,
        duration_sec=track.duration_sec,
        performing_artists=track.performing_artists,
        performing_artist_display=track.performing_artist_display,
        has_valid_duration=track.has_valid_duration,
        has_resolved_performer=track.has_resolved_performer,
        isrc=track.isrc,
        isrc_source=track.isrc_source,
        isrc_source_url=track.isrc_source_url,
        duration_source=track.duration_source,
        duration_source_url=track.duration_source_url,
        performer_source=track.performer_source,
        performer_source_url=track.performer_source_url,
        enrichment_notes=list(track.enrichment_notes),
        credits=buckets,
    )
