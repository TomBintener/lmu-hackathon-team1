import hashlib
import json
import logging
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.agent_client import AgentFallbackError, AgentFallbackProvider
from app.config import Settings
from app.discogs_client import ApiCacheStore, CatalogProvider, DiscogsError
from app.evaluator import string_similarity
from app.extractor import (
    CREDIT_BUCKETS,
    ExtractedCredit,
    ExtractedRelease,
    classify_role,
    extract_release,
    format_duration_raw,
    is_resolved_artist_name,
)
from app.fallback_providers import ItunesFallbackProvider
from app.models import NormalizedInput, ReleaseOut, TrackArtistOut, TrackCreditOut, TrackOut
from app.normalizer import normalize_text

logger = logging.getLogger(__name__)

LIVE_ALBUM_RE = re.compile(
    r"\b(?:live|unplugged|in concert|baloise session)\b", re.IGNORECASE
)
PAREN_LIVE_RE = re.compile(r"\((?:[^)]*\b(?:live|unplugged|in concert)\b[^)]*)\)", re.IGNORECASE)
PAREN_BRACKET_RE = re.compile(r"[\(\[][^\)\]]*[\)\]]")
QUALIFIER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("live", re.compile(r"\b(?:live|unplugged|in concert)\b", re.IGNORECASE)),
    (
        "edit",
        re.compile(
            r"\b(?:radio\s+edit|single\s+edit|7\s*[\"'\u2033]?\s*(?:edit|version)|short\s+edit|edit)\b",
            re.IGNORECASE,
        ),
    ),
    ("remix", re.compile(r"\b(?:club\s+mix|dub\s+mix|remix(?:ed)?|rmx)\b", re.IGNORECASE)),
    (
        "extended",
        re.compile(
            r"\b(?:extended|12\s*[\"'\u2033]?\s*(?:mix|version|edit)?|maxi(?:\s+version)?)\b",
            re.IGNORECASE,
        ),
    ),
    ("acoustic", re.compile(r"\b(?:acoustic|akustik)\b", re.IGNORECASE)),
    ("instrumental", re.compile(r"\binstrumental\b", re.IGNORECASE)),
    ("demo", re.compile(r"\bdemo\b", re.IGNORECASE)),
    ("medley", re.compile(r"\bmedley\b", re.IGNORECASE)),
    ("karaoke", re.compile(r"\bkaraoke\b", re.IGNORECASE)),
]

MB_ROLE_MAP = {
    "composer": "composer",
    "writer": "composer",
    "lyricist": "lyricist",
    "librettist": "lyricist",
    "translator": "lyricist",
    "producer": "producer",
    "co-producer": "producer",
    "executive producer": "producer",
}


@dataclass
class DonorTrack:
    disc_number: int = 1
    position: str = ""
    title: str = ""
    duration_sec: int | None = None
    performing_artists: list[TrackArtistOut] = field(default_factory=list)
    performing_artist_display: str = ""
    isrc: str = ""
    credits: list[ExtractedCredit] = field(default_factory=list)
    source_label: str = ""
    source_url: str = ""
    external_track_id: str = ""


@dataclass
class EnrichmentStats:
    tracks_total: int = 0
    durations_filled: int = 0
    performers_filled: int = 0
    credits_added: int = 0
    isrcs_filled: int = 0
    duration_coverage_before: float = 0.0
    duration_coverage_after: float = 0.0
    performer_coverage_before: float = 0.0
    performer_coverage_after: float = 0.0
    writer_credit_coverage_before: float = 0.0
    writer_credit_coverage_after: float = 0.0
    producer_credit_coverage_before: float = 0.0
    producer_credit_coverage_after: float = 0.0
    providers_used: list[str] = field(default_factory=list)
    agent_input_tokens: int = 0
    agent_output_tokens: int = 0
    agent_cost_usd: float = 0.0
    tracks: list[TrackOut] = field(default_factory=list)


def _build_cache_key(provider: str, endpoint: str, params: dict[str, Any] | None) -> str:
    raw = json.dumps(
        {"provider": provider, "endpoint": endpoint, "params": params or {}},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_track_coverages(tracks: list[TrackOut]) -> tuple[float, float, float, float]:
    total = len(tracks)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    dur = round(sum(1 for t in tracks if t.has_valid_duration) / total, 4)
    perf = round(sum(1 for t in tracks if t.has_resolved_performer) / total, 4)
    writers = round(
        sum(
            1
            for t in tracks
            if (t.credits.get("composers") or t.credits.get("lyricists"))
        )
        / total,
        4,
    )
    producers = round(
        sum(1 for t in tracks if t.credits.get("producers")) / total,
        4,
    )
    return dur, perf, writers, producers


def track_needs_album_enrichment(track: TrackOut) -> bool:
    has_writer = bool(track.credits.get("composers") or track.credits.get("lyricists"))
    has_producer = bool(track.credits.get("producers"))
    return (
        not track.has_valid_duration
        or not track.has_resolved_performer
        or not track.isrc
        or not has_writer
        or not has_producer
    )


def track_needs_core_enrichment(track: TrackOut) -> bool:
    return not track.has_valid_duration or not track.has_resolved_performer or not track.isrc


def is_live_release(
    title: str,
    input_title: str = "",
    formats: list[dict[str, Any]] | None = None,
    notes: str = "",
    format_tags: list[str] | None = None,
) -> bool:
    for blob in (title, input_title, notes):
        if blob and LIVE_ALBUM_RE.search(blob):
            return True
    for tag in format_tags or []:
        if LIVE_ALBUM_RE.search(tag):
            return True
    for fmt in formats or []:
        for desc in fmt.get("descriptions") or []:
            if LIVE_ALBUM_RE.search(str(desc)):
                return True
        if fmt.get("text") and LIVE_ALBUM_RE.search(str(fmt["text"])):
            return True
    return False


def extract_version_qualifiers(text: str) -> set[str]:
    found: set[str] = set()
    cleaned = normalize_text(text)
    if not cleaned:
        return found
    for name, pattern in QUALIFIER_PATTERNS:
        if pattern.search(cleaned):
            found.add(name)
    return found


def qualifiers_compatible(
    target_title: str, candidate_title: str, allow_live_tag_mismatch: bool = False
) -> bool:
    q_target = extract_version_qualifiers(target_title)
    q_cand = extract_version_qualifiers(candidate_title)
    if allow_live_tag_mismatch:
        q_target.discard("live")
        q_cand.discard("live")
    return q_target == q_cand


def _title_match_score(
    target_title: str, candidate_title: str, allow_live_tag_mismatch: bool = False
) -> float:
    base = string_similarity(target_title, candidate_title)
    if allow_live_tag_mismatch:
        stripped_target = PAREN_LIVE_RE.sub("", target_title).strip()
        stripped_cand = PAREN_LIVE_RE.sub("", candidate_title).strip()
        if stripped_target and stripped_cand:
            base = max(base, string_similarity(stripped_target, stripped_cand))
    return base


def align_album_tracks(
    target_tracks: list[TrackOut],
    donor_tracks: list[DonorTrack],
    trusted_release_link: bool = False,
    allow_live_tag_mismatch: bool = False,
) -> dict[int, DonorTrack]:
    """Align donor album tracks to target tracks without guessing across versions."""
    if not target_tracks or not donor_tracks:
        return {}

    candidates: list[tuple[float, int, int]] = []
    for t_idx, target in enumerate(target_tracks):
        for d_idx, donor in enumerate(donor_tracks):
            if not qualifiers_compatible(
                target.title, donor.title, allow_live_tag_mismatch=allow_live_tag_mismatch
            ):
                continue
            sim = _title_match_score(
                target.title, donor.title, allow_live_tag_mismatch=allow_live_tag_mismatch
            )
            if sim < 0.85:
                continue
            bonus = 0.03 if target.disc_number == donor.disc_number else 0.0
            if t_idx == d_idx:
                bonus += 0.02
            candidates.append((sim + bonus, t_idx, d_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)
    matched: dict[int, DonorTrack] = {}
    used_targets: set[int] = set()
    used_donors: set[int] = set()

    for _, t_idx, d_idx in candidates:
        if t_idx in used_targets or d_idx in used_donors:
            continue
        used_targets.add(t_idx)
        used_donors.add(d_idx)
        matched[target_tracks[t_idx].track_order] = donor_tracks[d_idx]

    if not matched:
        return {}

    min_required = 1 if trusted_release_link else max(1, math.ceil(0.60 * len(target_tracks)))
    if len(matched) < min_required:
        return {}
    return matched


def select_best_single_track(
    target_title: str,
    target_artist: str,
    candidates: list[DonorTrack],
    provider_name: str,
) -> tuple[DonorTrack | None, str | None]:
    """Pick the strongest unambiguous standalone track match, detecting >10s duration conflicts."""
    passing: list[tuple[float, DonorTrack]] = []
    for cand in candidates:
        if not qualifiers_compatible(target_title, cand.title):
            continue
        t_sim = string_similarity(target_title, cand.title)
        if t_sim < 0.88:
            continue
        if target_artist and is_resolved_artist_name(target_artist):
            if not cand.performing_artist_display:
                continue
            a_sim = string_similarity(target_artist, cand.performing_artist_display)
            if a_sim < 0.75:
                continue
            combined = 0.60 * t_sim + 0.40 * a_sim
        else:
            combined = t_sim
        passing.append((combined, cand))

    if not passing:
        return None, None

    passing.sort(key=lambda item: item[0], reverse=True)
    best_score, best = passing[0]
    close_durations = [
        cand.duration_sec
        for score, cand in passing
        if (best_score - score) <= 0.05
        and cand.duration_sec is not None
        and cand.duration_sec > 0
    ]
    if close_durations and (max(close_durations) - min(close_durations)) > 10:
        note = (
            f"Duration conflict on {provider_name} "
            f"({min(close_durations)}s vs {max(close_durations)}s); skipped duration"
        )
        safe_donor = DonorTrack(
            disc_number=best.disc_number,
            position=best.position,
            title=best.title,
            duration_sec=None,
            performing_artists=best.performing_artists,
            performing_artist_display=best.performing_artist_display,
            isrc=best.isrc,
            credits=best.credits,
            source_label=best.source_label,
            source_url=best.source_url,
        )
        return safe_donor, note
    return best, None


def _append_note(track: TrackOut, note: str) -> None:
    cleaned = (note or "").strip()
    if cleaned and cleaned not in track.enrichment_notes:
        track.enrichment_notes.append(cleaned)


def apply_donor_to_track(
    track: TrackOut, donor: DonorTrack, stats: EnrichmentStats
) -> bool:
    """Merge missing duration, performer, ISRC, and credits from donor into track."""
    changed = False

    if (
        not track.has_valid_duration
        and donor.duration_sec is not None
        and donor.duration_sec > 0
    ):
        track.duration_sec = donor.duration_sec
        track.duration_raw = format_duration_raw(donor.duration_sec)
        track.has_valid_duration = True
        track.duration_source = donor.source_label
        track.duration_source_url = donor.source_url
        stats.durations_filled += 1
        changed = True

    if (
        not track.has_resolved_performer
        and donor.performing_artist_display
        and is_resolved_artist_name(donor.performing_artist_display)
    ):
        track.performing_artists = list(donor.performing_artists) or [
            TrackArtistOut(name=donor.performing_artist_display, inherited=False)
        ]
        track.performing_artist_display = donor.performing_artist_display
        track.has_resolved_performer = True
        track.performer_source = donor.source_label
        track.performer_source_url = donor.source_url
        stats.performers_filled += 1
        changed = True

    if not track.isrc and donor.isrc:
        track.isrc = donor.isrc.strip().upper()
        track.isrc_source = donor.source_label
        track.isrc_source_url = donor.source_url
        stats.isrcs_filled += 1
        changed = True

    existing_keys: set[tuple[str, str]] = set()
    for bucket_items in track.credits.values():
        for c in bucket_items:
            existing_keys.add((c.artist_name.lower(), c.role_category.lower()))

    for credit in donor.credits:
        bucket_key = CREDIT_BUCKETS.get(credit.role_category, "other")
        dedup_key = (credit.artist_name.lower(), credit.role_category.lower())
        if dedup_key in existing_keys:
            continue
        existing_keys.add(dedup_key)
        track.credits.setdefault(bucket_key, []).append(
            TrackCreditOut(
                discogs_artist_id=credit.discogs_artist_id,
                artist_name=credit.artist_name,
                anv=credit.anv,
                role_raw=credit.role_raw,
                role_category=credit.role_category,
                credit_source=credit.credit_source or donor.source_label,
                source_url=credit.source_url or donor.source_url,
            )
        )
        stats.credits_added += 1
        changed = True

    return changed


def _extracted_release_to_donors(
    rel: ExtractedRelease, source_label: str | None = None
) -> list[DonorTrack]:
    label = source_label or rel.source_provider or "discogs"
    donors: list[DonorTrack] = []
    for t in rel.tracks:
        donors.append(
            DonorTrack(
                disc_number=t.disc_number,
                position=t.position,
                title=t.title,
                duration_sec=t.duration_sec if t.has_valid_duration else None,
                performing_artists=list(t.performing_artists),
                performing_artist_display=t.performing_artist_display if t.has_resolved_performer else "",
                isrc=t.isrc,
                credits=list(t.credits),
                source_label=label,
                source_url=rel.url,
            )
        )
    return donors


def _cached_http_json(
    client: httpx.Client,
    cache_store: ApiCacheStore | None,
    provider_name: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
    bypass_cache: bool = False,
    rate_lock: Any | None = None,
    last_request_ref: list[float] | None = None,
    min_interval_sec: float = 0.0,
    max_retries: int = 1,
) -> dict[str, Any] | None:
    cache_key = _build_cache_key(provider_name, endpoint, params)
    if cache_store is not None and not bypass_cache:
        cached = cache_store.get_api_cache(cache_key)
        if cached is not None:
            payload = cached.get("payload_json")
            return payload if isinstance(payload, dict) else None

    for attempt in range(max_retries + 1):
        if rate_lock is not None and last_request_ref is not None and min_interval_sec > 0:
            with rate_lock:
                elapsed = time.monotonic() - last_request_ref[0]
                if last_request_ref[0] > 0 and elapsed < min_interval_sec:
                    time.sleep(min_interval_sec - elapsed)
                last_request_ref[0] = time.monotonic()

        try:
            resp = client.get(endpoint, params=params)
        except Exception as exc:
            logger.warning("%s request error on %s: %s", provider_name, endpoint, exc)
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return None

        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt < max_retries:
                time.sleep(1.2 * (attempt + 1))
                continue
            logger.warning("%s returned HTTP %s on %s", provider_name, resp.status_code, endpoint)
            return None

        if resp.status_code == 404:
            if cache_store is not None:
                cache_store.set_api_cache(cache_key, endpoint, 404, {"_not_found": True})
            return None

        if resp.status_code >= 400:
            logger.warning("%s returned HTTP %s on %s", provider_name, resp.status_code, endpoint)
            return None

        try:
            payload = resp.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None

        if cache_store is not None:
            cache_store.set_api_cache(cache_key, endpoint, resp.status_code, payload)
        return payload
    return None


class DiscogsMasterVersionEnricher:
    """Tier 1: Fetch CD/File sibling versions sharing the matched release's master_id."""

    name = "discogs_master"

    def __init__(self, provider: CatalogProvider, max_versions: int = 2) -> None:
        self._provider = provider
        self._max_versions = max(1, max_versions)

    @staticmethod
    def _rank_version(ver: dict[str, Any]) -> tuple[int, int, int]:
        major = {str(m).lower() for m in (ver.get("major_formats") or [])}
        fmt_str = str(ver.get("format") or "").lower()
        fmt_score = 0
        if "cd" in major or "cd" in fmt_str:
            fmt_score = 30
        elif "file" in major or "digital" in fmt_str or "web" in fmt_str:
            fmt_score = 25
        elif "sacd" in fmt_str:
            fmt_score = 20
        elif "cassette" in major:
            fmt_score = 5

        country = str(ver.get("country") or "").lower()
        country_score = 3 if country in {"germany", "europe", "us", "uk"} else 0
        status_score = 2 if str(ver.get("status") or "").lower() == "accepted" else 0
        return (fmt_score, country_score, status_score)

    def fetch_sibling_releases(
        self,
        master_id: int,
        exclude_release_id: int | None,
        bypass_cache: bool = False,
    ) -> list[ExtractedRelease]:
        get_versions = getattr(self._provider, "get_master_versions", None)
        if not callable(get_versions):
            return []

        try:
            payload, _ = get_versions(master_id, per_page=25, bypass_cache=bypass_cache)
        except DiscogsError as exc:
            logger.warning("Discogs master versions failed for %s: %s", master_id, exc)
            return []

        versions = [
            v
            for v in (payload.get("versions") or [])
            if v.get("id") is not None and int(v["id"]) != exclude_release_id
        ]
        versions.sort(key=self._rank_version, reverse=True)

        extracted: list[ExtractedRelease] = []
        for ver in versions[: self._max_versions]:
            vid = int(ver["id"])
            try:
                raw_rel, _ = self._provider.get_release(vid, bypass_cache=bypass_cache)
            except DiscogsError as exc:
                logger.warning("Discogs master version fetch failed for %s: %s", vid, exc)
                continue
            if raw_rel and "id" in raw_rel:
                extracted.append(extract_release(raw_rel))
        return extracted


class MusicBrainzEnricher:
    """Tier 2a + 2b: Free MusicBrainz WS/2 enrichment for durations, artists, credits, and ISRCs."""

    name = "musicbrainz"

    def __init__(
        self,
        settings: Settings,
        cache_store: ApiCacheStore | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._cache = cache_store
        self._own_client = http_client is None
        ua = f"{settings.discogs_user_agent} (https://github.com/cloudIT4IPM/ai4-lmu-hackathon)"
        self._client = http_client or httpx.Client(
            base_url="https://musicbrainz.org/ws/2",
            timeout=settings.enrichment_timeout_sec,
            headers={"User-Agent": ua, "Accept": "application/json"},
        )
        self._rate_lock = threading.Lock()
        self._last_req = [0.0]

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def _get(
        self, endpoint: str, params: dict[str, Any], bypass_cache: bool = False
    ) -> dict[str, Any] | None:
        payload = _cached_http_json(
            client=self._client,
            cache_store=self._cache,
            provider_name=self.name,
            endpoint=endpoint,
            params=params,
            bypass_cache=bypass_cache,
            rate_lock=self._rate_lock,
            last_request_ref=self._last_req,
            min_interval_sec=self._settings.musicbrainz_min_interval_sec,
            max_retries=1,
        )
        if not payload or payload.get("_not_found"):
            return None
        return payload

    @staticmethod
    def _parse_artist_credit(
        artist_credit: list[dict[str, Any]] | None,
    ) -> tuple[list[TrackArtistOut], str]:
        artists: list[TrackArtistOut] = []
        chunks: list[str] = []
        for idx, item in enumerate(artist_credit or []):
            if not isinstance(item, dict):
                continue
            artist_obj = item.get("artist") or {}
            name = str(artist_obj.get("name") or item.get("name") or "").strip()
            joinphrase = str(item.get("joinphrase") or "")
            if not name:
                continue
            artists.append(TrackArtistOut(name=name, join=joinphrase.strip(), inherited=False))
            chunks.append(name)
            if idx < len(artist_credit or []) - 1:
                chunks.append(joinphrase if joinphrase else ", ")
        return artists, "".join(chunks).strip()

    @staticmethod
    def _parse_recording_credits(
        recording: dict[str, Any], source_url: str
    ) -> list[ExtractedCredit]:
        credits: list[ExtractedCredit] = []
        seen: set[tuple[str, str]] = set()

        def _add(artist_name: str, role_raw: str, category: str) -> None:
            name_clean = artist_name.strip()
            if not name_clean:
                return
            key = (name_clean.lower(), role_raw.lower())
            if key in seen:
                return
            seen.add(key)
            credits.append(
                ExtractedCredit(
                    discogs_artist_id=None,
                    artist_name=name_clean,
                    anv="",
                    role_raw=role_raw,
                    role_category=category,
                    credit_source="musicbrainz",
                    source_url=source_url,
                )
            )

        for rel in recording.get("relations") or []:
            ttype = str(rel.get("target-type") or "").lower()
            if ttype == "artist":
                artist_name = str((rel.get("artist") or {}).get("name") or "").strip()
                role_type = str(rel.get("type") or "").strip()
                attrs = [str(a).lower() for a in (rel.get("attributes") or [])]
                role_low = role_type.lower()
                if role_low in MB_ROLE_MAP:
                    cat = MB_ROLE_MAP[role_low]
                elif any("guest" in a or "featured" in a for a in attrs):
                    cat = "featured"
                else:
                    cat = classify_role(role_type)
                role_label = role_type.title() if role_type else "Credit"
                _add(artist_name, role_label, cat)
            elif ttype == "work":
                work = rel.get("work") or {}
                for wrel in work.get("relations") or []:
                    if str(wrel.get("target-type") or "").lower() != "artist":
                        continue
                    artist_name = str((wrel.get("artist") or {}).get("name") or "").strip()
                    wtype = str(wrel.get("type") or "").strip()
                    wlow = wtype.lower()
                    cat = MB_ROLE_MAP.get(wlow, classify_role(wtype))
                    role_label = wtype.title() if wtype else "Writer"
                    _add(artist_name, role_label, cat)

        return credits

    def _recording_to_donor(
        self,
        recording: dict[str, Any],
        source_url: str,
        disc_number: int = 1,
        position: str = "",
        track_title: str = "",
        track_length_ms: Any = None,
        track_artist_credit: list[dict[str, Any]] | None = None,
    ) -> DonorTrack:
        title = (track_title or str(recording.get("title") or "")).strip()
        len_ms = track_length_ms if track_length_ms is not None else recording.get("length")
        duration_sec: int | None = None
        if len_ms is not None:
            try:
                sec = int(round(float(len_ms) / 1000.0))
                if sec > 0:
                    duration_sec = sec
            except (TypeError, ValueError):
                duration_sec = None

        ac = track_artist_credit or recording.get("artist-credit") or []
        artists, artist_disp = self._parse_artist_credit(ac)
        isrcs = [str(i).strip().upper() for i in (recording.get("isrcs") or []) if str(i).strip()]
        credits = self._parse_recording_credits(recording, source_url)

        return DonorTrack(
            disc_number=disc_number,
            position=position,
            title=title,
            duration_sec=duration_sec,
            performing_artists=artists,
            performing_artist_display=artist_disp,
            isrc=isrcs[0] if isrcs else "",
            credits=credits,
            source_label=self.name,
            source_url=source_url,
            external_track_id=str(recording.get("id") or ""),
        )

    def fetch_album_donors(
        self,
        norm: NormalizedInput,
        target_is_live: bool,
        bypass_cache: bool = False,
    ) -> tuple[list[DonorTrack], bool, bool]:
        """Return (donor_tracks, trusted_barcode_match, donor_is_live)."""
        candidate_releases: list[tuple[dict[str, Any], bool]] = []

        for ean in norm.barcode_variants:
            payload = self._get(
                "/release/",
                {"query": f"barcode:{ean}", "fmt": "json", "limit": 5},
                bypass_cache=bypass_cache,
            )
            for rel in (payload or {}).get("releases") or []:
                candidate_releases.append((rel, True))
            if candidate_releases:
                break

        if not candidate_releases and norm.title_clean:
            title_q = norm.title_clean.replace('"', "")
            if norm.artist_clean and not norm.is_compilation:
                art_q = norm.artist_clean.replace('"', "")
                query = f'release:"{title_q}" AND artist:"{art_q}"'
            else:
                query = f'release:"{title_q}"'
            payload = self._get(
                "/release/",
                {"query": query, "fmt": "json", "limit": 5},
                bypass_cache=bypass_cache,
            )
            for rel in (payload or {}).get("releases") or []:
                candidate_releases.append((rel, False))

        if not candidate_releases:
            return [], False, False

        scored: list[tuple[float, str, bool, bool]] = []
        for rel, by_barcode in candidate_releases:
            mbid = str(rel.get("id") or "").strip()
            rel_title = str(rel.get("title") or "").strip()
            if not mbid or not rel_title:
                continue
            cand_live = is_live_release(rel_title)
            if not by_barcode and target_is_live != cand_live:
                continue
            t_sim = string_similarity(norm.title_clean, rel_title) if norm.title_clean else 0.7
            if not by_barcode and t_sim < 0.55:
                continue
            _, rel_artist = self._parse_artist_credit(rel.get("artist-credit"))
            if not by_barcode and norm.artist_clean and not norm.is_compilation:
                a_sim = string_similarity(norm.artist_clean, rel_artist)
                if a_sim < 0.50:
                    continue
            else:
                a_sim = 1.0 if by_barcode else 0.6
            media = rel.get("media") or []
            fmt_bonus = 0.05 if any("cd" in str(m.get("format") or "").lower() or "digital" in str(m.get("format") or "").lower() for m in media) else 0.0
            scored.append((t_sim + a_sim + fmt_bonus, mbid, by_barcode, cand_live))

        if not scored:
            return [], False, False
        scored.sort(key=lambda item: item[0], reverse=True)
        _, best_mbid, by_barcode, cand_live = scored[0]

        detail = self._get(
            f"/release/{best_mbid}",
            {
                "inc": "recordings+artist-credits+isrcs+recording-rels+work-rels+work-level-rels+artist-rels",
                "fmt": "json",
            },
            bypass_cache=bypass_cache,
        )
        if not detail:
            return [], False, False

        release_url = f"https://musicbrainz.org/release/{best_mbid}"
        donors: list[DonorTrack] = []
        for disc_idx, medium in enumerate(detail.get("media") or [], start=1):
            disc_num = int(medium.get("position") or disc_idx)
            for tr in medium.get("tracks") or []:
                rec = tr.get("recording") or {}
                donors.append(
                    self._recording_to_donor(
                        recording=rec,
                        source_url=release_url,
                        disc_number=disc_num,
                        position=str(tr.get("number") or ""),
                        track_title=str(tr.get("title") or rec.get("title") or "").strip(),
                        track_length_ms=tr.get("length"),
                        track_artist_credit=tr.get("artist-credit"),
                    )
                )
        return donors, by_barcode, cand_live

    def enrich_single_track(
        self,
        track_title: str,
        track_artist: str,
        need_details: bool = True,
        bypass_cache: bool = False,
    ) -> tuple[DonorTrack | None, str | None]:
        t_clean = track_title.replace('"', "").strip()
        a_clean = track_artist.replace('"', "").strip()
        if not t_clean or not is_resolved_artist_name(a_clean):
            return None, None

        payload = self._get(
            "/recording/",
            {"query": f'recording:"{t_clean}" AND artist:"{a_clean}"', "fmt": "json", "limit": 5},
            bypass_cache=bypass_cache,
        )
        recs = (payload or {}).get("recordings") or []
        if not recs:
            return None, None

        candidates = [
            self._recording_to_donor(
                recording=r,
                source_url=f"https://musicbrainz.org/recording/{r.get('id')}",
            )
            for r in recs
            if r.get("id")
        ]
        winner, conflict_note = select_best_single_track(
            track_title, track_artist, candidates, self.name
        )
        if winner is None:
            return None, conflict_note

        if need_details and winner.external_track_id:
            detail = self._get(
                f"/recording/{winner.external_track_id}",
                {"inc": "isrcs+artist-credits+artist-rels+work-rels+work-level-rels", "fmt": "json"},
                bypass_cache=bypass_cache,
            )
            if detail and detail.get("id"):
                detailed = self._recording_to_donor(
                    recording=detail,
                    source_url=f"https://musicbrainz.org/recording/{detail['id']}",
                )
                if conflict_note is not None:
                    detailed.duration_sec = None
                return detailed, conflict_note

        return winner, conflict_note


class DeezerEnricher:
    """Tier 2a + 2b: Free Deezer Public API enrichment (durations, artists, ISRCs)."""

    name = "deezer"

    def __init__(
        self,
        settings: Settings,
        cache_store: ApiCacheStore | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._cache = cache_store
        self._own_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url="https://api.deezer.com",
            timeout=settings.enrichment_timeout_sec,
            headers={"User-Agent": settings.discogs_user_agent},
        )

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def _get(
        self, endpoint: str, params: dict[str, Any] | None = None, bypass_cache: bool = False
    ) -> dict[str, Any] | None:
        payload = _cached_http_json(
            client=self._client,
            cache_store=self._cache,
            provider_name=self.name,
            endpoint=endpoint,
            params=params,
            bypass_cache=bypass_cache,
            max_retries=1,
        )
        if not payload or payload.get("_not_found") or "error" in payload:
            return None
        return payload

    def _album_to_donors(self, album_obj: dict[str, Any]) -> list[DonorTrack]:
        album_url = str(album_obj.get("link") or "")
        donors: list[DonorTrack] = []
        for idx, tr in enumerate((album_obj.get("tracks") or {}).get("data") or [], start=1):
            title = str(tr.get("title") or "").strip()
            if not title:
                continue
            dur_raw = tr.get("duration")
            dur_sec = int(dur_raw) if isinstance(dur_raw, int) and dur_raw > 0 else None
            artist_name = str((tr.get("artist") or {}).get("name") or "").strip()
            artists = (
                [TrackArtistOut(name=artist_name, inherited=False)]
                if is_resolved_artist_name(artist_name)
                else []
            )
            donors.append(
                DonorTrack(
                    disc_number=int(tr.get("disk_number") or 1),
                    position=str(tr.get("track_position") or idx),
                    title=title,
                    duration_sec=dur_sec,
                    performing_artists=artists,
                    performing_artist_display=artist_name if artists else "",
                    isrc=str(tr.get("isrc") or "").strip().upper(),
                    source_label=self.name,
                    source_url=str(tr.get("link") or album_url),
                    external_track_id=str(tr.get("id") or ""),
                )
            )
        return donors

    def fetch_album_donors(
        self,
        norm: NormalizedInput,
        target_is_live: bool,
        bypass_cache: bool = False,
    ) -> tuple[list[DonorTrack], bool, bool]:
        for ean in norm.barcode_variants:
            alb = self._get(f"/album/upc:{ean}", bypass_cache=bypass_cache)
            if alb and alb.get("id") and (alb.get("tracks") or {}).get("data"):
                alb_title = str(alb.get("title") or "")
                return self._album_to_donors(alb), True, is_live_release(alb_title)

        if not norm.title_clean:
            return [], False, False

        queries = []
        if norm.artist_clean and not norm.is_compilation:
            queries.append(f"{norm.artist_clean} {norm.title_clean}")
            queries.append(f'artist:"{norm.artist_clean}" album:"{norm.title_clean}"')
        else:
            queries.append(norm.title_clean)
            queries.append(f'album:"{norm.title_clean}"')

        scored: list[tuple[float, int, bool]] = []
        for q in queries:
            res = self._get("/search/album", params={"q": q, "limit": 5}, bypass_cache=bypass_cache)
            for item in (res or {}).get("data") or []:
                alb_id = item.get("id")
                alb_title = str(item.get("title") or "").strip()
                if not alb_id or not alb_title:
                    continue
                cand_live = is_live_release(alb_title)
                if target_is_live != cand_live:
                    continue
                t_sim = string_similarity(norm.title_clean, alb_title)
                if t_sim < 0.55:
                    continue
                alb_artist = str((item.get("artist") or {}).get("name") or "").strip()
                if norm.artist_clean and not norm.is_compilation:
                    a_sim = string_similarity(norm.artist_clean, alb_artist)
                    if a_sim < 0.50:
                        continue
                else:
                    a_sim = 0.6
                scored.append((t_sim + a_sim, int(alb_id), cand_live))
            if scored:
                break

        if not scored:
            return [], False, False
        scored.sort(key=lambda pair: pair[0], reverse=True)
        _, best_id, cand_live = scored[0]
        alb = self._get(f"/album/{best_id}", bypass_cache=bypass_cache)
        if not alb:
            return [], False, False
        return self._album_to_donors(alb), False, cand_live

    def fetch_track_isrc(
        self, deezer_track_id: str, bypass_cache: bool = False
    ) -> tuple[str, str]:
        if not deezer_track_id:
            return "", ""
        payload = self._get(f"/track/{deezer_track_id}", bypass_cache=bypass_cache)
        if not payload:
            return "", ""
        return (
            str(payload.get("isrc") or "").strip().upper(),
            str(payload.get("link") or ""),
        )

    def enrich_single_track(
        self,
        track_title: str,
        track_artist: str,
        need_isrc: bool = True,
        bypass_cache: bool = False,
    ) -> tuple[DonorTrack | None, str | None]:
        t_clean = track_title.replace('"', "").strip()
        a_clean = track_artist.replace('"', "").strip()
        if not t_clean or not is_resolved_artist_name(a_clean):
            return None, None

        base_title = re.sub(r"\s+", " ", PAREN_BRACKET_RE.sub(" ", t_clean)).strip()
        queries = [f"{a_clean} {t_clean}"]
        if base_title and base_title.lower() != t_clean.lower():
            queries.append(f"{a_clean} {base_title}")

        winner: DonorTrack | None = None
        conflict_note: str | None = None
        for q in queries:
            res = self._get(
                "/search/track",
                params={"q": q, "limit": 10},
                bypass_cache=bypass_cache,
            )
            items = (res or {}).get("data") or []
            if not items:
                continue

            candidates: list[DonorTrack] = []
            for tr in items:
                title = str(tr.get("title") or "").strip()
                if not title:
                    continue
                dur_raw = tr.get("duration")
                dur_sec = int(dur_raw) if isinstance(dur_raw, int) and dur_raw > 0 else None
                artist_name = str((tr.get("artist") or {}).get("name") or "").strip()
                artists = (
                    [TrackArtistOut(name=artist_name, inherited=False)]
                    if is_resolved_artist_name(artist_name)
                    else []
                )
                candidates.append(
                    DonorTrack(
                        title=title,
                        duration_sec=dur_sec,
                        performing_artists=artists,
                        performing_artist_display=artist_name if artists else "",
                        isrc=str(tr.get("isrc") or "").strip().upper(),
                        source_label=self.name,
                        source_url=str(tr.get("link") or ""),
                        external_track_id=str(tr.get("id") or ""),
                    )
                )

            winner, conflict_note = select_best_single_track(
                track_title, track_artist, candidates, self.name
            )
            if winner is not None or conflict_note is not None:
                break
        if winner is not None and need_isrc and not winner.isrc and winner.external_track_id:
            isrc, tr_link = self.fetch_track_isrc(
                winner.external_track_id, bypass_cache=bypass_cache
            )
            if isrc:
                winner.isrc = isrc
                if tr_link:
                    winner.source_url = tr_link
        return winner, conflict_note


class ItunesTrackEnricher:
    """Tier 2a + 2b: Reuses iTunes Search/Lookup for missing durations & performers."""

    name = "itunes"

    def __init__(
        self,
        settings: Settings,
        cache_store: ApiCacheStore | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._provider = ItunesFallbackProvider(
            settings, cache_store=cache_store, http_client=http_client
        )

    def close(self) -> None:
        self._provider.close()

    def fetch_album_donors(
        self,
        norm: NormalizedInput,
        target_is_live: bool,
        bypass_cache: bool = False,
    ) -> tuple[list[DonorTrack], bool]:
        cands, _, _ = self._provider.lookup_candidates(norm, bypass_cache=bypass_cache)
        for rel, _ in cands:
            cand_live = is_live_release(rel.title)
            if target_is_live != cand_live:
                continue
            return _extracted_release_to_donors(rel, source_label=self.name), cand_live
        return [], False

    def enrich_single_track(
        self,
        track_title: str,
        track_artist: str,
        bypass_cache: bool = False,
    ) -> tuple[DonorTrack | None, str | None]:
        if not track_title.strip() or not is_resolved_artist_name(track_artist):
            return None, None
        payload, _, err = self._provider._get_json_cached(
            "/search",
            {
                "term": f"{track_artist} {track_title}",
                "entity": "song",
                "country": "DE",
                "limit": 5,
            },
            bypass_cache=bypass_cache,
        )
        if err or not payload:
            return None, None

        candidates: list[DonorTrack] = []
        for item in payload.get("results") or []:
            if str(item.get("wrapperType") or "").lower() != "track":
                continue
            title = str(item.get("trackName") or "").strip()
            if not title:
                continue
            ms_raw = item.get("trackTimeMillis")
            dur_sec: int | None = None
            if ms_raw is not None:
                try:
                    sec = int(round(float(ms_raw) / 1000.0))
                    if sec > 0:
                        dur_sec = sec
                except (TypeError, ValueError):
                    dur_sec = None
            artist_name = str(item.get("artistName") or "").strip()
            artists = (
                [TrackArtistOut(name=artist_name, inherited=False)]
                if is_resolved_artist_name(artist_name)
                else []
            )
            candidates.append(
                DonorTrack(
                    title=title,
                    duration_sec=dur_sec,
                    performing_artists=artists,
                    performing_artist_display=artist_name if artists else "",
                    source_label=self.name,
                    source_url=str(item.get("trackViewUrl") or item.get("collectionViewUrl") or ""),
                )
            )
        return select_best_single_track(track_title, track_artist, candidates, self.name)


class GeminiTrackEnricher:
    """Tier 3: Batched grounded Google Search enrichment for remaining gaps."""

    name = "gemini_agent"

    def __init__(self, agent_provider: AgentFallbackProvider) -> None:
        self._agent = agent_provider

    def enrich_tracks_batch(
        self,
        norm: NormalizedInput,
        release_url: str,
        tracks: list[TrackOut],
    ) -> tuple[dict[int, DonorTrack], int, int, float]:
        enrich_fn = getattr(self._agent, "enrich_missing_tracks", None)
        if not callable(enrich_fn):
            return {}, 0, 0, 0.0

        missing_payload: list[dict[str, Any]] = []
        for t in tracks:
            if not track_needs_album_enrichment(t):
                continue
            missing_fields: list[str] = []
            if not t.has_valid_duration:
                missing_fields.append("length_sec")
            if not t.has_resolved_performer:
                missing_fields.append("artist")
            if not t.isrc:
                missing_fields.append("isrc")
            if not (t.credits.get("composers") or t.credits.get("lyricists")):
                missing_fields.append("composers/lyricists")
            if not t.credits.get("producers"):
                missing_fields.append("producers")
            missing_payload.append(
                {
                    "track_order": t.track_order,
                    "title": t.title,
                    "current_artist": t.performing_artist_display or norm.artist_clean,
                    "missing_fields": missing_fields,
                }
            )

        if not missing_payload:
            return {}, 0, 0, 0.0

        donors_by_order: dict[int, DonorTrack] = {}
        total_in = 0
        total_out = 0
        total_cost = 0.0
        for start in range(0, len(missing_payload), 25):
            chunk = missing_payload[start : start + 25]
            try:
                ans = enrich_fn(
                    barcode=norm.barcode_raw,
                    album=norm.album_title_raw or norm.title_clean,
                    artist=norm.album_artist_raw or norm.artist_clean,
                    release_url=release_url,
                    missing_tracks=chunk,
                )
            except AgentFallbackError as exc:
                logger.warning("Gemini track enrichment chunk failed: %s", exc)
                continue

            total_in += int(getattr(ans, "input_tokens", 0) or 0)
            total_out += int(getattr(ans, "output_tokens", 0) or 0)
            total_cost += float(getattr(ans, "cost_usd", 0.0) or 0.0)

            fallback_url = ans.sources[0] if ans.sources else ""
            for item in ans.tracks:
                src_url = (item.source_url or fallback_url or "").strip()
                if not src_url:
                    continue
                dur_sec = (
                    int(item.length_sec)
                    if item.length_sec is not None and item.length_sec > 0
                    else None
                )
                artist_clean = (item.artist or "").strip()
                artists = (
                    [TrackArtistOut(name=artist_clean, inherited=False)]
                    if is_resolved_artist_name(artist_clean)
                    else []
                )
                credits: list[ExtractedCredit] = []
                for role_list, cat, role_label in (
                    (item.composers, "composer", "Composer"),
                    (item.lyricists, "lyricist", "Lyricist"),
                    (item.producers, "producer", "Producer"),
                ):
                    for cname in role_list:
                        c_clean = (cname or "").strip()
                        if c_clean:
                            credits.append(
                                ExtractedCredit(
                                    discogs_artist_id=None,
                                    artist_name=c_clean,
                                    anv="",
                                    role_raw=role_label,
                                    role_category=cat,
                                    credit_source=self.name,
                                    source_url=src_url,
                                )
                            )
                donors_by_order[item.track_order] = DonorTrack(
                    title=item.title,
                    duration_sec=dur_sec,
                    performing_artists=artists,
                    performing_artist_display=artist_clean if artists else "",
                    isrc=(item.isrc or "").strip().upper(),
                    credits=credits,
                    source_label=self.name,
                    source_url=src_url,
                )
        return donors_by_order, total_in, total_out, round(total_cost, 6)


class TrackEnricher:
    """Orchestrates Tier 0 -> Tier 1 -> Tier 2a -> Tier 2b -> optional Tier 3 enrichment."""

    def __init__(
        self,
        settings: Settings,
        discogs_master: DiscogsMasterVersionEnricher | None = None,
        musicbrainz: MusicBrainzEnricher | None = None,
        deezer: DeezerEnricher | None = None,
        itunes: ItunesTrackEnricher | None = None,
        gemini: GeminiTrackEnricher | None = None,
    ) -> None:
        self._settings = settings
        self._discogs_master = discogs_master
        self._musicbrainz = musicbrainz
        self._deezer = deezer
        self._itunes = itunes
        self._gemini = gemini

    def close(self) -> None:
        for component in (self._musicbrainz, self._deezer, self._itunes):
            if component is not None:
                component.close()

    def enrich_tracks(
        self,
        tracks: list[TrackOut],
        norm: NormalizedInput,
        release_meta: ReleaseOut | None = None,
        release_notes: str = "",
        source_provider: str = "discogs",
        sibling_releases: list[ExtractedRelease] | None = None,
        agent_sources: list[str] | None = None,
        use_agent: bool = False,
        bypass_cache: bool = False,
        progress_cb: Callable[[str, int | None], None] | None = None,
    ) -> EnrichmentStats:
        copied = [t.model_copy(deep=True) for t in tracks]
        dur_before, perf_before, wri_before, prod_before = compute_track_coverages(copied)
        stats = EnrichmentStats(
            tracks_total=len(copied),
            duration_coverage_before=dur_before,
            performer_coverage_before=perf_before,
            writer_credit_coverage_before=wri_before,
            producer_credit_coverage_before=prod_before,
            tracks=copied,
        )
        if not copied:
            return stats

        providers_used: list[str] = []

        def _record_provider(name: str) -> None:
            if name not in providers_used:
                providers_used.append(name)

        def _apply_aligned(
            aligned: dict[int, DonorTrack], provider_name: str
        ) -> bool:
            any_changed = False
            for t in copied:
                donor = aligned.get(t.track_order)
                if donor is not None and apply_donor_to_track(t, donor, stats):
                    any_changed = True
            if any_changed:
                _record_provider(provider_name)
            return any_changed

        target_is_live = is_live_release(
            title=release_meta.title if release_meta else norm.album_title_raw,
            input_title=norm.album_title_raw,
            formats=release_meta.formats if release_meta else None,
            notes=release_notes,
            format_tags=norm.format_hint.tags,
        )

        # Tier 0a: Single-artist performer inheritance
        fallback_artist = ""
        if not norm.is_compilation:
            if norm.artist_clean and is_resolved_artist_name(norm.artist_clean):
                fallback_artist = norm.artist_clean
            elif release_meta and is_resolved_artist_name(release_meta.artist_display):
                fallback_artist = release_meta.artist_display

        if fallback_artist:
            default_url = (
                release_meta.url
                if release_meta
                else (agent_sources[0] if agent_sources else "")
            )
            inherited_any = False
            for t in copied:
                if not t.has_resolved_performer:
                    donor = DonorTrack(
                        performing_artists=[TrackArtistOut(name=fallback_artist, inherited=True)],
                        performing_artist_display=fallback_artist,
                        source_label="album_artist_inherited",
                        source_url=default_url,
                    )
                    if apply_donor_to_track(t, donor, stats):
                        inherited_any = True
            if inherited_any:
                _record_provider("album_artist_inherited")

        # Tier 0b: Already-stored sibling releases in PostgreSQL
        for sib in sibling_releases or []:
            if not any(track_needs_album_enrichment(t) for t in copied):
                break
            sib_live = is_live_release(sib.title, formats=sib.formats, notes=sib.notes)
            if target_is_live != sib_live:
                continue
            trusted = (
                release_meta is not None
                and release_meta.master_id is not None
                and sib.master_id == release_meta.master_id
            )
            aligned = align_album_tracks(
                copied,
                _extracted_release_to_donors(sib, f"{sib.source_provider}_sibling"),
                trusted_release_link=trusted,
                allow_live_tag_mismatch=(target_is_live and sib_live),
            )
            _apply_aligned(aligned, f"{sib.source_provider}_sibling")

        # Tier 1: Discogs master versions (CD / File sibling editions)
        if (
            any(track_needs_album_enrichment(t) for t in copied)
            and release_meta is not None
            and release_meta.master_id is not None
            and self._discogs_master is not None
        ):
            if progress_cb is not None:
                progress_cb(
                    f"Checking Discogs master #{release_meta.master_id} sibling editions", None
                )
            for ver_rel in self._discogs_master.fetch_sibling_releases(
                release_meta.master_id,
                exclude_release_id=release_meta.discogs_release_id,
                bypass_cache=bypass_cache,
            ):
                if not any(track_needs_album_enrichment(t) for t in copied):
                    break
                ver_live = is_live_release(
                    ver_rel.title, formats=ver_rel.formats, notes=ver_rel.notes
                )
                aligned = align_album_tracks(
                    copied,
                    _extracted_release_to_donors(ver_rel, "discogs_master"),
                    trusted_release_link=True,
                    allow_live_tag_mismatch=(target_is_live and ver_live),
                )
                _apply_aligned(aligned, "discogs_master")

        # Tier 2a: Album-level MusicBrainz -> Deezer -> iTunes
        mb_album_matched = False
        if any(track_needs_album_enrichment(t) for t in copied) and self._musicbrainz is not None:
            if progress_cb is not None:
                progress_cb("Checking MusicBrainz release metadata", None)
            mb_donors, mb_trusted, mb_live = self._musicbrainz.fetch_album_donors(
                norm, target_is_live=target_is_live, bypass_cache=bypass_cache
            )
            if mb_donors:
                aligned = align_album_tracks(
                    copied,
                    mb_donors,
                    trusted_release_link=mb_trusted,
                    allow_live_tag_mismatch=(target_is_live and (mb_live or mb_trusted)),
                )
                mb_album_matched = bool(aligned)
                _apply_aligned(aligned, "musicbrainz")

        if any(track_needs_album_enrichment(t) for t in copied) and self._deezer is not None:
            if progress_cb is not None:
                progress_cb("Checking Deezer album metadata", None)
            dz_donors, dz_trusted, dz_live = self._deezer.fetch_album_donors(
                norm, target_is_live=target_is_live, bypass_cache=bypass_cache
            )
            if dz_donors:
                aligned = align_album_tracks(
                    copied,
                    dz_donors,
                    trusted_release_link=dz_trusted,
                    allow_live_tag_mismatch=(target_is_live and (dz_live or dz_trusted)),
                )
                # Fetch per-track ISRC for aligned tracks still missing ISRC
                for t in copied:
                    donor = aligned.get(t.track_order)
                    if donor is not None and not t.isrc and not donor.isrc and donor.external_track_id:
                        isrc, tr_link = self._deezer.fetch_track_isrc(
                            donor.external_track_id, bypass_cache=bypass_cache
                        )
                        if isrc:
                            donor.isrc = isrc
                            if tr_link:
                                donor.source_url = tr_link
                _apply_aligned(aligned, "deezer")

        if (
            any(track_needs_core_enrichment(t) for t in copied)
            and source_provider != "itunes"
            and self._itunes is not None
        ):
            if progress_cb is not None:
                progress_cb("Checking iTunes album metadata", None)
            it_donors, it_live = self._itunes.fetch_album_donors(
                norm, target_is_live=target_is_live, bypass_cache=bypass_cache
            )
            if it_donors:
                aligned = align_album_tracks(
                    copied,
                    it_donors,
                    trusted_release_link=False,
                    allow_live_tag_mismatch=(target_is_live and it_live),
                )
                _apply_aligned(aligned, "itunes")

        # Tier 2b: Per-track external search (ONLY for non-live releases with resolved performers)
        if not target_is_live:
            max_track_lookups = max(0, self._settings.enrichment_max_per_track_lookups)
            if self._deezer is not None and any(
                track_needs_core_enrichment(t) and t.has_resolved_performer for t in copied
            ):
                if progress_cb is not None:
                    progress_cb("Searching Deezer per track for missing lengths/ISRCs", None)
                dz_calls = 0
                for t in copied:
                    if dz_calls >= max_track_lookups:
                        break
                    if not (track_needs_core_enrichment(t) and t.has_resolved_performer):
                        continue
                    dz_calls += 1
                    donor, note = self._deezer.enrich_single_track(
                        t.title,
                        t.performing_artist_display,
                        need_isrc=not t.isrc,
                        bypass_cache=bypass_cache,
                    )
                    if note:
                        _append_note(t, note)
                    if donor is not None and apply_donor_to_track(t, donor, stats):
                        _record_provider("deezer")

            if self._itunes is not None and any(
                (not t.has_valid_duration) and t.has_resolved_performer for t in copied
            ):
                if progress_cb is not None:
                    progress_cb("Searching iTunes per track for missing durations", None)
                it_calls = 0
                for t in copied:
                    if it_calls >= max_track_lookups:
                        break
                    if t.has_valid_duration or not t.has_resolved_performer:
                        continue
                    it_calls += 1
                    donor, note = self._itunes.enrich_single_track(
                        t.title,
                        t.performing_artist_display,
                        bypass_cache=bypass_cache,
                    )
                    if note:
                        _append_note(t, note)
                    if donor is not None and apply_donor_to_track(t, donor, stats):
                        _record_provider("itunes")

            max_mb_track_lookups = max(0, self._settings.enrichment_max_mb_track_lookups)
            if (
                not mb_album_matched
                and self._musicbrainz is not None
                and max_mb_track_lookups > 0
                and any(
                    (
                        not t.has_valid_duration
                        or not t.isrc
                        or not (t.credits.get("composers") or t.credits.get("lyricists"))
                    )
                    and t.has_resolved_performer
                    for t in copied
                )
            ):
                if progress_cb is not None:
                    progress_cb("Searching MusicBrainz recordings per track", None)
                mb_calls = 0
                for t in copied:
                    if mb_calls >= max_mb_track_lookups:
                        break
                    need_mb_track = (
                        not t.has_valid_duration
                        or not t.isrc
                        or not (t.credits.get("composers") or t.credits.get("lyricists"))
                    )
                    if not (need_mb_track and t.has_resolved_performer):
                        continue
                    mb_calls += 1
                    need_details = not t.isrc or not (
                        (t.credits.get("composers") or t.credits.get("lyricists"))
                        and t.credits.get("producers")
                    )
                    donor, note = self._musicbrainz.enrich_single_track(
                        t.title,
                        t.performing_artist_display,
                        need_details=need_details,
                        bypass_cache=bypass_cache,
                    )
                    if note:
                        _append_note(t, note)
                    if donor is not None and apply_donor_to_track(t, donor, stats):
                        _record_provider("musicbrainz")

        # Tier 3: Optional batched Gemini + Google Search enrichment
        if (
            use_agent
            and self._gemini is not None
            and any(track_needs_album_enrichment(t) for t in copied)
        ):
            if progress_cb is not None:
                progress_cb("Running targeted Gemini search for remaining missing track data", None)
            ref_url = (
                release_meta.url
                if release_meta
                else (agent_sources[0] if agent_sources else "")
            )
            donors_by_order, in_tok, out_tok, cost_usd = self._gemini.enrich_tracks_batch(
                norm=norm,
                release_url=ref_url,
                tracks=copied,
            )
            stats.agent_input_tokens += in_tok
            stats.agent_output_tokens += out_tok
            stats.agent_cost_usd = round(stats.agent_cost_usd + cost_usd, 6)
            _apply_aligned(donors_by_order, "gemini_agent")

        dur_after, perf_after, wri_after, prod_after = compute_track_coverages(copied)
        stats.duration_coverage_after = dur_after
        stats.performer_coverage_after = perf_after
        stats.writer_credit_coverage_after = wri_after
        stats.producer_credit_coverage_after = prod_after
        stats.providers_used = providers_used
        return stats


def build_default_track_enricher(
    settings: Settings,
    provider: CatalogProvider | None,
    cache_store: ApiCacheStore | None,
    agent_provider: AgentFallbackProvider | None = None,
) -> TrackEnricher:
    discogs_master = (
        DiscogsMasterVersionEnricher(
            provider, max_versions=settings.enrichment_max_master_versions
        )
        if provider is not None
        else None
    )
    musicbrainz = MusicBrainzEnricher(settings, cache_store=cache_store)
    deezer = DeezerEnricher(settings, cache_store=cache_store)
    itunes = ItunesTrackEnricher(settings, cache_store=cache_store)
    gemini = GeminiTrackEnricher(agent_provider) if agent_provider is not None else None
    return TrackEnricher(
        settings=settings,
        discogs_master=discogs_master,
        musicbrainz=musicbrainz,
        deezer=deezer,
        itunes=itunes,
        gemini=gemini,
    )
