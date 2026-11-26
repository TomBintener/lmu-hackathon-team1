import hashlib
import html
import json
import logging
import re
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.discogs_client import ApiCacheStore
from app.evaluator import string_similarity
from app.extractor import ExtractedRelease, extract_release
from app.models import NormalizedInput
from app.normalizer import normalize_text

logger = logging.getLogger(__name__)

JSON_LD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
DISC_H4_RE = re.compile(r"<h4[^>]*>(.*?)</h4>", re.IGNORECASE | re.DOTALL)
DISK_NUM_RE = re.compile(r"\b(?:Disk|Disc|CD|LP)\s*(\d+)\b", re.IGNORECASE)
RECORDING_LI_RE = re.compile(
    r'<li[^>]+MusicRecording[^>]*>(.*?)</li>', re.IGNORECASE | re.DOTALL
)
POS_B_RE = re.compile(r"<b>\s*(\d+)\s*</b>", re.IGNORECASE)
ITEMPROP_NAME_RE = re.compile(
    r'<span[^>]+itemprop=["\']name["\'][^>]*>(.*?)</span>',
    re.IGNORECASE | re.DOTALL,
)

COMPILATION_ARTIST_LOWER = {
    "various",
    "various artists",
    "va",
    "v.a.",
    "verschiedene interpret:innen",
    "verschiedene interpreten",
}


def make_fallback_release_id(provider: str, external_id: str) -> int:
    """Build a deterministic negative 56-bit BIGINT ID for non-Discogs releases."""
    digest = hashlib.sha256(f"{provider}:{external_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:7], "big") or 1
    return -value


def _build_cache_key(provider: str, params: dict[str, Any]) -> str:
    raw = json.dumps({"provider": provider, "params": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _clean_html_text(fragment: str) -> str:
    no_tags = TAG_RE.sub(" ", fragment or "")
    return re.sub(r"\s+", " ", html.unescape(no_tags)).strip()


def _format_millis(ms_raw: Any) -> str:
    if ms_raw is None:
        return ""
    try:
        total_sec = max(0, int(round(float(ms_raw) / 1000.0)))
    except (TypeError, ValueError):
        return ""
    if total_sec <= 0:
        return ""
    if total_sec >= 3600:
        hours, rem = divmod(total_sec, 3600)
        mins, secs = divmod(rem, 60)
        return f"{hours}:{mins:02d}:{secs:02d}"
    mins, secs = divmod(total_sec, 60)
    return f"{mins}:{secs:02d}"


class FallbackProvider(Protocol):
    name: str

    def lookup_candidates(
        self, norm: NormalizedInput, bypass_cache: bool = False
    ) -> tuple[list[tuple[ExtractedRelease, str]], list[dict[str, Any]], bool]: ...

    def close(self) -> None: ...


class JpcFallbackProvider:
    """EAN-based fallback lookup against jpc.de product microdata and tracklists."""

    name = "jpc"

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
            timeout=settings.fallback_timeout_sec,
            follow_redirects=True,
            headers={"User-Agent": settings.discogs_user_agent},
        )

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    @staticmethod
    def _extract_product_jsonld(page_html: str) -> dict[str, Any] | None:
        for block in JSON_LD_RE.findall(page_html):
            try:
                parsed = json.loads(block)
            except ValueError:
                continue
            candidates = parsed if isinstance(parsed, list) else [parsed]
            for obj in candidates:
                if isinstance(obj, dict) and str(obj.get("@type") or "").lower() == "product":
                    return obj
        return None

    @staticmethod
    def _infer_release_artist(
        page_html: str,
        final_url: str,
        title: str,
        norm: NormalizedInput,
    ) -> str:
        h1_match = H1_RE.search(page_html)
        if h1_match:
            h1_text = _clean_html_text(h1_match.group(1))
            if title and ":" in h1_text:
                prefix, rest = h1_text.split(":", 1)
                if title.lower() in rest.lower() and prefix.strip():
                    candidate_artist = prefix.strip()
                    if candidate_artist.lower() in COMPILATION_ARTIST_LOWER:
                        return "Various"
                    return candidate_artist
        if "/art/various-artists-" in final_url.lower() or norm.is_compilation:
            return "Various"
        return norm.artist_clean or "Various"

    @staticmethod
    def _extract_tracklist_and_disc_count(
        page_html: str, is_compilation: bool
    ) -> tuple[list[dict[str, Any]], int]:
        tl_marker = page_html.find('class="box tracklisting"')
        if tl_marker == -1:
            return [], 1
        next_box = page_html.find('<div class="box ', tl_marker + 25)
        tl_section = page_html[tl_marker:next_box] if next_box != -1 else page_html[tl_marker:]

        raw_playlists = tl_section.split('<div class="playlist"')[1:]
        if not raw_playlists:
            return [], 1

        multi_disc = len(raw_playlists) > 1
        tracklist: list[dict[str, Any]] = []
        disc_numbers: list[int] = []

        for idx, pl_chunk in enumerate(raw_playlists, start=1):
            ol_end = pl_chunk.find("</ol>")
            pl_body = pl_chunk[:ol_end] if ol_end != -1 else pl_chunk
            disc_num = idx
            h4_m = DISC_H4_RE.search(pl_body)
            if h4_m:
                disk_m = DISK_NUM_RE.search(_clean_html_text(h4_m.group(1)))
                if disk_m:
                    disc_num = int(disk_m.group(1))
            disc_numbers.append(disc_num)

            if multi_disc:
                tracklist.append({"type_": "heading", "position": "", "title": f"CD {disc_num}"})

            li_blocks = RECORDING_LI_RE.findall(pl_body)
            if not li_blocks:
                li_blocks = [f'<span itemprop="name">{m}</span>' for m in ITEMPROP_NAME_RE.findall(pl_body)]

            for track_idx, li_html in enumerate(li_blocks, start=1):
                name_m = ITEMPROP_NAME_RE.search(li_html)
                if not name_m:
                    continue
                raw_name = _clean_html_text(name_m.group(1))
                if not raw_name:
                    continue

                pos_m = POS_B_RE.search(li_html)
                pos_num = pos_m.group(1) if pos_m else str(track_idx)
                position = f"{disc_num}-{pos_num}" if multi_disc else pos_num

                track_entry: dict[str, Any] = {
                    "type_": "track",
                    "position": position,
                    "title": raw_name,
                    "duration": "",
                }
                if is_compilation and ": " in raw_name:
                    t_artist, t_title = raw_name.split(": ", 1)
                    if t_artist.strip() and t_title.strip():
                        track_entry["artists"] = [{"name": t_artist.strip()}]
                        track_entry["title"] = t_title.strip()
                tracklist.append(track_entry)

        has_tracks = any(item.get("type_") == "track" for item in tracklist)
        return (tracklist if has_tracks else []), max(disc_numbers, default=1)

    @classmethod
    def _parse_product_page(
        cls,
        ean: str,
        final_url: str,
        page_html: str,
        norm: NormalizedInput,
    ) -> dict[str, Any] | None:
        product = cls._extract_product_jsonld(page_html)
        if product is None:
            return None

        title = str(product.get("name") or "").strip()
        gtin = str(product.get("gtin") or product.get("gtin13") or "").strip()
        sku = str(product.get("sku") or gtin or ean).strip()
        category = str(product.get("category") or "").strip()
        if not title:
            return None

        release_artist = cls._infer_release_artist(page_html, final_url, title, norm)
        is_comp = normalize_text(release_artist) in COMPILATION_ARTIST_LOWER or (
            norm.is_compilation and release_artist == "Various"
        )
        tracklist, disc_count = cls._extract_tracklist_and_disc_count(page_html, is_comp)
        if not tracklist:
            return None

        cat_low = category.lower()
        medium = "Vinyl" if ("lp" in cat_low or "vinyl" in cat_low) else "CD"

        return {
            "id": make_fallback_release_id(cls.name, sku),
            "_source_provider": cls.name,
            "uri": final_url,
            "resource_url": "",
            "title": title,
            "artists": [{"name": release_artist}],
            "year": None,
            "country": "Germany",
            "formats": [{"name": medium, "qty": str(disc_count), "descriptions": []}],
            "identifiers": [{"type": "Barcode", "value": gtin or ean}],
            "labels": [],
            "data_quality": "External (jpc.de)",
            "community": {"status": "", "have": 0, "want": 0},
            "notes": "",
            "extraartists": [],
            "tracklist": tracklist,
        }

    def lookup_candidates(
        self, norm: NormalizedInput, bypass_cache: bool = False
    ) -> tuple[list[tuple[ExtractedRelease, str]], list[dict[str, Any]], bool]:
        if not norm.barcode_variants:
            return [], [], True

        queries: list[dict[str, Any]] = []
        all_from_cache = True

        for ean in norm.barcode_variants:
            cache_key = _build_cache_key(self.name, {"ean": ean})
            cached = (
                self._cache.get_api_cache(cache_key)
                if (self._cache is not None and not bypass_cache)
                else None
            )
            if cached is not None:
                payload = cached.get("payload_json") or {}
                raw_release = payload.get("raw_release")
                found = bool(payload.get("found") and isinstance(raw_release, dict))
                queries.append(
                    {
                        "tier": "fallback_jpc_ean",
                        "params": {"ean": ean},
                        "result_count": 1 if found else 0,
                        "from_cache": True,
                    }
                )
                if found:
                    return [(extract_release(raw_release), "jpc_ean")], queries, True
                continue

            all_from_cache = False
            url = f"https://www.jpc.de/s/{ean}"
            try:
                resp = self._client.get(url)
            except Exception as exc:
                logger.warning("JPC fallback request failed for EAN %s: %s", ean, exc)
                queries.append(
                    {
                        "tier": "fallback_jpc_ean",
                        "params": {"ean": ean},
                        "result_count": 0,
                        "from_cache": False,
                        "error": str(exc),
                    }
                )
                continue

            if resp.status_code == 404:
                if self._cache is not None:
                    self._cache.set_api_cache(
                        cache_key, url, 404, {"found": False, "final_url": str(resp.url), "raw_release": None}
                    )
                queries.append(
                    {
                        "tier": "fallback_jpc_ean",
                        "params": {"ean": ean},
                        "result_count": 0,
                        "from_cache": False,
                    }
                )
                continue

            if resp.status_code >= 400:
                queries.append(
                    {
                        "tier": "fallback_jpc_ean",
                        "params": {"ean": ean},
                        "result_count": 0,
                        "from_cache": False,
                        "error": f"HTTP {resp.status_code}",
                    }
                )
                continue

            raw_release = self._parse_product_page(ean, str(resp.url), resp.text, norm)
            if self._cache is not None:
                self._cache.set_api_cache(
                    cache_key,
                    url,
                    resp.status_code,
                    {
                        "found": raw_release is not None,
                        "final_url": str(resp.url),
                        "raw_release": raw_release,
                    },
                )
            queries.append(
                {
                    "tier": "fallback_jpc_ean",
                    "params": {"ean": ean},
                    "result_count": 1 if raw_release is not None else 0,
                    "from_cache": False,
                }
            )
            if raw_release is not None:
                return [(extract_release(raw_release), "jpc_ean")], queries, all_from_cache

        return [], queries, all_from_cache


class ItunesFallbackProvider:
    """Fallback lookup using the free iTunes Search and Lookup JSON APIs."""

    name = "itunes"

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
            base_url="https://itunes.apple.com",
            timeout=settings.fallback_timeout_sec,
            headers={"User-Agent": settings.discogs_user_agent},
        )

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def _get_json_cached(
        self,
        endpoint: str,
        params: dict[str, Any],
        bypass_cache: bool,
    ) -> tuple[dict[str, Any], bool, str | None]:
        cache_key = _build_cache_key(f"{self.name}:{endpoint}", params)
        if self._cache is not None and not bypass_cache:
            cached = self._cache.get_api_cache(cache_key)
            if cached is not None:
                return cached.get("payload_json") or {}, True, None

        try:
            resp = self._client.get(endpoint, params=params)
        except Exception as exc:
            return {}, False, str(exc)

        if resp.status_code >= 400:
            return {}, False, f"HTTP {resp.status_code}"

        try:
            payload = resp.json()
        except ValueError as exc:
            return {}, False, f"Invalid JSON: {exc}"

        if not isinstance(payload, dict):
            return {}, False, "Unexpected iTunes JSON shape"

        if self._cache is not None:
            self._cache.set_api_cache(cache_key, endpoint, resp.status_code, payload)
        return payload, False, None

    @classmethod
    def _build_raw_release(
        cls, collection_id: int, results: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        collection_meta: dict[str, Any] = {}
        songs: list[dict[str, Any]] = []
        for item in results:
            wtype = str(item.get("wrapperType") or "").lower()
            if wtype == "collection" and not collection_meta:
                collection_meta = item
            elif wtype == "track" and item.get("trackName"):
                songs.append(item)

        if not songs:
            return None

        songs.sort(
            key=lambda s: (
                int(s.get("discNumber") or 1),
                int(s.get("trackNumber") or 0),
            )
        )
        max_disc = max(int(s.get("discNumber") or 1) for s in songs)

        tracklist: list[dict[str, Any]] = []
        current_disc: int | None = None
        for idx, song in enumerate(songs, start=1):
            disc_num = int(song.get("discNumber") or 1)
            track_num = int(song.get("trackNumber") or idx)
            if max_disc > 1 and disc_num != current_disc:
                current_disc = disc_num
                tracklist.append({"type_": "heading", "position": "", "title": f"CD {disc_num}"})

            pos = f"{disc_num}-{track_num}" if max_disc > 1 else str(track_num)
            track_entry: dict[str, Any] = {
                "type_": "track",
                "position": pos,
                "title": str(song.get("trackName") or "").strip(),
                "duration": _format_millis(song.get("trackTimeMillis")),
            }
            track_artist = str(song.get("artistName") or "").strip()
            if track_artist:
                track_entry["artists"] = [{"name": track_artist}]
            tracklist.append(track_entry)

        title = str(
            collection_meta.get("collectionName")
            or songs[0].get("collectionName")
            or ""
        ).strip()
        if not title:
            return None

        artist_name = str(
            collection_meta.get("artistName")
            or songs[0].get("artistName")
            or "Various"
        ).strip()
        if artist_name.lower() in COMPILATION_ARTIST_LOWER:
            artist_name = "Various"

        rel_date = str(
            collection_meta.get("releaseDate")
            or songs[0].get("releaseDate")
            or ""
        )
        year = int(rel_date[:4]) if len(rel_date) >= 4 and rel_date[:4].isdigit() else None
        copyright_txt = str(collection_meta.get("copyright") or "").strip()
        url = str(
            collection_meta.get("collectionViewUrl")
            or songs[0].get("collectionViewUrl")
            or f"https://music.apple.com/de/album/{collection_id}"
        )

        return {
            "id": make_fallback_release_id(cls.name, str(collection_id)),
            "_source_provider": cls.name,
            "uri": url,
            "resource_url": "",
            "title": title,
            "artists": [{"name": artist_name}],
            "year": year,
            "country": str(collection_meta.get("country") or "DE"),
            "formats": [{"name": "File", "qty": str(max_disc), "descriptions": ["Digital"]}],
            "identifiers": [],
            "labels": [{"name": copyright_txt, "catno": ""}] if copyright_txt else [],
            "data_quality": "External (iTunes)",
            "community": {"status": "", "have": 0, "want": 0},
            "notes": "",
            "extraartists": [],
            "tracklist": tracklist,
        }

    def lookup_candidates(
        self, norm: NormalizedInput, bypass_cache: bool = False
    ) -> tuple[list[tuple[ExtractedRelease, str]], list[dict[str, Any]], bool]:
        if not norm.title_clean:
            return [], [], True

        term = (
            norm.title_clean
            if (norm.is_compilation or not norm.artist_clean)
            else f"{norm.artist_clean} {norm.title_clean}"
        )
        search_params = {
            "term": term,
            "entity": "album",
            "country": "DE",
            "limit": 5,
        }
        queries: list[dict[str, Any]] = []
        search_payload, search_cached, search_err = self._get_json_cached(
            "/search", search_params, bypass_cache=bypass_cache
        )
        all_from_cache = search_cached

        raw_collections = [
            r
            for r in (search_payload.get("results") or [])
            if str(r.get("wrapperType") or "").lower() == "collection"
            and r.get("collectionId") is not None
        ]
        search_query_entry: dict[str, Any] = {
            "tier": "fallback_itunes_search",
            "params": search_params,
            "result_count": len(raw_collections),
            "from_cache": search_cached,
        }
        if search_err:
            search_query_entry["error"] = search_err
        queries.append(search_query_entry)

        if search_err or not raw_collections:
            return [], queries, all_from_cache

        scored_ids: list[tuple[float, int]] = []
        for col in raw_collections:
            col_id = int(col["collectionId"])
            col_title = str(col.get("collectionName") or "")
            col_artist = str(col.get("artistName") or "")
            t_sim = string_similarity(norm.title_clean, col_title)
            if t_sim < 0.45:
                continue
            if norm.is_compilation:
                a_sim = 1.0 if col_artist.lower() in COMPILATION_ARTIST_LOWER else 0.3
            elif norm.artist_clean:
                a_sim = string_similarity(norm.artist_clean, col_artist)
            else:
                a_sim = 0.5
            scored_ids.append((t_sim + a_sim, col_id))

        scored_ids.sort(key=lambda pair: pair[0], reverse=True)
        limit = max(1, self._settings.fallback_max_candidates)
        candidates: list[tuple[ExtractedRelease, str]] = []

        for _, col_id in scored_ids[:limit]:
            lookup_params = {"id": str(col_id), "entity": "song", "country": "DE"}
            lookup_payload, lookup_cached, lookup_err = self._get_json_cached(
                "/lookup", lookup_params, bypass_cache=bypass_cache
            )
            if not lookup_cached:
                all_from_cache = False
            results = list(lookup_payload.get("results") or [])
            lookup_entry: dict[str, Any] = {
                "tier": "fallback_itunes_lookup",
                "params": lookup_params,
                "result_count": len(results),
                "from_cache": lookup_cached,
            }
            if lookup_err:
                lookup_entry["error"] = lookup_err
            queries.append(lookup_entry)
            if lookup_err or not results:
                continue

            raw_release = self._build_raw_release(col_id, results)
            if raw_release is not None:
                candidates.append((extract_release(raw_release), "itunes_search"))

        return candidates, queries, all_from_cache


def build_default_fallback_providers(
    settings: Settings, cache_store: ApiCacheStore | None = None
) -> list[FallbackProvider]:
    """Instantiate the default external catalog fallback chain when enabled."""
    if not settings.enable_external_fallbacks:
        return []
    return [
        JpcFallbackProvider(settings, cache_store=cache_store),
        ItunesFallbackProvider(settings, cache_store=cache_store),
    ]
