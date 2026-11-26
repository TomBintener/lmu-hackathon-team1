import json
from typing import Any

import httpx

from app.config import Settings
from app.fallback_providers import (
    ItunesFallbackProvider,
    JpcFallbackProvider,
    make_fallback_release_id,
)
from app.normalizer import normalize_input


class MemoryApiCache:
    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}

    def get_api_cache(self, cache_key: str) -> dict[str, Any] | None:
        return self.store.get(cache_key)

    def set_api_cache(
        self, cache_key: str, endpoint: str, status_code: int, payload: dict[str, Any]
    ) -> None:
        self.store[cache_key] = {
            "cache_key": cache_key,
            "endpoint": endpoint,
            "status_code": status_code,
            "payload_json": payload,
        }


def _jpc_compilation_html() -> str:
    product_ld = json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": "Schlager Arena 2025: Sommer Edition - Die größten Hits des Jahres",
            "gtin": "4099964172904",
            "sku": 12285766,
            "category": "CD",
        }
    )
    return f"""<!doctype html>
<html>
<head>
  <script type="application/ld+json">{product_ld}</script>
</head>
<body>
  <h1>Schlager Arena 2025: Sommer Edition - Die größten Hits des Jahres auf 2 CDs</h1>
  <div class="box tracklisting">
    <div class="playlist">
      <h4>Disk 1 von 2 (CD)</h4>
      <ol role="list">
        <li itemscope itemtype="https://schema.org/MusicRecording"><div class="tracks"><b>1</b><span><span itemprop="name">Artist One: Track Alpha</span></span></div></li>
      </ol>
    </div>
    <div class="playlist">
      <h4>Disk 2 von 2 (CD)</h4>
      <ol role="list">
        <li itemscope itemtype="https://schema.org/MusicRecording"><div class="tracks"><b>1</b><span><span itemprop="name">Artist Two: Track Beta</span></span></div></li>
      </ol>
    </div>
  </div>
  <div class="box safety"><h4>Bilder zur Produktsicherheit</h4><span itemprop="name">Ignore Me</span></div>
</body>
</html>"""


def _jpc_single_artist_html() -> str:
    product_ld = json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": "Aspiral",
            "gtin": "4065629733131",
            "sku": 12169985,
            "category": "CD",
        }
    )
    return f"""<!doctype html>
<html>
<head>
  <script type="application/ld+json">{product_ld}</script>
</head>
<body>
  <h1>Epica: Aspiral auf CD</h1>
  <div class="box tracklisting">
    <div class="playlist">
      <h4>Disk 1 von 1 (CD)</h4>
      <ol role="list">
        <li itemscope itemtype="https://schema.org/MusicRecording"><div class="tracks"><b>1</b><span><span itemprop="name">Cross The Divide</span></span></div></li>
        <li itemscope itemtype="https://schema.org/MusicRecording"><div class="tracks"><b>2</b><span><span itemprop="name">Arcana</span></span></div></li>
      </ol>
    </div>
  </div>
</body>
</html>"""


def test_make_fallback_release_id_is_deterministic_negative_bigint() -> None:
    id1 = make_fallback_release_id("jpc", "12285766")
    id2 = make_fallback_release_id("jpc", "12285766")
    id3 = make_fallback_release_id("itunes", "12285766")
    assert id1 == id2
    assert id1 < 0
    assert id3 < 0
    assert id1 != id3
    assert id1 >= -(2**63)


def test_jpc_fallback_parses_multidisc_compilation_single_artist_and_caches() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("/4099964172904"):
            return httpx.Response(200, text=_jpc_compilation_html())
        if request.url.path.endswith("/4065629733131"):
            return httpx.Response(200, text=_jpc_single_artist_html())
        return httpx.Response(404, text="Not found")

    cache = MemoryApiCache()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = JpcFallbackProvider(Settings(), cache_store=cache, http_client=client)

    norm_comp = normalize_input(
        "4099964172904",
        "SCHLAGER ARENA 2025 - SOMMER EDITON - Die größten Hits des Jahres",
        "Various Artists",
    )
    candidates, queries, from_cache = provider.lookup_candidates(norm_comp)
    assert from_cache is False
    assert len(candidates) == 1
    rel, method = candidates[0]
    assert method == "jpc_ean"
    assert rel.source_provider == "jpc"
    assert rel.discogs_release_id < 0
    assert rel.artist_norm == "various"
    assert rel.track_count == 2
    assert rel.tracks[0].disc_number == 1
    assert rel.tracks[0].performing_artist_display == "Artist One"
    assert rel.tracks[0].title == "Track Alpha"
    assert rel.tracks[1].disc_number == 2
    assert rel.tracks[1].performing_artist_display == "Artist Two"
    assert rel.tracks[1].title == "Track Beta"
    assert ("4099964172904", "4099964172904") in rel.barcode_entries
    assert len(calls) == 1
    assert queries[0]["from_cache"] is False

    # Second call uses api_cache without hitting HTTP transport
    candidates_cached, queries_cached, from_cache_2 = provider.lookup_candidates(norm_comp)
    assert from_cache_2 is True
    assert len(candidates_cached) == 1
    assert queries_cached[0]["from_cache"] is True
    assert len(calls) == 1

    # Single-artist album infers artist from H1
    norm_single = normalize_input("4065629733131", "Aspiral", "EPICA")
    single_cands, _, _ = provider.lookup_candidates(norm_single)
    assert len(single_cands) == 1
    assert single_cands[0][0].artist_display == "Epica"
    assert [t.title for t in single_cands[0][0].tracks] == ["Cross The Divide", "Arcana"]

    # 404 returns empty cleanly
    norm_404 = normalize_input("4000000000000", "Missing CD", "Nobody")
    miss_cands, miss_queries, _ = provider.lookup_candidates(norm_404)
    assert miss_cands == []
    assert miss_queries[0]["result_count"] == 0


def test_jpc_fallback_handles_network_errors_gracefully() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = JpcFallbackProvider(Settings(), http_client=client)
    norm = normalize_input("4099964172904", "Test", "Artist")
    cands, queries, from_cache = provider.lookup_candidates(norm)
    assert cands == []
    assert from_cache is False
    assert "error" in queries[0]


def test_itunes_fallback_search_lookup_prefilter_and_caching() -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, dict(request.url.params)))
        if request.url.path == "/search":
            term = request.url.params.get("term", "")
            if "Paula Linke" in term:
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "wrapperType": "collection",
                                "collectionId": 6798000134,
                                "artistName": "Paula Linke",
                                "collectionName": "Küsse! Mehr Küsse!",
                            }
                        ]
                    },
                )
            if "Matthias Petzold" in term:
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "wrapperType": "collection",
                                "collectionId": 999111,
                                "artistName": "Thomanerchor & Martin Petzold",
                                "collectionName": "Bach Cantatas Vol. 9",
                            }
                        ]
                    },
                )
        if request.url.path == "/lookup" and request.url.params.get("id") == "6798000134":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "wrapperType": "collection",
                            "collectionId": 6798000134,
                            "artistName": "Paula Linke",
                            "collectionName": "Küsse! Mehr Küsse!",
                            "releaseDate": "2026-02-14T08:00:00Z",
                            "country": "DEU",
                            "copyright": "℗ 2026 Paula Linke",
                            "collectionViewUrl": "https://music.apple.com/de/album/kuesse/6798000134",
                        },
                        {
                            "wrapperType": "track",
                            "discNumber": 1,
                            "trackNumber": 1,
                            "trackName": "Erster Kuss",
                            "artistName": "Paula Linke",
                            "trackTimeMillis": 218000,
                        },
                        {
                            "wrapperType": "track",
                            "discNumber": 1,
                            "trackNumber": 2,
                            "trackName": "Zweiter Kuss",
                            "artistName": "Paula Linke",
                            "trackTimeMillis": 175256,
                        },
                    ]
                },
            )
        return httpx.Response(503, json={})

    cache = MemoryApiCache()
    client = httpx.Client(base_url="https://itunes.apple.com", transport=httpx.MockTransport(handler))
    provider = ItunesFallbackProvider(Settings(), cache_store=cache, http_client=client)

    norm_paula = normalize_input("", "Küsse! Mehr Küsse!", "Paula Linke")
    cands, queries, from_cache = provider.lookup_candidates(norm_paula)
    assert from_cache is False
    assert len(cands) == 1
    rel, method = cands[0]
    assert method == "itunes_search"
    assert rel.source_provider == "itunes"
    assert rel.discogs_release_id == make_fallback_release_id("itunes", "6798000134")
    assert rel.year == 2026
    assert rel.track_count == 2
    assert rel.tracks[0].duration_raw == "3:38"
    assert rel.tracks[0].duration_sec == 218
    assert rel.tracks[0].performing_artist_display == "Paula Linke"
    assert len(queries) == 2
    assert len(calls) == 2

    # Second call uses api_cache
    cands_2, _, from_cache_2 = provider.lookup_candidates(norm_paula)
    assert from_cache_2 is True
    assert len(cands_2) == 1
    assert len(calls) == 2

    # Unrelated search hit is filtered out before making /lookup requests
    norm_petzold = normalize_input("", "Slow Migration", "Matthias Petzold")
    cands_p, queries_p, _ = provider.lookup_candidates(norm_petzold)
    assert cands_p == []
    assert len(queries_p) == 1
    assert [p for p, _ in calls] == ["/search", "/lookup", "/search"]
