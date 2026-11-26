import json
import threading
import time
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg_pool import ConnectionPool

from app import cli
from app.agent_client import AgentAnswer, AgentTrack, GeminiAgentClient
from app.config import Settings, get_settings
from app.enricher import (
    DeezerEnricher,
    DiscogsMasterVersionEnricher,
    DonorTrack,
    GeminiTrackEnricher,
    ItunesTrackEnricher,
    MusicBrainzEnricher,
    TrackEnricher,
    align_album_tracks,
    extract_version_qualifiers,
    is_live_release,
    qualifiers_compatible,
    select_best_single_track,
)
from app.main import create_app
from app.models import ReleaseOut, TrackArtistOut, TrackCreditOut, TrackOut
from app.normalizer import normalize_input
from app.repository import PipelineRepository
from tests.conftest import make_sample_release
from tests.test_end_to_end import FakeCatalogProvider

TEST_ENRICH_DB_NAME = "tracklist_test_enrich"


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


class _FakeModelsAPI:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _FakeGenaiClient:
    def __init__(self, responses: list[Any]) -> None:
        self.models = _FakeModelsAPI(responses)


def _replace_db_name(db_url: str, db_name: str) -> str:
    parts = urlsplit(db_url)
    return urlunsplit((parts.scheme, parts.netloc, f"/{db_name}", parts.query, parts.fragment))


def _terminate_and_drop(admin_url: str, db_name: str) -> None:
    with psycopg.connect(admin_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
                """,
                (db_name,),
            )
            cur.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name))
            )


def _truncate_db(db_url: str) -> None:
    with psycopg.connect(db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE TABLE queue_jobs, input_records, releases, api_cache RESTART IDENTITY CASCADE"
            )


def _wait_until(predicate, timeout: float = 8.0, interval: float = 0.05) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        val = predicate()
        if val:
            return val
        time.sleep(interval)
    raise AssertionError("Timed out waiting for async enrichment condition")


@pytest.fixture(scope="module")
def enrich_test_db_url():
    base_url = get_settings().database_url
    admin_url = _replace_db_name(base_url, "postgres")
    test_url = _replace_db_name(base_url, TEST_ENRICH_DB_NAME)

    _terminate_and_drop(admin_url, TEST_ENRICH_DB_NAME)
    with psycopg.connect(admin_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TEST_ENRICH_DB_NAME))
            )

    pool = ConnectionPool(conninfo=test_url, min_size=1, max_size=2, open=True)
    try:
        PipelineRepository(pool).ensure_schema()
    finally:
        pool.close()

    try:
        yield test_url
    finally:
        _terminate_and_drop(admin_url, TEST_ENRICH_DB_NAME)


# ------------------------------------------------------------------ #
# Unit Tests: Guardrails & Alignment
# ------------------------------------------------------------------ #


def test_version_qualifiers_live_detection_and_conflict_guard() -> None:
    assert extract_version_qualifiers('Rhythm Is A Dancer (7" Edit)') == {"edit"}
    assert extract_version_qualifiers("Double Vision (Live in Chicago)") == {"live"}
    assert extract_version_qualifiers("Che (Edgar Peng Remix)") == {"remix"}

    assert not qualifiers_compatible("Don't Walk, Run (Edit)", "Don't Walk, Run")
    assert qualifiers_compatible("Don't Walk, Run (Edit)", "Don't Walk, Run (Radio Edit)")
    assert not qualifiers_compatible("Double Vision", "Double Vision (Live)")
    assert qualifiers_compatible(
        "Double Vision", "Double Vision (Live)", allow_live_tag_mismatch=True
    )

    assert is_live_release("LIVE IN CHICAGO (2LP)")
    assert is_live_release(
        "Unknown Title",
        formats=[{"name": "Vinyl", "descriptions": ["LP", "Live"]}],
    )
    assert not is_live_release("Aspiral", formats=[{"name": "CD", "descriptions": ["Album"]}])

    # Conflict > 10s skips duration while preserving unambiguous donor identity
    winner, note = select_best_single_track(
        target_title="Freed From Desire",
        target_artist="Gala",
        candidates=[
            DonorTrack(
                title="Freed From Desire",
                duration_sec=213,
                performing_artist_display="Gala",
                isrc="IT00001",
                source_label="deezer",
                source_url="https://deezer.test/1",
            ),
            DonorTrack(
                title="Freed From Desire",
                duration_sec=255,
                performing_artist_display="Gala",
                isrc="IT00002",
                source_label="deezer",
                source_url="https://deezer.test/2",
            ),
        ],
        provider_name="deezer",
    )
    assert winner is not None
    assert winner.duration_sec is None
    assert note is not None and "Duration conflict" in note

    # Agreement <= 10s keeps duration
    ok_winner, ok_note = select_best_single_track(
        target_title="Freed From Desire",
        target_artist="Gala",
        candidates=[
            DonorTrack(
                title="Freed From Desire",
                duration_sec=213,
                performing_artist_display="Gala",
                source_label="deezer",
            ),
            DonorTrack(
                title="Freed From Desire",
                duration_sec=216,
                performing_artist_display="Gala",
                source_label="deezer",
            ),
        ],
        provider_name="deezer",
    )
    assert ok_winner is not None and ok_winner.duration_sec == 213
    assert ok_note is None

    # Album alignment requires >= 60% overlap when untrusted
    targets = [
        TrackOut(track_order=1, title="Track A"),
        TrackOut(track_order=2, title="Track B"),
        TrackOut(track_order=3, title="Track C"),
    ]
    partial = align_album_tracks(
        targets,
        [DonorTrack(title="Track A", duration_sec=200)],
        trusted_release_link=False,
    )
    assert partial == {}
    full = align_album_tracks(
        targets,
        [
            DonorTrack(title="Track A", duration_sec=200),
            DonorTrack(title="Track B", duration_sec=210),
        ],
        trusted_release_link=False,
    )
    assert set(full.keys()) == {1, 2}


# ------------------------------------------------------------------ #
# Unit Tests: Providers
# ------------------------------------------------------------------ #


def test_discogs_master_version_enricher_prioritizes_cd_and_file() -> None:
    class _MasterFakeProvider(FakeCatalogProvider):
        def __init__(self) -> None:
            super().__init__()
            self.versions_by_master: dict[int, list[dict[str, Any]]] = {}

        def get_master_versions(
            self, master_id: int, per_page: int = 25, bypass_cache: bool = False
        ) -> tuple[dict[str, Any], bool]:
            return {"versions": self.versions_by_master.get(master_id, [])}, False

    provider = _MasterFakeProvider()
    provider.versions_by_master[777] = [
        {"id": 1001, "format": "LP, Vinyl", "major_formats": ["Vinyl"], "country": "US"},
        {"id": 1002, "format": "CD, Album", "major_formats": ["CD"], "country": "Germany"},
        {"id": 1003, "format": "MP3, File", "major_formats": ["File"], "country": "Europe"},
    ]
    provider.releases[1002] = make_sample_release(
        release_id=1002,
        title="Italia",
        artist="Till Brönner",
        tracklist=[{"type_": "track", "position": "1", "title": "Estate", "duration": "5:30"}],
        extraartists=[
            {"id": 9, "name": "Bruno Martino", "anv": "", "role": "Written-By", "tracks": "1"}
        ],
    )
    provider.releases[1003] = make_sample_release(
        release_id=1003,
        title="Italia",
        artist="Till Brönner",
        tracklist=[{"type_": "track", "position": "1", "title": "Estate", "duration": "5:31"}],
        extraartists=[],
    )

    enricher = DiscogsMasterVersionEnricher(provider, max_versions=2)
    siblings = enricher.fetch_sibling_releases(master_id=777, exclude_release_id=1001)
    assert [s.discogs_release_id for s in siblings] == [1002, 1003]
    assert siblings[0].tracks[0].duration_sec == 330
    assert any(c.artist_name == "Bruno Martino" for c in siblings[0].tracks[0].credits)


def test_musicbrainz_enricher_album_per_track_and_cache() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.url.path}?{request.url.query.decode()}")
        if request.url.path == "/ws/2/release/":
            q = request.url.params.get("query", "")
            if "barcode:4065629733131" in q:
                return httpx.Response(
                    200,
                    json={
                        "releases": [
                            {
                                "id": "mb-rel-1",
                                "title": "Aspiral",
                                "artist-credit": [{"name": "Epica", "joinphrase": ""}],
                                "media": [{"format": "CD"}],
                            }
                        ]
                    },
                )
            return httpx.Response(200, json={"releases": []})
        if request.url.path == "/ws/2/release/mb-rel-1":
            return httpx.Response(
                200,
                json={
                    "id": "mb-rel-1",
                    "title": "Aspiral",
                    "media": [
                        {
                            "position": 1,
                            "tracks": [
                                {
                                    "number": "1",
                                    "title": "Cross The Divide",
                                    "length": 258000,
                                    "artist-credit": [{"name": "Epica", "joinphrase": ""}],
                                    "recording": {
                                        "id": "mb-rec-1",
                                        "title": "Cross The Divide",
                                        "isrcs": ["DED832400405"],
                                        "relations": [
                                            {
                                                "type": "producer",
                                                "target-type": "artist",
                                                "artist": {"name": "Joost van den Broek"},
                                            },
                                            {
                                                "type": "performance",
                                                "target-type": "work",
                                                "work": {
                                                    "relations": [
                                                        {
                                                            "type": "composer",
                                                            "target-type": "artist",
                                                            "artist": {"name": "Mark Jansen"},
                                                        },
                                                        {
                                                            "type": "lyricist",
                                                            "target-type": "artist",
                                                            "artist": {"name": "Simone Simons"},
                                                        },
                                                    ]
                                                },
                                            },
                                        ],
                                    },
                                }
                            ],
                        }
                    ],
                },
            )
        if request.url.path == "/ws/2/recording/":
            return httpx.Response(
                200,
                json={
                    "recordings": [
                        {
                            "id": "mb-rec-2",
                            "title": "Solo Song",
                            "length": 195000,
                            "artist-credit": [{"name": "Indie Artist", "joinphrase": ""}],
                            "isrcs": ["DEABC2400001"],
                        }
                    ]
                },
            )
        if request.url.path == "/ws/2/recording/mb-rec-2":
            return httpx.Response(
                200,
                json={
                    "id": "mb-rec-2",
                    "title": "Solo Song",
                    "length": 195000,
                    "artist-credit": [{"name": "Indie Artist", "joinphrase": ""}],
                    "isrcs": ["DEABC2400001"],
                    "relations": [
                        {
                            "type": "producer",
                            "target-type": "artist",
                            "artist": {"name": "Studio Producer"},
                        }
                    ],
                },
            )
        return httpx.Response(404)

    cache = MemoryApiCache()
    http = httpx.Client(base_url="https://musicbrainz.org/ws/2", transport=httpx.MockTransport(handler))
    settings = Settings(musicbrainz_min_interval_sec=0.0)
    mb = MusicBrainzEnricher(settings, cache_store=cache, http_client=http)

    norm = normalize_input("4065629733131", "Aspiral", "EPICA")
    donors, trusted, is_live = mb.fetch_album_donors(norm, target_is_live=False)
    assert trusted is True
    assert is_live is False
    assert len(donors) == 1
    assert donors[0].duration_sec == 258
    assert donors[0].isrc == "DED832400405"
    assert donors[0].performing_artist_display == "Epica"
    assert {(c.role_category, c.artist_name) for c in donors[0].credits} == {
        ("producer", "Joost van den Broek"),
        ("composer", "Mark Jansen"),
        ("lyricist", "Simone Simons"),
    }
    assert len(calls) == 2

    # Cached repeat hits 0 HTTP calls
    donors_2, _, _ = mb.fetch_album_donors(norm, target_is_live=False)
    assert len(donors_2) == 1
    assert len(calls) == 2

    # Per-track recording lookup
    tr_donor, note = mb.enrich_single_track("Solo Song", "Indie Artist", need_details=True)
    assert note is None
    assert tr_donor is not None
    assert tr_donor.duration_sec == 195
    assert tr_donor.isrc == "DEABC2400001"
    assert any(c.artist_name == "Studio Producer" for c in tr_donor.credits)
    http.close()


def test_deezer_and_itunes_enrichers_handle_upc_errors_isrc_and_tracks() -> None:
    def dz_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/album/upc:4029759207641":
            return httpx.Response(200, json={"error": {"type": "DataException", "code": 800}})
        if request.url.path == "/search/album":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": 785974581,
                            "title": "ITALIA",
                            "artist": {"name": "Till Brönner"},
                        }
                    ]
                },
            )
        if request.url.path == "/album/785974581":
            return httpx.Response(
                200,
                json={
                    "id": 785974581,
                    "title": "ITALIA",
                    "link": "https://www.deezer.com/album/785974581",
                    "tracks": {
                        "data": [
                            {
                                "id": 3453708121,
                                "title": "Estate",
                                "duration": 330,
                                "isrc": None,
                                "link": "https://www.deezer.com/track/3453708121",
                                "artist": {"name": "Till Brönner"},
                            }
                        ]
                    },
                },
            )
        if request.url.path == "/track/3453708121":
            return httpx.Response(
                200,
                json={
                    "id": 3453708121,
                    "isrc": "DEH842503686",
                    "link": "https://www.deezer.com/track/3453708121",
                },
            )
        if request.url.path == "/search/track":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": 999,
                            "title": "Sie!",
                            "duration": 211,
                            "link": "https://www.deezer.com/track/999",
                            "artist": {"name": "Thomas Anders"},
                        }
                    ]
                },
            )
        if request.url.path == "/track/999":
            return httpx.Response(200, json={"id": 999, "isrc": "DEZZZ2500009", "link": "https://www.deezer.com/track/999"})
        return httpx.Response(404)

    cache = MemoryApiCache()
    dz_http = httpx.Client(base_url="https://api.deezer.com", transport=httpx.MockTransport(dz_handler))
    dz = DeezerEnricher(Settings(), cache_store=cache, http_client=dz_http)

    norm = normalize_input("4029759207641", "ITALIA (LP)", "TILL BRÖNNER")
    donors, trusted, _ = dz.fetch_album_donors(norm, target_is_live=False)
    assert trusted is False
    assert len(donors) == 1
    assert donors[0].duration_sec == 330
    isrc, _ = dz.fetch_track_isrc(donors[0].external_track_id)
    assert isrc == "DEH842503686"

    tr_donor, note = dz.enrich_single_track("Sie!", "Thomas Anders", need_isrc=True)
    assert note is None
    assert tr_donor is not None and tr_donor.duration_sec == 211 and tr_donor.isrc == "DEZZZ2500009"
    dz_http.close()

    # iTunes single track search
    def it_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search" and request.url.params.get("entity") == "song":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "wrapperType": "track",
                            "trackName": "Schau mal herein",
                            "artistName": "Ross Antony & Linda Fäh",
                            "trackTimeMillis": 204000,
                            "trackViewUrl": "https://music.apple.com/de/album/x",
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"results": []})

    it_http = httpx.Client(base_url="https://itunes.apple.com", transport=httpx.MockTransport(it_handler))
    it = ItunesTrackEnricher(Settings(), cache_store=cache, http_client=it_http)
    it_donor, it_note = it.enrich_single_track("Schau mal herein", "Ross Antony & Linda Fäh")
    assert it_note is None
    assert it_donor is not None and it_donor.duration_sec == 204
    it_http.close()


def test_gemini_track_enricher_and_live_album_blocks_per_track_tier2b() -> None:
    class _CountingTrackProvider:
        def __init__(self, include_trusted: bool = True) -> None:
            self.album_calls = 0
            self.track_calls = 0
            self._include_trusted = include_trusted

        def fetch_album_donors(self, norm, target_is_live=False, bypass_cache=False):
            self.album_calls += 1
            return ([], False, False) if self._include_trusted else ([], False)

        def enrich_single_track(self, *args, **kwargs):
            self.track_calls += 1
            return None, None

    fake_genai = _FakeGenaiClient(
        [
            SimpleNamespace(
                text=json.dumps(
                    {
                        "tracks": [
                            {
                                "track_order": 1,
                                "title": "Double Vision",
                                "artist": "Foreigner",
                                "length_sec": 265,
                                "composers": ["Mick Jones", "Lou Gramm"],
                                "lyricists": ["Lou Gramm"],
                                "producers": ["Live Producer"],
                                "isrc": "USFRG2400001",
                                "source_url": "https://musicbrainz.org/release/live-foreigner",
                            }
                        ],
                        "sources": ["https://musicbrainz.org/release/live-foreigner"],
                    }
                ),
                candidates=[],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=100, candidates_token_count=50, thoughts_token_count=0
                ),
            )
        ]
    )
    agent = GeminiAgentClient(
        Settings(gemini_api_key="test-key", agent_max_attempts=1), genai_client=fake_genai
    )
    counting_dz = _CountingTrackProvider(include_trusted=True)
    counting_it = _CountingTrackProvider(include_trusted=False)
    enricher = TrackEnricher(
        settings=Settings(),
        deezer=counting_dz,  # type: ignore[arg-type]
        itunes=counting_it,  # type: ignore[arg-type]
        gemini=GeminiTrackEnricher(agent),
    )

    # Live album with missing duration: Tier 2b per-track MUST NOT be called; Tier 3 fills when use_agent=True
    live_norm = normalize_input("4029759205296", "LIVE IN CHICAGO (2LP)", "FOREIGNER")
    existing_track = TrackOut(
        track_order=1,
        position="A1",
        title="Double Vision",
        performing_artists=[TrackArtistOut(name="Foreigner", inherited=True)],
        performing_artist_display="Foreigner",
        has_valid_duration=False,
        has_resolved_performer=True,
    )
    stats_no_agent = enricher.enrich_tracks(
        tracks=[existing_track],
        norm=live_norm,
        release_meta=ReleaseOut(
            discogs_release_id=34744227,
            url="https://www.discogs.com/release/34744227",
            title="Live In Chicago",
            artist_display="Foreigner",
            formats=[{"name": "Vinyl", "qty": "2", "descriptions": ["LP", "Album"]}],
        ),
        use_agent=False,
    )
    assert counting_dz.album_calls == 1
    assert counting_dz.track_calls == 0
    assert counting_it.track_calls == 0
    assert stats_no_agent.durations_filled == 0

    stats_with_agent = enricher.enrich_tracks(
        tracks=[existing_track],
        norm=live_norm,
        release_meta=ReleaseOut(
            discogs_release_id=34744227,
            url="https://www.discogs.com/release/34744227",
            title="Live In Chicago",
            artist_display="Foreigner",
            formats=[{"name": "Vinyl", "qty": "2", "descriptions": ["LP", "Album"]}],
        ),
        use_agent=True,
    )
    assert counting_dz.track_calls == 0
    assert stats_with_agent.durations_filled == 1
    assert stats_with_agent.isrcs_filled == 1
    # 2 composers + 1 lyricist (Lou Gramm in a distinct role) + 1 producer
    assert stats_with_agent.credits_added == 4
    t_out = stats_with_agent.tracks[0]
    assert t_out.duration_sec == 265
    assert t_out.duration_source == "gemini_agent"
    assert t_out.duration_source_url == "https://musicbrainz.org/release/live-foreigner"
    assert t_out.isrc == "USFRG2400001"


# ------------------------------------------------------------------ #
# Integration Tests: Async Queue, PostgreSQL Backfill, Recovery, CLI
# ------------------------------------------------------------------ #


class _StubControlledEnricher(TrackEnricher):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings=settings)
        self.calls: list[dict[str, Any]] = []
        self.gates: dict[str, tuple[threading.Event, threading.Event]] = {}
        self.donors_by_title: dict[str, dict[int, DonorTrack]] = {}

    def enrich_tracks(
        self,
        tracks: list[TrackOut],
        norm,
        release_meta=None,
        release_notes: str = "",
        source_provider="discogs",
        sibling_releases=None,
        agent_sources=None,
        use_agent=False,
        bypass_cache=False,
        progress_cb=None,
    ):
        self.calls.append(
            {
                "title": norm.album_title_raw,
                "use_agent": use_agent,
                "bypass_cache": bypass_cache,
            }
        )
        gate = self.gates.get(norm.album_title_raw)
        if gate is not None:
            started, release = gate
            started.set()
            assert release.wait(timeout=5.0), "Timed out waiting for enrichment gate"

        # Temporarily run Tier 0 + apply injected stub donors per track_order
        stats = super().enrich_tracks(
            tracks=tracks,
            norm=norm,
            release_meta=release_meta,
            release_notes=release_notes,
            source_provider=source_provider,
            sibling_releases=sibling_releases,
            agent_sources=agent_sources,
            use_agent=False,
            bypass_cache=bypass_cache,
            progress_cb=progress_cb,
        )
        stub_map = self.donors_by_title.get(norm.album_title_raw, {})
        if stub_map:
            from app.enricher import apply_donor_to_track, compute_track_coverages

            changed = False
            for t in stats.tracks:
                donor = stub_map.get(t.track_order)
                if donor is not None and apply_donor_to_track(t, donor, stats):
                    changed = True
            if changed:
                if "stub_musicbrainz" not in stats.providers_used:
                    stats.providers_used.append("stub_musicbrainz")
                dur_a, perf_a, wri_a, prod_a = compute_track_coverages(stats.tracks)
                stats.duration_coverage_after = dur_a
                stats.performer_coverage_after = perf_a
                stats.writer_credit_coverage_after = wri_a
                stats.producer_credit_coverage_after = prod_a
        return stats


def test_async_enrich_queue_backfill_preserves_credits_and_handles_agent_records(
    enrich_test_db_url: str,
) -> None:
    _truncate_db(enrich_test_db_url)
    provider = FakeCatalogProvider()
    provider.search_by_tier_or_key = [
        ({"barcode": "9990004441111"}, [{"id": 9900401}]),
    ]
    provider.releases[9900401] = make_sample_release(
        release_id=9900401,
        title="__test_enrich_vinyl",
        artist="Vinyl Artist",
        barcode="9990004441111",
        tracklist=[
            {
                "type_": "track",
                "position": "A1",
                "title": "Silent Groove",
                "duration": "",
                "extraartists": [
                    {"id": 501, "name": "Existing Lyricist", "anv": "", "role": "Lyrics By"}
                ],
            },
            {"type_": "track", "position": "B1", "title": "Side B Cut", "duration": ""},
        ],
        extraartists=[],
    )

    class _FakeAgent:
        model_name = "gemini-test-enrich"

        def research_album(self, barcode: str, album: str, artist: str) -> AgentAnswer:
            return AgentAnswer(
                tracks=[
                    AgentTrack(disc=1, number=1, title="Agent Cut 1", length_sec=None),
                ],
                sources=["https://artist.example/album"],
            )

    settings = Settings(
        database_url=enrich_test_db_url,
        enable_agent_fallback=True,
        gemini_api_key="test-key",
        enable_queue_workers=True,
        discogs_queue_concurrency=1,
        agent_queue_concurrency=1,
        enrich_queue_concurrency=2,
        queue_poll_interval_sec=0.05,
        enable_track_enrichment=False,  # Seed pre-enrichment state first
        enable_idle_auto_enrichment=False,
    )
    from app.extractor import ExtractedCredit

    stub_enricher = _StubControlledEnricher(settings)
    stub_enricher.donors_by_title["__test_enrich_vinyl"] = {
        1: DonorTrack(
            title="Silent Groove",
            duration_sec=245,
            isrc="DEAAA2500001",
            credits=[
                ExtractedCredit(
                    discogs_artist_id=None,
                    artist_name="Enriched Composer",
                    anv="",
                    role_raw="Composer",
                    role_category="composer",
                    credit_source="musicbrainz",
                    source_url="https://musicbrainz.org/release/mb-401",
                ),
                ExtractedCredit(
                    discogs_artist_id=None,
                    artist_name="Enriched Producer",
                    anv="",
                    role_raw="Producer",
                    role_category="producer",
                    credit_source="musicbrainz",
                    source_url="https://musicbrainz.org/release/mb-401",
                ),
            ],
            source_label="musicbrainz",
            source_url="https://musicbrainz.org/release/mb-401",
        ),
        2: DonorTrack(
            title="Side B Cut",
            duration_sec=198,
            isrc="DEAAA2500002",
            source_label="deezer",
            source_url="https://www.deezer.com/track/402",
        ),
    }
    stub_enricher.donors_by_title["__test_enrich_agent_rec"] = {
        1: DonorTrack(
            title="Agent Cut 1",
            duration_sec=305,
            isrc="DEAGT2500001",
            source_label="deezer",
            source_url="https://www.deezer.com/track/901",
        )
    }

    app = create_app(
        settings=settings,
        provider=provider,
        agent_provider=_FakeAgent(),
        enricher=stub_enricher,
    )
    try:
        with TestClient(app) as client:
            # 1. Seed a matched Discogs vinyl record without durations + an agent record without duration
            r_vinyl = client.post(
                "/lookup",
                json={
                    "barcode": "9990004441111",
                    "album_title": "__test_enrich_vinyl",
                    "album_artist": "Vinyl Artist",
                },
            )
            assert r_vinyl.status_code == 200
            vinyl_id = r_vinyl.json()["input_id"]
            assert r_vinyl.json()["evaluation"]["duration_coverage"] == 0.0

            r_agent = client.post(
                "/lookup",
                json={
                    "barcode": "",
                    "album_title": "__test_enrich_agent_rec",
                    "album_artist": "Fantastischen Vier, Die",
                },
            )
            assert r_agent.status_code == 200
            agent_id = r_agent.json()["input_id"]
            # AgentTrack already inherited un-inverted single artist via Tier 0!
            assert (
                r_agent.json()["tracks"][0]["performing_artist_display"]
                == "Die Fantastischen Vier"
            )
            assert r_agent.json()["tracks"][0]["has_resolved_performer"] is True

            # 2. Enqueue async backfill of missing records via POST /jobs/enrich-missing
            app.state.settings.enable_track_enrichment = True
            enq = client.post("/jobs/enrich-missing")
            assert enq.status_code == 202
            job_id = enq.json()["id"]
            assert enq.json()["job_type"] == "track_enrich"
            assert enq.json()["total_items"] == 2

            detail = _wait_until(
                lambda: (
                    d
                    if (d := client.get(f"/jobs/{job_id}").json())["job"]["status"] == "completed"
                    else None
                )
            )
            assert detail["job"]["completed_items"] == 2
            assert detail["job"]["error_count"] == 0

            # 3. Verify vinyl record tracks, preserved credits, added credits, ISRC, and live coverages
            vinyl_detail = client.get(f"/records/{vinyl_id}").json()
            t1, t2 = vinyl_detail["tracks"]
            assert t1["duration_sec"] == 245
            assert t1["duration_raw"] == "4:05"
            assert t1["duration_source"] == "musicbrainz"
            assert t1["duration_source_url"] == "https://musicbrainz.org/release/mb-401"
            assert t1["isrc"] == "DEAAA2500001"
            # Existing Discogs credit preserved + MB credits added!
            assert any(c["artist_name"] == "Existing Lyricist" for c in t1["credits"]["lyricists"])
            assert any(c["artist_name"] == "Enriched Composer" for c in t1["credits"]["composers"])
            assert any(c["artist_name"] == "Enriched Producer" for c in t1["credits"]["producers"])
            assert t2["duration_sec"] == 198
            assert t2["duration_source"] == "deezer"
            assert vinyl_detail["runs"][0]["evaluation_summary_json"]["duration_coverage"] == 1.0

            # 4. Verify agent fallback record was enriched in agent_fallback_tracks
            agent_detail = client.get(f"/records/{agent_id}").json()
            at1 = agent_detail["tracks"][0]
            assert at1["duration_sec"] == 305
            assert at1["duration_source"] == "deezer"
            assert at1["isrc"] == "DEAGT2500001"
            assert at1["performing_artist_display"] == "Die Fantastischen Vier"

            # 5. Idempotency: second POST /jobs/enrich-missing sees 0 un-enriched records
            enq2 = client.post("/jobs/enrich-missing")
            assert enq2.status_code == 202
            assert enq2.json()["total_items"] == 0
            assert enq2.json()["status"] == "completed"

            # 6. Dashboard HTML renders provenance tags, ISRC, and Enrich buttons
            res_html = client.get("/results").text
            assert "Enrich Missing Data" in res_html
            det_html = client.get(f"/results/{vinyl_id}").text
            assert "Enrich Missing Tracks" in det_html
            assert "DEAAA2500001" in det_html
            assert "[musicbrainz]" in det_html
            assert "Existing Lyricist" in det_html
            q_html = client.get("/queue").text
            assert "Active Enrich" in q_html
            assert "Queued Enrich" in q_html

            # 7. Sync single-record enrich endpoint & error checks
            sync_res = client.post(f"/records/{vinyl_id}/enrich", json={"force_refresh": True})
            assert sync_res.status_code == 200
            assert sync_res.json()["duration_coverage_after"] == 1.0
            assert client.post("/jobs/enrich/99999999").status_code == 404
            assert client.post("/records/99999999/enrich").status_code == 404

            # 8. Automatic post-match async handoff: queued_discogs -> queued_enrich -> done
            provider.search_by_tier_or_key.append(
                ({"barcode": "9990004442222"}, [{"id": 9900402}])
            )
            provider.releases[9900402] = make_sample_release(
                release_id=9900402,
                title="__test_auto_handoff_enrich",
                artist="Handoff Band",
                barcode="9990004442222",
                tracklist=[{"type_": "track", "position": "1", "title": "Song X", "duration": ""}],
                extraartists=[],
            )
            stub_enricher.donors_by_title["__test_auto_handoff_enrich"] = {
                1: DonorTrack(
                    title="Song X",
                    duration_sec=222,
                    isrc="DEHND2500001",
                    source_label="deezer",
                    source_url="https://www.deezer.com/track/888",
                )
            }
            auto_job = client.post(
                "/jobs/lookup",
                json={
                    "barcode": "9990004442222",
                    "album_title": "__test_auto_handoff_enrich",
                    "album_artist": "Handoff Band",
                    "mode": "full",
                },
            )
            assert auto_job.status_code == 202
            auto_jid = auto_job.json()["job"]["id"]
            auto_done = _wait_until(
                lambda: (
                    d
                    if (d := client.get(f"/jobs/{auto_jid}").json())["job"]["status"] == "completed"
                    else None
                )
            )
            auto_input_id = auto_done["items"][0]["input_id"]
            auto_rec = client.get(f"/records/{auto_input_id}").json()
            assert auto_rec["tracks"][0]["duration_sec"] == 222
            assert auto_rec["tracks"][0]["isrc"] == "DEHND2500001"
    finally:
        _truncate_db(enrich_test_db_url)


def test_enrich_crash_recovery_cancel_and_cli_commands(
    enrich_test_db_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _truncate_db(enrich_test_db_url)
    provider = FakeCatalogProvider()
    provider.search_by_tier_or_key = [
        ({"barcode": "9990004443333"}, [{"id": 9900403}]),
    ]
    provider.releases[9900403] = make_sample_release(
        release_id=9900403,
        title="__test_cli_enrich",
        artist="CLI Enrich Band",
        barcode="9990004443333",
        tracklist=[{"type_": "track", "position": "1", "title": "CLI Cut", "duration": ""}],
        extraartists=[],
    )

    cli_settings = Settings(
        database_url=enrich_test_db_url,
        enable_agent_fallback=False,
        enable_queue_workers=False,
        enable_track_enrichment=False,
    )
    stub_enricher = _StubControlledEnricher(cli_settings)
    stub_enricher.donors_by_title["__test_cli_enrich"] = {
        1: DonorTrack(
            title="CLI Cut",
            duration_sec=180,
            isrc="DECLIE250001",
            source_label="musicbrainz",
            source_url="https://musicbrainz.org/release/cli-1",
        )
    }

    try:
        # 1. Seed record via CLI lookup
        code_l = cli.main(
            [
                "lookup",
                "--barcode",
                "9990004443333",
                "--title",
                "__test_cli_enrich",
                "--artist",
                "CLI Enrich Band",
            ],
            provider=provider,
            settings=cli_settings,
            enricher=stub_enricher,
        )
        assert code_l == 0
        lookup_out = json.loads(capsys.readouterr().out)
        input_id = lookup_out["input_id"]

        # 2. CLI enqueue-enrich
        code_enq = cli.main(
            ["enqueue-enrich", "--all-missing"],
            provider=provider,
            settings=cli_settings,
            enricher=stub_enricher,
        )
        assert code_enq == 0
        job_out = json.loads(capsys.readouterr().out)
        assert job_out["job_type"] == "track_enrich"
        assert job_out["total_items"] == 1

        # Simulate crash during checking_enrich -> QueueManager startup recovers & finishes it
        pool = ConnectionPool(conninfo=enrich_test_db_url, min_size=1, max_size=2, open=True)
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE queue_items SET stage = 'checking_enrich', worker_name = 'dead-enrich' WHERE job_id = %s",
                        (job_out["id"],),
                    )
                    cur.execute(
                        "UPDATE queue_jobs SET status = 'running' WHERE id = %s",
                        (job_out["id"],),
                    )
                conn.commit()
        finally:
            pool.close()

        app = create_app(
            settings=Settings(
                database_url=enrich_test_db_url,
                enable_agent_fallback=False,
                enable_queue_workers=True,
                enrich_queue_concurrency=1,
                queue_poll_interval_sec=0.05,
                enable_track_enrichment=True,
                enable_idle_auto_enrichment=False,
            ),
            provider=provider,
            enricher=stub_enricher,
        )
        with TestClient(app) as client:
            done = _wait_until(
                lambda: (
                    d
                    if (d := client.get(f"/jobs/{job_out['id']}").json())["job"]["status"]
                    == "completed"
                    else None
                )
            )
            assert done["job"]["completed_items"] == 1
            rec = client.get(f"/records/{input_id}").json()
            assert rec["tracks"][0]["duration_sec"] == 180
            assert rec["tracks"][0]["isrc"] == "DECLIE250001"

        # 2b. Verify queued_enrich cancellation while workers are stopped
        paused_app = create_app(
            settings=Settings(
                database_url=enrich_test_db_url,
                enable_agent_fallback=False,
                enable_queue_workers=False,
                enable_track_enrichment=True,
                enable_idle_auto_enrichment=False,
            ),
            provider=provider,
            enricher=stub_enricher,
        )
        with TestClient(paused_app) as pclient:
            enq_cancel = pclient.post(
                f"/jobs/enrich/{input_id}", json={"force_refresh": True}
            )
            assert enq_cancel.status_code == 202
            cancel_jid = enq_cancel.json()["job"]["id"]
            assert enq_cancel.json()["items"][0]["stage"] == "queued_enrich"
            cancelled = pclient.post(f"/jobs/{cancel_jid}/cancel")
            assert cancelled.status_code == 200
            assert cancelled.json()["status"] == "cancelled"
            assert cancelled.json()["cancelled_count"] == 1

        # 3. CLI sync enrich
        code_sync = cli.main(
            ["enrich", "--input-id", str(input_id), "--refresh"],
            provider=provider,
            settings=cli_settings,
            enricher=stub_enricher,
        )
        assert code_sync == 0
        sync_out = json.loads(capsys.readouterr().out)
        assert sync_out["duration_coverage_after"] == 1.0
    finally:
        _truncate_db(enrich_test_db_url)


def test_deezer_single_track_unquoted_and_parenthetical_fallback() -> None:
    seen_queries: list[str] = []

    def dz_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search/track":
            q = str(request.url.params.get("q") or "")
            seen_queries.append(q)
            assert 'artist:"' not in q
            assert 'track:"' not in q
            if q == "Gala Freed From Desire":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": 6907156,
                                "title": "Freed From Desire",
                                "duration": 213,
                                "artist": {"name": "Gala"},
                                "link": "https://www.deezer.com/track/6907156",
                            }
                        ]
                    },
                )
            if q == "Snap! Rhythm Is a Dancer (7 Edit)":
                return httpx.Response(200, json={"data": []})
            if q == "Snap! Rhythm Is a Dancer":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": 1001,
                                "title": "Rhythm Is a Dancer",
                                "duration": 315,
                                "artist": {"name": "Snap!"},
                                "link": "https://www.deezer.com/track/1001",
                            },
                            {
                                "id": 1002,
                                "title": 'Rhythm Is a Dancer (7" Edit)',
                                "duration": 225,
                                "artist": {"name": "Snap!"},
                                "link": "https://www.deezer.com/track/1002",
                            },
                        ]
                    },
                )
            return httpx.Response(200, json={"data": []})
        if request.url.path == "/track/6907156":
            return httpx.Response(
                200,
                json={
                    "id": 6907156,
                    "isrc": "ITA599600001",
                    "link": "https://www.deezer.com/track/6907156",
                },
            )
        if request.url.path == "/track/1002":
            return httpx.Response(
                200,
                json={
                    "id": 1002,
                    "isrc": "DEZZZ9200002",
                    "link": "https://www.deezer.com/track/1002",
                },
            )
        return httpx.Response(404)

    cache = MemoryApiCache()
    dz_http = httpx.Client(
        base_url="https://api.deezer.com", transport=httpx.MockTransport(dz_handler)
    )
    dz = DeezerEnricher(Settings(), cache_store=cache, http_client=dz_http)

    d1, n1 = dz.enrich_single_track("Freed From Desire", "Gala", need_isrc=True)
    assert n1 is None
    assert d1 is not None and d1.duration_sec == 213 and d1.isrc == "ITA599600001"

    d2, n2 = dz.enrich_single_track('Rhythm Is a Dancer (7" Edit)', "Snap!", need_isrc=True)
    assert n2 is None
    assert d2 is not None
    assert d2.title == 'Rhythm Is a Dancer (7" Edit)'
    assert d2.duration_sec == 225
    assert d2.isrc == "DEZZZ9200002"
    assert seen_queries == [
        "Gala Freed From Desire",
        "Snap! Rhythm Is a Dancer (7 Edit)",
        "Snap! Rhythm Is a Dancer",
    ]
    dz_http.close()

    # Release notes mentioning live recording must trigger live-release Tier 2b guardrail
    assert is_live_release(title="Greatest Hits", notes="Recorded live in Berlin 2019") is True
    blocking_dz = DeezerEnricher(
        Settings(),
        cache_store=MemoryApiCache(),
        http_client=httpx.Client(
            base_url="https://api.deezer.com",
            transport=httpx.MockTransport(
                lambda req: pytest.fail(f"Unexpected Tier 2b call on live notes: {req.url}")
                if req.url.path == "/search/track"
                else httpx.Response(200, json={"data": []})
            ),
        ),
    )
    enricher = TrackEnricher(settings=Settings(), deezer=blocking_dz)
    live_stats = enricher.enrich_tracks(
        tracks=[
            TrackOut(
                track_order=1,
                position="1",
                title="Encore Song",
                performing_artists=[TrackArtistOut(name="Tour Band", inherited=True)],
                performing_artist_display="Tour Band",
                has_resolved_performer=True,
            )
        ],
        norm=normalize_input("9990004440001", "Greatest Hits", "Tour Band"),
        release_meta=ReleaseOut(
            discogs_release_id=9900404,
            url="https://www.discogs.com/release/9900404",
            title="Greatest Hits",
            artist_display="Tour Band",
            formats=[{"name": "CD", "qty": "1", "descriptions": ["Album"]}],
        ),
        release_notes="Recorded live in Berlin",
        use_agent=False,
    )
    assert live_stats.durations_filled == 0


def test_gemini_enrich_tracks_batch_chunks_and_tracks_cost(
    enrich_test_db_url: str,
) -> None:
    _truncate_db(enrich_test_db_url)
    provider = FakeCatalogProvider()
    provider.search_by_tier_or_key = [
        ({"barcode": "9990004444444"}, [{"id": 9900405}]),
    ]
    provider.releases[9900405] = make_sample_release(
        release_id=9900405,
        title="__test_chunk30",
        artist="Chunk Orchestra",
        barcode="9990004444444",
        tracklist=[
            {"type_": "track", "position": str(i), "title": f"Movement {i}", "duration": ""}
            for i in range(1, 31)
        ],
        extraartists=[],
    )

    chunk1_tracks = [
        {
            "track_order": i,
            "title": f"Movement {i}",
            "artist": "Chunk Orchestra",
            "length_sec": 180 + i,
            "composers": [],
            "lyricists": [],
            "producers": [],
            "isrc": f"DECHK25000{i:02d}",
            "source_url": "https://musicbrainz.org/release/chunk-test",
        }
        for i in range(1, 26)
    ]
    chunk2_tracks = [
        {
            "track_order": i,
            "title": f"Movement {i}",
            "artist": "Chunk Orchestra",
            "length_sec": 180 + i,
            "composers": [],
            "lyricists": [],
            "producers": [],
            "isrc": f"DECHK25000{i:02d}",
            "source_url": "https://musicbrainz.org/release/chunk-test",
        }
        for i in range(26, 31)
    ]
    fake_genai = _FakeGenaiClient(
        [
            SimpleNamespace(
                text=json.dumps(
                    {
                        "tracks": chunk1_tracks,
                        "sources": ["https://musicbrainz.org/release/chunk-test"],
                    }
                ),
                candidates=[],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=200, candidates_token_count=100, thoughts_token_count=0
                ),
            ),
            SimpleNamespace(
                text=json.dumps(
                    {
                        "tracks": chunk2_tracks,
                        "sources": ["https://musicbrainz.org/release/chunk-test"],
                    }
                ),
                candidates=[],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=80, candidates_token_count=40, thoughts_token_count=0
                ),
            ),
        ]
    )
    settings = Settings(
        database_url=enrich_test_db_url,
        enable_agent_fallback=True,
        gemini_api_key="test-key",
        enable_queue_workers=False,
        enable_track_enrichment=False,
        enable_idle_auto_enrichment=False,
    )
    agent = GeminiAgentClient(
        Settings(gemini_api_key="test-key", agent_max_attempts=1), genai_client=fake_genai
    )
    enricher = TrackEnricher(settings=settings, gemini=GeminiTrackEnricher(agent))

    app = create_app(
        settings=settings,
        provider=provider,
        agent_provider=agent,
        enricher=enricher,
    )
    try:
        with TestClient(app) as client:
            r_seed = client.post(
                "/lookup",
                json={
                    "barcode": "9990004444444",
                    "album_title": "__test_chunk30",
                    "album_artist": "Chunk Orchestra",
                },
            )
            assert r_seed.status_code == 200
            input_id = r_seed.json()["input_id"]

            app.state.settings.enable_track_enrichment = True
            res = client.post(
                f"/records/{input_id}/enrich",
                json={"use_agent": True, "force_refresh": True},
            )
            assert res.status_code == 200
            body = res.json()
            assert len(fake_genai.models.calls) == 2
            assert body["tracks_total"] == 30
            assert body["durations_filled"] == 30
            assert body["isrcs_filled"] == 30
            assert body["agent_input_tokens"] == 280
            assert body["agent_output_tokens"] == 140
            assert body["agent_cost_usd"] > 0.0

            rec_detail = client.get(f"/records/{input_id}").json()
            assert rec_detail["record"]["agent_input_tokens"] == 280
            assert rec_detail["record"]["agent_output_tokens"] == 140
            assert rec_detail["record"]["agent_cost_usd"] == pytest.approx(
                body["agent_cost_usd"], rel=1e-5
            )
            rec_list = client.get("/records").json()
            matched_row = next(r for r in rec_list if r["input_id"] == input_id)
            assert matched_row["agent_input_tokens"] == 280
            assert matched_row["agent_output_tokens"] == 140
    finally:
        _truncate_db(enrich_test_db_url)


def test_write_job_reports_populates_coverages_and_featured_ui(
    enrich_test_db_url: str, tmp_path
) -> None:
    _truncate_db(enrich_test_db_url)
    provider = FakeCatalogProvider()
    provider.search_by_tier_or_key = [
        ({"barcode": "9990004445555"}, [{"id": 9900407}]),
    ]
    provider.releases[9900407] = make_sample_release(
        release_id=9900407,
        title="__test_report_album",
        artist="Report Artist",
        barcode="9990004445555",
        tracklist=[
            {
                "type_": "track",
                "position": "1",
                "title": "Featured Song",
                "duration": "3:20",
                "extraartists": [
                    {"id": 601, "name": "Song Writer", "anv": "", "role": "Written-By"},
                    {"id": 602, "name": "Beat Maker", "anv": "", "role": "Producer"},
                    {"id": 603, "name": "Guest Rapper", "anv": "", "role": "Featuring"},
                ],
            }
        ],
        extraartists=[],
    )

    csv_file = tmp_path / "mini.csv"
    csv_file.write_text(
        "barcode,album_title,album_artist\n"
        "9990004445555,__test_report_album,Report Artist\n"
        "9990004449999,__test_report_missing,Nobody\n",
        encoding="utf-8",
    )
    out_json = tmp_path / "mini_out.json"

    settings = Settings(
        database_url=enrich_test_db_url,
        enable_agent_fallback=False,
        enable_queue_workers=True,
        discogs_queue_concurrency=1,
        enrich_queue_concurrency=1,
        queue_poll_interval_sec=0.05,
        enable_track_enrichment=True,
        enable_idle_auto_enrichment=False,
    )
    stub_enricher = _StubControlledEnricher(settings)
    stub_enricher.donors_by_title["__test_report_album"] = {
        1: DonorTrack(
            title="Featured Song",
            duration_sec=200,
            isrc="DERPT2500001",
            source_label="deezer",
            source_url="https://www.deezer.com/track/777",
        )
    }

    app = create_app(
        settings=settings,
        provider=provider,
        enricher=stub_enricher,
    )
    try:
        with TestClient(app) as client:
            batch_res = client.post(
                "/jobs/batch",
                json={"csv_path": str(csv_file), "output_path": str(out_json)},
            )
            assert batch_res.status_code == 202
            job_id = batch_res.json()["id"]

            done = _wait_until(
                lambda: (
                    d
                    if (d := client.get(f"/jobs/{job_id}").json())["job"]["status"]
                    == "completed"
                    else None
                )
            )
            assert done["job"]["completed_items"] == 2
            assert out_json.exists()
            out_md = out_json.with_suffix(".md")
            assert out_md.exists()

            report = json.loads(out_json.read_text(encoding="utf-8"))
            outcomes = report["summary"]["outcomes"]
            row1, row2 = outcomes[0], outcomes[1]
            assert row1["match_confidence"] is not None and row1["match_confidence"] >= 0.8
            assert row1["duration_coverage"] == 1.0
            assert row1["performer_coverage"] == 1.0
            assert row1["writer_credit_coverage"] == 1.0
            assert row1["producer_credit_coverage"] == 1.0
            assert row1["isrc_coverage"] == 1.0
            assert row2["status"] == "not_found"
            assert len(row2["rejection_reasons"]) >= 1

            md_text = out_md.read_text(encoding="utf-8")
            assert "ISRC Cov" in md_text
            assert "100%" in md_text

            # Verify Featured column & Allow Gemini toggles in HTML dashboard
            det_html = client.get(f"/results/{row1['input_id']}").text
            assert "<th>Featured</th>" in det_html
            assert "Guest Rapper" in det_html
            assert "<th>Producers</th>" in det_html

            results_html = client.get("/results").text
            assert "enrichMissingUseAgent" in results_html
            assert "Allow Gemini" in results_html
            queue_html = client.get("/queue").text
            assert "enrichMissingUseAgent" in queue_html
            assert "Idle:${data.idle_auto_enrichment_enabled" in queue_html
    finally:
        _truncate_db(enrich_test_db_url)


def test_idle_queue_auto_enrichment_sweep(enrich_test_db_url: str) -> None:
    _truncate_db(enrich_test_db_url)
    provider = FakeCatalogProvider()
    provider.search_by_tier_or_key = [
        ({"barcode": "9990004446666"}, [{"id": 9900406}]),
    ]
    provider.releases[9900406] = make_sample_release(
        release_id=9900406,
        title="__test_idle_sweep",
        artist="Idle Artist",
        barcode="9990004446666",
        tracklist=[{"type_": "track", "position": "1", "title": "Idle Cut", "duration": ""}],
        extraartists=[],
    )

    settings = Settings(
        database_url=enrich_test_db_url,
        enable_agent_fallback=False,
        enable_queue_workers=True,
        discogs_queue_concurrency=1,
        enrich_queue_concurrency=1,
        queue_poll_interval_sec=0.05,
        enable_track_enrichment=False,  # Seed un-enriched matched release first
        enable_idle_auto_enrichment=True,
        idle_enrichment_interval_sec=0.1,
        idle_enrichment_batch_size=5,
    )
    stub_enricher = _StubControlledEnricher(settings)
    stub_enricher.donors_by_title["__test_idle_sweep"] = {
        1: DonorTrack(
            title="Idle Cut",
            duration_sec=240,
            isrc="DEIDL2500001",
            source_label="deezer",
            source_url="https://www.deezer.com/track/606",
        )
    }

    app = create_app(
        settings=settings,
        provider=provider,
        enricher=stub_enricher,
    )
    try:
        with TestClient(app) as client:
            r_seed = client.post(
                "/lookup",
                json={
                    "barcode": "9990004446666",
                    "album_title": "__test_idle_sweep",
                    "album_artist": "Idle Artist",
                },
            )
            assert r_seed.status_code == 200
            input_id = r_seed.json()["input_id"]
            assert r_seed.json()["evaluation"]["duration_coverage"] == 0.0

            # Enable enrichment and verify /jobs/status reports idle auto-enrichment enabled
            app.state.settings.enable_track_enrichment = True
            status_json = client.get("/jobs/status").json()
            assert status_json["idle_auto_enrichment_enabled"] is True

            # Wait for idle sweep worker (enrich-1) to automatically create & finish track_enrich_auto job
            auto_job = _wait_until(
                lambda: next(
                    (
                        j
                        for j in client.get("/jobs?limit=10").json()
                        if j["job_type"] == "track_enrich_auto" and j["status"] == "completed"
                    ),
                    None,
                ),
                timeout=6.0,
            )
            assert auto_job["completed_items"] == 1

            rec_after = client.get(f"/records/{input_id}").json()
            assert rec_after["tracks"][0]["duration_sec"] == 240
            assert rec_after["tracks"][0]["isrc"] == "DEIDL2500001"

            # Verify enriched_at was stamped and no duplicate idle job is spawned
            with psycopg.connect(enrich_test_db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT enriched_at FROM releases WHERE discogs_release_id = 9900406"
                    )
                    row = cur.fetchone()
                    assert row is not None and row[0] is not None

            time.sleep(0.25)
            auto_jobs = [
                j for j in client.get("/jobs?limit=10").json() if j["job_type"] == "track_enrich_auto"
            ]
            assert len(auto_jobs) == 1
    finally:
        _truncate_db(enrich_test_db_url)
