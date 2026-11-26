import csv
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool

from app import cli
from app.config import Settings, get_settings
from app.discogs_client import (
    DiscogsAuthError,
    DiscogsClient,
    DiscogsRateLimitError,
    DiscogsUpstreamError,
)
from app.main import create_app
from app.repository import PipelineRepository
from tests.conftest import make_sample_release


class FakeCatalogProvider:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.release_calls: list[int] = []
        self.search_responses: dict[str, list[dict[str, Any]]] = {}
        self.search_by_tier_or_key: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        self.releases: dict[int, dict[str, Any]] = {}
        self.fail_with: Exception | None = None

    def search_releases(
        self, params: dict[str, Any], bypass_cache: bool = False
    ) -> tuple[dict[str, Any], bool]:
        self.search_calls.append({"params": params, "bypass_cache": bypass_cache})
        if self.fail_with is not None:
            raise self.fail_with
        for expected_subset, results in self.search_by_tier_or_key:
            if all(params.get(k) == v for k, v in expected_subset.items()):
                return {"results": results}, False
        for key, results in self.search_responses.items():
            if key in str(params):
                return {"results": results}, False
        return {"results": []}, False

    def get_release(
        self, release_id: int, bypass_cache: bool = False
    ) -> tuple[dict[str, Any], bool]:
        self.release_calls.append(release_id)
        if self.fail_with is not None:
            raise self.fail_with
        return self.releases.get(release_id, {}), False


def _cleanup_test_rows(baseline_cache_keys: set[str] | None = None) -> None:
    settings = get_settings()
    pool = ConnectionPool(conninfo=settings.database_url, min_size=1, max_size=2, open=True)
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM input_records
                    WHERE album_title_raw LIKE '__test_%'
                       OR barcode_raw LIKE '999000%'
                       OR barcode_raw LIKE '0999000%'
                    """
                )
                cur.execute(
                    "DELETE FROM releases WHERE discogs_release_id BETWEEN 9900000 AND 9999999 OR title LIKE '__test_%'"
                )
                if baseline_cache_keys is not None:
                    cur.execute("SELECT cache_key FROM api_cache")
                    for (key,) in cur.fetchall():
                        if key not in baseline_cache_keys:
                            cur.execute("DELETE FROM api_cache WHERE cache_key = %s", (key,))
            conn.commit()
    finally:
        pool.close()


def _get_cache_keys() -> set[str]:
    settings = get_settings()
    pool = ConnectionPool(conninfo=settings.database_url, min_size=1, max_size=2, open=True)
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT cache_key FROM api_cache")
                return {r[0] for r in cur.fetchall()}
    finally:
        pool.close()


@pytest.fixture()
def pg_client():
    baseline_keys = _get_cache_keys()
    _cleanup_test_rows(baseline_keys)
    provider = FakeCatalogProvider()
    settings = Settings(
        database_url=get_settings().database_url,
        enable_agent_fallback=False,
        enable_queue_workers=False,
    )
    app = create_app(settings=settings, provider=provider)
    with TestClient(app) as client:
        yield client, provider, app
    _cleanup_test_rows(baseline_keys)


def test_real_discogs_client_with_pg_api_cache_and_upc_ean_dedup() -> None:
    baseline_keys = _get_cache_keys()
    _cleanup_test_rows(baseline_keys)

    http_calls: list[str] = []
    rel_payload = make_sample_release(
        release_id=9900001,
        title="__test_upc_album",
        artist="Test Artist",
        barcode="0999000111222",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        http_calls.append(str(request.url))
        if request.url.path == "/database/search":
            bc = request.url.params.get("barcode")
            if bc == "999000111222":
                return httpx.Response(200, json={"results": [{"id": 9900001}]})
            return httpx.Response(200, json={"results": []})
        if request.url.path == "/releases/9900001":
            return httpx.Response(200, json=rel_payload)
        return httpx.Response(404)

    settings = Settings(
        database_url=get_settings().database_url,
        discogs_token="test-token",
        discogs_user_agent="TestPipeline/1.0",
        enable_agent_fallback=False,
        enable_queue_workers=False,
    )
    pool = ConnectionPool(conninfo=settings.database_url, min_size=1, max_size=2, open=True)
    repo = PipelineRepository(pool)
    repo.ensure_schema()
    mock_http = httpx.Client(
        base_url="https://api.discogs.com", transport=httpx.MockTransport(handler)
    )
    discogs = DiscogsClient(settings, cache_store=repo, http_client=mock_http)
    app = create_app(settings=settings, provider=discogs)

    try:
        with TestClient(app) as client:
            # 1. Live Discogs call (12-digit UPC)
            r1 = client.post(
                "/lookup",
                json={
                    "barcode": "999000111222",
                    "album_title": "__test_upc_album",
                    "album_artist": "Test Artist",
                },
            )
            assert r1.status_code == 200
            assert r1.json()["source"] == "discogs_live"
            assert len(http_calls) == 2  # 1 search + 1 release detail

            # 2. 13-digit EAN variant hits release_barcodes locally (0 HTTP calls)
            r2 = client.post(
                "/lookup",
                json={
                    "barcode": "0999000111222",
                    "album_title": "__test_upc_album",
                    "album_artist": "Test Artist",
                },
            )
            assert r2.status_code == 200
            assert r2.json()["source"] == "database"
            assert len(http_calls) == 2

            # 3. force_refresh re-upserts release without duplicating tracks/credits
            r3 = client.post(
                "/lookup",
                json={
                    "barcode": "999000111222",
                    "album_title": "__test_upc_album",
                    "album_artist": "Test Artist",
                    "force_refresh": True,
                },
            )
            assert r3.status_code == 200
            assert len(r3.json()["tracks"]) == 4
            assert len(http_calls) == 4

            # 4. Delete releases/inputs row while keeping api_cache -> next lookup uses discogs_cached (0 HTTP calls)
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM input_records WHERE barcode_raw LIKE '%999000111222'")
                    cur.execute("DELETE FROM releases WHERE discogs_release_id = 9900001")
                conn.commit()

            r4 = client.post(
                "/lookup",
                json={
                    "barcode": "999000111222",
                    "album_title": "__test_upc_album",
                    "album_artist": "Test Artist",
                },
            )
            assert r4.status_code == 200
            assert r4.json()["source"] == "discogs_cached"
            assert len(http_calls) == 4
    finally:
        mock_http.close()
        pool.close()
        _cleanup_test_rows(baseline_keys)


def test_tier1_empty_stub_falls_back_to_tier2_and_tier3(pg_client) -> None:
    client, provider, _ = pg_client

    # Tier 1 barcode returns empty-tracklist stub 9900010; Tier 2 returns valid 9900011
    provider.search_by_tier_or_key = [
        ({"barcode": "9990002223334"}, [{"id": 9900010}]),
        (
            {"release_title": "__test_cascade_album", "artist": "Cascade Artist"},
            [{"id": 9900011}],
        ),
        ({"q": "Freetext Artist __test_freetext_album"}, [{"id": 9900012}]),
    ]
    provider.releases[9900010] = make_sample_release(
        release_id=9900010,
        title="__test_cascade_album",
        artist="Cascade Artist",
        barcode="9990002223334",
        tracklist=[],
    )
    provider.releases[9900011] = make_sample_release(
        release_id=9900011,
        title="__test_cascade_album",
        artist="Cascade Artist",
        barcode="",
    )
    provider.releases[9900012] = make_sample_release(
        release_id=9900012,
        title="__test_freetext_album",
        artist="Freetext Artist",
        barcode="",
    )

    res_cascade = client.post(
        "/lookup",
        json={
            "barcode": "9990002223334",
            "album_title": "__test_cascade_album",
            "album_artist": "Cascade Artist",
        },
    )
    assert res_cascade.status_code == 200
    body = res_cascade.json()
    assert body["release"]["discogs_release_id"] == 9900011
    assert body["status"] == "needs_review"
    assert any("barcode was not confirmed" in r.lower() for r in body["evaluation"]["review_reasons"])

    # Tier 3 freetext fallback when Tier 2 misses
    res_ft = client.post(
        "/lookup",
        json={
            "barcode": "",
            "album_title": "__test_freetext_album",
            "album_artist": "Freetext Artist",
        },
    )
    assert res_ft.status_code == 200
    assert res_ft.json()["evaluation"]["match_method"] == "freetext_q"


def test_404_ttl_cache_and_expiry(pg_client) -> None:
    client, provider, app = pg_client

    r1 = client.post(
        "/lookup",
        json={
            "barcode": "",
            "album_title": "__test_missing_ttl_album",
            "album_artist": "Missing Artist",
        },
    )
    assert r1.status_code == 404
    assert r1.json()["source"] == "discogs_live"
    input_id = r1.json()["input_id"]
    calls_after_first = len(provider.search_calls)

    # Repeat within TTL returns database_cache (0 new calls)
    r2 = client.post(
        "/lookup",
        json={
            "barcode": "",
            "album_title": "__test_missing_ttl_album",
            "album_artist": "Missing Artist",
        },
    )
    assert r2.status_code == 404
    assert r2.json()["source"] == "database_cache"
    assert len(provider.search_calls) == calls_after_first

    # Expire lookup_runs timestamp -> next lookup re-queries provider
    pool: ConnectionPool = app.state.pool
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE lookup_runs SET created_at = NOW() - INTERVAL '48 hours' WHERE input_id = %s",
                (input_id,),
            )
        conn.commit()

    r3 = client.post(
        "/lookup",
        json={
            "barcode": "",
            "album_title": "__test_missing_ttl_album",
            "album_artist": "Missing Artist",
        },
    )
    assert r3.status_code == 404
    assert r3.json()["source"] == "discogs_live"
    assert len(provider.search_calls) > calls_after_first


def test_offline_fallback_and_upstream_502_503_errors(pg_client) -> None:
    client, provider, _ = pg_client
    provider.search_responses["9990005556667"] = [{"id": 9900021}]
    provider.releases[9900021] = make_sample_release(
        release_id=9900021,
        title="__test_fallback_album",
        artist="Fallback Artist",
        barcode="9990005556667",
    )

    ok = client.post(
        "/lookup",
        json={
            "barcode": "9990005556667",
            "album_title": "__test_fallback_album",
            "album_artist": "Fallback Artist",
        },
    )
    assert ok.status_code == 200
    orig_checked = ok.json()["evaluation"]["candidates_checked"]

    # Rate limit on force_refresh serves fallback and preserves candidates_checked
    provider.fail_with = DiscogsRateLimitError("429 simulated")
    fb = client.post(
        "/lookup",
        json={
            "barcode": "9990005556667",
            "album_title": "__test_fallback_album",
            "album_artist": "Fallback Artist",
            "force_refresh": True,
        },
    )
    assert fb.status_code == 200
    assert fb.json()["source"] == "database_fallback"
    assert fb.json()["fallback_used"] is True
    assert fb.json()["evaluation"]["candidates_checked"] == orig_checked

    # Uncached record during 429 -> 503
    err_429 = client.post(
        "/lookup",
        json={"barcode": "", "album_title": "__test_uncached_429", "album_artist": "Artist"},
    )
    assert err_429.status_code == 503

    # Uncached record during 502/auth error -> 502
    provider.fail_with = DiscogsAuthError("Bad token", status_code=502)
    err_502 = client.post(
        "/lookup",
        json={"barcode": "", "album_title": "__test_uncached_502", "album_artist": "Artist"},
    )
    assert err_502.status_code == 502

    # Previously cached record still returns cleanly without force_refresh
    provider.fail_with = None
    still_ok = client.post(
        "/lookup",
        json={
            "barcode": "9990005556667",
            "album_title": "__test_fallback_album",
            "album_artist": "Fallback Artist",
        },
    )
    assert still_ok.status_code == 200
    assert still_ok.json()["source"] == "database"


def test_api_validation_404_routes_and_degraded_health(pg_client) -> None:
    client, _, app = pg_client

    assert client.post("/lookup", json={"barcode": "0", "album_title": ""}).status_code == 422
    assert client.get("/records/99999999").status_code == 404
    assert client.get("/releases/99999999").status_code == 404
    assert (
        client.post("/batch/csv", json={"csv_path": "/app/data/does_not_exist.csv"}).status_code
        == 404
    )

    orig_ping = app.state.repo.ping
    app.state.repo.ping = lambda: (_ for _ in ()).throw(RuntimeError("db down"))
    try:
        degraded = client.get("/health")
        assert degraded.status_code == 200
        assert degraded.json()["database_ok"] is False
        assert degraded.json()["status"] == "degraded"
    finally:
        app.state.repo.ping = orig_ping


def test_batch_csv_and_cli_end_to_end(
    pg_client, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client, provider, _ = pg_client
    provider.search_by_tier_or_key = [
        ({"barcode": "9990007778881"}, [{"id": 9900031}]),
        (
            {"release_title": "__test_batch_review", "artist": "Review Band"},
            [{"id": 9900032}, {"id": 9900033}],
        ),
    ]
    provider.releases[9900031] = make_sample_release(
        release_id=9900031,
        title="__test_batch_match",
        artist="Batch Artist",
        barcode="9990007778881",
    )
    provider.releases[9900032] = make_sample_release(
        release_id=9900032,
        title="__test_batch_review",
        artist="Review Band",
        barcode="",
        extra_track=False,
    )
    provider.releases[9900033] = make_sample_release(
        release_id=9900033,
        title="__test_batch_review",
        artist="Review Band",
        barcode="",
        extra_track=True,
    )

    csv_file = tmp_path / "batch.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["barcode", "album_title", "album_artist"])
        writer.writerow(["9990007778881", "__test_batch_match", "Batch Artist"])
        writer.writerow(["", "__test_batch_review", "Review Band"])
        writer.writerow(["", "__test_batch_missing", "Nobody"])
        writer.writerow(["0", "", "Invalid Row"])

    out_json = tmp_path / "out" / "batch_report.json"
    res = client.post(
        "/batch/csv",
        json={"csv_path": str(csv_file), "output_path": str(out_json)},
    )
    assert res.status_code == 200
    summary = res.json()
    assert summary["total_processed"] == 4
    assert summary["matched_count"] == 1
    assert summary["needs_review_count"] == 1
    assert summary["not_found_count"] == 1
    assert summary["error_count"] == 1
    assert summary["outcomes"][3]["status"] == "validation_error"
    assert out_json.exists()
    assert out_json.with_suffix(".md").exists()

    parsed_report = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(parsed_report["details"]) == 4

    cli_settings = Settings(
        database_url=get_settings().database_url,
        enable_agent_fallback=False,
    )

    # CLI end-to-end exit codes: 0 (match), 1 (404), 2 (upstream error), and batch subset
    code_ok = cli.main(
        [
            "lookup",
            "--barcode",
            "9990007778881",
            "--title",
            "__test_batch_match",
            "--artist",
            "Batch Artist",
        ],
        provider=provider,
        settings=cli_settings,
    )
    assert code_ok == 0
    capsys.readouterr()

    code_404 = cli.main(
        ["lookup", "--title", "__test_batch_missing", "--artist", "Nobody"],
        provider=provider,
        settings=cli_settings,
    )
    assert code_404 == 1
    capsys.readouterr()

    cli_out = tmp_path / "cli_report.json"
    code_batch = cli.main(
        ["batch", "--csv", str(csv_file), "--rows", "1,3", "--output", str(cli_out)],
        provider=provider,
        settings=cli_settings,
    )
    assert code_batch == 0
    assert cli_out.exists()
    capsys.readouterr()

    provider.fail_with = DiscogsUpstreamError("CLI simulated error")
    code_err = cli.main(
        ["lookup", "--title", "__test_cli_outage", "--artist", "Outage"],
        provider=provider,
        settings=cli_settings,
    )
    assert code_err == 2
    capsys.readouterr()
