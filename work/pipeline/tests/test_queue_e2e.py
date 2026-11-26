import csv
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg_pool import ConnectionPool

from app import cli
from app.agent_client import AgentAnswer, AgentTrack
from app.config import Settings, get_settings
from app.extractor import ExtractedRelease, extract_release
from app.fallback_providers import make_fallback_release_id
from app.main import create_app
from app.models import NormalizedInput
from app.repository import PipelineRepository
from tests.conftest import make_sample_release
from tests.test_end_to_end import FakeCatalogProvider

TEST_QUEUE_DB_NAME = "tracklist_test_queue"


class ControllableCatalogProvider(FakeCatalogProvider):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.blocked_titles: dict[str, tuple[threading.Event, threading.Event]] = {}

    def search_releases(
        self, params: dict[str, Any], bypass_cache: bool = False
    ) -> tuple[dict[str, Any], bool]:
        title = str(params.get("release_title") or "")
        if title in self.blocked_titles:
            started_evt, release_evt = self.blocked_titles[title]
            started_evt.set()
            assert release_evt.wait(timeout=5.0), f"Timed out waiting to release {title}"
        with self._lock:
            return super().search_releases(params, bypass_cache=bypass_cache)


class ControllableAgentProvider:
    def __init__(self, model_name: str = "gemini-queue-test") -> None:
        self._model_name = model_name
        self._lock = threading.Lock()
        self.calls: list[dict[str, str]] = []
        self.responses_by_title: dict[str, AgentAnswer | Exception] = {}
        self.hooks_by_title: dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    def research_album(self, barcode: str, album: str, artist: str) -> AgentAnswer:
        with self._lock:
            self.calls.append({"barcode": barcode, "album": album, "artist": artist})
        hook = self.hooks_by_title.get(album)
        if hook is not None:
            hook()
        outcome = self.responses_by_title.get(album, AgentAnswer())
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


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


def _truncate_test_db(db_url: str) -> None:
    with psycopg.connect(db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE TABLE queue_jobs, input_records, releases, api_cache RESTART IDENTITY CASCADE"
            )


@pytest.fixture(scope="module")
def queue_test_db_url():
    base_url = get_settings().database_url
    admin_url = _replace_db_name(base_url, "postgres")
    test_url = _replace_db_name(base_url, TEST_QUEUE_DB_NAME)

    _terminate_and_drop(admin_url, TEST_QUEUE_DB_NAME)
    with psycopg.connect(admin_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TEST_QUEUE_DB_NAME))
            )

    pool = ConnectionPool(conninfo=test_url, min_size=1, max_size=2, open=True)
    try:
        PipelineRepository(pool).ensure_schema()
    finally:
        pool.close()

    try:
        yield test_url
    finally:
        _terminate_and_drop(admin_url, TEST_QUEUE_DB_NAME)


def _wait_until(predicate, timeout: float = 8.0, interval: float = 0.05) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        val = predicate()
        if val:
            return val
        time.sleep(interval)
    raise AssertionError("Timed out waiting for async queue condition")


@pytest.fixture()
def pg_queue_client(queue_test_db_url: str):
    _truncate_test_db(queue_test_db_url)
    provider = ControllableCatalogProvider()
    agent = ControllableAgentProvider()
    settings = Settings(
        database_url=queue_test_db_url,
        enable_agent_fallback=True,
        gemini_api_key="test-gemini-key",
        enable_queue_workers=True,
        discogs_queue_concurrency=1,
        agent_queue_concurrency=2,
        queue_poll_interval_sec=0.05,
    )
    app = create_app(settings=settings, provider=provider, agent_provider=agent)
    with TestClient(app) as client:
        yield client, provider, agent, app
    _truncate_test_db(queue_test_db_url)


def test_two_stage_batch_queue_parallel_agent_workers_and_live_status(
    pg_queue_client, tmp_path: Path
) -> None:
    client, provider, agent, _ = pg_queue_client
    provider.search_by_tier_or_key = [
        ({"barcode": "9990009001111"}, [{"id": 9900301}]),
    ]
    provider.releases[9900301] = make_sample_release(
        release_id=9900301,
        title="__test_q_discogs_ok",
        artist="Discogs Band",
        barcode="9990009001111",
    )

    barrier = threading.Barrier(2, timeout=5.0)
    both_entered = threading.Event()
    release_event = threading.Event()
    entered_lock = threading.Lock()
    entered_count = 0

    def _parallel_hook() -> None:
        nonlocal entered_count
        barrier.wait()
        with entered_lock:
            entered_count += 1
            if entered_count == 2:
                both_entered.set()
        assert release_event.wait(timeout=5.0)

    agent.hooks_by_title["__test_q_agent_a"] = _parallel_hook
    agent.hooks_by_title["__test_q_agent_b"] = _parallel_hook
    agent.responses_by_title["__test_q_agent_a"] = AgentAnswer(
        tracks=[AgentTrack(disc=1, number=1, title="Agent Track A", length_sec=180)],
        sources=["https://musicbrainz.org/release/a"],
    )
    agent.responses_by_title["__test_q_agent_b"] = AgentAnswer(
        tracks=[AgentTrack(disc=1, number=1, title="Agent Track B", length_sec=200)],
        sources=["https://musicbrainz.org/release/b"],
    )
    agent.responses_by_title["__test_q_missing_all"] = AgentAnswer()

    csv_file = tmp_path / "queue_batch.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["barcode", "album_title", "album_artist"])
        writer.writerow(["9990009001111", "__test_q_discogs_ok", "Discogs Band"])
        writer.writerow(["", "__test_q_agent_a", "Agent A"])
        writer.writerow(["", "__test_q_agent_b", "Agent B"])
        writer.writerow(["", "__test_q_missing_all", "Ghost"])

    out_json = tmp_path / "queue_out" / "batch_report.json"
    enq = client.post(
        "/jobs/batch",
        json={"csv_path": str(csv_file), "output_path": str(out_json)},
    )
    assert enq.status_code == 202
    job_id = enq.json()["id"]
    assert enq.json()["total_items"] == 4

    # Both agent workers enter simultaneously -> proves parallel agent execution
    assert both_entered.wait(timeout=5.0)
    mid_status = client.get("/jobs/status").json()
    assert mid_status["workers_running"] is True
    assert mid_status["discogs_workers"] == 1
    assert mid_status["agent_workers"] == 2
    assert mid_status["counts"]["checking_agent"] == 2
    assert len(mid_status["active_items"]) >= 2

    release_event.set()

    def _done():
        d = client.get(f"/jobs/{job_id}").json()
        return d if d["job"]["status"] == "completed" else None

    detail = _wait_until(_done)
    job = detail["job"]
    assert job["completed_items"] == 4
    assert job["matched_count"] == 1
    assert job["needs_review_count"] == 2
    assert job["agent_recovered_count"] == 2
    assert job["not_found_count"] == 1
    assert job["error_count"] == 0
    assert job["progress_pct"] == 100.0
    assert out_json.exists()
    assert out_json.with_suffix(".md").exists()
    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    assert parsed["summary"]["matched_count"] == 1
    assert parsed["summary"]["needs_review_count"] == 2
    assert parsed["summary"]["not_found_count"] == 1


def test_agent_retry_single_duplicate_409_and_bulk_retry_not_found(
    pg_queue_client,
) -> None:
    client, provider, agent, _ = pg_queue_client

    # Seed two not_found records synchronously
    r1 = client.post(
        "/lookup",
        json={"barcode": "", "album_title": "__test_q_nf_1", "album_artist": "NF Artist 1"},
    )
    r2 = client.post(
        "/lookup",
        json={"barcode": "", "album_title": "__test_q_nf_2", "album_artist": "NF Artist 2"},
    )
    assert r1.status_code == 404
    assert r2.status_code == 404
    id1 = r1.json()["input_id"]
    id2 = r2.json()["input_id"]

    started_evt = threading.Event()
    release_evt = threading.Event()

    def _gate_nf1() -> None:
        started_evt.set()
        assert release_evt.wait(timeout=5.0)

    agent.hooks_by_title["__test_q_nf_1"] = _gate_nf1
    agent.responses_by_title["__test_q_nf_1"] = AgentAnswer(
        tracks=[AgentTrack(disc=1, number=1, title="Recovered NF 1", length_sec=195)],
        sources=["https://bandcamp.example/nf1"],
    )

    retry1 = client.post(f"/jobs/agent-retry/{id1}")
    assert retry1.status_code == 202
    retry_job_1 = retry1.json()["job"]["id"]
    assert started_evt.wait(timeout=5.0)

    # Duplicate enqueue while active returns 409 Conflict
    dup = client.post(f"/jobs/agent-retry/{id1}")
    assert dup.status_code == 409

    release_evt.set()
    _wait_until(lambda: client.get(f"/jobs/{retry_job_1}").json()["job"]["status"] == "completed")

    rec1_detail = client.get(f"/records/{id1}").json()
    assert rec1_detail["record"]["latest_status"] == "needs_review"
    assert rec1_detail["record"]["latest_source"] == "gemini_agent"
    assert rec1_detail["agent_fallback"] is not None

    # Bulk retry remaining not_found (id2) recovers via Discogs URL bridge
    provider.releases[9900302] = make_sample_release(
        release_id=9900302,
        title="__test_q_nf_2",
        artist="NF Artist 2",
        barcode="",
    )
    agent.responses_by_title["__test_q_nf_2"] = AgentAnswer(
        tracks=[AgentTrack(disc=1, number=1, title="Track One", length_sec=210)],
        sources=["https://www.discogs.com/release/9900302-NF-Artist-2"],
    )
    bulk = client.post("/jobs/retry-not-found")
    assert bulk.status_code == 202
    assert bulk.json()["total_items"] == 1
    bulk_id = bulk.json()["id"]

    bulk_done = _wait_until(
        lambda: (
            d
            if (d := client.get(f"/jobs/{bulk_id}").json())["job"]["status"] == "completed"
            else None
        )
    )
    assert bulk_done["job"]["agent_recovered_count"] == 1
    rec2_detail = client.get(f"/records/{id2}").json()
    assert rec2_detail["record"]["latest_source"] == "discogs_via_agent"
    assert rec2_detail["record"]["selected_release_id"] == 9900302

    # Second bulk retry sees 0 remaining not_found records
    bulk2 = client.post("/jobs/retry-not-found")
    assert bulk2.status_code == 202
    assert bulk2.json()["total_items"] == 0
    assert bulk2.json()["status"] == "completed"


def test_cancel_queued_job_prevents_pending_items_from_running(
    pg_queue_client, tmp_path: Path
) -> None:
    client, provider, _, _ = pg_queue_client
    first_started = threading.Event()
    first_release = threading.Event()
    provider.blocked_titles["__test_q_cancel_1"] = (first_started, first_release)
    provider.search_by_tier_or_key = [
        (
            {"release_title": "__test_q_cancel_1", "artist": "Artist 1"},
            [{"id": 9900309}],
        ),
    ]
    provider.releases[9900309] = make_sample_release(
        release_id=9900309,
        title="__test_q_cancel_1",
        artist="Artist 1",
        barcode="",
    )

    csv_file = tmp_path / "cancel.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["barcode", "album_title", "album_artist"])
        writer.writerow(["", "__test_q_cancel_1", "Artist 1"])
        writer.writerow(["", "__test_q_cancel_2", "Artist 2"])
        writer.writerow(["", "__test_q_cancel_3", "Artist 3"])
        writer.writerow(["", "__test_q_cancel_4", "Artist 4"])

    enq = client.post("/jobs/batch", json={"csv_path": str(csv_file), "output_path": ""})
    assert enq.status_code == 202
    job_id = enq.json()["id"]

    assert first_started.wait(timeout=5.0)
    cancel_res = client.post(f"/jobs/{job_id}/cancel")
    assert cancel_res.status_code == 200
    assert cancel_res.json()["status"] == "cancelled"

    first_release.set()

    detail = _wait_until(
        lambda: (
            d
            if (d := client.get(f"/jobs/{job_id}").json())["job"]["finished_at"] is not None
            else None
        )
    )
    assert detail["job"]["status"] == "cancelled"
    assert detail["job"]["cancelled_count"] == 3
    assert detail["job"]["completed_items"] == 4
    stages = [item["stage"] for item in detail["items"]]
    assert stages[1:] == ["cancelled", "cancelled", "cancelled"]
    searched_titles = [c["params"].get("release_title") for c in provider.search_calls]
    assert "__test_q_cancel_2" not in searched_titles
    assert "__test_q_cancel_3" not in searched_titles


def test_crash_recovery_resets_stale_checking_items(queue_test_db_url: str) -> None:
    _truncate_test_db(queue_test_db_url)

    settings = Settings(
        database_url=queue_test_db_url,
        enable_agent_fallback=True,
        gemini_api_key="test-gemini-key",
        enable_queue_workers=True,
        discogs_queue_concurrency=1,
        agent_queue_concurrency=1,
        queue_poll_interval_sec=0.05,
    )
    pool = ConnectionPool(conninfo=queue_test_db_url, min_size=1, max_size=2, open=True)
    repo = PipelineRepository(pool)

    job = repo.create_queue_job(
        job_type="batch_csv",
        label="Stale Recovery Test",
        items=[
            {
                "row_number": 1,
                "barcode": "",
                "album_title": "__test_q_stale_discogs",
                "album_artist": "Stale Band",
                "mode": "full",
                "stage": "checking_discogs",
                "status_message": "Simulated crash mid-discogs",
            },
            {
                "row_number": 2,
                "barcode": "",
                "album_title": "__test_q_stale_agent",
                "album_artist": "Stale Agent Band",
                "mode": "agent_only",
                "stage": "checking_agent",
                "status_message": "Simulated crash mid-agent",
            },
        ],
    )
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE queue_jobs SET status = 'running' WHERE id = %s", (job.id,)
            )
            cur.execute(
                "UPDATE queue_items SET worker_name = 'crashed-worker' WHERE job_id = %s",
                (job.id,),
            )
        conn.commit()
    pool.close()

    provider = ControllableCatalogProvider()
    provider.search_by_tier_or_key = [
        (
            {"release_title": "__test_q_stale_discogs", "artist": "Stale Band"},
            [{"id": 9900303}],
        ),
    ]
    provider.releases[9900303] = make_sample_release(
        release_id=9900303,
        title="__test_q_stale_discogs",
        artist="Stale Band",
        barcode="",
    )
    agent = ControllableAgentProvider()
    agent.responses_by_title["__test_q_stale_agent"] = AgentAnswer(
        tracks=[AgentTrack(disc=1, number=1, title="Recovered After Crash", length_sec=160)],
        sources=["https://recovered.test"],
    )

    try:
        app = create_app(settings=settings, provider=provider, agent_provider=agent)
        with TestClient(app) as client:
            detail = _wait_until(
                lambda: (
                    d
                    if (d := client.get(f"/jobs/{job.id}").json())["job"]["status"]
                    == "completed"
                    else None
                )
            )
            assert detail["job"]["completed_items"] == 2
            assert detail["job"]["matched_count"] == 1
            assert detail["job"]["needs_review_count"] == 1
            assert detail["job"]["agent_recovered_count"] == 1
    finally:
        _truncate_test_db(queue_test_db_url)


def test_queue_dashboard_html_and_error_routes(pg_queue_client) -> None:
    client, _, _, app = pg_queue_client

    # Seed a not_found record to verify Enqueue Agent buttons in /results and /results/{id}
    nf = client.post(
        "/lookup",
        json={"barcode": "", "album_title": "__test_q_ui_missing", "album_artist": "UI Ghost"},
    )
    assert nf.status_code == 404
    input_id = nf.json()["input_id"]

    # Enqueue a single full lookup via POST /jobs/lookup
    enq_single = client.post(
        "/jobs/lookup",
        json={
            "barcode": "",
            "album_title": "__test_q_single_ui",
            "album_artist": "Single UI Band",
            "mode": "full",
        },
    )
    assert enq_single.status_code == 202
    single_job_id = enq_single.json()["job"]["id"]
    _wait_until(
        lambda: client.get(f"/jobs/{single_job_id}").json()["job"]["status"] == "completed"
    )

    q_html = client.get("/queue")
    assert q_html.status_code == 200
    assert "Live Pipeline &amp; Agent Queue" in q_html.text
    assert "Active Workers" in q_html.text
    assert "Pending Queue" in q_html.text
    assert "Recent Jobs" in q_html.text
    assert "setInterval" in q_html.text
    assert "Retry All not_found with Agent" in q_html.text

    res_html = client.get("/results")
    assert res_html.status_code == 200
    assert "/queue" in res_html.text
    assert "queueBanner" in res_html.text
    assert f"enqueueAgent({input_id})" in res_html.text

    det_html = client.get(f"/results/{input_id}")
    assert det_html.status_code == 200
    assert "Enqueue Agent Fallback" in det_html.text

    # Error handling routes
    assert client.post("/jobs/agent-retry/99999999").status_code == 404
    assert (
        client.post("/jobs/batch", json={"csv_path": "/app/data/missing_queue.csv"}).status_code
        == 404
    )
    assert client.get("/jobs/99999999").status_code == 404
    assert client.post("/jobs/99999999/cancel").status_code == 404

    # When agent disabled, agent_only / retry-not-found return 400
    orig_flag = app.state.settings.enable_agent_fallback
    app.state.settings.enable_agent_fallback = False
    try:
        assert client.post("/jobs/retry-not-found").status_code == 400
        assert (
            client.post(
                "/jobs/lookup",
                json={
                    "barcode": "",
                    "album_title": "__test_q_disabled",
                    "album_artist": "Nobody",
                    "mode": "agent_only",
                },
            ).status_code
            == 400
        )
    finally:
        app.state.settings.enable_agent_fallback = orig_flag


def test_cli_enqueue_batch_and_retry_not_found(
    queue_test_db_url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _truncate_test_db(queue_test_db_url)

    csv_file = tmp_path / "cli_queue.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["barcode", "album_title", "album_artist"])
        writer.writerow(["", "__test_q_cli_1", "CLI Artist 1"])
        writer.writerow(["", "__test_q_cli_2", "CLI Artist 2"])

    cli_settings_disabled = Settings(
        database_url=queue_test_db_url,
        enable_agent_fallback=False,
        enable_queue_workers=False,
    )
    cli_settings_agent = Settings(
        database_url=queue_test_db_url,
        enable_agent_fallback=True,
        gemini_api_key="test-gemini-key",
        enable_queue_workers=False,
    )
    try:
        code = cli.main(
            [
                "enqueue-batch",
                "--csv",
                str(csv_file),
                "--output",
                str(tmp_path / "cli_q_out.json"),
            ],
            provider=FakeCatalogProvider(),
            settings=cli_settings_disabled,
        )
        assert code == 0
        out = capsys.readouterr().out
        job_payload = json.loads(out)
        assert job_payload["status"] == "queued"
        assert job_payload["total_items"] == 2

        # retry-not-found returns exit code 2 when agent disabled
        code_disabled = cli.main(
            ["retry-not-found"],
            provider=FakeCatalogProvider(),
            settings=cli_settings_disabled,
        )
        assert code_disabled == 2
        capsys.readouterr()

        # Seed a not_found record via CLI lookup and enqueue retry-not-found with agent enabled
        code_nf = cli.main(
            ["lookup", "--title", "__test_q_cli_nf", "--artist", "CLI NF"],
            provider=FakeCatalogProvider(),
            settings=cli_settings_disabled,
        )
        assert code_nf == 1
        capsys.readouterr()

        code_retry = cli.main(
            ["retry-not-found"],
            provider=FakeCatalogProvider(),
            agent_provider=ControllableAgentProvider(),
            settings=cli_settings_agent,
        )
        assert code_retry == 0
        retry_payload = json.loads(capsys.readouterr().out)
        assert retry_payload["job_type"] == "agent_retry"
        assert retry_payload["total_items"] == 1
        assert retry_payload["status"] == "queued"
    finally:
        _truncate_test_db(queue_test_db_url)


class _QueueFakeExternalFallback:
    name = "itunes"

    def __init__(self, releases_by_title: dict[str, tuple[ExtractedRelease, str]]) -> None:
        self.calls: list[str] = []
        self._releases = releases_by_title

    def lookup_candidates(
        self, norm: NormalizedInput, bypass_cache: bool = False
    ) -> tuple[list[tuple[ExtractedRelease, str]], list[dict[str, Any]], bool]:
        self.calls.append(norm.title_clean)
        hit = self._releases.get(norm.title_clean)
        queries = [
            {
                "tier": "fallback_itunes_test",
                "params": {"title": norm.title_clean},
                "result_count": 1 if hit else 0,
                "from_cache": False,
            }
        ]
        return ([hit] if hit else []), queries, False

    def close(self) -> None:
        return None


def test_queue_runs_external_fallback_between_discogs_and_agent(
    queue_test_db_url: str, tmp_path: Path
) -> None:
    _truncate_test_db(queue_test_db_url)
    provider = ControllableCatalogProvider()
    agent = ControllableAgentProvider()

    neg_id = make_fallback_release_id("itunes", "6798000134")
    raw_itunes = make_sample_release(
        release_id=neg_id,
        title="__test_q_ext_hit",
        artist="Queue Indie Artist",
        barcode="",
        notes="",
    )
    raw_itunes["_source_provider"] = "itunes"
    raw_itunes["uri"] = "https://music.apple.com/de/album/test/6798000134"
    ext_fb = _QueueFakeExternalFallback(
        {"__test_q_ext_hit": (extract_release(raw_itunes), "itunes_search")}
    )

    agent.responses_by_title["__test_q_ext_miss"] = AgentAnswer(
        tracks=[AgentTrack(disc=1, number=1, title="Agent Only Track", length_sec=160)],
        sources=["https://petzold.example/album"],
    )

    settings = Settings(
        database_url=queue_test_db_url,
        enable_agent_fallback=True,
        enable_external_fallbacks=True,
        gemini_api_key="test-gemini-key",
        enable_queue_workers=True,
        discogs_queue_concurrency=1,
        agent_queue_concurrency=1,
        queue_poll_interval_sec=0.05,
    )
    app = create_app(
        settings=settings,
        provider=provider,
        agent_provider=agent,
        fallback_providers=[ext_fb],
    )
    try:
        with TestClient(app) as client:
            csv_file = tmp_path / "q_mid_fallback.csv"
            with csv_file.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["barcode", "album_title", "album_artist"])
                writer.writerow(["", "__test_q_ext_hit", "Queue Indie Artist"])
                writer.writerow(["", "__test_q_ext_miss", "Queue Agent Artist"])

            out_json = tmp_path / "q_mid_report.json"
            enq = client.post(
                "/jobs/batch",
                json={"csv_path": str(csv_file), "output_path": str(out_json)},
            )
            assert enq.status_code == 202
            job_id = enq.json()["id"]

            detail = _wait_until(
                lambda: (
                    d
                    if (d := client.get(f"/jobs/{job_id}").json())["job"]["status"]
                    == "completed"
                    else None
                )
            )
            job = detail["job"]
            assert job["matched_count"] == 1
            assert job["needs_review_count"] == 1
            assert job["agent_recovered_count"] == 1
            assert len(ext_fb.calls) == 2
            assert len(agent.calls) == 1
            assert agent.calls[0]["album"] == "__test_q_ext_miss"

            items = detail["items"]
            assert items[0]["result_source"] == "itunes_live"
            assert items[0]["selected_release_id"] == neg_id
            assert items[0]["selected_agent_fallback_id"] is None
            assert items[1]["result_source"] == "gemini_agent"
            assert items[1]["selected_agent_fallback_id"] is not None

            report = json.loads(out_json.read_text(encoding="utf-8"))
            assert report["summary"]["outcomes"][0]["fallback_used"] is True
            assert report["summary"]["outcomes"][0]["source"] == "itunes_live"
    finally:
        _truncate_test_db(queue_test_db_url)


def test_restart_demo_resets_db_preserves_or_clears_cache_and_queues_csv(
    queue_test_db_url: str, tmp_path: Path
) -> None:
    _truncate_test_db(queue_test_db_url)
    provider = ControllableCatalogProvider()
    provider.search_by_tier_or_key = [
        ({"barcode": "9990006001111"}, [{"id": 9900601}]),
        ({"barcode": "9990006002222"}, [{"id": 9900602}]),
        ({"barcode": "9990006003333"}, [{"id": 9900603}]),
    ]
    provider.releases[9900601] = make_sample_release(
        release_id=9900601,
        title="__test_restart_old",
        artist="Old Band",
        barcode="9990006001111",
    )
    provider.releases[9900602] = make_sample_release(
        release_id=9900602,
        title="__test_restart_song_1",
        artist="New Band 1",
        barcode="9990006002222",
    )
    provider.releases[9900603] = make_sample_release(
        release_id=9900603,
        title="__test_restart_song_2",
        artist="New Band 2",
        barcode="9990006003333",
    )

    csv_file = tmp_path / "restart_demo.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["barcode", "album_title", "album_artist"])
        writer.writerow(["9990006002222", "__test_restart_song_1", "New Band 1"])
        writer.writerow(["9990006003333", "__test_restart_song_2", "New Band 2"])
    out_json = tmp_path / "restart_out.json"

    settings = Settings(
        database_url=queue_test_db_url,
        enable_agent_fallback=False,
        enable_queue_workers=True,
        discogs_queue_concurrency=1,
        queue_poll_interval_sec=0.05,
        enable_idle_auto_enrichment=False,
    )
    app = create_app(settings=settings, provider=provider)
    try:
        with TestClient(app) as client:
            # 1. Seed initial record, completed queue job, and api_cache entry
            seed = client.post(
                "/jobs/lookup",
                json={
                    "barcode": "9990006001111",
                    "album_title": "__test_restart_old",
                    "album_artist": "Old Band",
                    "mode": "full",
                },
            )
            assert seed.status_code == 202
            seed_jid = seed.json()["job"]["id"]
            _wait_until(
                lambda: client.get(f"/jobs/{seed_jid}").json()["job"]["status"]
                == "completed"
            )
            app.state.repo.set_api_cache(
                "restart-sentinel-key", "/sentinel", 200, {"ok": True}
            )

            # Missing CSV must fail 404 before wiping DB or cache
            missing_res = client.post(
                "/jobs/restart-demo",
                json={"csv_path": str(tmp_path / "does_not_exist.csv")},
            )
            assert missing_res.status_code == 404
            assert len(client.get("/records").json()) == 1
            assert app.state.repo.get_api_cache("restart-sentinel-key") is not None

            # Verify UI renders Restart controls on both /results and /queue
            res_html = client.get("/results").text
            queue_html = client.get("/queue").text
            for html_text in (res_html, queue_html):
                assert "Restart 25 Songs" in html_text
                assert "restartDemo()" in html_text
                assert "restartClearCache" in html_text

            # 2. Restart preserving api_cache (default demo mode)
            restart_keep = client.post(
                "/jobs/restart-demo",
                json={
                    "csv_path": str(csv_file),
                    "output_path": str(out_json),
                    "clear_api_cache": False,
                },
            )
            assert restart_keep.status_code == 202
            keep_job = restart_keep.json()
            assert keep_job["id"] == 1
            assert keep_job["total_items"] == 2

            done_keep = _wait_until(
                lambda: (
                    d
                    if (d := client.get("/jobs/1").json())["job"]["status"]
                    == "completed"
                    else None
                )
            )
            assert done_keep["job"]["completed_items"] == 2
            assert done_keep["job"]["matched_count"] == 2
            assert done_keep["job"]["error_count"] == 0
            assert out_json.exists()

            recs_after = client.get("/records").json()
            titles = {r["album_title_raw"] for r in recs_after}
            assert titles == {"__test_restart_song_1", "__test_restart_song_2"}
            assert "__test_restart_old" not in titles
            assert app.state.repo.get_api_cache("restart-sentinel-key") is not None

            # 3. Restart with clear_api_cache=True wipes api_cache too
            restart_wipe = client.post(
                "/jobs/restart-demo",
                json={
                    "csv_path": str(csv_file),
                    "output_path": str(out_json),
                    "clear_api_cache": True,
                },
            )
            assert restart_wipe.status_code == 202
            assert restart_wipe.json()["id"] == 1
            done_wipe = _wait_until(
                lambda: (
                    d
                    if (d := client.get("/jobs/1").json())["job"]["status"]
                    == "completed"
                    else None
                )
            )
            assert done_wipe["job"]["completed_items"] == 2
            assert app.state.repo.get_api_cache("restart-sentinel-key") is None
    finally:
        _truncate_test_db(queue_test_db_url)
