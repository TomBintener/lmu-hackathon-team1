import csv
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool

from app import cli
from app.agent_client import AgentAnswer, AgentFallbackError, AgentTrack
from app.config import Settings, get_settings
from app.discogs_client import DiscogsRateLimitError
from app.extractor import ExtractedRelease, extract_release
from app.fallback_providers import make_fallback_release_id
from app.main import create_app
from app.models import NormalizedInput
from tests.conftest import make_sample_release
from tests.test_end_to_end import FakeCatalogProvider, _cleanup_test_rows, _get_cache_keys


class FakeAgentProvider:
    def __init__(self, model_name: str = "gemini-test-model") -> None:
        self._model_name = model_name
        self.calls: list[dict[str, str]] = []
        self.responses_by_title: dict[str, AgentAnswer | Exception] = {}
        self.default_answer: AgentAnswer | Exception = AgentAnswer()

    @property
    def model_name(self) -> str:
        return self._model_name

    def research_album(self, barcode: str, album: str, artist: str) -> AgentAnswer:
        self.calls.append({"barcode": barcode, "album": album, "artist": artist})
        outcome = self.responses_by_title.get(album, self.default_answer)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture()
def pg_agent_client():
    baseline_keys = _get_cache_keys()
    _cleanup_test_rows(baseline_keys)
    provider = FakeCatalogProvider()
    agent = FakeAgentProvider()
    settings = Settings(
        database_url=get_settings().database_url,
        enable_agent_fallback=True,
        gemini_api_key="test-gemini-key",
        enable_queue_workers=False,
    )
    app = create_app(settings=settings, provider=provider, agent_provider=agent)
    with TestClient(app) as client:
        yield client, provider, agent, app
    _cleanup_test_rows(baseline_keys)


def test_agent_fallback_option_a_and_db_cache_hit(pg_agent_client) -> None:
    client, _, agent, app = pg_agent_client
    agent.responses_by_title["__test_agent_only_album"] = AgentAnswer(
        tracks=[
            AgentTrack(disc=1, number=1, title="Morning Light", length_sec=215),
            AgentTrack(disc=2, number=1, title="Evening Echo", length_sec=None),
        ],
        sources=["https://musicbrainz.org/release/11111111-1111-1111-1111-111111111111"],
    )

    r1 = client.post(
        "/lookup",
        json={
            "barcode": "9990008881112",
            "album_title": "__test_agent_only_album",
            "album_artist": "Agent Band",
        },
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["status"] == "needs_review"
    assert body1["source"] == "gemini_agent"
    assert body1["fallback_used"] is True
    assert body1["release"] is None
    assert body1["agent_result"] is not None
    fb_id = body1["agent_result"]["agent_fallback_id"]
    assert body1["agent_result"]["model_name"] == "gemini-test-model"
    assert body1["agent_result"]["track_count"] == 2
    assert len(body1["tracks"]) == 2
    assert body1["tracks"][0]["duration_raw"] == "3:35"
    assert body1["tracks"][0]["has_valid_duration"] is True
    assert body1["tracks"][1]["position"] == "2-1"
    assert body1["tracks"][1]["disc_label"] == "Disc 2"
    assert body1["evaluation"]["duration_coverage"] == 0.5
    assert body1["evaluation"]["match_method"] == "gemini_agent_search"
    assert len(agent.calls) == 1

    # Verify Option A isolation: releases stays strictly Discogs-only, agent tables populated
    pool: ConnectionPool = app.state.pool
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM releases WHERE title = '__test_agent_only_album'")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT COUNT(*) FROM agent_fallbacks WHERE id = %s", (fb_id,))
            assert cur.fetchone()[0] == 1
            cur.execute(
                "SELECT COUNT(*) FROM agent_fallback_tracks WHERE agent_fallback_id = %s",
                (fb_id,),
            )
            assert cur.fetchone()[0] == 2

    # Repeat lookup uses Layer 1 DB cache without invoking agent again
    r2 = client.post(
        "/lookup",
        json={
            "barcode": "9990008881112",
            "album_title": "__test_agent_only_album",
            "album_artist": "Agent Band",
        },
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["source"] == "database"
    assert body2["fallback_used"] is True
    assert body2["release"] is None
    assert body2["agent_result"]["agent_fallback_id"] == fb_id
    assert len(body2["tracks"]) == 2
    assert len(agent.calls) == 1


def test_agent_discogs_url_bridge_positive_and_rejected(pg_agent_client) -> None:
    client, provider, agent, app = pg_agent_client

    # Positive bridge: agent cites a valid Discogs release URL -> recovered into releases
    provider.releases[9900101] = make_sample_release(
        release_id=9900101,
        title="__test_bridge_album",
        artist="Bridge Artist",
        barcode="9990008882225",
    )
    agent.responses_by_title["__test_bridge_album"] = AgentAnswer(
        tracks=[AgentTrack(disc=1, number=1, title="Track One", length_sec=210)],
        sources=["https://www.discogs.com/de/release/9900101-Bridge-Artist-Test"],
    )

    res_pos = client.post(
        "/lookup",
        json={
            "barcode": "9990008882225",
            "album_title": "__test_bridge_album",
            "album_artist": "Bridge Artist",
        },
    )
    assert res_pos.status_code == 200
    b_pos = res_pos.json()
    assert b_pos["source"] == "discogs_via_agent"
    assert b_pos["fallback_used"] is True
    assert b_pos["agent_result"] is None
    assert b_pos["release"]["discogs_release_id"] == 9900101
    assert len(b_pos["tracks"]) == 4

    pool: ConnectionPool = app.state.pool
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM agent_fallbacks WHERE input_id = %s",
                (b_pos["input_id"],),
            )
            assert cur.fetchone()[0] == 0

    # Negative bridge: cited Discogs release fails evaluation -> degrades cleanly to Option A
    provider.releases[9900102] = make_sample_release(
        release_id=9900102,
        title="Completely Unrelated Album",
        artist="Someone Else",
        barcode="0001112223334",
    )
    agent.responses_by_title["__test_bridge_reject"] = AgentAnswer(
        tracks=[AgentTrack(disc=1, number=1, title="Real Song", length_sec=190)],
        sources=["https://www.discogs.com/release/9900102-Wrong"],
    )
    res_neg = client.post(
        "/lookup",
        json={
            "barcode": "",
            "album_title": "__test_bridge_reject",
            "album_artist": "Target Artist",
        },
    )
    assert res_neg.status_code == 200
    b_neg = res_neg.json()
    assert b_neg["source"] == "gemini_agent"
    assert b_neg["release"] is None
    assert b_neg["agent_result"] is not None
    assert len(b_neg["evaluation"]["rejection_reasons"]) >= 1


def test_discogs_error_agent_fallback_and_double_failure_503(pg_agent_client) -> None:
    client, provider, agent, _ = pg_agent_client
    provider.fail_with = DiscogsRateLimitError("429 simulated outage")

    # 1. Discogs 429 + Agent succeeds -> 200 gemini_agent_fallback (bridge skipped)
    agent.responses_by_title["__test_outage_recovered"] = AgentAnswer(
        tracks=[AgentTrack(disc=1, number=1, title="Outage Song", length_sec=200)],
        sources=["https://www.discogs.com/release/9900199-Ignored-During-Outage"],
    )
    ok = client.post(
        "/lookup",
        json={
            "barcode": "",
            "album_title": "__test_outage_recovered",
            "album_artist": "Outage Band",
        },
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["source"] == "gemini_agent_fallback"
    assert body["status"] == "needs_review"
    assert body["fallback_used"] is True
    assert "429 simulated outage" in (body["warning"] or "")
    assert body["agent_result"] is not None
    assert provider.release_calls == []

    # 2. Discogs 429 + Agent returns empty -> preserves 503
    agent.responses_by_title["__test_outage_empty"] = AgentAnswer()
    err1 = client.post(
        "/lookup",
        json={"barcode": "", "album_title": "__test_outage_empty", "album_artist": "Nobody"},
    )
    assert err1.status_code == 503

    # 3. Discogs 429 + Agent raises AgentFallbackError -> preserves 503
    agent.responses_by_title["__test_outage_err"] = AgentFallbackError("Gemini quota hit")
    err2 = client.post(
        "/lookup",
        json={"barcode": "", "album_title": "__test_outage_err", "album_artist": "Nobody"},
    )
    assert err2.status_code == 503


def test_discogs_not_found_and_agent_empty_or_error_returns_404_ttl_cached(
    pg_agent_client,
) -> None:
    client, _, agent, _ = pg_agent_client
    agent.responses_by_title["__test_both_missing"] = AgentAnswer()

    r1 = client.post(
        "/lookup",
        json={"barcode": "", "album_title": "__test_both_missing", "album_artist": "Ghost"},
    )
    assert r1.status_code == 404
    assert r1.json()["source"] == "discogs_live"
    assert any(q.get("tier") == "agent_fallback" for q in r1.json()["queries_attempted"])
    calls_after_first = len(agent.calls)

    # Repeat within TTL hits database_cache without calling agent again
    r2 = client.post(
        "/lookup",
        json={"barcode": "", "album_title": "__test_both_missing", "album_artist": "Ghost"},
    )
    assert r2.status_code == 404
    assert r2.json()["source"] == "database_cache"
    assert len(agent.calls) == calls_after_first

    # Agent error during Discogs not_found also degrades cleanly to 404
    agent.responses_by_title["__test_agent_boom_404"] = AgentFallbackError("Boom")
    r3 = client.post(
        "/lookup",
        json={"barcode": "", "album_title": "__test_agent_boom_404", "album_artist": "Ghost"},
    )
    assert r3.status_code == 404
    assert any("Boom" in str(q.get("error", "")) for q in r3.json()["queries_attempted"])


def test_dashboard_records_and_review_workflow_for_agent_fallback(pg_agent_client) -> None:
    client, _, agent, _ = pg_agent_client
    agent.responses_by_title["__test_review_agent_album"] = AgentAnswer(
        tracks=[
            AgentTrack(disc=1, number=1, title="First Cut", length_sec=185),
            AgentTrack(disc=1, number=2, title="Second Cut", length_sec=240),
        ],
        sources=[
            "https://musicbrainz.org/release/example",
            "javascript:alert(1)",
        ],
    )

    res = client.post(
        "/lookup",
        json={
            "barcode": "",
            "album_title": "__test_review_agent_album",
            "album_artist": "Review Agent Artist",
        },
    )
    assert res.status_code == 200
    input_id = res.json()["input_id"]
    fb_id = res.json()["agent_result"]["agent_fallback_id"]

    # /records and /records/{id}
    records = client.get("/records?status=needs_review").json()
    matching = [r for r in records if r["input_id"] == input_id]
    assert len(matching) == 1
    assert matching[0]["selected_agent_fallback_id"] == fb_id
    assert matching[0]["agent_source_url"] == "https://musicbrainz.org/release/example"

    detail = client.get(f"/records/{input_id}").json()
    assert detail["selected_release"] is None
    assert detail["agent_fallback"]["agent_fallback_id"] == fb_id
    assert len(detail["tracks"]) == 2

    # HTML dashboard renders Agent card, safe link only, and enabled Confirm Match button
    html_list = client.get("/results").text
    assert f"Agent #{fb_id}" in html_list

    html_detail = client.get(f"/results/{input_id}").text
    assert f"Agent #{fb_id}" in html_detail
    assert "gemini-test-model" in html_detail
    assert 'href="https://musicbrainz.org/release/example"' in html_detail
    assert 'href="javascript:alert(1)"' not in html_detail
    assert "<span>javascript:alert(1)</span>" in html_detail
    assert "onclick=\"submitReview('confirm', null)\" >Confirm Match" in html_detail

    # Confirm agent fallback -> promotes to matched and preserves duration_coverage on DB lookup
    rev_confirm = client.post(
        f"/records/{input_id}/review",
        json={"decision": "confirm", "reviewer_type": "human", "reviewer_name": "Alice"},
    )
    assert rev_confirm.status_code == 200
    assert rev_confirm.json()["selected_agent_fallback_id"] == fb_id

    cached_after_confirm = client.post(
        "/lookup",
        json={
            "barcode": "",
            "album_title": "__test_review_agent_album",
            "album_artist": "Review Agent Artist",
        },
    )
    assert cached_after_confirm.status_code == 200
    c_body = cached_after_confirm.json()
    assert c_body["status"] == "matched"
    assert c_body["source"] == "database"
    assert c_body["agent_result"]["agent_fallback_id"] == fb_id
    assert c_body["evaluation"]["duration_coverage"] == 1.0

    # Deny clears selection and puts record into rejected
    rev_deny = client.post(
        f"/records/{input_id}/review",
        json={"decision": "deny", "reviewer_type": "human", "note": "Wrong edition"},
    )
    assert rev_deny.status_code == 200
    detail_after_deny = client.get(f"/records/{input_id}").json()
    assert detail_after_deny["record"]["latest_status"] == "rejected"
    assert detail_after_deny["agent_fallback"] is None


def test_batch_csv_and_cli_mixed_discogs_and_agent_fallback(
    pg_agent_client, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client, provider, agent, _ = pg_agent_client
    provider.search_by_tier_or_key = [
        ({"barcode": "9990008883331"}, [{"id": 9900201}]),
    ]
    provider.releases[9900201] = make_sample_release(
        release_id=9900201,
        title="__test_mix_discogs",
        artist="Discogs Artist",
        barcode="9990008883331",
    )
    agent.responses_by_title["__test_mix_agent"] = AgentAnswer(
        tracks=[AgentTrack(disc=1, number=1, title="Agent Only Track", length_sec=150)],
        sources=["https://label.example/release"],
    )
    agent.responses_by_title["__test_mix_missing"] = AgentAnswer()

    csv_file = tmp_path / "mix.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["barcode", "album_title", "album_artist"])
        writer.writerow(["9990008883331", "__test_mix_discogs", "Discogs Artist"])
        writer.writerow(["", "__test_mix_agent", "Agent Artist"])
        writer.writerow(["", "__test_mix_missing", "Nobody"])

    out_json = tmp_path / "mix_report.json"
    res = client.post(
        "/batch/csv",
        json={"csv_path": str(csv_file), "output_path": str(out_json)},
    )
    assert res.status_code == 200
    summary = res.json()
    assert summary["total_processed"] == 3
    assert summary["matched_count"] == 1
    assert summary["needs_review_count"] == 1
    assert summary["not_found_count"] == 1
    assert summary["error_count"] == 0

    agent_outcome = summary["outcomes"][1]
    assert agent_outcome["source"] == "gemini_agent"
    assert agent_outcome["discogs_release_id"] is None
    assert agent_outcome["agent_fallback_id"] is not None
    assert agent_outcome["release_url"] == "https://label.example/release"
    assert agent_outcome["agent_sources"] == ["https://label.example/release"]

    parsed_report = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(parsed_report["details"]) == 3
    assert out_json.with_suffix(".md").exists()

    cli_settings = Settings(
        database_url=get_settings().database_url,
        enable_agent_fallback=True,
        gemini_api_key="test-gemini-key",
    )
    code_agent_ok = cli.main(
        ["lookup", "--title", "__test_mix_agent", "--artist", "Agent Artist"],
        provider=provider,
        agent_provider=agent,
        settings=cli_settings,
    )
    assert code_agent_ok == 0
    capsys.readouterr()

    code_404 = cli.main(
        ["lookup", "--title", "__test_mix_missing", "--artist", "Nobody"],
        provider=provider,
        agent_provider=agent,
        settings=cli_settings,
    )
    assert code_404 == 1
    capsys.readouterr()


class FakeExternalFallbackProvider:
    def __init__(self, name: str = "jpc") -> None:
        self.name = name
        self.calls: list[str] = []
        self.releases_by_title: dict[str, tuple[ExtractedRelease, str]] = {}

    def lookup_candidates(
        self, norm: NormalizedInput, bypass_cache: bool = False
    ) -> tuple[list[tuple[ExtractedRelease, str]], list[dict[str, Any]], bool]:
        self.calls.append(norm.title_clean)
        entry = self.releases_by_title.get(norm.title_clean)
        queries = [
            {
                "tier": f"fallback_{self.name}_test",
                "params": {"title": norm.title_clean},
                "result_count": 1 if entry else 0,
                "from_cache": False,
            }
        ]
        return ([entry] if entry else []), queries, False

    def close(self) -> None:
        return None


def test_external_fallback_sits_between_discogs_and_gemini_agent() -> None:
    baseline_keys = _get_cache_keys()
    _cleanup_test_rows(baseline_keys)
    provider = FakeCatalogProvider()
    agent = FakeAgentProvider()
    ext_fb = FakeExternalFallbackProvider(name="jpc")

    neg_id = make_fallback_release_id("jpc", "test-jpc-1")
    raw_fb = make_sample_release(
        release_id=neg_id,
        title="__test_ext_fallback_hit",
        artist="Fallback Band",
        barcode="9990007771111",
    )
    raw_fb["_source_provider"] = "jpc"
    raw_fb["uri"] = "https://www.jpc.de/jpcng/poprock/detail/-/art/test/hnum/1"
    ext_fb.releases_by_title["__test_ext_fallback_hit"] = (extract_release(raw_fb), "jpc_ean")

    agent.responses_by_title["__test_ext_fallback_miss"] = AgentAnswer(
        tracks=[AgentTrack(disc=1, number=1, title="Agent Rescued Track", length_sec=190)],
        sources=["https://petzold.example/release"],
    )

    settings = Settings(
        database_url=get_settings().database_url,
        enable_agent_fallback=True,
        enable_external_fallbacks=True,
        gemini_api_key="test-gemini-key",
        enable_queue_workers=False,
    )
    app = create_app(
        settings=settings,
        provider=provider,
        agent_provider=agent,
        fallback_providers=[ext_fb],
    )
    try:
        with TestClient(app) as client:
            # Case A: Discogs misses -> JPC fallback resolves -> Gemini Agent NOT called
            r1 = client.post(
                "/lookup",
                json={
                    "barcode": "9990007771111",
                    "album_title": "__test_ext_fallback_hit",
                    "album_artist": "Fallback Band",
                },
            )
            assert r1.status_code == 200
            b1 = r1.json()
            assert b1["status"] == "matched"
            assert b1["source"] == "jpc_live"
            assert b1["fallback_used"] is True
            assert b1["release"]["discogs_release_id"] == neg_id
            assert b1["agent_result"] is None
            assert len(ext_fb.calls) == 1
            assert len(agent.calls) == 0

            # Second lookup hits Layer 1 PostgreSQL DB cache
            r1_cached = client.post(
                "/lookup",
                json={
                    "barcode": "9990007771111",
                    "album_title": "__test_ext_fallback_hit",
                    "album_artist": "Fallback Band",
                },
            )
            assert r1_cached.status_code == 200
            assert r1_cached.json()["source"] == "database"
            assert len(agent.calls) == 0

            # Dashboard renders friendly JPC label instead of negative ID
            dash_html = client.get("/results").text
            assert ">JPC<" in dash_html

            # Case B: Discogs & external fallback miss -> escalates to Gemini Agent
            r2 = client.post(
                "/lookup",
                json={
                    "barcode": "",
                    "album_title": "__test_ext_fallback_miss",
                    "album_artist": "Rare Band",
                },
            )
            assert r2.status_code == 200
            b2 = r2.json()
            assert b2["source"] == "gemini_agent"
            assert b2["agent_result"] is not None
            assert len(agent.calls) == 1
            detail = client.get(f"/records/{b2['input_id']}").json()
            tiers = [q.get("tier") for q in detail["runs"][0]["queries_attempted_json"]]
            assert "fallback_jpc_test" in tiers
            assert "agent_fallback" in tiers
    finally:
        _cleanup_test_rows(baseline_keys)
