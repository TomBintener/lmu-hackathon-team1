import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app.agent_client import (
    AgentFallbackError,
    AgentTrack,
    AzureOpenAIAgentClient,
    GeminiAgentClient,
    _normalize_azure_responses_url,
    build_agent_client,
    dedupe_and_sort_tracks,
    extract_discogs_release_ids,
    grounding_urls,
    parse_answer,
)
from app.config import Settings


class FakeModelsAPI:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        next_item = self._responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


class FakeGenaiClient:
    def __init__(self, responses: list[Any]) -> None:
        self.models = FakeModelsAPI(responses)


def _make_response(text: str, uris: list[str] | None = None) -> Any:
    chunks = [SimpleNamespace(web=SimpleNamespace(uri=u)) for u in (uris or [])]
    candidate = SimpleNamespace(
        grounding_metadata=SimpleNamespace(grounding_chunks=chunks)
    )
    return SimpleNamespace(text=text, candidates=[candidate])


def test_parse_answer_raw_fenced_prose_and_errors() -> None:
    raw = '{"tracks":[{"disc":1,"number":1,"title":"Intro","length_sec":95}],"sources":["https://musicbrainz.org/1"]}'
    parsed_raw = parse_answer(raw)
    assert len(parsed_raw.tracks) == 1
    assert parsed_raw.tracks[0].title == "Intro"
    assert parsed_raw.tracks[0].length_sec == 95

    fenced = f"Here you go:\n```json\n{raw}\n```\nDone!"
    parsed_fenced = parse_answer(fenced)
    assert parsed_fenced.sources == ["https://musicbrainz.org/1"]

    with pytest.raises(ValueError):
        parse_answer("I could not find any JSON here.")

    with pytest.raises(ValueError):
        parse_answer("{not valid json}")


def test_extract_discogs_release_ids_and_dedupe() -> None:
    sources = [
        "https://musicbrainz.org/release/abc",
        "https://www.discogs.com/release/123456-Artist-Album",
        "http://discogs.com/de/release/98765-Another-Slug",
        "https://www.discogs.com/master/55555",
        "https://www.discogs.com/release/123456",
        "",
    ]
    assert extract_discogs_release_ids(sources) == [123456, 98765]


def test_dedupe_and_sort_tracks_and_grounding_urls() -> None:
    tracks = [
        AgentTrack(disc=2, number=1, title="Disc 2 First", length_sec=200),
        AgentTrack(disc=1, number=2, title="Second", length_sec=180),
        AgentTrack(disc=1, number=1, title="First", length_sec=120),
        AgentTrack(disc=1, number=1, title="Duplicate First", length_sec=999),
    ]
    cleaned = dedupe_and_sort_tracks(tracks)
    assert [(t.disc, t.number, t.title) for t in cleaned] == [
        (1, 1, "First"),
        (1, 2, "Second"),
        (2, 1, "Disc 2 First"),
    ]

    resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                grounding_metadata=SimpleNamespace(
                    grounding_chunks=[
                        SimpleNamespace(web=SimpleNamespace(uri="https://a.example")),
                        SimpleNamespace(web=None),
                        SimpleNamespace(web=SimpleNamespace(uri="https://a.example")),
                        SimpleNamespace(web=SimpleNamespace(uri="https://b.example")),
                    ]
                )
            ),
            SimpleNamespace(grounding_metadata=None),
        ]
    )
    assert grounding_urls(resp) == ["https://a.example", "https://b.example"]
    assert grounding_urls(SimpleNamespace(candidates=None)) == []


def test_gemini_agent_client_happy_path_grounding_and_barcode_zero() -> None:
    json_without_sources = (
        '{"tracks":['
        '{"disc":1,"number":2,"title":"Track B","length_sec":null},'
        '{"disc":1,"number":1,"title":"Track A","length_sec":210}'
        '],"sources":[]}'
    )
    fake_genai = FakeGenaiClient(
        [_make_response(json_without_sources, uris=["https://source-from-grounding.test/album"])]
    )
    settings = Settings(google_genai_model="gemini-3.8-flash", gemini_api_key="real-key")
    client = GeminiAgentClient(settings, genai_client=fake_genai)

    ans = client.research_album("0", "Test Album", "Test Artist")
    assert client.model_name == "gemini-3.8-flash"
    assert [t.title for t in ans.tracks] == ["Track A", "Track B"]
    assert ans.sources == ["https://source-from-grounding.test/album"]
    sent_prompt = fake_genai.models.calls[0]["contents"]
    assert "barcode (EAN/UPC): unknown" in sent_prompt


def test_gemini_agent_client_restructure_empty_retry_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.agent_client.time.sleep", lambda _: None)

    # 1. Free-text answer triggers _restructure() fallback
    fake_restructure = FakeGenaiClient(
        [
            _make_response("1. Alpha (3:00)\n2. Beta"),
            _make_response(
                '{"tracks":[{"disc":1,"number":1,"title":"Alpha","length_sec":180}],'
                '"sources":["https://restructured.test"]}'
            ),
        ]
    )
    c1 = GeminiAgentClient(Settings(agent_max_attempts=2), genai_client=fake_restructure)
    ans1 = c1.research_album("1234567890123", "Album", "Artist")
    assert len(fake_restructure.models.calls) == 2
    assert ans1.tracks[0].title == "Alpha"
    assert ans1.sources == ["https://restructured.test"]

    # 2. Empty tracks returned -> empty AgentAnswer
    fake_empty = FakeGenaiClient(
        [_make_response('{"tracks":[],"sources":["https://ignored.test"]}')]
    )
    c2 = GeminiAgentClient(Settings(agent_max_attempts=1), genai_client=fake_empty)
    ans2 = c2.research_album("", "Nothing", "Nobody")
    assert ans2.tracks == []
    assert ans2.sources == []

    # 3. Transient failure followed by recovery on retry
    fake_retry = FakeGenaiClient(
        [
            RuntimeError("429 temporary"),
            _make_response(
                '{"tracks":[{"disc":1,"number":1,"title":"Recovered","length_sec":100}],'
                '"sources":["https://ok.test"]}'
            ),
        ]
    )
    c3 = GeminiAgentClient(Settings(agent_max_attempts=2), genai_client=fake_retry)
    ans3 = c3.research_album("", "Retry Album", "Retry Artist")
    assert [t.title for t in ans3.tracks] == ["Recovered"]

    # 4. Persistent failure raises AgentFallbackError
    fake_fail = FakeGenaiClient([RuntimeError("err1"), RuntimeError("err2")])
    c4 = GeminiAgentClient(Settings(agent_max_attempts=2), genai_client=fake_fail)
    with pytest.raises(AgentFallbackError):
        c4.research_album("", "Fail Album", "Fail Artist")


def test_gemini_agent_client_requires_configured_key() -> None:
    c = GeminiAgentClient(Settings(gemini_api_key=""))
    with pytest.raises(AgentFallbackError):
        c.research_album("", "Album", "Artist")


def test_gemini_agent_client_extracts_token_usage_and_cost() -> None:
    usage = SimpleNamespace(
        prompt_token_count=1000,
        candidates_token_count=400,
        thoughts_token_count=100,
    )
    resp = SimpleNamespace(
        text='{"tracks": [{"disc": 1, "number": 1, "title": "Song A", "length_sec": 180}], "sources": ["https://example.com"]}',
        candidates=[],
        usage_metadata=usage,
    )
    client = GeminiAgentClient(Settings(), genai_client=FakeGenaiClient([resp]))
    ans = client.research_album("", "Album", "Artist")
    assert ans.input_tokens == 1000
    assert ans.output_tokens == 500
    # (1000 / 1_000_000) * 0.75 + (500 / 1_000_000) * 3.75 = 0.00075 + 0.001875 = 0.002625
    assert ans.cost_usd == pytest.approx(0.002625)


def test_normalize_azure_responses_url() -> None:
    assert (
        _normalize_azure_responses_url(
            "https://example.services.ai.azure.com/openai/v1/responses"
        )
        == "https://example.services.ai.azure.com/openai/v1/responses"
    )
    assert (
        _normalize_azure_responses_url("https://example.services.ai.azure.com/openai/v1/")
        == "https://example.services.ai.azure.com/openai/v1/responses"
    )
    assert (
        _normalize_azure_responses_url("https://example.services.ai.azure.com/")
        == "https://example.services.ai.azure.com/openai/v1/responses"
    )


def _azure_mock_client(
    handler: Any,
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_azure_openai_agent_client_happy_path_citations_and_cost() -> None:
    recorded: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        recorded.append(
            {
                "url": str(request.url),
                "api_key": request.headers.get("api-key"),
                "body": body,
            }
        )
        payload = {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [{"url": "https://www.discogs.com/release/12345"}],
                    },
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                '{"tracks":['
                                '{"disc":1,"number":2,"title":"Track B","length_sec":null},'
                                '{"disc":1,"number":1,"title":"Track A","length_sec":210}'
                                '],"sources":[]}'
                            ),
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://musicbrainz.org/release/abc",
                                }
                            ],
                        }
                    ],
                },
            ],
            "usage": {
                "input_tokens": 2000,
                "input_tokens_details": {"cached_tokens": 1000},
                "output_tokens": 500,
            },
        }
        return httpx.Response(200, json=payload)

    settings = Settings(
        azure_openai_endpoint="https://mock-foundry.services.ai.azure.com",
        azure_openai_api_key="test-azure-secret-key",
        azure_openai_model="gpt-5-mini",
        azure_openai_reasoning_effort="low",
    )
    http_client = _azure_mock_client(handler)
    client = AzureOpenAIAgentClient(settings, http_client=http_client)
    ans = client.research_album("0", "Test Album", "Test Artist")

    assert client.model_name == "gpt-5-mini"
    assert len(recorded) == 1
    assert recorded[0]["url"] == "https://mock-foundry.services.ai.azure.com/openai/v1/responses"
    assert recorded[0]["api_key"] == "test-azure-secret-key"
    assert recorded[0]["body"]["model"] == "gpt-5-mini"
    assert recorded[0]["body"]["tools"] == [{"type": "web_search"}]
    assert recorded[0]["body"]["reasoning"] == {"effort": "low"}
    assert "temperature" not in recorded[0]["body"]
    assert "barcode (EAN/UPC): unknown" in recorded[0]["body"]["input"]

    assert [t.title for t in ans.tracks] == ["Track A", "Track B"]
    assert ans.sources == [
        "https://www.discogs.com/release/12345",
        "https://musicbrainz.org/release/abc",
    ]
    assert ans.input_tokens == 2000
    assert ans.output_tokens == 500
    # (1000 * 0.25 + 1000 * 0.03 + 500 * 2.00) / 1_000_000 = 0.00128
    assert ans.cost_usd == pytest.approx(0.00128)


def test_azure_openai_agent_client_restructure_retry_and_permanent_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.agent_client.time.sleep", lambda _: None)

    base_settings = Settings(
        azure_openai_endpoint="https://mock-foundry.services.ai.azure.com/openai/v1/responses",
        azure_openai_api_key="fake-secret-value-never-leak",
        azure_openai_model="gpt-5-mini",
        agent_max_attempts=2,
    )

    # 1. Free-text answer triggers json_object restructure call without tools
    calls_1: list[dict[str, Any]] = []

    def handler_restructure(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls_1.append(body)
        if len(calls_1) == 1:
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output_text": "1. Alpha (3:00)\n2. Beta",
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output_text": (
                    '{"tracks":[{"disc":1,"number":1,"title":"Alpha","length_sec":180}],'
                    '"sources":["https://restructured.test"]}'
                ),
                "usage": {"input_tokens": 80, "output_tokens": 40},
            },
        )

    c1 = AzureOpenAIAgentClient(base_settings, http_client=_azure_mock_client(handler_restructure))
    ans1 = c1.research_album("1234567890123", "Album", "Artist")
    assert len(calls_1) == 2
    assert "tools" in calls_1[0]
    assert "tools" not in calls_1[1]
    assert calls_1[1]["text"] == {"format": {"type": "json_object"}}
    assert ans1.tracks[0].title == "Alpha"
    assert ans1.sources == ["https://restructured.test"]
    assert ans1.input_tokens == 180
    assert ans1.output_tokens == 90

    # 2. Permanent 401 fails immediately without retry and without leaking API key
    calls_2 = 0

    def handler_401(_request: httpx.Request) -> httpx.Response:
        nonlocal calls_2
        calls_2 += 1
        return httpx.Response(401, json={"error": {"message": "Access denied due to invalid key."}})

    c2 = AzureOpenAIAgentClient(base_settings, http_client=_azure_mock_client(handler_401))
    with pytest.raises(AgentFallbackError) as exc_info:
        c2.research_album("", "Auth Fail", "Artist")
    assert calls_2 == 1
    assert "fake-secret-value-never-leak" not in str(exc_info.value)

    # 3. Transient 429 retries and recovers
    calls_3 = 0

    def handler_429(_request: httpx.Request) -> httpx.Response:
        nonlocal calls_3
        calls_3 += 1
        if calls_3 == 1:
            return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output_text": (
                    '{"tracks":[{"disc":1,"number":1,"title":"Recovered","length_sec":100}],'
                    '"sources":["https://ok.test"]}'
                ),
                "usage": {"input_tokens": 50, "output_tokens": 25},
            },
        )

    c3 = AzureOpenAIAgentClient(base_settings, http_client=_azure_mock_client(handler_429))
    ans3 = c3.research_album("", "Retry Album", "Retry Artist")
    assert calls_3 == 2
    assert [t.title for t in ans3.tracks] == ["Recovered"]

    # 4. Persistent 503 raises AgentFallbackError after max attempts
    calls_4 = 0

    def handler_503(_request: httpx.Request) -> httpx.Response:
        nonlocal calls_4
        calls_4 += 1
        return httpx.Response(503, json={"error": {"message": "Service unavailable"}})

    c4 = AzureOpenAIAgentClient(base_settings, http_client=_azure_mock_client(handler_503))
    with pytest.raises(AgentFallbackError):
        c4.research_album("", "Fail Album", "Fail Artist")
    assert calls_4 == 2


def test_azure_openai_agent_client_enrich_missing_tracks() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output_text": (
                    '{"tracks":[{"track_order":1,"title":"Track 1","artist":"Artist 1",'
                    '"length_sec":205,"composers":["Comp A"],"lyricists":[],"producers":[],'
                    '"isrc":"DE1234567890","source_url":"https://musicbrainz.org/rec/1"}],'
                    '"sources":["https://musicbrainz.org/rec/1"]}'
                ),
                "usage": {"input_tokens": 1000, "output_tokens": 200},
            },
        )

    settings = Settings(
        azure_openai_endpoint="https://mock-foundry.services.ai.azure.com/openai/v1/responses",
        azure_openai_api_key="test-key",
        azure_openai_model="gpt-5-mini",
    )
    client = AzureOpenAIAgentClient(settings, http_client=_azure_mock_client(handler))
    ans = client.enrich_missing_tracks(
        barcode="123",
        album="Album",
        artist="Artist",
        release_url="https://discogs.com/release/1",
        missing_tracks=[{"track_order": 1, "title": "Track 1", "missing_fields": ["isrc"]}],
    )
    assert len(ans.tracks) == 1
    assert ans.tracks[0].isrc == "DE1234567890"
    assert ans.sources == ["https://musicbrainz.org/rec/1"]
    assert ans.cost_usd == pytest.approx(0.00065)


def test_settings_provider_resolution_and_build_agent_client() -> None:
    unconfigured = Settings()
    assert unconfigured.is_agent_configured() is False
    assert unconfigured.resolved_agent_provider() is None
    assert build_agent_client(unconfigured) is None

    placeholder_azure = Settings(
        azure_openai_endpoint="https://YOUR_RESOURCE.services.ai.azure.com/openai/v1/responses",
        azure_openai_api_key="PASTE_YOUR_AZURE_OPENAI_API_KEY",
    )
    assert placeholder_azure.is_azure_agent_configured() is False
    assert build_agent_client(placeholder_azure) is None

    both_auto = Settings(
        agent_provider="auto",
        azure_openai_endpoint="https://mock-foundry.services.ai.azure.com/openai/v1/responses",
        azure_openai_api_key="real-azure-key",
        azure_openai_model="gpt-5-mini",
        gemini_api_key="real-gemini-key",
    )
    assert both_auto.is_agent_configured() is True
    assert both_auto.resolved_agent_provider() == "azure"
    built_azure = build_agent_client(both_auto)
    assert isinstance(built_azure, AzureOpenAIAgentClient)
    assert built_azure.model_name == "gpt-5-mini"

    explicit_gemini = Settings(
        agent_provider="gemini",
        azure_openai_endpoint="https://mock-foundry.services.ai.azure.com/openai/v1/responses",
        azure_openai_api_key="real-azure-key",
        gemini_api_key="real-gemini-key",
    )
    assert explicit_gemini.resolved_agent_provider() == "gemini"
    assert isinstance(build_agent_client(explicit_gemini), GeminiAgentClient)

    disabled = Settings(
        enable_agent_fallback=False,
        azure_openai_endpoint="https://mock-foundry.services.ai.azure.com/openai/v1/responses",
        azure_openai_api_key="real-azure-key",
    )
    assert disabled.is_agent_configured() is False
    assert build_agent_client(disabled) is None
