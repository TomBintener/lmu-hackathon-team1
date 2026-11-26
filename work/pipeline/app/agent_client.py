import json
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - optional when running Azure-only
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from app.config import (
    PLACEHOLDER_AZURE_ENDPOINT_MARKERS,
    PLACEHOLDER_AZURE_KEYS,
    PLACEHOLDER_GEMINI_KEYS,
    Settings,
)
from app.cost import compute_token_cost

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

DISCOGS_RELEASE_URL_RE = re.compile(
    r"https?://(?:www\.)?discogs\.com/(?:[a-z]{2}/)?release/(\d+)", re.IGNORECASE
)

PROMPT = """You are a music metadata researcher. Find the track list of one specific release using web search.

Release (from our catalogue; it may contain typos or format hints like "(2CD)" or "(LP)"):
- barcode (EAN/UPC): {barcode}
- album title: {album}
- album artist: {artist}

Rules:
1. If a barcode is given, search for it first (MusicBrainz, Discogs, shops, label or artist pages). Then search by artist and title.
2. The barcode identifies the exact edition. Use the edition that matches it (e.g. deluxe vs. standard, 1 vs. 2 discs). Without a barcode, use hints in the title ("(2CD)", "(LP)", ...) to choose the edition.
3. Take tracks ONLY from pages that your search returned. Never write tracks from memory. If you cannot find a track list you trust, return an empty "tracks" list.
4. "Various", "VA" and similar mean a compilation. "Name, Die" means "Die Name".
5. Keep track titles exactly as written on the source and do not translate them. "disc" starts at 1; "number" is the position on that disc (for vinyl, count continuously per record: A1, A2, B1 -> 1, 2, 3). Set "length_sec" to the duration in seconds, or null if the source gives none.
6. Set "artist" to the performing artist(s) for each track (especially important on compilations), or null if the source gives none. Only include "composers", "lyricists", "producers", or "isrc" when explicitly stated on the returned source pages; otherwise leave them empty or null.
7. "sources" lists the URLs of the pages you took the tracks from.

Answer with a single JSON object and nothing else (no markdown, no comments):
{{"tracks": [{{"disc": 1, "number": 1, "title": "...", "artist": "...", "length_sec": 215, "composers": [], "lyricists": [], "producers": [], "isrc": null}}], "sources": ["https://..."]}}
"""

ENRICH_PROMPT = """You are a music metadata researcher. Find missing metadata for tracks on an already-matched release using web search.

Matched release:
- barcode (EAN/UPC): {barcode}
- album title: {album}
- album artist: {artist}
- reference release URL: {release_url}

Tracks needing enrichment (JSON list with track_order, title, current_artist, missing_fields):
{tracks_json}

Rules:
1. Only report metadata (artist, length_sec, composers, lyricists, producers, isrc) explicitly found on pages returned by web search for THIS exact recording/version. Never guess or fill from memory.
2. Do NOT transfer durations across incompatible versions (e.g. studio vs. live, album version vs. 7" Edit or Remix).
3. For every track where you find any missing field, set "source_url" to the exact web URL supporting that track's metadata. If no reliable source is found for a track, keep its fields null/empty.
4. Keep "track_order" identical to the input list.

Answer with a single JSON object and nothing else (no markdown, no comments):
{{"tracks": [{{"track_order": 1, "title": "...", "artist": null, "length_sec": null, "composers": [], "lyricists": [], "producers": [], "isrc": null, "source_url": "https://..."}}], "sources": ["https://..."]}}
"""


class AgentTrack(BaseModel):
    disc: int = 1
    number: int
    title: str
    artist: str | None = None
    length_sec: int | None = None
    composers: list[str] = Field(default_factory=list)
    lyricists: list[str] = Field(default_factory=list)
    producers: list[str] = Field(default_factory=list)
    isrc: str | None = None


class AgentAnswer(BaseModel):
    tracks: list[AgentTrack] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class AgentTrackEnrichment(BaseModel):
    track_order: int
    title: str = ""
    artist: str | None = None
    length_sec: int | None = None
    composers: list[str] = Field(default_factory=list)
    lyricists: list[str] = Field(default_factory=list)
    producers: list[str] = Field(default_factory=list)
    isrc: str | None = None
    source_url: str | None = None


class AgentEnrichmentAnswer(BaseModel):
    tracks: list[AgentTrackEnrichment] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def _extract_token_usage(response: Any) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from Gemini usage_metadata."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = (getattr(usage, "candidates_token_count", 0) or 0) + (
        getattr(usage, "thoughts_token_count", 0) or 0
    )
    return int(input_tokens), int(output_tokens)


class AgentFallbackError(Exception):
    """Raised when the agent fallback fails across all retry attempts."""


class _PermanentAgentError(Exception):
    """Non-retryable client/configuration/auth error from an upstream agent provider."""


class AgentFallbackProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def research_album(self, barcode: str, album: str, artist: str) -> AgentAnswer: ...


def parse_json_model(text: str, schema_cls: type[TModel]) -> TModel:
    """Extract and validate a JSON object into a Pydantic model."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in agent response.")
    return schema_cls.model_validate_json(cleaned[start : end + 1])


def parse_answer(text: str) -> AgentAnswer:
    """Extract and validate a JSON AgentAnswer from raw or fenced model output."""
    return parse_json_model(text, AgentAnswer)


def grounding_urls(response: Any) -> list[str]:
    """Extract unique web URIs returned in Gemini Google Search grounding metadata."""
    urls: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        metadata = getattr(candidate, "grounding_metadata", None)
        chunks = getattr(metadata, "grounding_chunks", None) if metadata else None
        for chunk in chunks or []:
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", None) if web else None
            if uri and uri not in urls:
                urls.append(uri)
    return urls


def dedupe_and_sort_tracks(tracks: list[AgentTrack]) -> list[AgentTrack]:
    """Sort tracks by (disc, number) and drop duplicate positions keeping the first."""
    deduped: dict[tuple[int, int], AgentTrack] = {}
    for track in sorted(tracks, key=lambda t: (t.disc, t.number)):
        deduped.setdefault((track.disc, track.number), track)
    return list(deduped.values())


def extract_discogs_release_ids(sources: list[str]) -> list[int]:
    """Extract unique positive Discogs release IDs from source URLs in appearance order."""
    ids: list[int] = []
    seen: set[int] = set()
    for src in sources:
        if not src:
            continue
        match = DISCOGS_RELEASE_URL_RE.search(src)
        if not match:
            continue
        rid = int(match.group(1))
        if rid > 0 and rid not in seen:
            seen.add(rid)
            ids.append(rid)
    return ids


def _normalize_azure_responses_url(endpoint: str) -> str:
    """Normalize an Azure AI Foundry / Azure OpenAI URL to .../openai/v1/responses."""
    cleaned = (endpoint or "").strip().rstrip("/")
    lowered = cleaned.lower()
    if lowered.endswith("/openai/v1/responses"):
        return cleaned
    if lowered.endswith("/openai/v1"):
        return f"{cleaned}/responses"
    return f"{cleaned}/openai/v1/responses"


def _extract_responses_text(payload: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI Responses API JSON payload."""
    top_text = payload.get("output_text")
    if isinstance(top_text, str) and top_text.strip():
        return top_text

    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            text = part.get("text")
            if isinstance(text, str) and text and part_type in ("output_text", "text", None):
                chunks.append(text)
    return "\n".join(chunks)


def _extract_responses_urls(payload: dict[str, Any]) -> list[str]:
    """Extract unique citation URLs from Responses API output annotations and search calls."""
    urls: list[str] = []

    def _add(url: Any) -> None:
        if isinstance(url, str):
            stripped = url.strip()
            if stripped and stripped not in urls:
                urls.append(stripped)

    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        if isinstance(action, dict):
            for src in action.get("sources") or []:
                if isinstance(src, dict):
                    _add(src.get("url"))
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            for ann in part.get("annotations") or []:
                if isinstance(ann, dict):
                    _add(ann.get("url"))
    return urls


def _extract_responses_usage(payload: dict[str, Any]) -> tuple[int, int, int]:
    """Extract (input_tokens, cached_input_tokens, output_tokens) from Responses API usage."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0
    input_tokens = int(usage.get("input_tokens") or 0)
    details = usage.get("input_tokens_details")
    cached_tokens = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    output_tokens = int(usage.get("output_tokens") or 0)
    return input_tokens, cached_tokens, output_tokens


def _extract_azure_error_message(response: httpx.Response) -> str:
    """Safely extract an error message from an Azure HTTP error response without leaking headers."""
    try:
        data = response.json()
    except ValueError:
        return (response.text or "").strip()[:300] or f"HTTP {response.status_code}"
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if data.get("message"):
            return str(data["message"])
    return (response.text or "").strip()[:300] or f"HTTP {response.status_code}"


def _build_research_answer(
    answer: AgentAnswer,
    fallback_urls: list[str],
    total_in: int,
    total_out: int,
    *,
    model: str | None = None,
    total_cached: int = 0,
) -> AgentAnswer:
    cost_usd = (
        compute_token_cost(
            total_in,
            total_out,
            model=model,
            cached_input_tokens=total_cached,
        )
        if (total_in > 0 or total_out > 0)
        else 0.0
    )
    tracks = dedupe_and_sort_tracks(answer.tracks)
    if not tracks:
        return AgentAnswer(
            input_tokens=total_in,
            output_tokens=total_out,
            cost_usd=cost_usd,
        )
    sources = [s for s in dict.fromkeys(src.strip() for src in answer.sources if src) if s]
    if not sources:
        sources = fallback_urls
    return AgentAnswer(
        tracks=tracks,
        sources=sources,
        input_tokens=total_in,
        output_tokens=total_out,
        cost_usd=cost_usd,
    )


def _build_enrichment_answer(
    parsed: AgentEnrichmentAnswer,
    fallback_urls: list[str],
    total_in: int,
    total_out: int,
    *,
    model: str | None = None,
    total_cached: int = 0,
) -> AgentEnrichmentAnswer:
    cost_usd = (
        compute_token_cost(
            total_in,
            total_out,
            model=model,
            cached_input_tokens=total_cached,
        )
        if (total_in > 0 or total_out > 0)
        else 0.0
    )
    sources = [s for s in dict.fromkeys(src.strip() for src in parsed.sources if src) if s]
    if not sources:
        sources = fallback_urls
    return AgentEnrichmentAnswer(
        tracks=parsed.tracks,
        sources=sources,
        input_tokens=total_in,
        output_tokens=total_out,
        cost_usd=cost_usd,
    )


class _BaseSearchAgentClient:
    """Shared retry, formatting, and post-processing logic for search-grounded agent clients."""

    _provider_label: str = "Agent"

    def __init__(self, settings: Settings, model_name: str) -> None:
        self._settings = settings
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def _run_with_retries(
        self,
        *,
        operation_label: str,
        fn: Callable[[int, int], TModel],
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> TModel:
        attempts = max(1, self._settings.agent_max_attempts)
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return fn(attempt, attempts)
            except _PermanentAgentError as exc:
                logger.warning(
                    "%s %s permanent failure: %s",
                    self._provider_label,
                    operation_label,
                    exc,
                )
                raise AgentFallbackError(
                    f"{self._provider_label} {operation_label} failed: {exc}"
                ) from exc
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "%s %s attempt %s/%s failed: %s",
                    self._provider_label,
                    operation_label,
                    attempt,
                    attempts,
                    exc,
                )
                if attempt < attempts:
                    delay = 2 ** (attempt + 1)
                    if on_progress is not None:
                        on_progress(
                            attempt,
                            attempts,
                            f"Attempt {attempt}/{attempts} failed; retrying in {delay}s",
                        )
                    time.sleep(delay)
        raise AgentFallbackError(
            f"{self._provider_label} {operation_label} failed after {attempts} attempt(s): {last_exc}"
        ) from last_exc


class GeminiAgentClient(_BaseSearchAgentClient):
    """Gemini + Google Search tracklist discovery fallback client."""

    _provider_label = "Gemini"

    def __init__(self, settings: Settings, genai_client: Any | None = None) -> None:
        super().__init__(settings, settings.google_genai_model)
        self._external_client = genai_client
        self._owned_client: Any | None = None
        self._client_lock = threading.Lock()

    def _get_client(self) -> Any:
        if self._external_client is not None:
            return self._external_client
        with self._client_lock:
            if self._owned_client is None:
                if genai is None:
                    raise _PermanentAgentError("google-genai package is not installed.")
                api_key = (self._settings.gemini_api_key or "").strip()
                if not api_key or api_key in PLACEHOLDER_GEMINI_KEYS:
                    raise _PermanentAgentError("GEMINI_API_KEY is not configured.")
                self._owned_client = genai.Client(vertexai=False, api_key=api_key)
            return self._owned_client

    def close(self) -> None:
        with self._client_lock:
            if self._owned_client is not None:
                close_fn = getattr(self._owned_client, "close", None)
                if callable(close_fn):
                    close_fn()
                self._owned_client = None

    def _restructure(
        self, client: Any, text: str, schema_cls: type[TModel] = AgentAnswer
    ) -> tuple[TModel, int, int]:
        if not text.strip():
            return schema_cls(), 0, 0
        response = client.models.generate_content(
            model=self._model_name,
            contents=(
                "Convert the following text into the JSON schema. Use only information "
                "contained in the text. If it contains no track list, return empty lists.\n\n"
                + text
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema_cls,
                temperature=0,
            ),
        )
        in_tok, out_tok = _extract_token_usage(response)
        return schema_cls.model_validate_json(response.text or "{}"), in_tok, out_tok

    def _run_grounded_json_prompt(
        self,
        client: Any,
        prompt: str,
        schema_cls: type[TModel],
        label: str,
        attempt: int = 1,
        attempts: int = 1,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> tuple[TModel, list[str], int, int]:
        if on_progress is not None:
            on_progress(
                attempt,
                attempts,
                f"Gemini + Google Search attempt {attempt}/{attempts} ({self._model_name})",
            )
        response = client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0,
            ),
        )
        total_in, total_out = _extract_token_usage(response)
        text = getattr(response, "text", None) or ""
        try:
            parsed = parse_json_model(text, schema_cls)
        except ValueError:
            logger.info("Agent output for %r required schema restructure fallback", label)
            if on_progress is not None:
                on_progress(
                    attempt,
                    attempts,
                    f"Attempt {attempt}/{attempts}: restructuring free-text output to JSON",
                )
            parsed, rest_in, rest_out = self._restructure(client, text, schema_cls=schema_cls)
            total_in += rest_in
            total_out += rest_out
        return parsed, grounding_urls(response), total_in, total_out

    def _research_once(
        self,
        client: Any,
        barcode: str,
        album: str,
        artist: str,
        attempt: int = 1,
        attempts: int = 1,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> AgentAnswer:
        prompt = PROMPT.format(
            barcode=barcode if barcode not in ("", "0") else "unknown",
            album=album,
            artist=artist,
        )
        answer, g_urls, total_in, total_out = self._run_grounded_json_prompt(
            client=client,
            prompt=prompt,
            schema_cls=AgentAnswer,
            label=album,
            attempt=attempt,
            attempts=attempts,
            on_progress=on_progress,
        )
        return _build_research_answer(
            answer,
            g_urls,
            total_in,
            total_out,
            model=self._model_name,
        )

    def enrich_missing_tracks(
        self,
        barcode: str,
        album: str,
        artist: str,
        release_url: str,
        missing_tracks: list[dict[str, Any]],
    ) -> AgentEnrichmentAnswer:
        """Run one batched grounded search to fill missing fields for incomplete tracks."""
        if not missing_tracks:
            return AgentEnrichmentAnswer()
        try:
            client = self._get_client()
        except _PermanentAgentError as exc:
            raise AgentFallbackError(str(exc)) from exc
        prompt = ENRICH_PROMPT.format(
            barcode=barcode if barcode not in ("", "0") else "unknown",
            album=album,
            artist=artist or "Various",
            release_url=release_url or "unknown",
            tracks_json=json.dumps(missing_tracks, ensure_ascii=False),
        )

        def _once(attempt: int, attempts: int) -> AgentEnrichmentAnswer:
            parsed, g_urls, total_in, total_out = self._run_grounded_json_prompt(
                client=client,
                prompt=prompt,
                schema_cls=AgentEnrichmentAnswer,
                label=f"enrich:{album}",
                attempt=attempt,
                attempts=attempts,
                on_progress=None,
            )
            return _build_enrichment_answer(
                parsed,
                g_urls,
                total_in,
                total_out,
                model=self._model_name,
            )

        return self._run_with_retries(
            operation_label=f"track enrichment ({artist} - {album})",
            fn=_once,
        )

    def research_album_with_progress(
        self,
        barcode: str,
        album: str,
        artist: str,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> AgentAnswer:
        try:
            client = self._get_client()
        except _PermanentAgentError as exc:
            raise AgentFallbackError(str(exc)) from exc

        def _once(attempt: int, attempts: int) -> AgentAnswer:
            logger.info(
                "Gemini agent research attempt %s/%s for %s - %s (barcode=%s)",
                attempt,
                attempts,
                artist,
                album,
                barcode or "-",
            )
            return self._research_once(
                client,
                barcode,
                album,
                artist,
                attempt=attempt,
                attempts=attempts,
                on_progress=on_progress,
            )

        return self._run_with_retries(
            operation_label=f"agent fallback ({artist} - {album})",
            fn=_once,
            on_progress=on_progress,
        )

    def research_album(self, barcode: str, album: str, artist: str) -> AgentAnswer:
        return self.research_album_with_progress(barcode, album, artist, on_progress=None)


class AzureOpenAIAgentClient(_BaseSearchAgentClient):
    """Azure AI Foundry OpenAI v1 Responses API tracklist discovery & enrichment client."""

    _provider_label = "Azure OpenAI"

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.Client | None = None,
    ) -> None:
        super().__init__(settings, settings.azure_openai_model)
        self._external_client = http_client
        self._owned_client: httpx.Client | None = None
        self._client_lock = threading.Lock()

    def _get_client_and_credentials(self) -> tuple[httpx.Client, str, str]:
        endpoint = (self._settings.azure_openai_endpoint or "").strip()
        api_key = (self._settings.azure_openai_api_key or "").strip()
        if not endpoint or any(m in endpoint for m in PLACEHOLDER_AZURE_ENDPOINT_MARKERS):
            raise _PermanentAgentError("AZURE_OPENAI_ENDPOINT is not configured.")
        if not api_key or api_key in PLACEHOLDER_AZURE_KEYS:
            raise _PermanentAgentError("AZURE_OPENAI_API_KEY is not configured.")

        url = _normalize_azure_responses_url(endpoint)
        if self._external_client is not None:
            return self._external_client, url, api_key
        with self._client_lock:
            if self._owned_client is None:
                self._owned_client = httpx.Client(timeout=self._settings.azure_openai_timeout_sec)
            return self._owned_client, url, api_key

    def close(self) -> None:
        with self._client_lock:
            if self._owned_client is not None:
                self._owned_client.close()
                self._owned_client = None

    def _build_payload(
        self,
        prompt: str,
        *,
        use_web_search: bool,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "input": prompt,
        }
        effort = (self._settings.azure_openai_reasoning_effort or "").strip().lower()
        if json_mode:
            payload["reasoning"] = {"effort": "low"}
            payload["text"] = {"format": {"type": "json_object"}}
        elif effort:
            payload["reasoning"] = {"effort": effort}

        if use_web_search and self._settings.azure_openai_enable_web_search:
            payload["tools"] = [{"type": "web_search"}]
        return payload

    def _post_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        client, url, api_key = self._get_client_and_credentials()
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
        }
        response = client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            msg = _extract_azure_error_message(response)
            if response.status_code in (408, 429) or response.status_code >= 500:
                raise RuntimeError(f"Azure OpenAI HTTP {response.status_code}: {msg}")
            raise _PermanentAgentError(f"Azure OpenAI HTTP {response.status_code}: {msg}")

        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Azure OpenAI returned non-object JSON payload.")

        status = str(data.get("status") or "").strip().lower()
        if status == "failed":
            err = data.get("error") if isinstance(data.get("error"), dict) else {}
            err_msg = str(err.get("message") or "Azure OpenAI response failed.")
            err_code = str(err.get("code") or "").lower()
            if any(token in err_code for token in ("rate_limit", "server_error", "timeout")):
                raise RuntimeError(f"Azure OpenAI transient failure: {err_msg}")
            raise _PermanentAgentError(err_msg)
        if status == "incomplete" and not _extract_responses_text(data).strip():
            details = data.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, dict) else "unknown"
            raise RuntimeError(f"Azure OpenAI response incomplete ({reason})")
        return data

    def _restructure(
        self, text: str, schema_cls: type[TModel] = AgentAnswer
    ) -> tuple[TModel, int, int, int]:
        if not text.strip():
            return schema_cls(), 0, 0, 0
        prompt = (
            "Convert the following text into a single valid JSON object matching the requested "
            "schema. Use only information contained in the text. If it contains no track list, "
            "return empty lists.\n\n" + text
        )
        data = self._post_response(
            self._build_payload(prompt, use_web_search=False, json_mode=True)
        )
        in_tok, cached_tok, out_tok = _extract_responses_usage(data)
        parsed = parse_json_model(_extract_responses_text(data), schema_cls)
        return parsed, in_tok, cached_tok, out_tok

    def _run_grounded_json_prompt(
        self,
        prompt: str,
        schema_cls: type[TModel],
        label: str,
        attempt: int = 1,
        attempts: int = 1,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> tuple[TModel, list[str], int, int, int]:
        if on_progress is not None:
            search_tag = " + web search" if self._settings.azure_openai_enable_web_search else ""
            on_progress(
                attempt,
                attempts,
                f"Azure {self._model_name}{search_tag} attempt {attempt}/{attempts}",
            )
        data = self._post_response(self._build_payload(prompt, use_web_search=True))
        total_in, total_cached, total_out = _extract_responses_usage(data)
        text = _extract_responses_text(data)
        urls = _extract_responses_urls(data)
        try:
            parsed = parse_json_model(text, schema_cls)
        except ValueError:
            logger.info("Azure agent output for %r required JSON restructure fallback", label)
            if on_progress is not None:
                on_progress(
                    attempt,
                    attempts,
                    f"Attempt {attempt}/{attempts}: restructuring free-text output to JSON",
                )
            parsed, rest_in, rest_cached, rest_out = self._restructure(text, schema_cls=schema_cls)
            total_in += rest_in
            total_cached += rest_cached
            total_out += rest_out
        return parsed, urls, total_in, total_cached, total_out

    def enrich_missing_tracks(
        self,
        barcode: str,
        album: str,
        artist: str,
        release_url: str,
        missing_tracks: list[dict[str, Any]],
    ) -> AgentEnrichmentAnswer:
        """Run one batched grounded search to fill missing fields for incomplete tracks."""
        if not missing_tracks:
            return AgentEnrichmentAnswer()
        prompt = ENRICH_PROMPT.format(
            barcode=barcode if barcode not in ("", "0") else "unknown",
            album=album,
            artist=artist or "Various",
            release_url=release_url or "unknown",
            tracks_json=json.dumps(missing_tracks, ensure_ascii=False),
        )

        def _once(attempt: int, attempts: int) -> AgentEnrichmentAnswer:
            parsed, urls, total_in, total_cached, total_out = self._run_grounded_json_prompt(
                prompt=prompt,
                schema_cls=AgentEnrichmentAnswer,
                label=f"enrich:{album}",
                attempt=attempt,
                attempts=attempts,
                on_progress=None,
            )
            return _build_enrichment_answer(
                parsed,
                urls,
                total_in,
                total_out,
                model=self._model_name,
                total_cached=total_cached,
            )

        return self._run_with_retries(
            operation_label=f"track enrichment ({artist} - {album})",
            fn=_once,
        )

    def research_album_with_progress(
        self,
        barcode: str,
        album: str,
        artist: str,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> AgentAnswer:
        prompt = PROMPT.format(
            barcode=barcode if barcode not in ("", "0") else "unknown",
            album=album,
            artist=artist,
        )

        def _once(attempt: int, attempts: int) -> AgentAnswer:
            logger.info(
                "Azure agent research attempt %s/%s for %s - %s (barcode=%s, model=%s)",
                attempt,
                attempts,
                artist,
                album,
                barcode or "-",
                self._model_name,
            )
            answer, urls, total_in, total_cached, total_out = self._run_grounded_json_prompt(
                prompt=prompt,
                schema_cls=AgentAnswer,
                label=album,
                attempt=attempt,
                attempts=attempts,
                on_progress=on_progress,
            )
            return _build_research_answer(
                answer,
                urls,
                total_in,
                total_out,
                model=self._model_name,
                total_cached=total_cached,
            )

        return self._run_with_retries(
            operation_label=f"agent fallback ({artist} - {album})",
            fn=_once,
            on_progress=on_progress,
        )

    def research_album(self, barcode: str, album: str, artist: str) -> AgentAnswer:
        return self.research_album_with_progress(barcode, album, artist, on_progress=None)


def build_agent_client(settings: Settings) -> GeminiAgentClient | AzureOpenAIAgentClient | None:
    """Instantiate the configured search-grounded agent client, or None if unconfigured."""
    if not settings.enable_agent_fallback:
        return None
    provider = settings.resolved_agent_provider()
    if provider == "azure":
        return AzureOpenAIAgentClient(settings)
    if provider == "gemini":
        return GeminiAgentClient(settings)
    return None
